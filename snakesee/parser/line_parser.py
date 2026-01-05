"""Log line parsing with context tracking."""

import logging
from dataclasses import dataclass
from dataclasses import field
from enum import Enum
from typing import NamedTuple

# Import regex patterns from patterns module (single source of truth)
from snakesee.parser.patterns import ERROR_IN_RULE_PATTERN
from snakesee.parser.patterns import FINISHED_JOB_PATTERN
from snakesee.parser.patterns import PROGRESS_PATTERN
from snakesee.parser.patterns import RULE_START_PATTERN
from snakesee.parser.patterns import TIMESTAMP_PATTERN

# Import helper functions from utils module (single source of truth)
from snakesee.parser.utils import _parse_positive_int
from snakesee.parser.utils import _parse_timestamp
from snakesee.parser.utils import _parse_wildcards

logger = logging.getLogger(__name__)


class ParseEventType(Enum):
    """Types of events that can be parsed from a log line."""

    TIMESTAMP = "timestamp"
    PROGRESS = "progress"
    RULE_START = "rule_start"
    WILDCARDS = "wildcards"
    THREADS = "threads"
    LOG_PATH = "log_path"
    JOBID = "jobid"
    JOB_FINISHED = "job_finished"
    ERROR = "error"


class ParseEvent(NamedTuple):
    """Result of parsing a log line."""

    event_type: ParseEventType
    data: dict[str, object]


@dataclass
class ParsingContext:
    """Current parsing state for multi-line log entries.

    Snakemake logs use multi-line blocks where context from earlier
    lines (rule, wildcards, etc.) applies to later lines (jobid).
    """

    rule: str | None = None
    jobid: str | None = None
    wildcards: dict[str, str] | None = None
    threads: int | None = None
    timestamp: float | None = None
    log_path: str | None = None

    # Pending error state - we defer ERROR emission until the error block is fully parsed
    # because jobid and log come AFTER "Error in rule X:" in the error block
    pending_error_rule: str | None = None
    error_jobid: str | None = None
    error_wildcards: dict[str, str] | None = None
    error_threads: int | None = None
    error_log_path: str | None = None

    def reset_for_new_rule(self, rule: str) -> None:
        """Reset context when entering a new rule block.

        Args:
            rule: Name of the new rule.
        """
        self.rule = rule
        self.jobid = None
        self.wildcards = None
        self.threads = None
        self.log_path = None

    def start_error_block(self, rule: str) -> None:
        """Start tracking a pending error block.

        Args:
            rule: Name of the rule that errored.
        """
        self.pending_error_rule = rule
        # Initialize with current context if rule matches
        if self.rule == rule:
            self.error_jobid = self.jobid
            self.error_wildcards = self.wildcards
            self.error_threads = self.threads
            self.error_log_path = self.log_path
        else:
            self.error_jobid = None
            self.error_wildcards = None
            self.error_threads = None
            self.error_log_path = None

    def get_pending_error(self) -> ParseEvent | None:
        """Get the pending error event and clear it.

        Returns:
            ParseEvent for the error, or None if no pending error.
        """
        if self.pending_error_rule is None:
            return None

        # Strip "(check log file(s) for error details)" suffix from error blocks
        log_path = self.error_log_path
        if log_path and " (check log file" in log_path:
            log_path = log_path.split(" (check log file")[0].strip()

        event = ParseEvent(
            ParseEventType.ERROR,
            {
                "rule": self.pending_error_rule,
                "jobid": self.error_jobid,
                "wildcards": self.error_wildcards,
                "threads": self.error_threads,
                "log_path": log_path,
            },
        )
        # Clear pending error state
        self.pending_error_rule = None
        self.error_jobid = None
        self.error_wildcards = None
        self.error_threads = None
        self.error_log_path = None
        return event

    def has_pending_error(self) -> bool:
        """Check if there's a pending error to emit."""
        return self.pending_error_rule is not None


@dataclass
class LogLineParser:
    """Parses individual Snakemake log lines.

    Maintains parsing context across lines to handle multi-line
    log entries where information spans multiple lines.
    """

    context: ParsingContext = field(default_factory=ParsingContext)

    def parse_line(self, line: str) -> list[ParseEvent]:
        """Parse a single log line and return structured events.

        Uses fast-path prefix checks to skip expensive regex operations
        for lines that can't possibly match.

        Updates internal context as needed for multi-line entries.
        May return multiple events when a pending error needs to be flushed.

        Args:
            line: Log line to parse.

        Returns:
            List of ParseEvents (may be empty, one, or two events).
        """
        line = line.rstrip("\n\r")
        events: list[ParseEvent] = []

        # Fast path: empty lines
        if not line:
            return events

        first_char = line[0]

        # Timestamp lines start with '[' - this ends error blocks
        if first_char == "[":
            if match := TIMESTAMP_PATTERN.match(line):
                # Flush any pending error before emitting timestamp
                if pending := self.context.get_pending_error():
                    events.append(pending)
                timestamp = _parse_timestamp(match.group(1))
                self.context.timestamp = timestamp
                events.append(ParseEvent(ParseEventType.TIMESTAMP, {"timestamp": timestamp}))
            return events

        # Indented lines (properties) start with space/tab
        if first_char in (" ", "\t"):
            event = self._parse_indented_line(line)
            if event:
                events.append(event)
            return events

        # Rule start: "rule X:" or "localrule X:" - this ends error blocks
        if first_char == "r" and line.startswith("rule "):
            if match := RULE_START_PATTERN.match(line):
                # Flush any pending error before emitting rule start
                if pending := self.context.get_pending_error():
                    events.append(pending)
                rule = match.group(1)
                self.context.reset_for_new_rule(rule)
                events.append(ParseEvent(ParseEventType.RULE_START, {"rule": rule}))
            return events

        if first_char == "l" and line.startswith("localrule "):
            if match := RULE_START_PATTERN.match(line):
                # Flush any pending error before emitting rule start
                if pending := self.context.get_pending_error():
                    events.append(pending)
                rule = match.group(1)
                self.context.reset_for_new_rule(rule)
                events.append(ParseEvent(ParseEventType.RULE_START, {"rule": rule}))
            return events

        # Finished job: "Finished job X" or "Finished jobid: X"
        if first_char == "F" and line.startswith("Finished "):
            if match := FINISHED_JOB_PATTERN.search(line):
                jobid = match.group(1)
                events.append(
                    ParseEvent(
                        ParseEventType.JOB_FINISHED,
                        {"jobid": jobid, "timestamp": self.context.timestamp},
                    )
                )
            return events

        # Error detection: "Error in rule X:" - starts a pending error block
        if first_char == "E" and line.startswith("Error in rule "):
            if match := ERROR_IN_RULE_PATTERN.search(line):
                # Flush any previous pending error before starting new one
                if pending := self.context.get_pending_error():
                    events.append(pending)
                rule = match.group(1)
                # Start pending error - we'll capture jobid/log from subsequent lines
                self.context.start_error_block(rule)
            return events

        # Progress line: "X of Y steps (Z%) done" - check with substring first
        if "steps" in line and "done" in line:
            if match := PROGRESS_PATTERN.search(line):
                completed = int(match.group(1))
                total = int(match.group(2))
                events.append(
                    ParseEvent(ParseEventType.PROGRESS, {"completed": completed, "total": total})
                )

        return events

    def flush_pending_error(self) -> ParseEvent | None:
        """Flush any pending error event.

        Call this at end of file or when needing to ensure all errors are emitted.

        Returns:
            ParseEvent for the pending error, or None if no pending error.
        """
        return self.context.get_pending_error()

    def _parse_indented_line(self, line: str) -> ParseEvent | None:
        """Parse indented property lines (wildcards, threads, log, jobid).

        Args:
            line: Indented log line starting with space/tab.

        Returns:
            ParseEvent if line contains a recognized property, None otherwise.
        """
        stripped = line.lstrip()

        # Check each property type by prefix (faster than regex)
        if stripped.startswith("wildcards:"):
            value = stripped[10:].strip()  # len('wildcards:') = 10
            wildcards = _parse_wildcards(value)
            self.context.wildcards = wildcards
            # Also update error context if in error block
            if self.context.has_pending_error():
                self.context.error_wildcards = wildcards
            return ParseEvent(
                ParseEventType.WILDCARDS,
                {"wildcards": wildcards, "jobid": self.context.jobid},
            )

        if stripped.startswith("threads:"):
            value = stripped[8:].strip()  # len('threads:') = 8
            threads = _parse_positive_int(value, "threads")
            if threads is not None:
                self.context.threads = threads
                # Also update error context if in error block
                if self.context.has_pending_error():
                    self.context.error_threads = threads
                return ParseEvent(
                    ParseEventType.THREADS,
                    {"threads": threads, "jobid": self.context.jobid},
                )
            return None

        if stripped.startswith("log:"):
            log_path = stripped[4:].strip()  # len('log:') = 4
            self.context.log_path = log_path
            # Also update error context if in error block
            if self.context.has_pending_error():
                self.context.error_log_path = log_path
            return ParseEvent(
                ParseEventType.LOG_PATH,
                {"log_path": log_path, "jobid": self.context.jobid},
            )

        if stripped.startswith("jobid:"):
            jobid = stripped[6:].strip()  # len('jobid:') = 6
            self.context.jobid = jobid
            # Also update error context if in error block
            if self.context.has_pending_error():
                self.context.error_jobid = jobid
            return ParseEvent(
                ParseEventType.JOBID,
                {
                    "jobid": jobid,
                    "rule": self.context.rule,
                    "wildcards": self.context.wildcards,
                    "threads": self.context.threads,
                    "timestamp": self.context.timestamp,
                    "log_path": self.context.log_path,
                },
            )

        return None

    def reset(self) -> None:
        """Reset parsing context."""
        self.context = ParsingContext()
