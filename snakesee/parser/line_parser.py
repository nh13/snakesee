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


@dataclass
class LogLineParser:
    """Parses individual Snakemake log lines.

    Maintains parsing context across lines to handle multi-line
    log entries where information spans multiple lines.
    """

    context: ParsingContext = field(default_factory=ParsingContext)

    def parse_line(self, line: str) -> ParseEvent | None:
        """Parse a single log line and return structured event.

        Uses fast-path prefix checks to skip expensive regex operations
        for lines that can't possibly match.

        Updates internal context as needed for multi-line entries.

        Args:
            line: Log line to parse.

        Returns:
            ParseEvent if the line contains relevant information, None otherwise.
        """
        line = line.rstrip("\n\r")

        # Fast path: empty lines
        if not line:
            return None

        first_char = line[0]

        # Timestamp lines start with '['
        if first_char == "[":
            if match := TIMESTAMP_PATTERN.match(line):
                timestamp = _parse_timestamp(match.group(1))
                self.context.timestamp = timestamp
                return ParseEvent(ParseEventType.TIMESTAMP, {"timestamp": timestamp})
            return None

        # Indented lines (properties) start with space/tab
        if first_char in (" ", "\t"):
            return self._parse_indented_line(line)

        # Rule start: "rule X:" or "localrule X:"
        if first_char == "r" and line.startswith("rule "):
            if match := RULE_START_PATTERN.match(line):
                rule = match.group(1)
                self.context.reset_for_new_rule(rule)
                return ParseEvent(ParseEventType.RULE_START, {"rule": rule})
            return None

        if first_char == "l" and line.startswith("localrule "):
            if match := RULE_START_PATTERN.match(line):
                rule = match.group(1)
                self.context.reset_for_new_rule(rule)
                return ParseEvent(ParseEventType.RULE_START, {"rule": rule})
            return None

        # Finished job: "Finished job X" or "Finished jobid: X"
        if first_char == "F" and line.startswith("Finished "):
            if match := FINISHED_JOB_PATTERN.search(line):
                jobid = match.group(1)
                return ParseEvent(
                    ParseEventType.JOB_FINISHED,
                    {"jobid": jobid, "timestamp": self.context.timestamp},
                )
            return None

        # Error detection: "Error in rule X:"
        if first_char == "E" and line.startswith("Error in rule "):
            if match := ERROR_IN_RULE_PATTERN.search(line):
                rule = match.group(1)
                # Use context if error rule matches current context
                if self.context.rule == rule:
                    return ParseEvent(
                        ParseEventType.ERROR,
                        {
                            "rule": rule,
                            "jobid": self.context.jobid,
                            "wildcards": self.context.wildcards,
                            "threads": self.context.threads,
                            "log_path": self.context.log_path,
                        },
                    )
                return ParseEvent(
                    ParseEventType.ERROR,
                    {
                        "rule": rule,
                        "jobid": None,
                        "wildcards": None,
                        "threads": None,
                        "log_path": None,
                    },
                )
            return None

        # Progress line: "X of Y steps (Z%) done" - check with substring first
        if "steps" in line and "done" in line:
            if match := PROGRESS_PATTERN.search(line):
                completed = int(match.group(1))
                total = int(match.group(2))
                return ParseEvent(ParseEventType.PROGRESS, {"completed": completed, "total": total})

        return None

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
            return ParseEvent(
                ParseEventType.WILDCARDS,
                {"wildcards": wildcards, "jobid": self.context.jobid},
            )

        if stripped.startswith("threads:"):
            value = stripped[8:].strip()  # len('threads:') = 8
            threads = _parse_positive_int(value, "threads")
            if threads is not None:
                self.context.threads = threads
                return ParseEvent(
                    ParseEventType.THREADS,
                    {"threads": threads, "jobid": self.context.jobid},
                )
            return None

        if stripped.startswith("log:"):
            log_path = stripped[4:].strip()  # len('log:') = 4
            self.context.log_path = log_path
            return ParseEvent(
                ParseEventType.LOG_PATH,
                {"log_path": log_path, "jobid": self.context.jobid},
            )

        if stripped.startswith("jobid:"):
            jobid = stripped[6:].strip()  # len('jobid:') = 6
            self.context.jobid = jobid
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
