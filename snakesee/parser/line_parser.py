"""Log line parsing with context tracking."""

import logging
import time
from dataclasses import dataclass
from dataclasses import field
from enum import Enum
from typing import NamedTuple

# Import regex patterns from patterns module (single source of truth)
from snakesee.parser.patterns import ERROR_IN_RULE_PATTERN
from snakesee.parser.patterns import FINISHED_JOB_PATTERN
from snakesee.parser.patterns import JOBID_PATTERN
from snakesee.parser.patterns import LOG_PATTERN
from snakesee.parser.patterns import PROGRESS_PATTERN
from snakesee.parser.patterns import RULE_START_PATTERN
from snakesee.parser.patterns import THREADS_PATTERN
from snakesee.parser.patterns import TIMESTAMP_PATTERN
from snakesee.parser.patterns import WILDCARDS_PATTERN

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

        Updates internal context as needed for multi-line entries.

        Args:
            line: Log line to parse.

        Returns:
            ParseEvent if the line contains relevant information, None otherwise.
        """
        line = line.rstrip("\n\r")

        # Check for timestamp
        if match := TIMESTAMP_PATTERN.match(line):
            timestamp = _parse_timestamp(match.group(1))
            self.context.timestamp = timestamp
            return ParseEvent(ParseEventType.TIMESTAMP, {"timestamp": timestamp})

        # Check for progress
        if match := PROGRESS_PATTERN.search(line):
            completed = int(match.group(1))
            total = int(match.group(2))
            return ParseEvent(ParseEventType.PROGRESS, {"completed": completed, "total": total})

        # Track current rule being executed
        if match := RULE_START_PATTERN.match(line):
            rule = match.group(1)
            self.context.reset_for_new_rule(rule)
            return ParseEvent(ParseEventType.RULE_START, {"rule": rule})

        # Capture wildcards within rule block
        if match := WILDCARDS_PATTERN.match(line):
            wildcards = _parse_wildcards(match.group(1))
            self.context.wildcards = wildcards
            return ParseEvent(
                ParseEventType.WILDCARDS,
                {"wildcards": wildcards, "jobid": self.context.jobid},
            )

        # Capture threads within rule block
        if match := THREADS_PATTERN.match(line):
            threads = _parse_positive_int(match.group(1), "threads")
            if threads is not None:
                self.context.threads = threads
                return ParseEvent(
                    ParseEventType.THREADS,
                    {"threads": threads, "jobid": self.context.jobid},
                )
            return None

        # Capture log path within rule block
        if match := LOG_PATTERN.match(line):
            log_path = match.group(1).strip()
            self.context.log_path = log_path
            return ParseEvent(
                ParseEventType.LOG_PATH,
                {"log_path": log_path, "jobid": self.context.jobid},
            )

        # Capture jobid within rule block
        if match := JOBID_PATTERN.match(line):
            jobid = match.group(1)
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

        # Track finished jobs
        if match := FINISHED_JOB_PATTERN.search(line):
            jobid = match.group(1)
            return ParseEvent(
                ParseEventType.JOB_FINISHED,
                {"jobid": jobid, "timestamp": self.context.timestamp},
            )

        # Detect errors
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

    def reset(self) -> None:
        """Reset parsing context."""
        self.context = ParsingContext()


def _parse_timestamp(timestamp_str: str) -> float:
    """Parse a snakemake timestamp string to Unix timestamp.

    Args:
        timestamp_str: String like "Mon Dec 15 22:34:30 2025"

    Returns:
        Unix timestamp (seconds since epoch).
    """
    try:
        return time.mktime(time.strptime(timestamp_str, "%a %b %d %H:%M:%S %Y"))
    except (ValueError, OverflowError):
        return 0.0


def _parse_wildcards(wildcards_str: str) -> dict[str, str]:
    """Parse a wildcards string into a dictionary.

    Args:
        wildcards_str: String like "sample=A, batch=1"

    Returns:
        Dictionary like {"sample": "A", "batch": "1"}
    """
    wildcards: dict[str, str] = {}
    for part in wildcards_str.split(","):
        part = part.strip()
        if "=" in part:
            key, value = part.split("=", 1)
            wildcards[key.strip()] = value.strip()
    return wildcards


def _parse_positive_int(value: str, field_name: str = "value") -> int | None:
    """Parse a string as a positive integer with validation.

    Args:
        value: String to parse.
        field_name: Name of field for logging.

    Returns:
        Parsed integer if valid and positive, None otherwise.
    """
    try:
        result = int(value)
        if result <= 0:
            logger.debug("Invalid %s: %d (must be positive)", field_name, result)
            return None
        return result
    except ValueError:
        logger.debug("Could not parse %s as integer: %s", field_name, value)
        return None
