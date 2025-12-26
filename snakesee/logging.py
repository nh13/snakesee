"""Logging configuration for snakesee.

This module provides centralized logging configuration with support for
both human-readable and structured (JSON) output formats.

Usage:
    from snakesee.logging import configure_logging

    # Basic configuration
    configure_logging(level="INFO")

    # JSON output for log aggregation
    configure_logging(level="DEBUG", json_output=True)
"""

import json
import logging
import sys
from datetime import datetime
from datetime import timezone
from typing import ClassVar
from typing import TextIO


class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured log output.

    Produces JSON lines suitable for log aggregation systems like
    Elasticsearch, Splunk, or CloudWatch.

    Each log entry includes:
    - timestamp: ISO 8601 format with timezone
    - level: Log level name
    - logger: Logger name
    - message: Log message
    - Additional fields from the record's extra dict
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format a log record as JSON."""
        log_entry: dict[str, object] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add location info if available
        if record.pathname and record.lineno:
            log_entry["location"] = {
                "file": record.pathname,
                "line": record.lineno,
                "function": record.funcName,
            }

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        # Add any extra fields
        extra_fields = {
            key: value
            for key, value in record.__dict__.items()
            if key
            not in {
                "name",
                "msg",
                "args",
                "created",
                "filename",
                "funcName",
                "levelname",
                "levelno",
                "lineno",
                "module",
                "msecs",
                "pathname",
                "process",
                "processName",
                "relativeCreated",
                "stack_info",
                "exc_info",
                "exc_text",
                "thread",
                "threadName",
                "taskName",
                "message",
            }
        }
        if extra_fields:
            log_entry["extra"] = extra_fields

        return json.dumps(log_entry, default=str)


class ColoredFormatter(logging.Formatter):
    """Colored formatter for human-readable console output."""

    COLORS: ClassVar[dict[str, str]] = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET: ClassVar[str] = "\033[0m"

    def __init__(self, fmt: str | None = None, use_color: bool = True) -> None:
        super().__init__(fmt)
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        """Format a log record with optional color."""
        if self.use_color and record.levelname in self.COLORS:
            levelname = f"{self.COLORS[record.levelname]}{record.levelname}{self.RESET}"
            record = logging.makeLogRecord(record.__dict__)
            record.levelname = levelname
        return super().format(record)


def configure_logging(
    level: str | int = "WARNING",
    json_output: bool = False,
    stream: TextIO | None = None,
    include_timestamp: bool = True,
) -> None:
    """
    Configure logging for snakesee.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL or int).
        json_output: If True, output structured JSON logs.
        stream: Output stream. Defaults to stderr.
        include_timestamp: Include timestamp in human-readable output.
    """
    if stream is None:
        stream = sys.stderr

    # Convert string level to int if needed
    if isinstance(level, str):
        numeric_level = getattr(logging, level.upper(), None)
        if numeric_level is None:
            logging.warning("Invalid log level '%s', defaulting to WARNING", level)
            numeric_level = logging.WARNING
        level = numeric_level

    # Get the root snakesee logger
    logger = logging.getLogger("snakesee")
    logger.setLevel(level)

    # Remove existing handlers
    logger.handlers.clear()

    # Create handler
    handler = logging.StreamHandler(stream)
    handler.setLevel(level)

    # Set formatter
    if json_output:
        handler.setFormatter(StructuredFormatter())
    else:
        fmt_parts = []
        if include_timestamp:
            fmt_parts.append("%(asctime)s")
        fmt_parts.extend(["%(levelname)s", "%(name)s", "%(message)s"])
        fmt = " - ".join(fmt_parts)

        use_color = hasattr(stream, "isatty") and stream.isatty()
        handler.setFormatter(ColoredFormatter(fmt, use_color=use_color))

    logger.addHandler(handler)

    # Don't propagate to root logger
    logger.propagate = False


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a snakesee module.

    Args:
        name: Module name (will be prefixed with 'snakesee.').

    Returns:
        Logger instance.

    Example:
        logger = get_logger(__name__)
        logger.info("Processing file", extra={"file": path})
    """
    if not name.startswith("snakesee"):
        name = f"snakesee.{name}"
    return logging.getLogger(name)
