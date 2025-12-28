"""Centralized formatting utilities for snakesee.

This module provides consistent formatting functions for durations, sizes,
percentages, and status display throughout the application.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Literal

# Time conversion constants
SECONDS_PER_MINUTE: int = 60
SECONDS_PER_HOUR: int = 3600


class StatusColor(str, Enum):
    """Standard colors for workflow/job status display."""

    RUNNING = "green"
    COMPLETED = "blue"
    FAILED = "red"
    INCOMPLETE = "yellow"
    UNKNOWN = "yellow"
    PENDING = "dim"
    ERROR = "red"


@dataclass(frozen=True)
class StatusStyle:
    """Complete styling for a status indicator."""

    color: str
    icon: str = ""
    bold: bool = False

    @property
    def rich_style(self) -> str:
        """Return Rich markup style string."""
        style = self.color
        if self.bold:
            style = f"bold {style}"
        return style


# Standard status styles
STATUS_STYLES: dict[str, StatusStyle] = {
    "running": StatusStyle(color="green", icon="[yellow]~[/]", bold=False),
    "completed": StatusStyle(color="blue", icon="[green]v[/]", bold=False),
    "failed": StatusStyle(color="red", icon="[red]x[/]", bold=True),
    "error": StatusStyle(color="red", icon="[red]x[/]", bold=True),
    "incomplete": StatusStyle(color="yellow", icon="[yellow]![/]", bold=False),
    "unknown": StatusStyle(color="yellow", icon="[yellow]?[/]", bold=False),
    "pending": StatusStyle(color="dim", icon="[dim]o[/]", bold=False),
}


def get_status_style(status: str) -> StatusStyle:
    """Get the style for a given status.

    Args:
        status: Status string (case-insensitive).

    Returns:
        StatusStyle for the status, or default unknown style.
    """
    return STATUS_STYLES.get(status.lower(), STATUS_STYLES["unknown"])


def get_status_color(status: str) -> str:
    """Get just the color for a status (backward compatibility).

    Args:
        status: Status string (case-insensitive).

    Returns:
        Color name for Rich markup.
    """
    return get_status_style(status).color


# =============================================================================
# Duration Formatting
# =============================================================================


def format_duration(
    seconds: float,
    precision: Literal["auto", "seconds", "minutes", "hours"] = "auto",
) -> str:
    """Format seconds as human-readable duration.

    Args:
        seconds: Duration in seconds.
        precision: Level of precision for output.
            - "auto": Choose based on magnitude
            - "seconds": Always include seconds
            - "minutes": Include minutes, drop seconds for large values
            - "hours": Only show hours for very large values

    Returns:
        Formatted duration string (e.g., "5s", "2m 30s", "1h 15m").

    Examples:
        >>> format_duration(45)
        '45s'
        >>> format_duration(125)
        '2m 5s'
        >>> format_duration(3725)
        '1h 2m'
        >>> format_duration(float("inf"))
        'unknown'
    """
    if seconds == float("inf"):
        return "unknown"
    if seconds < 0:
        return "0s"

    if precision == "auto":
        if seconds < SECONDS_PER_MINUTE:
            return f"{int(seconds)}s"
        if seconds < SECONDS_PER_HOUR:
            minutes = int(seconds // SECONDS_PER_MINUTE)
            secs = int(seconds % SECONDS_PER_MINUTE)
            return f"{minutes}m {secs}s" if secs > 0 else f"{minutes}m"
        hours = int(seconds // SECONDS_PER_HOUR)
        minutes = int((seconds % SECONDS_PER_HOUR) // SECONDS_PER_MINUTE)
        return f"{hours}h {minutes}m" if minutes > 0 else f"{hours}h"

    if precision == "seconds":
        if seconds < SECONDS_PER_MINUTE:
            return f"{int(seconds)}s"
        if seconds < SECONDS_PER_HOUR:
            minutes = int(seconds // SECONDS_PER_MINUTE)
            secs = int(seconds % SECONDS_PER_MINUTE)
            return f"{minutes}m {secs}s"
        hours = int(seconds // SECONDS_PER_HOUR)
        minutes = int((seconds % SECONDS_PER_HOUR) // SECONDS_PER_MINUTE)
        secs = int(seconds % SECONDS_PER_MINUTE)
        if secs > 0:
            return f"{hours}h {minutes}m {secs}s"
        return f"{hours}h {minutes}m" if minutes > 0 else f"{hours}h"

    if precision == "minutes":
        if seconds < SECONDS_PER_MINUTE:
            return f"{int(seconds)}s"
        minutes = int(seconds // SECONDS_PER_MINUTE)
        if seconds < SECONDS_PER_HOUR:
            return f"{minutes}m"
        hours = int(seconds // SECONDS_PER_HOUR)
        minutes = int((seconds % SECONDS_PER_HOUR) // SECONDS_PER_MINUTE)
        return f"{hours}h {minutes}m" if minutes > 0 else f"{hours}h"

    # precision == "hours"
    if seconds < SECONDS_PER_HOUR:
        return f"{int(seconds / SECONDS_PER_MINUTE)}m"
    hours_float = seconds / SECONDS_PER_HOUR
    return f"{hours_float:.1f}h"


def format_duration_range(
    lower: float,
    upper: float,
    separator: str = " - ",
) -> str:
    """Format a duration range.

    Args:
        lower: Lower bound in seconds.
        upper: Upper bound in seconds.
        separator: String between lower and upper.

    Returns:
        Formatted range string (e.g., "2m - 5m").

    Examples:
        >>> format_duration_range(60, 180)
        '1m - 3m'
        >>> format_duration_range(30, 90)
        '30s - 1m 30s'
        >>> format_duration_range(3600, 7200)
        '1h - 2h'
    """
    return f"{format_duration(max(0, lower))}{separator}{format_duration(upper)}"


# =============================================================================
# Size Formatting
# =============================================================================

_SIZE_UNITS = ["B", "KB", "MB", "GB", "TB", "PB"]


def format_size(
    bytes_: int | float,
    precision: int = 1,
    binary: bool = True,
) -> str:
    """Format bytes as human-readable size.

    Args:
        bytes_: Size in bytes.
        precision: Decimal places for non-integer results.
        binary: Use binary (1024) or SI (1000) units.

    Returns:
        Formatted size string (e.g., "1.5 GB", "256 KB").

    Examples:
        >>> format_size(1024)
        '1 KB'
        >>> format_size(1536, precision=2)
        '1.50 KB'
        >>> format_size(0)
        '0 B'
    """
    if bytes_ < 0:
        return "0 B"
    if bytes_ == 0:
        return "0 B"

    divisor = 1024.0 if binary else 1000.0

    for unit in _SIZE_UNITS[:-1]:
        if abs(bytes_) < divisor:
            if bytes_ == int(bytes_):
                return f"{int(bytes_)} {unit}"
            return f"{bytes_:.{precision}f} {unit}"
        bytes_ /= divisor

    return f"{bytes_:.{precision}f} {_SIZE_UNITS[-1]}"


def format_size_rate(
    bytes_per_second: float,
    precision: int = 1,
) -> str:
    """Format bytes per second as human-readable rate.

    Args:
        bytes_per_second: Rate in bytes per second.
        precision: Decimal places for output.

    Returns:
        Formatted rate string (e.g., "1.5 MB/s").

    Examples:
        >>> format_size_rate(1048576)
        '1 MB/s'
        >>> format_size_rate(1536)
        '1.5 KB/s'
        >>> format_size_rate(0)
        '0 B/s'
    """
    if bytes_per_second < 0:
        return "0 B/s"
    return f"{format_size(bytes_per_second, precision)}/s"


# =============================================================================
# Percentage and Count Formatting
# =============================================================================


def format_percentage(
    value: float,
    precision: int = 1,
    include_symbol: bool = True,
) -> str:
    """Format a value as a percentage.

    Args:
        value: Value between 0.0 and 1.0 (or 0-100 if > 1).
        precision: Decimal places.
        include_symbol: Whether to include the % symbol.

    Returns:
        Formatted percentage string.

    Examples:
        >>> format_percentage(0.756)
        '75.6%'
        >>> format_percentage(0.756, precision=0)
        '76%'
        >>> format_percentage(0.756, include_symbol=False)
        '75.6'
    """
    # Handle both 0-1 and 0-100 ranges
    if value <= 1.0:
        value *= 100

    value = max(0.0, min(100.0, value))

    formatted = f"{value:.{precision}f}"
    if include_symbol:
        formatted += "%"

    return formatted


def format_count(
    current: int,
    total: int,
    include_percentage: bool = True,
) -> str:
    """Format a count as "X of Y (Z%)".

    Args:
        current: Current count.
        total: Total count.
        include_percentage: Whether to include percentage.

    Returns:
        Formatted count string.

    Examples:
        >>> format_count(5, 10)
        '5 of 10 (50.0%)'
        >>> format_count(5, 10, include_percentage=False)
        '5 of 10'
    """
    if include_percentage and total > 0:
        pct = (current / total) * 100
        return f"{current} of {total} ({pct:.1f}%)"
    return f"{current} of {total}"


def format_count_compact(current: int, total: int) -> str:
    """Format a count as "X/Y".

    Args:
        current: Current count.
        total: Total count.

    Returns:
        Compact count string.

    Examples:
        >>> format_count_compact(5, 10)
        '5/10'
    """
    return f"{current}/{total}"


# =============================================================================
# ETA Formatting
# =============================================================================


def format_eta(
    seconds_remaining: float,
    lower_bound: float | None = None,
    upper_bound: float | None = None,
    confidence: float = 1.0,
) -> str:
    """Format an ETA with optional range and confidence indication.

    Args:
        seconds_remaining: Expected seconds remaining.
        lower_bound: Optimistic estimate (optional).
        upper_bound: Pessimistic estimate (optional).
        confidence: Confidence level (0.0 to 1.0).

    Returns:
        Formatted ETA string with appropriate caveats.

    Examples:
        >>> format_eta(300, confidence=0.9)
        '~5m'
        >>> format_eta(300, lower_bound=180, upper_bound=420, confidence=0.5)
        '3m - 7m'
        >>> format_eta(300, confidence=0.2)
        '~5m (rough)'
    """
    if seconds_remaining == float("inf"):
        return "unknown"

    expected_str = format_duration(seconds_remaining)

    # High confidence, narrow range: just show estimate
    if confidence > 0.7:
        if lower_bound is not None and upper_bound is not None:
            if seconds_remaining > 0:
                range_width = (upper_bound - lower_bound) / seconds_remaining
                if range_width < 0.3:
                    return f"~{expected_str}"
        else:
            return f"~{expected_str}"

    # Medium confidence: show range if available
    if confidence > 0.4 and lower_bound is not None and upper_bound is not None:
        return format_duration_range(lower_bound, upper_bound)

    # Low confidence: show with caveat
    if confidence > 0.1:
        return f"~{expected_str} (rough)"

    return f"~{expected_str} (very rough)"


# =============================================================================
# Wildcard Formatting
# =============================================================================


def format_wildcards(
    wildcards: dict[str, str] | None,
    separator: str = ", ",
    max_length: int | None = None,
) -> str:
    """Format wildcards dictionary as a string.

    Args:
        wildcards: Dictionary of wildcard key-value pairs.
        separator: String between key=value pairs.
        max_length: Maximum output length (truncate with "...").

    Returns:
        Formatted wildcards string.

    Examples:
        >>> format_wildcards({"sample": "A", "batch": "1"})
        'sample=A, batch=1'
        >>> format_wildcards(None)
        ''
    """
    if not wildcards:
        return ""

    result = separator.join(f"{k}={v}" for k, v in wildcards.items())

    if max_length is not None and len(result) > max_length:
        return result[: max_length - 3] + "..."

    return result
