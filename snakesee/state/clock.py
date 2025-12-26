"""Injectable clock for testable time handling.

This module provides a Clock protocol and implementations that allow
time-dependent code to be tested deterministically.

Example usage:
    # Production code
    from snakesee.state import get_clock

    def calculate_elapsed(start_time: float) -> float:
        return get_clock().now() - start_time

    # Test code
    from snakesee.state import FrozenClock, set_clock

    def test_elapsed():
        clock = FrozenClock(1000.0)
        set_clock(clock)

        assert calculate_elapsed(900.0) == 100.0

        clock.advance(50.0)
        assert calculate_elapsed(900.0) == 150.0
"""

import threading
import time as _time
from typing import Protocol


class Clock(Protocol):
    """Protocol for injectable time sources.

    This enables deterministic testing by allowing tests to provide
    a controlled time source instead of using real wall-clock time.
    """

    def now(self) -> float:
        """Return current time as Unix timestamp (seconds since epoch)."""
        ...

    def monotonic(self) -> float:
        """Return monotonic clock value for measuring durations.

        This is not affected by system clock adjustments and is suitable
        for measuring elapsed time.
        """
        ...


class SystemClock:
    """Default clock implementation using system time.

    This is the production implementation that delegates to the
    standard library's time module.
    """

    def now(self) -> float:
        """Return current time as Unix timestamp."""
        return _time.time()

    def monotonic(self) -> float:
        """Return monotonic clock value."""
        return _time.monotonic()


class FrozenClock:
    """Clock frozen at a specific time for testing.

    Useful for testing time-dependent logic without flakiness.

    Attributes:
        frozen_time: The frozen Unix timestamp.
        frozen_monotonic: The frozen monotonic value.

    Example:
        clock = FrozenClock(1700000000.0)
        assert clock.now() == 1700000000.0

        clock.advance(60.0)  # Advance by 1 minute
        assert clock.now() == 1700000060.0
    """

    def __init__(
        self,
        frozen_time: float | None = None,
        frozen_monotonic: float | None = None,
    ) -> None:
        """Initialize with specific frozen times.

        Args:
            frozen_time: Unix timestamp to freeze at. Defaults to current time.
            frozen_monotonic: Monotonic value to freeze at. Defaults to 0.0.
        """
        self._time = frozen_time if frozen_time is not None else _time.time()
        self._monotonic = frozen_monotonic if frozen_monotonic is not None else 0.0

    def now(self) -> float:
        """Return the frozen time."""
        return self._time

    def monotonic(self) -> float:
        """Return the frozen monotonic value."""
        return self._monotonic

    def advance(self, seconds: float) -> None:
        """Advance the frozen time by the given number of seconds.

        Args:
            seconds: Number of seconds to advance (can be negative).
        """
        self._time += seconds
        self._monotonic += seconds

    def set_time(self, timestamp: float) -> None:
        """Set the frozen time to a specific timestamp.

        Args:
            timestamp: Unix timestamp to set.
        """
        self._time = timestamp

    def set_monotonic(self, value: float) -> None:
        """Set the frozen monotonic value.

        Args:
            value: Monotonic value to set.
        """
        self._monotonic = value


class OffsetClock:
    """Clock with a fixed offset from system time.

    Useful for simulating time shifts without fully freezing time.

    Attributes:
        offset: Seconds to add to system time.
    """

    def __init__(self, offset: float = 0.0) -> None:
        """Initialize with an offset.

        Args:
            offset: Seconds to add to current time (negative for past).
        """
        self._offset = offset

    def now(self) -> float:
        """Return system time plus offset."""
        return _time.time() + self._offset

    def monotonic(self) -> float:
        """Return system monotonic (offset doesn't apply to durations)."""
        return _time.monotonic()


# Default global clock instance with thread-safe access
_clock_lock = threading.RLock()
_default_clock: Clock = SystemClock()


def get_clock() -> Clock:
    """Get the current default clock.

    Returns:
        The currently configured clock instance.
    """
    with _clock_lock:
        return _default_clock


def set_clock(clock: Clock) -> None:
    """Set the default clock (primarily for testing).

    Args:
        clock: Clock instance to use as default.
    """
    global _default_clock
    with _clock_lock:
        _default_clock = clock


def reset_clock() -> None:
    """Reset the default clock to SystemClock."""
    global _default_clock
    with _clock_lock:
        _default_clock = SystemClock()


class DurationResult:
    """Result of a duration calculation with validation status.

    This class provides a way to handle duration calculations that may
    encounter clock skew or other issues without raising exceptions.

    Attributes:
        value: The calculated duration (>= 0, clamped if negative).
        raw_value: The original calculated value before clamping.
        is_valid: True if the duration was valid (non-negative).
        context: Optional context description.
    """

    __slots__ = ("value", "raw_value", "is_valid", "context")

    def __init__(
        self,
        value: float,
        raw_value: float,
        is_valid: bool,
        context: str | None = None,
    ) -> None:
        self.value = value
        self.raw_value = raw_value
        self.is_valid = is_valid
        self.context = context

    def __repr__(self) -> str:
        return (
            f"DurationResult(value={self.value}, raw_value={self.raw_value}, "
            f"is_valid={self.is_valid}, context={self.context!r})"
        )


class ClockUtils:
    """Centralized utilities for clock and duration handling.

    This class provides consistent handling of duration calculations,
    including validation and clock skew detection. It centralizes logic
    that was previously duplicated in JobInfo.elapsed(), JobInfo.duration(),
    and RuleTimingStats._time_weighted_mean().

    Example:
        # Get validated duration
        result = ClockUtils.calculate_duration(start_time, end_time, "job foo")
        if not result.is_valid:
            logger.warning("Clock skew detected for %s", result.context)
        return result.value

        # Or use the elapsed time helper
        elapsed = ClockUtils.elapsed_since(start_time, context="job bar")
    """

    @staticmethod
    def calculate_duration(
        start_time: float,
        end_time: float,
        context: str | None = None,
    ) -> DurationResult:
        """Calculate duration between two timestamps with validation.

        This is the primary method for calculating durations. It handles
        clock skew by clamping negative durations to zero.

        Args:
            start_time: Start timestamp (Unix epoch seconds).
            end_time: End timestamp (Unix epoch seconds).
            context: Optional description for logging (e.g., "job foo").

        Returns:
            DurationResult with validated duration and status.
        """
        raw_duration = end_time - start_time
        is_valid = raw_duration >= 0
        clamped = max(0.0, raw_duration)
        return DurationResult(
            value=clamped,
            raw_value=raw_duration,
            is_valid=is_valid,
            context=context,
        )

    @staticmethod
    def elapsed_since(
        start_time: float,
        context: str | None = None,
        clock: Clock | None = None,
    ) -> DurationResult:
        """Calculate elapsed time since a start timestamp.

        Args:
            start_time: Start timestamp (Unix epoch seconds).
            context: Optional description for logging.
            clock: Optional clock instance (defaults to global clock).

        Returns:
            DurationResult with validated elapsed time and status.
        """
        if clock is None:
            clock = get_clock()
        now = clock.now()
        return ClockUtils.calculate_duration(start_time, now, context)

    @staticmethod
    def validate_duration(
        duration: float,
        context: str | None = None,
    ) -> DurationResult:
        """Validate a pre-calculated duration value.

        Args:
            duration: Duration value in seconds.
            context: Optional description for logging.

        Returns:
            DurationResult with validated duration and status.
        """
        is_valid = duration >= 0
        clamped = max(0.0, duration)
        return DurationResult(
            value=clamped,
            raw_value=duration,
            is_valid=is_valid,
            context=context,
        )

    @staticmethod
    def age_seconds(
        timestamp: float,
        clock: Clock | None = None,
    ) -> float:
        """Calculate age in seconds (time since timestamp).

        This is a convenience method for calculating how old a timestamp is.
        The result is always non-negative.

        Args:
            timestamp: Unix timestamp to calculate age for.
            clock: Optional clock instance (defaults to global clock).

        Returns:
            Age in seconds (>= 0).
        """
        if clock is None:
            clock = get_clock()
        return max(0.0, clock.now() - timestamp)
