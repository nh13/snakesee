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


# Default global clock instance
_default_clock: Clock = SystemClock()


def get_clock() -> Clock:
    """Get the current default clock.

    Returns:
        The currently configured clock instance.
    """
    return _default_clock


def set_clock(clock: Clock) -> None:
    """Set the default clock (primarily for testing).

    Args:
        clock: Clock instance to use as default.
    """
    global _default_clock
    _default_clock = clock


def reset_clock() -> None:
    """Reset the default clock to SystemClock."""
    global _default_clock
    _default_clock = SystemClock()
