"""Tests for the Clock protocol and implementations."""

import time
from collections.abc import Generator

import pytest

from snakesee.state.clock import FrozenClock
from snakesee.state.clock import OffsetClock
from snakesee.state.clock import SystemClock
from snakesee.state.clock import get_clock
from snakesee.state.clock import reset_clock
from snakesee.state.clock import set_clock


class TestSystemClock:
    """Tests for SystemClock."""

    def test_now_returns_current_time(self) -> None:
        """Test that now() returns approximately current time."""
        clock = SystemClock()
        before = time.time()
        result = clock.now()
        after = time.time()
        assert before <= result <= after

    def test_monotonic_returns_value(self) -> None:
        """Test that monotonic() returns a value."""
        clock = SystemClock()
        result = clock.monotonic()
        assert result >= 0


class TestFrozenClock:
    """Tests for FrozenClock."""

    def test_frozen_at_specific_time(self) -> None:
        """Test clock frozen at a specific time."""
        clock = FrozenClock(1700000000.0)
        assert clock.now() == 1700000000.0

    def test_frozen_at_current_time_by_default(self) -> None:
        """Test clock freezes at current time if not specified."""
        before = time.time()
        clock = FrozenClock()
        after = time.time()
        assert before <= clock.now() <= after

    def test_monotonic_default_is_zero(self) -> None:
        """Test monotonic defaults to 0."""
        clock = FrozenClock(1700000000.0)
        assert clock.monotonic() == 0.0

    def test_monotonic_custom_value(self) -> None:
        """Test monotonic can be set to custom value."""
        clock = FrozenClock(frozen_time=1700000000.0, frozen_monotonic=100.0)
        assert clock.monotonic() == 100.0

    def test_advance_time(self) -> None:
        """Test advancing frozen time."""
        clock = FrozenClock(1000.0)
        clock.advance(60.0)
        assert clock.now() == 1060.0

    def test_advance_negative(self) -> None:
        """Test advancing time by negative amount."""
        clock = FrozenClock(1000.0)
        clock.advance(-30.0)
        assert clock.now() == 970.0

    def test_advance_also_advances_monotonic(self) -> None:
        """Test that advance also advances monotonic time."""
        clock = FrozenClock(frozen_time=1000.0, frozen_monotonic=50.0)
        clock.advance(25.0)
        assert clock.monotonic() == 75.0

    def test_set_time(self) -> None:
        """Test setting time to a specific value."""
        clock = FrozenClock(1000.0)
        clock.set_time(2000.0)
        assert clock.now() == 2000.0

    def test_set_monotonic(self) -> None:
        """Test setting monotonic to a specific value."""
        clock = FrozenClock()
        clock.set_monotonic(500.0)
        assert clock.monotonic() == 500.0


class TestOffsetClock:
    """Tests for OffsetClock."""

    def test_zero_offset(self) -> None:
        """Test clock with zero offset returns current time."""
        clock = OffsetClock(0.0)
        before = time.time()
        result = clock.now()
        after = time.time()
        assert before <= result <= after

    def test_positive_offset(self) -> None:
        """Test clock with positive offset returns future time."""
        clock = OffsetClock(100.0)
        current = time.time()
        result = clock.now()
        # Should be approximately 100 seconds ahead
        assert result > current + 99
        assert result < current + 101

    def test_negative_offset(self) -> None:
        """Test clock with negative offset returns past time."""
        clock = OffsetClock(-100.0)
        current = time.time()
        result = clock.now()
        # Should be approximately 100 seconds behind
        assert result < current - 99
        assert result > current - 101

    def test_monotonic_not_affected_by_offset(self) -> None:
        """Test that monotonic is not affected by offset."""
        clock = OffsetClock(1000.0)
        before = time.monotonic()
        result = clock.monotonic()
        after = time.monotonic()
        assert before <= result <= after


class TestGlobalClock:
    """Tests for global clock functions."""

    def test_default_clock_is_system_clock(self) -> None:
        """Test that default clock is SystemClock."""
        reset_clock()
        clock = get_clock()
        assert isinstance(clock, SystemClock)

    def test_set_clock(self) -> None:
        """Test setting a custom clock."""
        frozen = FrozenClock(12345.0)
        set_clock(frozen)
        assert get_clock().now() == 12345.0
        reset_clock()

    def test_reset_clock(self) -> None:
        """Test resetting clock to default."""
        set_clock(FrozenClock(12345.0))
        reset_clock()
        assert isinstance(get_clock(), SystemClock)


@pytest.fixture(autouse=True)
def cleanup_clock() -> Generator[None, None, None]:
    """Reset clock after each test."""
    yield
    reset_clock()
