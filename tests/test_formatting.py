"""Tests for the formatting module."""

from snakesee.formatting import format_count
from snakesee.formatting import format_count_compact
from snakesee.formatting import format_duration
from snakesee.formatting import format_duration_range
from snakesee.formatting import format_eta
from snakesee.formatting import format_percentage
from snakesee.formatting import format_size
from snakesee.formatting import format_wildcards
from snakesee.formatting import get_status_color
from snakesee.formatting import get_status_style


class TestFormatDuration:
    """Tests for format_duration function."""

    def test_seconds_only(self) -> None:
        """Test formatting seconds only."""
        assert format_duration(0) == "0s"
        assert format_duration(45) == "45s"
        assert format_duration(59) == "59s"

    def test_minutes_and_seconds(self) -> None:
        """Test formatting minutes and seconds."""
        assert format_duration(60) == "1m"
        assert format_duration(90) == "1m 30s"
        assert format_duration(125) == "2m 5s"
        assert format_duration(3599) == "59m 59s"

    def test_hours_and_minutes(self) -> None:
        """Test formatting hours and minutes."""
        assert format_duration(3600) == "1h"
        assert format_duration(3725) == "1h 2m"
        assert format_duration(7200) == "2h"
        assert format_duration(7260) == "2h 1m"

    def test_infinity(self) -> None:
        """Test formatting infinity."""
        assert format_duration(float("inf")) == "unknown"

    def test_negative(self) -> None:
        """Test formatting negative values."""
        assert format_duration(-10) == "0s"

    def test_precision_seconds(self) -> None:
        """Test precision='seconds' always includes seconds."""
        assert format_duration(45, precision="seconds") == "45s"
        assert format_duration(125, precision="seconds") == "2m 5s"
        assert format_duration(3725, precision="seconds") == "1h 2m 5s"
        assert format_duration(3720, precision="seconds") == "1h 2m"

    def test_precision_minutes(self) -> None:
        """Test precision='minutes' drops seconds for larger values."""
        assert format_duration(45, precision="minutes") == "45s"
        assert format_duration(125, precision="minutes") == "2m"
        assert format_duration(3725, precision="minutes") == "1h 2m"

    def test_precision_hours(self) -> None:
        """Test precision='hours' shows decimal hours."""
        assert format_duration(3600, precision="hours") == "1.0h"
        assert format_duration(5400, precision="hours") == "1.5h"
        assert format_duration(1800, precision="hours") == "30m"


class TestFormatDurationRange:
    """Tests for format_duration_range function."""

    def test_basic_range(self) -> None:
        """Test basic range formatting."""
        assert format_duration_range(60, 180) == "1m - 3m"

    def test_custom_separator(self) -> None:
        """Test custom separator."""
        assert format_duration_range(60, 180, " to ") == "1m to 3m"

    def test_negative_lower_clamped(self) -> None:
        """Test negative lower bound is clamped to 0."""
        assert format_duration_range(-10, 60) == "0s - 1m"


class TestFormatSize:
    """Tests for format_size function."""

    def test_bytes(self) -> None:
        """Test formatting bytes."""
        assert format_size(0) == "0 B"
        assert format_size(512) == "512 B"
        assert format_size(1023) == "1023 B"

    def test_kilobytes(self) -> None:
        """Test formatting kilobytes."""
        assert format_size(1024) == "1 KB"  # Exact integer shows without decimal
        assert format_size(1536) == "1.5 KB"

    def test_megabytes(self) -> None:
        """Test formatting megabytes."""
        assert format_size(1024 * 1024) == "1 MB"  # Exact integer shows without decimal

    def test_gigabytes(self) -> None:
        """Test formatting gigabytes."""
        assert format_size(1024 * 1024 * 1024) == "1 GB"  # Exact integer shows without decimal

    def test_precision(self) -> None:
        """Test custom precision."""
        assert format_size(1536, precision=2) == "1.50 KB"
        assert format_size(1536, precision=0) == "2 KB"

    def test_negative(self) -> None:
        """Test negative value returns 0 B."""
        assert format_size(-100) == "0 B"


class TestFormatPercentage:
    """Tests for format_percentage function."""

    def test_basic(self) -> None:
        """Test basic percentage formatting."""
        assert format_percentage(0.5) == "50.0%"
        assert format_percentage(0.756) == "75.6%"
        assert format_percentage(1.0) == "100.0%"

    def test_precision(self) -> None:
        """Test custom precision."""
        assert format_percentage(0.756, precision=0) == "76%"
        assert format_percentage(0.756, precision=2) == "75.60%"

    def test_without_symbol(self) -> None:
        """Test without percent symbol."""
        assert format_percentage(0.5, include_symbol=False) == "50.0"

    def test_handles_0_to_100_range(self) -> None:
        """Test that values > 1 are treated as percentages."""
        assert format_percentage(50.0) == "50.0%"


class TestFormatCount:
    """Tests for format_count function."""

    def test_basic(self) -> None:
        """Test basic count formatting."""
        assert format_count(5, 10) == "5 of 10 (50.0%)"
        assert format_count(0, 10) == "0 of 10 (0.0%)"
        assert format_count(10, 10) == "10 of 10 (100.0%)"

    def test_without_percentage(self) -> None:
        """Test without percentage."""
        assert format_count(5, 10, include_percentage=False) == "5 of 10"

    def test_zero_total(self) -> None:
        """Test with zero total."""
        assert format_count(0, 0, include_percentage=True) == "0 of 0"


class TestFormatCountCompact:
    """Tests for format_count_compact function."""

    def test_basic(self) -> None:
        """Test compact count formatting."""
        assert format_count_compact(5, 10) == "5/10"
        assert format_count_compact(0, 0) == "0/0"


class TestFormatEta:
    """Tests for format_eta function."""

    def test_high_confidence(self) -> None:
        """Test high confidence shows simple estimate."""
        assert format_eta(300, confidence=0.9) == "~5m"
        assert format_eta(3600, confidence=0.8) == "~1h"

    def test_medium_confidence_with_range(self) -> None:
        """Test medium confidence shows range when available."""
        result = format_eta(300, lower_bound=180, upper_bound=420, confidence=0.5)
        assert result == "3m - 7m"

    def test_low_confidence(self) -> None:
        """Test low confidence shows rough estimate."""
        assert format_eta(300, confidence=0.2) == "~5m (rough)"

    def test_very_low_confidence(self) -> None:
        """Test very low confidence shows very rough estimate."""
        assert format_eta(300, confidence=0.05) == "~5m (very rough)"

    def test_infinity(self) -> None:
        """Test infinity returns unknown."""
        assert format_eta(float("inf")) == "unknown"


class TestFormatWildcards:
    """Tests for format_wildcards function."""

    def test_basic(self) -> None:
        """Test basic wildcard formatting."""
        assert format_wildcards({"sample": "A"}) == "sample=A"
        assert format_wildcards({"sample": "A", "batch": "1"}) == "sample=A, batch=1"

    def test_none(self) -> None:
        """Test None wildcards returns empty string."""
        assert format_wildcards(None) == ""

    def test_empty(self) -> None:
        """Test empty wildcards returns empty string."""
        assert format_wildcards({}) == ""

    def test_custom_separator(self) -> None:
        """Test custom separator."""
        assert format_wildcards({"a": "1", "b": "2"}, separator="; ") == "a=1; b=2"

    def test_max_length(self) -> None:
        """Test max_length truncation."""
        result = format_wildcards({"sample": "very_long_value"}, max_length=10)
        assert len(result) <= 10
        assert result.endswith("...")


class TestGetStatusStyle:
    """Tests for get_status_style function."""

    def test_running(self) -> None:
        """Test running status style."""
        style = get_status_style("running")
        assert style.color == "green"

    def test_completed(self) -> None:
        """Test completed status style."""
        style = get_status_style("completed")
        assert style.color == "blue"

    def test_failed(self) -> None:
        """Test failed status style."""
        style = get_status_style("failed")
        assert style.color == "red"
        assert style.bold is True

    def test_case_insensitive(self) -> None:
        """Test case insensitivity."""
        assert get_status_style("RUNNING").color == "green"
        assert get_status_style("Running").color == "green"

    def test_unknown_returns_default(self) -> None:
        """Test unknown status returns default style."""
        style = get_status_style("nonexistent")
        assert style.color == "yellow"  # unknown color


class TestGetStatusColor:
    """Tests for get_status_color function."""

    def test_returns_color_string(self) -> None:
        """Test that it returns just the color string."""
        assert get_status_color("running") == "green"
        assert get_status_color("failed") == "red"
