"""Tests for monitor data models."""

import time
from pathlib import Path

from snakesee.models import JobInfo
from snakesee.models import RuleTimingStats
from snakesee.models import TimeEstimate
from snakesee.models import WorkflowProgress
from snakesee.models import WorkflowStatus
from snakesee.models import format_duration


class TestFormatDuration:
    """Tests for the format_duration function."""

    def test_seconds(self) -> None:
        """Test formatting seconds."""
        assert format_duration(5) == "5s"
        assert format_duration(59) == "59s"

    def test_minutes(self) -> None:
        """Test formatting minutes."""
        assert format_duration(60) == "1m"
        assert format_duration(90) == "1m 30s"
        assert format_duration(3599) == "59m 59s"

    def test_hours(self) -> None:
        """Test formatting hours."""
        assert format_duration(3600) == "1h"
        assert format_duration(3660) == "1h 1m"
        assert format_duration(7200) == "2h"

    def test_edge_cases(self) -> None:
        """Test edge cases."""
        assert format_duration(0) == "0s"
        assert format_duration(-1) == "0s"
        assert format_duration(float("inf")) == "unknown"


class TestJobInfo:
    """Tests for the JobInfo dataclass."""

    def test_elapsed_no_start(self) -> None:
        """Test elapsed time with no start time."""
        job = JobInfo(rule="test")
        assert job.elapsed is None

    def test_elapsed_running(self) -> None:
        """Test elapsed time for running job."""
        start = time.time() - 10
        job = JobInfo(rule="test", start_time=start)
        elapsed = job.elapsed
        assert elapsed is not None
        assert 9 < elapsed < 12  # Allow some tolerance

    def test_duration_complete(self) -> None:
        """Test duration for completed job."""
        job = JobInfo(rule="test", start_time=100.0, end_time=200.0)
        assert job.duration == 100.0

    def test_duration_incomplete(self) -> None:
        """Test duration for incomplete job."""
        job = JobInfo(rule="test", start_time=100.0)
        assert job.duration is None


class TestRuleTimingStats:
    """Tests for the RuleTimingStats dataclass."""

    def test_empty_stats(self) -> None:
        """Test stats with no data."""
        stats = RuleTimingStats(rule="test")
        assert stats.count == 0
        assert stats.mean_duration == 0.0
        assert stats.std_dev == 0.0

    def test_single_observation(self) -> None:
        """Test stats with single observation."""
        stats = RuleTimingStats(rule="test", durations=[10.0])
        assert stats.count == 1
        assert stats.mean_duration == 10.0
        assert stats.std_dev == 0.0  # Need at least 2 for std_dev

    def test_multiple_observations(self) -> None:
        """Test stats with multiple observations."""
        stats = RuleTimingStats(rule="test", durations=[10.0, 20.0, 30.0])
        assert stats.count == 3
        assert stats.mean_duration == 20.0
        assert stats.std_dev > 0

    def test_weighted_mean_position_based(self) -> None:
        """Test weighted mean with position-based weighting (no timestamps)."""
        stats = RuleTimingStats(rule="test", durations=[10.0, 20.0, 30.0])
        # Weights: 1, 2, 4 (exponential)
        # Weighted sum: 10*1 + 20*2 + 30*4 = 170
        # Sum of weights: 1 + 2 + 4 = 7
        # Weighted mean: 170/7 ≈ 24.29
        assert 24 < stats.weighted_mean() < 25

    def test_weighted_mean_time_based(self) -> None:
        """Test weighted mean with time-based weighting using timestamps."""
        now = time.time()
        day = 86400
        # Recent run (0 days ago) should have highest weight
        # Old run (14 days ago) should have lowest weight
        stats = RuleTimingStats(
            rule="test",
            durations=[10.0, 30.0],  # old: 10s, recent: 30s
            timestamps=[now - 14 * day, now],  # 14 days ago, now
        )
        # With 7-day half-life, 14-day old data has weight ~0.25, today's has weight ~1.0
        # Should be closer to 30 than to 20 (the simple mean)
        weighted = stats.weighted_mean(half_life_days=7.0)
        assert weighted > 20.0  # More than simple mean
        assert weighted > 25.0  # Significantly weighted toward recent

    def test_weighted_mean_custom_half_life(self) -> None:
        """Test weighted mean respects custom half-life parameter."""
        now = time.time()
        day = 86400
        stats = RuleTimingStats(
            rule="test",
            durations=[10.0, 30.0],  # old: 10s, recent: 30s
            timestamps=[now - 7 * day, now],  # 7 days ago, now
        )
        # With 7-day half-life, 7-day old data has weight 0.5, today's has weight 1.0
        # Weighted mean = (10 * 0.5 + 30 * 1.0) / 1.5 = 35/1.5 ≈ 23.33
        weighted_7d = stats.weighted_mean(half_life_days=7.0)
        assert 23 < weighted_7d < 24

        # With 1-day half-life, old data is heavily discounted
        weighted_1d = stats.weighted_mean(half_life_days=1.0)
        assert weighted_1d > weighted_7d  # Even more weighted toward recent

        # With 30-day half-life, old data retains more weight
        weighted_30d = stats.weighted_mean(half_life_days=30.0)
        assert weighted_30d < weighted_7d  # Closer to simple mean

    def test_weighted_mean_falls_back_without_timestamps(self) -> None:
        """Test weighted mean falls back to position-based when timestamps missing."""
        stats = RuleTimingStats(
            rule="test",
            durations=[10.0, 20.0, 30.0],
            timestamps=[],  # No timestamps
        )
        # Should use position-based weighting
        weighted = stats.weighted_mean()
        assert 24 < weighted < 25  # Same as position-based test

    def test_high_variance(self) -> None:
        """Test high variance detection."""
        low_var = RuleTimingStats(rule="test", durations=[10.0, 10.5, 10.2])
        high_var = RuleTimingStats(rule="test", durations=[10.0, 50.0, 100.0])

        assert not low_var.is_high_variance
        assert high_var.is_high_variance

    def test_recency_factor_no_timestamps(self) -> None:
        """Test recency factor returns 0.5 when no timestamps."""
        stats = RuleTimingStats(rule="test", durations=[10.0, 20.0])
        assert stats.recency_factor() == 0.5

    def test_recency_factor_recent_data(self) -> None:
        """Test recency factor is high for recent data."""
        now = time.time()
        stats = RuleTimingStats(
            rule="test",
            durations=[10.0, 20.0],
            timestamps=[now - 3600, now],  # 1 hour ago, now
        )
        assert stats.recency_factor() > 0.9  # Very recent

    def test_recency_factor_old_data(self) -> None:
        """Test recency factor is low for old data."""
        now = time.time()
        day = 86400
        stats = RuleTimingStats(
            rule="test",
            durations=[10.0, 20.0],
            timestamps=[now - 30 * day, now - 28 * day],  # 30 and 28 days ago
        )
        assert stats.recency_factor() < 0.3  # Old data

    def test_recent_consistency_no_data(self) -> None:
        """Test recent consistency returns 0.5 when no data."""
        stats = RuleTimingStats(rule="test", durations=[])
        assert stats.recent_consistency() == 0.5

    def test_recent_consistency_consistent(self) -> None:
        """Test recent consistency is high for consistent runs."""
        now = time.time()
        stats = RuleTimingStats(
            rule="test",
            durations=[10.0, 10.1, 9.9, 10.0],
            timestamps=[now - 3600, now - 2400, now - 1200, now],
        )
        assert stats.recent_consistency() > 0.9  # Very consistent

    def test_recent_consistency_variable(self) -> None:
        """Test recent consistency is lower for variable runs."""
        now = time.time()
        stats = RuleTimingStats(
            rule="test",
            durations=[5.0, 20.0, 10.0, 30.0],  # High variance
            timestamps=[now - 3600, now - 2400, now - 1200, now],
        )
        assert stats.recent_consistency() < 0.7  # Variable data


class TestWildcardTimingStats:
    """Tests for the WildcardTimingStats class."""

    def test_init(self) -> None:
        """Test basic initialization."""
        from snakesee.models import WildcardTimingStats

        wts = WildcardTimingStats(rule="align", wildcard_key="sample")
        assert wts.rule == "align"
        assert wts.wildcard_key == "sample"
        assert wts.stats_by_value == {}

    def test_get_stats_for_value_not_found(self) -> None:
        """Test get_stats_for_value returns None for unknown value."""
        from snakesee.models import WildcardTimingStats

        wts = WildcardTimingStats(rule="align", wildcard_key="sample")
        assert wts.get_stats_for_value("unknown") is None

    def test_get_stats_for_value_insufficient_samples(self) -> None:
        """Test get_stats_for_value returns None for insufficient samples."""
        from snakesee.models import WildcardTimingStats

        wts = WildcardTimingStats(
            rule="align",
            wildcard_key="sample",
            stats_by_value={
                "A": RuleTimingStats(rule="align:sample=A", durations=[100.0, 110.0]),
            },
        )
        # Only 2 samples, need at least 3
        assert wts.get_stats_for_value("A") is None

    def test_get_stats_for_value_sufficient_samples(self) -> None:
        """Test get_stats_for_value returns stats when sufficient samples."""
        from snakesee.models import WildcardTimingStats

        wts = WildcardTimingStats(
            rule="align",
            wildcard_key="sample",
            stats_by_value={
                "A": RuleTimingStats(rule="align:sample=A", durations=[100.0, 110.0, 105.0]),
            },
        )
        stats = wts.get_stats_for_value("A")
        assert stats is not None
        assert stats.count == 3

    def test_get_most_predictive_key_empty(self) -> None:
        """Test get_most_predictive_key with no stats."""
        from snakesee.models import WildcardTimingStats

        result = WildcardTimingStats.get_most_predictive_key({})
        assert result is None

    def test_get_most_predictive_key_single_value(self) -> None:
        """Test get_most_predictive_key with only one value per key."""
        from snakesee.models import WildcardTimingStats

        wts = WildcardTimingStats(
            rule="align",
            wildcard_key="sample",
            stats_by_value={
                "A": RuleTimingStats(rule="align:sample=A", durations=[100.0]),
            },
        )
        # Need at least 2 values to compare
        result = WildcardTimingStats.get_most_predictive_key({"sample": wts})
        assert result is None

    def test_get_most_predictive_key_high_variance(self) -> None:
        """Test get_most_predictive_key identifies key with high variance."""
        from snakesee.models import WildcardTimingStats

        # sample key: A=100, B=500 (high between-group variance)
        sample_wts = WildcardTimingStats(
            rule="align",
            wildcard_key="sample",
            stats_by_value={
                "A": RuleTimingStats(rule="align:sample=A", durations=[100.0]),
                "B": RuleTimingStats(rule="align:sample=B", durations=[500.0]),
            },
        )

        # batch key: 1=100, 2=110 (low between-group variance)
        batch_wts = WildcardTimingStats(
            rule="align",
            wildcard_key="batch",
            stats_by_value={
                "1": RuleTimingStats(rule="align:batch=1", durations=[100.0]),
                "2": RuleTimingStats(rule="align:batch=2", durations=[110.0]),
            },
        )

        result = WildcardTimingStats.get_most_predictive_key(
            {
                "sample": sample_wts,
                "batch": batch_wts,
            }
        )
        assert result == "sample"  # Higher variance between A and B


class TestTimeEstimate:
    """Tests for the TimeEstimate dataclass."""

    def test_format_eta_high_confidence(self) -> None:
        """Test ETA formatting with high confidence."""
        estimate = TimeEstimate(
            seconds_remaining=300,
            lower_bound=280,
            upper_bound=320,
            confidence=0.8,
            method="weighted",
        )
        assert estimate.format_eta() == "~5m"

    def test_format_eta_medium_confidence(self) -> None:
        """Test ETA formatting with medium confidence."""
        estimate = TimeEstimate(
            seconds_remaining=300,
            lower_bound=200,
            upper_bound=400,
            confidence=0.5,
            method="weighted",
        )
        eta = estimate.format_eta()
        assert "-" in eta  # Shows range

    def test_format_eta_low_confidence(self) -> None:
        """Test ETA formatting with low confidence."""
        estimate = TimeEstimate(
            seconds_remaining=300,
            lower_bound=100,
            upper_bound=600,
            confidence=0.2,
            method="simple",
        )
        assert "rough" in estimate.format_eta()

    def test_format_eta_unknown(self) -> None:
        """Test ETA formatting when unknown."""
        estimate = TimeEstimate(
            seconds_remaining=float("inf"),
            lower_bound=0,
            upper_bound=float("inf"),
            confidence=0.0,
            method="bootstrap",
        )
        assert estimate.format_eta() == "unknown"


class TestWorkflowProgress:
    """Tests for the WorkflowProgress dataclass."""

    def test_percent_complete(self) -> None:
        """Test percent complete calculation."""
        progress = WorkflowProgress(
            workflow_dir=Path("."),
            status=WorkflowStatus.RUNNING,
            total_jobs=100,
            completed_jobs=25,
        )
        assert progress.percent_complete == 25.0

    def test_percent_complete_zero_total(self) -> None:
        """Test percent complete with zero total."""
        progress = WorkflowProgress(
            workflow_dir=Path("."),
            status=WorkflowStatus.UNKNOWN,
            total_jobs=0,
            completed_jobs=0,
        )
        assert progress.percent_complete == 0.0

    def test_pending_jobs(self) -> None:
        """Test pending jobs calculation."""
        progress = WorkflowProgress(
            workflow_dir=Path("."),
            status=WorkflowStatus.RUNNING,
            total_jobs=100,
            completed_jobs=25,
            running_jobs=[JobInfo(rule="test")] * 5,
        )
        assert progress.pending_jobs == 70  # 100 - 25 - 5

    def test_elapsed_seconds(self) -> None:
        """Test elapsed seconds calculation."""
        start = time.time() - 60
        progress = WorkflowProgress(
            workflow_dir=Path("."),
            status=WorkflowStatus.RUNNING,
            total_jobs=100,
            completed_jobs=25,
            start_time=start,
        )
        elapsed = progress.elapsed_seconds
        assert elapsed is not None
        assert 59 < elapsed < 62
