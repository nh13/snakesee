"""Tests for the time estimator."""

from pathlib import Path

import pytest

from snakesee.estimator import TimeEstimator
from snakesee.models import JobInfo
from snakesee.models import RuleTimingStats
from snakesee.models import WorkflowProgress
from snakesee.models import WorkflowStatus


class TestTimeEstimator:
    """Tests for the TimeEstimator class."""

    def test_init_empty(self) -> None:
        """Test initialization with no data."""
        estimator = TimeEstimator()
        assert estimator.rule_stats == {}

    def test_init_with_stats(self) -> None:
        """Test initialization with pre-loaded stats."""
        stats = {"align": RuleTimingStats(rule="align", durations=[100.0])}
        estimator = TimeEstimator(rule_stats=stats)
        assert "align" in estimator.rule_stats

    def test_load_from_metadata(self, metadata_dir: Path) -> None:
        """Test loading stats from metadata directory."""
        estimator = TimeEstimator()
        estimator.load_from_metadata(metadata_dir)

        assert "align" in estimator.rule_stats
        assert estimator.rule_stats["align"].count == 5
        assert estimator.rule_stats["align"].mean_duration == 100.0

        assert "sort" in estimator.rule_stats
        assert estimator.rule_stats["sort"].count == 3
        assert estimator.rule_stats["sort"].mean_duration == 50.0

    def test_global_mean_empty(self) -> None:
        """Test global mean with no data returns default."""
        estimator = TimeEstimator()
        assert estimator.global_mean_duration() == 60.0

    def test_global_mean_with_data(self, metadata_dir: Path) -> None:
        """Test global mean calculation."""
        estimator = TimeEstimator()
        estimator.load_from_metadata(metadata_dir)

        # (5 * 100 + 3 * 50) / 8 = 81.25
        assert estimator.global_mean_duration() == pytest.approx(81.25)

    def test_estimate_no_jobs(self) -> None:
        """Test estimation with no jobs to do."""
        estimator = TimeEstimator()
        progress = WorkflowProgress(
            workflow_dir=Path("."),
            status=WorkflowStatus.COMPLETED,
            total_jobs=0,
            completed_jobs=0,
        )

        estimate = estimator.estimate_remaining(progress)
        assert estimate.seconds_remaining == 0.0
        assert estimate.confidence == 1.0
        assert estimate.method == "complete"

    def test_estimate_already_complete(self) -> None:
        """Test estimation when already complete."""
        estimator = TimeEstimator()
        progress = WorkflowProgress(
            workflow_dir=Path("."),
            status=WorkflowStatus.COMPLETED,
            total_jobs=10,
            completed_jobs=10,
        )

        estimate = estimator.estimate_remaining(progress)
        assert estimate.seconds_remaining == 0.0
        assert estimate.method == "complete"

    def test_estimate_no_progress(self) -> None:
        """Test estimation with no completed jobs."""
        estimator = TimeEstimator()
        progress = WorkflowProgress(
            workflow_dir=Path("."),
            status=WorkflowStatus.RUNNING,
            total_jobs=10,
            completed_jobs=0,
            running_jobs=[JobInfo(rule="test")],
        )

        estimate = estimator.estimate_remaining(progress)
        assert estimate.seconds_remaining > 0
        assert estimate.confidence < 0.1
        assert estimate.method == "bootstrap"

    def test_estimate_simple(self) -> None:
        """Test simple estimation fallback."""
        estimator = TimeEstimator()  # No historical data
        progress = WorkflowProgress(
            workflow_dir=Path("."),
            status=WorkflowStatus.RUNNING,
            total_jobs=10,
            completed_jobs=5,
            start_time=0.0,  # Started at epoch
        )
        # Pretend 100s have elapsed for 5 jobs = 20s per job
        # So 5 remaining jobs = 100s remaining

        estimate = estimator.estimate_remaining(progress)
        assert estimate.method == "simple"
        # Can't test exact value since it depends on current time

    def test_estimate_weighted(self, metadata_dir: Path) -> None:
        """Test weighted estimation with historical data."""
        estimator = TimeEstimator()
        estimator.load_from_metadata(metadata_dir)

        # Create progress with some completed jobs
        progress = WorkflowProgress(
            workflow_dir=Path("."),
            status=WorkflowStatus.RUNNING,
            total_jobs=10,
            completed_jobs=5,
            running_jobs=[],
            recent_completions=[
                JobInfo(rule="align", start_time=1000.0, end_time=1100.0),
                JobInfo(rule="align", start_time=1100.0, end_time=1200.0),
                JobInfo(rule="sort", start_time=1200.0, end_time=1250.0),
            ],
            start_time=1000.0,
        )

        estimate = estimator.estimate_remaining(progress)
        assert estimate.method == "weighted"
        assert estimate.seconds_remaining > 0
        assert estimate.confidence > 0

    def test_estimate_with_running_elapsed(self, metadata_dir: Path) -> None:
        """Test estimation subtracts elapsed time for running jobs."""
        estimator = TimeEstimator()
        estimator.load_from_metadata(metadata_dir)

        progress = WorkflowProgress(
            workflow_dir=Path("."),
            status=WorkflowStatus.RUNNING,
            total_jobs=10,
            completed_jobs=8,
            running_jobs=[JobInfo(rule="align")],
            recent_completions=[
                JobInfo(rule="align", start_time=1000.0, end_time=1100.0),
            ],
            start_time=1000.0,
        )

        # With elapsed time for the running job
        running_elapsed = {"align": 50.0}  # 50s of 100s expected elapsed
        estimate = estimator.estimate_remaining(progress, running_elapsed)

        # Should account for the 50s already elapsed
        assert estimate.seconds_remaining > 0

    def test_infer_pending_rules(self) -> None:
        """Test pending rule inference."""
        estimator = TimeEstimator()

        # 3 align, 2 sort completed -> 60% align, 40% sort
        completed = {"align": 3, "sort": 2}
        pending = estimator._infer_pending_rules(completed, 10)

        # Should maintain roughly same proportion
        assert pending["align"] >= pending["sort"]
        assert sum(pending.values()) <= 10

    def test_infer_pending_rules_empty(self) -> None:
        """Test pending rule inference with no data."""
        estimator = TimeEstimator()
        pending = estimator._infer_pending_rules({}, 10)
        assert pending == {}

    def test_get_estimate_for_job_no_data(self) -> None:
        """Test get_estimate_for_job with no historical data."""
        estimator = TimeEstimator()
        mean, var = estimator.get_estimate_for_job("unknown_rule")
        assert mean == 60.0  # Default global mean
        assert var > 0

    def test_get_estimate_for_job_with_rule_stats(self) -> None:
        """Test get_estimate_for_job uses rule-level stats."""
        stats = {"align": RuleTimingStats(rule="align", durations=[100.0, 110.0, 90.0])}
        estimator = TimeEstimator(rule_stats=stats)

        mean, var = estimator.get_estimate_for_job("align")
        # Should use weighted mean, not simple mean
        assert 90 < mean < 110
        assert var > 0

    def test_wildcard_conditioning_disabled(self) -> None:
        """Test that wildcard conditioning is off by default."""
        estimator = TimeEstimator()
        assert estimator.use_wildcard_conditioning is False

    def test_wildcard_conditioning_enabled(self) -> None:
        """Test wildcard conditioning can be enabled."""
        estimator = TimeEstimator(use_wildcard_conditioning=True)
        assert estimator.use_wildcard_conditioning is True


class TestThreadAwareETA:
    """Tests for thread-aware ETA estimation."""

    def test_thread_stats_initialized_empty(self) -> None:
        """Test thread_stats is initialized as empty dict."""
        estimator = TimeEstimator()
        assert estimator.thread_stats == {}

    def test_get_estimate_with_exact_thread_match(self) -> None:
        """Test get_estimate_for_job uses exact thread match."""
        from snakesee.models import ThreadTimingStats

        estimator = TimeEstimator()

        # Set up thread stats: align with 4 threads takes 50s
        thread_stats = ThreadTimingStats(rule="align")
        thread_stats.stats_by_threads[4] = RuleTimingStats(
            rule="align", durations=[50.0, 52.0, 48.0]
        )
        estimator.thread_stats["align"] = thread_stats

        # Also set up rule stats with different timing (100s)
        estimator.rule_stats["align"] = RuleTimingStats(
            rule="align", durations=[100.0, 100.0, 100.0]
        )

        # With threads=4, should use thread-specific stats (~50s)
        mean, var = estimator.get_estimate_for_job("align", threads=4)
        assert 45 < mean < 55  # Should be around 50s, not 100s
        assert var > 0

    def test_get_estimate_with_thread_fallback(self) -> None:
        """Test get_estimate_for_job falls back to aggregate when no exact match."""
        from snakesee.models import ThreadTimingStats

        estimator = TimeEstimator()

        # Set up thread stats: align has 1-thread (100s) and 8-thread (20s) data
        thread_stats = ThreadTimingStats(rule="align")
        thread_stats.stats_by_threads[1] = RuleTimingStats(rule="align", durations=[100.0])
        thread_stats.stats_by_threads[8] = RuleTimingStats(rule="align", durations=[20.0])
        estimator.thread_stats["align"] = thread_stats

        # Request 4 threads - should fall back to aggregate (~60s average)
        mean, var = estimator.get_estimate_for_job("align", threads=4)
        # Should be aggregate of 100 and 20 = 60
        assert 55 < mean < 65
        assert var > 0

    def test_get_estimate_no_thread_data_falls_to_rule_stats(self) -> None:
        """Test get_estimate_for_job falls back to rule_stats when no thread data."""
        estimator = TimeEstimator()

        # Only rule stats, no thread stats (with varying durations for non-zero variance)
        estimator.rule_stats["align"] = RuleTimingStats(
            rule="align", durations=[95.0, 100.0, 105.0]
        )

        # With threads=4 but no thread data, should use rule stats
        mean, var = estimator.get_estimate_for_job("align", threads=4)
        assert 95 < mean < 105  # Should be around 100s
        assert var > 0

    def test_get_estimate_no_threads_param_uses_rule_stats(self) -> None:
        """Test get_estimate_for_job without threads uses rule_stats."""
        from snakesee.models import ThreadTimingStats

        estimator = TimeEstimator()

        # Set up thread stats with different timing
        thread_stats = ThreadTimingStats(rule="align")
        thread_stats.stats_by_threads[4] = RuleTimingStats(rule="align", durations=[50.0])
        estimator.thread_stats["align"] = thread_stats

        # Set up rule stats
        estimator.rule_stats["align"] = RuleTimingStats(rule="align", durations=[100.0])

        # Without threads param, should use rule stats
        mean, var = estimator.get_estimate_for_job("align")
        assert 95 < mean < 105  # Should be around 100s
        assert var > 0

    def test_thread_stats_takes_priority_over_wildcards(self) -> None:
        """Test thread-specific stats take priority over wildcard conditioning."""
        from snakesee.models import ThreadTimingStats
        from snakesee.models import WildcardTimingStats

        estimator = TimeEstimator(use_wildcard_conditioning=True)

        # Set up thread stats: 30s with 4 threads
        thread_stats = ThreadTimingStats(rule="align")
        thread_stats.stats_by_threads[4] = RuleTimingStats(rule="align", durations=[30.0])
        estimator.thread_stats["align"] = thread_stats

        # Set up wildcard stats: sample1 takes 200s
        wc_stats = WildcardTimingStats(rule="align", wildcard_key="sample")
        wc_stats.stats_by_value["sample1"] = RuleTimingStats(rule="align", durations=[200.0])
        estimator.wildcard_stats["align"] = {"sample": wc_stats}

        # Thread stats should take priority
        mean, var = estimator.get_estimate_for_job(
            "align", wildcards={"sample": "sample1"}, threads=4
        )
        assert 25 < mean < 35  # Should be around 30s, not 200s

    def test_single_sample_thread_variance(self) -> None:
        """Test variance calculation for single-sample thread stats."""
        from snakesee.models import ThreadTimingStats

        estimator = TimeEstimator()

        # Single sample with exact thread match
        thread_stats = ThreadTimingStats(rule="align")
        thread_stats.stats_by_threads[4] = RuleTimingStats(
            rule="align",
            durations=[100.0],  # Single sample
        )
        estimator.thread_stats["align"] = thread_stats

        mean, var = estimator.get_estimate_for_job("align", threads=4)
        assert 95 < mean < 105
        # With single sample, variance should be (mean * 0.2)^2 = (100 * 0.2)^2 = 400
        assert 380 < var < 420

    def test_aggregate_fallback_variance(self) -> None:
        """Test variance calculation for aggregate fallback (no exact thread match)."""
        from snakesee.models import ThreadTimingStats

        estimator = TimeEstimator()

        # Single sample with NO exact thread match (will aggregate)
        thread_stats = ThreadTimingStats(rule="align")
        thread_stats.stats_by_threads[8] = RuleTimingStats(
            rule="align",
            durations=[100.0],  # Single sample, different thread count
        )
        estimator.thread_stats["align"] = thread_stats

        mean, var = estimator.get_estimate_for_job("align", threads=4)  # No match for 4
        assert 95 < mean < 105
        # With aggregate fallback single sample, variance should be (mean * 0.3)^2 = 900
        assert 880 < var < 920
