"""Tests for the time estimator."""

from pathlib import Path

import pytest

from snakesee.estimator import TimeEstimator
from snakesee.models import JobInfo
from snakesee.models import WorkflowProgress
from snakesee.models import WorkflowStatus


class TestTimeEstimator:
    """Tests for the TimeEstimator class."""

    def test_init_empty(self) -> None:
        """Test initialization with no data."""
        estimator = TimeEstimator()
        assert estimator.rule_stats == {}

    def test_init_with_registry(self) -> None:
        """Test initialization with pre-loaded registry."""
        from snakesee.state.rule_registry import RuleRegistry

        registry = RuleRegistry()
        registry.record_completion("align", 100.0, 1000.0)
        estimator = TimeEstimator(rule_registry=registry)
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
        from snakesee.state.rule_registry import RuleRegistry

        registry = RuleRegistry()
        registry.record_completion("align", 100.0, 1000.0)
        registry.record_completion("align", 110.0, 2000.0)
        registry.record_completion("align", 90.0, 3000.0)
        estimator = TimeEstimator(rule_registry=registry)

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
        from snakesee.state.rule_registry import RuleRegistry

        registry = RuleRegistry()
        # Thread-specific: align with 4 threads takes ~50s
        registry.record_completion("align", 50.0, 1000.0, threads=4)
        registry.record_completion("align", 52.0, 2000.0, threads=4)
        registry.record_completion("align", 48.0, 3000.0, threads=4)
        # Also non-thread data at 100s
        registry.record_completion("align", 100.0, 4000.0)
        registry.record_completion("align", 100.0, 5000.0)
        registry.record_completion("align", 100.0, 6000.0)

        estimator = TimeEstimator(rule_registry=registry)

        # With threads=4, should use thread-specific stats (~50s)
        mean, var = estimator.get_estimate_for_job("align", threads=4)
        assert 45 < mean < 55  # Should be around 50s, not 100s
        assert var > 0

    def test_get_estimate_with_thread_fallback(self) -> None:
        """Test get_estimate_for_job falls back to aggregate when no exact match."""
        from snakesee.state.rule_registry import RuleRegistry

        registry = RuleRegistry()
        # Thread stats: 1-thread (100s) and 8-thread (20s), no 4-thread
        registry.record_completion("align", 100.0, 1000.0, threads=1)
        registry.record_completion("align", 20.0, 2000.0, threads=8)

        estimator = TimeEstimator(rule_registry=registry)

        # Request 4 threads - should fall back to aggregate (~60s average)
        mean, var = estimator.get_estimate_for_job("align", threads=4)
        # Should be aggregate of 100 and 20 = 60
        assert 55 < mean < 65
        assert var > 0

    def test_get_estimate_no_thread_data_falls_to_rule_stats(self) -> None:
        """Test get_estimate_for_job falls back to rule_stats when no thread data."""
        from snakesee.state.rule_registry import RuleRegistry

        registry = RuleRegistry()
        # Only aggregate stats, no thread-specific data
        registry.record_completion("align", 95.0, 1000.0)
        registry.record_completion("align", 100.0, 2000.0)
        registry.record_completion("align", 105.0, 3000.0)

        estimator = TimeEstimator(rule_registry=registry)

        # With threads=4 but no thread data, should use rule stats
        mean, var = estimator.get_estimate_for_job("align", threads=4)
        assert 95 < mean < 105  # Should be around 100s
        assert var > 0

    def test_get_estimate_no_threads_param_uses_rule_stats(self) -> None:
        """Test get_estimate_for_job without threads uses rule_stats."""
        from snakesee.state.rule_registry import RuleRegistry

        registry = RuleRegistry()
        # Thread-specific data: 50s with 4 threads
        registry.record_completion("align", 50.0, 1000.0, threads=4)
        # Aggregate: 100s (no threads specified, contributes to aggregate)
        registry.record_completion("align", 100.0, 2000.0)

        estimator = TimeEstimator(rule_registry=registry)

        # Without threads param, should use aggregate rule stats
        mean, var = estimator.get_estimate_for_job("align")
        # Aggregate of 50 and 100 = 75
        assert 70 < mean < 80
        assert var > 0

    def test_thread_stats_takes_priority_over_wildcards(self) -> None:
        """Test thread-specific stats take priority over wildcard conditioning."""
        from snakesee.state.rule_registry import RuleRegistry

        registry = RuleRegistry()
        # Thread stats: 30s with 4 threads
        registry.record_completion("align", 30.0, 1000.0, threads=4)
        # Wildcard stats: sample1 takes 200s (no threads)
        registry.record_completion("align", 200.0, 2000.0, wildcards={"sample": "sample1"})

        estimator = TimeEstimator(rule_registry=registry, use_wildcard_conditioning=True)

        # Thread stats should take priority
        mean, var = estimator.get_estimate_for_job(
            "align", wildcards={"sample": "sample1"}, threads=4
        )
        assert 25 < mean < 35  # Should be around 30s, not 200s

    def test_single_sample_thread_variance(self) -> None:
        """Test variance calculation for single-sample thread stats."""
        from snakesee.state.rule_registry import RuleRegistry

        registry = RuleRegistry()
        # Single sample with threads=4
        registry.record_completion("align", 100.0, 1000.0, threads=4)

        estimator = TimeEstimator(rule_registry=registry)

        mean, var = estimator.get_estimate_for_job("align", threads=4)
        assert 95 < mean < 105
        # With single sample, variance should be (mean * 0.2)^2 = (100 * 0.2)^2 = 400
        assert 380 < var < 420

    def test_aggregate_fallback_variance(self) -> None:
        """Test variance calculation for aggregate fallback (no exact thread match)."""
        from snakesee.state.rule_registry import RuleRegistry

        registry = RuleRegistry()
        # Single sample with threads=8 (no match for threads=4)
        registry.record_completion("align", 100.0, 1000.0, threads=8)

        estimator = TimeEstimator(rule_registry=registry)

        mean, var = estimator.get_estimate_for_job("align", threads=4)  # No match for 4
        assert 95 < mean < 105
        # With aggregate fallback single sample, variance should be (mean * 0.3)^2 = 900
        assert 880 < var < 920


class TestWildcardConditioning:
    """Tests for wildcard-conditioned ETA estimation."""

    def test_wildcard_conditioning_with_match(self) -> None:
        """Test get_estimate_for_job uses wildcard-specific stats when available."""
        from snakesee.state.rule_registry import RuleRegistry

        registry = RuleRegistry()
        # sample1 takes ~50s, sample2 takes ~200s
        # Need at least 3 samples per value for MIN_SAMPLES_FOR_CONDITIONING
        for val in [50.0, 52.0, 48.0]:
            registry.record_completion("align", val, 1000.0, wildcards={"sample": "sample1"})
        for val in [200.0, 210.0, 190.0]:
            registry.record_completion("align", val, 2000.0, wildcards={"sample": "sample2"})

        estimator = TimeEstimator(rule_registry=registry, use_wildcard_conditioning=True)

        # With sample1 wildcard, should use wildcard-specific stats (~50s)
        mean, var = estimator.get_estimate_for_job("align", wildcards={"sample": "sample1"})
        assert 45 < mean < 55  # Should be around 50s, not 200s or aggregate
        assert var > 0

        # With sample2 wildcard, should use wildcard-specific stats (~200s)
        mean2, var2 = estimator.get_estimate_for_job("align", wildcards={"sample": "sample2"})
        assert 190 < mean2 < 210
        assert var2 > 0

    def test_wildcard_conditioning_no_match_falls_to_rule(self) -> None:
        """Test wildcard conditioning falls back to rule stats when no match."""
        from snakesee.state.rule_registry import RuleRegistry

        registry = RuleRegistry()
        # Known wildcard values
        for val in [50.0, 52.0, 48.0]:
            registry.record_completion("align", val, 1000.0, wildcards={"sample": "sample1"})
        for val in [200.0, 210.0, 190.0]:
            registry.record_completion("align", val, 2000.0, wildcards={"sample": "sample2"})
        # Also add some aggregate data at ~100s
        for val in [100.0, 105.0, 95.0]:
            registry.record_completion("align", val, 3000.0)

        estimator = TimeEstimator(rule_registry=registry, use_wildcard_conditioning=True)

        # With unknown sample3 wildcard, should fall back to aggregate stats
        mean, var = estimator.get_estimate_for_job("align", wildcards={"sample": "sample3"})
        # Aggregate of all samples (50s, 200s, 100s) - complex weighting, just check bounds
        assert 50 < mean < 200  # Should not match sample1 or sample2 exactly
        assert var > 0

    def test_pending_jobs_use_wildcard_conditioning(self) -> None:
        """Test estimate_remaining uses wildcard-specific estimates for pending jobs."""
        from snakesee.state.rule_registry import RuleRegistry

        registry = RuleRegistry()
        # sample1 takes ~10s, sample2 takes ~100s
        # Need at least 3 samples per value for MIN_SAMPLES_FOR_CONDITIONING
        for val in [10.0, 12.0, 8.0]:
            registry.record_completion("align", val, 1000.0, wildcards={"sample": "sample1"})
        for val in [100.0, 110.0, 90.0]:
            registry.record_completion("align", val, 2000.0, wildcards={"sample": "sample2"})

        estimator = TimeEstimator(rule_registry=registry, use_wildcard_conditioning=True)

        # Create progress with pending jobs that have wildcards
        progress = WorkflowProgress(
            workflow_dir=Path("."),
            status=WorkflowStatus.RUNNING,
            total_jobs=5,
            completed_jobs=2,
            running_jobs=[],
            recent_completions=[
                JobInfo(rule="align", start_time=1000.0, end_time=1010.0),
                JobInfo(rule="align", start_time=1010.0, end_time=1020.0),
            ],
            # Two pending jobs: one sample1 (~10s), one sample2 (~100s)
            pending_jobs_list=[
                JobInfo(rule="align", wildcards={"sample": "sample1"}),
                JobInfo(rule="align", wildcards={"sample": "sample2"}),
            ],
            start_time=1000.0,
        )

        estimate = estimator.estimate_remaining(progress)
        assert estimate.method == "weighted"
        # Should be around 10s + 100s = 110s (before parallelism adjustment)
        # With parallelism of ~1, estimate should be roughly 110s
        assert estimate.seconds_remaining > 50  # Not just using sample1
        assert estimate.seconds_remaining < 200  # Not double-counting or overestimating

    def test_pending_jobs_fallback_without_wildcards(self) -> None:
        """Test estimate_remaining falls back to rule stats when pending_jobs_list is empty."""
        from snakesee.state.rule_registry import RuleRegistry

        registry = RuleRegistry()
        # Add some rule-level stats (average ~50s)
        for val in [45.0, 50.0, 55.0]:
            registry.record_completion("align", val, 1000.0 + val)

        estimator = TimeEstimator(rule_registry=registry, use_wildcard_conditioning=True)

        # Create progress without pending_jobs_list
        progress = WorkflowProgress(
            workflow_dir=Path("."),
            status=WorkflowStatus.RUNNING,
            total_jobs=5,
            completed_jobs=2,
            running_jobs=[],
            recent_completions=[
                JobInfo(rule="align", start_time=1000.0, end_time=1050.0),
                JobInfo(rule="align", start_time=1050.0, end_time=1100.0),
            ],
            # Empty pending_jobs_list - should use rule-level inference
            pending_jobs_list=[],
            start_time=1000.0,
        )

        estimate = estimator.estimate_remaining(progress)
        assert estimate.method == "weighted"
        # Should use rule-level stats (~50s per job, 3 pending jobs before parallelism)
        assert estimate.seconds_remaining > 0


class TestCombinationConditioning:
    """Tests for full wildcard+threads combination matching."""

    def test_combination_matching_exact(self) -> None:
        """Test get_estimate_for_job uses combination stats when available."""
        from snakesee.state.rule_registry import RuleRegistry

        registry = RuleRegistry()
        # sample1 + threads=4 takes ~10s
        for val in [10.0, 11.0, 9.0]:
            registry.record_completion(
                "align", val, 1000.0, threads=4, wildcards={"sample": "sample1"}
            )
        # sample1 + threads=8 takes ~5s (faster with more threads)
        for val in [5.0, 5.5, 4.5]:
            registry.record_completion(
                "align", val, 2000.0, threads=8, wildcards={"sample": "sample1"}
            )
        # sample2 + threads=4 takes ~100s (larger sample)
        for val in [100.0, 105.0, 95.0]:
            registry.record_completion(
                "align", val, 3000.0, threads=4, wildcards={"sample": "sample2"}
            )

        estimator = TimeEstimator(rule_registry=registry, use_wildcard_conditioning=True)

        # sample1 + threads=4 should get ~10s
        mean, _var = estimator.get_estimate_for_job(
            "align", wildcards={"sample": "sample1"}, threads=4
        )
        assert 8 < mean < 12  # Should be around 10s

        # sample1 + threads=8 should get ~5s
        mean2, _var2 = estimator.get_estimate_for_job(
            "align", wildcards={"sample": "sample1"}, threads=8
        )
        assert 4 < mean2 < 6  # Should be around 5s

        # sample2 + threads=4 should get ~100s
        mean3, _var3 = estimator.get_estimate_for_job(
            "align", wildcards={"sample": "sample2"}, threads=4
        )
        assert 90 < mean3 < 110  # Should be around 100s

    def test_combination_fallback_to_wildcard(self) -> None:
        """Test combination falls back to wildcard stats when no exact match."""
        from snakesee.state.rule_registry import RuleRegistry

        registry = RuleRegistry()
        # Only have sample1 + threads=4 combination data
        for val in [10.0, 11.0, 9.0]:
            registry.record_completion(
                "align", val, 1000.0, threads=4, wildcards={"sample": "sample1"}
            )
        # Add more sample1 data without threads=8 (enough for wildcard conditioning)
        for val in [12.0, 13.0, 11.0]:
            registry.record_completion(
                "align", val, 2000.0, threads=2, wildcards={"sample": "sample1"}
            )

        estimator = TimeEstimator(rule_registry=registry, use_wildcard_conditioning=True)

        # sample1 + threads=16 has no exact combination match
        # Should fall back to wildcard stats for sample1 (mix of thread counts)
        mean, _var = estimator.get_estimate_for_job(
            "align", wildcards={"sample": "sample1"}, threads=16
        )
        # Should be around the wildcard average for sample1 (~10-12s range)
        assert 8 < mean < 14


class TestRuleRegistryIntegration:
    """Tests for TimeEstimator integration with RuleRegistry (Phase 11)."""

    def test_estimator_with_rule_registry(self) -> None:
        """Test TimeEstimator uses RuleRegistry when provided."""
        from snakesee.state.rule_registry import RuleRegistry

        # Create RuleRegistry with data
        registry = RuleRegistry()
        registry.record_completion("align", 100.0, 1000.0)
        registry.record_completion("align", 110.0, 2000.0)
        registry.record_completion("sort", 50.0, 3000.0)

        # Create TimeEstimator with registry
        estimator = TimeEstimator(rule_registry=registry)

        # Should use registry data via _effective_rule_stats
        mean, var = estimator.get_estimate_for_job("align")
        assert 100 < mean < 115  # Weighted mean of 100, 110
        assert var > 0

    def test_estimator_rule_registry_thread_stats(self) -> None:
        """Test TimeEstimator uses thread stats from RuleRegistry."""
        from snakesee.state.rule_registry import RuleRegistry

        # Create RuleRegistry with thread-specific data
        registry = RuleRegistry()
        registry.record_completion("align", 50.0, 1000.0, threads=4)
        registry.record_completion("align", 52.0, 2000.0, threads=4)
        registry.record_completion("align", 100.0, 3000.0, threads=1)

        # Create TimeEstimator with registry
        estimator = TimeEstimator(rule_registry=registry)

        # With threads=4, should use thread-specific stats (~50s)
        mean4, _ = estimator.get_estimate_for_job("align", threads=4)
        assert 45 < mean4 < 60

        # With threads=1, should use thread-specific stats (~100s)
        mean1, _ = estimator.get_estimate_for_job("align", threads=1)
        assert 95 < mean1 < 110

    def test_estimator_rule_registry_global_mean(self) -> None:
        """Test TimeEstimator global_mean_duration uses RuleRegistry."""
        from snakesee.state.rule_registry import RuleRegistry

        # Create RuleRegistry with data
        registry = RuleRegistry()
        registry.record_completion("align", 100.0, 1000.0)
        registry.record_completion("sort", 50.0, 2000.0)

        # Create TimeEstimator with registry
        estimator = TimeEstimator(rule_registry=registry)

        # Global mean should be (100 + 50) / 2 = 75
        assert estimator.global_mean_duration() == pytest.approx(75.0)

    def test_estimator_creates_internal_registry(self) -> None:
        """Test TimeEstimator creates internal registry when none provided."""
        # Create TimeEstimator without explicit registry
        estimator = TimeEstimator()

        # Should have an internal registry
        assert estimator._rule_registry is not None
        assert estimator.rule_stats == {}

        # Can load data into it
        estimator._rule_registry.record_completion("align", 100.0, 1000.0)
        assert "align" in estimator.rule_stats

    def test_global_mean_duration_caching(self) -> None:
        """Test that global_mean_duration uses caching for efficiency."""
        from snakesee.state.rule_registry import RuleRegistry

        registry = RuleRegistry()
        registry.record_completion("align", 100.0, 1000.0)
        registry.record_completion("sort", 50.0, 2000.0)

        estimator = TimeEstimator(rule_registry=registry)

        # First call computes and caches
        mean1 = estimator.global_mean_duration()
        assert mean1 == pytest.approx(75.0)

        # Second call should return cached value
        mean2 = estimator.global_mean_duration()
        assert mean2 == pytest.approx(75.0)

        # Cache should be valid
        assert estimator._global_mean_cache == pytest.approx(75.0)
        assert estimator._global_mean_cache_sample_count == 2

        # Adding new data should invalidate cache
        registry.record_completion("call", 125.0, 3000.0)
        mean3 = estimator.global_mean_duration()
        assert mean3 == pytest.approx(91.67, rel=0.01)  # (100+50+125)/3
        assert estimator._global_mean_cache_sample_count == 3


class TestEstimatorHelperMethods:
    """Tests for extracted helper methods in TimeEstimator."""

    def test_estimate_parallelism_with_history(self) -> None:
        """Test parallelism estimation from historical completion rate."""
        from snakesee.state.rule_registry import RuleRegistry

        registry = RuleRegistry()
        estimator = TimeEstimator(rule_registry=registry)

        # Create progress with history: 10 jobs in 100 seconds = 0.1 jobs/sec
        # With global mean of 60s, parallelism = 0.1 * 60 = 6 -> sqrt(6) ≈ 2.45
        import tempfile

        tmp_dir = Path(tempfile.gettempdir())
        progress = WorkflowProgress(
            workflow_dir=tmp_dir,
            status=WorkflowStatus.RUNNING,
            total_jobs=20,
            completed_jobs=10,
            running_jobs=[],
            start_time=0.0,
        )
        # Mock elapsed time
        progress = WorkflowProgress(
            workflow_dir=tmp_dir,
            status=WorkflowStatus.RUNNING,
            total_jobs=20,
            completed_jobs=10,
            running_jobs=[],
            start_time=0.0,
        )

        # Test with no elapsed time (fallback path)
        parallelism = estimator._estimate_parallelism(progress, 60.0)
        assert parallelism >= 1.0

    def test_estimate_parallelism_fallback(self) -> None:
        """Test parallelism fallback when no history available."""
        estimator = TimeEstimator()

        running = [
            JobInfo(rule="align", job_id="1"),
            JobInfo(rule="align", job_id="2"),
            JobInfo(rule="align", job_id="3"),
            JobInfo(rule="align", job_id="4"),
        ]

        import tempfile

        progress = WorkflowProgress(
            workflow_dir=Path(tempfile.gettempdir()),
            status=WorkflowStatus.RUNNING,
            total_jobs=10,
            completed_jobs=0,
            running_jobs=running,
        )

        parallelism = estimator._estimate_parallelism(progress, 60.0)
        # With 4 running jobs, fallback is min(4, sqrt(4)) = 2
        assert parallelism == pytest.approx(2.0)

    def test_calculate_confidence_scores_empty(self) -> None:
        """Test confidence calculation with no data."""
        estimator = TimeEstimator()

        sample_conf, recency, consistency, coverage = estimator._calculate_confidence_scores({}, {})

        assert sample_conf == 0.0
        assert recency == 0.5  # Default when no data
        assert consistency == 0.5  # Default when no data
        assert coverage == 0.0

    def test_calculate_confidence_scores_with_data(self) -> None:
        """Test confidence calculation with data."""
        from snakesee.state.rule_registry import RuleRegistry

        registry = RuleRegistry()
        # Add consistent timing data
        for i in range(5):
            registry.record_completion("align", 100.0 + i, 1000.0 + i * 100)

        estimator = TimeEstimator(rule_registry=registry)
        rule_completed = {"align": 5}

        sample_conf, recency, consistency, coverage = estimator._calculate_confidence_scores(
            rule_completed, estimator.rule_stats
        )

        assert sample_conf > 0.0
        assert recency > 0.0
        assert consistency > 0.0
        assert coverage > 0.0


class TestEstimatorFuzzyMatching:
    """Tests for TimeEstimator fuzzy matching behavior."""

    def test_fuzzy_match_similar_rule(self) -> None:
        """Test fuzzy matching finds similar rule names."""
        from snakesee.state.rule_registry import RuleRegistry

        registry = RuleRegistry()
        registry.record_completion("align_reads", 100.0, 1000.0)

        estimator = TimeEstimator(rule_registry=registry)

        # Similar rule name should match via fuzzy matching
        duration, _variance = estimator.get_estimate_for_job(
            rule="align_read",  # One character different
            threads=None,
            wildcards=None,
            input_size=None,
        )

        # Should get estimate from fuzzy match (with higher variance)
        assert duration > 0

    def test_no_fuzzy_match_very_different_rule(self) -> None:
        """Test fuzzy matching doesn't match very different rules."""
        from snakesee.state.rule_registry import RuleRegistry

        registry = RuleRegistry()
        registry.record_completion("align", 100.0, 1000.0)

        estimator = TimeEstimator(rule_registry=registry)

        # Very different rule name should fall back to global mean
        duration, _variance = estimator.get_estimate_for_job(
            rule="completely_different_name",
            threads=None,
            wildcards=None,
            input_size=None,
        )

        # Should get fallback estimate (global mean)
        assert duration > 0

    def test_code_hash_match_priority(self) -> None:
        """Test that code hash matching takes priority over fuzzy matching."""
        from snakesee.state.rule_registry import RuleRegistry

        registry = RuleRegistry()
        registry.record_completion("rule_v1", 50.0, 1000.0)
        registry.record_completion("rule_v2", 200.0, 2000.0)

        # Simulate code hash mapping - rule_new shares hash with rule_v2
        # Both rules must be in the same hash group for the lookup to work
        estimator = TimeEstimator(rule_registry=registry)
        estimator.code_hash_to_rules = {"hash123": {"rule_new", "rule_v2"}}

        duration, _ = estimator.get_estimate_for_job(
            rule="rule_new",
            threads=None,
            wildcards=None,
            input_size=None,
        )

        # Should use rule_v2's timing (200s) due to code hash match
        assert duration == pytest.approx(200.0, rel=0.1)


class TestEstimatorCacheBehavior:
    """Tests for TimeEstimator cache behavior."""

    def test_global_mean_cache_hit(self) -> None:
        """Test that global mean cache is reused on repeated calls."""
        from snakesee.state.rule_registry import RuleRegistry

        registry = RuleRegistry()
        registry.record_completion("align", 100.0, 1000.0)

        estimator = TimeEstimator(rule_registry=registry)

        # First call computes and caches
        mean1 = estimator.global_mean_duration()
        cache_count1 = estimator._global_mean_cache_sample_count

        # Second call should hit cache
        mean2 = estimator.global_mean_duration()
        cache_count2 = estimator._global_mean_cache_sample_count

        assert mean1 == mean2
        assert cache_count1 == cache_count2

    def test_rule_stats_cache_invalidation(self) -> None:
        """Test that adding new timing data invalidates cache."""
        from snakesee.state.rule_registry import RuleRegistry

        registry = RuleRegistry()
        registry.record_completion("align", 100.0, 1000.0)

        estimator = TimeEstimator(rule_registry=registry)

        mean1 = estimator.global_mean_duration()
        assert mean1 == pytest.approx(100.0)

        # Add new data
        registry.record_completion("sort", 200.0, 2000.0)

        # Cache should be invalidated, new mean computed
        mean2 = estimator.global_mean_duration()
        assert mean2 == pytest.approx(150.0)  # (100 + 200) / 2

    def test_estimate_job_uses_cached_stats(self) -> None:
        """Test that get_estimate_for_job uses cached rule_stats."""
        from snakesee.state.rule_registry import RuleRegistry

        registry = RuleRegistry()
        registry.record_completion("align", 100.0, 1000.0)
        registry.record_completion("align", 120.0, 2000.0)

        estimator = TimeEstimator(rule_registry=registry)

        # First estimate
        dur1, _ = estimator.get_estimate_for_job("align", None, None, None)

        # Second estimate should use cached stats
        dur2, _ = estimator.get_estimate_for_job("align", None, None, None)

        assert dur1 == dur2
