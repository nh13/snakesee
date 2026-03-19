"""Tests for thread-aware parallelism estimation."""

import tempfile
import time
from pathlib import Path

import orjson
import pytest

from snakesee.estimation.data_loader import HistoricalDataLoader
from snakesee.estimation.estimator import TimeEstimator
from snakesee.models import JobInfo
from snakesee.models import WorkflowProgress
from snakesee.models import WorkflowStatus
from snakesee.state.rule_registry import RuleRegistry
from snakesee.state.rule_registry import RuleStatistics


class TestTypicalThreads:
    """Tests for RuleStatistics.typical_threads property."""

    def test_typical_threads_no_data(self) -> None:
        """Default to 1.0 when no thread data is available."""
        stats = RuleStatistics(rule="align")
        assert stats.typical_threads == 1.0

    def test_typical_threads_single_thread_count(self) -> None:
        """Single thread count returns that count."""
        stats = RuleStatistics(rule="align")
        for _ in range(5):
            stats.record_completion(duration=10.0, timestamp=time.time(), threads=8)
        assert stats.typical_threads == 8.0

    def test_typical_threads_weighted_average(self) -> None:
        """Weighted average across multiple thread counts."""
        stats = RuleStatistics(rule="align")
        # 3 runs with 4 threads, 1 run with 8 threads
        for _ in range(3):
            stats.record_completion(duration=10.0, timestamp=time.time(), threads=4)
        stats.record_completion(duration=5.0, timestamp=time.time(), threads=8)
        # Expected: (4*3 + 8*1) / 4 = 20/4 = 5.0
        assert stats.typical_threads == pytest.approx(5.0)

    def test_registry_typical_threads(self) -> None:
        """RuleRegistry.typical_threads convenience method."""
        registry = RuleRegistry()
        registry.record_completion("align", 10.0, time.time(), threads=4)
        registry.record_completion("align", 10.0, time.time(), threads=4)
        assert registry.typical_threads("align") == 4.0
        # Unknown rule defaults to 1.0
        assert registry.typical_threads("unknown_rule") == 1.0


class TestEstimateCores:
    """Tests for TimeEstimator._estimate_cores."""

    def test_estimate_cores_from_max_observed(self) -> None:
        """Uses max_observed_thread_sum when available."""
        estimator = TimeEstimator()
        progress = WorkflowProgress(
            workflow_dir=Path(tempfile.gettempdir()),
            status=WorkflowStatus.RUNNING,
            total_jobs=20,
            completed_jobs=5,
            running_jobs=[],
            max_observed_thread_sum=64.0,
        )
        assert estimator._estimate_cores(progress) == 64.0

    def test_estimate_cores_fallback_to_running(self) -> None:
        """Falls back to current running job threads when no max observed."""
        estimator = TimeEstimator()
        running = [
            JobInfo(rule="align", job_id="1", threads=8),
            JobInfo(rule="align", job_id="2", threads=8),
        ]
        progress = WorkflowProgress(
            workflow_dir=Path(tempfile.gettempdir()),
            status=WorkflowStatus.RUNNING,
            total_jobs=10,
            completed_jobs=0,
            running_jobs=running,
        )
        assert estimator._estimate_cores(progress) == 16.0

    def test_estimate_cores_no_threads_defaults_to_one(self) -> None:
        """Running jobs with unknown threads default to 1 each."""
        estimator = TimeEstimator()
        running = [
            JobInfo(rule="align", job_id="1"),
            JobInfo(rule="align", job_id="2"),
            JobInfo(rule="align", job_id="3"),
        ]
        progress = WorkflowProgress(
            workflow_dir=Path(tempfile.gettempdir()),
            status=WorkflowStatus.RUNNING,
            total_jobs=10,
            completed_jobs=0,
            running_jobs=running,
        )
        assert estimator._estimate_cores(progress) == 3.0

    def test_estimate_cores_no_running_jobs(self) -> None:
        """Returns 1.0 when no running jobs and no max observed."""
        estimator = TimeEstimator()
        progress = WorkflowProgress(
            workflow_dir=Path(tempfile.gettempdir()),
            status=WorkflowStatus.RUNNING,
            total_jobs=10,
            completed_jobs=0,
            running_jobs=[],
        )
        assert estimator._estimate_cores(progress) == 1.0


class TestThreadSecondsEstimation:
    """Tests for thread-seconds based estimation."""

    def test_thread_seconds_estimation(self) -> None:
        """4-thread jobs with 16 cores should give correct wall-clock estimate."""
        registry = RuleRegistry()
        now = time.time()
        # Record historical data: align rule takes 100s with 4 threads
        for i in range(5):
            registry.record_completion("align", 100.0, now - i * 200, threads=4)

        estimator = TimeEstimator(rule_registry=registry)

        # 8 pending 4-thread jobs, running on 16 cores
        pending = [
            JobInfo(rule="align", job_id=str(i), threads=4) for i in range(8)
        ]
        progress = WorkflowProgress(
            workflow_dir=Path(tempfile.gettempdir()),
            status=WorkflowStatus.RUNNING,
            total_jobs=10,
            completed_jobs=2,
            running_jobs=[],
            pending_jobs_list=pending,
            recent_completions=[
                JobInfo(rule="align", job_id="c1", start_time=now - 200, end_time=now - 100),
                JobInfo(rule="align", job_id="c2", start_time=now - 100, end_time=now),
            ],
            max_observed_thread_sum=16.0,
        )

        estimate = estimator.estimate_remaining(progress)
        # 8 jobs * 100s * 4 threads = 3200 thread-seconds / 16 cores = 200s
        assert estimate.seconds_remaining == pytest.approx(200.0, rel=0.3)
        assert estimate.inferred_cores == 16.0

    def test_mixed_thread_counts(self) -> None:
        """Mix of 1-thread and 8-thread jobs with correct estimation."""
        registry = RuleRegistry()
        now = time.time()
        # sort rule: 1 thread, 50s each
        for i in range(5):
            registry.record_completion("sort", 50.0, now - i * 100, threads=1)
        # align rule: 8 threads, 200s each
        for i in range(5):
            registry.record_completion("align", 200.0, now - i * 100, threads=8)

        estimator = TimeEstimator(rule_registry=registry)

        # 4 pending sort jobs (1 thread) + 2 pending align jobs (8 threads)
        pending = [
            JobInfo(rule="sort", job_id=f"s{i}", threads=1) for i in range(4)
        ] + [
            JobInfo(rule="align", job_id=f"a{i}", threads=8) for i in range(2)
        ]
        progress = WorkflowProgress(
            workflow_dir=Path(tempfile.gettempdir()),
            status=WorkflowStatus.RUNNING,
            total_jobs=12,
            completed_jobs=6,
            running_jobs=[],
            pending_jobs_list=pending,
            recent_completions=[
                JobInfo(rule="sort", job_id="c1", start_time=now - 100, end_time=now - 50),
            ],
            max_observed_thread_sum=16.0,
        )

        estimate = estimator.estimate_remaining(progress)
        # sort: 4 * 50s * 1 = 200 thread-seconds
        # align: 2 * 200s * 8 = 3200 thread-seconds
        # total: 3400 / 16 = 212.5s
        assert estimate.seconds_remaining == pytest.approx(212.5, rel=0.3)

    def test_single_threaded_backward_compatible(self) -> None:
        """Single-threaded workflows should give same results as before."""
        registry = RuleRegistry()
        now = time.time()
        # All jobs 1 thread (default)
        for i in range(5):
            registry.record_completion("sort", 60.0, now - i * 100)

        estimator = TimeEstimator(rule_registry=registry)

        # 5 pending 1-thread jobs, 1 running (1 thread)
        pending = [
            JobInfo(rule="sort", job_id=str(i)) for i in range(5)
        ]
        running = [
            JobInfo(rule="sort", job_id="r1", start_time=now - 30),
        ]
        progress = WorkflowProgress(
            workflow_dir=Path(tempfile.gettempdir()),
            status=WorkflowStatus.RUNNING,
            total_jobs=8,
            completed_jobs=2,
            running_jobs=running,
            pending_jobs_list=pending,
            recent_completions=[
                JobInfo(rule="sort", job_id="c1", start_time=now - 120, end_time=now - 60),
            ],
        )

        estimate = estimator.estimate_remaining(progress)
        # With 1 running job (1 thread), cores = 1
        # 5 pending * 60s * 1 + ~30s remaining * 1 = ~330 thread-seconds / 1 core
        assert estimate.seconds_remaining > 0
        assert estimate.inferred_cores == 1.0


class TestEventsLoaderPassesThreads:
    """Test that data_loader passes threads from events to registry."""

    def test_events_loader_passes_threads(self) -> None:
        """load_from_events should pass threads to record_completion."""
        registry = RuleRegistry()
        loader = HistoricalDataLoader(registry)

        # Write a temporary events file with threads
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            event = {
                "event_type": "job_finished",
                "rule_name": "align",
                "duration": 100.0,
                "timestamp": time.time(),
                "threads": 4,
                "wildcards": {"sample": "A"},
            }
            f.write(orjson.dumps(event).decode() + "\n")
            events_path = Path(f.name)

        try:
            loader.load_from_events(events_path)
        finally:
            events_path.unlink()

        # Verify threads were recorded
        stats = registry.get("align")
        assert stats is not None
        assert stats.by_threads is not None
        assert 4 in stats.by_threads.stats_by_threads
        assert stats.typical_threads == 4.0
