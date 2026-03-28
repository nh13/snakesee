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


class TestEstimatedCores:
    """Tests for TimeEstimator.estimated_cores property."""

    def test_tracks_peak_thread_sum(self) -> None:
        """Peak thread sum is retained across estimate_remaining calls."""
        estimator = TimeEstimator()
        running_16 = [
            JobInfo(rule="a", job_id="1", threads=8),
            JobInfo(rule="a", job_id="2", threads=8),
        ]
        running_4 = [JobInfo(rule="a", job_id="1", threads=4)]

        progress_16 = WorkflowProgress(
            workflow_dir=Path(tempfile.gettempdir()),
            status=WorkflowStatus.RUNNING,
            total_jobs=20,
            completed_jobs=5,
            running_jobs=running_16,
        )
        progress_4 = WorkflowProgress(
            workflow_dir=Path(tempfile.gettempdir()),
            status=WorkflowStatus.RUNNING,
            total_jobs=20,
            completed_jobs=10,
            running_jobs=running_4,
        )

        estimator.estimate_remaining(progress_16)
        assert estimator.estimated_cores == 16.0
        # Peak is retained even with fewer running jobs
        estimator.estimate_remaining(progress_4)
        assert estimator.estimated_cores == 16.0

    def test_fallback_to_running(self) -> None:
        """Falls back to current running job threads when no prior peak."""
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
        estimator.estimate_remaining(progress)
        assert estimator.estimated_cores == 16.0

    def test_no_threads_defaults_to_one(self) -> None:
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
        estimator.estimate_remaining(progress)
        assert estimator.estimated_cores == 3.0

    def test_no_running_jobs(self) -> None:
        """Returns 1.0 when no running jobs and no prior peak."""
        estimator = TimeEstimator()
        assert estimator.estimated_cores == 1.0

    def test_job_threads_zero(self) -> None:
        """threads=0 should return 0.0, not fall back to typical_threads."""
        estimator = TimeEstimator()
        job = JobInfo(rule="align", job_id="1", threads=0)
        assert estimator._job_threads(job) == 0.0


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
        # Seed the peak thread sum by estimating with 16 threads of running jobs
        seed_progress = WorkflowProgress(
            workflow_dir=Path(tempfile.gettempdir()),
            status=WorkflowStatus.RUNNING,
            total_jobs=10,
            completed_jobs=0,
            running_jobs=[JobInfo(rule="align", job_id=f"s{i}", threads=4) for i in range(4)],
        )
        estimator.estimate_remaining(seed_progress)

        # 8 pending 4-thread jobs, running on 16 cores
        pending = [JobInfo(rule="align", job_id=str(i), threads=4) for i in range(8)]
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
        estimator._max_observed_thread_sum = 16.0

        # 4 pending sort jobs (1 thread) + 2 pending align jobs (8 threads)
        pending = [JobInfo(rule="sort", job_id=f"s{i}", threads=1) for i in range(4)] + [
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
        pending = [JobInfo(rule="sort", job_id=str(i)) for i in range(5)]
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


class TestParseCoresFromLog:
    """Tests for parse_cores_from_log function."""

    def test_parses_cores_line(self, tmp_path: Path) -> None:
        """Parse 'Provided cores: N' from a Snakemake log."""
        from snakesee.parser import parse_cores_from_log

        log_file = tmp_path / "test.log"
        log_file.write_text(
            "Building DAG of jobs...\n"
            "Provided cores: 16\n"
            "Rules claiming more threads will be scaled down.\n"
        )
        assert parse_cores_from_log(log_file) == 16

    def test_parses_cores_with_hint(self, tmp_path: Path) -> None:
        """Parse cores line with parallelism hint (cores=1)."""
        from snakesee.parser import parse_cores_from_log

        log_file = tmp_path / "test.log"
        log_file.write_text("Provided cores: 1 (use --cores to define parallelism)\n")
        assert parse_cores_from_log(log_file) == 1

    def test_returns_none_when_missing(self, tmp_path: Path) -> None:
        """Return None when no cores line is present."""
        from snakesee.parser import parse_cores_from_log

        log_file = tmp_path / "test.log"
        log_file.write_text("Building DAG of jobs...\nrule align:\n    jobid: 1\n")
        assert parse_cores_from_log(log_file) is None

    def test_returns_none_for_missing_file(self, tmp_path: Path) -> None:
        """Return None for a nonexistent file."""
        from snakesee.parser import parse_cores_from_log

        assert parse_cores_from_log(tmp_path / "nonexistent.log") is None


class TestProvidedCores:
    """Tests for TimeEstimator.set_provided_cores."""

    def test_provided_cores_overrides_observed(self) -> None:
        """Provided cores takes precedence over peak observed thread sum."""
        estimator = TimeEstimator()
        estimator._max_observed_thread_sum = 8.0
        estimator.set_provided_cores(32)
        assert estimator.estimated_cores == 32.0

    def test_observed_used_when_no_provided(self) -> None:
        """Peak observed thread sum used when no provided cores."""
        estimator = TimeEstimator()
        estimator._max_observed_thread_sum = 12.0
        assert estimator.estimated_cores == 12.0

    def test_provided_cores_used_in_estimation(self) -> None:
        """Provided cores flows through to TimeEstimate.inferred_cores."""
        registry = RuleRegistry()
        now = time.time()
        for i in range(5):
            registry.record_completion("align", 100.0, now - i * 200, threads=4)

        estimator = TimeEstimator(rule_registry=registry)
        estimator.set_provided_cores(32)

        pending = [JobInfo(rule="align", job_id=str(i), threads=4) for i in range(8)]
        progress = WorkflowProgress(
            workflow_dir=Path(tempfile.gettempdir()),
            status=WorkflowStatus.RUNNING,
            total_jobs=10,
            completed_jobs=2,
            running_jobs=[],
            pending_jobs_list=pending,
            recent_completions=[
                JobInfo(rule="align", job_id="c1", start_time=now - 200, end_time=now - 100),
            ],
        )
        estimate = estimator.estimate_remaining(progress)
        assert estimate.inferred_cores == 32.0
        # 8 jobs * 100s * 4 threads = 3200 thread-seconds / 32 cores = 100s
        assert estimate.seconds_remaining == pytest.approx(100.0, rel=0.3)


class TestSimpleEstimationThreadSeconds:
    """Tests for thread-seconds model in _estimate_simple."""

    def test_simple_uses_thread_seconds_with_job_list(self) -> None:
        """_estimate_simple uses thread-seconds when pending job list is available."""
        from snakesee.state.clock import FrozenClock
        from snakesee.state.clock import reset_clock
        from snakesee.state.clock import set_clock

        clock = FrozenClock(1200.0)
        set_clock(clock)

        try:
            estimator = TimeEstimator()
            estimator._max_observed_thread_sum = 16.0

            # 4 completed jobs in 200s with 16 cores
            pending = [JobInfo(rule="align", job_id=str(i), threads=4) for i in range(4)]
            progress = WorkflowProgress(
                workflow_dir=Path(tempfile.gettempdir()),
                status=WorkflowStatus.RUNNING,
                total_jobs=8,
                completed_jobs=4,
                running_jobs=[],
                pending_jobs_list=pending,
                start_time=1000.0,
            )

            estimate = estimator.estimate_remaining(progress)
            assert estimate.method == "simple"
            assert estimate.seconds_remaining > 0
            assert estimate.inferred_cores == 16.0
        finally:
            reset_clock()

    def test_simple_fallback_without_job_list(self) -> None:
        """_estimate_simple falls back to linear extrapolation without job list."""
        from snakesee.state.clock import FrozenClock
        from snakesee.state.clock import reset_clock
        from snakesee.state.clock import set_clock

        clock = FrozenClock(1200.0)
        set_clock(clock)

        try:
            estimator = TimeEstimator()

            progress = WorkflowProgress(
                workflow_dir=Path(tempfile.gettempdir()),
                status=WorkflowStatus.RUNNING,
                total_jobs=10,
                completed_jobs=5,
                running_jobs=[],
                start_time=1000.0,
            )

            estimate = estimator.estimate_remaining(progress)
            assert estimate.method == "simple"
            # 200s elapsed, 5 done → 40s/job × 5 remaining = 200s
            assert estimate.seconds_remaining == pytest.approx(200.0, rel=0.01)
        finally:
            reset_clock()


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
