"""Tests for the WorkflowState module."""

from pathlib import Path

import pytest

from snakesee.models import WorkflowStatus
from snakesee.state.clock import FrozenClock
from snakesee.state.config import EstimationConfig
from snakesee.state.job_registry import Job
from snakesee.state.job_registry import JobStatus
from snakesee.state.workflow_state import WorkflowState


class TestWorkflowStateCreation:
    """Tests for WorkflowState creation."""

    def test_create_basic(self, tmp_path: Path) -> None:
        """Test basic creation from workflow directory."""
        state = WorkflowState.create(tmp_path)
        assert state.paths.workflow_dir == tmp_path
        assert state.jobs is not None
        assert state.rules is not None
        assert state.status == WorkflowStatus.UNKNOWN

    def test_create_with_clock(self, tmp_path: Path) -> None:
        """Test creation with custom clock."""
        clock = FrozenClock(1000.0)
        state = WorkflowState.create(tmp_path, clock=clock)
        assert state.clock is clock

    def test_create_with_config(self, tmp_path: Path) -> None:
        """Test creation with custom config."""
        config = EstimationConfig(default_global_mean=120.0)
        state = WorkflowState.create(tmp_path, config=config)
        assert state.config is config
        assert state.config.default_global_mean == 120.0


class TestWorkflowStateProperties:
    """Tests for WorkflowState properties."""

    def test_completed_count(self, tmp_path: Path) -> None:
        """Test completed job count."""
        state = WorkflowState.create(tmp_path)
        state.jobs.add(Job(key="1", rule="align", status=JobStatus.COMPLETED))
        state.jobs.add(Job(key="2", rule="sort", status=JobStatus.COMPLETED))
        state.jobs.add(Job(key="3", rule="call", status=JobStatus.RUNNING))

        assert state.completed_count == 2

    def test_failed_count(self, tmp_path: Path) -> None:
        """Test failed job count."""
        state = WorkflowState.create(tmp_path)
        state.jobs.add(Job(key="1", rule="align", status=JobStatus.FAILED))
        state.jobs.add(Job(key="2", rule="sort", status=JobStatus.COMPLETED))

        assert state.failed_count == 1

    def test_running_count(self, tmp_path: Path) -> None:
        """Test running job count."""
        state = WorkflowState.create(tmp_path)
        state.jobs.add(Job(key="1", rule="align", status=JobStatus.RUNNING))
        state.jobs.add(Job(key="2", rule="sort", status=JobStatus.RUNNING))
        state.jobs.add(Job(key="3", rule="call", status=JobStatus.PENDING))

        assert state.running_count == 2

    def test_pending_count(self, tmp_path: Path) -> None:
        """Test pending job count."""
        state = WorkflowState.create(tmp_path)
        state.jobs.add(Job(key="1", rule="align", status=JobStatus.PENDING))
        state.jobs.add(Job(key="2", rule="sort", status=JobStatus.PENDING))
        state.jobs.add(Job(key="3", rule="call", status=JobStatus.RUNNING))

        assert state.pending_count == 2

    def test_progress_fraction_with_total(self, tmp_path: Path) -> None:
        """Test progress fraction calculation."""
        state = WorkflowState.create(tmp_path)
        state.total_jobs = 10
        state.jobs.add(Job(key="1", rule="align", status=JobStatus.COMPLETED))
        state.jobs.add(Job(key="2", rule="sort", status=JobStatus.COMPLETED))

        assert state.progress_fraction == pytest.approx(0.2)

    def test_progress_fraction_no_total(self, tmp_path: Path) -> None:
        """Test progress fraction without total jobs."""
        state = WorkflowState.create(tmp_path)
        state.jobs.add(Job(key="1", rule="align", status=JobStatus.COMPLETED))

        assert state.progress_fraction == 0.0

    def test_elapsed_seconds(self, tmp_path: Path) -> None:
        """Test elapsed time calculation."""
        clock = FrozenClock(1100.0)
        state = WorkflowState.create(tmp_path, clock=clock)
        state.start_time = 1000.0

        assert state.elapsed_seconds == pytest.approx(100.0)

    def test_elapsed_seconds_no_start(self, tmp_path: Path) -> None:
        """Test elapsed time without start time."""
        state = WorkflowState.create(tmp_path)
        assert state.elapsed_seconds is None


class TestWorkflowStateStatus:
    """Tests for workflow status updates."""

    def test_status_unknown_no_total(self, tmp_path: Path) -> None:
        """Test status is unknown without total jobs."""
        state = WorkflowState.create(tmp_path)
        state.update_status()
        assert state.status == WorkflowStatus.UNKNOWN

    def test_status_running(self, tmp_path: Path) -> None:
        """Test running status when jobs are running."""
        state = WorkflowState.create(tmp_path)
        state.total_jobs = 5
        state.jobs.add(Job(key="1", rule="align", status=JobStatus.RUNNING))
        state.update_status()
        assert state.status == WorkflowStatus.RUNNING

    def test_status_completed(self, tmp_path: Path) -> None:
        """Test completed status when all jobs done."""
        state = WorkflowState.create(tmp_path)
        state.total_jobs = 2
        state.jobs.add(Job(key="1", rule="align", status=JobStatus.COMPLETED))
        state.jobs.add(Job(key="2", rule="sort", status=JobStatus.COMPLETED))
        state.update_status()
        assert state.status == WorkflowStatus.COMPLETED

    def test_status_failed(self, tmp_path: Path) -> None:
        """Test failed status when jobs have failed."""
        state = WorkflowState.create(tmp_path)
        state.total_jobs = 5
        state.jobs.add(Job(key="1", rule="align", status=JobStatus.COMPLETED))
        state.jobs.add(Job(key="2", rule="sort", status=JobStatus.FAILED))
        state.update_status()
        assert state.status == WorkflowStatus.FAILED

    def test_status_not_started(self, tmp_path: Path) -> None:
        """Test not started status."""
        state = WorkflowState.create(tmp_path)
        state.total_jobs = 5
        state.update_status()
        assert state.status == WorkflowStatus.NOT_STARTED


class TestWorkflowStateJobCompletion:
    """Tests for job completion tracking."""

    def test_record_job_completion(self, tmp_path: Path) -> None:
        """Test recording job completion updates rule stats."""
        state = WorkflowState.create(tmp_path)
        job = Job(
            key="1",
            rule="align",
            status=JobStatus.COMPLETED,
            start_time=100.0,
            end_time=200.0,
            threads=4,
        )
        state.jobs.add(job)

        state.record_job_completion("1")

        rule_stats = state.rules.get("align")
        assert rule_stats is not None
        assert rule_stats.aggregate.count == 1
        assert rule_stats.aggregate.durations[0] == 100.0

    def test_record_job_completion_nonexistent(self, tmp_path: Path) -> None:
        """Test recording completion for non-existent job."""
        state = WorkflowState.create(tmp_path)
        # Should not raise
        state.record_job_completion("nonexistent")
        assert state.rules.total_sample_count() == 0


class TestWorkflowStateBackwardCompat:
    """Tests for backward compatibility."""

    def test_to_progress(self, tmp_path: Path) -> None:
        """Test conversion to WorkflowProgress."""
        state = WorkflowState.create(tmp_path)
        state.total_jobs = 5
        state.start_time = 1000.0
        state.jobs.add(
            Job(
                key="1",
                rule="align",
                status=JobStatus.RUNNING,
                start_time=1050.0,
                threads=4,
            )
        )
        state.jobs.add(Job(key="2", rule="sort", status=JobStatus.COMPLETED))
        state.jobs.add(Job(key="3", rule="call", status=JobStatus.FAILED))

        progress = state.to_progress()

        assert progress.total_jobs == 5
        assert progress.completed_jobs == 1
        assert progress.failed_jobs == 1
        assert len(progress.running_jobs) == 1
        assert progress.running_jobs[0].rule == "align"
        assert progress.start_time == 1000.0


class TestWorkflowStateClear:
    """Tests for state clearing."""

    def test_clear(self, tmp_path: Path) -> None:
        """Test clearing all state."""
        state = WorkflowState.create(tmp_path)
        state.total_jobs = 5
        state.start_time = 1000.0
        state.status = WorkflowStatus.RUNNING
        state.jobs.add(Job(key="1", rule="align", status=JobStatus.RUNNING))
        state.rules.record_completion("align", 100.0, 1000.0)

        state.clear()

        assert len(state.jobs) == 0
        assert len(state.rules) == 0
        assert state.status == WorkflowStatus.UNKNOWN
        assert state.total_jobs is None
        assert state.start_time is None
