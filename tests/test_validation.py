"""Tests for validation module."""

from pathlib import Path

from snakesee.events import EventType
from snakesee.events import SnakeseeEvent
from snakesee.models import JobInfo
from snakesee.models import WorkflowProgress
from snakesee.models import WorkflowStatus
from snakesee.validation import Discrepancy
from snakesee.validation import EventAccumulator
from snakesee.validation import ValidationLogger
from snakesee.validation import compare_states


class TestEventAccumulator:
    """Tests for EventAccumulator class."""

    def test_workflow_started(self) -> None:
        """Test handling workflow started event."""
        acc = EventAccumulator()
        event = SnakeseeEvent(
            event_type=EventType.WORKFLOW_STARTED,
            timestamp=1000.0,
            workflow_id="test-123",
        )
        acc.process_event(event)

        assert acc.workflow_started is True
        assert acc.workflow_start_time == 1000.0

    def test_progress_event(self) -> None:
        """Test handling progress event."""
        acc = EventAccumulator()
        event = SnakeseeEvent(
            event_type=EventType.PROGRESS,
            timestamp=1000.0,
            completed_jobs=5,
            total_jobs=10,
        )
        acc.process_event(event)

        assert acc.total_jobs == 10
        assert acc.completed_jobs == 5

    def test_job_lifecycle(self) -> None:
        """Test full job lifecycle: submitted -> started -> finished."""
        acc = EventAccumulator()

        # Submit job
        acc.process_event(
            SnakeseeEvent(
                event_type=EventType.JOB_SUBMITTED,
                timestamp=1000.0,
                job_id=1,
                rule_name="align",
                wildcards=(("sample", "A"),),
            )
        )
        assert len(acc.jobs) == 1
        assert acc.jobs[1].status == "submitted"
        assert acc.jobs[1].rule_name == "align"

        # Start job
        acc.process_event(
            SnakeseeEvent(
                event_type=EventType.JOB_STARTED,
                timestamp=1001.0,
                job_id=1,
            )
        )
        assert acc.jobs[1].status == "running"
        assert acc.jobs[1].start_time == 1001.0

        # Finish job
        acc.process_event(
            SnakeseeEvent(
                event_type=EventType.JOB_FINISHED,
                timestamp=1100.0,
                job_id=1,
                duration=99.0,
            )
        )
        assert acc.jobs[1].status == "finished"
        assert acc.jobs[1].end_time == 1100.0
        assert acc.jobs[1].duration == 99.0

    def test_job_error(self) -> None:
        """Test job error handling."""
        acc = EventAccumulator()

        # Submit and start
        acc.process_event(
            SnakeseeEvent(
                event_type=EventType.JOB_SUBMITTED,
                timestamp=1000.0,
                job_id=1,
                rule_name="sort",
            )
        )
        acc.process_event(
            SnakeseeEvent(
                event_type=EventType.JOB_STARTED,
                timestamp=1001.0,
                job_id=1,
            )
        )

        # Error
        acc.process_event(
            SnakeseeEvent(
                event_type=EventType.JOB_ERROR,
                timestamp=1050.0,
                job_id=1,
                error_message="Out of memory",
            )
        )

        assert acc.jobs[1].status == "error"
        assert acc.jobs[1].error_message == "Out of memory"

    def test_running_jobs_property(self) -> None:
        """Test running_jobs property."""
        acc = EventAccumulator()

        # Add two jobs, one running, one finished
        acc.process_event(
            SnakeseeEvent(
                event_type=EventType.JOB_SUBMITTED,
                timestamp=1000.0,
                job_id=1,
                rule_name="align",
            )
        )
        acc.process_event(
            SnakeseeEvent(
                event_type=EventType.JOB_STARTED,
                timestamp=1001.0,
                job_id=1,
            )
        )
        acc.process_event(
            SnakeseeEvent(
                event_type=EventType.JOB_SUBMITTED,
                timestamp=1000.0,
                job_id=2,
                rule_name="sort",
            )
        )
        acc.process_event(
            SnakeseeEvent(
                event_type=EventType.JOB_STARTED,
                timestamp=1001.0,
                job_id=2,
            )
        )
        acc.process_event(
            SnakeseeEvent(
                event_type=EventType.JOB_FINISHED,
                timestamp=1100.0,
                job_id=2,
            )
        )

        running = acc.running_jobs
        assert len(running) == 1
        assert running[0].job_id == 1

    def test_failed_jobs_property(self) -> None:
        """Test failed_jobs property."""
        acc = EventAccumulator()

        acc.process_event(
            SnakeseeEvent(
                event_type=EventType.JOB_ERROR,
                timestamp=1050.0,
                job_id=1,
                rule_name="sort",
                error_message="Failed",
            )
        )

        failed = acc.failed_jobs
        assert len(failed) == 1
        assert failed[0].job_id == 1


class TestCompareStates:
    """Tests for compare_states function."""

    def _make_progress(
        self,
        tmp_path: Path,
        total: int = 10,
        completed: int = 5,
        running: list[JobInfo] | None = None,
        failed: int = 0,
        failed_list: list[JobInfo] | None = None,
    ) -> WorkflowProgress:
        """Create a WorkflowProgress for testing."""
        return WorkflowProgress(
            workflow_dir=tmp_path,
            status=WorkflowStatus.RUNNING,
            total_jobs=total,
            completed_jobs=completed,
            failed_jobs=failed,
            failed_jobs_list=failed_list or [],
            running_jobs=running or [],
            recent_completions=[],
        )

    def test_matching_states(self, tmp_path: Path) -> None:
        """Test that matching states produce no discrepancies."""
        acc = EventAccumulator()
        acc.workflow_started = True
        acc.total_jobs = 10
        acc.completed_jobs = 5

        progress = self._make_progress(tmp_path, total=10, completed=5)

        discrepancies = compare_states(acc, progress)
        # Should only have count-related discrepancies if any
        assert all(d.category != "total_jobs" for d in discrepancies)
        assert all(d.category != "completed_jobs" for d in discrepancies)

    def test_total_jobs_mismatch(self, tmp_path: Path) -> None:
        """Test detection of total jobs mismatch."""
        acc = EventAccumulator()
        acc.workflow_started = True
        acc.total_jobs = 10
        acc.completed_jobs = 5

        progress = self._make_progress(tmp_path, total=15, completed=5)

        discrepancies = compare_states(acc, progress)
        total_disc = [d for d in discrepancies if d.category == "total_jobs"]
        assert len(total_disc) == 1
        assert total_disc[0].event_value == 10
        assert total_disc[0].parsed_value == 15

    def test_running_count_mismatch(self, tmp_path: Path) -> None:
        """Test detection of running job count mismatch."""
        acc = EventAccumulator()
        acc.workflow_started = True
        # Add a running job to accumulator
        acc.process_event(
            SnakeseeEvent(
                event_type=EventType.JOB_SUBMITTED,
                timestamp=1000.0,
                job_id=1,
                rule_name="align",
            )
        )
        acc.process_event(
            SnakeseeEvent(
                event_type=EventType.JOB_STARTED,
                timestamp=1001.0,
                job_id=1,
            )
        )

        # But parsed state shows no running jobs
        progress = self._make_progress(tmp_path, running=[])

        discrepancies = compare_states(acc, progress)
        count_disc = [d for d in discrepancies if d.category == "running_count"]
        assert len(count_disc) == 1
        assert count_disc[0].event_value == 1
        assert count_disc[0].parsed_value == 0

    def test_missing_running_job(self, tmp_path: Path) -> None:
        """Test detection of job running in events but not in parsed."""
        acc = EventAccumulator()
        acc.workflow_started = True
        acc.process_event(
            SnakeseeEvent(
                event_type=EventType.JOB_SUBMITTED,
                timestamp=1000.0,
                job_id=42,
                rule_name="align",
            )
        )
        acc.process_event(
            SnakeseeEvent(
                event_type=EventType.JOB_STARTED,
                timestamp=1001.0,
                job_id=42,
            )
        )

        progress = self._make_progress(tmp_path, running=[])

        discrepancies = compare_states(acc, progress)
        missing = [d for d in discrepancies if d.category == "missing_running_job"]
        assert len(missing) == 1
        assert missing[0].job_id == 42
        assert missing[0].rule_name == "align"

    def test_extra_running_job(self, tmp_path: Path) -> None:
        """Test detection of job in parsed but not tracked by events."""
        acc = EventAccumulator()
        acc.workflow_started = True

        # Parsed state has a running job
        running_job = JobInfo(rule="align", job_id="42", start_time=1000.0)
        progress = self._make_progress(tmp_path, running=[running_job])

        discrepancies = compare_states(acc, progress)
        extra = [d for d in discrepancies if d.category == "extra_running_job"]
        assert len(extra) == 1
        assert extra[0].job_id == 42

    def test_failed_job_mismatch(self, tmp_path: Path) -> None:
        """Test detection of failed job mismatch."""
        acc = EventAccumulator()
        acc.workflow_started = True
        acc.process_event(
            SnakeseeEvent(
                event_type=EventType.JOB_ERROR,
                timestamp=1050.0,
                job_id=1,
                rule_name="sort",
                error_message="Failed",
            )
        )

        # Parsed state shows no failed jobs
        progress = self._make_progress(tmp_path, failed=0, failed_list=[])

        discrepancies = compare_states(acc, progress)
        missing = [d for d in discrepancies if d.category == "missing_failed_job"]
        assert len(missing) == 1
        assert missing[0].job_id == 1


class TestDiscrepancy:
    """Tests for Discrepancy dataclass."""

    def test_to_dict_minimal(self) -> None:
        """Test to_dict with minimal fields."""
        d = Discrepancy(
            category="test",
            severity="warning",
            message="Test message",
        )
        result = d.to_dict()
        assert result["category"] == "test"
        assert result["severity"] == "warning"
        assert result["message"] == "Test message"
        assert "job_id" not in result

    def test_to_dict_full(self) -> None:
        """Test to_dict with all fields."""
        d = Discrepancy(
            category="running_count",
            severity="error",
            message="Mismatch",
            event_value=5,
            parsed_value=3,
            job_id=42,
            rule_name="align",
            wildcards={"sample": "A"},
        )
        result = d.to_dict()
        assert result["event_value"] == 5
        assert result["parsed_value"] == 3
        assert result["job_id"] == 42
        assert result["rule_name"] == "align"
        assert result["wildcards"] == {"sample": "A"}


class TestValidationLogger:
    """Tests for ValidationLogger class."""

    def test_creates_log_file(self, tmp_path: Path) -> None:
        """Test that logger creates log file."""
        logger = ValidationLogger(tmp_path)
        logger.log_session_start()
        logger.close()

        log_file = tmp_path / ".snakesee_validation.log"
        assert log_file.exists()
        content = log_file.read_text()
        assert "VALIDATION SESSION STARTED" in content

    def test_logs_discrepancy(self, tmp_path: Path) -> None:
        """Test logging a discrepancy."""
        logger = ValidationLogger(tmp_path)
        logger.log_discrepancy(
            Discrepancy(
                category="running_count",
                severity="warning",
                message="Count mismatch",
                event_value=5,
                parsed_value=3,
                job_id=42,
                rule_name="align",
            )
        )
        logger.close()

        log_file = tmp_path / ".snakesee_validation.log"
        content = log_file.read_text()
        assert "running_count" in content
        assert "Count mismatch" in content
        assert "job_id=42" in content
        assert "rule=align" in content
        assert "events=5" in content
        assert "parsed=3" in content

    def test_logs_summary(self, tmp_path: Path) -> None:
        """Test logging summary."""
        logger = ValidationLogger(tmp_path)

        acc = EventAccumulator()
        acc.total_jobs = 10
        acc.completed_jobs = 5

        progress = WorkflowProgress(
            workflow_dir=tmp_path,
            status=WorkflowStatus.RUNNING,
            total_jobs=10,
            completed_jobs=5,
            running_jobs=[],
        )

        logger.log_summary(acc, progress)
        logger.close()

        log_file = tmp_path / ".snakesee_validation.log"
        content = log_file.read_text()
        assert "EVENT STATE" in content
        assert "PARSED STATE" in content
