"""Tests for validation module."""

from pathlib import Path

import pytest

from snakesee.events import EventType
from snakesee.events import SnakeseeEvent
from snakesee.exceptions import InvalidParameterError
from snakesee.models import JobInfo
from snakesee.models import WorkflowProgress
from snakesee.models import WorkflowStatus
from snakesee.validation import Discrepancy
from snakesee.validation import EventAccumulator
from snakesee.validation import ValidationLogger
from snakesee.validation import compare_states
from snakesee.validation import require_in_range
from snakesee.validation import require_non_negative
from snakesee.validation import require_not_empty
from snakesee.validation import require_positive
from snakesee.validation import validate_in_range
from snakesee.validation import validate_non_negative
from snakesee.validation import validate_not_empty
from snakesee.validation import validate_positive


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


class TestOutOfOrderEvents:
    """Tests for handling out-of-order events."""

    def test_finished_before_started(self) -> None:
        """Test handling finish event before start event."""
        acc = EventAccumulator()

        # Submit job
        acc.process_event(
            SnakeseeEvent(
                event_type=EventType.JOB_SUBMITTED,
                timestamp=1000.0,
                job_id=1,
                rule_name="align",
            )
        )

        # Finish arrives before start (out of order)
        acc.process_event(
            SnakeseeEvent(
                event_type=EventType.JOB_FINISHED,
                timestamp=1100.0,
                job_id=1,
                duration=50.0,
            )
        )

        # Job should still be marked as finished
        assert acc.jobs[1].status == "finished"
        assert acc.jobs[1].end_time == 1100.0

    def test_error_before_started(self) -> None:
        """Test handling error event before start event."""
        acc = EventAccumulator()

        # Submit job
        acc.process_event(
            SnakeseeEvent(
                event_type=EventType.JOB_SUBMITTED,
                timestamp=1000.0,
                job_id=1,
                rule_name="align",
            )
        )

        # Error arrives before start
        acc.process_event(
            SnakeseeEvent(
                event_type=EventType.JOB_ERROR,
                timestamp=1050.0,
                job_id=1,
            )
        )

        # Job should be marked as error
        assert acc.jobs[1].status == "error"

    def test_duplicate_finish_events(self) -> None:
        """Test handling duplicate finish events."""
        acc = EventAccumulator()

        # Full lifecycle
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
                event_type=EventType.JOB_FINISHED,
                timestamp=1100.0,
                job_id=1,
                duration=99.0,
            )
        )

        # Duplicate finish event
        acc.process_event(
            SnakeseeEvent(
                event_type=EventType.JOB_FINISHED,
                timestamp=1101.0,
                job_id=1,
                duration=100.0,
            )
        )

        # Should still be finished with original values
        assert acc.jobs[1].status == "finished"

    def test_event_for_unknown_job(self) -> None:
        """Test handling events for jobs not previously submitted."""
        acc = EventAccumulator()

        # Start event for unknown job
        acc.process_event(
            SnakeseeEvent(
                event_type=EventType.JOB_STARTED,
                timestamp=1001.0,
                job_id=999,
            )
        )

        # Should create job entry
        assert 999 in acc.jobs
        assert acc.jobs[999].start_time == 1001.0

    def test_progress_decreases(self) -> None:
        """Test handling progress that decreases (shouldn't happen but handle gracefully)."""
        acc = EventAccumulator()

        # Initial progress
        acc.process_event(
            SnakeseeEvent(
                event_type=EventType.PROGRESS,
                timestamp=1000.0,
                completed_jobs=10,
                total_jobs=20,
            )
        )
        assert acc.completed_jobs == 10

        # Progress decreases (anomaly)
        acc.process_event(
            SnakeseeEvent(
                event_type=EventType.PROGRESS,
                timestamp=1001.0,
                completed_jobs=5,  # Less than before
                total_jobs=20,
            )
        )

        # Should accept the new value (no special handling)
        assert acc.completed_jobs == 5

    def test_timestamps_out_of_order(self) -> None:
        """Test events with timestamps out of order.

        When events arrive out of order, later events can overwrite earlier states.
        This tests that the accumulator handles this gracefully without crashing.
        """
        acc = EventAccumulator()

        # Events arrive out of timestamp order
        acc.process_event(
            SnakeseeEvent(
                event_type=EventType.JOB_FINISHED,
                timestamp=1100.0,
                job_id=1,
                duration=99.0,
            )
        )
        acc.process_event(
            SnakeseeEvent(
                event_type=EventType.JOB_SUBMITTED,
                timestamp=1000.0,  # Earlier timestamp
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

        # Started event arrives after finished, so status is running
        # This is expected - events are processed in arrival order, not timestamp order
        assert acc.jobs[1].rule_name == "align"
        assert acc.jobs[1].status == "running"


# =============================================================================
# Tests for Parameter Validation Functions
# =============================================================================


class TestRequirePositive:
    """Tests for require_positive function."""

    def test_positive_value(self) -> None:
        """Test that positive values pass through."""
        assert require_positive(5.0, "test") == 5.0
        assert require_positive(1, "test") == 1
        assert require_positive(0.001, "test") == 0.001

    def test_zero_raises(self) -> None:
        """Test that zero raises InvalidParameterError."""
        with pytest.raises(InvalidParameterError) as exc_info:
            require_positive(0, "param")
        assert exc_info.value.parameter == "param"
        assert exc_info.value.value == 0

    def test_negative_raises(self) -> None:
        """Test that negative values raise InvalidParameterError."""
        with pytest.raises(InvalidParameterError):
            require_positive(-5.0, "param")


class TestRequireNonNegative:
    """Tests for require_non_negative function."""

    def test_positive_value(self) -> None:
        """Test that positive values pass through."""
        assert require_non_negative(5.0, "test") == 5.0

    def test_zero_passes(self) -> None:
        """Test that zero passes through."""
        assert require_non_negative(0, "test") == 0
        assert require_non_negative(0.0, "test") == 0.0

    def test_negative_raises(self) -> None:
        """Test that negative values raise InvalidParameterError."""
        with pytest.raises(InvalidParameterError) as exc_info:
            require_non_negative(-1.0, "param")
        assert exc_info.value.parameter == "param"


class TestRequireNotEmpty:
    """Tests for require_not_empty function."""

    def test_non_empty_list(self) -> None:
        """Test that non-empty lists pass through."""
        result = require_not_empty([1, 2, 3], "items")
        assert result == [1, 2, 3]

    def test_non_empty_tuple(self) -> None:
        """Test that non-empty tuples pass through."""
        result = require_not_empty((1, 2), "items")
        assert result == (1, 2)

    def test_empty_list_raises(self) -> None:
        """Test that empty lists raise InvalidParameterError."""
        with pytest.raises(InvalidParameterError) as exc_info:
            require_not_empty([], "items")
        assert exc_info.value.parameter == "items"

    def test_empty_tuple_raises(self) -> None:
        """Test that empty tuples raise InvalidParameterError."""
        with pytest.raises(InvalidParameterError):
            require_not_empty((), "items")


class TestRequireInRange:
    """Tests for require_in_range function."""

    def test_value_in_range(self) -> None:
        """Test that values in range pass through."""
        assert require_in_range(5, "x", min_value=0, max_value=10) == 5
        assert require_in_range(0, "x", min_value=0, max_value=10) == 0
        assert require_in_range(10, "x", min_value=0, max_value=10) == 10

    def test_below_min_raises(self) -> None:
        """Test that values below minimum raise InvalidParameterError."""
        with pytest.raises(InvalidParameterError) as exc_info:
            require_in_range(-1, "x", min_value=0)
        assert ">= 0" in exc_info.value.constraint

    def test_above_max_raises(self) -> None:
        """Test that values above maximum raise InvalidParameterError."""
        with pytest.raises(InvalidParameterError) as exc_info:
            require_in_range(11, "x", max_value=10)
        assert "<= 10" in exc_info.value.constraint

    def test_no_constraints(self) -> None:
        """Test that any value passes with no constraints."""
        assert require_in_range(-100, "x") == -100
        assert require_in_range(1000, "x") == 1000


class TestValidatePositiveDecorator:
    """Tests for validate_positive decorator."""

    def test_decorator_passes_valid_values(self) -> None:
        """Test that decorator allows positive values."""

        @validate_positive("x", "y")
        def func(x: float, y: int) -> float:
            return x + y

        assert func(1.0, 2) == 3.0

    def test_decorator_rejects_zero(self) -> None:
        """Test that decorator rejects zero."""

        @validate_positive("x")
        def func(x: float) -> float:
            return x

        with pytest.raises(InvalidParameterError):
            func(0.0)

    def test_decorator_rejects_negative(self) -> None:
        """Test that decorator rejects negative values."""

        @validate_positive("x")
        def func(x: float) -> float:
            return x

        with pytest.raises(InvalidParameterError):
            func(-1.0)

    def test_decorator_allows_none(self) -> None:
        """Test that decorator allows None values."""

        @validate_positive("x")
        def func(x: float | None = None) -> float | None:
            return x

        assert func(None) is None


class TestValidateNonNegativeDecorator:
    """Tests for validate_non_negative decorator."""

    def test_decorator_passes_valid_values(self) -> None:
        """Test that decorator allows non-negative values."""

        @validate_non_negative("x")
        def func(x: float) -> float:
            return x

        assert func(0.0) == 0.0
        assert func(5.0) == 5.0

    def test_decorator_rejects_negative(self) -> None:
        """Test that decorator rejects negative values."""

        @validate_non_negative("x")
        def func(x: float) -> float:
            return x

        with pytest.raises(InvalidParameterError):
            func(-0.001)


class TestValidateNotEmptyDecorator:
    """Tests for validate_not_empty decorator."""

    def test_decorator_passes_valid_values(self) -> None:
        """Test that decorator allows non-empty sequences."""

        @validate_not_empty("items")
        def func(items: list[int]) -> int:
            return sum(items)

        assert func([1, 2, 3]) == 6

    def test_decorator_rejects_empty(self) -> None:
        """Test that decorator rejects empty sequences."""

        @validate_not_empty("items")
        def func(items: list[int]) -> int:
            return sum(items)

        with pytest.raises(InvalidParameterError):
            func([])


class TestValidateInRangeDecorator:
    """Tests for validate_in_range decorator."""

    def test_decorator_passes_valid_values(self) -> None:
        """Test that decorator allows values in range."""

        @validate_in_range("confidence", min_value=0.0, max_value=1.0)
        def func(confidence: float) -> float:
            return confidence

        assert func(0.5) == 0.5
        assert func(0.0) == 0.0
        assert func(1.0) == 1.0

    def test_decorator_rejects_out_of_range(self) -> None:
        """Test that decorator rejects values out of range."""

        @validate_in_range("confidence", min_value=0.0, max_value=1.0)
        def func(confidence: float) -> float:
            return confidence

        with pytest.raises(InvalidParameterError):
            func(-0.1)
        with pytest.raises(InvalidParameterError):
            func(1.1)


class TestEventAccumulatorAdvanced:
    """Additional tests for EventAccumulator edge cases."""

    def test_process_events_batch(self) -> None:
        """Test batch processing of events."""
        acc = EventAccumulator()
        events = [
            SnakeseeEvent(
                event_type=EventType.JOB_SUBMITTED,
                timestamp=1000.0,
                job_id=1,
                rule_name="align",
            ),
            SnakeseeEvent(
                event_type=EventType.JOB_STARTED,
                timestamp=1001.0,
                job_id=1,
            ),
            SnakeseeEvent(
                event_type=EventType.JOB_FINISHED,
                timestamp=1100.0,
                job_id=1,
                duration=99.0,
            ),
        ]
        acc.process_events(events)

        assert len(acc.jobs) == 1
        assert acc.jobs[1].status == "finished"

    def test_prune_completed_jobs(self) -> None:
        """Test pruning of completed jobs when over limit."""
        acc = EventAccumulator(max_jobs=5)

        # Add 10 completed jobs
        for i in range(10):
            acc.process_event(
                SnakeseeEvent(
                    event_type=EventType.JOB_SUBMITTED,
                    timestamp=float(1000 + i),
                    job_id=i,
                    rule_name="job",
                )
            )
            acc.process_event(
                SnakeseeEvent(
                    event_type=EventType.JOB_FINISHED,
                    timestamp=float(1100 + i),
                    job_id=i,
                )
            )
        acc._prune_if_needed()

        # Should have pruned to max_jobs
        assert len(acc.jobs) <= 5

    def test_submitted_jobs_property(self) -> None:
        """Test submitted_jobs property."""
        acc = EventAccumulator()
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
                event_type=EventType.JOB_SUBMITTED,
                timestamp=1001.0,
                job_id=2,
                rule_name="sort",
            )
        )

        submitted = acc.submitted_jobs
        assert len(submitted) == 2

    def test_finished_jobs_property(self) -> None:
        """Test finished_jobs property."""
        acc = EventAccumulator()
        acc.process_event(
            SnakeseeEvent(
                event_type=EventType.JOB_FINISHED,
                timestamp=1100.0,
                job_id=1,
                rule_name="align",
            )
        )

        finished = acc.finished_jobs
        assert len(finished) == 1


class TestValidationLoggerAdvanced:
    """Additional tests for ValidationLogger edge cases."""

    def test_log_discrepancy_with_wildcards(self, tmp_path: Path) -> None:
        """Test logging discrepancy with wildcards."""
        logger = ValidationLogger(tmp_path)
        logger.log_discrepancy(
            Discrepancy(
                category="timing_mismatch",
                severity="info",
                message="Start time differs",
                wildcards={"sample": "A", "lane": "L001"},
            )
        )
        logger.close()

        log_file = tmp_path / ".snakesee_validation.log"
        content = log_file.read_text()
        assert "sample=A" in content
        assert "lane=L001" in content

    def test_log_discrepancies_batch(self, tmp_path: Path) -> None:
        """Test logging multiple discrepancies."""
        logger = ValidationLogger(tmp_path)
        discrepancies = [
            Discrepancy(category="cat1", severity="warning", message="msg1"),
            Discrepancy(category="cat2", severity="error", message="msg2"),
        ]
        logger.log_discrepancies(discrepancies)
        logger.close()

        log_file = tmp_path / ".snakesee_validation.log"
        content = log_file.read_text()
        assert "cat1" in content
        assert "cat2" in content


class TestCompareStatesAdvanced:
    """Additional tests for compare_states edge cases."""

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

    def test_start_time_mismatch(self, tmp_path: Path) -> None:
        """Test detection of start time mismatch between event and parsed job."""
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

        # Parsed job has different start time (more than 5 seconds difference)
        running_job = JobInfo(rule="align", job_id="42", start_time=1010.0)
        progress = self._make_progress(tmp_path, running=[running_job])

        discrepancies = compare_states(acc, progress)
        timing_disc = [d for d in discrepancies if d.category == "start_time_mismatch"]
        assert len(timing_disc) == 1

    def test_completed_jobs_mismatch(self, tmp_path: Path) -> None:
        """Test detection of completed jobs count mismatch."""
        acc = EventAccumulator()
        acc.workflow_started = True
        acc.completed_jobs = 10

        progress = self._make_progress(tmp_path, completed=5)

        discrepancies = compare_states(acc, progress)
        comp_disc = [d for d in discrepancies if d.category == "completed_jobs"]
        assert len(comp_disc) == 1
        assert comp_disc[0].event_value == 10
        assert comp_disc[0].parsed_value == 5

    def test_failed_count_mismatch(self, tmp_path: Path) -> None:
        """Test detection of failed job count mismatch."""
        acc = EventAccumulator()
        acc.workflow_started = True
        acc.process_event(
            SnakeseeEvent(
                event_type=EventType.JOB_ERROR,
                timestamp=1050.0,
                job_id=1,
                rule_name="sort",
            )
        )
        acc.process_event(
            SnakeseeEvent(
                event_type=EventType.JOB_ERROR,
                timestamp=1060.0,
                job_id=2,
                rule_name="align",
            )
        )

        # Parsed state shows only 1 failed job
        progress = self._make_progress(tmp_path, failed=1)

        discrepancies = compare_states(acc, progress)
        failed_disc = [d for d in discrepancies if d.category == "failed_count"]
        assert len(failed_disc) == 1
        assert failed_disc[0].event_value == 2
        assert failed_disc[0].parsed_value == 1
