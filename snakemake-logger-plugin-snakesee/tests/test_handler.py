"""Tests for LogHandler."""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from snakemake_logger_plugin_snakesee.events import EventType, SnakeseeEvent
from snakemake_logger_plugin_snakesee.handler import LogEvent, LogHandler
from snakemake_logger_plugin_snakesee.settings import LogHandlerSettings


def create_mock_record(**kwargs: Any) -> MagicMock:
    """Create a mock log record with the given attributes."""
    record = MagicMock()
    for key, value in kwargs.items():
        setattr(record, key, value)
    return record


def create_handler(tmp_path: Path) -> LogHandler:
    """Create a LogHandler with test settings."""
    settings = LogHandlerSettings(event_file=Path(".snakesee_events.jsonl"))
    common_settings = MagicMock()
    common_settings.directory = tmp_path
    return LogHandler(settings=settings, common_settings=common_settings)


def read_events(tmp_path: Path) -> list[SnakeseeEvent]:
    """Read all events from the event file."""
    event_file = tmp_path / ".snakesee_events.jsonl"
    if not event_file.exists():
        return []
    events = []
    for line in event_file.read_text().strip().split("\n"):
        if line:
            events.append(SnakeseeEvent.from_json(line))
    return events


class TestLogHandler:
    """Tests for LogHandler class."""

    def test_handler_properties(self, tmp_path: Path) -> None:
        """Test handler property values."""
        handler = create_handler(tmp_path)
        assert handler.writes_to_stream is False
        assert handler.writes_to_file is True
        assert handler.has_filter is True
        assert handler.has_formatter is False
        assert handler.needs_rulegraph is False
        handler.close()

    def test_emit_ignores_none_event(self, tmp_path: Path) -> None:
        """Test that records without event attribute are ignored."""
        handler = create_handler(tmp_path)
        record = create_mock_record()
        del record.event  # No event attribute

        handler.emit(record)
        handler.close()

        events = read_events(tmp_path)
        assert len(events) == 0

    def test_emit_job_info(self, tmp_path: Path) -> None:
        """Test handling JOB_INFO event."""
        handler = create_handler(tmp_path)

        # Create mock wildcards object
        wildcards = MagicMock()
        wildcards.__dict__ = {"sample": "A", "lane": "1"}

        record = create_mock_record(
            event=LogEvent.JOB_INFO,
            jobid=42,
            name="align",
            wildcards=wildcards,
            threads=4,
            input=["input.fastq"],
            output=["output.bam"],
        )

        handler.emit(record)
        handler.close()

        events = read_events(tmp_path)
        assert len(events) == 1
        assert events[0].event_type == EventType.JOB_SUBMITTED
        assert events[0].job_id == 42
        assert events[0].rule_name == "align"
        assert events[0].wildcards == {"sample": "A", "lane": "1"}
        assert events[0].threads == 4
        assert events[0].input_files == ["input.fastq"]
        assert events[0].output_files == ["output.bam"]

    def test_emit_job_started(self, tmp_path: Path) -> None:
        """Test handling JOB_STARTED event."""
        handler = create_handler(tmp_path)

        record = create_mock_record(
            event=LogEvent.JOB_STARTED,
            job_ids=[42, 43],
        )

        handler.emit(record)
        handler.close()

        events = read_events(tmp_path)
        assert len(events) == 2
        assert events[0].event_type == EventType.JOB_STARTED
        assert events[0].job_id == 42
        assert events[1].job_id == 43

    def test_emit_job_finished(self, tmp_path: Path) -> None:
        """Test handling JOB_FINISHED event with duration calculation."""
        handler = create_handler(tmp_path)

        # First emit JOB_INFO to register the job
        info_record = create_mock_record(
            event=LogEvent.JOB_INFO,
            jobid=42,
            name="align",
        )
        handler.emit(info_record)

        # Then emit JOB_STARTED to record start time
        start_record = create_mock_record(
            event=LogEvent.JOB_STARTED,
            job_ids=[42],
        )
        handler.emit(start_record)

        # Then emit JOB_FINISHED
        finish_record = create_mock_record(
            event=LogEvent.JOB_FINISHED,
            jobid=42,
        )
        handler.emit(finish_record)
        handler.close()

        events = read_events(tmp_path)
        assert len(events) == 3

        finished_event = events[2]
        assert finished_event.event_type == EventType.JOB_FINISHED
        assert finished_event.job_id == 42
        assert finished_event.rule_name == "align"
        # Duration should be calculated (will be very small in tests)
        assert finished_event.duration is not None
        assert finished_event.duration >= 0

    def test_emit_job_error(self, tmp_path: Path) -> None:
        """Test handling JOB_ERROR event."""
        handler = create_handler(tmp_path)

        record = create_mock_record(
            event=LogEvent.JOB_ERROR,
            jobid=42,
            name="sort",
            msg="Sort failed: out of memory",
        )

        handler.emit(record)
        handler.close()

        events = read_events(tmp_path)
        assert len(events) == 1
        assert events[0].event_type == EventType.JOB_ERROR
        assert events[0].job_id == 42
        assert events[0].rule_name == "sort"
        assert events[0].error_message == "Sort failed: out of memory"

    def test_emit_progress(self, tmp_path: Path) -> None:
        """Test handling PROGRESS event."""
        handler = create_handler(tmp_path)

        record = create_mock_record(
            event=LogEvent.PROGRESS,
            done=5,
            total=10,
        )

        handler.emit(record)
        handler.close()

        events = read_events(tmp_path)
        assert len(events) == 1
        assert events[0].event_type == EventType.PROGRESS
        assert events[0].completed_jobs == 5
        assert events[0].total_jobs == 10

    def test_emit_workflow_started(self, tmp_path: Path) -> None:
        """Test handling WORKFLOW_STARTED event."""
        handler = create_handler(tmp_path)

        record = create_mock_record(
            event=LogEvent.WORKFLOW_STARTED,
            workflow_id="abc123",
        )

        handler.emit(record)
        handler.close()

        events = read_events(tmp_path)
        assert len(events) == 1
        assert events[0].event_type == EventType.WORKFLOW_STARTED
        assert events[0].workflow_id == "abc123"

    def test_wildcards_as_dict(self, tmp_path: Path) -> None:
        """Test extracting wildcards when they're a plain dict."""
        handler = create_handler(tmp_path)

        record = create_mock_record(
            event=LogEvent.JOB_INFO,
            jobid=42,
            name="align",
            wildcards={"sample": "A"},
        )

        handler.emit(record)
        handler.close()

        events = read_events(tmp_path)
        assert events[0].wildcards == {"sample": "A"}

    def test_full_workflow_sequence(self, tmp_path: Path) -> None:
        """Test a realistic sequence of workflow events."""
        handler = create_handler(tmp_path)

        # Workflow started
        handler.emit(create_mock_record(event=LogEvent.WORKFLOW_STARTED))

        # Progress: 0/2
        handler.emit(create_mock_record(event=LogEvent.PROGRESS, done=0, total=2))

        # Job 1 submitted
        handler.emit(
            create_mock_record(
                event=LogEvent.JOB_INFO,
                jobid=1,
                name="align",
                wildcards={"sample": "A"},
            )
        )

        # Job 1 started
        handler.emit(create_mock_record(event=LogEvent.JOB_STARTED, job_ids=[1]))

        # Job 1 finished
        handler.emit(create_mock_record(event=LogEvent.JOB_FINISHED, jobid=1))

        # Progress: 1/2
        handler.emit(create_mock_record(event=LogEvent.PROGRESS, done=1, total=2))

        # Job 2 submitted
        handler.emit(
            create_mock_record(
                event=LogEvent.JOB_INFO,
                jobid=2,
                name="sort",
                wildcards={"sample": "A"},
            )
        )

        # Job 2 started
        handler.emit(create_mock_record(event=LogEvent.JOB_STARTED, job_ids=[2]))

        # Job 2 finished
        handler.emit(create_mock_record(event=LogEvent.JOB_FINISHED, jobid=2))

        # Progress: 2/2
        handler.emit(create_mock_record(event=LogEvent.PROGRESS, done=2, total=2))

        handler.close()

        events = read_events(tmp_path)
        assert len(events) == 10

        # Check event sequence
        assert events[0].event_type == EventType.WORKFLOW_STARTED
        assert events[1].event_type == EventType.PROGRESS
        assert events[2].event_type == EventType.JOB_SUBMITTED
        assert events[3].event_type == EventType.JOB_STARTED
        assert events[4].event_type == EventType.JOB_FINISHED
        assert events[5].event_type == EventType.PROGRESS
