"""Tests for snakesee event reading."""

from pathlib import Path

import pytest

from snakesee.events import EVENT_FILE_NAME
from snakesee.events import EventReader
from snakesee.events import EventType
from snakesee.events import SnakeseeEvent
from snakesee.events import get_event_file_path


class TestEventType:
    """Tests for EventType enum."""

    def test_event_type_values(self) -> None:
        """Test that event types have expected string values."""
        assert EventType.WORKFLOW_STARTED.value == "workflow_started"
        assert EventType.JOB_SUBMITTED.value == "job_submitted"
        assert EventType.JOB_STARTED.value == "job_started"
        assert EventType.JOB_FINISHED.value == "job_finished"
        assert EventType.JOB_ERROR.value == "job_error"
        assert EventType.PROGRESS.value == "progress"

    def test_event_type_is_string(self) -> None:
        """Test that EventType inherits from str."""
        assert isinstance(EventType.PROGRESS, str)
        assert EventType.PROGRESS == "progress"


class TestSnakeseeEvent:
    """Tests for SnakeseeEvent dataclass."""

    def test_minimal_event(self) -> None:
        """Test creating event with only required fields."""
        event = SnakeseeEvent(
            event_type=EventType.PROGRESS,
            timestamp=1234567890.123,
        )
        assert event.event_type == EventType.PROGRESS
        assert event.timestamp == 1234567890.123
        assert event.job_id is None

    def test_frozen_dataclass(self) -> None:
        """Test that SnakeseeEvent is frozen (immutable)."""
        event = SnakeseeEvent(
            event_type=EventType.PROGRESS,
            timestamp=1234567890.123,
        )
        with pytest.raises(AttributeError):
            event.timestamp = 0.0  # type: ignore[misc]

    def test_from_json_minimal(self) -> None:
        """Test JSON deserialization with minimal fields."""
        json_str = '{"event_type":"progress","timestamp":1234567890.123}'
        event = SnakeseeEvent.from_json(json_str)

        assert event.event_type == EventType.PROGRESS
        assert event.timestamp == 1234567890.123
        assert event.job_id is None

    def test_from_json_with_wildcards(self) -> None:
        """Test JSON deserialization with wildcards."""
        json_str = (
            '{"event_type":"job_submitted","timestamp":1234567890.123,'
            '"job_id":42,"rule_name":"align","wildcards":{"sample":"A","lane":"1"}}'
        )
        event = SnakeseeEvent.from_json(json_str)

        assert event.event_type == EventType.JOB_SUBMITTED
        assert event.job_id == 42
        assert event.rule_name == "align"
        # Wildcards are converted to sorted tuples
        assert event.wildcards is not None
        assert event.wildcards_dict == {"sample": "A", "lane": "1"}

    def test_from_json_with_files(self) -> None:
        """Test JSON deserialization with file lists."""
        json_str = (
            '{"event_type":"job_submitted","timestamp":1234567890.123,'
            '"input_files":["a.txt","b.txt"],"output_files":["c.txt"]}'
        )
        event = SnakeseeEvent.from_json(json_str)

        assert event.input_files == ("a.txt", "b.txt")
        assert event.output_files == ("c.txt",)

    def test_from_json_with_resources(self) -> None:
        """Test JSON deserialization with resources."""
        json_str = (
            '{"event_type":"job_submitted","timestamp":1234567890.123,'
            '"resources":{"threads":4,"mem_mb":8000}}'
        )
        event = SnakeseeEvent.from_json(json_str)

        assert event.resources == (("mem_mb", 8000), ("threads", 4))

    def test_from_json_invalid_event_type(self) -> None:
        """Test that invalid event type raises ValueError."""
        json_str = '{"event_type":"invalid","timestamp":1234567890.123}'
        with pytest.raises(ValueError):
            SnakeseeEvent.from_json(json_str)

    def test_wildcards_dict_none(self) -> None:
        """Test wildcards_dict property when wildcards is None."""
        event = SnakeseeEvent(
            event_type=EventType.PROGRESS,
            timestamp=1234567890.123,
        )
        assert event.wildcards_dict is None

    def test_wildcards_dict_conversion(self) -> None:
        """Test wildcards_dict property converts tuples to dict."""
        event = SnakeseeEvent(
            event_type=EventType.JOB_SUBMITTED,
            timestamp=1234567890.123,
            wildcards=(("lane", "1"), ("sample", "A")),
        )
        assert event.wildcards_dict == {"lane": "1", "sample": "A"}


class TestEventReader:
    """Tests for EventReader class."""

    def test_read_nonexistent_file(self, tmp_path: Path) -> None:
        """Test reading from nonexistent file returns empty list."""
        event_file = tmp_path / "events.jsonl"
        reader = EventReader(event_file)

        events = reader.read_new_events()
        assert events == []

    def test_read_empty_file(self, tmp_path: Path) -> None:
        """Test reading from empty file returns empty list."""
        event_file = tmp_path / "events.jsonl"
        event_file.touch()

        reader = EventReader(event_file)
        events = reader.read_new_events()
        assert events == []

    def test_read_single_event(self, tmp_path: Path) -> None:
        """Test reading a single event."""
        event_file = tmp_path / "events.jsonl"
        event_file.write_text(
            '{"event_type":"progress","timestamp":1234567890.123,"total_jobs":10}\n'
        )

        reader = EventReader(event_file)
        events = reader.read_new_events()

        assert len(events) == 1
        assert events[0].event_type == EventType.PROGRESS
        assert events[0].total_jobs == 10

    def test_read_multiple_events(self, tmp_path: Path) -> None:
        """Test reading multiple events."""
        event_file = tmp_path / "events.jsonl"
        event_file.write_text(
            '{"event_type":"workflow_started","timestamp":1.0}\n'
            '{"event_type":"job_submitted","timestamp":2.0,"job_id":1}\n'
            '{"event_type":"job_started","timestamp":3.0,"job_id":1}\n'
        )

        reader = EventReader(event_file)
        events = reader.read_new_events()

        assert len(events) == 3
        assert events[0].event_type == EventType.WORKFLOW_STARTED
        assert events[1].event_type == EventType.JOB_SUBMITTED
        assert events[2].event_type == EventType.JOB_STARTED

    def test_incremental_reading(self, tmp_path: Path) -> None:
        """Test that reader only returns new events on subsequent calls."""
        event_file = tmp_path / "events.jsonl"

        # Write initial events
        with open(event_file, "w") as f:
            f.write('{"event_type":"progress","timestamp":1.0,"total_jobs":10}\n')

        reader = EventReader(event_file)
        events = reader.read_new_events()
        assert len(events) == 1

        # Second read should return empty (no new events)
        events = reader.read_new_events()
        assert len(events) == 0

        # Append new event
        with open(event_file, "a") as f:
            f.write('{"event_type":"job_started","timestamp":2.0,"job_id":1}\n')

        # Should only get the new event
        events = reader.read_new_events()
        assert len(events) == 1
        assert events[0].event_type == EventType.JOB_STARTED

    def test_skip_malformed_lines(self, tmp_path: Path) -> None:
        """Test that malformed JSON lines are skipped."""
        event_file = tmp_path / "events.jsonl"
        event_file.write_text(
            '{"event_type":"progress","timestamp":1.0}\n'
            "not valid json\n"
            '{"invalid": "missing required fields"}\n'
            '{"event_type":"job_started","timestamp":3.0,"job_id":1}\n'
        )

        reader = EventReader(event_file)
        events = reader.read_new_events()

        # Should skip the invalid lines and return the valid ones
        assert len(events) == 2
        assert events[0].event_type == EventType.PROGRESS
        assert events[1].event_type == EventType.JOB_STARTED

    def test_skip_empty_lines(self, tmp_path: Path) -> None:
        """Test that empty lines are skipped."""
        event_file = tmp_path / "events.jsonl"
        event_file.write_text(
            '{"event_type":"progress","timestamp":1.0}\n'
            "\n"
            "   \n"
            '{"event_type":"job_started","timestamp":2.0,"job_id":1}\n'
        )

        reader = EventReader(event_file)
        events = reader.read_new_events()

        assert len(events) == 2

    def test_reset(self, tmp_path: Path) -> None:
        """Test that reset allows re-reading all events."""
        event_file = tmp_path / "events.jsonl"
        event_file.write_text(
            '{"event_type":"progress","timestamp":1.0}\n'
            '{"event_type":"job_started","timestamp":2.0,"job_id":1}\n'
        )

        reader = EventReader(event_file)

        # Read all events
        events = reader.read_new_events()
        assert len(events) == 2

        # Second read should be empty
        events = reader.read_new_events()
        assert len(events) == 0

        # Reset and read again
        reader.reset()
        events = reader.read_new_events()
        assert len(events) == 2

    def test_has_events_nonexistent(self, tmp_path: Path) -> None:
        """Test has_events for nonexistent file."""
        event_file = tmp_path / "events.jsonl"
        reader = EventReader(event_file)
        assert reader.has_events is False

    def test_has_events_empty(self, tmp_path: Path) -> None:
        """Test has_events for empty file."""
        event_file = tmp_path / "events.jsonl"
        event_file.touch()
        reader = EventReader(event_file)
        assert reader.has_events is False

    def test_has_events_with_content(self, tmp_path: Path) -> None:
        """Test has_events for file with content."""
        event_file = tmp_path / "events.jsonl"
        event_file.write_text('{"event_type":"progress","timestamp":1.0}\n')
        reader = EventReader(event_file)
        assert reader.has_events is True


class TestGetEventFilePath:
    """Tests for get_event_file_path function."""

    def test_returns_correct_path(self, tmp_path: Path) -> None:
        """Test that correct event file path is returned."""
        path = get_event_file_path(tmp_path)
        assert path == tmp_path / EVENT_FILE_NAME
        assert path.name == ".snakesee_events.jsonl"
