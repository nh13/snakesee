"""Tests for EventWriter."""

from pathlib import Path

from snakemake_logger_plugin_snakesee.events import EventType, SnakeseeEvent
from snakemake_logger_plugin_snakesee.writer import EventWriter


class TestEventWriter:
    """Tests for EventWriter class."""

    def test_write_single_event(self, tmp_path: Path) -> None:
        """Test writing a single event."""
        event_file = tmp_path / "events.jsonl"
        event = SnakeseeEvent(
            event_type=EventType.PROGRESS,
            timestamp=1234567890.123,
            completed_jobs=1,
            total_jobs=10,
        )

        with EventWriter(event_file) as writer:
            writer.write(event)

        content = event_file.read_text()
        assert content.strip() == event.to_json()

    def test_write_multiple_events(self, tmp_path: Path) -> None:
        """Test writing multiple events."""
        event_file = tmp_path / "events.jsonl"
        events = [
            SnakeseeEvent(
                event_type=EventType.WORKFLOW_STARTED,
                timestamp=1234567890.0,
            ),
            SnakeseeEvent(
                event_type=EventType.JOB_SUBMITTED,
                timestamp=1234567890.1,
                job_id=1,
                rule_name="align",
            ),
            SnakeseeEvent(
                event_type=EventType.JOB_STARTED,
                timestamp=1234567890.2,
                job_id=1,
            ),
        ]

        with EventWriter(event_file) as writer:
            for event in events:
                writer.write(event)

        lines = event_file.read_text().strip().split("\n")
        assert len(lines) == 3

        for i, line in enumerate(lines):
            parsed = SnakeseeEvent.from_json(line)
            assert parsed.timestamp == events[i].timestamp

    def test_buffering(self, tmp_path: Path) -> None:
        """Test that buffering works correctly."""
        event_file = tmp_path / "events.jsonl"

        # Buffer size of 3 means events are written in batches
        writer = EventWriter(event_file, buffer_size=3)

        # Write 2 events - should not be flushed yet
        writer.write(SnakeseeEvent(event_type=EventType.PROGRESS, timestamp=1.0, total_jobs=10))
        writer.write(SnakeseeEvent(event_type=EventType.PROGRESS, timestamp=2.0, total_jobs=10))

        # File should not exist yet or be empty
        if event_file.exists():
            assert event_file.read_text() == ""

        # Write 3rd event - should trigger flush
        writer.write(SnakeseeEvent(event_type=EventType.PROGRESS, timestamp=3.0, total_jobs=10))

        # Now all 3 events should be written
        lines = event_file.read_text().strip().split("\n")
        assert len(lines) == 3

        writer.close()

    def test_close_flushes_buffer(self, tmp_path: Path) -> None:
        """Test that close() flushes remaining buffered events."""
        event_file = tmp_path / "events.jsonl"

        writer = EventWriter(event_file, buffer_size=10)
        writer.write(SnakeseeEvent(event_type=EventType.PROGRESS, timestamp=1.0, total_jobs=10))

        # Before close, file might not have content
        writer.close()

        # After close, event should be written
        lines = event_file.read_text().strip().split("\n")
        assert len(lines) == 1

    def test_context_manager(self, tmp_path: Path) -> None:
        """Test context manager usage."""
        event_file = tmp_path / "events.jsonl"

        with EventWriter(event_file) as writer:
            writer.write(SnakeseeEvent(event_type=EventType.PROGRESS, timestamp=1.0, total_jobs=10))

        # After exiting context, file should have content
        assert event_file.exists()
        lines = event_file.read_text().strip().split("\n")
        assert len(lines) == 1

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        """Test that parent directories are created if they don't exist."""
        event_file = tmp_path / "nested" / "path" / "events.jsonl"

        with EventWriter(event_file) as writer:
            writer.write(SnakeseeEvent(event_type=EventType.PROGRESS, timestamp=1.0, total_jobs=10))

        assert event_file.exists()

    def test_append_to_existing_file(self, tmp_path: Path) -> None:
        """Test appending to an existing file."""
        event_file = tmp_path / "events.jsonl"

        # Write first event
        with EventWriter(event_file) as writer:
            writer.write(SnakeseeEvent(event_type=EventType.PROGRESS, timestamp=1.0, total_jobs=10))

        # Append second event
        with EventWriter(event_file) as writer:
            writer.write(SnakeseeEvent(event_type=EventType.PROGRESS, timestamp=2.0, total_jobs=10))

        # Both events should be present
        lines = event_file.read_text().strip().split("\n")
        assert len(lines) == 2

    def test_immediate_flush_default(self, tmp_path: Path) -> None:
        """Test that default buffer_size=1 causes immediate flush."""
        event_file = tmp_path / "events.jsonl"

        writer = EventWriter(event_file)  # Default buffer_size=1
        writer.write(SnakeseeEvent(event_type=EventType.PROGRESS, timestamp=1.0, total_jobs=10))

        # Should be immediately written
        lines = event_file.read_text().strip().split("\n")
        assert len(lines) == 1

        writer.close()

    def test_truncate_clears_existing_file(self, tmp_path: Path) -> None:
        """Test that truncate clears existing file content."""
        event_file = tmp_path / "events.jsonl"

        # Write some initial events
        with EventWriter(event_file) as writer:
            writer.write(SnakeseeEvent(event_type=EventType.PROGRESS, timestamp=1.0, total_jobs=10))
            writer.write(SnakeseeEvent(event_type=EventType.PROGRESS, timestamp=2.0, total_jobs=10))

        # Verify initial content
        lines = event_file.read_text().strip().split("\n")
        assert len(lines) == 2

        # Truncate and write new event
        writer = EventWriter(event_file)
        writer.truncate()
        writer.write(SnakeseeEvent(event_type=EventType.WORKFLOW_STARTED, timestamp=3.0))
        writer.close()

        # Only new event should be present
        lines = event_file.read_text().strip().split("\n")
        assert len(lines) == 1
        parsed = SnakeseeEvent.from_json(lines[0])
        assert parsed.event_type == EventType.WORKFLOW_STARTED
        assert parsed.timestamp == 3.0
