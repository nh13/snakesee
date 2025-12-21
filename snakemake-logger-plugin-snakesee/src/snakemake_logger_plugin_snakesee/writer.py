"""Event writer for snakesee logger plugin."""

import fcntl
import os
from pathlib import Path
from typing import TextIO

from snakemake_logger_plugin_snakesee.events import SnakeseeEvent


class EventWriter:
    """Thread-safe JSONL event writer with file locking.

    Writes events to a JSON Lines file with proper file locking
    to support concurrent access from multiple processes.

    Attributes:
        path: Path to the event file.
        buffer_size: Number of events to buffer before flushing.
    """

    def __init__(self, path: Path, buffer_size: int = 1) -> None:
        """Initialize the event writer.

        Args:
            path: Path to the event file.
            buffer_size: Number of events to buffer before flushing.
                Default is 1 (immediate flush).
        """
        self.path = path
        self.buffer_size = buffer_size
        self._buffer: list[SnakeseeEvent] = []
        self._file: TextIO | None = None

    def _ensure_open(self) -> TextIO:
        """Ensure the file is open for appending.

        Returns:
            The open file handle.
        """
        if self._file is None:
            # Ensure parent directory exists
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._file = open(self.path, "a", encoding="utf-8")
        return self._file

    def truncate(self) -> None:
        """Truncate the event file to clear stale data from previous runs.

        Should be called when a new workflow starts to ensure fresh state.
        """
        # Close existing handle if open
        if self._file is not None:
            self._file.close()
            self._file = None

        # Truncate by opening in write mode
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.path, "w", encoding="utf-8")

    def write(self, event: SnakeseeEvent) -> None:
        """Write an event to the file.

        Events are buffered and flushed when the buffer is full.

        Args:
            event: The event to write.
        """
        self._buffer.append(event)
        if len(self._buffer) >= self.buffer_size:
            self.flush()

    def flush(self) -> None:
        """Flush buffered events to disk.

        Uses file locking to ensure safe concurrent access.
        """
        if not self._buffer:
            return

        f = self._ensure_open()

        # Use file locking for safe concurrent access
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            for event in self._buffer:
                f.write(event.to_json() + "\n")
            f.flush()
            os.fsync(f.fileno())
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)

        self._buffer.clear()

    def close(self) -> None:
        """Close the file and flush any remaining events."""
        self.flush()
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self) -> "EventWriter":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        """Context manager exit."""
        self.close()
