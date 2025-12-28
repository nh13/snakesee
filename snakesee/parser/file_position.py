"""File position tracking with rotation detection."""

from pathlib import Path


class LogFilePosition:
    """Tracks file position and detects log rotation/truncation.

    Handles common scenarios like log file rotation (inode change)
    and file truncation (size decrease).

    Attributes:
        path: Path to the file being tracked.
    """

    __slots__ = ("path", "_offset", "_inode")

    def __init__(self, path: Path) -> None:
        """Initialize position tracker.

        Args:
            path: Path to the file to track.
        """
        self.path = path
        self._offset: int = 0
        self._inode: int | None = None

    @property
    def offset(self) -> int:
        """Current read position in the file."""
        return self._offset

    @offset.setter
    def offset(self, value: int) -> None:
        """Set the current read position."""
        self._offset = value

    def check_rotation(self) -> bool:
        """Check if the file has been rotated or truncated.

        Detects rotation by checking for inode change or file truncation
        (current size less than last known position).

        Returns:
            True if the file was rotated and position was reset.
        """
        try:
            stat = self.path.stat()
            current_inode = stat.st_ino
            current_size = stat.st_size

            if self._inode is not None and (
                current_inode != self._inode or current_size < self._offset
            ):
                self.reset()
                return True

            self._inode = current_inode
            return False
        except FileNotFoundError:
            return False
        except OSError:
            return False

    def reset(self) -> None:
        """Reset position to start of file."""
        self._offset = 0
        self._inode = None

    def clamp_to_size(self, file_size: int) -> None:
        """Clamp offset to file bounds.

        Args:
            file_size: Current file size (must be non-negative).
        """
        if file_size < 0:
            file_size = 0
        self._offset = min(self._offset, file_size)
