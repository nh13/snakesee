"""Type aliases for common callback and data patterns.

This module provides centralized type definitions for commonly used
callback signatures and data structures throughout the snakesee codebase.
"""

from collections.abc import Callable
from typing import NamedTuple

# Callback for reporting progress during long-running operations.
# Args: (current_item: int, total_items: int)
ProgressCallback = Callable[[int, int], None]


class TimingRecord(NamedTuple):
    """A single timing measurement with its timestamp.

    Attributes:
        duration: Duration of the operation in seconds.
        timestamp: Unix timestamp when the operation completed.
    """

    duration: float
    timestamp: float
