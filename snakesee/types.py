"""Type aliases for common callback and data patterns.

This module provides centralized type definitions for commonly used
callback signatures and data structures throughout the snakesee codebase.
"""

from collections.abc import Callable
from typing import Any
from typing import NamedTuple
from typing import TypedDict

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


# =============================================================================
# TypedDict classes for Snakemake log handler messages
# =============================================================================


class SnakemakeLogMessage(TypedDict, total=False):
    """Base structure for Snakemake log handler messages.

    All fields are optional since different log levels provide different data.
    """

    level: str
    jobid: int
    name: str
    rule: str
    wildcards: Any  # Can be object with __dict__ or dict
    threads: int
    resources: Any  # Can be object with __dict__ or dict
    input: list[str] | tuple[str, ...]
    output: list[str] | tuple[str, ...]
    done: int
    total: int
    msg: str
    message: str


class JobEventData(TypedDict, total=False):
    """Data structure for job-related events."""

    event_type: str
    timestamp: float
    job_id: int | None
    rule_name: str | None
    wildcards: dict[str, str] | None
    threads: int | None
    resources: dict[str, Any] | None
    input_files: list[str] | None
    output_files: list[str] | None
    duration: float | None
    error_message: str | None


class ProgressEventData(TypedDict):
    """Data structure for progress events."""

    event_type: str
    timestamp: float
    completed_jobs: int
    total_jobs: int


class WorkflowEventData(TypedDict):
    """Data structure for workflow lifecycle events."""

    event_type: str
    timestamp: float


# =============================================================================
# TypedDict classes for metadata and parsing
# =============================================================================


class MetadataDict(TypedDict, total=False):
    """Structure of Snakemake metadata JSON files."""

    starttime: float
    endtime: float
    rule: str
    input: list[str]
    output: list[str]
    params: dict[str, Any]
    wildcards: dict[str, str]
    resources: dict[str, Any]
    code: str
    version: str
    log: list[str]


class EventDict(TypedDict, total=False):
    """Structure of snakesee event records in .snakesee_events.jsonl."""

    event_type: str
    timestamp: float
    job_id: int | None
    rule_name: str | None
    wildcards: dict[str, str] | None
    threads: int | None
    resources: dict[str, Any] | None
    input_files: list[str] | None
    output_files: list[str] | None
    duration: float | None
    error_message: str | None
    completed_jobs: int | None
    total_jobs: int | None


# =============================================================================
# TypedDict classes for estimation
# =============================================================================


class EstimateResult(TypedDict):
    """Result of a time estimation calculation."""

    expected: float
    variance: float
    method: str
    confidence: float
