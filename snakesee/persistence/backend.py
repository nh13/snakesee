"""Persistence backend protocol and auto-detection."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Protocol

if TYPE_CHECKING:
    from snakesee.types import ProgressCallback

from snakesee.parser.metadata import MetadataRecord


class PersistenceBackend(Protocol):
    """Protocol for reading Snakemake workflow metadata.

    Implementations provide access to job metadata, incomplete markers,
    and lock state from either the filesystem or SQLite backends.
    """

    def iterate_metadata(
        self,
        progress_callback: ProgressCallback | None = None,
    ) -> Iterator[MetadataRecord]:
        """Iterate over all job metadata records.

        Args:
            progress_callback: Optional callback(current, total) for progress.

        Yields:
            MetadataRecord for each completed job.
        """
        ...

    def has_incomplete_jobs(self) -> bool:
        """Check whether any jobs are currently marked incomplete."""
        ...

    def iterate_incomplete_jobs(
        self,
        min_start_time: float | None = None,
    ) -> Iterator[IncompleteJob]:
        """Iterate over incomplete (in-progress) job markers.

        Args:
            min_start_time: If set, only yield jobs started at or after this time.

        Yields:
            IncompleteJob for each in-progress job.
        """
        ...

    def has_locks(self) -> bool:
        """Check whether any workflow locks are held."""
        ...


@dataclass(frozen=True, slots=True)
class IncompleteJob:
    """A job marked as incomplete/in-progress.

    Attributes:
        start_time: Approximate start time (mtime for FS, starttime for DB).
        output_file: Output file path if known.
        rule: Rule name if known (DB backend provides this; FS does not).
        external_jobid: External executor job ID if known.
    """

    start_time: float | None = None
    output_file: Path | None = None
    rule: str | None = None
    external_jobid: str | None = None


def detect_backend(workflow_dir: Path) -> PersistenceBackend:
    """Auto-detect and return the appropriate persistence backend.

    Prefers the SQLite DB backend when .snakemake/metadata.db exists.
    Falls back to filesystem backend otherwise.

    Args:
        workflow_dir: Root workflow directory containing .snakemake/.

    Returns:
        A PersistenceBackend implementation.
    """
    from snakesee.state.paths import WorkflowPaths

    paths = WorkflowPaths(workflow_dir)

    if paths.metadata_db.exists():
        from snakesee.persistence.db import DbPersistence

        return DbPersistence(paths)

    from snakesee.persistence.fs import FsPersistence

    return FsPersistence(paths)
