"""Filesystem persistence backend.

Reads Snakemake metadata from the traditional .snakemake/metadata/,
.snakemake/incomplete/, and .snakemake/locks/ directory layout.
"""

from __future__ import annotations

import base64
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

from snakesee.parser.metadata import MetadataRecord
from snakesee.parser.metadata import parse_metadata_files_full
from snakesee.persistence.backend import IncompleteJob

if TYPE_CHECKING:
    from snakesee.state.paths import WorkflowPaths
    from snakesee.types import ProgressCallback

logger = logging.getLogger(__name__)


class FsPersistence:
    """Filesystem-based persistence backend.

    Wraps the existing metadata file iteration, incomplete marker scanning,
    and lock file detection into the PersistenceBackend interface.
    """

    def __init__(self, paths: WorkflowPaths) -> None:
        self._paths = paths

    def iterate_metadata(
        self,
        progress_callback: ProgressCallback | None = None,
    ) -> Iterator[MetadataRecord]:
        """Iterate over metadata files in .snakemake/metadata/."""
        if not self._paths.has_metadata:
            return
        yield from parse_metadata_files_full(self._paths.metadata_dir, progress_callback)

    def has_locks(self) -> bool:
        """Check for lock files in .snakemake/locks/."""
        locks_dir = self._paths.locks_dir
        if not locks_dir.exists():
            return False
        try:
            return any(locks_dir.iterdir())
        except OSError:
            return False

    def has_incomplete_jobs(self) -> bool:
        """Check for incomplete markers in .snakemake/incomplete/."""
        inc_dir = self._paths.incomplete_dir
        if not inc_dir.exists():
            return False
        try:
            return any(
                marker.is_file() and marker.name != "migration_underway"
                for marker in inc_dir.rglob("*")
            )
        except OSError:
            return False

    def iterate_incomplete_jobs(
        self,
        min_start_time: float | None = None,
    ) -> Iterator[IncompleteJob]:
        """Iterate over incomplete markers in .snakemake/incomplete/.

        Each marker filename is a base64-encoded output file path.
        The marker's mtime approximates when the job started.
        """
        inc_dir = self._paths.incomplete_dir
        if not inc_dir.exists():
            return

        for marker in inc_dir.rglob("*"):
            if not marker.is_file() or marker.name == "migration_underway":
                continue

            try:
                marker_mtime = marker.stat().st_mtime
            except OSError:
                continue

            if min_start_time is not None and marker_mtime < min_start_time:
                continue

            output_file: Path | None = None
            try:
                decoded = base64.b64decode(marker.name).decode("utf-8")
                output_file = Path(decoded)
            except (ValueError, UnicodeDecodeError) as e:
                logger.warning("Failed to decode base64 marker filename %s: %s", marker.name, e)

            yield IncompleteJob(
                start_time=marker_mtime,
                output_file=output_file,
            )
