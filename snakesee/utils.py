"""Shared utility functions for snakesee.

This module consolidates common utilities used across multiple modules
to avoid duplication and ensure consistent behavior.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any

if TYPE_CHECKING:
    from snakesee.types import ProgressCallback

logger = logging.getLogger(__name__)


def safe_mtime(path: Path) -> float:
    """Get file modification time, returning 0.0 if file doesn't exist.

    This handles the common race condition where a file may be deleted
    between checking for existence and reading its mtime.

    Args:
        path: Path to the file.

    Returns:
        The file's modification time as a Unix timestamp, or 0.0 if the
        file doesn't exist.
    """
    try:
        return path.stat().st_mtime
    except FileNotFoundError:
        return 0.0


def iterate_metadata_files(
    metadata_dir: Path,
    progress_callback: ProgressCallback | None = None,
) -> Iterator[tuple[Path, dict[str, Any]]]:
    """Iterate metadata files with optional progress reporting.

    Iterates over all files in the metadata directory, parsing each as JSON.
    Invalid files (non-JSON or unreadable) are silently skipped with debug logging.

    Args:
        metadata_dir: Path to .snakemake/metadata/ directory.
        progress_callback: Optional callback(current, total) for progress reporting.
            If provided, files are pre-enumerated for accurate total count.

    Yields:
        Tuples of (file_path, parsed_json_data) for each valid metadata file.
    """
    if not metadata_dir.exists():
        return

    # Get file list upfront if progress is requested for accurate reporting
    if progress_callback is not None:
        files = [f for f in metadata_dir.rglob("*") if f.is_file()]
        total = len(files)
    else:
        files = None
        total = 0

    file_iter = files if files is not None else metadata_dir.rglob("*")

    for i, meta_file in enumerate(file_iter):
        if files is None and not meta_file.is_file():
            continue

        if progress_callback is not None:
            progress_callback(i + 1, total)

        try:
            data = json.loads(meta_file.read_text())
            yield meta_file, data
        except json.JSONDecodeError as e:
            logger.debug("Malformed JSON in metadata file %s: %s", meta_file, e)
            continue
        except OSError as e:
            logger.debug("Error reading metadata file %s: %s", meta_file, e)
            continue
