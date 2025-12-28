"""Metadata file parsing for Snakemake workflows.

This module handles parsing of .snakemake/metadata/ files which contain
information about completed jobs, including timing, wildcards, and code.

Note: Currently (Snakemake <= 8.x), metadata files do NOT store wildcards.
Wildcards are only available from live log events during the current session.
This means combination-based estimates (wildcard+threads) only work for jobs
that ran in the current session.

TODO: Once https://github.com/snakemake/snakemake/pull/3888 is merged and
released, metadata files will include wildcards, enabling historical
combination-based estimates across sessions.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from snakesee.models import JobInfo
from snakesee.utils import iterate_metadata_files

if TYPE_CHECKING:
    from snakesee.types import ProgressCallback

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class MetadataRecord:
    """Single metadata file parsed data for efficient single-pass collection.

    Contains all fields needed by various collection functions so we only
    read each metadata file once.
    """

    rule: str
    start_time: float | None = None
    end_time: float | None = None
    wildcards: dict[str, str] | None = None
    input_size: int | None = None
    code_hash: str | None = None

    @property
    def duration(self) -> float | None:
        """Calculate duration from start and end times."""
        if self.start_time is not None and self.end_time is not None:
            return self.end_time - self.start_time
        return None

    def to_job_info(self) -> JobInfo:
        """Convert to JobInfo for compatibility with existing code."""
        return JobInfo(
            rule=self.rule,
            start_time=self.start_time,
            end_time=self.end_time,
            wildcards=self.wildcards,
            input_size=self.input_size,
        )


def _calculate_input_size(input_files: list[str] | None) -> int | None:
    """Calculate total input size from file list.

    Args:
        input_files: List of input file paths from metadata.

    Returns:
        Total size in bytes, or None if not a valid list or any file is missing.
    """
    if not isinstance(input_files, list) or not input_files:
        return None

    total_size = 0
    for f in input_files:
        try:
            total_size += Path(f).stat().st_size
        except OSError:
            return None
    return total_size


def parse_metadata_files(
    metadata_dir: Path,
    progress_callback: ProgressCallback | None = None,
) -> Iterator[JobInfo]:
    """
    Parse completed job information from Snakemake metadata files.

    Reads JSON metadata files from .snakemake/metadata/ to extract
    timing information for completed jobs, including input file sizes.

    Args:
        metadata_dir: Path to .snakemake/metadata/ directory.
        progress_callback: Optional callback(current, total) for progress reporting.

    Yields:
        JobInfo instances for each completed job found.
    """
    for _path, data in iterate_metadata_files(metadata_dir, progress_callback):
        rule = data.get("rule")
        starttime = data.get("starttime")
        endtime = data.get("endtime")

        if rule is not None and starttime is not None and endtime is not None:
            # Extract wildcards if present (Snakemake stores as dict)
            wildcards_data = data.get("wildcards")
            wildcards: dict[str, str] | None = None
            if isinstance(wildcards_data, dict):
                wildcards = {str(k): str(v) for k, v in wildcards_data.items()}

            yield JobInfo(
                rule=rule,
                start_time=starttime,
                end_time=endtime,
                wildcards=wildcards,
                input_size=_calculate_input_size(data.get("input")),
            )


def parse_metadata_files_full(
    metadata_dir: Path,
    progress_callback: ProgressCallback | None = None,
) -> Iterator[MetadataRecord]:
    """
    Parse all metadata from Snakemake metadata files in a single pass.

    This is more efficient than calling parse_metadata_files and
    collect_rule_code_hashes separately, as it reads each file only once.

    Args:
        metadata_dir: Path to .snakemake/metadata/ directory.
        progress_callback: Optional callback(current, total) for progress reporting.

    Yields:
        MetadataRecord instances containing timing and code hash data.
    """
    for _path, data in iterate_metadata_files(metadata_dir, progress_callback):
        rule = data.get("rule")
        if rule is None:
            continue

        # Extract timing data
        starttime = data.get("starttime")
        endtime = data.get("endtime")

        # Extract wildcards if present
        wildcards_data = data.get("wildcards")
        wildcards: dict[str, str] | None = None
        if isinstance(wildcards_data, dict):
            wildcards = {str(k): str(v) for k, v in wildcards_data.items()}

        # Extract and hash code
        code_hash: str | None = None
        code = data.get("code")
        if code:
            normalized_code = " ".join(code.split())
            code_hash = hashlib.sha256(normalized_code.encode()).hexdigest()[:16]

        yield MetadataRecord(
            rule=rule,
            start_time=starttime,
            end_time=endtime,
            wildcards=wildcards,
            input_size=_calculate_input_size(data.get("input")),
            code_hash=code_hash,
        )


def collect_rule_code_hashes(
    metadata_dir: Path,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, set[str]]:
    """
    Collect code hashes for each rule from metadata files.

    This enables detection of renamed rules by matching their shell code.
    If two rules have the same code hash, they are likely the same rule
    that was renamed.

    Args:
        metadata_dir: Path to .snakemake/metadata/ directory.
        progress_callback: Optional callback(current, total) for progress reporting.

    Returns:
        Dictionary mapping code_hash -> set of rule names that use that code.
    """
    hash_to_rules: dict[str, set[str]] = {}

    if not metadata_dir.exists():
        return hash_to_rules

    # Use optimized iterate_metadata_files (6-7x faster than rglob)
    for _path, data in iterate_metadata_files(
        metadata_dir,
        progress_callback,
        sort_by_mtime=False,  # Order doesn't matter for code hash collection
        use_cache=False,  # We need to read code field which isn't cached
    ):
        rule = data.get("rule")
        code = data.get("code")

        if rule and code:
            # Normalize whitespace before hashing to handle formatting differences
            normalized_code = " ".join(code.split())
            code_hash = hashlib.sha256(normalized_code.encode()).hexdigest()[:16]

            if code_hash not in hash_to_rules:
                hash_to_rules[code_hash] = set()
            hash_to_rules[code_hash].add(rule)

    return hash_to_rules
