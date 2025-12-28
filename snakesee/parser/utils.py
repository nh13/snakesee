"""Utility functions for parsing.

Small helper functions used across parser modules.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _parse_wildcards(wildcards_str: str) -> dict[str, str]:
    """
    Parse a wildcards string into a dictionary.

    Args:
        wildcards_str: String like "sample=A, batch=1"

    Returns:
        Dictionary like {"sample": "A", "batch": "1"}

    Examples:
        >>> _parse_wildcards("sample=A, batch=1")
        {'sample': 'A', 'batch': '1'}
        >>> _parse_wildcards("id=test_123")
        {'id': 'test_123'}
        >>> _parse_wildcards("key=value=with=equals")
        {'key': 'value=with=equals'}
    """
    wildcards: dict[str, str] = {}
    # Split by comma, then parse key=value pairs
    for part in wildcards_str.split(","):
        part = part.strip()
        if "=" in part:
            key, value = part.split("=", 1)
            wildcards[key.strip()] = value.strip()
    return wildcards


def _parse_positive_int(value: str, field_name: str = "value") -> int | None:
    """Parse a string as a positive integer with validation.

    Args:
        value: String to parse.
        field_name: Name of field for logging.

    Returns:
        Parsed integer if valid and positive, None otherwise.
    """
    try:
        result = int(value)
        if result <= 0:
            logger.debug("Invalid %s: %d (must be positive)", field_name, result)
            return None
        return result
    except ValueError:
        logger.debug("Could not parse %s as integer: %s", field_name, value)
        return None


def _parse_non_negative_int(value: str, field_name: str = "value") -> int | None:
    """Parse a string as a non-negative integer with validation.

    Args:
        value: String to parse.
        field_name: Name of field for logging.

    Returns:
        Parsed integer if valid and >= 0, None otherwise.
    """
    try:
        result = int(value)
        if result < 0:
            logger.debug("Invalid %s: %d (must be non-negative)", field_name, result)
            return None
        return result
    except ValueError:
        logger.debug("Could not parse %s as integer: %s", field_name, value)
        return None


def _parse_timestamp(timestamp_str: str) -> float | None:
    """Parse a snakemake log timestamp into Unix time.

    Args:
        timestamp_str: Timestamp string like "Mon Dec 15 22:34:30 2025"

    Returns:
        Unix timestamp as float, or None if parsing fails.
    """
    from datetime import datetime

    try:
        # Format: "Mon Dec 15 22:34:30 2025"
        dt = datetime.strptime(timestamp_str, "%a %b %d %H:%M:%S %Y")
        return dt.timestamp()
    except ValueError:
        return None


def calculate_input_size(file_paths: list[Path]) -> int | None:
    """
    Calculate total size of input files.

    Args:
        file_paths: List of input file paths.

    Returns:
        Total size in bytes, or None if any file doesn't exist.
    """
    total_size = 0
    for path in file_paths:
        try:
            total_size += path.stat().st_size
        except OSError:
            return None  # File doesn't exist or can't be accessed
    return total_size if file_paths else None


def estimate_input_size_from_output(
    output_path: Path,
    workflow_dir: Path,
) -> int | None:
    """
    Try to estimate input size by looking for related input files.

    This is a heuristic that works for common bioinformatics patterns where
    output files are derived from inputs with predictable naming conventions.

    Examples:
        - sample.sorted.bam -> sample.bam
        - sample.fastq.gz -> looks for sample.fq.gz, sample.fastq.gz
        - sample.vcf.gz -> sample.bam

    Args:
        output_path: Path to the output file.
        workflow_dir: Workflow root directory.

    Returns:
        Estimated input size in bytes, or None if not determinable.
    """
    # Common input file patterns relative to output
    suffixes_to_strip = [
        ".sorted.bam",
        ".sorted",
        ".trimmed",
        ".filtered",
        ".dedup",
        ".aligned",
    ]

    name = output_path.name

    # Try stripping common suffixes to find input
    for suffix in suffixes_to_strip:
        if name.endswith(suffix):
            input_name = name[: -len(suffix)]
            # Try common extensions
            for ext in [".bam", ".fastq.gz", ".fq.gz", ".fa.gz", ".fasta.gz"]:
                candidate = workflow_dir / (input_name + ext)
                if candidate.exists():
                    try:
                        return candidate.stat().st_size
                    except OSError:
                        continue

    # No input found
    return None
