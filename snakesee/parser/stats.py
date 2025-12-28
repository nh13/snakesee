"""Timing statistics collection for Snakemake workflows.

This module collects and aggregates timing statistics from metadata files
for use in time estimation.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from snakesee.models import RuleTimingStats
from snakesee.models import WildcardTimingStats
from snakesee.parser.metadata import parse_metadata_files

if TYPE_CHECKING:
    from snakesee.types import ProgressCallback


def collect_rule_timing_stats(
    metadata_dir: Path,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, RuleTimingStats]:
    """
    Collect historical timing statistics per rule from metadata.

    Aggregates all completed job timings by rule name, sorted chronologically
    by end time. Includes timestamps for time-based weighted estimation.
    Input sizes are included when available from job metadata.

    Args:
        metadata_dir: Path to .snakemake/metadata/ directory.
        progress_callback: Optional callback(current, total) for progress reporting.

    Returns:
        Dictionary mapping rule names to their timing statistics.
    """
    # First, collect all jobs with their timing info
    # rule -> [(duration, end_time, input_size), ...]
    jobs_by_rule: dict[str, list[tuple[float, float, int | None]]] = {}

    for job in parse_metadata_files(metadata_dir, progress_callback=progress_callback):
        duration = job.duration
        end_time = job.end_time
        if duration is None or end_time is None:
            continue

        if job.rule not in jobs_by_rule:
            jobs_by_rule[job.rule] = []
        jobs_by_rule[job.rule].append((duration, end_time, job.input_size))

    # Build stats with sorted durations, timestamps, and input sizes
    stats: dict[str, RuleTimingStats] = {}
    for rule, timing_tuples in jobs_by_rule.items():
        # Sort by end_time (oldest first) for consistent ordering
        timing_tuples.sort(key=lambda x: x[1])

        durations = [t[0] for t in timing_tuples]
        timestamps = [t[1] for t in timing_tuples]
        input_sizes = [t[2] for t in timing_tuples]

        stats[rule] = RuleTimingStats(
            rule=rule,
            durations=durations,
            timestamps=timestamps,
            input_sizes=input_sizes,
        )

    return stats


def _build_wildcard_stats_for_key(
    rule: str,
    wc_key: str,
    wc_values: dict[str, list[tuple[float, float]]],
) -> WildcardTimingStats:
    """Build WildcardTimingStats from collected timing pairs.

    Args:
        rule: Rule name.
        wc_key: Wildcard key name.
        wc_values: Dict of wildcard value -> list of (duration, timestamp) pairs.

    Returns:
        WildcardTimingStats for this wildcard key.
    """
    stats_by_value: dict[str, RuleTimingStats] = {}

    for wc_value, timing_pairs in wc_values.items():
        # Sort by end_time (use sorted() to avoid mutating caller's data)
        sorted_pairs = sorted(timing_pairs, key=lambda x: x[1])
        durations = [pair[0] for pair in sorted_pairs]
        timestamps = [pair[1] for pair in sorted_pairs]

        stats_by_value[wc_value] = RuleTimingStats(
            rule=f"{rule}:{wc_key}={wc_value}",
            durations=durations,
            timestamps=timestamps,
        )

    return WildcardTimingStats(
        rule=rule,
        wildcard_key=wc_key,
        stats_by_value=stats_by_value,
    )


def collect_wildcard_timing_stats(
    metadata_dir: Path,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, dict[str, WildcardTimingStats]]:
    """Collect timing statistics per rule, conditioned on wildcards.

    Groups execution times by (rule, wildcard_key, wildcard_value) for rules
    that have wildcards in their metadata.

    Args:
        metadata_dir: Path to .snakemake/metadata/ directory.
        progress_callback: Optional callback(current, total) for progress reporting.

    Returns:
        Nested dictionary: rule -> wildcard_key -> WildcardTimingStats
    """
    # Collect all jobs with wildcards
    # Structure: rule -> wildcard_key -> wildcard_value -> [(duration, end_time), ...]
    data: dict[str, dict[str, dict[str, list[tuple[float, float]]]]] = {}

    for job in parse_metadata_files(metadata_dir, progress_callback=progress_callback):
        duration = job.duration
        end_time = job.end_time
        if duration is None or end_time is None or not job.wildcards:
            continue

        if job.rule not in data:
            data[job.rule] = {}

        for wc_key, wc_value in job.wildcards.items():
            if wc_key not in data[job.rule]:
                data[job.rule][wc_key] = {}
            if wc_value not in data[job.rule][wc_key]:
                data[job.rule][wc_key][wc_value] = []

            data[job.rule][wc_key][wc_value].append((duration, end_time))

    # Build WildcardTimingStats objects
    result: dict[str, dict[str, WildcardTimingStats]] = {}
    for rule, wc_keys in data.items():
        result[rule] = {
            wc_key: _build_wildcard_stats_for_key(rule, wc_key, wc_values)
            for wc_key, wc_values in wc_keys.items()
        }

    return result
