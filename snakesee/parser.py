"""Parsers for Snakemake log files and metadata."""

import json
import re
from collections.abc import Iterator
from pathlib import Path

from snakesee.models import JobInfo
from snakesee.models import RuleTimingStats
from snakesee.models import WildcardTimingStats
from snakesee.models import WorkflowProgress
from snakesee.models import WorkflowStatus

# Pattern: "15 of 50 steps (30%) done"
PROGRESS_PATTERN = re.compile(r"(\d+) of (\d+) steps \((\d+(?:\.\d+)?)%\) done")

# Pattern for rule start: "rule align:" or "localrule all:"
RULE_START_PATTERN = re.compile(r"(?:local)?rule (\w+):")

# Pattern for job ID in log: "    jobid: 5"
JOBID_PATTERN = re.compile(r"\s+jobid:\s*(\d+)")

# Pattern for finished job: "Finished job 5." or "[date] Finished job 5."
FINISHED_JOB_PATTERN = re.compile(r"Finished job (\d+)\.")

# Pattern for error in job: "Error in rule X" or job failure indicators
ERROR_PATTERN = re.compile(r"Error in rule (\w+):|Error executing rule|RuleException")

# Pattern for "Error in rule X:" specifically (to capture rule name)
ERROR_IN_RULE_PATTERN = re.compile(r"Error in rule (\w+):")

# Pattern for timestamp lines: "[Mon Dec 15 22:34:30 2025]"
TIMESTAMP_PATTERN = re.compile(r"\[(\w{3} \w{3} +\d+ \d+:\d+:\d+ \d+)\]")

# Pattern for wildcards line: "    wildcards: sample=A, batch=1"
WILDCARDS_PATTERN = re.compile(r"\s+wildcards:\s*(.+)")


def _parse_wildcards(wildcards_str: str) -> dict[str, str]:
    """
    Parse a wildcards string into a dictionary.

    Args:
        wildcards_str: String like "sample=A, batch=1"

    Returns:
        Dictionary like {"sample": "A", "batch": "1"}
    """
    wildcards: dict[str, str] = {}
    # Split by comma, then parse key=value pairs
    for part in wildcards_str.split(","):
        part = part.strip()
        if "=" in part:
            key, value = part.split("=", 1)
            wildcards[key.strip()] = value.strip()
    return wildcards


def find_latest_log(snakemake_dir: Path) -> Path | None:
    """
    Find the most recent snakemake log file.

    Args:
        snakemake_dir: Path to the .snakemake directory.

    Returns:
        Path to the most recent log file, or None if no logs exist.
    """
    log_dir = snakemake_dir / "log"
    if not log_dir.exists():
        return None
    logs = sorted(log_dir.glob("*.snakemake.log"), key=lambda p: p.stat().st_mtime)
    return logs[-1] if logs else None


def parse_progress_from_log(log_path: Path) -> tuple[int, int]:
    """
    Parse current progress from a snakemake log file.

    Reads the log file and finds the most recent progress line.

    Args:
        log_path: Path to the snakemake log file.

    Returns:
        Tuple of (completed_count, total_count). Returns (0, 0) if no progress found.
    """
    completed, total = 0, 0
    try:
        for line in log_path.read_text().splitlines():
            if match := PROGRESS_PATTERN.search(line):
                completed = int(match.group(1))
                total = int(match.group(2))
    except OSError:
        pass
    return completed, total


def parse_rules_from_log(log_path: Path) -> dict[str, int]:
    """
    Parse rule execution counts from a snakemake log file.

    Counts how many times each rule appears to have completed based on log entries.

    Args:
        log_path: Path to the snakemake log file.

    Returns:
        Dictionary mapping rule names to completion counts.
    """
    rule_counts: dict[str, int] = {}
    current_rule: str | None = None

    try:
        for line in log_path.read_text().splitlines():
            # Track current rule being executed
            if match := RULE_START_PATTERN.match(line):
                current_rule = match.group(1)
            # Count "Finished job" as rule completion
            elif "Finished job" in line and current_rule is not None:
                rule_counts[current_rule] = rule_counts.get(current_rule, 0) + 1
    except OSError:
        pass

    return rule_counts


def parse_metadata_files(metadata_dir: Path) -> Iterator[JobInfo]:  # noqa: C901
    """
    Parse completed job information from Snakemake metadata files.

    Reads JSON metadata files from .snakemake/metadata/ to extract
    timing information for completed jobs, including input file sizes.

    Args:
        metadata_dir: Path to .snakemake/metadata/ directory.

    Yields:
        JobInfo instances for each completed job found.
    """
    if not metadata_dir.exists():
        return

    for meta_file in metadata_dir.rglob("*"):
        if not meta_file.is_file():
            continue
        try:
            data = json.loads(meta_file.read_text())
            rule = data.get("rule")
            starttime = data.get("starttime")
            endtime = data.get("endtime")

            if rule is not None and starttime is not None and endtime is not None:
                # Extract wildcards if present (Snakemake stores as dict)
                wildcards_data = data.get("wildcards")
                wildcards: dict[str, str] | None = None
                if isinstance(wildcards_data, dict):
                    wildcards = {str(k): str(v) for k, v in wildcards_data.items()}

                # Extract input files and calculate total size
                input_size: int | None = None
                input_files = data.get("input")
                if isinstance(input_files, list) and input_files:
                    total_size = 0
                    all_found = True
                    for f in input_files:
                        try:
                            total_size += Path(f).stat().st_size
                        except OSError:
                            all_found = False
                            break
                    if all_found:
                        input_size = total_size

                yield JobInfo(
                    rule=rule,
                    start_time=starttime,
                    end_time=endtime,
                    wildcards=wildcards,
                    input_size=input_size,
                )
        except (json.JSONDecodeError, OSError, KeyError):
            continue


def parse_running_jobs_from_log(log_path: Path) -> list[JobInfo]:
    """
    Parse currently running jobs by analyzing the log file.

    Tracks jobs that have started (rule + jobid) but not yet finished.

    Args:
        log_path: Path to the snakemake log file.

    Returns:
        List of JobInfo for jobs that appear to be running.
    """
    # Track started jobs: jobid -> (rule, start_line_num, wildcards)
    started_jobs: dict[str, tuple[str, int, dict[str, str] | None]] = {}
    finished_jobids: set[str] = set()

    current_rule: str | None = None
    current_jobid: str | None = None
    current_wildcards: dict[str, str] | None = None

    try:
        lines = log_path.read_text().splitlines()
        for line_num, line in enumerate(lines):
            # Track current rule being executed
            if match := RULE_START_PATTERN.match(line):
                current_rule = match.group(1)
                current_jobid = None  # Reset jobid for new rule block
                current_wildcards = None

            # Capture wildcards within rule block
            elif match := WILDCARDS_PATTERN.match(line):
                current_wildcards = _parse_wildcards(match.group(1))

            # Capture jobid within rule block
            elif match := JOBID_PATTERN.match(line):
                current_jobid = match.group(1)
                if current_rule is not None and current_jobid not in started_jobs:
                    started_jobs[current_jobid] = (current_rule, line_num, current_wildcards)

            # Track finished jobs
            elif match := FINISHED_JOB_PATTERN.search(line):
                finished_jobids.add(match.group(1))

    except OSError:
        return []

    # Jobs that started but haven't finished are running
    running: list[JobInfo] = []
    log_mtime = log_path.stat().st_mtime if log_path.exists() else None

    for jobid, (rule, _line_num, wildcards) in started_jobs.items():
        if jobid not in finished_jobids:
            # Estimate start time from log modification time (approximate)
            running.append(
                JobInfo(
                    rule=rule,
                    job_id=jobid,
                    start_time=log_mtime,  # Approximate; could improve with timestamps
                    wildcards=wildcards,
                )
            )

    return running


def parse_failed_jobs_from_log(log_path: Path) -> list[JobInfo]:
    """
    Parse failed jobs from the snakemake log file.

    Looks for "Error in rule X:" patterns and extracts the rule name
    and associated job ID when available. Useful for --keep-going workflows.

    Args:
        log_path: Path to the snakemake log file.

    Returns:
        List of JobInfo for jobs that failed.
    """
    failed_jobs: list[JobInfo] = []
    seen_failures: set[tuple[str, str | None]] = set()  # (rule, jobid) pairs

    # Track context: rule, jobid, and wildcards for each job block
    current_rule: str | None = None
    current_jobid: str | None = None
    current_wildcards: dict[str, str] | None = None

    try:
        lines = log_path.read_text().splitlines()
        for line in lines:
            # Track current rule
            if match := RULE_START_PATTERN.match(line):
                current_rule = match.group(1)
                current_jobid = None
                current_wildcards = None

            # Capture wildcards
            elif match := WILDCARDS_PATTERN.match(line):
                current_wildcards = _parse_wildcards(match.group(1))

            # Capture jobid
            elif match := JOBID_PATTERN.match(line):
                current_jobid = match.group(1)

            # Detect errors
            elif match := ERROR_IN_RULE_PATTERN.search(line):
                rule = match.group(1)
                # Use context jobid/wildcards if the error rule matches current context
                jobid = current_jobid if current_rule == rule else None
                wildcards = current_wildcards if current_rule == rule else None
                key = (rule, jobid)

                if key not in seen_failures:
                    seen_failures.add(key)
                    failed_jobs.append(
                        JobInfo(
                            rule=rule,
                            job_id=jobid,
                            wildcards=wildcards,
                        )
                    )

    except OSError:
        pass

    return failed_jobs


def parse_incomplete_jobs(incomplete_dir: Path) -> Iterator[JobInfo]:
    """
    Parse currently running jobs from incomplete markers.

    Snakemake creates marker files in .snakemake/incomplete/ for jobs that
    are in progress. The file modification time indicates when the job started.

    Note: This is a fallback method. Prefer parse_running_jobs_from_log()
    which provides rule names.

    Args:
        incomplete_dir: Path to .snakemake/incomplete/ directory.

    Yields:
        JobInfo instances for each in-progress job.
    """
    if not incomplete_dir.exists():
        return

    for marker in incomplete_dir.rglob("*"):
        if marker.is_file() and marker.name != "migration_underway":
            try:
                # The marker's mtime is approximately when the job started
                yield JobInfo(
                    rule="unknown",  # Cannot determine rule from marker filename
                    start_time=marker.stat().st_mtime,
                )
            except OSError:
                continue


def _parse_timestamp(timestamp_str: str) -> float | None:
    """Parse a snakemake log timestamp into Unix time."""
    from datetime import datetime

    try:
        # Format: "Mon Dec 15 22:34:30 2025"
        dt = datetime.strptime(timestamp_str, "%a %b %d %H:%M:%S %Y")
        return dt.timestamp()
    except ValueError:
        return None


def parse_completed_jobs_from_log(log_path: Path) -> list[JobInfo]:
    """
    Parse completed jobs with timing from a snakemake log file.

    Extracts job start/end times from log timestamps to reconstruct
    job durations. This is useful for historical logs where metadata
    may have been overwritten by later runs.

    Args:
        log_path: Path to the snakemake log file.

    Returns:
        List of JobInfo for completed jobs with timing information.
    """
    completed_jobs: list[JobInfo] = []

    # Track started jobs: jobid -> (rule, start_time, wildcards)
    started_jobs: dict[str, tuple[str, float, dict[str, str] | None]] = {}
    current_rule: str | None = None
    current_timestamp: float | None = None
    current_wildcards: dict[str, str] | None = None

    try:
        lines = log_path.read_text().splitlines()
        for line in lines:
            # Check for timestamp
            if match := TIMESTAMP_PATTERN.match(line):
                current_timestamp = _parse_timestamp(match.group(1))

            # Track current rule being executed
            elif match := RULE_START_PATTERN.match(line):
                current_rule = match.group(1)
                current_wildcards = None

            # Capture wildcards within rule block
            elif match := WILDCARDS_PATTERN.match(line):
                current_wildcards = _parse_wildcards(match.group(1))

            # Capture jobid within rule block
            elif match := JOBID_PATTERN.match(line):
                jobid = match.group(1)
                if current_rule is not None and current_timestamp is not None:
                    started_jobs[jobid] = (current_rule, current_timestamp, current_wildcards)

            # Track finished jobs
            elif match := FINISHED_JOB_PATTERN.search(line):
                jobid = match.group(1)
                if jobid in started_jobs and current_timestamp is not None:
                    rule, start_time, wildcards = started_jobs[jobid]
                    completed_jobs.append(
                        JobInfo(
                            rule=rule,
                            job_id=jobid,
                            start_time=start_time,
                            end_time=current_timestamp,
                            wildcards=wildcards,
                        )
                    )

    except OSError:
        pass

    return completed_jobs


def is_workflow_running(snakemake_dir: Path, stale_threshold: float = 60.0) -> bool:
    """
    Check if a workflow is currently running.

    Uses two signals:
    1. Lock files exist in .snakemake/locks/
    2. Log file was recently modified (within stale_threshold seconds)

    This handles the case where snakemake is killed without cleaning up locks.

    Args:
        snakemake_dir: Path to the .snakemake directory.
        stale_threshold: Seconds since last log modification before considering
            the workflow stale/dead. Default 60 seconds.

    Returns:
        True if workflow appears to be actively running, False otherwise.
    """
    import time

    locks_dir = snakemake_dir / "locks"
    if not locks_dir.exists():
        return False

    try:
        has_locks = any(locks_dir.iterdir())
    except OSError:
        return False

    if not has_locks:
        return False

    # Lock files exist - check if the log is still being updated
    log_file = find_latest_log(snakemake_dir)
    if log_file is None:
        # No log file but locks exist - assume running (early startup)
        return True

    try:
        log_mtime = log_file.stat().st_mtime
        age = time.time() - log_mtime
        # If log hasn't been modified recently, workflow is likely dead
        return age < stale_threshold
    except OSError:
        # Can't stat log file - assume running to be safe
        return True


def collect_rule_timing_stats(metadata_dir: Path) -> dict[str, RuleTimingStats]:
    """
    Collect historical timing statistics per rule from metadata.

    Aggregates all completed job timings by rule name, sorted chronologically
    by end time. Includes timestamps for time-based weighted estimation.
    Input sizes are included when available from job metadata.

    Args:
        metadata_dir: Path to .snakemake/metadata/ directory.

    Returns:
        Dictionary mapping rule names to their timing statistics.
    """
    # First, collect all jobs with their timing info
    # rule -> [(duration, end_time, input_size), ...]
    jobs_by_rule: dict[str, list[tuple[float, float, int | None]]] = {}

    for job in parse_metadata_files(metadata_dir):
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


def collect_wildcard_timing_stats(  # noqa: C901
    metadata_dir: Path,
) -> dict[str, dict[str, WildcardTimingStats]]:
    """
    Collect timing statistics per rule, conditioned on wildcards.

    Groups execution times by (rule, wildcard_key, wildcard_value) for rules
    that have wildcards in their metadata.

    Args:
        metadata_dir: Path to .snakemake/metadata/ directory.

    Returns:
        Nested dictionary: rule -> wildcard_key -> WildcardTimingStats
    """
    # Collect all jobs with wildcards
    # Structure: rule -> wildcard_key -> wildcard_value -> [(duration, end_time), ...]
    data: dict[str, dict[str, dict[str, list[tuple[float, float]]]]] = {}

    for job in parse_metadata_files(metadata_dir):
        duration = job.duration
        end_time = job.end_time
        if duration is None or end_time is None:
            continue
        if not job.wildcards:
            continue  # Skip jobs without wildcards

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
        result[rule] = {}
        for wc_key, wc_values in wc_keys.items():
            stats_by_value: dict[str, RuleTimingStats] = {}

            for wc_value, timing_pairs in wc_values.items():
                # Sort by end_time
                timing_pairs.sort(key=lambda x: x[1])

                durations = [pair[0] for pair in timing_pairs]
                timestamps = [pair[1] for pair in timing_pairs]

                stats_by_value[wc_value] = RuleTimingStats(
                    rule=f"{rule}:{wc_key}={wc_value}",
                    durations=durations,
                    timestamps=timestamps,
                )

            result[rule][wc_key] = WildcardTimingStats(
                rule=rule,
                wildcard_key=wc_key,
                stats_by_value=stats_by_value,
            )

    return result


def _filter_completions_by_timeframe(
    completions: list[JobInfo],
    log_path: Path,
    cutoff_time: float | None = None,
) -> list[JobInfo]:
    """
    Filter completions to jobs that completed during a specific workflow run.

    Args:
        completions: List of all completed jobs from metadata.
        log_path: Path to the log file.
        cutoff_time: Optional upper bound (e.g., when next log started).
                     If None, no upper bound is applied.

    Returns:
        Filtered list of completions that occurred during this workflow run.
    """
    try:
        log_start = log_path.stat().st_ctime
        if cutoff_time is not None:
            return [
                j
                for j in completions
                if j.end_time is not None and log_start <= j.end_time < cutoff_time
            ]
        else:
            return [j for j in completions if j.end_time is not None and j.end_time >= log_start]
    except OSError:
        return []  # Can't determine timeframe - return empty to avoid stale data


def parse_workflow_state(
    workflow_dir: Path,
    log_file: Path | None = None,
    cutoff_time: float | None = None,
) -> WorkflowProgress:
    """
    Parse complete workflow state from .snakemake directory.

    Combines information from log files, metadata, incomplete markers,
    and lock files to build a complete picture of workflow state.

    Args:
        workflow_dir: Root directory containing .snakemake/.
        log_file: Optional specific log file to parse. If None, uses the latest.
        cutoff_time: Optional upper bound for filtering completions (e.g., when
                     the next log started). Used for "time machine" view of
                     historical logs.

    Returns:
        Current workflow state as a WorkflowProgress instance.
    """
    snakemake_dir = workflow_dir / ".snakemake"

    # Use specified log file or find latest
    log_path = log_file if log_file is not None else find_latest_log(snakemake_dir)
    is_latest_log = log_file is None or log_file == find_latest_log(snakemake_dir)

    # Determine status from lock files (only relevant for latest log)
    if is_latest_log and is_workflow_running(snakemake_dir):
        status = WorkflowStatus.RUNNING
    else:
        status = WorkflowStatus.COMPLETED

    # Parse progress from log file
    completed, total = (0, 0) if log_path is None else parse_progress_from_log(log_path)

    # Get workflow start time from log file creation
    start_time: float | None = None
    if log_path is not None:
        try:
            start_time = log_path.stat().st_ctime
        except OSError:
            pass

    # Parse completed jobs for recent completions
    metadata_dir = snakemake_dir / "metadata"
    all_completions = list(parse_metadata_files(metadata_dir))

    # Filter completions to the relevant timeframe
    completions: list[JobInfo] = []
    if log_path is not None:
        filtered = _filter_completions_by_timeframe(all_completions, log_path, cutoff_time)
        if filtered:
            completions = filtered
        else:
            # No matching metadata - parse completions from the log file
            # This handles: no metadata dir, timing mismatches, historical logs
            completions = parse_completed_jobs_from_log(log_path)

    completions.sort(key=lambda j: j.end_time or 0, reverse=True)

    # Parse running jobs from log file (provides rule names)
    running: list[JobInfo] = []
    failed_list: list[JobInfo] = []
    if log_path is not None:
        running = parse_running_jobs_from_log(log_path)
        failed_list = parse_failed_jobs_from_log(log_path)

    # If we have running jobs in the log, override status to RUNNING
    # (handles case where lock-based detection fails due to stale threshold)
    if running and is_latest_log:
        status = WorkflowStatus.RUNNING

    # Check for failures: either from parsed errors or from incomplete progress
    failed_jobs = len(failed_list)
    if failed_jobs > 0:
        status = WorkflowStatus.FAILED if status != WorkflowStatus.RUNNING else status
    elif status == WorkflowStatus.COMPLETED and completed < total and not running:
        # Fallback: if workflow stopped but not all jobs completed, assume failure
        # Only apply if there are no running jobs (avoids false positives)
        status = WorkflowStatus.FAILED
        failed_jobs = total - completed

    return WorkflowProgress(
        workflow_dir=workflow_dir,
        status=status,
        total_jobs=total,
        completed_jobs=completed,
        failed_jobs=failed_jobs,
        failed_jobs_list=failed_list,
        running_jobs=running,
        recent_completions=completions[:10],
        start_time=start_time,
        log_file=log_path,
    )


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
