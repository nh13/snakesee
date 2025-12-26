"""Parsers for Snakemake log files and metadata."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from snakesee.models import JobInfo
from snakesee.models import RuleTimingStats
from snakesee.models import WildcardTimingStats
from snakesee.models import WorkflowProgress
from snakesee.models import WorkflowStatus
from snakesee.parser.log_reader import IncrementalLogReader
from snakesee.utils import iterate_metadata_files

if TYPE_CHECKING:
    from snakesee.types import ProgressCallback

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
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


# Pattern: "15 of 50 steps (30%) done"
PROGRESS_PATTERN = re.compile(r"(\d+) of (\d+) steps \((\d+(?:\.\d+)?)%\) done")

# Pattern for rule start: "rule align:" or "localrule all:"
RULE_START_PATTERN = re.compile(r"(?:local)?rule (\w+):")

# Pattern for job ID in log: "    jobid: 5"
JOBID_PATTERN = re.compile(r"\s+jobid:\s*(\d+)")

# Pattern for finished job:
# - Old format: "Finished job 5." or "[date] Finished job 5."
# - Snakemake 9.x format: "Finished jobid: 5" or "Finished jobid: 5 (Rule: name)"
FINISHED_JOB_PATTERN = re.compile(r"Finished (?:job |jobid:\s*)(\d+)")

# Pattern for error in job: "Error in rule X" or job failure indicators
ERROR_PATTERN = re.compile(r"Error in rule (\w+):|Error executing rule|RuleException")

# Pattern for "Error in rule X:" specifically (to capture rule name)
ERROR_IN_RULE_PATTERN = re.compile(r"Error in rule (\w+):")

# Pattern for timestamp lines: "[Mon Dec 15 22:34:30 2025]"
TIMESTAMP_PATTERN = re.compile(r"\[(\w{3} \w{3} +\d+ \d+:\d+:\d+ \d+)\]")

# Pattern for wildcards line: "    wildcards: sample=A, batch=1"
WILDCARDS_PATTERN = re.compile(r"\s+wildcards:\s*(.+)")

# Pattern for threads line: "    threads: 4"
THREADS_PATTERN = re.compile(r"\s+threads:\s*(\d+)")

# Pattern for log line: "    log: logs/sample.log"
LOG_PATTERN = re.compile(r"\s+log:\s*(.+)")


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


def parse_job_stats_from_log(log_path: Path) -> set[str]:
    """
    Parse the 'Job stats' table from a Snakemake log to get the set of rules.

    The job stats table appears at the start of a Snakemake run and lists all
    rules that will be executed along with their job counts.

    Args:
        log_path: Path to the Snakemake log file.

    Returns:
        Set of rule names from the job stats table. Returns empty set if
        the table cannot be found or parsed.
    """
    rules: set[str] = set()

    try:
        content = log_path.read_text(errors="ignore")
    except OSError:
        return rules

    lines = content.splitlines()
    in_job_stats = False
    past_header = False

    for line in lines:
        # Look for the start of job stats table
        if line.strip() == "Job stats:":
            in_job_stats = True
            continue

        if not in_job_stats:
            continue

        stripped = line.strip()

        # Skip empty lines
        if not stripped:
            # Empty line after data means end of table
            if past_header and rules:
                break
            continue

        # Skip header line (job count)
        if stripped.startswith("job") and "count" in stripped:
            continue

        # Skip separator line (dashes)
        if stripped.startswith("-"):
            past_header = True
            continue

        # Parse data row: "rule_name    count"
        if past_header:
            parts = stripped.split()
            if len(parts) >= 2:
                rule_name = parts[0]
                # Skip 'total' row
                if rule_name != "total":
                    rules.add(rule_name)

    return rules


def parse_job_stats_counts_from_log(log_path: Path) -> dict[str, int]:
    """
    Parse the 'Job stats' table from a Snakemake log to get rule -> job count mapping.

    The job stats table appears at the start of a Snakemake run and lists all
    rules that will be executed along with their job counts. This function
    captures both the rule names AND their expected counts.

    Args:
        log_path: Path to the Snakemake log file.

    Returns:
        Dictionary mapping rule names to their expected job counts.
        Returns empty dict if the table cannot be found or parsed.
    """
    counts: dict[str, int] = {}

    try:
        content = log_path.read_text(errors="ignore")
    except OSError:
        return counts

    lines = content.splitlines()
    in_job_stats = False
    past_header = False

    for line in lines:
        # Look for the start of job stats table
        if line.strip() == "Job stats:":
            in_job_stats = True
            continue

        if not in_job_stats:
            continue

        stripped = line.strip()

        # Skip empty lines
        if not stripped:
            # Empty line after data means end of table
            if past_header and counts:
                break
            continue

        # Skip header line (job count)
        if stripped.startswith("job") and "count" in stripped:
            continue

        # Skip separator line (dashes)
        if stripped.startswith("-"):
            past_header = True
            continue

        # Parse data row: "rule_name    count"
        if past_header:
            parts = stripped.split()
            if len(parts) >= 2:
                rule_name = parts[0]
                # Skip 'total' row
                if rule_name != "total":
                    try:
                        count = int(parts[-1])
                        counts[rule_name] = count
                    except ValueError:
                        # Skip if count isn't a valid integer
                        pass

    return counts


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
        try:
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
        except KeyError:
            continue


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
        try:
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
        except KeyError:
            continue


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

    # Get file list - count upfront if progress is requested
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
            rule = data.get("rule")
            code = data.get("code")

            if rule and code:
                # Normalize whitespace before hashing to handle formatting differences
                normalized_code = " ".join(code.split())
                code_hash = hashlib.sha256(normalized_code.encode()).hexdigest()[:16]

                if code_hash not in hash_to_rules:
                    hash_to_rules[code_hash] = set()
                hash_to_rules[code_hash].add(rule)

        except json.JSONDecodeError as e:
            logger.debug("Malformed JSON in metadata file %s: %s", meta_file, e)
            continue
        except OSError as e:
            logger.debug("Error reading metadata file %s: %s", meta_file, e)
            continue
        except KeyError as e:
            logger.debug("Missing key in metadata file %s: %s", meta_file, e)
            continue

    return hash_to_rules


def parse_running_jobs_from_log(log_path: Path) -> list[JobInfo]:
    """
    Parse currently running jobs by analyzing the log file.

    Tracks jobs that have started (rule + jobid) but not yet finished.

    Args:
        log_path: Path to the snakemake log file.

    Returns:
        List of JobInfo for jobs that appear to be running.
    """
    # Track started jobs: jobid -> (rule, start_line_num, wildcards, threads)
    started_jobs: dict[str, tuple[str, int, dict[str, str] | None, int | None]] = {}
    # Job logs: jobid -> log_path (separate lookup by unique jobid)
    job_logs: dict[str, str] = {}
    finished_jobids: set[str] = set()

    current_rule: str | None = None
    current_jobid: str | None = None
    current_wildcards: dict[str, str] | None = None
    current_threads: int | None = None
    current_log_path: str | None = None

    try:
        lines = log_path.read_text().splitlines()
        for line_num, line in enumerate(lines):
            # Track current rule being executed
            if match := RULE_START_PATTERN.match(line):
                current_rule = match.group(1)
                current_jobid = None  # Reset jobid for new rule block
                current_wildcards = None
                current_threads = None
                current_log_path = None

            # Capture wildcards within rule block
            elif match := WILDCARDS_PATTERN.match(line):
                current_wildcards = _parse_wildcards(match.group(1))

            # Capture threads within rule block
            elif match := THREADS_PATTERN.match(line):
                current_threads = int(match.group(1))
                # Update already-stored job if threads comes after jobid
                if current_jobid and current_jobid in started_jobs:
                    rule, ln, wc, _ = started_jobs[current_jobid]
                    started_jobs[current_jobid] = (rule, ln, wc, current_threads)

            # Capture log path within rule block - store by jobid
            elif match := LOG_PATTERN.match(line):
                current_log_path = match.group(1).strip()
                if current_jobid:
                    job_logs[current_jobid] = current_log_path

            # Capture jobid within rule block
            elif match := JOBID_PATTERN.match(line):
                current_jobid = match.group(1)
                if current_rule is not None and current_jobid not in started_jobs:
                    started_jobs[current_jobid] = (
                        current_rule,
                        line_num,
                        current_wildcards,
                        current_threads,
                    )
                # Store log by jobid if we already captured it
                if current_log_path:
                    job_logs[current_jobid] = current_log_path

            # Track finished jobs
            elif match := FINISHED_JOB_PATTERN.search(line):
                finished_jobids.add(match.group(1))

    except OSError:
        return []

    # Jobs that started but haven't finished are running
    running: list[JobInfo] = []
    log_mtime = log_path.stat().st_mtime if log_path.exists() else None

    for jobid, (rule, _line_num, wildcards, threads) in started_jobs.items():
        if jobid not in finished_jobids:
            # Look up log by jobid - unique within this run
            job_log = job_logs.get(jobid)
            running.append(
                JobInfo(
                    rule=rule,
                    job_id=jobid,
                    start_time=log_mtime,  # Approximate; could improve with timestamps
                    wildcards=wildcards,
                    threads=threads,
                    log_file=Path(job_log) if job_log else None,
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

    # Track context: rule, jobid, wildcards, threads, and log for each job block
    current_rule: str | None = None
    current_jobid: str | None = None
    current_wildcards: dict[str, str] | None = None
    current_threads: int | None = None
    current_log_path: str | None = None
    # Job logs: jobid -> log_path (separate lookup by unique jobid)
    job_logs: dict[str, str] = {}

    try:
        lines = log_path.read_text().splitlines()
        for line in lines:
            # Track current rule
            if match := RULE_START_PATTERN.match(line):
                current_rule = match.group(1)
                current_jobid = None
                current_wildcards = None
                current_threads = None
                current_log_path = None

            # Capture wildcards
            elif match := WILDCARDS_PATTERN.match(line):
                current_wildcards = _parse_wildcards(match.group(1))

            # Capture threads
            elif match := THREADS_PATTERN.match(line):
                current_threads = int(match.group(1))

            # Capture log path within rule block - store by jobid
            elif match := LOG_PATTERN.match(line):
                current_log_path = match.group(1).strip()
                if current_jobid:
                    job_logs[current_jobid] = current_log_path

            # Capture jobid
            elif match := JOBID_PATTERN.match(line):
                current_jobid = match.group(1)
                # Store log by jobid if we already captured it
                if current_log_path:
                    job_logs[current_jobid] = current_log_path

            # Detect errors
            elif match := ERROR_IN_RULE_PATTERN.search(line):
                rule = match.group(1)
                # Use context jobid/wildcards/threads/log if the error rule matches current context
                jobid = current_jobid if current_rule == rule else None
                wildcards = current_wildcards if current_rule == rule else None
                threads = current_threads if current_rule == rule else None
                # Look up log by jobid - unique within this run
                job_log = job_logs.get(jobid) if jobid else None
                key = (rule, jobid)

                if key not in seen_failures:
                    seen_failures.add(key)
                    failed_jobs.append(
                        JobInfo(
                            rule=rule,
                            job_id=jobid,
                            wildcards=wildcards,
                            threads=threads,
                            log_file=Path(job_log) if job_log else None,
                        )
                    )

    except OSError:
        pass

    return failed_jobs


def parse_incomplete_jobs(
    incomplete_dir: Path, min_start_time: float | None = None
) -> Iterator[JobInfo]:
    """
    Parse currently running jobs from incomplete markers.

    Snakemake creates marker files in .snakemake/incomplete/ for jobs that
    are in progress. The marker filename is base64-encoded output file path.
    The file modification time indicates when the job started.

    Note: This is a fallback method. Prefer parse_running_jobs_from_log()
    which provides rule names.

    Args:
        incomplete_dir: Path to .snakemake/incomplete/ directory.
        min_start_time: If provided, only yield markers with mtime >= this time.
            Used to filter out stale markers from previous workflow runs.

    Yields:
        JobInfo instances for each in-progress job.
    """
    import base64

    if not incomplete_dir.exists():
        return

    for marker in incomplete_dir.rglob("*"):
        if marker.is_file() and marker.name != "migration_underway":
            try:
                marker_mtime = marker.stat().st_mtime

                # Skip markers that are older than the current workflow run
                if min_start_time is not None and marker_mtime < min_start_time:
                    continue

                # Decode the base64 filename to get the output file path
                output_file: Path | None = None
                try:
                    decoded = base64.b64decode(marker.name).decode("utf-8")
                    output_file = Path(decoded)
                except (ValueError, UnicodeDecodeError):
                    pass  # Keep output_file as None if decoding fails

                # The marker's mtime is approximately when the job started
                yield JobInfo(
                    rule="unknown",  # Cannot determine rule from marker filename
                    start_time=marker_mtime,
                    output_file=output_file,
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


def _get_first_log_timestamp(log_path: Path) -> float | None:
    """Extract the first timestamp from a snakemake log file.

    This provides a more accurate workflow start time than the file's ctime,
    since the log file may be created before jobs actually start.
    """
    try:
        with log_path.open() as f:
            for line in f:
                if match := TIMESTAMP_PATTERN.match(line):
                    return _parse_timestamp(match.group(1))
    except OSError:
        pass
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

    # Track started jobs: jobid -> (rule, start_time, wildcards, threads)
    started_jobs: dict[str, tuple[str, float, dict[str, str] | None, int | None]] = {}
    # Job logs: jobid -> log_path (separate lookup by unique jobid)
    job_logs: dict[str, str] = {}
    current_rule: str | None = None
    current_timestamp: float | None = None
    current_wildcards: dict[str, str] | None = None
    current_threads: int | None = None
    current_jobid: str | None = None
    current_log_path: str | None = None

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
                current_threads = None
                current_jobid = None
                current_log_path = None

            # Capture wildcards within rule block
            elif match := WILDCARDS_PATTERN.match(line):
                current_wildcards = _parse_wildcards(match.group(1))

            # Capture threads within rule block
            elif match := THREADS_PATTERN.match(line):
                current_threads = int(match.group(1))
                # Update already-stored job if threads comes after jobid
                if current_jobid and current_jobid in started_jobs:
                    rule, ts, wc, _ = started_jobs[current_jobid]
                    started_jobs[current_jobid] = (rule, ts, wc, current_threads)

            # Capture log path within rule block - store by jobid
            elif match := LOG_PATTERN.match(line):
                current_log_path = match.group(1).strip()
                if current_jobid:
                    job_logs[current_jobid] = current_log_path

            # Capture jobid within rule block
            elif match := JOBID_PATTERN.match(line):
                current_jobid = match.group(1)
                if current_rule is not None and current_timestamp is not None:
                    started_jobs[current_jobid] = (
                        current_rule,
                        current_timestamp,
                        current_wildcards,
                        current_threads,
                    )
                # Store log by jobid if we already captured it
                if current_log_path:
                    job_logs[current_jobid] = current_log_path

            # Track finished jobs
            elif match := FINISHED_JOB_PATTERN.search(line):
                jobid = match.group(1)
                if jobid in started_jobs and current_timestamp is not None:
                    rule, start_time, wildcards, threads = started_jobs[jobid]
                    # Look up log by jobid - unique within this run
                    job_log = job_logs.get(jobid)
                    completed_jobs.append(
                        JobInfo(
                            rule=rule,
                            job_id=jobid,
                            start_time=start_time,
                            end_time=current_timestamp,
                            wildcards=wildcards,
                            threads=threads,
                            log_file=Path(job_log) if job_log else None,
                        )
                    )

    except OSError:
        pass

    return completed_jobs


def parse_threads_from_log(log_path: Path) -> dict[str, int]:
    """
    Parse a jobid -> threads mapping from a snakemake log file.

    This is used to augment metadata completions with thread info,
    since metadata files don't store the threads directive.

    Args:
        log_path: Path to the snakemake log file.

    Returns:
        Dictionary mapping job_id to thread count.
    """
    threads_map: dict[str, int] = {}
    current_jobid: str | None = None
    current_threads: int | None = None

    try:
        for line in log_path.read_text().splitlines():
            # Track current rule (resets context)
            if RULE_START_PATTERN.match(line):
                current_jobid = None
                current_threads = None

            # Capture threads
            elif match := THREADS_PATTERN.match(line):
                current_threads = int(match.group(1))
                # Update already-stored job if threads comes after jobid
                if current_jobid and current_jobid not in threads_map:
                    threads_map[current_jobid] = current_threads

            # Capture jobid
            elif match := JOBID_PATTERN.match(line):
                current_jobid = match.group(1)
                if current_threads is not None and current_jobid not in threads_map:
                    threads_map[current_jobid] = current_threads

    except OSError:
        pass

    return threads_map


def is_workflow_running(snakemake_dir: Path, stale_threshold: float = 1800.0) -> bool:
    """
    Check if a workflow is currently running.

    Uses multiple signals:
    1. Lock files exist in .snakemake/locks/
    2. Incomplete markers exist in .snakemake/incomplete/ (jobs in progress)
    3. Log file was recently modified (within stale_threshold seconds)

    If locks AND incomplete markers both exist, the workflow is definitely running
    (incomplete markers are created when jobs start and removed when they finish).
    Log freshness is only used as a fallback when there are no incomplete markers.

    Args:
        snakemake_dir: Path to the .snakemake directory.
        stale_threshold: Seconds since last log modification before considering
            the workflow stale/dead. Default 1800 seconds (30 minutes).

    Returns:
        True if workflow appears to be actively running, False otherwise.
    """
    from snakesee.state.clock import get_clock
    from snakesee.state.paths import WorkflowPaths

    locks_dir = snakemake_dir / "locks"
    if not locks_dir.exists():
        return False

    try:
        has_locks = any(locks_dir.iterdir())
    except OSError:
        return False

    if not has_locks:
        return False

    # If locks exist AND incomplete markers exist, workflow is definitely running
    # (incomplete markers are created when jobs start, removed when they finish)
    incomplete_dir = snakemake_dir / "incomplete"
    if incomplete_dir.exists():
        try:
            has_incomplete = any(incomplete_dir.iterdir())
            if has_incomplete:
                return True
        except OSError:
            pass

    # Fall back to log freshness check when no incomplete markers
    # snakemake_dir is .snakemake, so parent is workflow_dir
    paths = WorkflowPaths(snakemake_dir.parent)
    log_file = paths.find_latest_log()
    if log_file is None:
        # No log file but locks exist - assume running (early startup)
        return True

    try:
        log_mtime = log_file.stat().st_mtime
        age = get_clock().now() - log_mtime
        # If log hasn't been modified recently, workflow is likely dead
        return age < stale_threshold
    except OSError:
        # Can't stat log file - assume running to be safe
        return True


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


def collect_wildcard_timing_stats(  # noqa: C901
    metadata_dir: Path,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, dict[str, WildcardTimingStats]]:
    """
    Collect timing statistics per rule, conditioned on wildcards.

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


def _augment_completions_with_threads(completions: list[JobInfo], log_path: Path) -> list[JobInfo]:
    """Augment completions with threads from log parsing.

    Metadata completions don't have threads - match by rule + end_time.
    """
    log_completions = parse_completed_jobs_from_log(log_path)
    # Build lookup: (rule, end_time_rounded) -> threads
    threads_lookup: dict[tuple[str, int], int] = {}
    for lc in log_completions:
        if lc.threads is not None and lc.end_time is not None:
            key = (lc.rule, int(lc.end_time))
            threads_lookup[key] = lc.threads

    if not threads_lookup:
        return completions

    augmented: list[JobInfo] = []
    for job in completions:
        threads = job.threads
        if threads is None and job.end_time is not None:
            key = (job.rule, int(job.end_time))
            threads = threads_lookup.get(key)
        if threads is not None and job.threads is None:
            job = JobInfo(
                rule=job.rule,
                job_id=job.job_id,
                start_time=job.start_time,
                end_time=job.end_time,
                output_file=job.output_file,
                wildcards=job.wildcards,
                input_size=job.input_size,
                threads=threads,
            )
        augmented.append(job)
    return augmented


def parse_workflow_state(
    workflow_dir: Path,
    log_file: Path | None = None,
    cutoff_time: float | None = None,
    log_reader: IncrementalLogReader | None = None,
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
        log_reader: Optional incremental log reader for efficient polling.
                    When provided, uses cached state instead of re-parsing
                    the entire log file on each call.

    Returns:
        Current workflow state as a WorkflowProgress instance.
    """
    from snakesee.state.paths import WorkflowPaths

    paths = WorkflowPaths(workflow_dir)
    snakemake_dir = paths.snakemake_dir

    # Use specified log file or find latest
    latest_log = paths.find_latest_log()
    log_path = log_file if log_file is not None else latest_log
    is_latest_log = log_file is None or log_file == latest_log

    # Determine status from lock files (only relevant for latest log)
    workflow_is_running = is_latest_log and is_workflow_running(snakemake_dir)
    if workflow_is_running:
        status = WorkflowStatus.RUNNING
    else:
        status = WorkflowStatus.COMPLETED

    # Update incremental reader if provided and log path matches/changes
    if log_reader is not None and log_path is not None:
        log_reader.set_log_path(log_path)
        log_reader.read_new_lines()

    # Parse progress from log file (or use reader state)
    if log_reader is not None and log_path is not None:
        completed, total = log_reader.progress
    else:
        completed, total = (0, 0) if log_path is None else parse_progress_from_log(log_path)

    # Get workflow start time for filtering incomplete markers
    # Prefer the first timestamp in the log (when jobs actually started) over file ctime
    start_time: float | None = None
    if log_path is not None:
        start_time = _get_first_log_timestamp(log_path)
        if start_time is None:
            # Fall back to file ctime if no timestamps in log yet
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
        elif log_reader is not None:
            # Use incremental reader's completed jobs
            completions = log_reader.completed_jobs
        else:
            # No matching metadata - parse completions from the log file
            # This handles: no metadata dir, timing mismatches, historical logs
            completions = parse_completed_jobs_from_log(log_path)

    # Augment completions with threads from log (metadata doesn't have threads)
    if log_path is not None and completions:
        completions = _augment_completions_with_threads(completions, log_path)

    completions.sort(key=lambda j: j.end_time or 0, reverse=True)

    # Parse running jobs from log file (provides rule names)
    running: list[JobInfo] = []
    failed_list: list[JobInfo] = []
    if log_path is not None:
        if log_reader is not None:
            running = log_reader.running_jobs
            failed_list = log_reader.failed_jobs
        else:
            running = parse_running_jobs_from_log(log_path)
            failed_list = parse_failed_jobs_from_log(log_path)

    # Check for incomplete markers (jobs that were in progress when workflow was interrupted)
    incomplete_dir = snakemake_dir / "incomplete"
    incomplete_list = (
        list(parse_incomplete_jobs(incomplete_dir, min_start_time=start_time))
        if is_latest_log
        else []
    )

    # Remove failed jobs from the running list - a job can't be both running and failed
    failed_job_ids = {job.job_id for job in failed_list if job.job_id is not None}
    if failed_job_ids:
        running = [job for job in running if job.job_id not in failed_job_ids]

    # If workflow is not actually running, clear "running" jobs from log parsing
    # (those are orphaned jobs, not truly running - they're reflected in incomplete_list)
    if not workflow_is_running:
        running = []
    else:
        # If workflow IS running, incomplete markers represent running jobs, not
        # interrupted ones - clear them to avoid double-counting
        incomplete_list = []

    # Determine final status based on running jobs, failures, and incomplete markers
    failed_jobs = len(failed_list)
    if running and is_latest_log and workflow_is_running:
        # Workflow is actively running jobs
        status = WorkflowStatus.RUNNING
    elif failed_jobs > 0:
        # No running jobs but has failures
        status = WorkflowStatus.FAILED
    elif incomplete_list:
        # Workflow has incomplete markers = interrupted
        status = WorkflowStatus.INCOMPLETE
    elif completed < total and not workflow_is_running:
        # Fallback: if workflow stopped but not all jobs completed, it was interrupted
        status = WorkflowStatus.INCOMPLETE

    return WorkflowProgress(
        workflow_dir=workflow_dir,
        status=status,
        total_jobs=total,
        completed_jobs=completed,
        failed_jobs=failed_jobs,
        failed_jobs_list=failed_list,
        incomplete_jobs_list=incomplete_list,
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
