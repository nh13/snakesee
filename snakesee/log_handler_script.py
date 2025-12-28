"""Log handler script for Snakemake 8.x --log-handler-script integration.

This script converts Snakemake log messages to the same event format used by
the snakesee logger plugin (Snakemake 9+), enabling real-time monitoring
with snakesee for Snakemake 8.x users.

Usage:
    snakemake --log-handler-script $(snakesee log-handler-path) --cores 4

Note on execution tracking:
    This script is optimized for local execution where jobs start immediately
    after submission. For cluster/cloud executors (SLURM, AWS Batch, etc.),
    jobs may be queued before running. Since Snakemake 8.x doesn't provide a
    reliable signal for when a queued job actually starts executing, we emit
    job_started immediately upon job_info. This means "running" jobs may
    actually still be queued on cluster systems.

    For accurate queue vs running tracking on clusters, consider using
    Snakemake 9+ with the logger plugin (--logger snakesee).
"""

from __future__ import annotations

import atexit
import fcntl
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any
from typing import TextIO

# Logger for this module - debug level by default to avoid noise
logger = logging.getLogger(__name__)

# Event file name (must match snakesee/events.py EVENT_FILE_NAME)
EVENT_FILE_NAME = ".snakesee_events.jsonl"

# Compact JSON separators (no spaces after : or ,)
JSON_SEPARATORS: tuple[str, str] = (",", ":")

# Global state for tracking job start times and metadata
# Protected by _state_lock for thread safety
_state_lock = threading.Lock()
_job_start_times: dict[int, float] = {}
_job_rules: dict[int, str] = {}
_job_wildcards: dict[int, dict[str, str]] = {}
_job_threads: dict[int, int] = {}
_event_file: TextIO | None = None
_workflow_started_emitted: bool = False
_workflow_lock_fd: int | None = None  # File descriptor for workflow-level locking


def _close_event_file() -> None:
    """Close the event file handle on exit."""
    global _event_file, _workflow_lock_fd
    if _event_file is not None:
        try:
            _event_file.close()
        except OSError as e:
            logger.debug("Failed to close event file: %s", e)
        _event_file = None

    # Also close the workflow lock FD if open
    if _workflow_lock_fd is not None:
        try:
            fcntl.flock(_workflow_lock_fd, fcntl.LOCK_UN)
        except OSError:
            pass
        try:
            os.close(_workflow_lock_fd)
        except OSError as e:
            logger.debug("Failed to close workflow lock: %s", e)
        _workflow_lock_fd = None


# Register cleanup on exit
atexit.register(_close_event_file)


def _get_event_file() -> TextIO:
    """Get or open the event file handle."""
    global _event_file
    if _event_file is None:
        event_path = Path.cwd() / EVENT_FILE_NAME
        _event_file = open(event_path, "a", encoding="utf-8")
    return _event_file


# Default timeout for file lock operations (seconds)
FILE_LOCK_TIMEOUT: float = 5.0


def _acquire_lock_with_timeout(fd: int, timeout: float = FILE_LOCK_TIMEOUT) -> bool:
    """Acquire an exclusive file lock with timeout using exponential backoff.

    Args:
        fd: File descriptor to lock.
        timeout: Maximum time to wait for lock in seconds.

    Returns:
        True if lock was acquired, False if timeout expired.
    """
    deadline = time.time() + timeout
    sleep_time = 0.001  # Start with 1ms
    max_sleep = 0.1  # Cap at 100ms
    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except BlockingIOError:
            remaining = deadline - time.time()
            if remaining <= 0:
                return False
            # Exponential backoff with cap, bounded by remaining time
            time.sleep(min(sleep_time, max_sleep, remaining))
            sleep_time *= 1.5


# Maximum retries for event writing
MAX_EVENT_WRITE_RETRIES: int = 3


def _write_event(event_data: dict[str, Any]) -> None:
    """Write an event to the event file with proper locking and retry logic.

    Uses exponential backoff on lock acquisition failure, with up to
    MAX_EVENT_WRITE_RETRIES attempts.
    """
    f = _get_event_file()
    # Remove None values to reduce file size
    event_data = {k: v for k, v in event_data.items() if v is not None}
    line = json.dumps(event_data, separators=JSON_SEPARATORS, default=str) + "\n"

    fd = f.fileno()

    # Loop-based retry with exponential backoff
    for retry_count in range(MAX_EVENT_WRITE_RETRIES + 1):
        if _acquire_lock_with_timeout(fd):
            break

        if retry_count < MAX_EVENT_WRITE_RETRIES:
            # Exponential backoff: 100ms, 200ms, 400ms
            backoff = 0.1 * (2**retry_count)
            logger.debug(
                "Lock acquisition failed, retrying in %.1fs (attempt %d/%d)",
                backoff,
                retry_count + 1,
                MAX_EVENT_WRITE_RETRIES,
            )
            time.sleep(backoff)
    else:
        # Exhausted all retries without acquiring lock
        logger.warning(
            "Failed to acquire event file lock after %d attempts, event may be lost",
            MAX_EVENT_WRITE_RETRIES,
        )
        return

    try:
        f.write(line)
        f.flush()
        os.fsync(fd)
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)


def _extract_wildcards(msg: dict[str, Any]) -> dict[str, str] | None:
    """Extract wildcards from a message."""
    wc = msg.get("wildcards")
    if wc is None:
        return None

    if hasattr(wc, "__dict__"):
        return {k: str(v) for k, v in vars(wc).items() if not k.startswith("_")}
    elif isinstance(wc, dict):
        return {str(k): str(v) for k, v in wc.items()}

    return None


def _extract_resources(msg: dict[str, Any]) -> dict[str, Any] | None:
    """Extract resources from a message."""
    res = msg.get("resources")
    if res is None:
        return None

    def extract_value(v: Any) -> str | int | float | bool | None:
        if isinstance(v, (str, int, float, bool, type(None))):
            return v
        try:
            s = str(v)
            if s.startswith("<") and "object at" in s:
                return None
            return s
        except (TypeError, ValueError, AttributeError):
            return None

    result: dict[str, Any] = {}
    if hasattr(res, "__dict__"):
        for k, v in vars(res).items():
            if not k.startswith("_"):
                extracted = extract_value(v)
                if extracted is not None:
                    result[k] = extracted
    elif isinstance(res, dict):
        for k, v in res.items():
            extracted = extract_value(v)
            if extracted is not None:
                result[k] = extracted

    return result if result else None


def _handle_job_info(msg: dict[str, Any], timestamp: float) -> None:
    """Handle job submission (job_info level).

    Since Snakemake 8.x doesn't emit separate job_started events,
    we emit both job_submitted and job_started here. For local execution,
    jobs start immediately after submission.
    """
    jobid = msg.get("jobid")
    rule_name = msg.get("name") or msg.get("rule", "unknown")
    wildcards = _extract_wildcards(msg)
    threads = msg.get("threads")
    resources = _extract_resources(msg)

    # Store for later correlation (thread-safe)
    if jobid is not None:
        with _state_lock:
            _job_rules[jobid] = rule_name
            if wildcards:
                _job_wildcards[jobid] = wildcards
            if threads is not None:
                _job_threads[jobid] = threads
            _job_start_times[jobid] = timestamp

    # Extract file lists
    input_files = None
    if msg.get("input"):
        inp = msg["input"]
        if isinstance(inp, (list, tuple)):
            input_files = [str(f) for f in inp]

    output_files = None
    if msg.get("output"):
        out = msg["output"]
        if isinstance(out, (list, tuple)):
            output_files = [str(f) for f in out]

    _write_event(
        {
            "event_type": "job_submitted",
            "timestamp": timestamp,
            "job_id": jobid,
            "rule_name": rule_name,
            "wildcards": wildcards,
            "threads": threads,
            "resources": resources,
            "input_files": input_files,
            "output_files": output_files,
        }
    )

    # Emit job_started immediately since Snakemake 8.x doesn't send
    # separate job_started events, and local jobs start immediately
    if jobid is not None:
        _write_event(
            {
                "event_type": "job_started",
                "timestamp": timestamp,
                "job_id": jobid,
                "rule_name": rule_name,
                "wildcards": wildcards,
                "threads": threads,
            }
        )


def _handle_job_started(msg: dict[str, Any], timestamp: float) -> None:
    """Handle job execution start."""
    jobid = msg.get("jobid")
    if jobid is None:
        return

    # Thread-safe state access
    with _state_lock:
        # Only set start time if not already set by _handle_job_info
        if jobid not in _job_start_times:
            _job_start_times[jobid] = timestamp
        rule_name = _job_rules.get(jobid)
        wildcards = _job_wildcards.get(jobid)
        threads = _job_threads.get(jobid)

    _write_event(
        {
            "event_type": "job_started",
            "timestamp": timestamp,
            "job_id": jobid,
            "rule_name": rule_name,
            "wildcards": wildcards,
            "threads": threads,
        }
    )


def _handle_job_finished(msg: dict[str, Any], timestamp: float) -> None:
    """Handle job completion."""
    jobid = msg.get("jobid")
    if jobid is None:
        return

    # Thread-safe state cleanup
    with _state_lock:
        start_time = _job_start_times.pop(jobid, None)
        rule_name = _job_rules.pop(jobid, None)
        wildcards = _job_wildcards.pop(jobid, None)
        threads = _job_threads.pop(jobid, None)

    duration = timestamp - start_time if start_time else None

    _write_event(
        {
            "event_type": "job_finished",
            "timestamp": timestamp,
            "job_id": jobid,
            "rule_name": rule_name,
            "wildcards": wildcards,
            "threads": threads,
            "duration": duration,
        }
    )


def _handle_job_error(msg: dict[str, Any], timestamp: float) -> None:
    """Handle job failure."""
    jobid = msg.get("jobid")
    # Only emit job_error events for actual jobs (with job_id)
    # RuleException messages come through without job_id and should be ignored
    # since the actual job failure event follows with the job_id
    if jobid is None:
        return

    error_msg = msg.get("msg") or msg.get("message")
    if error_msg is not None:
        error_msg = str(error_msg)

    # Thread-safe state cleanup (jobid is guaranteed non-None after early return)
    with _state_lock:
        start_time = _job_start_times.pop(jobid, None)
        wildcards = _job_wildcards.pop(jobid, None)
        threads = _job_threads.pop(jobid, None)
        # Use rule from message, falling back to stored value
        rule_name = msg.get("name") or msg.get("rule") or _job_rules.pop(jobid, None)

    duration = timestamp - start_time if start_time else None

    _write_event(
        {
            "event_type": "job_error",
            "timestamp": timestamp,
            "job_id": jobid,
            "rule_name": rule_name,
            "wildcards": wildcards,
            "threads": threads,
            "duration": duration,
            "error_message": error_msg,
        }
    )


def _handle_progress(msg: dict[str, Any], timestamp: float) -> None:
    """Handle workflow progress update."""
    done = msg.get("done", 0)
    total = msg.get("total", 0)

    _write_event(
        {
            "event_type": "progress",
            "timestamp": timestamp,
            "completed_jobs": done,
            "total_jobs": total,
        }
    )


def _ensure_workflow_started(timestamp: float) -> None:
    """Emit workflow_started event once at the beginning.

    This also clears any stale events from previous workflow runs by
    truncating the event file before writing the first event.

    Uses file locking to ensure atomicity when multiple processes
    might start simultaneously.
    """
    global _workflow_started_emitted, _event_file, _workflow_lock_fd
    if _workflow_started_emitted:
        return

    event_path = Path.cwd() / EVENT_FILE_NAME
    lock_path = Path.cwd() / ".snakesee_events.lock"

    # Acquire exclusive lock to make check-and-truncate atomic
    lock_fd: int | None = None
    try:
        lock_fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR)
        if not _acquire_lock_with_timeout(lock_fd, FILE_LOCK_TIMEOUT * 2):
            logger.warning("Failed to acquire workflow lock within timeout")
            # Close the FD to prevent leak on timeout
            try:
                os.close(lock_fd)
            except OSError:
                pass
            return

        # Double-check after acquiring lock (another process may have initialized)
        # mypy doesn't understand global can change between checks (thread safety pattern)
        if _workflow_started_emitted:
            return  # type: ignore[unreachable]

        # Close existing file handle if open (prevents file descriptor leak)
        existing_file = _event_file
        if existing_file is not None:
            try:
                existing_file.close()
            except OSError:
                pass
            _event_file = None

        # Truncate event file to clear stale data from previous runs
        try:
            _event_file = open(event_path, "w", encoding="utf-8")
            _write_event(
                {
                    "event_type": "workflow_started",
                    "timestamp": timestamp,
                }
            )
            _workflow_started_emitted = True
            _workflow_lock_fd = lock_fd  # Store for later cleanup
            lock_fd = None  # Prevent closing in finally since we're keeping it
        except Exception:
            # Clean up event file on failure to prevent leak
            if _event_file is not None:
                try:
                    _event_file.close()
                except OSError:
                    pass
                _event_file = None
            raise
    finally:
        # Release and close lock if we still own it
        if lock_fd is not None:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
            except OSError:
                pass
            try:
                os.close(lock_fd)
            except OSError:
                pass


def log_handler(msg: dict[str, Any]) -> None:
    """Main log handler function called by Snakemake for every log message.

    This function is invoked by Snakemake when using --log-handler-script.
    It converts log messages to snakesee events for real-time monitoring.

    Args:
        msg: Dictionary containing log message data with keys like:
            - level: Log level (job_info, job_finished, error, progress, etc.)
            - jobid: Job identifier
            - name/rule: Rule name
            - wildcards: Job wildcards
            - threads: Thread count
            - resources: Resource requirements
            - input/output: File lists
            - done/total: Progress counts
    """
    level = msg.get("level")
    if level is None:
        return

    timestamp = time.time()

    # Emit workflow_started on first message
    _ensure_workflow_started(timestamp)

    # Map log levels to event handlers
    if level == "job_info":
        _handle_job_info(msg, timestamp)
    elif level == "job_started":
        _handle_job_started(msg, timestamp)
    elif level == "job_finished":
        _handle_job_finished(msg, timestamp)
    elif level in ("job_error", "error"):
        _handle_job_error(msg, timestamp)
    elif level == "progress":
        _handle_progress(msg, timestamp)
