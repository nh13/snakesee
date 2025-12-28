"""Coordinator for incremental log reading."""

import logging
import threading
from pathlib import Path

from snakesee.models import JobInfo
from snakesee.parser.failure_tracker import FailureTracker
from snakesee.parser.file_position import LogFilePosition
from snakesee.parser.job_tracker import JobLifecycleTracker
from snakesee.parser.line_parser import LogLineParser
from snakesee.parser.line_parser import ParseEventType

logger = logging.getLogger(__name__)


class IncrementalLogReader:
    """Streaming reader for Snakemake log files with position tracking.

    Reads log lines incrementally, tracking the current file position to only
    parse new content on subsequent calls. Maintains cumulative state for
    running jobs, completed jobs, failed jobs, and progress.

    Handles log file rotation by detecting inode changes or file truncation.

    This is a coordinator class that delegates to specialized components:
    - LogFilePosition: File position tracking and rotation detection
    - LogLineParser: Log line parsing with context tracking
    - JobLifecycleTracker: Job start/finish tracking
    - FailureTracker: Failure deduplication

    Attributes:
        log_path: Path to the log file being monitored.
    """

    def __init__(self, log_path: Path) -> None:
        """Initialize the incremental log reader.

        Args:
            log_path: Path to the Snakemake log file.
        """
        self.log_path = log_path
        self._lock = threading.RLock()

        # Delegate components
        self._position = LogFilePosition(log_path)
        self._parser = LogLineParser()
        self._jobs = JobLifecycleTracker()
        self._failures = FailureTracker()

        # Progress state
        self._completed: int = 0
        self._total: int = 0

    def reset(self) -> None:
        """Reset reader to start of file and clear all state.

        Call this when switching to a different log file or to re-read
        from the beginning.
        """
        with self._lock:
            self._reset_unlocked()

    def set_log_path(self, log_path: Path) -> None:
        """Change the log file being monitored.

        Resets all state if the path changes.

        Args:
            log_path: New log file path.
        """
        with self._lock:
            if log_path != self.log_path:
                self.log_path = log_path
                self._position = LogFilePosition(log_path)
                self._reset_unlocked()

    def _reset_unlocked(self) -> None:
        """Reset state without acquiring lock (caller must hold lock)."""
        self._position.reset()
        self._parser.reset()
        self._jobs.reset()
        self._failures.reset()
        self._completed = 0
        self._total = 0

    def read_new_lines(self) -> int:
        """Read and parse new lines from the log file.

        Updates internal state based on new log content. This method
        should be called periodically to process new log entries.

        Returns:
            Number of new lines processed.
        """
        if not self.log_path.exists():
            return 0

        with self._lock:
            if self._position.check_rotation():
                # File was rotated - reset all state
                self._parser.reset()
                self._jobs.reset()
                self._failures.reset()
                self._completed = 0
                self._total = 0

            lines_processed = 0
            try:
                with open(self.log_path, "r", encoding="utf-8", errors="replace") as f:
                    # Clamp offset to file bounds
                    file_size = f.seek(0, 2)
                    self._position.clamp_to_size(file_size)
                    f.seek(self._position.offset)

                    for line in f:
                        self._process_line(line)
                        lines_processed += 1

                    self._position.offset = f.tell()
            except FileNotFoundError:
                pass
            except PermissionError as e:
                logger.warning("Permission denied reading log file %s: %s", self.log_path, e)
            except OSError as e:
                logger.warning("Error reading log file %s: %s", self.log_path, e)

            return lines_processed

    def _process_line(self, line: str) -> None:
        """Process a parsed line and update state.

        Args:
            line: Raw log line to process.
        """
        event = self._parser.parse_line(line)
        if event is None:
            return

        if event.event_type == ParseEventType.PROGRESS:
            completed = event.data["completed"]
            total = event.data["total"]
            if isinstance(completed, int) and isinstance(total, int):
                self._completed = completed
                self._total = total

        elif event.event_type == ParseEventType.JOBID:
            rule = event.data.get("rule")
            if rule is not None:
                jobid = str(event.data["jobid"])
                if not self._jobs.is_job_started(jobid):
                    timestamp = event.data.get("timestamp")
                    wildcards = event.data.get("wildcards")
                    threads = event.data.get("threads")
                    self._jobs.start_job(
                        jobid=jobid,
                        rule=str(rule),
                        start_time=float(timestamp)
                        if isinstance(timestamp, (int, float))
                        else None,
                        wildcards=wildcards if isinstance(wildcards, dict) else None,
                        threads=threads if isinstance(threads, int) else None,
                    )
                log_path = event.data.get("log_path")
                if isinstance(log_path, str):
                    self._jobs.set_job_log(jobid, log_path)

        elif event.event_type == ParseEventType.WILDCARDS:
            wc_jobid = event.data.get("jobid")
            wc_wildcards = event.data.get("wildcards")
            if isinstance(wc_jobid, str) and isinstance(wc_wildcards, dict):
                self._jobs.update_job(wc_jobid, wildcards=wc_wildcards)

        elif event.event_type == ParseEventType.THREADS:
            th_jobid = event.data.get("jobid")
            th_threads = event.data.get("threads")
            if isinstance(th_jobid, str) and isinstance(th_threads, int):
                self._jobs.update_job(th_jobid, threads=th_threads)

        elif event.event_type == ParseEventType.LOG_PATH:
            lp_jobid = event.data.get("jobid")
            lp_path = event.data.get("log_path")
            if isinstance(lp_jobid, str) and isinstance(lp_path, str):
                self._jobs.set_job_log(lp_jobid, lp_path)

        elif event.event_type == ParseEventType.JOB_FINISHED:
            fin_jobid = str(event.data["jobid"])
            fin_timestamp = event.data.get("timestamp")
            fin_end = float(fin_timestamp) if isinstance(fin_timestamp, (int, float)) else None
            self._jobs.finish_job(fin_jobid, end_time=fin_end)

        elif event.event_type == ParseEventType.ERROR:
            err_rule = str(event.data["rule"])
            err_jobid = event.data.get("jobid")
            err_log = event.data.get("log_path")
            err_wildcards = event.data.get("wildcards")
            err_threads = event.data.get("threads")
            self._failures.record_failure(
                rule=err_rule,
                jobid=str(err_jobid) if err_jobid is not None else None,
                wildcards=err_wildcards if isinstance(err_wildcards, dict) else None,
                threads=err_threads if isinstance(err_threads, int) else None,
                log_file=Path(err_log) if isinstance(err_log, str) else None,
            )

    @property
    def progress(self) -> tuple[int, int]:
        """Get current workflow progress.

        Returns:
            Tuple of (completed_count, total_count).
        """
        with self._lock:
            return self._completed, self._total

    @property
    def running_jobs(self) -> list[JobInfo]:
        """Get list of currently running jobs.

        Returns:
            List of JobInfo for jobs that started but haven't finished.
        """
        with self._lock:
            return self._jobs.get_running_jobs()

    @property
    def completed_jobs(self) -> list[JobInfo]:
        """Get list of completed jobs with timing info.

        Returns:
            List of JobInfo for completed jobs, sorted by end time (newest first).
        """
        with self._lock:
            return self._jobs.get_completed_jobs()

    @property
    def failed_jobs(self) -> list[JobInfo]:
        """Get list of failed jobs.

        Returns:
            List of JobInfo for jobs that encountered errors.
        """
        with self._lock:
            return self._failures.get_failed_jobs()
