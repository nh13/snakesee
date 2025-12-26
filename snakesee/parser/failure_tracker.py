"""Failure tracking with deduplication."""

from pathlib import Path

from snakesee.models import JobInfo


class FailureTracker:
    """Tracks job failures with deduplication.

    Prevents duplicate failure entries for the same rule/jobid combination.
    """

    __slots__ = ("_failed_jobs", "_seen_failures")

    def __init__(self) -> None:
        """Initialize the failure tracker."""
        self._failed_jobs: list[JobInfo] = []
        self._seen_failures: set[tuple[str, str | None]] = set()

    def record_failure(
        self,
        rule: str,
        jobid: str | None = None,
        wildcards: dict[str, str] | None = None,
        threads: int | None = None,
        log_file: Path | None = None,
    ) -> bool:
        """Record a job failure.

        Args:
            rule: Name of the failed rule.
            jobid: Job identifier (if known).
            wildcards: Wildcard values.
            threads: Thread count.
            log_file: Path to the log file.

        Returns:
            True if this is a new failure, False if duplicate.
        """
        key = (rule, jobid)
        if key in self._seen_failures:
            return False

        self._seen_failures.add(key)
        self._failed_jobs.append(
            JobInfo(
                rule=rule,
                job_id=jobid,
                wildcards=wildcards,
                threads=threads,
                log_file=log_file,
            )
        )
        return True

    def get_failed_jobs(self) -> list[JobInfo]:
        """Get list of failed jobs.

        Returns:
            List of JobInfo for failed jobs.
        """
        return list(self._failed_jobs)

    def reset(self) -> None:
        """Clear all failure tracking state."""
        self._failed_jobs.clear()
        self._seen_failures.clear()
