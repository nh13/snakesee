"""Job lifecycle tracking from start to completion."""

from pathlib import Path
from typing import TypedDict

from snakesee.models import JobInfo


class StartedJobData(TypedDict):
    """Data for a job that has started but not finished."""

    rule: str
    start_time: float | None
    wildcards: dict[str, str] | None
    threads: int | None


class JobLifecycleTracker:
    """Tracks jobs from start to completion.

    Maintains state for started jobs, finished job IDs, and completed
    jobs with full timing information.
    """

    __slots__ = (
        "_completed_jobs",
        "_finished_jobids",
        "_job_logs",
        "_started_jobs",
    )

    def __init__(self) -> None:
        """Initialize the job tracker."""
        self._started_jobs: dict[str, StartedJobData] = {}
        self._finished_jobids: set[str] = set()
        self._completed_jobs: list[JobInfo] = []
        self._job_logs: dict[str, str] = {}

    def start_job(
        self,
        jobid: str,
        rule: str,
        start_time: float | None = None,
        wildcards: dict[str, str] | None = None,
        threads: int | None = None,
    ) -> None:
        """Record a job starting.

        Args:
            jobid: Unique job identifier.
            rule: Name of the rule.
            start_time: Unix timestamp when job started.
            wildcards: Wildcard values for this job.
            threads: Number of threads allocated.
        """
        if jobid not in self._started_jobs:
            self._started_jobs[jobid] = StartedJobData(
                rule=rule,
                start_time=start_time,
                wildcards=wildcards,
                threads=threads,
            )

    def update_job(
        self,
        jobid: str,
        wildcards: dict[str, str] | None = None,
        threads: int | None = None,
    ) -> None:
        """Update an existing job's metadata.

        Args:
            jobid: Job identifier.
            wildcards: Wildcard values to update.
            threads: Thread count to update.
        """
        if jobid in self._started_jobs:
            if wildcards is not None:
                self._started_jobs[jobid]["wildcards"] = wildcards
            if threads is not None:
                self._started_jobs[jobid]["threads"] = threads

    def set_job_log(self, jobid: str, log_path: str) -> None:
        """Set the log file path for a job.

        Args:
            jobid: Job identifier.
            log_path: Path to the log file.
        """
        self._job_logs[jobid] = log_path

    def get_job_log(self, jobid: str) -> str | None:
        """Get the log file path for a job.

        Args:
            jobid: Job identifier.

        Returns:
            Log file path or None if not set.
        """
        return self._job_logs.get(jobid)

    def finish_job(self, jobid: str, end_time: float | None = None) -> JobInfo | None:
        """Record a job finishing.

        Args:
            jobid: Job identifier.
            end_time: Unix timestamp when job finished.

        Returns:
            JobInfo for the completed job, or None if job was not tracked.
        """
        self._finished_jobids.add(jobid)

        if jobid in self._started_jobs:
            job_data = self._started_jobs[jobid]
            log_path = self._job_logs.get(jobid)
            job_info = JobInfo(
                rule=job_data["rule"],
                job_id=jobid,
                start_time=job_data["start_time"],
                end_time=end_time,
                wildcards=job_data["wildcards"],
                threads=job_data["threads"],
                log_file=Path(log_path) if log_path else None,
            )
            self._completed_jobs.append(job_info)
            # Clean up started job data to prevent memory growth
            del self._started_jobs[jobid]
            # Also clean up the job log entry
            self._job_logs.pop(jobid, None)
            return job_info
        return None

    def get_running_jobs(self) -> list[JobInfo]:
        """Get list of jobs that started but haven't finished.

        Returns:
            List of JobInfo for running jobs.
        """
        running: list[JobInfo] = []
        for jobid, job_data in self._started_jobs.items():
            if jobid not in self._finished_jobids:
                log_path = self._job_logs.get(jobid)
                running.append(
                    JobInfo(
                        rule=job_data["rule"],
                        job_id=jobid,
                        start_time=job_data["start_time"],
                        wildcards=job_data["wildcards"],
                        threads=job_data["threads"],
                        log_file=Path(log_path) if log_path else None,
                    )
                )
        return running

    def get_completed_jobs(self) -> list[JobInfo]:
        """Get list of completed jobs sorted by end time (newest first).

        Returns:
            List of JobInfo for completed jobs.
        """
        return sorted(
            self._completed_jobs,
            key=lambda j: j.end_time or 0,
            reverse=True,
        )

    def is_job_started(self, jobid: str) -> bool:
        """Check if a job has been started.

        Args:
            jobid: Job identifier.

        Returns:
            True if the job has been started.
        """
        return jobid in self._started_jobs

    def reset(self) -> None:
        """Clear all job tracking state."""
        self._started_jobs.clear()
        self._finished_jobids.clear()
        self._completed_jobs.clear()
        self._job_logs.clear()
