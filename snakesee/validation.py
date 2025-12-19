"""Validation module for comparing event-based state with parsed state.

This module helps identify discrepancies between job status tracking via
the logger plugin events and log/metadata parsing, enabling bug detection
in either approach.
"""

import logging
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any

from snakesee.events import EventType
from snakesee.events import SnakeseeEvent
from snakesee.models import WorkflowProgress

# Validation log file name
VALIDATION_LOG_NAME = ".snakesee_validation.log"


@dataclass
class JobState:
    """Tracked state for a single job from events."""

    job_id: int
    rule_name: str
    wildcards: dict[str, str] | None = None
    status: str = "submitted"  # submitted, running, finished, error
    submit_time: float | None = None
    start_time: float | None = None
    end_time: float | None = None
    duration: float | None = None
    error_message: str | None = None


@dataclass
class EventAccumulator:
    """Accumulates events to build a complete workflow state.

    This tracks the full history of job state changes from events,
    allowing comparison with the parsed state at any point in time.
    """

    # Job states keyed by job_id
    jobs: dict[int, JobState] = field(default_factory=dict)

    # Workflow-level state
    total_jobs: int = 0
    completed_jobs: int = 0
    workflow_started: bool = False
    workflow_start_time: float | None = None

    def process_event(self, event: SnakeseeEvent) -> None:
        """Process a single event and update accumulated state."""
        if event.event_type == EventType.WORKFLOW_STARTED:
            self.workflow_started = True
            self.workflow_start_time = event.timestamp

        elif event.event_type == EventType.PROGRESS:
            if event.total_jobs is not None:
                self.total_jobs = event.total_jobs
            if event.completed_jobs is not None:
                self.completed_jobs = event.completed_jobs

        elif event.event_type == EventType.JOB_SUBMITTED:
            if event.job_id is not None:
                self.jobs[event.job_id] = JobState(
                    job_id=event.job_id,
                    rule_name=event.rule_name or "unknown",
                    wildcards=event.wildcards_dict,
                    status="submitted",
                    submit_time=event.timestamp,
                )

        elif event.event_type == EventType.JOB_STARTED:
            if event.job_id is not None:
                if event.job_id in self.jobs:
                    self.jobs[event.job_id].status = "running"
                    self.jobs[event.job_id].start_time = event.timestamp
                else:
                    # Job started without being submitted (shouldn't happen)
                    self.jobs[event.job_id] = JobState(
                        job_id=event.job_id,
                        rule_name=event.rule_name or "unknown",
                        wildcards=event.wildcards_dict,
                        status="running",
                        start_time=event.timestamp,
                    )

        elif event.event_type == EventType.JOB_FINISHED:
            if event.job_id is not None:
                if event.job_id in self.jobs:
                    self.jobs[event.job_id].status = "finished"
                    self.jobs[event.job_id].end_time = event.timestamp
                    self.jobs[event.job_id].duration = event.duration
                else:
                    self.jobs[event.job_id] = JobState(
                        job_id=event.job_id,
                        rule_name=event.rule_name or "unknown",
                        status="finished",
                        end_time=event.timestamp,
                        duration=event.duration,
                    )

        elif event.event_type == EventType.JOB_ERROR:
            if event.job_id is not None:
                if event.job_id in self.jobs:
                    self.jobs[event.job_id].status = "error"
                    self.jobs[event.job_id].end_time = event.timestamp
                    self.jobs[event.job_id].duration = event.duration
                    self.jobs[event.job_id].error_message = event.error_message
                else:
                    self.jobs[event.job_id] = JobState(
                        job_id=event.job_id,
                        rule_name=event.rule_name or "unknown",
                        status="error",
                        end_time=event.timestamp,
                        duration=event.duration,
                        error_message=event.error_message,
                    )

    def process_events(self, events: list[SnakeseeEvent]) -> None:
        """Process multiple events."""
        for event in events:
            self.process_event(event)

    @property
    def running_jobs(self) -> list[JobState]:
        """Get list of currently running jobs."""
        return [j for j in self.jobs.values() if j.status == "running"]

    @property
    def finished_jobs(self) -> list[JobState]:
        """Get list of finished jobs."""
        return [j for j in self.jobs.values() if j.status == "finished"]

    @property
    def failed_jobs(self) -> list[JobState]:
        """Get list of failed jobs."""
        return [j for j in self.jobs.values() if j.status == "error"]

    @property
    def submitted_jobs(self) -> list[JobState]:
        """Get list of submitted (pending) jobs."""
        return [j for j in self.jobs.values() if j.status == "submitted"]


@dataclass
class Discrepancy:
    """A discrepancy between event-based and parsed state."""

    category: str  # e.g., "running_count", "job_missing", "timing_mismatch"
    severity: str  # "info", "warning", "error"
    message: str
    event_value: Any = None
    parsed_value: Any = None
    job_id: int | None = None
    rule_name: str | None = None
    wildcards: dict[str, str] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result: dict[str, Any] = {
            "category": self.category,
            "severity": self.severity,
            "message": self.message,
        }
        if self.event_value is not None:
            result["event_value"] = self.event_value
        if self.parsed_value is not None:
            result["parsed_value"] = self.parsed_value
        if self.job_id is not None:
            result["job_id"] = self.job_id
        if self.rule_name is not None:
            result["rule_name"] = self.rule_name
        if self.wildcards is not None:
            result["wildcards"] = self.wildcards
        return result


class ValidationLogger:
    """Logs validation discrepancies to a file."""

    def __init__(self, workflow_dir: Path) -> None:
        """Initialize the validation logger.

        Args:
            workflow_dir: Path to the workflow directory.
        """
        self.log_path = workflow_dir / VALIDATION_LOG_NAME
        self._logger = logging.getLogger("snakesee.validation")
        self._handler: logging.FileHandler | None = None
        self._setup_logger()

    def _setup_logger(self) -> None:
        """Set up file logging."""
        self._handler = logging.FileHandler(self.log_path, mode="a")
        self._handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        self._logger.addHandler(self._handler)
        self._logger.setLevel(logging.DEBUG)

    def log_session_start(self) -> None:
        """Log the start of a validation session."""
        self._logger.info("=" * 60)
        self._logger.info("VALIDATION SESSION STARTED")
        self._logger.info("=" * 60)

    def log_discrepancy(self, discrepancy: Discrepancy) -> None:
        """Log a single discrepancy."""
        level = {
            "info": logging.INFO,
            "warning": logging.WARNING,
            "error": logging.ERROR,
        }.get(discrepancy.severity, logging.INFO)

        # Build detailed message
        parts = [f"[{discrepancy.category}] {discrepancy.message}"]

        if discrepancy.job_id is not None:
            parts.append(f"job_id={discrepancy.job_id}")
        if discrepancy.rule_name is not None:
            parts.append(f"rule={discrepancy.rule_name}")
        if discrepancy.wildcards:
            wc_str = ",".join(f"{k}={v}" for k, v in discrepancy.wildcards.items())
            parts.append(f"wildcards={{{wc_str}}}")
        if discrepancy.event_value is not None:
            parts.append(f"events={discrepancy.event_value}")
        if discrepancy.parsed_value is not None:
            parts.append(f"parsed={discrepancy.parsed_value}")

        self._logger.log(level, " | ".join(parts))

    def log_discrepancies(self, discrepancies: list[Discrepancy]) -> None:
        """Log multiple discrepancies."""
        for d in discrepancies:
            self.log_discrepancy(d)

    def log_summary(self, event_state: EventAccumulator, parsed: WorkflowProgress) -> None:
        """Log a summary of current state from both sources."""
        self._logger.debug(
            f"EVENT STATE: total={event_state.total_jobs}, "
            f"completed={event_state.completed_jobs}, "
            f"running={len(event_state.running_jobs)}, "
            f"failed={len(event_state.failed_jobs)}"
        )
        self._logger.debug(
            f"PARSED STATE: total={parsed.total_jobs}, "
            f"completed={parsed.completed_jobs}, "
            f"running={len(parsed.running_jobs)}, "
            f"failed={parsed.failed_jobs}"
        )

    def close(self) -> None:
        """Close the log handler."""
        if self._handler:
            self._logger.removeHandler(self._handler)
            self._handler.close()


def compare_states(event_state: EventAccumulator, parsed: WorkflowProgress) -> list[Discrepancy]:
    """Compare event-accumulated state with parsed state.

    Args:
        event_state: State accumulated from events.
        parsed: State from log/metadata parsing.

    Returns:
        List of discrepancies found.
    """
    discrepancies: list[Discrepancy] = []

    # Compare total jobs
    if event_state.total_jobs != parsed.total_jobs and event_state.total_jobs > 0:
        discrepancies.append(
            Discrepancy(
                category="total_jobs",
                severity="warning",
                message="Total job count mismatch",
                event_value=event_state.total_jobs,
                parsed_value=parsed.total_jobs,
            )
        )

    # Compare completed jobs
    if event_state.completed_jobs != parsed.completed_jobs and event_state.completed_jobs > 0:
        discrepancies.append(
            Discrepancy(
                category="completed_jobs",
                severity="warning",
                message="Completed job count mismatch",
                event_value=event_state.completed_jobs,
                parsed_value=parsed.completed_jobs,
            )
        )

    # Compare running job count
    event_running = len(event_state.running_jobs)
    parsed_running = len(parsed.running_jobs)
    if event_running != parsed_running:
        discrepancies.append(
            Discrepancy(
                category="running_count",
                severity="warning",
                message="Running job count mismatch",
                event_value=event_running,
                parsed_value=parsed_running,
            )
        )

    # Compare failed job count
    event_failed = len(event_state.failed_jobs)
    if event_failed != parsed.failed_jobs:
        discrepancies.append(
            Discrepancy(
                category="failed_count",
                severity="warning",
                message="Failed job count mismatch",
                event_value=event_failed,
                parsed_value=parsed.failed_jobs,
            )
        )

    # Check for jobs in events but not in parsed running list
    parsed_running_ids = {j.job_id for j in parsed.running_jobs if j.job_id}
    for event_job in event_state.running_jobs:
        job_id_str = str(event_job.job_id)
        if job_id_str not in parsed_running_ids:
            discrepancies.append(
                Discrepancy(
                    category="missing_running_job",
                    severity="error",
                    message="Job running per events but not found in parsed running list",
                    job_id=event_job.job_id,
                    rule_name=event_job.rule_name,
                    wildcards=event_job.wildcards,
                    event_value="running",
                    parsed_value="not found",
                )
            )

    # Check for jobs in parsed running list but not in events
    event_running_ids = {j.job_id for j in event_state.running_jobs}
    for parsed_job in parsed.running_jobs:
        if parsed_job.job_id:
            try:
                job_id_int = int(parsed_job.job_id)
                if job_id_int not in event_running_ids:
                    # Only report if we've seen any events for this workflow
                    if event_state.workflow_started:
                        discrepancies.append(
                            Discrepancy(
                                category="extra_running_job",
                                severity="warning",
                                message="Job in parsed running list but not tracked by events",
                                job_id=job_id_int,
                                rule_name=parsed_job.rule,
                                wildcards=parsed_job.wildcards,
                                event_value="not tracked",
                                parsed_value="running",
                            )
                        )
            except ValueError:
                pass  # Non-integer job ID

    # Check for timing discrepancies in running jobs
    for event_job in event_state.running_jobs:
        job_id_str = str(event_job.job_id)
        for parsed_job in parsed.running_jobs:
            if parsed_job.job_id == job_id_str:
                # Compare start times
                if event_job.start_time is not None and parsed_job.start_time is not None:
                    time_diff = abs(event_job.start_time - parsed_job.start_time)
                    if time_diff > 5.0:  # More than 5 seconds difference
                        discrepancies.append(
                            Discrepancy(
                                category="start_time_mismatch",
                                severity="info",
                                message=f"Start time differs by {time_diff:.1f}s",
                                job_id=event_job.job_id,
                                rule_name=event_job.rule_name,
                                wildcards=event_job.wildcards,
                                event_value=event_job.start_time,
                                parsed_value=parsed_job.start_time,
                            )
                        )
                break

    # Check for failed jobs in events but not in parsed
    parsed_failed_ids = {j.job_id for j in parsed.failed_jobs_list if j.job_id}
    for job in event_state.failed_jobs:
        job_id_str = str(job.job_id)
        if job_id_str not in parsed_failed_ids:
            discrepancies.append(
                Discrepancy(
                    category="missing_failed_job",
                    severity="error",
                    message="Job failed per events but not in parsed failed list",
                    job_id=job.job_id,
                    rule_name=job.rule_name,
                    wildcards=job.wildcards,
                    event_value=f"error: {job.error_message}",
                    parsed_value="not found",
                )
            )

    return discrepancies
