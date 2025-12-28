"""Unified workflow state container.

This module provides the top-level WorkflowState class that serves as
the single source of truth for all workflow state, combining job tracking,
rule statistics, paths, and configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from snakesee.models import WorkflowProgress
    from snakesee.models import WorkflowStatus
    from snakesee.state.clock import Clock
    from snakesee.state.config import EstimationConfig
    from snakesee.state.job_registry import JobRegistry
    from snakesee.state.paths import WorkflowPaths
    from snakesee.state.rule_registry import RuleRegistry


def _default_status() -> WorkflowStatus:
    """Return default status (deferred import to avoid circular import)."""
    from snakesee.models import WorkflowStatus

    return WorkflowStatus.UNKNOWN


@dataclass
class WorkflowState:
    """Unified container for all workflow state.

    This is the single source of truth for workflow state, consolidating
    what was previously scattered across IncrementalLogReader, WorkflowProgress,
    TUI, EventAccumulator, and TimeEstimator.

    Attributes:
        paths: Centralized path resolution.
        jobs: Registry of all job state.
        rules: Registry of all rule timing statistics.
        clock: Injectable time source.
        config: Estimation configuration.
        status: Overall workflow status.
        total_jobs: Total number of jobs in workflow (if known).
        start_time: Workflow start timestamp.
        current_log: Path to current log file being monitored.
    """

    paths: WorkflowPaths
    jobs: JobRegistry
    rules: RuleRegistry
    clock: Clock
    config: EstimationConfig
    status: WorkflowStatus = field(default_factory=_default_status)
    total_jobs: int | None = None
    start_time: float | None = None
    current_log: Path | None = None

    @property
    def completed_count(self) -> int:
        """Number of completed jobs."""
        return len(self.jobs.completed())

    @property
    def failed_count(self) -> int:
        """Number of failed jobs."""
        return len(self.jobs.failed())

    @property
    def running_count(self) -> int:
        """Number of currently running jobs."""
        return len(self.jobs.running())

    @property
    def pending_count(self) -> int:
        """Number of pending jobs."""
        from snakesee.state.job_registry import JobStatus

        return len([j for j in self.jobs.all_jobs() if j.status == JobStatus.PENDING])

    @property
    def progress_fraction(self) -> float:
        """Fraction of jobs completed (0.0 to 1.0)."""
        if self.total_jobs is None or self.total_jobs == 0:
            return 0.0
        return self.completed_count / self.total_jobs

    @property
    def elapsed_seconds(self) -> float | None:
        """Elapsed time since workflow start."""
        if self.start_time is None:
            return None
        return self.clock.now() - self.start_time

    def update_status(self) -> None:
        """Update workflow status based on current state."""
        from snakesee.models import WorkflowStatus

        if self.total_jobs is None:
            # Unknown total - check if we have failures anyway
            if self.failed_count > 0:
                self.status = WorkflowStatus.FAILED
            elif self.running_count > 0:
                self.status = WorkflowStatus.RUNNING
            else:
                self.status = WorkflowStatus.UNKNOWN
        elif self.running_count > 0:
            self.status = WorkflowStatus.RUNNING
        elif self.failed_count > 0 and self.completed_count < self.total_jobs:
            self.status = WorkflowStatus.FAILED
        elif self.completed_count >= self.total_jobs:
            self.status = WorkflowStatus.COMPLETED
        elif self.completed_count == 0 and self.running_count == 0:
            self.status = WorkflowStatus.NOT_STARTED
        else:
            # Check for stale workflow
            if self.elapsed_seconds is not None:
                if self.elapsed_seconds > self.config.time.stale_workflow_threshold:
                    self.status = WorkflowStatus.STALE
                else:
                    self.status = WorkflowStatus.RUNNING
            else:
                # No start time - assume incomplete
                self.status = WorkflowStatus.INCOMPLETE

    def record_job_completion(self, job_key: str) -> None:
        """Record a job completion and update rule statistics.

        Args:
            job_key: Key of the completed job.
        """
        job = self.jobs.get(job_key)
        if job is not None and job.duration is not None:
            self.rules.record_job_completion(job)

    def to_progress(self) -> WorkflowProgress:
        """Convert to WorkflowProgress for backward compatibility.

        Returns:
            WorkflowProgress snapshot of current state.
        """
        from snakesee.models import WorkflowProgress

        return WorkflowProgress(
            workflow_dir=self.paths.workflow_dir,
            status=self.status,
            total_jobs=self.total_jobs or 0,
            completed_jobs=self.completed_count,
            running_jobs=[job.to_job_info() for job in self.jobs.running()],
            failed_jobs=self.failed_count,
            failed_jobs_list=[job.to_job_info() for job in self.jobs.failed()],
            pending_jobs_list=[job.to_job_info() for job in self.jobs.submitted()],
            start_time=self.start_time,
            log_file=self.current_log,
        )

    def clear(self) -> None:
        """Clear all state."""
        from snakesee.models import WorkflowStatus

        self.jobs.clear()
        self.rules.clear()
        self.status = WorkflowStatus.UNKNOWN
        self.total_jobs = None
        self.start_time = None
        self.current_log = None

    @classmethod
    def create(
        cls,
        workflow_dir: Path,
        clock: Clock | None = None,
        config: EstimationConfig | None = None,
    ) -> WorkflowState:
        """Create a new WorkflowState for a workflow directory.

        Args:
            workflow_dir: Path to the workflow directory.
            clock: Optional clock for time operations.
            config: Optional estimation configuration.

        Returns:
            New WorkflowState instance.
        """
        from snakesee.state.clock import get_clock
        from snakesee.state.config import DEFAULT_CONFIG
        from snakesee.state.job_registry import JobRegistry
        from snakesee.state.paths import WorkflowPaths
        from snakesee.state.rule_registry import RuleRegistry

        actual_clock = clock or get_clock()
        actual_config = config or DEFAULT_CONFIG

        return cls(
            paths=WorkflowPaths(workflow_dir=workflow_dir),
            jobs=JobRegistry(),
            rules=RuleRegistry(config=actual_config),
            clock=actual_clock,
            config=actual_config,
        )
