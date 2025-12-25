"""Centralized state management for snakesee.

This module provides a unified architecture for managing workflow state,
replacing scattered state across multiple components with a single source of truth.

Key classes:
- Clock: Injectable time source for testability
- EstimationConfig: Centralized configuration for estimation
- WorkflowPaths: Centralized path resolution
- JobRegistry: Single source of truth for job state (Phase 6)
- RuleRegistry: Centralized rule statistics (Phase 7)
- WorkflowState: Top-level state container (Phase 8)
"""

from snakesee.state.clock import Clock
from snakesee.state.clock import FrozenClock
from snakesee.state.clock import OffsetClock
from snakesee.state.clock import SystemClock
from snakesee.state.clock import get_clock
from snakesee.state.clock import reset_clock
from snakesee.state.clock import set_clock
from snakesee.state.config import ConfidenceThresholds
from snakesee.state.config import ConfidenceWeights
from snakesee.state.config import EstimationConfig
from snakesee.state.config import TimeConstants
from snakesee.state.config import VarianceMultipliers
from snakesee.state.job_registry import Job
from snakesee.state.job_registry import JobRegistry
from snakesee.state.job_registry import JobStatus
from snakesee.state.paths import WorkflowPaths
from snakesee.state.rule_registry import RuleRegistry
from snakesee.state.rule_registry import RuleStatistics
from snakesee.state.workflow_state import WorkflowState

__all__ = [
    # Clock
    "Clock",
    "SystemClock",
    "FrozenClock",
    "OffsetClock",
    "get_clock",
    "set_clock",
    "reset_clock",
    # Config
    "EstimationConfig",
    "VarianceMultipliers",
    "ConfidenceWeights",
    "ConfidenceThresholds",
    "TimeConstants",
    # Paths
    "WorkflowPaths",
    # Job Registry
    "Job",
    "JobRegistry",
    "JobStatus",
    # Rule Registry
    "RuleRegistry",
    "RuleStatistics",
    # Workflow State
    "WorkflowState",
    # Note: WorkflowStatus is in snakesee.models, not here
]
