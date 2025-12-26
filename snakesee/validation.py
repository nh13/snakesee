"""Validation module for snakesee.

This module provides:
1. Parameter validation decorators and utilities for function arguments
2. State comparison between event-based tracking and log/metadata parsing

This module re-exports from focused submodules for backward compatibility:
    - snakesee.parameter_validation: Decorators and validation functions
    - snakesee.state_comparison: Event vs parsed state comparison

Related modules:
    snakesee.events: Event types and SnakeseeEvent dataclass
    snakesee.parser: Log file parsing for comparison
    snakesee.models: WorkflowProgress for state comparison
    snakesee.exceptions: InvalidParameterError for validation failures
"""

# Re-export from parameter_validation
from snakesee.parameter_validation import require_in_range
from snakesee.parameter_validation import require_non_negative
from snakesee.parameter_validation import require_not_empty
from snakesee.parameter_validation import require_positive
from snakesee.parameter_validation import validate_in_range
from snakesee.parameter_validation import validate_non_negative
from snakesee.parameter_validation import validate_not_empty
from snakesee.parameter_validation import validate_positive

# Re-export from state_comparison
from snakesee.state_comparison import DEFAULT_MAX_JOBS
from snakesee.state_comparison import VALIDATION_LOG_NAME
from snakesee.state_comparison import Discrepancy
from snakesee.state_comparison import DiscrepancyValue
from snakesee.state_comparison import EventAccumulator
from snakesee.state_comparison import JobState
from snakesee.state_comparison import ValidationLogger
from snakesee.state_comparison import compare_states

__all__ = [
    # Parameter validation decorators
    "validate_positive",
    "validate_non_negative",
    "validate_not_empty",
    "validate_in_range",
    # Inline validation functions
    "require_positive",
    "require_non_negative",
    "require_not_empty",
    "require_in_range",
    # State comparison
    "JobState",
    "EventAccumulator",
    "Discrepancy",
    "DiscrepancyValue",
    "ValidationLogger",
    "compare_states",
    "VALIDATION_LOG_NAME",
    "DEFAULT_MAX_JOBS",
]
