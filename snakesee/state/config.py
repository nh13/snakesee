"""Centralized configuration for time estimation.

This module consolidates all estimation-related configuration that was
previously scattered across multiple files as magic numbers.

The EstimationConfig dataclass contains all thresholds, multipliers,
and settings used by the time estimation system.
"""

from dataclasses import dataclass
from dataclasses import field
from typing import Literal

from snakesee.constants import STALE_WORKFLOW_THRESHOLD_SECONDS

WeightingStrategy = Literal["time", "index"]


@dataclass(frozen=True)
class VarianceMultipliers:
    """Variance multipliers for different estimation scenarios.

    When only one observation exists, variance is computed as (mean * multiplier)^2.
    Lower multipliers = tighter confidence bounds.

    Attributes:
        exact_thread_match: High confidence - exact thread count match.
        exact_wildcard_match: High confidence - exact wildcard match.
        size_scaled: Medium confidence - input-size scaling.
        rule_fallback: Standard - rule-level fallback.
        aggregate_fallback: Standard - aggregate across thread counts.
        fuzzy_match: Lower confidence - fuzzy rule matching.
        global_fallback: Lowest confidence - no rule data.
        bootstrap: No progress yet - initial estimate.
    """

    exact_thread_match: float = 0.2
    exact_wildcard_match: float = 0.2
    size_scaled: float = 0.25
    rule_fallback: float = 0.3
    aggregate_fallback: float = 0.3
    fuzzy_match: float = 0.4
    global_fallback: float = 0.5
    bootstrap: float = 0.5


@dataclass(frozen=True)
class ConfidenceWeights:
    """Weights for computing overall confidence score.

    Used in _estimate_weighted() to combine multiple factors.
    All weights should sum to 1.0.

    Attributes:
        sample_size: Weight for historical sample count.
        recency: Weight for data recency.
        consistency: Weight for recent run consistency.
        data_coverage: Weight for fraction of rules with data.

    Raises:
        ValueError: If weights do not sum to 1.0.
    """

    sample_size: float = 0.4
    recency: float = 0.3
    consistency: float = 0.2
    data_coverage: float = 0.1

    def __post_init__(self) -> None:
        """Validate that weights sum to 1.0."""
        total = self.sample_size + self.recency + self.consistency + self.data_coverage
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"Confidence weights must sum to 1.0, got {total}")


@dataclass(frozen=True)
class ConfidenceThresholds:
    """Thresholds for confidence-based decisions.

    Attributes:
        size_scaling_min: Min confidence to use size-scaled estimate.
        high_confidence: Show simple estimate without range.
        medium_confidence: Show range.
        low_confidence: Show "rough" caveat.
        max_confidence: Cap to avoid overconfidence.
        bootstrap_confidence: Confidence when no progress yet.
    """

    size_scaling_min: float = 0.3
    high_confidence: float = 0.7
    medium_confidence: float = 0.4
    low_confidence: float = 0.1
    max_confidence: float = 0.9
    bootstrap_confidence: float = 0.05


@dataclass(frozen=True)
class TimeConstants:
    """Time-related constants used throughout the codebase.

    Attributes:
        seconds_per_day: Number of seconds in a day.
        seconds_per_hour: Number of seconds in an hour.
        seconds_per_minute: Number of seconds in a minute.
        stale_workflow_threshold: Seconds before workflow is considered stale.
        timing_mismatch_tolerance: Tolerance for validation timing comparisons.
    """

    seconds_per_day: int = 86400
    seconds_per_hour: int = 3600
    seconds_per_minute: int = 60
    stale_workflow_threshold: float = STALE_WORKFLOW_THRESHOLD_SECONDS
    timing_mismatch_tolerance: float = 5.0  # seconds


@dataclass(frozen=True)
class EstimationConfig:
    """Central configuration for time estimation.

    This class consolidates all estimation-related configuration that was
    previously scattered across multiple files as magic numbers.

    Attributes:
        weighting_strategy: How to weight historical data ("time" or "index").
        half_life_days: Half-life in days for time-based weighting.
        half_life_logs: Half-life in run count for index-based weighting.
        variance: Variance multiplier settings.
        confidence_weights: Weights for confidence calculation.
        confidence_thresholds: Decision thresholds based on confidence.
        time: Time-related constants.
        min_samples_for_conditioning: Minimum samples for wildcard conditioning.
        min_pairs_for_size_scaling: Minimum pairs for size correlation.
        high_variance_cv: Coefficient of variation threshold for "high variance".
        default_global_mean: Fallback duration when no data available.
        parallelism_max: Maximum effective parallelism for estimation.

    Raises:
        ValueError: If any parameter is invalid (e.g., non-positive half_life,
            parallelism_max < parallelism_min, etc.).
    """

    weighting_strategy: WeightingStrategy = "index"
    half_life_days: float = 7.0
    half_life_logs: int = 10

    variance: VarianceMultipliers = field(default_factory=VarianceMultipliers)
    confidence_weights: ConfidenceWeights = field(default_factory=ConfidenceWeights)
    confidence_thresholds: ConfidenceThresholds = field(default_factory=ConfidenceThresholds)
    time: TimeConstants = field(default_factory=TimeConstants)

    # Sample size thresholds
    # Synced with constants.MIN_SAMPLES_FOR_CONDITIONING
    min_samples_for_conditioning: int = 3
    min_pairs_for_size_scaling: int = 3
    min_samples_for_confidence: int = 10

    # Variance thresholds
    high_variance_cv: float = 0.5
    variance_ratio_threshold: float = 0.1

    # Default values
    default_global_mean: float = 60.0
    default_recency_factor: float = 0.5

    # Parallelism estimation bounds
    parallelism_min: float = 1.0
    parallelism_max: float = 8.0
    parallelism_fallback_max: float = 4.0

    # Size scaling bounds
    size_dampening_power: float = 0.5
    size_ratio_min: float = 0.5
    size_ratio_max: float = 2.0
    size_confidence_max: float = 0.8
    size_confidence_divisor: int = 10

    # Bootstrap estimation bounds (when no jobs completed)
    bootstrap_lower_multiplier: float = 0.2
    bootstrap_upper_multiplier: float = 3.0

    # Simple estimate bounds (used when no historical data)
    simple_estimate_confidence_cap: float = 0.7
    simple_estimate_jobs_divisor: int = 20

    # Fuzzy rule matching
    fuzzy_match_max_distance: int = 3

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.half_life_logs <= 0:
            raise ValueError(f"half_life_logs must be > 0, got {self.half_life_logs}")
        if self.half_life_days <= 0:
            raise ValueError(f"half_life_days must be > 0, got {self.half_life_days}")
        if self.parallelism_max < self.parallelism_min:
            raise ValueError(
                f"parallelism_max ({self.parallelism_max}) must be >= "
                f"parallelism_min ({self.parallelism_min})"
            )
        if self.parallelism_min <= 0:
            raise ValueError(f"parallelism_min must be > 0, got {self.parallelism_min}")
        if self.default_global_mean <= 0:
            raise ValueError(f"default_global_mean must be > 0, got {self.default_global_mean}")


# Global default instance
DEFAULT_CONFIG = EstimationConfig()
