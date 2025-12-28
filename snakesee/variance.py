"""Centralized variance and confidence calculations for time estimation.

This module encapsulates all uncertainty/variance calculation logic that was
previously duplicated across the estimator code.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from snakesee.models import RuleTimingStats
    from snakesee.state.config import EstimationConfig


@dataclass(frozen=True)
class VarianceResult:
    """Result of variance calculation with context.

    Attributes:
        variance: The computed variance value.
        source: How the variance was computed ("observed", "single_sample", "fallback").
        multiplier_used: The multiplier that was applied (if any).
    """

    variance: float
    source: str
    multiplier_used: float | None = None


@dataclass(frozen=True)
class ConfidenceResult:
    """Result of confidence calculation with breakdown.

    Attributes:
        confidence: Overall confidence score (0.0 to 1.0).
        sample_factor: Contribution from sample size.
        recency_factor: Contribution from data recency.
        consistency_factor: Contribution from run consistency.
        coverage_factor: Contribution from data coverage.
    """

    confidence: float
    sample_factor: float
    recency_factor: float
    consistency_factor: float
    coverage_factor: float


class VarianceCalculator:
    """Calculator for estimation uncertainty and confidence.

    This class centralizes all variance and confidence calculations that
    were previously scattered throughout the estimator code.

    Attributes:
        config: EstimationConfig containing all thresholds and multipliers.

    Example:
        from snakesee.state import EstimationConfig
        from snakesee.variance import VarianceCalculator

        calc = VarianceCalculator()

        # Calculate variance from statistics
        result = calc.calculate_variance(stats, scenario="rule_fallback")
        print(f"Variance: {result.variance}, Source: {result.source}")

        # Calculate confidence
        conf = calc.calculate_confidence(
            sample_count=20,
            recency_factor=0.8,
            consistency_factor=0.9,
            coverage_factor=0.7
        )
        print(f"Confidence: {conf.confidence}")
    """

    def __init__(self, config: EstimationConfig | None = None) -> None:
        """Initialize the calculator.

        Args:
            config: Configuration to use. Defaults to DEFAULT_CONFIG.
        """
        from snakesee.state.config import DEFAULT_CONFIG

        self.config = config or DEFAULT_CONFIG

    # =========================================================================
    # Core Variance Calculation
    # =========================================================================

    def calculate_variance(
        self,
        stats: RuleTimingStats,
        scenario: str = "rule_fallback",
        mean_override: float | None = None,
    ) -> VarianceResult:
        """Calculate variance for a set of timing statistics.

        Uses observed standard deviation when multiple samples are available,
        otherwise falls back to a multiplier-based estimate.

        Args:
            stats: Timing statistics to compute variance from.
            scenario: The estimation scenario (determines fallback multiplier).
                Valid values: "exact_thread_match", "exact_wildcard_match",
                "size_scaled", "rule_fallback", "aggregate_fallback",
                "fuzzy_match", "global_fallback", "bootstrap"
            mean_override: Optional mean to use instead of stats.weighted_mean().

        Returns:
            VarianceResult with the computed variance.
        """
        if stats.count > 1:
            return VarianceResult(
                variance=stats.std_dev**2,
                source="observed",
                multiplier_used=None,
            )

        # Single sample or no data: use multiplier
        multiplier = self._get_multiplier(scenario)
        mean = mean_override if mean_override is not None else stats.mean_duration

        if mean <= 0:
            mean = self.config.default_global_mean

        return VarianceResult(
            variance=(mean * multiplier) ** 2,
            source="single_sample" if stats.count == 1 else "fallback",
            multiplier_used=multiplier,
        )

    def calculate_variance_from_mean(
        self,
        mean: float,
        scenario: str = "global_fallback",
    ) -> VarianceResult:
        """Calculate variance when only mean is available (no stats).

        Args:
            mean: The mean duration.
            scenario: The estimation scenario.

        Returns:
            VarianceResult with fallback variance.
        """
        multiplier = self._get_multiplier(scenario)
        return VarianceResult(
            variance=(mean * multiplier) ** 2,
            source="fallback",
            multiplier_used=multiplier,
        )

    def adjust_variance_for_size_scaling(
        self,
        base_variance: float,
        size_confidence: float,
    ) -> float:
        """Adjust variance when using size-scaled estimates.

        Higher size-scaling confidence reduces variance.

        Args:
            base_variance: The base variance from stats.
            size_confidence: Confidence in the size scaling (0.0 to 1.0).

        Returns:
            Adjusted variance.
        """
        # Reduce variance based on size-scaling confidence
        # Formula: variance * (1 - confidence * 0.5)
        adjustment = 1.0 - (size_confidence * 0.5)
        return base_variance * adjustment

    def _get_multiplier(self, scenario: str) -> float:
        """Get the variance multiplier for a scenario.

        Args:
            scenario: The estimation scenario name.

        Returns:
            The multiplier value from config.
        """
        multipliers = self.config.variance
        return getattr(multipliers, scenario, multipliers.global_fallback)

    # =========================================================================
    # Confidence Calculation
    # =========================================================================

    def calculate_confidence(
        self,
        sample_count: int,
        recency_factor: float,
        consistency_factor: float,
        coverage_factor: float,
    ) -> ConfidenceResult:
        """Calculate overall confidence from component factors.

        Uses weighted combination of factors as defined in config.

        Args:
            sample_count: Total number of historical samples.
            recency_factor: How recent the data is (0.0 to 1.0).
            consistency_factor: How consistent recent runs are (0.0 to 1.0).
            coverage_factor: Fraction of rules with data (0.0 to 1.0).

        Returns:
            ConfidenceResult with overall confidence and breakdown.
        """
        weights = self.config.confidence_weights
        thresholds = self.config.confidence_thresholds

        # Normalize sample count to 0-1 range
        sample_factor = min(1.0, sample_count / self.config.min_samples_for_confidence)

        # Compute weighted sum
        confidence = (
            weights.sample_size * sample_factor
            + weights.recency * recency_factor
            + weights.consistency * consistency_factor
            + weights.data_coverage * coverage_factor
        )

        # Cap at maximum confidence
        confidence = min(thresholds.max_confidence, confidence)

        return ConfidenceResult(
            confidence=confidence,
            sample_factor=sample_factor,
            recency_factor=recency_factor,
            consistency_factor=consistency_factor,
            coverage_factor=coverage_factor,
        )

    def calculate_simple_confidence(
        self,
        completed_jobs: int,
        divisor: int | None = None,
        max_confidence: float | None = None,
    ) -> float:
        """Calculate simple confidence based on completed job count.

        Used for simple linear estimation when no historical data.

        Args:
            completed_jobs: Number of completed jobs.
            divisor: Denominator for the confidence fraction. Uses config default.
            max_confidence: Maximum confidence value. Uses config default.

        Returns:
            Confidence value.
        """
        if divisor is None:
            divisor = self.config.simple_estimate_jobs_divisor
        if max_confidence is None:
            max_confidence = self.config.simple_estimate_confidence_cap
        return min(max_confidence, completed_jobs / divisor)

    # =========================================================================
    # Aggregate Variance
    # =========================================================================

    def aggregate_variances(
        self,
        variances: list[float],
        counts: list[int] | None = None,
    ) -> float:
        """Aggregate multiple variances (e.g., across pending jobs).

        For independent jobs, variances add linearly.
        If counts are provided, each variance is multiplied by its count.

        Args:
            variances: List of variance values.
            counts: Optional counts for each variance (default: 1 each).

        Returns:
            Aggregated variance.
        """
        if counts is None:
            counts = [1] * len(variances)

        return sum(v * c for v, c in zip(variances, counts, strict=True))

    def std_dev_from_variance(self, total_variance: float) -> float:
        """Convert total variance to standard deviation.

        Args:
            total_variance: The total variance.

        Returns:
            Standard deviation (sqrt of variance).
        """
        return math.sqrt(total_variance) if total_variance > 0 else 0.0

    # =========================================================================
    # Confidence Bounds
    # =========================================================================

    def calculate_bounds(
        self,
        estimate: float,
        std_dev: float,
        num_std_devs: float = 2.0,
    ) -> tuple[float, float]:
        """Calculate confidence bounds from estimate and std dev.

        Args:
            estimate: The central estimate.
            std_dev: Standard deviation.
            num_std_devs: Number of standard deviations for bounds.

        Returns:
            Tuple of (lower_bound, upper_bound).
        """
        lower = max(0, estimate - num_std_devs * std_dev)
        upper = estimate + num_std_devs * std_dev
        return lower, upper

    def calculate_bootstrap_bounds(
        self,
        estimate: float,
    ) -> tuple[float, float]:
        """Calculate wide bounds for bootstrap estimation (no progress yet).

        Args:
            estimate: The central estimate.

        Returns:
            Tuple of (lower_bound, upper_bound).
        """
        # Very wide bounds: 20% to 300% of estimate
        return estimate * 0.2, estimate * 3.0


# Default calculator instance
_default_calculator: VarianceCalculator | None = None
_calculator_lock = threading.Lock()


def get_calculator(config: EstimationConfig | None = None) -> VarianceCalculator:
    """Get a variance calculator instance.

    Args:
        config: Optional config. If None, uses/creates default calculator.

    Returns:
        VarianceCalculator instance.
    """
    global _default_calculator

    if config is not None:
        return VarianceCalculator(config)

    if _default_calculator is None:
        with _calculator_lock:
            # Double-check after acquiring lock
            if _default_calculator is None:
                _default_calculator = VarianceCalculator()

    return _default_calculator
