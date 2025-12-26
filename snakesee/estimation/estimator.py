"""Time estimation for Snakemake workflow progress."""

import math
from pathlib import Path
from statistics import mean
from typing import TYPE_CHECKING

from snakesee.estimation.data_loader import HistoricalDataLoader
from snakesee.estimation.pending_inferrer import PendingRuleInferrer
from snakesee.estimation.rule_matcher import RuleMatchingEngine
from snakesee.estimation.rule_matcher import levenshtein_distance
from snakesee.models import RuleTimingStats
from snakesee.models import ThreadTimingStats
from snakesee.models import TimeEstimate
from snakesee.models import WeightingStrategy
from snakesee.models import WildcardTimingStats
from snakesee.models import WorkflowProgress
from snakesee.state.config import DEFAULT_CONFIG
from snakesee.state.config import EstimationConfig

if TYPE_CHECKING:
    from snakesee.state.rule_registry import RuleRegistry
    from snakesee.types import ProgressCallback

# Backward compatibility alias
_levenshtein_distance = levenshtein_distance


class TimeEstimator:
    """
    Estimates remaining workflow time from historical data.

    Uses per-rule timing statistics from previous workflow runs to estimate
    how long the remaining jobs will take. Falls back to simple linear
    estimation when historical data is unavailable.

    Attributes:
        rule_stats: Dictionary mapping rule names to their timing statistics.
        thread_stats: Dictionary mapping rule names to thread-conditioned timing stats.
        wildcard_stats: Nested dict of wildcard-conditioned timing stats.
        use_wildcard_conditioning: Whether to use wildcard-specific estimates.
        config: Centralized estimation configuration.
    """

    def __init__(
        self,
        use_wildcard_conditioning: bool = False,
        half_life_days: float | None = None,
        weighting_strategy: WeightingStrategy | None = None,
        half_life_logs: int | None = None,
        config: EstimationConfig | None = None,
        rule_registry: "RuleRegistry | None" = None,
    ) -> None:
        """
        Initialize the estimator.

        Args:
            use_wildcard_conditioning: Whether to use wildcard-specific timing.
            half_life_days: Half-life in days for time-based weighting.
                           After this many days, a run's weight is halved.
                           Only used when weighting_strategy="time".
            weighting_strategy: Strategy for weighting historical data.
                              "time" - decay based on wall-clock time (good for stable pipelines)
                              "index" - decay based on run count (good for active development)
            half_life_logs: Half-life in log count for index-based weighting.
                           After this many runs, a run's weight is halved.
                           Only used when weighting_strategy="index".
            config: Centralized estimation configuration. If not provided, uses
                   DEFAULT_CONFIG with any explicit parameters overriding it.
            rule_registry: RuleRegistry for centralized statistics. If not provided,
                          creates an internal registry.
        """
        from snakesee.state.rule_registry import RuleRegistry

        # Use provided config or default
        self.config = config if config is not None else DEFAULT_CONFIG

        # Centralized registry - create internal one if not provided
        self._rule_registry: RuleRegistry = rule_registry or RuleRegistry(config=self.config)

        # Cache for global_mean_duration (invalidated when sample count changes)
        self._global_mean_cache: float | None = None
        self._global_mean_cache_sample_count: int = 0

        self.current_rules: set[str] | None = None  # Rules in current workflow (for filtering)
        self.code_hash_to_rules: dict[str, set[str]] = {}  # For renamed rule detection
        self.expected_job_counts: dict[str, int] | None = None  # Expected counts from Job stats
        self.use_wildcard_conditioning = use_wildcard_conditioning

        # Use explicit params if provided, otherwise use config values
        self.weighting_strategy: WeightingStrategy = (
            weighting_strategy if weighting_strategy is not None else self.config.weighting_strategy
        )
        self.half_life_days = (
            half_life_days if half_life_days is not None else self.config.half_life_days
        )
        self.half_life_logs = (
            half_life_logs if half_life_logs is not None else self.config.half_life_logs
        )

        # Helper components
        self._rule_matcher = RuleMatchingEngine(max_distance=self.config.fuzzy_match_max_distance)
        self._pending_inferrer = PendingRuleInferrer()
        self._data_loader: HistoricalDataLoader | None = None

    @property
    def rule_stats(self) -> dict[str, RuleTimingStats]:
        """Get rule stats dict from the registry.

        Returns a dict view for backward compatibility with code that reads rule_stats.
        """
        return self._rule_registry.to_rule_stats_dict()

    @rule_stats.setter
    def rule_stats(self, value: dict[str, RuleTimingStats]) -> None:
        """Set rule stats by loading into the registry.

        Supports backward compatibility with code that sets rule_stats directly.
        """
        self._rule_registry.load_from_rule_stats(value)

    @property
    def thread_stats(self) -> dict[str, ThreadTimingStats]:
        """Get thread stats dict from the registry."""
        return self._rule_registry.to_thread_stats_dict()

    @property
    def wildcard_stats(self) -> dict[str, dict[str, WildcardTimingStats]]:
        """Get wildcard stats dict from the registry."""
        return self._rule_registry.to_wildcard_stats_dict()

    def _get_data_loader(self) -> HistoricalDataLoader:
        """Get or create the data loader."""
        if self._data_loader is None:
            self._data_loader = HistoricalDataLoader(
                registry=self._rule_registry,
                use_wildcard_conditioning=self.use_wildcard_conditioning,
            )
        return self._data_loader

    def load_from_metadata(
        self,
        metadata_dir: Path,
        progress_callback: "ProgressCallback | None" = None,
    ) -> None:
        """
        Load historical execution times from .snakemake/metadata/.

        Uses a single-pass parser for efficiency - reads each metadata file
        only once to collect timing stats, code hashes, and wildcard stats.
        Data is recorded directly into the RuleRegistry.

        Args:
            metadata_dir: Path to .snakemake/metadata/ directory.
            progress_callback: Optional callback(current, total) for progress reporting.
        """
        loader = self._get_data_loader()
        loader.load_from_metadata(metadata_dir, progress_callback)
        self.code_hash_to_rules = loader.code_hash_to_rules

    def load_from_events(self, events_file: Path) -> None:
        """
        Load historical execution times from a snakesee events file.

        Parses the .snakesee_events.jsonl file to extract job durations from
        job_finished events. Data is recorded directly into the RuleRegistry.

        Args:
            events_file: Path to .snakesee_events.jsonl file.
        """
        loader = self._get_data_loader()
        has_wildcards = loader.load_from_events(events_file)

        # Auto-enable wildcard conditioning if we have wildcard data
        if has_wildcards:
            self.use_wildcard_conditioning = True

    def _find_rules_by_code_hash(self, rule: str) -> list[str]:
        """
        Find other rules that share the same code hash as the given rule.

        This helps detect renamed rules - if two rules have the same shell
        code but different names, they are likely the same rule renamed.

        Args:
            rule: The rule name to look up.

        Returns:
            List of other rule names that share the same code hash.
            Empty list if no code hash data or no matches.
        """
        effective_stats = self.rule_stats
        for _code_hash, rules in self.code_hash_to_rules.items():
            if rule in rules:
                # Return other rules in the same hash group
                return [r for r in rules if r != rule and r in effective_stats]
        return []

    def _find_similar_rule(
        self, rule: str, max_distance: int | None = None
    ) -> tuple[str, RuleTimingStats] | None:
        """
        Find the most similar known rule using code hash and Levenshtein distance.

        First checks if any known rule shares the same code hash (renamed rule).
        Then falls back to Levenshtein distance for similar names.

        Args:
            rule: The unknown rule name to match.
            max_distance: Maximum edit distance to consider a match.
                          Defaults to config.fuzzy_match_max_distance.

        Returns:
            Tuple of (matched_rule_name, stats) if a similar rule is found,
            None otherwise.
        """
        effective_stats = self.rule_stats
        if not effective_stats:
            return None

        return self._rule_matcher.find_best_match(
            rule=rule,
            code_hash_to_rules=self.code_hash_to_rules,
            rule_stats=effective_stats,
            max_distance=max_distance,
        )

    def _get_weighted_mean(self, stats: RuleTimingStats) -> float:
        """Get weighted mean using configured strategy."""
        return stats.weighted_mean(
            half_life_days=self.half_life_days,
            strategy=self.weighting_strategy,
            half_life_logs=self.half_life_logs,
        )

    def _get_size_scaled_estimate(
        self, stats: RuleTimingStats, input_size: int
    ) -> tuple[float, float]:
        """Get size-scaled estimate using configured strategy."""
        return stats.size_scaled_estimate(
            input_size=input_size,
            half_life_days=self.half_life_days,
            strategy=self.weighting_strategy,
            half_life_logs=self.half_life_logs,
        )

    def _get_recency_factor(self, stats: RuleTimingStats) -> float:
        """Get recency factor using configured strategy."""
        return stats.recency_factor(
            half_life_days=self.half_life_days,
            strategy=self.weighting_strategy,
            half_life_logs=self.half_life_logs,
        )

    def get_estimate_for_job(
        self,
        rule: str,
        wildcards: dict[str, str] | None = None,
        input_size: int | None = None,
        threads: int | None = None,
    ) -> tuple[float, float]:
        """
        Get expected duration and variance for a specific job.

        Uses thread-specific timing if available, then wildcard-specific timing
        if enabled, otherwise falls back to rule-level statistics.
        Optionally scales by input file size.

        Args:
            rule: The rule name.
            wildcards: Optional wildcard values for the job.
            input_size: Optional input file size in bytes for size-scaled estimates.
            threads: Optional thread count for thread-specific estimates.

        Returns:
            Tuple of (expected_duration, variance).
        """
        global_mean = self.global_mean_duration()

        # Try thread-specific stats first (highest priority for running job ETA)
        effective_thread_stats = self.thread_stats
        if threads is not None and rule in effective_thread_stats:
            thread_stats, matched_threads = effective_thread_stats[rule].get_best_match(threads)
            if thread_stats is not None:
                thread_mean = self._get_weighted_mean(thread_stats)
                var_mult = self.config.variance
                # Tighter variance for exact thread match, wider for aggregate fallback
                if matched_threads is not None:
                    thread_var = (
                        thread_stats.std_dev**2
                        if thread_stats.count > 1
                        else (thread_mean * var_mult.exact_thread_match) ** 2
                    )
                else:
                    # Aggregate fallback - use standard variance
                    thread_var = (
                        thread_stats.std_dev**2
                        if thread_stats.count > 1
                        else (thread_mean * var_mult.aggregate_fallback) ** 2
                    )
                return thread_mean, thread_var

        # Try wildcard-specific stats if enabled
        effective_wildcard_stats = self.wildcard_stats
        if self.use_wildcard_conditioning and wildcards and rule in effective_wildcard_stats:
            rule_wc_stats = effective_wildcard_stats[rule]

            # Find the most predictive wildcard key for this rule
            best_key = WildcardTimingStats.get_most_predictive_key(rule_wc_stats)

            if best_key is not None and best_key in wildcards:
                wc_value = wildcards[best_key]
                wts = rule_wc_stats[best_key]
                value_stats = wts.get_stats_for_value(wc_value)

                if value_stats is not None:
                    # Use wildcard-specific statistics
                    rule_mean = self._get_weighted_mean(value_stats)
                    rule_var = (
                        value_stats.std_dev**2
                        if value_stats.count > 1
                        else (rule_mean * self.config.variance.exact_wildcard_match) ** 2
                    )
                    return rule_mean, rule_var

        # Fall back to rule-level stats
        effective_stats = self.rule_stats
        if rule in effective_stats:
            stats = effective_stats[rule]

            # Try size-scaled estimate if input size is available
            if input_size is not None and input_size > 0:
                scaled_est, confidence = self._get_size_scaled_estimate(stats, input_size)
                size_conf_threshold = self.config.confidence_thresholds.size_scaling_min
                if confidence > size_conf_threshold:
                    # Reduce variance for size-scaled estimates
                    rule_var = (
                        stats.std_dev**2 * (1 - confidence * 0.5)
                        if stats.count > 1
                        else (scaled_est * self.config.variance.size_scaled) ** 2
                    )
                    return scaled_est, rule_var

            # Standard rule-level estimate
            rule_mean = self._get_weighted_mean(stats)
            rule_var = (
                stats.std_dev**2
                if stats.count > 1
                else (rule_mean * self.config.variance.rule_fallback) ** 2
            )
            return rule_mean, rule_var

        # Try fuzzy matching for renamed/similar rules before falling back to global mean
        similar = self._find_similar_rule(rule)
        if similar is not None:
            matched_rule, stats = similar
            rule_mean = self._get_weighted_mean(stats)
            # Wider variance for fuzzy matches due to uncertainty
            rule_var = (
                stats.std_dev**2
                if stats.count > 1
                else (rule_mean * self.config.variance.fuzzy_match) ** 2
            )
            return rule_mean, rule_var

        # No data available, use global mean
        return global_mean, (global_mean * self.config.variance.global_fallback) ** 2

    def global_mean_duration(self) -> float:
        """
        Get the global average duration across all known rules.

        Used as a fallback when a specific rule has no historical data.
        Result is cached and invalidated when sample count changes.

        Returns:
            Average duration in seconds, or config.default_global_mean if no data.
        """
        # Check cache validity using sample count as version indicator
        current_sample_count = self._rule_registry.total_sample_count()
        if (
            self._global_mean_cache is not None
            and current_sample_count == self._global_mean_cache_sample_count
        ):
            return self._global_mean_cache

        # Recalculate and cache
        effective_stats = self.rule_stats
        all_durations = [d for stats in effective_stats.values() for d in stats.durations]
        self._global_mean_cache = (
            mean(all_durations) if all_durations else self.config.default_global_mean
        )
        self._global_mean_cache_sample_count = current_sample_count
        return self._global_mean_cache

    def estimate_remaining(
        self,
        progress: WorkflowProgress,
        running_elapsed: dict[str, float] | None = None,
    ) -> TimeEstimate:
        """
        Estimate remaining time for a workflow.

        Uses one of several estimation strategies depending on available data:
        1. "weighted" - Uses per-rule historical timing with exponential weighting
        2. "simple" - Falls back to average time per completed step

        Args:
            progress: Current workflow progress state.
            running_elapsed: Optional dict mapping rule names to seconds already
                elapsed for running jobs. Used to subtract from estimates.

        Returns:
            TimeEstimate with expected time, confidence bounds, and method.
        """
        running_elapsed = running_elapsed or {}

        # Handle edge case: no jobs to do
        if progress.total_jobs == 0:
            return TimeEstimate(
                seconds_remaining=0.0,
                lower_bound=0.0,
                upper_bound=0.0,
                confidence=1.0,
                method="complete",
            )

        # Handle edge case: workflow complete
        if progress.completed_jobs >= progress.total_jobs:
            return TimeEstimate(
                seconds_remaining=0.0,
                lower_bound=0.0,
                upper_bound=0.0,
                confidence=1.0,
                method="complete",
            )

        # Handle edge case: nothing completed yet
        if progress.completed_jobs == 0:
            return self._estimate_no_progress(progress)

        # Try weighted estimation with historical data
        if self.rule_stats:
            return self._estimate_weighted(progress, running_elapsed)

        # Fall back to simple estimation
        return self._estimate_simple(progress)

    def _estimate_no_progress(self, progress: WorkflowProgress) -> TimeEstimate:
        """
        Estimate when no jobs have completed yet.

        Args:
            progress: Current workflow progress.

        Returns:
            TimeEstimate with very low confidence.
        """
        global_mean = self.global_mean_duration()
        estimate = global_mean * progress.total_jobs

        # Adjust for parallelism if jobs are running
        parallelism = max(1, len(progress.running_jobs))
        estimate = estimate / math.sqrt(parallelism)

        return TimeEstimate(
            seconds_remaining=estimate,
            lower_bound=estimate * self.config.bootstrap_lower_multiplier,
            upper_bound=estimate * self.config.bootstrap_upper_multiplier,
            confidence=self.config.confidence_thresholds.bootstrap_confidence,
            method="bootstrap",
        )

    def _estimate_simple(self, progress: WorkflowProgress) -> TimeEstimate:
        """
        Simple linear estimation based on average time per step.

        Args:
            progress: Current workflow progress.

        Returns:
            TimeEstimate using simple extrapolation.
        """
        elapsed = progress.elapsed_seconds
        if elapsed is None or elapsed <= 0:
            return self._estimate_no_progress(progress)

        # Guard against division by zero when no jobs completed yet
        if progress.completed_jobs <= 0:
            return self._estimate_no_progress(progress)

        avg_time_per_step = elapsed / progress.completed_jobs
        remaining_steps = progress.total_jobs - progress.completed_jobs
        estimate = avg_time_per_step * remaining_steps

        # Confidence grows with more completed jobs
        confidence = min(
            self.config.simple_estimate_confidence_cap,
            progress.completed_jobs / self.config.simple_estimate_jobs_divisor,
        )

        # Wider bounds for fewer samples
        uncertainty = max(0.3, 1.0 - (progress.completed_jobs / progress.total_jobs))

        return TimeEstimate(
            seconds_remaining=estimate,
            lower_bound=estimate * (1 - uncertainty),
            upper_bound=estimate * (1 + uncertainty * 2),
            confidence=confidence,
            method="simple",
        )

    def _estimate_parallelism(self, progress: WorkflowProgress, global_mean: float) -> float:
        """Estimate effective parallelism from workflow progress.

        Uses historical completion rate when available, otherwise falls back
        to current running job count with conservative clamping.

        Args:
            progress: Current workflow progress.
            global_mean: Global mean job duration.

        Returns:
            Estimated parallelism factor (>= 1.0).
        """
        elapsed = progress.elapsed_seconds
        if elapsed and elapsed > 0 and progress.completed_jobs > 0:
            # Infer historical parallelism from completion rate
            historical_rate = progress.completed_jobs / elapsed
            if historical_rate > 0:
                avg_job_time = global_mean
                historical_parallelism = historical_rate * avg_job_time
                # Clamp to reasonable range and use sqrt for conservative adjustment
                return max(1.0, min(8.0, math.sqrt(historical_parallelism)))
            return 1.0
        # Fall back to current running jobs but cap the effect
        return max(1.0, min(4.0, math.sqrt(len(progress.running_jobs))))

    def _calculate_confidence_scores(
        self,
        rule_completed: dict[str, int],
        effective_stats: dict[str, RuleTimingStats],
    ) -> tuple[float, float, float, float]:
        """Calculate confidence component scores.

        Args:
            rule_completed: Dict of rule name to completion count.
            effective_stats: Dict of rule name to timing stats.

        Returns:
            Tuple of (sample_confidence, avg_recency, avg_consistency, data_coverage).
        """
        data_coverage = len(rule_completed) / max(1, len(effective_stats))
        total_samples = sum(s.count for s in effective_stats.values())
        sample_confidence = min(1.0, total_samples / self.config.min_samples_for_confidence)

        # Calculate average recency and consistency across rules with data
        recency_scores: list[float] = []
        consistency_scores: list[float] = []
        for rule in rule_completed:
            if rule in effective_stats:
                stats = effective_stats[rule]
                recency_scores.append(self._get_recency_factor(stats))
                consistency_scores.append(stats.recent_consistency())

        avg_recency = sum(recency_scores) / len(recency_scores) if recency_scores else 0.5
        avg_consistency = (
            sum(consistency_scores) / len(consistency_scores) if consistency_scores else 0.5
        )

        return sample_confidence, avg_recency, avg_consistency, data_coverage

    def _estimate_weighted(
        self,
        progress: WorkflowProgress,
        running_elapsed: dict[str, float],  # noqa: ARG002 - kept for API compatibility
    ) -> TimeEstimate:
        """Weighted estimation using per-rule historical timing.

        Uses exponentially weighted moving averages favoring recent executions.

        Args:
            progress: Current workflow progress.
            running_elapsed: Dict mapping rule names to elapsed seconds (unused, for API compat).

        Returns:
            TimeEstimate using weighted historical data.
        """
        # Get rule completion counts from recent completions
        rule_completed: dict[str, int] = {}
        for job in progress.recent_completions:
            rule_completed[job.rule] = rule_completed.get(job.rule, 0) + 1

        # Count running jobs by rule
        running_by_rule: dict[str, int] = {}
        for job in progress.running_jobs:
            running_by_rule[job.rule] = running_by_rule.get(job.rule, 0) + 1

        # Only augment with historical counts if we don't have expected_job_counts
        # (to avoid skewing proportional inference with historical execution counts)
        effective_stats = self.rule_stats
        if not self.expected_job_counts:
            for rule, stats in effective_stats.items():
                if rule not in rule_completed:
                    rule_completed[rule] = stats.count

        # Estimate pending rule distribution
        # If expected_job_counts is set, _infer_pending_rules will use exact calculation
        pending = progress.total_jobs - progress.completed_jobs - len(progress.running_jobs)
        pending_rules = self._infer_pending_rules(
            rule_completed, pending, self.current_rules, running_by_rule
        )

        # Calculate expected remaining time
        total_expected = 0.0
        total_variance = 0.0
        global_mean = self.global_mean_duration()

        for rule, count in pending_rules.items():
            if rule in effective_stats:
                stats = effective_stats[rule]
                rule_mean = self._get_weighted_mean(stats)
                rule_var = (
                    stats.std_dev**2
                    if stats.count > 1
                    else (rule_mean * self.config.variance.rule_fallback) ** 2
                )
            else:
                # Unknown rule: use global mean with higher uncertainty
                rule_mean = global_mean
                rule_var = (global_mean * self.config.variance.global_fallback) ** 2

            total_expected += count * rule_mean
            total_variance += count * rule_var

        # Add time for running jobs - use wildcard-specific estimates when available
        for job in progress.running_jobs:
            # Use get_estimate_for_job which handles wildcard conditioning
            expected, variance = self.get_estimate_for_job(
                rule=job.rule,
                wildcards=job.wildcards,
                input_size=job.input_size,
                threads=job.threads,
            )

            # Use actual elapsed time if available
            elapsed = job.elapsed or 0.0
            remaining = max(0, expected - elapsed)
            total_expected += remaining
            total_variance += variance

        # Estimate parallelism and apply adjustment
        parallelism = self._estimate_parallelism(progress, global_mean)
        effective_time = total_expected / parallelism

        # Calculate confidence bounds
        fallback_std = effective_time * self.config.variance.rule_fallback
        std_dev = math.sqrt(total_variance) if total_variance > 0 else fallback_std
        lower = max(0, effective_time - 2 * std_dev)
        upper = effective_time + 2 * std_dev

        # Calculate confidence from multiple factors
        sample_conf, avg_recency, avg_consistency, data_coverage = (
            self._calculate_confidence_scores(rule_completed, effective_stats)
        )
        weights = self.config.confidence_weights
        base_confidence = (
            weights.sample_size * sample_conf
            + weights.recency * avg_recency
            + weights.consistency * avg_consistency
            + weights.data_coverage * data_coverage
        )
        confidence = min(self.config.confidence_thresholds.max_confidence, base_confidence)

        return TimeEstimate(
            seconds_remaining=effective_time,
            lower_bound=lower,
            upper_bound=upper,
            confidence=confidence,
            method="weighted",
        )

    def _infer_pending_rules(
        self,
        completed_by_rule: dict[str, int],
        pending_count: int,
        current_rules: set[str] | None = None,
        running_by_rule: dict[str, int] | None = None,
    ) -> dict[str, int]:
        """
        Infer the composition of pending rules.

        If expected_job_counts is available (from Job stats table), uses exact
        calculation: pending = expected - completed - running.
        Otherwise falls back to proportional inference.

        Args:
            completed_by_rule: Count of completed jobs per rule.
            pending_count: Total number of pending jobs.
            current_rules: Optional set of rules that exist in the current workflow.
                If provided, only rules in this set will be included in inference.
                This filters out deleted rules from historical data.
            running_by_rule: Optional count of running jobs per rule.

        Returns:
            Estimated count of pending jobs per rule.
        """
        return self._pending_inferrer.infer(
            completed_by_rule=completed_by_rule,
            pending_count=pending_count,
            expected_job_counts=self.expected_job_counts,
            current_rules=current_rules,
            running_by_rule=running_by_rule,
        )
