"""Time estimation for Snakemake workflow progress."""

import math
from pathlib import Path
from statistics import mean

from snakesee.models import RuleTimingStats
from snakesee.models import TimeEstimate
from snakesee.models import WildcardTimingStats
from snakesee.models import WorkflowProgress
from snakesee.parser import collect_rule_timing_stats
from snakesee.parser import collect_wildcard_timing_stats


class TimeEstimator:
    """
    Estimates remaining workflow time from historical data.

    Uses per-rule timing statistics from previous workflow runs to estimate
    how long the remaining jobs will take. Falls back to simple linear
    estimation when historical data is unavailable.

    Attributes:
        rule_stats: Dictionary mapping rule names to their timing statistics.
        wildcard_stats: Nested dict of wildcard-conditioned timing stats.
        use_wildcard_conditioning: Whether to use wildcard-specific estimates.
        half_life_days: Half-life in days for temporal weighting.
    """

    def __init__(
        self,
        rule_stats: dict[str, RuleTimingStats] | None = None,
        use_wildcard_conditioning: bool = False,
        half_life_days: float = 7.0,
    ) -> None:
        """
        Initialize the estimator.

        Args:
            rule_stats: Pre-loaded rule timing statistics. If None, must call
                load_from_metadata() before estimating.
            use_wildcard_conditioning: Whether to use wildcard-specific timing.
            half_life_days: Half-life in days for temporal weighting.
                           After this many days, a run's weight is halved.
        """
        self.rule_stats: dict[str, RuleTimingStats] = rule_stats or {}
        self.wildcard_stats: dict[str, dict[str, WildcardTimingStats]] = {}
        self.use_wildcard_conditioning = use_wildcard_conditioning
        self.half_life_days = half_life_days

    def load_from_metadata(self, metadata_dir: Path) -> None:
        """
        Load historical execution times from .snakemake/metadata/.

        Args:
            metadata_dir: Path to .snakemake/metadata/ directory.
        """
        self.rule_stats = collect_rule_timing_stats(metadata_dir)

        # Also load wildcard stats if conditioning is enabled
        if self.use_wildcard_conditioning:
            self.wildcard_stats = collect_wildcard_timing_stats(metadata_dir)

    def get_estimate_for_job(
        self,
        rule: str,
        wildcards: dict[str, str] | None = None,
        input_size: int | None = None,
    ) -> tuple[float, float]:
        """
        Get expected duration and variance for a specific job.

        Uses wildcard-specific timing if available and enabled, otherwise
        falls back to rule-level statistics. Optionally scales by input file size.

        Args:
            rule: The rule name.
            wildcards: Optional wildcard values for the job.
            input_size: Optional input file size in bytes for size-scaled estimates.

        Returns:
            Tuple of (expected_duration, variance).
        """
        global_mean = self.global_mean_duration()

        # Try wildcard-specific stats if enabled
        if self.use_wildcard_conditioning and wildcards and rule in self.wildcard_stats:
            rule_wc_stats = self.wildcard_stats[rule]

            # Find the most predictive wildcard key for this rule
            best_key = WildcardTimingStats.get_most_predictive_key(rule_wc_stats)

            if best_key is not None and best_key in wildcards:
                wc_value = wildcards[best_key]
                wts = rule_wc_stats[best_key]
                value_stats = wts.get_stats_for_value(wc_value)

                if value_stats is not None:
                    # Use wildcard-specific statistics
                    rule_mean = value_stats.weighted_mean(self.half_life_days)
                    rule_var = (
                        value_stats.std_dev**2
                        if value_stats.count > 1
                        else (rule_mean * 0.2) ** 2  # Tighter variance for wildcard match
                    )
                    return rule_mean, rule_var

        # Fall back to rule-level stats
        if rule in self.rule_stats:
            stats = self.rule_stats[rule]

            # Try size-scaled estimate if input size is available
            if input_size is not None and input_size > 0:
                scaled_est, confidence = stats.size_scaled_estimate(input_size, self.half_life_days)
                if confidence > 0.3:  # Only use if we have reasonable confidence
                    # Reduce variance for size-scaled estimates
                    rule_var = (
                        stats.std_dev**2 * (1 - confidence * 0.5)
                        if stats.count > 1
                        else (scaled_est * 0.25) ** 2
                    )
                    return scaled_est, rule_var

            # Standard rule-level estimate
            rule_mean = stats.weighted_mean(self.half_life_days)
            rule_var = stats.std_dev**2 if stats.count > 1 else (rule_mean * 0.3) ** 2
            return rule_mean, rule_var

        # No data available, use global mean
        return global_mean, (global_mean * 0.5) ** 2

    def global_mean_duration(self) -> float:
        """
        Get the global average duration across all known rules.

        Used as a fallback when a specific rule has no historical data.

        Returns:
            Average duration in seconds, or 60.0 if no data available.
        """
        all_durations = [d for stats in self.rule_stats.values() for d in stats.durations]
        return mean(all_durations) if all_durations else 60.0

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
            lower_bound=estimate * 0.2,
            upper_bound=estimate * 3.0,
            confidence=0.05,
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

        avg_time_per_step = elapsed / progress.completed_jobs
        remaining_steps = progress.total_jobs - progress.completed_jobs
        estimate = avg_time_per_step * remaining_steps

        # Confidence grows with more completed jobs
        confidence = min(0.7, progress.completed_jobs / 20)

        # Wider bounds for fewer samples
        uncertainty = max(0.3, 1.0 - (progress.completed_jobs / progress.total_jobs))

        return TimeEstimate(
            seconds_remaining=estimate,
            lower_bound=estimate * (1 - uncertainty),
            upper_bound=estimate * (1 + uncertainty * 2),
            confidence=confidence,
            method="simple",
        )

    def _estimate_weighted(  # noqa: C901
        self,
        progress: WorkflowProgress,
        running_elapsed: dict[str, float],
    ) -> TimeEstimate:
        """
        Weighted estimation using per-rule historical timing.

        Uses exponentially weighted moving averages favoring recent executions.

        Args:
            progress: Current workflow progress.
            running_elapsed: Dict mapping rule names to elapsed seconds.

        Returns:
            TimeEstimate using weighted historical data.
        """
        # Get rule completion counts from recent completions
        rule_completed: dict[str, int] = {}
        for job in progress.recent_completions:
            rule_completed[job.rule] = rule_completed.get(job.rule, 0) + 1

        # Estimate pending rule distribution (assume proportional to completed)
        pending = progress.total_jobs - progress.completed_jobs - len(progress.running_jobs)
        pending_rules = self._infer_pending_rules(rule_completed, pending)

        # Calculate expected remaining time
        total_expected = 0.0
        total_variance = 0.0
        global_mean = self.global_mean_duration()

        for rule, count in pending_rules.items():
            if rule in self.rule_stats:
                stats = self.rule_stats[rule]
                rule_mean = stats.weighted_mean(self.half_life_days)
                rule_var = stats.std_dev**2 if stats.count > 1 else (rule_mean * 0.3) ** 2
            else:
                # Unknown rule: use global mean with higher uncertainty
                rule_mean = global_mean
                rule_var = (global_mean * 0.5) ** 2

            total_expected += count * rule_mean
            total_variance += count * rule_var

        # Add time for running jobs (subtract elapsed time)
        for rule, elapsed in running_elapsed.items():
            if rule in self.rule_stats:
                expected = self.rule_stats[rule].weighted_mean(self.half_life_days)
            else:
                expected = global_mean
            remaining = max(0, expected - elapsed)
            total_expected += remaining
            total_variance += (expected * 0.3) ** 2  # Uncertainty for running jobs

        # Add time for running jobs we don't have elapsed info for
        unknown_running = len(progress.running_jobs) - len(running_elapsed)
        if unknown_running > 0:
            total_expected += unknown_running * global_mean * 0.5  # Assume halfway done
            total_variance += unknown_running * (global_mean * 0.5) ** 2

        # Estimate parallelism from historical completion rate (more stable than current)
        # Use a conservative parallelism estimate based on completed jobs and elapsed time
        elapsed = progress.elapsed_seconds
        if elapsed and elapsed > 0 and progress.completed_jobs > 0:
            # Infer historical parallelism from completion rate
            historical_rate = progress.completed_jobs / elapsed
            if historical_rate > 0:
                avg_job_time = global_mean
                historical_parallelism = historical_rate * avg_job_time
                # Clamp to reasonable range and use sqrt for conservative adjustment
                parallelism = max(1.0, min(8.0, math.sqrt(historical_parallelism)))
            else:
                parallelism = 1.0
        else:
            # Fall back to current running jobs but cap the effect
            parallelism = max(1.0, min(4.0, math.sqrt(len(progress.running_jobs))))

        # Apply parallelism adjustment conservatively (use sqrt to dampen swings)
        effective_time = total_expected / parallelism

        # Calculate confidence bounds
        std_dev = math.sqrt(total_variance) if total_variance > 0 else effective_time * 0.3
        lower = max(0, effective_time - 2 * std_dev)
        upper = effective_time + 2 * std_dev

        # Confidence based on data quality, recency, and consistency
        data_coverage = len(rule_completed) / max(1, len(self.rule_stats))
        sample_confidence = min(1.0, sum(s.count for s in self.rule_stats.values()) / 10)

        # Calculate average recency and consistency across rules with data
        recency_scores: list[float] = []
        consistency_scores: list[float] = []
        for rule in rule_completed:
            if rule in self.rule_stats:
                stats = self.rule_stats[rule]
                recency_scores.append(stats.recency_factor(self.half_life_days))
                consistency_scores.append(stats.recent_consistency())

        avg_recency = sum(recency_scores) / len(recency_scores) if recency_scores else 0.5
        avg_consistency = (
            sum(consistency_scores) / len(consistency_scores) if consistency_scores else 0.5
        )

        # Combine factors: 40% sample size, 30% recency, 20% consistency, 10% coverage
        base_confidence = (
            0.4 * sample_confidence
            + 0.3 * avg_recency
            + 0.2 * avg_consistency
            + 0.1 * data_coverage
        )
        confidence = min(0.9, base_confidence)

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
    ) -> dict[str, int]:
        """
        Infer the composition of pending rules.

        Assumes pending jobs follow the same proportion as completed jobs.
        This works well for regular workflows (e.g., same rules per sample).

        Args:
            completed_by_rule: Count of completed jobs per rule.
            pending_count: Total number of pending jobs.

        Returns:
            Estimated count of pending jobs per rule.
        """
        if not completed_by_rule or pending_count <= 0:
            return {}

        total_completed = sum(completed_by_rule.values())
        if total_completed == 0:
            return {}

        pending_rules: dict[str, int] = {}
        for rule, count in completed_by_rule.items():
            proportion = count / total_completed
            pending_rules[rule] = max(1, int(pending_count * proportion))

        return pending_rules
