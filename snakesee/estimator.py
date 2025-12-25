"""Time estimation for Snakemake workflow progress."""

import math
from collections.abc import Callable
from pathlib import Path
from statistics import mean

from snakesee.models import RuleTimingStats
from snakesee.models import ThreadTimingStats
from snakesee.models import TimeEstimate
from snakesee.models import WeightingStrategy
from snakesee.models import WildcardTimingStats
from snakesee.models import WorkflowProgress
from snakesee.parser import parse_metadata_files_full


def _levenshtein_distance(s1: str, s2: str) -> int:
    """
    Calculate the Levenshtein distance between two strings.

    Args:
        s1: First string.
        s2: Second string.

    Returns:
        The minimum number of edits (insertions, deletions, substitutions)
        needed to transform s1 into s2.
    """
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row: list[int] = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            # Cost is 0 if characters match, 1 otherwise
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


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
        weighting_strategy: Strategy for weighting historical data ("time" or "index").
        half_life_days: Half-life in days for time-based weighting.
        half_life_logs: Half-life in log count for index-based weighting.
    """

    def __init__(
        self,
        rule_stats: dict[str, RuleTimingStats] | None = None,
        use_wildcard_conditioning: bool = False,
        half_life_days: float = 7.0,
        weighting_strategy: WeightingStrategy = "index",
        half_life_logs: int = 10,
    ) -> None:
        """
        Initialize the estimator.

        Args:
            rule_stats: Pre-loaded rule timing statistics. If None, must call
                load_from_metadata() before estimating.
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
        """
        self.rule_stats: dict[str, RuleTimingStats] = rule_stats or {}
        self.thread_stats: dict[str, ThreadTimingStats] = {}
        self.wildcard_stats: dict[str, dict[str, WildcardTimingStats]] = {}
        self.current_rules: set[str] | None = None  # Rules in current workflow (for filtering)
        self.code_hash_to_rules: dict[str, set[str]] = {}  # For renamed rule detection
        self.expected_job_counts: dict[str, int] | None = None  # Expected counts from Job stats
        self.use_wildcard_conditioning = use_wildcard_conditioning
        self.weighting_strategy = weighting_strategy
        self.half_life_days = half_life_days
        self.half_life_logs = half_life_logs

    def load_from_metadata(
        self,
        metadata_dir: Path,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> None:
        """
        Load historical execution times from .snakemake/metadata/.

        Uses a single-pass parser for efficiency - reads each metadata file
        only once to collect timing stats, code hashes, and wildcard stats.

        Args:
            metadata_dir: Path to .snakemake/metadata/ directory.
            progress_callback: Optional callback(current, total) for progress reporting.
        """
        # Collect all data in temporary structures
        # rule -> [(duration, end_time, input_size), ...]
        jobs_by_rule: dict[str, list[tuple[float, float, int | None]]] = {}
        # code_hash -> set of rule names
        hash_to_rules: dict[str, set[str]] = {}
        # rule -> wildcard_key -> wildcard_value -> [(duration, end_time), ...]
        wildcard_data: dict[str, dict[str, dict[str, list[tuple[float, float]]]]] = {}

        # Single pass through all metadata files
        for record in parse_metadata_files_full(metadata_dir, progress_callback):
            duration = record.duration
            end_time = record.end_time

            # Collect timing stats
            if duration is not None and end_time is not None:
                if record.rule not in jobs_by_rule:
                    jobs_by_rule[record.rule] = []
                jobs_by_rule[record.rule].append((duration, end_time, record.input_size))

                # Collect wildcard stats if enabled
                if self.use_wildcard_conditioning and record.wildcards:
                    if record.rule not in wildcard_data:
                        wildcard_data[record.rule] = {}
                    for wc_key, wc_value in record.wildcards.items():
                        if wc_key not in wildcard_data[record.rule]:
                            wildcard_data[record.rule][wc_key] = {}
                        if wc_value not in wildcard_data[record.rule][wc_key]:
                            wildcard_data[record.rule][wc_key][wc_value] = []
                        wildcard_data[record.rule][wc_key][wc_value].append((duration, end_time))

            # Collect code hashes
            if record.code_hash:
                if record.code_hash not in hash_to_rules:
                    hash_to_rules[record.code_hash] = set()
                hash_to_rules[record.code_hash].add(record.rule)

        # Build RuleTimingStats from collected data
        self.rule_stats = {}
        for rule, timing_tuples in jobs_by_rule.items():
            timing_tuples.sort(key=lambda x: x[1])  # Sort by end_time
            durations = [t[0] for t in timing_tuples]
            timestamps = [t[1] for t in timing_tuples]
            input_sizes = [t[2] for t in timing_tuples]
            self.rule_stats[rule] = RuleTimingStats(
                rule=rule,
                durations=durations,
                timestamps=timestamps,
                input_sizes=input_sizes,
            )

        # Store code hashes
        self.code_hash_to_rules = hash_to_rules

        # Build WildcardTimingStats if enabled
        if self.use_wildcard_conditioning:
            self.wildcard_stats = {}
            for rule, wc_keys in wildcard_data.items():
                self.wildcard_stats[rule] = {}
                for wc_key, wc_values in wc_keys.items():
                    stats_by_value: dict[str, RuleTimingStats] = {}
                    for wc_value, timing_pairs in wc_values.items():
                        timing_pairs.sort(key=lambda x: x[1])
                        durations = [pair[0] for pair in timing_pairs]
                        timestamps = [pair[1] for pair in timing_pairs]
                        stats_by_value[wc_value] = RuleTimingStats(
                            rule=f"{rule}:{wc_key}={wc_value}",
                            durations=durations,
                            timestamps=timestamps,
                        )
                    self.wildcard_stats[rule][wc_key] = WildcardTimingStats(
                        rule=rule,
                        wildcard_key=wc_key,
                        stats_by_value=stats_by_value,
                    )

    def load_from_events(self, events_file: Path) -> None:  # noqa: C901
        """
        Load historical execution times from a snakesee events file.

        Parses the .snakesee_events.jsonl file to extract job durations from
        job_finished events. This complements or replaces metadata-based loading.
        Also builds wildcard-specific timing stats for accurate per-sample estimates.

        Args:
            events_file: Path to .snakesee_events.jsonl file.
        """
        import json

        if not events_file.exists():
            return

        # Collect timing data: rule -> [(duration, timestamp), ...]
        jobs_by_rule: dict[str, list[tuple[float, float]]] = {}
        # Collect wildcard data: rule -> wildcard_key -> wildcard_value -> [(duration, timestamp)]
        wildcard_data: dict[str, dict[str, dict[str, list[tuple[float, float]]]]] = {}

        try:
            for line in events_file.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # Only process job_finished events with duration
                if event.get("event_type") != "job_finished":
                    continue
                duration = event.get("duration")
                timestamp = event.get("timestamp")
                rule_name = event.get("rule_name")
                wildcards = event.get("wildcards")

                if duration is None or timestamp is None or rule_name is None:
                    continue

                if rule_name not in jobs_by_rule:
                    jobs_by_rule[rule_name] = []
                jobs_by_rule[rule_name].append((duration, timestamp))

                # Collect wildcard-specific timing
                if wildcards and isinstance(wildcards, dict):
                    if rule_name not in wildcard_data:
                        wildcard_data[rule_name] = {}
                    for wc_key, wc_value in wildcards.items():
                        if wc_key not in wildcard_data[rule_name]:
                            wildcard_data[rule_name][wc_key] = {}
                        if wc_value not in wildcard_data[rule_name][wc_key]:
                            wildcard_data[rule_name][wc_key][wc_value] = []
                        wildcard_data[rule_name][wc_key][wc_value].append((duration, timestamp))

        except OSError:
            return

        # Merge into rule_stats (add to existing or create new)
        for rule, timing_tuples in jobs_by_rule.items():
            timing_tuples.sort(key=lambda x: x[1])  # Sort by timestamp
            durations = [t[0] for t in timing_tuples]
            timestamps = [t[1] for t in timing_tuples]

            if rule in self.rule_stats:
                # Merge with existing stats
                existing = self.rule_stats[rule]
                # Combine and re-sort by timestamp
                combined = list(zip(existing.durations, existing.timestamps, strict=False))
                combined.extend(timing_tuples)
                combined.sort(key=lambda x: x[1])
                self.rule_stats[rule] = RuleTimingStats(
                    rule=rule,
                    durations=[t[0] for t in combined],
                    timestamps=[t[1] for t in combined],
                    input_sizes=existing.input_sizes,  # Preserve from metadata
                )
            else:
                # Create new stats
                self.rule_stats[rule] = RuleTimingStats(
                    rule=rule,
                    durations=durations,
                    timestamps=timestamps,
                )

        # Build wildcard stats from events (enables wildcard conditioning)
        for rule, wc_keys in wildcard_data.items():
            if rule not in self.wildcard_stats:
                self.wildcard_stats[rule] = {}
            for wc_key, wc_values in wc_keys.items():
                stats_by_value: dict[str, RuleTimingStats] = {}
                for wc_value, timing_pairs in wc_values.items():
                    timing_pairs.sort(key=lambda x: x[1])
                    durations = [pair[0] for pair in timing_pairs]
                    timestamps = [pair[1] for pair in timing_pairs]
                    stats_by_value[wc_value] = RuleTimingStats(
                        rule=f"{rule}:{wc_key}={wc_value}",
                        durations=durations,
                        timestamps=timestamps,
                    )
                # Merge or create WildcardTimingStats
                if wc_key in self.wildcard_stats[rule]:
                    # Merge with existing
                    existing_wts = self.wildcard_stats[rule][wc_key]
                    for wc_value, new_stats in stats_by_value.items():
                        if wc_value in existing_wts.stats_by_value:
                            # Combine durations/timestamps
                            old = existing_wts.stats_by_value[wc_value]
                            combined = list(zip(old.durations, old.timestamps, strict=False))
                            new_pairs = zip(new_stats.durations, new_stats.timestamps, strict=False)
                            combined.extend(new_pairs)
                            combined.sort(key=lambda x: x[1])
                            existing_wts.stats_by_value[wc_value] = RuleTimingStats(
                                rule=f"{rule}:{wc_key}={wc_value}",
                                durations=[t[0] for t in combined],
                                timestamps=[t[1] for t in combined],
                            )
                        else:
                            existing_wts.stats_by_value[wc_value] = new_stats
                else:
                    self.wildcard_stats[rule][wc_key] = WildcardTimingStats(
                        rule=rule,
                        wildcard_key=wc_key,
                        stats_by_value=stats_by_value,
                    )

        # Auto-enable wildcard conditioning if we have wildcard data
        if wildcard_data:
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
        for _code_hash, rules in self.code_hash_to_rules.items():
            if rule in rules:
                # Return other rules in the same hash group
                return [r for r in rules if r != rule and r in self.rule_stats]
        return []

    def _find_similar_rule(
        self, rule: str, max_distance: int = 3
    ) -> tuple[str, RuleTimingStats] | None:
        """
        Find the most similar known rule using code hash and Levenshtein distance.

        First checks if any known rule shares the same code hash (renamed rule).
        Then falls back to Levenshtein distance for similar names.

        Args:
            rule: The unknown rule name to match.
            max_distance: Maximum edit distance to consider a match (default 3).

        Returns:
            Tuple of (matched_rule_name, stats) if a similar rule is found,
            None otherwise.
        """
        if not self.rule_stats:
            return None

        # First, try code hash matching (exact code match = renamed rule)
        hash_matches = self._find_rules_by_code_hash(rule)
        if hash_matches:
            # Use the first match (could merge stats in future)
            matched_rule = hash_matches[0]
            return matched_rule, self.rule_stats[matched_rule]

        # Fall back to Levenshtein distance
        best_match: str | None = None
        best_distance = max_distance + 1

        for known_rule in self.rule_stats:
            distance = _levenshtein_distance(rule, known_rule)
            if distance < best_distance:
                best_distance = distance
                best_match = known_rule

        if best_match is not None and best_distance <= max_distance:
            return best_match, self.rule_stats[best_match]

        return None

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
        if threads is not None and rule in self.thread_stats:
            thread_stats, matched_threads = self.thread_stats[rule].get_best_match(threads)
            if thread_stats is not None:
                thread_mean = self._get_weighted_mean(thread_stats)
                # Tighter variance for exact thread match, wider for aggregate fallback
                if matched_threads is not None:
                    thread_var = (
                        thread_stats.std_dev**2
                        if thread_stats.count > 1
                        else (thread_mean * 0.2) ** 2
                    )
                else:
                    # Aggregate fallback - use standard variance
                    thread_var = (
                        thread_stats.std_dev**2
                        if thread_stats.count > 1
                        else (thread_mean * 0.3) ** 2
                    )
                return thread_mean, thread_var

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
                    rule_mean = self._get_weighted_mean(value_stats)
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
                scaled_est, confidence = self._get_size_scaled_estimate(stats, input_size)
                if confidence > 0.3:  # Only use if we have reasonable confidence
                    # Reduce variance for size-scaled estimates
                    rule_var = (
                        stats.std_dev**2 * (1 - confidence * 0.5)
                        if stats.count > 1
                        else (scaled_est * 0.25) ** 2
                    )
                    return scaled_est, rule_var

            # Standard rule-level estimate
            rule_mean = self._get_weighted_mean(stats)
            rule_var = stats.std_dev**2 if stats.count > 1 else (rule_mean * 0.3) ** 2
            return rule_mean, rule_var

        # Try fuzzy matching for renamed/similar rules before falling back to global mean
        similar = self._find_similar_rule(rule)
        if similar is not None:
            matched_rule, stats = similar
            rule_mean = self._get_weighted_mean(stats)
            # Wider variance for fuzzy matches due to uncertainty
            rule_var = stats.std_dev**2 if stats.count > 1 else (rule_mean * 0.4) ** 2
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

        # Count running jobs by rule
        running_by_rule: dict[str, int] = {}
        for job in progress.running_jobs:
            running_by_rule[job.rule] = running_by_rule.get(job.rule, 0) + 1

        # Only augment with historical counts if we don't have expected_job_counts
        # (to avoid skewing proportional inference with historical execution counts)
        if not self.expected_job_counts:
            for rule, stats in self.rule_stats.items():
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
            if rule in self.rule_stats:
                stats = self.rule_stats[rule]
                rule_mean = self._get_weighted_mean(stats)
                rule_var = stats.std_dev**2 if stats.count > 1 else (rule_mean * 0.3) ** 2
            else:
                # Unknown rule: use global mean with higher uncertainty
                rule_mean = global_mean
                rule_var = (global_mean * 0.5) ** 2

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
                recency_scores.append(self._get_recency_factor(stats))
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
        if pending_count <= 0:
            return {}

        running_by_rule = running_by_rule or {}

        # If we have expected job counts from Job stats table, use exact calculation
        if self.expected_job_counts:
            pending_rules: dict[str, int] = {}
            for rule, expected in self.expected_job_counts.items():
                completed = completed_by_rule.get(rule, 0)
                running = running_by_rule.get(rule, 0)
                remaining = expected - completed - running
                if remaining > 0:
                    pending_rules[rule] = remaining
            return pending_rules

        # Fall back to proportional inference
        if not completed_by_rule:
            return {}

        # Filter out deleted rules if current_rules is provided
        if current_rules is not None:
            completed_by_rule = {r: c for r, c in completed_by_rule.items() if r in current_rules}

        total_completed = sum(completed_by_rule.values())
        if total_completed == 0:
            return {}

        pending_rules = {}
        for rule, count in completed_by_rule.items():
            proportion = count / total_completed
            estimated = round(pending_count * proportion)
            if estimated > 0:
                pending_rules[rule] = estimated

        return pending_rules
