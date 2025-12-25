"""Rule registry for centralized rule timing statistics.

This module provides a single source of truth for all rule timing data,
consolidating aggregate, thread-level, and wildcard-level statistics.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from snakesee.models import RuleTimingStats
    from snakesee.models import ThreadTimingStats
    from snakesee.models import WildcardTimingStats
    from snakesee.state.config import EstimationConfig
    from snakesee.state.job_registry import Job


def _make_empty_rule_stats(rule: str = "") -> RuleTimingStats:
    """Factory function to create empty RuleTimingStats."""
    from snakesee.models import RuleTimingStats

    return RuleTimingStats(rule=rule)


@dataclass
class RuleStatistics:
    """Complete timing statistics for a single rule.

    Combines aggregate, thread-level, and wildcard-level statistics
    in a single container.

    Attributes:
        rule: The rule name.
        aggregate: Overall timing statistics for the rule.
        by_threads: Thread-specific timing statistics.
        by_wildcard: Wildcard-specific timing statistics.
        expected_count: Expected number of jobs for this rule (if known).
    """

    rule: str
    aggregate: RuleTimingStats = field(default_factory=_make_empty_rule_stats)
    by_threads: ThreadTimingStats | None = None
    by_wildcard: dict[str, WildcardTimingStats] = field(default_factory=dict)
    expected_count: int | None = None

    def __post_init__(self) -> None:
        """Ensure aggregate has correct rule name."""
        from snakesee.models import RuleTimingStats

        if self.aggregate.rule != self.rule:
            # Create new RuleTimingStats with correct rule name
            self.aggregate = RuleTimingStats(
                rule=self.rule,
                durations=list(self.aggregate.durations),
                timestamps=list(self.aggregate.timestamps),
                input_sizes=list(self.aggregate.input_sizes),
            )

    def record_completion(
        self,
        duration: float,
        timestamp: float,
        threads: int | None = None,
        wildcards: dict[str, str] | None = None,
        input_size: int | None = None,
    ) -> None:
        """Record a job completion.

        Args:
            duration: Job duration in seconds.
            timestamp: Completion timestamp.
            threads: Number of threads used.
            wildcards: Wildcard values for the job.
            input_size: Input file size in bytes.
        """
        from snakesee.models import RuleTimingStats
        from snakesee.models import ThreadTimingStats
        from snakesee.models import WildcardTimingStats

        # Update aggregate stats
        self.aggregate.durations.append(duration)
        self.aggregate.timestamps.append(timestamp)
        self.aggregate.input_sizes.append(input_size)

        # Update thread-specific stats
        if threads is not None:
            if self.by_threads is None:
                self.by_threads = ThreadTimingStats(rule=self.rule, stats_by_threads={})
            # Get or create stats for this thread count
            if threads not in self.by_threads.stats_by_threads:
                self.by_threads.stats_by_threads[threads] = RuleTimingStats(
                    rule=f"{self.rule}@{threads}t"
                )
            thread_stats = self.by_threads.stats_by_threads[threads]
            thread_stats.durations.append(duration)
            thread_stats.timestamps.append(timestamp)

        # Update wildcard-specific stats (for each wildcard key)
        if wildcards:
            for wc_key, wc_value in wildcards.items():
                if wc_key not in self.by_wildcard:
                    self.by_wildcard[wc_key] = WildcardTimingStats(
                        rule=self.rule,
                        wildcard_key=wc_key,
                        stats_by_value={},
                    )
                wts = self.by_wildcard[wc_key]
                # Access stats_by_value directly (get_stats_for_value has min sample check)
                if wc_value not in wts.stats_by_value:
                    wts.stats_by_value[wc_value] = RuleTimingStats(
                        rule=f"{self.rule}:{wc_key}={wc_value}"
                    )
                value_stats = wts.stats_by_value[wc_value]
                value_stats.durations.append(duration)
                value_stats.timestamps.append(timestamp)


class RuleRegistry:
    """Central registry for all rule timing statistics.

    Consolidates what was previously scattered across TimeEstimator's
    rule_stats, thread_stats, and wildcard_stats dictionaries.

    Example:
        >>> registry = RuleRegistry()
        >>> stats = registry.get_or_create("align")
        >>> stats.record_completion(duration=100.0, timestamp=time.time(), threads=4)
        >>> mean, var = registry.get_estimate("align", threads=4)
    """

    def __init__(self, config: EstimationConfig | None = None) -> None:
        """Initialize empty registry.

        Args:
            config: Optional estimation configuration.
        """
        from snakesee.state.config import DEFAULT_CONFIG

        self.config = config or DEFAULT_CONFIG
        self._rules: dict[str, RuleStatistics] = {}
        self._global_mean_cache: float | None = None
        self._cache_valid: bool = False

    def __len__(self) -> int:
        """Return number of rules in registry."""
        return len(self._rules)

    def __contains__(self, rule: str) -> bool:
        """Check if rule exists in registry."""
        return rule in self._rules

    def get(self, rule: str) -> RuleStatistics | None:
        """Get statistics for a rule."""
        return self._rules.get(rule)

    def get_or_create(self, rule: str) -> RuleStatistics:
        """Get existing statistics or create new ones.

        Args:
            rule: Rule name.

        Returns:
            RuleStatistics for the rule.
        """
        if rule not in self._rules:
            self._rules[rule] = RuleStatistics(rule=rule)
            self._cache_valid = False
        return self._rules[rule]

    def record_completion(
        self,
        rule: str,
        duration: float,
        timestamp: float,
        threads: int | None = None,
        wildcards: dict[str, str] | None = None,
        input_size: int | None = None,
    ) -> None:
        """Record a job completion.

        Args:
            rule: Rule name.
            duration: Job duration in seconds.
            timestamp: Completion timestamp.
            threads: Number of threads used.
            wildcards: Wildcard values.
            input_size: Input file size.
        """
        stats = self.get_or_create(rule)
        stats.record_completion(duration, timestamp, threads, wildcards, input_size)
        self._cache_valid = False

    def record_job_completion(self, job: Job) -> None:
        """Record completion from a Job object.

        Args:
            job: Completed job with timing data.
        """
        if job.duration is None:
            return

        self.record_completion(
            rule=job.rule,
            duration=job.duration,
            timestamp=job.end_time or 0.0,
            threads=job.threads,
            wildcards=job.wildcards if job.wildcards else None,
            input_size=job.input_size,
        )

    def global_mean_duration(self) -> float:
        """Get global mean duration across all rules.

        Returns:
            Mean duration in seconds, or config default if no data.
        """
        if self._cache_valid and self._global_mean_cache is not None:
            return self._global_mean_cache

        all_durations: list[float] = []
        for stats in self._rules.values():
            all_durations.extend(stats.aggregate.durations)

        if all_durations:
            self._global_mean_cache = sum(all_durations) / len(all_durations)
        else:
            self._global_mean_cache = self.config.default_global_mean

        self._cache_valid = True
        return self._global_mean_cache

    def set_expected_counts(self, counts: dict[str, int]) -> None:
        """Set expected job counts for rules.

        Args:
            counts: Dictionary of rule name to expected count.
        """
        for rule, count in counts.items():
            stats = self.get_or_create(rule)
            stats.expected_count = count

    def clear(self) -> None:
        """Clear all statistics."""
        self._rules.clear()
        self._cache_valid = False
        self._global_mean_cache = None

    # Backward compatibility methods

    def to_rule_stats_dict(self) -> dict[str, RuleTimingStats]:
        """Convert to dict for backward compatibility with TimeEstimator."""
        return {name: stats.aggregate for name, stats in self._rules.items()}

    def to_thread_stats_dict(self) -> dict[str, ThreadTimingStats]:
        """Convert to thread stats dict for backward compatibility."""
        result: dict[str, ThreadTimingStats] = {}
        for name, stats in self._rules.items():
            if stats.by_threads is not None:
                result[name] = stats.by_threads
        return result

    def to_wildcard_stats_dict(self) -> dict[str, dict[str, WildcardTimingStats]]:
        """Convert to wildcard stats dict for backward compatibility."""
        result: dict[str, dict[str, WildcardTimingStats]] = {}
        for name, stats in self._rules.items():
            if stats.by_wildcard:
                result[name] = stats.by_wildcard
        return result

    def load_from_rule_stats(
        self,
        rule_stats: dict[str, RuleTimingStats],
        thread_stats: dict[str, ThreadTimingStats] | None = None,
        wildcard_stats: dict[str, dict[str, WildcardTimingStats]] | None = None,
    ) -> None:
        """Load from existing stats dictionaries.

        Args:
            rule_stats: Aggregate rule timing stats.
            thread_stats: Thread-specific stats.
            wildcard_stats: Wildcard-specific stats.
        """
        for rule, stats in rule_stats.items():
            rule_stat = self.get_or_create(rule)
            rule_stat.aggregate = stats

        if thread_stats:
            for rule, tstats in thread_stats.items():
                rule_stat = self.get_or_create(rule)
                rule_stat.by_threads = tstats

        if wildcard_stats:
            for rule, wstats in wildcard_stats.items():
                rule_stat = self.get_or_create(rule)
                rule_stat.by_wildcard = wstats

        self._cache_valid = False

    def all_rules(self) -> list[str]:
        """Get all rule names."""
        return list(self._rules.keys())

    def total_sample_count(self) -> int:
        """Get total number of samples across all rules."""
        return sum(stats.aggregate.count for stats in self._rules.values())
