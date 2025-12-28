"""Inference of pending rule distribution for time estimation."""


class PendingRuleInferrer:
    """Infers the distribution of pending jobs by rule.

    When we know the total pending count but not the breakdown by rule,
    this class infers the distribution based on:
    1. Expected job counts (from Snakemake's Job stats table) if available
    2. Proportional inference from completed job distribution otherwise
    """

    def infer(
        self,
        completed_by_rule: dict[str, int],
        pending_count: int,
        expected_job_counts: dict[str, int] | None = None,
        current_rules: set[str] | None = None,
        running_by_rule: dict[str, int] | None = None,
    ) -> dict[str, int]:
        """Infer the distribution of pending jobs by rule.

        Args:
            completed_by_rule: Count of completed jobs per rule.
            pending_count: Total number of pending jobs.
            expected_job_counts: Expected counts from Job stats table (most accurate).
            current_rules: Set of rules in current workflow (filters deleted rules).
            running_by_rule: Count of running jobs per rule.

        Returns:
            Estimated count of pending jobs per rule.
        """
        if pending_count <= 0:
            return {}

        running_by_rule = running_by_rule or {}

        # Use exact calculation if we have expected job counts
        if expected_job_counts:
            return self._exact_calculation(
                expected_job_counts,
                completed_by_rule,
                running_by_rule,
            )

        # Fall back to proportional inference
        return self._proportional_inference(
            completed_by_rule,
            pending_count,
            current_rules,
        )

    def _exact_calculation(
        self,
        expected_job_counts: dict[str, int],
        completed_by_rule: dict[str, int],
        running_by_rule: dict[str, int],
    ) -> dict[str, int]:
        """Calculate pending using expected - completed - running."""
        pending_rules: dict[str, int] = {}

        for rule, expected in expected_job_counts.items():
            completed = completed_by_rule.get(rule, 0)
            running = running_by_rule.get(rule, 0)
            remaining = expected - completed - running
            if remaining > 0:
                pending_rules[rule] = remaining

        return pending_rules

    def _proportional_inference(
        self,
        completed_by_rule: dict[str, int],
        pending_count: int,
        current_rules: set[str] | None,
    ) -> dict[str, int]:
        """Infer pending distribution proportionally to completed jobs.

        Note: Due to rounding, the sum of returned values may not exactly
        equal pending_count. This is expected and the estimation handles
        this gracefully.
        """
        if not completed_by_rule:
            return {}

        # Filter out deleted rules if current_rules is provided
        if current_rules is not None:
            completed_by_rule = {r: c for r, c in completed_by_rule.items() if r in current_rules}

        total_completed = sum(completed_by_rule.values())
        if total_completed == 0:
            return {}

        pending_rules: dict[str, int] = {}
        for rule, count in completed_by_rule.items():
            proportion = count / total_completed
            estimated = round(pending_count * proportion)
            if estimated > 0:
                pending_rules[rule] = estimated

        return pending_rules
