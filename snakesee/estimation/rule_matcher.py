"""Rule matching utilities for fuzzy matching and code hash lookup."""

import functools

from snakesee.models import RuleTimingStats


@functools.lru_cache(maxsize=256)
def levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate the Levenshtein distance between two strings.

    Results are cached for efficiency when comparing the same rule names
    multiple times during estimation.

    Args:
        s1: First string.
        s2: Second string.

    Returns:
        The minimum number of edits (insertions, deletions, substitutions)
        needed to transform s1 into s2.
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row: list[int] = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


class RuleMatchingEngine:
    """Matches rules by name similarity and code hash.

    Used to find timing data for renamed rules or similar rules
    when exact matches aren't available.
    """

    def __init__(self, max_distance: int = 3) -> None:
        """Initialize the matcher.

        Args:
            max_distance: Maximum Levenshtein distance for fuzzy matches.
        """
        self.max_distance = max_distance

    def find_by_code_hash(
        self,
        rule: str,
        code_hash_to_rules: dict[str, set[str]],
        known_rules: set[str],
    ) -> str | None:
        """Find a rule with matching code hash.

        When multiple rules share the same code hash and are in known_rules,
        returns the lexicographically smallest rule name for deterministic behavior.

        Args:
            rule: Rule name to look up.
            code_hash_to_rules: Mapping of code hashes to rule sets.
            known_rules: Set of rules with available stats.

        Returns:
            Name of matching rule (lexicographically smallest if multiple),
            or None if not found.
        """
        for _hash, rules in code_hash_to_rules.items():
            if rule in rules:
                # Find all candidate rules that match criteria
                candidates = {r for r in rules if r != rule and r in known_rules}
                if candidates:
                    # Return lexicographically smallest for deterministic selection
                    return min(candidates)
        return None

    def find_similar(
        self,
        rule: str,
        known_rules: set[str],
        max_distance: int | None = None,
    ) -> tuple[str, int] | None:
        """Find similar rule by Levenshtein distance.

        When multiple rules have the same distance, returns the lexicographically
        smallest one for deterministic behavior.

        Args:
            rule: Rule name to match.
            known_rules: Set of rules to search.
            max_distance: Maximum distance (uses instance default if None).

        Returns:
            Tuple of (matched_rule, distance), or None if no match.
        """
        if max_distance is None:
            max_distance = self.max_distance

        best_match: str | None = None
        best_distance = max_distance + 1

        for known_rule in known_rules:
            distance = levenshtein_distance(rule, known_rule)
            # Prefer lower distance, then lexicographically smaller name for ties
            if distance < best_distance or (
                distance == best_distance and best_match is not None and known_rule < best_match
            ):
                best_distance = distance
                best_match = known_rule

        if best_match is not None and best_distance <= max_distance:
            return best_match, best_distance

        return None

    def find_best_match(
        self,
        rule: str,
        code_hash_to_rules: dict[str, set[str]],
        rule_stats: dict[str, RuleTimingStats],
        max_distance: int | None = None,
    ) -> tuple[str, RuleTimingStats] | None:
        """Find the best matching rule using code hash then fuzzy matching.

        Args:
            rule: Rule name to match.
            code_hash_to_rules: Mapping of code hashes to rule sets.
            rule_stats: Available rule statistics.
            max_distance: Maximum Levenshtein distance.

        Returns:
            Tuple of (matched_rule, stats), or None if no match.
        """
        known_rules = set(rule_stats.keys())

        # Try code hash first (exact code = renamed rule)
        hash_match = self.find_by_code_hash(rule, code_hash_to_rules, known_rules)
        if hash_match is not None:
            return hash_match, rule_stats[hash_match]

        # Fall back to fuzzy name matching
        similar = self.find_similar(rule, known_rules, max_distance)
        if similar is not None:
            matched_rule, _distance = similar
            return matched_rule, rule_stats[matched_rule]

        return None
