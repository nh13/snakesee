"""Tests for estimation package components."""

import json
from pathlib import Path

import pytest

from snakesee.estimation.data_loader import HistoricalDataLoader
from snakesee.estimation.pending_inferrer import PendingRuleInferrer
from snakesee.estimation.rule_matcher import RuleMatchingEngine
from snakesee.estimation.rule_matcher import levenshtein_distance
from snakesee.models import RuleTimingStats
from snakesee.state.rule_registry import RuleRegistry


class TestLevenshteinDistance:
    """Tests for levenshtein_distance function."""

    def test_identical_strings(self) -> None:
        assert levenshtein_distance("hello", "hello") == 0

    def test_empty_strings(self) -> None:
        assert levenshtein_distance("", "") == 0
        assert levenshtein_distance("hello", "") == 5
        assert levenshtein_distance("", "hello") == 5

    def test_single_insertion(self) -> None:
        assert levenshtein_distance("hello", "hellos") == 1

    def test_single_deletion(self) -> None:
        assert levenshtein_distance("hello", "hell") == 1

    def test_single_substitution(self) -> None:
        assert levenshtein_distance("hello", "hallo") == 1

    def test_multiple_edits(self) -> None:
        assert levenshtein_distance("kitten", "sitting") == 3

    def test_cached_results(self) -> None:
        # Clear cache first
        levenshtein_distance.cache_clear()
        # First call
        levenshtein_distance("rule_a", "rule_b")
        # Second call should hit cache
        levenshtein_distance("rule_a", "rule_b")
        assert levenshtein_distance.cache_info().hits >= 1


class TestRuleMatchingEngine:
    """Tests for RuleMatchingEngine class."""

    def test_init_default_max_distance(self) -> None:
        engine = RuleMatchingEngine()
        assert engine.max_distance == 3

    def test_init_custom_max_distance(self) -> None:
        engine = RuleMatchingEngine(max_distance=5)
        assert engine.max_distance == 5

    def test_find_by_code_hash_match(self) -> None:
        engine = RuleMatchingEngine()
        code_hash_to_rules: dict[str, set[str]] = {
            "abc123": {"rule_a", "rule_b"},
            "def456": {"rule_c"},
        }
        known_rules = {"rule_b", "rule_c"}

        result = engine.find_by_code_hash("rule_a", code_hash_to_rules, known_rules)
        assert result == "rule_b"

    def test_find_by_code_hash_no_match_different_hash(self) -> None:
        engine = RuleMatchingEngine()
        code_hash_to_rules: dict[str, set[str]] = {"abc123": {"rule_a"}}
        known_rules = {"rule_b"}

        result = engine.find_by_code_hash("rule_a", code_hash_to_rules, known_rules)
        assert result is None

    def test_find_by_code_hash_no_match_same_rule(self) -> None:
        engine = RuleMatchingEngine()
        code_hash_to_rules: dict[str, set[str]] = {"abc123": {"rule_a"}}
        known_rules = {"rule_a"}

        result = engine.find_by_code_hash("rule_a", code_hash_to_rules, known_rules)
        assert result is None

    def test_find_similar_exact_match(self) -> None:
        engine = RuleMatchingEngine()
        known_rules = {"rule_a", "rule_b"}

        result = engine.find_similar("rule_a", known_rules)
        assert result == ("rule_a", 0)

    def test_find_similar_one_edit(self) -> None:
        engine = RuleMatchingEngine()
        known_rules = {"rule_a", "rule_b"}

        result = engine.find_similar("rule_c", known_rules)
        # Distance of 1 from rule_a or rule_b
        assert result is not None
        assert result[1] == 1

    def test_find_similar_no_match_too_far(self) -> None:
        engine = RuleMatchingEngine(max_distance=1)
        known_rules = {"rule_a", "rule_b"}

        result = engine.find_similar("completely_different", known_rules)
        assert result is None

    def test_find_similar_custom_max_distance(self) -> None:
        engine = RuleMatchingEngine(max_distance=3)
        known_rules = {"rule_a"}

        result = engine.find_similar("rule_a", known_rules, max_distance=0)
        assert result == ("rule_a", 0)

        result = engine.find_similar("rule_b", known_rules, max_distance=0)
        assert result is None

    def test_find_best_match_code_hash_priority(self) -> None:
        engine = RuleMatchingEngine()
        code_hash_to_rules: dict[str, set[str]] = {"abc123": {"rule_a", "rule_a_v2"}}
        rule_stats = {
            "rule_a_v2": RuleTimingStats(rule="rule_a_v2", durations=[10.0], timestamps=[1000.0]),
            "rule_b": RuleTimingStats(rule="rule_b", durations=[20.0], timestamps=[1000.0]),
        }

        result = engine.find_best_match("rule_a", code_hash_to_rules, rule_stats)
        assert result is not None
        assert result[0] == "rule_a_v2"

    def test_find_best_match_fuzzy_fallback(self) -> None:
        engine = RuleMatchingEngine()
        code_hash_to_rules: dict[str, set[str]] = {}
        rule_stats = {
            "rule_a": RuleTimingStats(rule="rule_a", durations=[10.0], timestamps=[1000.0]),
        }

        result = engine.find_best_match("rule_b", code_hash_to_rules, rule_stats)
        assert result is not None
        assert result[0] == "rule_a"

    def test_find_best_match_no_match(self) -> None:
        engine = RuleMatchingEngine(max_distance=1)
        code_hash_to_rules: dict[str, set[str]] = {}
        rule_stats = {
            "rule_a": RuleTimingStats(rule="rule_a", durations=[10.0], timestamps=[1000.0]),
        }

        result = engine.find_best_match("completely_different", code_hash_to_rules, rule_stats)
        assert result is None


class TestPendingRuleInferrer:
    """Tests for PendingRuleInferrer class."""

    def test_infer_zero_pending(self) -> None:
        inferrer = PendingRuleInferrer()
        result = inferrer.infer(
            completed_by_rule={"rule_a": 10},
            pending_count=0,
        )
        assert result == {}

    def test_infer_negative_pending(self) -> None:
        inferrer = PendingRuleInferrer()
        result = inferrer.infer(
            completed_by_rule={"rule_a": 10},
            pending_count=-5,
        )
        assert result == {}

    def test_infer_with_expected_counts(self) -> None:
        inferrer = PendingRuleInferrer()
        result = inferrer.infer(
            completed_by_rule={"rule_a": 5, "rule_b": 3},
            pending_count=12,
            expected_job_counts={"rule_a": 10, "rule_b": 5, "rule_c": 2},
        )
        # Expected: rule_a: 10-5=5, rule_b: 5-3=2, rule_c: 2-0=2
        assert result == {"rule_a": 5, "rule_b": 2, "rule_c": 2}

    def test_infer_with_expected_counts_and_running(self) -> None:
        inferrer = PendingRuleInferrer()
        result = inferrer.infer(
            completed_by_rule={"rule_a": 5},
            pending_count=3,
            expected_job_counts={"rule_a": 10, "rule_b": 5},
            running_by_rule={"rule_a": 2, "rule_b": 3},
        )
        # Expected: rule_a: 10-5-2=3, rule_b: 5-0-3=2
        assert result == {"rule_a": 3, "rule_b": 2}

    def test_infer_proportional_empty_completed(self) -> None:
        inferrer = PendingRuleInferrer()
        result = inferrer.infer(
            completed_by_rule={},
            pending_count=10,
        )
        assert result == {}

    def test_infer_proportional_basic(self) -> None:
        inferrer = PendingRuleInferrer()
        result = inferrer.infer(
            completed_by_rule={"rule_a": 6, "rule_b": 4},
            pending_count=10,
        )
        # rule_a: 60% of 10 = 6, rule_b: 40% of 10 = 4
        assert result == {"rule_a": 6, "rule_b": 4}

    def test_infer_proportional_filters_current_rules(self) -> None:
        inferrer = PendingRuleInferrer()
        result = inferrer.infer(
            completed_by_rule={"rule_a": 6, "rule_b": 4, "deleted_rule": 5},
            pending_count=10,
            current_rules={"rule_a", "rule_b"},
        )
        # Only rule_a and rule_b, total completed is 10
        # rule_a: 60% of 10 = 6, rule_b: 40% of 10 = 4
        assert result == {"rule_a": 6, "rule_b": 4}

    def test_infer_proportional_rounds_to_zero(self) -> None:
        inferrer = PendingRuleInferrer()
        result = inferrer.infer(
            completed_by_rule={"rule_a": 100, "rule_b": 1},
            pending_count=1,
        )
        # rule_a: 99% of 1 = 1 (rounded), rule_b: 1% of 1 = 0 (rounded)
        assert result == {"rule_a": 1}


class TestHistoricalDataLoader:
    """Tests for HistoricalDataLoader class."""

    def test_init(self) -> None:
        registry = RuleRegistry()
        loader = HistoricalDataLoader(registry)
        assert loader._registry is registry
        assert loader.use_wildcard_conditioning is False

    def test_init_with_wildcard_conditioning(self) -> None:
        registry = RuleRegistry()
        loader = HistoricalDataLoader(registry, use_wildcard_conditioning=True)
        assert loader.use_wildcard_conditioning is True

    def test_load_from_metadata(self, tmp_path: Path) -> None:
        metadata_dir = tmp_path / "metadata"
        metadata_dir.mkdir()

        # Create a metadata file
        meta_file = metadata_dir / "abc123"
        meta_file.write_text(
            json.dumps(
                {
                    "rule": "test_rule",
                    "starttime": 1000.0,
                    "endtime": 1010.0,
                    "code": "echo hello",  # Code is hashed to generate code_hash
                }
            )
        )

        registry = RuleRegistry()
        loader = HistoricalDataLoader(registry)
        loader.load_from_metadata(metadata_dir)

        rule_stats = registry.to_rule_stats_dict()
        assert "test_rule" in rule_stats
        stats = rule_stats["test_rule"]
        assert len(stats.durations) == 1
        assert stats.durations[0] == pytest.approx(10.0, rel=0.01)
        # Code hash is computed from the "code" field
        assert len(loader.code_hash_to_rules) == 1
        assert "test_rule" in next(iter(loader.code_hash_to_rules.values()))

    def test_load_from_metadata_with_wildcards(self, tmp_path: Path) -> None:
        metadata_dir = tmp_path / "metadata"
        metadata_dir.mkdir()

        meta_file = metadata_dir / "abc123"
        meta_file.write_text(
            json.dumps(
                {
                    "rule": "test_rule",
                    "starttime": 1000.0,
                    "endtime": 1010.0,
                    "wildcards": {"sample": "A"},
                }
            )
        )

        registry = RuleRegistry()
        loader = HistoricalDataLoader(registry, use_wildcard_conditioning=True)
        loader.load_from_metadata(metadata_dir)

        wc_stats = registry.to_wildcard_stats_dict()
        assert "test_rule" in wc_stats
        assert "sample" in wc_stats["test_rule"]

    def test_load_from_events(self, tmp_path: Path) -> None:
        events_file = tmp_path / "events.jsonl"
        events = [
            {"event_type": "job_started", "rule_name": "test"},
            {
                "event_type": "job_finished",
                "rule_name": "test_rule",
                "duration": 10.0,
                "timestamp": 1000.0,
            },
        ]
        events_file.write_text("\n".join(json.dumps(e) for e in events))

        registry = RuleRegistry()
        loader = HistoricalDataLoader(registry)
        has_wildcards = loader.load_from_events(events_file)

        assert has_wildcards is False
        rule_stats = registry.to_rule_stats_dict()
        assert "test_rule" in rule_stats
        stats = rule_stats["test_rule"]
        assert len(stats.durations) == 1
        assert stats.durations[0] == pytest.approx(10.0)

    def test_load_from_events_with_wildcards(self, tmp_path: Path) -> None:
        events_file = tmp_path / "events.jsonl"
        event = {
            "event_type": "job_finished",
            "rule_name": "test_rule",
            "duration": 10.0,
            "timestamp": 1000.0,
            "wildcards": {"sample": "A"},
        }
        events_file.write_text(json.dumps(event))

        registry = RuleRegistry()
        loader = HistoricalDataLoader(registry)
        has_wildcards = loader.load_from_events(events_file)

        assert has_wildcards is True

    def test_load_from_events_nonexistent(self, tmp_path: Path) -> None:
        events_file = tmp_path / "nonexistent.jsonl"
        registry = RuleRegistry()
        loader = HistoricalDataLoader(registry)
        has_wildcards = loader.load_from_events(events_file)

        assert has_wildcards is False

    def test_load_from_events_malformed_json(self, tmp_path: Path) -> None:
        events_file = tmp_path / "events.jsonl"
        events_file.write_text("not json\n{}\n")

        registry = RuleRegistry()
        loader = HistoricalDataLoader(registry)
        has_wildcards = loader.load_from_events(events_file)

        assert has_wildcards is False

    def test_load_from_events_missing_required_fields(self, tmp_path: Path) -> None:
        events_file = tmp_path / "events.jsonl"
        events = [
            {"event_type": "job_finished"},  # Missing duration, timestamp, rule_name
            {"event_type": "job_finished", "rule_name": "test"},  # Missing duration
        ]
        events_file.write_text("\n".join(json.dumps(e) for e in events))

        registry = RuleRegistry()
        loader = HistoricalDataLoader(registry)
        loader.load_from_events(events_file)

        # No valid events should be loaded
        rule_stats = registry.to_rule_stats_dict()
        assert "test" not in rule_stats
