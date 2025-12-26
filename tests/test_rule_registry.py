"""Tests for the RuleRegistry module."""

import pytest

from snakesee.state.config import EstimationConfig
from snakesee.state.job_registry import Job
from snakesee.state.job_registry import JobStatus
from snakesee.state.rule_registry import RuleRegistry
from snakesee.state.rule_registry import RuleStatistics


class TestRuleStatistics:
    """Tests for RuleStatistics dataclass."""

    def test_creation(self) -> None:
        """Test basic creation."""
        stats = RuleStatistics(rule="align")
        assert stats.rule == "align"
        assert stats.aggregate.rule == "align"
        assert stats.by_threads is None
        assert stats.by_wildcard == {}

    def test_record_completion_basic(self) -> None:
        """Test recording a basic completion."""
        stats = RuleStatistics(rule="align")
        stats.record_completion(duration=100.0, timestamp=1000.0)

        assert stats.aggregate.count == 1
        assert stats.aggregate.durations == [100.0]
        assert stats.aggregate.timestamps == [1000.0]

    def test_record_completion_with_threads(self) -> None:
        """Test recording completion with threads."""
        stats = RuleStatistics(rule="align")
        stats.record_completion(duration=100.0, timestamp=1000.0, threads=4)

        assert stats.by_threads is not None
        thread_stats, _matched = stats.by_threads.get_best_match(4)
        assert thread_stats is not None
        assert thread_stats.count == 1

    def test_record_completion_with_wildcards(self) -> None:
        """Test recording completion with wildcards."""
        stats = RuleStatistics(rule="align")
        stats.record_completion(duration=100.0, timestamp=1000.0, wildcards={"sample": "A"})

        assert "sample" in stats.by_wildcard
        wts = stats.by_wildcard["sample"]
        # Access stats_by_value directly since get_stats_for_value requires 3+ samples
        assert "A" in wts.stats_by_value
        value_stats = wts.stats_by_value["A"]
        assert value_stats.count == 1

    def test_record_multiple_completions(self) -> None:
        """Test recording multiple completions."""
        stats = RuleStatistics(rule="align")
        stats.record_completion(duration=100.0, timestamp=1000.0)
        stats.record_completion(duration=120.0, timestamp=2000.0)
        stats.record_completion(duration=110.0, timestamp=3000.0)

        assert stats.aggregate.count == 3
        assert stats.aggregate.mean_duration == pytest.approx(110.0)


class TestRuleRegistry:
    """Tests for RuleRegistry."""

    def test_empty_registry(self) -> None:
        """Test empty registry."""
        registry = RuleRegistry()
        assert len(registry) == 0
        assert "align" not in registry

    def test_get_nonexistent(self) -> None:
        """Test getting nonexistent rule."""
        registry = RuleRegistry()
        assert registry.get("align") is None

    def test_get_or_create_new(self) -> None:
        """Test creating new rule."""
        registry = RuleRegistry()
        stats = registry.get_or_create("align")

        assert stats.rule == "align"
        assert "align" in registry
        assert len(registry) == 1

    def test_get_or_create_existing(self) -> None:
        """Test getting existing rule."""
        registry = RuleRegistry()
        stats1 = registry.get_or_create("align")
        stats2 = registry.get_or_create("align")

        assert stats1 is stats2

    def test_record_completion(self) -> None:
        """Test recording completion through registry."""
        registry = RuleRegistry()
        registry.record_completion(
            rule="align",
            duration=100.0,
            timestamp=1000.0,
            threads=4,
        )

        stats = registry.get("align")
        assert stats is not None
        assert stats.aggregate.count == 1

    def test_record_job_completion(self) -> None:
        """Test recording from Job object."""
        registry = RuleRegistry()
        job = Job(
            key="1",
            rule="align",
            status=JobStatus.COMPLETED,
            start_time=100.0,
            end_time=200.0,
            threads=4,
            wildcards={"sample": "A"},
        )

        registry.record_job_completion(job)

        stats = registry.get("align")
        assert stats is not None
        assert stats.aggregate.count == 1
        assert stats.aggregate.durations[0] == 100.0
        assert stats.by_threads is not None

    def test_global_mean_empty(self) -> None:
        """Test global mean with no data."""
        registry = RuleRegistry()
        mean = registry.global_mean_duration()
        assert mean == 60.0  # Default from config

    def test_global_mean_with_data(self) -> None:
        """Test global mean with data."""
        registry = RuleRegistry()
        registry.record_completion("align", 100.0, 1000.0)
        registry.record_completion("sort", 200.0, 2000.0)

        mean = registry.global_mean_duration()
        assert mean == pytest.approx(150.0)

    def test_global_mean_caching(self) -> None:
        """Test global mean caching."""
        registry = RuleRegistry()
        registry.record_completion("align", 100.0, 1000.0)

        mean1 = registry.global_mean_duration()
        mean2 = registry.global_mean_duration()
        assert mean1 == mean2

        # Cache should invalidate on new data
        registry.record_completion("sort", 200.0, 2000.0)
        mean3 = registry.global_mean_duration()
        assert mean3 == pytest.approx(150.0)

    def test_set_expected_counts(self) -> None:
        """Test setting expected job counts."""
        registry = RuleRegistry()
        registry.set_expected_counts({"align": 10, "sort": 5})

        align = registry.get("align")
        sort = registry.get("sort")
        assert align is not None
        assert align.expected_count == 10
        assert sort is not None
        assert sort.expected_count == 5

    def test_clear(self) -> None:
        """Test clearing registry."""
        registry = RuleRegistry()
        registry.record_completion("align", 100.0, 1000.0)

        registry.clear()

        assert len(registry) == 0
        assert registry.get("align") is None

    def test_all_rules(self) -> None:
        """Test getting all rule names."""
        registry = RuleRegistry()
        registry.record_completion("align", 100.0, 1000.0)
        registry.record_completion("sort", 200.0, 2000.0)

        rules = registry.all_rules()
        assert set(rules) == {"align", "sort"}

    def test_total_sample_count(self) -> None:
        """Test total sample count."""
        registry = RuleRegistry()
        registry.record_completion("align", 100.0, 1000.0)
        registry.record_completion("align", 110.0, 2000.0)
        registry.record_completion("sort", 200.0, 3000.0)

        assert registry.total_sample_count() == 3


class TestRuleRegistryBackwardCompat:
    """Tests for backward compatibility methods."""

    def test_to_rule_stats_dict(self) -> None:
        """Test converting to rule stats dict."""
        registry = RuleRegistry()
        registry.record_completion("align", 100.0, 1000.0)
        registry.record_completion("sort", 200.0, 2000.0)

        stats_dict = registry.to_rule_stats_dict()

        assert "align" in stats_dict
        assert "sort" in stats_dict
        assert stats_dict["align"].count == 1
        assert stats_dict["sort"].count == 1

    def test_to_thread_stats_dict(self) -> None:
        """Test converting to thread stats dict."""
        registry = RuleRegistry()
        registry.record_completion("align", 100.0, 1000.0, threads=4)
        registry.record_completion("sort", 200.0, 2000.0)  # No threads

        thread_dict = registry.to_thread_stats_dict()

        assert "align" in thread_dict
        assert "sort" not in thread_dict

    def test_to_wildcard_stats_dict(self) -> None:
        """Test converting to wildcard stats dict."""
        registry = RuleRegistry()
        registry.record_completion("align", 100.0, 1000.0, wildcards={"sample": "A"})
        registry.record_completion("sort", 200.0, 2000.0)  # No wildcards

        wc_dict = registry.to_wildcard_stats_dict()

        assert "align" in wc_dict
        assert "sort" not in wc_dict
        assert "sample" in wc_dict["align"]

    def test_load_from_rule_stats(self) -> None:
        """Test loading from existing stats dicts."""
        from snakesee.models import RuleTimingStats

        registry = RuleRegistry()
        rule_stats = {
            "align": RuleTimingStats(
                rule="align", durations=[100.0, 110.0], timestamps=[1000.0, 2000.0]
            )
        }

        registry.load_from_rule_stats(rule_stats)

        stats = registry.get("align")
        assert stats is not None
        assert stats.aggregate.count == 2


class TestRuleRegistryConfig:
    """Tests for configuration integration."""

    def test_custom_config(self) -> None:
        """Test with custom config."""
        config = EstimationConfig(default_global_mean=120.0)
        registry = RuleRegistry(config=config)

        mean = registry.global_mean_duration()
        assert mean == 120.0

    def test_default_config(self) -> None:
        """Test with default config."""
        registry = RuleRegistry()
        assert registry.config is not None
        assert registry.config.default_global_mean == 60.0
