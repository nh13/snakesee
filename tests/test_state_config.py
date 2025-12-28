"""Tests for EstimationConfig and related configuration classes."""

import pytest

from snakesee.state.config import DEFAULT_CONFIG
from snakesee.state.config import ConfidenceThresholds
from snakesee.state.config import ConfidenceWeights
from snakesee.state.config import EstimationConfig
from snakesee.state.config import TimeConstants
from snakesee.state.config import VarianceMultipliers


class TestVarianceMultipliers:
    """Tests for VarianceMultipliers."""

    def test_default_values(self) -> None:
        """Test default multiplier values."""
        mult = VarianceMultipliers()
        assert mult.exact_thread_match == 0.2
        assert mult.exact_wildcard_match == 0.2
        assert mult.size_scaled == 0.25
        assert mult.rule_fallback == 0.3
        assert mult.aggregate_fallback == 0.3
        assert mult.fuzzy_match == 0.4
        assert mult.global_fallback == 0.5
        assert mult.bootstrap == 0.5

    def test_custom_values(self) -> None:
        """Test custom multiplier values."""
        mult = VarianceMultipliers(exact_thread_match=0.1, global_fallback=0.6)
        assert mult.exact_thread_match == 0.1
        assert mult.global_fallback == 0.6

    def test_frozen(self) -> None:
        """Test that VarianceMultipliers is frozen."""
        mult = VarianceMultipliers()
        with pytest.raises(AttributeError):
            mult.exact_thread_match = 0.5  # type: ignore[misc]


class TestConfidenceWeights:
    """Tests for ConfidenceWeights."""

    def test_default_values(self) -> None:
        """Test default weight values."""
        weights = ConfidenceWeights()
        assert weights.sample_size == 0.4
        assert weights.recency == 0.3
        assert weights.consistency == 0.2
        assert weights.data_coverage == 0.1

    def test_weights_sum_to_one(self) -> None:
        """Test that default weights sum to 1.0."""
        weights = ConfidenceWeights()
        total = weights.sample_size + weights.recency + weights.consistency + weights.data_coverage
        assert total == pytest.approx(1.0)

    def test_invalid_weights_raises_error(self) -> None:
        """Test that weights not summing to 1.0 raises ValueError."""
        with pytest.raises(ValueError, match="must sum to 1.0"):
            ConfidenceWeights(sample_size=0.5, recency=0.5, consistency=0.5, data_coverage=0.5)

    def test_custom_valid_weights(self) -> None:
        """Test custom weights that sum to 1.0."""
        weights = ConfidenceWeights(
            sample_size=0.25, recency=0.25, consistency=0.25, data_coverage=0.25
        )
        total = weights.sample_size + weights.recency + weights.consistency + weights.data_coverage
        assert total == pytest.approx(1.0)


class TestConfidenceThresholds:
    """Tests for ConfidenceThresholds."""

    def test_default_values(self) -> None:
        """Test default threshold values."""
        thresholds = ConfidenceThresholds()
        assert thresholds.size_scaling_min == 0.3
        assert thresholds.high_confidence == 0.7
        assert thresholds.medium_confidence == 0.4
        assert thresholds.low_confidence == 0.1
        assert thresholds.max_confidence == 0.9
        assert thresholds.bootstrap_confidence == 0.05


class TestTimeConstants:
    """Tests for TimeConstants."""

    def test_default_values(self) -> None:
        """Test default time constants."""
        constants = TimeConstants()
        assert constants.seconds_per_day == 86400
        assert constants.seconds_per_hour == 3600
        assert constants.seconds_per_minute == 60
        assert constants.stale_workflow_threshold == 1800.0
        assert constants.timing_mismatch_tolerance == 5.0

    def test_seconds_per_day_is_correct(self) -> None:
        """Test that seconds_per_day is 24 * 60 * 60."""
        constants = TimeConstants()
        assert constants.seconds_per_day == 24 * 60 * 60

    def test_seconds_per_hour_is_correct(self) -> None:
        """Test that seconds_per_hour is 60 * 60."""
        constants = TimeConstants()
        assert constants.seconds_per_hour == 60 * 60


class TestEstimationConfig:
    """Tests for EstimationConfig."""

    def test_default_values(self) -> None:
        """Test default config values."""
        config = EstimationConfig()
        assert config.weighting_strategy == "index"
        assert config.half_life_days == 7.0
        assert config.half_life_logs == 10
        assert config.min_samples_for_conditioning == 3
        assert config.high_variance_cv == 0.5
        assert config.default_global_mean == 60.0

    def test_nested_defaults(self) -> None:
        """Test nested dataclass defaults."""
        config = EstimationConfig()
        assert isinstance(config.variance, VarianceMultipliers)
        assert isinstance(config.confidence_weights, ConfidenceWeights)
        assert isinstance(config.confidence_thresholds, ConfidenceThresholds)
        assert isinstance(config.time, TimeConstants)

    def test_custom_values(self) -> None:
        """Test custom config values."""
        config = EstimationConfig(
            weighting_strategy="time",
            half_life_days=14.0,
            half_life_logs=20,
        )
        assert config.weighting_strategy == "time"
        assert config.half_life_days == 14.0
        assert config.half_life_logs == 20

    def test_custom_nested_values(self) -> None:
        """Test custom nested dataclass values."""
        custom_variance = VarianceMultipliers(exact_thread_match=0.1)
        config = EstimationConfig(variance=custom_variance)
        assert config.variance.exact_thread_match == 0.1

    def test_frozen(self) -> None:
        """Test that EstimationConfig is frozen."""
        config = EstimationConfig()
        with pytest.raises(AttributeError):
            config.half_life_days = 14.0  # type: ignore[misc]

    def test_half_life_logs_must_be_positive(self) -> None:
        """Test that half_life_logs must be > 0."""
        with pytest.raises(ValueError, match="half_life_logs must be > 0"):
            EstimationConfig(half_life_logs=0)
        with pytest.raises(ValueError, match="half_life_logs must be > 0"):
            EstimationConfig(half_life_logs=-1)

    def test_half_life_days_must_be_positive(self) -> None:
        """Test that half_life_days must be > 0."""
        with pytest.raises(ValueError, match="half_life_days must be > 0"):
            EstimationConfig(half_life_days=0.0)
        with pytest.raises(ValueError, match="half_life_days must be > 0"):
            EstimationConfig(half_life_days=-1.0)

    def test_parallelism_max_must_be_gte_min(self) -> None:
        """Test that parallelism_max >= parallelism_min."""
        with pytest.raises(ValueError, match="parallelism_max.*must be >= parallelism_min"):
            EstimationConfig(parallelism_min=10.0, parallelism_max=5.0)

    def test_parallelism_min_must_be_positive(self) -> None:
        """Test that parallelism_min must be > 0."""
        with pytest.raises(ValueError, match="parallelism_min must be > 0"):
            EstimationConfig(parallelism_min=0.0)
        with pytest.raises(ValueError, match="parallelism_min must be > 0"):
            EstimationConfig(parallelism_min=-1.0)

    def test_default_global_mean_must_be_positive(self) -> None:
        """Test that default_global_mean must be > 0."""
        with pytest.raises(ValueError, match="default_global_mean must be > 0"):
            EstimationConfig(default_global_mean=0.0)
        with pytest.raises(ValueError, match="default_global_mean must be > 0"):
            EstimationConfig(default_global_mean=-10.0)


class TestDefaultConfig:
    """Tests for DEFAULT_CONFIG global instance."""

    def test_default_config_exists(self) -> None:
        """Test that DEFAULT_CONFIG is an EstimationConfig."""
        assert isinstance(DEFAULT_CONFIG, EstimationConfig)

    def test_default_config_has_expected_values(self) -> None:
        """Test that DEFAULT_CONFIG has default values."""
        assert DEFAULT_CONFIG.weighting_strategy == "index"
        assert DEFAULT_CONFIG.half_life_days == 7.0
