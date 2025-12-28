"""Tests for the VarianceCalculator module."""

import pytest

from snakesee.models import RuleTimingStats
from snakesee.state.config import EstimationConfig
from snakesee.variance import ConfidenceResult
from snakesee.variance import VarianceCalculator
from snakesee.variance import VarianceResult
from snakesee.variance import get_calculator


class TestVarianceResult:
    """Tests for VarianceResult dataclass."""

    def test_creation(self) -> None:
        """Test creating a VarianceResult."""
        result = VarianceResult(variance=100.0, source="observed")
        assert result.variance == 100.0
        assert result.source == "observed"
        assert result.multiplier_used is None

    def test_with_multiplier(self) -> None:
        """Test creating a VarianceResult with multiplier."""
        result = VarianceResult(variance=100.0, source="fallback", multiplier_used=0.3)
        assert result.multiplier_used == 0.3


class TestConfidenceResult:
    """Tests for ConfidenceResult dataclass."""

    def test_creation(self) -> None:
        """Test creating a ConfidenceResult."""
        result = ConfidenceResult(
            confidence=0.75,
            sample_factor=0.8,
            recency_factor=0.7,
            consistency_factor=0.9,
            coverage_factor=0.6,
        )
        assert result.confidence == 0.75
        assert result.sample_factor == 0.8


class TestVarianceCalculator:
    """Tests for VarianceCalculator class."""

    def test_default_config(self) -> None:
        """Test calculator uses default config."""
        calc = VarianceCalculator()
        assert calc.config is not None
        assert calc.config.default_global_mean == 60.0

    def test_custom_config(self) -> None:
        """Test calculator with custom config."""
        config = EstimationConfig(default_global_mean=120.0)
        calc = VarianceCalculator(config)
        assert calc.config.default_global_mean == 120.0


class TestCalculateVariance:
    """Tests for calculate_variance method."""

    def test_multiple_samples_uses_observed(self) -> None:
        """Test that multiple samples use observed std dev."""
        calc = VarianceCalculator()
        stats = RuleTimingStats(rule="test", durations=[100.0, 110.0, 90.0])

        result = calc.calculate_variance(stats)
        assert result.source == "observed"
        assert result.multiplier_used is None
        # Variance should be std_dev^2
        assert result.variance == pytest.approx(stats.std_dev**2)

    def test_single_sample_uses_multiplier(self) -> None:
        """Test that single sample uses multiplier."""
        calc = VarianceCalculator()
        stats = RuleTimingStats(rule="test", durations=[100.0])

        result = calc.calculate_variance(stats, scenario="rule_fallback")
        assert result.source == "single_sample"
        assert result.multiplier_used == 0.3  # rule_fallback default
        # Variance should be (mean * multiplier)^2
        expected = (100.0 * 0.3) ** 2
        assert result.variance == pytest.approx(expected)

    def test_no_samples_uses_fallback(self) -> None:
        """Test that no samples uses fallback."""
        calc = VarianceCalculator()
        stats = RuleTimingStats(rule="test", durations=[])

        result = calc.calculate_variance(stats, scenario="global_fallback")
        assert result.source == "fallback"
        assert result.multiplier_used == 0.5  # global_fallback default
        # Uses default_global_mean
        expected = (60.0 * 0.5) ** 2
        assert result.variance == pytest.approx(expected)

    def test_mean_override(self) -> None:
        """Test mean_override parameter."""
        calc = VarianceCalculator()
        stats = RuleTimingStats(rule="test", durations=[100.0])

        result = calc.calculate_variance(stats, scenario="rule_fallback", mean_override=200.0)
        expected = (200.0 * 0.3) ** 2
        assert result.variance == pytest.approx(expected)

    def test_different_scenarios(self) -> None:
        """Test different scenario multipliers."""
        calc = VarianceCalculator()
        stats = RuleTimingStats(rule="test", durations=[100.0])

        # Each scenario should use its own multiplier
        exact = calc.calculate_variance(stats, scenario="exact_thread_match")
        assert exact.multiplier_used == 0.2

        bootstrap = calc.calculate_variance(stats, scenario="bootstrap")
        assert bootstrap.multiplier_used == 0.5


class TestCalculateVarianceFromMean:
    """Tests for calculate_variance_from_mean method."""

    def test_basic(self) -> None:
        """Test basic variance from mean."""
        calc = VarianceCalculator()
        result = calc.calculate_variance_from_mean(100.0, scenario="global_fallback")

        assert result.source == "fallback"
        assert result.multiplier_used == 0.5
        expected = (100.0 * 0.5) ** 2
        assert result.variance == pytest.approx(expected)


class TestAdjustVarianceForSizeScaling:
    """Tests for adjust_variance_for_size_scaling method."""

    def test_high_confidence_reduces_variance(self) -> None:
        """Test that high confidence reduces variance."""
        calc = VarianceCalculator()
        base_variance = 100.0

        result = calc.adjust_variance_for_size_scaling(base_variance, size_confidence=1.0)
        # With 100% confidence, variance reduced by 50%
        assert result == pytest.approx(50.0)

    def test_zero_confidence_no_change(self) -> None:
        """Test that zero confidence doesn't change variance."""
        calc = VarianceCalculator()
        base_variance = 100.0

        result = calc.adjust_variance_for_size_scaling(base_variance, size_confidence=0.0)
        assert result == pytest.approx(100.0)

    def test_partial_confidence(self) -> None:
        """Test partial confidence."""
        calc = VarianceCalculator()
        base_variance = 100.0

        result = calc.adjust_variance_for_size_scaling(base_variance, size_confidence=0.5)
        # adjustment = 1 - 0.5 * 0.5 = 0.75
        assert result == pytest.approx(75.0)


class TestCalculateConfidence:
    """Tests for calculate_confidence method."""

    def test_basic(self) -> None:
        """Test basic confidence calculation."""
        calc = VarianceCalculator()
        result = calc.calculate_confidence(
            sample_count=10,
            recency_factor=0.8,
            consistency_factor=0.9,
            coverage_factor=0.7,
        )

        assert isinstance(result, ConfidenceResult)
        assert 0.0 <= result.confidence <= 1.0
        assert result.sample_factor == 1.0  # 10 / 10 (min_samples_for_confidence)

    def test_max_confidence_capped(self) -> None:
        """Test that confidence is capped at max."""
        calc = VarianceCalculator()
        result = calc.calculate_confidence(
            sample_count=100,  # Very high
            recency_factor=1.0,
            consistency_factor=1.0,
            coverage_factor=1.0,
        )

        assert result.confidence <= 0.9  # max_confidence

    def test_low_sample_count(self) -> None:
        """Test low sample count reduces confidence."""
        calc = VarianceCalculator()
        result = calc.calculate_confidence(
            sample_count=2,
            recency_factor=1.0,
            consistency_factor=1.0,
            coverage_factor=1.0,
        )

        # sample_factor = 2 / 10 = 0.2
        assert result.sample_factor == pytest.approx(0.2)


class TestCalculateSimpleConfidence:
    """Tests for calculate_simple_confidence method."""

    def test_basic(self) -> None:
        """Test basic simple confidence."""
        calc = VarianceCalculator()

        assert calc.calculate_simple_confidence(0) == 0.0
        assert calc.calculate_simple_confidence(10) == pytest.approx(0.5)
        assert calc.calculate_simple_confidence(20) == pytest.approx(0.7)  # capped at max

    def test_custom_params(self) -> None:
        """Test custom parameters."""
        calc = VarianceCalculator()

        result = calc.calculate_simple_confidence(10, divisor=10, max_confidence=0.9)
        assert result == pytest.approx(0.9)  # 10/10 = 1.0, capped at 0.9


class TestAggregateVariances:
    """Tests for aggregate_variances method."""

    def test_basic(self) -> None:
        """Test basic variance aggregation."""
        calc = VarianceCalculator()

        result = calc.aggregate_variances([100.0, 200.0, 300.0])
        assert result == pytest.approx(600.0)

    def test_with_counts(self) -> None:
        """Test variance aggregation with counts."""
        calc = VarianceCalculator()

        result = calc.aggregate_variances([100.0, 200.0], counts=[2, 3])
        # 100 * 2 + 200 * 3 = 800
        assert result == pytest.approx(800.0)

    def test_empty(self) -> None:
        """Test empty variance list."""
        calc = VarianceCalculator()
        result = calc.aggregate_variances([])
        assert result == 0.0


class TestStdDevFromVariance:
    """Tests for std_dev_from_variance method."""

    def test_basic(self) -> None:
        """Test basic std dev calculation."""
        calc = VarianceCalculator()

        assert calc.std_dev_from_variance(100.0) == pytest.approx(10.0)
        assert calc.std_dev_from_variance(0.0) == 0.0

    def test_negative_returns_zero(self) -> None:
        """Test negative variance returns 0."""
        calc = VarianceCalculator()
        assert calc.std_dev_from_variance(-100.0) == 0.0


class TestCalculateBounds:
    """Tests for calculate_bounds method."""

    def test_basic(self) -> None:
        """Test basic bounds calculation."""
        calc = VarianceCalculator()

        lower, upper = calc.calculate_bounds(100.0, 10.0)
        assert lower == pytest.approx(80.0)
        assert upper == pytest.approx(120.0)

    def test_custom_num_std_devs(self) -> None:
        """Test custom number of std devs."""
        calc = VarianceCalculator()

        lower, upper = calc.calculate_bounds(100.0, 10.0, num_std_devs=1.0)
        assert lower == pytest.approx(90.0)
        assert upper == pytest.approx(110.0)

    def test_lower_clamped_to_zero(self) -> None:
        """Test lower bound is clamped to 0."""
        calc = VarianceCalculator()

        lower, upper = calc.calculate_bounds(10.0, 20.0)
        assert lower == 0.0


class TestCalculateBootstrapBounds:
    """Tests for calculate_bootstrap_bounds method."""

    def test_basic(self) -> None:
        """Test bootstrap bounds."""
        calc = VarianceCalculator()

        lower, upper = calc.calculate_bootstrap_bounds(100.0)
        assert lower == pytest.approx(20.0)  # 20%
        assert upper == pytest.approx(300.0)  # 300%


class TestGetCalculator:
    """Tests for get_calculator function."""

    def test_returns_calculator(self) -> None:
        """Test get_calculator returns a VarianceCalculator."""
        calc = get_calculator()
        assert isinstance(calc, VarianceCalculator)

    def test_with_config(self) -> None:
        """Test get_calculator with custom config."""
        config = EstimationConfig(default_global_mean=999.0)
        calc = get_calculator(config)
        assert calc.config.default_global_mean == 999.0

    def test_caches_default(self) -> None:
        """Test that default calculator is cached."""
        calc1 = get_calculator()
        calc2 = get_calculator()
        assert calc1 is calc2
