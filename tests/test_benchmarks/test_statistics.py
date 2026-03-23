"""Tests for statistical analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ep2_crypto.benchmarks.statistics import StatisticalAnalyzer


@pytest.fixture
def analyzer() -> StatisticalAnalyzer:
    return StatisticalAnalyzer(confidence_level=0.95)


class TestSharpeCI:
    def test_ci_contains_point_estimate(self, analyzer: StatisticalAnalyzer) -> None:
        rng = np.random.default_rng(42)
        returns = pd.Series(rng.normal(0.0001, 0.002, 5000))
        result = analyzer.sharpe_confidence_interval(returns, method="analytical")
        assert result["analytical"]["ci_lower"] <= result["sharpe_ratio"]
        assert result["analytical"]["ci_upper"] >= result["sharpe_ratio"]

    def test_wider_ci_with_fewer_obs(self, analyzer: StatisticalAnalyzer) -> None:
        rng = np.random.default_rng(42)
        returns_long = pd.Series(rng.normal(0.0001, 0.002, 10000))
        returns_short = pd.Series(rng.normal(0.0001, 0.002, 100))
        ci_long = analyzer.sharpe_confidence_interval(returns_long, method="analytical")
        ci_short = analyzer.sharpe_confidence_interval(returns_short, method="analytical")
        width_long = ci_long["analytical"]["ci_upper"] - ci_long["analytical"]["ci_lower"]
        width_short = ci_short["analytical"]["ci_upper"] - ci_short["analytical"]["ci_lower"]
        assert width_short > width_long


class TestSignificanceTests:
    def test_strong_signal_significant(self, analyzer: StatisticalAnalyzer) -> None:
        # Very high Sharpe should be significant
        rng = np.random.default_rng(42)
        returns = pd.Series(rng.normal(0.001, 0.001, 5000))
        result = analyzer.test_sharpe_vs_zero(returns)
        assert result["significant"]
        assert result["p_value"] < 0.01

    def test_zero_mean_not_significant(self, analyzer: StatisticalAnalyzer) -> None:
        rng = np.random.default_rng(42)
        returns = pd.Series(rng.normal(0, 0.002, 5000))
        result = analyzer.test_sharpe_vs_zero(returns)
        # With zero mean, p-value should be high (usually > 0.05)
        # Not guaranteed due to randomness, but highly likely
        assert result["p_value"] > 0.01

    def test_sharpe_difference(self, analyzer: StatisticalAnalyzer) -> None:
        rng = np.random.default_rng(42)
        a = pd.Series(rng.normal(0.001, 0.002, 5000))
        b = pd.Series(rng.normal(0.0, 0.002, 5000))
        result = analyzer.test_sharpe_difference(a, b)
        assert result["sharpe_a_annual"] > result["sharpe_b_annual"]


class TestMultipleTesting:
    def test_holm_correction(self, analyzer: StatisticalAnalyzer) -> None:
        p_values = {"a": 0.01, "b": 0.03, "c": 0.06}
        corrected = analyzer.multiple_testing_correction(p_values, method="holm")
        # Adjusted p-values should be >= original
        for name in p_values:
            assert corrected[name]["adjusted_p"] >= corrected[name]["original_p"]

    def test_bonferroni_correction(self, analyzer: StatisticalAnalyzer) -> None:
        p_values = {"a": 0.01, "b": 0.03, "c": 0.06}
        corrected = analyzer.multiple_testing_correction(p_values, method="bonferroni")
        assert corrected["a"]["adjusted_p"] == pytest.approx(0.03, abs=1e-10)


class TestDrawdownAnalysis:
    def test_positive_max_drawdown(self, analyzer: StatisticalAnalyzer) -> None:
        rng = np.random.default_rng(42)
        returns = pd.Series(rng.normal(0, 0.002, 3000))
        result = analyzer.drawdown_analysis(returns)
        assert result["max_drawdown"] >= 0

    def test_monotonic_increase_no_drawdown(self, analyzer: StatisticalAnalyzer) -> None:
        returns = pd.Series(np.full(100, 0.001))
        result = analyzer.drawdown_analysis(returns)
        assert result["max_drawdown"] == pytest.approx(0.0, abs=1e-10)


class TestReturnDistribution:
    def test_fat_tails_detection(self, analyzer: StatisticalAnalyzer) -> None:
        rng = np.random.default_rng(42)
        # Student-t with low df has fat tails
        returns = pd.Series(rng.standard_t(df=3, size=10000) * 0.002)
        result = analyzer.return_distribution_analysis(returns)
        assert result["excess_kurtosis"] > 3  # Normal = 0
        assert result["fat_tails"]

    def test_normal_returns(self, analyzer: StatisticalAnalyzer) -> None:
        rng = np.random.default_rng(42)
        returns = pd.Series(rng.normal(0, 0.002, 10000))
        result = analyzer.return_distribution_analysis(returns)
        assert result["excess_kurtosis"] < 2  # Should be near 0
