"""Tests for statistical validation suite."""

from __future__ import annotations

import numpy as np
import pytest

from ep2_crypto.backtest.validation import (
    Verdict,
    block_bootstrap_ci,
    deflated_sharpe_ratio,
    permutation_test,
    probabilistic_sharpe_ratio,
    run_validation_suite,
    walk_forward_stability,
)


# ---------------------------------------------------------------------------
# PSR
# ---------------------------------------------------------------------------
class TestPSR:
    def test_strong_edge_high_psr(self) -> None:
        """Strong edge → PSR close to 1."""
        rng = np.random.default_rng(42)
        returns = rng.normal(0.0005, 0.001, size=50_000)
        psr = probabilistic_sharpe_ratio(returns)
        assert psr > 0.99

    def test_no_edge_low_psr(self) -> None:
        """Zero-mean returns → PSR well below 0.95 (not significant)."""
        rng = np.random.default_rng(42)
        returns = rng.normal(0.0, 0.001, size=10_000)
        psr = probabilistic_sharpe_ratio(returns)
        assert psr < 0.95  # not significant

    def test_negative_edge_low_psr(self) -> None:
        """Negative-mean returns → PSR < 0.5."""
        rng = np.random.default_rng(42)
        returns = rng.normal(-0.0003, 0.001, size=10_000)
        psr = probabilistic_sharpe_ratio(returns)
        assert psr < 0.1

    def test_short_series_uninformative(self) -> None:
        """Very short series → PSR = 0.5."""
        returns = np.array([0.001, -0.001, 0.002])
        assert probabilistic_sharpe_ratio(returns) == 0.5

    def test_benchmark_comparison(self) -> None:
        """PSR against a high benchmark should be lower than against zero."""
        rng = np.random.default_rng(42)
        # Weak edge so PSR doesn't saturate at 1.0
        returns = rng.normal(0.00005, 0.001, size=10_000)
        psr_zero = probabilistic_sharpe_ratio(returns, benchmark_sharpe=0.0)
        psr_high = probabilistic_sharpe_ratio(returns, benchmark_sharpe=5.0)
        assert psr_high < psr_zero


# ---------------------------------------------------------------------------
# DSR
# ---------------------------------------------------------------------------
class TestDSR:
    def test_single_trial_equals_psr(self) -> None:
        """With 1 trial, DSR should equal PSR."""
        rng = np.random.default_rng(42)
        returns = rng.normal(0.0003, 0.001, size=10_000)
        psr = probabilistic_sharpe_ratio(returns)
        dsr = deflated_sharpe_ratio(returns, n_trials=1)
        assert abs(psr - dsr) < 0.01

    def test_more_trials_lower_dsr(self) -> None:
        """More trials → lower DSR (stricter test)."""
        rng = np.random.default_rng(42)
        # Weak edge so DSR doesn't saturate at 1.0
        returns = rng.normal(0.00005, 0.001, size=10_000)
        dsr_1 = deflated_sharpe_ratio(returns, n_trials=1)
        dsr_100 = deflated_sharpe_ratio(returns, n_trials=100)
        assert dsr_100 < dsr_1

    def test_strong_edge_survives_multiple_testing(self) -> None:
        """Very strong edge should still have high DSR even with many trials."""
        rng = np.random.default_rng(42)
        returns = rng.normal(0.001, 0.001, size=50_000)
        dsr = deflated_sharpe_ratio(returns, n_trials=50)
        assert dsr > 0.9


# ---------------------------------------------------------------------------
# Permutation Test
# ---------------------------------------------------------------------------
class TestPermutationTest:
    def test_significant_signal(self) -> None:
        """A signal with real timing should have p < 0.05."""
        rng = np.random.default_rng(42)
        n = 5000
        raw_returns = rng.normal(0, 0.01, n)
        # Signal that predicts direction (imperfect oracle)
        positions = np.sign(raw_returns + rng.normal(0, 0.005, n))
        result = permutation_test(raw_returns, positions, n_permutations=1000, seed=42)
        assert result["p_value"] < 0.05

    def test_random_signal_not_significant(self) -> None:
        """Random positions should have p > 0.05 typically."""
        rng = np.random.default_rng(42)
        n = 5000
        raw_returns = rng.normal(0, 0.01, n)
        positions = rng.choice([-1.0, 0.0, 1.0], size=n)
        result = permutation_test(raw_returns, positions, n_permutations=1000, seed=42)
        assert result["p_value"] > 0.01  # not significant

    def test_output_format(self) -> None:
        rng = np.random.default_rng(42)
        n = 500
        raw_returns = rng.normal(0, 0.01, n)
        positions = rng.choice([-1.0, 1.0], size=n)
        result = permutation_test(raw_returns, positions, n_permutations=100)
        assert "observed_sharpe" in result
        assert "p_value" in result
        assert 0 <= result["p_value"] <= 1


# ---------------------------------------------------------------------------
# Block Bootstrap
# ---------------------------------------------------------------------------
class TestBlockBootstrap:
    def test_positive_edge_ci_above_zero(self) -> None:
        """Strong positive edge → CI lower bound > 0."""
        rng = np.random.default_rng(42)
        returns = rng.normal(0.0005, 0.001, size=10_000)
        boot = block_bootstrap_ci(returns, n_iterations=1000, seed=42)
        assert boot["ci_lower"] > 0
        assert boot["ci_upper"] > boot["ci_lower"]

    def test_zero_edge_ci_straddles_zero(self) -> None:
        """Zero-mean returns → CI should straddle zero."""
        rng = np.random.default_rng(42)
        returns = rng.normal(0.0, 0.001, size=5000)
        boot = block_bootstrap_ci(returns, n_iterations=1000, seed=42)
        assert boot["ci_lower"] < 0 < boot["ci_upper"]

    def test_output_keys(self) -> None:
        rng = np.random.default_rng(42)
        returns = rng.normal(0, 0.001, 1000)
        boot = block_bootstrap_ci(returns, n_iterations=100)
        assert "mean_sharpe" in boot
        assert "ci_lower" in boot
        assert "ci_upper" in boot
        assert "std_sharpe" in boot

    def test_short_series(self) -> None:
        returns = np.array([0.001, -0.001])
        boot = block_bootstrap_ci(returns, n_iterations=100)
        assert "mean_sharpe" in boot


# ---------------------------------------------------------------------------
# Walk-Forward Stability
# ---------------------------------------------------------------------------
class TestWFStability:
    def test_stable_folds(self) -> None:
        """Consistent fold Sharpes → CV < 0.5."""
        sharpes = [1.5, 1.6, 1.4, 1.5, 1.7, 1.3, 1.5, 1.6]
        result = walk_forward_stability(sharpes)
        assert result["stable"] is True
        assert result["cv"] < 0.5

    def test_unstable_folds(self) -> None:
        """Widely varying Sharpes → CV > 0.5."""
        sharpes = [3.0, -2.0, 5.0, -1.0, 4.0, -3.0]
        result = walk_forward_stability(sharpes)
        assert result["stable"] is False

    def test_pct_positive(self) -> None:
        sharpes = [1.0, 2.0, -0.5, 1.5, 0.3]
        result = walk_forward_stability(sharpes)
        assert result["pct_positive"] == 0.8

    def test_single_fold(self) -> None:
        result = walk_forward_stability([1.5])
        assert result["cv"] == float("inf")
        assert result["stable"] is False


# ---------------------------------------------------------------------------
# Combined Validation Suite
# ---------------------------------------------------------------------------
class TestValidationSuite:
    def test_genuine_edge_verdict(self) -> None:
        """Strong edge should get GENUINE_EDGE verdict."""
        rng = np.random.default_rng(42)
        n = 20_000
        raw_returns = rng.normal(0, 0.01, n)
        # Semi-oracle signal
        positions = np.sign(raw_returns + rng.normal(0, 0.003, n))
        strategy_returns = raw_returns * positions

        result = run_validation_suite(
            returns=strategy_returns,
            positions=positions,
            raw_returns=raw_returns,
            fold_sharpes=[2.0, 1.8, 2.2, 1.9, 2.1],
            n_trials=1,
            n_permutations=500,
            n_bootstrap=500,
            seed=42,
        )
        assert result.verdict == Verdict.GENUINE_EDGE
        assert result.tests_passed > 0

    def test_noise_verdict(self) -> None:
        """Random signal should get LIKELY_NOISE or INCONCLUSIVE."""
        rng = np.random.default_rng(42)
        n = 5000
        returns = rng.normal(0, 0.001, n)

        result = run_validation_suite(
            returns=returns,
            fold_sharpes=[-0.5, 0.2, -0.3, 0.1, -0.4],
            n_permutations=200,
            n_bootstrap=200,
            seed=42,
        )
        assert result.verdict in (Verdict.LIKELY_NOISE, Verdict.INCONCLUSIVE)

    def test_summary_string(self) -> None:
        rng = np.random.default_rng(42)
        returns = rng.normal(0.0001, 0.001, 5000)
        result = run_validation_suite(
            returns=returns,
            n_permutations=100,
            n_bootstrap=100,
        )
        summary = result.summary()
        assert "Verdict" in summary
        assert "PSR" in summary
        assert "DSR" in summary

    def test_minimal_input(self) -> None:
        """Should handle minimal data without error."""
        returns = np.array([0.001, -0.001, 0.002, -0.0005] * 10)
        result = run_validation_suite(
            returns=returns,
            n_permutations=50,
            n_bootstrap=50,
        )
        assert isinstance(result.verdict, Verdict)

    def test_details_populated(self) -> None:
        rng = np.random.default_rng(42)
        returns = rng.normal(0, 0.001, 5000)
        result = run_validation_suite(
            returns=returns,
            n_permutations=100,
            n_bootstrap=100,
        )
        assert "psr" in result.details
        assert "dsr" in result.details
        assert "bootstrap" in result.details
