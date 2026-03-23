"""Tests for the final performance report (OOS Sharpe CI, DSR, MC ruin, verdict)."""

from __future__ import annotations

import numpy as np

from ep2_crypto.backtest.performance_report import (
    MonteCarloRuin,
    PerformanceReport,
    PerformanceReporter,
    Verdict,
    run_monte_carlo_ruin,
)

# ---------------------------------------------------------------------------
# Synthetic return generators
# ---------------------------------------------------------------------------


def _good_returns(n: int = 5000, seed: int = 0) -> np.ndarray:
    """Returns with a genuine edge: Sharpe ~ 2 after costs."""
    rng = np.random.default_rng(seed)
    return rng.normal(0.0008, 0.002, n)  # mean >> std/sqrt(n)


def _noise_returns(n: int = 5000, seed: int = 0) -> np.ndarray:
    """Returns with no edge: mean ~ 0."""
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 0.002, n)


def _bad_returns(n: int = 5000, seed: int = 0) -> np.ndarray:
    """Returns with negative edge."""
    rng = np.random.default_rng(seed)
    return rng.normal(-0.0008, 0.002, n)


# ---------------------------------------------------------------------------
# MonteCarloRuin
# ---------------------------------------------------------------------------


class TestRunMonteCarloRuin:
    def test_positive_edge_low_ruin_prob(self):
        trade_returns = np.random.default_rng(0).normal(0.003, 0.01, 1000)
        mc = run_monte_carlo_ruin(trade_returns, n_paths=500, n_trades_per_path=200, seed=0)
        assert mc.ruin_probability < 0.20  # good edge → low ruin

    def test_negative_edge_high_ruin_prob(self):
        trade_returns = np.random.default_rng(0).normal(-0.005, 0.01, 1000)
        mc = run_monte_carlo_ruin(trade_returns, n_paths=500, n_trades_per_path=200, seed=0)
        assert mc.ruin_probability > mc.ruin_threshold or mc.median_max_drawdown > 0.1

    def test_empty_returns_no_error(self):
        mc = run_monte_carlo_ruin(np.array([]), n_paths=100, n_trades_per_path=100)
        assert mc.ruin_probability == 0.0
        assert mc.passed

    def test_single_return(self):
        mc = run_monte_carlo_ruin(np.array([0.001]), n_paths=100, n_trades_per_path=50)
        assert 0.0 <= mc.ruin_probability <= 1.0

    def test_p95_ge_median(self):
        trade_returns = np.random.default_rng(0).normal(0.001, 0.01, 500)
        mc = run_monte_carlo_ruin(trade_returns, n_paths=500, n_trades_per_path=200, seed=0)
        assert mc.p95_max_drawdown >= mc.median_max_drawdown

    def test_ruin_probability_in_unit_interval(self):
        trade_returns = np.random.default_rng(0).normal(0, 0.01, 300)
        mc = run_monte_carlo_ruin(trade_returns, n_paths=200, n_trades_per_path=100)
        assert 0.0 <= mc.ruin_probability <= 1.0

    def test_passed_flag_matches_threshold(self):
        trade_returns = np.full(100, 0.005)  # guaranteed no ruin
        mc = run_monte_carlo_ruin(
            trade_returns, n_paths=200, n_trades_per_path=50, ruin_threshold=0.20
        )
        assert mc.passed  # 0% ruin probability

    def test_summary_string_not_empty(self):
        mc = run_monte_carlo_ruin(np.full(100, 0.001), n_paths=50, n_trades_per_path=50)
        s = mc.summary()
        assert len(s) > 20
        assert "PASS" in s or "FAIL" in s

    def test_reproducible(self):
        data = np.random.default_rng(7).normal(0, 0.01, 200)
        mc1 = run_monte_carlo_ruin(data, n_paths=100, n_trades_per_path=100, seed=7)
        mc2 = run_monte_carlo_ruin(data, n_paths=100, n_trades_per_path=100, seed=7)
        assert mc1.ruin_probability == mc2.ruin_probability


# ---------------------------------------------------------------------------
# PerformanceReporter._score_criteria
# ---------------------------------------------------------------------------


class TestScoreCriteria:
    def test_all_pass(self):
        passed, total = PerformanceReporter._score_criteria(
            sharpe_ci_lower=0.5,
            dsr=0.96,
            psr=0.97,
            permutation_p=0.02,
            wf_cv=0.3,
            breakeven_bps=20.0,
            mc_ruin_passed=True,
        )
        assert passed == 7
        assert total == 7

    def test_all_fail(self):
        passed, total = PerformanceReporter._score_criteria(
            sharpe_ci_lower=-0.5,
            dsr=0.50,
            psr=0.50,
            permutation_p=0.50,
            wf_cv=2.0,
            breakeven_bps=5.0,
            mc_ruin_passed=False,
        )
        assert passed == 0
        assert total == 7

    def test_partial_pass(self):
        passed, total = PerformanceReporter._score_criteria(
            sharpe_ci_lower=0.1,  # pass
            dsr=0.96,  # pass
            psr=0.50,  # fail
            permutation_p=0.50,  # fail
            wf_cv=0.3,  # pass
            breakeven_bps=5.0,  # fail
            mc_ruin_passed=True,  # pass
        )
        assert passed == 4
        assert total == 7


# ---------------------------------------------------------------------------
# PerformanceReporter._compute_verdict
# ---------------------------------------------------------------------------


class TestComputeVerdict:
    def test_deploy_at_6_of_7(self):
        assert PerformanceReporter._compute_verdict(6, 7) == Verdict.DEPLOY

    def test_deploy_at_7_of_7(self):
        assert PerformanceReporter._compute_verdict(7, 7) == Verdict.DEPLOY

    def test_caution_at_4_of_7(self):
        assert PerformanceReporter._compute_verdict(4, 7) == Verdict.CAUTION

    def test_caution_at_5_of_7(self):
        assert PerformanceReporter._compute_verdict(5, 7) == Verdict.CAUTION

    def test_do_not_deploy_at_0(self):
        assert PerformanceReporter._compute_verdict(0, 7) == Verdict.DO_NOT_DEPLOY

    def test_do_not_deploy_at_3(self):
        assert PerformanceReporter._compute_verdict(3, 7) == Verdict.DO_NOT_DEPLOY


# ---------------------------------------------------------------------------
# PerformanceReporter.generate — minimal runs (fast)
# ---------------------------------------------------------------------------


class TestPerformanceReporterGenerate:
    """Uses small bootstrap/MC to keep tests fast."""

    def _reporter(self) -> PerformanceReporter:
        return PerformanceReporter(
            n_bootstrap=500,
            n_permutations=200,
            n_mc_paths=200,
            n_mc_trades=200,
            seed=42,
        )

    def test_returns_performance_report_type(self):
        r = _good_returns(n=2000)
        report = self._reporter().generate(r)
        assert isinstance(report, PerformanceReport)

    def test_verdict_field_is_verdict_enum(self):
        r = _good_returns(n=2000)
        report = self._reporter().generate(r)
        assert isinstance(report.verdict, Verdict)

    def test_good_returns_high_sharpe(self):
        r = _good_returns(n=3000)
        report = self._reporter().generate(r)
        # lo_corrected_sharpe returns a per-bar-scale value; good returns should be > 0
        assert report.sharpe_lo_corrected > 0.0

    def test_noise_returns_low_sharpe(self):
        r = _noise_returns(n=3000, seed=99)
        report = self._reporter().generate(r)
        assert report.sharpe_lo_corrected < 1.0

    def test_ci_lower_lt_upper(self):
        r = _good_returns(n=2000)
        report = self._reporter().generate(r)
        assert report.sharpe_ci_lower < report.sharpe_ci_upper

    def test_good_returns_ci_lower_positive(self):
        r = _good_returns(n=3000, seed=0)
        report = self._reporter().generate(r)
        assert report.sharpe_ci_lower > 0

    def test_dsr_in_unit_interval(self):
        r = _good_returns(n=2000)
        report = self._reporter().generate(r)
        assert 0.0 <= report.dsr <= 1.0

    def test_psr_in_unit_interval(self):
        r = _good_returns(n=2000)
        report = self._reporter().generate(r)
        assert 0.0 <= report.psr <= 1.0

    def test_max_drawdown_in_unit_interval(self):
        r = _good_returns(n=2000)
        report = self._reporter().generate(r)
        assert 0.0 <= report.max_drawdown <= 1.0

    def test_monte_carlo_present(self):
        r = _good_returns(n=2000)
        report = self._reporter().generate(r)
        assert isinstance(report.monte_carlo, MonteCarloRuin)
        assert 0.0 <= report.monte_carlo.ruin_probability <= 1.0

    def test_regime_breakdown_when_labels_provided(self):
        np.random.default_rng(0)
        n = 2000
        r = _good_returns(n)
        labels = (np.arange(n) // (n // 3)).astype(np.int32)
        report = self._reporter().generate(r, regime_labels=labels)
        assert len(report.regime_breakdown) >= 1

    def test_empty_regime_breakdown_when_no_labels(self):
        r = _good_returns(n=2000)
        report = self._reporter().generate(r, regime_labels=None)
        assert report.regime_breakdown == {}

    def test_wf_stability_with_fold_sharpes(self):
        r = _good_returns(n=2000)
        fold_sharpes = [1.5, 1.8, 1.2, 2.0, 1.6]
        report = self._reporter().generate(r, fold_sharpes=fold_sharpes)
        assert report.wf_stability_cv < float("inf")

    def test_wf_stability_without_fold_sharpes(self):
        r = _good_returns(n=2000)
        report = self._reporter().generate(r, fold_sharpes=None)
        # wf_cv defaults to inf when no fold sharpes
        assert report.wf_stability_cv == float("inf")

    def test_bad_returns_do_not_deploy(self):
        r = _bad_returns(n=3000)
        report = self._reporter().generate(r)
        assert report.verdict in (Verdict.CAUTION, Verdict.DO_NOT_DEPLOY)

    def test_good_returns_deploy_or_caution(self):
        r = _good_returns(n=4000, seed=0)
        report = self._reporter().generate(r)
        assert report.verdict in (Verdict.DEPLOY, Verdict.CAUTION)

    def test_to_dict_no_validation_key(self):
        r = _good_returns(n=2000)
        report = self._reporter().generate(r)
        d = report.to_dict()
        assert "validation" not in d

    def test_to_dict_contains_required_fields(self):
        r = _good_returns(n=2000)
        report = self._reporter().generate(r)
        d = report.to_dict()
        for field in (
            "sharpe_lo_corrected",
            "dsr",
            "psr",
            "max_drawdown",
            "verdict",
            "criteria_passed",
            "criteria_total",
        ):
            assert field in d

    def test_summary_contains_verdict(self):
        r = _good_returns(n=2000)
        report = self._reporter().generate(r)
        s = report.summary()
        assert "VERDICT" in s
        assert report.verdict.value in s

    def test_to_markdown_contains_table(self):
        r = _good_returns(n=2000)
        report = self._reporter().generate(r)
        md = report.to_markdown()
        assert "|" in md
        assert "Sharpe" in md

    def test_n_trials_affects_dsr(self):
        r = _good_returns(n=2000)
        rep = self._reporter()
        report_1 = rep.generate(r, n_trials=1)
        report_100 = rep.generate(r, n_trials=100)
        # More trials → lower DSR (harder to pass)
        assert report_1.dsr >= report_100.dsr

    def test_custom_trade_returns_used_for_mc(self):
        r = _good_returns(n=2000)
        trade_ret = np.full(100, 0.005)  # guaranteed positive
        report = self._reporter().generate(r, trade_returns=trade_ret)
        assert report.monte_carlo.passed

    def test_cost_sensitivity_populated(self):
        r = _good_returns(n=2000)
        report = self._reporter().generate(r)
        assert len(report.cost_sensitivity_rows) >= 2

    def test_breakeven_positive_for_good_returns(self):
        r = _good_returns(n=3000)
        report = self._reporter().generate(r)
        assert report.breakeven_cost_bps > 0
