"""Tests for historical and synthetic stress testing framework."""

from __future__ import annotations

import numpy as np

from ep2_crypto.backtest.metrics import BacktestResult
from ep2_crypto.backtest.stress_tests import (
    HistoricalScenario,
    HistoricalStressScenario,
    StressReport,
    StressTestResult,
    StressTestRunner,
    SyntheticScenario,
    SyntheticStressScenario,
    _closes_to_ohlcv,
    _evaluate_stress_result,
    _make_crash_path,
)

# ---------------------------------------------------------------------------
# Helper: build a minimal BacktestResult stub
# ---------------------------------------------------------------------------


def _stub_result(
    max_drawdown: float = 0.05,
    total_trades: int = 50,
    sharpe: float = 1.0,
    total_return: float = 0.10,
) -> BacktestResult:
    import numpy as np

    from ep2_crypto.backtest.metrics import compute_backtest_result

    rng = np.random.default_rng(0)
    returns = rng.normal(0.0002, 0.002, 500)
    compute_backtest_result(returns)
    # Patch drawdown and trades (we only care about these for pass/fail)
    return BacktestResult(
        sharpe_ratio=sharpe,
        sharpe_raw=sharpe,
        sortino_ratio=sharpe * 1.5,
        calmar_ratio=sharpe / max(max_drawdown, 0.001),
        total_return=total_return,
        annualized_return=total_return,
        cvar_5pct=-0.01,
        max_drawdown=max_drawdown,
        max_drawdown_duration_bars=100,
        avg_drawdown=max_drawdown / 2,
        total_trades=total_trades,
        win_rate=0.52,
        profit_factor=1.2,
        expectancy_per_trade=3.0,
        avg_win_bps=10.0,
        avg_loss_bps=-8.0,
        avg_bars_per_trade=4.0,
        trades_per_day=10.0,
        total_fee_usd=100.0,
        total_slippage_usd=50.0,
        total_funding_usd=20.0,
        total_cost_usd=170.0,
        skewness=-0.2,
        kurtosis=3.5,
    )


# ---------------------------------------------------------------------------
# Price path generators
# ---------------------------------------------------------------------------


class TestMakeCrashPath:
    def test_output_shape(self):
        closes = _make_crash_path(
            n_bars=200, start_price=30_000.0, crash_fraction=0.30, crash_start=50, crash_end=130
        )
        assert len(closes) == 200

    def test_crash_reduces_price(self):
        closes = _make_crash_path(
            n_bars=300, start_price=30_000.0, crash_fraction=0.40, crash_start=50, crash_end=200
        )
        pre_crash = closes[49]
        post_crash = closes[200]
        assert post_crash < pre_crash * 0.80  # at least 20% drop

    def test_no_negative_prices(self):
        closes = _make_crash_path(
            n_bars=200, start_price=30_000.0, crash_fraction=0.99, crash_start=10, crash_end=100
        )
        assert np.all(closes > 0)

    def test_reproducible_with_seed(self):
        c1 = _make_crash_path(100, 30_000.0, 0.3, 20, 70, seed=7)
        c2 = _make_crash_path(100, 30_000.0, 0.3, 20, 70, seed=7)
        assert np.allclose(c1, c2)


class TestClosesToOHLCV:
    def test_output_shapes_consistent(self):
        closes = np.linspace(30_000.0, 29_000.0, 100)
        o, h, l, c, v, ts = _closes_to_ohlcv(closes)
        for arr in (o, h, l, c, v):
            assert len(arr) == 100
        assert len(ts) == 100

    def test_high_ge_close(self):
        closes = np.full(50, 30_000.0)
        _o, h, _l, c, _v, _ts = _closes_to_ohlcv(closes)
        assert np.all(h >= c * 0.99)

    def test_low_le_close(self):
        closes = np.full(50, 30_000.0)
        _o, _h, l, c, _v, _ts = _closes_to_ohlcv(closes)
        assert np.all(l <= c * 1.01)

    def test_timestamps_monotonically_increasing(self):
        closes = np.ones(50) * 30_000.0
        _, _, _, _, _, ts = _closes_to_ohlcv(closes)
        assert np.all(np.diff(ts) > 0)


# ---------------------------------------------------------------------------
# HistoricalStressScenario
# ---------------------------------------------------------------------------


class TestHistoricalStressScenario:
    def setup_method(self):
        self.gen = HistoricalStressScenario()

    def test_all_scenarios_generate(self):
        for scenario in HistoricalScenario:
            result = self.gen.generate(scenario, seed=0)
            assert len(result) == 9  # 6 OHLCV + 2 signals + meta

    def test_covid_crash_depth(self):
        _, _, _, closes, _, _, _, _, _meta = self.gen.generate(
            HistoricalScenario.COVID_CRASH_MAR2020, seed=0
        )
        peak = closes[:300].max()
        trough = closes[200:800].min()
        assert (peak - trough) / peak > 0.30  # at least 30% drop

    def test_ftx_crash_moderate_depth(self):
        _, _, _, closes, _, _, _, _, _meta = self.gen.generate(
            HistoricalScenario.FTX_COLLAPSE_NOV2022, seed=0
        )
        peak = closes[:600].max()
        trough = closes[500:2000].min()
        assert (peak - trough) / peak > 0.10  # at least 10% drop

    def test_meta_contains_description(self):
        _, _, _, _, _, _, _, _, meta = self.gen.generate(
            HistoricalScenario.CHINA_BAN_MAY2021, seed=0
        )
        assert "description" in meta
        assert len(meta["description"]) > 5

    def test_reproducible_with_seed(self):
        r1 = self.gen.generate(HistoricalScenario.ATH_CORRECTION_MAR2024, seed=3)
        r2 = self.gen.generate(HistoricalScenario.ATH_CORRECTION_MAR2024, seed=3)
        assert np.allclose(r1[3], r2[3])  # closes match

    def test_outputs_have_no_nan(self):
        opens, highs, lows, closes, volumes, _ts, _sigs, _confs, _ = self.gen.generate(
            HistoricalScenario.COVID_CRASH_MAR2020, seed=0
        )
        for arr in (opens, highs, lows, closes, volumes):
            assert not np.any(np.isnan(arr))


# ---------------------------------------------------------------------------
# SyntheticStressScenario
# ---------------------------------------------------------------------------


class TestSyntheticStressScenario:
    def setup_method(self):
        self.gen = SyntheticStressScenario()

    def test_zero_vol_produces_flat_window(self):
        _opens, _highs, _lows, closes, _volumes, _ts, _sigs, _confs = self.gen.zero_volatility_48h(
            seed=0
        )
        zero_window = closes[200:776]
        vol = np.std(np.diff(np.log(zero_window)))
        normal_vol = np.std(np.diff(np.log(closes[:200])))
        assert vol < normal_vol * 0.2  # zero-vol window is much quieter

    def test_flash_crashes_produce_drops(self):
        _, _, _, closes, _, _, _, _ = self.gen.ten_flash_crashes(seed=0)
        log_returns = np.diff(np.log(closes))
        # At least 5 bars with drops > 3%
        big_drops = (log_returns < -0.03).sum()
        assert big_drops >= 5

    def test_correlation_drop_output_length(self):
        tup = self.gen.correlation_drop(n_bars=500, seed=0)
        assert len(tup[3]) == 500  # closes

    def test_high_funding_returns_funding_array(self):
        result = self.gen.high_funding_rate(n_bars=200, seed=0)
        assert len(result) == 9  # extra funding_rates array
        funding_rates = result[8]
        assert funding_rates is not None
        assert len(funding_rates) == 200
        assert funding_rates.max() > 0

    def test_high_funding_spikes_at_8h_intervals(self):
        result = self.gen.high_funding_rate(n_bars=400, seed=0)
        funding_rates = result[8]
        assert funding_rates[0] > 0 or funding_rates[96] > 0

    def test_broken_model_all_long_signals(self):
        _, _, _, _, _, _, signals, confidences = self.gen.broken_model(seed=0)
        # After warm-up, all signals are +1
        assert np.all(signals[10:] == 1)
        assert np.all(confidences[10:] > 0.5)

    def test_no_negative_prices_any_scenario(self):
        scenarios = [
            self.gen.zero_volatility_48h(seed=1),
            self.gen.ten_flash_crashes(seed=1),
            self.gen.correlation_drop(seed=1),
            self.gen.broken_model(seed=1),
        ]
        for tup in scenarios:
            closes = tup[3]
            assert np.all(closes > 0)


# ---------------------------------------------------------------------------
# _evaluate_stress_result
# ---------------------------------------------------------------------------


class TestEvaluateStressResult:
    def test_low_dd_passes(self):
        result = _stub_result(max_drawdown=0.05, total_trades=50)
        sr = _evaluate_stress_result("test", result, n_bars=500, max_allowed_dd=0.20)
        assert sr.passed

    def test_high_dd_without_kill_switch_fails(self):
        result = _stub_result(max_drawdown=0.30, total_trades=500)
        sr = _evaluate_stress_result("test", result, n_bars=500, max_allowed_dd=0.20)
        assert not sr.passed

    def test_kill_switch_fired_high_dd_still_can_pass(self):
        # Kill switch fires early (few trades) + moderate DD <= threshold + buffer
        result = _stub_result(max_drawdown=0.18, total_trades=5)
        sr = _evaluate_stress_result(
            "test", result, n_bars=600, max_allowed_dd=0.20, kill_switch_dd_threshold=0.15
        )
        assert sr.passed

    def test_result_contains_scenario_name(self):
        result = _stub_result()
        sr = _evaluate_stress_result("covid_crash", result, n_bars=1000)
        assert sr.scenario == "covid_crash"

    def test_kill_switch_proxy_fires_when_few_trades(self):
        result = _stub_result(total_trades=3)
        sr = _evaluate_stress_result("x", result, n_bars=600)
        assert sr.kill_switch_fired

    def test_kill_switch_not_fired_when_many_trades(self):
        result = _stub_result(total_trades=200)
        sr = _evaluate_stress_result("x", result, n_bars=600)
        assert not sr.kill_switch_fired

    def test_str_representation(self):
        result = _stub_result()
        sr = _evaluate_stress_result("scenario_x", result, n_bars=500)
        s = str(sr)
        assert "scenario_x" in s
        assert "PASS" in s or "FAIL" in s


# ---------------------------------------------------------------------------
# StressReport
# ---------------------------------------------------------------------------


class TestStressReport:
    def _make_report(self) -> StressReport:
        results = [
            StressTestResult("a", True, 0.05, False, 50, 1.0, 0.10, 500),
            StressTestResult("b", True, 0.10, True, 10, 0.5, 0.03, 600),
            StressTestResult("c", False, 0.30, False, 200, -0.5, -0.15, 400),
        ]
        return StressReport(results=results, max_allowed_dd=0.20)

    def test_n_passed_correct(self):
        r = self._make_report()
        assert r.n_passed == 2

    def test_n_failed_correct(self):
        r = self._make_report()
        assert r.n_failed == 1

    def test_all_passed_false(self):
        r = self._make_report()
        assert not r.all_passed

    def test_all_passed_true_when_all_pass(self):
        results = [
            StressTestResult("a", True, 0.05, False, 50, 1.0, 0.10, 500),
        ]
        r = StressReport(results=results)
        assert r.all_passed

    def test_summary_contains_verdict(self):
        r = self._make_report()
        s = r.summary()
        assert "FAILED" in s or "PASS" in s

    def test_to_dict_length(self):
        r = self._make_report()
        d = r.to_dict()
        assert len(d) == 3

    def test_to_dict_contains_fields(self):
        r = self._make_report()
        for entry in r.to_dict():
            assert "scenario" in entry
            assert "passed" in entry
            assert "max_drawdown" in entry


# ---------------------------------------------------------------------------
# StressTestRunner — historical (fast: run one scenario only)
# ---------------------------------------------------------------------------


class TestStressTestRunnerHistorical:
    def setup_method(self):
        self.runner = StressTestRunner(initial_equity=50_000.0, seed=42)

    def test_single_historical_scenario_runs(self):
        results = self.runner.run_historical([HistoricalScenario.ATH_CORRECTION_MAR2024])
        assert len(results) == 1
        r = results[0]
        assert r.scenario == HistoricalScenario.ATH_CORRECTION_MAR2024.value
        assert isinstance(r.passed, bool)

    def test_historical_result_has_valid_dd(self):
        results = self.runner.run_historical([HistoricalScenario.ATH_CORRECTION_MAR2024])
        assert 0.0 <= results[0].max_drawdown <= 1.0

    def test_ftx_crash_passes_with_risk_engine(self):
        results = self.runner.run_historical([HistoricalScenario.FTX_COLLAPSE_NOV2022])
        r = results[0]
        # With default risk config (15% DD halt), should pass
        assert r.passed

    def test_multiple_historical_scenarios(self):
        results = self.runner.run_historical(
            [
                HistoricalScenario.CHINA_BAN_MAY2021,
                HistoricalScenario.ATH_CORRECTION_MAR2024,
            ]
        )
        assert len(results) == 2


# ---------------------------------------------------------------------------
# StressTestRunner — synthetic (fast: run zero-vol + broken-model)
# ---------------------------------------------------------------------------


class TestStressTestRunnerSynthetic:
    def setup_method(self):
        self.runner = StressTestRunner(initial_equity=50_000.0, seed=0)

    def test_zero_vol_scenario_runs(self):
        results = self.runner.run_synthetic([SyntheticScenario.ZERO_VOLATILITY_48H])
        assert len(results) == 1
        assert isinstance(results[0].passed, bool)

    def test_broken_model_kill_switch_fires(self):
        results = self.runner.run_synthetic([SyntheticScenario.BROKEN_MODEL])
        r = results[0]
        # Kill switch or low DD: should pass with default risk config
        assert r.passed

    def test_high_funding_scenario_runs(self):
        results = self.runner.run_synthetic([SyntheticScenario.HIGH_FUNDING_RATE])
        assert len(results) == 1

    def test_flash_crash_scenario_runs(self):
        results = self.runner.run_synthetic([SyntheticScenario.TEN_FLASH_CRASHES])
        assert len(results) == 1

    def test_correlation_drop_runs(self):
        results = self.runner.run_synthetic([SyntheticScenario.CORRELATION_DROP])
        assert len(results) == 1


# ---------------------------------------------------------------------------
# StressTestRunner — run_all (just verify structure, not all scenarios)
# ---------------------------------------------------------------------------


class TestStressTestRunnerRunAll:
    def test_run_all_returns_stress_report(self):
        runner = StressTestRunner(initial_equity=50_000.0, seed=0)
        report = runner.run_all(
            historical_scenarios=[HistoricalScenario.ATH_CORRECTION_MAR2024],
            synthetic_scenarios=[SyntheticScenario.ZERO_VOLATILITY_48H],
        )
        assert isinstance(report, StressReport)
        assert len(report.results) == 2

    def test_stress_report_summary_not_empty(self):
        runner = StressTestRunner(initial_equity=50_000.0, seed=0)
        report = runner.run_all(
            historical_scenarios=[HistoricalScenario.ATH_CORRECTION_MAR2024],
            synthetic_scenarios=[SyntheticScenario.BROKEN_MODEL],
        )
        s = report.summary()
        assert len(s) > 50
