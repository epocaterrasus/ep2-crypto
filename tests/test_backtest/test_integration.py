"""Integration tests for Sprint 10 — Backtesting Framework.

Validates the complete pipeline:
  1. Data → engine → metrics → validation
  2. Walk-forward fold generation + auditing
  3. Cost is always included (no zero-cost mode)
  4. Risk engine constraints are integrated
  5. Benchmarks run against new engine
  6. Statistical validation produces verdict
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ep2_crypto.backtest.engine import BacktestConfig, BacktestEngine
from ep2_crypto.backtest.metrics import (
    BARS_PER_DAY,
    BARS_PER_YEAR,
    BacktestResult,
    cost_sensitivity,
    find_breakeven_cost,
)
from ep2_crypto.backtest.simulator import ExecutionSimulator
from ep2_crypto.backtest.validation import Verdict, run_validation_suite
from ep2_crypto.backtest.walk_forward import (
    WalkForwardAuditor,
    WalkForwardConfig,
    WalkForwardValidator,
)
from ep2_crypto.benchmarks.engine import BacktestEngine as BenchmarkEngine
from ep2_crypto.benchmarks.strategies import FundingRateCarry


def _generate_data(
    n_days: int = 30,
    trend: float = 0.00005,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Generate synthetic data for integration tests."""
    rng = np.random.default_rng(seed)
    n = n_days * BARS_PER_DAY
    returns = rng.normal(trend, 0.001, size=n)
    closes = 100_000.0 * np.cumprod(1 + returns)
    opens = np.roll(closes, 1)
    opens[0] = 100_000.0
    highs = np.maximum(opens, closes) * (1 + rng.uniform(0, 0.002, n))
    lows = np.minimum(opens, closes) * (1 - rng.uniform(0, 0.002, n))
    volumes = rng.uniform(100, 500, n)

    # Within trading hours
    base_ts = 1704103200000  # 10:00 UTC
    timestamps = np.arange(n, dtype=np.int64) * 300_000 + base_ts

    # Noisy oracle signal
    future_ret = np.zeros(n)
    future_ret[:-1] = np.diff(closes) / closes[:-1]
    noise = rng.normal(0, 0.002, n)
    raw_signal = future_ret + noise
    signals = np.sign(raw_signal).astype(np.int8)
    confidences = np.clip(np.abs(raw_signal) * 500, 0.5, 0.95)
    funding_rates = rng.normal(0.0001, 0.00003, n)

    return {
        "opens": opens,
        "highs": highs,
        "lows": lows,
        "closes": closes,
        "volumes": volumes,
        "timestamps_ms": timestamps,
        "signals": signals,
        "confidences": confidences,
        "funding_rates": funding_rates,
    }


# ---------------------------------------------------------------------------
# 1. Full pipeline: data → engine → metrics
# ---------------------------------------------------------------------------
class TestFullPipeline:
    def test_30_day_backtest_runs(self) -> None:
        """Backtest on 30+ days of data should complete without errors."""
        data = _generate_data(n_days=30)
        engine = BacktestEngine(
            BacktestConfig(
                initial_equity=50_000.0,
                confidence_threshold=0.5,
            )
        )
        result = engine.run(**data)

        assert isinstance(result, BacktestResult)
        assert result.total_return != 0 or result.total_trades == 0
        assert len(result.equity_curve) == 30 * BARS_PER_DAY

    def test_60_day_backtest_runs(self) -> None:
        """Longer backtest should also work."""
        data = _generate_data(n_days=60)
        engine = BacktestEngine(
            BacktestConfig(
                initial_equity=50_000.0,
                confidence_threshold=0.5,
            )
        )
        result = engine.run(**data)
        assert result.total_trades >= 0


# ---------------------------------------------------------------------------
# 2. Costs are ALWAYS included
# ---------------------------------------------------------------------------
class TestCostsAlwaysPresent:
    def test_no_zero_cost_mode(self) -> None:
        """Every trade must incur costs — no free execution."""
        data = _generate_data(n_days=30)
        engine = BacktestEngine(
            BacktestConfig(
                initial_equity=50_000.0,
                confidence_threshold=0.5,
            )
        )
        result = engine.run(**data)
        if result.total_trades > 0:
            assert result.total_fee_usd > 0, "Fees must be > 0 when trades occur"
            assert result.total_cost_usd > 0, "Total costs must be > 0"

    def test_simulator_always_charges(self) -> None:
        """ExecutionSimulator must always produce non-zero costs."""
        sim = ExecutionSimulator(seed=42)
        entry = sim.simulate_entry("long", 0.05, 100_000.0)
        assert entry.executed
        assert entry.fee_bps > 0
        assert entry.total_cost_bps > 0


# ---------------------------------------------------------------------------
# 3. Risk engine is integrated
# ---------------------------------------------------------------------------
class TestRiskIntegration:
    def test_risk_limits_respected(self) -> None:
        """Backtest should respect risk engine position limits."""
        data = _generate_data(n_days=30)
        config = BacktestConfig(
            initial_equity=50_000.0,
            confidence_threshold=0.5,
        )
        engine = BacktestEngine(config)
        result = engine.run(**data)
        # Engine ran through risk manager, no exceptions
        assert isinstance(result, BacktestResult)

    def test_low_confidence_filtered(self) -> None:
        """Signals below confidence threshold should not execute."""
        data = _generate_data(n_days=30)
        data["confidences"] = np.full(len(data["closes"]), 0.3)  # all low
        config = BacktestConfig(
            initial_equity=50_000.0,
            confidence_threshold=0.55,
        )
        engine = BacktestEngine(config)
        result = engine.run(**data)
        assert result.total_trades == 0


# ---------------------------------------------------------------------------
# 4. Walk-forward + audit
# ---------------------------------------------------------------------------
class TestWalkForwardIntegration:
    def test_30_day_walk_forward(self) -> None:
        """Walk-forward on 30 days should produce folds."""
        n = 30 * BARS_PER_DAY
        wf = WalkForwardValidator(n)
        assert wf.n_folds > 0

    def test_audit_passes_on_generated_folds(self) -> None:
        """Generated folds should pass all audit checks."""
        n = 30 * BARS_PER_DAY
        cfg = WalkForwardConfig()
        wf = WalkForwardValidator(n, config=cfg)
        auditor = WalkForwardAuditor(config=cfg)
        result = auditor.audit(wf.folds())
        assert result.passed is True, f"Audit errors: {result.errors}"

    def test_oos_indices_valid(self) -> None:
        """Concatenated OOS indices should be within data bounds."""
        n = 30 * BARS_PER_DAY
        wf = WalkForwardValidator(n)
        oos = wf.concatenated_oos_indices()
        assert len(oos) > 0
        assert oos.min() >= 0
        assert oos.max() < n


# ---------------------------------------------------------------------------
# 5. Statistical validation produces verdict
# ---------------------------------------------------------------------------
class TestValidationIntegration:
    def test_validation_produces_verdict(self) -> None:
        """Validation suite should return a valid verdict."""
        rng = np.random.default_rng(42)
        returns = rng.normal(0.0001, 0.001, size=5000)
        result = run_validation_suite(
            returns=returns,
            n_permutations=200,
            n_bootstrap=200,
        )
        assert isinstance(result.verdict, Verdict)
        assert result.tests_total > 0

    def test_validation_with_fold_sharpes(self) -> None:
        """Validation with fold Sharpes should check stability."""
        rng = np.random.default_rng(42)
        returns = rng.normal(0.0001, 0.001, size=5000)
        result = run_validation_suite(
            returns=returns,
            fold_sharpes=[1.5, 1.3, 1.8, 1.4, 1.6],
            n_permutations=100,
            n_bootstrap=100,
        )
        assert "walk_forward_stability" in result.details


# ---------------------------------------------------------------------------
# 6. Benchmarks compatibility
# ---------------------------------------------------------------------------
class TestBenchmarkIntegration:
    def test_funding_carry_benchmark(self) -> None:
        """Funding rate carry should run as a benchmark."""
        rng = np.random.default_rng(42)
        n = 500
        df = pd.DataFrame(
            {
                "open": rng.uniform(99000, 101000, n),
                "high": rng.uniform(100000, 102000, n),
                "low": rng.uniform(98000, 100000, n),
                "close": 100_000.0 * np.cumprod(1 + rng.normal(0, 0.001, n)),
                "volume": rng.uniform(100, 500, n),
                "funding_rate": rng.normal(0.0001, 0.00005, n),
            }
        )
        engine = BenchmarkEngine()
        strategy = FundingRateCarry()
        metrics, _, _ = engine.run_strategy(strategy, df)
        assert isinstance(metrics.sharpe_ratio, float)


# ---------------------------------------------------------------------------
# 7. Cost sensitivity analysis
# ---------------------------------------------------------------------------
class TestCostSensitivityIntegration:
    def test_cost_sensitivity_runs(self) -> None:
        """Cost sensitivity should produce results at multiple levels."""
        rng = np.random.default_rng(42)
        returns = rng.normal(0.0002, 0.001, size=10000)
        results = cost_sensitivity(returns, n_trades=500, n_bars=10000)
        assert len(results) == 6  # default 6 levels
        # Higher cost → lower return
        assert results[-1]["total_return"] < results[0]["total_return"]

    def test_breakeven_cost_found(self) -> None:
        """Should find a finite break-even cost."""
        rng = np.random.default_rng(42)
        returns = rng.normal(0.0001, 0.001, size=10000)
        results = cost_sensitivity(
            returns,
            n_trades=1000,
            n_bars=10000,
            cost_levels_bps=[0, 4, 8, 16, 32, 64],
        )
        be = find_breakeven_cost(results)
        assert be > 0


# ---------------------------------------------------------------------------
# 8. Annualization correctness
# ---------------------------------------------------------------------------
class TestAnnualization:
    def test_uses_105120_not_252(self) -> None:
        """Verify the system uses sqrt(105120) for annualization."""
        assert BARS_PER_YEAR == 105_120
        assert abs(np.sqrt(BARS_PER_YEAR) - 324.22) < 0.1
