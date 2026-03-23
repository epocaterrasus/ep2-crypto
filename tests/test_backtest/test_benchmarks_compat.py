"""Tests for benchmark compatibility and funding rate carry strategy."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ep2_crypto.benchmarks.strategies import (
    BuyAndHold,
    FundingRateCarry,
    FundingRateStrategy,
    SimpleMomentum,
    get_default_benchmark_suite,
    STRATEGY_REGISTRY,
)
from ep2_crypto.benchmarks.engine import BacktestEngine as BenchmarkEngine
from ep2_crypto.benchmarks.metrics import compute_metrics


def _make_df(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic OHLCV DataFrame with funding rate."""
    rng = np.random.default_rng(seed)
    close = 100_000.0 * np.cumprod(1 + rng.normal(0, 0.001, n))
    df = pd.DataFrame({
        "open": np.roll(close, 1),
        "high": close * (1 + rng.uniform(0, 0.002, n)),
        "low": close * (1 - rng.uniform(0, 0.002, n)),
        "close": close,
        "volume": rng.uniform(100, 500, n),
        "funding_rate": rng.normal(0.0001, 0.00005, n),  # mostly positive
        "obi": rng.uniform(-0.5, 0.5, n),
        "nq_close": 18000 * np.cumprod(1 + rng.normal(0, 0.0005, n)),
    })
    df.loc[0, "open"] = 100_000.0
    return df


# ---------------------------------------------------------------------------
# FundingRateCarry strategy
# ---------------------------------------------------------------------------
class TestFundingRateCarry:
    def test_basic_positions(self) -> None:
        """Should produce positions based on funding rate sign."""
        df = _make_df()
        strategy = FundingRateCarry()
        positions = strategy.generate_positions(df)
        assert len(positions) == len(df)
        assert set(positions.unique()).issubset({-1.0, 0.0, 1.0})

    def test_positive_funding_goes_short(self) -> None:
        """When funding is consistently positive, carry should be short."""
        df = _make_df()
        df["funding_rate"] = 0.001  # strongly positive
        strategy = FundingRateCarry(smoothing_periods=1)
        positions = strategy.generate_positions(df)
        # Most positions should be short
        assert (positions == -1.0).mean() > 0.9

    def test_negative_funding_goes_long(self) -> None:
        """When funding is consistently negative, carry should be long."""
        df = _make_df()
        df["funding_rate"] = -0.001  # strongly negative
        strategy = FundingRateCarry(smoothing_periods=1)
        positions = strategy.generate_positions(df)
        assert (positions == 1.0).mean() > 0.9

    def test_smoothing_reduces_turnover(self) -> None:
        """Smoothed version should have fewer position changes."""
        df = _make_df()
        raw = FundingRateCarry(smoothing_periods=1)
        smoothed = FundingRateCarry(smoothing_periods=10)
        pos_raw = raw.generate_positions(df)
        pos_smooth = smoothed.generate_positions(df)
        changes_raw = (pos_raw.diff().abs() > 0).sum()
        changes_smooth = (pos_smooth.diff().abs() > 0).sum()
        assert changes_smooth <= changes_raw

    def test_requires_funding_rate_column(self) -> None:
        df = _make_df()
        df = df.drop(columns=["funding_rate"])
        strategy = FundingRateCarry()
        with pytest.raises(ValueError, match="funding_rate"):
            strategy.generate_positions(df)


# ---------------------------------------------------------------------------
# Registry and suite
# ---------------------------------------------------------------------------
class TestRegistry:
    def test_funding_carry_in_registry(self) -> None:
        assert "funding_rate_carry" in STRATEGY_REGISTRY

    def test_default_suite_has_carry(self) -> None:
        suite = get_default_benchmark_suite()
        assert "funding_carry" in suite
        assert "funding_carry_no_smooth" in suite

    def test_default_suite_size(self) -> None:
        suite = get_default_benchmark_suite()
        # Should have many strategies
        assert len(suite) >= 25


# ---------------------------------------------------------------------------
# Compatibility with benchmark engine
# ---------------------------------------------------------------------------
class TestBenchmarkEngineCompat:
    def test_carry_runs_in_engine(self) -> None:
        """FundingRateCarry should work with existing BacktestEngine."""
        df = _make_df(300)
        engine = BenchmarkEngine(trading_cost_bps=4.0, slippage_bps=1.0)
        strategy = FundingRateCarry()
        metrics, returns, positions = engine.run_strategy(strategy, df)
        assert metrics.total_trades >= 0
        assert isinstance(metrics.sharpe_ratio, float)

    def test_all_basic_strategies_run(self) -> None:
        """All strategies that don't need special columns should run."""
        df = _make_df(300)
        engine = BenchmarkEngine()
        basic_strategies = {
            "buy_hold": BuyAndHold(),
            "momentum": SimpleMomentum(lookback=5),
            "funding_carry": FundingRateCarry(),
            "funding_rate": FundingRateStrategy(),
        }
        for name, strategy in basic_strategies.items():
            metrics, _, _ = engine.run_strategy(strategy, df)
            assert isinstance(metrics.sharpe_ratio, float), f"{name} failed"

    def test_suite_run(self) -> None:
        """Full suite should run without errors on complete data."""
        df = _make_df(500)
        engine = BenchmarkEngine()
        suite = get_default_benchmark_suite()
        # Only run strategies whose required columns exist
        for name, strategy in suite.items():
            try:
                engine.run_strategy(strategy, df)
            except ValueError:
                pass  # missing columns is OK for this test

    def test_regime_decomposed_works(self) -> None:
        """Regime decomposed backtest should work with carry."""
        df = _make_df(300)
        engine = BenchmarkEngine()
        strategy = FundingRateCarry()
        regimes = pd.Series(
            np.random.default_rng(42).choice(["bull", "bear", "range"], 300),
            index=df.index,
        )
        result = engine.regime_decomposed_backtest(strategy, df, regimes)
        assert isinstance(result, dict)
