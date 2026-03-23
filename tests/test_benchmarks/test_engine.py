"""Tests for the backtest engine."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ep2_crypto.benchmarks.data import generate_synthetic_btc
from ep2_crypto.benchmarks.engine import BacktestEngine
from ep2_crypto.benchmarks.strategies import (
    BuyAndHold,
    OracleStrategy,
    SimpleMomentum,
    get_default_benchmark_suite,
)


@pytest.fixture
def engine() -> BacktestEngine:
    return BacktestEngine(trading_cost_bps=3.0, slippage_bps=1.0)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return generate_synthetic_btc(n_bars=2000, seed=42)


class TestBacktestEngine:
    def test_run_strategy(self, engine: BacktestEngine, sample_df: pd.DataFrame) -> None:
        metrics, returns, positions = engine.run_strategy(BuyAndHold(), sample_df)
        assert len(returns) == len(sample_df)
        assert len(positions) == len(sample_df)
        assert hasattr(metrics, "sharpe_ratio")

    def test_oracle_beats_all(self, sample_df: pd.DataFrame) -> None:
        """Oracle with perfect foresight should have positive Sharpe at zero cost.

        With realistic costs, oracle may have negative Sharpe because it
        changes position every bar (high turnover). This is a real insight:
        even perfect foresight is limited by execution costs.
        """
        zero_cost_engine = BacktestEngine(trading_cost_bps=0.0, slippage_bps=0.0)
        metrics, _, _ = zero_cost_engine.run_strategy(OracleStrategy(), sample_df)
        assert metrics.sharpe_ratio > 0

    def test_run_suite(self, engine: BacktestEngine, sample_df: pd.DataFrame) -> None:
        suite = {"bh": BuyAndHold(), "mom5": SimpleMomentum(5)}
        results = engine.run_suite(sample_df, suite)
        assert "bh" in results
        assert "mom5" in results
        assert "metrics" in results["bh"]

    def test_results_to_dataframe(self, engine: BacktestEngine, sample_df: pd.DataFrame) -> None:
        suite = {"bh": BuyAndHold(), "mom5": SimpleMomentum(5)}
        results = engine.run_suite(sample_df, suite)
        df = engine.results_to_dataframe(results)
        assert "sharpe_ratio" in df.columns
        assert len(df) == 2

    def test_random_distribution(self, engine: BacktestEngine, sample_df: pd.DataFrame) -> None:
        dist = engine.run_random_distribution(sample_df, n_simulations=50, hold_bars=3)
        assert "sharpe_distribution" in dist
        assert len(dist["sharpe_distribution"]) == 50
        assert "p95_sharpe" in dist
        assert "p99_sharpe" in dist

    def test_compare_to_random(self, engine: BacktestEngine, sample_df: pd.DataFrame) -> None:
        dist = engine.run_random_distribution(sample_df, n_simulations=50)
        result = engine.compare_to_random(5.0, dist)  # Very high Sharpe
        assert result["significant_at_01"]

    def test_regime_decomposed(self, engine: BacktestEngine, sample_df: pd.DataFrame) -> None:
        # Create simple regime labels
        regimes = pd.Series("normal", index=sample_df.index)
        regimes.iloc[:500] = "trending"
        regimes.iloc[500:1000] = "ranging"

        strategy = SimpleMomentum(lookback=5)
        results = engine.regime_decomposed_backtest(strategy, sample_df, regimes)
        assert "trending" in results
        assert "ranging" in results
        assert "normal" in results
