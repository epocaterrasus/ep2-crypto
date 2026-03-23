"""Tests for metrics computation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ep2_crypto.benchmarks.metrics import compute_metrics, BARS_PER_DAY


class TestComputeMetrics:
    def test_profitable_strategy(self) -> None:
        """Positive-mean returns with variance should give positive Sharpe."""
        n = 5000
        idx = pd.date_range("2024-01-01", periods=n, freq="5min")
        rng = np.random.default_rng(42)
        # Positive mean with some variance so std > 0
        returns = pd.Series(rng.normal(0.0005, 0.001, n), index=idx)
        positions = pd.Series(np.ones(n), index=idx)
        metrics = compute_metrics(returns, positions, trading_cost_bps=0)
        assert metrics.sharpe_ratio > 0
        assert metrics.total_return > 0

    def test_unprofitable_strategy(self) -> None:
        """Negative-mean returns with variance should give negative Sharpe."""
        n = 5000
        idx = pd.date_range("2024-01-01", periods=n, freq="5min")
        rng = np.random.default_rng(42)
        returns = pd.Series(rng.normal(-0.0005, 0.001, n), index=idx)
        positions = pd.Series(np.ones(n), index=idx)
        metrics = compute_metrics(returns, positions, trading_cost_bps=0)
        assert metrics.sharpe_ratio < 0
        assert metrics.total_return < 0

    def test_flat_position_zero_trades(self) -> None:
        """All-flat positions should produce zero trades."""
        n = 1000
        idx = pd.date_range("2024-01-01", periods=n, freq="5min")
        returns = pd.Series(np.zeros(n), index=idx)
        positions = pd.Series(np.zeros(n), index=idx)
        metrics = compute_metrics(returns, positions)
        assert metrics.total_trades == 0

    def test_trading_costs_reduce_returns(self) -> None:
        """Higher costs should reduce total return."""
        n = 5000
        idx = pd.date_range("2024-01-01", periods=n, freq="5min")
        rng = np.random.default_rng(42)
        returns = pd.Series(rng.normal(0.0001, 0.001, n), index=idx)
        # Frequent position changes to amplify cost effect
        positions = pd.Series(rng.choice([-1.0, 1.0], n), index=idx)

        m_no_cost = compute_metrics(returns, positions, trading_cost_bps=0)
        m_high_cost = compute_metrics(returns, positions, trading_cost_bps=10)
        assert m_no_cost.total_return > m_high_cost.total_return

    def test_empty_returns(self) -> None:
        """Empty input should not crash."""
        metrics = compute_metrics(pd.Series(dtype=float), pd.Series(dtype=float))
        assert metrics.total_trades == 0

    def test_max_drawdown_nonnegative(self) -> None:
        """Max drawdown should always be >= 0."""
        n = 3000
        idx = pd.date_range("2024-01-01", periods=n, freq="5min")
        rng = np.random.default_rng(42)
        returns = pd.Series(rng.normal(0, 0.002, n), index=idx)
        positions = pd.Series(np.ones(n), index=idx)
        metrics = compute_metrics(returns, positions)
        assert metrics.max_drawdown >= 0

    def test_win_rate_bounds(self) -> None:
        """Win rate should be between 0 and 1."""
        n = 3000
        idx = pd.date_range("2024-01-01", periods=n, freq="5min")
        rng = np.random.default_rng(42)
        returns = pd.Series(rng.normal(0, 0.002, n), index=idx)
        positions = pd.Series(rng.choice([-1.0, 0.0, 1.0], n), index=idx)
        metrics = compute_metrics(returns, positions)
        assert 0 <= metrics.win_rate <= 1
