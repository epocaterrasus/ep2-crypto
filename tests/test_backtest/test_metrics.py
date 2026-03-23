"""Tests for backtest metrics module."""

from __future__ import annotations

import numpy as np
import pytest

from ep2_crypto.backtest.metrics import (
    BARS_PER_YEAR,
    SQRT_BARS_PER_YEAR,
    BacktestResult,
    TradeRecord,
    analyze_trades,
    compute_backtest_result,
    cost_sensitivity,
    cvar,
    find_breakeven_cost,
    lo_corrected_sharpe,
    max_drawdown_info,
    regime_metrics,
    rolling_sharpe,
    sortino_ratio,
)


# ---------------------------------------------------------------------------
# Constants verification
# ---------------------------------------------------------------------------
class TestConstants:
    def test_bars_per_year(self) -> None:
        assert BARS_PER_YEAR == 288 * 365  # 105,120

    def test_sqrt_bars_per_year(self) -> None:
        expected = np.sqrt(105_120)
        assert abs(SQRT_BARS_PER_YEAR - expected) < 0.01
        assert abs(SQRT_BARS_PER_YEAR - 324.22) < 0.1

    def test_never_use_252(self) -> None:
        """Verify annualization uses 105,120 not 252."""
        assert BARS_PER_YEAR != 252
        assert SQRT_BARS_PER_YEAR > 300  # sqrt(252) ≈ 15.87


# ---------------------------------------------------------------------------
# Lo-corrected Sharpe
# ---------------------------------------------------------------------------
class TestLoSharpe:
    def test_zero_returns(self) -> None:
        returns = np.zeros(1000)
        assert lo_corrected_sharpe(returns) == 0.0

    def test_constant_positive_returns(self) -> None:
        """Constant returns → zero std → zero Sharpe."""
        returns = np.full(1000, 0.0001)
        assert lo_corrected_sharpe(returns) == 0.0

    def test_positive_edge_gives_positive_sharpe(self) -> None:
        rng = np.random.default_rng(42)
        # Mean = 0.0001, std = 0.001 → raw Sharpe ≈ 0.1 * sqrt(105120)
        returns = rng.normal(0.0001, 0.001, size=5000)
        sharpe = lo_corrected_sharpe(returns)
        assert sharpe > 0

    def test_negative_edge_gives_negative_sharpe(self) -> None:
        rng = np.random.default_rng(42)
        returns = rng.normal(-0.0001, 0.001, size=5000)
        sharpe = lo_corrected_sharpe(returns)
        assert sharpe < 0

    def test_autocorrelated_returns_lower_sharpe(self) -> None:
        """Autocorrelated returns should have LOWER Lo-corrected Sharpe than raw."""
        rng = np.random.default_rng(42)
        n = 10_000
        noise = rng.normal(0.0001, 0.001, size=n)
        # Add positive autocorrelation
        returns = np.zeros(n)
        returns[0] = noise[0]
        for i in range(1, n):
            returns[i] = 0.3 * returns[i - 1] + noise[i]

        lo_sharpe = lo_corrected_sharpe(returns, max_lag=6)
        # Raw sharpe (no correction)
        mean_r = returns.mean()
        std_r = returns.std(ddof=1)
        raw_sharpe = mean_r / std_r * SQRT_BARS_PER_YEAR

        # Lo-correction should reduce the Sharpe for positively autocorrelated returns
        assert abs(lo_sharpe) <= abs(raw_sharpe) * 1.1  # allow small tolerance

    def test_short_series_fallback(self) -> None:
        """Very short series should fall back to raw Sharpe."""
        returns = np.array([0.001, -0.0005, 0.002])
        sharpe = lo_corrected_sharpe(returns, max_lag=6)
        assert isinstance(sharpe, float)


# ---------------------------------------------------------------------------
# Sortino
# ---------------------------------------------------------------------------
class TestSortino:
    def test_all_positive_returns(self) -> None:
        """All positive returns → capped Sortino (no downside)."""
        returns = np.array([0.001, 0.002, 0.001, 0.003] * 100)
        sort = sortino_ratio(returns)
        assert sort == 10.0  # capped

    def test_mixed_returns(self) -> None:
        rng = np.random.default_rng(42)
        returns = rng.normal(0.0001, 0.001, size=5000)
        sort = sortino_ratio(returns)
        assert sort > 0

    def test_empty_returns(self) -> None:
        assert sortino_ratio(np.array([])) == 0.0


# ---------------------------------------------------------------------------
# CVaR
# ---------------------------------------------------------------------------
class TestCVaR:
    def test_known_distribution(self) -> None:
        """Uniform returns [0.01, ..., 1.00] → worst 5% = mean of [0.01..0.05]."""
        returns = np.arange(1, 101) / 100.0
        cv = cvar(returns, alpha=0.05)
        # Worst 5% = indices 0-4 → values 0.01..0.05 → mean = 0.03
        assert abs(cv - 0.03) < 0.001

    def test_negative_returns_cvar(self) -> None:
        """CVaR of negative returns should be negative."""
        rng = np.random.default_rng(42)
        returns = rng.normal(-0.001, 0.01, size=1000)
        cv = cvar(returns)
        assert cv < 0

    def test_empty_returns(self) -> None:
        assert cvar(np.array([])) == 0.0


# ---------------------------------------------------------------------------
# Max Drawdown
# ---------------------------------------------------------------------------
class TestMaxDrawdown:
    def test_monotonically_increasing(self) -> None:
        """No drawdown if equity only goes up."""
        equity = np.linspace(1.0, 2.0, 100)
        max_dd, duration, avg_dd = max_drawdown_info(equity)
        assert max_dd == 0.0
        assert duration == 0

    def test_known_drawdown(self) -> None:
        """Equity 1.0 → 1.5 → 1.2 → 1.6: drawdown = (1.5-1.2)/1.5 = 20%."""
        equity = np.array([1.0, 1.25, 1.5, 1.3, 1.2, 1.4, 1.6])
        max_dd, duration, avg_dd = max_drawdown_info(equity)
        assert abs(max_dd - 0.2) < 0.001
        # Duration: bars 3,4 are in drawdown from peak at bar 2
        assert duration >= 2

    def test_total_wipeout(self) -> None:
        equity = np.array([1.0, 0.8, 0.5, 0.2, 0.1])
        max_dd, _, _ = max_drawdown_info(equity)
        assert max_dd > 0.89


# ---------------------------------------------------------------------------
# Rolling Sharpe
# ---------------------------------------------------------------------------
class TestRollingSharpe:
    def test_short_series_all_nan(self) -> None:
        """Series shorter than window should be all NaN."""
        returns = np.random.default_rng(42).normal(0, 0.001, 100)
        rs = rolling_sharpe(returns, window_bars=200)
        assert np.all(np.isnan(rs))

    def test_output_shape(self) -> None:
        returns = np.random.default_rng(42).normal(0, 0.001, 10000)
        rs = rolling_sharpe(returns, window_bars=500)
        assert len(rs) == len(returns)
        # First 499 should be NaN, rest should be numbers
        assert np.all(np.isnan(rs[:499]))
        assert not np.isnan(rs[499])


# ---------------------------------------------------------------------------
# Trade Analysis
# ---------------------------------------------------------------------------
class TestAnalyzeTrades:
    def _make_trade(self, ret: float, bars: int = 3) -> TradeRecord:
        return TradeRecord(
            entry_bar=0,
            exit_bar=bars,
            side="long",
            entry_price=100_000.0,
            exit_price=100_000.0 * (1 + ret),
            quantity=0.01,
            pnl_usd=100_000.0 * 0.01 * ret,
            return_pct=ret,
            bars_held=bars,
        )

    def test_all_winners(self) -> None:
        trades = [self._make_trade(0.001) for _ in range(10)]
        stats = analyze_trades(trades)
        assert stats["win_rate"] == 1.0
        assert stats["profit_factor"] > 0

    def test_all_losers(self) -> None:
        trades = [self._make_trade(-0.001) for _ in range(10)]
        stats = analyze_trades(trades)
        assert stats["win_rate"] == 0.0
        assert stats["profit_factor"] == 0.0

    def test_mixed_trades(self) -> None:
        trades = [
            self._make_trade(0.002),
            self._make_trade(-0.001),
            self._make_trade(0.003),
            self._make_trade(-0.0005),
        ]
        stats = analyze_trades(trades)
        assert stats["win_rate"] == 0.5
        assert stats["profit_factor"] > 1.0  # winners > losers
        assert stats["expectancy_bps"] > 0

    def test_empty_trades(self) -> None:
        stats = analyze_trades([])
        assert stats["win_rate"] == 0.0


# ---------------------------------------------------------------------------
# Regime Metrics
# ---------------------------------------------------------------------------
class TestRegimeMetrics:
    def test_two_regimes(self) -> None:
        rng = np.random.default_rng(42)
        n = 1000
        returns = rng.normal(0.0001, 0.001, size=n)
        labels = np.array([0] * 500 + [1] * 500, dtype=np.int32)
        # Make regime 1 more profitable
        returns[500:] += 0.0002

        rm = regime_metrics(returns, labels)
        assert 0 in rm
        assert 1 in rm
        assert rm[1]["sharpe"] > rm[0]["sharpe"]

    def test_none_labels(self) -> None:
        returns = np.random.default_rng(42).normal(0, 0.001, 100)
        assert regime_metrics(returns, None) == {}


# ---------------------------------------------------------------------------
# Cost Sensitivity
# ---------------------------------------------------------------------------
class TestCostSensitivity:
    def test_higher_cost_lower_sharpe(self) -> None:
        rng = np.random.default_rng(42)
        returns = rng.normal(0.0002, 0.001, size=10000)
        results = cost_sensitivity(returns, n_trades=500, n_bars=10000)
        sharpes = [r["sharpe"] for r in results]
        # Should be monotonically decreasing
        for i in range(1, len(sharpes)):
            assert sharpes[i] <= sharpes[i - 1] + 0.1  # small tolerance

    def test_breakeven_cost(self) -> None:
        """Should find a finite break-even cost for a profitable strategy."""
        rng = np.random.default_rng(42)
        returns = rng.normal(0.0001, 0.001, size=10000)
        results = cost_sensitivity(
            returns, n_trades=1000, n_bars=10000,
            cost_levels_bps=[0, 2, 4, 8, 16, 32, 64],
        )
        be = find_breakeven_cost(results)
        assert be > 0
        assert be < 100  # should be finite

    def test_always_positive_sharpe(self) -> None:
        """Very profitable strategy → break-even = inf at tested levels."""
        rng = np.random.default_rng(42)
        # Strong edge with small noise → all cost levels still positive
        returns = rng.normal(0.01, 0.001, size=10000)
        results = cost_sensitivity(returns, n_trades=10, n_bars=10000)
        be = find_breakeven_cost(results)
        assert be == float("inf")


# ---------------------------------------------------------------------------
# Compute Backtest Result (integration)
# ---------------------------------------------------------------------------
class TestComputeBacktestResult:
    def test_basic_computation(self) -> None:
        rng = np.random.default_rng(42)
        returns = rng.normal(0.0001, 0.001, size=5000)
        result = compute_backtest_result(returns)

        assert isinstance(result, BacktestResult)
        assert result.sharpe_ratio != 0
        assert result.total_return != 0
        assert result.max_drawdown >= 0
        assert len(result.equity_curve) == 5000

    def test_with_trades(self) -> None:
        rng = np.random.default_rng(42)
        returns = rng.normal(0.0001, 0.001, size=5000)
        trades = [
            TradeRecord(
                entry_bar=i * 10, exit_bar=i * 10 + 5,
                side="long", entry_price=100_000.0,
                exit_price=100_000.0 * (1 + rng.normal(0.001, 0.005)),
                quantity=0.01, pnl_usd=rng.normal(5, 50),
                return_pct=rng.normal(0.001, 0.005), bars_held=5,
            )
            for i in range(100)
        ]
        result = compute_backtest_result(returns, trades=trades)
        assert result.total_trades == 100
        assert result.win_rate > 0

    def test_with_costs(self) -> None:
        returns = np.random.default_rng(42).normal(0.0001, 0.001, size=5000)
        result = compute_backtest_result(
            returns,
            total_fee_usd=1000.0,
            total_slippage_usd=500.0,
            total_funding_usd=200.0,
        )
        assert result.total_cost_usd == 1700.0
        assert result.total_fee_usd == 1000.0

    def test_empty_returns(self) -> None:
        result = compute_backtest_result(np.array([]))
        assert result.total_trades == 0
        assert result.sharpe_ratio == 0.0

    def test_summary_string(self) -> None:
        returns = np.random.default_rng(42).normal(0.0001, 0.001, size=5000)
        result = compute_backtest_result(returns)
        summary = result.summary()
        assert "Sharpe" in summary
        assert "Sortino" in summary
        assert "CVaR" in summary

    def test_to_dict_excludes_arrays(self) -> None:
        returns = np.random.default_rng(42).normal(0.0001, 0.001, size=100)
        result = compute_backtest_result(returns)
        d = result.to_dict()
        assert "equity_curve" not in d
        assert "rolling_sharpe_30d" not in d
        assert "sharpe_ratio" in d
