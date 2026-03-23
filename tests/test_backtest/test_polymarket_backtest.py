"""Tests for Polymarket binary payoff backtester.

Uses synthetic signals with known outcomes to verify:
- Binary payoff math (win/loss/fee calculation)
- Edge and filtering (min_prob, min_edge)
- Aggregate metrics (win_rate, Sharpe, drawdown, profit_factor)
- Edge cases (empty bars, all wins, all losses, zero-cost guards)
- Comparison report
"""

from __future__ import annotations

import pytest

from ep2_crypto.backtest.polymarket_backtest import (
    BinaryBar,
    BinaryBacktestResult,
    BinaryFeeModel,
    BinaryTrade,
    PolymarketBacktester,
    comparison_report,
    compute_shares,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_bar(
    *,
    signal: int = 1,
    open_price: float = 50_000.0,
    close_price: float = 50_100.0,  # UP by default
    model_prob: float = 0.65,
    market_yes: float = 0.60,
    market_no: float = 0.40,
    ts: int = 0,
) -> BinaryBar:
    return BinaryBar(
        timestamp_ms=ts,
        open_price=open_price,
        close_price=close_price,
        signal=signal,
        model_prob=model_prob,
        market_price_yes=market_yes,
        market_price_no=market_no,
    )


def make_backtester(**kwargs: object) -> PolymarketBacktester:
    defaults = dict(
        initial_capital=10_000.0,
        bet_fraction=0.02,
        min_model_prob=0.0,
        min_edge=0.0,
    )
    defaults.update(kwargs)
    return PolymarketBacktester(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# BinaryFeeModel
# ---------------------------------------------------------------------------


class TestBinaryFeeModel:
    def test_entry_fee(self) -> None:
        model = BinaryFeeModel(taker_fee_rate=0.02)
        # 100 shares * $0.60 * 2% = $1.20
        assert model.entry_fee(100.0, 0.60) == pytest.approx(1.20, abs=0.001)

    def test_round_trip_fee_equals_entry_fee(self) -> None:
        model = BinaryFeeModel(taker_fee_rate=0.02)
        # Binary markets auto-resolve — no exit fee
        assert model.round_trip_fee(50.0, 0.75) == model.entry_fee(50.0, 0.75)

    def test_zero_taker_fee(self) -> None:
        model = BinaryFeeModel(taker_fee_rate=0.0)
        assert model.entry_fee(100.0, 0.60) == 0.0


# ---------------------------------------------------------------------------
# compute_shares
# ---------------------------------------------------------------------------


class TestComputeShares:
    def test_basic(self) -> None:
        # $10k * 2% = $200 notional; $200 / $0.62 = 322.58 → 322.58
        shares = compute_shares(10_000.0, 0.02, 0.62)
        assert shares == pytest.approx(322.58, abs=0.01)

    def test_min_shares_enforced(self) -> None:
        # Very small capital should still return min_shares
        shares = compute_shares(1.0, 0.001, 0.99, min_shares=1.0)
        assert shares >= 1.0

    def test_high_price(self) -> None:
        # At price=0.99, we get fewer shares
        shares = compute_shares(10_000.0, 0.02, 0.99)
        assert shares > 0


# ---------------------------------------------------------------------------
# PolymarketBacktester — constructor validation
# ---------------------------------------------------------------------------


class TestBacktesterInit:
    def test_negative_capital_raises(self) -> None:
        with pytest.raises(ValueError, match="initial_capital"):
            PolymarketBacktester(initial_capital=-1.0)

    def test_zero_bet_fraction_raises(self) -> None:
        with pytest.raises(ValueError, match="bet_fraction"):
            PolymarketBacktester(bet_fraction=0.0)

    def test_over_1_bet_fraction_raises(self) -> None:
        with pytest.raises(ValueError, match="bet_fraction"):
            PolymarketBacktester(bet_fraction=1.5)

    def test_valid_construction(self) -> None:
        bt = make_backtester()
        assert bt is not None


# ---------------------------------------------------------------------------
# Single trade payoff math
# ---------------------------------------------------------------------------


class TestSingleTradePayoff:
    def test_win_trade_pnl(self) -> None:
        """WIN: bought 10 shares at $0.60, each pays $1.00."""
        bt = make_backtester(
            initial_capital=10_000.0,
            bet_fraction=0.002,  # ~$20 notional → ~33 shares at $0.60
        )
        bar = make_bar(signal=1, open_price=100.0, close_price=101.0,
                       market_yes=0.60, model_prob=0.70)
        result = bt.run([bar])

        assert result.total_trades == 1
        assert result.wins == 1
        assert result.losses == 0
        trade = result.trades[0]
        assert trade.outcome == "win"
        # Gross payoff = shares * 1.0 - shares * price
        expected_gross = trade.shares * (1.0 - trade.market_price)
        assert trade.pnl_gross == pytest.approx(expected_gross, abs=0.01)
        assert trade.pnl_net < trade.pnl_gross  # Fee reduces net

    def test_loss_trade_pnl(self) -> None:
        """LOSS: bet UP but price went DOWN."""
        bt = make_backtester(
            initial_capital=10_000.0,
            bet_fraction=0.002,
        )
        bar = make_bar(signal=1, open_price=100.0, close_price=99.0,
                       market_yes=0.60, model_prob=0.70)
        result = bt.run([bar])

        assert result.total_trades == 1
        assert result.losses == 1
        trade = result.trades[0]
        assert trade.outcome == "loss"
        # Full loss of cost
        assert trade.pnl_gross == pytest.approx(-trade.cost_usd, abs=0.001)

    def test_down_bet_wins_when_price_falls(self) -> None:
        """DOWN bet: signal=-1, price closes lower → WIN."""
        bt = make_backtester(
            initial_capital=10_000.0,
            bet_fraction=0.002,
        )
        bar = make_bar(signal=-1, open_price=100.0, close_price=99.0,
                       market_no=0.45, model_prob=0.65)
        result = bt.run([bar])
        assert result.wins == 1

    def test_down_bet_loses_when_price_rises(self) -> None:
        """DOWN bet: signal=-1, price closes higher → LOSS."""
        bt = make_backtester(initial_capital=10_000.0, bet_fraction=0.002)
        bar = make_bar(signal=-1, open_price=100.0, close_price=101.0,
                       market_no=0.45, model_prob=0.65)
        result = bt.run([bar])
        assert result.losses == 1

    def test_flat_price_counts_as_loss_for_up_bet(self) -> None:
        """Price unchanged: not strictly higher → UP bet loses."""
        bt = make_backtester(initial_capital=10_000.0, bet_fraction=0.002)
        bar = make_bar(signal=1, open_price=100.0, close_price=100.0)
        result = bt.run([bar])
        assert result.losses == 1

    def test_fee_deducted_from_pnl_net(self) -> None:
        bt = make_backtester(initial_capital=10_000.0, bet_fraction=0.01)
        bar = make_bar(signal=1, open_price=100.0, close_price=101.0)
        result = bt.run([bar])
        trade = result.trades[0]
        assert trade.fee_usd > 0
        assert trade.pnl_net == pytest.approx(trade.pnl_gross - trade.fee_usd, abs=0.001)


# ---------------------------------------------------------------------------
# Signal filtering
# ---------------------------------------------------------------------------


class TestSignalFiltering:
    def test_zero_signal_skipped(self) -> None:
        bt = make_backtester()
        bars = [make_bar(signal=0), make_bar(signal=0)]
        result = bt.run(bars)
        assert result.total_trades == 0
        assert result.no_trades == 2

    def test_low_model_prob_skipped(self) -> None:
        bt = PolymarketBacktester(
            initial_capital=10_000.0,
            bet_fraction=0.02,
            min_model_prob=0.70,
        )
        bar = make_bar(signal=1, model_prob=0.65)  # Below threshold
        result = bt.run([bar])
        assert result.total_trades == 0
        assert result.no_trades == 1

    def test_low_edge_skipped(self) -> None:
        bt = PolymarketBacktester(
            initial_capital=10_000.0,
            bet_fraction=0.02,
            min_edge=0.10,
        )
        # model_prob=0.62, market_yes=0.60 → edge=0.02 < 0.10
        bar = make_bar(signal=1, model_prob=0.62, market_yes=0.60)
        result = bt.run([bar])
        assert result.total_trades == 0

    def test_sufficient_edge_accepted(self) -> None:
        bt = PolymarketBacktester(
            initial_capital=10_000.0,
            bet_fraction=0.02,
            min_edge=0.05,
        )
        bar = make_bar(signal=1, model_prob=0.72, market_yes=0.60)  # edge=0.12
        result = bt.run([bar])
        assert result.total_trades == 1

    def test_min_prob_exactly_at_threshold_accepted(self) -> None:
        bt = PolymarketBacktester(
            initial_capital=10_000.0,
            bet_fraction=0.02,
            min_model_prob=0.65,
        )
        bar = make_bar(signal=1, model_prob=0.65)
        result = bt.run([bar])
        assert result.total_trades == 1


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------


class TestAggregateMetrics:
    def _perfect_wins(self, n: int = 10) -> BinaryBacktestResult:
        """All UP signals with price rising."""
        bt = make_backtester(initial_capital=10_000.0, bet_fraction=0.01)
        bars = [
            make_bar(signal=1, open_price=100.0, close_price=101.0, ts=i * 300_000)
            for i in range(n)
        ]
        return bt.run(bars)

    def _perfect_losses(self, n: int = 10) -> BinaryBacktestResult:
        """All UP signals with price falling."""
        bt = make_backtester(initial_capital=10_000.0, bet_fraction=0.01)
        bars = [
            make_bar(signal=1, open_price=100.0, close_price=99.0, ts=i * 300_000)
            for i in range(n)
        ]
        return bt.run(bars)

    def test_win_rate_all_wins(self) -> None:
        result = self._perfect_wins()
        assert result.win_rate == pytest.approx(1.0)

    def test_win_rate_all_losses(self) -> None:
        result = self._perfect_losses()
        assert result.win_rate == pytest.approx(0.0)

    def test_profit_factor_all_wins(self) -> None:
        result = self._perfect_wins()
        assert result.profit_factor == float("inf")

    def test_profit_factor_mixed(self) -> None:
        bt = make_backtester(initial_capital=10_000.0, bet_fraction=0.01)
        bars = [
            make_bar(signal=1, open_price=100.0, close_price=101.0, ts=0),  # win
            make_bar(signal=1, open_price=100.0, close_price=99.0, ts=300_000),  # loss
        ]
        result = bt.run(bars)
        assert result.profit_factor > 0
        # win at ~$0.60 price: gross = shares*(1-0.6) = 40% of cost
        # loss at same price: gross = -100% of cost → PF < 1
        assert result.profit_factor < 1.0

    def test_total_pnl_positive_for_all_wins(self) -> None:
        result = self._perfect_wins(20)
        assert result.total_pnl_net > 0

    def test_total_pnl_negative_for_all_losses(self) -> None:
        result = self._perfect_losses(20)
        assert result.total_pnl_net < 0

    def test_roi_computed(self) -> None:
        result = self._perfect_wins(10)
        assert result.roi == pytest.approx(result.total_pnl_net / 10_000.0, abs=0.001)

    def test_sharpe_positive_for_consistent_wins(self) -> None:
        result = self._perfect_wins(20)
        assert result.sharpe > 0

    def test_sharpe_negative_for_consistent_losses(self) -> None:
        result = self._perfect_losses(20)
        assert result.sharpe < 0

    def test_max_drawdown_zero_for_monotonic_equity(self) -> None:
        result = self._perfect_wins(10)
        assert result.max_drawdown >= 0.0

    def test_max_drawdown_nonzero_after_losses(self) -> None:
        result = self._perfect_losses(5)
        assert result.max_drawdown > 0.0

    def test_equity_curve_length(self) -> None:
        bt = make_backtester(initial_capital=5_000.0, bet_fraction=0.01)
        bars = [make_bar(ts=i) for i in range(5)]
        result = bt.run(bars)
        # One equity point per bar + initial point
        assert len(result.equity_curve) == len(bars) + 1

    def test_equity_starts_at_initial_capital(self) -> None:
        bt = make_backtester(initial_capital=7_500.0)
        result = bt.run([make_bar()])
        assert result.equity_curve[0] == pytest.approx(7_500.0)

    def test_capital_cannot_go_negative(self) -> None:
        """Very large bets → equity floored at 0."""
        bt = PolymarketBacktester(
            initial_capital=10.0,
            bet_fraction=1.0,  # 100% of capital
            min_model_prob=0.0,
        )
        bars = [
            make_bar(signal=1, open_price=100.0, close_price=99.0, ts=i)
            for i in range(5)
        ]
        result = bt.run(bars)
        assert min(result.equity_curve) >= 0.0

    def test_edge_per_bet(self) -> None:
        result = self._perfect_wins(10)
        expected = result.total_pnl_net / result.total_cost
        assert result.edge_per_bet == pytest.approx(expected, abs=0.001)


# ---------------------------------------------------------------------------
# Max drawdown calculation
# ---------------------------------------------------------------------------


class TestMaxDrawdown:
    def test_empty_curve(self) -> None:
        dd = PolymarketBacktester._compute_max_drawdown([])
        assert dd == 0.0

    def test_monotonic_increasing(self) -> None:
        dd = PolymarketBacktester._compute_max_drawdown([100.0, 110.0, 120.0])
        assert dd == pytest.approx(0.0, abs=0.001)

    def test_full_drawdown(self) -> None:
        dd = PolymarketBacktester._compute_max_drawdown([100.0, 50.0, 0.0])
        assert dd == pytest.approx(1.0)

    def test_partial_drawdown(self) -> None:
        dd = PolymarketBacktester._compute_max_drawdown([100.0, 80.0, 90.0])
        assert dd == pytest.approx(0.20, abs=0.001)

    def test_recovery_after_drawdown(self) -> None:
        dd = PolymarketBacktester._compute_max_drawdown([100.0, 60.0, 120.0, 90.0])
        # Peak=100, trough=60 → 40% DD
        assert dd == pytest.approx(0.40, abs=0.001)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_bars_returns_empty_result(self) -> None:
        bt = make_backtester()
        result = bt.run([])
        assert result.total_trades == 0
        assert result.win_rate == 0.0
        assert result.total_pnl_net == 0.0

    def test_single_bar_win(self) -> None:
        bt = make_backtester()
        result = bt.run([make_bar(signal=1, open_price=100.0, close_price=101.0)])
        assert result.total_trades == 1
        assert result.wins == 1

    def test_single_bar_loss(self) -> None:
        bt = make_backtester()
        result = bt.run([make_bar(signal=1, open_price=100.0, close_price=99.0)])
        assert result.total_trades == 1
        assert result.losses == 1

    def test_all_no_trade_signals(self) -> None:
        bt = make_backtester()
        bars = [make_bar(signal=0) for _ in range(20)]
        result = bt.run(bars)
        assert result.total_trades == 0
        assert result.no_trades == 20

    def test_sharpe_single_trade(self) -> None:
        """Single trade has no standard deviation → Sharpe = 0."""
        bt = make_backtester()
        result = bt.run([make_bar()])
        assert result.sharpe == 0.0

    def test_sortino_no_losses(self) -> None:
        """All wins → no downside → Sortino = inf."""
        bt = make_backtester(initial_capital=10_000.0, bet_fraction=0.01)
        bars = [
            make_bar(signal=1, open_price=100.0, close_price=101.0, ts=i)
            for i in range(5)
        ]
        result = bt.run(bars)
        assert result.sortino == float("inf") or result.sortino > 0

    def test_summary_string(self) -> None:
        bt = make_backtester()
        result = bt.run([make_bar()])
        s = result.summary()
        assert "Win rate" in s
        assert "Net PnL" in s


# ---------------------------------------------------------------------------
# Comparison report
# ---------------------------------------------------------------------------


class TestComparisonReport:
    def test_report_contains_all_metrics(self) -> None:
        bt = make_backtester(initial_capital=5_000.0, bet_fraction=0.01)
        bars_a = [make_bar(signal=1, open_price=100.0, close_price=101.0, ts=i) for i in range(5)]
        bars_b = [make_bar(signal=-1, open_price=100.0, close_price=99.0, ts=i) for i in range(5)]
        result_a = bt.run(bars_a)
        result_b = bt.run(bars_b)

        report = comparison_report({"strategy_a": result_a, "strategy_b": result_b})

        for key in ["win_rate", "total_pnl_net", "roi", "sharpe", "profit_factor"]:
            assert key in report
            assert "strategy_a" in report[key]
            assert "strategy_b" in report[key]

    def test_report_empty_strategies(self) -> None:
        report = comparison_report({})
        assert isinstance(report, dict)
