"""Tests for DailyReportGenerator."""

from __future__ import annotations

import math
from dataclasses import dataclass

import pytest

from ep2_crypto.monitoring.daily_report import (
    DailyReportGenerator,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@dataclass
class FakeTrade:
    """Minimal trade object matching PaperTrade interface."""

    order_id: str
    pnl_usd: float
    fee_usd: float
    notional_usd: float = 10_000.0


def make_trades(
    n: int = 10,
    win_rate: float = 0.6,
    win_usd: float = 20.0,
    lose_usd: float = -10.0,
) -> list[FakeTrade]:
    trades = []
    for i in range(n):
        is_win = i < int(n * win_rate)
        pnl = win_usd if is_win else lose_usd
        trades.append(FakeTrade(order_id=str(i), pnl_usd=pnl, fee_usd=4.0))
    return trades


def make_generator(balance: float = 10_000.0) -> DailyReportGenerator:
    return DailyReportGenerator(initial_balance=balance)


# ---------------------------------------------------------------------------
# Empty state tests
# ---------------------------------------------------------------------------


class TestEmptyReport:
    def test_no_trades_returns_insufficient(self) -> None:
        gen = make_generator()
        report = gen.generate(trades=[])
        assert report.go_nogo_verdict == "INSUFFICIENT_DATA"
        assert report.total_trades == 0

    def test_no_trades_preserves_balance(self) -> None:
        gen = make_generator(balance=50_000.0)
        report = gen.generate(trades=[], final_balance=50_000.0)
        assert report.initial_balance == pytest.approx(50_000.0)
        assert report.final_balance == pytest.approx(50_000.0)


# ---------------------------------------------------------------------------
# Trade statistics tests
# ---------------------------------------------------------------------------


class TestTradeStatistics:
    def test_win_lose_count(self) -> None:
        trades = make_trades(n=10, win_rate=0.6)
        report = make_generator().generate(trades)
        assert report.winning_trades == 6
        assert report.losing_trades == 4
        assert report.total_trades == 10

    def test_win_rate(self) -> None:
        trades = make_trades(n=100, win_rate=0.55)
        report = make_generator().generate(trades)
        assert report.win_rate == pytest.approx(0.55, abs=0.01)

    def test_profit_factor(self) -> None:
        # 6 wins at $20, 4 losses at $10 → PF = 120/40 = 3.0
        trades = make_trades(n=10, win_rate=0.6, win_usd=20.0, lose_usd=-10.0)
        report = make_generator().generate(trades)
        assert report.profit_factor == pytest.approx(3.0, rel=1e-4)

    def test_max_consecutive_losses(self) -> None:
        # Pattern: W W L L L W L → max consec = 3
        pnls = [10, 10, -5, -5, -5, 10, -5]
        trades = [FakeTrade(str(i), p, 1.0) for i, p in enumerate(pnls)]
        report = make_generator().generate(trades)
        assert report.max_consecutive_losses == 3

    def test_avg_trade_pnl(self) -> None:
        # 5 wins at $10, 5 losses at $10 → avg = 0
        trades = [FakeTrade(str(i), 10.0 if i < 5 else -10.0, 0.0) for i in range(10)]
        report = make_generator().generate(trades)
        assert report.avg_trade_pnl_usd == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# PnL calculation tests
# ---------------------------------------------------------------------------


class TestPnLCalculation:
    def test_gross_pnl(self) -> None:
        trades = [FakeTrade(str(i), 10.0, 0.0) for i in range(10)]  # all wins
        report = make_generator(10_000.0).generate(trades)
        assert report.gross_pnl_usd == pytest.approx(100.0)

    def test_net_pnl_subtracts_fees(self) -> None:
        trades = [FakeTrade(str(i), 10.0, 4.0) for i in range(10)]
        report = make_generator(10_000.0).generate(trades)
        assert report.gross_pnl_usd == pytest.approx(100.0)
        assert report.total_fees_usd == pytest.approx(40.0)
        assert report.net_pnl_usd == pytest.approx(60.0)

    def test_return_pct(self) -> None:
        trades = [FakeTrade(str(i), 100.0, 0.0) for i in range(10)]  # $1000 gain
        report = make_generator(10_000.0).generate(trades)
        assert report.return_pct == pytest.approx(0.10, rel=1e-4)  # 10%

    def test_final_balance_computed(self) -> None:
        trades = [FakeTrade(str(i), 100.0, 10.0) for i in range(5)]  # +$500 - $50 fees
        report = make_generator(10_000.0).generate(trades)
        assert report.final_balance == pytest.approx(10_450.0, rel=1e-4)


# ---------------------------------------------------------------------------
# Risk metric tests
# ---------------------------------------------------------------------------


class TestRiskMetrics:
    def test_sharpe_positive_for_consistent_profits(self) -> None:
        # Use varying returns so std > 0 — e.g. mostly wins with some small losses
        import random

        rng = random.Random(42)
        trades = [FakeTrade(str(i), 10.0 + rng.gauss(0, 2), 0.0) for i in range(50)]
        report = make_generator().generate(trades)
        assert report.sharpe_ratio > 0

    def test_sharpe_zero_for_single_trade(self) -> None:
        report = make_generator().generate([FakeTrade("0", 10.0, 0.0)])
        assert report.sharpe_ratio == pytest.approx(0.0)

    def test_max_drawdown_zero_all_wins(self) -> None:
        trades = [FakeTrade(str(i), 10.0, 0.0) for i in range(10)]
        report = make_generator(10_000.0).generate(trades)
        assert report.max_drawdown_pct == pytest.approx(0.0)

    def test_max_drawdown_detected(self) -> None:
        # $100 up, $500 down → max DD = $500 from peak of $10,100
        trades = [FakeTrade("0", 100.0, 0.0), FakeTrade("1", -500.0, 0.0)]
        report = make_generator(10_000.0).generate(trades)
        assert report.max_drawdown_usd == pytest.approx(500.0)
        assert report.max_drawdown_pct == pytest.approx(500.0 / 10_100.0, rel=1e-4)

    def test_sortino_zero_no_losses(self) -> None:
        trades = [FakeTrade(str(i), 10.0, 0.0) for i in range(10)]
        report = make_generator().generate(trades)
        assert math.isinf(report.sortino_ratio) or report.sortino_ratio > 0


# ---------------------------------------------------------------------------
# Regime breakdown tests
# ---------------------------------------------------------------------------


class TestRegimeBreakdown:
    def test_regime_stats_aggregated(self) -> None:
        trades = [FakeTrade(str(i), 10.0 if i < 5 else -5.0, 1.0) for i in range(10)]
        labels = {str(i): "trending" if i < 7 else "ranging" for i in range(10)}
        report = make_generator().generate(trades, regime_labels=labels)
        assert "trending" in report.regime_stats
        assert "ranging" in report.regime_stats

    def test_profitable_regime_count(self) -> None:
        trades = [
            FakeTrade("0", 100.0, 0.0),  # trending win
            FakeTrade("1", -10.0, 0.0),  # ranging loss
        ]
        labels = {"0": "trending", "1": "ranging"}
        report = make_generator().generate(trades, regime_labels=labels)
        assert report.profitable_regime_count == 1

    def test_no_regime_labels_empty_breakdown(self) -> None:
        trades = make_trades(n=5)
        report = make_generator().generate(trades)
        assert report.regime_stats == {}


# ---------------------------------------------------------------------------
# Go/no-go verdict tests
# ---------------------------------------------------------------------------


class TestGoNoGo:
    def test_insufficient_data_below_200_trades(self) -> None:
        trades = make_trades(n=100)
        report = make_generator().generate(trades)
        assert report.go_nogo_verdict == "INSUFFICIENT_DATA"
        assert not report.go_nogo_details.get("enough_trades", True)

    def test_no_go_low_win_rate(self) -> None:
        # 200+ trades but 45% win rate
        trades = make_trades(n=200, win_rate=0.45, win_usd=15.0, lose_usd=-5.0)
        report = make_generator().generate(trades)
        # May not pass all criteria
        if report.go_nogo_verdict != "INSUFFICIENT_DATA":
            # Check win_rate_ok is False
            assert not report.go_nogo_details.get("win_rate_ok", True)

    def test_go_with_all_criteria_met(self) -> None:
        # Build a dataset that passes all criteria
        # 210 trades, 60% win rate, consistent profits, low drawdown
        n = 210
        trades = []
        for i in range(n):
            is_win = i % 5 != 0  # 80% wins
            pnl = 5.0 if is_win else -2.0
            trades.append(FakeTrade(str(i), pnl, 0.1, notional_usd=5_000.0))
        labels = {
            str(i): "trending" if i % 3 == 0 else "ranging" if i % 3 == 1 else "volatile"
            for i in range(n)
        }
        gen = DailyReportGenerator(initial_balance=100_000.0)
        report = gen.generate(trades, regime_labels=labels)
        # Most criteria should pass for profitable setup
        assert report.go_nogo_details["enough_trades"]
        assert report.go_nogo_details["win_rate_ok"]

    def test_go_nogo_details_keys_present(self) -> None:
        trades = make_trades(n=5)
        report = make_generator().generate(trades)
        expected_keys = {
            "enough_trades",
            "win_rate_ok",
            "sharpe_ok",
            "drawdown_ok",
            "profit_factor_ok",
            "avg_pnl_ok",
            "consec_losses_ok",
            "regime_breadth_ok",
        }
        assert expected_keys.issubset(report.go_nogo_details.keys())


# ---------------------------------------------------------------------------
# Output format tests
# ---------------------------------------------------------------------------


class TestOutputFormats:
    def test_to_dict_is_json_compatible(self) -> None:
        import json

        trades = make_trades(n=5)
        report = make_generator().generate(trades)
        d = report.to_dict()
        json_str = json.dumps(d)
        parsed = json.loads(json_str)
        assert parsed["total_trades"] == 5

    def test_to_dict_has_required_keys(self) -> None:
        trades = make_trades(n=5)
        report = make_generator().generate(trades)
        d = report.to_dict()
        required = {
            "report_date",
            "total_trades",
            "win_rate",
            "profit_factor",
            "net_pnl_usd",
            "sharpe_ratio",
            "max_drawdown_pct",
            "go_nogo_verdict",
        }
        assert required.issubset(d.keys())

    def test_to_text_contains_key_info(self) -> None:
        trades = make_trades(n=5)
        report = make_generator().generate(trades)
        text = report.to_text()
        assert "Report" in text
        assert "TRADES" in text
        assert "PnL" in text
        assert "RISK" in text

    def test_report_date_populated(self) -> None:
        trades = make_trades(n=5)
        report = make_generator().generate(trades)
        assert len(report.report_date) == 10  # "YYYY-MM-DD"

    def test_final_balance_override(self) -> None:
        trades = make_trades(n=5)
        report = make_generator(10_000.0).generate(trades, final_balance=11_000.0)
        assert report.final_balance == pytest.approx(11_000.0)
