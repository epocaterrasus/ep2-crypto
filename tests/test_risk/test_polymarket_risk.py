"""Tests for PolymarketRiskAdapter: binary risk management."""

from __future__ import annotations

import sqlite3
import time

import pytest

from ep2_crypto.risk.config import RiskConfig
from ep2_crypto.risk.kill_switches import SwitchName  # noqa: F401 — Literal type
from ep2_crypto.risk.polymarket_risk import (
    BinaryDirection,
    BinarySignal,
    PolymarketRiskAdapter,
)


def _make_signal(
    direction: BinaryDirection = BinaryDirection.UP,
    model_prob: float = 0.55,
    market_price: float = 0.50,
) -> BinarySignal:
    return BinarySignal(
        direction=direction,
        model_prob=model_prob,
        market_price=market_price,
        timestamp_ms=int(time.time() * 1000),
        window_end_ms=int(time.time() * 1000) + 300_000,
    )


def _make_config(**overrides: object) -> RiskConfig:
    defaults = {
        "daily_loss_limit": 0.02,
        "weekly_loss_limit": 0.05,
        "max_drawdown_halt": 0.15,
        "consecutive_loss_limit": 10,
        "max_trades_per_day": 30,
    }
    defaults.update(overrides)
    return RiskConfig(**defaults)  # type: ignore[arg-type]


def _make_adapter(
    initial_equity: float = 10_000.0,
    **config_overrides: object,
) -> PolymarketRiskAdapter:
    conn = sqlite3.connect(":memory:")
    return PolymarketRiskAdapter(
        config=_make_config(**config_overrides),
        initial_equity=initial_equity,
        conn=conn,
    )


# ---------------------------------------------------------------------------
# Basic approval flow
# ---------------------------------------------------------------------------


class TestBasicApproval:
    def test_approve_with_edge(self) -> None:
        adapter = _make_adapter()
        signal = _make_signal(model_prob=0.55, market_price=0.50)
        decision = adapter.approve_trade(signal)
        assert decision.approved
        assert decision.shares > 0
        assert decision.cost_usd > 0
        assert decision.max_loss_usd == decision.cost_usd

    def test_reject_no_edge(self) -> None:
        adapter = _make_adapter()
        signal = _make_signal(model_prob=0.50, market_price=0.50)
        decision = adapter.approve_trade(signal)
        assert not decision.approved
        assert "no_edge" in decision.reason

    def test_reject_position_open(self) -> None:
        adapter = _make_adapter()
        signal = _make_signal()
        adapter.approve_trade(signal)
        adapter.on_trade_opened()

        decision = adapter.approve_trade(signal)
        assert not decision.approved
        assert "already open" in decision.reason

    def test_position_clears_after_resolution(self) -> None:
        adapter = _make_adapter()
        signal = _make_signal()
        adapter.approve_trade(signal)
        adapter.on_trade_opened()
        adapter.on_trade_resolved(won=True, pnl=50.0)

        decision = adapter.approve_trade(signal)
        assert decision.approved


# ---------------------------------------------------------------------------
# Kill switches
# ---------------------------------------------------------------------------


class TestKillSwitches:
    def test_daily_loss_triggers(self) -> None:
        adapter = _make_adapter(daily_loss_limit=0.02)
        # Lose 2% = $200
        adapter.on_trade_opened()
        adapter.on_trade_resolved(won=False, pnl=-200.0)

        signal = _make_signal()
        decision = adapter.approve_trade(signal)
        assert not decision.approved
        assert not decision.kill_switch_ok

    def test_consecutive_loss_triggers(self) -> None:
        adapter = _make_adapter(consecutive_loss_limit=3)
        for _i in range(3):
            adapter.on_trade_opened()
            adapter.on_trade_resolved(won=False, pnl=-10.0)

        signal = _make_signal()
        decision = adapter.approve_trade(signal)
        assert not decision.approved

    def test_consecutive_losses_reset_on_win(self) -> None:
        adapter = _make_adapter(consecutive_loss_limit=5)
        # 2 losses
        for _ in range(2):
            adapter.on_trade_opened()
            adapter.on_trade_resolved(won=False, pnl=-10.0)

        assert adapter.consecutive_losses == 2

        # 1 win resets
        adapter.on_trade_opened()
        adapter.on_trade_resolved(won=True, pnl=20.0)
        assert adapter.consecutive_losses == 0

    def test_emergency_halt(self) -> None:
        adapter = _make_adapter()
        adapter.trigger_emergency("test halt")

        signal = _make_signal()
        decision = adapter.approve_trade(signal)
        assert not decision.approved

    def test_kill_switch_reset(self) -> None:
        adapter = _make_adapter(daily_loss_limit=0.02)
        adapter.on_trade_opened()
        adapter.on_trade_resolved(won=False, pnl=-200.0)

        signal = _make_signal()
        assert not adapter.approve_trade(signal).approved

        adapter.reset_kill_switch("daily_loss", "manual reset after review")
        # Kill switch should be cleared now
        signal2 = _make_signal()
        decision2 = adapter.approve_trade(signal2)
        # May still reject for other reasons but kill_switch_ok should be True
        assert decision2.kill_switch_ok


# ---------------------------------------------------------------------------
# Max trades per day
# ---------------------------------------------------------------------------


class TestMaxTrades:
    def test_max_trades_blocks(self) -> None:
        adapter = _make_adapter(max_trades_per_day=3)
        for _ in range(3):
            adapter.on_trade_opened()
            adapter.on_trade_resolved(won=True, pnl=10.0)

        signal = _make_signal()
        decision = adapter.approve_trade(signal)
        assert not decision.approved
        assert "Max trades" in decision.reason

    def test_daily_reset_clears_counter(self) -> None:
        adapter = _make_adapter(max_trades_per_day=3)
        for _ in range(3):
            adapter.on_trade_opened()
            adapter.on_trade_resolved(won=True, pnl=10.0)

        adapter.reset_daily()

        signal = _make_signal()
        decision = adapter.approve_trade(signal)
        assert decision.approved


# ---------------------------------------------------------------------------
# Drawdown gate
# ---------------------------------------------------------------------------


class TestDrawdownGate:
    def test_drawdown_reduces_size(self) -> None:
        adapter = _make_adapter(max_drawdown_halt=0.15)
        signal = _make_signal(model_prob=0.60, market_price=0.50)

        # Trade with no drawdown
        decision_full = adapter.approve_trade(signal)
        assert decision_full.approved
        full_cost = decision_full.cost_usd

        # Simulate drawdown
        adapter.on_trade_opened()
        adapter.on_trade_resolved(won=False, pnl=-500.0)  # 5% DD

        decision_reduced = adapter.approve_trade(signal)
        if decision_reduced.approved:
            assert decision_reduced.cost_usd < full_cost


# ---------------------------------------------------------------------------
# Equity tracking
# ---------------------------------------------------------------------------


class TestEquityTracking:
    def test_equity_increases_on_win(self) -> None:
        adapter = _make_adapter()
        adapter.on_trade_opened()
        adapter.on_trade_resolved(won=True, pnl=50.0)
        assert adapter.equity == 10_050.0

    def test_equity_decreases_on_loss(self) -> None:
        adapter = _make_adapter()
        adapter.on_trade_opened()
        adapter.on_trade_resolved(won=False, pnl=-30.0)
        assert adapter.equity == 9_970.0

    def test_daily_pnl_tracks(self) -> None:
        adapter = _make_adapter()
        adapter.on_trade_opened()
        adapter.on_trade_resolved(won=True, pnl=50.0)
        adapter.on_trade_opened()
        adapter.on_trade_resolved(won=False, pnl=-30.0)
        assert adapter.daily_pnl == pytest.approx(20.0)

    def test_daily_pnl_resets(self) -> None:
        adapter = _make_adapter()
        adapter.on_trade_opened()
        adapter.on_trade_resolved(won=True, pnl=50.0)
        adapter.reset_daily()
        assert adapter.daily_pnl == 0.0


# ---------------------------------------------------------------------------
# Direction handling
# ---------------------------------------------------------------------------


class TestDirectionHandling:
    def test_up_direction(self) -> None:
        adapter = _make_adapter()
        signal = _make_signal(direction=BinaryDirection.UP, model_prob=0.55)
        decision = adapter.approve_trade(signal)
        assert decision.approved
        assert decision.direction == BinaryDirection.UP

    def test_down_direction(self) -> None:
        adapter = _make_adapter()
        signal = _make_signal(direction=BinaryDirection.DOWN, model_prob=0.55)
        decision = adapter.approve_trade(signal)
        assert decision.approved
        assert decision.direction == BinaryDirection.DOWN


# ---------------------------------------------------------------------------
# Full scenario: normal trading day
# ---------------------------------------------------------------------------


class TestFullScenario:
    def test_normal_trading_day(self) -> None:
        """Simulate a day with wins and losses."""
        adapter = _make_adapter(max_trades_per_day=30, daily_loss_limit=0.10)

        wins = 0
        losses = 0

        for i in range(10):
            signal = _make_signal(model_prob=0.55)
            decision = adapter.approve_trade(signal)

            if not decision.approved:
                break

            adapter.on_trade_opened()

            # Alternate win/loss
            if i % 3 != 0:
                adapter.on_trade_resolved(won=True, pnl=decision.cost_usd * 0.8)
                wins += 1
            else:
                adapter.on_trade_resolved(won=False, pnl=-decision.cost_usd)
                losses += 1

        assert wins > 0
        assert losses > 0
        assert adapter.equity != 10_000.0  # Changed

    def test_crash_scenario(self) -> None:
        """Simulate consecutive losses triggering kill switch."""
        adapter = _make_adapter(consecutive_loss_limit=5, daily_loss_limit=0.10)

        for _i in range(5):
            signal = _make_signal(model_prob=0.55)
            decision = adapter.approve_trade(signal)
            if not decision.approved:
                break
            adapter.on_trade_opened()
            adapter.on_trade_resolved(won=False, pnl=-50.0)

        # After 5 consecutive losses, should be halted
        signal = _make_signal(model_prob=0.55)
        decision = adapter.approve_trade(signal)
        assert not decision.approved
