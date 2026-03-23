"""Tests for RiskManager: full orchestration of all risk components."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime

import numpy as np
import pytest

from ep2_crypto.risk.config import RiskConfig
from ep2_crypto.risk.risk_manager import (
    RiskActionType,
    RiskManager,
    SignalInput,
)


@pytest.fixture
def conn() -> sqlite3.Connection:
    return sqlite3.connect(":memory:")


@pytest.fixture
def config() -> RiskConfig:
    return RiskConfig(
        enforce_trading_hours=False,  # Simplify tests
        min_volatility_ann=0.0,  # Don't block on vol in unit tests
        max_volatility_ann=10.0,
    )


@pytest.fixture
def rm(conn: sqlite3.Connection, config: RiskConfig) -> RiskManager:
    return RiskManager(conn, initial_equity=50_000.0, config=config)


def _make_signal(
    direction: str = "long",
    confidence: float = 0.80,
    timestamp_ms: int | None = None,
) -> SignalInput:
    if timestamp_ms is None:
        timestamp_ms = int(
            datetime(2026, 3, 25, 14, 0, tzinfo=UTC).timestamp() * 1000
        )
    return SignalInput(
        direction=direction,
        confidence=confidence,
        timestamp_ms=timestamp_ms,
    )


def _make_prices(
    n: int = 300, base: float = 67000.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    closes = base + np.cumsum(rng.normal(0, 50, n))
    highs = closes + rng.uniform(30, 150, n)
    lows = closes - rng.uniform(30, 150, n)
    return closes, highs, lows


class TestApproveTradeBasic:
    def test_approve_valid_signal(self, rm: RiskManager) -> None:
        closes, highs, lows = _make_prices()
        signal = _make_signal()
        decision = rm.approve_trade(signal, closes, highs, lows, len(closes) - 1)
        assert decision.approved
        assert decision.quantity_btc > 0
        assert decision.stop_price > 0
        assert decision.reason == "All risk checks passed"

    def test_reject_when_position_open(self, rm: RiskManager) -> None:
        closes, highs, lows = _make_prices()
        signal = _make_signal()
        decision = rm.approve_trade(signal, closes, highs, lows, len(closes) - 1)
        assert decision.approved

        # Open position
        rm.on_trade_opened(
            "long", decision.quantity_btc, 67000.0,
            signal.timestamp_ms, decision.stop_price,
        )

        # Try to open another
        decision2 = rm.approve_trade(signal, closes, highs, lows, len(closes) - 1)
        assert not decision2.approved
        assert "already open" in decision2.reason

    def test_reject_invalid_direction(self, rm: RiskManager) -> None:
        closes, highs, lows = _make_prices()
        signal = _make_signal(direction="sideways")
        decision = rm.approve_trade(signal, closes, highs, lows, len(closes) - 1)
        assert not decision.approved


class TestKillSwitchIntegration:
    def test_daily_loss_halts_trading(self, rm: RiskManager) -> None:
        closes, highs, lows = _make_prices()

        # Manually open a large position (bypass sizer to control size)
        # Need loss > 3% of $50k = $1500
        # 0.05 BTC * (67000 - 30000) = $1850 loss
        ts = int(datetime(2026, 3, 25, 14, 0, tzinfo=UTC).timestamp() * 1000)
        rm.on_trade_opened("long", 0.05, 67000.0, ts, 65000.0)
        rm.on_trade_closed(30000.0, ts + 300_000)  # $1850 loss = 3.7%

        # Now try to open another trade
        signal2 = _make_signal()
        decision2 = rm.approve_trade(signal2, closes, highs, lows, len(closes) - 1)
        assert not decision2.approved
        assert "Kill switch" in decision2.reason

    def test_emergency_kill_switch(self, rm: RiskManager) -> None:
        rm.trigger_emergency("Manual test halt")
        closes, highs, lows = _make_prices()
        decision = rm.approve_trade(
            _make_signal(), closes, highs, lows, len(closes) - 1
        )
        assert not decision.approved
        assert "Kill switch" in decision.reason

    def test_reset_kill_switch_allows_trading(self, rm: RiskManager) -> None:
        rm.trigger_emergency("Test halt")
        rm.reset_kill_switch("emergency", "Test complete, safe to resume")

        closes, highs, lows = _make_prices()
        decision = rm.approve_trade(
            _make_signal(), closes, highs, lows, len(closes) - 1
        )
        assert decision.approved


class TestMaxTradesPerDay:
    def test_exceeding_daily_trade_limit(
        self, conn: sqlite3.Connection
    ) -> None:
        config = RiskConfig(
            enforce_trading_hours=False,
            min_volatility_ann=0.0,
            max_volatility_ann=10.0,
            max_trades_per_day=2,
        )
        rm = RiskManager(conn, initial_equity=50_000.0, config=config)
        closes, highs, lows = _make_prices()
        idx = len(closes) - 1

        for i in range(2):
            signal = _make_signal(timestamp_ms=1700000000000 + i * 600_000)
            d = rm.approve_trade(signal, closes, highs, lows, idx)
            assert d.approved
            rm.on_trade_opened(
                "long", d.quantity_btc, 67000.0,
                signal.timestamp_ms, d.stop_price,
            )
            rm.on_trade_closed(67100.0, signal.timestamp_ms + 300_000)

        # Third trade should be rejected
        signal3 = _make_signal(timestamp_ms=1700001200000)
        d3 = rm.approve_trade(signal3, closes, highs, lows, idx)
        assert not d3.approved
        assert "Max trades" in d3.reason


class TestOnBar:
    def test_max_holding_period_triggers_close(self, rm: RiskManager) -> None:
        closes, highs, lows = _make_prices()
        idx = len(closes) - 1

        signal = _make_signal()
        d = rm.approve_trade(signal, closes, highs, lows, idx)
        rm.on_trade_opened(
            "long", d.quantity_btc, 67000.0,
            signal.timestamp_ms, d.stop_price,
        )

        # Simulate 6 bars
        actions = []
        for i in range(7):
            bar_ts = signal.timestamp_ms + (i + 1) * 300_000
            bar_actions = rm.on_bar(
                67000.0, 67100.0, 66900.0, bar_ts,
                closes, highs, lows, idx,
            )
            actions.extend(bar_actions)

        close_actions = [a for a in actions if a.action == RiskActionType.CLOSE_POSITION]
        assert len(close_actions) >= 1
        assert "holding period" in close_actions[0].reason

    def test_stop_loss_triggers_close(self, rm: RiskManager) -> None:
        closes, highs, lows = _make_prices()
        idx = len(closes) - 1

        signal = _make_signal()
        d = rm.approve_trade(signal, closes, highs, lows, idx)
        rm.on_trade_opened(
            "long", d.quantity_btc, 67000.0,
            signal.timestamp_ms, d.stop_price,
        )

        # Bar with low below stop price
        bar_ts = signal.timestamp_ms + 300_000
        actions = rm.on_bar(
            66000.0, 67100.0, d.stop_price - 100, bar_ts,
            closes, highs, lows, idx,
        )

        close_actions = [a for a in actions if a.action == RiskActionType.CLOSE_POSITION]
        assert len(close_actions) >= 1
        assert "Stop loss" in close_actions[0].reason

    def test_no_action_when_flat(self, rm: RiskManager) -> None:
        closes, highs, lows = _make_prices()
        idx = len(closes) - 1
        actions = rm.on_bar(
            67000.0, 67100.0, 66900.0, 1700000000000,
            closes, highs, lows, idx,
        )
        assert actions == []


class TestOnBarKillSwitch:
    def test_kill_switch_during_on_bar_forces_close(self, rm: RiskManager) -> None:
        """If kill switch triggers during on_bar while position is open, force close."""
        closes, highs, lows = _make_prices()
        idx = len(closes) - 1
        signal = _make_signal()
        d = rm.approve_trade(signal, closes, highs, lows, idx)
        rm.on_trade_opened(
            "long", d.quantity_btc, 67000.0,
            signal.timestamp_ms, d.stop_price,
        )
        # Trigger emergency while position is open
        rm.trigger_emergency("Test during bar")
        bar_ts = signal.timestamp_ms + 300_000
        actions = rm.on_bar(
            67000.0, 67100.0, 66900.0, bar_ts,
            closes, highs, lows, idx,
        )
        close_actions = [a for a in actions if a.action == RiskActionType.CLOSE_POSITION]
        assert len(close_actions) >= 1
        assert "Kill switch" in close_actions[0].reason


class TestShortStopLoss:
    def test_short_stop_loss_on_bar_high(self, rm: RiskManager) -> None:
        closes, highs, lows = _make_prices()
        idx = len(closes) - 1
        signal = _make_signal(direction="short")
        d = rm.approve_trade(signal, closes, highs, lows, idx)
        assert d.approved
        rm.on_trade_opened(
            "short", d.quantity_btc, 67000.0,
            signal.timestamp_ms, d.stop_price,
        )
        # Bar high exceeds short stop price
        bar_ts = signal.timestamp_ms + 300_000
        actions = rm.on_bar(
            67000.0, d.stop_price + 500, 66800.0, bar_ts,
            closes, highs, lows, idx,
        )
        close_actions = [a for a in actions if a.action == RiskActionType.CLOSE_POSITION]
        assert len(close_actions) >= 1
        assert "Stop loss" in close_actions[0].reason


class TestVolatilityGuardIntegration:
    def test_vol_guard_rejection_has_state(self) -> None:
        conn = sqlite3.connect(":memory:")
        config = RiskConfig(
            enforce_trading_hours=True,
            min_volatility_ann=0.0,
            max_volatility_ann=10.0,
        )
        rm = RiskManager(conn, initial_equity=50_000.0, config=config)
        closes, highs, lows = _make_prices()
        # Use a timestamp outside trading hours (3:00 UTC)
        ts_ms = int(datetime(2026, 3, 25, 3, 0, tzinfo=UTC).timestamp() * 1000)
        signal = SignalInput(direction="long", confidence=0.80, timestamp_ms=ts_ms)
        decision = rm.approve_trade(signal, closes, highs, lows, len(closes) - 1)
        assert not decision.approved
        assert not decision.volatility_ok


class TestWeeklyReset:
    def test_weekly_counters_reset(self, rm: RiskManager) -> None:
        closes, highs, lows = _make_prices()
        signal = _make_signal()
        d = rm.approve_trade(signal, closes, highs, lows, len(closes) - 1)
        rm.on_trade_opened("long", d.quantity_btc, 67000.0, signal.timestamp_ms, d.stop_price)
        rm.on_trade_closed(67100.0, signal.timestamp_ms + 300_000)
        state = rm.get_risk_state()
        assert state.weekly_pnl != 0 or state.trades_today > 0
        rm.reset_weekly_counters()
        state = rm.get_risk_state()
        assert state.weekly_pnl == pytest.approx(0.0)


class TestDrawdownGateIntegration:
    def test_drawdown_gate_blocks_at_max_dd(self) -> None:
        conn = sqlite3.connect(":memory:")
        config = RiskConfig(
            enforce_trading_hours=False,
            min_volatility_ann=0.0,
            max_volatility_ann=10.0,
        )
        rm = RiskManager(conn, initial_equity=50_000.0, config=config)

        # Lose 15%+ to trigger drawdown halt
        ts = int(datetime(2026, 3, 25, 14, 0, tzinfo=UTC).timestamp() * 1000)
        rm.on_trade_opened("long", 1.0, 67000.0, ts, 55000.0)
        rm.on_trade_closed(56000.0, ts + 300_000)  # lose $11000 = 22% of 50k

        closes, highs, lows = _make_prices()
        signal = _make_signal()
        decision = rm.approve_trade(signal, closes, highs, lows, len(closes) - 1)
        assert not decision.approved


class TestRiskState:
    def test_get_risk_state(self, rm: RiskManager) -> None:
        state = rm.get_risk_state()
        assert state.equity == pytest.approx(50_000.0)
        assert state.daily_pnl == 0.0
        assert state.consecutive_losses == 0
        assert state.trades_today == 0
        assert len(state.kill_switches) == 5


class TestDayWeekReset:
    def test_daily_reset(self, rm: RiskManager) -> None:
        closes, highs, lows = _make_prices()
        signal = _make_signal()
        d = rm.approve_trade(signal, closes, highs, lows, len(closes) - 1)
        rm.on_trade_opened(
            "long", d.quantity_btc, 67000.0,
            signal.timestamp_ms, d.stop_price,
        )
        rm.on_trade_closed(67100.0, signal.timestamp_ms + 300_000)
        assert rm.get_risk_state().trades_today == 1

        rm.reset_daily_counters()
        assert rm.get_risk_state().trades_today == 0


class TestConfigValidation:
    def test_valid_config(self) -> None:
        config = RiskConfig()
        assert config.daily_loss_limit == 0.03

    def test_invalid_daily_loss(self) -> None:
        with pytest.raises((ValueError, Exception)):
            RiskConfig(daily_loss_limit=0.0)

    def test_invalid_max_position(self) -> None:
        with pytest.raises((ValueError, Exception)):
            RiskConfig(max_position_fraction=1.5)

    def test_invalid_vol_range(self) -> None:
        with pytest.raises((ValueError, Exception)):
            RiskConfig(min_volatility_ann=2.0, max_volatility_ann=1.0)
