"""Stress scenario tests for the risk engine.

Simulates extreme market conditions to verify the system survives:
1. COVID-style crash (20% drop in 30 min)
2. Flash crash (sudden 20% spike then recovery)
3. 10 consecutive losses in 1 hour
4. Slow bleed over 14 days
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime

import numpy as np

from ep2_crypto.risk.config import RiskConfig
from ep2_crypto.risk.risk_manager import (
    RiskActionType,
    RiskManager,
    SignalInput,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _config(**overrides: object) -> RiskConfig:
    defaults = {
        "enforce_trading_hours": False,
        "min_volatility_ann": 0.0,
        "max_volatility_ann": 100.0,  # Don't block on vol in stress tests
    }
    defaults.update(overrides)
    return RiskConfig(**defaults)


def _signal(direction: str = "long", confidence: float = 0.80, ts_ms: int = 0) -> SignalInput:
    if ts_ms == 0:
        ts_ms = int(datetime(2026, 3, 25, 14, 0, tzinfo=UTC).timestamp() * 1000)
    return SignalInput(direction=direction, confidence=confidence, timestamp_ms=ts_ms)


def _make_crash_prices(
    n_bars: int,
    start_price: float,
    crash_start_bar: int,
    crash_pct: float,
    crash_duration_bars: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate prices with a crash starting at crash_start_bar."""
    closes = np.full(n_bars, start_price)
    # Normal price before crash
    rng = np.random.default_rng(42)
    for i in range(1, crash_start_bar):
        closes[i] = closes[i - 1] * (1 + rng.normal(0, 0.001))

    # Crash phase: drop crash_pct over crash_duration_bars
    for i in range(crash_duration_bars):
        bar_idx = crash_start_bar + i
        if bar_idx >= n_bars:
            break
        progress = (i + 1) / crash_duration_bars
        closes[bar_idx] = start_price * (1 - crash_pct * progress)

    # Post-crash: stay low
    post_crash_price = start_price * (1 - crash_pct)
    for i in range(crash_start_bar + crash_duration_bars, n_bars):
        closes[i] = post_crash_price * (1 + rng.normal(0, 0.001))

    highs = closes * (1 + np.abs(rng.normal(0, 0.002, n_bars)))
    lows = closes * (1 - np.abs(rng.normal(0, 0.002, n_bars)))
    return closes, highs, lows


def _make_slow_bleed_prices(
    n_bars: int,
    start_price: float,
    daily_loss_pct: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate prices with steady daily decline."""
    rng = np.random.default_rng(42)
    bar_loss = daily_loss_pct / 288  # per 5-min bar
    closes = np.zeros(n_bars)
    closes[0] = start_price
    for i in range(1, n_bars):
        closes[i] = closes[i - 1] * (1 - bar_loss + rng.normal(0, 0.0005))
    highs = closes * (1 + np.abs(rng.normal(0, 0.001, n_bars)))
    lows = closes * (1 - np.abs(rng.normal(0, 0.001, n_bars)))
    return closes, highs, lows


# ---------------------------------------------------------------------------
# Scenario 1: COVID-style crash — 20% drop in 6 bars (30 min)
# ---------------------------------------------------------------------------


class TestCOVIDCrash:
    def test_stop_loss_fires_during_crash(self) -> None:
        """When position is open and market drops 20%, stop should fire."""
        conn = sqlite3.connect(":memory:")
        rm = RiskManager(conn, initial_equity=50_000.0, config=_config())

        n_bars = 300
        closes, highs, lows = _make_crash_prices(
            n_bars,
            start_price=67000.0,
            crash_start_bar=200,
            crash_pct=0.20,
            crash_duration_bars=6,
        )

        # Approve and open a trade before the crash
        signal = _signal(ts_ms=1700000000000)
        idx = 190
        decision = rm.approve_trade(signal, closes, highs, lows, idx)
        assert decision.approved

        rm.on_trade_opened(
            "long",
            decision.quantity_btc,
            closes[idx],
            signal.timestamp_ms,
            decision.stop_price,
        )

        # Walk through crash bars
        stop_fired = False
        for i in range(191, 210):
            bar_ts = signal.timestamp_ms + (i - 190) * 300_000
            actions = rm.on_bar(
                closes[i],
                highs[i],
                lows[i],
                bar_ts,
                closes,
                highs,
                lows,
                i,
            )
            for action in actions:
                if action.action == RiskActionType.CLOSE_POSITION:
                    stop_fired = True
                    break
            if stop_fired:
                break

        assert stop_fired, "Stop loss should fire during 20% crash"

    def test_system_survives_with_limited_drawdown(self) -> None:
        """After COVID crash, equity loss should be bounded by risk limits."""
        conn = sqlite3.connect(":memory:")
        initial_equity = 50_000.0
        rm = RiskManager(conn, initial_equity=initial_equity, config=_config())

        closes, highs, lows = _make_crash_prices(
            300,
            start_price=67000.0,
            crash_start_bar=200,
            crash_pct=0.20,
            crash_duration_bars=6,
        )

        # Open and immediately face crash
        signal = _signal(ts_ms=1700000000000)
        decision = rm.approve_trade(signal, closes, highs, lows, 199)
        if decision.approved:
            rm.on_trade_opened(
                "long",
                decision.quantity_btc,
                closes[199],
                signal.timestamp_ms,
                decision.stop_price,
            )

            # Process crash bars
            for i in range(200, 210):
                bar_ts = signal.timestamp_ms + (i - 199) * 300_000
                actions = rm.on_bar(
                    closes[i],
                    highs[i],
                    lows[i],
                    bar_ts,
                    closes,
                    highs,
                    lows,
                    i,
                )
                for action in actions:
                    if action.action == RiskActionType.CLOSE_POSITION:
                        rm.on_trade_closed(closes[i], bar_ts)
                        break

        # Equity should not have dropped more than 15%
        state = rm.get_risk_state()
        loss_pct = (initial_equity - state.equity) / initial_equity
        assert loss_pct < 0.15, f"Lost {loss_pct:.1%} — should be bounded by risk limits"


# ---------------------------------------------------------------------------
# Scenario 2: Flash crash — 20% drop and immediate recovery
# ---------------------------------------------------------------------------


class TestFlashCrash:
    def test_stop_fires_within_first_bar(self) -> None:
        """In a flash crash, the stop should trigger on the first crash bar."""
        conn = sqlite3.connect(":memory:")
        rm = RiskManager(conn, initial_equity=50_000.0, config=_config())

        closes, highs, lows = _make_crash_prices(
            300,
            start_price=67000.0,
            crash_start_bar=200,
            crash_pct=0.20,
            crash_duration_bars=1,  # instant
        )

        signal = _signal(ts_ms=1700000000000)
        decision = rm.approve_trade(signal, closes, highs, lows, 199)
        assert decision.approved

        rm.on_trade_opened(
            "long",
            decision.quantity_btc,
            closes[199],
            signal.timestamp_ms,
            decision.stop_price,
        )

        # Single crash bar
        bar_ts = signal.timestamp_ms + 300_000
        actions = rm.on_bar(
            closes[200],
            highs[200],
            lows[200],
            bar_ts,
            closes,
            highs,
            lows,
            200,
        )

        close_actions = [a for a in actions if a.action == RiskActionType.CLOSE_POSITION]
        assert len(close_actions) >= 1, "Stop should fire on first crash bar"


# ---------------------------------------------------------------------------
# Scenario 3: 10 consecutive losses in 1 hour
# ---------------------------------------------------------------------------


class TestConsecutiveLosses:
    def test_kill_switch_triggers_after_consecutive_losses(self) -> None:
        conn = sqlite3.connect(":memory:")
        config = _config(
            consecutive_loss_limit=10,
            max_trades_per_day=50,
        )
        rm = RiskManager(conn, initial_equity=50_000.0, config=config)

        closes, highs, lows = _make_crash_prices(
            300,
            start_price=67000.0,
            crash_start_bar=100,
            crash_pct=0.05,
            crash_duration_bars=100,
        )

        base_ts = int(datetime(2026, 3, 25, 14, 0, tzinfo=UTC).timestamp() * 1000)
        losses = 0
        blocked = False

        for trade_num in range(15):
            signal = _signal(ts_ms=base_ts + trade_num * 360_000)
            idx = min(50 + trade_num * 5, 290)
            decision = rm.approve_trade(signal, closes, highs, lows, idx)

            if not decision.approved:
                blocked = True
                break

            entry_price = closes[idx]
            rm.on_trade_opened(
                "long",
                decision.quantity_btc,
                entry_price,
                signal.timestamp_ms,
                decision.stop_price,
            )

            # Simulate a small loss
            exit_price = entry_price * 0.998  # 0.2% loss per trade
            rm.on_trade_closed(exit_price, signal.timestamp_ms + 300_000)
            losses += 1

        assert blocked or losses >= 10, (
            "Should be blocked by kill switch or have recorded enough losses"
        )
        state = rm.get_risk_state()
        # Either consecutive loss or daily loss should have triggered
        assert state.consecutive_losses >= 10 or any(
            ks.state.value == "triggered" for ks in state.kill_switches
        )


# ---------------------------------------------------------------------------
# Scenario 4: Slow bleed over 14 days — drawdown gate activates
# ---------------------------------------------------------------------------


class TestSlowBleed:
    def test_drawdown_gate_reduces_size_during_bleed(self) -> None:
        """Over a slow bleed, the drawdown gate should progressively reduce size."""
        conn = sqlite3.connect(":memory:")
        rm = RiskManager(conn, initial_equity=50_000.0, config=_config())

        # 14 days of data = 4032 bars, with 0.3% daily loss
        n_bars = 4100
        closes, highs, lows = _make_slow_bleed_prices(
            n_bars,
            start_price=67000.0,
            daily_loss_pct=0.003,
        )

        base_ts = int(datetime(2026, 3, 25, 0, 0, tzinfo=UTC).timestamp() * 1000)
        sizes_over_time: list[float] = []

        for bar in range(300, min(n_bars, 1500), 100):
            bar_ts = base_ts + bar * 300_000

            # Process the bar (no position, just update state)
            rm.on_bar(closes[bar], highs[bar], lows[bar], bar_ts, closes, highs, lows, bar)

            signal = _signal(ts_ms=bar_ts)
            decision = rm.approve_trade(signal, closes, highs, lows, bar)
            if decision.approved:
                sizes_over_time.append(decision.quantity_btc)
                # Don't actually open — just measure approved size
            else:
                sizes_over_time.append(0.0)

        # As the bleed continues, sizes should generally decrease
        if len(sizes_over_time) > 3:
            early_sizes = sizes_over_time[:3]
            late_sizes = sizes_over_time[-3:]
            # Late sizes should be smaller or zero (due to kill switches)
            early_avg = sum(early_sizes) / len(early_sizes) if early_sizes else 0
            late_avg = sum(late_sizes) / len(late_sizes) if late_sizes else 0
            assert late_avg <= early_avg, (
                f"Late sizes ({late_avg:.6f}) should be <= early sizes ({early_avg:.6f})"
            )


# ---------------------------------------------------------------------------
# Meta: all stress scenarios complete without exceptions
# ---------------------------------------------------------------------------


class TestStressNoExceptions:
    def test_rapid_open_close_cycle(self) -> None:
        """Rapidly open and close 30 positions — no crashes."""
        conn = sqlite3.connect(":memory:")
        config = _config(max_trades_per_day=50)
        rm = RiskManager(conn, initial_equity=50_000.0, config=config)
        closes, highs, lows = _make_crash_prices(
            300,
            start_price=67000.0,
            crash_start_bar=250,
            crash_pct=0.05,
            crash_duration_bars=10,
        )

        base_ts = int(datetime(2026, 3, 25, 14, 0, tzinfo=UTC).timestamp() * 1000)
        for i in range(30):
            signal = _signal(ts_ms=base_ts + i * 600_000)
            decision = rm.approve_trade(signal, closes, highs, lows, 100)
            if not decision.approved:
                break
            rm.on_trade_opened(
                "long",
                decision.quantity_btc,
                67000.0,
                signal.timestamp_ms,
                decision.stop_price,
            )
            # Alternating wins and losses
            exit_price = 67100.0 if i % 3 != 0 else 66900.0
            rm.on_trade_closed(exit_price, signal.timestamp_ms + 300_000)

    def test_extreme_vol_does_not_crash(self) -> None:
        """Pass extreme prices through the system — no exceptions."""
        conn = sqlite3.connect(":memory:")
        rm = RiskManager(conn, initial_equity=50_000.0, config=_config())

        rng = np.random.default_rng(42)
        n = 300
        closes = 67000.0 + np.cumsum(rng.normal(0, 500, n))  # High volatility
        closes = np.maximum(closes, 1000.0)  # Floor at $1000
        highs = closes * 1.05
        lows = closes * 0.95

        base_ts = int(datetime(2026, 3, 25, 14, 0, tzinfo=UTC).timestamp() * 1000)
        for i in range(50, n, 10):
            bar_ts = base_ts + i * 300_000
            rm.on_bar(closes[i], highs[i], lows[i], bar_ts, closes, highs, lows, i)
