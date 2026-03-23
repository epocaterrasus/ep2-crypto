"""Hypothesis property-based tests for the risk engine.

These tests verify invariants that must hold across all possible inputs:
- Position size NEVER exceeds max cap
- After kill switch, ZERO trades approved until reset
- Drawdown multiplier always in [0, 1]
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from ep2_crypto.risk.config import RiskConfig
from ep2_crypto.risk.drawdown_gate import DrawdownGate
from ep2_crypto.risk.position_sizer import PositionSizer
from ep2_crypto.risk.risk_manager import RiskManager, SignalInput

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _prices(n: int = 100, base: float = 67000.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    closes = base + np.cumsum(rng.normal(0, 50, n))
    highs = closes + rng.uniform(30, 150, n)
    lows = closes - rng.uniform(30, 150, n)
    return closes, highs, lows


def _signal(
    direction: str = "long",
    confidence: float = 0.80,
) -> SignalInput:
    ts = int(datetime(2026, 3, 25, 14, 0, tzinfo=UTC).timestamp() * 1000)
    return SignalInput(direction=direction, confidence=confidence, timestamp_ms=ts)


# ---------------------------------------------------------------------------
# Property: Position size NEVER exceeds 5% of equity
# ---------------------------------------------------------------------------


@given(
    confidence=st.floats(min_value=0.01, max_value=1.0),
    equity=st.floats(min_value=1000.0, max_value=10_000_000.0),
    win_rate=st.floats(min_value=0.51, max_value=0.90),
    payoff=st.floats(min_value=0.5, max_value=5.0),
    dd_mult=st.floats(min_value=0.01, max_value=1.0),
)
@settings(max_examples=500)
def test_position_size_never_exceeds_cap(
    confidence: float,
    equity: float,
    win_rate: float,
    payoff: float,
    dd_mult: float,
) -> None:
    sizer = PositionSizer()
    closes, highs, lows = _prices()
    result = sizer.compute(
        "long",
        confidence,
        equity,
        67000.0,
        highs,
        lows,
        closes,
        current_idx=99,
        drawdown_multiplier=dd_mult,
        win_rate=win_rate,
        payoff_ratio=payoff,
    )
    if not result.rejected:
        assert result.position_fraction <= 0.05 + 1e-9, (
            f"Position fraction {result.position_fraction} exceeded 5% cap"
        )
        assert result.notional_usd <= equity * 0.05 + 1.0, (
            f"Notional {result.notional_usd} exceeded 5% of equity {equity}"
        )


# ---------------------------------------------------------------------------
# Property: After kill switch trigger, ZERO trades approved
# ---------------------------------------------------------------------------


@given(
    confidence=st.floats(min_value=0.01, max_value=1.0),
)
@settings(max_examples=200)
def test_kill_switch_blocks_all_trades(
    confidence: float,
) -> None:
    conn = sqlite3.connect(":memory:")
    equity = 50_000.0
    config = RiskConfig(
        enforce_trading_hours=False,
        min_volatility_ann=0.0,
        max_volatility_ann=10.0,
        daily_loss_limit=0.03,
    )
    rm = RiskManager(conn, initial_equity=equity, config=config)

    # Open a large position and lose >3% of equity ($1500+)
    # 0.5 BTC at 67000, exit at 64000 -> loss = 0.5 * 3000 = $1500 = 3% of $50k
    ts = int(datetime(2026, 3, 25, 14, 0, tzinfo=UTC).timestamp() * 1000)
    rm.on_trade_opened("long", 0.5, 67000.0, ts, 60000.0)
    rm.on_trade_closed(64000.0, ts + 300_000)  # $1500 loss = exactly 3%

    # Now try to trade — should be blocked by daily loss kill switch
    closes, highs, lows = _prices()
    signal = _signal(confidence=confidence)
    decision = rm.approve_trade(signal, closes, highs, lows, 99)
    assert not decision.approved, (
        "Trade approved after $1500 loss on $50k equity should be blocked by daily limit"
    )


# ---------------------------------------------------------------------------
# Property: Drawdown multiplier always in [0, 1]
# ---------------------------------------------------------------------------


@given(
    equity_changes=st.lists(
        st.floats(min_value=-0.20, max_value=0.10),
        min_size=5,
        max_size=50,
    ),
)
@settings(max_examples=300)
def test_drawdown_multiplier_always_bounded(
    equity_changes: list[float],
) -> None:
    conn = sqlite3.connect(":memory:")
    gate = DrawdownGate(conn, initial_equity=100_000.0)
    equity = 100_000.0

    for change in equity_changes:
        equity = max(1.0, equity * (1 + change))
        mult = gate.update(equity)
        assert 0.0 <= mult <= 1.0, f"Multiplier {mult} out of [0, 1] bounds at equity {equity}"


# ---------------------------------------------------------------------------
# Property: Kelly fraction always non-negative
# ---------------------------------------------------------------------------


@given(
    win_rate=st.floats(min_value=0.0, max_value=1.0),
    payoff=st.floats(min_value=0.01, max_value=10.0),
)
@settings(max_examples=300)
def test_kelly_fraction_never_negative(
    win_rate: float,
    payoff: float,
) -> None:
    sizer = PositionSizer()
    closes, highs, lows = _prices()
    result = sizer.compute(
        "long",
        0.80,
        50_000.0,
        67000.0,
        highs,
        lows,
        closes,
        current_idx=99,
        win_rate=win_rate,
        payoff_ratio=payoff,
    )
    assert result.raw_kelly_fraction >= 0.0


# ---------------------------------------------------------------------------
# Property: Risk per trade never exceeds max_risk_per_trade
# ---------------------------------------------------------------------------


@given(
    confidence=st.floats(min_value=0.01, max_value=1.0),
    equity=st.floats(min_value=5000.0, max_value=1_000_000.0),
    win_rate=st.floats(min_value=0.51, max_value=0.90),
)
@settings(max_examples=300)
def test_risk_per_trade_never_exceeds_cap(
    confidence: float,
    equity: float,
    win_rate: float,
) -> None:
    sizer = PositionSizer(max_risk_per_trade=0.01)
    closes, highs, lows = _prices()
    result = sizer.compute(
        "long",
        confidence,
        equity,
        67000.0,
        highs,
        lows,
        closes,
        current_idx=99,
        win_rate=win_rate,
        payoff_ratio=1.0,
    )
    if not result.rejected:
        max_risk_usd = 0.01 * equity
        assert result.risk_per_trade_usd <= max_risk_usd + 1.0, (
            f"Risk {result.risk_per_trade_usd} exceeded cap {max_risk_usd}"
        )
