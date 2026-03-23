"""Golden dataset tests for the risk engine.

20+ parameterized cases with known inputs -> expected outputs.
These NEVER change — regression protection.
"""

from __future__ import annotations

import math
import sqlite3
from datetime import UTC, datetime

import numpy as np
import pytest

from ep2_crypto.risk.config import RiskConfig
from ep2_crypto.risk.drawdown_gate import CONVEX_K, DrawdownGate
from ep2_crypto.risk.kill_switches import KillSwitchManager
from ep2_crypto.risk.position_sizer import PositionSizer
from ep2_crypto.risk.risk_manager import RiskManager, SignalInput

# ---------------------------------------------------------------------------
# Helper: price arrays
# ---------------------------------------------------------------------------

def _prices(n: int = 300, base: float = 67000.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    closes = base + np.cumsum(rng.normal(0, 50, n))
    highs = closes + rng.uniform(30, 150, n)
    lows = closes - rng.uniform(30, 150, n)
    return closes, highs, lows


def _signal(
    direction: str = "long",
    confidence: float = 0.80,
    win_rate: float | None = None,
    payoff_ratio: float | None = None,
) -> SignalInput:
    ts = int(datetime(2026, 3, 25, 14, 0, tzinfo=UTC).timestamp() * 1000)
    return SignalInput(
        direction=direction,
        confidence=confidence,
        timestamp_ms=ts,
        win_rate=win_rate,
        payoff_ratio=payoff_ratio,
    )


def _rm(conn: sqlite3.Connection, equity: float = 50_000.0, **overrides: object) -> RiskManager:
    defaults = {
        "enforce_trading_hours": False,
        "min_volatility_ann": 0.0,
        "max_volatility_ann": 10.0,
    }
    defaults.update(overrides)
    config = RiskConfig(**defaults)
    return RiskManager(conn, initial_equity=equity, config=config)


# ---------------------------------------------------------------------------
# Golden: Convex drawdown formula at exact percentages
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "dd_pct, expected_mult",
    [
        (0.0, 1.0),
        (3.0, math.pow(1.0 - 0.03 / 0.15, CONVEX_K)),   # ~0.7155
        (5.0, math.pow(1.0 - 0.05 / 0.15, CONVEX_K)),   # ~0.5443
        (8.0, math.pow(1.0 - 0.08 / 0.15, CONVEX_K)),   # ~0.3200
        (10.0, math.pow(1.0 - 0.10 / 0.15, CONVEX_K)),  # ~0.1925
        (12.0, math.pow(1.0 - 0.12 / 0.15, CONVEX_K)),  # ~0.0894
        (15.0, 0.0),
        (20.0, 0.0),
    ],
    ids=["dd=0%", "dd=3%", "dd=5%", "dd=8%", "dd=10%", "dd=12%", "dd=15%", "dd=20%"],
)
def test_golden_convex_drawdown(dd_pct: float, expected_mult: float) -> None:
    conn = sqlite3.connect(":memory:")
    gate = DrawdownGate(conn, initial_equity=100_000.0)
    gate.update(100_000.0)
    equity = 100_000.0 * (1 - dd_pct / 100)
    mult = gate.update(equity)
    assert mult == pytest.approx(expected_mult, abs=0.01)


# ---------------------------------------------------------------------------
# Golden: Kelly fraction computation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "win_rate, payoff, expected_kelly",
    [
        (0.55, 1.0, 0.10),   # f = (0.55*1 - 0.45)/1 = 0.10
        (0.52, 1.0, 0.04),   # f = (0.52 - 0.48)/1 = 0.04
        (0.60, 1.5, 0.3333), # f = (0.60*1.5 - 0.40)/1.5 = 0.3333
        (0.50, 1.0, 0.0),    # no edge
        (0.45, 1.0, 0.0),    # negative edge -> 0
        (0.55, 2.0, 0.325),  # f = (0.55*2 - 0.45)/2 = 0.325
    ],
    ids=["55/1.0", "52/1.0", "60/1.5", "50/1.0", "45/1.0", "55/2.0"],
)
def test_golden_kelly_fraction(win_rate: float, payoff: float, expected_kelly: float) -> None:
    sizer = PositionSizer()
    result = sizer.compute(
        "long", 0.80, 50_000.0, 67000.0,
        *_prices()[1::-1], _prices()[0],  # highs, lows, closes
        current_idx=299,
        win_rate=win_rate, payoff_ratio=payoff,
    )
    if expected_kelly <= 0:
        assert result.rejected
    else:
        assert not result.rejected
        assert result.raw_kelly_fraction == pytest.approx(expected_kelly, abs=0.01)


# ---------------------------------------------------------------------------
# Golden: Full approve_trade decisions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "equity, confidence, direction, should_approve, reason_contains",
    [
        # Normal conditions
        (50_000.0, 0.80, "long", True, "passed"),
        (50_000.0, 0.80, "short", True, "passed"),
        # Low confidence still approved (sizer handles scaling)
        (50_000.0, 0.30, "long", True, "passed"),
        # Zero confidence rejected
        (50_000.0, 0.0, "long", False, ""),
        # Invalid direction
        (50_000.0, 0.80, "sideways", False, ""),
        # Zero equity
        (0.0, 0.80, "long", False, ""),
    ],
    ids=[
        "normal_long", "normal_short", "low_conf",
        "zero_conf", "invalid_dir", "zero_equity",
    ],
)
def test_golden_approve_trade(
    equity: float,
    confidence: float,
    direction: str,
    should_approve: bool,
    reason_contains: str,
) -> None:
    conn = sqlite3.connect(":memory:")
    if equity <= 0:
        # RiskManager rejects non-positive equity at construction
        with pytest.raises(ValueError):
            _rm(conn, equity=equity)
        return
    rm = _rm(conn, equity=equity)
    closes, highs, lows = _prices()
    signal = _signal(direction=direction, confidence=confidence)
    decision = rm.approve_trade(signal, closes, highs, lows, len(closes) - 1)
    assert decision.approved == should_approve
    if should_approve:
        assert decision.quantity_btc > 0
        assert decision.stop_price > 0


# ---------------------------------------------------------------------------
# Golden: Kill switch exact thresholds
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "switch, check_value, should_trigger",
    [
        ("daily_loss", -0.029, False),   # Just below 3%
        ("daily_loss", -0.030, True),    # Exactly 3%
        ("daily_loss", -0.031, True),    # Just above 3%
        ("weekly_loss", -0.049, False),  # Below 5%
        ("weekly_loss", -0.050, True),   # Exactly 5%
        ("max_drawdown", 0.149, False),  # Below 15%
        ("max_drawdown", 0.150, True),   # Exactly 15%
        ("consecutive_loss", 14, False), # Below 15
        ("consecutive_loss", 15, True),  # Exactly 15
    ],
    ids=[
        "daily_below", "daily_exact", "daily_above",
        "weekly_below", "weekly_exact",
        "dd_below", "dd_exact",
        "consec_below", "consec_exact",
    ],
)
def test_golden_kill_switch_thresholds(
    switch: str,
    check_value: float,
    should_trigger: bool,
) -> None:
    conn = sqlite3.connect(":memory:")
    thresholds = {
        "daily_loss": 0.03,
        "weekly_loss": 0.05,
        "max_drawdown": 0.15,
        "consecutive_loss": 15.0,
        "emergency": 1.0,
    }
    ks = KillSwitchManager(conn, thresholds)

    if switch == "daily_loss":
        result = ks.check_daily_loss(check_value)
    elif switch == "weekly_loss":
        result = ks.check_weekly_loss(check_value)
    elif switch == "max_drawdown":
        result = ks.check_max_drawdown(check_value)
    elif switch == "consecutive_loss":
        result = ks.check_consecutive_losses(int(check_value))
    else:
        raise ValueError(f"Unknown switch: {switch}")

    assert result == should_trigger


# ---------------------------------------------------------------------------
# Golden: Position size never exceeds 5% cap
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "equity, confidence, win_rate, payoff",
    [
        (100_000.0, 1.0, 0.80, 2.0),   # Very high edge
        (100_000.0, 1.0, 0.70, 1.5),   # High edge
        (50_000.0, 0.95, 0.60, 1.2),   # Moderate edge
        (10_000.0, 0.80, 0.55, 1.0),   # Small account
    ],
    ids=["high_edge_100k", "med_edge_100k", "mod_edge_50k", "small_10k"],
)
def test_golden_position_cap(
    equity: float, confidence: float, win_rate: float, payoff: float,
) -> None:
    sizer = PositionSizer()
    closes, highs, lows = _prices()
    result = sizer.compute(
        "long", confidence, equity, 67000.0,
        highs, lows, closes, current_idx=299,
        win_rate=win_rate, payoff_ratio=payoff,
    )
    assert not result.rejected
    assert result.position_fraction <= 0.05 + 1e-9
    assert result.notional_usd <= equity * 0.05 + 1.0


# ---------------------------------------------------------------------------
# Golden: Stop loss direction
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "direction, price",
    [("long", 67000.0), ("short", 67000.0)],
    ids=["long_stop_below", "short_stop_above"],
)
def test_golden_stop_direction(direction: str, price: float) -> None:
    sizer = PositionSizer()
    closes, highs, lows = _prices(base=price)
    result = sizer.compute(
        direction, 0.80, 50_000.0, price,
        highs, lows, closes, current_idx=299,
    )
    assert not result.rejected
    if direction == "long":
        assert result.stop_price < price
    else:
        assert result.stop_price > price
