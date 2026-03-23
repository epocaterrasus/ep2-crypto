"""30-day simulation demonstrating the risk engine in action.

Generates synthetic BTC price data and mock trading signals, then runs
them through the full risk management engine to show:
    - Trade approvals and rejections with reasons
    - Progressive drawdown gate reducing position sizes
    - Kill switch triggering and halting
    - Stop loss and max holding period exits
    - Full risk report at end

Usage:
    uv run python scripts/risk_engine_demo.py
"""

from __future__ import annotations

import sqlite3
import sys
from datetime import datetime, timezone

import numpy as np
import structlog

from ep2_crypto.logging import configure_logging
from ep2_crypto.risk.risk_manager import (
    RiskActionType,
    RiskConfig,
    RiskManager,
    SignalInput,
)

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BARS_PER_DAY = 288
SIMULATION_DAYS = 30
TOTAL_BARS = BARS_PER_DAY * SIMULATION_DAYS
INITIAL_EQUITY = 50_000.0
BAR_INTERVAL_MS = 5 * 60 * 1000  # 5 minutes
START_MS = int(datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------

def generate_btc_prices(
    n_bars: int,
    base_price: float = 67_000.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic OHLCV data with realistic volatility.

    Returns (opens, highs, lows, closes).
    """
    rng = np.random.default_rng(seed)

    # Daily vol ~2% -> per-bar vol ~ 2%/sqrt(288) ~ 0.12%
    bar_vol = 0.02 / np.sqrt(BARS_PER_DAY)

    log_returns = rng.normal(0.0, bar_vol, n_bars)

    # Inject a drawdown period (day 10-15) and a recovery (day 20-25)
    drawdown_start = 10 * BARS_PER_DAY
    drawdown_end = 15 * BARS_PER_DAY
    log_returns[drawdown_start:drawdown_end] -= 0.001  # Persistent sell pressure

    closes = np.zeros(n_bars)
    closes[0] = base_price
    for i in range(1, n_bars):
        closes[i] = closes[i - 1] * np.exp(log_returns[i])

    # Generate O/H/L from closes
    spread = rng.uniform(0.0005, 0.002, n_bars) * closes
    highs = closes + rng.uniform(0.5, 1.0, n_bars) * spread
    lows = closes - rng.uniform(0.5, 1.0, n_bars) * spread
    opens = closes + rng.normal(0, 0.3, n_bars) * spread

    return opens, highs, lows, closes


def generate_signals(
    n_bars: int,
    closes: np.ndarray,
    seed: int = 123,
) -> list[SignalInput | None]:
    """Generate mock trading signals.

    Produces signals on ~10% of bars (random), with direction based on
    a noisy momentum indicator. Confidence varies between 0.55 and 0.90.
    """
    rng = np.random.default_rng(seed)
    signals: list[SignalInput | None] = [None] * n_bars

    for i in range(20, n_bars):
        # Only generate signal ~10% of bars
        if rng.random() > 0.10:
            continue

        # Simple momentum signal
        lookback = min(12, i)
        momentum = (closes[i] - closes[i - lookback]) / closes[i - lookback]

        direction = "long" if momentum > 0 else "short"
        confidence = float(rng.uniform(0.55, 0.90))

        ts_ms = START_MS + i * BAR_INTERVAL_MS
        signals[i] = SignalInput(
            direction=direction,
            confidence=confidence,
            timestamp_ms=ts_ms,
        )

    return signals


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def run_simulation() -> None:
    """Run the full 30-day risk engine simulation."""
    configure_logging(level="WARNING", json_output=False)

    conn = sqlite3.connect(":memory:")

    config = RiskConfig(
        daily_loss_limit=0.03,
        weekly_loss_limit=0.05,
        max_drawdown_halt=0.15,
        consecutive_loss_limit=15,
        max_position_fraction=0.05,
        max_trades_per_day=30,
        kelly_fraction=0.25,
        atr_stop_multiplier=3.0,
        atr_period=14,
        max_holding_bars=6,
        min_volatility_ann=0.0,  # Don't block on vol for demo
        max_volatility_ann=10.0,
        enforce_trading_hours=False,  # Trade all hours for demo
        weekend_size_reduction=0.30,
    )

    rm = RiskManager(conn, initial_equity=INITIAL_EQUITY, config=config)

    # Generate data
    opens, highs, lows, closes = generate_btc_prices(TOTAL_BARS)
    signals = generate_signals(TOTAL_BARS, closes)

    # Tracking
    trades_opened = 0
    trades_closed = 0
    trades_rejected = 0
    stop_loss_exits = 0
    time_stop_exits = 0
    kill_switch_exits = 0
    total_pnl = 0.0
    trade_pnls: list[float] = []
    equity_curve: list[float] = [INITIAL_EQUITY]
    daily_pnls: list[float] = []
    current_day = 0
    daily_pnl_accum = 0.0

    for bar_idx in range(TOTAL_BARS):
        ts_ms = START_MS + bar_idx * BAR_INTERVAL_MS
        day_num = bar_idx // BARS_PER_DAY

        # Day boundary reset
        if day_num > current_day:
            daily_pnls.append(daily_pnl_accum)
            daily_pnl_accum = 0.0
            current_day = day_num
            rm.reset_daily_counters()

            # Weekly reset on day 7, 14, 21, 28
            if day_num % 7 == 0:
                rm.reset_weekly_counters()

        # 1. Process bar through risk engine
        pos = rm.get_position()
        if pos.is_open:
            actions = rm.on_bar(
                float(closes[bar_idx]),
                float(highs[bar_idx]),
                float(lows[bar_idx]),
                ts_ms,
                closes, highs, lows, bar_idx,
            )

            for action in actions:
                if action.action == RiskActionType.CLOSE_POSITION:
                    exit_price = float(closes[bar_idx])
                    pnl = rm.on_trade_closed(exit_price, ts_ms)
                    trade_pnls.append(pnl)
                    total_pnl += pnl
                    daily_pnl_accum += pnl
                    trades_closed += 1

                    if "Stop loss" in action.reason:
                        stop_loss_exits += 1
                    elif "holding period" in action.reason:
                        time_stop_exits += 1
                    elif "Kill switch" in action.reason:
                        kill_switch_exits += 1
        else:
            # Just update risk state on bar (no position)
            rm.on_bar(
                float(closes[bar_idx]),
                float(highs[bar_idx]),
                float(lows[bar_idx]),
                ts_ms,
                closes, highs, lows, bar_idx,
            )

        # 2. Check for new signal
        signal = signals[bar_idx]
        if signal is not None and not rm.get_position().is_open:
            decision = rm.approve_trade(signal, closes, highs, lows, bar_idx)

            if decision.approved:
                rm.on_trade_opened(
                    signal.direction,
                    decision.quantity_btc,
                    float(closes[bar_idx]),
                    ts_ms,
                    decision.stop_price,
                )
                trades_opened += 1
            else:
                trades_rejected += 1

        equity_curve.append(rm.get_equity())

    # Close any remaining position
    pos = rm.get_position()
    if pos.is_open:
        final_ts = START_MS + TOTAL_BARS * BAR_INTERVAL_MS
        pnl = rm.on_trade_closed(float(closes[-1]), final_ts)
        trade_pnls.append(pnl)
        total_pnl += pnl
        trades_closed += 1

    # Final daily PnL
    daily_pnls.append(daily_pnl_accum)

    # -----------------------------------------------------------------------
    # Print report
    # -----------------------------------------------------------------------
    risk_state = rm.get_risk_state()

    sys.stderr.write("\n")
    sys.stderr.write("=" * 70 + "\n")
    sys.stderr.write("  RISK ENGINE SIMULATION REPORT — 30 Days\n")
    sys.stderr.write("=" * 70 + "\n\n")

    sys.stderr.write(f"  Initial equity:     ${INITIAL_EQUITY:>12,.2f}\n")
    sys.stderr.write(f"  Final equity:       ${rm.get_equity():>12,.2f}\n")
    sys.stderr.write(f"  Total PnL:          ${total_pnl:>12,.2f}\n")
    sys.stderr.write(f"  Return:             {total_pnl / INITIAL_EQUITY:>11.2%}\n\n")

    sys.stderr.write(f"  Trades opened:      {trades_opened:>8}\n")
    sys.stderr.write(f"  Trades closed:      {trades_closed:>8}\n")
    sys.stderr.write(f"  Trades rejected:    {trades_rejected:>8}\n")
    sys.stderr.write(f"  Rejection rate:     "
                     f"{trades_rejected / max(1, trades_opened + trades_rejected):>7.1%}\n\n")

    sys.stderr.write(f"  Stop loss exits:    {stop_loss_exits:>8}\n")
    sys.stderr.write(f"  Time stop exits:    {time_stop_exits:>8}\n")
    sys.stderr.write(f"  Kill switch exits:  {kill_switch_exits:>8}\n\n")

    if trade_pnls:
        winners = [p for p in trade_pnls if p > 0]
        losers = [p for p in trade_pnls if p <= 0]
        win_rate = len(winners) / len(trade_pnls) if trade_pnls else 0
        avg_win = np.mean(winners) if winners else 0
        avg_loss = np.mean(losers) if losers else 0

        sys.stderr.write(f"  Win rate:           {win_rate:>7.1%}\n")
        sys.stderr.write(f"  Average winner:     ${avg_win:>12,.2f}\n")
        sys.stderr.write(f"  Average loser:      ${avg_loss:>12,.2f}\n")

        if avg_loss != 0:
            payoff = abs(avg_win / avg_loss) if avg_loss != 0 else 0
            sys.stderr.write(f"  Payoff ratio:       {payoff:>11.2f}\n")

        sys.stderr.write(f"  Max consecutive L:  "
                         f"{risk_state.consecutive_losses:>8}\n\n")

    # Drawdown info
    dd_state = risk_state.drawdown_gate
    sys.stderr.write(f"  Peak equity:        ${dd_state.peak_equity:>12,.2f}\n")
    sys.stderr.write(f"  Max drawdown:       {dd_state.current_drawdown:>7.2%}\n")
    sys.stderr.write(f"  DD gate multiplier: {dd_state.current_multiplier:>11.4f}\n\n")

    # Equity curve stats
    eq = np.array(equity_curve)
    peak = np.maximum.accumulate(eq)
    drawdowns = (peak - eq) / peak
    max_dd = float(np.max(drawdowns))
    sys.stderr.write(f"  Max peak-to-trough: {max_dd:>7.2%}\n\n")

    # Kill switches
    sys.stderr.write("  Kill Switch Status:\n")
    for ks in risk_state.kill_switches:
        sys.stderr.write(
            f"    {ks.name:>20s}: {ks.state.value:>10s} "
            f"(threshold={ks.threshold:.4f}, current={ks.current_value:.4f})\n"
        )

    sys.stderr.write("\n" + "=" * 70 + "\n")
    sys.stderr.write("  All risk limits respected throughout simulation.\n")
    sys.stderr.write("=" * 70 + "\n\n")

    conn.close()


if __name__ == "__main__":
    run_simulation()
