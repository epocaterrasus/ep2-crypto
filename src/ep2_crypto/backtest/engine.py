"""Event-driven backtest engine.

Processes OHLCV bars sequentially with:
  - Next-bar-open execution (signal at bar t → fill at open of bar t+1)
  - Full risk engine integration (approve_trade, on_bar, kill switches)
  - Execution simulator (slippage, fees, partial fill, funding)
  - Per-bar state persistence (equity, position, margin)
  - Numba-compiled inner loop for performance-critical path

Usage:
    engine = BacktestEngine(initial_equity=50_000.0)
    result = engine.run(ohlcv_data, signals, timestamps)
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

import numba
import numpy as np
import structlog
from numpy.typing import NDArray

from ep2_crypto.backtest.metrics import (
    BacktestResult,
    TradeRecord,
    compute_backtest_result,
)
from ep2_crypto.backtest.simulator import ExecutionSimulator, FundingAccumulator
from ep2_crypto.risk.config import RiskConfig
from ep2_crypto.risk.risk_manager import (
    RiskAction,
    RiskActionType,
    RiskManager,
    SignalInput,
)

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Numba-compiled inner loop for equity tracking
# ---------------------------------------------------------------------------
@numba.njit(cache=True)
def _update_equity_numba(
    equity: float,
    position_qty: float,
    position_side: int,  # 1=long, -1=short, 0=flat
    bar_close: float,
    prev_close: float,
) -> float:
    """Update equity from bar price movement (Numba-compiled).

    Args:
        equity: Current equity in USD.
        position_qty: Absolute position size in BTC.
        position_side: +1 long, -1 short, 0 flat.
        bar_close: Current bar close price.
        prev_close: Previous bar close price.

    Returns:
        Updated equity.
    """
    if position_side == 0 or position_qty <= 0 or prev_close <= 0:
        return equity
    price_change = bar_close - prev_close
    pnl = position_qty * price_change * position_side
    return equity + pnl


@numba.njit(cache=True)
def _compute_bar_returns_numba(
    closes: numba.float64[:],
    positions: numba.float64[:],
    n: int,
) -> numba.float64[:]:
    """Compute per-bar returns from close prices and position sizes (Numba).

    positions[i] = signed quantity (+ for long, - for short)
    """
    returns = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        if closes[i - 1] > 0 and positions[i] != 0.0:
            price_ret = (closes[i] - closes[i - 1]) / closes[i - 1]
            returns[i] = price_ret * np.sign(positions[i])
    return returns


# ---------------------------------------------------------------------------
# Engine configuration
# ---------------------------------------------------------------------------
@dataclass
class BacktestConfig:
    """Configuration for the backtest engine."""

    initial_equity: float = 50_000.0
    risk_config: RiskConfig | None = None

    # Execution
    seed: int = 42

    # Signal interpretation
    confidence_threshold: float = 0.55
    default_win_rate: float = 0.53
    default_payoff_ratio: float = 1.1

    # Funding rate
    default_funding_rate: float = 0.0001  # 0.01% per 8h


# ---------------------------------------------------------------------------
# Per-bar state snapshot
# ---------------------------------------------------------------------------
@dataclass
class BarState:
    """State snapshot at a single bar."""

    bar_idx: int
    timestamp_ms: int
    equity: float
    position_side: int  # +1, -1, 0
    position_qty: float
    position_entry_price: float
    unrealized_pnl: float
    cumulative_fees: float
    cumulative_slippage: float
    cumulative_funding: float


# ---------------------------------------------------------------------------
# Backtest Engine
# ---------------------------------------------------------------------------
class BacktestEngine:
    """Event-driven backtest engine with risk management integration.

    Flow per bar:
      1. Risk engine on_bar() — check stops, update state
      2. If signal present and no position: approve_trade() → execute
      3. If risk action requires close: execute exit
      4. Update equity from mark-to-market
      5. Record state snapshot
    """

    def __init__(self, config: BacktestConfig | None = None) -> None:
        self._config = config or BacktestConfig()
        self._simulator = ExecutionSimulator(seed=self._config.seed)
        self._funding = FundingAccumulator()

    def run(
        self,
        opens: NDArray[np.float64],
        highs: NDArray[np.float64],
        lows: NDArray[np.float64],
        closes: NDArray[np.float64],
        volumes: NDArray[np.float64],
        timestamps_ms: NDArray[np.int64],
        signals: NDArray[np.int8] | None = None,
        confidences: NDArray[np.float64] | None = None,
        funding_rates: NDArray[np.float64] | None = None,
        regime_labels: NDArray[np.int32] | None = None,
    ) -> BacktestResult:
        """Run backtest over OHLCV data.

        Args:
            opens, highs, lows, closes, volumes: OHLCV arrays (same length).
            timestamps_ms: Bar timestamps in milliseconds.
            signals: Per-bar signal: +1=long, -1=short, 0=no signal.
            confidences: Per-bar confidence [0,1].
            funding_rates: Per-bar funding rate (applied at settlement times).
            regime_labels: Per-bar regime label (for breakdown metrics).

        Returns:
            BacktestResult with comprehensive metrics.
        """
        n = len(closes)
        if n < 2:
            return compute_backtest_result(np.array([]))

        # Default signals: no signal
        if signals is None:
            signals = np.zeros(n, dtype=np.int8)
        if confidences is None:
            confidences = np.full(n, 0.6)
        if funding_rates is None:
            funding_rates = np.full(n, self._config.default_funding_rate)

        # Initialize risk manager with in-memory SQLite
        conn = sqlite3.connect(":memory:")
        risk_config = self._config.risk_config or RiskConfig()
        risk_mgr = RiskManager(
            conn=conn,
            initial_equity=self._config.initial_equity,
            config=risk_config,
        )

        # State tracking
        equity = self._config.initial_equity
        per_bar_returns = np.zeros(n, dtype=np.float64)
        position_side: int = 0  # +1, -1, 0
        position_qty: float = 0.0
        position_entry_price: float = 0.0
        position_entry_bar: int = 0

        # Cost accumulators
        total_fees: float = 0.0
        total_slippage: float = 0.0
        total_funding: float = 0.0

        # Trade records
        trades: list[TradeRecord] = []

        # Pending signal (for next-bar-open execution)
        pending_signal: int = 0
        pending_confidence: float = 0.0

        # Bar-by-bar processing
        for i in range(1, n):
            bar_close = closes[i]
            bar_open = opens[i]
            bar_high = highs[i]
            bar_low = lows[i]
            bar_vol_usd = float(volumes[i] * closes[i])
            bar_ts = int(timestamps_ms[i])
            prev_close = closes[i - 1]

            # 1. Execute pending signal at this bar's open
            if pending_signal != 0 and position_side == 0:
                direction = "long" if pending_signal > 0 else "short"
                signal_input = SignalInput(
                    direction=direction,
                    confidence=pending_confidence,
                    timestamp_ms=bar_ts,
                    win_rate=self._config.default_win_rate,
                    payoff_ratio=self._config.default_payoff_ratio,
                )

                decision = risk_mgr.approve_trade(
                    signal_input, closes[:i + 1], highs[:i + 1],
                    lows[:i + 1], i,
                )

                if decision.approved and decision.quantity_btc > 0:
                    result = self._simulator.simulate_entry(
                        side=direction,
                        desired_quantity=decision.quantity_btc,
                        price=bar_open,
                        bar_volume_usd=bar_vol_usd,
                    )
                    if result.executed:
                        position_side = 1 if direction == "long" else -1
                        position_qty = result.fill_quantity
                        position_entry_price = result.fill_price
                        position_entry_bar = i
                        total_fees += result.fee_usd
                        total_slippage += result.slippage_bps * result.fill_notional_usd * 1e-4

                        risk_mgr.on_trade_opened(
                            side=direction,
                            quantity=result.fill_quantity,
                            entry_price=result.fill_price,
                            entry_time_ms=bar_ts,
                            stop_price=decision.stop_price,
                        )

            pending_signal = 0
            pending_confidence = 0.0

            # 2. Risk engine on_bar — check stops, holding limits
            risk_actions: list[RiskAction] = []
            if position_side != 0:
                risk_actions = risk_mgr.on_bar(
                    bar_close, bar_high, bar_low, bar_ts,
                    closes[:i + 1], highs[:i + 1], lows[:i + 1], i,
                )

            # 3. Handle risk actions (forced exits)
            force_close = any(
                a.action == RiskActionType.CLOSE_POSITION
                for a in risk_actions
            )

            if force_close and position_side != 0:
                side_str = "long" if position_side > 0 else "short"
                exit_result = self._simulator.simulate_exit(
                    side=side_str,
                    quantity=position_qty,
                    price=bar_close,
                    bar_volume_usd=bar_vol_usd,
                )
                if exit_result.executed:
                    pnl = self._compute_trade_pnl(
                        position_side, position_qty,
                        position_entry_price, exit_result.fill_price,
                    )
                    equity += pnl - exit_result.fee_usd
                    total_fees += exit_result.fee_usd
                    slip_usd = exit_result.slippage_bps * exit_result.fill_notional_usd * 1e-4
                    total_slippage += slip_usd

                    risk_mgr.on_trade_closed(exit_result.fill_price, bar_ts)

                    trades.append(TradeRecord(
                        entry_bar=position_entry_bar,
                        exit_bar=i,
                        side=side_str,
                        entry_price=position_entry_price,
                        exit_price=exit_result.fill_price,
                        quantity=position_qty,
                        pnl_usd=pnl,
                        return_pct=pnl / (position_qty * position_entry_price),
                        bars_held=i - position_entry_bar,
                        exit_cost_bps=exit_result.total_cost_bps,
                    ))

                    position_side = 0
                    position_qty = 0.0
                    position_entry_price = 0.0

            # 4. Funding rate at settlement times
            if position_side != 0 and i > 0:
                prev_ts = int(timestamps_ms[i - 1])
                if self._funding.is_settlement_bar(bar_ts, prev_ts):
                    notional = position_qty * bar_close
                    is_long = position_side > 0
                    payment = self._funding.funding_payment(
                        notional, is_long, float(funding_rates[i]),
                    )
                    equity -= payment
                    total_funding += payment

            # 5. Mark-to-market equity update
            if position_side != 0 and prev_close > 0:
                equity_before = equity
                equity = _update_equity_numba(
                    equity, position_qty, position_side,
                    bar_close, prev_close,
                )
                if equity_before > 0:
                    per_bar_returns[i] = (equity - equity_before) / equity_before

            # 6. Record new signal for next-bar execution
            above_threshold = confidences[i] >= self._config.confidence_threshold
            if signals[i] != 0 and position_side == 0 and above_threshold:
                pending_signal = int(signals[i])
                pending_confidence = float(confidences[i])

            # 7. Check if we should exit on signal reversal
            if position_side != 0 and signals[i] != 0:
                signal_dir = 1 if signals[i] > 0 else -1
                if signal_dir != position_side:
                    # Signal reversal → close position at next bar
                    side_str = "long" if position_side > 0 else "short"
                    exit_result = self._simulator.simulate_exit(
                        side=side_str,
                        quantity=position_qty,
                        price=bar_close,
                        bar_volume_usd=bar_vol_usd,
                    )
                    if exit_result.executed:
                        pnl = self._compute_trade_pnl(
                            position_side, position_qty,
                            position_entry_price, exit_result.fill_price,
                        )
                        equity += pnl - exit_result.fee_usd
                        total_fees += exit_result.fee_usd

                        risk_mgr.on_trade_closed(exit_result.fill_price, bar_ts)

                        trades.append(TradeRecord(
                            entry_bar=position_entry_bar,
                            exit_bar=i,
                            side=side_str,
                            entry_price=position_entry_price,
                            exit_price=exit_result.fill_price,
                            quantity=position_qty,
                            pnl_usd=pnl,
                            return_pct=pnl / (position_qty * position_entry_price),
                            bars_held=i - position_entry_bar,
                        ))

                        position_side = 0
                        position_qty = 0.0
                        position_entry_price = 0.0

                        # Queue opposite signal
                        if confidences[i] >= self._config.confidence_threshold:
                            pending_signal = int(signals[i])
                            pending_confidence = float(confidences[i])

        # Close any remaining position at last bar
        if position_side != 0:
            side_str = "long" if position_side > 0 else "short"
            exit_result = self._simulator.simulate_exit(
                side=side_str,
                quantity=position_qty,
                price=closes[-1],
                bar_volume_usd=float(volumes[-1] * closes[-1]),
            )
            if exit_result.executed:
                pnl = self._compute_trade_pnl(
                    position_side, position_qty,
                    position_entry_price, exit_result.fill_price,
                )
                equity += pnl - exit_result.fee_usd
                total_fees += exit_result.fee_usd

                trades.append(TradeRecord(
                    entry_bar=position_entry_bar,
                    exit_bar=n - 1,
                    side=side_str,
                    entry_price=position_entry_price,
                    exit_price=exit_result.fill_price,
                    quantity=position_qty,
                    pnl_usd=pnl,
                    return_pct=pnl / (position_qty * position_entry_price),
                    bars_held=n - 1 - position_entry_bar,
                ))

        conn.close()

        # Compute comprehensive metrics
        result = compute_backtest_result(
            returns=per_bar_returns,
            trades=trades,
            regime_labels=regime_labels,
            total_fee_usd=total_fees,
            total_slippage_usd=total_slippage,
            total_funding_usd=total_funding,
        )

        logger.info(
            "backtest_complete",
            n_bars=n,
            n_trades=len(trades),
            sharpe=result.sharpe_ratio,
            total_return=result.total_return,
            max_drawdown=result.max_drawdown,
            total_costs=result.total_cost_usd,
        )

        return result

    @staticmethod
    def _compute_trade_pnl(
        side: int,
        quantity: float,
        entry_price: float,
        exit_price: float,
    ) -> float:
        """Compute PnL for a trade (excluding costs)."""
        return quantity * (exit_price - entry_price) * side
