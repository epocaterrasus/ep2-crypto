"""Risk manager: orchestrates all risk components for trade approval and bar updates.

This is the top-level class that the backtest engine and live trading loop
call. It wraps:
    - PositionTracker: position state and mark-to-market
    - KillSwitchManager: daily/weekly/drawdown/consecutive loss halts
    - DrawdownGate: progressive position size reduction
    - VolatilityGuard: vol range, trading hours, weekend, funding proximity
    - PositionSizer: Kelly sizing, ATR stops, max caps

Two main entry points:
    approve_trade(signal) -> TradeDecision
        Called before entering a trade. Returns approved/rejected with size.

    on_bar(bar) -> list[RiskAction]
        Called on every 5-min bar. Updates state, checks stops, may force exits.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import structlog

from ep2_crypto.risk.config import RiskConfig
from ep2_crypto.risk.drawdown_gate import DrawdownGate, DrawdownGateState
from ep2_crypto.risk.kill_switches import (
    KillSwitchManager,
    KillSwitchStatus,
    SwitchName,
)
from ep2_crypto.risk.position_sizer import PositionSizer, SizingResult
from ep2_crypto.risk.position_tracker import PositionSide, PositionState, PositionTracker
from ep2_crypto.risk.volatility_guard import VolatilityGuard, VolatilityGuardState

if TYPE_CHECKING:
    import sqlite3

    import numpy as np
    from numpy.typing import NDArray

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


class RiskActionType(Enum):
    NONE = "none"
    CLOSE_POSITION = "close_position"
    REDUCE_POSITION = "reduce_position"


@dataclass
class RiskAction:
    """An action the risk engine requires the execution layer to take."""

    action: RiskActionType
    reason: str
    urgency: str = "normal"  # "normal", "high", "immediate"


@dataclass
class TradeDecision:
    """Result of approve_trade(): whether to enter and at what size."""

    approved: bool
    quantity_btc: float = 0.0
    notional_usd: float = 0.0
    stop_price: float = 0.0
    position_fraction: float = 0.0
    reason: str = ""
    sizing_details: SizingResult | None = None

    # Breakdown of what each gate decided (for logging/debugging)
    kill_switch_ok: bool = True
    volatility_ok: bool = True
    drawdown_multiplier: float = 1.0
    vol_guard_state: VolatilityGuardState | None = None


@dataclass
class RiskState:
    """Complete snapshot of risk engine state for monitoring/health endpoints."""

    position: PositionState
    kill_switches: list[KillSwitchStatus]
    drawdown_gate: DrawdownGateState
    equity: float
    daily_pnl: float
    weekly_pnl: float
    consecutive_losses: int
    trades_today: int
    vol_guard: VolatilityGuardState | None = None


@dataclass
class SignalInput:
    """Input signal from the prediction model to the risk engine."""

    direction: str  # "long" or "short"
    confidence: float  # 0.0 to 1.0
    timestamp_ms: int
    win_rate: float | None = None  # historical, for Kelly
    payoff_ratio: float | None = None  # historical, for Kelly


# ---------------------------------------------------------------------------
# SQLite persistence for risk events
# ---------------------------------------------------------------------------

RISK_EVENT_TABLE = """
CREATE TABLE IF NOT EXISTS risk_events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp_ms    INTEGER NOT NULL,
    event_type      TEXT    NOT NULL,
    direction       TEXT,
    confidence      REAL,
    approved        INTEGER NOT NULL,
    quantity_btc    REAL,
    stop_price      REAL,
    reason          TEXT,
    equity          REAL,
    drawdown        REAL,
    vol_annualized  REAL
)
"""


# ---------------------------------------------------------------------------
# RiskManager
# ---------------------------------------------------------------------------


class RiskManager:
    """Orchestrates all risk components.

    Initialization requires:
        - SQLite connection (for state persistence)
        - Initial equity (capital at start)
        - Risk configuration (thresholds, sizing params)

    Example:
        rm = RiskManager(conn, initial_equity=50_000.0, config=risk_config)

        # Before each trade:
        decision = rm.approve_trade(signal, closes, highs, lows, idx)
        if decision.approved:
            execute_trade(decision)

        # On every bar:
        actions = rm.on_bar(bar_close, bar_high, bar_low, bar_ts, closes, highs, lows, idx)
        for action in actions:
            handle_action(action)
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        initial_equity: float,
        config: RiskConfig | None = None,
    ) -> None:
        if initial_equity <= 0:
            raise ValueError(f"initial_equity must be positive, got {initial_equity}")

        self._conn = conn
        self._equity = initial_equity
        self._initial_equity = initial_equity
        self._config = config or RiskConfig()

        # Ensure event table exists
        self._conn.execute(RISK_EVENT_TABLE)
        self._conn.commit()

        # Initialize sub-components
        self._position = PositionTracker(conn)
        self._kill_switches = KillSwitchManager(
            conn,
            thresholds={
                "daily_loss": self._config.daily_loss_limit,
                "weekly_loss": self._config.weekly_loss_limit,
                "max_drawdown": self._config.max_drawdown_halt,
                "consecutive_loss": float(self._config.consecutive_loss_limit),
                "emergency": 1.0,
            },
        )
        self._drawdown_gate = DrawdownGate(
            conn,
            initial_equity=initial_equity,
            halt_threshold=self._config.max_drawdown_halt,
            cooldown_bars=self._config.drawdown_cooldown_bars,
        )
        self._vol_guard = VolatilityGuard(
            min_vol_annualized=self._config.min_volatility_ann,
            max_vol_annualized=self._config.max_volatility_ann,
            trading_start_hour_utc=self._config.trading_start_hour_utc,
            trading_end_hour_utc=self._config.trading_end_hour_utc,
            weekend_reduction=self._config.weekend_size_reduction,
            enforce_trading_hours=self._config.enforce_trading_hours,
        )
        self._sizer = PositionSizer(
            kelly_fraction=self._config.kelly_fraction,
            max_position_fraction=self._config.max_position_fraction,
            min_btc_quantity=self._config.min_btc_quantity,
            atr_multiplier=self._config.atr_stop_multiplier,
            atr_period=self._config.atr_period,
            max_holding_bars=self._config.max_holding_bars,
        )

        # Tracking counters
        self._daily_pnl: float = 0.0
        self._weekly_pnl: float = 0.0
        self._consecutive_losses: int = 0
        self._trades_today: int = 0
        self._last_stop_price: float = 0.0

        logger.info(
            "risk_manager_initialized",
            initial_equity=initial_equity,
            daily_loss_limit=self._config.daily_loss_limit,
            weekly_loss_limit=self._config.weekly_loss_limit,
            max_drawdown=self._config.max_drawdown_halt,
            max_trades_per_day=self._config.max_trades_per_day,
        )

    # -- Main entry points ----------------------------------------------------

    def approve_trade(
        self,
        signal: SignalInput,
        closes: NDArray[np.float64],
        highs: NDArray[np.float64],
        lows: NDArray[np.float64],
        current_idx: int,
    ) -> TradeDecision:
        """Evaluate whether a trade signal should be executed.

        Runs all risk checks in order:
            1. Position check (already have one open?)
            2. Kill switches (any triggered?)
            3. Max trades per day
            4. Volatility guard (vol in range? hours ok?)
            5. Drawdown gate (get multiplier)
            6. Position sizing (Kelly * confidence * dd_mult * caps)
            7. Exposure check (within limits?)

        Args:
            signal: Direction, confidence, and optional Kelly inputs.
            closes: Full close price array.
            highs: Full high price array.
            lows: Full low price array.
            current_idx: Current bar index.

        Returns:
            TradeDecision with approval status, size, and stop price.
        """
        # Step 0: Check if position already open
        pos = self._position.get_state()
        if pos.is_open:
            return self._reject_trade(signal, "Position already open")

        # Step 1: Kill switches
        if self._kill_switches.any_triggered():
            triggered = self._kill_switches.get_triggered_names()
            return self._reject_trade(signal, f"Kill switch(es) triggered: {', '.join(triggered)}")

        # Step 2: Max trades per day
        if self._trades_today >= self._config.max_trades_per_day:
            return self._reject_trade(
                signal,
                f"Max trades per day reached ({self._trades_today}/"
                f"{self._config.max_trades_per_day})",
            )

        # Step 3: Volatility guard
        vol_state = self._vol_guard.check(closes, current_idx, signal.timestamp_ms)
        if not vol_state.can_trade:
            decision = self._reject_trade(
                signal,
                vol_state.rejection_reason or "Volatility guard rejected",
            )
            decision.vol_guard_state = vol_state
            decision.volatility_ok = False
            return decision

        # Step 4: Drawdown gate multiplier
        dd_mult = self._drawdown_gate.get_multiplier()
        if dd_mult <= 0.0:
            return self._reject_trade(
                signal,
                f"Drawdown gate halted trading (multiplier={dd_mult:.4f})",
            )

        # Step 5: Compute time multiplier (weekend + funding)
        time_mult = vol_state.weekend_multiplier * vol_state.funding_proximity_multiplier

        # Step 6: Position sizing
        current_price = float(closes[current_idx])
        sizing = self._sizer.compute(
            signal_direction=signal.direction,
            confidence=signal.confidence,
            equity=self._equity,
            current_price=current_price,
            highs=highs,
            lows=lows,
            closes=closes,
            current_idx=current_idx,
            drawdown_multiplier=dd_mult,
            time_multiplier=time_mult,
            win_rate=signal.win_rate,
            payoff_ratio=signal.payoff_ratio,
        )

        if sizing.rejected:
            return self._reject_trade(
                signal,
                sizing.rejection_reason or "Position sizer rejected",
            )

        # All checks passed
        decision = TradeDecision(
            approved=True,
            quantity_btc=sizing.quantity_btc,
            notional_usd=sizing.notional_usd,
            stop_price=sizing.stop_price,
            position_fraction=sizing.position_fraction,
            reason="All risk checks passed",
            sizing_details=sizing,
            kill_switch_ok=True,
            volatility_ok=True,
            drawdown_multiplier=dd_mult,
            vol_guard_state=vol_state,
        )

        self._log_event(
            event_type="trade_approved",
            direction=signal.direction,
            confidence=signal.confidence,
            approved=True,
            quantity_btc=sizing.quantity_btc,
            stop_price=sizing.stop_price,
            reason="all_checks_passed",
            vol_ann=vol_state.rolling_vol_annualized,
        )

        logger.info(
            "trade_approved",
            direction=signal.direction,
            confidence=round(signal.confidence, 4),
            quantity_btc=round(sizing.quantity_btc, 6),
            notional_usd=round(sizing.notional_usd, 2),
            stop_price=round(sizing.stop_price, 2),
            dd_multiplier=round(dd_mult, 4),
            vol_ann=round(vol_state.rolling_vol_annualized, 4),
        )

        return decision

    def on_trade_opened(
        self,
        side: str,
        quantity: float,
        entry_price: float,
        entry_time_ms: int,
        stop_price: float,
    ) -> None:
        """Notify risk engine that a trade was executed.

        Called by the execution layer AFTER the fill is confirmed.
        """
        self._position.open_position(side, quantity, entry_price, entry_time_ms)
        self._last_stop_price = stop_price
        self._trades_today += 1

    def on_trade_closed(
        self,
        exit_price: float,
        exit_time_ms: int,
    ) -> float:
        """Notify risk engine that a position was closed. Returns realized PnL."""
        pnl = self._position.close_position(exit_price, exit_time_ms)

        # Update tracking
        self._daily_pnl += pnl
        self._weekly_pnl += pnl
        self._equity += pnl

        # Update consecutive losses
        if pnl < 0:
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0

        # Notify drawdown gate
        self._drawdown_gate.on_trade_result(profitable=(pnl > 0))

        # Check kill switches after the trade
        self._check_all_kill_switches()

        logger.info(
            "trade_closed_risk_update",
            pnl=round(pnl, 2),
            equity=round(self._equity, 2),
            daily_pnl=round(self._daily_pnl, 2),
            consecutive_losses=self._consecutive_losses,
        )

        return pnl

    def on_bar(
        self,
        bar_close: float,
        bar_high: float,
        bar_low: float,
        bar_timestamp_ms: int,
        closes: NDArray[np.float64],
        highs: NDArray[np.float64],
        lows: NDArray[np.float64],
        current_idx: int,
    ) -> list[RiskAction]:
        """Process a new bar: update state, check stops, may force exits.

        Called on every 5-min bar by the backtest engine or live loop.

        Returns:
            List of actions for the execution layer (empty if no action needed).
        """
        actions: list[RiskAction] = []

        # 1. Mark position to market
        pos = self._position.mark_to_market(bar_close, bar_timestamp_ms)

        # 2. Update equity with unrealized PnL
        unrealized = pos.unrealized_pnl if pos.is_open else 0.0
        effective_equity = self._equity + unrealized

        # 3. Update drawdown gate
        self._drawdown_gate.update(effective_equity)

        # 4. Check kill switches
        self._check_all_kill_switches()
        if self._kill_switches.any_triggered() and pos.is_open:
            triggered = self._kill_switches.get_triggered_names()
            actions.append(
                RiskAction(
                    action=RiskActionType.CLOSE_POSITION,
                    reason=f"Kill switch triggered: {', '.join(triggered)}",
                    urgency="immediate",
                )
            )
            return actions

        if not pos.is_open:
            return actions

        # 5. Check max holding period
        if pos.bars_held >= self._sizer.max_holding_bars:
            actions.append(
                RiskAction(
                    action=RiskActionType.CLOSE_POSITION,
                    reason=(
                        f"Max holding period exceeded: {pos.bars_held} bars "
                        f">= {self._sizer.max_holding_bars}"
                    ),
                    urgency="high",
                )
            )
            return actions

        # 6. Check stop loss
        if self._last_stop_price > 0:
            long_stopped = pos.side == PositionSide.LONG and bar_low <= self._last_stop_price
            short_stopped = pos.side == PositionSide.SHORT and bar_high >= self._last_stop_price
            stopped = long_stopped or short_stopped

            if stopped:
                actions.append(
                    RiskAction(
                        action=RiskActionType.CLOSE_POSITION,
                        reason=(
                            f"Stop loss hit: stop={self._last_stop_price:.2f}, "
                            f"bar_low={bar_low:.2f}, bar_high={bar_high:.2f}"
                        ),
                        urgency="immediate",
                    )
                )
                return actions

        return actions

    # -- State access ---------------------------------------------------------

    def get_risk_state(self) -> RiskState:
        """Return complete risk engine state for monitoring."""
        pos = self._position.get_state()
        unrealized = pos.unrealized_pnl if pos.is_open else 0.0

        return RiskState(
            position=pos,
            kill_switches=self._kill_switches.get_all_status(),
            drawdown_gate=self._drawdown_gate.get_state(),
            equity=self._equity + unrealized,
            daily_pnl=self._daily_pnl + unrealized,
            weekly_pnl=self._weekly_pnl + unrealized,
            consecutive_losses=self._consecutive_losses,
            trades_today=self._trades_today,
        )

    def get_equity(self) -> float:
        """Return current equity (realized only, no unrealized)."""
        return self._equity

    def get_position(self) -> PositionState:
        """Return current position state."""
        return self._position.get_state()

    # -- Day/week boundary management -----------------------------------------

    def reset_daily_counters(self) -> None:
        """Call at day boundary (00:00 UTC) to reset daily limits."""
        logger.info(
            "daily_counters_reset",
            previous_daily_pnl=round(self._daily_pnl, 2),
            trades_today=self._trades_today,
        )
        self._daily_pnl = 0.0
        self._trades_today = 0
        self._position.reset_daily_pnl()

    def reset_weekly_counters(self) -> None:
        """Call at week boundary (Monday 00:00 UTC) to reset weekly limits."""
        logger.info(
            "weekly_counters_reset",
            previous_weekly_pnl=round(self._weekly_pnl, 2),
        )
        self._weekly_pnl = 0.0

    # -- Kill switch delegation -----------------------------------------------

    def reset_kill_switch(self, switch_name: SwitchName, reason: str) -> None:
        """Reset a specific kill switch. Requires reason string."""
        self._kill_switches.reset(switch_name, reason)

    def trigger_emergency(self, reason: str) -> None:
        """Trigger emergency kill switch."""
        self._kill_switches.trigger_emergency(reason, equity=self._equity)

    # -- Private helpers -------------------------------------------------------

    def _reject_trade(self, signal: SignalInput, reason: str) -> TradeDecision:
        """Create a rejected trade decision and log it."""
        self._log_event(
            event_type="trade_rejected",
            direction=signal.direction,
            confidence=signal.confidence,
            approved=False,
            quantity_btc=0.0,
            stop_price=0.0,
            reason=reason,
        )
        logger.info(
            "trade_rejected",
            direction=signal.direction,
            confidence=round(signal.confidence, 4),
            reason=reason,
        )
        return TradeDecision(approved=False, reason=reason)

    def _check_all_kill_switches(self) -> None:
        """Run all kill switch checks against current state."""
        self._kill_switches.check_daily_loss(
            daily_pnl_fraction=self._daily_pnl / self._initial_equity,
            equity=self._equity,
        )
        self._kill_switches.check_weekly_loss(
            weekly_pnl_fraction=self._weekly_pnl / self._initial_equity,
            equity=self._equity,
        )
        self._kill_switches.check_max_drawdown(
            drawdown_fraction=self._drawdown_gate.get_drawdown(),
            equity=self._equity,
        )
        self._kill_switches.check_consecutive_losses(
            consecutive_losses=self._consecutive_losses,
            equity=self._equity,
        )

    def _log_event(
        self,
        event_type: str,
        direction: str | None = None,
        confidence: float | None = None,
        approved: bool = False,
        quantity_btc: float | None = None,
        stop_price: float | None = None,
        reason: str | None = None,
        vol_ann: float | None = None,
    ) -> None:
        """Persist a risk event to SQLite."""
        now_ms = int(time.time() * 1000)
        dd = self._drawdown_gate.get_drawdown()
        self._conn.execute(
            "INSERT INTO risk_events "
            "(timestamp_ms, event_type, direction, confidence, approved, "
            "quantity_btc, stop_price, reason, equity, drawdown, vol_annualized) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                now_ms,
                event_type,
                direction,
                confidence,
                int(approved),
                quantity_btc,
                stop_price,
                reason,
                self._equity,
                dd,
                vol_ann,
            ),
        )
        self._conn.commit()
