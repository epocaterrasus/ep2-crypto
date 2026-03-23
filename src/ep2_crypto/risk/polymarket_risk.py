"""Risk adapter for Polymarket binary prediction markets.

Reuses the existing kill switch and drawdown gate infrastructure but
replaces the perps-oriented position sizing and stop-loss logic with
binary-outcome-specific equivalents.

Key differences from perps risk management:
- No margin, no liquidation, no funding rate costs
- Max loss is known upfront (= cost of shares)
- No stop-loss needed (market resolves in 5 min)
- No holding period limit (5-min resolution is the exit)
- Position sizing via binary Kelly, not ATR-based
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

import structlog

from ep2_crypto.risk.binary_position_sizer import BinaryPositionSizer, BinarySizingResult
from ep2_crypto.risk.drawdown_gate import DrawdownGate
from ep2_crypto.risk.kill_switches import KillSwitchManager, SwitchName

if TYPE_CHECKING:
    import sqlite3

    from ep2_crypto.risk.config import RiskConfig

logger = structlog.get_logger(__name__)


class BinaryDirection(StrEnum):
    UP = "up"
    DOWN = "down"


@dataclass
class BinarySignal:
    """Signal from the ML pipeline for a binary prediction market."""

    direction: BinaryDirection  # Model's predicted direction
    model_prob: float  # Model's estimated probability (0-1)
    market_price: float  # Current market price of the share (0-1)
    timestamp_ms: int
    window_end_ms: int  # When the 5-min window resolves


@dataclass
class BinaryTradeDecision:
    """Decision on whether to enter a binary bet."""

    approved: bool
    direction: BinaryDirection = BinaryDirection.UP
    shares: float = 0.0
    cost_usd: float = 0.0
    max_loss_usd: float = 0.0
    expected_value: float = 0.0
    reason: str = ""
    sizing: BinarySizingResult | None = None

    # Gate breakdown
    kill_switch_ok: bool = True
    drawdown_multiplier: float = 1.0


class PolymarketRiskAdapter:
    """Risk management for Polymarket binary prediction markets.

    Reuses kill switches and drawdown gate from the perps risk engine.
    Replaces position sizing with binary Kelly criterion.
    No margin, stop-loss, or volatility guard needed.
    """

    def __init__(
        self,
        config: RiskConfig,
        initial_equity: float,
        conn: sqlite3.Connection,
        sizer: BinaryPositionSizer | None = None,
    ) -> None:
        self._config = config
        self._equity = initial_equity
        self._peak_equity = initial_equity
        self._daily_pnl: float = 0.0
        self._weekly_pnl: float = 0.0
        self._consecutive_losses: int = 0
        self._trades_today: int = 0
        self._position_open: bool = False
        self._conn = conn

        # Reuse existing infrastructure
        thresholds: dict[SwitchName, float] = {
            "daily_loss": config.daily_loss_limit,
            "weekly_loss": config.weekly_loss_limit,
            "max_drawdown": config.max_drawdown_halt,
            "consecutive_loss": float(config.consecutive_loss_limit),
            "emergency": 1.0,
        }
        self._kill_switches = KillSwitchManager(conn=conn, thresholds=thresholds)
        self._drawdown_gate = DrawdownGate(
            conn=conn,
            initial_equity=initial_equity,
            halt_threshold=config.max_drawdown_halt,
            cooldown_bars=config.drawdown_cooldown_bars,
        )

        # Binary-specific sizer
        self._sizer = sizer or BinaryPositionSizer()

        logger.info(
            "polymarket_risk_initialized",
            initial_equity=initial_equity,
            daily_loss_limit=config.daily_loss_limit,
            max_drawdown=config.max_drawdown_halt,
        )

    @property
    def equity(self) -> float:
        return self._equity

    @property
    def daily_pnl(self) -> float:
        return self._daily_pnl

    @property
    def consecutive_losses(self) -> int:
        return self._consecutive_losses

    @property
    def position_open(self) -> bool:
        return self._position_open

    def approve_trade(self, signal: BinarySignal) -> BinaryTradeDecision:
        """Evaluate whether a binary bet should be placed.

        Checks:
            1. Position already open? (one at a time)
            2. Kill switches triggered?
            3. Max trades per day?
            4. Drawdown gate multiplier
            5. Binary position sizing (Kelly)

        Args:
            signal: Direction, model probability, and market price.

        Returns:
            BinaryTradeDecision with approval and sizing details.
        """
        # Step 1: Position check
        if self._position_open:
            return self._reject(signal, "Position already open")

        # Step 2: Kill switches
        if self._kill_switches.any_triggered():
            triggered = self._kill_switches.get_triggered_names()
            decision = self._reject(
                signal,
                f"Kill switch(es) triggered: {', '.join(triggered)}",
            )
            decision.kill_switch_ok = False
            return decision

        # Step 3: Max trades per day
        if self._trades_today >= self._config.max_trades_per_day:
            return self._reject(
                signal,
                f"Max trades per day ({self._trades_today}/{self._config.max_trades_per_day})",
            )

        # Step 4: Drawdown gate
        dd_mult = self._drawdown_gate.get_multiplier()
        if dd_mult <= 0.0:
            return self._reject(
                signal,
                f"Drawdown gate halted trading (multiplier={dd_mult:.4f})",
            )

        # Step 5: Binary position sizing
        sizing = self._sizer.compute(
            model_prob=signal.model_prob,
            market_price=signal.market_price,
            bankroll=self._equity,
            drawdown_multiplier=dd_mult,
        )

        if sizing.rejected:
            return self._reject(
                signal,
                sizing.rejection_reason or "Position sizer rejected",
            )

        decision = BinaryTradeDecision(
            approved=True,
            direction=signal.direction,
            shares=sizing.shares,
            cost_usd=sizing.cost_usd,
            max_loss_usd=sizing.max_loss_usd,
            expected_value=sizing.expected_value,
            reason="All risk checks passed",
            sizing=sizing,
            kill_switch_ok=True,
            drawdown_multiplier=dd_mult,
        )

        logger.info(
            "binary_trade_approved",
            direction=signal.direction.value,
            model_prob=round(signal.model_prob, 4),
            market_price=round(signal.market_price, 4),
            shares=round(sizing.shares, 2),
            cost_usd=round(sizing.cost_usd, 2),
            dd_mult=round(dd_mult, 4),
        )

        return decision

    def on_trade_opened(self) -> None:
        """Notify that a binary bet was placed."""
        self._position_open = True
        self._trades_today += 1

    def on_trade_resolved(self, won: bool, pnl: float) -> None:
        """Notify that a binary bet resolved.

        Args:
            won: Whether the bet was correct.
            pnl: Realized PnL (positive if won, negative = -cost if lost).
        """
        self._position_open = False
        self._daily_pnl += pnl
        self._weekly_pnl += pnl
        self._equity += pnl

        # Update peak equity
        if self._equity > self._peak_equity:
            self._peak_equity = self._equity

        # Consecutive losses
        if pnl < 0:
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0

        # Drawdown gate
        self._drawdown_gate.update(self._equity)
        self._drawdown_gate.on_trade_result(profitable=won)

        # Kill switches
        self._check_kill_switches()

        logger.info(
            "binary_trade_resolved",
            won=won,
            pnl=round(pnl, 2),
            equity=round(self._equity, 2),
            daily_pnl=round(self._daily_pnl, 2),
            consecutive_losses=self._consecutive_losses,
        )

    def reset_daily(self) -> None:
        """Reset daily counters (call at midnight UTC)."""
        self._daily_pnl = 0.0
        self._trades_today = 0

    def reset_weekly(self) -> None:
        """Reset weekly counters (call at Monday 00:00 UTC)."""
        self._weekly_pnl = 0.0

    def trigger_emergency(self, reason: str) -> None:
        """Emergency halt: trigger the emergency kill switch."""
        self._kill_switches.trigger_emergency(reason, equity=self._equity)
        self._position_open = False
        logger.critical("emergency_halt_polymarket", reason=reason)

    def reset_kill_switch(self, name: SwitchName, reason: str) -> None:
        """Reset a specific kill switch with reason."""
        self._kill_switches.reset(name, reason)

    def _check_kill_switches(self) -> None:
        """Check all kill switch thresholds after a trade."""
        ref = self._peak_equity if self._peak_equity > 0 else 1.0

        # Daily loss as negative fraction
        daily_frac = self._daily_pnl / ref  # negative when losing
        self._kill_switches.check_daily_loss(daily_frac, equity=self._equity)

        # Weekly loss
        weekly_frac = self._weekly_pnl / ref
        self._kill_switches.check_weekly_loss(weekly_frac, equity=self._equity)

        # Max drawdown (positive fraction)
        dd = (self._peak_equity - self._equity) / ref if ref > 0 else 0.0
        self._kill_switches.check_max_drawdown(dd, equity=self._equity)

        # Consecutive losses
        self._kill_switches.check_consecutive_losses(self._consecutive_losses, equity=self._equity)

    def _reject(self, signal: BinarySignal, reason: str) -> BinaryTradeDecision:
        logger.debug(
            "binary_trade_rejected",
            direction=signal.direction.value,
            reason=reason,
        )
        return BinaryTradeDecision(
            approved=False,
            direction=signal.direction,
            reason=reason,
        )
