"""Real-time position state tracking with mark-to-market and persistence.

Tracks a single BTC/USDT perpetual futures position: entry, PnL, duration,
max adverse excursion. Serializes to SQLite on every state change.

Thread safety: All public methods acquire a threading.Lock. The backtest
engine runs single-threaded so the lock is uncontended; in live trading
the lock prevents data races between the bar handler and the order-fill
callback.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    import sqlite3

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


class PositionSide(Enum):
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


@dataclass
class PositionState:
    """Immutable snapshot of position at a point in time."""

    symbol: str = "BTC/USDT:USDT"
    side: PositionSide = PositionSide.FLAT
    quantity: float = 0.0  # BTC amount (always positive; side encodes direction)
    entry_price: float = 0.0
    entry_time_ms: int = 0  # Unix ms when position was opened
    current_price: float = 0.0
    unrealized_pnl: float = 0.0  # USD PnL at current_price
    max_adverse_excursion: float = 0.0  # Worst unrealized PnL seen (always <= 0)
    max_favorable_excursion: float = 0.0  # Best unrealized PnL seen (always >= 0)
    realized_pnl: float = 0.0  # Cumulative realized PnL for the day
    bars_held: int = 0

    @property
    def is_open(self) -> bool:
        return self.side != PositionSide.FLAT and self.quantity > 0.0

    @property
    def notional_usd(self) -> float:
        """Dollar notional of the position."""
        return self.quantity * self.entry_price

    def holding_duration_ms(self, now_ms: int) -> int:
        """Milliseconds since entry. Returns 0 if flat."""
        if not self.is_open:
            return 0
        return now_ms - self.entry_time_ms


# ---------------------------------------------------------------------------
# SQLite persistence schema
# ---------------------------------------------------------------------------

POSITION_TABLE = """
CREATE TABLE IF NOT EXISTS risk_position (
    id              INTEGER PRIMARY KEY CHECK (id = 1),
    symbol          TEXT    NOT NULL,
    side            TEXT    NOT NULL,
    quantity        REAL    NOT NULL,
    entry_price     REAL    NOT NULL,
    entry_time_ms   INTEGER NOT NULL,
    current_price   REAL    NOT NULL,
    unrealized_pnl  REAL    NOT NULL,
    max_adverse_excursion   REAL NOT NULL,
    max_favorable_excursion REAL NOT NULL,
    realized_pnl    REAL    NOT NULL,
    bars_held       INTEGER NOT NULL,
    updated_at_ms   INTEGER NOT NULL
)
"""


# ---------------------------------------------------------------------------
# PositionTracker
# ---------------------------------------------------------------------------


class PositionTracker:
    """Tracks a single open position with real-time mark-to-market.

    Design decisions:
    - Single-position only (max 1 open at a time per REQUIREMENTS RR-1.2).
    - State written to SQLite on every mutation so it survives crashes.
    - Thread-safe via a reentrant lock.
    - All PnL in USD.

    Usage (backtest):
        tracker = PositionTracker(conn)
        tracker.open_position("long", 0.05, 67000.0, bar_time_ms)
        for bar in bars:
            tracker.mark_to_market(bar.close, bar.timestamp_ms)
        result = tracker.close_position(bar.close, bar.timestamp_ms)

    Usage (live):
        tracker = PositionTracker(conn)
        # Called from order-fill callback
        tracker.open_position("long", fill.qty, fill.price, fill.timestamp_ms)
        # Called every 5-min bar
        tracker.mark_to_market(bar.close, bar.timestamp_ms)
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self._lock = threading.RLock()
        self._state = PositionState()
        self._ensure_table()
        self._load_state()

    # -- Public API -----------------------------------------------------------

    def open_position(
        self,
        side: str,
        quantity: float,
        entry_price: float,
        entry_time_ms: int,
        symbol: str = "BTC/USDT:USDT",
    ) -> PositionState:
        """Open a new position. Raises if a position is already open.

        Args:
            side: "long" or "short"
            quantity: BTC amount (positive)
            entry_price: Fill price in USD
            entry_time_ms: Timestamp of fill
            symbol: Trading pair

        Returns:
            Snapshot of the newly opened position.

        Raises:
            ValueError: If a position is already open, or invalid side.
        """
        with self._lock:
            if self._state.is_open:
                raise ValueError(
                    f"Cannot open position: already {self._state.side.value} "
                    f"{self._state.quantity} @ {self._state.entry_price}"
                )
            if side not in ("long", "short"):
                raise ValueError(f"Invalid side: {side!r}. Must be 'long' or 'short'.")
            if quantity <= 0:
                raise ValueError(f"Quantity must be positive, got {quantity}")
            if entry_price <= 0:
                raise ValueError(f"Entry price must be positive, got {entry_price}")

            self._state = PositionState(
                symbol=symbol,
                side=PositionSide(side),
                quantity=quantity,
                entry_price=entry_price,
                entry_time_ms=entry_time_ms,
                current_price=entry_price,
                unrealized_pnl=0.0,
                max_adverse_excursion=0.0,
                max_favorable_excursion=0.0,
                realized_pnl=self._state.realized_pnl,  # carry forward daily realized
                bars_held=0,
            )
            self._persist()

            logger.info(
                "position_opened",
                side=side,
                quantity=quantity,
                entry_price=entry_price,
                notional_usd=quantity * entry_price,
            )
            return self.get_state()

    def close_position(
        self,
        exit_price: float,
        exit_time_ms: int,
    ) -> float:
        """Close the current position and return realized PnL.

        Args:
            exit_price: Price at which to close.
            exit_time_ms: Timestamp of exit.

        Returns:
            Realized PnL in USD for this trade.

        Raises:
            ValueError: If no position is open.
        """
        with self._lock:
            if not self._state.is_open:
                raise ValueError("Cannot close: no open position.")

            trade_pnl = self._compute_pnl(exit_price)
            duration_ms = exit_time_ms - self._state.entry_time_ms

            logger.info(
                "position_closed",
                side=self._state.side.value,
                quantity=self._state.quantity,
                entry_price=self._state.entry_price,
                exit_price=exit_price,
                trade_pnl=round(trade_pnl, 2),
                duration_ms=duration_ms,
                bars_held=self._state.bars_held,
                max_adverse_excursion=round(self._state.max_adverse_excursion, 2),
                max_favorable_excursion=round(self._state.max_favorable_excursion, 2),
            )

            new_realized = self._state.realized_pnl + trade_pnl
            self._state = PositionState(
                symbol=self._state.symbol,
                side=PositionSide.FLAT,
                quantity=0.0,
                entry_price=0.0,
                entry_time_ms=0,
                current_price=exit_price,
                unrealized_pnl=0.0,
                max_adverse_excursion=0.0,
                max_favorable_excursion=0.0,
                realized_pnl=new_realized,
                bars_held=0,
            )
            self._persist()
            return trade_pnl

    def mark_to_market(self, price: float, timestamp_ms: int) -> PositionState:
        """Update position with latest market price. Call on every bar.

        Also increments bars_held counter. Updates MAE/MFE.

        Args:
            price: Current market price.
            timestamp_ms: Current bar timestamp.

        Returns:
            Updated position state snapshot.
        """
        with self._lock:
            self._state.current_price = price

            if self._state.is_open:
                pnl = self._compute_pnl(price)
                self._state.unrealized_pnl = pnl
                self._state.bars_held += 1

                # Track max adverse / favorable excursion
                if pnl < self._state.max_adverse_excursion:
                    self._state.max_adverse_excursion = pnl
                if pnl > self._state.max_favorable_excursion:
                    self._state.max_favorable_excursion = pnl

            self._persist()
            return self.get_state()

    def get_state(self) -> PositionState:
        """Return an immutable snapshot of current position state."""
        with self._lock:
            return PositionState(
                symbol=self._state.symbol,
                side=self._state.side,
                quantity=self._state.quantity,
                entry_price=self._state.entry_price,
                entry_time_ms=self._state.entry_time_ms,
                current_price=self._state.current_price,
                unrealized_pnl=self._state.unrealized_pnl,
                max_adverse_excursion=self._state.max_adverse_excursion,
                max_favorable_excursion=self._state.max_favorable_excursion,
                realized_pnl=self._state.realized_pnl,
                bars_held=self._state.bars_held,
            )

    def get_holding_duration_ms(self) -> int:
        """Return how long the current position has been held, in ms."""
        with self._lock:
            if not self._state.is_open:
                return 0
            now_ms = int(time.time() * 1000)
            return self._state.holding_duration_ms(now_ms)

    def get_holding_bars(self) -> int:
        """Return number of bars the current position has been held."""
        with self._lock:
            return self._state.bars_held

    def reset_daily_pnl(self) -> None:
        """Reset the realized PnL accumulator (call at day boundary)."""
        with self._lock:
            self._state.realized_pnl = 0.0
            self._persist()
            logger.info("daily_pnl_reset")

    # -- Private helpers ------------------------------------------------------

    def _compute_pnl(self, price: float) -> float:
        """Compute unrealized PnL at given price."""
        if self._state.side == PositionSide.LONG:
            return self._state.quantity * (price - self._state.entry_price)
        elif self._state.side == PositionSide.SHORT:
            return self._state.quantity * (self._state.entry_price - price)
        return 0.0

    def _ensure_table(self) -> None:
        """Create the persistence table if it does not exist."""
        self._conn.execute(POSITION_TABLE)
        self._conn.commit()

    def _persist(self) -> None:
        """Write current state to SQLite (upsert single row with id=1)."""
        now_ms = int(time.time() * 1000)
        self._conn.execute(
            "INSERT OR REPLACE INTO risk_position "
            "(id, symbol, side, quantity, entry_price, entry_time_ms, "
            "current_price, unrealized_pnl, max_adverse_excursion, "
            "max_favorable_excursion, realized_pnl, bars_held, updated_at_ms) "
            "VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                self._state.symbol,
                self._state.side.value,
                self._state.quantity,
                self._state.entry_price,
                self._state.entry_time_ms,
                self._state.current_price,
                self._state.unrealized_pnl,
                self._state.max_adverse_excursion,
                self._state.max_favorable_excursion,
                self._state.realized_pnl,
                self._state.bars_held,
                now_ms,
            ),
        )
        self._conn.commit()

    def _load_state(self) -> None:
        """Restore state from SQLite on startup. If no row, start flat."""
        row = self._conn.execute(
            "SELECT symbol, side, quantity, entry_price, entry_time_ms, "
            "current_price, unrealized_pnl, max_adverse_excursion, "
            "max_favorable_excursion, realized_pnl, bars_held "
            "FROM risk_position WHERE id = ?",
            (1,),
        ).fetchone()

        if row is None:
            logger.info("position_tracker_init", state="no_persisted_state_found")
            return

        self._state = PositionState(
            symbol=row[0],
            side=PositionSide(row[1]),
            quantity=row[2],
            entry_price=row[3],
            entry_time_ms=row[4],
            current_price=row[5],
            unrealized_pnl=row[6],
            max_adverse_excursion=row[7],
            max_favorable_excursion=row[8],
            realized_pnl=row[9],
            bars_held=row[10],
        )
        logger.info(
            "position_tracker_init",
            state="restored",
            side=self._state.side.value,
            quantity=self._state.quantity,
            entry_price=self._state.entry_price,
        )
