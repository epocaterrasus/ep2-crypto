"""Kill switch manager: halt trading when loss limits are breached.

Each kill switch follows a state machine:
    ARMED -> TRIGGERED -> RESET (back to ARMED)

Transitions:
    ARMED -> TRIGGERED: automatic when threshold breached
    TRIGGERED -> ARMED: requires explicit reset() with a reason string

All state is persisted to SQLite on every transition so that a process
crash cannot silently re-enable trading. A TRIGGERED switch blocks all
new trades until a human resets it.

Kill switches implemented:
    1. DailyLoss     - daily realized + unrealized loss exceeds threshold
    2. WeeklyLoss    - rolling 7-day loss exceeds threshold
    3. MaxDrawdown   - peak-to-trough equity drawdown exceeds threshold
    4. ConsecutiveLoss - N consecutive losing trades
    5. Emergency      - manual halt, closes all positions
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Literal

import structlog

if TYPE_CHECKING:
    import sqlite3

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


class SwitchState(Enum):
    ARMED = "armed"
    TRIGGERED = "triggered"


SwitchName = Literal[
    "daily_loss",
    "weekly_loss",
    "max_drawdown",
    "consecutive_loss",
    "emergency",
]

ALL_SWITCH_NAMES: list[SwitchName] = [
    "daily_loss",
    "weekly_loss",
    "max_drawdown",
    "consecutive_loss",
    "emergency",
]


@dataclass
class KillSwitchStatus:
    """Snapshot of a single kill switch."""

    name: SwitchName
    state: SwitchState
    threshold: float
    current_value: float
    triggered_at_ms: int | None
    triggered_reason: str | None
    reset_at_ms: int | None
    reset_reason: str | None


# ---------------------------------------------------------------------------
# SQLite persistence
# ---------------------------------------------------------------------------

KILL_SWITCH_TABLE = """
CREATE TABLE IF NOT EXISTS risk_kill_switches (
    name            TEXT PRIMARY KEY,
    state           TEXT    NOT NULL,
    threshold       REAL    NOT NULL,
    current_value   REAL    NOT NULL DEFAULT 0.0,
    triggered_at_ms INTEGER,
    triggered_reason TEXT,
    reset_at_ms     INTEGER,
    reset_reason    TEXT
)
"""

KILL_SWITCH_LOG_TABLE = """
CREATE TABLE IF NOT EXISTS risk_kill_switch_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp_ms    INTEGER NOT NULL,
    switch_name     TEXT    NOT NULL,
    old_state       TEXT    NOT NULL,
    new_state       TEXT    NOT NULL,
    threshold       REAL    NOT NULL,
    current_value   REAL    NOT NULL,
    reason          TEXT,
    equity          REAL
)
"""


# ---------------------------------------------------------------------------
# KillSwitchManager
# ---------------------------------------------------------------------------


class KillSwitchManager:
    """Manages all kill switches with persistent state.

    Configuration comes from a dict of thresholds:
        {
            "daily_loss": 0.03,       # 3% of capital
            "weekly_loss": 0.05,      # 5% of capital
            "max_drawdown": 0.15,     # 15% from peak
            "consecutive_loss": 15,   # 15 losing trades in a row
            "emergency": 1.0,         # always threshold=1, triggered manually
        }

    The manager checks thresholds and triggers switches automatically.
    Reset MUST be explicit with a reason string.
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        thresholds: dict[SwitchName, float],
    ) -> None:
        self._conn = conn
        self._thresholds = thresholds
        self._ensure_tables()
        self._initialize_switches()

    # -- Public API -----------------------------------------------------------

    def any_triggered(self) -> bool:
        """Return True if ANY kill switch is in TRIGGERED state."""
        row = self._conn.execute(
            "SELECT count(*) FROM risk_kill_switches WHERE state = ?",
            (SwitchState.TRIGGERED.value,),
        ).fetchone()
        return row[0] > 0

    def get_triggered_names(self) -> list[SwitchName]:
        """Return names of all currently triggered switches."""
        rows = self._conn.execute(
            "SELECT name FROM risk_kill_switches WHERE state = ?",
            (SwitchState.TRIGGERED.value,),
        ).fetchall()
        return [row[0] for row in rows]

    def get_all_status(self) -> list[KillSwitchStatus]:
        """Return status of every kill switch."""
        rows = self._conn.execute(
            "SELECT name, state, threshold, current_value, "
            "triggered_at_ms, triggered_reason, reset_at_ms, reset_reason "
            "FROM risk_kill_switches ORDER BY name"
        ).fetchall()
        return [
            KillSwitchStatus(
                name=row[0],
                state=SwitchState(row[1]),
                threshold=row[2],
                current_value=row[3],
                triggered_at_ms=row[4],
                triggered_reason=row[5],
                reset_at_ms=row[6],
                reset_reason=row[7],
            )
            for row in rows
        ]

    def check_daily_loss(
        self,
        daily_pnl_fraction: float,
        equity: float | None = None,
    ) -> bool:
        """Check if daily loss threshold is breached.

        Args:
            daily_pnl_fraction: Today's P&L as fraction of capital (negative = loss).
            equity: Current equity for logging.

        Returns:
            True if switch was triggered (or was already triggered).
        """
        return self._check_and_maybe_trigger(
            "daily_loss",
            current_value=abs(min(daily_pnl_fraction, 0.0)),
            equity=equity,
            reason=f"Daily loss {daily_pnl_fraction:.4f} exceeded threshold",
        )

    def check_weekly_loss(
        self,
        weekly_pnl_fraction: float,
        equity: float | None = None,
    ) -> bool:
        """Check weekly (rolling 7-day) loss threshold."""
        return self._check_and_maybe_trigger(
            "weekly_loss",
            current_value=abs(min(weekly_pnl_fraction, 0.0)),
            equity=equity,
            reason=f"Weekly loss {weekly_pnl_fraction:.4f} exceeded threshold",
        )

    def check_max_drawdown(
        self,
        drawdown_fraction: float,
        equity: float | None = None,
    ) -> bool:
        """Check max drawdown threshold.

        Args:
            drawdown_fraction: Current drawdown as positive fraction (0.05 = 5%).
        """
        return self._check_and_maybe_trigger(
            "max_drawdown",
            current_value=drawdown_fraction,
            equity=equity,
            reason=f"Drawdown {drawdown_fraction:.4f} exceeded threshold",
        )

    def check_consecutive_losses(
        self,
        consecutive_losses: int,
        equity: float | None = None,
    ) -> bool:
        """Check consecutive loss count threshold."""
        return self._check_and_maybe_trigger(
            "consecutive_loss",
            current_value=float(consecutive_losses),
            equity=equity,
            reason=f"Consecutive losses {consecutive_losses} exceeded threshold",
        )

    def trigger_emergency(
        self,
        reason: str,
        equity: float | None = None,
    ) -> None:
        """Manually trigger the emergency kill switch.

        This should also close all open positions (caller responsibility).
        """
        self._force_trigger("emergency", reason=reason, equity=equity)

    def reset(
        self,
        switch_name: SwitchName,
        reason: str,
    ) -> None:
        """Reset a triggered kill switch back to ARMED.

        Args:
            switch_name: Which switch to reset.
            reason: Human-readable explanation for why it is safe to resume.

        Raises:
            ValueError: If the switch is not currently triggered.
        """
        row = self._conn.execute(
            "SELECT state FROM risk_kill_switches WHERE name = ?",
            (switch_name,),
        ).fetchone()
        if row is None:
            raise ValueError(f"Unknown kill switch: {switch_name}")
        if row[0] != SwitchState.TRIGGERED.value:
            raise ValueError(
                f"Cannot reset {switch_name}: current state is {row[0]}, not triggered"
            )

        now_ms = int(time.time() * 1000)
        self._conn.execute(
            "UPDATE risk_kill_switches SET state = ?, current_value = 0.0, "
            "reset_at_ms = ?, reset_reason = ? WHERE name = ?",
            (SwitchState.ARMED.value, now_ms, reason, switch_name),
        )
        self._log_transition(
            switch_name=switch_name,
            old_state=SwitchState.TRIGGERED,
            new_state=SwitchState.ARMED,
            threshold=self._thresholds[switch_name],
            current_value=0.0,
            reason=f"RESET: {reason}",
            equity=None,
        )
        self._conn.commit()

        logger.info(
            "kill_switch_reset",
            switch=switch_name,
            reason=reason,
        )

    def reset_all(self, reason: str) -> int:
        """Reset all triggered switches. Returns count of switches reset."""
        triggered = self.get_triggered_names()
        for name in triggered:
            self.reset(name, reason)
        return len(triggered)

    # -- Update current values (for display, without triggering) ---------------

    def update_current_value(self, switch_name: SwitchName, value: float) -> None:
        """Update the current_value field for monitoring display only.

        Does NOT trigger the switch. Use the check_* methods for that.
        """
        self._conn.execute(
            "UPDATE risk_kill_switches SET current_value = ? WHERE name = ?",
            (value, switch_name),
        )
        self._conn.commit()

    # -- Private helpers -------------------------------------------------------

    def _check_and_maybe_trigger(
        self,
        switch_name: SwitchName,
        current_value: float,
        equity: float | None,
        reason: str,
    ) -> bool:
        """Check threshold and trigger if breached. Returns True if triggered."""
        row = self._conn.execute(
            "SELECT state, threshold FROM risk_kill_switches WHERE name = ?",
            (switch_name,),
        ).fetchone()
        if row is None:
            logger.warning("kill_switch_not_found", switch=switch_name)
            return False

        state = SwitchState(row[0])
        threshold = row[1]

        # Update current value regardless
        self._conn.execute(
            "UPDATE risk_kill_switches SET current_value = ? WHERE name = ?",
            (current_value, switch_name),
        )

        if state == SwitchState.TRIGGERED:
            self._conn.commit()
            return True

        if current_value >= threshold:
            self._force_trigger(switch_name, reason=reason, equity=equity)
            return True

        self._conn.commit()
        return False

    def _force_trigger(
        self,
        switch_name: SwitchName,
        reason: str,
        equity: float | None,
    ) -> None:
        """Force a switch into TRIGGERED state."""
        now_ms = int(time.time() * 1000)

        row = self._conn.execute(
            "SELECT state, threshold, current_value FROM risk_kill_switches WHERE name = ?",
            (switch_name,),
        ).fetchone()
        if row is None:
            return

        old_state = SwitchState(row[0])
        threshold = row[1]
        current_value = row[2]

        self._conn.execute(
            "UPDATE risk_kill_switches SET state = ?, "
            "triggered_at_ms = ?, triggered_reason = ? WHERE name = ?",
            (SwitchState.TRIGGERED.value, now_ms, reason, switch_name),
        )
        self._log_transition(
            switch_name=switch_name,
            old_state=old_state,
            new_state=SwitchState.TRIGGERED,
            threshold=threshold,
            current_value=current_value,
            reason=reason,
            equity=equity,
        )
        self._conn.commit()

        logger.warning(
            "kill_switch_triggered",
            switch=switch_name,
            reason=reason,
            threshold=threshold,
            current_value=current_value,
            equity=equity,
        )

    def _log_transition(
        self,
        switch_name: str,
        old_state: SwitchState,
        new_state: SwitchState,
        threshold: float,
        current_value: float,
        reason: str | None,
        equity: float | None,
    ) -> None:
        """Append to the kill switch log table."""
        now_ms = int(time.time() * 1000)
        self._conn.execute(
            "INSERT INTO risk_kill_switch_log "
            "(timestamp_ms, switch_name, old_state, new_state, "
            "threshold, current_value, reason, equity) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                now_ms,
                switch_name,
                old_state.value,
                new_state.value,
                threshold,
                current_value,
                reason,
                equity,
            ),
        )

    def _ensure_tables(self) -> None:
        self._conn.execute(KILL_SWITCH_TABLE)
        self._conn.execute(KILL_SWITCH_LOG_TABLE)
        self._conn.commit()

    def _initialize_switches(self) -> None:
        """Insert any missing switches with ARMED state."""
        for name in ALL_SWITCH_NAMES:
            threshold = self._thresholds.get(name, 1.0)
            self._conn.execute(
                "INSERT OR IGNORE INTO risk_kill_switches "
                "(name, state, threshold, current_value) VALUES (?, ?, ?, 0.0)",
                (name, SwitchState.ARMED.value, threshold),
            )
            # Update threshold if config changed (but do NOT reset state)
            self._conn.execute(
                "UPDATE risk_kill_switches SET threshold = ? WHERE name = ?",
                (threshold, name),
            )
        self._conn.commit()

        statuses = self.get_all_status()
        for s in statuses:
            logger.info(
                "kill_switch_loaded",
                switch=s.name,
                state=s.state.value,
                threshold=s.threshold,
            )
