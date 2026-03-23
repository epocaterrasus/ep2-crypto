"""Drawdown-based position size gating with convex multiplier.

Tracks peak equity and computes a position-size multiplier that decreases
continuously as drawdown deepens. Uses a convex decay (k=1.5) so the
reduction accelerates as losses grow — protecting capital more aggressively
during deep drawdowns.

Two independent reduction factors:
    1. Depth-based (convex): multiplier = max(0, (1 - dd/max_dd)^1.5)
    2. Duration-based: how many bars spent underwater (independent of depth)

Final multiplier = min(depth_mult, duration_mult)

Graduated re-entry protocol:
    After drawdown recovery, position sizes ramp back through 5 phases:
    Phase 0: 10% -> Phase 1: 25% -> Phase 2: 50% -> Phase 3: 75% -> Phase 4: 100%
    Each phase requires profitability to advance; a losing trade drops back.
    Minimum cooldown_bars at each phase before advancing.
"""

from __future__ import annotations

import math
import sqlite3
import time
from dataclasses import dataclass

import structlog

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Convex exponent per CLAUDE.md / SPRINTS.md spec
CONVEX_K = 1.5

# Recovery phases: fraction of full size allowed at each phase
RECOVERY_PHASES = (0.10, 0.25, 0.50, 0.75, 1.00)

# Duration-based reduction thresholds (bars -> multiplier)
# 3 days = 3 * 288 = 864 bars -> 80%
# 7 days = 7 * 288 = 2016 bars -> 40%
# 14 days = 14 * 288 = 4032 bars -> 0% (halt)
DURATION_THRESHOLDS: list[tuple[int, float]] = [
    (864, 0.80),   # 3 days underwater
    (2016, 0.40),  # 7 days underwater
    (4032, 0.00),  # 14 days underwater -> halt
]


# ---------------------------------------------------------------------------
# SQLite persistence
# ---------------------------------------------------------------------------

DRAWDOWN_GATE_TABLE = """
CREATE TABLE IF NOT EXISTS risk_drawdown_gate (
    id              INTEGER PRIMARY KEY CHECK (id = 1),
    peak_equity     REAL    NOT NULL,
    current_equity  REAL    NOT NULL,
    current_drawdown REAL   NOT NULL,
    depth_multiplier REAL   NOT NULL,
    duration_multiplier REAL NOT NULL,
    final_multiplier REAL   NOT NULL,
    bars_underwater INTEGER NOT NULL,
    recovery_phase  INTEGER NOT NULL,
    recovery_wins   INTEGER NOT NULL,
    cooldown_remaining_bars INTEGER NOT NULL,
    updated_at_ms   INTEGER NOT NULL
)
"""


@dataclass
class DrawdownGateState:
    """Snapshot of drawdown gate status."""

    peak_equity: float
    current_equity: float
    current_drawdown: float  # as positive fraction (0.05 = 5%)
    depth_multiplier: float  # from convex formula
    duration_multiplier: float  # from underwater duration
    current_multiplier: float  # final = min(depth, duration) * recovery
    bars_underwater: int
    recovery_phase: int  # 0-4 index into RECOVERY_PHASES
    recovery_wins: int
    cooldown_remaining_bars: int


# ---------------------------------------------------------------------------
# DrawdownGate
# ---------------------------------------------------------------------------

class DrawdownGate:
    """Progressive position size reduction based on equity drawdown.

    The gate computes a continuous multiplier between 0.0 and 1.0 that
    scales position sizes. As drawdown deepens, the multiplier shrinks.

    Two independent reduction mechanisms:
        1. Depth: multiplier = max(0, (1 - dd/halt_threshold)^1.5)
        2. Duration: bars spent underwater -> tiered reduction

    Final multiplier = min(depth, duration) * recovery_phase_cap

    Configuration:
        halt_threshold: drawdown fraction at which depth multiplier = 0 (default 0.15)
        cooldown_bars: minimum bars at each recovery phase before advancing (default 5)
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        initial_equity: float,
        halt_threshold: float = 0.15,
        cooldown_bars: int = 5,
    ) -> None:
        if initial_equity <= 0:
            raise ValueError(f"initial_equity must be positive, got {initial_equity}")
        if halt_threshold <= 0 or halt_threshold > 1.0:
            raise ValueError(f"halt_threshold must be in (0, 1], got {halt_threshold}")

        self._conn = conn
        self._halt_threshold = halt_threshold
        self._cooldown_bars = cooldown_bars

        # State
        self._peak_equity = initial_equity
        self._current_equity = initial_equity
        self._depth_multiplier = 1.0
        self._duration_multiplier = 1.0
        self._final_multiplier = 1.0
        self._bars_underwater = 0
        self._recovery_phase = len(RECOVERY_PHASES) - 1  # start at full (phase 4)
        self._recovery_wins = 0
        self._cooldown_remaining = 0
        self._in_recovery = False

        self._ensure_table()
        self._load_state(initial_equity)

    # -- Public API -----------------------------------------------------------

    def update(self, equity: float) -> float:
        """Update with new equity value and return the position size multiplier.

        Call this on every bar after computing equity.

        Args:
            equity: Current total equity (capital + unrealized PnL).

        Returns:
            Multiplier between 0.0 and 1.0 to scale position sizes.
        """
        self._current_equity = equity

        # Update peak (only ratchets up, never down)
        if equity > self._peak_equity:
            self._peak_equity = equity

        # Compute drawdown as positive fraction
        dd = self._compute_dd()

        # --- Depth-based multiplier (convex k=1.5) ---
        self._depth_multiplier = self._compute_depth_multiplier(dd)

        # --- Duration-based multiplier ---
        if dd > 0:
            self._bars_underwater += 1
        else:
            self._bars_underwater = 0

        self._duration_multiplier = self._compute_duration_multiplier(self._bars_underwater)

        # --- Combine: min of depth and duration ---
        raw_mult = min(self._depth_multiplier, self._duration_multiplier)

        # --- Recovery gating ---
        if raw_mult >= 1.0 and not self._in_recovery:
            # No drawdown, no recovery needed
            self._final_multiplier = 1.0
        elif raw_mult < self._final_multiplier or (raw_mult <= 0.0):
            # Drawdown deepening: apply immediately, enter recovery mode
            self._final_multiplier = max(0.0, raw_mult)
            if not self._in_recovery and raw_mult < 1.0:
                self._in_recovery = True
                self._recovery_phase = 0
                self._recovery_wins = 0
                self._cooldown_remaining = self._cooldown_bars
                logger.info(
                    "drawdown_gate_recovery_entered",
                    multiplier=round(self._final_multiplier, 4),
                    drawdown=round(dd, 4),
                )
        else:
            # Recovery: raw_mult >= final_multiplier, cap by recovery phase
            if self._in_recovery:
                phase_cap = RECOVERY_PHASES[self._recovery_phase]
                self._final_multiplier = min(raw_mult, phase_cap)
            else:
                self._final_multiplier = raw_mult

        # Tick cooldown
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1

        self._persist()
        return self._final_multiplier

    def on_trade_result(self, profitable: bool) -> None:
        """Notify the gate of a trade result for recovery tracking.

        Call this after each trade closes. During recovery, profitable
        trades advance toward full size; losing trades set back.

        Args:
            profitable: True if the trade was profitable.
        """
        if not self._in_recovery:
            return

        if profitable:
            self._recovery_wins += 1
            # Can advance phase if cooldown expired and enough wins
            if (
                self._cooldown_remaining <= 0
                and self._recovery_wins >= 2
                and self._recovery_phase < len(RECOVERY_PHASES) - 1
            ):
                self._recovery_phase += 1
                self._recovery_wins = 0
                self._cooldown_remaining = self._cooldown_bars
                logger.info(
                    "drawdown_gate_phase_advance",
                    new_phase=self._recovery_phase,
                    phase_cap=RECOVERY_PHASES[self._recovery_phase],
                )

                # Full recovery reached
                if self._recovery_phase == len(RECOVERY_PHASES) - 1:
                    self._in_recovery = False
                    logger.info("drawdown_gate_recovery_complete")
        else:
            # Losing trade: drop back one phase (minimum phase 0)
            if self._recovery_phase > 0:
                self._recovery_phase -= 1
                self._cooldown_remaining = self._cooldown_bars
                logger.info(
                    "drawdown_gate_phase_setback",
                    new_phase=self._recovery_phase,
                    phase_cap=RECOVERY_PHASES[self._recovery_phase],
                )
            self._recovery_wins = 0

        self._persist()

    def get_state(self) -> DrawdownGateState:
        """Return current drawdown gate state snapshot."""
        dd = self._compute_dd()
        return DrawdownGateState(
            peak_equity=self._peak_equity,
            current_equity=self._current_equity,
            current_drawdown=dd,
            depth_multiplier=self._depth_multiplier,
            duration_multiplier=self._duration_multiplier,
            current_multiplier=self._final_multiplier,
            bars_underwater=self._bars_underwater,
            recovery_phase=self._recovery_phase,
            recovery_wins=self._recovery_wins,
            cooldown_remaining_bars=self._cooldown_remaining,
        )

    def get_multiplier(self) -> float:
        """Return current position size multiplier (0.0 to 1.0)."""
        return self._final_multiplier

    def get_drawdown(self) -> float:
        """Return current drawdown as positive fraction."""
        return self._compute_dd()

    # -- Private helpers -------------------------------------------------------

    def _compute_dd(self) -> float:
        """Compute drawdown as positive fraction."""
        if self._peak_equity <= 0:
            return 0.0
        dd = (self._peak_equity - self._current_equity) / self._peak_equity
        return max(dd, 0.0)

    def _compute_depth_multiplier(self, drawdown: float) -> float:
        """Compute depth-based multiplier using convex formula (k=1.5).

        multiplier = max(0, (1 - dd / halt_threshold)^1.5)

        With halt=0.15:
            dd=0.00 -> 1.000
            dd=0.03 -> ~0.714 (71.4%)
            dd=0.05 -> ~0.544 (54.4%)
            dd=0.08 -> ~0.320 (32.0%)
            dd=0.12 -> ~0.089 (8.9%)
            dd=0.15 -> 0.000
        """
        if drawdown <= 0:
            return 1.0
        if drawdown >= self._halt_threshold:
            return 0.0

        ratio = 1.0 - drawdown / self._halt_threshold
        return max(0.0, math.pow(ratio, CONVEX_K))

    def _compute_duration_multiplier(self, bars_underwater: int) -> float:
        """Compute duration-based multiplier from underwater bar count.

        Linearly interpolates between threshold points:
            0 bars -> 1.0 (full)
            864 bars (3 days) -> 0.80
            2016 bars (7 days) -> 0.40
            4032 bars (14 days) -> 0.00 (halt)
        """
        if bars_underwater <= 0:
            return 1.0

        # Walk through thresholds
        prev_bars = 0
        prev_mult = 1.0
        for threshold_bars, threshold_mult in DURATION_THRESHOLDS:
            if bars_underwater <= threshold_bars:
                # Linear interpolation between prev and current threshold
                frac = (bars_underwater - prev_bars) / (threshold_bars - prev_bars)
                return prev_mult + frac * (threshold_mult - prev_mult)
            prev_bars = threshold_bars
            prev_mult = threshold_mult

        # Beyond last threshold -> halt
        return 0.0

    def _ensure_table(self) -> None:
        self._conn.execute(DRAWDOWN_GATE_TABLE)
        self._conn.commit()

    def _persist(self) -> None:
        now_ms = int(time.time() * 1000)
        dd = self._compute_dd()
        self._conn.execute(
            "INSERT OR REPLACE INTO risk_drawdown_gate "
            "(id, peak_equity, current_equity, current_drawdown, "
            "depth_multiplier, duration_multiplier, final_multiplier, "
            "bars_underwater, recovery_phase, recovery_wins, "
            "cooldown_remaining_bars, updated_at_ms) "
            "VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                self._peak_equity,
                self._current_equity,
                dd,
                self._depth_multiplier,
                self._duration_multiplier,
                self._final_multiplier,
                self._bars_underwater,
                self._recovery_phase,
                self._recovery_wins,
                self._cooldown_remaining,
                now_ms,
            ),
        )
        self._conn.commit()

    def _load_state(self, initial_equity: float) -> None:
        row = self._conn.execute(
            "SELECT peak_equity, current_equity, depth_multiplier, "
            "duration_multiplier, final_multiplier, bars_underwater, "
            "recovery_phase, recovery_wins, cooldown_remaining_bars "
            "FROM risk_drawdown_gate WHERE id = ?",
            (1,),
        ).fetchone()

        if row is None:
            logger.info(
                "drawdown_gate_init",
                state="fresh",
                initial_equity=initial_equity,
            )
            self._persist()
            return

        self._peak_equity = max(row[0], initial_equity)
        self._current_equity = row[1]
        self._depth_multiplier = row[2]
        self._duration_multiplier = row[3]
        self._final_multiplier = row[4]
        self._bars_underwater = row[5]
        self._recovery_phase = row[6]
        self._recovery_wins = row[7]
        self._cooldown_remaining = row[8]
        self._in_recovery = self._recovery_phase < len(RECOVERY_PHASES) - 1

        logger.info(
            "drawdown_gate_init",
            state="restored",
            peak_equity=self._peak_equity,
            multiplier=self._final_multiplier,
            recovery_phase=self._recovery_phase,
            bars_underwater=self._bars_underwater,
        )
