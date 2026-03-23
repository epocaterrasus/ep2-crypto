"""Tests for DrawdownGate: convex reduction (k=1.5), duration-based, recovery."""

from __future__ import annotations

import math
import sqlite3

import pytest

from ep2_crypto.risk.drawdown_gate import (
    CONVEX_K,
    DrawdownGate,
)


@pytest.fixture
def conn() -> sqlite3.Connection:
    return sqlite3.connect(":memory:")


@pytest.fixture
def gate(conn: sqlite3.Connection) -> DrawdownGate:
    return DrawdownGate(conn, initial_equity=100_000.0)


# ---------------------------------------------------------------------------
# Depth-based convex multiplier (k=1.5)
# ---------------------------------------------------------------------------

class TestConvexMultiplierCurve:
    """Verify the convex formula: multiplier = max(0, (1 - dd/max_dd)^1.5)."""

    def test_zero_drawdown_full_size(self, gate: DrawdownGate) -> None:
        mult = gate.update(100_000.0)
        assert mult == pytest.approx(1.0)

    def test_new_high_resets_peak(self, gate: DrawdownGate) -> None:
        gate.update(100_000.0)
        gate.update(105_000.0)
        mult = gate.update(105_000.0)
        assert mult == pytest.approx(1.0)
        state = gate.get_state()
        assert state.peak_equity == pytest.approx(105_000.0)

    def test_three_percent_drawdown(self, gate: DrawdownGate) -> None:
        """At 3% DD: multiplier = (1 - 0.03/0.15)^1.5 = 0.8^1.5 ≈ 0.7155."""
        gate.update(100_000.0)
        mult = gate.update(97_000.0)
        expected = math.pow(1.0 - 0.03 / 0.15, CONVEX_K)
        assert mult == pytest.approx(expected, abs=0.01)

    def test_five_percent_drawdown(self, gate: DrawdownGate) -> None:
        """At 5% DD: multiplier = (1 - 0.05/0.15)^1.5 = (2/3)^1.5 ≈ 0.5443."""
        gate.update(100_000.0)
        mult = gate.update(95_000.0)
        expected = math.pow(1.0 - 0.05 / 0.15, CONVEX_K)
        assert mult == pytest.approx(expected, abs=0.01)

    def test_eight_percent_drawdown(self, gate: DrawdownGate) -> None:
        """At 8% DD: ≈ 0.320."""
        gate.update(100_000.0)
        mult = gate.update(92_000.0)
        expected = math.pow(1.0 - 0.08 / 0.15, CONVEX_K)
        assert mult == pytest.approx(expected, abs=0.01)

    def test_twelve_percent_drawdown(self, gate: DrawdownGate) -> None:
        """At 12% DD: ≈ 0.089."""
        gate.update(100_000.0)
        mult = gate.update(88_000.0)
        expected = math.pow(1.0 - 0.12 / 0.15, CONVEX_K)
        assert mult == pytest.approx(expected, abs=0.01)

    def test_fifteen_percent_drawdown_halts(self, gate: DrawdownGate) -> None:
        gate.update(100_000.0)
        mult = gate.update(85_000.0)
        assert mult == pytest.approx(0.0)

    def test_beyond_halt_stays_zero(self, gate: DrawdownGate) -> None:
        gate.update(100_000.0)
        mult = gate.update(80_000.0)
        assert mult == pytest.approx(0.0)

    def test_multiplier_monotonically_decreases(self, conn: sqlite3.Connection) -> None:
        gate = DrawdownGate(conn, initial_equity=100_000.0)
        gate.update(100_000.0)
        prev_mult = 1.0
        for dd_pct in range(1, 16):
            equity = 100_000.0 * (1 - dd_pct / 100)
            mult = gate.update(equity)
            assert mult <= prev_mult, (
                f"Multiplier increased at {dd_pct}% drawdown: {mult} > {prev_mult}"
            )
            prev_mult = mult


# ---------------------------------------------------------------------------
# Duration-based multiplier
# ---------------------------------------------------------------------------

class TestDurationMultiplier:
    def test_short_underwater_no_reduction(self, gate: DrawdownGate) -> None:
        gate.update(100_000.0)
        # 10 bars underwater (much less than 3 days = 864)
        for _ in range(10):
            gate.update(99_000.0)
        state = gate.get_state()
        # Duration mult should still be near 1.0
        assert state.duration_multiplier > 0.99

    def test_three_day_underwater_eighty_percent(
        self, conn: sqlite3.Connection
    ) -> None:
        gate = DrawdownGate(conn, initial_equity=100_000.0)
        gate.update(100_000.0)
        # Simulate 864 bars at slight drawdown (1%)
        for _ in range(864):
            gate.update(99_000.0)
        state = gate.get_state()
        assert state.duration_multiplier == pytest.approx(0.80, abs=0.01)

    def test_duration_independent_of_depth(self, conn: sqlite3.Connection) -> None:
        """Duration multiplier only depends on time, not drawdown depth."""
        gate = DrawdownGate(conn, initial_equity=100_000.0)
        gate.update(100_000.0)
        # Small drawdown (0.1%) for many bars
        for _ in range(864):
            gate.update(99_900.0)
        state = gate.get_state()
        # Duration should still apply even with tiny depth
        assert state.duration_multiplier == pytest.approx(0.80, abs=0.01)
        assert state.bars_underwater == 864


# ---------------------------------------------------------------------------
# Recovery protocol
# ---------------------------------------------------------------------------

class TestGraduatedRecovery:
    def test_recovery_does_not_snap_back(self, gate: DrawdownGate) -> None:
        gate.update(100_000.0)
        gate.update(90_000.0)  # 10% drawdown

        # Equity recovers fully but multiplier capped by recovery phase
        mult = gate.update(100_000.0)
        # Phase 0 cap = 10%, so mult should be <= 0.10
        assert mult <= 0.10 + 0.01

    def test_recovery_phases_advance_with_wins(
        self, conn: sqlite3.Connection
    ) -> None:
        gate = DrawdownGate(conn, initial_equity=100_000.0, cooldown_bars=0)
        gate.update(100_000.0)
        gate.update(90_000.0)  # Enter recovery
        gate.update(100_000.0)  # Equity recovers

        # Phase 0: cap 10%
        assert gate.get_multiplier() <= 0.10 + 0.01

        # 2 wins to advance to phase 1 (25%)
        gate.on_trade_result(profitable=True)
        gate.on_trade_result(profitable=True)
        gate.update(100_000.0)
        assert gate.get_multiplier() <= 0.25 + 0.01

        # 2 more wins to advance to phase 2 (50%)
        gate.on_trade_result(profitable=True)
        gate.on_trade_result(profitable=True)
        gate.update(100_000.0)
        assert gate.get_multiplier() <= 0.50 + 0.01

    def test_loss_during_recovery_drops_phase(
        self, conn: sqlite3.Connection
    ) -> None:
        gate = DrawdownGate(conn, initial_equity=100_000.0, cooldown_bars=0)
        gate.update(100_000.0)
        gate.update(90_000.0)
        gate.update(100_000.0)

        # Advance to phase 1
        gate.on_trade_result(profitable=True)
        gate.on_trade_result(profitable=True)
        state = gate.get_state()
        assert state.recovery_phase == 1

        # Losing trade drops back to phase 0
        gate.on_trade_result(profitable=False)
        state = gate.get_state()
        assert state.recovery_phase == 0

    def test_full_recovery_to_100_percent(
        self, conn: sqlite3.Connection
    ) -> None:
        gate = DrawdownGate(conn, initial_equity=100_000.0, cooldown_bars=0)
        gate.update(100_000.0)
        gate.update(90_000.0)  # Enter recovery
        gate.update(100_000.0)  # Equity recovers

        # Advance through all 5 phases (2 wins per phase)
        for _phase in range(4):  # 4 transitions to reach phase 4
            gate.on_trade_result(profitable=True)
            gate.on_trade_result(profitable=True)
            gate.update(100_000.0)

        assert gate.get_multiplier() == pytest.approx(1.0)

    def test_cooldown_prevents_premature_advance(
        self, conn: sqlite3.Connection
    ) -> None:
        gate = DrawdownGate(conn, initial_equity=100_000.0, cooldown_bars=5)
        gate.update(100_000.0)
        gate.update(90_000.0)
        gate.update(100_000.0)

        # 2 wins immediately — but cooldown hasn't expired
        gate.on_trade_result(profitable=True)
        gate.on_trade_result(profitable=True)
        state = gate.get_state()
        # Should still be phase 0 because cooldown hasn't expired
        assert state.recovery_phase == 0

        # Tick through cooldown
        for _ in range(5):
            gate.update(100_000.0)

        # Now wins should advance
        gate.on_trade_result(profitable=True)
        gate.on_trade_result(profitable=True)
        state = gate.get_state()
        assert state.recovery_phase == 1


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_state_survives_restart(self, conn: sqlite3.Connection) -> None:
        g1 = DrawdownGate(conn, initial_equity=100_000.0)
        g1.update(100_000.0)
        g1.update(92_000.0)
        mult1 = g1.get_multiplier()

        g2 = DrawdownGate(conn, initial_equity=100_000.0)
        assert g2.get_multiplier() == pytest.approx(mult1, rel=0.01)

    def test_bars_underwater_persists(self, conn: sqlite3.Connection) -> None:
        g1 = DrawdownGate(conn, initial_equity=100_000.0)
        g1.update(100_000.0)
        for _ in range(50):
            g1.update(99_000.0)

        g2 = DrawdownGate(conn, initial_equity=100_000.0)
        state = g2.get_state()
        assert state.bars_underwater == 50


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_negative_equity_rejected(self, conn: sqlite3.Connection) -> None:
        with pytest.raises(ValueError, match="positive"):
            DrawdownGate(conn, initial_equity=-1000.0)

    def test_invalid_halt_threshold(self, conn: sqlite3.Connection) -> None:
        with pytest.raises(ValueError):
            DrawdownGate(conn, initial_equity=100_000.0, halt_threshold=0.0)

    def test_invalid_halt_threshold_too_high(self, conn: sqlite3.Connection) -> None:
        with pytest.raises(ValueError):
            DrawdownGate(conn, initial_equity=100_000.0, halt_threshold=1.5)


# ---------------------------------------------------------------------------
# State snapshot
# ---------------------------------------------------------------------------

class TestGetState:
    def test_state_reflects_drawdown(self, gate: DrawdownGate) -> None:
        gate.update(100_000.0)
        gate.update(95_000.0)
        state = gate.get_state()
        assert state.current_drawdown == pytest.approx(0.05, rel=0.01)
        assert state.peak_equity == pytest.approx(100_000.0)
        assert state.current_equity == pytest.approx(95_000.0)

    def test_state_includes_all_fields(self, gate: DrawdownGate) -> None:
        gate.update(100_000.0)
        gate.update(95_000.0)
        state = gate.get_state()
        assert hasattr(state, "depth_multiplier")
        assert hasattr(state, "duration_multiplier")
        assert hasattr(state, "bars_underwater")
        assert hasattr(state, "recovery_phase")
