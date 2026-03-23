"""Persistence tests: kill switches, position tracker, drawdown gate survive restart.

Uses file-backed SQLite (not :memory:) to verify real persistence.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from ep2_crypto.risk.drawdown_gate import DrawdownGate
from ep2_crypto.risk.kill_switches import KillSwitchManager
from ep2_crypto.risk.position_tracker import PositionSide, PositionTracker


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "risk_test.db"


# ---------------------------------------------------------------------------
# Kill switch persistence (file-backed)
# ---------------------------------------------------------------------------

class TestKillSwitchPersistence:
    def test_triggered_state_survives_file_close_reopen(
        self, db_path: Path,
    ) -> None:
        """Trigger -> close connection -> reopen -> verify still triggered."""
        thresholds = {
            "daily_loss": 0.03,
            "weekly_loss": 0.05,
            "max_drawdown": 0.15,
            "consecutive_loss": 15.0,
            "emergency": 1.0,
        }

        # Session 1: trigger daily loss
        conn1 = sqlite3.connect(str(db_path))
        ks1 = KillSwitchManager(conn1, thresholds)
        ks1.check_daily_loss(-0.05)
        assert ks1.any_triggered()
        conn1.close()

        # Session 2: reopen, verify persisted
        conn2 = sqlite3.connect(str(db_path))
        ks2 = KillSwitchManager(conn2, thresholds)
        assert ks2.any_triggered()
        assert "daily_loss" in ks2.get_triggered_names()
        conn2.close()

    def test_reset_state_survives_file_close_reopen(
        self, db_path: Path,
    ) -> None:
        thresholds = {
            "daily_loss": 0.03,
            "weekly_loss": 0.05,
            "max_drawdown": 0.15,
            "consecutive_loss": 15.0,
            "emergency": 1.0,
        }

        # Trigger then reset
        conn1 = sqlite3.connect(str(db_path))
        ks1 = KillSwitchManager(conn1, thresholds)
        ks1.check_daily_loss(-0.05)
        ks1.reset("daily_loss", "Reviewed after test")
        assert not ks1.any_triggered()
        conn1.close()

        # Verify reset persisted
        conn2 = sqlite3.connect(str(db_path))
        ks2 = KillSwitchManager(conn2, thresholds)
        assert not ks2.any_triggered()
        conn2.close()

    def test_multiple_switches_persist(self, db_path: Path) -> None:
        thresholds = {
            "daily_loss": 0.03,
            "weekly_loss": 0.05,
            "max_drawdown": 0.15,
            "consecutive_loss": 15.0,
            "emergency": 1.0,
        }

        conn1 = sqlite3.connect(str(db_path))
        ks1 = KillSwitchManager(conn1, thresholds)
        ks1.check_daily_loss(-0.05)
        ks1.check_weekly_loss(-0.06)
        assert len(ks1.get_triggered_names()) == 2
        conn1.close()

        conn2 = sqlite3.connect(str(db_path))
        ks2 = KillSwitchManager(conn2, thresholds)
        triggered = ks2.get_triggered_names()
        assert "daily_loss" in triggered
        assert "weekly_loss" in triggered
        conn2.close()

    def test_audit_log_persists(self, db_path: Path) -> None:
        thresholds = {
            "daily_loss": 0.03,
            "weekly_loss": 0.05,
            "max_drawdown": 0.15,
            "consecutive_loss": 15.0,
            "emergency": 1.0,
        }

        conn1 = sqlite3.connect(str(db_path))
        ks1 = KillSwitchManager(conn1, thresholds)
        ks1.check_daily_loss(-0.05)
        ks1.reset("daily_loss", "Test reset")
        conn1.close()

        conn2 = sqlite3.connect(str(db_path))
        rows = conn2.execute(
            "SELECT switch_name, old_state, new_state FROM risk_kill_switch_log"
        ).fetchall()
        # At least 2 entries: ARMED->TRIGGERED, TRIGGERED->ARMED
        assert len(rows) >= 2
        conn2.close()


# ---------------------------------------------------------------------------
# Position tracker persistence (file-backed)
# ---------------------------------------------------------------------------

class TestPositionTrackerPersistence:
    def test_open_position_survives_crash(self, db_path: Path) -> None:
        """Simulate crash: open position -> close conn -> reopen -> verify."""
        conn1 = sqlite3.connect(str(db_path))
        t1 = PositionTracker(conn1)
        t1.open_position("long", 0.05, 67000.0, 1700000000000)
        t1.mark_to_market(67500.0, 1700000300000)
        conn1.close()

        conn2 = sqlite3.connect(str(db_path))
        t2 = PositionTracker(conn2)
        state = t2.get_state()
        assert state.side == PositionSide.LONG
        assert state.quantity == pytest.approx(0.05)
        assert state.entry_price == pytest.approx(67000.0)
        assert state.bars_held == 1
        conn2.close()

    def test_realized_pnl_persists_across_restart(self, db_path: Path) -> None:
        conn1 = sqlite3.connect(str(db_path))
        t1 = PositionTracker(conn1)
        t1.open_position("long", 0.05, 67000.0, 1700000000000)
        pnl = t1.close_position(67500.0, 1700000300000)
        assert pnl == pytest.approx(25.0)
        conn1.close()

        conn2 = sqlite3.connect(str(db_path))
        t2 = PositionTracker(conn2)
        state = t2.get_state()
        assert not state.is_open
        assert state.realized_pnl == pytest.approx(25.0)
        conn2.close()

    def test_daily_pnl_reset_is_intentional(self, db_path: Path) -> None:
        """Daily PnL reset is a documented, intentional operation."""
        conn1 = sqlite3.connect(str(db_path))
        t1 = PositionTracker(conn1)
        t1.open_position("long", 0.05, 67000.0, 1700000000000)
        t1.close_position(67500.0, 1700000300000)
        t1.reset_daily_pnl()
        conn1.close()

        conn2 = sqlite3.connect(str(db_path))
        t2 = PositionTracker(conn2)
        assert t2.get_state().realized_pnl == pytest.approx(0.0)
        conn2.close()


# ---------------------------------------------------------------------------
# Drawdown gate persistence (file-backed)
# ---------------------------------------------------------------------------

class TestDrawdownGatePersistence:
    def test_drawdown_state_survives_restart(self, db_path: Path) -> None:
        conn1 = sqlite3.connect(str(db_path))
        g1 = DrawdownGate(conn1, initial_equity=100_000.0)
        g1.update(100_000.0)
        g1.update(92_000.0)  # 8% drawdown
        mult1 = g1.get_multiplier()
        state1 = g1.get_state()
        conn1.close()

        conn2 = sqlite3.connect(str(db_path))
        g2 = DrawdownGate(conn2, initial_equity=100_000.0)
        assert g2.get_multiplier() == pytest.approx(mult1, rel=0.01)
        state2 = g2.get_state()
        assert state2.bars_underwater == state1.bars_underwater
        conn2.close()

    def test_peak_equity_persists(self, db_path: Path) -> None:
        conn1 = sqlite3.connect(str(db_path))
        g1 = DrawdownGate(conn1, initial_equity=100_000.0)
        g1.update(110_000.0)  # New peak
        conn1.close()

        conn2 = sqlite3.connect(str(db_path))
        g2 = DrawdownGate(conn2, initial_equity=100_000.0)
        state = g2.get_state()
        assert state.peak_equity >= 110_000.0
        conn2.close()
