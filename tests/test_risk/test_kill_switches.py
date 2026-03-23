"""Tests for KillSwitchManager: trigger, persist, reset behavior."""

from __future__ import annotations

import sqlite3

import pytest

from ep2_crypto.risk.kill_switches import (
    ALL_SWITCH_NAMES,
    KillSwitchManager,
    SwitchState,
)


@pytest.fixture
def conn() -> sqlite3.Connection:
    return sqlite3.connect(":memory:")


@pytest.fixture
def default_thresholds() -> dict[str, float]:
    return {
        "daily_loss": 0.03,
        "weekly_loss": 0.05,
        "max_drawdown": 0.15,
        "consecutive_loss": 15.0,
        "emergency": 1.0,
    }


@pytest.fixture
def manager(conn: sqlite3.Connection, default_thresholds: dict[str, float]) -> KillSwitchManager:
    return KillSwitchManager(conn, default_thresholds)


class TestInitialization:
    def test_all_switches_initialized_armed(self, manager: KillSwitchManager) -> None:
        statuses = manager.get_all_status()
        assert len(statuses) == len(ALL_SWITCH_NAMES)
        for s in statuses:
            assert s.state == SwitchState.ARMED

    def test_none_triggered_initially(self, manager: KillSwitchManager) -> None:
        assert not manager.any_triggered()
        assert manager.get_triggered_names() == []


class TestDailyLossSwitch:
    def test_no_trigger_below_threshold(self, manager: KillSwitchManager) -> None:
        result = manager.check_daily_loss(-0.02)  # 2% loss < 3% threshold
        assert not result

    def test_trigger_at_threshold(self, manager: KillSwitchManager) -> None:
        result = manager.check_daily_loss(-0.03)  # exactly 3%
        assert result
        assert manager.any_triggered()
        assert "daily_loss" in manager.get_triggered_names()

    def test_trigger_above_threshold(self, manager: KillSwitchManager) -> None:
        result = manager.check_daily_loss(-0.05)  # 5% > 3%
        assert result

    def test_positive_pnl_does_not_trigger(self, manager: KillSwitchManager) -> None:
        result = manager.check_daily_loss(0.10)
        assert not result


class TestWeeklyLossSwitch:
    def test_trigger_at_threshold(self, manager: KillSwitchManager) -> None:
        result = manager.check_weekly_loss(-0.05)
        assert result
        assert "weekly_loss" in manager.get_triggered_names()


class TestMaxDrawdownSwitch:
    def test_trigger_at_threshold(self, manager: KillSwitchManager) -> None:
        result = manager.check_max_drawdown(0.15)
        assert result
        assert "max_drawdown" in manager.get_triggered_names()

    def test_no_trigger_below(self, manager: KillSwitchManager) -> None:
        result = manager.check_max_drawdown(0.10)
        assert not result


class TestConsecutiveLossSwitch:
    def test_trigger_at_count(self, manager: KillSwitchManager) -> None:
        result = manager.check_consecutive_losses(15)
        assert result
        assert "consecutive_loss" in manager.get_triggered_names()

    def test_no_trigger_below(self, manager: KillSwitchManager) -> None:
        result = manager.check_consecutive_losses(14)
        assert not result


class TestEmergencySwitch:
    def test_manual_trigger(self, manager: KillSwitchManager) -> None:
        manager.trigger_emergency("System test")
        assert manager.any_triggered()
        assert "emergency" in manager.get_triggered_names()


class TestResetBehavior:
    def test_reset_requires_triggered_state(self, manager: KillSwitchManager) -> None:
        with pytest.raises(ValueError, match="not triggered"):
            manager.reset("daily_loss", "Testing reset on armed switch")

    def test_reset_returns_to_armed(self, manager: KillSwitchManager) -> None:
        manager.check_daily_loss(-0.05)
        assert manager.any_triggered()
        manager.reset("daily_loss", "Reviewed and safe to resume")
        assert not manager.any_triggered()

    def test_reset_logs_reason(self, manager: KillSwitchManager) -> None:
        manager.check_daily_loss(-0.05)
        manager.reset("daily_loss", "Manual review completed")
        statuses = manager.get_all_status()
        daily = next(s for s in statuses if s.name == "daily_loss")
        assert daily.reset_reason == "Manual review completed"

    def test_reset_all(self, manager: KillSwitchManager) -> None:
        manager.check_daily_loss(-0.05)
        manager.check_weekly_loss(-0.06)
        assert len(manager.get_triggered_names()) == 2
        count = manager.reset_all("Full system reset after review")
        assert count == 2
        assert not manager.any_triggered()


class TestPersistence:
    def test_triggered_state_survives_restart(
        self,
        conn: sqlite3.Connection,
        default_thresholds: dict[str, float],
    ) -> None:
        m1 = KillSwitchManager(conn, default_thresholds)
        m1.check_daily_loss(-0.05)
        assert m1.any_triggered()

        # Simulate restart
        m2 = KillSwitchManager(conn, default_thresholds)
        assert m2.any_triggered()
        assert "daily_loss" in m2.get_triggered_names()

    def test_reset_state_survives_restart(
        self,
        conn: sqlite3.Connection,
        default_thresholds: dict[str, float],
    ) -> None:
        m1 = KillSwitchManager(conn, default_thresholds)
        m1.check_daily_loss(-0.05)
        m1.reset("daily_loss", "Reviewed")

        m2 = KillSwitchManager(conn, default_thresholds)
        assert not m2.any_triggered()


class TestStaysTriggered:
    def test_already_triggered_stays_triggered(self, manager: KillSwitchManager) -> None:
        manager.check_daily_loss(-0.05)
        # Even if loss improves, switch stays triggered
        result = manager.check_daily_loss(-0.01)
        assert result  # still triggered

    def test_multiple_checks_do_not_duplicate(self, manager: KillSwitchManager) -> None:
        manager.check_daily_loss(-0.05)
        manager.check_daily_loss(-0.05)
        assert len(manager.get_triggered_names()) == 1
