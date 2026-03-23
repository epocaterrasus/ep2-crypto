"""Tests for PositionTracker: open, close, mark-to-market, persistence."""

from __future__ import annotations

import sqlite3

import pytest

from ep2_crypto.risk.position_tracker import PositionSide, PositionState, PositionTracker


@pytest.fixture
def conn() -> sqlite3.Connection:
    return sqlite3.connect(":memory:")


@pytest.fixture
def tracker(conn: sqlite3.Connection) -> PositionTracker:
    return PositionTracker(conn)


class TestPositionTrackerOpenClose:
    def test_initial_state_is_flat(self, tracker: PositionTracker) -> None:
        state = tracker.get_state()
        assert state.side == PositionSide.FLAT
        assert state.quantity == 0.0
        assert not state.is_open

    def test_open_long_position(self, tracker: PositionTracker) -> None:
        ts = 1700000000000
        state = tracker.open_position("long", 0.05, 67000.0, ts)
        assert state.side == PositionSide.LONG
        assert state.quantity == 0.05
        assert state.entry_price == 67000.0
        assert state.entry_time_ms == ts
        assert state.is_open

    def test_open_short_position(self, tracker: PositionTracker) -> None:
        state = tracker.open_position("short", 0.1, 68000.0, 1700000000000)
        assert state.side == PositionSide.SHORT
        assert state.quantity == 0.1

    def test_cannot_open_when_position_exists(self, tracker: PositionTracker) -> None:
        tracker.open_position("long", 0.05, 67000.0, 1700000000000)
        with pytest.raises(ValueError, match="Cannot open position"):
            tracker.open_position("short", 0.05, 68000.0, 1700000300000)

    def test_close_long_positive_pnl(self, tracker: PositionTracker) -> None:
        tracker.open_position("long", 0.05, 67000.0, 1700000000000)
        pnl = tracker.close_position(67500.0, 1700000300000)
        assert pnl == pytest.approx(0.05 * 500.0, rel=1e-6)  # $25
        state = tracker.get_state()
        assert not state.is_open
        assert state.realized_pnl == pytest.approx(25.0, rel=1e-6)

    def test_close_long_negative_pnl(self, tracker: PositionTracker) -> None:
        tracker.open_position("long", 0.05, 67000.0, 1700000000000)
        pnl = tracker.close_position(66000.0, 1700000300000)
        assert pnl == pytest.approx(-50.0, rel=1e-6)

    def test_close_short_positive_pnl(self, tracker: PositionTracker) -> None:
        tracker.open_position("short", 0.1, 68000.0, 1700000000000)
        pnl = tracker.close_position(67000.0, 1700000300000)
        assert pnl == pytest.approx(100.0, rel=1e-6)

    def test_close_short_negative_pnl(self, tracker: PositionTracker) -> None:
        tracker.open_position("short", 0.1, 68000.0, 1700000000000)
        pnl = tracker.close_position(69000.0, 1700000300000)
        assert pnl == pytest.approx(-100.0, rel=1e-6)

    def test_cannot_close_when_flat(self, tracker: PositionTracker) -> None:
        with pytest.raises(ValueError, match="no open position"):
            tracker.close_position(67000.0, 1700000000000)

    def test_invalid_side(self, tracker: PositionTracker) -> None:
        with pytest.raises(ValueError, match="Invalid side"):
            tracker.open_position("invalid", 0.05, 67000.0, 1700000000000)

    def test_zero_quantity_rejected(self, tracker: PositionTracker) -> None:
        with pytest.raises(ValueError, match="positive"):
            tracker.open_position("long", 0.0, 67000.0, 1700000000000)

    def test_negative_price_rejected(self, tracker: PositionTracker) -> None:
        with pytest.raises(ValueError, match="positive"):
            tracker.open_position("long", 0.05, -1.0, 1700000000000)


class TestMarkToMarket:
    def test_mtm_updates_unrealized_pnl(self, tracker: PositionTracker) -> None:
        tracker.open_position("long", 0.05, 67000.0, 1700000000000)
        state = tracker.mark_to_market(67500.0, 1700000300000)
        assert state.unrealized_pnl == pytest.approx(25.0, rel=1e-6)
        assert state.bars_held == 1

    def test_mtm_tracks_max_adverse_excursion(self, tracker: PositionTracker) -> None:
        tracker.open_position("long", 0.05, 67000.0, 1700000000000)
        tracker.mark_to_market(66500.0, 1700000300000)  # -25 loss
        tracker.mark_to_market(67500.0, 1700000600000)  # +25 gain
        state = tracker.get_state()
        assert state.max_adverse_excursion == pytest.approx(-25.0, rel=1e-6)
        assert state.max_favorable_excursion == pytest.approx(25.0, rel=1e-6)

    def test_mtm_increments_bars_held(self, tracker: PositionTracker) -> None:
        tracker.open_position("long", 0.05, 67000.0, 1700000000000)
        for i in range(5):
            tracker.mark_to_market(67000.0, 1700000000000 + (i + 1) * 300000)
        assert tracker.get_holding_bars() == 5

    def test_mtm_on_flat_position(self, tracker: PositionTracker) -> None:
        state = tracker.mark_to_market(67000.0, 1700000000000)
        assert state.unrealized_pnl == 0.0
        assert state.bars_held == 0


class TestPersistence:
    def test_state_survives_restart(self, conn: sqlite3.Connection) -> None:
        t1 = PositionTracker(conn)
        t1.open_position("long", 0.05, 67000.0, 1700000000000)
        t1.mark_to_market(67500.0, 1700000300000)

        # Simulate restart: create new tracker on same connection
        t2 = PositionTracker(conn)
        state = t2.get_state()
        assert state.side == PositionSide.LONG
        assert state.quantity == 0.05
        assert state.entry_price == 67000.0

    def test_realized_pnl_persists(self, conn: sqlite3.Connection) -> None:
        t1 = PositionTracker(conn)
        t1.open_position("long", 0.05, 67000.0, 1700000000000)
        t1.close_position(67500.0, 1700000300000)

        t2 = PositionTracker(conn)
        state = t2.get_state()
        assert state.realized_pnl == pytest.approx(25.0, rel=1e-6)

    def test_daily_pnl_reset(self, tracker: PositionTracker) -> None:
        tracker.open_position("long", 0.05, 67000.0, 1700000000000)
        tracker.close_position(67500.0, 1700000300000)
        assert tracker.get_state().realized_pnl == pytest.approx(25.0, rel=1e-6)
        tracker.reset_daily_pnl()
        assert tracker.get_state().realized_pnl == 0.0


class TestPositionStateProperties:
    def test_notional_usd(self) -> None:
        state = PositionState(
            side=PositionSide.LONG,
            quantity=0.05,
            entry_price=67000.0,
        )
        assert state.notional_usd == pytest.approx(3350.0)

    def test_holding_duration(self) -> None:
        state = PositionState(
            side=PositionSide.LONG,
            quantity=0.05,
            entry_price=67000.0,
            entry_time_ms=1700000000000,
        )
        duration = state.holding_duration_ms(1700000300000)
        assert duration == 300000

    def test_flat_holding_duration(self) -> None:
        state = PositionState()
        assert state.holding_duration_ms(1700000300000) == 0
