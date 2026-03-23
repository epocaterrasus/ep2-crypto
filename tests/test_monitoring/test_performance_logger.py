"""Tests for the performance logger (flight recorder)."""

from __future__ import annotations

import sqlite3

import pytest

from ep2_crypto.monitoring.performance_logger import (
    BarStateRecord,
    Direction,
    PerformanceLogger,
    TradeRecord,
)


@pytest.fixture()
def conn() -> sqlite3.Connection:
    """In-memory SQLite connection."""
    c = sqlite3.connect(":memory:")
    yield c
    c.close()


@pytest.fixture()
def perf_logger(conn: sqlite3.Connection) -> PerformanceLogger:
    return PerformanceLogger(conn)


class TestTableCreation:
    def test_trade_log_table_exists(self, conn: sqlite3.Connection) -> None:
        PerformanceLogger(conn)
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='trade_log'"
        )
        assert cursor.fetchone() is not None

    def test_bar_state_table_exists(self, conn: sqlite3.Connection) -> None:
        PerformanceLogger(conn)
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='bar_state_log'"
        )
        assert cursor.fetchone() is not None

    def test_indices_created(self, conn: sqlite3.Connection) -> None:
        PerformanceLogger(conn)
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_%'"
        )
        indices = {row[0] for row in cursor.fetchall()}
        assert "idx_trade_log_ts" in indices
        assert "idx_bar_state_log_ts" in indices

    def test_idempotent_creation(self, conn: sqlite3.Connection) -> None:
        PerformanceLogger(conn)
        PerformanceLogger(conn)  # Should not raise


class TestTradeLogging:
    def test_log_basic_trade(self, perf_logger: PerformanceLogger) -> None:
        record = TradeRecord(
            timestamp_ms=1_000_000,
            direction=Direction.LONG,
            predicted_confidence=0.75,
            predicted_magnitude=0.002,
            entry_price=50000.0,
            position_size=0.01,
        )
        row_id = perf_logger.log_trade(record)
        assert row_id > 0

    def test_log_trade_with_full_fields(self, perf_logger: PerformanceLogger) -> None:
        record = TradeRecord(
            timestamp_ms=2_000_000,
            direction=Direction.SHORT,
            predicted_confidence=0.82,
            predicted_magnitude=0.003,
            actual_direction=Direction.SHORT,
            actual_return=-0.0025,
            pnl=12.50,
            slippage_expected=0.0002,
            slippage_actual=0.0003,
            latency_ms=45.2,
            regime="trending",
            features={"obi_1_3": 0.35, "rsi_14": 72.0},
            entry_price=50000.0,
            exit_price=49875.0,
            position_size=0.05,
            meta_info={"model_version": "v1.2"},
        )
        row_id = perf_logger.log_trade(record)
        assert row_id > 0

    def test_trade_count_increments(self, perf_logger: PerformanceLogger) -> None:
        for i in range(5):
            perf_logger.log_trade(
                TradeRecord(
                    timestamp_ms=i * 1000,
                    direction=Direction.LONG,
                    predicted_confidence=0.6,
                    predicted_magnitude=0.001,
                )
            )
        assert perf_logger.session_trade_count == 5
        assert perf_logger.get_trade_count() == 5

    def test_update_trade_outcome(self, perf_logger: PerformanceLogger) -> None:
        record = TradeRecord(
            timestamp_ms=3_000_000,
            direction=Direction.LONG,
            predicted_confidence=0.70,
            predicted_magnitude=0.002,
            entry_price=50000.0,
        )
        trade_id = perf_logger.log_trade(record)

        perf_logger.update_trade_outcome(
            trade_id=trade_id,
            actual_direction=Direction.LONG,
            actual_return=0.002,
            pnl=10.0,
            slippage_actual=0.0001,
            exit_price=50100.0,
        )

        trades = perf_logger.query_trades()
        assert len(trades) == 1
        assert trades[0]["actual_direction"] == "long"
        assert trades[0]["pnl"] == 10.0
        assert trades[0]["exit_price"] == 50100.0


class TestBarStateLogging:
    def test_log_basic_bar_state(self, perf_logger: PerformanceLogger) -> None:
        record = BarStateRecord(
            timestamp_ms=1_000_000,
            bar_close=50000.0,
        )
        row_id = perf_logger.log_bar_state(record)
        assert row_id > 0

    def test_log_full_bar_state(self, perf_logger: PerformanceLogger) -> None:
        record = BarStateRecord(
            timestamp_ms=2_000_000,
            bar_close=50100.0,
            regime="trending",
            regime_confidence=0.85,
            model_prediction="long",
            model_confidence=0.72,
            risk_state={"daily_pnl": -0.005, "drawdown": 0.02},
            feature_values={"obi": 0.3, "rsi": 65.0, "vol": 0.35},
            kill_switch_active=False,
            drawdown_multiplier=0.95,
            volatility_ann=0.42,
            position_open=True,
            equity=100000.0,
        )
        row_id = perf_logger.log_bar_state(record)
        assert row_id > 0

    def test_bar_count_increments(self, perf_logger: PerformanceLogger) -> None:
        for i in range(10):
            perf_logger.log_bar_state(
                BarStateRecord(timestamp_ms=i * 300_000, bar_close=50000.0 + i)
            )
        assert perf_logger.session_bar_count == 10
        assert perf_logger.get_bar_count() == 10


class TestQueries:
    def _insert_trades(self, perf_logger: PerformanceLogger, n: int = 10) -> list[int]:
        ids = []
        for i in range(n):
            pnl = 10.0 if i % 3 != 0 else -5.0  # ~67% win rate
            trade_id = perf_logger.log_trade(
                TradeRecord(
                    timestamp_ms=(i + 1) * 300_000,
                    direction=Direction.LONG if i % 2 == 0 else Direction.SHORT,
                    predicted_confidence=0.6 + i * 0.02,
                    predicted_magnitude=0.001,
                    actual_direction=Direction.LONG if i % 2 == 0 else Direction.SHORT,
                    actual_return=0.001 if i % 3 != 0 else -0.0005,
                    pnl=pnl,
                    entry_price=50000.0,
                )
            )
            ids.append(trade_id)
        return ids

    def test_query_trades_all(self, perf_logger: PerformanceLogger) -> None:
        self._insert_trades(perf_logger, 5)
        trades = perf_logger.query_trades()
        assert len(trades) == 5

    def test_query_trades_time_range(self, perf_logger: PerformanceLogger) -> None:
        self._insert_trades(perf_logger, 10)
        trades = perf_logger.query_trades(start_ms=300_000 * 3, end_ms=300_000 * 7)
        assert all(300_000 * 3 <= t["timestamp_ms"] <= 300_000 * 7 for t in trades)

    def test_query_trades_limit(self, perf_logger: PerformanceLogger) -> None:
        self._insert_trades(perf_logger, 10)
        trades = perf_logger.query_trades(limit=3)
        assert len(trades) == 3

    def test_query_bar_states_all(self, perf_logger: PerformanceLogger) -> None:
        for i in range(5):
            perf_logger.log_bar_state(
                BarStateRecord(timestamp_ms=(i + 1) * 300_000, bar_close=50000.0)
            )
        states = perf_logger.query_bar_states()
        assert len(states) == 5

    def test_query_bar_states_time_range(self, perf_logger: PerformanceLogger) -> None:
        for i in range(10):
            perf_logger.log_bar_state(
                BarStateRecord(timestamp_ms=(i + 1) * 300_000, bar_close=50000.0)
            )
        states = perf_logger.query_bar_states(start_ms=300_000 * 3, end_ms=300_000 * 7)
        assert all(300_000 * 3 <= s["timestamp_ms"] <= 300_000 * 7 for s in states)

    def test_get_recent_pnl(self, perf_logger: PerformanceLogger) -> None:
        self._insert_trades(perf_logger, 10)
        pnls = perf_logger.get_recent_pnl(5)
        assert len(pnls) == 5
        assert all(isinstance(p, float) for p in pnls)

    def test_get_recent_returns(self, perf_logger: PerformanceLogger) -> None:
        self._insert_trades(perf_logger, 10)
        returns = perf_logger.get_recent_returns(5)
        assert len(returns) == 5

    def test_get_win_rate(self, perf_logger: PerformanceLogger) -> None:
        self._insert_trades(perf_logger, 10)
        wr = perf_logger.get_win_rate(10)
        assert wr is not None
        assert 0.0 <= wr <= 1.0

    def test_get_win_rate_empty(self, perf_logger: PerformanceLogger) -> None:
        assert perf_logger.get_win_rate() is None

    def test_get_completed_trade_count(self, perf_logger: PerformanceLogger) -> None:
        self._insert_trades(perf_logger, 5)
        assert perf_logger.get_completed_trade_count() == 5

        # Log one without outcome
        perf_logger.log_trade(
            TradeRecord(
                timestamp_ms=999_000_000,
                direction=Direction.LONG,
                predicted_confidence=0.6,
                predicted_magnitude=0.001,
            )
        )
        assert perf_logger.get_completed_trade_count() == 5
        assert perf_logger.get_trade_count() == 6

    def test_get_latest_bar_state(self, perf_logger: PerformanceLogger) -> None:
        for i in range(5):
            perf_logger.log_bar_state(
                BarStateRecord(
                    timestamp_ms=(i + 1) * 300_000,
                    bar_close=50000.0 + i * 100,
                )
            )
        latest = perf_logger.get_latest_bar_state()
        assert latest is not None
        assert latest["timestamp_ms"] == 5 * 300_000

    def test_get_latest_bar_state_empty(self, perf_logger: PerformanceLogger) -> None:
        assert perf_logger.get_latest_bar_state() is None

    def test_get_cumulative_pnl(self, perf_logger: PerformanceLogger) -> None:
        self._insert_trades(perf_logger, 10)
        cum_pnl = perf_logger.get_cumulative_pnl()
        assert isinstance(cum_pnl, float)

    def test_get_cumulative_pnl_empty(self, perf_logger: PerformanceLogger) -> None:
        assert perf_logger.get_cumulative_pnl() == 0.0

    def test_get_accuracy(self, perf_logger: PerformanceLogger) -> None:
        self._insert_trades(perf_logger, 10)
        acc = perf_logger.get_accuracy()
        assert acc is not None
        # All predictions match actual in our test data
        assert acc == 1.0

    def test_get_accuracy_empty(self, perf_logger: PerformanceLogger) -> None:
        assert perf_logger.get_accuracy() is None


class TestEdgeCases:
    def test_features_json_roundtrip(self, perf_logger: PerformanceLogger) -> None:
        features = {"obi_1_3": 0.35, "rsi_14": 72.0, "vol_ewma": 0.42}
        perf_logger.log_trade(
            TradeRecord(
                timestamp_ms=1_000_000,
                direction=Direction.LONG,
                predicted_confidence=0.7,
                predicted_magnitude=0.002,
                features=features,
            )
        )
        trades = perf_logger.query_trades()
        import json

        stored_features = json.loads(trades[0]["features_json"])
        assert stored_features == features

    def test_meta_info_json_roundtrip(self, perf_logger: PerformanceLogger) -> None:
        meta = {"model_version": "v1.2", "retrain_id": 42}
        perf_logger.log_trade(
            TradeRecord(
                timestamp_ms=1_000_000,
                direction=Direction.SHORT,
                predicted_confidence=0.8,
                predicted_magnitude=0.003,
                meta_info=meta,
            )
        )
        trades = perf_logger.query_trades()
        import json

        stored_meta = json.loads(trades[0]["meta_json"])
        assert stored_meta == meta

    def test_risk_state_json_roundtrip(self, perf_logger: PerformanceLogger) -> None:
        risk = {"daily_pnl": -0.01, "kill_switches": ["none"]}
        perf_logger.log_bar_state(
            BarStateRecord(
                timestamp_ms=1_000_000,
                bar_close=50000.0,
                risk_state=risk,
            )
        )
        states = perf_logger.query_bar_states()
        import json

        stored_risk = json.loads(states[0]["risk_state_json"])
        assert stored_risk == risk

    def test_direction_enum_values(self) -> None:
        assert Direction.LONG.value == "long"
        assert Direction.SHORT.value == "short"
        assert Direction.FLAT.value == "flat"

    def test_write_latency(self, perf_logger: PerformanceLogger) -> None:
        latency = perf_logger.measure_latency()
        # Should be under 10ms (10,000 microseconds) on any reasonable system
        assert latency < 10_000
