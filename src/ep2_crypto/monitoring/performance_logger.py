"""Flight recorder: logs every trade and every bar state snapshot.

Everything else in monitoring depends on this data — it is the single source
of truth for what the system actually did vs what it predicted.
"""

from __future__ import annotations

import json
import sqlite3  # noqa: TC003
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class Direction(StrEnum):
    """Trade direction."""

    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


@dataclass(frozen=True)
class TradeRecord:
    """Immutable record of a single trade."""

    timestamp_ms: int
    direction: Direction
    predicted_confidence: float
    predicted_magnitude: float
    actual_direction: Direction | None = None
    actual_return: float | None = None
    pnl: float | None = None
    slippage_expected: float | None = None
    slippage_actual: float | None = None
    latency_ms: float | None = None
    regime: str | None = None
    features: dict[str, float] = field(default_factory=dict)
    entry_price: float | None = None
    exit_price: float | None = None
    position_size: float | None = None
    meta_info: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BarStateRecord:
    """Immutable snapshot of system state at a single bar."""

    timestamp_ms: int
    bar_close: float
    regime: str | None = None
    regime_confidence: float | None = None
    model_prediction: str | None = None
    model_confidence: float | None = None
    risk_state: dict[str, Any] = field(default_factory=dict)
    feature_values: dict[str, float] = field(default_factory=dict)
    kill_switch_active: bool = False
    drawdown_multiplier: float | None = None
    volatility_ann: float | None = None
    position_open: bool = False
    equity: float | None = None


_TRADE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS trade_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp_ms INTEGER NOT NULL,
    direction TEXT NOT NULL,
    predicted_confidence REAL NOT NULL,
    predicted_magnitude REAL NOT NULL,
    actual_direction TEXT,
    actual_return REAL,
    pnl REAL,
    slippage_expected REAL,
    slippage_actual REAL,
    latency_ms REAL,
    regime TEXT,
    features_json TEXT,
    entry_price REAL,
    exit_price REAL,
    position_size REAL,
    meta_json TEXT,
    created_at REAL NOT NULL DEFAULT (strftime('%s', 'now'))
)
"""

_TRADE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_trade_log_ts ON trade_log (timestamp_ms)
"""

_BAR_STATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS bar_state_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp_ms INTEGER NOT NULL,
    bar_close REAL NOT NULL,
    regime TEXT,
    regime_confidence REAL,
    model_prediction TEXT,
    model_confidence REAL,
    risk_state_json TEXT,
    feature_values_json TEXT,
    kill_switch_active INTEGER NOT NULL DEFAULT 0,
    drawdown_multiplier REAL,
    volatility_ann REAL,
    position_open INTEGER NOT NULL DEFAULT 0,
    equity REAL,
    created_at REAL NOT NULL DEFAULT (strftime('%s', 'now'))
)
"""

_BAR_STATE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_bar_state_log_ts ON bar_state_log (timestamp_ms)
"""

_INSERT_TRADE_SQL = """
INSERT INTO trade_log (
    timestamp_ms, direction, predicted_confidence, predicted_magnitude,
    actual_direction, actual_return, pnl, slippage_expected, slippage_actual,
    latency_ms, regime, features_json, entry_price, exit_price,
    position_size, meta_json
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

_INSERT_BAR_STATE_SQL = """
INSERT INTO bar_state_log (
    timestamp_ms, bar_close, regime, regime_confidence,
    model_prediction, model_confidence, risk_state_json,
    feature_values_json, kill_switch_active, drawdown_multiplier,
    volatility_ann, position_open, equity
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


class PerformanceLogger:
    """Flight recorder that persists every trade and bar state to SQLite.

    Thread-safe via SQLite's WAL mode. All queries are parameterized.
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self._ensure_tables()
        self._trade_count = 0
        self._bar_count = 0

    def _ensure_tables(self) -> None:
        """Create tables and indices if they don't exist."""
        self._conn.execute(_TRADE_TABLE_SQL)
        self._conn.execute(_TRADE_INDEX_SQL)
        self._conn.execute(_BAR_STATE_TABLE_SQL)
        self._conn.execute(_BAR_STATE_INDEX_SQL)
        self._conn.commit()

    def log_trade(self, record: TradeRecord) -> int:
        """Log a trade record. Returns the row ID."""
        cursor = self._conn.execute(
            _INSERT_TRADE_SQL,
            (
                record.timestamp_ms,
                record.direction.value,
                record.predicted_confidence,
                record.predicted_magnitude,
                record.actual_direction.value if record.actual_direction else None,
                record.actual_return,
                record.pnl,
                record.slippage_expected,
                record.slippage_actual,
                record.latency_ms,
                record.regime,
                json.dumps(record.features) if record.features else None,
                record.entry_price,
                record.exit_price,
                record.position_size,
                json.dumps(record.meta_info) if record.meta_info else None,
            ),
        )
        self._conn.commit()
        self._trade_count += 1
        row_id = cursor.lastrowid or 0
        logger.info(
            "trade_logged",
            trade_id=row_id,
            direction=record.direction.value,
            confidence=record.predicted_confidence,
            pnl=record.pnl,
        )
        return row_id

    def log_bar_state(self, record: BarStateRecord) -> int:
        """Log a bar state snapshot. Returns the row ID."""
        cursor = self._conn.execute(
            _INSERT_BAR_STATE_SQL,
            (
                record.timestamp_ms,
                record.bar_close,
                record.regime,
                record.regime_confidence,
                record.model_prediction,
                record.model_confidence,
                json.dumps(record.risk_state) if record.risk_state else None,
                json.dumps(record.feature_values) if record.feature_values else None,
                int(record.kill_switch_active),
                record.drawdown_multiplier,
                record.volatility_ann,
                int(record.position_open),
                record.equity,
            ),
        )
        self._conn.commit()
        self._bar_count += 1
        return cursor.lastrowid or 0

    def update_trade_outcome(
        self,
        trade_id: int,
        actual_direction: Direction,
        actual_return: float,
        pnl: float,
        slippage_actual: float | None = None,
        exit_price: float | None = None,
    ) -> None:
        """Update a trade with its actual outcome (called after trade closes)."""
        self._conn.execute(
            """UPDATE trade_log SET
                actual_direction = ?, actual_return = ?, pnl = ?,
                slippage_actual = ?, exit_price = ?
            WHERE id = ?""",
            (
                actual_direction.value,
                actual_return,
                pnl,
                slippage_actual,
                exit_price,
                trade_id,
            ),
        )
        self._conn.commit()
        logger.info(
            "trade_outcome_updated",
            trade_id=trade_id,
            actual_direction=actual_direction.value,
            pnl=pnl,
        )

    def query_trades(
        self,
        start_ms: int | None = None,
        end_ms: int | None = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        """Query trade records within a time range."""
        conditions: list[str] = []
        params: list[Any] = []
        if start_ms is not None:
            conditions.append("timestamp_ms >= ?")
            params.append(start_ms)
        if end_ms is not None:
            conditions.append("timestamp_ms <= ?")
            params.append(end_ms)

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        sql = f"SELECT * FROM trade_log {where} ORDER BY timestamp_ms DESC LIMIT ?"  # noqa: S608
        params.append(limit)

        cursor = self._conn.execute(sql, params)
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        return [dict(zip(columns, row, strict=True)) for row in rows]

    def query_bar_states(
        self,
        start_ms: int | None = None,
        end_ms: int | None = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        """Query bar state snapshots within a time range."""
        conditions: list[str] = []
        params: list[Any] = []
        if start_ms is not None:
            conditions.append("timestamp_ms >= ?")
            params.append(start_ms)
        if end_ms is not None:
            conditions.append("timestamp_ms <= ?")
            params.append(end_ms)

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        sql = f"SELECT * FROM bar_state_log {where} ORDER BY timestamp_ms DESC LIMIT ?"  # noqa: S608
        params.append(limit)

        cursor = self._conn.execute(sql, params)
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        return [dict(zip(columns, row, strict=True)) for row in rows]

    def get_recent_pnl(self, n_trades: int = 50) -> list[float]:
        """Get the PnL of the N most recent completed trades."""
        cursor = self._conn.execute(
            "SELECT pnl FROM trade_log WHERE pnl IS NOT NULL "
            "ORDER BY timestamp_ms DESC LIMIT ?",
            (n_trades,),
        )
        return [row[0] for row in cursor.fetchall()]

    def get_recent_returns(self, n_trades: int = 50) -> list[float]:
        """Get actual returns of the N most recent completed trades."""
        cursor = self._conn.execute(
            "SELECT actual_return FROM trade_log WHERE actual_return IS NOT NULL "
            "ORDER BY timestamp_ms DESC LIMIT ?",
            (n_trades,),
        )
        return [row[0] for row in cursor.fetchall()]

    def get_win_rate(self, n_trades: int = 50) -> float | None:
        """Calculate win rate over the last N completed trades."""
        cursor = self._conn.execute(
            "SELECT pnl FROM trade_log WHERE pnl IS NOT NULL "
            "ORDER BY timestamp_ms DESC LIMIT ?",
            (n_trades,),
        )
        pnls = [row[0] for row in cursor.fetchall()]
        if not pnls:
            return None
        wins = sum(1 for p in pnls if p > 0)
        return wins / len(pnls)

    def get_trade_count(self) -> int:
        """Total number of trades logged (including those without outcomes)."""
        cursor = self._conn.execute("SELECT COUNT(*) FROM trade_log")
        return cursor.fetchone()[0]

    def get_completed_trade_count(self) -> int:
        """Number of trades with outcomes filled in."""
        cursor = self._conn.execute(
            "SELECT COUNT(*) FROM trade_log WHERE pnl IS NOT NULL"
        )
        return cursor.fetchone()[0]

    def get_bar_count(self) -> int:
        """Total number of bar states logged."""
        cursor = self._conn.execute("SELECT COUNT(*) FROM bar_state_log")
        return cursor.fetchone()[0]

    def get_latest_bar_state(self) -> dict[str, Any] | None:
        """Get the most recent bar state snapshot."""
        cursor = self._conn.execute(
            "SELECT * FROM bar_state_log ORDER BY timestamp_ms DESC LIMIT 1"
        )
        row = cursor.fetchone()
        if row is None:
            return None
        columns = [desc[0] for desc in cursor.description]
        return dict(zip(columns, row, strict=True))

    def get_cumulative_pnl(self) -> float:
        """Sum of all completed trade PnLs."""
        cursor = self._conn.execute(
            "SELECT COALESCE(SUM(pnl), 0.0) FROM trade_log WHERE pnl IS NOT NULL"
        )
        return cursor.fetchone()[0]

    def get_accuracy(self, n_trades: int | None = None) -> float | None:
        """Directional accuracy: how often predicted direction matched actual."""
        limit_clause = f"LIMIT {n_trades}" if n_trades else ""
        cursor = self._conn.execute(
            f"SELECT direction, actual_direction FROM trade_log "  # noqa: S608
            f"WHERE actual_direction IS NOT NULL "
            f"ORDER BY timestamp_ms DESC {limit_clause}"
        )
        rows = cursor.fetchall()
        if not rows:
            return None
        correct = sum(1 for pred, actual in rows if pred == actual)
        return correct / len(rows)

    @property
    def session_trade_count(self) -> int:
        """Trades logged in this session (since init)."""
        return self._trade_count

    @property
    def session_bar_count(self) -> int:
        """Bar states logged in this session (since init)."""
        return self._bar_count

    def measure_latency(self) -> float:
        """Measure write latency for a single trade log (microseconds)."""
        start = time.perf_counter()
        record = TradeRecord(
            timestamp_ms=0,
            direction=Direction.FLAT,
            predicted_confidence=0.0,
            predicted_magnitude=0.0,
        )
        self.log_trade(record)
        elapsed_us = (time.perf_counter() - start) * 1_000_000
        # Clean up the benchmark row
        self._conn.execute(
            "DELETE FROM trade_log WHERE timestamp_ms = 0 AND direction = 'flat'"
        )
        self._conn.commit()
        self._trade_count -= 1
        return elapsed_us
