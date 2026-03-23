"""Tests for database schema creation via DatabaseConnection + create_tables."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from ep2_crypto.config import DatabaseConfig
from ep2_crypto.db.connection import DatabaseConnection
from ep2_crypto.db.schema import _HYPERTABLE_TABLES, _INDEXES, create_tables

EXPECTED_TABLES = [
    "agg_trades",
    "cross_market",
    "feature_snapshot",
    "funding_rate",
    "liquidation",
    "ohlcv",
    "onchain_whale",
    "open_interest",
    "orderbook_snapshot",
    "prediction",
    "regime_label",
]


def _make_sqlite_db(path: str | Path = ":memory:") -> DatabaseConnection:
    """Return an initialised in-memory SQLite DatabaseConnection."""
    cfg = DatabaseConfig(backend="sqlite", sqlite_path=Path(str(path)))
    db = DatabaseConnection(cfg)
    create_tables(db)
    return db


class TestCreateTables:
    def test_creates_all_tables(self) -> None:
        db = _make_sqlite_db()
        conn = db.get_connection()
        tables = sorted(
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
            ).fetchall()
        )
        for expected in EXPECTED_TABLES:
            assert expected in tables, f"Missing table: {expected}"

    def test_table_count_matches(self) -> None:
        db = _make_sqlite_db()
        conn = db.get_connection()
        count = conn.execute(
            "SELECT count(*) FROM sqlite_master WHERE type = 'table'"
        ).fetchone()[0]
        assert count == len(EXPECTED_TABLES)

    def test_indexes_created(self) -> None:
        db = _make_sqlite_db()
        conn = db.get_connection()
        indexes = [
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'index' AND name LIKE 'idx_%'"
            ).fetchall()
        ]
        assert len(indexes) == len(_INDEXES)

    def test_wal_mode_enabled(self) -> None:
        db = _make_sqlite_db()
        conn = db.get_connection()
        journal = conn.execute("PRAGMA journal_mode").fetchone()[0]
        # In-memory databases always use "memory" journal mode — WAL is set for file DBs.
        assert journal in ("wal", "memory")

    def test_foreign_keys_enabled(self) -> None:
        db = _make_sqlite_db()
        conn = db.get_connection()
        fk = conn.execute("PRAGMA foreign_keys").fetchone()[0]
        assert fk == 1

    def test_idempotent_creation(self) -> None:
        """Calling create_tables twice must not raise."""
        db = _make_sqlite_db()
        create_tables(db)  # second call
        conn = db.get_connection()
        count = conn.execute(
            "SELECT count(*) FROM sqlite_master WHERE type = 'table'"
        ).fetchone()[0]
        assert count == len(EXPECTED_TABLES)

    def test_ohlcv_primary_key(self) -> None:
        db = _make_sqlite_db()
        conn = db.get_connection()
        conn.execute(
            "INSERT INTO ohlcv VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (1000, "BTC/USDT", "5m", 100.0, 101.0, 99.0, 100.5, 10.0, 1005.0, 50),
        )
        conn.commit()
        # Duplicate should be silently replaced via INSERT OR REPLACE — confirm row still exists.
        conn.execute(
            "INSERT OR REPLACE INTO ohlcv VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (1000, "BTC/USDT", "5m", 100.0, 101.0, 99.0, 100.5, 10.0, 1005.0, 50),
        )
        conn.commit()
        count = conn.execute("SELECT count(*) FROM ohlcv").fetchone()[0]
        assert count == 1

    def test_row_factory_set(self) -> None:
        db = _make_sqlite_db()
        conn = db.get_connection()
        assert conn.row_factory == sqlite3.Row

    def test_file_based_wal(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        db = _make_sqlite_db(db_path)
        conn = db.get_connection()
        journal = conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert journal == "wal"
        db.close()

    def test_all_expected_hypertable_tables_present(self) -> None:
        """Verify the hypertable list covers all time-series tables."""
        assert set(EXPECTED_TABLES) == set(_HYPERTABLE_TABLES)

    def test_placeholder_sqlite(self) -> None:
        db = _make_sqlite_db()
        assert db.placeholder == "?"

    def test_fmt_substitution(self) -> None:
        db = _make_sqlite_db()
        result = db.fmt("SELECT * FROM ohlcv WHERE symbol = {p}")
        assert result == "SELECT * FROM ohlcv WHERE symbol = ?"

    def test_backend_property(self) -> None:
        db = _make_sqlite_db()
        assert db.backend == "sqlite"
