"""Tests for database schema creation."""

import sqlite3

from ep2_crypto.db.schema import (
    ALL_TABLES,
    INDEXES,
    create_tables,
)

EXPECTED_TABLES = [
    "ohlcv",
    "orderbook_snapshot",
    "agg_trades",
    "funding_rate",
    "open_interest",
    "liquidation",
    "cross_market",
    "onchain_whale",
    "regime_label",
    "prediction",
    "feature_snapshot",
]


class TestCreateTables:
    def test_creates_all_tables(self):
        conn = create_tables(":memory:")
        tables = [
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
            ).fetchall()
        ]
        for expected in EXPECTED_TABLES:
            assert expected in tables, f"Missing table: {expected}"
        conn.close()

    def test_table_count_matches(self):
        conn = create_tables(":memory:")
        count = conn.execute("SELECT count(*) FROM sqlite_master WHERE type = 'table'").fetchone()[
            0
        ]
        assert count == len(EXPECTED_TABLES)
        conn.close()

    def test_indexes_created(self):
        conn = create_tables(":memory:")
        indexes = [
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'index' AND name LIKE 'idx_%'"
            ).fetchall()
        ]
        assert len(indexes) == len(INDEXES)
        conn.close()

    def test_wal_mode_enabled(self):
        conn = create_tables(":memory:")
        # In-memory databases can't use WAL, but the PRAGMA is set
        # Test with a file-based db would verify this
        journal = conn.execute("PRAGMA journal_mode").fetchone()[0]
        # :memory: returns "memory" for journal_mode
        assert journal in ("wal", "memory")
        conn.close()

    def test_foreign_keys_enabled(self):
        conn = create_tables(":memory:")
        fk = conn.execute("PRAGMA foreign_keys").fetchone()[0]
        assert fk == 1
        conn.close()

    def test_idempotent_creation(self):
        conn = create_tables(":memory:")
        # Run creation DDL again — should not raise
        for table_ddl in ALL_TABLES:
            conn.execute(table_ddl)
        conn.commit()
        count = conn.execute("SELECT count(*) FROM sqlite_master WHERE type = 'table'").fetchone()[
            0
        ]
        assert count == len(EXPECTED_TABLES)
        conn.close()

    def test_ohlcv_primary_key(self):
        conn = create_tables(":memory:")
        conn.execute(
            "INSERT INTO ohlcv VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (1000, "BTC/USDT", "5m", 100.0, 101.0, 99.0, 100.5, 10.0, 1005.0, 50),
        )
        # Duplicate should raise
        try:
            conn.execute(
                "INSERT INTO ohlcv VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (1000, "BTC/USDT", "5m", 100.0, 101.0, 99.0, 100.5, 10.0, 1005.0, 50),
            )
            raise AssertionError("Should have raised IntegrityError")
        except sqlite3.IntegrityError:
            pass
        conn.close()

    def test_row_factory_set(self):
        conn = create_tables(":memory:")
        assert conn.row_factory == sqlite3.Row
        conn.close()

    def test_file_based_wal(self, tmp_path):
        db_path = tmp_path / "test.db"
        conn = create_tables(db_path)
        journal = conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert journal == "wal"
        conn.close()
