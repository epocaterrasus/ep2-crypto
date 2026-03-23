#!/usr/bin/env python3
"""Migrate ep2-crypto data from SQLite to TimescaleDB.

Creates TimescaleDB hypertables for all time-series tables, then
copies data from a local SQLite file into TimescaleDB.

Usage:
    # Dry run — show what would be migrated, touch nothing
    uv run python scripts/migrate_to_timescale.py --dry-run

    # Create hypertables only (no data migration)
    uv run python scripts/migrate_to_timescale.py --schema-only

    # Full migration (schema + data)
    uv run python scripts/migrate_to_timescale.py \\
        --sqlite-path ./ep2_crypto.db \\
        --pg-url "postgresql://ep2:secret@localhost:5432/ep2_crypto"

    # Migrate specific tables only
    uv run python scripts/migrate_to_timescale.py \\
        --tables ohlcv,prediction,regime_label

Environment variables (override --pg-url):
    EP2_DB_URL   PostgreSQL connection string

Required packages (already in project deps):
    psycopg2-binary, structlog
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from dataclasses import dataclass
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Table configuration
# ---------------------------------------------------------------------------

@dataclass
class TableConfig:
    """Configuration for migrating one table to TimescaleDB."""
    name: str
    time_column: str      # Column used as hypertable time dimension
    chunk_interval: str   # TimescaleDB chunk interval (e.g. '7 days')
    create_ddl: str       # PostgreSQL CREATE TABLE statement
    space_column: str | None = None  # Optional second partitioning dimension


_TABLE_CONFIGS: list[TableConfig] = [
    TableConfig(
        name="ohlcv",
        time_column="ts",
        chunk_interval="7 days",
        create_ddl="""
CREATE TABLE IF NOT EXISTS ohlcv (
    ts              TIMESTAMPTZ NOT NULL,
    symbol          TEXT        NOT NULL,
    interval        TEXT        NOT NULL,
    open            DOUBLE PRECISION NOT NULL,
    high            DOUBLE PRECISION NOT NULL,
    low             DOUBLE PRECISION NOT NULL,
    close           DOUBLE PRECISION NOT NULL,
    volume          DOUBLE PRECISION NOT NULL,
    quote_volume    DOUBLE PRECISION,
    trades_count    INTEGER,
    PRIMARY KEY (ts, symbol, interval)
)""",
    ),
    TableConfig(
        name="orderbook_snapshot",
        time_column="ts",
        chunk_interval="1 day",
        create_ddl="""
CREATE TABLE IF NOT EXISTS orderbook_snapshot (
    ts              TIMESTAMPTZ NOT NULL,
    symbol          TEXT        NOT NULL,
    bid_prices      TEXT        NOT NULL,
    bid_sizes       TEXT        NOT NULL,
    ask_prices      TEXT        NOT NULL,
    ask_sizes       TEXT        NOT NULL,
    mid_price       DOUBLE PRECISION NOT NULL,
    spread          DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (ts, symbol)
)""",
    ),
    TableConfig(
        name="agg_trades",
        time_column="ts",
        chunk_interval="1 day",
        create_ddl="""
CREATE TABLE IF NOT EXISTS agg_trades (
    ts              TIMESTAMPTZ NOT NULL,
    symbol          TEXT        NOT NULL,
    price           DOUBLE PRECISION NOT NULL,
    quantity        DOUBLE PRECISION NOT NULL,
    is_buyer_maker  BOOLEAN     NOT NULL,
    trade_id        TEXT,
    PRIMARY KEY (ts, symbol, trade_id)
)""",
    ),
    TableConfig(
        name="funding_rate",
        time_column="ts",
        chunk_interval="7 days",
        create_ddl="""
CREATE TABLE IF NOT EXISTS funding_rate (
    ts              TIMESTAMPTZ NOT NULL,
    symbol          TEXT        NOT NULL,
    funding_rate    DOUBLE PRECISION NOT NULL,
    mark_price      DOUBLE PRECISION,
    index_price     DOUBLE PRECISION,
    PRIMARY KEY (ts, symbol)
)""",
    ),
    TableConfig(
        name="open_interest",
        time_column="ts",
        chunk_interval="7 days",
        create_ddl="""
CREATE TABLE IF NOT EXISTS open_interest (
    ts              TIMESTAMPTZ NOT NULL,
    symbol          TEXT        NOT NULL,
    open_interest   DOUBLE PRECISION NOT NULL,
    oi_value_usd    DOUBLE PRECISION,
    PRIMARY KEY (ts, symbol)
)""",
    ),
    TableConfig(
        name="liquidation",
        time_column="ts",
        chunk_interval="1 day",
        create_ddl="""
CREATE TABLE IF NOT EXISTS liquidation (
    ts              TIMESTAMPTZ NOT NULL,
    symbol          TEXT        NOT NULL,
    side            TEXT        NOT NULL,
    price           DOUBLE PRECISION NOT NULL,
    quantity        DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (ts, symbol, side, price)
)""",
    ),
    TableConfig(
        name="cross_market",
        time_column="ts",
        chunk_interval="7 days",
        create_ddl="""
CREATE TABLE IF NOT EXISTS cross_market (
    ts              TIMESTAMPTZ NOT NULL,
    symbol          TEXT        NOT NULL,
    price           DOUBLE PRECISION NOT NULL,
    source          TEXT        NOT NULL,
    PRIMARY KEY (ts, symbol, source)
)""",
    ),
    TableConfig(
        name="onchain_whale",
        time_column="ts",
        chunk_interval="7 days",
        create_ddl="""
CREATE TABLE IF NOT EXISTS onchain_whale (
    ts              TIMESTAMPTZ NOT NULL,
    tx_hash         TEXT        NOT NULL,
    value_btc       DOUBLE PRECISION NOT NULL,
    fee_rate        DOUBLE PRECISION,
    is_exchange_flow BOOLEAN,
    PRIMARY KEY (ts, tx_hash)
)""",
    ),
    TableConfig(
        name="regime_label",
        time_column="ts",
        chunk_interval="7 days",
        create_ddl="""
CREATE TABLE IF NOT EXISTS regime_label (
    ts               TIMESTAMPTZ NOT NULL,
    symbol           TEXT        NOT NULL,
    regime           TEXT        NOT NULL,
    hmm_state        INTEGER,
    hmm_prob         DOUBLE PRECISION,
    bocpd_run_length DOUBLE PRECISION,
    garch_vol        DOUBLE PRECISION,
    efficiency_ratio DOUBLE PRECISION,
    PRIMARY KEY (ts, symbol)
)""",
    ),
    TableConfig(
        name="prediction",
        time_column="ts",
        chunk_interval="7 days",
        create_ddl="""
CREATE TABLE IF NOT EXISTS prediction (
    ts                   TIMESTAMPTZ NOT NULL,
    symbol               TEXT        NOT NULL,
    direction            TEXT        NOT NULL,
    confidence           DOUBLE PRECISION NOT NULL,
    calibrated_prob_up   DOUBLE PRECISION,
    calibrated_prob_down DOUBLE PRECISION,
    position_size        DOUBLE PRECISION,
    regime               TEXT,
    model_version        TEXT,
    PRIMARY KEY (ts, symbol)
)""",
    ),
    TableConfig(
        name="feature_snapshot",
        time_column="ts",
        chunk_interval="1 day",
        create_ddl="""
CREATE TABLE IF NOT EXISTS feature_snapshot (
    ts              TIMESTAMPTZ NOT NULL,
    symbol          TEXT        NOT NULL,
    features_json   TEXT        NOT NULL,
    PRIMARY KEY (ts, symbol)
)""",
    ),
]

# Map SQLite table name → TimescaleDB TableConfig
_CONFIG_BY_NAME: dict[str, TableConfig] = {c.name: c for c in _TABLE_CONFIGS}

# SQLite column name → TimescaleDB column name (renames)
_SQLITE_RENAMES: dict[str, dict[str, str]] = {
    "ohlcv": {"timestamp_ms": "ts"},
    "orderbook_snapshot": {"timestamp_ms": "ts"},
    "agg_trades": {"timestamp_ms": "ts", "is_buyer_maker": "is_buyer_maker"},
    "funding_rate": {"timestamp_ms": "ts"},
    "open_interest": {"timestamp_ms": "ts"},
    "liquidation": {"timestamp_ms": "ts"},
    "cross_market": {"timestamp_ms": "ts"},
    "onchain_whale": {"timestamp_ms": "ts"},
    "regime_label": {"timestamp_ms": "ts"},
    "prediction": {"timestamp_ms": "ts"},
    "feature_snapshot": {"timestamp_ms": "ts"},
}


# ---------------------------------------------------------------------------
# Core migration logic
# ---------------------------------------------------------------------------

def _ms_to_timestamptz(ms: int) -> str:
    """Convert Unix milliseconds to ISO-8601 string for PostgreSQL TIMESTAMPTZ."""
    from datetime import datetime, timezone
    dt = datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)
    return dt.isoformat()


def create_hypertables(pg_conn: Any, tables: list[str], dry_run: bool = False) -> None:
    """Create TimescaleDB tables and convert them to hypertables.

    Args:
        pg_conn: psycopg2 connection to TimescaleDB (None in dry-run mode).
        tables: List of table names to create.
        dry_run: If True, print SQL without executing.
    """
    cur = pg_conn.cursor() if pg_conn is not None else None

    for cfg in _TABLE_CONFIGS:
        if cfg.name not in tables:
            continue

        # Create table
        ddl = cfg.create_ddl.strip()
        if dry_run:
            logger.info("dry_run_create_table", table=cfg.name, ddl=ddl[:80])
            logger.info("dry_run_hypertable", table=cfg.name,
                        chunk_interval=cfg.chunk_interval, time_col=cfg.time_column)
            continue

        cur.execute(ddl)
        logger.info("table_created", table=cfg.name)

        # Convert to hypertable
        cur.execute(
            f"SELECT create_hypertable('{cfg.name}', '{cfg.time_column}', "
            f"chunk_time_interval => INTERVAL '{cfg.chunk_interval}', "
            f"if_not_exists => TRUE)"
        )
        logger.info("hypertable_created", table=cfg.name, chunk_interval=cfg.chunk_interval)

        # Enable compression
        try:
            cur.execute(
                f"ALTER TABLE {cfg.name} SET ("
                f"timescaledb.compress, "
                f"timescaledb.compress_orderby = '{cfg.time_column} DESC'"
                f")"
            )
            logger.info("compression_enabled", table=cfg.name)
        except Exception as exc:
            logger.warning("compression_skipped", table=cfg.name, reason=str(exc))

    if not dry_run:
        pg_conn.commit()


def migrate_table(
    sqlite_conn: sqlite3.Connection,
    pg_conn: Any,
    table_name: str,
    *,
    batch_size: int = 5000,
    dry_run: bool = False,
) -> int:
    """Migrate one table from SQLite to TimescaleDB.

    Args:
        sqlite_conn: Open SQLite connection.
        pg_conn: Open psycopg2 connection to TimescaleDB.
        table_name: Table to migrate.
        batch_size: Rows per INSERT batch.
        dry_run: If True, count rows but don't insert.

    Returns:
        Number of rows migrated.
    """
    cfg = _CONFIG_BY_NAME.get(table_name)
    if cfg is None:
        logger.warning("unknown_table_skipped", table=table_name)
        return 0

    renames = _SQLITE_RENAMES.get(table_name, {})

    if dry_run or sqlite_conn is None:
        logger.info("dry_run_would_migrate", table=table_name)
        return 0

    # Get SQLite columns
    cur_sqlite = sqlite_conn.cursor()
    cur_sqlite.execute(f"SELECT * FROM {table_name} LIMIT 0")  # noqa: S608
    sqlite_cols = [d[0] for d in cur_sqlite.description]

    # Map to TimescaleDB column names
    pg_cols = [renames.get(c, c) for c in sqlite_cols]

    # Count total rows
    total = cur_sqlite.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]  # noqa: S608
    if total == 0:
        logger.info("table_empty_skipped", table=table_name)
        return 0

    logger.info("migrating_table", table=table_name, rows=total)

    # Build INSERT statement
    placeholders = ", ".join(["%s"] * len(pg_cols))
    col_list = ", ".join(pg_cols)
    insert_sql = f"INSERT INTO {table_name} ({col_list}) VALUES ({placeholders}) ON CONFLICT DO NOTHING"

    cur_pg = pg_conn.cursor()
    migrated = 0
    offset = 0

    while offset < total:
        rows = cur_sqlite.execute(
            f"SELECT * FROM {table_name} LIMIT ? OFFSET ?",  # noqa: S608
            (batch_size, offset),
        ).fetchall()

        if not rows:
            break

        # Transform rows: convert timestamp_ms → TIMESTAMPTZ ISO string
        transformed = []
        for row in rows:
            row_dict = dict(zip(sqlite_cols, row))
            if "timestamp_ms" in row_dict:
                row_dict["ts"] = _ms_to_timestamptz(int(row_dict.pop("timestamp_ms")))
            if "is_buyer_maker" in row_dict:
                row_dict["is_buyer_maker"] = bool(row_dict["is_buyer_maker"])
            if "is_exchange_flow" in row_dict and row_dict["is_exchange_flow"] is not None:
                row_dict["is_exchange_flow"] = bool(row_dict["is_exchange_flow"])
            # Reorder to match pg_cols order
            transformed.append(tuple(row_dict.get(c.replace("ts", "ts"), row_dict.get(c)) for c in pg_cols))

        cur_pg.executemany(insert_sql, transformed)
        pg_conn.commit()

        migrated += len(rows)
        offset += batch_size
        logger.info("batch_migrated", table=table_name, migrated=migrated, total=total)

    logger.info("table_migration_complete", table=table_name, migrated=migrated)
    return migrated


def run_migration(
    sqlite_path: str,
    pg_url: str,
    tables: list[str],
    *,
    schema_only: bool = False,
    dry_run: bool = False,
    batch_size: int = 5000,
) -> dict[str, int]:
    """Run full migration from SQLite to TimescaleDB.

    Args:
        sqlite_path: Path to SQLite database file.
        pg_url: PostgreSQL connection URL.
        tables: List of table names to migrate.
        schema_only: If True, only create schema (no data copy).
        dry_run: If True, show what would happen without executing.
        batch_size: Rows per INSERT batch.

    Returns:
        Dict mapping table_name → rows_migrated.
    """
    results: dict[str, int] = {}

    # Connect to SQLite
    if not dry_run and not schema_only:
        import os
        if not os.path.exists(sqlite_path):
            msg = f"SQLite file not found: {sqlite_path}"
            raise FileNotFoundError(msg)
        sqlite_conn = sqlite3.connect(sqlite_path)
        sqlite_conn.row_factory = sqlite3.Row
        logger.info("sqlite_connected", path=sqlite_path)
    else:
        sqlite_conn = None

    # Connect to TimescaleDB (requires psycopg2 unless dry-run)
    if not dry_run:
        try:
            import psycopg2
        except ImportError as exc:
            msg = "psycopg2 is required: uv add psycopg2-binary"
            raise RuntimeError(msg) from exc
        pg_conn = psycopg2.connect(pg_url)
        logger.info("timescaledb_connected", url=pg_url.split("@")[-1])  # hide credentials
    else:
        pg_conn = None

    try:
        # Step 1: Create hypertables
        create_hypertables(pg_conn, tables, dry_run=dry_run)

        if schema_only:
            logger.info("schema_only_complete", tables=tables)
            return {t: 0 for t in tables}

        # Step 2: Migrate data
        for table in tables:
            count = migrate_table(
                sqlite_conn,
                pg_conn,
                table,
                batch_size=batch_size,
                dry_run=dry_run,
            )
            results[table] = count

    finally:
        if sqlite_conn:
            sqlite_conn.close()
        if pg_conn:
            pg_conn.close()

    total_rows = sum(results.values())
    logger.info("migration_complete", total_rows=total_rows, tables=len(results))
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import os

    parser = argparse.ArgumentParser(
        description="Migrate ep2-crypto data from SQLite to TimescaleDB",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--sqlite-path",
        default="./ep2_crypto.db",
        help="Path to SQLite database (default: ./ep2_crypto.db)",
    )
    parser.add_argument(
        "--pg-url",
        default=os.environ.get("EP2_DB_URL", "postgresql://ep2:ep2_secret@localhost:5432/ep2_crypto"),
        help="PostgreSQL connection URL (default: $EP2_DB_URL or localhost)",
    )
    parser.add_argument(
        "--tables",
        default=",".join(_CONFIG_BY_NAME.keys()),
        help="Comma-separated table names to migrate (default: all)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5000,
        help="Rows per INSERT batch (default: 5000)",
    )
    parser.add_argument(
        "--schema-only",
        action="store_true",
        help="Only create hypertables, do not copy data",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without executing",
    )

    args = parser.parse_args()
    tables = [t.strip() for t in args.tables.split(",") if t.strip()]

    unknown = [t for t in tables if t not in _CONFIG_BY_NAME]
    if unknown:
        logger.error("unknown_tables", tables=unknown, valid=list(_CONFIG_BY_NAME.keys()))
        sys.exit(1)

    if args.dry_run:
        logger.info("dry_run_mode", tables=tables)

    results = run_migration(
        sqlite_path=args.sqlite_path,
        pg_url=args.pg_url,
        tables=tables,
        schema_only=args.schema_only,
        dry_run=args.dry_run,
        batch_size=args.batch_size,
    )

    if not args.dry_run and not args.schema_only:
        print("\nMigration summary:")
        print(f"{'Table':<25} {'Rows':>10}")
        print("-" * 36)
        for table, count in results.items():
            print(f"{table:<25} {count:>10,}")
        print("-" * 36)
        print(f"{'TOTAL':<25} {sum(results.values()):>10,}")


if __name__ == "__main__":
    main()
