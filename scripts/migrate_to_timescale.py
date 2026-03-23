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
    is_hypertable: bool = True  # False for reference/lookup tables


_TABLE_CONFIGS: list[TableConfig] = [
    TableConfig(
        name="ohlcv",
        time_column="ts",
        chunk_interval="7 days",
        create_ddl="""
CREATE TABLE IF NOT EXISTS ohlcv (
    ts              TIMESTAMPTZ      NOT NULL,
    symbol          TEXT             NOT NULL,
    interval        TEXT             NOT NULL,
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
    ts              TIMESTAMPTZ      NOT NULL,
    symbol          TEXT             NOT NULL,
    bid_prices      TEXT             NOT NULL,
    bid_sizes       TEXT             NOT NULL,
    ask_prices      TEXT             NOT NULL,
    ask_sizes       TEXT             NOT NULL,
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
    ts              TIMESTAMPTZ      NOT NULL,
    symbol          TEXT             NOT NULL,
    price           DOUBLE PRECISION NOT NULL,
    quantity        DOUBLE PRECISION NOT NULL,
    is_buyer_maker  BOOLEAN          NOT NULL,
    trade_id        TEXT             NOT NULL DEFAULT '',
    PRIMARY KEY (ts, symbol, trade_id)
)""",
    ),
    TableConfig(
        name="funding_rate",
        time_column="ts",
        chunk_interval="7 days",
        create_ddl="""
CREATE TABLE IF NOT EXISTS funding_rate (
    ts              TIMESTAMPTZ      NOT NULL,
    symbol          TEXT             NOT NULL,
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
    ts              TIMESTAMPTZ      NOT NULL,
    symbol          TEXT             NOT NULL,
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
    ts              TIMESTAMPTZ      NOT NULL,
    symbol          TEXT             NOT NULL,
    side            TEXT             NOT NULL,
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
    ts              TIMESTAMPTZ      NOT NULL,
    symbol          TEXT             NOT NULL,
    price           DOUBLE PRECISION NOT NULL,
    source          TEXT             NOT NULL,
    PRIMARY KEY (ts, symbol, source)
)""",
    ),
    TableConfig(
        name="onchain_whale",
        time_column="ts",
        chunk_interval="7 days",
        create_ddl="""
CREATE TABLE IF NOT EXISTS onchain_whale (
    ts               TIMESTAMPTZ      NOT NULL,
    tx_hash          TEXT             NOT NULL,
    value_btc        DOUBLE PRECISION NOT NULL,
    fee_rate         DOUBLE PRECISION,
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
    ts               TIMESTAMPTZ      NOT NULL,
    symbol           TEXT             NOT NULL,
    regime           TEXT             NOT NULL,
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
    ts                   TIMESTAMPTZ      NOT NULL,
    symbol               TEXT             NOT NULL,
    direction            TEXT             NOT NULL,
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
    ts            TIMESTAMPTZ NOT NULL,
    symbol        TEXT        NOT NULL,
    features_json JSONB       NOT NULL,
    PRIMARY KEY (ts, symbol)
)""",
    ),
    # ---------------------------------------------------------------------------
    # Monitoring tables — from monitoring/performance_logger.py
    # ---------------------------------------------------------------------------
    TableConfig(
        name="trade_log",
        time_column="ts",
        chunk_interval="7 days",
        create_ddl="""
CREATE TABLE IF NOT EXISTS trade_log (
    id                   BIGSERIAL        PRIMARY KEY,
    ts                   TIMESTAMPTZ      NOT NULL,
    direction            TEXT             NOT NULL,
    predicted_confidence DOUBLE PRECISION NOT NULL,
    predicted_magnitude  DOUBLE PRECISION NOT NULL,
    actual_direction     TEXT,
    actual_return        DOUBLE PRECISION,
    pnl                  DOUBLE PRECISION,
    slippage_expected    DOUBLE PRECISION,
    slippage_actual      DOUBLE PRECISION,
    latency_ms           DOUBLE PRECISION,
    regime               TEXT,
    features_json        JSONB,
    entry_price          DOUBLE PRECISION,
    exit_price           DOUBLE PRECISION,
    position_size        DOUBLE PRECISION,
    meta_json            JSONB,
    created_at           TIMESTAMPTZ NOT NULL DEFAULT NOW()
)""",
    ),
    TableConfig(
        name="bar_state_log",
        time_column="ts",
        chunk_interval="7 days",
        create_ddl="""
CREATE TABLE IF NOT EXISTS bar_state_log (
    id                   BIGSERIAL        PRIMARY KEY,
    ts                   TIMESTAMPTZ      NOT NULL,
    bar_close            DOUBLE PRECISION NOT NULL,
    regime               TEXT,
    regime_confidence    DOUBLE PRECISION,
    model_prediction     TEXT,
    model_confidence     DOUBLE PRECISION,
    risk_state_json      JSONB,
    feature_values_json  JSONB,
    kill_switch_active   BOOLEAN          NOT NULL DEFAULT FALSE,
    drawdown_multiplier  DOUBLE PRECISION,
    volatility_ann       DOUBLE PRECISION,
    position_open        BOOLEAN          NOT NULL DEFAULT FALSE,
    equity               DOUBLE PRECISION,
    created_at           TIMESTAMPTZ      NOT NULL DEFAULT NOW()
)""",
    ),
    # ---------------------------------------------------------------------------
    # Polymarket tables — prediction market data storage
    # ---------------------------------------------------------------------------
    TableConfig(
        name="polymarket_markets",
        time_column="ts",
        chunk_interval="30 days",
        create_ddl="""
CREATE TABLE IF NOT EXISTS polymarket_markets (
    ts               TIMESTAMPTZ      NOT NULL,
    condition_id     TEXT             NOT NULL,
    question         TEXT             NOT NULL,
    end_date_iso     TEXT,
    active           BOOLEAN          NOT NULL DEFAULT TRUE,
    resolved         BOOLEAN          NOT NULL DEFAULT FALSE,
    resolution       TEXT,
    yes_price        DOUBLE PRECISION,
    no_price         DOUBLE PRECISION,
    volume_usd       DOUBLE PRECISION,
    liquidity_usd    DOUBLE PRECISION,
    raw_json         JSONB,
    PRIMARY KEY (ts, condition_id)
)""",
    ),
    TableConfig(
        name="polymarket_positions",
        time_column="ts",
        chunk_interval="30 days",
        create_ddl="""
CREATE TABLE IF NOT EXISTS polymarket_positions (
    ts               TIMESTAMPTZ      NOT NULL,
    condition_id     TEXT             NOT NULL,
    side             TEXT             NOT NULL,
    size             DOUBLE PRECISION NOT NULL,
    avg_price        DOUBLE PRECISION NOT NULL,
    current_price    DOUBLE PRECISION,
    unrealized_pnl   DOUBLE PRECISION,
    realized_pnl     DOUBLE PRECISION,
    status           TEXT             NOT NULL DEFAULT 'open',
    PRIMARY KEY (ts, condition_id, side)
)""",
    ),
]

# Map table name → TableConfig
_CONFIG_BY_NAME: dict[str, TableConfig] = {c.name: c for c in _TABLE_CONFIGS}

# SQLite timestamp column → TimescaleDB column renames (per table)
_SQLITE_RENAMES: dict[str, dict[str, str]] = {
    "ohlcv": {"timestamp_ms": "ts"},
    "orderbook_snapshot": {"timestamp_ms": "ts"},
    "agg_trades": {"timestamp_ms": "ts"},
    "funding_rate": {"timestamp_ms": "ts"},
    "open_interest": {"timestamp_ms": "ts"},
    "liquidation": {"timestamp_ms": "ts"},
    "cross_market": {"timestamp_ms": "ts"},
    "onchain_whale": {"timestamp_ms": "ts"},
    "regime_label": {"timestamp_ms": "ts"},
    "prediction": {"timestamp_ms": "ts"},
    "feature_snapshot": {"timestamp_ms": "ts"},
    "trade_log": {"timestamp_ms": "ts"},
    "bar_state_log": {"timestamp_ms": "ts"},
}

# Tables that have no SQLite source (Polymarket tables are PG-only)
_PG_ONLY_TABLES: frozenset[str] = frozenset({"polymarket_markets", "polymarket_positions"})

# ---------------------------------------------------------------------------
# Additional indexes (beyond the hypertable default index on time column)
# ---------------------------------------------------------------------------

_EXTRA_INDEXES: list[tuple[str, str]] = [
    # (index_name, CREATE INDEX statement)
    (
        "idx_ohlcv_symbol_ts",
        "CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_ts ON ohlcv (symbol, ts DESC)",
    ),
    (
        "idx_orderbook_snapshot_symbol_ts",
        "CREATE INDEX IF NOT EXISTS idx_orderbook_snapshot_symbol_ts ON orderbook_snapshot (symbol, ts DESC)",
    ),
    (
        "idx_agg_trades_symbol_ts",
        "CREATE INDEX IF NOT EXISTS idx_agg_trades_symbol_ts ON agg_trades (symbol, ts DESC)",
    ),
    (
        "idx_funding_rate_symbol_ts",
        "CREATE INDEX IF NOT EXISTS idx_funding_rate_symbol_ts ON funding_rate (symbol, ts DESC)",
    ),
    (
        "idx_open_interest_symbol_ts",
        "CREATE INDEX IF NOT EXISTS idx_open_interest_symbol_ts ON open_interest (symbol, ts DESC)",
    ),
    (
        "idx_liquidation_symbol_ts",
        "CREATE INDEX IF NOT EXISTS idx_liquidation_symbol_ts ON liquidation (symbol, ts DESC)",
    ),
    (
        "idx_cross_market_symbol_ts",
        "CREATE INDEX IF NOT EXISTS idx_cross_market_symbol_ts ON cross_market (symbol, ts DESC)",
    ),
    (
        "idx_regime_label_symbol_ts",
        "CREATE INDEX IF NOT EXISTS idx_regime_label_symbol_ts ON regime_label (symbol, ts DESC)",
    ),
    (
        "idx_prediction_symbol_ts",
        "CREATE INDEX IF NOT EXISTS idx_prediction_symbol_ts ON prediction (symbol, ts DESC)",
    ),
    (
        "idx_feature_snapshot_symbol_ts",
        "CREATE INDEX IF NOT EXISTS idx_feature_snapshot_symbol_ts ON feature_snapshot (symbol, ts DESC)",
    ),
    (
        "idx_trade_log_ts",
        "CREATE INDEX IF NOT EXISTS idx_trade_log_ts ON trade_log (ts DESC)",
    ),
    (
        "idx_bar_state_log_ts",
        "CREATE INDEX IF NOT EXISTS idx_bar_state_log_ts ON bar_state_log (ts DESC)",
    ),
    (
        "idx_polymarket_markets_condition",
        "CREATE INDEX IF NOT EXISTS idx_polymarket_markets_condition ON polymarket_markets (condition_id, ts DESC)",
    ),
    (
        "idx_polymarket_positions_condition",
        "CREATE INDEX IF NOT EXISTS idx_polymarket_positions_condition ON polymarket_positions (condition_id, ts DESC)",
    ),
    (
        # GIN index on JSONB feature snapshots for attribute queries
        "idx_feature_snapshot_gin",
        "CREATE INDEX IF NOT EXISTS idx_feature_snapshot_gin ON feature_snapshot USING GIN (features_json)",
    ),
]


# ---------------------------------------------------------------------------
# Core migration logic
# ---------------------------------------------------------------------------

def _ms_to_timestamptz(ms: int) -> str:
    """Convert Unix milliseconds to ISO-8601 string for PostgreSQL TIMESTAMPTZ."""
    from datetime import datetime, timezone
    dt = datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)
    return dt.isoformat()


def _validate_identifier(name: str, allowlist: set[str]) -> str:
    """Validate a SQL identifier against an allowlist.

    Table names and column names cannot be parameterized in PostgreSQL.
    We mitigate injection risk by validating against a hardcoded allowlist
    derived from _TABLE_CONFIGS, which is fully controlled in source code.

    Args:
        name: Identifier to validate.
        allowlist: Set of permitted names.

    Returns:
        The validated name, unchanged.

    Raises:
        ValueError: If name is not in the allowlist.
    """
    if name not in allowlist:
        msg = f"Identifier '{name}' is not in the allowlist. Refusing to execute."
        raise ValueError(msg)
    return name


_VALID_TABLES: set[str] = set(_CONFIG_BY_NAME.keys())
_VALID_TIME_COLS: set[str] = {"ts"}  # Only 'ts' is used across all configs


def create_schema(pg_conn: Any, tables: list[str], *, dry_run: bool = False) -> None:
    """Create TimescaleDB extension, tables, hypertables, and indexes.

    This function is idempotent — it uses IF NOT EXISTS throughout and
    passes if_not_exists => TRUE to create_hypertable.

    Args:
        pg_conn: psycopg2 connection to TimescaleDB (None in dry-run mode).
        tables: List of table names to create.
        dry_run: If True, log SQL without executing.
    """
    cur = pg_conn.cursor() if pg_conn is not None else None

    # Step 1: Ensure TimescaleDB extension is installed
    ext_sql = "CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE"
    if dry_run:
        logger.info("dry_run_sql", sql=ext_sql)
    else:
        assert cur is not None
        cur.execute(ext_sql)
        logger.info("timescaledb_extension_ready")

    # Step 2: Create tables
    for cfg in _TABLE_CONFIGS:
        if cfg.name not in tables:
            continue

        ddl = cfg.create_ddl.strip()
        if dry_run:
            logger.info("dry_run_create_table", table=cfg.name, ddl=ddl[:120])
        else:
            assert cur is not None
            cur.execute(ddl)
            logger.info("table_created_or_exists", table=cfg.name)

    # Step 3: Convert time-series tables to hypertables
    for cfg in _TABLE_CONFIGS:
        if cfg.name not in tables or not cfg.is_hypertable:
            continue

        # Validate identifiers against allowlist before interpolating into SQL.
        # These values come exclusively from _TABLE_CONFIGS in source code,
        # never from user input or external data.
        table_name = _validate_identifier(cfg.name, _VALID_TABLES)
        time_col = _validate_identifier(cfg.time_column, _VALID_TIME_COLS)

        hypertable_sql = (
            f"SELECT create_hypertable("  # noqa: S608
            f"'{table_name}', '{time_col}', "
            f"chunk_time_interval => INTERVAL '{cfg.chunk_interval}', "
            f"if_not_exists => TRUE"
            f")"
        )
        if dry_run:
            logger.info(
                "dry_run_hypertable",
                table=cfg.name,
                time_col=cfg.time_column,
                chunk_interval=cfg.chunk_interval,
            )
        else:
            assert cur is not None
            cur.execute(hypertable_sql)
            logger.info(
                "hypertable_created_or_exists",
                table=cfg.name,
                chunk_interval=cfg.chunk_interval,
            )

    # Step 4: Enable compression on hypertables (best-effort; skipped if already set)
    for cfg in _TABLE_CONFIGS:
        if cfg.name not in tables or not cfg.is_hypertable:
            continue

        table_name = _validate_identifier(cfg.name, _VALID_TABLES)
        time_col = _validate_identifier(cfg.time_column, _VALID_TIME_COLS)

        compress_sql = (
            f"ALTER TABLE {table_name} SET ("  # noqa: S608
            f"timescaledb.compress, "
            f"timescaledb.compress_orderby = '{time_col} DESC'"
            f")"
        )
        if dry_run:
            logger.info("dry_run_compression", table=cfg.name)
        else:
            assert cur is not None
            try:
                cur.execute(compress_sql)
                logger.info("compression_enabled", table=cfg.name)
            except Exception as exc:
                # Compression fails harmlessly if already enabled or on CE
                pg_conn.rollback()
                logger.warning("compression_skipped", table=cfg.name, reason=str(exc))
                # Restart the transaction after rollback
                cur = pg_conn.cursor()

    # Step 5: Create secondary indexes
    for index_name, index_sql in _EXTRA_INDEXES:
        # Only create indexes for tables we are actually setting up
        table_from_index = index_name.split("idx_", 1)[-1].rsplit("_", 1)[0]
        # Map index → table: idx_ohlcv_symbol_ts → "ohlcv_symbol" → skip if table not in tables
        # We use a simpler approach: always create all indexes, skip if table not included
        # by checking if any table from `tables` is a prefix of the index name
        target_tables = [t for t in tables if index_name.startswith(f"idx_{t}")]
        if not target_tables and table_from_index not in tables:
            continue

        if dry_run:
            logger.info("dry_run_index", index=index_name)
        else:
            assert cur is not None
            try:
                cur.execute(index_sql)
                logger.info("index_created_or_exists", index=index_name)
            except Exception as exc:
                logger.warning("index_skipped", index=index_name, reason=str(exc))

    if not dry_run and pg_conn is not None:
        pg_conn.commit()
        logger.info("schema_committed")


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

    if table_name in _PG_ONLY_TABLES:
        logger.info("pg_only_table_skipped_during_migration", table=table_name)
        return 0

    renames = _SQLITE_RENAMES.get(table_name, {})

    if dry_run or sqlite_conn is None:
        logger.info("dry_run_would_migrate", table=table_name)
        return 0

    # Validate table name against allowlist before using in query
    validated_table = _validate_identifier(table_name, _VALID_TABLES)

    # Get SQLite columns
    cur_sqlite = sqlite_conn.cursor()
    cur_sqlite.execute(f"SELECT * FROM {validated_table} LIMIT 0")  # noqa: S608
    sqlite_cols = [d[0] for d in cur_sqlite.description]

    # Map SQLite column names to TimescaleDB column names
    pg_cols = [renames.get(c, c) for c in sqlite_cols]

    # Count total rows
    total_row = cur_sqlite.execute(
        f"SELECT COUNT(*) FROM {validated_table}"  # noqa: S608
    ).fetchone()
    total = total_row[0] if total_row else 0
    if total == 0:
        logger.info("table_empty_skipped", table=table_name)
        return 0

    logger.info("migrating_table", table=table_name, rows=total)

    # Build INSERT with ON CONFLICT DO NOTHING for idempotency
    placeholders = ", ".join(["%s"] * len(pg_cols))
    col_list = ", ".join(pg_cols)
    insert_sql = (
        f"INSERT INTO {validated_table} ({col_list}) VALUES ({placeholders})"  # noqa: S608
        f" ON CONFLICT DO NOTHING"
    )

    cur_pg = pg_conn.cursor()
    migrated = 0
    offset = 0

    while offset < total:
        rows = cur_sqlite.execute(
            f"SELECT * FROM {validated_table} LIMIT ? OFFSET ?",  # noqa: S608
            (batch_size, offset),
        ).fetchall()

        if not rows:
            break

        transformed: list[tuple[Any, ...]] = []
        for row in rows:
            row_dict: dict[str, Any] = dict(zip(sqlite_cols, row))

            # Convert Unix-ms integer → ISO-8601 TIMESTAMPTZ string
            if "timestamp_ms" in row_dict:
                row_dict["ts"] = _ms_to_timestamptz(int(row_dict.pop("timestamp_ms")))

            # SQLite stores booleans as integers
            if "is_buyer_maker" in row_dict and row_dict["is_buyer_maker"] is not None:
                row_dict["is_buyer_maker"] = bool(row_dict["is_buyer_maker"])
            if "is_exchange_flow" in row_dict and row_dict["is_exchange_flow"] is not None:
                row_dict["is_exchange_flow"] = bool(row_dict["is_exchange_flow"])
            if "kill_switch_active" in row_dict and row_dict["kill_switch_active"] is not None:
                row_dict["kill_switch_active"] = bool(row_dict["kill_switch_active"])
            if "position_open" in row_dict and row_dict["position_open"] is not None:
                row_dict["position_open"] = bool(row_dict["position_open"])

            # trade_id NULL → empty string (NOT NULL DEFAULT '' in PG schema)
            if "trade_id" in row_dict and row_dict["trade_id"] is None:
                row_dict["trade_id"] = ""

            # Reorder values to match pg_cols order
            transformed.append(tuple(row_dict.get(c) for c in pg_cols))

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

    This function is idempotent — it can be run multiple times safely.
    Tables already migrated will have rows skipped via ON CONFLICT DO NOTHING.

    Args:
        sqlite_path: Path to SQLite database file.
        pg_url: PostgreSQL connection URL (read from EP2_DB_URL env var if not given).
        tables: List of table names to migrate.
        schema_only: If True, only create schema (no data copy).
        dry_run: If True, show what would happen without executing.
        batch_size: Rows per INSERT batch.

    Returns:
        Dict mapping table_name → rows_migrated.
    """
    results: dict[str, int] = {}

    # Connect to SQLite (not needed for schema-only or dry-run)
    sqlite_conn: sqlite3.Connection | None = None
    if not dry_run and not schema_only:
        import os
        if not os.path.exists(sqlite_path):
            msg = f"SQLite file not found: {sqlite_path}"
            raise FileNotFoundError(msg)
        sqlite_conn = sqlite3.connect(sqlite_path)
        logger.info("sqlite_connected", path=sqlite_path)

    # Connect to TimescaleDB
    pg_conn: Any = None
    if not dry_run:
        try:
            import psycopg2  # type: ignore[import-untyped]
        except ImportError as exc:
            msg = "psycopg2 is required: uv add psycopg2-binary"
            raise RuntimeError(msg) from exc

        # Log the host portion only — never log credentials
        host_hint = pg_url.split("@")[-1] if "@" in pg_url else pg_url
        pg_conn = psycopg2.connect(pg_url)
        logger.info("timescaledb_connected", host=host_hint)

    try:
        # Step 1: Create schema (extension + tables + hypertables + indexes)
        create_schema(pg_conn, tables, dry_run=dry_run)

        if schema_only:
            logger.info("schema_only_complete", tables=tables)
            return {t: 0 for t in tables}

        # Step 2: Migrate data from SQLite → TimescaleDB
        for table in tables:
            count = migrate_table(
                sqlite_conn,  # type: ignore[arg-type]
                pg_conn,
                table,
                batch_size=batch_size,
                dry_run=dry_run,
            )
            results[table] = count

    finally:
        if sqlite_conn is not None:
            sqlite_conn.close()
        if pg_conn is not None:
            pg_conn.close()

    total_rows = sum(results.values())
    logger.info("migration_complete", total_rows=total_rows, tables_processed=len(results))
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
        help="Only create schema (extension, tables, hypertables, indexes) — do not copy data",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without executing any SQL",
    )

    args = parser.parse_args()
    tables = [t.strip() for t in args.tables.split(",") if t.strip()]

    unknown = [t for t in tables if t not in _CONFIG_BY_NAME]
    if unknown:
        logger.error("unknown_tables", tables=unknown, valid=list(_CONFIG_BY_NAME.keys()))
        sys.exit(1)

    if args.dry_run:
        logger.info("dry_run_mode_active", tables=tables)

    results = run_migration(
        sqlite_path=args.sqlite_path,
        pg_url=args.pg_url,
        tables=tables,
        schema_only=args.schema_only,
        dry_run=args.dry_run,
        batch_size=args.batch_size,
    )

    if not args.dry_run and not args.schema_only:
        logger.info(
            "migration_summary",
            results={table: count for table, count in results.items()},
            total_rows=sum(results.values()),
        )


if __name__ == "__main__":
    main()
