"""Database schema for all ep2-crypto data types.

SQLite (dev): timestamps as INTEGER (unix ms), REAL for floats.
PostgreSQL/TimescaleDB (prod): timestamps as BIGINT, DOUBLE PRECISION for floats.

INTEGER in PostgreSQL is 32-bit (max ~2.1 billion); unix-ms values (~1.74e12) require BIGINT.
"""

from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from pathlib import Path

logger = structlog.get_logger(__name__)

# -- Table definitions --------------------------------------------------------

OHLCV_TABLE = """
CREATE TABLE IF NOT EXISTS ohlcv (
    timestamp_ms    INTEGER NOT NULL,
    symbol          TEXT    NOT NULL,
    interval        TEXT    NOT NULL,
    open            REAL    NOT NULL,
    high            REAL    NOT NULL,
    low             REAL    NOT NULL,
    close           REAL    NOT NULL,
    volume          REAL    NOT NULL,
    quote_volume    REAL,
    trades_count    INTEGER,
    PRIMARY KEY (timestamp_ms, symbol, interval)
)
"""

ORDERBOOK_TABLE = """
CREATE TABLE IF NOT EXISTS orderbook_snapshot (
    timestamp_ms    INTEGER NOT NULL,
    symbol          TEXT    NOT NULL,
    bid_prices      TEXT    NOT NULL,
    bid_sizes       TEXT    NOT NULL,
    ask_prices      TEXT    NOT NULL,
    ask_sizes       TEXT    NOT NULL,
    mid_price       REAL    NOT NULL,
    spread          REAL    NOT NULL,
    PRIMARY KEY (timestamp_ms, symbol)
)
"""

TRADES_TABLE = """
CREATE TABLE IF NOT EXISTS agg_trades (
    timestamp_ms    INTEGER NOT NULL,
    symbol          TEXT    NOT NULL,
    price           REAL    NOT NULL,
    quantity        REAL    NOT NULL,
    is_buyer_maker  INTEGER NOT NULL,
    trade_id        TEXT,
    PRIMARY KEY (timestamp_ms, symbol, trade_id)
)
"""

FUNDING_RATE_TABLE = """
CREATE TABLE IF NOT EXISTS funding_rate (
    timestamp_ms    INTEGER NOT NULL,
    symbol          TEXT    NOT NULL,
    funding_rate    REAL    NOT NULL,
    mark_price      REAL,
    index_price     REAL,
    PRIMARY KEY (timestamp_ms, symbol)
)
"""

OPEN_INTEREST_TABLE = """
CREATE TABLE IF NOT EXISTS open_interest (
    timestamp_ms    INTEGER NOT NULL,
    symbol          TEXT    NOT NULL,
    open_interest   REAL    NOT NULL,
    oi_value_usd    REAL,
    PRIMARY KEY (timestamp_ms, symbol)
)
"""

LIQUIDATION_TABLE = """
CREATE TABLE IF NOT EXISTS liquidation (
    timestamp_ms    INTEGER NOT NULL,
    symbol          TEXT    NOT NULL,
    side            TEXT    NOT NULL,
    price           REAL    NOT NULL,
    quantity        REAL    NOT NULL,
    PRIMARY KEY (timestamp_ms, symbol, side, price)
)
"""

CROSS_MARKET_TABLE = """
CREATE TABLE IF NOT EXISTS cross_market (
    timestamp_ms    INTEGER NOT NULL,
    symbol          TEXT    NOT NULL,
    price           REAL    NOT NULL,
    source          TEXT    NOT NULL,
    PRIMARY KEY (timestamp_ms, symbol, source)
)
"""

ONCHAIN_TABLE = """
CREATE TABLE IF NOT EXISTS onchain_whale (
    timestamp_ms    INTEGER NOT NULL,
    tx_hash         TEXT    NOT NULL,
    value_btc       REAL    NOT NULL,
    fee_rate        REAL,
    is_exchange_flow INTEGER,
    PRIMARY KEY (timestamp_ms, tx_hash)
)
"""

REGIME_TABLE = """
CREATE TABLE IF NOT EXISTS regime_label (
    timestamp_ms    INTEGER NOT NULL,
    symbol          TEXT    NOT NULL,
    regime          TEXT    NOT NULL,
    hmm_state       INTEGER,
    hmm_prob        REAL,
    bocpd_run_length REAL,
    garch_vol       REAL,
    efficiency_ratio REAL,
    PRIMARY KEY (timestamp_ms, symbol)
)
"""

PREDICTION_TABLE = """
CREATE TABLE IF NOT EXISTS prediction (
    timestamp_ms        INTEGER NOT NULL,
    symbol              TEXT    NOT NULL,
    direction           TEXT    NOT NULL,
    confidence          REAL    NOT NULL,
    calibrated_prob_up  REAL,
    calibrated_prob_down REAL,
    position_size       REAL,
    regime              TEXT,
    model_version       TEXT,
    PRIMARY KEY (timestamp_ms, symbol)
)
"""

FEATURE_SNAPSHOT_TABLE = """
CREATE TABLE IF NOT EXISTS feature_snapshot (
    timestamp_ms    INTEGER NOT NULL,
    symbol          TEXT    NOT NULL,
    features_json   TEXT    NOT NULL,
    PRIMARY KEY (timestamp_ms, symbol)
)
"""

ALL_TABLES = [
    OHLCV_TABLE,
    ORDERBOOK_TABLE,
    TRADES_TABLE,
    FUNDING_RATE_TABLE,
    OPEN_INTEREST_TABLE,
    LIQUIDATION_TABLE,
    CROSS_MARKET_TABLE,
    ONCHAIN_TABLE,
    REGIME_TABLE,
    PREDICTION_TABLE,
    FEATURE_SNAPSHOT_TABLE,
]

# -- Index definitions --------------------------------------------------------

INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_ohlcv_ts ON ohlcv (timestamp_ms)",
    "CREATE INDEX IF NOT EXISTS idx_orderbook_ts ON orderbook_snapshot (timestamp_ms)",
    "CREATE INDEX IF NOT EXISTS idx_trades_ts ON agg_trades (timestamp_ms)",
    "CREATE INDEX IF NOT EXISTS idx_funding_ts ON funding_rate (timestamp_ms)",
    "CREATE INDEX IF NOT EXISTS idx_oi_ts ON open_interest (timestamp_ms)",
    "CREATE INDEX IF NOT EXISTS idx_liquidation_ts ON liquidation (timestamp_ms)",
    "CREATE INDEX IF NOT EXISTS idx_cross_market_ts ON cross_market (timestamp_ms)",
    "CREATE INDEX IF NOT EXISTS idx_regime_ts ON regime_label (timestamp_ms)",
    "CREATE INDEX IF NOT EXISTS idx_prediction_ts ON prediction (timestamp_ms)",
]

# -- PRAGMA settings ----------------------------------------------------------

PRAGMA_SETTINGS = [
    "PRAGMA journal_mode = WAL",
    "PRAGMA synchronous = NORMAL",
    "PRAGMA cache_size = -64000",  # 64 MB
    "PRAGMA busy_timeout = 5000",
    "PRAGMA journal_size_limit = 67108864",  # 64 MB
    "PRAGMA mmap_size = 268435456",  # 256 MB
    "PRAGMA temp_store = MEMORY",
    "PRAGMA foreign_keys = ON",
]


# -- PostgreSQL table definitions (BIGINT for ms timestamps, DOUBLE PRECISION for floats) ------

_PG_OHLCV = """
CREATE TABLE IF NOT EXISTS ohlcv (
    timestamp_ms    BIGINT           NOT NULL,
    symbol          TEXT             NOT NULL,
    interval        TEXT             NOT NULL,
    open            DOUBLE PRECISION NOT NULL,
    high            DOUBLE PRECISION NOT NULL,
    low             DOUBLE PRECISION NOT NULL,
    close           DOUBLE PRECISION NOT NULL,
    volume          DOUBLE PRECISION NOT NULL,
    quote_volume    DOUBLE PRECISION,
    trades_count    INTEGER,
    PRIMARY KEY (timestamp_ms, symbol, interval)
)
"""

_PG_ORDERBOOK = """
CREATE TABLE IF NOT EXISTS orderbook_snapshot (
    timestamp_ms    BIGINT           NOT NULL,
    symbol          TEXT             NOT NULL,
    bid_prices      TEXT             NOT NULL,
    bid_sizes       TEXT             NOT NULL,
    ask_prices      TEXT             NOT NULL,
    ask_sizes       TEXT             NOT NULL,
    mid_price       DOUBLE PRECISION NOT NULL,
    spread          DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (timestamp_ms, symbol)
)
"""

_PG_TRADES = """
CREATE TABLE IF NOT EXISTS agg_trades (
    timestamp_ms    BIGINT           NOT NULL,
    symbol          TEXT             NOT NULL,
    price           DOUBLE PRECISION NOT NULL,
    quantity        DOUBLE PRECISION NOT NULL,
    is_buyer_maker  INTEGER          NOT NULL,
    trade_id        TEXT,
    PRIMARY KEY (timestamp_ms, symbol, trade_id)
)
"""

_PG_FUNDING = """
CREATE TABLE IF NOT EXISTS funding_rate (
    timestamp_ms    BIGINT           NOT NULL,
    symbol          TEXT             NOT NULL,
    funding_rate    DOUBLE PRECISION NOT NULL,
    mark_price      DOUBLE PRECISION,
    index_price     DOUBLE PRECISION,
    PRIMARY KEY (timestamp_ms, symbol)
)
"""

_PG_OI = """
CREATE TABLE IF NOT EXISTS open_interest (
    timestamp_ms    BIGINT           NOT NULL,
    symbol          TEXT             NOT NULL,
    open_interest   DOUBLE PRECISION NOT NULL,
    oi_value_usd    DOUBLE PRECISION,
    PRIMARY KEY (timestamp_ms, symbol)
)
"""

_PG_LIQUIDATION = """
CREATE TABLE IF NOT EXISTS liquidation (
    timestamp_ms    BIGINT           NOT NULL,
    symbol          TEXT             NOT NULL,
    side            TEXT             NOT NULL,
    price           DOUBLE PRECISION NOT NULL,
    quantity        DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (timestamp_ms, symbol, side, price)
)
"""

_PG_CROSS_MARKET = """
CREATE TABLE IF NOT EXISTS cross_market (
    timestamp_ms    BIGINT           NOT NULL,
    symbol          TEXT             NOT NULL,
    price           DOUBLE PRECISION NOT NULL,
    source          TEXT             NOT NULL,
    PRIMARY KEY (timestamp_ms, symbol, source)
)
"""

_PG_ONCHAIN = """
CREATE TABLE IF NOT EXISTS onchain_whale (
    timestamp_ms     BIGINT           NOT NULL,
    tx_hash          TEXT             NOT NULL,
    value_btc        DOUBLE PRECISION NOT NULL,
    fee_rate         DOUBLE PRECISION,
    is_exchange_flow INTEGER,
    PRIMARY KEY (timestamp_ms, tx_hash)
)
"""

_PG_REGIME = """
CREATE TABLE IF NOT EXISTS regime_label (
    timestamp_ms     BIGINT           NOT NULL,
    symbol           TEXT             NOT NULL,
    regime           TEXT             NOT NULL,
    hmm_state        INTEGER,
    hmm_prob         DOUBLE PRECISION,
    bocpd_run_length DOUBLE PRECISION,
    garch_vol        DOUBLE PRECISION,
    efficiency_ratio DOUBLE PRECISION,
    PRIMARY KEY (timestamp_ms, symbol)
)
"""

_PG_PREDICTION = """
CREATE TABLE IF NOT EXISTS prediction (
    timestamp_ms         BIGINT           NOT NULL,
    symbol               TEXT             NOT NULL,
    direction            TEXT             NOT NULL,
    confidence           DOUBLE PRECISION NOT NULL,
    calibrated_prob_up   DOUBLE PRECISION,
    calibrated_prob_down DOUBLE PRECISION,
    position_size        DOUBLE PRECISION,
    regime               TEXT,
    model_version        TEXT,
    PRIMARY KEY (timestamp_ms, symbol)
)
"""

_PG_FEATURE_SNAPSHOT = """
CREATE TABLE IF NOT EXISTS feature_snapshot (
    timestamp_ms    BIGINT NOT NULL,
    symbol          TEXT   NOT NULL,
    features_json   TEXT   NOT NULL,
    PRIMARY KEY (timestamp_ms, symbol)
)
"""

_PG_ALL_TABLES = [
    _PG_OHLCV,
    _PG_ORDERBOOK,
    _PG_TRADES,
    _PG_FUNDING,
    _PG_OI,
    _PG_LIQUIDATION,
    _PG_CROSS_MARKET,
    _PG_ONCHAIN,
    _PG_REGIME,
    _PG_PREDICTION,
    _PG_FEATURE_SNAPSHOT,
]

# Indexes are identical for both backends (PostgreSQL supports IF NOT EXISTS since 9.5)
_PG_INDEXES = INDEXES  # same DDL works


def create_postgres_tables(dsn: str) -> Any:
    """Create all tables and indexes in a PostgreSQL/TimescaleDB database.

    Args:
        dsn: PostgreSQL connection string, e.g.
             "postgresql://ep2:secret@localhost:5432/ep2_crypto"

    Returns:
        Open psycopg2 connection with all tables created.
    """
    try:
        import psycopg2  # type: ignore[import-untyped]
    except ImportError:
        msg = "psycopg2-binary required for PostgreSQL. Run: uv sync"
        raise ImportError(msg) from None

    conn = psycopg2.connect(dsn)
    conn.autocommit = False
    cur = conn.cursor()

    for ddl in _PG_ALL_TABLES:
        cur.execute(ddl)

    for idx_ddl in _PG_INDEXES:
        cur.execute(idx_ddl)

    conn.commit()

    cur.execute(
        "SELECT count(*) FROM information_schema.tables "
        "WHERE table_schema = 'public' AND table_type = 'BASE TABLE'"
    )
    row = cur.fetchone()
    table_count = row[0] if row else 0
    cur.close()

    logger.info("postgres_db_initialized", dsn=_redact_dsn(dsn), tables=table_count)
    return conn


def _redact_dsn(dsn: str) -> str:
    if "@" in dsn:
        prefix, rest = dsn.rsplit("@", 1)
        if ":" in prefix:
            scheme_user, _ = prefix.rsplit(":", 1)
            return f"{scheme_user}:***@{rest}"
    return dsn


def create_tables(db_path: Path | str) -> sqlite3.Connection:
    """Create all tables, indexes, and set PRAGMA options.

    Args:
        db_path: Path to SQLite database file. Use ":memory:" for in-memory.

    Returns:
        Open database connection with all tables created.
    """
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    for pragma in PRAGMA_SETTINGS:
        conn.execute(pragma)

    for table_ddl in ALL_TABLES:
        conn.execute(table_ddl)

    for index_ddl in INDEXES:
        conn.execute(index_ddl)

    conn.commit()

    table_count = conn.execute(
        "SELECT count(*) FROM sqlite_master WHERE type = ?", ("table",)
    ).fetchone()[0]

    logger.info("database_initialized", path=str(db_path), tables=table_count)

    return conn
