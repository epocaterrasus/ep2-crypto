"""Database schema for ep2-crypto.

SQLite is used in development; TimescaleDB (PostgreSQL) is used in production.
All timestamps are stored as INTEGER (Unix milliseconds) in SQLite.
TimescaleDB uses TIMESTAMPTZ — the migration script handles conversion.

TimescaleDB DDL lives in scripts/migrate_to_timescale.py.
This file owns the SQLite schema and the create_tables() helper used by dev
and test code.
"""

from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING

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
    trade_id        TEXT    NOT NULL DEFAULT '',
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
    timestamp_ms     INTEGER NOT NULL,
    tx_hash          TEXT    NOT NULL,
    value_btc        REAL    NOT NULL,
    fee_rate         REAL,
    is_exchange_flow INTEGER,
    PRIMARY KEY (timestamp_ms, tx_hash)
)
"""

REGIME_TABLE = """
CREATE TABLE IF NOT EXISTS regime_label (
    timestamp_ms     INTEGER NOT NULL,
    symbol           TEXT    NOT NULL,
    regime           TEXT    NOT NULL,
    hmm_state        INTEGER,
    hmm_prob         REAL,
    bocpd_run_length REAL,
    garch_vol        REAL,
    efficiency_ratio REAL,
    PRIMARY KEY (timestamp_ms, symbol)
)
"""

PREDICTION_TABLE = """
CREATE TABLE IF NOT EXISTS prediction (
    timestamp_ms         INTEGER NOT NULL,
    symbol               TEXT    NOT NULL,
    direction            TEXT    NOT NULL,
    confidence           REAL    NOT NULL,
    calibrated_prob_up   REAL,
    calibrated_prob_down REAL,
    position_size        REAL,
    regime               TEXT,
    model_version        TEXT,
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

# -- Monitoring tables (also defined in monitoring/performance_logger.py) ------
# Canonical DDL lives here; performance_logger.py creates its own SQLite
# connection independently and duplicates these definitions for isolation.

TRADE_LOG_TABLE = """
CREATE TABLE IF NOT EXISTS trade_log (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp_ms         INTEGER NOT NULL,
    direction            TEXT    NOT NULL,
    predicted_confidence REAL    NOT NULL,
    predicted_magnitude  REAL    NOT NULL,
    actual_direction     TEXT,
    actual_return        REAL,
    pnl                  REAL,
    slippage_expected    REAL,
    slippage_actual      REAL,
    latency_ms           REAL,
    regime               TEXT,
    features_json        TEXT,
    entry_price          REAL,
    exit_price           REAL,
    position_size        REAL,
    meta_json            TEXT,
    created_at           REAL    NOT NULL DEFAULT (strftime('%s', 'now'))
)
"""

BAR_STATE_LOG_TABLE = """
CREATE TABLE IF NOT EXISTS bar_state_log (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp_ms         INTEGER NOT NULL,
    bar_close            REAL    NOT NULL,
    regime               TEXT,
    regime_confidence    REAL,
    model_prediction     TEXT,
    model_confidence     REAL,
    risk_state_json      TEXT,
    feature_values_json  TEXT,
    kill_switch_active   INTEGER NOT NULL DEFAULT 0,
    drawdown_multiplier  REAL,
    volatility_ann       REAL,
    position_open        INTEGER NOT NULL DEFAULT 0,
    equity               REAL,
    created_at           REAL    NOT NULL DEFAULT (strftime('%s', 'now'))
)
"""

# -- Polymarket tables ---------------------------------------------------------
# SQLite versions for dev/test; TimescaleDB versions are in migrate_to_timescale.py.

POLYMARKET_MARKETS_TABLE = """
CREATE TABLE IF NOT EXISTS polymarket_markets (
    timestamp_ms    INTEGER NOT NULL,
    condition_id    TEXT    NOT NULL,
    question        TEXT    NOT NULL,
    end_date_iso    TEXT,
    active          INTEGER NOT NULL DEFAULT 1,
    resolved        INTEGER NOT NULL DEFAULT 0,
    resolution      TEXT,
    yes_price       REAL,
    no_price        REAL,
    volume_usd      REAL,
    liquidity_usd   REAL,
    raw_json        TEXT,
    PRIMARY KEY (timestamp_ms, condition_id)
)
"""

POLYMARKET_POSITIONS_TABLE = """
CREATE TABLE IF NOT EXISTS polymarket_positions (
    timestamp_ms    INTEGER NOT NULL,
    condition_id    TEXT    NOT NULL,
    side            TEXT    NOT NULL,
    size            REAL    NOT NULL,
    avg_price       REAL    NOT NULL,
    current_price   REAL,
    unrealized_pnl  REAL,
    realized_pnl    REAL,
    status          TEXT    NOT NULL DEFAULT 'open',
    PRIMARY KEY (timestamp_ms, condition_id, side)
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
    TRADE_LOG_TABLE,
    BAR_STATE_LOG_TABLE,
    POLYMARKET_MARKETS_TABLE,
    POLYMARKET_POSITIONS_TABLE,
]

# -- Index definitions --------------------------------------------------------
# All time-series tables: compound (symbol, timestamp_ms) index for the
# most common query pattern (WHERE symbol = ? AND timestamp_ms BETWEEN ? AND ?).
# Single-column timestamp_ms indexes are retained for queries without a symbol filter.

INDEXES = [
    # Single-column time indexes
    "CREATE INDEX IF NOT EXISTS idx_ohlcv_ts ON ohlcv (timestamp_ms)",
    "CREATE INDEX IF NOT EXISTS idx_orderbook_ts ON orderbook_snapshot (timestamp_ms)",
    "CREATE INDEX IF NOT EXISTS idx_trades_ts ON agg_trades (timestamp_ms)",
    "CREATE INDEX IF NOT EXISTS idx_funding_ts ON funding_rate (timestamp_ms)",
    "CREATE INDEX IF NOT EXISTS idx_oi_ts ON open_interest (timestamp_ms)",
    "CREATE INDEX IF NOT EXISTS idx_liquidation_ts ON liquidation (timestamp_ms)",
    "CREATE INDEX IF NOT EXISTS idx_cross_market_ts ON cross_market (timestamp_ms)",
    "CREATE INDEX IF NOT EXISTS idx_regime_ts ON regime_label (timestamp_ms)",
    "CREATE INDEX IF NOT EXISTS idx_prediction_ts ON prediction (timestamp_ms)",
    "CREATE INDEX IF NOT EXISTS idx_trade_log_ts ON trade_log (timestamp_ms)",
    "CREATE INDEX IF NOT EXISTS idx_bar_state_log_ts ON bar_state_log (timestamp_ms)",
    "CREATE INDEX IF NOT EXISTS idx_polymarket_markets_ts ON polymarket_markets (timestamp_ms)",
    "CREATE INDEX IF NOT EXISTS idx_polymarket_positions_ts ON polymarket_positions (timestamp_ms)",
    # Compound (symbol, timestamp_ms) indexes — covers the repository.py query pattern
    "CREATE INDEX IF NOT EXISTS idx_ohlcv_sym_ts ON ohlcv (symbol, timestamp_ms)",
    "CREATE INDEX IF NOT EXISTS idx_orderbook_sym_ts ON orderbook_snapshot (symbol, timestamp_ms)",
    "CREATE INDEX IF NOT EXISTS idx_trades_sym_ts ON agg_trades (symbol, timestamp_ms)",
    "CREATE INDEX IF NOT EXISTS idx_funding_sym_ts ON funding_rate (symbol, timestamp_ms)",
    "CREATE INDEX IF NOT EXISTS idx_oi_sym_ts ON open_interest (symbol, timestamp_ms)",
    "CREATE INDEX IF NOT EXISTS idx_liquidation_sym_ts ON liquidation (symbol, timestamp_ms)",
    "CREATE INDEX IF NOT EXISTS idx_cross_market_sym_ts ON cross_market (symbol, timestamp_ms)",
    "CREATE INDEX IF NOT EXISTS idx_regime_sym_ts ON regime_label (symbol, timestamp_ms)",
    "CREATE INDEX IF NOT EXISTS idx_prediction_sym_ts ON prediction (symbol, timestamp_ms)",
    # Polymarket condition_id lookup
    "CREATE INDEX IF NOT EXISTS idx_polymarket_markets_cid ON polymarket_markets (condition_id, timestamp_ms)",
    "CREATE INDEX IF NOT EXISTS idx_polymarket_positions_cid ON polymarket_positions (condition_id, timestamp_ms)",
]

# -- PRAGMA settings ----------------------------------------------------------

PRAGMA_SETTINGS = [
    "PRAGMA journal_mode = WAL",
    "PRAGMA synchronous = NORMAL",
    "PRAGMA cache_size = -64000",        # 64 MB
    "PRAGMA busy_timeout = 5000",
    "PRAGMA journal_size_limit = 67108864",  # 64 MB
    "PRAGMA mmap_size = 268435456",          # 256 MB
    "PRAGMA temp_store = MEMORY",
    "PRAGMA foreign_keys = ON",
]


def create_tables(db_path: "Path | str") -> sqlite3.Connection:
    """Create all tables, indexes, and set PRAGMA options.

    This function is idempotent — it uses IF NOT EXISTS throughout and
    can be called multiple times without side effects.

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
