"""Backfill historical BTC data from Binance Futures, Bybit, and yfinance.

Data sources (all free, no API keys):
  1. Binance Futures REST: 1m OHLCV klines for BTCUSDT perpetual
  2. Binance Futures REST: Funding rate history (every 8h)
  3. Bybit REST: Open interest history (5min intervals)
  4. yfinance: NQ, Gold, DXY, ETH-USD daily OHLCV

Backend selection (controlled by environment variables, see config.py):
  EP2_DB_BACKEND=sqlite          -> uses EP2_DB_SQLITE_PATH (default: data/ep2_crypto.db)
  EP2_DB_BACKEND=timescaledb     -> uses EP2_DB_TIMESCALEDB_URL (PostgreSQL DSN)

Usage:
  uv run python scripts/collect_history.py --start 2019-09-01 --end today
  uv run python scripts/collect_history.py --start 2024-01-01 --end today
  uv run python scripts/collect_history.py --resume
"""

from __future__ import annotations

import argparse
import asyncio
import signal
import sys
import time
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, AsyncGenerator

import httpx
import structlog
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so we can import ep2_crypto
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from ep2_crypto.config import DatabaseConfig  # noqa: E402
from ep2_crypto.db.repository import Repository  # noqa: E402
from ep2_crypto.db.schema import create_tables  # noqa: E402

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SYMBOL = "BTCUSDT"
DEFAULT_START = "2019-09-01"

BINANCE_FAPI = "https://fapi.binance.com"
BYBIT_API = "https://api.bybit.com"

# Binance Futures weight limits:
#   /fapi/v1/klines costs 2 weight per call
#   Hard limit: 2400 weight/min -> 1200 calls/min max
#   Safe target: 600 calls/min -> 1 call per 0.10s
#   We use 0.12s to stay well under with burst headroom.
BINANCE_DELAY_S = 0.12

# Bybit: 120 requests/min -> 1 per 0.5s, we use 0.55s for headroom
BYBIT_DELAY_S = 0.55

BATCH_COMMIT_SIZE = 10_000
PROGRESS_LOG_INTERVAL = 1_000  # structlog checkpoint every N rows

CROSS_MARKET_SYMBOLS: dict[str, str] = {
    "NQ=F": "yfinance",
    "GC=F": "yfinance",
    "DX-Y.NYB": "yfinance",
    "ETH-USD": "yfinance",
}

# Tracks whether a SIGINT has been received so collectors can exit cleanly.
_shutdown_requested = False


def _request_shutdown(signum: int, frame: object) -> None:
    global _shutdown_requested  # noqa: PLW0603
    _shutdown_requested = True
    logger.warning("sigint_received", msg="Graceful shutdown requested; finishing current batch")


signal.signal(signal.SIGINT, _request_shutdown)


# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------


def _parse_date(date_str: str) -> datetime:
    """Parse a date string (YYYY-MM-DD or 'today') to a UTC datetime."""
    if date_str.lower() == "today":
        return datetime.now(tz=UTC).replace(hour=0, minute=0, second=0, microsecond=0)
    return datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=UTC)


def _dt_to_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def _ms_to_dt(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1000, tz=UTC)


# ---------------------------------------------------------------------------
# Database connection — supports SQLite and TimescaleDB (PostgreSQL)
# ---------------------------------------------------------------------------


def _build_repo(db_config: DatabaseConfig) -> tuple[Repository, Any]:
    """Create and return (Repository, raw_connection) for the configured backend.

    For SQLite the raw connection is a sqlite3.Connection.
    For TimescaleDB the raw connection is a psycopg2 connection.
    The caller owns the lifecycle (must close after use).
    """
    if db_config.backend == "timescaledb":
        try:
            import psycopg2  # type: ignore[import-untyped]
            import psycopg2.extras  # type: ignore[import-untyped]
        except ImportError as exc:
            logger.error(
                "psycopg2_not_installed",
                hint="uv sync --extra timescaledb  (or: pip install psycopg2-binary)",
            )
            raise SystemExit(1) from exc

        dsn = db_config.timescaledb_url.get_secret_value()
        if not dsn:
            logger.error(
                "missing_timescaledb_url",
                hint="Set EP2_DB_TIMESCALEDB_URL=postgresql://user:pass@host/dbname",
            )
            raise SystemExit(1)

        conn = psycopg2.connect(dsn)
        conn.autocommit = False
        _ensure_postgres_schema(conn)
        repo = _PostgresRepository(conn)
        logger.info("database_ready", backend="timescaledb")
        return repo, conn  # type: ignore[return-value]

    # Default: SQLite
    sqlite_path = db_config.sqlite_path
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    sqlite_conn = create_tables(sqlite_path)
    repo = Repository(sqlite_conn)
    logger.info("database_ready", backend="sqlite", path=str(sqlite_path))
    return repo, sqlite_conn


def _ensure_postgres_schema(conn: Any) -> None:
    """Create tables in PostgreSQL if they do not yet exist.

    Uses PostgreSQL-compatible DDL (ON CONFLICT, BIGINT instead of INTEGER).
    TimescaleDB hypertables are created for time-series tables.
    """
    ddl_statements = [
        # OHLCV
        """
        CREATE TABLE IF NOT EXISTS ohlcv (
            timestamp_ms    BIGINT  NOT NULL,
            symbol          TEXT    NOT NULL,
            interval        TEXT    NOT NULL,
            open            DOUBLE PRECISION NOT NULL,
            high            DOUBLE PRECISION NOT NULL,
            low             DOUBLE PRECISION NOT NULL,
            close           DOUBLE PRECISION NOT NULL,
            volume          DOUBLE PRECISION NOT NULL,
            quote_volume    DOUBLE PRECISION,
            trades_count    INTEGER,
            PRIMARY KEY (timestamp_ms, symbol, interval)
        )
        """,
        # funding_rate
        """
        CREATE TABLE IF NOT EXISTS funding_rate (
            timestamp_ms    BIGINT  NOT NULL,
            symbol          TEXT    NOT NULL,
            funding_rate    DOUBLE PRECISION NOT NULL,
            mark_price      DOUBLE PRECISION,
            index_price     DOUBLE PRECISION,
            PRIMARY KEY (timestamp_ms, symbol)
        )
        """,
        # open_interest
        """
        CREATE TABLE IF NOT EXISTS open_interest (
            timestamp_ms    BIGINT  NOT NULL,
            symbol          TEXT    NOT NULL,
            open_interest   DOUBLE PRECISION NOT NULL,
            oi_value_usd    DOUBLE PRECISION,
            PRIMARY KEY (timestamp_ms, symbol)
        )
        """,
        # cross_market
        """
        CREATE TABLE IF NOT EXISTS cross_market (
            timestamp_ms    BIGINT  NOT NULL,
            symbol          TEXT    NOT NULL,
            price           DOUBLE PRECISION NOT NULL,
            source          TEXT    NOT NULL,
            PRIMARY KEY (timestamp_ms, symbol, source)
        )
        """,
    ]
    indexes = [
        "CREATE INDEX IF NOT EXISTS idx_ohlcv_ts ON ohlcv (timestamp_ms)",
        "CREATE INDEX IF NOT EXISTS idx_funding_ts ON funding_rate (timestamp_ms)",
        "CREATE INDEX IF NOT EXISTS idx_oi_ts ON open_interest (timestamp_ms)",
        "CREATE INDEX IF NOT EXISTS idx_cm_ts ON cross_market (timestamp_ms)",
    ]

    with conn.cursor() as cur:
        for stmt in ddl_statements:
            cur.execute(stmt)
        for idx in indexes:
            cur.execute(idx)
        # Attempt to create TimescaleDB hypertables; ignore if already done or
        # if TimescaleDB extension is not available.
        for table in ("ohlcv", "funding_rate", "open_interest", "cross_market"):
            try:
                cur.execute(
                    "SELECT create_hypertable(%s, 'timestamp_ms', if_not_exists => TRUE, "
                    "chunk_time_interval => 86400000)",
                    (table,),
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "hypertable_skip",
                    table=table,
                    reason=str(exc),
                )
                conn.rollback()
    conn.commit()
    logger.info("postgres_schema_ready")


class _PostgresRepository:
    """Thin wrapper that exposes the same batch-insert interface as Repository
    but uses PostgreSQL-compatible ON CONFLICT syntax."""

    def __init__(self, conn: Any) -> None:
        self._conn = conn

    # -- last timestamp -------------------------------------------------------

    def last_timestamp(self, table: str, symbol: str) -> int | None:
        allowed = {"ohlcv", "funding_rate", "open_interest", "cross_market"}
        if table not in allowed:
            raise ValueError(f"Unknown table: {table}")
        with self._conn.cursor() as cur:
            cur.execute(
                f"SELECT MAX(timestamp_ms) FROM {table} WHERE symbol = %s",  # noqa: S608
                (symbol,),
            )
            row = cur.fetchone()
        return row[0] if row and row[0] is not None else None

    # -- OHLCV ----------------------------------------------------------------

    def insert_ohlcv_batch(self, rows: list[tuple[Any, ...]]) -> int:
        sql = (
            "INSERT INTO ohlcv VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) "
            "ON CONFLICT (timestamp_ms, symbol, interval) DO UPDATE SET "
            "open=EXCLUDED.open, high=EXCLUDED.high, low=EXCLUDED.low, "
            "close=EXCLUDED.close, volume=EXCLUDED.volume, "
            "quote_volume=EXCLUDED.quote_volume, trades_count=EXCLUDED.trades_count"
        )
        import psycopg2.extras  # type: ignore[import-untyped]

        with self._conn.cursor() as cur:
            psycopg2.extras.execute_batch(cur, sql, rows, page_size=2000)
        self._conn.commit()
        return len(rows)

    # -- Funding rate ---------------------------------------------------------

    def insert_funding_batch(self, rows: list[tuple[Any, ...]]) -> int:
        sql = (
            "INSERT INTO funding_rate VALUES (%s,%s,%s,%s,%s) "
            "ON CONFLICT (timestamp_ms, symbol) DO UPDATE SET "
            "funding_rate=EXCLUDED.funding_rate, mark_price=EXCLUDED.mark_price, "
            "index_price=EXCLUDED.index_price"
        )
        import psycopg2.extras  # type: ignore[import-untyped]

        with self._conn.cursor() as cur:
            psycopg2.extras.execute_batch(cur, sql, rows, page_size=2000)
        self._conn.commit()
        return len(rows)

    # -- Open interest --------------------------------------------------------

    def insert_oi_batch(self, rows: list[tuple[Any, ...]]) -> int:
        sql = (
            "INSERT INTO open_interest VALUES (%s,%s,%s,%s) "
            "ON CONFLICT (timestamp_ms, symbol) DO UPDATE SET "
            "open_interest=EXCLUDED.open_interest, oi_value_usd=EXCLUDED.oi_value_usd"
        )
        import psycopg2.extras  # type: ignore[import-untyped]

        with self._conn.cursor() as cur:
            psycopg2.extras.execute_batch(cur, sql, rows, page_size=2000)
        self._conn.commit()
        return len(rows)

    # -- Cross-market ---------------------------------------------------------

    def insert_cross_market_batch(self, rows: list[tuple[Any, ...]]) -> int:
        sql = (
            "INSERT INTO cross_market VALUES (%s,%s,%s,%s) "
            "ON CONFLICT (timestamp_ms, symbol, source) DO UPDATE SET "
            "price=EXCLUDED.price"
        )
        import psycopg2.extras  # type: ignore[import-untyped]

        with self._conn.cursor() as cur:
            psycopg2.extras.execute_batch(cur, sql, rows, page_size=2000)
        self._conn.commit()
        return len(rows)

    # -- Summary queries ------------------------------------------------------

    def count_and_range(
        self, table: str, symbol: str
    ) -> tuple[int, int | None, int | None]:
        """Return (count, min_ts_ms, max_ts_ms) for a symbol in a table."""
        allowed = {"ohlcv", "funding_rate", "open_interest", "cross_market"}
        if table not in allowed:
            raise ValueError(f"Unknown table: {table}")
        with self._conn.cursor() as cur:
            cur.execute(
                f"SELECT COUNT(*), MIN(timestamp_ms), MAX(timestamp_ms) "  # noqa: S608
                f"FROM {table} WHERE symbol = %s",
                (symbol,),
            )
            row = cur.fetchone()
        if row:
            return (row[0] or 0, row[1], row[2])
        return (0, None, None)

    def cross_market_counts(self) -> list[tuple[str, int]]:
        with self._conn.cursor() as cur:
            cur.execute("SELECT symbol, COUNT(*) FROM cross_market GROUP BY symbol")
            return cur.fetchall()


# ---------------------------------------------------------------------------
# Unified adapter so collectors work with either backend
# ---------------------------------------------------------------------------


class _DbAdapter:
    """Abstracts over SQLite Repository and _PostgresRepository for collectors.

    Collectors call adapter methods; this class routes to the correct
    backend and SQL dialect.
    """

    def __init__(self, repo: Repository | _PostgresRepository, backend: str) -> None:
        self._repo = repo
        self._backend = backend

    @property
    def is_postgres(self) -> bool:
        return self._backend == "timescaledb"

    def last_timestamp(self, table: str, symbol: str) -> int | None:
        if self.is_postgres:
            return self._repo.last_timestamp(table, symbol)  # type: ignore[union-attr]
        # SQLite path via raw connection (Repository exposes connection property)
        allowed = {"ohlcv", "funding_rate", "open_interest", "cross_market"}
        if table not in allowed:
            raise ValueError(f"Unknown table: {table}")
        row = self._repo.connection.execute(  # type: ignore[union-attr]
            f"SELECT MAX(timestamp_ms) FROM {table} WHERE symbol = ?",  # noqa: S608
            (symbol,),
        ).fetchone()
        return row[0] if row and row[0] is not None else None

    def insert_ohlcv_batch(self, rows: list[tuple[Any, ...]]) -> int:
        if self.is_postgres:
            return self._repo.insert_ohlcv_batch(rows)  # type: ignore[union-attr]
        return self._repo.insert_ohlcv_batch(rows)  # type: ignore[union-attr]

    def insert_funding_batch(self, rows: list[tuple[Any, ...]]) -> int:
        if self.is_postgres:
            return self._repo.insert_funding_batch(rows)  # type: ignore[union-attr]
        # SQLite: Repository doesn't have a batch funding method; add inline
        self._repo.connection.executemany(  # type: ignore[union-attr]
            "INSERT OR REPLACE INTO funding_rate VALUES (?, ?, ?, ?, ?)",
            rows,
        )
        self._repo.connection.commit()  # type: ignore[union-attr]
        return len(rows)

    def insert_oi_batch(self, rows: list[tuple[Any, ...]]) -> int:
        if self.is_postgres:
            return self._repo.insert_oi_batch(rows)  # type: ignore[union-attr]
        self._repo.connection.executemany(  # type: ignore[union-attr]
            "INSERT OR REPLACE INTO open_interest VALUES (?, ?, ?, ?)",
            rows,
        )
        self._repo.connection.commit()  # type: ignore[union-attr]
        return len(rows)

    def insert_cross_market_batch(self, rows: list[tuple[Any, ...]]) -> int:
        if self.is_postgres:
            return self._repo.insert_cross_market_batch(rows)  # type: ignore[union-attr]
        self._repo.connection.executemany(  # type: ignore[union-attr]
            "INSERT OR REPLACE INTO cross_market VALUES (?, ?, ?, ?)",
            rows,
        )
        self._repo.connection.commit()  # type: ignore[union-attr]
        return len(rows)

    def count_and_range(
        self, table: str, symbol: str
    ) -> tuple[int, int | None, int | None]:
        if self.is_postgres:
            return self._repo.count_and_range(table, symbol)  # type: ignore[union-attr]
        allowed = {"ohlcv", "funding_rate", "open_interest", "cross_market"}
        if table not in allowed:
            raise ValueError(f"Unknown table: {table}")
        conn = self._repo.connection  # type: ignore[union-attr]
        count = conn.execute(
            f"SELECT COUNT(*) FROM {table} WHERE symbol = ?",  # noqa: S608
            (symbol,),
        ).fetchone()[0]
        row = conn.execute(
            f"SELECT MIN(timestamp_ms), MAX(timestamp_ms) FROM {table} WHERE symbol = ?",  # noqa: S608
            (symbol,),
        ).fetchone()
        min_ts = row[0] if row else None
        max_ts = row[1] if row else None
        return (count, min_ts, max_ts)

    def cross_market_counts(self) -> list[tuple[str, int]]:
        if self.is_postgres:
            return self._repo.cross_market_counts()  # type: ignore[union-attr]
        conn = self._repo.connection  # type: ignore[union-attr]
        return conn.execute(
            "SELECT symbol, COUNT(*) FROM cross_market GROUP BY symbol"
        ).fetchall()

    def close(self) -> None:
        if self.is_postgres:
            self._repo._conn.close()  # type: ignore[union-attr]
        else:
            self._repo.connection.close()  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Retry / HTTP helpers
# ---------------------------------------------------------------------------
MAX_RETRIES = 5
BACKOFF_BASE = 2.0
BACKOFF_MAX_S = 60.0


async def _fetch_json(
    client: httpx.AsyncClient,
    url: str,
    params: dict[str, Any],
    *,
    delay: float = 0.12,
) -> Any:
    """Fetch JSON with exponential backoff retry on rate-limit and transient errors."""
    for attempt in range(MAX_RETRIES):
        try:
            resp = await client.get(url, params=params, timeout=30.0)
            if resp.status_code == 429:
                retry_after = resp.headers.get("Retry-After")
                wait = float(retry_after) if retry_after else BACKOFF_BASE ** (attempt + 2)
                wait = min(wait, BACKOFF_MAX_S)
                logger.warning("rate_limited", url=url, wait_s=wait, attempt=attempt + 1)
                await asyncio.sleep(wait)
                continue
            if resp.status_code in (418, 503):
                # Binance IP ban or service unavailable — back off longer
                wait = min(BACKOFF_BASE ** (attempt + 3), BACKOFF_MAX_S)
                logger.warning(
                    "server_unavailable",
                    status=resp.status_code,
                    url=url,
                    wait_s=wait,
                )
                await asyncio.sleep(wait)
                continue
            resp.raise_for_status()
            await asyncio.sleep(delay)
            return resp.json()
        except (httpx.HTTPStatusError, httpx.ReadTimeout, httpx.ConnectTimeout) as exc:
            wait = min(BACKOFF_BASE ** (attempt + 1), BACKOFF_MAX_S)
            logger.warning(
                "fetch_error",
                url=url,
                attempt=attempt + 1,
                max_attempts=MAX_RETRIES,
                error=str(exc),
                wait_s=wait,
            )
            if attempt == MAX_RETRIES - 1:
                logger.error("fetch_failed_permanently", url=url, params=params)
                return None
            await asyncio.sleep(wait)
    return None


# ---------------------------------------------------------------------------
# 1. Binance Futures: 1m OHLCV klines
# ---------------------------------------------------------------------------
async def collect_ohlcv(
    adapter: _DbAdapter,
    client: httpx.AsyncClient,
    start_ms: int,
    end_ms: int,
) -> int:
    """Collect 1m OHLCV klines from Binance Futures and persist via repository."""
    url = f"{BINANCE_FAPI}/fapi/v1/klines"
    current_ms = start_ms
    total_rows = 0
    batch: list[tuple[Any, ...]] = []
    rows_since_last_log = 0

    total_bars_est = (end_ms - start_ms) // 60_000
    pbar = tqdm(total=total_bars_est, desc="OHLCV 1m", unit="bars")

    try:
        while current_ms < end_ms and not _shutdown_requested:
            params: dict[str, Any] = {
                "symbol": SYMBOL,
                "interval": "1m",
                "startTime": current_ms,
                "endTime": end_ms,
                "limit": 1500,
            }
            data = await _fetch_json(client, url, params, delay=BINANCE_DELAY_S)
            if data is None:
                # Permanent failure for this chunk — skip forward to avoid infinite loop
                logger.warning("ohlcv_chunk_skipped", start_ms=current_ms, bars=1500)
                current_ms += 1500 * 60_000
                pbar.update(1500)
                continue

            if len(data) == 0:
                break

            for k in data:
                # Binance kline format:
                # [open_time, open, high, low, close, volume, close_time,
                #  quote_volume, trades_count, ...]
                row = (
                    int(k[0]),          # timestamp_ms (open time)
                    SYMBOL,             # symbol
                    "1m",               # interval
                    float(k[1]),        # open
                    float(k[2]),        # high
                    float(k[3]),        # low
                    float(k[4]),        # close
                    float(k[5]),        # volume
                    float(k[7]),        # quote_volume
                    int(k[8]),          # trades_count
                )
                batch.append(row)

            fetched = len(data)
            pbar.update(fetched)
            rows_since_last_log += fetched
            current_ms = int(data[-1][0]) + 60_000

            if len(batch) >= BATCH_COMMIT_SIZE:
                adapter.insert_ohlcv_batch(batch)
                total_rows += len(batch)
                if rows_since_last_log >= PROGRESS_LOG_INTERVAL:
                    logger.info(
                        "ohlcv_progress",
                        total_rows=total_rows,
                        up_to=_ms_to_dt(current_ms).isoformat(),
                    )
                    rows_since_last_log = 0
                batch = []

        # Final flush
        if batch:
            adapter.insert_ohlcv_batch(batch)
            total_rows += len(batch)

    finally:
        pbar.close()

    logger.info("ohlcv_complete", total_rows=total_rows, symbol=SYMBOL)
    return total_rows


# ---------------------------------------------------------------------------
# 2. Binance Futures: Funding rate history
# ---------------------------------------------------------------------------
async def collect_funding(
    adapter: _DbAdapter,
    client: httpx.AsyncClient,
    start_ms: int,
    end_ms: int,
) -> int:
    """Collect funding rate history from Binance Futures."""
    url = f"{BINANCE_FAPI}/fapi/v1/fundingRate"
    current_ms = start_ms
    total_rows = 0
    batch: list[tuple[Any, ...]] = []

    total_est = max(1, (end_ms - start_ms) // (8 * 3600 * 1000))
    pbar = tqdm(total=total_est, desc="Funding rate", unit="rows")

    try:
        while current_ms < end_ms and not _shutdown_requested:
            params: dict[str, Any] = {
                "symbol": SYMBOL,
                "startTime": current_ms,
                "endTime": end_ms,
                "limit": 1000,
            }
            data = await _fetch_json(client, url, params, delay=BINANCE_DELAY_S)
            if data is None:
                logger.warning("funding_chunk_skipped", start_ms=current_ms)
                current_ms += 1000 * 8 * 3600 * 1000
                continue

            if len(data) == 0:
                break

            for item in data:
                row = (
                    int(item["fundingTime"]),
                    SYMBOL,
                    float(item["fundingRate"]),
                    float(item["markPrice"]) if item.get("markPrice") else None,
                    None,  # index_price not in this endpoint
                )
                batch.append(row)

            pbar.update(len(data))
            current_ms = int(data[-1]["fundingTime"]) + 1

            if len(batch) >= BATCH_COMMIT_SIZE:
                adapter.insert_funding_batch(batch)
                total_rows += len(batch)
                logger.info("funding_progress", total_rows=total_rows)
                batch = []

            if len(data) < 1000:
                break

        if batch:
            adapter.insert_funding_batch(batch)
            total_rows += len(batch)

    finally:
        pbar.close()

    logger.info("funding_complete", total_rows=total_rows, symbol=SYMBOL)
    return total_rows


# ---------------------------------------------------------------------------
# 3. Bybit REST: Open interest history
# ---------------------------------------------------------------------------
async def collect_open_interest(
    adapter: _DbAdapter,
    client: httpx.AsyncClient,
    start_ms: int,
    end_ms: int,
) -> int:
    """Collect open interest history from Bybit (5min intervals)."""
    url = f"{BYBIT_API}/v5/market/open-interest"
    current_ms = start_ms
    total_rows = 0
    batch: list[tuple[Any, ...]] = []

    total_est = max(1, (end_ms - start_ms) // (5 * 60_000))
    pbar = tqdm(total=total_est, desc="Open interest", unit="rows")

    try:
        while current_ms < end_ms and not _shutdown_requested:
            params: dict[str, Any] = {
                "category": "linear",
                "symbol": SYMBOL,
                "intervalTime": "5min",
                "startTime": current_ms,
                "endTime": min(current_ms + 200 * 5 * 60_000, end_ms),
                "limit": 200,
            }
            data = await _fetch_json(client, url, params, delay=BYBIT_DELAY_S)
            if data is None or data.get("retCode") != 0:
                err_msg = data.get("retMsg", "unknown") if data else "no response"
                logger.warning("bybit_oi_error", msg=err_msg, start_ms=current_ms)
                current_ms += 200 * 5 * 60_000
                pbar.update(200)
                continue

            result_list: list[dict[str, Any]] = data.get("result", {}).get("list", [])
            if not result_list:
                current_ms += 200 * 5 * 60_000
                pbar.update(200)
                continue

            for item in result_list:
                ts = int(item["timestamp"])
                oi = float(item["openInterest"])
                batch.append((ts, SYMBOL, oi, None))

            pbar.update(len(result_list))

            timestamps = [int(item["timestamp"]) for item in result_list]
            newest = max(timestamps)
            current_ms = newest + 5 * 60_000

            if len(batch) >= BATCH_COMMIT_SIZE:
                adapter.insert_oi_batch(batch)
                total_rows += len(batch)
                logger.info("oi_progress", total_rows=total_rows)
                batch = []

            if len(result_list) < 200:
                break

        if batch:
            adapter.insert_oi_batch(batch)
            total_rows += len(batch)

    finally:
        pbar.close()

    logger.info("open_interest_complete", total_rows=total_rows, symbol=SYMBOL)
    return total_rows


# ---------------------------------------------------------------------------
# 4. yfinance: Cross-market daily data (runs in executor to avoid blocking loop)
# ---------------------------------------------------------------------------
def _collect_cross_market_sync(
    adapter: _DbAdapter,
    start_date: str,
    end_date: str,
) -> int:
    """Synchronous cross-market collection — called via run_in_executor."""
    try:
        import yfinance as yf  # type: ignore[import-untyped]
    except ImportError:
        logger.error("yfinance_not_installed", hint="uv sync --extra ingest")
        return 0

    total_rows = 0

    for symbol, source in tqdm(CROSS_MARKET_SYMBOLS.items(), desc="Cross-market", unit="sym"):
        if _shutdown_requested:
            break
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date, interval="1d")
            if df.empty:
                logger.warning("yfinance_no_data", symbol=symbol)
                continue

            batch: list[tuple[Any, ...]] = []
            for idx, row in df.iterrows():
                close_val = row["Close"]
                # Skip rows with NaN close price
                if close_val is None or (isinstance(close_val, float) and close_val != close_val):
                    continue
                ts_ms = int(idx.timestamp() * 1000)  # type: ignore[union-attr]
                batch.append((ts_ms, symbol, float(close_val), source))

            if batch:
                adapter.insert_cross_market_batch(batch)
                total_rows += len(batch)
                logger.info(
                    "cross_market_symbol_done",
                    symbol=symbol,
                    rows=len(batch),
                )
        except Exception:
            logger.exception("cross_market_error", symbol=symbol)

    logger.info("cross_market_complete", total_rows=total_rows)
    return total_rows


async def collect_cross_market(
    adapter: _DbAdapter,
    start_date: str,
    end_date: str,
) -> int:
    """Async wrapper — runs yfinance in the default thread pool executor."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None,
        _collect_cross_market_sync,
        adapter,
        start_date,
        end_date,
    )


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def log_summary(adapter: _DbAdapter) -> None:
    """Log a summary of all collected data using structlog."""
    tables = {
        "ohlcv": SYMBOL,
        "funding_rate": SYMBOL,
        "open_interest": SYMBOL,
    }

    logger.info("collection_summary_start", separator="=" * 60)

    for table, symbol in tables.items():
        count, min_ts, max_ts = adapter.count_and_range(table, symbol)
        if count > 0 and min_ts is not None and max_ts is not None:
            start_dt = _ms_to_dt(min_ts).strftime("%Y-%m-%d")
            end_dt = _ms_to_dt(max_ts).strftime("%Y-%m-%d")
            logger.info(
                "table_summary",
                table=table,
                rows=count,
                range_start=start_dt,
                range_end=end_dt,
            )
        else:
            logger.info("table_summary", table=table, rows=0, range="N/A")

    for row in adapter.cross_market_counts():
        logger.info("table_summary", table="cross_market", symbol=row[0], rows=row[1])

    logger.info("collection_summary_end", separator="=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def async_main(args: argparse.Namespace) -> None:
    """Run all collectors."""
    db_config = DatabaseConfig()
    repo, _raw_conn = _build_repo(db_config)
    adapter = _DbAdapter(repo, db_config.backend)

    logger.info(
        "collector_start",
        backend=db_config.backend,
        symbol=SYMBOL,
    )

    try:
        # Determine date range
        if args.resume:
            last_ohlcv = adapter.last_timestamp("ohlcv", SYMBOL)
            if last_ohlcv is not None:
                start_ms = last_ohlcv + 60_000  # next minute after last stored bar
                logger.info("resuming_from", timestamp=_ms_to_dt(start_ms).isoformat())
            else:
                start_ms = _dt_to_ms(_parse_date(DEFAULT_START))
                logger.info("no_existing_data_starting_fresh", start=DEFAULT_START)
        else:
            start_ms = _dt_to_ms(_parse_date(args.start))

        end_ms = _dt_to_ms(_parse_date(args.end))

        if start_ms >= end_ms:
            logger.info("already_up_to_date", start_ms=start_ms, end_ms=end_ms)
            return

        start_dt = _ms_to_dt(start_ms)
        end_dt = _ms_to_dt(end_ms)

        logger.info(
            "collection_range",
            start=start_dt.strftime("%Y-%m-%d"),
            end=end_dt.strftime("%Y-%m-%d"),
            days=round((end_ms - start_ms) / 86_400_000, 1),
        )

        t0 = time.monotonic()

        async with httpx.AsyncClient(
            headers={"User-Agent": "ep2-crypto/0.1"},
            follow_redirects=True,
        ) as client:
            # OHLCV first (largest dataset)
            ohlcv_rows = await collect_ohlcv(adapter, client, start_ms, end_ms)

            if _shutdown_requested:
                logger.warning("shutdown_after_ohlcv")
                return

            # Funding + OI can run concurrently (different endpoints)
            funding_rows, oi_rows = await asyncio.gather(
                collect_funding(adapter, client, start_ms, end_ms),
                collect_open_interest(adapter, client, start_ms, end_ms),
            )

        if not _shutdown_requested:
            # yfinance runs in executor — won't block the event loop
            cm_rows = await collect_cross_market(
                adapter,
                start_dt.strftime("%Y-%m-%d"),
                end_dt.strftime("%Y-%m-%d"),
            )
        else:
            cm_rows = 0

        elapsed = time.monotonic() - t0
        logger.info(
            "collection_finished",
            elapsed_min=round(elapsed / 60, 1),
            ohlcv_rows=ohlcv_rows,
            funding_rows=funding_rows,
            oi_rows=oi_rows,
            cross_market_rows=cm_rows,
            shutdown_requested=_shutdown_requested,
        )

        log_summary(adapter)

    finally:
        adapter.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill historical BTC data for ep2-crypto",
    )
    parser.add_argument(
        "--start",
        default=DEFAULT_START,
        help=f"Start date YYYY-MM-DD (default: {DEFAULT_START})",
    )
    parser.add_argument(
        "--end",
        default="today",
        help="End date YYYY-MM-DD or 'today' (default: today)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last stored timestamp in DB (queries MAX(timestamp_ms) from ohlcv)",
    )
    args = parser.parse_args()

    logger.info("starting_collect_history", start=args.start, end=args.end, resume=args.resume)
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
