"""Backfill historical BTC data from Binance Futures, Bybit, and yfinance.

Data sources (all free, no API keys):
  1. Binance Futures REST: 1m OHLCV klines for BTCUSDT perpetual
  2. Binance Futures REST: Funding rate history (every 8h)
  3. Bybit REST: Open interest history (5min intervals)
  4. yfinance: NQ, Gold, DXY, ETH-USD daily OHLCV

Usage:
  uv run python scripts/collect_history.py --start 2019-09-01 --end today
  uv run python scripts/collect_history.py --start 2024-01-01 --end today
  uv run python scripts/collect_history.py --resume

Environment:
  EP2_DB_URL  SQLite path (default: data/history.db) OR PostgreSQL DSN
              postgresql://user:pass@host:5432/dbname
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import structlog
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so we can import ep2_crypto
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from ep2_crypto.db.connection import DBConnection, _is_postgres_url  # noqa: E402
from ep2_crypto.db.schema import create_postgres_tables, create_tables  # noqa: E402

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SYMBOL = "BTCUSDT"
DEFAULT_START = "2019-09-01"
DEFAULT_DB = str(_PROJECT_ROOT / "data" / "history.db")

BINANCE_FAPI = "https://fapi.binance.com"
BYBIT_API = "https://api.bybit.com"

# Binance: each klines request = 5 weight, limit 2400/min -> ~480 req/min safe
# We'll be conservative: 400 requests/min -> 1 request per 0.15s
BINANCE_DELAY_S = 0.15
# Bybit: 120 requests/min
BYBIT_DELAY_S = 0.55

BATCH_COMMIT_SIZE = 10_000

CROSS_MARKET_SYMBOLS = {
    "NQ=F": "yfinance",
    "GC=F": "yfinance",
    "DX-Y.NYB": "yfinance",
    "ETH-USD": "yfinance",
}


def _parse_date(date_str: str) -> datetime:
    """Parse a date string (YYYY-MM-DD or 'today') to a UTC datetime."""
    if date_str.lower() == "today":
        return datetime.now(tz=UTC).replace(hour=0, minute=0, second=0, microsecond=0)
    return datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=UTC)


def _dt_to_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def _ms_to_dt(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1000, tz=UTC)


def _resolve_db_url(args_db: str | None) -> str:
    """Resolve the database URL from args or environment.

    Returns a PostgreSQL DSN or a SQLite file path string.
    """
    url = args_db or os.environ.get("EP2_DB_URL", DEFAULT_DB)
    if not _is_postgres_url(url):
        # Treat as a file path — ensure parent directory exists
        path = Path(url)
        path.parent.mkdir(parents=True, exist_ok=True)
        return str(path)
    return url


def _get_connection(db_url: str) -> DBConnection:
    """Open database connection, creating schema if needed."""
    if _is_postgres_url(db_url):
        create_postgres_tables(db_url)
    else:
        create_tables(db_url)  # SQLite — creates file + schema
    return DBConnection(db_url)


def _last_timestamp(conn: DBConnection, table: str, symbol: str) -> int | None:
    """Get the last timestamp_ms for a symbol in a table (validated name)."""
    allowed = {"ohlcv", "funding_rate", "open_interest", "cross_market"}
    if table not in allowed:
        msg = f"Unknown table: {table}"
        raise ValueError(msg)
    row = conn.execute(
        f"SELECT MAX(timestamp_ms) FROM {table} WHERE symbol = ?",  # noqa: S608
        (symbol,),
    ).fetchone()
    return row[0] if row and row[0] is not None else None


# ---------------------------------------------------------------------------
# Retry / HTTP helpers
# ---------------------------------------------------------------------------
MAX_RETRIES = 3
BACKOFF_BASE = 2.0


async def _fetch_json(
    client: httpx.AsyncClient,
    url: str,
    params: dict[str, Any],
    *,
    delay: float = 0.15,
) -> Any:
    """Fetch JSON with exponential backoff retry."""
    for attempt in range(MAX_RETRIES):
        try:
            resp = await client.get(url, params=params, timeout=30.0)
            if resp.status_code == 429:
                wait = BACKOFF_BASE ** (attempt + 1)
                logger.warning("rate_limited", url=url, wait_s=wait)
                await asyncio.sleep(wait)
                continue
            resp.raise_for_status()
            await asyncio.sleep(delay)
            return resp.json()
        except (httpx.HTTPStatusError, httpx.ReadTimeout, httpx.ConnectTimeout) as exc:
            wait = BACKOFF_BASE ** (attempt + 1)
            logger.warning(
                "fetch_error",
                url=url,
                attempt=attempt + 1,
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
    conn: DBConnection,
    client: httpx.AsyncClient,
    start_ms: int,
    end_ms: int,
) -> int:
    """Collect 1m OHLCV klines from Binance Futures."""
    url = f"{BINANCE_FAPI}/fapi/v1/klines"
    current_ms = start_ms
    total_rows = 0
    batch: list[tuple[Any, ...]] = []

    # Estimate total 1m bars for progress
    total_bars_est = (end_ms - start_ms) // 60_000
    pbar = tqdm(total=total_bars_est, desc="OHLCV 1m", unit="bars")

    while current_ms < end_ms:
        params = {
            "symbol": SYMBOL,
            "interval": "1m",
            "startTime": current_ms,
            "endTime": end_ms,
            "limit": 1500,
        }
        data = await _fetch_json(client, url, params, delay=BINANCE_DELAY_S)
        if not data:
            # Skip this chunk, advance 1500 minutes
            current_ms += 1500 * 60_000
            pbar.update(1500)
            continue

        if len(data) == 0:
            break

        for k in data:
            # Binance kline: [open_time, open, high, low, close, volume,
            #   close_time, quote_volume, trades_count, ...]
            row = (
                int(k[0]),  # timestamp_ms (open time)
                SYMBOL,  # symbol
                "1m",  # interval
                float(k[1]),  # open
                float(k[2]),  # high
                float(k[3]),  # low
                float(k[4]),  # close
                float(k[5]),  # volume
                float(k[7]),  # quote_volume
                int(k[8]),  # trades_count
            )
            batch.append(row)

        pbar.update(len(data))

        # Advance past last candle
        current_ms = int(data[-1][0]) + 60_000

        # Checkpoint
        if len(batch) >= BATCH_COMMIT_SIZE:
            conn.executemany(
                "INSERT OR REPLACE INTO ohlcv VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                batch,
            )
            conn.commit()
            total_rows += len(batch)
            batch = []

    # Final flush
    if batch:
        conn.executemany(
            "INSERT OR REPLACE INTO ohlcv VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            batch,
        )
        conn.commit()
        total_rows += len(batch)

    pbar.close()
    logger.info("ohlcv_complete", total_rows=total_rows)
    return total_rows


# ---------------------------------------------------------------------------
# 2. Binance Futures: Funding rate history
# ---------------------------------------------------------------------------
async def collect_funding(
    conn: DBConnection,
    client: httpx.AsyncClient,
    start_ms: int,
    end_ms: int,
) -> int:
    """Collect funding rate history from Binance Futures."""
    url = f"{BINANCE_FAPI}/fapi/v1/fundingRate"
    current_ms = start_ms
    total_rows = 0
    batch: list[tuple[Any, ...]] = []

    # ~3 per day, estimate total
    total_est = max(1, (end_ms - start_ms) // (8 * 3600 * 1000))
    pbar = tqdm(total=total_est, desc="Funding rate", unit="rows")

    while current_ms < end_ms:
        params = {
            "symbol": SYMBOL,
            "startTime": current_ms,
            "endTime": end_ms,
            "limit": 1000,
        }
        data = await _fetch_json(client, url, params, delay=BINANCE_DELAY_S)
        if not data:
            # Advance ~1000 funding periods (8h each)
            current_ms += 1000 * 8 * 3600 * 1000
            continue

        if len(data) == 0:
            break

        for item in data:
            row = (
                int(item["fundingTime"]),
                SYMBOL,
                float(item["fundingRate"]),
                float(item.get("markPrice", 0)) if item.get("markPrice") else None,
                None,  # index_price not in this endpoint
            )
            batch.append(row)

        pbar.update(len(data))
        current_ms = int(data[-1]["fundingTime"]) + 1

        if len(batch) >= BATCH_COMMIT_SIZE:
            conn.executemany(
                "INSERT OR REPLACE INTO funding_rate VALUES (?, ?, ?, ?, ?)",
                batch,
            )
            conn.commit()
            total_rows += len(batch)
            batch = []

        # Fewer requests needed, no need to be aggressive
        if len(data) < 1000:
            break

    if batch:
        conn.executemany(
            "INSERT OR REPLACE INTO funding_rate VALUES (?, ?, ?, ?, ?)",
            batch,
        )
        conn.commit()
        total_rows += len(batch)

    pbar.close()
    logger.info("funding_complete", total_rows=total_rows)
    return total_rows


# ---------------------------------------------------------------------------
# 3. Bybit REST: Open interest history
# ---------------------------------------------------------------------------
async def collect_open_interest(
    conn: DBConnection,
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

    while current_ms < end_ms:
        params = {
            "category": "linear",
            "symbol": SYMBOL,
            "intervalTime": "5min",
            "startTime": current_ms,
            "endTime": min(current_ms + 200 * 5 * 60_000, end_ms),
            "limit": 200,
        }
        data = await _fetch_json(client, url, params, delay=BYBIT_DELAY_S)
        if not data or data.get("retCode") != 0:
            err_msg = data.get("retMsg", "unknown") if data else "no response"
            logger.warning("bybit_error", msg=err_msg, start=current_ms)
            current_ms += 200 * 5 * 60_000
            pbar.update(200)
            continue

        result_list = data.get("result", {}).get("list", [])
        if not result_list:
            # Bybit may not have data for older dates, advance
            current_ms += 200 * 5 * 60_000
            pbar.update(200)
            continue

        for item in result_list:
            ts = int(item["timestamp"])
            oi = float(item["openInterest"])
            batch.append((ts, SYMBOL, oi, None))

        pbar.update(len(result_list))

        # Bybit returns newest first, so find the newest timestamp to advance
        timestamps = [int(item["timestamp"]) for item in result_list]
        newest = max(timestamps)

        # Advance past the newest point returned
        current_ms = newest + 5 * 60_000

        if len(batch) >= BATCH_COMMIT_SIZE:
            conn.executemany(
                "INSERT OR REPLACE INTO open_interest VALUES (?, ?, ?, ?)",
                batch,
            )
            conn.commit()
            total_rows += len(batch)
            batch = []

        if len(result_list) < 200:
            # Reached current time
            break

    if batch:
        conn.executemany(
            "INSERT OR REPLACE INTO open_interest VALUES (?, ?, ?, ?)",
            batch,
        )
        conn.commit()
        total_rows += len(batch)

    pbar.close()
    logger.info("open_interest_complete", total_rows=total_rows)
    return total_rows


# ---------------------------------------------------------------------------
# 4. yfinance: Cross-market daily data
# ---------------------------------------------------------------------------
def collect_cross_market(
    conn: DBConnection,
    start_date: str,
    end_date: str,
) -> int:
    """Collect cross-market daily data via yfinance (synchronous)."""
    try:
        import yfinance as yf
    except ImportError:
        logger.error("yfinance_not_installed", hint="uv sync --extra ingest")
        return 0

    total_rows = 0

    for symbol, source in tqdm(CROSS_MARKET_SYMBOLS.items(), desc="Cross-market", unit="sym"):
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
                # Convert index to ms timestamp
                ts_ms = int(idx.timestamp() * 1000)  # type: ignore[union-attr]
                # Store close price in cross_market table
                batch.append((ts_ms, symbol, float(close_val), source))

            if batch:
                conn.executemany(
                    "INSERT OR REPLACE INTO cross_market VALUES (?, ?, ?, ?)",
                    batch,
                )
                conn.commit()
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


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def print_summary(conn: DBConnection) -> None:
    """Log a summary of all collected data."""
    tables = {
        "ohlcv": SYMBOL,
        "funding_rate": SYMBOL,
        "open_interest": SYMBOL,
    }

    logger.info("=" * 60)
    logger.info("COLLECTION SUMMARY")
    logger.info("=" * 60)

    for table, symbol in tables.items():
        count = conn.execute(
            f"SELECT COUNT(*) FROM {table} WHERE symbol = ?",  # noqa: S608
            (symbol,),
        ).fetchone()[0]
        min_max = conn.execute(
            f"SELECT MIN(timestamp_ms), MAX(timestamp_ms) FROM {table} WHERE symbol = ?",  # noqa: S608
            (symbol,),
        ).fetchone()

        if count > 0 and min_max[0] is not None:
            start_dt = _ms_to_dt(min_max[0]).strftime("%Y-%m-%d")
            end_dt = _ms_to_dt(min_max[1]).strftime("%Y-%m-%d")
            logger.info(
                "table_summary",
                table=table,
                rows=f"{count:,}",
                range=f"{start_dt} to {end_dt}",
            )
        else:
            logger.info("table_summary", table=table, rows=0, range="N/A")

    # Cross-market: per symbol
    cm_rows = conn.execute(
        "SELECT symbol, COUNT(*) FROM cross_market GROUP BY symbol"
    ).fetchall()
    for row in cm_rows:
        logger.info(
            "table_summary",
            table="cross_market",
            symbol=row[0],
            rows=f"{row[1]:,}",
        )

    total_count = sum(
        conn.execute(
            f"SELECT COUNT(*) FROM {t}"  # noqa: S608
        ).fetchone()[0]
        for t in ["ohlcv", "funding_rate", "open_interest", "cross_market"]
    )
    logger.info("total_rows", total=f"{total_count:,}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def async_main(args: argparse.Namespace) -> None:
    """Run all collectors."""
    db_url = _resolve_db_url(args.db)
    conn = _get_connection(db_url)
    logger.info("database_ready", url=db_url if not _is_postgres_url(db_url) else "postgresql://***")

    # Determine date range
    if args.resume:
        # Find the latest timestamp across key tables
        last_ohlcv = _last_timestamp(conn, "ohlcv", SYMBOL)
        if last_ohlcv is not None:
            start_ms = last_ohlcv + 60_000  # next minute after last stored
            logger.info("resuming_from", timestamp=_ms_to_dt(start_ms).isoformat())
        else:
            start_ms = _dt_to_ms(_parse_date(DEFAULT_START))
            logger.info("no_existing_data_starting_fresh", start=DEFAULT_START)
    else:
        start_ms = _dt_to_ms(_parse_date(args.start))

    end_ms = _dt_to_ms(_parse_date(args.end))
    start_dt = _ms_to_dt(start_ms)
    end_dt = _ms_to_dt(end_ms)

    logger.info(
        "collection_range",
        start=start_dt.strftime("%Y-%m-%d"),
        end=end_dt.strftime("%Y-%m-%d"),
        days=(end_ms - start_ms) / (86_400_000),
    )

    t0 = time.monotonic()

    # Async collectors (Binance + Bybit)
    async with httpx.AsyncClient(
        headers={"User-Agent": "ep2-crypto/0.1"},
        follow_redirects=True,
    ) as client:
        # Run OHLCV first (largest dataset), then funding + OI in parallel
        ohlcv_rows = await collect_ohlcv(conn, client, start_ms, end_ms)

        funding_rows, oi_rows = await asyncio.gather(
            collect_funding(conn, client, start_ms, end_ms),
            collect_open_interest(conn, client, start_ms, end_ms),
        )

    # yfinance is synchronous
    cm_rows = collect_cross_market(
        conn,
        start_dt.strftime("%Y-%m-%d"),
        end_dt.strftime("%Y-%m-%d"),
    )

    elapsed = time.monotonic() - t0
    logger.info(
        "collection_finished",
        elapsed_min=f"{elapsed / 60:.1f}",
        ohlcv=ohlcv_rows,
        funding=funding_rows,
        open_interest=oi_rows,
        cross_market=cm_rows,
    )

    print_summary(conn)
    conn.close()


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
        help="Resume from last stored timestamp in DB",
    )
    parser.add_argument(
        "--db",
        default=None,
        help=f"Override DB path/URL (default: {DEFAULT_DB} or EP2_DB_URL env)",
    )
    args = parser.parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
