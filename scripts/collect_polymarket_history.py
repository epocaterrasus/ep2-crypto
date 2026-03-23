"""Backfill historical 5-minute BTC Up/Down markets from Polymarket.

Queries the Gamma API (public, no auth) for resolved BTC 5-minute
prediction markets and stores results in SQLite for later joining
with BTC OHLCV data.

Usage:
    uv run python scripts/collect_polymarket_history.py --start 2026-02-01
    uv run python scripts/collect_polymarket_history.py --resume
    uv run python scripts/collect_polymarket_history.py --derive-from-btc --start 2026-01-01
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sqlite3
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aiohttp
import structlog
from tqdm import tqdm

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GAMMA_API_BASE = "https://gamma-api.polymarket.com"
WINDOW_SECONDS = 300  # 5 minutes
MAX_CONCURRENT = 25
# 300 requests per 10 seconds => 30 per second => ~33ms between requests
RATE_LIMIT_PER_10S = 300
CHECKPOINT_INTERVAL = 1000

# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

POLYMARKET_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS polymarket_5m_history (
    window_ts       INTEGER PRIMARY KEY,
    slug            TEXT NOT NULL,
    outcome         TEXT,
    yes_close_price REAL,
    no_close_price  REAL,
    volume          REAL,
    resolved        INTEGER DEFAULT 0,
    condition_id    TEXT
)
"""

POLYMARKET_INDEX_DDL = (
    "CREATE INDEX IF NOT EXISTS idx_poly5m_resolved ON polymarket_5m_history (resolved)"
)


def get_db_path() -> Path:
    """Resolve DB path from EP2_DB_URL env or default."""
    raw = os.environ.get("EP2_DB_URL", "")
    if raw:
        return Path(raw)
    return Path("data/history.db")


def init_db(db_path: Path) -> sqlite3.Connection:
    """Open DB, create polymarket table if needed, return connection."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA busy_timeout = 5000")
    conn.execute(POLYMARKET_TABLE_DDL)
    conn.execute(POLYMARKET_INDEX_DDL)
    conn.commit()
    logger.info("db_initialized", path=str(db_path))
    return conn


def upsert_rows(conn: sqlite3.Connection, rows: list[dict[str, Any]]) -> int:
    """Insert or replace rows into polymarket_5m_history."""
    if not rows:
        return 0
    conn.executemany(
        "INSERT OR REPLACE INTO polymarket_5m_history "
        "(window_ts, slug, outcome, yes_close_price, no_close_price, "
        "volume, resolved, condition_id) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (
                r["window_ts"],
                r["slug"],
                r.get("outcome"),
                r.get("yes_close_price"),
                r.get("no_close_price"),
                r.get("volume"),
                1 if r.get("resolved") else 0,
                r.get("condition_id"),
            )
            for r in rows
        ],
    )
    conn.commit()
    return len(rows)


def get_last_stored_ts(conn: sqlite3.Connection) -> int | None:
    """Return the most recent window_ts in the table, or None."""
    row = conn.execute("SELECT MAX(window_ts) AS max_ts FROM polymarket_5m_history").fetchone()
    if row and row["max_ts"] is not None:
        return int(row["max_ts"])
    return None


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------


class RateLimiter:
    """Token bucket rate limiter: max `rate` requests per `per` seconds."""

    def __init__(self, rate: int, per: float) -> None:
        self._rate = rate
        self._per = per
        self._semaphore = asyncio.Semaphore(rate)
        self._timestamps: list[float] = []
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Wait until a request slot is available."""
        while True:
            async with self._lock:
                now = time.monotonic()
                # Purge timestamps older than the window
                self._timestamps = [t for t in self._timestamps if now - t < self._per]
                if len(self._timestamps) < self._rate:
                    self._timestamps.append(now)
                    return
                # Calculate wait time until oldest timestamp expires
                wait_time = self._per - (now - self._timestamps[0])
            await asyncio.sleep(max(wait_time, 0.01))


# ---------------------------------------------------------------------------
# Gamma API fetcher
# ---------------------------------------------------------------------------


def build_slug(window_ts: int) -> str:
    """Build the deterministic market slug for a 5-min window."""
    aligned = (window_ts // WINDOW_SECONDS) * WINDOW_SECONDS
    return f"btc-updown-5m-{aligned}"


def parse_event_response(data: dict[str, Any], window_ts: int) -> dict[str, Any] | None:
    """Parse a Gamma API event response into a row dict."""
    # The /events endpoint returns an event with nested markets
    markets: list[dict[str, Any]] = data.get("markets", [])
    if not markets:
        return None

    # Take the first (and usually only) market
    market = markets[0]
    tokens: list[dict[str, Any]] = market.get("tokens", [])

    yes_token = next(
        (t for t in tokens if t.get("outcome", "").lower() in ("yes", "up")),
        {},
    )
    no_token = next(
        (t for t in tokens if t.get("outcome", "").lower() in ("no", "down")),
        {},
    )

    # Determine resolved outcome
    resolved = market.get("closed", False) or market.get("resolved", False)
    outcome: str | None = None
    if resolved:
        winner = market.get("winner", "").lower()
        if winner in ("yes", "up"):
            outcome = "up"
        elif winner in ("no", "down"):
            outcome = "down"

    return {
        "window_ts": window_ts,
        "slug": market.get("slug", build_slug(window_ts)),
        "outcome": outcome,
        "yes_close_price": float(yes_token.get("price", 0.0)) if yes_token else None,
        "no_close_price": float(no_token.get("price", 0.0)) if no_token else None,
        "volume": float(market.get("volume", 0.0) or 0.0),
        "resolved": resolved,
        "condition_id": market.get("conditionId", market.get("condition_id", "")),
    }


async def fetch_window(
    session: aiohttp.ClientSession,
    rate_limiter: RateLimiter,
    window_ts: int,
) -> dict[str, Any] | None:
    """Fetch a single 5-min window from the Gamma API.

    Returns parsed row dict or None if no market exists for this window.
    """
    slug = build_slug(window_ts)
    url = f"{GAMMA_API_BASE}/events"
    params = {"slug": slug}

    await rate_limiter.acquire()

    try:
        async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=15)) as resp:
            if resp.status == 404:
                return None
            if resp.status == 429:
                # Rate limited — back off and retry once
                retry_after = float(resp.headers.get("Retry-After", "2"))
                logger.warning("rate_limited", retry_after=retry_after, slug=slug)
                await asyncio.sleep(retry_after)
                await rate_limiter.acquire()
                timeout = aiohttp.ClientTimeout(total=15)
                async with session.get(url, params=params, timeout=timeout) as retry_resp:
                    if retry_resp.status != 200:
                        return None
                    data = await retry_resp.json()
            elif resp.status != 200:
                return None
            else:
                data = await resp.json()

        # Gamma /events returns a list — find matching event
        if isinstance(data, list):
            if not data:
                return None
            # Find the event whose slug matches
            event = next(
                (e for e in data if slug in e.get("slug", "")),
                data[0] if data else None,
            )
            if event is None:
                return None
            return parse_event_response(event, window_ts)
        elif isinstance(data, dict):
            return parse_event_response(data, window_ts)
        return None

    except TimeoutError:
        logger.debug("fetch_timeout", slug=slug)
        return None
    except aiohttp.ClientError as exc:
        logger.debug("fetch_error", slug=slug, error=str(exc))
        return None


# ---------------------------------------------------------------------------
# Main collection loop
# ---------------------------------------------------------------------------


async def collect_polymarket_history(
    start_ts: int,
    end_ts: int,
    conn: sqlite3.Connection,
) -> dict[str, Any]:
    """Iterate backwards from end_ts to start_ts, fetching each 5-min window.

    Returns summary statistics.
    """
    # Generate all window timestamps (descending)
    # Align to 5-min boundaries
    end_aligned = (end_ts // WINDOW_SECONDS) * WINDOW_SECONDS
    start_aligned = (start_ts // WINDOW_SECONDS) * WINDOW_SECONDS

    windows: list[int] = []
    ts = end_aligned
    while ts >= start_aligned:
        windows.append(ts)
        ts -= WINDOW_SECONDS

    total_windows = len(windows)
    logger.info(
        "collection_started",
        total_windows=total_windows,
        start=datetime.fromtimestamp(start_aligned, tz=UTC).isoformat(),
        end=datetime.fromtimestamp(end_aligned, tz=UTC).isoformat(),
    )

    rate_limiter = RateLimiter(rate=RATE_LIMIT_PER_10S, per=10.0)
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    found_rows: list[dict[str, Any]] = []
    pending_buffer: list[dict[str, Any]] = []
    total_found = 0
    total_resolved = 0
    total_up = 0

    pbar = tqdm(total=total_windows, desc="Fetching Polymarket windows", unit="win")

    async with aiohttp.ClientSession() as session:

        async def fetch_with_semaphore(wts: int) -> dict[str, Any] | None:
            async with semaphore:
                return await fetch_window(session, rate_limiter, wts)

        # Process in batches to allow checkpointing
        batch_size = CHECKPOINT_INTERVAL
        for batch_start in range(0, total_windows, batch_size):
            batch = windows[batch_start : batch_start + batch_size]
            tasks = [fetch_with_semaphore(wts) for wts in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                pbar.update(1)
                if isinstance(result, Exception):
                    logger.debug("batch_item_error", error=str(result))
                    continue
                if result is not None:
                    pending_buffer.append(result)
                    total_found += 1
                    if result.get("resolved"):
                        total_resolved += 1
                    if result.get("outcome") == "up":
                        total_up += 1

            # Checkpoint: flush to DB
            if pending_buffer:
                inserted = upsert_rows(conn, pending_buffer)
                logger.info(
                    "checkpoint",
                    inserted=inserted,
                    total_found=total_found,
                    progress=f"{batch_start + len(batch)}/{total_windows}",
                )
                found_rows.extend(pending_buffer)
                pending_buffer = []

    pbar.close()

    # Final flush
    if pending_buffer:
        upsert_rows(conn, pending_buffer)
        found_rows.extend(pending_buffer)

    up_rate = (total_up / total_resolved * 100) if total_resolved > 0 else 0.0

    summary = {
        "total_windows_scanned": total_windows,
        "total_found": total_found,
        "total_resolved": total_resolved,
        "total_up": total_up,
        "up_win_rate_pct": round(up_rate, 2),
        "start_date": datetime.fromtimestamp(start_aligned, tz=UTC).isoformat(),
        "end_date": datetime.fromtimestamp(end_aligned, tz=UTC).isoformat(),
    }

    logger.info("collection_complete", **summary)
    return summary


# ---------------------------------------------------------------------------
# Derive synthetic outcomes from BTC OHLCV
# ---------------------------------------------------------------------------


def derive_from_btc_ohlcv(
    conn: sqlite3.Connection,
    start_ts: int,
    end_ts: int,
) -> dict[str, Any]:
    """Create synthetic Polymarket-like outcomes from ohlcv_1m or ohlcv table.

    For each 5-min window, compare close at window_end vs open at window_start.
    If close > open => "up", else "down".
    """
    start_ms = start_ts * 1000
    end_ms = end_ts * 1000

    # Try to query 1-minute OHLCV data (interval = '1m')
    rows = conn.execute(
        "SELECT timestamp_ms, open, close FROM ohlcv "
        "WHERE symbol = ? AND interval = ? "
        "AND timestamp_ms >= ? AND timestamp_ms < ? "
        "ORDER BY timestamp_ms",
        ("BTC/USDT:USDT", "1m", start_ms, end_ms),
    ).fetchall()

    if not rows:
        # Fallback: try 5-minute bars directly
        rows = conn.execute(
            "SELECT timestamp_ms, open, close FROM ohlcv "
            "WHERE symbol = ? AND interval = ? "
            "AND timestamp_ms >= ? AND timestamp_ms < ? "
            "ORDER BY timestamp_ms",
            ("BTC/USDT:USDT", "5m", start_ms, end_ms),
        ).fetchall()
        if not rows:
            logger.warning(
                "no_ohlcv_data_found",
                start_ms=start_ms,
                end_ms=end_ms,
            )
            return {"derived_count": 0, "error": "No OHLCV data found"}

        # 5m bars: each row IS a window
        derived_rows: list[dict[str, Any]] = []
        for row in rows:
            ts_s = row["timestamp_ms"] // 1000
            window_ts = (ts_s // WINDOW_SECONDS) * WINDOW_SECONDS
            outcome = "up" if row["close"] > row["open"] else "down"
            derived_rows.append(
                {
                    "window_ts": window_ts,
                    "slug": build_slug(window_ts),
                    "outcome": outcome,
                    "yes_close_price": None,
                    "no_close_price": None,
                    "volume": None,
                    "resolved": True,
                    "condition_id": "derived-from-btc-ohlcv",
                }
            )

        inserted = upsert_rows(conn, derived_rows)
        total_up = sum(1 for r in derived_rows if r["outcome"] == "up")
        up_rate = (total_up / len(derived_rows) * 100) if derived_rows else 0.0

        summary = {
            "derived_count": inserted,
            "total_up": total_up,
            "total_down": inserted - total_up,
            "up_rate_pct": round(up_rate, 2),
            "source": "5m_bars",
        }
        logger.info("derive_complete", **summary)
        return summary

    # 1-minute bars: aggregate into 5-min windows
    # Build a dict: window_ts -> list of (timestamp_ms, open, close)
    window_bars: dict[int, list[tuple[int, float, float]]] = {}
    for row in rows:
        ts_s = row["timestamp_ms"] // 1000
        window_ts = (ts_s // WINDOW_SECONDS) * WINDOW_SECONDS
        if window_ts not in window_bars:
            window_bars[window_ts] = []
        window_bars[window_ts].append(
            (row["timestamp_ms"], float(row["open"]), float(row["close"]))
        )

    derived_rows = []
    for window_ts, bars in sorted(window_bars.items()):
        if len(bars) < 3:
            # Skip incomplete windows (need at least 3 of 5 minutes)
            continue
        bars_sorted = sorted(bars, key=lambda x: x[0])
        window_open = bars_sorted[0][1]  # open of first bar
        window_close = bars_sorted[-1][2]  # close of last bar
        outcome = "up" if window_close > window_open else "down"
        derived_rows.append(
            {
                "window_ts": window_ts,
                "slug": build_slug(window_ts),
                "outcome": outcome,
                "yes_close_price": None,
                "no_close_price": None,
                "volume": None,
                "resolved": True,
                "condition_id": "derived-from-btc-ohlcv",
            }
        )

    inserted = upsert_rows(conn, derived_rows)
    total_up = sum(1 for r in derived_rows if r["outcome"] == "up")
    up_rate = (total_up / len(derived_rows) * 100) if derived_rows else 0.0

    summary = {
        "derived_count": inserted,
        "total_up": total_up,
        "total_down": inserted - total_up,
        "up_rate_pct": round(up_rate, 2),
        "source": "1m_bars_aggregated",
    }
    logger.info("derive_complete", **summary)
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill Polymarket 5-minute BTC Up/Down market history",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Start date (YYYY-MM-DD). Default: 2026-02-01",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="End date (YYYY-MM-DD). Default: now",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the last stored window timestamp",
    )
    parser.add_argument(
        "--derive-from-btc",
        action="store_true",
        help="Derive synthetic outcomes from BTC OHLCV data instead of Polymarket API",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=None,
        help="Override database path (default: EP2_DB_URL env or data/history.db)",
    )
    return parser.parse_args()


def main() -> None:
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.PrintLoggerFactory(),
    )

    args = parse_args()

    # Resolve DB path
    db_path = Path(args.db_path) if args.db_path else get_db_path()

    conn = init_db(db_path)

    # Resolve time range
    now_ts = int(time.time())

    if args.end:
        end_dt = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=UTC)
        end_ts = int(end_dt.timestamp())
    else:
        end_ts = now_ts

    if args.resume:
        last_ts = get_last_stored_ts(conn)
        if last_ts is not None:
            # Resume from the window after the last stored one
            start_ts = last_ts + WINDOW_SECONDS
            logger.info(
                "resuming",
                from_ts=last_ts,
                from_date=datetime.fromtimestamp(last_ts, tz=UTC).isoformat(),
            )
        else:
            # No data yet, use default start
            start_dt = datetime.strptime("2026-02-01", "%Y-%m-%d").replace(tzinfo=UTC)
            start_ts = int(start_dt.timestamp())
            logger.info("no_previous_data_found, starting_from_default")
    elif args.start:
        start_dt = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=UTC)
        start_ts = int(start_dt.timestamp())
    else:
        start_dt = datetime.strptime("2026-02-01", "%Y-%m-%d").replace(tzinfo=UTC)
        start_ts = int(start_dt.timestamp())

    if start_ts >= end_ts:
        logger.info("nothing_to_collect", start_ts=start_ts, end_ts=end_ts)
        conn.close()
        return

    try:
        if args.derive_from_btc:
            summary = derive_from_btc_ohlcv(conn, start_ts, end_ts)
        else:
            summary = asyncio.run(collect_polymarket_history(start_ts, end_ts, conn))

        # Print summary to structured log
        logger.info("final_summary", **summary)
    except KeyboardInterrupt:
        logger.info("collection_interrupted_by_user")
    except Exception as exc:
        logger.error("collection_failed", error=str(exc), exc_info=True)
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    main()
