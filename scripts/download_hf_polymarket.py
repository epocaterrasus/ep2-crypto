"""Download BTC 5-min market data from HuggingFace SII-WANGZJ/Polymarket_data.

Uses streaming mode so we never download the full 107GB dataset.
Filters for BTC up/down 5-min markets only.

Stores results in polymarket_5m_history table (same schema as collect_polymarket_history.py).

Usage:
    uv run python scripts/download_hf_polymarket.py
    uv run python scripts/download_hf_polymarket.py --dry-run
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

import structlog

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

logger = structlog.get_logger(__name__)

# BTC 5-min market name patterns (Polymarket uses various slugs)
BTC_5MIN_KEYWORDS = [
    "btc-up", "btc-down", "bitcoin-up", "bitcoin-down",
    "will-btc", "will-bitcoin", "btc-usd", "bitcoin-usd",
]

WINDOW_SECONDS = 300


def _get_db_conn() -> tuple[Any, bool]:
    """Return (connection, is_postgres)."""
    backend = os.environ.get("EP2_DB_BACKEND", "sqlite").lower()
    if backend == "timescaledb":
        import psycopg2  # type: ignore[import]
        pg_url = os.environ.get(
            "EP2_DB_TIMESCALEDB_URL",
            "postgresql://ep2:ep2_secret@localhost:5432/ep2_crypto",
        )
        conn = psycopg2.connect(pg_url)
        conn.autocommit = False
        return conn, True

    from ep2_crypto.db.schema import create_tables
    db_path = Path(os.environ.get("EP2_DB_URL", "data/history.db"))
    return create_tables(db_path), False


def _ensure_table(conn: Any, is_postgres: bool) -> None:
    """Create polymarket_5m_history if it doesn't exist."""
    if is_postgres:
        ddl = """
        CREATE TABLE IF NOT EXISTS polymarket_5m_history (
            window_ts       TIMESTAMPTZ      PRIMARY KEY,
            slug            TEXT             NOT NULL,
            outcome         TEXT,
            yes_close_price DOUBLE PRECISION,
            no_close_price  DOUBLE PRECISION,
            volume          DOUBLE PRECISION,
            resolved        BOOLEAN          DEFAULT FALSE,
            condition_id    TEXT
        )"""
    else:
        ddl = """
        CREATE TABLE IF NOT EXISTS polymarket_5m_history (
            window_ts       INTEGER PRIMARY KEY,
            slug            TEXT NOT NULL,
            outcome         TEXT,
            yes_close_price REAL,
            no_close_price  REAL,
            volume          REAL,
            resolved        INTEGER DEFAULT 0,
            condition_id    TEXT
        )"""
    cur = conn.cursor()
    cur.execute(ddl)
    conn.commit()


def _is_btc_5min_market(row: dict[str, Any]) -> bool:
    """Return True if this market looks like a BTC 5-min up/down market."""
    slug = str(row.get("slug", "") or row.get("market_slug", "") or "").lower()
    question = str(row.get("question", "") or row.get("title", "") or "").lower()

    # Must mention BTC or bitcoin
    if not any(k in slug or k in question for k in ["btc", "bitcoin"]):
        return False

    # Must be a short-window (5 min) market — slug usually contains time info
    # or market title mentions "5 minutes" / "5-minute"
    text = slug + " " + question
    is_short = any(k in text for k in ["5-min", "5min", "5 min", "300s", "up-or-down"])

    return is_short


def _upsert_row(conn: Any, row: dict[str, Any], is_postgres: bool) -> None:
    """Upsert a single row into polymarket_5m_history."""
    cur = conn.cursor()
    if is_postgres:
        cur.execute(
            """
            INSERT INTO polymarket_5m_history
                (window_ts, slug, outcome, yes_close_price, no_close_price,
                 volume, resolved, condition_id)
            VALUES (to_timestamp(%s), %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (window_ts) DO UPDATE SET
                yes_close_price = EXCLUDED.yes_close_price,
                no_close_price  = EXCLUDED.no_close_price,
                volume          = EXCLUDED.volume,
                resolved        = EXCLUDED.resolved,
                outcome         = EXCLUDED.outcome
            """,
            (
                row["window_ts"],
                row["slug"],
                row.get("outcome"),
                row.get("yes_close_price"),
                row.get("no_close_price"),
                row.get("volume"),
                row.get("resolved", False),
                row.get("condition_id", ""),
            ),
        )
    else:
        cur.execute(
            """
            INSERT OR REPLACE INTO polymarket_5m_history
                (window_ts, slug, outcome, yes_close_price, no_close_price,
                 volume, resolved, condition_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                int(row["window_ts"]),
                row["slug"],
                row.get("outcome"),
                row.get("yes_close_price"),
                row.get("no_close_price"),
                row.get("volume"),
                1 if row.get("resolved") else 0,
                row.get("condition_id", ""),
            ),
        )


def download_btc_markets(dry_run: bool = False) -> dict[str, int]:
    """Stream SII-WANGZJ/Polymarket_data and extract BTC 5-min markets."""
    try:
        from datasets import load_dataset  # type: ignore[import]
    except ImportError:
        logger.error(
            "huggingface_datasets_not_installed",
            hint="uv add datasets",
        )
        raise

    logger.info("loading_hf_dataset_streaming", dataset="SII-WANGZJ/Polymarket_data")

    # Stream all splits; the dataset has 'train' and potentially others
    try:
        ds = load_dataset(
            "SII-WANGZJ/Polymarket_data",
            split="train",
            streaming=True,
            trust_remote_code=True,
        )
    except Exception as exc:
        logger.error("hf_dataset_load_failed", error=str(exc))
        raise

    conn: Any = None
    is_postgres = False
    if not dry_run:
        conn, is_postgres = _get_db_conn()
        _ensure_table(conn, is_postgres)

    scanned = 0
    btc_found = 0
    inserted = 0
    batch: list[dict[str, Any]] = []
    BATCH_SIZE = 500

    def flush_batch() -> int:
        nonlocal inserted
        if dry_run or not batch:
            return 0
        for row in batch:
            _upsert_row(conn, row, is_postgres)
        conn.commit()
        count = len(batch)
        batch.clear()
        inserted += count
        return count

    try:
        for record in ds:
            scanned += 1

            if scanned % 100_000 == 0:
                logger.info(
                    "scanning_progress",
                    scanned=scanned,
                    btc_found=btc_found,
                    inserted=inserted,
                )

            if not _is_btc_5min_market(record):
                continue

            btc_found += 1

            # Extract the close time as the window_ts
            close_time = record.get("end_date_iso") or record.get("close_time") or record.get("end_time")
            if close_time is None:
                continue

            try:
                if isinstance(close_time, (int, float)):
                    window_ts = int(close_time)
                else:
                    from datetime import datetime, UTC
                    dt = datetime.fromisoformat(str(close_time).replace("Z", "+00:00"))
                    window_ts = int(dt.timestamp())
                # Align to 5-min boundary
                window_ts = (window_ts // WINDOW_SECONDS) * WINDOW_SECONDS
            except (ValueError, TypeError):
                continue

            yes_price = record.get("yes_price") or record.get("yes_close_price")
            no_price = record.get("no_price") or record.get("no_close_price")
            volume = record.get("volume") or record.get("volume_usd")
            outcome = record.get("outcome") or record.get("resolution")
            condition_id = record.get("condition_id") or record.get("conditionId", "")
            slug = record.get("slug") or record.get("market_slug", f"hf-btc-{window_ts}")

            row: dict[str, Any] = {
                "window_ts": window_ts,
                "slug": slug,
                "outcome": str(outcome).lower() if outcome else None,
                "yes_close_price": float(yes_price) if yes_price is not None else None,
                "no_close_price": float(no_price) if no_price is not None else None,
                "volume": float(volume) if volume is not None else None,
                "resolved": outcome is not None,
                "condition_id": str(condition_id),
            }

            if dry_run:
                if btc_found <= 5:
                    logger.info("dry_run_sample", **row)
            else:
                batch.append(row)
                if len(batch) >= BATCH_SIZE:
                    flush_batch()

    except KeyboardInterrupt:
        logger.info("download_interrupted")
    finally:
        if batch:
            flush_batch()
        if conn and not dry_run:
            conn.close()

    summary = {
        "scanned": scanned,
        "btc_found": btc_found,
        "inserted": inserted,
    }
    logger.info("download_complete", **summary)
    return summary


def main() -> None:
    import structlog
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.PrintLoggerFactory(),
    )

    parser = argparse.ArgumentParser(description="Download BTC Polymarket data from HuggingFace")
    parser.add_argument("--dry-run", action="store_true", help="Scan only, don't write to DB")
    args = parser.parse_args()

    download_btc_markets(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
