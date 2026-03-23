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
# BTC hourly up/down markets (HF dataset has hourly, not 5-min markets)
BTC_UPDOWN_KEYWORDS = ["bitcoin-up-or-down", "btc-up-or-down", "bitcoin up or down", "btc up or down"]

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


def _is_btc_updown_market(row: dict[str, Any]) -> bool:
    """Return True if this is a Bitcoin Up or Down direction market."""
    slug = str(row.get("slug", "") or "").lower()
    question = str(row.get("question", "") or "").lower()
    text = slug + " " + question
    return any(k in text for k in BTC_UPDOWN_KEYWORDS)


def _parse_outcome_prices(outcome_prices_str: str) -> tuple[float | None, float | None]:
    """Parse outcome_prices string like \"['0.48', '0.52']\" into (yes, no) floats."""
    try:
        import ast
        prices = ast.literal_eval(str(outcome_prices_str))
        yes_p = float(prices[0]) if prices else None
        no_p = float(prices[1]) if len(prices) > 1 else None
        return yes_p, no_p
    except Exception:
        return None, None


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

            if not _is_btc_updown_market(record):
                continue

            btc_found += 1

            # Extract the market end time as window_ts
            end_date = record.get("end_date") or record.get("end_date_iso") or record.get("close_time")
            if end_date is None:
                continue

            try:
                from datetime import datetime, UTC
                if isinstance(end_date, (int, float)):
                    window_ts = int(end_date)
                else:
                    dt = datetime.fromisoformat(str(end_date).replace("Z", "+00:00"))
                    window_ts = int(dt.timestamp())
                # Align to nearest 5-min boundary
                window_ts = (window_ts // WINDOW_SECONDS) * WINDOW_SECONDS
            except (ValueError, TypeError):
                continue

            # outcome_prices is a string like "['0.48', '0.52']"
            yes_price, no_price = _parse_outcome_prices(record.get("outcome_prices", ""))

            # Determine outcome from prices: yes_price=1.0 → Up won, yes_price=0.0 → Down won
            outcome: str | None = None
            resolved = str(record.get("closed", "0")) == "1"
            if resolved and yes_price is not None:
                if yes_price >= 0.99:
                    outcome = "up"
                elif yes_price <= 0.01:
                    outcome = "down"

            volume = record.get("volume")
            condition_id = record.get("condition_id", "")
            slug = record.get("slug", f"hf-btc-{window_ts}")

            row: dict[str, Any] = {
                "window_ts": window_ts,
                "slug": slug,
                "outcome": outcome,
                "yes_close_price": yes_price,
                "no_close_price": no_price,
                "volume": float(volume) if volume is not None else None,
                "resolved": resolved,
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
