"""Full training pipeline: data → features → walk-forward → ensemble → calibration → save.

Usage:
    uv run python scripts/train.py                     # Full training on all data
    uv run python scripts/train.py --days 60           # Last 60 days only
    uv run python scripts/train.py --output /app/models/  # Custom output directory
    uv run python scripts/train.py --skip-gru          # Skip GRU (faster, no GPU needed)

Environment variables (from DatabaseConfig in config.py):
    EP2_DB_BACKEND           sqlite | timescaledb  (default: sqlite)
    EP2_DB_SQLITE_PATH       path to SQLite file   (default: data/ep2_crypto.db)
    EP2_DB_TIMESCALEDB_URL   PostgreSQL DSN         (required when backend=timescaledb)

Model output:
    EP2_MODEL_DIR            override default output dir (default: /app/models in Docker,
                             models/ in local dev). CLI --output takes precedence.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Union

import numpy as np
import structlog

if TYPE_CHECKING:
    from numpy.typing import NDArray

from ep2_crypto.monitoring.alerts import Alert, AlertManager, AlertTier, RateLimiter, TelegramSender

logger = structlog.get_logger(__name__)

# Bars per day for 5-min candles in 24/7 crypto
BARS_PER_DAY = 288

# Default model output directory: prefer /app/models/ (Docker volume) in production,
# fall back to models/ for local dev.  CLI --output overrides both.
DEFAULT_MODEL_DIR = os.environ.get("EP2_MODEL_DIR", "/app/models" if os.path.isdir("/app") else "models")


# ---------------------------------------------------------------------------
# Database connection
# ---------------------------------------------------------------------------

def _build_alert_manager() -> AlertManager:
    """Build an AlertManager from env vars (gracefully disabled if not configured).

    Uses unlimited rate limits — this is a training job with infrequent sends.
    EP2_TELEGRAM_BOT_TOKEN and EP2_TELEGRAM_CHAT_ID must both be set to enable.
    """
    bot_token = os.environ.get("EP2_TELEGRAM_BOT_TOKEN", "")
    chat_id = os.environ.get("EP2_TELEGRAM_CHAT_ID", "")
    sender = TelegramSender(bot_token=bot_token, chat_id=chat_id)
    unlimited = RateLimiter(
        limits={t: 0 for t in AlertTier}  # 0 = unlimited for all tiers
    )
    mgr = AlertManager(senders=[sender], rate_limiter=unlimited)
    if sender.enabled:
        logger.info("telegram_alerts_enabled")
    else:
        logger.info("telegram_alerts_disabled", hint="Set EP2_TELEGRAM_BOT_TOKEN + EP2_TELEGRAM_CHAT_ID to enable")
    return mgr


def get_db_connection() -> Any:
    """Connect to the database using DatabaseConfig env-var settings.

    Reads EP2_DB_BACKEND to choose the driver:
      - "sqlite"      → sqlite3, path from EP2_DB_SQLITE_PATH
      - "timescaledb" → psycopg2, DSN from EP2_DB_TIMESCALEDB_URL

    Returns a connection object. For PostgreSQL the connection is NOT
    autocommit so callers must use explicit transactions or call commit().
    The type is annotated as Any to accommodate both sqlite3.Connection and
    psycopg2.connection without a hard psycopg2 import at the module level.
    """
    backend = os.environ.get("EP2_DB_BACKEND", "sqlite").lower()

    if backend == "timescaledb":
        dsn = os.environ.get("EP2_DB_TIMESCALEDB_URL", "")
        if not dsn:
            msg = (
                "EP2_DB_TIMESCALEDB_URL is not set. "
                "Set it to a valid PostgreSQL DSN, e.g. "
                "postgresql://user:pass@host:5432/dbname"
            )
            raise RuntimeError(msg)
        try:
            import psycopg2
            import psycopg2.extras  # noqa: F401 — registers DictCursor etc.
        except ImportError as exc:
            logger.error(
                "psycopg2_not_installed",
                hint="uv add psycopg2-binary  or  pip install psycopg2-binary",
            )
            raise RuntimeError("psycopg2 is required for TimescaleDB backend") from exc

        conn = psycopg2.connect(dsn)
        conn.autocommit = False
        logger.info("db_connected", backend="timescaledb")
        return conn

    # Default: SQLite
    import sqlite3

    sqlite_path = os.environ.get("EP2_DB_SQLITE_PATH", "data/ep2_crypto.db")
    conn = sqlite3.connect(sqlite_path)
    conn.row_factory = sqlite3.Row
    logger.info("db_connected", backend="sqlite", path=sqlite_path)
    return conn


def _is_postgres(conn: Any) -> bool:
    """Return True if *conn* is a psycopg2 (PostgreSQL) connection."""
    try:
        import psycopg2  # noqa: PLC0415
        return isinstance(conn, psycopg2.extensions.connection)
    except ImportError:
        return False


def _adapt_placeholders(query: str, conn: Any) -> str:
    """Convert %s placeholders (Postgres style) to ? (SQLite style) if needed."""
    if _is_postgres(conn):
        return query  # psycopg2 already uses %s
    return query.replace("%s", "?")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_ohlcv(
    conn: Any,
    days: int | None = None,
) -> dict[str, NDArray[np.float64]]:
    """Load OHLCV data from database.

    Queries 1-minute bars (stored as interval='1m') which are later
    aggregated to 5-minute bars by aggregate_to_5min().

    Returns dict with: timestamps_ms, opens, highs, lows, closes, volumes
    """
    if days:
        end_ms = int(time.time() * 1000)
        start_ms = end_ms - (days * 24 * 60 * 60 * 1000)
        query = _adapt_placeholders(
            "SELECT timestamp_ms, open, high, low, close, volume FROM ohlcv "
            "WHERE symbol = %s AND interval = %s AND timestamp_ms >= %s AND timestamp_ms < %s "
            "ORDER BY timestamp_ms",
            conn,
        )
        cur = conn.cursor()
        cur.execute(query, ("BTCUSDT", "1m", start_ms, end_ms))
    else:
        query = _adapt_placeholders(
            "SELECT timestamp_ms, open, high, low, close, volume FROM ohlcv "
            "WHERE symbol = %s AND interval = %s ORDER BY timestamp_ms",
            conn,
        )
        cur = conn.cursor()
        cur.execute(query, ("BTCUSDT", "1m"))

    rows = cur.fetchall()
    if not rows:
        raise ValueError(
            "No OHLCV data found in database. "
            "Run scripts/collect_history.py first to backfill historical data."
        )

    n = len(rows)
    timestamps = np.empty(n, dtype=np.int64)
    opens = np.empty(n, dtype=np.float64)
    highs = np.empty(n, dtype=np.float64)
    lows = np.empty(n, dtype=np.float64)
    closes = np.empty(n, dtype=np.float64)
    volumes = np.empty(n, dtype=np.float64)

    for i, row in enumerate(rows):
        timestamps[i] = row[0]
        opens[i] = row[1]
        highs[i] = row[2]
        lows[i] = row[3]
        closes[i] = row[4]
        volumes[i] = row[5]

    logger.info(
        "ohlcv_loaded",
        rows=n,
        days=round(n / BARS_PER_DAY / 5, 1),  # 5 1m bars per 5-min bar
        start_ms=int(timestamps[0]),
        end_ms=int(timestamps[-1]),
    )
    return {
        "timestamps_ms": timestamps,
        "opens": opens,
        "highs": highs,
        "lows": lows,
        "closes": closes,
        "volumes": volumes,
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_to_5min(data: dict[str, NDArray[Any]]) -> dict[str, NDArray[Any]]:
    """Aggregate 1-minute bars to 5-minute bars."""
    n = len(data["closes"])
    n5 = (n // 5) * 5  # Truncate to multiple of 5

    timestamps = data["timestamps_ms"][:n5].reshape(-1, 5)[:, 0]
    opens = data["opens"][:n5].reshape(-1, 5)[:, 0]
    highs = data["highs"][:n5].reshape(-1, 5).max(axis=1)
    lows = data["lows"][:n5].reshape(-1, 5).min(axis=1)
    closes = data["closes"][:n5].reshape(-1, 5)[:, -1]
    volumes = data["volumes"][:n5].reshape(-1, 5).sum(axis=1)

    logger.info("aggregated_to_5min", bars_1m=n, bars_5m=len(closes))
    return {
        "timestamps_ms": timestamps,
        "opens": opens,
        "highs": highs,
        "lows": lows,
        "closes": closes,
        "volumes": volumes,
    }


# ---------------------------------------------------------------------------
# Feature & label computation
# ---------------------------------------------------------------------------

def load_polymarket(
    conn: Any,
    timestamps_ms: "NDArray[np.int64]",
) -> dict[str, "NDArray[np.float64]"]:
    """Load Polymarket 5-min market data aligned to OHLCV bar timestamps.

    Queries ``polymarket_5m_history`` and aligns yes_close_price and volume to
    the OHLCV timestamp array. Missing windows are filled with NaN.

    Returns dict with keys: poly_yes_prices, poly_volumes, poly_resolved.
    Returns empty dict (all NaN arrays) if the table doesn't exist or has no data.
    """
    n = len(timestamps_ms)
    nan_arrays: dict[str, NDArray[np.float64]] = {
        "poly_yes_prices": np.full(n, np.nan, dtype=np.float64),
        "poly_volumes": np.full(n, np.nan, dtype=np.float64),
        "poly_resolved": np.full(n, np.nan, dtype=np.float64),
    }

    is_pg = _is_postgres(conn)
    try:
        cur = conn.cursor()
        if is_pg:
            cur.execute(
                "SELECT extract(epoch FROM window_ts)::bigint * 1000 AS ts_ms, "
                "yes_close_price, volume, resolved "
                "FROM polymarket_5m_history ORDER BY window_ts"
            )
        else:
            cur.execute(
                "SELECT window_ts * 1000 AS ts_ms, yes_close_price, volume, resolved "
                "FROM polymarket_5m_history ORDER BY window_ts"
            )
        rows = cur.fetchall()
    except Exception as exc:
        logger.info("polymarket_table_not_found", reason=str(exc))
        return nan_arrays

    if not rows:
        logger.info("polymarket_no_data")
        return nan_arrays

    # Build sorted list of (ts_ms, yes_price, volume, resolved) for binary-search alignment.
    # Data may be a mix of 5-min and hourly granularity (5-min from Gamma API,
    # hourly from HuggingFace). Forward-fill ensures hourly data covers all
    # 5-min bars within the hour.
    poly_sorted: list[tuple[int, float, float, float]] = []
    for row in rows:
        ts_ms = int(row[0])
        bar_ms = (ts_ms // 300_000) * 300_000
        yes_p = float(row[1]) if row[1] is not None else float("nan")
        vol = float(row[2]) if row[2] is not None else float("nan")
        resolved = 1.0 if row[3] else 0.0
        poly_sorted.append((bar_ms, yes_p, vol, resolved))
    poly_sorted.sort(key=lambda x: x[0])

    poly_ts_arr = np.array([p[0] for p in poly_sorted], dtype=np.int64)

    # Align to OHLCV timestamps using searchsorted for O(log n) per bar.
    # For each OHLCV bar, take the most recent Polymarket entry that ended
    # BEFORE this bar (lag-1 in feature computation handles bar T → T+1 pred).
    yes_prices = np.full(n, np.nan, dtype=np.float64)
    volumes = np.full(n, np.nan, dtype=np.float64)
    resolved_arr = np.full(n, np.nan, dtype=np.float64)

    matched = 0
    for i, ts_ms in enumerate(timestamps_ms):
        bar_ms = int((ts_ms // 300_000) * 300_000)
        # Find the rightmost poly entry whose ts <= bar_ms
        idx = int(np.searchsorted(poly_ts_arr, bar_ms, side="right")) - 1
        if idx >= 0:
            _, yes_p, vol, res = poly_sorted[idx]
            if np.isfinite(yes_p):
                yes_prices[i], volumes[i], resolved_arr[i] = yes_p, vol, res
                matched += 1

    logger.info(
        "polymarket_loaded",
        total_poly_rows=len(rows),
        matched_bars=matched,
        coverage_pct=round(matched / max(n, 1) * 100, 1),
    )
    return {
        "poly_yes_prices": yes_prices,
        "poly_volumes": volumes,
        "poly_resolved": resolved_arr,
    }


def compute_features(
    data: dict[str, NDArray[Any]],
    conn: Any | None = None,
) -> tuple[NDArray[np.float64], list[str]]:
    """Compute feature matrix from OHLCV data.

    Returns (X, feature_names) where X is (n_bars, n_features).
    """
    from ep2_crypto.features.pipeline import FeaturePipeline

    pipeline = FeaturePipeline()
    warmup = pipeline.warmup_bars

    n = len(data["closes"])
    logger.info("computing_features", n_bars=n, warmup=warmup, n_features=pipeline.n_features)

    # Load optional Polymarket data (gracefully absent before Feb 2025)
    poly_kwargs: dict[str, NDArray[np.float64]] = {}
    if conn is not None:
        poly_kwargs = load_polymarket(conn, data["timestamps_ms"])

    X = pipeline.compute_batch(
        timestamps=data["timestamps_ms"],
        opens=data["opens"],
        highs=data["highs"],
        lows=data["lows"],
        closes=data["closes"],
        volumes=data["volumes"],
        **poly_kwargs,
    )

    feature_names = pipeline.output_names
    logger.info("features_computed", shape=X.shape, n_feature_names=len(feature_names))
    return X, feature_names


def compute_labels(
    data: dict[str, NDArray[Any]],
) -> tuple[NDArray[np.int8], NDArray[np.float64]]:
    """Compute ternary labels using triple barrier method.

    Returns (labels, returns) where labels are -1/0/+1.
    """
    from ep2_crypto.models.labeling import BarrierConfig, label_triple_barrier

    config = BarrierConfig()
    labels, returns, _ = label_triple_barrier(
        closes=data["closes"],
        highs=data["highs"],
        lows=data["lows"],
        config=config,
    )

    n_up = int((labels == 1).sum())
    n_flat = int((labels == 0).sum())
    n_down = int((labels == -1).sum())
    total = max(len(labels), 1)
    logger.info(
        "labels_computed",
        total=len(labels),
        up=n_up,
        flat=n_flat,
        down=n_down,
        up_pct=round(n_up / total * 100, 1),
    )
    return labels, returns


# ---------------------------------------------------------------------------
# Walk-forward training
# ---------------------------------------------------------------------------

def train_walk_forward(
    X: NDArray[np.float64],
    y: NDArray[np.int8],
    feature_names: list[str],
    output_dir: Path,
    skip_gru: bool = False,
    alert_manager: AlertManager | None = None,
) -> dict[str, Any]:
    """Run walk-forward training of all models.

    Returns dict with final models and OOF predictions.
    The existing bug fixes are preserved:
      - Zero-weight dummy class injection for LightGBM and CatBoost
        (prevents crash when FLAT class is absent from a training window)
      - LightGBM warm-start class conflict fix
      - Stacking API fix (base_model_names kwarg)
    """
    from ep2_crypto.backtest.walk_forward import WalkForwardConfig, WalkForwardValidator
    from ep2_crypto.models.calibration import IsotonicCalibrator
    from ep2_crypto.models.catboost_direction import CatBoostDirectionModel
    from ep2_crypto.models.lgbm_direction import LGBMDirectionModel
    from ep2_crypto.models.stacking import StackingEnsemble

    n_bars = len(y)
    wf_config = WalkForwardConfig()
    validator = WalkForwardValidator(n_bars=n_bars, config=wf_config)
    folds = validator.folds()

    logger.info("walk_forward_start", n_folds=len(folds), n_bars=n_bars)

    if alert_manager is not None:
        alert_manager.send(Alert(
            tier=AlertTier.INFO,
            title="Training started",
            message=f"Walk-forward: {len(folds)} folds, {n_bars:,} bars",
        ))

    # Collect OOF predictions for stacking
    oof_lgbm = np.zeros((n_bars, 3), dtype=np.float64)
    oof_catboost = np.zeros((n_bars, 3), dtype=np.float64)
    oof_mask = np.zeros(n_bars, dtype=bool)

    # Track models from last fold for saving
    last_lgbm: Union[LGBMDirectionModel, None] = None
    last_catboost: Union[CatBoostDirectionModel, None] = None
    prev_lgbm_model: Union[LGBMDirectionModel, None] = None

    fold_sharpes: list[float] = []

    for fold in folds:
        fold_start = time.time()
        train_idx = slice(fold.train_start, fold.train_end)
        test_idx = slice(fold.test_start, fold.test_end)

        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        # Split train into train/val (last 20% of train for early stopping)
        val_size = max(1, int(len(y_train) * 0.2))
        X_tr, y_tr = X_train[:-val_size], y_train[:-val_size]
        X_val, y_val = X_train[-val_size:], y_train[-val_size:]

        # --- LightGBM ---
        # Bug fix (bc894ae): warm-start is only passed when the previous model
        # exists AND its class set matches the current fold's classes.
        lgbm = LGBMDirectionModel()
        lgbm.train(X_tr, y_tr, X_val, y_val, feature_names=feature_names, init_model=prev_lgbm_model)
        lgbm_proba = lgbm.predict_proba(X_test)
        oof_lgbm[test_idx] = lgbm_proba
        prev_lgbm_model = lgbm
        last_lgbm = lgbm

        # --- CatBoost ---
        # Bug fix (ad4a5bf): CatBoostDirectionModel injects zero-weight dummy
        # rows for any missing class to prevent a crash when FLAT is absent.
        catboost = CatBoostDirectionModel()
        catboost.train(X_tr, y_tr, X_val, y_val)
        catboost_proba = catboost.predict_proba(X_test)
        oof_catboost[test_idx] = catboost_proba
        last_catboost = catboost

        oof_mask[test_idx] = True

        # Fold metrics
        lgbm_preds = lgbm.predict(X_test)
        accuracy = float((lgbm_preds == y_test).mean())
        fold_time = time.time() - fold_start

        # Sharpe approximation — use sqrt(105_120) per ADR-002, never sqrt(252)
        correct = (lgbm_preds == y_test).astype(float)
        returns_sim = np.where(correct, 0.001, -0.001)
        fold_sharpe = float(
            returns_sim.mean() / max(float(returns_sim.std()), 1e-10) * np.sqrt(105_120)
        )
        fold_sharpes.append(fold_sharpe)

        logger.info(
            "fold_complete",
            fold=fold.fold_idx,
            accuracy=round(accuracy, 4),
            sharpe=round(fold_sharpe, 2),
            train_size=fold.train_size,
            test_size=fold.test_size,
            time_s=round(fold_time, 1),
        )

        # Telegram notification every 50 folds (and on the final fold)
        if alert_manager is not None:
            is_final = fold.fold_idx == folds[-1].fold_idx
            if fold.fold_idx % 50 == 0 or is_final:
                mean_so_far = float(np.mean(fold_sharpes))
                alert_manager.send(Alert(
                    tier=AlertTier.INFO,
                    title=f"Fold {fold.fold_idx}/{len(folds)} complete",
                    message=(
                        f"acc={accuracy:.4f}  sharpe={fold_sharpe:.2f}\n"
                        f"mean_sharpe_so_far={mean_so_far:.2f}  ({fold.fold_idx}/{len(folds)} folds)"
                    ),
                ))

    # --- Stacking Ensemble ---
    logger.info("training_stacking_ensemble")
    mask = oof_mask
    # Pass a list of per-model OOF arrays (not pre-hstacked) so stacking.train()
    # can call np.hstack() internally and correctly set _n_base_models=2.
    base_probas_list = [oof_lgbm[mask], oof_catboost[mask]]
    y_oof = y[mask]

    stacking = StackingEnsemble()
    stacking_metrics = stacking.train(base_probas_list, y_oof, base_model_names=["lgbm", "catboost"])
    logger.info("stacking_trained", metrics=stacking_metrics)

    if alert_manager is not None:
        alert_manager.send(Alert(
            tier=AlertTier.INFO,
            title="Stacking ensemble trained",
            message=f"meta_train_acc={stacking_metrics.get('meta_train_accuracy', 0):.4f}  n_base_models={int(stacking_metrics.get('n_base_models', 0))}",
        ))

    # --- Calibration ---
    logger.info("training_calibrator")
    stacking_probas = stacking.predict_proba(base_probas_list)
    calibrator = IsotonicCalibrator()
    cal_metrics = calibrator.fit(stacking_probas, y_oof)
    logger.info("calibrator_trained", metrics=cal_metrics)

    # --- Save models to output_dir ---
    output_dir.mkdir(parents=True, exist_ok=True)

    if last_lgbm:
        lgbm_path = output_dir / "lgbm_direction.bin"
        last_lgbm.save(lgbm_path)
        logger.info("model_saved", model="lgbm", path=str(lgbm_path))

    if last_catboost:
        catboost_path = output_dir / "catboost_direction.bin"
        last_catboost.save(catboost_path)
        logger.info("model_saved", model="catboost", path=str(catboost_path))

    stacking_path = output_dir / "stacking_ensemble.pkl"
    calibrator_path = output_dir / "calibrator.pkl"
    stacking.save(stacking_path)
    calibrator.save(calibrator_path)
    logger.info(
        "all_models_saved",
        output_dir=str(output_dir),
        stacking=str(stacking_path),
        calibrator=str(calibrator_path),
    )

    # --- Summary ---
    mean_sharpe = float(np.mean(fold_sharpes))
    std_sharpe = float(np.std(fold_sharpes))
    cv_sharpe = std_sharpe / max(abs(mean_sharpe), 1e-10)

    logger.info(
        "training_summary",
        n_folds=len(folds),
        mean_sharpe=round(mean_sharpe, 2),
        std_sharpe=round(std_sharpe, 2),
        cv_sharpe=round(cv_sharpe, 2),
        stable=cv_sharpe < 0.5,
    )

    if alert_manager is not None:
        alert_manager.send(Alert(
            tier=AlertTier.INFO,
            title="Training complete",
            message=(
                f"n_folds={len(folds)}\n"
                f"mean_sharpe={mean_sharpe:.2f}  std={std_sharpe:.2f}\n"
                f"cv_sharpe={cv_sharpe:.2f}  stable={cv_sharpe < 0.5}"
            ),
        ))

    return {
        "lgbm": last_lgbm,
        "catboost": last_catboost,
        "stacking": stacking,
        "calibrator": calibrator,
        "fold_sharpes": fold_sharpes,
        "mean_sharpe": mean_sharpe,
        "oof_lgbm": oof_lgbm,
        "oof_catboost": oof_catboost,
        "oof_mask": oof_mask,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train ep2-crypto models")
    parser.add_argument(
        "--days",
        type=int,
        default=None,
        help="Use last N days of data (default: all)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_MODEL_DIR,
        help=(
            f"Output directory for trained models "
            f"(default: {DEFAULT_MODEL_DIR}, override via EP2_MODEL_DIR env var)"
        ),
    )
    parser.add_argument("--skip-gru", action="store_true", help="Skip GRU training (faster)")
    args = parser.parse_args()

    output_dir = Path(args.output)
    logger.info(
        "training_pipeline_start",
        days=args.days,
        output=str(output_dir),
        backend=os.environ.get("EP2_DB_BACKEND", "sqlite"),
    )

    # 1. Load data
    conn = get_db_connection()
    try:
        raw_data = load_ohlcv(conn, days=args.days)

        # 2. Aggregate to 5-min bars
        data = aggregate_to_5min(raw_data)

        # 3. Compute features (conn kept open for Polymarket data load)
        X, feature_names = compute_features(data, conn=conn)
    finally:
        conn.close()

    # 4. Compute labels
    y, _returns = compute_labels(data)

    # 5. Align X and y (features may have NaN warmup at start)
    # Find first row with no NaN
    valid_start = 0
    for i in range(len(X)):
        if not np.any(np.isnan(X[i])):
            valid_start = i
            break

    # Labels may be NaN at the end (triple barrier can't resolve last bars)
    valid_end = len(y)
    for i in range(len(y) - 1, -1, -1):
        if y[i] != 0 or i < len(y) - 50:  # Allow some flat labels
            valid_end = i + 1
            break

    # Use the intersection
    start = valid_start
    end = min(valid_end, len(X), len(y))
    X = X[start:end]
    y = y[start:end]

    # Replace any remaining NaN with 0 (tree models handle this well)
    nan_count = int(np.isnan(X).sum())
    if nan_count > 0:
        logger.warning("filling_nans", count=nan_count)
        X = np.nan_to_num(X, nan=0.0)

    logger.info("data_aligned", start=start, end=end, X_shape=X.shape, y_shape=y.shape)

    # 6. Walk-forward training
    alert_manager = _build_alert_manager()
    pipeline_start = time.time()
    results = train_walk_forward(X, y, feature_names, output_dir, skip_gru=args.skip_gru, alert_manager=alert_manager)
    pipeline_time = time.time() - pipeline_start

    # 7. Final report — use structlog, never print()
    logger.info(
        "training_complete",
        total_time_min=round(pipeline_time / 60, 1),
        n_bars=X.shape[0],
        n_features=X.shape[1],
        n_folds=len(results["fold_sharpes"]),
        mean_sharpe=round(results["mean_sharpe"], 2),
        models_saved=str(output_dir),
    )


if __name__ == "__main__":
    main()
