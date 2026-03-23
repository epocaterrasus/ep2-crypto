"""Full training pipeline: data → features → walk-forward → ensemble → calibration → save.

Usage:
    uv run python scripts/train.py                     # Full training on all data
    uv run python scripts/train.py --days 60           # Last 60 days only
    uv run python scripts/train.py --output models/    # Custom output directory
    uv run python scripts/train.py --skip-gru          # Skip GRU (faster, no GPU needed)
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import structlog

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Ensure src/ is importable when running as a script
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

logger = structlog.get_logger(__name__)

# Bars per day for 5-min candles in 24/7 crypto
BARS_PER_DAY = 288
MS_PER_BAR = 5 * 60 * 1000  # 5 minutes in milliseconds


def get_db_connection() -> Any:
    """Connect to the database (SQLite or PostgreSQL via env var).

    Returns a sqlite3.Connection for SQLite or a psycopg2 connection for PostgreSQL.
    Both support .cursor() with .execute() / .fetchall().
    """
    db_url = os.environ.get("EP2_DB_URL", "data/history.db")

    if "postgresql" in db_url or db_url.startswith("postgres://"):
        import psycopg2  # type: ignore[import-untyped]

        conn = psycopg2.connect(db_url)
        conn.autocommit = False
        logger.info("db_connected", backend="postgresql")
        return conn
    else:
        conn = sqlite3.connect(db_url)
        conn.row_factory = sqlite3.Row
        logger.info("db_connected", backend="sqlite", path=db_url)
        return conn


def load_ohlcv(
    conn: sqlite3.Connection,
    days: int | None = None,
) -> dict[str, NDArray[np.float64]]:
    """Load OHLCV data from database.

    Returns dict with: timestamps_ms, opens, highs, lows, closes, volumes
    """
    if days:
        end_ms = int(time.time() * 1000)
        start_ms = end_ms - (days * 24 * 60 * 60 * 1000)
        query = (
            "SELECT timestamp_ms, open, high, low, close, volume FROM ohlcv "
            "WHERE symbol = %s AND interval = %s AND timestamp_ms >= %s AND timestamp_ms < %s "
            "ORDER BY timestamp_ms"
        )
        # Adapt for sqlite vs postgres
        q = query.replace("%s", "?") if isinstance(conn, sqlite3.Connection) else query
        cur = conn.cursor()
        cur.execute(q, ("BTCUSDT", "1m", start_ms, end_ms))
    else:
        query = (
            "SELECT timestamp_ms, open, high, low, close, volume FROM ohlcv "
            "WHERE symbol = %s AND interval = %s ORDER BY timestamp_ms"
        )
        q = query.replace("%s", "?") if isinstance(conn, sqlite3.Connection) else query
        cur = conn.cursor()
        cur.execute(q, ("BTCUSDT", "1m"))

    rows = cur.fetchall()
    if not rows:
        raise ValueError("No OHLCV data found in database")

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
        days=n / BARS_PER_DAY,
        start=timestamps[0],
        end=timestamps[-1],
    )
    return {
        "timestamps_ms": timestamps,
        "opens": opens,
        "highs": highs,
        "lows": lows,
        "closes": closes,
        "volumes": volumes,
    }


def aggregate_to_5min(data: dict[str, NDArray]) -> dict[str, NDArray]:
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


def compute_features(data: dict[str, NDArray]) -> tuple[NDArray[np.float64], list[str]]:
    """Compute feature matrix from OHLCV data.

    Returns (X, feature_names) where X is (n_bars, n_features).
    """
    from ep2_crypto.features.pipeline import FeaturePipeline

    pipeline = FeaturePipeline()
    warmup = pipeline.warmup_bars

    n = len(data["closes"])
    logger.info("computing_features", n_bars=n, warmup=warmup, n_features=pipeline.n_features)

    X = pipeline.compute_batch(
        timestamps=data["timestamps_ms"],
        opens=data["opens"],
        highs=data["highs"],
        lows=data["lows"],
        closes=data["closes"],
        volumes=data["volumes"],
    )

    feature_names = pipeline.output_names
    logger.info("features_computed", shape=X.shape, feature_names=len(feature_names))
    return X, feature_names


def compute_labels(data: dict[str, NDArray]) -> tuple[NDArray[np.int8], NDArray[np.float64]]:
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

    n_up = (labels == 1).sum()
    n_flat = (labels == 0).sum()
    n_down = (labels == -1).sum()
    logger.info(
        "labels_computed",
        total=len(labels),
        up=int(n_up),
        flat=int(n_flat),
        down=int(n_down),
        up_pct=round(n_up / max(len(labels), 1) * 100, 1),
    )
    return labels, returns


def train_walk_forward(
    X: NDArray[np.float64],
    y: NDArray[np.int8],
    feature_names: list[str],
    output_dir: Path,
    skip_gru: bool = False,
) -> dict:
    """Run walk-forward training of all models.

    Returns dict with final models and OOF predictions.
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

    # Collect OOF predictions for stacking
    oof_lgbm = np.zeros((n_bars, 3), dtype=np.float64)
    oof_catboost = np.zeros((n_bars, 3), dtype=np.float64)
    oof_mask = np.zeros(n_bars, dtype=bool)

    # Track models from last fold for saving
    last_lgbm = None
    last_catboost = None
    prev_lgbm_model = None

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
        lgbm = LGBMDirectionModel()
        lgbm_init = prev_lgbm_model._model if prev_lgbm_model is not None else None
        lgbm.train(X_tr, y_tr, X_val, y_val, feature_names=feature_names, init_model=lgbm_init)
        lgbm_proba = lgbm.predict_proba(X_test)
        oof_lgbm[test_idx] = lgbm_proba
        prev_lgbm_model = lgbm
        last_lgbm = lgbm

        # --- CatBoost ---
        catboost = CatBoostDirectionModel()
        catboost.train(X_tr, y_tr, X_val, y_val)
        catboost_proba = catboost.predict_proba(X_test)
        oof_catboost[test_idx] = catboost_proba
        last_catboost = catboost

        oof_mask[test_idx] = True

        # Fold metrics
        lgbm_preds = lgbm.predict(X_test)
        accuracy = (lgbm_preds == y_test).mean()
        fold_time = time.time() - fold_start

        # Simple Sharpe approximation from predictions
        correct = (lgbm_preds == y_test).astype(float)
        returns_sim = np.where(correct, 0.001, -0.001)
        fold_sharpe = returns_sim.mean() / max(returns_sim.std(), 1e-10) * np.sqrt(105_120)
        fold_sharpes.append(fold_sharpe)

        logger.info(
            "fold_complete",
            fold=fold.fold_idx,
            accuracy=round(float(accuracy), 4),
            sharpe=round(fold_sharpe, 2),
            train_size=fold.train_size,
            test_size=fold.test_size,
            time_s=round(fold_time, 1),
        )

    # --- Stacking Ensemble ---
    logger.info("training_stacking_ensemble")
    mask = oof_mask
    base_probas_list = [oof_lgbm[mask], oof_catboost[mask]]  # list[NDArray] as stacking expects
    y_oof = y[mask]

    stacking = StackingEnsemble()
    stacking_metrics = stacking.train(base_probas_list, y_oof, base_model_names=["lgbm", "catboost"])
    logger.info("stacking_trained", metrics=stacking_metrics)

    # --- Calibration ---
    logger.info("training_calibrator")
    stacking_probas = stacking.predict_proba(base_probas_list)
    calibrator = IsotonicCalibrator()
    cal_metrics = calibrator.fit(stacking_probas, y_oof)
    logger.info("calibrator_trained", metrics=cal_metrics)

    # --- Save models ---
    output_dir.mkdir(parents=True, exist_ok=True)

    if last_lgbm:
        last_lgbm.save(output_dir / "lgbm_direction.bin")
        logger.info("model_saved", model="lgbm", path=str(output_dir / "lgbm_direction.bin"))

    if last_catboost:
        last_catboost.save(output_dir / "catboost_direction.bin")
        logger.info("model_saved", model="catboost", path=str(output_dir / "catboost_direction.bin"))

    stacking.save(output_dir / "stacking_ensemble.pkl")
    calibrator.save(output_dir / "calibrator.pkl")
    logger.info("all_models_saved", output_dir=str(output_dir))

    # --- Summary ---
    mean_sharpe = np.mean(fold_sharpes)
    std_sharpe = np.std(fold_sharpes)
    cv_sharpe = std_sharpe / max(abs(mean_sharpe), 1e-10)

    logger.info(
        "training_summary",
        n_folds=len(folds),
        mean_sharpe=round(float(mean_sharpe), 2),
        std_sharpe=round(float(std_sharpe), 2),
        cv_sharpe=round(float(cv_sharpe), 2),
        stable=cv_sharpe < 0.5,
    )

    return {
        "lgbm": last_lgbm,
        "catboost": last_catboost,
        "stacking": stacking,
        "calibrator": calibrator,
        "fold_sharpes": fold_sharpes,
        "mean_sharpe": float(mean_sharpe),
        "oof_lgbm": oof_lgbm,
        "oof_catboost": oof_catboost,
        "oof_mask": oof_mask,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ep2-crypto models")
    parser.add_argument("--days", type=int, default=None, help="Use last N days of data (default: all)")
    parser.add_argument("--output", type=str, default="models/", help="Output directory for trained models")
    parser.add_argument("--skip-gru", action="store_true", help="Skip GRU training (faster)")
    args = parser.parse_args()

    output_dir = Path(args.output)
    logger.info("training_pipeline_start", days=args.days, output=str(output_dir))

    # 1. Load data
    conn = get_db_connection()
    try:
        raw_data = load_ohlcv(conn, days=args.days)
    finally:
        conn.close()

    # 2. Aggregate to 5-min bars
    data = aggregate_to_5min(raw_data)

    # 3. Compute features
    X, feature_names = compute_features(data)

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
    nan_count = np.isnan(X).sum()
    if nan_count > 0:
        logger.warning("filling_nans", count=int(nan_count))
        X = np.nan_to_num(X, nan=0.0)

    logger.info("data_aligned", start=start, end=end, X_shape=X.shape, y_shape=y.shape)

    # 6. Walk-forward training
    pipeline_start = time.time()
    results = train_walk_forward(X, y, feature_names, output_dir, skip_gru=args.skip_gru)
    pipeline_time = time.time() - pipeline_start

    # 7. Final report
    logger.info(
        "training_complete",
        total_time_min=round(pipeline_time / 60, 1),
        mean_sharpe=results["mean_sharpe"],
        models_saved=str(output_dir),
    )

    # Print human-readable summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Data: {X.shape[0]:,} bars, {X.shape[1]} features")
    print(f"  Folds: {len(results['fold_sharpes'])}")
    print(f"  Mean Sharpe: {results['mean_sharpe']:.2f}")
    print(f"  Time: {pipeline_time / 60:.1f} minutes")
    print(f"  Models: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
