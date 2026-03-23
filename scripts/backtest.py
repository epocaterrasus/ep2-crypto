"""CLI entry point for running backtests.

Usage:
    uv run python scripts/backtest.py --help
    uv run python scripts/backtest.py --days 30
    uv run python scripts/backtest.py --source db --days 90
    uv run python scripts/backtest.py --source db --start 2024-01-01 --end 2024-12-31
    uv run python scripts/backtest.py --validate --n-permutations 5000
    uv run python scripts/backtest.py --cost-sensitivity
    uv run python scripts/backtest.py --venue polymarket --source db
    uv run python scripts/backtest.py --output results/backtest_2024.json
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import structlog

from ep2_crypto.backtest.engine import BacktestConfig, BacktestEngine
from ep2_crypto.backtest.metrics import (
    BARS_PER_DAY,
    cost_sensitivity,
    find_breakeven_cost,
)
from ep2_crypto.backtest.polymarket_backtest import (
    BinaryBar,
    BinaryFeeModel,
    PolymarketBacktester,
)
from ep2_crypto.backtest.validation import run_validation_suite
from ep2_crypto.backtest.walk_forward import WalkForwardConfig, WalkForwardValidator
from ep2_crypto.db.repository import Repository
from ep2_crypto.db.schema import create_tables

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = structlog.get_logger(__name__)

# Minimum days recommended to span both a bull and bear market segment.
# 180 days gives ~6 months of data, which typically includes regime variety.
MIN_DAYS_FOR_REGIME_COVERAGE = 180

# Environment variable name used by train.py — kept consistent.
_DB_URL_ENV = "EP2_DB_URL"


# ---------------------------------------------------------------------------
# Database connection helpers
# ---------------------------------------------------------------------------

def _get_db_connection() -> sqlite3.Connection:
    """Open database connection from EP2_DB_URL env var or default SQLite path."""
    db_url = os.environ.get(_DB_URL_ENV, "data/ep2_crypto.db")

    if "postgresql" in db_url or "timescale" in db_url:
        try:
            import psycopg2

            conn = psycopg2.connect(db_url)
            conn.autocommit = False
            logger.info("db_connected", backend="timescaledb", url_prefix=db_url[:30])
            return conn  # type: ignore[return-value]
        except ImportError as exc:
            logger.error(
                "psycopg2_not_installed",
                hint="pip install psycopg2-binary",
            )
            raise RuntimeError("psycopg2 required for TimescaleDB backend") from exc
    else:
        db_path = Path(db_url)
        if not db_path.exists():
            logger.warning(
                "db_file_not_found",
                path=str(db_path),
                hint="Run scripts/collect_history.py first to populate the database",
            )
        conn = create_tables(db_path)
        logger.info("db_connected", backend="sqlite", path=str(db_path))
        return conn


def _is_sqlite_conn(conn: sqlite3.Connection) -> bool:
    return isinstance(conn, sqlite3.Connection)


# ---------------------------------------------------------------------------
# Real data loading via repository layer
# ---------------------------------------------------------------------------

def load_real_data(
    conn: sqlite3.Connection,
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
) -> dict[str, NDArray]:
    """Load OHLCV + funding data from DB using the parameterized repository.

    Returns dict compatible with BacktestEngine.run() signature.
    """
    repo = Repository(conn)

    ohlcv_rows = repo.query_ohlcv(symbol, interval, start_ms, end_ms)
    if not ohlcv_rows:
        msg = (
            f"No OHLCV data found for symbol={symbol!r} interval={interval!r} "
            f"between {start_ms} and {end_ms}. "
            "Run scripts/collect_history.py to backfill data."
        )
        raise ValueError(msg)

    n = len(ohlcv_rows)
    timestamps = np.empty(n, dtype=np.int64)
    opens = np.empty(n, dtype=np.float64)
    highs = np.empty(n, dtype=np.float64)
    lows = np.empty(n, dtype=np.float64)
    closes = np.empty(n, dtype=np.float64)
    volumes = np.empty(n, dtype=np.float64)

    for i, row in enumerate(ohlcv_rows):
        timestamps[i] = row["timestamp_ms"]
        opens[i] = row["open"]
        highs[i] = row["high"]
        lows[i] = row["low"]
        closes[i] = row["close"]
        volumes[i] = row["volume"]

    logger.info(
        "ohlcv_loaded",
        rows=n,
        days=round(n / BARS_PER_DAY, 1),
        start_ms=int(timestamps[0]),
        end_ms=int(timestamps[-1]),
        symbol=symbol,
        interval=interval,
    )

    # Load funding rates — forward-fill gaps (funding settles every 8h)
    funding_rows = repo.query_funding_rate(symbol, start_ms, end_ms)
    funding_rates = np.full(n, 0.0001, dtype=np.float64)  # default 0.01%/8h

    if funding_rows:
        # Build timestamp→rate lookup then map onto bar timestamps
        funding_lookup: dict[int, float] = {
            r["timestamp_ms"]: r["funding_rate"] for r in funding_rows
        }
        last_rate = 0.0001
        for i in range(n):
            ts = int(timestamps[i])
            if ts in funding_lookup:
                last_rate = funding_lookup[ts]
            funding_rates[i] = last_rate

        logger.info("funding_loaded", rows=len(funding_rows))

    return {
        "opens": opens,
        "highs": highs,
        "lows": lows,
        "closes": closes,
        "volumes": volumes,
        "timestamps_ms": timestamps,
        "funding_rates": funding_rates,
    }


# ---------------------------------------------------------------------------
# Synthetic data (demo / CI only — not for production evaluation)
# ---------------------------------------------------------------------------

def generate_synthetic_data(
    n_days: int = 30,
    trend: float = 0.00005,
    seed: int = 42,
) -> dict[str, NDArray]:
    """Generate synthetic OHLCV data for demo/testing.

    WARNING: This path produces an unrealistically clean signal (noisy oracle).
    Do NOT use synthetic data for production strategy evaluation.
    Use --source db with real data instead.
    """
    logger.warning(
        "using_synthetic_data",
        reason="source=synthetic selected — results are NOT production-grade",
    )
    rng = np.random.default_rng(seed)
    n = n_days * BARS_PER_DAY
    returns = rng.normal(trend, 0.001, size=n)
    closes = 100_000.0 * np.cumprod(1 + returns)

    opens = np.roll(closes, 1)
    opens[0] = 100_000.0
    highs = np.maximum(opens, closes) * (1 + rng.uniform(0, 0.002, n))
    lows = np.minimum(opens, closes) * (1 - rng.uniform(0, 0.002, n))
    volumes = rng.uniform(100, 500, n)

    # Start at 10:00 UTC (within trading hours)
    base_ts = 1704103200000
    timestamps = np.arange(n, dtype=np.int64) * 300_000 + base_ts

    # Synthetic signals (noisy oracle — NOT from trained model)
    future_ret = np.zeros(n)
    future_ret[:-1] = np.diff(closes) / closes[:-1]
    noise = rng.normal(0, 0.002, n)
    raw_signal = future_ret + noise
    signals = np.sign(raw_signal).astype(np.int8)
    confidences = np.clip(np.abs(raw_signal) * 500, 0.5, 0.95)

    funding_rates = rng.normal(0.0001, 0.00003, n)

    return {
        "opens": opens,
        "highs": highs,
        "lows": lows,
        "closes": closes,
        "volumes": volumes,
        "timestamps_ms": timestamps,
        "signals": signals,
        "confidences": confidences,
        "funding_rates": funding_rates,
    }


# ---------------------------------------------------------------------------
# Model signal loading
# ---------------------------------------------------------------------------

def load_model_signals(
    data: dict[str, NDArray],
    model_dir: Path,
) -> tuple[NDArray[np.int8], NDArray[np.float64]]:
    """Load trained model predictions for the given OHLCV data.

    Attempts to load the stacking ensemble from model_dir. Falls back to
    zero signals (no trades) if models are not found, with a clear warning.

    Args:
        data: OHLCV arrays from load_real_data().
        model_dir: Directory containing serialized model artifacts.

    Returns:
        (signals, confidences) arrays of length n_bars.
    """
    n = len(data["closes"])
    ensemble_path = model_dir / "ensemble" / "stacking_meta.pkl"

    if not ensemble_path.exists():
        logger.warning(
            "model_not_found",
            path=str(ensemble_path),
            fallback="zero_signals",
            hint="Run scripts/train.py first to train and save models",
        )
        return np.zeros(n, dtype=np.int8), np.zeros(n, dtype=np.float64)

    try:
        from ep2_crypto.features.pipeline import FeaturePipeline
        from ep2_crypto.models.stacking import StackingEnsemble

        pipeline = FeaturePipeline()
        X = pipeline.compute_batch(
            timestamps=data["timestamps_ms"],
            opens=data["opens"],
            highs=data["highs"],
            lows=data["lows"],
            closes=data["closes"],
            volumes=data["volumes"],
        )

        ensemble = StackingEnsemble.load(ensemble_path)
        proba = ensemble.predict_proba(X)  # shape (n, 3): [down, flat, up]

        # ternary: -1=down (class 0), 0=flat (class 1), +1=up (class 2)
        predicted_class = np.argmax(proba, axis=1) - 1  # map 0,1,2 → -1,0,+1
        signals = predicted_class.astype(np.int8)
        # Confidence = max probability across directional classes (not flat)
        directional_prob = np.maximum(proba[:, 0], proba[:, 2])
        confidences = directional_prob.astype(np.float64)

        logger.info(
            "model_signals_loaded",
            n_bars=n,
            n_signals=int(np.count_nonzero(signals)),
            model_path=str(ensemble_path),
        )
        return signals, confidences

    except Exception:
        logger.exception(
            "model_load_failed",
            path=str(ensemble_path),
            fallback="zero_signals",
        )
        return np.zeros(n, dtype=np.int8), np.zeros(n, dtype=np.float64)


# ---------------------------------------------------------------------------
# Polymarket venue helpers
# ---------------------------------------------------------------------------

def build_binary_bars(
    data: dict[str, NDArray],
    signals: NDArray[np.int8],
    confidences: NDArray[np.float64],
    market_spread: float = 0.02,
) -> list[BinaryBar]:
    """Convert OHLCV + signal arrays into BinaryBar list for PolymarketBacktester.

    Market prices are approximated: YES ≈ model_prob, NO ≈ 1 - model_prob,
    with a configurable spread applied symmetrically. In production, real
    Polymarket order book prices would be fetched from the API.
    """
    bars: list[BinaryBar] = []
    n = len(data["closes"])
    for i in range(n):
        sig = int(signals[i])
        prob = float(confidences[i])
        # Approximate market prices from model confidence
        yes_price = max(0.01, min(0.99, prob - market_spread / 2))
        no_price = max(0.01, min(0.99, (1.0 - prob) - market_spread / 2))
        bars.append(
            BinaryBar(
                timestamp_ms=int(data["timestamps_ms"][i]),
                open_price=float(data["opens"][i]),
                close_price=float(data["closes"][i]),
                signal=sig,
                model_prob=prob,
                market_price_yes=yes_price,
                market_price_no=no_price,
            )
        )
    return bars


# ---------------------------------------------------------------------------
# Regime coverage check
# ---------------------------------------------------------------------------

def _warn_if_insufficient_regime_coverage(n_days: int) -> None:
    """Emit a warning when the backtest period is too short for regime diversity."""
    if n_days < MIN_DAYS_FOR_REGIME_COVERAGE:
        logger.warning(
            "insufficient_regime_coverage",
            days=n_days,
            minimum_recommended=MIN_DAYS_FOR_REGIME_COVERAGE,
            reason=(
                "Backtest period is less than 180 days. "
                "Results may not span both bull and bear market segments. "
                "Use --days 365 or --start/--end spanning at least one full market cycle."
            ),
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run backtest with statistical validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Demo with synthetic data (no DB required):
  uv run python scripts/backtest.py --days 30

  # Real data from configured DB (EP2_DB_URL env var):
  uv run python scripts/backtest.py --source db --days 90

  # Specific date range (real data):
  uv run python scripts/backtest.py --source db --start 2024-01-01 --end 2024-12-31

  # Polymarket binary payoff simulation:
  uv run python scripts/backtest.py --venue polymarket --source db --days 90

  # Full validation + save results:
  uv run python scripts/backtest.py --source db --validate --output results/bt.json
""",
    )

    # Data source
    parser.add_argument(
        "--source",
        choices=["synthetic", "db"],
        default="synthetic",
        help=(
            "Data source: 'synthetic' uses generated data (demo only), "
            "'db' loads from the database configured via EP2_DB_URL env var "
            "(default: synthetic)"
        ),
    )
    parser.add_argument(
        "--symbol",
        default="BTCUSDT",
        help="Trading symbol (default: BTCUSDT)",
    )
    parser.add_argument(
        "--interval",
        default="5m",
        help="Bar interval stored in DB (default: 5m)",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Start date for backtest, format YYYY-MM-DD (requires --source db)",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="End date for backtest, format YYYY-MM-DD (requires --source db)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help=(
            "Number of trailing days of data when --start/--end not specified "
            "(default: 30). Recommended >=180 to cover bull+bear regimes."
        ),
    )

    # Venue
    parser.add_argument(
        "--venue",
        choices=["futures", "polymarket"],
        default="futures",
        help=(
            "Trading venue: 'futures' uses standard BTC perp backtest engine, "
            "'polymarket' uses binary payoff backtester with ~2%% fee structure "
            "(default: futures)"
        ),
    )

    # Model
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(os.environ.get("EP2_MODEL_DIR", "models")),
        help=(
            "Directory containing trained model artifacts "
            "(default: EP2_MODEL_DIR env var or 'models/'). "
            "If no model is found, zero signals are used."
        ),
    )
    parser.add_argument(
        "--no-model",
        action="store_true",
        help="Skip model loading and use zero signals (benchmark: no-trade baseline)",
    )

    # Backtest config
    parser.add_argument(
        "--equity",
        type=float,
        default=50_000.0,
        help="Initial equity in USD (default: 50000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.60,
        help="Minimum confidence to trade (default: 0.60)",
    )

    # Polymarket-specific
    parser.add_argument(
        "--bet-fraction",
        type=float,
        default=0.02,
        help="Fraction of capital per Polymarket binary bet (default: 0.02 = 2%%)",
    )

    # Analysis flags
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run statistical validation suite (PSR, DSR, permutation, bootstrap)",
    )
    parser.add_argument(
        "--n-permutations",
        type=int,
        default=1000,
        help="Permutation test iterations (default: 1000)",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=1000,
        help="Bootstrap iterations (default: 1000)",
    )
    parser.add_argument(
        "--cost-sensitivity",
        action="store_true",
        help="Run cost sensitivity analysis across 0–20 bps round-trip costs",
    )
    parser.add_argument(
        "--walk-forward",
        action="store_true",
        help="Display walk-forward fold structure for the loaded dataset",
    )

    # Output
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Save full results to this JSON file path (default: stdout only)",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    # -- Resolve date range ---------------------------------------------------
    if args.start or args.end:
        if args.source != "db":
            logger.error(
                "date_range_requires_db_source",
                hint="Use --source db with --start/--end",
            )
            raise SystemExit(1)
        fmt = "%Y-%m-%d"
        start_dt = datetime.strptime(args.start, fmt).replace(tzinfo=timezone.utc)
        end_dt = datetime.strptime(args.end, fmt).replace(tzinfo=timezone.utc)
        start_ms = int(start_dt.timestamp() * 1000)
        end_ms = int(end_dt.timestamp() * 1000)
        n_days = max(1, (end_ms - start_ms) // (24 * 60 * 60 * 1000))
    else:
        end_ms = int(time.time() * 1000)
        start_ms = end_ms - args.days * 24 * 60 * 60 * 1000
        n_days = args.days

    _warn_if_insufficient_regime_coverage(n_days)

    # -- Load data ------------------------------------------------------------
    if args.source == "db":
        logger.info(
            "loading_real_data",
            source="db",
            symbol=args.symbol,
            interval=args.interval,
            days=n_days,
        )
        conn = _get_db_connection()
        try:
            data = load_real_data(
                conn,
                symbol=args.symbol,
                interval=args.interval,
                start_ms=start_ms,
                end_ms=end_ms,
            )
        finally:
            conn.close()
    else:
        logger.info("loading_synthetic_data", days=args.days)
        data = generate_synthetic_data(n_days=args.days, seed=args.seed)

    n_bars = len(data["closes"])
    logger.info("data_loaded", n_bars=n_bars, days=round(n_bars / BARS_PER_DAY, 1))

    # -- Load signals ---------------------------------------------------------
    if "signals" in data and "confidences" in data:
        # Synthetic path already includes signals
        signals: NDArray[np.int8] = data["signals"]
        confidences: NDArray[np.float64] = data["confidences"]
    elif args.no_model:
        logger.info("model_skipped", reason="--no-model flag set")
        signals = np.zeros(n_bars, dtype=np.int8)
        confidences = np.zeros(n_bars, dtype=np.float64)
    else:
        signals, confidences = load_model_signals(data, args.model_dir)

    # -- Run backtest ---------------------------------------------------------
    results_dict: dict = {}

    if args.venue == "polymarket":
        logger.info(
            "running_polymarket_backtest",
            n_bars=n_bars,
            bet_fraction=args.bet_fraction,
            initial_capital=args.equity,
        )
        bars = build_binary_bars(data, signals, confidences)
        backtester = PolymarketBacktester(
            initial_capital=args.equity,
            bet_fraction=args.bet_fraction,
            fee_model=BinaryFeeModel(taker_fee_rate=0.02),
            min_model_prob=args.confidence_threshold,
        )
        poly_result = backtester.run(bars)

        logger.info(
            "polymarket_backtest_complete",
            total_trades=poly_result.total_trades,
            win_rate=round(poly_result.win_rate, 4),
            net_pnl=round(poly_result.total_pnl_net, 2),
            roi=round(poly_result.roi, 4),
            sharpe=round(poly_result.sharpe, 3),
            max_drawdown=round(poly_result.max_drawdown, 4),
        )
        logger.info("polymarket_summary", summary=poly_result.summary())

        results_dict = {
            "venue": "polymarket",
            "source": args.source,
            "n_bars": n_bars,
            "n_days": n_days,
            "initial_capital": args.equity,
            "bet_fraction": args.bet_fraction,
            "total_trades": poly_result.total_trades,
            "wins": poly_result.wins,
            "losses": poly_result.losses,
            "win_rate": poly_result.win_rate,
            "total_pnl_net": poly_result.total_pnl_net,
            "total_fees": poly_result.total_fees,
            "roi": poly_result.roi,
            "sharpe": poly_result.sharpe,
            "sortino": poly_result.sortino,
            "profit_factor": poly_result.profit_factor,
            "max_drawdown": poly_result.max_drawdown,
            "peak_equity": poly_result.peak_equity,
            "edge_per_bet": poly_result.edge_per_bet,
        }

    else:
        # Futures perpetual backtest
        config = BacktestConfig(
            initial_equity=args.equity,
            seed=args.seed,
            confidence_threshold=args.confidence_threshold,
        )
        engine = BacktestEngine(config)

        logger.info("running_futures_backtest", n_bars=n_bars)
        result = engine.run(
            opens=data["opens"],
            highs=data["highs"],
            lows=data["lows"],
            closes=data["closes"],
            volumes=data["volumes"],
            timestamps_ms=data["timestamps_ms"],
            signals=signals,
            confidences=confidences,
            funding_rates=data.get("funding_rates"),
        )

        logger.info("backtest_results", summary=result.summary())

        # Walk-forward fold display
        if args.walk_forward:
            wf_config = WalkForwardConfig()
            wf = WalkForwardValidator(n_bars, config=wf_config)
            logger.info(
                "walk_forward_folds",
                total_folds=wf.n_folds,
                train_days=wf_config.train_days,
                test_days=wf_config.test_days,
                purge_bars=wf_config.purge_bars,
                embargo_bars=wf_config.embargo_bars,
            )
            for fold in wf.folds()[:5]:
                logger.info(
                    "fold_info",
                    fold_idx=fold.fold_idx,
                    train=f"[{fold.train_start}:{fold.train_end}]",
                    test=f"[{fold.test_start}:{fold.test_end}]",
                )
            if wf.n_folds > 5:
                logger.info("walk_forward_truncated", remaining=wf.n_folds - 5)

        # Cost sensitivity
        cost_sens_rows: list[dict] = []
        if args.cost_sensitivity:
            if len(result.equity_curve) > 0:
                gross = np.diff(result.equity_curve) / result.equity_curve[:-1]
                gross = np.concatenate([[0.0], gross])
                sens = cost_sensitivity(
                    gross,
                    result.total_trades,
                    len(gross),
                    cost_levels_bps=[0, 4, 8, 12, 16, 20],
                )
                cost_sens_rows = sens
                for entry in sens:
                    logger.info(
                        "cost_sensitivity_row",
                        cost_bps=entry["cost_bps"],
                        sharpe=round(entry["sharpe"], 3),
                        total_return=round(entry["total_return"], 4),
                        annualized_return=round(entry["annualized_return"], 4),
                    )
                be = find_breakeven_cost(sens)
                logger.info("breakeven_cost_bps", bps=round(be, 1))
            else:
                logger.warning("cost_sensitivity_skipped", reason="empty equity curve")

        # Statistical validation
        val_dict: dict = {}
        if args.validate:
            returns = np.diff(result.equity_curve) / result.equity_curve[:-1]
            returns = np.concatenate([[0.0], returns])

            val = run_validation_suite(
                returns=returns,
                n_permutations=args.n_permutations,
                n_bootstrap=args.n_bootstrap,
                seed=args.seed,
            )
            logger.info("validation_results", summary=val.summary())
            val_dict = {
                "verdict": val.verdict.value,
                "psr": val.probabilistic_sharpe_ratio,
                "dsr": val.deflated_sharpe_ratio,
                "permutation_p_value": val.permutation_p_value,
                "bootstrap_sharpe_ci_low": val.bootstrap_sharpe_ci[0],
                "bootstrap_sharpe_ci_high": val.bootstrap_sharpe_ci[1],
                "walk_forward_cv": val.walk_forward_cv,
            }

        results_dict = {
            "venue": "futures",
            "source": args.source,
            "symbol": args.symbol,
            "interval": args.interval,
            "n_bars": n_bars,
            "n_days": n_days,
            "initial_equity": args.equity,
            "confidence_threshold": args.confidence_threshold,
            **result.to_dict(),
            "cost_sensitivity": cost_sens_rows,
            "validation": val_dict,
        }

    # -- Persist results to JSON if requested ---------------------------------
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as fh:
            json.dump(results_dict, fh, indent=2, default=str)
        logger.info("results_saved", path=str(args.output))


if __name__ == "__main__":
    main()
