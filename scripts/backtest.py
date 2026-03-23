"""CLI entry point for running backtests.

Usage:
    uv run python scripts/backtest.py --help
    uv run python scripts/backtest.py --days 30
    uv run python scripts/backtest.py --validate --n-permutations 5000
    uv run python scripts/backtest.py --cost-sensitivity
"""

from __future__ import annotations

import argparse

import numpy as np
import structlog

from ep2_crypto.backtest.engine import BacktestConfig, BacktestEngine
from ep2_crypto.backtest.metrics import (
    BARS_PER_DAY,
    cost_sensitivity,
    find_breakeven_cost,
)
from ep2_crypto.backtest.validation import run_validation_suite
from ep2_crypto.backtest.walk_forward import WalkForwardConfig, WalkForwardValidator

logger = structlog.get_logger(__name__)


def generate_synthetic_data(
    n_days: int = 30,
    trend: float = 0.00005,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Generate synthetic OHLCV data for demo/testing.

    In production, this would load from the database.
    """
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

    # Synthetic signals (noisy oracle)
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run backtest with statistical validation",
    )
    parser.add_argument(
        "--days", type=int, default=30,
        help="Number of days of data (default: 30)",
    )
    parser.add_argument(
        "--equity", type=float, default=50_000.0,
        help="Initial equity in USD (default: 50000)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--confidence-threshold", type=float, default=0.55,
        help="Minimum confidence to trade (default: 0.55)",
    )
    parser.add_argument(
        "--validate", action="store_true",
        help="Run statistical validation suite",
    )
    parser.add_argument(
        "--n-permutations", type=int, default=1000,
        help="Permutation test iterations (default: 1000)",
    )
    parser.add_argument(
        "--n-bootstrap", type=int, default=1000,
        help="Bootstrap iterations (default: 1000)",
    )
    parser.add_argument(
        "--cost-sensitivity", action="store_true",
        help="Run cost sensitivity analysis",
    )
    parser.add_argument(
        "--walk-forward", action="store_true",
        help="Show walk-forward fold info",
    )

    args = parser.parse_args()

    # Generate data
    logger.info("generating_data", days=args.days)
    data = generate_synthetic_data(n_days=args.days, seed=args.seed)

    # Run backtest
    config = BacktestConfig(
        initial_equity=args.equity,
        seed=args.seed,
        confidence_threshold=args.confidence_threshold,
    )
    engine = BacktestEngine(config)

    logger.info("running_backtest", n_bars=len(data["closes"]))
    result = engine.run(**data)

    # Print results
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)
    print(result.summary())

    # Walk-forward info
    if args.walk_forward:
        print("\n" + "=" * 60)
        print("WALK-FORWARD FOLDS")
        print("=" * 60)
        wf_config = WalkForwardConfig()
        wf = WalkForwardValidator(len(data["closes"]), config=wf_config)
        print(f"Total folds: {wf.n_folds}")
        for fold in wf.folds()[:5]:
            print(
                f"  Fold {fold.fold_idx}: train [{fold.train_start}:{fold.train_end}] "
                f"test [{fold.test_start}:{fold.test_end}]"
            )
        if wf.n_folds > 5:
            print(f"  ... and {wf.n_folds - 5} more folds")

    # Cost sensitivity
    if args.cost_sensitivity:
        print("\n" + "=" * 60)
        print("COST SENSITIVITY")
        print("=" * 60)
        if len(result.equity_curve) > 0:
            # Approximate gross returns by adding back costs
            gross = np.diff(result.equity_curve) / result.equity_curve[:-1]
            gross = np.concatenate([[0.0], gross])
            sens = cost_sensitivity(
                gross, result.total_trades, len(gross),
                cost_levels_bps=[0, 4, 8, 12, 16, 20],
            )
            for entry in sens:
                print(
                    f"  {entry['cost_bps']:5.0f} bps: "
                    f"Sharpe={entry['sharpe']:7.3f}  "
                    f"Return={entry['total_return']:8.2%}"
                )
            be = find_breakeven_cost(sens)
            print(f"\n  Break-even cost: {be:.1f} bps")

    # Statistical validation
    if args.validate:
        print("\n" + "=" * 60)
        print("STATISTICAL VALIDATION")
        print("=" * 60)
        # Get per-bar returns from equity curve
        returns = np.diff(result.equity_curve) / result.equity_curve[:-1]
        returns = np.concatenate([[0.0], returns])

        val = run_validation_suite(
            returns=returns,
            n_permutations=args.n_permutations,
            n_bootstrap=args.n_bootstrap,
            seed=args.seed,
        )
        print(val.summary())

    print()


if __name__ == "__main__":
    main()
