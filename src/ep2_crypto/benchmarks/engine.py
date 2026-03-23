"""Backtest engine — runs strategies against OHLCV data and computes metrics.

Handles:
- Position-weighted returns with trading costs
- Multiple strategy comparison
- Random strategy distribution generation
- Regime-decomposed evaluation
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from ep2_crypto.benchmarks.metrics import BacktestMetrics, compute_metrics
from ep2_crypto.benchmarks.strategies import (
    BenchmarkStrategy,
    RandomEntry,
    get_default_benchmark_suite,
)

logger = logging.getLogger(__name__)


class BacktestEngine:
    """Run benchmarks against price data and collect results."""

    def __init__(
        self,
        trading_cost_bps: float = 3.0,
        slippage_bps: float = 1.0,
    ) -> None:
        """
        Args:
            trading_cost_bps: Exchange fee in bps per side (e.g., 3 = 0.03% = Binance taker).
            slippage_bps: Estimated slippage in bps per trade.
        """
        self.total_cost_bps = trading_cost_bps + slippage_bps

    def run_strategy(
        self,
        strategy: BenchmarkStrategy,
        df: pd.DataFrame,
    ) -> tuple[BacktestMetrics, pd.Series, pd.Series]:
        """Run a single strategy and return (metrics, returns, positions).

        Args:
            strategy: Strategy instance with generate_positions method.
            df: OHLCV DataFrame with at minimum close column.

        Returns:
            Tuple of (BacktestMetrics, strategy_returns Series, positions Series).
        """
        positions = strategy.generate_positions(df)
        raw_returns = df["close"].pct_change().fillna(0.0)
        strategy_returns = raw_returns * positions

        metrics = compute_metrics(
            returns=strategy_returns,
            positions=positions,
            trading_cost_bps=self.total_cost_bps,
        )
        return metrics, strategy_returns, positions

    def run_suite(
        self,
        df: pd.DataFrame,
        suite: dict[str, BenchmarkStrategy] | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Run all benchmarks and return a results dictionary.

        Args:
            df: OHLCV DataFrame.
            suite: Optional custom suite. Defaults to get_default_benchmark_suite().

        Returns:
            Dict mapping strategy_name -> {metrics, returns, positions}.
        """
        if suite is None:
            suite = get_default_benchmark_suite()

        results: dict[str, dict[str, Any]] = {}
        for name, strategy in suite.items():
            try:
                metrics, returns, positions = self.run_strategy(strategy, df)
                results[name] = {
                    "metrics": metrics,
                    "returns": returns,
                    "positions": positions,
                    "strategy": strategy,
                }
                logger.info(
                    "Strategy %s: Sharpe=%.3f, Return=%.2f%%, MaxDD=%.2f%%",
                    name,
                    metrics.sharpe_ratio,
                    metrics.total_return * 100,
                    metrics.max_drawdown * 100,
                )
            except (ValueError, KeyError) as e:
                logger.warning("Strategy %s failed: %s", name, str(e))
                results[name] = {"error": str(e)}

        return results

    def run_random_distribution(
        self,
        df: pd.DataFrame,
        n_simulations: int = 1000,
        hold_bars: int = 3,
        entry_probability: float = 0.1,
    ) -> dict[str, Any]:
        """Run many random entry simulations to build the null distribution.

        Returns:
            Dict with:
              - sharpe_distribution: array of Sharpe ratios
              - return_distribution: array of total returns
              - p95_sharpe: 95th percentile Sharpe (strategy must beat this)
              - p99_sharpe: 99th percentile Sharpe
              - mean_sharpe: average random Sharpe
              - std_sharpe: std of random Sharpe
        """
        sharpes = np.zeros(n_simulations)
        total_returns = np.zeros(n_simulations)

        raw_returns = df["close"].pct_change().fillna(0.0)

        for i in range(n_simulations):
            strategy = RandomEntry(
                hold_bars=hold_bars,
                seed=i,
                entry_probability=entry_probability,
            )
            positions = strategy.generate_positions(df)
            strat_returns = raw_returns * positions

            metrics = compute_metrics(
                returns=strat_returns,
                positions=positions,
                trading_cost_bps=self.total_cost_bps,
            )
            sharpes[i] = metrics.sharpe_ratio
            total_returns[i] = metrics.total_return

        return {
            "sharpe_distribution": sharpes,
            "return_distribution": total_returns,
            "p95_sharpe": float(np.percentile(sharpes, 95)),
            "p99_sharpe": float(np.percentile(sharpes, 99)),
            "mean_sharpe": float(sharpes.mean()),
            "std_sharpe": float(sharpes.std()),
            "mean_return": float(total_returns.mean()),
            "std_return": float(total_returns.std()),
        }

    def compare_to_random(
        self,
        strategy_sharpe: float,
        random_dist: dict[str, Any],
    ) -> dict[str, Any]:
        """Compare a strategy's Sharpe to the random distribution.

        Returns significance metrics.
        """
        sharpes = random_dist["sharpe_distribution"]
        z_score = (strategy_sharpe - sharpes.mean()) / max(sharpes.std(), 1e-10)
        p_value = float((sharpes >= strategy_sharpe).mean())

        return {
            "strategy_sharpe": strategy_sharpe,
            "random_mean_sharpe": random_dist["mean_sharpe"],
            "random_std_sharpe": random_dist["std_sharpe"],
            "z_score": float(z_score),
            "p_value": p_value,
            "significant_at_05": p_value < 0.05,
            "significant_at_01": p_value < 0.01,
            "beats_p95": strategy_sharpe > random_dist["p95_sharpe"],
            "beats_p99": strategy_sharpe > random_dist["p99_sharpe"],
        }

    def regime_decomposed_backtest(
        self,
        strategy: BenchmarkStrategy,
        df: pd.DataFrame,
        regime_labels: pd.Series,
    ) -> dict[str, BacktestMetrics]:
        """Run backtest decomposed by regime labels.

        Args:
            strategy: Strategy to evaluate.
            df: Full OHLCV DataFrame.
            regime_labels: Series with same index as df, values like
                          'trending_up', 'trending_down', 'ranging', etc.

        Returns:
            Dict mapping regime_name -> BacktestMetrics for that regime.
        """
        positions = strategy.generate_positions(df)
        raw_returns = df["close"].pct_change().fillna(0.0)
        strategy_returns = raw_returns * positions

        results: dict[str, BacktestMetrics] = {}
        for regime_name in regime_labels.unique():
            mask = regime_labels == regime_name
            if mask.sum() < 10:
                continue
            regime_ret = strategy_returns[mask]
            regime_pos = positions[mask]
            metrics = compute_metrics(
                returns=regime_ret,
                positions=regime_pos,
                trading_cost_bps=self.total_cost_bps,
            )
            results[regime_name] = metrics

        return results

    def results_to_dataframe(
        self,
        results: dict[str, dict[str, Any]],
    ) -> pd.DataFrame:
        """Convert suite results to a summary DataFrame for comparison."""
        rows = []
        for name, result in results.items():
            if "error" in result:
                continue
            m: BacktestMetrics = result["metrics"]
            rows.append({"strategy": name, **m.to_dict()})
        return pd.DataFrame(rows).set_index("strategy").sort_values("sharpe_ratio", ascending=False)
