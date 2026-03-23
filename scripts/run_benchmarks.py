"""Run all benchmark strategies and produce the full statistical report.

Usage:
    uv run python scripts/run_benchmarks.py [--n-bars 8640] [--n-random-sims 1000]

Outputs a comprehensive report covering all 12 analysis items.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

import numpy as np
import pandas as pd

# Add src to path for development
sys.path.insert(0, "src")

from ep2_crypto.benchmarks.data import generate_synthetic_btc
from ep2_crypto.benchmarks.engine import BacktestEngine
from ep2_crypto.benchmarks.metrics import BARS_PER_DAY, BacktestMetrics
from ep2_crypto.benchmarks.statistics import StatisticalAnalyzer
from ep2_crypto.benchmarks.strategies import (
    BuyAndHold,
    CrossMarketLeadLag,
    FundingRateStrategy,
    MeanReversionRSI,
    MovingAverageCrossover,
    NaiveEnsemble,
    OracleStrategy,
    OrderBookImbalance,
    RandomEntry,
    SimpleMomentum,
    VolatilityBreakout,
    get_default_benchmark_suite,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def separator(title: str) -> None:
    logger.info("")
    logger.info("=" * 80)
    logger.info("  %s", title)
    logger.info("=" * 80)


def run_all_benchmarks(n_bars: int, n_random_sims: int) -> None:
    start_time = time.time()

    # --- Generate data ---
    separator("DATA GENERATION")
    logger.info("Generating %d bars (%.1f days) of synthetic 5-min BTC data...", n_bars, n_bars / BARS_PER_DAY)
    df = generate_synthetic_btc(n_bars=n_bars, regime_switching=True, seed=42)
    logger.info("Price range: $%.0f - $%.0f", df["close"].min(), df["close"].max())
    logger.info("Volatility (annualized): %.1f%%", df["close"].pct_change().std() * np.sqrt(BARS_PER_DAY * 365.25) * 100)

    engine = BacktestEngine(trading_cost_bps=3.0, slippage_bps=1.0)
    analyzer = StatisticalAnalyzer(confidence_level=0.95)

    # ===================================================================
    # 1. BUY AND HOLD
    # ===================================================================
    separator("1. BUY AND HOLD")
    for leverage in [1.0, 2.0, 5.0]:
        strategy = BuyAndHold(leverage=leverage)
        metrics, returns, positions = engine.run_strategy(strategy, df)
        logger.info("--- Buy & Hold %.0fx ---", leverage)
        logger.info("\n%s", metrics.summary())
        ci = analyzer.sharpe_confidence_interval(returns)
        if "analytical" in ci:
            logger.info(
                "Sharpe 95%% CI (analytical): [%.3f, %.3f]",
                ci["analytical"]["ci_lower"],
                ci["analytical"]["ci_upper"],
            )
        logger.info(
            "WHY HARD TO BEAT: In bull markets, buy-and-hold captures 100%% of "
            "the move with zero trading costs. At %.0fx leverage, volatility drag "
            "reduces returns: the higher the leverage, the worse the drag at 5-min "
            "compounding frequency.",
            leverage,
        )

    # ===================================================================
    # 2. SIMPLE MOMENTUM
    # ===================================================================
    separator("2. SIMPLE MOMENTUM")
    best_momentum_sharpe = -np.inf
    best_momentum_n = 1
    momentum_results = {}
    for n in [1, 3, 5, 10, 20]:
        strategy = SimpleMomentum(lookback=n)
        metrics, returns, positions = engine.run_strategy(strategy, df)
        momentum_results[n] = (metrics, returns, positions)
        logger.info("--- Momentum N=%d ---", n)
        logger.info("  Sharpe: %.3f | Return: %.2f%% | MaxDD: %.2f%% | Trades/day: %.1f",
                     metrics.sharpe_ratio, metrics.total_return * 100,
                     metrics.max_drawdown * 100, metrics.trades_per_day)
        sig = analyzer.test_sharpe_vs_zero(returns)
        logger.info("  Significance vs zero: t=%.3f, p=%.4f, significant=%s",
                     sig["test_statistic"], sig["p_value"], sig["significant"])
        if metrics.sharpe_ratio > best_momentum_sharpe:
            best_momentum_sharpe = metrics.sharpe_ratio
            best_momentum_n = n

    logger.info("BEST MOMENTUM: N=%d (Sharpe=%.3f)", best_momentum_n, best_momentum_sharpe)
    logger.info(
        "WHY HARD TO BEAT: Short lookback (N=1-3) captures micro-momentum in "
        "trending bars but gets destroyed by mean reversion in ranging markets. "
        "Longer lookback (N=10-20) is smoother but lags significantly at 5-min. "
        "Beating momentum proves the ML system handles regime transitions better."
    )

    # ===================================================================
    # 3. MEAN REVERSION (RSI)
    # ===================================================================
    separator("3. MEAN REVERSION (RSI)")
    for oversold, overbought in [(30, 70), (20, 80)]:
        strategy = MeanReversionRSI(period=14, oversold=oversold, overbought=overbought)
        metrics, returns, positions = engine.run_strategy(strategy, df)
        logger.info("--- RSI %d/%d ---", oversold, overbought)
        logger.info("  Sharpe: %.3f | Return: %.2f%% | MaxDD: %.2f%% | WinRate: %.2f%%",
                     metrics.sharpe_ratio, metrics.total_return * 100,
                     metrics.max_drawdown * 100, metrics.win_rate * 100)
        sig = analyzer.test_sharpe_vs_zero(returns)
        logger.info("  Significance: p=%.4f", sig["p_value"])

    logger.info(
        "WHY EASY/HARD TO BEAT: RSI mean reversion works in range-bound regimes "
        "but gets destroyed in strong trends (buys the dip that keeps dipping). "
        "At 5-min BTC, most of the time is trending at micro-scale, making pure "
        "RSI unprofitable after costs. Beating RSI is relatively easy; the real "
        "test is beating it in ranging regimes specifically."
    )

    # ===================================================================
    # 4. MOVING AVERAGE CROSSOVER
    # ===================================================================
    separator("4. MOVING AVERAGE CROSSOVER")
    ma_results = {}
    for fast, slow in [(5, 20), (10, 50), (20, 100)]:
        for ma_type in ["ema", "sma"]:
            strategy = MovingAverageCrossover(fast, slow, ma_type)
            metrics, returns, positions = engine.run_strategy(strategy, df)
            key = f"{ma_type}_{fast}_{slow}"
            ma_results[key] = (metrics, returns, positions)
            logger.info("--- %s %d/%d ---", ma_type.upper(), fast, slow)
            logger.info("  Sharpe: %.3f | Return: %.2f%% | MaxDD: %.2f%% | Trades/day: %.1f",
                         metrics.sharpe_ratio, metrics.total_return * 100,
                         metrics.max_drawdown * 100, metrics.trades_per_day)

    logger.info(
        "WHY HARD TO BEAT: MA crossover is the most widely known strategy, so "
        "if markets are efficient, its edge should be arbitraged away. At 5-min, "
        "fast crossovers (5/20) generate many trades with high costs; slow "
        "crossovers (20/100) miss intraday moves. Beating this proves the ML "
        "system adds value beyond simple trend direction."
    )

    # ===================================================================
    # 5. RANDOM ENTRY (TRUE NULL HYPOTHESIS)
    # ===================================================================
    separator("5. RANDOM ENTRY — TRUE NULL HYPOTHESIS")
    logger.info("Running %d random simulations (hold=3 bars)...", n_random_sims)
    random_dist = engine.run_random_distribution(
        df, n_simulations=n_random_sims, hold_bars=3, entry_probability=0.1,
    )
    logger.info("Random Strategy Distribution:")
    logger.info("  Mean Sharpe:  %.3f (+/- %.3f)", random_dist["mean_sharpe"], random_dist["std_sharpe"])
    logger.info("  95th pctile:  %.3f", random_dist["p95_sharpe"])
    logger.info("  99th pctile:  %.3f", random_dist["p99_sharpe"])
    logger.info("  Mean Return:  %.4f%% (+/- %.4f%%)", random_dist["mean_return"] * 100, random_dist["std_return"] * 100)

    # Test different hold periods
    for hold in [1, 2, 3, 6]:
        dist = engine.run_random_distribution(
            df, n_simulations=200, hold_bars=hold, entry_probability=0.1,
        )
        logger.info("  Hold=%d bars: Mean Sharpe=%.3f, P95=%.3f, P99=%.3f",
                     hold, dist["mean_sharpe"], dist["p95_sharpe"], dist["p99_sharpe"])

    logger.info(
        "WHY THIS IS THE MOST IMPORTANT BENCHMARK: If your strategy cannot beat "
        "the 99th percentile of random entry (Sharpe > %.3f), it has NO "
        "statistical edge. The p-value of your strategy vs this distribution "
        "is the single most important number in the entire system. "
        "Any Sharpe below %.3f is indistinguishable from luck.",
        random_dist["p99_sharpe"],
        random_dist["p95_sharpe"],
    )

    # ===================================================================
    # 6. VOLATILITY BREAKOUT
    # ===================================================================
    separator("6. VOLATILITY BREAKOUT")
    for atr_mult, hold in [(2.0, 3), (1.5, 5), (3.0, 2)]:
        strategy = VolatilityBreakout(atr_multiplier=atr_mult, hold_bars=hold)
        metrics, returns, positions = engine.run_strategy(strategy, df)
        logger.info("--- Breakout %.1fATR, hold %d ---", atr_mult, hold)
        logger.info("  Sharpe: %.3f | Return: %.2f%% | MaxDD: %.2f%% | Trades/day: %.1f",
                     metrics.sharpe_ratio, metrics.total_return * 100,
                     metrics.max_drawdown * 100, metrics.trades_per_day)

    logger.info(
        "WHY HARD TO BEAT: Volatility breakouts capture large moves (news, "
        "liquidation cascades). But false breakouts are common, and slippage "
        "is highest during volatile bars. Beating this proves the ML system "
        "distinguishes real breakouts from noise."
    )

    # ===================================================================
    # 7. FUNDING RATE
    # ===================================================================
    separator("7. FUNDING RATE STRATEGY")
    strategy = FundingRateStrategy(long_threshold=-0.0003, short_threshold=0.0005)
    metrics, returns, positions = engine.run_strategy(strategy, df)
    logger.info("\n%s", metrics.summary())
    sig = analyzer.test_sharpe_vs_zero(returns)
    logger.info("Significance: p=%.4f", sig["p_value"])
    logger.info(
        "WHY HARD TO BEAT: Funding rate captures crowd sentiment at 8h cycles. "
        "Contrarian funding is well-known and traded by many desks, reducing "
        "edge. But it's a slow signal — few trades, so low statistical power "
        "to measure performance. Beating this proves the ML system captures "
        "positioning dynamics better than a single threshold."
    )

    # ===================================================================
    # 8. ORDER BOOK IMBALANCE
    # ===================================================================
    separator("8. ORDER BOOK IMBALANCE")
    for threshold in [0.2, 0.3, 0.5]:
        strategy = OrderBookImbalance(long_threshold=threshold, short_threshold=-threshold)
        metrics, returns, positions = engine.run_strategy(strategy, df)
        logger.info("--- OBI threshold +/-%.1f ---", threshold)
        logger.info("  Sharpe: %.3f | Return: %.2f%% | MaxDD: %.2f%% | Trades/day: %.1f",
                     metrics.sharpe_ratio, metrics.total_return * 100,
                     metrics.max_drawdown * 100, metrics.trades_per_day)

    logger.info(
        "WHY HARD TO BEAT: OBI at 5-min aggregation is a smoothed version of "
        "the true microstructure signal (which operates at sub-second). The "
        "signal has predictive power but decays fast. Beating OBI proves the "
        "ML system extracts more from order flow — multi-level book features, "
        "trade flow imbalance, or absorption detection."
    )

    # ===================================================================
    # 9. CROSS-MARKET LEAD-LAG
    # ===================================================================
    separator("9. CROSS-MARKET LEAD-LAG (NQ -> BTC)")
    for lag in [1, 2, 3, 5]:
        strategy = CrossMarketLeadLag(lag_bars=lag)
        metrics, returns, positions = engine.run_strategy(strategy, df)
        # Directional accuracy
        raw_ret = df["close"].pct_change().fillna(0)
        pos = positions
        correct = ((raw_ret > 0) & (pos > 0)) | ((raw_ret < 0) & (pos < 0))
        accuracy = correct[pos != 0].mean() if (pos != 0).sum() > 0 else 0
        logger.info("--- NQ lag=%d bars ---", lag)
        logger.info("  Sharpe: %.3f | Return: %.2f%% | Accuracy: %.2f%%",
                     metrics.sharpe_ratio, metrics.total_return * 100, accuracy * 100)

    logger.info(
        "WHY HARD TO BEAT: NQ-BTC lead-lag is a documented phenomenon during "
        "US hours. But the lag is short (1-2 bars at 5-min) and diminishes "
        "outside US hours. Beating this proves the ML system captures "
        "cross-market dynamics beyond simple directional following."
    )

    # ===================================================================
    # 10. NAIVE ENSEMBLE
    # ===================================================================
    separator("10. NAIVE ENSEMBLE")
    # Pick top 3 individual strategies by Sharpe
    individual_results = {}
    candidates = [
        ("momentum_best", SimpleMomentum(lookback=best_momentum_n)),
        ("rsi_30_70", MeanReversionRSI(period=14, oversold=30, overbought=70)),
        ("ema_10_50", MovingAverageCrossover(10, 50, "ema")),
        ("vol_breakout", VolatilityBreakout(atr_multiplier=2.0, hold_bars=3)),
        ("lead_lag_1", CrossMarketLeadLag(lag_bars=1)),
    ]
    for name, strat in candidates:
        m, r, p = engine.run_strategy(strat, df)
        individual_results[name] = (m, r, p, strat)

    # Sort by Sharpe, take top 3
    sorted_results = sorted(individual_results.items(), key=lambda x: x[1][0].sharpe_ratio, reverse=True)
    top3_names = [s[0] for s in sorted_results[:3]]
    top3_strategies = [individual_results[n][3] for n in top3_names]
    logger.info("Top 3 individual strategies: %s", top3_names)
    for name in top3_names:
        logger.info("  %s: Sharpe=%.3f", name, individual_results[name][0].sharpe_ratio)

    ensemble = NaiveEnsemble(strategies=top3_strategies)
    metrics, returns, positions = engine.run_strategy(ensemble, df)
    logger.info("--- Naive Ensemble (majority vote of top 3) ---")
    logger.info("\n%s", metrics.summary())

    beats_any = all(
        metrics.sharpe_ratio > individual_results[n][0].sharpe_ratio
        for n in top3_names
    )
    logger.info("Ensemble beats ALL top-3 individuals: %s", beats_any)
    logger.info(
        "WHY HARDEST TO BEAT: Naive combination already captures diversification "
        "benefit. If majority vote beats each component, that proves signals are "
        "complementary. Beating the ensemble proves the ML system's learned "
        "weighting is superior to simple voting — the highest benchmark bar."
    )

    # ===================================================================
    # 11. ORACLE (UPPER BOUND)
    # ===================================================================
    separator("11. ORACLE STRATEGY (UPPER BOUND)")
    oracle = OracleStrategy()
    metrics, returns, positions = engine.run_strategy(oracle, df)
    logger.info("\n%s", metrics.summary())
    logger.info(
        "INTERPRETATION: This is the CEILING. No real strategy can beat this. "
        "Oracle Sharpe = %.3f after costs. Any strategy claiming higher has a bug. "
        "If your best strategy has Sharpe X, then X/%.3f = %.1f%% of theoretical "
        "maximum edge is captured.",
        metrics.sharpe_ratio,
        metrics.sharpe_ratio,
        0,  # Placeholder — would be filled with actual strategy sharpe
    )

    # ===================================================================
    # 12. FULL STATISTICAL ANALYSIS
    # ===================================================================
    separator("12. COMPREHENSIVE STATISTICAL ANALYSIS")

    # Run full suite
    suite = get_default_benchmark_suite()
    suite_results = engine.run_suite(df, suite)
    summary_df = engine.results_to_dataframe(suite_results)
    logger.info("BENCHMARK SUMMARY TABLE:")
    logger.info("\n%s", summary_df.to_string())

    # Statistical report
    report = analyzer.full_benchmark_report(suite_results)

    logger.info("\nSHARPE CONFIDENCE INTERVALS (95%%):")
    for name, r in report.items():
        if "error" in r:
            continue
        ci = r["sharpe_ci"]
        sharpe = ci["sharpe_ratio"]
        if "analytical" in ci:
            a = ci["analytical"]
            logger.info("  %-25s Sharpe=%.3f  CI=[%.3f, %.3f]",
                         name, sharpe, a["ci_lower"], a["ci_upper"])

    logger.info("\nSIGNIFICANCE TESTS (H0: Sharpe = 0):")
    p_values = {}
    for name, r in report.items():
        if "error" in r:
            continue
        sig = r["significance"]
        logger.info("  %-25s t=%.3f  p=%.4f  %s",
                     name, sig["test_statistic"], sig["p_value"],
                     "***" if sig["p_value"] < 0.01 else ("**" if sig["p_value"] < 0.05 else ""))
        p_values[name] = sig["p_value"]

    # Multiple testing correction
    if p_values:
        corrected = analyzer.multiple_testing_correction(p_values, method="holm")
        logger.info("\nHOLM-CORRECTED P-VALUES:")
        for name, c in sorted(corrected.items(), key=lambda x: x[1]["adjusted_p"]):
            logger.info("  %-25s original=%.4f  corrected=%.4f  %s",
                         name, c["original_p"], c["adjusted_p"],
                         "SIG" if c["significant"] else "")

    # Hardest benchmarks to beat
    logger.info("\nBENCHMARK DIFFICULTY RANKING (by Sharpe):")
    for idx, row in summary_df.head(10).iterrows():
        logger.info("  %-25s Sharpe=%.3f  Return=%.2f%%  MaxDD=%.2f%%",
                     idx, row["sharpe_ratio"], row["total_return"] * 100,
                     row["max_drawdown"] * 100)

    # --- What each benchmark proves ---
    separator("WHAT BEATING EACH BENCHMARK PROVES")
    proofs = {
        "Buy & Hold": "The ML system adds value beyond passive exposure — it times entries/exits.",
        "Simple Momentum": "The ML captures momentum better than raw return sign — likely through nonlinear interactions.",
        "RSI Mean Reversion": "The ML distinguishes regimes OR times mean-reversion entries better than fixed thresholds.",
        "MA Crossover": "The ML adds value beyond simple trend direction — better entry/exit timing or multi-factor conditioning.",
        "Random Entry": "THE ML HAS A REAL STATISTICAL EDGE. This is the minimum bar. Everything else is bonus.",
        "Volatility Breakout": "The ML distinguishes real breakouts from noise — likely using order flow or cross-market context.",
        "Funding Rate": "The ML captures crowd positioning better than a single contrarian threshold.",
        "Order Book Imbalance": "The ML extracts more from order flow than simple imbalance — multi-level features matter.",
        "Cross-Market Lead-Lag": "The ML captures cross-market dynamics beyond directional following — nonlinear conditional relationships.",
        "Naive Ensemble": "The ML's learned combination is superior to simple voting — THE HIGHEST BAR.",
        "Oracle": "If you beat this, you have a bug. This is the ceiling.",
    }
    for name, proof in proofs.items():
        logger.info("  %-22s %s", name + ":", proof)

    elapsed = time.time() - start_time
    separator(f"COMPLETE — {elapsed:.1f}s elapsed")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run benchmark strategies")
    parser.add_argument("--n-bars", type=int, default=288 * 30, help="Number of 5-min bars (default: 30 days)")
    parser.add_argument("--n-random-sims", type=int, default=1000, help="Number of random simulations")
    args = parser.parse_args()
    run_all_benchmarks(n_bars=args.n_bars, n_random_sims=args.n_random_sims)


if __name__ == "__main__":
    main()
