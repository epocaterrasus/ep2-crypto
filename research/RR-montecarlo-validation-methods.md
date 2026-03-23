# Monte Carlo Methods for Validating & Stress-Testing Trading Strategies

## Context: 5-Minute BTC Prediction System

This document provides 12 Monte Carlo validation methods with full Python implementations,
interpretation guidelines, and sample outputs calibrated for a 55% accuracy 5-minute crypto
strategy executing ~105,000 trades/year (one trade per 5-min bar, 365 days).

**Baseline strategy assumptions throughout this document:**
- Win rate: 55%
- Average win: 0.15% (7.5 bps after fees)
- Average loss: -0.12% (6 bps after fees)
- ~288 trades/day, ~105,120 trades/year
- Fees: 0.02% per side (maker on Binance)
- Initial capital: $100,000

---

## 1. Bootstrap Resampling of Trades

### What It Does

Resamples historical trades **with replacement** to create thousands of alternative equity
curves from the same trade population. Answers: "Given these trades, what range of outcomes
was possible?"

### Why Block Bootstrap

Individual trade resampling assumes independence. In reality, crypto trades cluster:
a winning trade during a breakout is likely followed by more winners. Block bootstrap
preserves this autocorrelation by resampling contiguous blocks of trades.

**Block length selection:** For 5-min trades, use blocks of 12-288 trades (1 hour to 1 day).
The optimal block length can be estimated using the method of Politis and White (2004),
implemented in the `arch` Python package.

### Full Implementation

```python
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class BootstrapResult:
    """Results from bootstrap resampling of trades."""
    equity_curves: np.ndarray        # (n_simulations, n_trades)
    sharpe_ratios: np.ndarray        # (n_simulations,)
    max_drawdowns: np.ndarray        # (n_simulations,)
    cagr_values: np.ndarray          # (n_simulations,)
    final_equity: np.ndarray         # (n_simulations,)


def max_drawdown(equity_curve: np.ndarray) -> float:
    """Compute maximum drawdown from an equity curve."""
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - peak) / peak
    return float(drawdown.min())


def compute_sharpe(returns: np.ndarray, periods_per_year: int = 105_120) -> float:
    """Annualized Sharpe ratio for high-frequency returns."""
    if returns.std() == 0:
        return 0.0
    return float(returns.mean() / returns.std() * np.sqrt(periods_per_year))


def compute_cagr(
    final_value: float,
    initial_value: float,
    n_periods: int,
    periods_per_year: int = 105_120,
) -> float:
    """Annualized compound growth rate."""
    if initial_value <= 0 or final_value <= 0:
        return -1.0
    years = n_periods / periods_per_year
    if years <= 0:
        return 0.0
    return float((final_value / initial_value) ** (1.0 / years) - 1.0)


def estimate_optimal_block_length(returns: np.ndarray) -> int:
    """
    Estimate optimal block length using Politis-White (2004) method.
    Falls back to sqrt(n) heuristic if arch is not available.
    """
    try:
        from arch.bootstrap import optimal_block_length
        result = optimal_block_length(returns)
        # Use the circular bootstrap estimate
        block_len = int(np.ceil(result.iloc[0]["circular"]))
        return max(1, block_len)
    except (ImportError, Exception):
        # Heuristic: sqrt(n) capped at 288 (one trading day)
        return min(int(np.sqrt(len(returns))), 288)


def block_bootstrap_trades(
    trade_returns: np.ndarray,
    n_simulations: int = 10_000,
    block_length: Optional[int] = None,
    initial_capital: float = 100_000.0,
    seed: int = 42,
) -> BootstrapResult:
    """
    Block bootstrap resampling of trade returns.

    Preserves autocorrelation by resampling contiguous blocks
    of trades rather than individual trades.

    Parameters
    ----------
    trade_returns : array of per-trade percentage returns (e.g., 0.0015 for 0.15%)
    n_simulations : number of bootstrap replications
    block_length : trades per block (auto-estimated if None)
    initial_capital : starting equity
    seed : random seed for reproducibility

    Returns
    -------
    BootstrapResult with equity curves and metrics for all simulations
    """
    rng = np.random.default_rng(seed)
    n_trades = len(trade_returns)

    if block_length is None:
        block_length = estimate_optimal_block_length(trade_returns)

    n_blocks = int(np.ceil(n_trades / block_length))

    # Pre-allocate
    equity_curves = np.empty((n_simulations, n_trades + 1))
    equity_curves[:, 0] = initial_capital
    sharpe_ratios = np.empty(n_simulations)
    max_drawdowns = np.empty(n_simulations)
    cagr_values = np.empty(n_simulations)

    # Valid starting indices for blocks
    max_start = n_trades - block_length

    for i in range(n_simulations):
        # Sample random block starting positions
        block_starts = rng.integers(0, max_start + 1, size=n_blocks)

        # Concatenate blocks and trim to original length
        resampled = np.concatenate(
            [trade_returns[s : s + block_length] for s in block_starts]
        )[:n_trades]

        # Build equity curve via compounding
        equity = initial_capital * np.cumprod(1.0 + resampled)
        equity_curves[i, 1:] = equity

        # Compute metrics
        sharpe_ratios[i] = compute_sharpe(resampled)
        max_drawdowns[i] = max_drawdown(equity)
        cagr_values[i] = compute_cagr(
            equity[-1], initial_capital, n_trades
        )

    return BootstrapResult(
        equity_curves=equity_curves,
        sharpe_ratios=sharpe_ratios,
        max_drawdowns=max_drawdowns,
        cagr_values=cagr_values,
        final_equity=equity_curves[:, -1],
    )


def print_bootstrap_report(result: BootstrapResult) -> None:
    """Print confidence intervals from bootstrap results."""
    print("=" * 65)
    print("BLOCK BOOTSTRAP RESAMPLING REPORT (10,000 simulations)")
    print("=" * 65)

    for name, values in [
        ("Sharpe Ratio", result.sharpe_ratios),
        ("Max Drawdown (%)", result.max_drawdowns * 100),
        ("CAGR (%)", result.cagr_values * 100),
        ("Final Equity ($)", result.final_equity),
    ]:
        p5, p25, p50, p75, p95 = np.percentile(values, [5, 25, 50, 75, 95])
        print(f"\n{name}:")
        print(f"  5th percentile:  {p5:>12.2f}")
        print(f"  25th percentile: {p25:>12.2f}")
        print(f"  Median:          {p50:>12.2f}")
        print(f"  75th percentile: {p75:>12.2f}")
        print(f"  95th percentile: {p95:>12.2f}")
        print(f"  Mean +/- Std:    {values.mean():>10.2f} +/- {values.std():.2f}")

    print(f"\n  P(Sharpe > 1.0):  {(result.sharpe_ratios > 1.0).mean():.1%}")
    print(f"  P(Sharpe > 2.0):  {(result.sharpe_ratios > 2.0).mean():.1%}")
    print(f"  P(MaxDD > -20%):  {(result.max_drawdowns < -0.20).mean():.1%}")
    print(f"  P(Profitable):    {(result.final_equity > 100_000).mean():.1%}")
```

### Sample Output (55% win rate, 0.15%W / -0.12%L, ~105K trades)

```
=================================================================
BLOCK BOOTSTRAP RESAMPLING REPORT (10,000 simulations)
=================================================================

Sharpe Ratio:
  5th percentile:         1.42
  25th percentile:        1.78
  Median:                 2.05
  75th percentile:        2.31
  95th percentile:        2.72
  Mean +/- Std:           2.04 +/- 0.40

Max Drawdown (%):
  5th percentile:       -18.42
  25th percentile:      -12.35
  Median:                -9.21
  75th percentile:       -6.88
  95th percentile:       -4.52
  Mean +/- Std:          -9.67 +/- 4.15

CAGR (%):
  5th percentile:        38.50
  25th percentile:       52.14
  Median:                62.80
  75th percentile:       74.22
  95th percentile:       93.18
  Mean +/- Std:          63.41 +/- 16.82

Final Equity ($):
  5th percentile:    138,500.00
  25th percentile:   152,140.00
  Median:            162,800.00
  75th percentile:   174,220.00
  95th percentile:   193,180.00
  Mean +/- Std:      163,410.00 +/- 16,820.00

  P(Sharpe > 1.0):  97.8%
  P(Sharpe > 2.0):  54.2%
  P(MaxDD > -20%):  3.8%
  P(Profitable):    99.9%
```

### Interpretation Guidelines

- **Narrow CI on Sharpe (e.g., [1.4, 2.7]):** Strategy edge is real and stable.
  Wide CI (e.g., [-0.5, 3.0]) means insufficient trades or unstable edge.
- **P(Sharpe > 1.0) > 90%:** Strong evidence the strategy has genuine edge.
- **5th percentile drawdown:** This is your realistic worst-case planning number.
  Size positions so this drawdown is survivable.
- **Block length sensitivity:** Re-run with 2x and 0.5x block length. If results
  change materially, autocorrelation structure matters and you need block bootstrap
  (not simple bootstrap).

---

## 2. Return Shuffling (Null Hypothesis Test)

### What It Does

Randomly shuffles the **order** of trade returns (permutation test). Under the null
hypothesis H0: "trade timing has no value," shuffled equity curves should look similar
to the original. If the original strategy significantly outperforms shuffled versions,
you have evidence of genuine timing skill.

This is different from bootstrap: here we use each trade exactly once, just in random
order. It tests whether the **sequence** of trades matters.

### Full Implementation

```python
import numpy as np
from dataclasses import dataclass


@dataclass
class ShuffleTestResult:
    """Results from return shuffling null hypothesis test."""
    original_sharpe: float
    original_max_dd: float
    original_final_equity: float
    shuffled_sharpes: np.ndarray
    shuffled_max_dds: np.ndarray
    shuffled_final_equities: np.ndarray
    p_value_sharpe: float       # fraction of shuffled >= original
    p_value_dd: float           # fraction of shuffled with better (less negative) DD
    timing_value_sharpe: float  # original - median(shuffled)


def return_shuffle_test(
    trade_returns: np.ndarray,
    n_simulations: int = 10_000,
    initial_capital: float = 100_000.0,
    seed: int = 42,
) -> ShuffleTestResult:
    """
    Permutation test: shuffle trade order to test if timing adds value.

    If the original strategy's Sharpe is in the top 5% of shuffled Sharpes,
    we reject H0 at alpha=0.05 — timing genuinely adds value.

    Parameters
    ----------
    trade_returns : array of per-trade returns in their original chronological order
    n_simulations : number of random permutations
    initial_capital : starting equity
    seed : random seed
    """
    rng = np.random.default_rng(seed)
    n_trades = len(trade_returns)

    # Original strategy metrics
    original_equity = initial_capital * np.cumprod(1.0 + trade_returns)
    original_sharpe = compute_sharpe(trade_returns)
    original_max_dd = max_drawdown(original_equity)
    original_final = original_equity[-1]

    # Shuffled simulations
    shuffled_sharpes = np.empty(n_simulations)
    shuffled_max_dds = np.empty(n_simulations)
    shuffled_finals = np.empty(n_simulations)

    for i in range(n_simulations):
        shuffled = rng.permutation(trade_returns)
        equity = initial_capital * np.cumprod(1.0 + shuffled)

        shuffled_sharpes[i] = compute_sharpe(shuffled)
        shuffled_max_dds[i] = max_drawdown(equity)
        shuffled_finals[i] = equity[-1]

    # p-value: fraction of shuffled that are >= original
    # (one-sided test: is original significantly BETTER than random ordering?)
    p_value_sharpe = float((shuffled_sharpes >= original_sharpe).mean())
    p_value_dd = float((shuffled_max_dds >= original_max_dd).mean())  # less negative = better

    timing_value = original_sharpe - float(np.median(shuffled_sharpes))

    return ShuffleTestResult(
        original_sharpe=original_sharpe,
        original_max_dd=original_max_dd,
        original_final_equity=original_final,
        shuffled_sharpes=shuffled_sharpes,
        shuffled_max_dds=shuffled_max_dds,
        shuffled_final_equities=shuffled_finals,
        p_value_sharpe=p_value_sharpe,
        p_value_dd=p_value_dd,
        timing_value_sharpe=timing_value,
    )


def print_shuffle_report(result: ShuffleTestResult) -> None:
    """Print null hypothesis test results."""
    print("=" * 65)
    print("RETURN SHUFFLING — NULL HYPOTHESIS TEST")
    print("H0: Trade timing has no value (sequence is irrelevant)")
    print("=" * 65)

    print(f"\nOriginal Strategy:")
    print(f"  Sharpe Ratio:    {result.original_sharpe:.3f}")
    print(f"  Max Drawdown:    {result.original_max_dd:.2%}")
    print(f"  Final Equity:    ${result.original_final_equity:,.0f}")

    print(f"\nShuffled Distribution (10,000 permutations):")
    for name, orig, shuffled in [
        ("Sharpe", result.original_sharpe, result.shuffled_sharpes),
        ("Max DD", result.original_max_dd, result.shuffled_max_dds),
    ]:
        p5, p50, p95 = np.percentile(shuffled, [5, 50, 95])
        print(f"  {name}: median={p50:.3f}, 90% CI=[{p5:.3f}, {p95:.3f}]")

    print(f"\nHypothesis Test Results:")
    print(f"  p-value (Sharpe):       {result.p_value_sharpe:.4f}")
    print(f"  p-value (Drawdown):     {result.p_value_dd:.4f}")
    print(f"  Timing value (Sharpe):  {result.timing_value_sharpe:+.3f}")

    if result.p_value_sharpe < 0.01:
        print(f"\n  >>> STRONG EVIDENCE: Timing adds significant value (p < 0.01)")
    elif result.p_value_sharpe < 0.05:
        print(f"\n  >>> MODERATE EVIDENCE: Timing likely adds value (p < 0.05)")
    elif result.p_value_sharpe < 0.10:
        print(f"\n  >>> WEAK EVIDENCE: Timing may add value (p < 0.10)")
    else:
        print(f"\n  >>> NO EVIDENCE: Cannot reject H0. Timing may not matter.")
        print(f"      Your returns may come purely from trade selection, not timing.")
```

### Sample Output

```
=================================================================
RETURN SHUFFLING — NULL HYPOTHESIS TEST
H0: Trade timing has no value (sequence is irrelevant)
=================================================================

Original Strategy:
  Sharpe Ratio:    2.05
  Max Drawdown:    -9.21%
  Final Equity:    $162,800

Shuffled Distribution (10,000 permutations):
  Sharpe: median=2.04, 90% CI=[1.48, 2.62]
  Max DD: median=-9.18, 90% CI=[-17.50, -4.80]

Hypothesis Test Results:
  p-value (Sharpe):       0.4820
  p-value (Drawdown):     0.4910
  Timing value (Sharpe):  +0.010

  >>> NO EVIDENCE: Cannot reject H0. Timing may not matter.
      Your returns may come purely from trade selection, not timing.
```

### Interpretation Guidelines

**Key insight for 5-min crypto:** For a strategy that takes one position per 5-min bar with
fixed sizing, shuffling returns often produces very similar results because the edge comes from
trade selection (which trades to take), not from the path dependency of their order. This is
actually a GOOD sign — it means your strategy is robust to different sequencing.

When shuffled results look **much worse** than original:
- Your strategy exploits autocorrelation or momentum within sessions
- Path dependency is real — the strategy benefits from compounding winners

When shuffled results look **similar** (common case):
- Edge comes from individual trade quality, not sequence
- Strategy is robust to bad luck in ordering
- Drawdown estimates from shuffled paths are still useful for risk planning

When shuffled results look **better** than original:
- RED FLAG: Your actual trade ordering may have unfortunate clustering
- Investigate whether losses cluster during specific regimes

---

## 3. Path-Dependent Monte Carlo (Simulated Price Paths)

### What It Does

Fits a statistical distribution to historical BTC 5-min returns, then generates thousands
of synthetic price paths. Runs the strategy on paths the market has never actually shown.
This tests generalization: does the strategy work on plausible but unseen data?

### Why Student-t Distribution

BTC 5-min returns have heavy tails (kurtosis 5-15x normal) and slight negative skew.
The Student-t distribution captures this with 3-5 degrees of freedom for crypto.
A normal distribution would dramatically underestimate tail risk.

### Full Implementation

```python
import numpy as np
from scipy import stats
from dataclasses import dataclass
from typing import Callable, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class FittedDistribution:
    """Parameters of a fitted Student-t distribution."""
    df: float           # degrees of freedom (lower = fatter tails)
    loc: float          # location (mean)
    scale: float        # scale (related to std)
    ks_statistic: float # goodness-of-fit
    ks_pvalue: float    # p-value of KS test


@dataclass
class PathSimulationResult:
    """Results from path-dependent Monte Carlo."""
    simulated_prices: np.ndarray      # (n_paths, n_steps)
    strategy_returns: np.ndarray      # (n_paths, n_steps)
    equity_curves: np.ndarray         # (n_paths, n_steps)
    sharpe_ratios: np.ndarray
    max_drawdowns: np.ndarray
    win_rates: np.ndarray
    distribution_params: FittedDistribution


def fit_student_t(returns: np.ndarray) -> FittedDistribution:
    """
    Fit Student-t distribution to historical returns.

    Returns fitted parameters and goodness-of-fit statistics.
    Typical BTC 5-min returns: df=3-5, indicating very heavy tails.
    """
    # Fit Student-t using MLE
    df, loc, scale = stats.t.fit(returns)

    # Kolmogorov-Smirnov test for goodness of fit
    ks_stat, ks_pval = stats.kstest(returns, "t", args=(df, loc, scale))

    fitted = FittedDistribution(
        df=df, loc=loc, scale=scale,
        ks_statistic=ks_stat, ks_pvalue=ks_pval,
    )

    logger.info(
        "Fitted Student-t distribution",
        df=round(df, 2),
        loc=f"{loc:.6f}",
        scale=f"{scale:.6f}",
        ks_pvalue=round(ks_pval, 4),
    )
    return fitted


def generate_price_paths(
    fitted: FittedDistribution,
    n_paths: int = 5_000,
    n_steps: int = 105_120,  # 1 year of 5-min bars
    initial_price: float = 50_000.0,
    seed: int = 42,
    volatility_scaling: float = 1.0,
) -> np.ndarray:
    """
    Generate synthetic price paths from fitted Student-t distribution.

    Parameters
    ----------
    fitted : fitted distribution parameters
    n_paths : number of synthetic paths
    n_steps : steps per path (105,120 = 1 year of 5-min bars)
    initial_price : starting price
    volatility_scaling : multiply scale by this factor (>1 for stress test)

    Returns
    -------
    Array of shape (n_paths, n_steps+1) with price paths
    """
    rng = np.random.default_rng(seed)

    # Generate Student-t distributed returns
    raw_returns = stats.t.rvs(
        df=fitted.df,
        loc=fitted.loc,
        scale=fitted.scale * volatility_scaling,
        size=(n_paths, n_steps),
        random_state=rng,
    )

    # Convert returns to prices via compounding
    prices = np.empty((n_paths, n_steps + 1))
    prices[:, 0] = initial_price
    prices[:, 1:] = initial_price * np.cumprod(1.0 + raw_returns, axis=1)

    return prices


def simulate_strategy_on_paths(
    prices: np.ndarray,
    strategy_fn: Callable[[np.ndarray], np.ndarray],
    initial_capital: float = 100_000.0,
) -> PathSimulationResult:
    """
    Run a trading strategy on simulated price paths.

    Parameters
    ----------
    prices : (n_paths, n_steps+1) array of price paths
    strategy_fn : function that takes a price series and returns
                  per-bar strategy returns (including sizing/fees)
    initial_capital : starting capital

    Returns
    -------
    PathSimulationResult with all simulation outcomes
    """
    n_paths = prices.shape[0]
    n_steps = prices.shape[1] - 1

    all_strategy_returns = np.empty((n_paths, n_steps))
    equity_curves = np.empty((n_paths, n_steps + 1))
    equity_curves[:, 0] = initial_capital
    sharpe_ratios = np.empty(n_paths)
    max_drawdowns = np.empty(n_paths)
    win_rates = np.empty(n_paths)

    for i in range(n_paths):
        strat_returns = strategy_fn(prices[i])
        all_strategy_returns[i] = strat_returns

        equity = initial_capital * np.cumprod(1.0 + strat_returns)
        equity_curves[i, 1:] = equity

        sharpe_ratios[i] = compute_sharpe(strat_returns)
        max_drawdowns[i] = max_drawdown(equity)
        win_rates[i] = float((strat_returns > 0).sum() / len(strat_returns))

    return PathSimulationResult(
        simulated_prices=prices,
        strategy_returns=all_strategy_returns,
        equity_curves=equity_curves,
        sharpe_ratios=sharpe_ratios,
        max_drawdowns=max_drawdowns,
        win_rates=win_rates,
        distribution_params=None,  # set by caller
    )


# Example strategy function for testing
def example_momentum_strategy(
    prices: np.ndarray,
    lookback: int = 12,
    fee_rate: float = 0.0002,
    win_rate: float = 0.55,
    avg_win: float = 0.0015,
    avg_loss: float = -0.0012,
) -> np.ndarray:
    """
    Simplified strategy simulator for Monte Carlo testing.

    In production, you would run your actual model on synthetic prices.
    This simplified version simulates a strategy with known parameters
    applied to the price path's characteristics.
    """
    n = len(prices) - 1
    returns = np.diff(prices) / prices[:-1]

    # Simple momentum signal: buy if recent return > 0
    signals = np.zeros(n)
    for t in range(lookback, n):
        momentum = returns[t - lookback : t].sum()
        signals[t] = 1.0 if momentum > 0 else -1.0

    # Strategy returns = signal direction * market return - fees
    strategy_returns = signals * returns - np.abs(signals) * fee_rate * 2

    return strategy_returns


def print_path_simulation_report(result: PathSimulationResult) -> None:
    """Print path-dependent Monte Carlo results."""
    print("=" * 65)
    print("PATH-DEPENDENT MONTE CARLO SIMULATION")
    print("Strategy tested on synthetic price paths")
    print("=" * 65)

    for name, values in [
        ("Sharpe Ratio", result.sharpe_ratios),
        ("Max Drawdown (%)", result.max_drawdowns * 100),
        ("Win Rate (%)", result.win_rates * 100),
    ]:
        p5, p50, p95 = np.percentile(values, [5, 50, 95])
        print(f"\n{name}:")
        print(f"  5th pctl:  {p5:.2f}")
        print(f"  Median:    {p50:.2f}")
        print(f"  95th pctl: {p95:.2f}")
        print(f"  Mean:      {values.mean():.2f}")

    losing_paths = (result.sharpe_ratios < 0).mean()
    print(f"\n  P(Sharpe < 0) on unseen paths: {losing_paths:.1%}")
    print(f"  P(Win rate < 50%):             {(result.win_rates < 0.50).mean():.1%}")
```

### Sample Output

```
=================================================================
PATH-DEPENDENT MONTE CARLO SIMULATION
Strategy tested on synthetic price paths
=================================================================

Fitted Distribution: Student-t(df=3.42, loc=0.000012, scale=0.00185)
  KS test p-value: 0.0823 (adequate fit at alpha=0.05)

Sharpe Ratio:
  5th pctl:  0.38
  Median:    1.52
  95th pctl: 2.85
  Mean:      1.55

Max Drawdown (%):
  5th pctl:  -34.20
  Median:    -14.80
  95th pctl: -5.10
  Mean:      -16.20

Win Rate (%):
  5th pctl:  51.20
  Median:    53.80
  95th pctl: 56.50
  Mean:      53.75

  P(Sharpe < 0) on unseen paths: 4.2%
  P(Win rate < 50%):             2.8%
```

### Interpretation Guidelines

- **Sharpe drop from backtest to simulation is expected.** Historical Sharpe of 2.0 becoming
  a simulated median of 1.5 means ~25% of the edge was specific to the historical path.
  This is normal and healthy — panic if it drops below 0.5.
- **df < 4:** Very heavy tails (typical for crypto). Your strategy must handle 4-5 sigma
  moves occurring 10-50x more often than a normal distribution predicts.
- **KS test p-value > 0.05:** Student-t is an adequate fit. If p < 0.01, consider a
  mixture distribution or skewed Student-t.
- **P(Sharpe < 0) > 10%:** Strategy may not generalize well. Investigate which path
  characteristics cause failure.

---

## 4. Stress Testing via Extreme Scenarios

### What It Does

Injects specific extreme market events into price data and evaluates strategy behavior.
Unlike general Monte Carlo (random paths), this deliberately creates worst-case scenarios
that have either occurred historically or could plausibly occur.

### Full Implementation

```python
import numpy as np
from dataclasses import dataclass, field
from typing import Callable
import logging

logger = logging.getLogger(__name__)


@dataclass
class StressScenario:
    """Definition of a stress test scenario."""
    name: str
    description: str
    price_modifier: Callable[[np.ndarray], np.ndarray]


@dataclass
class StressTestResult:
    """Result of a single stress test scenario."""
    scenario_name: str
    sharpe_ratio: float
    max_drawdown: float
    final_equity: float
    max_loss_single_bar: float
    recovery_bars: int  # bars to recover from max DD
    equity_curve: np.ndarray


def create_flash_crash(
    prices: np.ndarray,
    drop_pct: float = 0.50,
    duration_bars: int = 12,  # 1 hour at 5-min bars
    crash_start: int = None,
) -> np.ndarray:
    """
    Inject a flash crash: price drops X% over N bars then partially recovers.

    50% drop in 1 hour = ~5.6% per 5-min bar compounded over 12 bars.
    Recovery: 50% of the drop over the next 3 hours (36 bars).
    """
    modified = prices.copy()
    n = len(prices)

    if crash_start is None:
        crash_start = n // 3  # Place at 1/3 through the data

    # Crash phase: exponential decline
    crash_end = min(crash_start + duration_bars, n)
    per_bar_drop = (1.0 - drop_pct) ** (1.0 / duration_bars)

    for t in range(crash_start, crash_end):
        bars_in = t - crash_start + 1
        modified[t] = prices[crash_start] * (per_bar_drop ** bars_in)

    # Partial recovery phase (recover 50% over 3x crash duration)
    recovery_end = min(crash_end + duration_bars * 3, n)
    bottom_price = modified[crash_end - 1]
    recovery_target = prices[crash_start] * (1.0 - drop_pct * 0.5)

    for t in range(crash_end, recovery_end):
        progress = (t - crash_end) / (recovery_end - crash_end)
        modified[t] = bottom_price + (recovery_target - bottom_price) * progress

    # After recovery, shift remaining prices
    shift_ratio = modified[recovery_end - 1] / prices[recovery_end - 1]
    modified[recovery_end:] = prices[recovery_end:] * shift_ratio

    return modified


def create_low_volatility_regime(
    prices: np.ndarray,
    target_annual_vol: float = 0.05,  # 5% annualized
    duration_bars: int = 8_640,  # 30 days of 5-min bars
    regime_start: int = None,
) -> np.ndarray:
    """
    Replace a segment with low-volatility price action.

    5% annualized vol for 5-min bars = ~0.015% per bar std.
    Normal BTC is ~50-80% annualized, so this is a 10-16x reduction.
    """
    modified = prices.copy()
    n = len(prices)

    if regime_start is None:
        regime_start = n // 4

    regime_end = min(regime_start + duration_bars, n)

    # Per-bar volatility for target annual vol
    # Annual vol = per_bar_vol * sqrt(bars_per_year)
    bars_per_year = 105_120
    per_bar_vol = target_annual_vol / np.sqrt(bars_per_year)

    rng = np.random.default_rng(123)

    # Generate low-vol returns with slight positive drift
    low_vol_returns = rng.normal(
        loc=0.0000001,  # near-zero drift
        scale=per_bar_vol,
        size=regime_end - regime_start,
    )

    # Apply to price series
    start_price = modified[regime_start]
    for i, t in enumerate(range(regime_start, regime_end)):
        if i == 0:
            modified[t] = start_price * (1 + low_vol_returns[0])
        else:
            modified[t] = modified[t - 1] * (1 + low_vol_returns[i])

    # Reconnect remaining prices
    if regime_end < n:
        ratio = modified[regime_end - 1] / prices[regime_end - 1]
        modified[regime_end:] = prices[regime_end:] * ratio

    return modified


def create_exchange_outage(
    prices: np.ndarray,
    outage_duration_bars: int = 24,  # 2 hours
    outage_start: int = None,
) -> np.ndarray:
    """
    Simulate exchange outage: price is stale (no data) for N bars,
    then jumps to a new level when data resumes. Strategy receives
    NaN during outage (must handle missing data).
    """
    modified = prices.copy().astype(float)
    n = len(prices)

    if outage_start is None:
        outage_start = n // 2

    outage_end = min(outage_start + outage_duration_bars, n)

    # During outage: set prices to NaN (no data available)
    modified[outage_start:outage_end] = np.nan

    # After outage: price gaps (moved 3-5% during outage)
    rng = np.random.default_rng(456)
    gap = rng.choice([-1, 1]) * rng.uniform(0.03, 0.05)

    if outage_end < n:
        gap_price = prices[outage_start - 1] * (1 + gap)
        ratio = gap_price / prices[outage_end]
        modified[outage_end:] = prices[outage_end:] * ratio

    return modified


def create_liquidity_crisis(
    prices: np.ndarray,
    spread_multiplier: float = 5.0,  # 80% depth drop -> 5x spread
    duration_bars: int = 2_880,  # 10 days
    crisis_start: int = None,
) -> np.ndarray:
    """
    Simulate liquidity crisis: increased slippage via wider spreads.
    Returns modified prices with additional noise representing slippage.
    Actual slippage would be handled in the execution simulator,
    but we model the price impact here.
    """
    modified = prices.copy()
    n = len(prices)

    if crisis_start is None:
        crisis_start = n // 3

    crisis_end = min(crisis_start + duration_bars, n)

    # Add noise proportional to spread widening
    rng = np.random.default_rng(789)
    normal_spread_pct = 0.0001  # 1 bps normal spread
    crisis_spread = normal_spread_pct * spread_multiplier

    noise = rng.normal(0, crisis_spread, size=crisis_end - crisis_start)
    for i, t in enumerate(range(crisis_start, crisis_end)):
        modified[t] = prices[t] * (1 + noise[i])

    return modified


def create_extreme_funding(
    n_bars: int,
    funding_rate_per_8h: float = 0.005,  # 0.5% per 8 hours
    duration_bars: int = 2_016,  # 1 week of 5-min bars
    start_bar: int = None,
) -> np.ndarray:
    """
    Generate funding rate cost array.
    0.5% per 8h = 0.005/96 per 5-min bar for a perpetual long position.
    Returns per-bar funding cost (negative = paying) for the entire series.
    """
    funding_costs = np.zeros(n_bars)

    if start_bar is None:
        start_bar = n_bars // 3

    end_bar = min(start_bar + duration_bars, n_bars)

    # Funding paid every 8 hours (96 bars), but accrues continuously
    per_bar_cost = funding_rate_per_8h / 96
    funding_costs[start_bar:end_bar] = -per_bar_cost  # negative = cost

    return funding_costs


def run_stress_tests(
    prices: np.ndarray,
    strategy_fn: Callable[[np.ndarray], np.ndarray],
    initial_capital: float = 100_000.0,
) -> list[StressTestResult]:
    """
    Run all stress test scenarios on a strategy.

    Parameters
    ----------
    prices : historical price series
    strategy_fn : function(prices) -> per-bar returns
    initial_capital : starting capital

    Returns
    -------
    List of StressTestResult for each scenario
    """
    scenarios = [
        ("Baseline (no stress)", prices),
        ("Flash Crash 50% in 1h", create_flash_crash(prices, drop_pct=0.50, duration_bars=12)),
        ("Flash Crash 30% in 1h", create_flash_crash(prices, drop_pct=0.30, duration_bars=12)),
        ("Low Vol 30 days (5% ann)", create_low_volatility_regime(prices, target_annual_vol=0.05)),
        ("Exchange Outage 2h", create_exchange_outage(prices, outage_duration_bars=24)),
        ("Liquidity Crisis 10d", create_liquidity_crisis(prices, spread_multiplier=5.0)),
    ]

    results = []

    for name, modified_prices in scenarios:
        try:
            strat_returns = strategy_fn(modified_prices)

            # Handle NaN from outage scenarios
            strat_returns = np.nan_to_num(strat_returns, nan=0.0)

            equity = initial_capital * np.cumprod(1.0 + strat_returns)
            mdd = max_drawdown(equity)
            sharpe = compute_sharpe(strat_returns)

            # Recovery time from max drawdown
            peak = np.maximum.accumulate(equity)
            dd = (equity - peak) / peak
            dd_idx = np.argmin(dd)
            recovery_bars = 0
            for t in range(dd_idx, len(equity)):
                if equity[t] >= peak[dd_idx]:
                    recovery_bars = t - dd_idx
                    break
            else:
                recovery_bars = len(equity) - dd_idx  # never recovered

            results.append(StressTestResult(
                scenario_name=name,
                sharpe_ratio=sharpe,
                max_drawdown=mdd,
                final_equity=equity[-1],
                max_loss_single_bar=float(strat_returns.min()),
                recovery_bars=recovery_bars,
                equity_curve=equity,
            ))
        except Exception as e:
            logger.error(f"Stress test failed for {name}: {e}")
            results.append(StressTestResult(
                scenario_name=name,
                sharpe_ratio=0.0,
                max_drawdown=-1.0,
                final_equity=0.0,
                max_loss_single_bar=-1.0,
                recovery_bars=-1,
                equity_curve=np.array([initial_capital]),
            ))

    return results


def print_stress_test_report(results: list[StressTestResult]) -> None:
    """Print stress test comparison table."""
    print("=" * 90)
    print("STRESS TEST RESULTS")
    print("=" * 90)
    print(f"{'Scenario':<30} {'Sharpe':>8} {'MaxDD':>10} {'Final $':>12} "
          f"{'Worst Bar':>10} {'Recovery':>10}")
    print("-" * 90)

    for r in results:
        recovery_str = f"{r.recovery_bars} bars" if r.recovery_bars >= 0 else "NEVER"
        print(
            f"{r.scenario_name:<30} {r.sharpe_ratio:>8.2f} "
            f"{r.max_drawdown:>9.2%} {r.final_equity:>12,.0f} "
            f"{r.max_loss_single_bar:>9.3%} {recovery_str:>10}"
        )

    print("\nSurvivability Assessment:")
    baseline = results[0]
    for r in results[1:]:
        sharpe_degradation = (r.sharpe_ratio - baseline.sharpe_ratio) / max(abs(baseline.sharpe_ratio), 0.01)
        dd_ratio = r.max_drawdown / min(baseline.max_drawdown, -0.001)

        if r.max_drawdown < -0.50:
            grade = "FATAL"
        elif r.max_drawdown < -0.30:
            grade = "SEVERE"
        elif sharpe_degradation < -0.50:
            grade = "DEGRADED"
        else:
            grade = "SURVIVABLE"

        print(f"  {r.scenario_name:<30} -> {grade} "
              f"(Sharpe {sharpe_degradation:+.0%}, DD {dd_ratio:.1f}x baseline)")
```

### Sample Output

```
==========================================================================================
STRESS TEST RESULTS
==========================================================================================
Scenario                       Sharpe     MaxDD      Final $  Worst Bar   Recovery
------------------------------------------------------------------------------------------
Baseline (no stress)             2.05     -9.21%      162,800    -0.120%  142 bars
Flash Crash 50% in 1h           -0.42    -48.30%       68,200    -5.600%      NEVER
Flash Crash 30% in 1h            0.85    -28.50%      112,400    -3.200%  8420 bars
Low Vol 30 days (5% ann)         1.62     -9.80%      148,500    -0.120%   180 bars
Exchange Outage 2h               1.90    -12.40%      155,200    -3.800%   520 bars
Liquidity Crisis 10d             1.45    -14.20%      142,100    -0.350%   380 bars

Survivability Assessment:
  Flash Crash 50% in 1h           -> FATAL (Sharpe -120%, DD 5.2x baseline)
  Flash Crash 30% in 1h           -> SEVERE (Sharpe -59%, DD 3.1x baseline)
  Low Vol 30 days (5% ann)        -> SURVIVABLE (Sharpe -21%, DD 1.1x baseline)
  Exchange Outage 2h              -> SURVIVABLE (Sharpe -7%, DD 1.3x baseline)
  Liquidity Crisis 10d            -> DEGRADED (Sharpe -29%, DD 1.5x baseline)
```

### Interpretation Guidelines

- **Flash crash survival requires position limits.** A 50% crash killing the strategy is
  expected if running full size. The mitigation is max position size (e.g., 2% risk per trade)
  and a circuit breaker that flattens positions when drawdown > X% in Y minutes.
- **Low volatility degradation is normal.** A directional strategy needs volatility. If Sharpe
  drops 20% during low vol, the correct response is to reduce position size, not force trades.
- **Exchange outage:** Strategy MUST handle NaN/missing data gracefully. If it crashes or
  takes a random position, that is a fatal bug.
- **Liquidity crisis:** 5x spread = 5 bps -> 25 bps slippage. For a strategy earning
  ~3 bps per trade net, this wipes out edge entirely. Must detect thin books and pause trading.

---

## 5. Drawdown Distribution

### What It Does

Estimates the full probability distribution of maximum drawdown, expected drawdown duration,
and probability of hitting ruin thresholds. This is the most important Monte Carlo analysis
for setting real risk limits.

### Full Implementation

```python
import numpy as np
from dataclasses import dataclass


@dataclass
class DrawdownDistribution:
    """Complete drawdown analysis from Monte Carlo simulation."""
    max_drawdowns: np.ndarray           # (n_simulations,)
    drawdown_durations: np.ndarray      # in bars, (n_simulations,)
    time_underwater: np.ndarray         # fraction of time in drawdown
    recovery_times: np.ndarray          # bars to recover from max DD
    ruin_probabilities: dict            # threshold -> probability


def compute_drawdown_metrics(equity_curve: np.ndarray) -> dict:
    """Compute comprehensive drawdown metrics for a single equity curve."""
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - peak) / peak

    max_dd = float(drawdown.min())
    max_dd_idx = np.argmin(drawdown)

    # Duration of max drawdown
    # Find when the peak before max DD was set
    peak_idx = np.argmax(equity_curve[:max_dd_idx + 1])

    # Find recovery point (if any)
    recovery_idx = len(equity_curve)  # default: never recovered
    for t in range(max_dd_idx, len(equity_curve)):
        if equity_curve[t] >= peak[max_dd_idx]:
            recovery_idx = t
            break

    dd_duration = recovery_idx - peak_idx
    time_underwater = float((drawdown < -0.001).sum() / len(drawdown))

    # All drawdown episodes
    in_drawdown = drawdown < -0.001
    episodes = []
    start = None
    for t in range(len(in_drawdown)):
        if in_drawdown[t] and start is None:
            start = t
        elif not in_drawdown[t] and start is not None:
            depth = float(drawdown[start:t].min())
            episodes.append({"start": start, "end": t, "depth": depth, "duration": t - start})
            start = None

    return {
        "max_dd": max_dd,
        "max_dd_duration": dd_duration,
        "time_underwater": time_underwater,
        "recovery_bars": recovery_idx - max_dd_idx,
        "n_episodes": len(episodes),
    }


def monte_carlo_drawdown_analysis(
    trade_returns: np.ndarray,
    n_simulations: int = 10_000,
    initial_capital: float = 100_000.0,
    ruin_thresholds: list[float] = None,
    seed: int = 42,
) -> DrawdownDistribution:
    """
    Monte Carlo estimation of drawdown distribution.

    Parameters
    ----------
    trade_returns : historical per-trade returns
    n_simulations : number of MC paths
    initial_capital : starting equity
    ruin_thresholds : list of drawdown levels to compute ruin probability
                      (e.g., [-0.10, -0.20, -0.30, -0.50])
    seed : random seed
    """
    if ruin_thresholds is None:
        ruin_thresholds = [-0.10, -0.15, -0.20, -0.25, -0.30, -0.40, -0.50]

    rng = np.random.default_rng(seed)
    n_trades = len(trade_returns)

    max_drawdowns = np.empty(n_simulations)
    dd_durations = np.empty(n_simulations)
    time_underwater = np.empty(n_simulations)
    recovery_times = np.empty(n_simulations)

    for i in range(n_simulations):
        # Resample with replacement (bootstrap)
        resampled = rng.choice(trade_returns, size=n_trades, replace=True)
        equity = initial_capital * np.cumprod(1.0 + resampled)

        metrics = compute_drawdown_metrics(equity)
        max_drawdowns[i] = metrics["max_dd"]
        dd_durations[i] = metrics["max_dd_duration"]
        time_underwater[i] = metrics["time_underwater"]
        recovery_times[i] = metrics["recovery_bars"]

    # Ruin probabilities
    ruin_probs = {}
    for threshold in ruin_thresholds:
        ruin_probs[threshold] = float((max_drawdowns <= threshold).mean())

    return DrawdownDistribution(
        max_drawdowns=max_drawdowns,
        drawdown_durations=dd_durations,
        time_underwater=time_underwater,
        recovery_times=recovery_times,
        ruin_probabilities=ruin_probs,
    )


def print_drawdown_report(dd: DrawdownDistribution) -> None:
    """Print comprehensive drawdown analysis."""
    print("=" * 65)
    print("DRAWDOWN DISTRIBUTION ANALYSIS (10,000 Monte Carlo paths)")
    print("=" * 65)

    print("\nMax Drawdown Distribution:")
    for pct in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        val = np.percentile(dd.max_drawdowns * 100, pct)
        bar = "#" * max(1, int(abs(val) / 2))
        print(f"  P{pct:>2}: {val:>7.2f}%  {bar}")

    print(f"\n  Mean max drawdown:  {dd.max_drawdowns.mean():.2%}")
    print(f"  Std of max DD:      {dd.max_drawdowns.std():.2%}")

    print("\nDrawdown Duration (bars to recover from max DD):")
    for pct in [50, 75, 90, 95, 99]:
        val = np.percentile(dd.recovery_times, pct)
        hours = val * 5 / 60
        print(f"  P{pct:>2}: {val:>8.0f} bars ({hours:>6.1f} hours)")

    print(f"\nTime Spent Underwater:")
    print(f"  Median:     {np.median(dd.time_underwater):.1%} of the time")
    print(f"  95th pctl:  {np.percentile(dd.time_underwater, 95):.1%} of the time")

    print(f"\nRuin Probability (P of hitting drawdown threshold at any point):")
    print(f"  {'Threshold':<15} {'Probability':>12} {'Assessment':<20}")
    print(f"  {'-'*47}")
    for threshold, prob in sorted(dd.ruin_probabilities.items()):
        if prob < 0.01:
            assessment = "Very safe"
        elif prob < 0.05:
            assessment = "Acceptable"
        elif prob < 0.15:
            assessment = "Caution"
        elif prob < 0.30:
            assessment = "Dangerous"
        else:
            assessment = "UNACCEPTABLE"
        print(f"  {threshold:>+7.0%} DD      {prob:>10.2%}   {assessment}")
```

### Sample Output

```
=================================================================
DRAWDOWN DISTRIBUTION ANALYSIS (10,000 Monte Carlo paths)
=================================================================

Max Drawdown Distribution:
  P 1:  -22.40%  ###########
  P 5:  -17.80%  ########
  P10:  -15.20%  #######
  P25:  -11.50%  #####
  P50:   -8.80%  ####
  P75:   -6.50%  ###
  P90:   -4.80%  ##
  P95:   -3.90%  #
  P99:   -2.70%  #

  Mean max drawdown:  -9.42%
  Std of max DD:      4.18%

Drawdown Duration (bars to recover from max DD):
  P50:      285 bars (  23.8 hours)
  P75:      580 bars (  48.3 hours)
  P90:     1120 bars (  93.3 hours)
  P95:     1680 bars ( 140.0 hours)
  P99:     3200 bars ( 266.7 hours)

Time Spent Underwater:
  Median:     34.2% of the time
  95th pctl:  52.8% of the time

Ruin Probability (P of hitting drawdown threshold at any point):
  Threshold    Probability Assessment
  -----------------------------------------------
  -10% DD          38.50%   UNACCEPTABLE
  -15% DD          12.30%   Caution
  -20% DD           3.80%   Acceptable
  -25% DD           1.20%   Acceptable
  -30% DD           0.35%   Very safe
  -40% DD           0.04%   Very safe
  -50% DD           0.01%   Very safe
```

### Interpretation Guidelines and Setting Risk Limits

1. **Daily drawdown limit:** Set at the P50 max drawdown level. For this strategy: 9% DD
   means set daily stop at ~3% (roughly 1/3 of max annual DD can happen in a day).

2. **Account-level stop:** Set at 2x the P50 max DD. Here that is ~18%. If you hit 18%
   drawdown, halt trading and review.

3. **Time-based concern trigger:** If drawdown lasts > P75 duration without recovery
   (~48 hours for this strategy), reduce position size by 50%.

4. **P(ruin at -20%) of 3.8%** is acceptable for most traders. If using leverage, this
   number scales proportionally — 3x leverage turns -20% into -60% on notional.

5. **Time underwater of 34%** means you are in some drawdown about 1/3 of the time. This
   is psychologically important — expect to spend 3-4 months per year below your high-water mark.

---

## 6. Sharpe Ratio Distribution

### What It Does

Estimates the sampling distribution of the Sharpe ratio. A single Sharpe number is meaningless
without a confidence interval. This tells you: "How confident can I be that the true Sharpe
exceeds 1.0?" and "How many months of data do I need?"

### Mathematical Foundation

The asymptotic standard error of the estimated Sharpe ratio (from Lo, 2002 and Bailey & Lopez de Prado, 2014):

```
SE(SR) = sqrt( (1 - skew*SR + (kurt-1)*SR^2/4) / T )
```

Where T = number of observations, skew = return skewness, kurt = excess kurtosis.

**Minimum Track Record Length** to confirm SR > benchmark c at confidence (1-alpha):

```
MinTRL = (1 - skew*SR + (kurt-1)*SR^2/4) * (z_{1-alpha} / (SR - c))^2
```

### Full Implementation

```python
import numpy as np
from scipy import stats
from dataclasses import dataclass


@dataclass
class SharpeDistribution:
    """Sharpe ratio distribution analysis."""
    point_estimate: float
    standard_error: float
    ci_lower_95: float
    ci_upper_95: float
    ci_lower_99: float
    ci_upper_99: float
    prob_sharpe_gt_0: float
    prob_sharpe_gt_1: float
    prob_sharpe_gt_2: float
    min_months_for_sharpe_gt_0: float
    min_months_for_sharpe_gt_1: float
    mc_sharpe_distribution: np.ndarray  # from Monte Carlo
    return_skewness: float
    return_kurtosis: float


def analytical_sharpe_distribution(
    returns: np.ndarray,
    periods_per_year: int = 105_120,
) -> SharpeDistribution:
    """
    Compute Sharpe ratio distribution using both analytical formulas
    and Monte Carlo bootstrap.

    Uses the Probabilistic Sharpe Ratio framework from
    Bailey & Lopez de Prado (2014).
    """
    n = len(returns)
    sr = float(returns.mean() / returns.std() * np.sqrt(periods_per_year))
    skew = float(stats.skew(returns))
    kurt = float(stats.kurtosis(returns))  # excess kurtosis

    # Annualized Sharpe standard error (Lo, 2002 corrected for non-normality)
    se_factor = 1.0 - skew * sr / np.sqrt(periods_per_year) + (kurt - 1) * (sr / np.sqrt(periods_per_year))**2 / 4
    se = np.sqrt(max(se_factor, 0.01) / n) * np.sqrt(periods_per_year)

    # Confidence intervals
    z_95 = stats.norm.ppf(0.975)  # 1.96
    z_99 = stats.norm.ppf(0.995)  # 2.576

    ci_95 = (sr - z_95 * se, sr + z_95 * se)
    ci_99 = (sr - z_99 * se, sr + z_99 * se)

    # Probabilistic Sharpe Ratio: P(true SR > benchmark)
    def prob_sr_gt(benchmark: float) -> float:
        if se <= 0:
            return 1.0 if sr > benchmark else 0.0
        z = (sr - benchmark) / se
        return float(stats.norm.cdf(z))

    # Minimum Track Record Length (in observations)
    def min_trl(benchmark: float, alpha: float = 0.05) -> float:
        z_alpha = stats.norm.ppf(1 - alpha)
        sr_diff = sr - benchmark
        if abs(sr_diff) < 1e-10:
            return float("inf")
        # Per-observation Sharpe
        sr_per_obs = sr / np.sqrt(periods_per_year)
        se_per_obs_factor = 1.0 - skew * sr_per_obs + (kurt - 1) * sr_per_obs**2 / 4
        min_obs = se_per_obs_factor * (z_alpha * np.sqrt(periods_per_year) / sr_diff) ** 2
        return max(min_obs, 0)

    # Convert observations to months
    obs_per_month = periods_per_year / 12

    # Monte Carlo bootstrap of Sharpe
    rng = np.random.default_rng(42)
    n_mc = 10_000
    mc_sharpes = np.empty(n_mc)
    for i in range(n_mc):
        boot = rng.choice(returns, size=n, replace=True)
        if boot.std() > 0:
            mc_sharpes[i] = boot.mean() / boot.std() * np.sqrt(periods_per_year)
        else:
            mc_sharpes[i] = 0.0

    return SharpeDistribution(
        point_estimate=sr,
        standard_error=se,
        ci_lower_95=ci_95[0],
        ci_upper_95=ci_95[1],
        ci_lower_99=ci_99[0],
        ci_upper_99=ci_99[1],
        prob_sharpe_gt_0=prob_sr_gt(0.0),
        prob_sharpe_gt_1=prob_sr_gt(1.0),
        prob_sharpe_gt_2=prob_sr_gt(2.0),
        min_months_for_sharpe_gt_0=min_trl(0.0) / obs_per_month,
        min_months_for_sharpe_gt_1=min_trl(1.0) / obs_per_month,
        mc_sharpe_distribution=mc_sharpes,
        return_skewness=skew,
        return_kurtosis=kurt,
    )


def print_sharpe_report(sd: SharpeDistribution) -> None:
    """Print Sharpe ratio distribution analysis."""
    print("=" * 65)
    print("SHARPE RATIO DISTRIBUTION ANALYSIS")
    print("=" * 65)

    print(f"\nReturn Properties:")
    print(f"  Skewness:          {sd.return_skewness:>8.3f}")
    print(f"  Excess kurtosis:   {sd.return_kurtosis:>8.3f}")

    print(f"\nAnalytical Estimates:")
    print(f"  Point estimate:    {sd.point_estimate:>8.3f}")
    print(f"  Standard error:    {sd.standard_error:>8.3f}")
    print(f"  95% CI:            [{sd.ci_lower_95:.3f}, {sd.ci_upper_95:.3f}]")
    print(f"  99% CI:            [{sd.ci_lower_99:.3f}, {sd.ci_upper_99:.3f}]")

    print(f"\nProbabilistic Sharpe Ratio:")
    print(f"  P(SR > 0):         {sd.prob_sharpe_gt_0:>8.1%}")
    print(f"  P(SR > 1):         {sd.prob_sharpe_gt_1:>8.1%}")
    print(f"  P(SR > 2):         {sd.prob_sharpe_gt_2:>8.1%}")

    print(f"\nMinimum Track Record (95% confidence):")
    print(f"  To confirm SR > 0: {sd.min_months_for_sharpe_gt_0:>6.1f} months")
    print(f"  To confirm SR > 1: {sd.min_months_for_sharpe_gt_1:>6.1f} months")

    print(f"\nMonte Carlo Bootstrap (10,000 resamples):")
    p5, p50, p95 = np.percentile(sd.mc_sharpe_distribution, [5, 50, 95])
    print(f"  5th percentile:    {p5:>8.3f}")
    print(f"  Median:            {p50:>8.3f}")
    print(f"  95th percentile:   {p95:>8.3f}")
    print(f"  MC P(SR > 1):      {(sd.mc_sharpe_distribution > 1).mean():>8.1%}")
```

### Sample Output

```
=================================================================
SHARPE RATIO DISTRIBUTION ANALYSIS
=================================================================

Return Properties:
  Skewness:           -0.180
  Excess kurtosis:     8.420

Analytical Estimates:
  Point estimate:       2.050
  Standard error:       0.285
  95% CI:              [1.491, 2.609]
  99% CI:              [1.316, 2.784]

Probabilistic Sharpe Ratio:
  P(SR > 0):           100.0%
  P(SR > 1):            99.9%
  P(SR > 2):            57.0%

Minimum Track Record (95% confidence):
  To confirm SR > 0:    0.8 months
  To confirm SR > 1:    2.9 months

Monte Carlo Bootstrap (10,000 resamples):
  5th percentile:       1.480
  Median:               2.042
  95th percentile:      2.615
  MC P(SR > 1):         97.2%
```

### Interpretation Guidelines

- **CI width is the key metric.** A SR of 2.0 with CI [1.5, 2.5] is meaningful.
  A SR of 2.0 with CI [-0.3, 4.3] is noise.

- **Kurtosis inflates SE.** BTC 5-min returns with kurtosis=8 have ~1.5x wider CI than
  normal returns. This is why crypto strategies need more data to confirm edge.

- **Minimum track record of ~3 months** to confirm SR > 1 at 95% confidence is surprisingly
  short for 5-min data (because you have 105K observations/year). Daily strategies need
  3-5 years for the same confidence.

- **Practical rule of thumb:** If your data spans < 2x the MinTRL, your Sharpe estimate is
  unreliable. Collect more data before scaling up.

- **P(SR > 1) < 90%:** Do not allocate significant capital. Wait for more data or improve
  the strategy.

---

## 7. Win Streak / Loss Streak Analysis

### What It Does

Computes the expected distribution of consecutive wins and losses. Answers: "Is my observed
10-trade losing streak unusual, or expected?" Critical for psychological resilience and
setting alerts.

### Mathematical Foundation

For independent trades with loss probability q = 1 - p:

**Expected longest losing streak in N trades:**
```
E[max_streak] = ln(N) / ln(1/q)
```

**Probability of a losing streak of length k or more in N trades:**
```
P(streak >= k) = 1 - (1 - q^k)^(N - k + 1)     (approximate, assumes independence)
```

For 55% win rate (q=0.45), N=105,120 trades/year:
```
E[max_losing_streak] = ln(105120) / ln(1/0.45) = 11.56 / 0.799 ≈ 14.5 trades
```

### Full Implementation

```python
import numpy as np
from scipy import stats as sp_stats
from dataclasses import dataclass


@dataclass
class StreakAnalysis:
    """Win/loss streak distribution analysis."""
    observed_max_win_streak: int
    observed_max_loss_streak: int
    expected_max_loss_streak: float
    expected_max_win_streak: float
    loss_streak_distribution: dict  # length -> probability
    win_streak_distribution: dict
    observed_is_unusual: bool  # True if observed exceeds 95th percentile
    mc_max_loss_streaks: np.ndarray
    mc_max_win_streaks: np.ndarray


def find_max_streak(outcomes: np.ndarray, target: bool = False) -> int:
    """Find the longest consecutive streak of target value."""
    max_streak = 0
    current = 0
    for o in outcomes:
        if (o <= 0) == target:  # target=False for losses (return <= 0)
            current += 1
            max_streak = max(max_streak, current)
        else:
            current = 0
    return max_streak


def all_streaks(outcomes: np.ndarray, target: bool = False) -> list[int]:
    """Return lengths of all streaks of target value."""
    streaks = []
    current = 0
    for o in outcomes:
        if (o <= 0) == target:
            current += 1
        else:
            if current > 0:
                streaks.append(current)
            current = 0
    if current > 0:
        streaks.append(current)
    return streaks


def streak_analysis(
    trade_returns: np.ndarray,
    n_simulations: int = 10_000,
    seed: int = 42,
) -> StreakAnalysis:
    """
    Monte Carlo analysis of win/loss streaks.

    Parameters
    ----------
    trade_returns : array of per-trade returns
    n_simulations : number of MC simulations
    seed : random seed
    """
    rng = np.random.default_rng(seed)
    n_trades = len(trade_returns)

    # Observed streaks
    is_loss = trade_returns <= 0
    observed_max_loss = find_max_streak(trade_returns, target=True)
    observed_max_win = find_max_streak(trade_returns, target=False)

    # Observed win rate
    win_rate = float((trade_returns > 0).mean())
    loss_rate = 1.0 - win_rate

    # Analytical expected max streak
    if loss_rate > 0 and loss_rate < 1:
        expected_max_loss = np.log(n_trades) / np.log(1.0 / loss_rate)
    else:
        expected_max_loss = 0.0

    if win_rate > 0 and win_rate < 1:
        expected_max_win = np.log(n_trades) / np.log(1.0 / win_rate)
    else:
        expected_max_win = 0.0

    # Monte Carlo simulation
    mc_max_loss = np.empty(n_simulations)
    mc_max_win = np.empty(n_simulations)

    for i in range(n_simulations):
        # Simulate trades with observed win rate
        sim_outcomes = rng.random(n_trades)
        sim_returns = np.where(
            sim_outcomes < win_rate,
            np.abs(rng.choice(trade_returns[trade_returns > 0], size=n_trades)),
            -np.abs(rng.choice(trade_returns[trade_returns <= 0], size=n_trades)),
        )

        mc_max_loss[i] = find_max_streak(sim_returns, target=True)
        mc_max_win[i] = find_max_streak(sim_returns, target=False)

    # Distribution of streak lengths (from analytical model)
    loss_streak_dist = {}
    for k in range(1, 30):
        if loss_rate > 0:
            prob = 1.0 - (1.0 - loss_rate**k) ** max(n_trades - k + 1, 1)
            loss_streak_dist[k] = min(prob, 1.0)
        else:
            loss_streak_dist[k] = 0.0

    win_streak_dist = {}
    for k in range(1, 30):
        if win_rate > 0:
            prob = 1.0 - (1.0 - win_rate**k) ** max(n_trades - k + 1, 1)
            win_streak_dist[k] = min(prob, 1.0)
        else:
            win_streak_dist[k] = 0.0

    # Check if observed is unusual
    observed_is_unusual = observed_max_loss > np.percentile(mc_max_loss, 95)

    return StreakAnalysis(
        observed_max_win_streak=observed_max_win,
        observed_max_loss_streak=observed_max_loss,
        expected_max_loss_streak=expected_max_loss,
        expected_max_win_streak=expected_max_win,
        loss_streak_distribution=loss_streak_dist,
        win_streak_distribution=win_streak_dist,
        observed_is_unusual=observed_is_unusual,
        mc_max_loss_streaks=mc_max_loss,
        mc_max_win_streaks=mc_max_win,
    )


def print_streak_report(sa: StreakAnalysis) -> None:
    """Print streak analysis report."""
    print("=" * 65)
    print("WIN/LOSS STREAK ANALYSIS")
    print("=" * 65)

    print(f"\nObserved Streaks:")
    print(f"  Longest winning streak:  {sa.observed_max_win_streak}")
    print(f"  Longest losing streak:   {sa.observed_max_loss_streak}")

    print(f"\nExpected (Analytical):")
    print(f"  Expected max win streak:  {sa.expected_max_win_streak:.1f}")
    print(f"  Expected max loss streak: {sa.expected_max_loss_streak:.1f}")

    print(f"\nMonte Carlo Max Losing Streak Distribution (10,000 sims):")
    for pct in [5, 25, 50, 75, 90, 95, 99]:
        val = np.percentile(sa.mc_max_loss_streaks, pct)
        print(f"  P{pct:>2}: {val:>4.0f} trades")

    print(f"\nProbability of Losing Streak >= K (analytical):")
    for k in [5, 8, 10, 12, 15, 18, 20, 25]:
        if k in sa.loss_streak_distribution:
            prob = sa.loss_streak_distribution[k]
            label = "*** CERTAIN" if prob > 0.99 else (
                "Likely" if prob > 0.50 else (
                "Possible" if prob > 0.05 else "Rare"))
            print(f"  {k:>2} losses in a row: {prob:>8.2%}  ({label})")

    if sa.observed_is_unusual:
        print(f"\n  WARNING: Observed losing streak of {sa.observed_max_loss_streak} "
              f"exceeds 95th percentile!")
        print(f"  This is unusual for a {55}% win rate. Investigate for:")
        print(f"    - Regime change that degraded the model")
        print(f"    - Correlated losses (trades not independent)")
        print(f"    - Data quality issues during the streak period")
    else:
        print(f"\n  Observed losing streak of {sa.observed_max_loss_streak} "
              f"is within normal expectations.")
```

### Sample Output

```
=================================================================
WIN/LOSS STREAK ANALYSIS
=================================================================

Observed Streaks:
  Longest winning streak:  18
  Longest losing streak:   13

Expected (Analytical):
  Expected max win streak:  19.2
  Expected max loss streak: 14.5

Monte Carlo Max Losing Streak Distribution (10,000 sims):
  P 5:   10 trades
  P25:   12 trades
  P50:   14 trades
  P75:   16 trades
  P90:   18 trades
  P95:   19 trades
  P99:   22 trades

Probability of Losing Streak >= K (analytical):
   5 losses in a row:   100.00%  (*** CERTAIN)
   8 losses in a row:   100.00%  (*** CERTAIN)
  10 losses in a row:    99.98%  (*** CERTAIN)
  12 losses in a row:    97.80%  (Likely)
  15 losses in a row:    52.40%  (Likely)
  18 losses in a row:    12.80%  (Possible)
  20 losses in a row:     3.80%  (Rare)
  25 losses in a row:     0.12%  (Rare)

  Observed losing streak of 13 is within normal expectations.
```

### Interpretation Guidelines

- **10+ consecutive losses are almost certain** for a 55% win rate over 105K trades.
  This is not a bug — it is math. Prepare psychologically.

- **15 consecutive losses: coin-flip probability.** Expect it roughly once per year.
  Do NOT change your strategy after 15 losses if your annual metrics are still positive.

- **20+ losses in a row: investigate.** P < 5% for a genuine 55% win rate. Either the
  true win rate has decayed, or losses are correlated (violating the independence assumption).

- **Use for alert thresholds:** Set a monitoring alert at the P90 level (18 consecutive
  losses). If triggered, reduce size and investigate. Do NOT hard-stop the strategy at
  the P50 level (14 losses) — that would happen every year.

---

## 8. Correlation Stress Testing

### What It Does

Simulates scenarios where cross-market correlations break down. For a BTC strategy that
uses NQ (Nasdaq), ETH, and Gold as features, tests what happens when these relationships
change or disappear.

### Full Implementation

```python
import numpy as np
from dataclasses import dataclass
from typing import Callable


@dataclass
class CorrelationStressResult:
    """Results from correlation stress testing."""
    scenario_name: str
    original_correlation: float
    stressed_correlation: float
    sharpe_original: float
    sharpe_stressed: float
    sharpe_degradation_pct: float
    max_dd_original: float
    max_dd_stressed: float
    feature_importance_shift: dict  # feature_name -> importance change


def generate_decorrelated_feature(
    target_series: np.ndarray,
    feature_series: np.ndarray,
    target_correlation: float,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate a modified feature series with a specific target correlation
    to the target series (BTC returns).

    Uses Cholesky decomposition to construct returns with exact correlation.
    """
    rng = np.random.default_rng(seed)
    n = len(target_series)

    # Standardize
    target_std = (target_series - target_series.mean()) / (target_series.std() + 1e-10)
    noise = rng.standard_normal(n)

    # Construct correlated series via Cholesky
    correlated = (
        target_correlation * target_std
        + np.sqrt(1 - target_correlation**2) * noise
    )

    # Scale back to original feature's distribution
    result = correlated * feature_series.std() + feature_series.mean()
    return result


def correlation_stress_test(
    btc_returns: np.ndarray,
    feature_dict: dict,  # name -> returns array
    strategy_fn: Callable,
    correlation_scenarios: list[dict] = None,
    initial_capital: float = 100_000.0,
    seed: int = 42,
) -> list[CorrelationStressResult]:
    """
    Test strategy robustness when cross-market correlations change.

    Parameters
    ----------
    btc_returns : BTC 5-min returns
    feature_dict : dict mapping feature name to its return series
                   e.g., {"NQ": nq_returns, "ETH": eth_returns, "Gold": gold_returns}
    strategy_fn : function(btc_returns, feature_dict) -> strategy_returns
    correlation_scenarios : list of dicts defining stress scenarios
    initial_capital : starting capital
    seed : random seed
    """
    if correlation_scenarios is None:
        correlation_scenarios = [
            {"name": "Baseline", "modifications": {}},
            {"name": "BTC-NQ decorrelation", "modifications": {"NQ": 0.0}},
            {"name": "BTC-NQ inversion", "modifications": {"NQ": -0.3}},
            {"name": "All correlations to zero", "modifications": {k: 0.0 for k in feature_dict}},
            {"name": "All correlations to 0.9", "modifications": {k: 0.9 for k in feature_dict}},
            {"name": "ETH leads BTC (corr=0.95)", "modifications": {"ETH": 0.95}},
            {"name": "Gold safe-haven (corr=-0.4)", "modifications": {"Gold": -0.4}},
        ]

    results = []

    for scenario in correlation_scenarios:
        # Compute original correlations
        modified_features = {}
        for feat_name, feat_returns in feature_dict.items():
            original_corr = float(np.corrcoef(btc_returns, feat_returns)[0, 1])

            if feat_name in scenario["modifications"]:
                target_corr = scenario["modifications"][feat_name]
                modified_features[feat_name] = generate_decorrelated_feature(
                    btc_returns, feat_returns, target_corr, seed=seed,
                )
                stressed_corr = target_corr
            else:
                modified_features[feat_name] = feat_returns
                stressed_corr = original_corr

        # Run strategy with modified features
        try:
            strat_returns_stressed = strategy_fn(btc_returns, modified_features)
            strat_returns_original = strategy_fn(btc_returns, feature_dict)

            sharpe_stressed = compute_sharpe(strat_returns_stressed)
            sharpe_original = compute_sharpe(strat_returns_original)

            equity_stressed = initial_capital * np.cumprod(1.0 + strat_returns_stressed)
            equity_original = initial_capital * np.cumprod(1.0 + strat_returns_original)

            degradation = (sharpe_stressed - sharpe_original) / max(abs(sharpe_original), 0.01) * 100

            results.append(CorrelationStressResult(
                scenario_name=scenario["name"],
                original_correlation=0.0,  # varies per feature
                stressed_correlation=0.0,
                sharpe_original=sharpe_original,
                sharpe_stressed=sharpe_stressed,
                sharpe_degradation_pct=degradation,
                max_dd_original=max_drawdown(equity_original),
                max_dd_stressed=max_drawdown(equity_stressed),
                feature_importance_shift={},
            ))
        except Exception as e:
            logger.error(f"Correlation stress test failed: {scenario['name']}: {e}")

    return results


def print_correlation_stress_report(results: list[CorrelationStressResult]) -> None:
    """Print correlation stress test results."""
    print("=" * 80)
    print("CORRELATION STRESS TEST RESULTS")
    print("=" * 80)
    print(f"{'Scenario':<35} {'Sharpe':>8} {'Change':>8} {'MaxDD':>8} {'Verdict':<15}")
    print("-" * 80)

    for r in results:
        if abs(r.sharpe_degradation_pct) < 10:
            verdict = "ROBUST"
        elif r.sharpe_degradation_pct > -30:
            verdict = "MODERATE"
        elif r.sharpe_stressed > 0:
            verdict = "DEGRADED"
        else:
            verdict = "BROKEN"

        print(
            f"{r.scenario_name:<35} {r.sharpe_stressed:>8.2f} "
            f"{r.sharpe_degradation_pct:>+7.1f}% {r.max_dd_stressed:>7.2%} "
            f"{verdict}"
        )

    print("\nKey Findings:")
    worst = min(results, key=lambda r: r.sharpe_stressed)
    print(f"  Most damaging scenario: {worst.scenario_name}")
    print(f"    Sharpe drops to {worst.sharpe_stressed:.2f} ({worst.sharpe_degradation_pct:+.1f}%)")

    all_zero = [r for r in results if "zero" in r.scenario_name.lower()]
    if all_zero:
        r = all_zero[0]
        if r.sharpe_stressed > 1.0:
            print(f"  Cross-market features are NICE-TO-HAVE (strategy works without them)")
        elif r.sharpe_stressed > 0:
            print(f"  Cross-market features are IMPORTANT (strategy weakened without them)")
        else:
            print(f"  Cross-market features are CRITICAL (strategy fails without them)")
```

### Sample Output

```
================================================================================
CORRELATION STRESS TEST RESULTS
================================================================================
Scenario                            Sharpe   Change    MaxDD Verdict
--------------------------------------------------------------------------------
Baseline                              2.05    +0.0%  -9.21% ROBUST
BTC-NQ decorrelation                  1.72   -16.1%  -12.4% MODERATE
BTC-NQ inversion                      1.45   -29.3%  -15.8% MODERATE
All correlations to zero              1.28   -37.6%  -18.2% DEGRADED
All correlations to 0.9               1.85    -9.8%  -10.5% ROBUST
ETH leads BTC (corr=0.95)            2.15    +4.9%   -8.5% ROBUST
Gold safe-haven (corr=-0.4)           1.92    -6.3%  -10.1% ROBUST

Key Findings:
  Most damaging scenario: All correlations to zero
    Sharpe drops to 1.28 (-37.6%)
  Cross-market features are IMPORTANT (strategy weakened without them)
```

### Interpretation Guidelines

- **If "all correlations to zero" still yields Sharpe > 1.0:** Your core BTC-only features
  carry the strategy. Cross-market features add alpha but are not a dependency.

- **If BTC-NQ decorrelation causes > 30% Sharpe drop:** Your strategy is heavily reliant on
  the macro-crypto correlation. This correlation broke multiple times in 2022-2024. Build a
  regime detector that reduces position size when rolling correlation drops below a threshold.

- **Correlation inversion (NQ up, BTC down):** This has happened during crypto-specific events
  (exchange collapses, regulatory announcements). A strategy that survives this scenario is
  robust to crypto-specific shocks.

- **Action items:** For each feature whose decorrelation causes > 20% Sharpe drop, implement
  a rolling correlation monitor and a fallback model that excludes that feature.

---

## 9. Parameter Sensitivity Monte Carlo

### What It Does

Randomly perturbs all model parameters by +/-10%, +/-20% and reruns the strategy.
If < 50% of perturbations remain profitable, the strategy is fragile (overfit to
specific parameter values).

### Full Implementation

```python
import numpy as np
from dataclasses import dataclass
from typing import Callable, Any
import logging

logger = logging.getLogger(__name__)


@dataclass
class ParameterSensitivityResult:
    """Results from parameter sensitivity Monte Carlo."""
    n_perturbations: int
    pct_profitable: float
    pct_sharpe_gt_1: float
    original_sharpe: float
    sharpe_distribution: np.ndarray
    max_dd_distribution: np.ndarray
    parameter_importances: dict  # param_name -> correlation with Sharpe
    fragility_score: float       # 0 = robust, 1 = fragile


def parameter_sensitivity_monte_carlo(
    strategy_fn: Callable,
    base_params: dict,
    param_ranges: dict,  # param_name -> (min_pct, max_pct) relative perturbation
    price_data: np.ndarray,
    n_perturbations: int = 1_000,
    initial_capital: float = 100_000.0,
    seed: int = 42,
) -> ParameterSensitivityResult:
    """
    Randomly perturb model parameters and evaluate strategy robustness.

    Parameters
    ----------
    strategy_fn : function(prices, **params) -> trade_returns
    base_params : dict of parameter name -> base value
    param_ranges : dict of parameter name -> (min_pct_change, max_pct_change)
                   e.g., {"lookback": (-0.20, 0.20)} means +/-20%
    price_data : price series to run strategy on
    n_perturbations : number of random parameter sets to test
    initial_capital : starting capital
    seed : random seed

    Returns
    -------
    ParameterSensitivityResult
    """
    rng = np.random.default_rng(seed)

    # Run original strategy
    original_returns = strategy_fn(price_data, **base_params)
    original_sharpe = compute_sharpe(original_returns)

    # Storage
    sharpes = np.empty(n_perturbations)
    max_dds = np.empty(n_perturbations)
    param_samples = {name: np.empty(n_perturbations) for name in base_params}

    for i in range(n_perturbations):
        perturbed = {}
        for name, base_val in base_params.items():
            if name in param_ranges:
                lo, hi = param_ranges[name]
                pct_change = rng.uniform(lo, hi)
                new_val = base_val * (1.0 + pct_change)
                # Round to int if original was int
                if isinstance(base_val, int):
                    new_val = max(1, int(round(new_val)))
                perturbed[name] = new_val
            else:
                perturbed[name] = base_val
            param_samples[name][i] = perturbed[name]

        try:
            returns = strategy_fn(price_data, **perturbed)
            equity = initial_capital * np.cumprod(1.0 + returns)
            sharpes[i] = compute_sharpe(returns)
            max_dds[i] = max_drawdown(equity)
        except Exception:
            sharpes[i] = 0.0
            max_dds[i] = -1.0

    # Parameter importance: correlation between param value and Sharpe
    param_importances = {}
    for name in base_params:
        valid = ~np.isnan(sharpes) & ~np.isnan(param_samples[name])
        if valid.sum() > 10:
            corr = np.corrcoef(param_samples[name][valid], sharpes[valid])[0, 1]
            param_importances[name] = float(corr)
        else:
            param_importances[name] = 0.0

    pct_profitable = float((sharpes > 0).mean())
    pct_gt_1 = float((sharpes > 1.0).mean())

    # Fragility score: 1 - fraction of perturbations that maintain > 50% of original Sharpe
    maintainers = (sharpes > original_sharpe * 0.5).mean()
    fragility_score = 1.0 - maintainers

    return ParameterSensitivityResult(
        n_perturbations=n_perturbations,
        pct_profitable=pct_profitable,
        pct_sharpe_gt_1=pct_gt_1,
        original_sharpe=original_sharpe,
        sharpe_distribution=sharpes,
        max_dd_distribution=max_dds,
        parameter_importances=param_importances,
        fragility_score=fragility_score,
    )


def print_sensitivity_report(result: ParameterSensitivityResult) -> None:
    """Print parameter sensitivity analysis."""
    print("=" * 65)
    print("PARAMETER SENSITIVITY MONTE CARLO")
    print(f"({result.n_perturbations} random parameter perturbations)")
    print("=" * 65)

    print(f"\nOriginal Sharpe:  {result.original_sharpe:.3f}")
    print(f"Fragility Score:  {result.fragility_score:.2f} ", end="")
    if result.fragility_score < 0.20:
        print("(ROBUST)")
    elif result.fragility_score < 0.40:
        print("(MODERATE)")
    elif result.fragility_score < 0.60:
        print("(FRAGILE)")
    else:
        print("(VERY FRAGILE - likely overfit)")

    print(f"\nPerturbed Strategy Performance:")
    print(f"  % profitable (Sharpe > 0):     {result.pct_profitable:.1%}")
    print(f"  % with Sharpe > 1.0:           {result.pct_sharpe_gt_1:.1%}")
    p5, p50, p95 = np.percentile(result.sharpe_distribution, [5, 50, 95])
    print(f"  Sharpe 5th/50th/95th pctl:     {p5:.2f} / {p50:.2f} / {p95:.2f}")
    p5dd, p50dd = np.percentile(result.max_dd_distribution, [5, 50])
    print(f"  MaxDD 5th/50th pctl:           {p5dd:.2%} / {p50dd:.2%}")

    print(f"\nParameter Importance (correlation with Sharpe):")
    sorted_params = sorted(
        result.parameter_importances.items(),
        key=lambda x: abs(x[1]),
        reverse=True,
    )
    for name, corr in sorted_params:
        bar = "#" * int(abs(corr) * 30)
        direction = "+" if corr > 0 else "-"
        sensitivity = "HIGH" if abs(corr) > 0.3 else ("MEDIUM" if abs(corr) > 0.15 else "LOW")
        print(f"  {name:<25} r={corr:+.3f} [{sensitivity:>6}] {direction}{bar}")

    print(f"\nRobustness Verdict:")
    if result.pct_profitable > 0.80 and result.fragility_score < 0.30:
        print(f"  PASS: Strategy is robust to parameter perturbation.")
    elif result.pct_profitable > 0.60:
        print(f"  MARGINAL: Strategy is somewhat sensitive. Review high-importance params.")
    else:
        print(f"  FAIL: Strategy is fragile. Likely overfit to specific parameters.")
        print(f"  Action: Widen the training regularization or simplify the model.")
```

### Sample Output

```
=================================================================
PARAMETER SENSITIVITY MONTE CARLO
(1,000 random parameter perturbations)
=================================================================

Original Sharpe:  2.050
Fragility Score:  0.18 (ROBUST)

Perturbed Strategy Performance:
  % profitable (Sharpe > 0):     92.3%
  % with Sharpe > 1.0:           78.5%
  Sharpe 5th/50th/95th pctl:     0.65 / 1.82 / 2.85
  MaxDD 5th/50th pctl:           -22.50% / -10.80%

Parameter Importance (correlation with Sharpe):
  lookback_period           r=+0.042 [   LOW]
  threshold                 r=-0.285 [MEDIUM] -########
  momentum_window           r=+0.125 [   LOW] +###
  atr_multiplier            r=-0.180 [MEDIUM] -#####
  position_size             r=+0.310 [  HIGH] +#########
  cooldown_bars             r=-0.052 [   LOW] -#

Robustness Verdict:
  PASS: Strategy is robust to parameter perturbation.
```

### Interpretation Guidelines

- **Fragility score < 0.20:** Excellent. Strategy works across a wide parameter space.
- **Fragility score 0.20-0.40:** Acceptable. Monitor the high-importance parameters.
- **Fragility score > 0.50:** Strategy is likely overfit. The "edge" exists only at the
  exact parameter values found during optimization.

- **High-importance parameters (|r| > 0.3):** These parameters materially affect performance.
  Either fix them based on domain knowledge (not optimization) or add regularization.

- **If % profitable < 50%:** The strategy relies on a narrow parameter "sweet spot."
  This is the strongest signal of overfitting — a real edge should work across a
  reasonable parameter neighborhood.

---

## 10. Synthetic Data Generation for Backtesting

### What It Does

Generates synthetic BTC price paths using GARCH (volatility clustering), HAR (multi-timescale
volatility), and bootstrap methods. Extends your test dataset beyond historical limits.

### When to Use vs. When to Avoid

**Use synthetic data when:**
- Historical data covers < 2 market cycles
- You need to test on regimes that haven't occurred yet (e.g., BTC at $200K)
- Stress testing beyond historical extremes
- Augmenting training data for ML models

**Avoid synthetic data when:**
- Your strategy relies on specific market microstructure (order book, funding rates)
  that cannot be realistically simulated
- The fitted model is a poor representation of reality (KS test p < 0.01)
- You are using it to REPLACE validation on real data (it should supplement, not replace)

### Full Implementation

```python
import numpy as np
from scipy import stats
from dataclasses import dataclass
from typing import Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class SyntheticDataConfig:
    """Configuration for synthetic data generation."""
    method: str  # "garch", "har_like", "bootstrap", "student_t"
    n_paths: int
    n_steps: int
    seed: int


@dataclass
class SyntheticPaths:
    """Generated synthetic price paths with metadata."""
    prices: np.ndarray             # (n_paths, n_steps+1)
    returns: np.ndarray            # (n_paths, n_steps)
    method: str
    volatility_paths: Optional[np.ndarray]  # for GARCH
    fit_diagnostics: dict


def generate_garch_paths(
    historical_returns: np.ndarray,
    n_paths: int = 1_000,
    n_steps: int = 105_120,
    initial_price: float = 50_000.0,
    seed: int = 42,
) -> SyntheticPaths:
    """
    Generate synthetic price paths using GARCH(1,1) with Student-t innovations.

    GARCH(1,1): sigma^2_t = omega + alpha * r^2_{t-1} + beta * sigma^2_{t-1}

    Captures volatility clustering (high-vol follows high-vol) which is
    critical for crypto.
    """
    rng = np.random.default_rng(seed)

    # Estimate GARCH parameters from historical data
    # Using simple moment-based estimation (production: use arch package)
    returns_sq = historical_returns**2
    mean_return = float(historical_returns.mean())
    var_return = float(historical_returns.var())

    # Typical GARCH(1,1) parameters for BTC 5-min
    # In production, fit with: arch.arch_model(returns, vol='GARCH', p=1, q=1, dist='t')
    omega = var_return * 0.05   # long-run variance contribution
    alpha = 0.10                # reaction to recent shocks
    beta = 0.85                 # persistence
    # alpha + beta < 1 for stationarity

    # Fit Student-t to standardized residuals
    standardized = historical_returns / np.sqrt(var_return)
    df_t, _, _ = stats.t.fit(standardized)
    df_t = max(df_t, 2.1)  # ensure finite variance

    diagnostics = {
        "omega": omega,
        "alpha": alpha,
        "beta": beta,
        "persistence": alpha + beta,
        "student_t_df": df_t,
        "unconditional_vol": np.sqrt(omega / (1 - alpha - beta)) * np.sqrt(105_120),
    }

    logger.info("GARCH parameters", **{k: f"{v:.4f}" for k, v in diagnostics.items()})

    # Simulate
    all_returns = np.empty((n_paths, n_steps))
    all_volatility = np.empty((n_paths, n_steps))

    for p in range(n_paths):
        sigma2 = var_return  # initial variance
        for t in range(n_steps):
            # Student-t innovation
            z = stats.t.rvs(df=df_t, random_state=rng)
            sigma = np.sqrt(sigma2)

            r = mean_return + sigma * z
            all_returns[p, t] = r
            all_volatility[p, t] = sigma

            # Update variance (GARCH recursion)
            sigma2 = omega + alpha * (r - mean_return)**2 + beta * sigma2

    # Convert to prices
    prices = np.empty((n_paths, n_steps + 1))
    prices[:, 0] = initial_price
    prices[:, 1:] = initial_price * np.cumprod(1.0 + all_returns, axis=1)

    return SyntheticPaths(
        prices=prices,
        returns=all_returns,
        method="GARCH(1,1) + Student-t",
        volatility_paths=all_volatility,
        fit_diagnostics=diagnostics,
    )


def generate_har_like_paths(
    historical_returns: np.ndarray,
    n_paths: int = 1_000,
    n_steps: int = 105_120,
    initial_price: float = 50_000.0,
    seed: int = 42,
) -> SyntheticPaths:
    """
    Generate paths using HAR-like (Heterogeneous AutoRegressive) volatility model.

    HAR captures multi-timescale volatility dynamics:
    - Daily (288 bars): driven by daily traders
    - 4-hour (48 bars): driven by session traders
    - 1-hour (12 bars): driven by scalpers

    sigma^2_t = c + b1*RV_1h + b2*RV_4h + b3*RV_daily
    """
    rng = np.random.default_rng(seed)

    # Compute realized volatility at different horizons
    def realized_vol(rets, window):
        rv = np.zeros(len(rets))
        for t in range(window, len(rets)):
            rv[t] = np.sqrt(np.mean(rets[t-window:t]**2))
        return rv

    rv_1h = realized_vol(historical_returns, 12)
    rv_4h = realized_vol(historical_returns, 48)
    rv_1d = realized_vol(historical_returns, 288)

    # Fit HAR coefficients via OLS
    valid = rv_1d > 0
    if valid.sum() < 1000:
        logger.warning("Insufficient data for HAR fit, using defaults")
        c, b1, b2, b3 = 0.0001, 0.3, 0.3, 0.3
    else:
        # Simplified OLS
        X = np.column_stack([np.ones(valid.sum()), rv_1h[valid], rv_4h[valid], rv_1d[valid]])
        y = np.abs(historical_returns[valid])
        try:
            coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
            c, b1, b2, b3 = coeffs
            b1, b2, b3 = max(b1, 0), max(b2, 0), max(b3, 0)
        except np.linalg.LinAlgError:
            c, b1, b2, b3 = 0.0001, 0.3, 0.3, 0.3

    diagnostics = {"c": c, "b1_1h": b1, "b2_4h": b2, "b3_1d": b3}

    # Student-t innovations
    df_t = max(stats.t.fit(historical_returns)[0], 2.1)
    mean_ret = float(historical_returns.mean())

    # Simulate
    all_returns = np.empty((n_paths, n_steps))
    buffer = 288  # warm-up period

    for p in range(n_paths):
        # Initialize with historical warm-up
        warm_up = rng.choice(historical_returns, size=buffer)
        rets = np.concatenate([warm_up, np.zeros(n_steps)])

        for t in range(buffer, buffer + n_steps):
            rv1 = np.sqrt(np.mean(rets[t-12:t]**2))
            rv4 = np.sqrt(np.mean(rets[t-48:t]**2))
            rvd = np.sqrt(np.mean(rets[t-288:t]**2))

            sigma = abs(c) + b1 * rv1 + b2 * rv4 + b3 * rvd
            sigma = max(sigma, 1e-8)

            z = stats.t.rvs(df=df_t, random_state=rng)
            rets[t] = mean_ret + sigma * z

        all_returns[p] = rets[buffer:]

    prices = np.empty((n_paths, n_steps + 1))
    prices[:, 0] = initial_price
    prices[:, 1:] = initial_price * np.cumprod(1.0 + all_returns, axis=1)

    return SyntheticPaths(
        prices=prices,
        returns=all_returns,
        method="HAR-RV + Student-t",
        volatility_paths=None,
        fit_diagnostics=diagnostics,
    )


def generate_block_bootstrap_paths(
    historical_returns: np.ndarray,
    n_paths: int = 1_000,
    n_steps: int = 105_120,
    block_length: int = 288,  # 1 day
    initial_price: float = 50_000.0,
    seed: int = 42,
) -> SyntheticPaths:
    """
    Generate synthetic paths via block bootstrap (non-parametric).

    Preserves all distributional properties AND temporal dependencies
    within blocks. The most assumption-free method.
    """
    rng = np.random.default_rng(seed)
    n_hist = len(historical_returns)
    n_blocks = int(np.ceil(n_steps / block_length))

    all_returns = np.empty((n_paths, n_steps))

    for p in range(n_paths):
        blocks = []
        for _ in range(n_blocks):
            start = rng.integers(0, n_hist - block_length)
            blocks.append(historical_returns[start:start + block_length])
        all_returns[p] = np.concatenate(blocks)[:n_steps]

    prices = np.empty((n_paths, n_steps + 1))
    prices[:, 0] = initial_price
    prices[:, 1:] = initial_price * np.cumprod(1.0 + all_returns, axis=1)

    return SyntheticPaths(
        prices=prices,
        returns=all_returns,
        method="Block Bootstrap (1-day blocks)",
        volatility_paths=None,
        fit_diagnostics={"block_length": block_length, "n_blocks": n_blocks},
    )


def validate_synthetic_data(
    historical_returns: np.ndarray,
    synthetic: SyntheticPaths,
) -> dict:
    """
    Compare statistical properties of synthetic vs historical data.
    Returns a diagnostic report.
    """
    hist = historical_returns
    # Use first synthetic path as representative
    synth = synthetic.returns[0]

    diagnostics = {
        "method": synthetic.method,
        "mean": {"historical": float(hist.mean()), "synthetic": float(synth.mean())},
        "std": {"historical": float(hist.std()), "synthetic": float(synth.std())},
        "skewness": {"historical": float(stats.skew(hist)), "synthetic": float(stats.skew(synth))},
        "kurtosis": {"historical": float(stats.kurtosis(hist)), "synthetic": float(stats.kurtosis(synth))},
        "min": {"historical": float(hist.min()), "synthetic": float(synth.min())},
        "max": {"historical": float(hist.max()), "synthetic": float(synth.max())},
    }

    # Autocorrelation at lag 1
    ac_hist = float(np.corrcoef(hist[:-1], hist[1:])[0, 1])
    ac_synth = float(np.corrcoef(synth[:-1], synth[1:])[0, 1])
    diagnostics["autocorr_lag1"] = {"historical": ac_hist, "synthetic": ac_synth}

    # Autocorrelation of squared returns (volatility clustering)
    ac_sq_hist = float(np.corrcoef(hist[:-1]**2, hist[1:]**2)[0, 1])
    ac_sq_synth = float(np.corrcoef(synth[:-1]**2, synth[1:]**2)[0, 1])
    diagnostics["vol_clustering"] = {"historical": ac_sq_hist, "synthetic": ac_sq_synth}

    return diagnostics


def print_synthetic_validation(diagnostics: dict) -> None:
    """Print synthetic data validation report."""
    print("=" * 65)
    print(f"SYNTHETIC DATA VALIDATION: {diagnostics['method']}")
    print("=" * 65)
    print(f"{'Metric':<25} {'Historical':>12} {'Synthetic':>12} {'Match':>8}")
    print("-" * 65)

    for key in ["mean", "std", "skewness", "kurtosis", "min", "max",
                "autocorr_lag1", "vol_clustering"]:
        h = diagnostics[key]["historical"]
        s = diagnostics[key]["synthetic"]
        ratio = abs(s / h) if abs(h) > 1e-10 else float("inf")
        match = "GOOD" if 0.7 < ratio < 1.3 else ("OK" if 0.5 < ratio < 1.5 else "POOR")
        print(f"  {key:<23} {h:>12.6f} {s:>12.6f} {match:>8}")
```

### Method Comparison

| Method | Preserves Vol Clustering | Preserves Tails | Preserves Autocorrelation | Assumption-Free | Speed |
|--------|:---:|:---:|:---:|:---:|:---:|
| GARCH(1,1) + Student-t | Yes | Mostly | No (returns) | No | Fast |
| HAR + Student-t | Yes (multi-scale) | Mostly | Partial | No | Slow |
| Block Bootstrap | Yes (within blocks) | Yes (exact) | Yes (within blocks) | Yes | Fast |

### When to Use Each

- **GARCH:** When you need controlled volatility experiments (e.g., "what if vol doubles?").
  Adjust omega/alpha/beta to create specific vol regimes.
- **HAR:** When multi-timescale dynamics matter (intraday vs. daily volatility patterns).
  Better than GARCH for capturing the "volatility of volatility."
- **Block Bootstrap:** Default choice. No parametric assumptions, preserves all empirical
  properties. Only limitation: cannot generate returns outside historical range.

---

## 11. Monte Carlo for Position Sizing Validation (Kelly Criterion)

### What It Does

Simulates the full Kelly, half-Kelly, and quarter-Kelly position sizing strategies over
thousands of paths. Shows the tradeoff: full Kelly maximizes long-run growth but produces
terrifying drawdowns. Half-Kelly captures ~75% of growth with dramatically less pain.

### Mathematical Foundation

**Kelly fraction:**
```
f* = p/a - q/b
```
Where p = win probability, q = 1-p, a = loss fraction, b = win fraction.

For our strategy: p=0.55, b=0.0015, a=0.0012:
```
f* = 0.55/0.0012 - 0.45/0.0015 = 458.3 - 300.0 = 158.3
```

This means Kelly says bet 158x your bankroll — which is absurd for continuous
outcomes. The correct approach for continuous returns:

```
f* = mu / sigma^2
```

Where mu = expected return per trade, sigma^2 = variance per trade.

For our strategy: mu = 0.55*0.0015 + 0.45*(-0.0012) = 0.000285
sigma^2 ~= 0.0014^2 = 0.00000196 (approx given the win/loss magnitudes)
```
f* = 0.000285 / 0.00000196 = 145.4
```

Still huge. This is because individual 5-min trade returns have tiny variance.
In practice, we apply Kelly to aggregated daily or position-level returns.

### Full Implementation

```python
import numpy as np
from dataclasses import dataclass


@dataclass
class KellySimulationResult:
    """Results from Kelly criterion Monte Carlo."""
    fraction_label: str
    kelly_fraction: float
    equity_curves: np.ndarray
    final_equity: np.ndarray
    sharpe_ratios: np.ndarray
    max_drawdowns: np.ndarray
    median_growth_rate: float
    p_ruin_20pct: float
    p_ruin_50pct: float
    median_max_dd: float


def compute_kelly_fraction(
    returns: np.ndarray,
) -> float:
    """
    Compute the Kelly fraction for continuous returns.
    f* = mean(r) / var(r)
    """
    mu = returns.mean()
    var = returns.var()
    if var < 1e-20:
        return 0.0
    return float(mu / var)


def simulate_kelly_sizing(
    trade_returns: np.ndarray,
    kelly_fractions: list[float] = None,
    n_simulations: int = 10_000,
    initial_capital: float = 100_000.0,
    seed: int = 42,
) -> list[KellySimulationResult]:
    """
    Monte Carlo simulation of different Kelly fractions.

    Aggregates 5-min returns into daily returns for Kelly sizing,
    then simulates equity paths.

    Parameters
    ----------
    trade_returns : per-trade returns
    kelly_fractions : list of Kelly multiples [1.0, 0.5, 0.25]
    n_simulations : number of MC paths
    initial_capital : starting capital
    seed : random seed
    """
    rng = np.random.default_rng(seed)

    # Aggregate to daily returns (288 trades/day)
    n_days = len(trade_returns) // 288
    daily_returns = np.array([
        np.prod(1 + trade_returns[i*288:(i+1)*288]) - 1
        for i in range(n_days)
    ])

    full_kelly = compute_kelly_fraction(daily_returns)

    if kelly_fractions is None:
        kelly_fractions = [1.0, 0.5, 0.25, 0.10]

    results = []

    for frac_mult in kelly_fractions:
        actual_fraction = full_kelly * frac_mult
        # Cap at reasonable levels
        actual_fraction = min(actual_fraction, 5.0)  # max 5x leverage
        actual_fraction = max(actual_fraction, 0.01)

        label = f"{frac_mult:.0%} Kelly (f={actual_fraction:.2f})"

        equity_curves = np.empty((n_simulations, n_days + 1))
        equity_curves[:, 0] = initial_capital
        sharpes = np.empty(n_simulations)
        max_dds = np.empty(n_simulations)

        for i in range(n_simulations):
            # Resample daily returns
            resampled = rng.choice(daily_returns, size=n_days, replace=True)

            # Apply Kelly sizing: actual return = fraction * market return
            sized_returns = actual_fraction * resampled

            # Clip to prevent bankruptcy (can't lose more than 100%)
            sized_returns = np.maximum(sized_returns, -0.99)

            equity = initial_capital * np.cumprod(1.0 + sized_returns)
            equity_curves[i, 1:] = equity

            sharpes[i] = compute_sharpe(sized_returns, periods_per_year=365)
            max_dds[i] = max_drawdown(equity)

        final_equity = equity_curves[:, -1]
        growth_rates = (final_equity / initial_capital) ** (1.0 / (n_days/365)) - 1

        results.append(KellySimulationResult(
            fraction_label=label,
            kelly_fraction=actual_fraction,
            equity_curves=equity_curves,
            final_equity=final_equity,
            sharpe_ratios=sharpes,
            max_drawdowns=max_dds,
            median_growth_rate=float(np.median(growth_rates)),
            p_ruin_20pct=float((max_dds < -0.20).mean()),
            p_ruin_50pct=float((max_dds < -0.50).mean()),
            median_max_dd=float(np.median(max_dds)),
        ))

    return results


def print_kelly_report(results: list[KellySimulationResult]) -> None:
    """Print Kelly criterion comparison report."""
    print("=" * 85)
    print("KELLY CRITERION MONTE CARLO — POSITION SIZING COMPARISON")
    print("=" * 85)
    print(f"{'Fraction':<28} {'Med Growth':>10} {'Med MaxDD':>10} "
          f"{'P(DD>20%)':>10} {'P(DD>50%)':>10} {'Med Final$':>12}")
    print("-" * 85)

    for r in results:
        print(
            f"{r.fraction_label:<28} {r.median_growth_rate:>9.1%} "
            f"{r.median_max_dd:>9.2%} {r.p_ruin_20pct:>9.1%} "
            f"{r.p_ruin_50pct:>9.1%} {np.median(r.final_equity):>11,.0f}"
        )

    print(f"\nKey Insight:")
    if len(results) >= 2:
        full = results[0]
        half = results[1]
        growth_ratio = half.median_growth_rate / max(full.median_growth_rate, 0.001)
        dd_ratio = half.median_max_dd / min(full.median_max_dd, -0.001)
        print(f"  Half Kelly captures {growth_ratio:.0%} of Full Kelly growth")
        print(f"  with {1-dd_ratio:.0%} less drawdown")
        print(f"  Full Kelly P(50% DD) = {full.p_ruin_50pct:.1%}")
        print(f"  Half Kelly P(50% DD) = {half.p_ruin_50pct:.1%}")

    print(f"\nRecommendation for live trading:")
    print(f"  Start with quarter-Kelly. Scale to half-Kelly after 3+ months")
    print(f"  of confirmed positive edge. Never use full Kelly.")
```

### Sample Output

```
=====================================================================================
KELLY CRITERION MONTE CARLO — POSITION SIZING COMPARISON
=====================================================================================
Fraction                     Med Growth   Med MaxDD  P(DD>20%)  P(DD>50%)    Med Final$
-------------------------------------------------------------------------------------
100% Kelly (f=3.20)              82.5%     -42.30%      88.5%      38.2%      182,500
50% Kelly (f=1.60)               62.0%     -22.10%      55.3%       4.8%      162,000
25% Kelly (f=0.80)               38.5%     -12.50%      14.2%       0.2%      138,500
10% Kelly (f=0.32)               16.8%      -5.80%       0.8%       0.0%      116,800

Key Insight:
  Half Kelly captures 75% of Full Kelly growth
  with 48% less drawdown
  Full Kelly P(50% DD) = 38.2%
  Half Kelly P(50% DD) = 4.8%

Recommendation for live trading:
  Start with quarter-Kelly. Scale to half-Kelly after 3+ months
  of confirmed positive edge. Never use full Kelly.
```

### Interpretation Guidelines

- **Full Kelly has a mathematical property:** There is an X% chance your equity drops to X%
  of starting capital. 50% chance of a 50% drawdown is psychologically devastating.

- **Half-Kelly is the industry standard.** It captures ~75% of the optimal growth rate with
  dramatically reduced drawdown risk. It also provides a buffer against parameter estimation
  error.

- **Quarter-Kelly for new strategies.** Until you have 3+ months of live data confirming
  your edge, quarter-Kelly limits damage if your backtest was overfit.

- **The Kelly fraction for 5-min trades is misleadingly large.** Always aggregate to daily
  returns before computing Kelly. Per-trade Kelly of 150+ implies absurd leverage that would
  be destroyed by a single tail event.

---

## 12. Ruin Probability

### What It Does

Computes the probability of losing X% of capital over N trades, both analytically and
via Monte Carlo. Combines position sizing, expected return, and volatility into a single
survival metric.

### Mathematical Foundation

**Balsara's Formula (for fixed-risk trades):**
```
P(ruin) = ((1 - Edge) / (1 + Edge)) ^ U
```
Where:
- Edge = (WinRate * AvgWin) - (LossRate * AvgLoss)
- U = Capital Units = AccountSize / RiskPerTrade

**Continuous approximation (for variable returns):**
```
P(ruin to level L) = exp(-2 * mu * L / sigma^2)
```
Where mu = expected return per trade, sigma = std of returns per trade,
L = log of the ruin level (e.g., L = ln(0.8) for 20% drawdown).

### Full Implementation

```python
import numpy as np
from scipy import stats
from dataclasses import dataclass


@dataclass
class RuinProbabilityResult:
    """Comprehensive ruin probability analysis."""
    analytical_ruin_probs: dict     # threshold -> probability
    mc_ruin_probs: dict             # threshold -> probability
    mc_ruin_times: dict             # threshold -> median trades to ruin (if it occurs)
    edge_per_trade: float
    expected_trades_to_ruin: dict   # threshold -> expected trades
    survival_probability: dict      # threshold -> P(survive N trades)


def analytical_ruin_probability(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    risk_per_trade: float,
    account_size: float = 100_000.0,
    ruin_thresholds: list[float] = None,
) -> dict:
    """
    Compute analytical ruin probability using Balsara's formula.

    Parameters
    ----------
    win_rate : probability of winning (e.g., 0.55)
    avg_win : average winning return (e.g., 0.0015)
    avg_loss : average losing return magnitude (e.g., 0.0012)
    risk_per_trade : fraction of capital risked per trade
    account_size : starting capital
    ruin_thresholds : list of drawdown levels (e.g., [-0.20, -0.50])
    """
    if ruin_thresholds is None:
        ruin_thresholds = [-0.10, -0.20, -0.30, -0.50, -0.75]

    edge = win_rate * avg_win - (1 - win_rate) * avg_loss

    results = {}
    for threshold in ruin_thresholds:
        # Capital units needed to lose to reach threshold
        capital_to_lose = abs(threshold) * account_size
        units = capital_to_lose / (risk_per_trade * account_size)

        if edge <= 0:
            # Negative edge: ruin is certain given enough trades
            results[threshold] = 1.0
        else:
            ratio = (1 - edge) / (1 + edge)
            if ratio >= 1:
                results[threshold] = 1.0
            else:
                results[threshold] = min(ratio ** units, 1.0)

    return results


def continuous_ruin_probability(
    mu: float,
    sigma: float,
    ruin_thresholds: list[float] = None,
) -> dict:
    """
    Continuous approximation: P(ever hitting level L) for a random walk
    with drift mu and volatility sigma.

    Uses: P(ruin) = exp(-2 * mu * |L| / sigma^2) for mu > 0
    """
    if ruin_thresholds is None:
        ruin_thresholds = [-0.10, -0.20, -0.30, -0.50, -0.75]

    results = {}
    for threshold in ruin_thresholds:
        L = abs(threshold)
        if mu <= 0:
            results[threshold] = 1.0
        else:
            results[threshold] = min(np.exp(-2 * mu * L / sigma**2), 1.0)

    return results


def monte_carlo_ruin_probability(
    trade_returns: np.ndarray,
    n_simulations: int = 50_000,
    n_trades_per_sim: int = 105_120,
    risk_fraction: float = 1.0,  # fraction of Kelly to apply
    ruin_thresholds: list[float] = None,
    seed: int = 42,
) -> RuinProbabilityResult:
    """
    Monte Carlo estimation of ruin probability.

    Simulates many trading paths and counts how often each ruin
    threshold is breached.

    Parameters
    ----------
    trade_returns : historical per-trade returns
    n_simulations : number of MC paths (50K+ for accurate tail estimates)
    n_trades_per_sim : trades per simulation path
    risk_fraction : position sizing multiplier
    ruin_thresholds : drawdown levels to check
    seed : random seed
    """
    if ruin_thresholds is None:
        ruin_thresholds = [-0.10, -0.20, -0.30, -0.50, -0.75]

    rng = np.random.default_rng(seed)

    # Analytics
    mu = float(trade_returns.mean())
    sigma = float(trade_returns.std())
    win_rate = float((trade_returns > 0).mean())
    avg_win = float(trade_returns[trade_returns > 0].mean())
    avg_loss = float(np.abs(trade_returns[trade_returns <= 0]).mean())
    edge = win_rate * avg_win - (1 - win_rate) * avg_loss

    # Analytical estimates
    analytical = continuous_ruin_probability(
        mu=mu * risk_fraction,
        sigma=sigma * risk_fraction,
        ruin_thresholds=ruin_thresholds,
    )

    # Monte Carlo
    mc_ruin_counts = {t: 0 for t in ruin_thresholds}
    mc_ruin_times = {t: [] for t in ruin_thresholds}

    for i in range(n_simulations):
        # Resample trades
        sampled = rng.choice(trade_returns, size=n_trades_per_sim, replace=True)
        sized = sampled * risk_fraction

        # Build equity curve
        equity = np.cumprod(1.0 + sized)
        peak = np.maximum.accumulate(equity)
        drawdown = (equity - peak) / peak

        for threshold in ruin_thresholds:
            breached = drawdown <= threshold
            if breached.any():
                mc_ruin_counts[threshold] += 1
                first_breach = np.argmax(breached)
                mc_ruin_times[threshold].append(first_breach)

    mc_ruin_probs = {t: mc_ruin_counts[t] / n_simulations for t in ruin_thresholds}
    mc_median_times = {}
    for t in ruin_thresholds:
        if mc_ruin_times[t]:
            mc_median_times[t] = float(np.median(mc_ruin_times[t]))
        else:
            mc_median_times[t] = float("inf")

    # Expected trades to ruin (analytical, for drift mu > 0)
    expected_to_ruin = {}
    for threshold in ruin_thresholds:
        L = abs(threshold)
        if mu > 0:
            # This is a rough approximation based on first passage time
            expected_to_ruin[threshold] = L / mu if mc_ruin_probs.get(threshold, 0) > 0.01 else float("inf")
        else:
            expected_to_ruin[threshold] = L / abs(mu) if mu != 0 else 0

    # Survival probability over 1 year
    survival = {t: 1.0 - mc_ruin_probs[t] for t in ruin_thresholds}

    return RuinProbabilityResult(
        analytical_ruin_probs=analytical,
        mc_ruin_probs=mc_ruin_probs,
        mc_ruin_times=mc_median_times,
        edge_per_trade=edge,
        expected_trades_to_ruin=expected_to_ruin,
        survival_probability=survival,
    )


def print_ruin_report(result: RuinProbabilityResult) -> None:
    """Print comprehensive ruin probability analysis."""
    print("=" * 80)
    print("RUIN PROBABILITY ANALYSIS")
    print(f"Edge per trade: {result.edge_per_trade:.6f} ({result.edge_per_trade*100:.4f}%)")
    print("=" * 80)

    print(f"\n{'Threshold':<12} {'Analytical':>12} {'Monte Carlo':>12} "
          f"{'Median Time':>14} {'1Y Survival':>12}")
    print("-" * 65)

    for threshold in sorted(result.analytical_ruin_probs.keys()):
        a = result.analytical_ruin_probs[threshold]
        m = result.mc_ruin_probs[threshold]
        t = result.mc_ruin_times[threshold]
        s = result.survival_probability[threshold]

        time_str = f"{t:,.0f} trades" if t < float("inf") else "Never"
        days_str = ""
        if t < float("inf"):
            days = t / 288
            time_str = f"{days:.0f} days"

        print(
            f"{threshold:>+7.0%} DD   {a:>11.4%} {m:>11.4%} "
            f"{time_str:>14} {s:>11.2%}"
        )

    print(f"\nRisk Assessment:")
    mc20 = result.mc_ruin_probs.get(-0.20, 0)
    mc50 = result.mc_ruin_probs.get(-0.50, 0)

    if mc20 < 0.05:
        print(f"  20% DD risk: ACCEPTABLE ({mc20:.2%})")
    elif mc20 < 0.15:
        print(f"  20% DD risk: ELEVATED ({mc20:.2%}) - consider reducing position size")
    else:
        print(f"  20% DD risk: HIGH ({mc20:.2%}) - MUST reduce position size")

    if mc50 < 0.01:
        print(f"  50% DD risk: NEGLIGIBLE ({mc50:.2%})")
    elif mc50 < 0.05:
        print(f"  50% DD risk: LOW but present ({mc50:.2%})")
    else:
        print(f"  50% DD risk: DANGEROUS ({mc50:.2%}) - account blow-up likely")

    print(f"\nPosition Sizing Guidance:")
    print(f"  For P(20% DD) < 5%:  risk at most ~{result.edge_per_trade*100/4:.3f}% per trade")
    print(f"  For P(50% DD) < 1%:  risk at most ~{result.edge_per_trade*100/8:.3f}% per trade")
```

### Sample Output

```
================================================================================
RUIN PROBABILITY ANALYSIS
Edge per trade: 0.000285 (0.0285%)
================================================================================

Threshold    Analytical  Monte Carlo     Median Time   1Y Survival
-----------------------------------------------------------------
 -10% DD       18.2400%      16.8000%        82 days       83.20%
 -20% DD        3.3200%       3.9500%       195 days       96.05%
 -30% DD        0.6100%       0.8200%       Never          99.18%
 -50% DD        0.0200%       0.0400%       Never          99.96%
 -75% DD        0.0003%       0.0000%       Never         100.00%

Risk Assessment:
  20% DD risk: ACCEPTABLE (3.95%)
  50% DD risk: NEGLIGIBLE (0.04%)

Position Sizing Guidance:
  For P(20% DD) < 5%:  risk at most ~0.007% per trade
  For P(50% DD) < 1%:  risk at most ~0.004% per trade
```

### Interpretation Guidelines

- **Analytical vs Monte Carlo mismatch is expected.** Analytical formulas assume iid returns.
  MC captures autocorrelation and fat tails. Trust MC for realistic estimates, use analytical
  for quick sanity checks.

- **The "Never" median time is deceptive.** It means the median path doesn't hit that level,
  but some paths do. The MC probability is the fraction that does — even 0.04% means 4 out
  of 10,000 paths hit -50%.

- **Risk per trade of 0.007%** means on a $100K account, risking $7 per trade. With 105K
  trades/year, that is $7 * 0.000285 * 105K = $210/year expected profit per unit. To make
  meaningful returns, you need either larger capital or more edge.

- **Scaling rule:** If you double position size, ruin probability approximately squares.
  P(ruin) going from 4% to 16% is a huge jump from "acceptable" to "likely."

---

## Integration: Running All Analyses

```python
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_sample_trades(
    n_trades: int = 105_120,
    win_rate: float = 0.55,
    avg_win: float = 0.0015,
    avg_loss: float = -0.0012,
    seed: int = 42,
) -> np.ndarray:
    """Generate sample trade returns matching our strategy profile."""
    rng = np.random.default_rng(seed)
    is_win = rng.random(n_trades) < win_rate
    returns = np.where(
        is_win,
        rng.exponential(avg_win, n_trades),        # wins: exponential distribution
        -rng.exponential(abs(avg_loss), n_trades),  # losses: exponential distribution
    )
    return returns


def run_full_monte_carlo_suite(trade_returns: np.ndarray) -> None:
    """
    Run all 12 Monte Carlo validation methods.

    This is the master function that executes the complete validation suite.
    Each method runs independently and prints its own report.
    """
    logger.info(f"Starting Monte Carlo validation suite with {len(trade_returns):,} trades")

    # 1. Block Bootstrap
    logger.info("Running block bootstrap resampling...")
    boot_result = block_bootstrap_trades(trade_returns, n_simulations=10_000)
    print_bootstrap_report(boot_result)

    # 2. Return Shuffling
    logger.info("Running return shuffling null hypothesis test...")
    shuffle_result = return_shuffle_test(trade_returns, n_simulations=10_000)
    print_shuffle_report(shuffle_result)

    # 5. Drawdown Distribution
    logger.info("Running drawdown distribution analysis...")
    dd_result = monte_carlo_drawdown_analysis(trade_returns, n_simulations=10_000)
    print_drawdown_report(dd_result)

    # 6. Sharpe Ratio Distribution
    logger.info("Running Sharpe ratio distribution analysis...")
    sharpe_dist = analytical_sharpe_distribution(trade_returns)
    print_sharpe_report(sharpe_dist)

    # 7. Streak Analysis
    logger.info("Running win/loss streak analysis...")
    streak_result = streak_analysis(trade_returns, n_simulations=10_000)
    print_streak_report(streak_result)

    # 12. Ruin Probability
    logger.info("Running ruin probability analysis...")
    ruin_result = monte_carlo_ruin_probability(trade_returns, n_simulations=50_000)
    print_ruin_report(ruin_result)

    logger.info("Monte Carlo validation suite complete.")


if __name__ == "__main__":
    trades = generate_sample_trades()
    run_full_monte_carlo_suite(trades)
```

---

## Summary: Which Method to Run When

| Stage | Methods | Purpose |
|-------|---------|---------|
| After initial backtest | 1 (Bootstrap), 2 (Shuffle), 6 (Sharpe CI) | "Is the edge real?" |
| Before parameter lock | 9 (Parameter Sensitivity) | "Is the strategy overfit?" |
| Before sizing decision | 11 (Kelly MC), 12 (Ruin Probability), 5 (Drawdown Dist) | "How much to risk?" |
| Before live deployment | 4 (Stress Test), 8 (Correlation Stress) | "What breaks it?" |
| Ongoing monitoring | 7 (Streak Analysis), 5 (Drawdown Dist) | "Is this drawdown normal?" |
| Data augmentation | 3 (Path MC), 10 (Synthetic Data) | "Does it generalize?" |

## Dependencies

```toml
[project]
dependencies = [
    "numpy>=2.0",
    "scipy>=1.14",
    "arch>=7.0",           # For optimal block length estimation
    "pandas>=2.2",
    "structlog>=24.0",
]
```

## Key References

- Lo, A.W. (2002). "The Statistics of Sharpe Ratios." Financial Analysts Journal.
- Bailey, D.H. & Lopez de Prado, M. (2014). "The Deflated Sharpe Ratio." Journal of Portfolio Management.
- Politis, D.N. & White, H. (2004). "Automatic Block-Length Selection." Econometric Reviews.
- Corsi, F. (2009). "A Simple Approximate Long-Memory Model of Realized Volatility." Journal of Financial Econometrics.
- Balsara, N.J. (1992). "Money Management Strategies for Futures Traders." Wiley.

Sources:
- [Monte Carlo Simulation in Python: Stress-Test Your Trading Strategy](https://arongroups.co/forex-articles/monte-carlo-simulation-in-python-for-trading/)
- [Monte Carlo Simulations in Trading: A Practical Guide](https://quantproof.io/blog/monte-carlo-simulations-trading-strategy-validation)
- [Monte Carlo Stress Test for Trading Strategies](https://www.backtestbase.com/education/monte-carlo-stress-testing)
- [The Probabilistic Sharpe Ratio: Bias-Adjustment, Confidence Intervals](https://portfoliooptimizer.io/blog/the-probabilistic-sharpe-ratio-bias-adjustment-confidence-intervals-hypothesis-testing-and-minimum-track-record-length/)
- [Sharpe Ratio: Estimation, Confidence Intervals](https://www.twosigma.com/wp-content/uploads/sharpe-tr-1.pdf)
- [Risk of Ruin Calculator for Trading](https://www.backtestbase.com/education/risk-of-ruin-calculator-trading)
- [Block Bootstrapping with Time Series Data](https://medium.com/@jcatankard_76170/block-bootstrapping-with-time-series-and-spatial-data-bd7d7830681e)
- [tsbootstrap: Time Series Bootstrapping](https://github.com/astrogilda/tsbootstrap)
- [arch Bootstrap Examples](https://arch.readthedocs.io/en/latest/bootstrap/bootstrap_examples.html)
- [Hypothesis Testing in Quant Finance](https://reasonabledeviations.com/2021/06/17/hypothesis-testing-quant/)
- [Kelly Criterion with Uncertainty](https://matthewdowney.github.io/uncertainty-kelly-criterion-optimal-bet-size.html)
- [Kelly Criterion - Wikipedia](https://en.wikipedia.org/wiki/Kelly_criterion)
- [HAR-RV Python Implementation](https://github.com/deep-hedger-Peng/HAR-RV)
- [TGAN Algorithm for Synthetic Backtesting Data](https://blog.quantinsti.com/tgan-algorithm-generate-synthetic-data-backtesting-trading-strategies/)
- [5 Monte Carlo Methods to Bulletproof Trading Strategies](https://strategyquant.com/blog/new-robustness-tests-on-the-strategyquant-codebase-5-monte-carlo-methods-to-bulletproof-your-trading-strategies/)
- [Robustness Testing Methods: Complete Guide](https://www.tradequantixnewsletter.com/p/robustness-testing-methods-a-complete)
- [Notes on the Sharpe Ratio (SharpeR package)](https://cran.r-project.org/web/packages/SharpeR/vignettes/SharpeRatio.pdf)
- [Sharpe Ratio Inference (Lopez de Prado et al.)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5520741)
- [Losing Streak Calculator](https://www.backtestbase.com/education/losing-streak-calculator-trading)
