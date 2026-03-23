"""Backtest performance metrics with Lo-corrected Sharpe and regime breakdowns.

Key differences from benchmarks/metrics.py:
  - Lo-corrected Sharpe (accounts for autocorrelation in returns)
  - CVaR (Conditional Value at Risk) at 5%
  - Rolling 30-day Sharpe series
  - Regime-specific metric breakdowns
  - Cost sensitivity analysis
  - Trade-level statistics with expectancy

Annualization: sqrt(105,120) for 24/7 crypto. NEVER sqrt(252).
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import structlog

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants — 5-min BTC bars, 24/7
# ---------------------------------------------------------------------------
BARS_PER_DAY: int = 288
BARS_PER_YEAR: float = BARS_PER_DAY * 365.0  # 105,120
SQRT_BARS_PER_YEAR: float = np.sqrt(BARS_PER_YEAR)  # ~324.22
RISK_FREE_ANNUAL: float = 0.05
RISK_FREE_PER_BAR: float = (1 + RISK_FREE_ANNUAL) ** (1 / BARS_PER_YEAR) - 1


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class TradeRecord:
    """A single completed trade."""

    entry_bar: int
    exit_bar: int
    side: str  # "long" or "short"
    entry_price: float
    exit_price: float
    quantity: float
    pnl_usd: float
    return_pct: float
    bars_held: int
    entry_cost_bps: float = 0.0
    exit_cost_bps: float = 0.0
    funding_paid_usd: float = 0.0


@dataclasses.dataclass(frozen=True)
class BacktestResult:
    """Comprehensive backtest performance summary."""

    # Core risk-adjusted ratios
    sharpe_ratio: float  # Lo-corrected
    sharpe_raw: float  # uncorrected
    sortino_ratio: float
    calmar_ratio: float

    # Returns
    total_return: float
    annualized_return: float
    cvar_5pct: float  # Conditional VaR at 5% (per-bar)

    # Drawdown
    max_drawdown: float
    max_drawdown_duration_bars: int
    avg_drawdown: float

    # Trade stats
    total_trades: int
    win_rate: float
    profit_factor: float
    expectancy_per_trade: float  # avg PnL per trade in bps
    avg_win_bps: float
    avg_loss_bps: float
    avg_bars_per_trade: float
    trades_per_day: float

    # Cost breakdown
    total_fee_usd: float
    total_slippage_usd: float
    total_funding_usd: float
    total_cost_usd: float

    # Distribution
    skewness: float
    kurtosis: float

    # Series (not in __eq__)
    equity_curve: NDArray[np.float64] = dataclasses.field(
        default_factory=lambda: np.array([]), repr=False, compare=False
    )
    rolling_sharpe_30d: NDArray[np.float64] = dataclasses.field(
        default_factory=lambda: np.array([]), repr=False, compare=False
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict, excluding large arrays."""
        d = dataclasses.asdict(self)
        d.pop("equity_curve", None)
        d.pop("rolling_sharpe_30d", None)
        return d

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"Sharpe (Lo-corrected): {self.sharpe_ratio:>10.3f}",
            f"Sharpe (raw):          {self.sharpe_raw:>10.3f}",
            f"Sortino:               {self.sortino_ratio:>10.3f}",
            f"Calmar:                {self.calmar_ratio:>10.3f}",
            f"Annual Return:         {self.annualized_return:>10.2%}",
            f"Total Return:          {self.total_return:>10.2%}",
            f"Max Drawdown:          {self.max_drawdown:>10.2%}",
            f"Max DD Duration:       {self.max_drawdown_duration_bars:>10d} bars"
            f" ({self.max_drawdown_duration_bars / BARS_PER_DAY:.1f} days)",
            f"CVaR (5%):             {self.cvar_5pct:>10.4%}",
            f"Win Rate:              {self.win_rate:>10.2%}",
            f"Profit Factor:         {self.profit_factor:>10.3f}",
            f"Expectancy/trade:      {self.expectancy_per_trade:>10.2f} bps",
            f"Trades/Day:            {self.trades_per_day:>10.2f}",
            f"Total Trades:          {self.total_trades:>10d}",
            f"Total Costs:           ${self.total_cost_usd:>10.2f}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core metric computations
# ---------------------------------------------------------------------------
def lo_corrected_sharpe(
    returns: NDArray[np.float64],
    max_lag: int = 6,
) -> float:
    """Sharpe ratio with Lo (2002) autocorrelation correction.

    The naive Sharpe overestimates when returns are autocorrelated.
    Lo's correction: SR_corrected = SR_naive * sqrt(q / eta(q))
    where eta(q) = q + 2 * sum_{k=1}^{q-1} (q-k) * rho_k

    Args:
        returns: Per-bar excess returns.
        max_lag: Number of lags for ACF estimation.

    Returns:
        Lo-corrected annualized Sharpe ratio.
    """
    n = len(returns)
    if n < max_lag + 2:
        return _raw_sharpe(returns)

    mean_r = returns.mean()
    std_r = returns.std(ddof=1)
    if std_r < 1e-15:
        return 0.0

    raw_sr = mean_r / std_r

    # Estimate autocorrelation at lags 1..max_lag
    demeaned = returns - mean_r
    var = np.sum(demeaned**2) / n

    if var < 1e-30:
        return 0.0

    q = int(BARS_PER_YEAR)
    eta = float(q)
    for k in range(1, min(max_lag + 1, q)):
        if k >= n:
            break
        acf_k = np.sum(demeaned[k:] * demeaned[:-k]) / (n * var)
        eta += 2.0 * (q - k) * acf_k

    if eta <= 0:
        return _raw_sharpe(returns)

    correction = np.sqrt(q / eta)
    return float(raw_sr * np.sqrt(q) * correction / np.sqrt(q))


def _raw_sharpe(returns: NDArray[np.float64]) -> float:
    """Uncorrected annualized Sharpe."""
    if len(returns) < 2:
        return 0.0
    mean_r = returns.mean()
    std_r = returns.std(ddof=1)
    if std_r < 1e-15:
        return 0.0
    return float(mean_r / std_r * SQRT_BARS_PER_YEAR)


def sortino_ratio(returns: NDArray[np.float64]) -> float:
    """Annualized Sortino ratio (downside deviation only)."""
    if len(returns) < 2:
        return 0.0
    downside = returns[returns < 0]
    if len(downside) == 0:
        return 0.0 if returns.mean() <= 0 else 10.0  # cap
    downside_std = np.sqrt(np.mean(downside**2))
    if downside_std < 1e-15:
        return 0.0
    return float(returns.mean() / downside_std * SQRT_BARS_PER_YEAR)


def cvar(returns: NDArray[np.float64], alpha: float = 0.05) -> float:
    """Conditional Value at Risk (Expected Shortfall) at alpha level.

    Returns the mean of the worst alpha-fraction of returns.
    Result is negative (a loss).
    """
    if len(returns) == 0:
        return 0.0
    cutoff = int(max(1, len(returns) * alpha))
    sorted_returns = np.sort(returns)
    return float(sorted_returns[:cutoff].mean())


def max_drawdown_info(
    equity_curve: NDArray[np.float64],
) -> tuple[float, int, float]:
    """Compute max drawdown, max duration, and average drawdown.

    Args:
        equity_curve: Cumulative equity (e.g. starting at 1.0).

    Returns:
        (max_drawdown_fraction, max_duration_bars, avg_drawdown_fraction)
    """
    if len(equity_curve) < 2:
        return 0.0, 0, 0.0

    running_max = np.maximum.accumulate(equity_curve)
    dd_series = (equity_curve - running_max) / np.where(running_max > 0, running_max, 1.0)

    max_dd = float(abs(dd_series.min()))

    # Duration: longest streak below running max
    in_dd = equity_curve < running_max
    if not np.any(in_dd):
        return max_dd, 0, 0.0

    # Group consecutive drawdown bars
    changes = np.diff(in_dd.astype(int), prepend=0)
    group_ids = np.cumsum(changes != 0)
    dd_groups = group_ids[in_dd]
    if len(dd_groups) == 0:
        return max_dd, 0, 0.0

    _unique, counts = np.unique(dd_groups, return_counts=True)
    max_duration = int(counts.max())

    avg_dd = float(abs(dd_series[in_dd].mean())) if np.any(in_dd) else 0.0

    return max_dd, max_duration, avg_dd


def rolling_sharpe(
    returns: NDArray[np.float64],
    window_bars: int = BARS_PER_DAY * 30,
) -> NDArray[np.float64]:
    """Rolling annualized Sharpe ratio.

    Args:
        returns: Per-bar returns.
        window_bars: Rolling window size (default: 30 days).

    Returns:
        Array of rolling Sharpe values (NaN where insufficient data).
    """
    n = len(returns)
    result = np.full(n, np.nan)

    if n < window_bars:
        return result

    # Use cumulative sums for efficient rolling mean/std
    for i in range(window_bars, n + 1):
        window = returns[i - window_bars : i]
        mean_w = window.mean()
        std_w = window.std(ddof=1)
        if std_w > 1e-15:
            result[i - 1] = float(mean_w / std_w * SQRT_BARS_PER_YEAR)
        else:
            result[i - 1] = 0.0

    return result


# ---------------------------------------------------------------------------
# Trade-level analysis
# ---------------------------------------------------------------------------
def analyze_trades(
    trades: list[TradeRecord],
) -> dict[str, float]:
    """Compute trade-level statistics from a list of TradeRecords.

    Returns dict with: win_rate, profit_factor, expectancy_bps,
    avg_win_bps, avg_loss_bps, avg_bars_per_trade.
    """
    if not trades:
        return {
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "expectancy_bps": 0.0,
            "avg_win_bps": 0.0,
            "avg_loss_bps": 0.0,
            "avg_bars_per_trade": 0.0,
        }

    returns_bps = np.array([t.return_pct * 10_000 for t in trades])
    wins = returns_bps[returns_bps > 0]
    losses = returns_bps[returns_bps <= 0]

    win_rate = len(wins) / len(returns_bps)
    avg_win = float(wins.mean()) if len(wins) > 0 else 0.0
    avg_loss = float(losses.mean()) if len(losses) > 0 else 0.0

    gross_profit = float(wins.sum()) if len(wins) > 0 else 0.0
    gross_loss = float(abs(losses.sum())) if len(losses) > 0 else 1e-10
    profit_factor = gross_profit / gross_loss

    expectancy = float(returns_bps.mean())
    avg_bars = float(np.mean([t.bars_held for t in trades]))

    return {
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "expectancy_bps": expectancy,
        "avg_win_bps": avg_win,
        "avg_loss_bps": avg_loss,
        "avg_bars_per_trade": avg_bars,
    }


# ---------------------------------------------------------------------------
# Regime breakdowns
# ---------------------------------------------------------------------------
def regime_metrics(
    returns: NDArray[np.float64],
    regime_labels: NDArray[np.int32] | None,
) -> dict[int, dict[str, float]]:
    """Compute per-regime performance metrics.

    Args:
        returns: Per-bar returns.
        regime_labels: Integer regime label per bar (same length as returns).

    Returns:
        Dict mapping regime_id -> {sharpe, return, max_dd, n_bars}.
    """
    if regime_labels is None or len(regime_labels) != len(returns):
        return {}

    results: dict[int, dict[str, float]] = {}
    for regime_id in np.unique(regime_labels):
        mask = regime_labels == regime_id
        regime_ret = returns[mask]
        n = int(mask.sum())
        if n < 10:
            continue

        excess = regime_ret - RISK_FREE_PER_BAR
        sharpe = _raw_sharpe(excess)
        total_ret = float((1 + regime_ret).prod() - 1)
        eq = np.cumprod(1 + regime_ret)
        max_dd, _, _ = max_drawdown_info(eq)

        results[int(regime_id)] = {
            "sharpe": sharpe,
            "total_return": total_ret,
            "max_drawdown": max_dd,
            "n_bars": n,
            "mean_return_bps": float(regime_ret.mean() * 10_000),
        }

    return results


# ---------------------------------------------------------------------------
# Cost sensitivity analysis
# ---------------------------------------------------------------------------
def cost_sensitivity(
    gross_returns: NDArray[np.float64],
    n_trades: int,
    n_bars: int,
    cost_levels_bps: list[float] | None = None,
) -> list[dict[str, float]]:
    """Run backtest at multiple cost levels to find break-even point.

    Approximation: subtracts cost_per_bar = (n_trades * cost_bps) / n_bars
    from each bar's return.

    Args:
        gross_returns: Per-bar gross returns (before costs).
        n_trades: Total number of trades.
        n_bars: Total number of bars.
        cost_levels_bps: Round-trip costs to test (default: 0,4,8,12,16,20).

    Returns:
        List of {cost_bps, sharpe, total_return, annualized_return}.
    """
    if cost_levels_bps is None:
        cost_levels_bps = [0.0, 4.0, 8.0, 12.0, 16.0, 20.0]

    results = []
    for cost_bps in cost_levels_bps:
        cost_per_trade = cost_bps * 1e-4
        total_cost_frac = cost_per_trade * n_trades
        cost_per_bar = total_cost_frac / max(n_bars, 1)

        net_returns = gross_returns - cost_per_bar
        excess = net_returns - RISK_FREE_PER_BAR
        sharpe = _raw_sharpe(excess)

        cum = float(np.prod(1 + net_returns) - 1)
        try:
            ann = (1 + cum) ** (BARS_PER_YEAR / max(n_bars, 1)) - 1
        except (OverflowError, FloatingPointError):
            ann = float("inf") if cum > 0 else float("-inf")

        results.append(
            {
                "cost_bps": cost_bps,
                "sharpe": sharpe,
                "total_return": cum,
                "annualized_return": ann,
            }
        )

    return results


def find_breakeven_cost(
    sensitivity: list[dict[str, float]],
) -> float:
    """Find the cost level where Sharpe crosses zero.

    Returns bps value (interpolated), or inf if always positive.
    """
    prev = None
    for entry in sorted(sensitivity, key=lambda x: x["cost_bps"]):
        if prev is not None and prev["sharpe"] > 0 >= entry["sharpe"]:
            # Linear interpolation
            s1, c1 = prev["sharpe"], prev["cost_bps"]
            s2, c2 = entry["sharpe"], entry["cost_bps"]
            denom = s1 - s2
            if abs(denom) < 1e-12:
                return c1
            return c1 + s1 * (c2 - c1) / denom
        prev = entry

    # All positive
    if sensitivity and all(e["sharpe"] > 0 for e in sensitivity):
        return float("inf")
    return 0.0


# ---------------------------------------------------------------------------
# Master computation
# ---------------------------------------------------------------------------
def compute_backtest_result(
    returns: NDArray[np.float64],
    trades: list[TradeRecord] | None = None,
    regime_labels: NDArray[np.int32] | None = None,
    total_fee_usd: float = 0.0,
    total_slippage_usd: float = 0.0,
    total_funding_usd: float = 0.0,
) -> BacktestResult:
    """Compute comprehensive backtest metrics from per-bar net returns.

    Args:
        returns: Per-bar net returns (after all costs).
        trades: Optional list of TradeRecords for trade-level stats.
        regime_labels: Optional regime labels per bar.
        total_fee_usd: Total fees paid in USD.
        total_slippage_usd: Total slippage in USD.
        total_funding_usd: Total funding paid in USD.

    Returns:
        BacktestResult with all metrics.
    """
    n = len(returns)
    if n == 0:
        return _empty_result()

    excess = returns - RISK_FREE_PER_BAR

    # Sharpe
    sharpe_lo = lo_corrected_sharpe(excess)
    sharpe_uncorrected = _raw_sharpe(excess)

    # Sortino
    sort = sortino_ratio(excess)

    # Returns
    equity = np.cumprod(1 + returns)
    total_ret = float(equity[-1] - 1)
    ann_ret = (1 + total_ret) ** (BARS_PER_YEAR / n) - 1

    # Drawdown
    max_dd, max_dd_dur, avg_dd = max_drawdown_info(equity)

    # Calmar
    calmar = ann_ret / max_dd if max_dd > 1e-10 else 0.0

    # CVaR
    cvar_5 = cvar(returns, alpha=0.05)

    # Rolling Sharpe
    roll_sharpe = rolling_sharpe(excess)

    # Trade stats
    trade_stats = analyze_trades(trades or [])
    n_trades = len(trades) if trades else 0
    n_days = n / BARS_PER_DAY
    trades_per_day = n_trades / max(n_days, 1e-10)

    # Distribution
    skew = float(pd.Series(returns).skew()) if n > 2 else 0.0
    kurt = float(pd.Series(returns).kurtosis()) if n > 3 else 0.0

    total_cost = total_fee_usd + total_slippage_usd + total_funding_usd

    return BacktestResult(
        sharpe_ratio=sharpe_lo,
        sharpe_raw=sharpe_uncorrected,
        sortino_ratio=sort,
        calmar_ratio=calmar,
        total_return=total_ret,
        annualized_return=ann_ret,
        cvar_5pct=cvar_5,
        max_drawdown=max_dd,
        max_drawdown_duration_bars=max_dd_dur,
        avg_drawdown=avg_dd,
        total_trades=n_trades,
        win_rate=trade_stats["win_rate"],
        profit_factor=trade_stats["profit_factor"],
        expectancy_per_trade=trade_stats["expectancy_bps"],
        avg_win_bps=trade_stats["avg_win_bps"],
        avg_loss_bps=trade_stats["avg_loss_bps"],
        avg_bars_per_trade=trade_stats["avg_bars_per_trade"],
        trades_per_day=trades_per_day,
        total_fee_usd=total_fee_usd,
        total_slippage_usd=total_slippage_usd,
        total_funding_usd=total_funding_usd,
        total_cost_usd=total_cost,
        skewness=skew,
        kurtosis=kurt,
        equity_curve=equity,
        rolling_sharpe_30d=roll_sharpe,
    )


def _empty_result() -> BacktestResult:
    """Return an empty BacktestResult for edge cases."""
    return BacktestResult(
        sharpe_ratio=0.0,
        sharpe_raw=0.0,
        sortino_ratio=0.0,
        calmar_ratio=0.0,
        total_return=0.0,
        annualized_return=0.0,
        cvar_5pct=0.0,
        max_drawdown=0.0,
        max_drawdown_duration_bars=0,
        avg_drawdown=0.0,
        total_trades=0,
        win_rate=0.0,
        profit_factor=0.0,
        expectancy_per_trade=0.0,
        avg_win_bps=0.0,
        avg_loss_bps=0.0,
        avg_bars_per_trade=0.0,
        trades_per_day=0.0,
        total_fee_usd=0.0,
        total_slippage_usd=0.0,
        total_funding_usd=0.0,
        total_cost_usd=0.0,
        skewness=0.0,
        kurtosis=0.0,
    )
