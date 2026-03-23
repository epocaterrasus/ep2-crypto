"""Performance metrics for benchmark and strategy evaluation.

All metrics operate on pandas Series of per-bar returns and position signals.
Designed for 5-minute BTC bars (288 bars/day, ~105,120 bars/year).
"""

from __future__ import annotations

import dataclasses
import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)

# 5-min bars: 288 per day, 365.25 days/year
BARS_PER_DAY = 288
BARS_PER_YEAR = BARS_PER_DAY * 365.25
RISK_FREE_RATE_ANNUAL = 0.05  # ~5% risk-free (T-bill proxy)
RISK_FREE_PER_BAR = (1 + RISK_FREE_RATE_ANNUAL) ** (1 / BARS_PER_YEAR) - 1


@dataclasses.dataclass(frozen=True)
class BacktestMetrics:
    """Complete performance summary for a strategy backtest."""

    sharpe_ratio: float
    annualized_return: float
    total_return: float
    max_drawdown: float
    max_drawdown_duration_bars: int
    win_rate: float
    profit_factor: float
    trades_per_day: float
    total_trades: int
    avg_trade_return: float
    avg_win: float
    avg_loss: float
    calmar_ratio: float
    sortino_ratio: float
    skewness: float
    kurtosis: float
    avg_bars_in_trade: float

    def to_dict(self) -> dict[str, float]:
        return dataclasses.asdict(self)

    def summary(self) -> str:
        lines = [
            f"Sharpe Ratio:       {self.sharpe_ratio:>10.3f}",
            f"Sortino Ratio:      {self.sortino_ratio:>10.3f}",
            f"Calmar Ratio:       {self.calmar_ratio:>10.3f}",
            f"Annual Return:      {self.annualized_return:>10.2%}",
            f"Total Return:       {self.total_return:>10.2%}",
            f"Max Drawdown:       {self.max_drawdown:>10.2%}",
            f"Max DD Duration:    {self.max_drawdown_duration_bars:>10d} bars ({self.max_drawdown_duration_bars / BARS_PER_DAY:.1f} days)",
            f"Win Rate:           {self.win_rate:>10.2%}",
            f"Profit Factor:      {self.profit_factor:>10.3f}",
            f"Trades/Day:         {self.trades_per_day:>10.2f}",
            f"Total Trades:       {self.total_trades:>10d}",
            f"Avg Trade Return:   {self.avg_trade_return:>10.4%}",
            f"Avg Win:            {self.avg_win:>10.4%}",
            f"Avg Loss:           {self.avg_loss:>10.4%}",
            f"Avg Bars/Trade:     {self.avg_bars_in_trade:>10.1f}",
            f"Skewness:           {self.skewness:>10.3f}",
            f"Kurtosis:           {self.kurtosis:>10.3f}",
        ]
        return "\n".join(lines)


def compute_metrics(
    returns: pd.Series,
    positions: pd.Series,
    trading_cost_bps: float = 3.0,
) -> BacktestMetrics:
    """Compute full metrics from per-bar strategy returns and position signals.

    Args:
        returns: Per-bar log returns of the strategy (already position-weighted).
        positions: Position signal per bar (-1, 0, or +1). Used to count trades
                   and compute trade-level statistics.
        trading_cost_bps: Round-trip cost in basis points, applied per trade entry.
    """
    if len(returns) == 0 or len(positions) == 0:
        logger.warning("Empty returns or positions passed to compute_metrics")
        return _empty_metrics()

    returns = returns.copy()
    positions = positions.copy()

    # --- Apply trading costs at every position change ---
    position_changes = positions.diff().fillna(positions.iloc[0]).abs()
    # Each unit of position change costs half the round-trip (entry OR exit)
    cost_per_bar = position_changes * (trading_cost_bps / 10_000)
    net_returns = returns - cost_per_bar

    # --- Sharpe ratio (annualized) ---
    excess = net_returns - RISK_FREE_PER_BAR
    sharpe = _annualized_sharpe(excess)

    # --- Sortino ratio ---
    downside = excess[excess < 0]
    downside_std = np.sqrt((downside**2).mean()) if len(downside) > 0 else 1e-10
    sortino = (excess.mean() * np.sqrt(BARS_PER_YEAR)) / downside_std

    # --- Returns ---
    cumulative = (1 + net_returns).cumprod()
    total_return = cumulative.iloc[-1] - 1 if len(cumulative) > 0 else 0.0
    n_bars = len(net_returns)
    annualized_return = (1 + total_return) ** (BARS_PER_YEAR / max(n_bars, 1)) - 1

    # --- Drawdown ---
    running_max = cumulative.cummax()
    drawdown_series = (cumulative - running_max) / running_max
    max_drawdown = abs(drawdown_series.min()) if len(drawdown_series) > 0 else 0.0

    # Max drawdown duration (in bars)
    dd_duration = _max_drawdown_duration(cumulative)

    # Calmar
    calmar = annualized_return / max_drawdown if max_drawdown > 1e-10 else 0.0

    # --- Trade-level stats ---
    trade_boundaries = positions.diff().fillna(positions.iloc[0])
    trade_starts = trade_boundaries[trade_boundaries != 0].index
    trades = _extract_trades(net_returns, positions, trade_starts)

    total_trades = len(trades)
    n_days = n_bars / BARS_PER_DAY
    trades_per_day = total_trades / max(n_days, 1e-10)

    if total_trades > 0:
        trade_returns = np.array([t["return"] for t in trades])
        wins = trade_returns[trade_returns > 0]
        losses = trade_returns[trade_returns <= 0]
        win_rate = len(wins) / total_trades
        avg_trade_return = float(trade_returns.mean())
        avg_win = float(wins.mean()) if len(wins) > 0 else 0.0
        avg_loss = float(losses.mean()) if len(losses) > 0 else 0.0
        gross_profit = wins.sum() if len(wins) > 0 else 0.0
        gross_loss = abs(losses.sum()) if len(losses) > 0 else 1e-10
        profit_factor = gross_profit / gross_loss
        avg_bars_in_trade = float(np.mean([t["bars"] for t in trades]))
    else:
        win_rate = 0.0
        avg_trade_return = 0.0
        avg_win = 0.0
        avg_loss = 0.0
        profit_factor = 0.0
        avg_bars_in_trade = 0.0

    return BacktestMetrics(
        sharpe_ratio=float(sharpe),
        annualized_return=float(annualized_return),
        total_return=float(total_return),
        max_drawdown=float(max_drawdown),
        max_drawdown_duration_bars=int(dd_duration),
        win_rate=float(win_rate),
        profit_factor=float(profit_factor),
        trades_per_day=float(trades_per_day),
        total_trades=int(total_trades),
        avg_trade_return=float(avg_trade_return),
        avg_win=float(avg_win),
        avg_loss=float(avg_loss),
        calmar_ratio=float(calmar),
        sortino_ratio=float(sortino),
        skewness=float(net_returns.skew()),
        kurtosis=float(net_returns.kurtosis()),
        avg_bars_in_trade=float(avg_bars_in_trade),
    )


def _annualized_sharpe(excess_returns: pd.Series) -> float:
    """Annualized Sharpe from per-bar excess returns."""
    if len(excess_returns) < 2:
        return 0.0
    mean = excess_returns.mean()
    std = excess_returns.std(ddof=1)
    if std < 1e-15:
        return 0.0
    return float(mean / std * np.sqrt(BARS_PER_YEAR))


def _max_drawdown_duration(cumulative: pd.Series) -> int:
    """Longest streak (in bars) below the running maximum."""
    running_max = cumulative.cummax()
    in_drawdown = cumulative < running_max
    if not in_drawdown.any():
        return 0
    groups = (~in_drawdown).cumsum()
    dd_groups = groups[in_drawdown]
    if len(dd_groups) == 0:
        return 0
    return int(dd_groups.value_counts().max())


def _extract_trades(
    returns: pd.Series,
    positions: pd.Series,
    trade_starts: pd.Index,
) -> list[dict]:
    """Extract individual trades from position changes."""
    trades = []
    pos_array = positions.values
    ret_array = returns.values

    # Find indices where position changes (trade boundaries)
    change_mask = np.diff(pos_array, prepend=pos_array[0] - 1) != 0
    change_indices = np.where(change_mask)[0]

    for i in range(len(change_indices)):
        start = change_indices[i]
        end = change_indices[i + 1] if i + 1 < len(change_indices) else len(pos_array)
        if pos_array[start] == 0:
            continue  # Skip flat periods
        trade_ret = ret_array[start:end].sum()
        trades.append({"return": float(trade_ret), "bars": int(end - start)})

    return trades


def _empty_metrics() -> BacktestMetrics:
    return BacktestMetrics(
        sharpe_ratio=0.0,
        annualized_return=0.0,
        total_return=0.0,
        max_drawdown=0.0,
        max_drawdown_duration_bars=0,
        win_rate=0.0,
        profit_factor=0.0,
        trades_per_day=0.0,
        total_trades=0,
        avg_trade_return=0.0,
        avg_win=0.0,
        avg_loss=0.0,
        calmar_ratio=0.0,
        sortino_ratio=0.0,
        skewness=0.0,
        kurtosis=0.0,
        avg_bars_in_trade=0.0,
    )
