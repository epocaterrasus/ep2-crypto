"""Polymarket binary payoff backtester.

Simulates trading 5-minute BTC binary prediction markets using historical
directional signals. Models the binary payoff structure exactly:
- WIN: cost of shares → $1.00/share (payout = 1/price - 1 return)
- LOSS: cost of shares → $0.00 (full loss of premium)
- Fee: ~2% of notional each way

This is NOT a continuous PnL simulation. Each trade is a separate binary
bet that resolves in 5 minutes. The backtest runs signal × history and
produces metrics appropriate for binary market evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class BinaryBar:
    """A single 5-minute bar for binary payoff simulation."""

    timestamp_ms: int
    open_price: float  # BTC price at bar open
    close_price: float  # BTC price at bar close (= resolution price)
    signal: int  # +1 = UP, -1 = DOWN, 0 = NO TRADE
    model_prob: float  # Model's probability for predicted direction (0-1)
    market_price_yes: float  # Polymarket YES share price at bar open (0-1)
    market_price_no: float  # Polymarket NO share price at bar open (0-1)


@dataclass
class BinaryTrade:
    """A single completed binary bet."""

    timestamp_ms: int
    signal: int  # +1 or -1
    direction: str  # "yes" or "no"
    market_price: float  # Share price paid
    shares: float  # Number of shares bought
    cost_usd: float  # Total cost (shares × price)
    fee_usd: float  # Round-trip fee
    outcome: str  # "win" or "loss"
    pnl_gross: float  # Gross payoff before fees
    pnl_net: float  # Net payoff after fees
    price_at_entry: float  # BTC price when bet was placed
    price_at_resolution: float  # BTC price at resolution


@dataclass
class BinaryBacktestResult:
    """Aggregate results from a binary payoff backtest."""

    # Summary stats
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    no_trades: int = 0

    # PnL
    total_pnl_net: float = 0.0
    total_cost: float = 0.0
    total_fees: float = 0.0
    max_drawdown: float = 0.0
    peak_equity: float = 0.0

    # Per-trade stats
    avg_win_usd: float = 0.0
    avg_loss_usd: float = 0.0
    profit_factor: float = 0.0

    # Rate metrics
    win_rate: float = 0.0
    edge_per_bet: float = 0.0  # Expected value per $1 wagered

    # Risk metrics
    sharpe: float = 0.0  # Annualized over trade sequence
    sortino: float = 0.0
    roi: float = 0.0  # Net PnL / total cost

    # Trade list
    trades: list[BinaryTrade] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"Binary Backtest: {self.total_trades} trades | "
            f"Win rate: {self.win_rate:.1%} | "
            f"Net PnL: ${self.total_pnl_net:.2f} | "
            f"ROI: {self.roi:.1%} | "
            f"Sharpe: {self.sharpe:.2f} | "
            f"Max DD: {self.max_drawdown:.1%}"
        )


# ---------------------------------------------------------------------------
# Fee model
# ---------------------------------------------------------------------------


@dataclass
class BinaryFeeModel:
    """Fee model for Polymarket binary markets.

    Polymarket charges ~2% of notional as a taker fee.
    There is no fee on resolution (winner receives $1/share automatically).
    """

    taker_fee_rate: float = 0.02  # 2% of notional

    def entry_fee(self, shares: float, price: float) -> float:
        return round(shares * price * self.taker_fee_rate, 6)

    def round_trip_fee(self, shares: float, price: float) -> float:
        # Polymarket only charges on entry (no exit fee — market auto-resolves)
        return self.entry_fee(shares, price)


# ---------------------------------------------------------------------------
# Position sizer (simple for backtesting)
# ---------------------------------------------------------------------------


def compute_shares(
    capital: float,
    fraction: float,
    market_price: float,
    min_shares: float = 1.0,
) -> float:
    """Compute how many shares to buy given a capital fraction.

    Args:
        capital: Current equity in USD.
        fraction: Fraction of capital to risk (e.g. 0.02 = 2%).
        market_price: Current YES/NO share price (0-1).
        min_shares: Minimum order size.

    Returns:
        Number of shares (floored to 2 decimal places).
    """
    notional = capital * fraction
    shares = notional / market_price
    return max(round(shares, 2), min_shares)


# ---------------------------------------------------------------------------
# Core backtester
# ---------------------------------------------------------------------------


class PolymarketBacktester:
    """Simulate binary market betting using historical directional signals.

    Usage:
        backtester = PolymarketBacktester(
            initial_capital=10_000.0,
            bet_fraction=0.02,
            fee_model=BinaryFeeModel(),
        )
        result = backtester.run(bars)
    """

    def __init__(
        self,
        initial_capital: float = 10_000.0,
        bet_fraction: float = 0.02,
        fee_model: BinaryFeeModel | None = None,
        min_model_prob: float = 0.55,  # Skip bets below this confidence
        min_edge: float = 0.0,  # Skip if model_prob - market_price < min_edge
    ) -> None:
        if initial_capital <= 0:
            raise ValueError("initial_capital must be positive")
        if not 0 < bet_fraction <= 1.0:
            raise ValueError("bet_fraction must be in (0, 1]")

        self._initial_capital = initial_capital
        self._bet_fraction = bet_fraction
        self._fee_model = fee_model or BinaryFeeModel()
        self._min_model_prob = min_model_prob
        self._min_edge = min_edge

    def run(self, bars: list[BinaryBar]) -> BinaryBacktestResult:
        """Simulate all bars and return aggregate results.

        Each bar with signal != 0 is a potential bet. The bar's close price
        determines resolution: UP bet wins if close > open, DOWN if close < open.
        """
        if not bars:
            return BinaryBacktestResult()

        capital = self._initial_capital
        trades: list[BinaryTrade] = []
        equity_curve: list[float] = [capital]
        no_trades = 0

        for bar in bars:
            if bar.signal == 0:
                no_trades += 1
                equity_curve.append(capital)
                continue

            direction = "yes" if bar.signal == 1 else "no"
            market_price = (
                bar.market_price_yes if direction == "yes" else bar.market_price_no
            )

            # Skip low-confidence or low-edge bets
            if bar.model_prob < self._min_model_prob:
                no_trades += 1
                equity_curve.append(capital)
                continue

            edge = bar.model_prob - market_price
            if edge < self._min_edge:
                no_trades += 1
                equity_curve.append(capital)
                continue

            shares = compute_shares(capital, self._bet_fraction, market_price)
            cost = round(shares * market_price, 6)
            fee = self._fee_model.round_trip_fee(shares, market_price)

            # Resolve: UP wins if close > open, DOWN wins if close < open
            price_went_up = bar.close_price > bar.open_price
            won = (bar.signal == 1 and price_went_up) or (
                bar.signal == -1 and not price_went_up
            )

            if won:
                gross_payoff = shares * 1.0  # Shares pay out $1 each
                pnl_gross = gross_payoff - cost
                pnl_net = pnl_gross - fee
                outcome = "win"
            else:
                pnl_gross = -cost
                pnl_net = -cost - fee
                outcome = "loss"

            capital = max(0.0, capital + pnl_net)

            trade = BinaryTrade(
                timestamp_ms=bar.timestamp_ms,
                signal=bar.signal,
                direction=direction,
                market_price=market_price,
                shares=shares,
                cost_usd=cost,
                fee_usd=fee,
                outcome=outcome,
                pnl_gross=pnl_gross,
                pnl_net=pnl_net,
                price_at_entry=bar.open_price,
                price_at_resolution=bar.close_price,
            )
            trades.append(trade)
            equity_curve.append(capital)

        return self._compute_result(
            trades=trades,
            equity_curve=equity_curve,
            no_trades=no_trades,
        )

    def _compute_result(
        self,
        trades: list[BinaryTrade],
        equity_curve: list[float],
        no_trades: int,
    ) -> BinaryBacktestResult:
        result = BinaryBacktestResult(
            total_trades=len(trades),
            no_trades=no_trades,
            trades=trades,
            equity_curve=equity_curve,
        )

        if not trades:
            return result

        wins = [t for t in trades if t.outcome == "win"]
        losses = [t for t in trades if t.outcome == "loss"]

        result.wins = len(wins)
        result.losses = len(losses)
        result.total_pnl_net = sum(t.pnl_net for t in trades)
        result.total_cost = sum(t.cost_usd for t in trades)
        result.total_fees = sum(t.fee_usd for t in trades)

        result.avg_win_usd = (
            float(np.mean([t.pnl_net for t in wins])) if wins else 0.0
        )
        result.avg_loss_usd = (
            float(np.mean([t.pnl_net for t in losses])) if losses else 0.0
        )

        gross_wins = sum(t.pnl_net for t in wins)
        gross_losses = abs(sum(t.pnl_net for t in losses))
        result.profit_factor = (
            gross_wins / gross_losses if gross_losses > 0 else float("inf")
        )

        result.win_rate = len(wins) / len(trades)
        result.edge_per_bet = (
            result.total_pnl_net / result.total_cost if result.total_cost > 0 else 0.0
        )
        result.roi = result.total_pnl_net / self._initial_capital

        result.max_drawdown = self._compute_max_drawdown(equity_curve)
        result.peak_equity = max(equity_curve)

        pnl_series = np.array([t.pnl_net for t in trades])
        result.sharpe = self._compute_sharpe(pnl_series)
        result.sortino = self._compute_sortino(pnl_series)

        return result

    @staticmethod
    def _compute_max_drawdown(equity_curve: list[float]) -> float:
        """Maximum peak-to-trough drawdown as a fraction of peak equity."""
        if not equity_curve:
            return 0.0
        arr = np.array(equity_curve, dtype=float)
        peaks = np.maximum.accumulate(arr)
        drawdowns = (peaks - arr) / np.where(peaks > 0, peaks, 1.0)
        return float(np.max(drawdowns))

    @staticmethod
    def _compute_sharpe(pnl_series: np.ndarray) -> float:
        """Per-trade Sharpe (not annualized — binary bets don't have a fixed horizon)."""
        if len(pnl_series) < 2:
            return 0.0
        mean = float(np.mean(pnl_series))
        std = float(np.std(pnl_series, ddof=1))
        if std == 0:
            return 0.0
        return mean / std

    @staticmethod
    def _compute_sortino(pnl_series: np.ndarray) -> float:
        """Per-trade Sortino ratio (downside deviation only)."""
        if len(pnl_series) < 2:
            return 0.0
        mean = float(np.mean(pnl_series))
        downside = pnl_series[pnl_series < 0]
        if len(downside) == 0:
            return float("inf")
        downside_std = float(np.std(downside, ddof=1))
        if downside_std == 0:
            return 0.0
        return mean / downside_std


# ---------------------------------------------------------------------------
# Comparison report
# ---------------------------------------------------------------------------


def comparison_report(
    results: dict[str, BinaryBacktestResult],
) -> dict[str, Any]:
    """Compare multiple strategy results side-by-side.

    Args:
        results: Mapping of strategy_name → BinaryBacktestResult.

    Returns:
        Dict with per-metric comparison rows.
    """
    metrics = ["win_rate", "total_pnl_net", "roi", "sharpe", "sortino",
               "profit_factor", "max_drawdown", "total_trades", "edge_per_bet"]

    report: dict[str, Any] = {}
    for metric in metrics:
        report[metric] = {
            name: round(getattr(res, metric), 4)
            for name, res in results.items()
        }
    return report
