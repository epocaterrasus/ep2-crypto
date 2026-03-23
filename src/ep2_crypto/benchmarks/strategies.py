"""Benchmark strategies — vectorized implementations.

Each strategy takes a DataFrame with at minimum columns:
    open, high, low, close, volume
at 5-minute granularity and returns a pd.Series of positions (-1, 0, +1).

Some strategies require additional columns (funding_rate, obi, nq_close).
"""

from __future__ import annotations

import abc
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class BenchmarkStrategy(abc.ABC):
    """Base class for all benchmark strategies."""

    name: str = "base"

    @abc.abstractmethod
    def generate_positions(self, df: pd.DataFrame) -> pd.Series:
        """Return position series: +1 long, -1 short, 0 flat."""

    def validate_columns(self, df: pd.DataFrame, required: list[str]) -> None:
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"{self.name} requires columns {missing} not found in DataFrame")


# ---------------------------------------------------------------------------
# 1. Buy and Hold
# ---------------------------------------------------------------------------
class BuyAndHold(BenchmarkStrategy):
    """Always long. The simplest possible benchmark.

    Why it matters: Any directional strategy that cannot beat buy-and-hold in
    a bull market (after costs) is adding complexity for no value. In a bear
    market, buy-and-hold loses, so beating it is trivial — regime matters.

    Leverage variants (2x, 5x) show the volatility drag of leveraged holds,
    which is substantial at 5-min granularity due to compounding.
    """

    name = "buy_and_hold"

    def __init__(self, leverage: float = 1.0) -> None:
        self.leverage = leverage

    def generate_positions(self, df: pd.DataFrame) -> pd.Series:
        self.validate_columns(df, ["close"])
        return pd.Series(self.leverage, index=df.index, name="position")


# ---------------------------------------------------------------------------
# 2. Simple Momentum
# ---------------------------------------------------------------------------
class SimpleMomentum(BenchmarkStrategy):
    """If last N bars were net positive, go long; if negative, go short.

    This captures the simplest possible trend-following signal. At 5-min
    BTC, short lookback (N=1-3) captures micro-momentum but is noisy;
    longer lookback (N=10-20) is smoother but lags.

    Beating this proves the ML system captures momentum better than a
    raw return sign, likely through nonlinear feature interactions.
    """

    name = "simple_momentum"

    def __init__(self, lookback: int = 5) -> None:
        if lookback < 1:
            raise ValueError("lookback must be >= 1")
        self.lookback = lookback

    def generate_positions(self, df: pd.DataFrame) -> pd.Series:
        self.validate_columns(df, ["close"])
        returns = df["close"].pct_change()
        rolling_ret = returns.rolling(self.lookback).sum()
        positions = np.sign(rolling_ret)
        return pd.Series(positions, index=df.index, name="position").fillna(0.0)


# ---------------------------------------------------------------------------
# 3. Mean Reversion (RSI)
# ---------------------------------------------------------------------------
class MeanReversionRSI(BenchmarkStrategy):
    """Buy when RSI < oversold, sell when RSI > overbought.

    RSI is computed vectorized using Wilder's smoothing (EMA with
    alpha = 1/period). At 5-min BTC, mean reversion works in range-bound
    regimes but gets destroyed in trending regimes.

    Beating this proves the ML system can distinguish regimes OR capture
    mean-reversion timing better than a fixed-threshold RSI.
    """

    name = "mean_reversion_rsi"

    def __init__(
        self,
        period: int = 14,
        oversold: float = 30.0,
        overbought: float = 70.0,
    ) -> None:
        self.period = period
        self.oversold = oversold
        self.overbought = overbought

    def generate_positions(self, df: pd.DataFrame) -> pd.Series:
        self.validate_columns(df, ["close"])
        rsi = self._compute_rsi(df["close"], self.period)

        positions = pd.Series(0.0, index=df.index, name="position")
        positions[rsi < self.oversold] = 1.0  # Buy oversold
        positions[rsi > self.overbought] = -1.0  # Sell overbought

        # Forward-fill: hold position until opposite signal
        positions = positions.replace(0.0, np.nan).ffill().fillna(0.0)
        return positions

    @staticmethod
    def _compute_rsi(close: pd.Series, period: int) -> pd.Series:
        """Vectorized RSI using Wilder's smoothing (exponential)."""
        delta = close.diff()
        gain = delta.clip(lower=0.0)
        loss = (-delta).clip(lower=0.0)
        alpha = 1.0 / period
        avg_gain = gain.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, 1e-10)
        return 100.0 - (100.0 / (1.0 + rs))


# ---------------------------------------------------------------------------
# 4. Moving Average Crossover
# ---------------------------------------------------------------------------
class MovingAverageCrossover(BenchmarkStrategy):
    """Classic SMA or EMA crossover: long when fast > slow, short when fast < slow.

    The most widely known technical strategy. At 5-min BTC:
    - 5/20: responsive but whipsaw-heavy in sideways markets
    - 10/50: balanced, classic scalper setup
    - 20/100: smoother but misses many intraday moves

    Beating this proves the ML system adds value beyond trend direction,
    likely through entry/exit timing or multi-factor conditioning.
    """

    name = "ma_crossover"

    def __init__(
        self,
        fast_period: int = 10,
        slow_period: int = 50,
        ma_type: str = "ema",
    ) -> None:
        if fast_period >= slow_period:
            raise ValueError("fast_period must be < slow_period")
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.ma_type = ma_type

    def generate_positions(self, df: pd.DataFrame) -> pd.Series:
        self.validate_columns(df, ["close"])
        close = df["close"]
        if self.ma_type == "sma":
            fast = close.rolling(self.fast_period).mean()
            slow = close.rolling(self.slow_period).mean()
        else:
            fast = close.ewm(span=self.fast_period, adjust=False).mean()
            slow = close.ewm(span=self.slow_period, adjust=False).mean()

        positions = pd.Series(0.0, index=df.index, name="position")
        positions[fast > slow] = 1.0
        positions[fast < slow] = -1.0
        return positions.fillna(0.0)


# ---------------------------------------------------------------------------
# 5. Random Entry with Fixed Exit
# ---------------------------------------------------------------------------
class RandomEntry(BenchmarkStrategy):
    """Random direction, hold for N bars, exit. The TRUE null hypothesis.

    This is the most important benchmark. A strategy that cannot beat random
    entry has NO statistical edge — it's just lucky. We run many simulations
    to build a distribution of random strategy returns.

    Beating this (with statistical significance, p < 0.01) is the minimum
    bar for any ML strategy to be considered real.
    """

    name = "random_entry"

    def __init__(
        self,
        hold_bars: int = 3,
        seed: int = 42,
        entry_probability: float = 0.1,
    ) -> None:
        self.hold_bars = hold_bars
        self.seed = seed
        self.entry_probability = entry_probability

    def generate_positions(self, df: pd.DataFrame) -> pd.Series:
        self.validate_columns(df, ["close"])
        rng = np.random.default_rng(self.seed)
        n = len(df)

        positions = np.zeros(n)
        i = 0
        while i < n:
            if rng.random() < self.entry_probability:
                direction = rng.choice([-1.0, 1.0])
                end = min(i + self.hold_bars, n)
                positions[i:end] = direction
                i = end
            else:
                i += 1

        return pd.Series(positions, index=df.index, name="position")

    def generate_distribution(
        self,
        df: pd.DataFrame,
        n_simulations: int = 1000,
    ) -> list[pd.Series]:
        """Run many random simulations for statistical comparison."""
        results = []
        for sim in range(n_simulations):
            self.seed = sim
            positions = self.generate_positions(df)
            returns = df["close"].pct_change().fillna(0.0) * positions
            results.append(returns)
        return results


# ---------------------------------------------------------------------------
# 6. Volatility Breakout
# ---------------------------------------------------------------------------
class VolatilityBreakout(BenchmarkStrategy):
    """Enter when price moves > K * ATR in a single bar. Ride for N bars.

    A classic breakout strategy. At 5-min BTC, this captures large-move bars
    (news, liquidation cascades, whale entries). The challenge is that many
    breakouts are false (revert quickly) and slippage is highest during
    volatile moves.

    Beating this proves the ML system can distinguish real breakouts from
    noise, likely using order flow or cross-market context.
    """

    name = "volatility_breakout"

    def __init__(
        self,
        atr_period: int = 14,
        atr_multiplier: float = 2.0,
        hold_bars: int = 3,
    ) -> None:
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.hold_bars = hold_bars

    def generate_positions(self, df: pd.DataFrame) -> pd.Series:
        self.validate_columns(df, ["high", "low", "close"])
        atr = self._compute_atr(df, self.atr_period)
        bar_move = df["close"] - df["close"].shift(1)

        positions = np.zeros(len(df))
        threshold = atr * self.atr_multiplier

        # Vectorized signal generation
        long_signal = bar_move > threshold
        short_signal = bar_move < -threshold

        signal_indices = np.where(long_signal | short_signal)[0]
        for idx in signal_indices:
            direction = 1.0 if long_signal.iloc[idx] else -1.0
            end = min(idx + self.hold_bars, len(df))
            # Only set if not already in a trade
            if positions[idx] == 0:
                positions[idx:end] = direction

        return pd.Series(positions, index=df.index, name="position")

    @staticmethod
    def _compute_atr(df: pd.DataFrame, period: int) -> pd.Series:
        """Average True Range — vectorized."""
        high = df["high"]
        low = df["low"]
        prev_close = df["close"].shift(1)
        tr = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        return tr.ewm(span=period, adjust=False).mean()


# ---------------------------------------------------------------------------
# 7. Funding Rate Strategy
# ---------------------------------------------------------------------------
class FundingRateStrategy(BenchmarkStrategy):
    """Contrarian funding: short when funding high, long when funding negative.

    Funding rate reflects market consensus. When funding is very positive,
    longs are paying shorts — market is overleveraged long. Going short is
    contrarian and historically profitable on 8h funding cycles.

    At 5-min, the signal is slow (funding updates every 8h on most exchanges),
    so this strategy trades infrequently but captures regime-level sentiment.

    Beating this proves the ML system captures crowd positioning better
    than a single funding-rate threshold.
    """

    name = "funding_rate"

    def __init__(
        self,
        long_threshold: float = -0.0003,
        short_threshold: float = 0.0005,
    ) -> None:
        self.long_threshold = long_threshold
        self.short_threshold = short_threshold

    def generate_positions(self, df: pd.DataFrame) -> pd.Series:
        self.validate_columns(df, ["funding_rate"])
        fr = df["funding_rate"]
        positions = pd.Series(0.0, index=df.index, name="position")
        positions[fr > self.short_threshold] = -1.0
        positions[fr < self.long_threshold] = 1.0
        # Forward-fill: hold until opposite signal
        positions = positions.replace(0.0, np.nan).ffill().fillna(0.0)
        return positions


# ---------------------------------------------------------------------------
# 8. Order Book Imbalance Strategy
# ---------------------------------------------------------------------------
class OrderBookImbalance(BenchmarkStrategy):
    """Go long when OBI > threshold, short when OBI < -threshold.

    Order Book Imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume)
    Range: [-1, +1]. High OBI means more buying pressure on the book.

    This is the simplest microstructure strategy. At 5-min BTC, OBI is
    noisy but captures short-term supply/demand. The signal decays fast
    (sub-second at L1), so 5-min aggregated OBI is quite smoothed.

    Beating this proves the ML system extracts more from order flow
    than a simple imbalance ratio — likely through multi-level book
    features, trade flow imbalance, or absorption detection.
    """

    name = "order_book_imbalance"

    def __init__(
        self,
        long_threshold: float = 0.3,
        short_threshold: float = -0.3,
    ) -> None:
        self.long_threshold = long_threshold
        self.short_threshold = short_threshold

    def generate_positions(self, df: pd.DataFrame) -> pd.Series:
        self.validate_columns(df, ["obi"])
        obi = df["obi"]
        positions = pd.Series(0.0, index=df.index, name="position")
        positions[obi > self.long_threshold] = 1.0
        positions[obi < self.short_threshold] = -1.0
        return positions


# ---------------------------------------------------------------------------
# 9. Cross-Market Lead-Lag
# ---------------------------------------------------------------------------
class CrossMarketLeadLag(BenchmarkStrategy):
    """Follow NQ (Nasdaq futures) direction with a lag.

    NQ leads BTC during US trading hours by 2-5 minutes on average.
    This simple strategy just buys BTC if NQ went up in the last N bars.

    At 5-min, the lag is 1-2 bars. Outside US hours, the signal is weak.
    In risk-off regimes, correlation increases (both dump together).

    Beating this proves the ML system captures cross-market dynamics
    beyond simple directional following — likely through nonlinear
    conditional relationships or multi-asset feature interactions.
    """

    name = "cross_market_lead_lag"

    def __init__(self, lag_bars: int = 1) -> None:
        if lag_bars < 1:
            raise ValueError("lag_bars must be >= 1")
        self.lag_bars = lag_bars

    def generate_positions(self, df: pd.DataFrame) -> pd.Series:
        self.validate_columns(df, ["nq_close"])
        nq_returns = df["nq_close"].pct_change()
        # Use lagged NQ return as signal for BTC position
        lagged_signal = nq_returns.shift(self.lag_bars)
        positions = np.sign(lagged_signal)
        return pd.Series(positions, index=df.index, name="position").fillna(0.0)


# ---------------------------------------------------------------------------
# 10. Naive Ensemble
# ---------------------------------------------------------------------------
class NaiveEnsemble(BenchmarkStrategy):
    """Majority vote of multiple benchmark strategies.

    This tests whether naive combination beats any single benchmark.
    If a simple majority vote of 3 strategies beats each individually,
    that proves diversification works — but the ML system should still
    beat the ensemble through learned optimal weighting and nonlinear
    combination.

    Beating this is the highest bar among benchmarks — it proves the ML
    system's combination method is superior to simple voting.
    """

    name = "naive_ensemble"

    def __init__(self, strategies: list[BenchmarkStrategy] | None = None) -> None:
        self.strategies = strategies or []

    def generate_positions(self, df: pd.DataFrame) -> pd.Series:
        if not self.strategies:
            raise ValueError("NaiveEnsemble requires at least one strategy")

        all_positions = pd.DataFrame(index=df.index)
        for i, strategy in enumerate(self.strategies):
            try:
                all_positions[f"s_{i}"] = strategy.generate_positions(df)
            except (ValueError, KeyError) as e:
                logger.warning("Strategy %s failed in ensemble: %s", strategy.name, str(e))
                all_positions[f"s_{i}"] = 0.0

        # Majority vote: sign of sum
        vote = all_positions.sum(axis=1)
        positions = np.sign(vote)
        return pd.Series(positions, index=df.index, name="position").fillna(0.0)


# ---------------------------------------------------------------------------
# 11. Oracle (Perfect Foresight)
# ---------------------------------------------------------------------------
class OracleStrategy(BenchmarkStrategy):
    """Perfect foresight: always know the next bar's direction.

    This is the theoretical CEILING. No real strategy can beat this.
    It shows the maximum possible Sharpe after realistic costs.
    If the oracle Sharpe is 5.0 after costs, any strategy claiming
    Sharpe > 5.0 has a bug.

    The gap between oracle and your best strategy shows how much
    edge remains to be captured. If oracle Sharpe = 8 and your
    strategy Sharpe = 2, there's still 75% of theoretical edge
    uncaptured.
    """

    name = "oracle"

    def generate_positions(self, df: pd.DataFrame) -> pd.Series:
        self.validate_columns(df, ["close"])
        # Position at bar t earns return at bar t (= close[t]/close[t-1] - 1).
        # Oracle knows current bar's return, so position[t] = sign(return[t]).
        # This is equivalent to knowing the close before the bar ends.
        current_return = df["close"].pct_change()
        positions = np.sign(current_return)
        positions = pd.Series(positions, index=df.index, name="position").fillna(0.0)
        return positions


# ---------------------------------------------------------------------------
# 12. Funding Rate Carry
# ---------------------------------------------------------------------------
class FundingRateCarry(BenchmarkStrategy):
    """Hold the side that COLLECTS funding payments.

    When funding rate is positive, shorts receive from longs → go short.
    When funding rate is negative, longs receive from shorts → go long.

    This is a surprisingly strong baseline (Sharpe 1.0-2.0 historically)
    because it harvests the persistent bias of crypto futures markets
    where longs consistently overpay. Beating this proves the ML system
    adds alpha beyond simple carry.

    Uses smoothed funding to avoid excessive whipsawing.
    """

    name = "funding_rate_carry"

    def __init__(
        self,
        smoothing_periods: int = 3,
        entry_threshold: float = 0.00005,
    ) -> None:
        self.smoothing_periods = smoothing_periods
        self.entry_threshold = entry_threshold

    def generate_positions(self, df: pd.DataFrame) -> pd.Series:
        self.validate_columns(df, ["funding_rate"])
        fr = df["funding_rate"]

        # Smooth funding rate to reduce noise
        if self.smoothing_periods > 1:
            smoothed = fr.rolling(self.smoothing_periods, min_periods=1).mean()
        else:
            smoothed = fr

        positions = pd.Series(0.0, index=df.index, name="position")
        # Short when funding positive (collect from longs)
        positions[smoothed > self.entry_threshold] = -1.0
        # Long when funding negative (collect from shorts)
        positions[smoothed < -self.entry_threshold] = 1.0
        # Forward-fill: hold until signal changes
        positions = positions.replace(0.0, np.nan).ffill().fillna(0.0)
        return positions


# ---------------------------------------------------------------------------
# Strategy registry for convenience
# ---------------------------------------------------------------------------
STRATEGY_REGISTRY: dict[str, type[BenchmarkStrategy]] = {
    "buy_and_hold": BuyAndHold,
    "simple_momentum": SimpleMomentum,
    "mean_reversion_rsi": MeanReversionRSI,
    "ma_crossover": MovingAverageCrossover,
    "random_entry": RandomEntry,
    "volatility_breakout": VolatilityBreakout,
    "funding_rate": FundingRateStrategy,
    "order_book_imbalance": OrderBookImbalance,
    "cross_market_lead_lag": CrossMarketLeadLag,
    "naive_ensemble": NaiveEnsemble,
    "oracle": OracleStrategy,
    "funding_rate_carry": FundingRateCarry,
}


def get_default_benchmark_suite(
    momentum_lookbacks: list[int] | None = None,
    ma_combos: list[tuple[int, int]] | None = None,
) -> dict[str, BenchmarkStrategy]:
    """Return a dict of named strategies covering the full benchmark suite.

    Includes parameter sweeps for momentum and MA crossover.
    """
    if momentum_lookbacks is None:
        momentum_lookbacks = [1, 3, 5, 10, 20]
    if ma_combos is None:
        ma_combos = [(5, 20), (10, 50), (20, 100)]

    suite: dict[str, BenchmarkStrategy] = {}

    # Buy and hold variants
    for lev in [1.0, 2.0, 5.0]:
        suite[f"buy_hold_{lev}x"] = BuyAndHold(leverage=lev)

    # Momentum sweep
    for n in momentum_lookbacks:
        suite[f"momentum_N{n}"] = SimpleMomentum(lookback=n)

    # RSI mean reversion
    suite["rsi_30_70"] = MeanReversionRSI(period=14, oversold=30, overbought=70)
    suite["rsi_20_80"] = MeanReversionRSI(period=14, oversold=20, overbought=80)

    # MA crossover sweep
    for fast, slow in ma_combos:
        suite[f"ema_{fast}_{slow}"] = MovingAverageCrossover(fast, slow, "ema")
        suite[f"sma_{fast}_{slow}"] = MovingAverageCrossover(fast, slow, "sma")

    # Random entry (multiple hold periods)
    for hold in [1, 2, 3, 6]:
        suite[f"random_hold{hold}"] = RandomEntry(hold_bars=hold, seed=42)

    # Volatility breakout
    suite["vol_breakout_2atr_3bar"] = VolatilityBreakout(atr_multiplier=2.0, hold_bars=3)
    suite["vol_breakout_1.5atr_5bar"] = VolatilityBreakout(atr_multiplier=1.5, hold_bars=5)

    # Funding rate carry
    suite["funding_carry"] = FundingRateCarry()
    suite["funding_carry_no_smooth"] = FundingRateCarry(smoothing_periods=1)

    # Oracle
    suite["oracle"] = OracleStrategy()

    return suite
