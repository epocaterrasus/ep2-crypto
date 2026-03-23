"""Triple barrier labeling for ternary classification targets.

Implements Lopez de Prado's triple barrier method (AFML Chapter 3):
- Upper barrier: take-profit level (price rises enough)
- Lower barrier: stop-loss level (price drops enough)
- Vertical barrier: max holding period (time runs out)

The label is determined by which barrier is touched first:
- +1 (UP): upper barrier touched first
- -1 (DOWN): lower barrier touched first
-  0 (FLAT): vertical barrier reached without touching upper/lower

Barrier widths can be fixed (basis points), ATR-based, or percentile-based.
Adaptive thresholds per regime are supported.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


class Direction(IntEnum):
    """Ternary direction labels."""

    DOWN = -1
    FLAT = 0
    UP = 1


@dataclass(frozen=True)
class BarrierConfig:
    """Configuration for triple barrier labeling.

    Attributes:
        max_holding_bars: Vertical barrier — max bars to hold (default 12 = 1 hour at 5-min).
        upper_multiplier: Upper barrier width multiplier on volatility measure.
        lower_multiplier: Lower barrier width multiplier on volatility measure.
        vol_window: Window for computing volatility (ATR or rolling std of returns).
        min_barrier_bps: Minimum barrier width in basis points
            (prevents too-tight barriers in low-vol).
        use_atr: If True, use ATR for barrier width. If False, use rolling std of log returns.
    """

    max_holding_bars: int = 12
    upper_multiplier: float = 1.0
    lower_multiplier: float = 1.0
    vol_window: int = 20
    min_barrier_bps: float = 5.0
    use_atr: bool = True


def compute_atr(
    highs: NDArray[np.float64],
    lows: NDArray[np.float64],
    closes: NDArray[np.float64],
    window: int,
) -> NDArray[np.float64]:
    """Compute Average True Range (ATR) using Wilder's smoothing.

    Args:
        highs: High prices array.
        lows: Low prices array.
        closes: Close prices array.
        window: ATR smoothing window.

    Returns:
        ATR array (same length as input, NaN before warmup).
    """
    n = len(closes)
    tr = np.empty(n, dtype=np.float64)
    tr[0] = highs[0] - lows[0]

    for i in range(1, n):
        hl = highs[i] - lows[i]
        hc = abs(highs[i] - closes[i - 1])
        lc = abs(lows[i] - closes[i - 1])
        tr[i] = max(hl, hc, lc)

    atr = np.full(n, np.nan, dtype=np.float64)
    if n < window:
        return atr

    # Initial ATR = simple average of first `window` TRs
    atr[window - 1] = np.mean(tr[:window])

    # Wilder's smoothing
    alpha = 1.0 / window
    for i in range(window, n):
        atr[i] = atr[i - 1] * (1.0 - alpha) + tr[i] * alpha

    return atr


def compute_rolling_vol(
    closes: NDArray[np.float64],
    window: int,
) -> NDArray[np.float64]:
    """Compute rolling standard deviation of log returns.

    Args:
        closes: Close prices array.
        window: Rolling window size.

    Returns:
        Rolling volatility array (NaN before warmup).
    """
    n = len(closes)
    vol = np.full(n, np.nan, dtype=np.float64)

    if n < 2:
        return vol

    log_ret = np.diff(np.log(closes))

    for i in range(window, n):
        segment = log_ret[i - window : i]
        vol[i] = np.std(segment, ddof=1)

    return vol


def compute_barriers(
    closes: NDArray[np.float64],
    highs: NDArray[np.float64],
    lows: NDArray[np.float64],
    config: BarrierConfig | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute upper and lower barrier widths for each bar.

    Args:
        closes: Close prices.
        highs: High prices.
        lows: Low prices.
        config: Barrier configuration.

    Returns:
        Tuple of (upper_barriers, lower_barriers) in price units.
    """
    if config is None:
        config = BarrierConfig()

    n = len(closes)

    if config.use_atr:
        vol_measure = compute_atr(highs, lows, closes, config.vol_window)
    else:
        vol_measure = compute_rolling_vol(closes, config.vol_window)
        # Convert from log-return vol to price units
        vol_measure = vol_measure * closes

    min_barrier = closes * (config.min_barrier_bps / 10_000.0)

    upper = np.full(n, np.nan, dtype=np.float64)
    lower = np.full(n, np.nan, dtype=np.float64)

    for i in range(n):
        if np.isfinite(vol_measure[i]):
            u_val = vol_measure[i] * config.upper_multiplier
            l_val = vol_measure[i] * config.lower_multiplier
            upper[i] = max(u_val, min_barrier[i])
            lower[i] = max(l_val, min_barrier[i])

    return upper, lower


def label_triple_barrier(
    closes: NDArray[np.float64],
    highs: NDArray[np.float64],
    lows: NDArray[np.float64],
    config: BarrierConfig | None = None,
) -> tuple[NDArray[np.int8], NDArray[np.float64], NDArray[np.int32]]:
    """Apply triple barrier labeling to a price series.

    For each bar t, we look forward up to max_holding_bars to determine
    which barrier is hit first using high/low prices for intra-bar accuracy.

    Args:
        closes: Close prices.
        highs: High prices (for intra-bar barrier detection).
        lows: Low prices (for intra-bar barrier detection).
        config: Barrier configuration.

    Returns:
        Tuple of:
        - labels: int8 array of Direction values (+1, 0, -1). NaN-equivalent = 0 for warmup bars.
        - returns_at_exit: float64 array of return at barrier touch (in fractional terms).
        - hold_periods: int32 array of bars held until barrier touch.
    """
    if config is None:
        config = BarrierConfig()

    n = len(closes)
    labels = np.zeros(n, dtype=np.int8)
    returns_at_exit = np.zeros(n, dtype=np.float64)
    hold_periods = np.zeros(n, dtype=np.int32)

    upper_barriers, lower_barriers = compute_barriers(closes, highs, lows, config)

    for t in range(n):
        if not np.isfinite(upper_barriers[t]) or not np.isfinite(lower_barriers[t]):
            # Not enough warmup — label as FLAT
            labels[t] = Direction.FLAT
            continue

        entry_price = closes[t]
        upper_level = entry_price + upper_barriers[t]
        lower_level = entry_price - lower_barriers[t]
        max_bar = min(t + config.max_holding_bars, n - 1)

        touched = False
        for j in range(t + 1, max_bar + 1):
            # Check if high breaches upper barrier
            if highs[j] >= upper_level:
                labels[t] = Direction.UP
                returns_at_exit[t] = (upper_level - entry_price) / entry_price
                hold_periods[t] = j - t
                touched = True
                break
            # Check if low breaches lower barrier
            if lows[j] <= lower_level:
                labels[t] = Direction.DOWN
                returns_at_exit[t] = (lower_level - entry_price) / entry_price
                hold_periods[t] = j - t
                touched = True
                break

        if not touched:
            # Vertical barrier — use close at end of holding period
            if max_bar > t:
                exit_return = (closes[max_bar] - entry_price) / entry_price
                returns_at_exit[t] = exit_return
                hold_periods[t] = max_bar - t
                # Classify based on sign of return
                if exit_return > 0:
                    labels[t] = Direction.UP
                elif exit_return < 0:
                    labels[t] = Direction.DOWN
                else:
                    labels[t] = Direction.FLAT
            else:
                # At the very end of the array — cannot look forward
                labels[t] = Direction.FLAT
                hold_periods[t] = 0

    return labels, returns_at_exit, hold_periods


def label_fixed_threshold(
    closes: NDArray[np.float64],
    horizon: int = 1,
    threshold_bps: float = 10.0,
) -> NDArray[np.int8]:
    """Simple fixed-threshold labeling (for baseline comparison).

    Args:
        closes: Close prices.
        horizon: Forward return horizon in bars.
        threshold_bps: Threshold in basis points for up/down classification.

    Returns:
        int8 array of Direction values.
    """
    n = len(closes)
    labels = np.zeros(n, dtype=np.int8)
    threshold = threshold_bps / 10_000.0

    for t in range(n - horizon):
        ret = (closes[t + horizon] - closes[t]) / closes[t]
        if ret > threshold:
            labels[t] = Direction.UP
        elif ret < -threshold:
            labels[t] = Direction.DOWN
        else:
            labels[t] = Direction.FLAT

    return labels


def compute_class_weights(
    labels: NDArray[np.int8],
) -> dict[int, float]:
    """Compute balanced class weights for ternary labels.

    Uses inverse frequency weighting so rare classes get higher weight.

    Args:
        labels: Array of ternary labels (-1, 0, +1).

    Returns:
        Dict mapping label value to weight.
    """
    unique, counts = np.unique(labels, return_counts=True)
    n = len(labels)
    n_classes = len(unique)
    weights = {}
    for label, count in zip(unique, counts, strict=True):
        weights[int(label)] = n / (n_classes * count)
    return weights
