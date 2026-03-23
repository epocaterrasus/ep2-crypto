"""Momentum feature computers: ROC, RSI, linear regression slope, quantile rank.

Momentum and mean-reversion indicators for directional prediction.
All computations use only data at times <= current index (no look-ahead).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ep2_crypto.features.base import FeatureComputer

if TYPE_CHECKING:
    from numpy.typing import NDArray


class ROCComputer(FeatureComputer):
    """Rate of Change at multiple lookback periods.

    ROC_k = (close[t] - close[t-k]) / close[t-k]

    Computed at 1, 3, 6, and 12 bar lookbacks.
    """

    @property
    def name(self) -> str:
        return "roc"

    @property
    def warmup_bars(self) -> int:
        return 13  # Need 12 bars lookback + current

    def compute(
        self,
        idx: int,
        timestamps: NDArray[np.int64],
        opens: NDArray[np.float64],
        highs: NDArray[np.float64],
        lows: NDArray[np.float64],
        closes: NDArray[np.float64],
        volumes: NDArray[np.float64],
        **kwargs: NDArray[np.float64] | None,
    ) -> dict[str, float]:
        nan_result = {
            "roc_1": float("nan"),
            "roc_3": float("nan"),
            "roc_6": float("nan"),
            "roc_12": float("nan"),
        }

        if idx < self.warmup_bars - 1:
            return nan_result

        current = float(closes[idx])
        result: dict[str, float] = {}

        for period in [1, 3, 6, 12]:
            past = float(closes[idx - period])
            result[f"roc_{period}"] = (current - past) / past if past > 0 else 0.0

        return result

    def output_names(self) -> list[str]:
        return ["roc_1", "roc_3", "roc_6", "roc_12"]


class RSIComputer(FeatureComputer):
    """Relative Strength Index (Wilder's smoothing).

    RSI = 100 - 100 / (1 + RS)
    RS = avg_gain / avg_loss over window

    Uses exponential moving average (Wilder's smoothing) for stability.
    Bounded [0, 100].
    """

    def __init__(self, window: int = 14) -> None:
        self._window = window

    @property
    def name(self) -> str:
        return "rsi"

    @property
    def warmup_bars(self) -> int:
        return self._window + 1

    def compute(
        self,
        idx: int,
        timestamps: NDArray[np.int64],
        opens: NDArray[np.float64],
        highs: NDArray[np.float64],
        lows: NDArray[np.float64],
        closes: NDArray[np.float64],
        volumes: NDArray[np.float64],
        **kwargs: NDArray[np.float64] | None,
    ) -> dict[str, float]:
        if idx < self.warmup_bars - 1:
            return {"rsi": float("nan")}

        # Compute price changes from start
        changes = np.diff(closes[:idx + 1])

        # Wilder's smoothing: initialize with SMA, then EMA
        gains = np.where(changes > 0, changes, 0.0)
        losses = np.where(changes < 0, -changes, 0.0)

        # Initial average (SMA of first window)
        avg_gain = float(np.mean(gains[:self._window]))
        avg_loss = float(np.mean(losses[:self._window]))

        # Wilder's EMA for subsequent values
        alpha = 1.0 / self._window
        for i in range(self._window, len(gains)):
            avg_gain = avg_gain * (1 - alpha) + gains[i] * alpha
            avg_loss = avg_loss * (1 - alpha) + losses[i] * alpha

        if avg_loss == 0:
            rsi = 100.0 if avg_gain > 0 else 50.0
        else:
            rs = avg_gain / avg_loss
            rsi = 100.0 - 100.0 / (1.0 + rs)

        return {"rsi": rsi}

    def output_names(self) -> list[str]:
        return ["rsi"]


class LinRegSlopeComputer(FeatureComputer):
    """Linear regression slope of close prices over rolling window.

    Measures the trend strength and direction. Normalized by price
    to make it comparable across price levels.

    slope = Cov(t, price) / Var(t) / mean_price
    """

    def __init__(self, window: int = 20) -> None:
        self._window = window

    @property
    def name(self) -> str:
        return "linreg_slope"

    @property
    def warmup_bars(self) -> int:
        return self._window

    def compute(
        self,
        idx: int,
        timestamps: NDArray[np.int64],
        opens: NDArray[np.float64],
        highs: NDArray[np.float64],
        lows: NDArray[np.float64],
        closes: NDArray[np.float64],
        volumes: NDArray[np.float64],
        **kwargs: NDArray[np.float64] | None,
    ) -> dict[str, float]:
        if idx < self.warmup_bars - 1:
            return {"linreg_slope": float("nan")}

        start = idx - self._window + 1
        y = closes[start:idx + 1]
        x = np.arange(self._window, dtype=np.float64)

        # Linear regression via covariance
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        cov_xy = float(np.sum((x - x_mean) * (y - y_mean)))
        var_x = float(np.sum((x - x_mean) ** 2))

        if var_x == 0:
            return {"linreg_slope": 0.0}

        slope = cov_xy / var_x
        # Normalize by mean price
        mean_price = float(y_mean)
        normalized_slope = slope / mean_price if mean_price > 0 else 0.0

        return {"linreg_slope": normalized_slope}

    def output_names(self) -> list[str]:
        return ["linreg_slope"]


class QuantileRankComputer(FeatureComputer):
    """Price quantile rank over rolling window.

    Rank of current close within rolling window, normalized to [0, 1].
    0 = lowest price in window, 1 = highest.

    Useful for mean-reversion signals.
    """

    def __init__(self, window: int = 60) -> None:
        self._window = window

    @property
    def name(self) -> str:
        return "quantile_rank"

    @property
    def warmup_bars(self) -> int:
        return self._window

    def compute(
        self,
        idx: int,
        timestamps: NDArray[np.int64],
        opens: NDArray[np.float64],
        highs: NDArray[np.float64],
        lows: NDArray[np.float64],
        closes: NDArray[np.float64],
        volumes: NDArray[np.float64],
        **kwargs: NDArray[np.float64] | None,
    ) -> dict[str, float]:
        if idx < self.warmup_bars - 1:
            return {"quantile_rank": float("nan")}

        start = idx - self._window + 1
        window_data = closes[start:idx + 1]
        current = float(closes[idx])

        # Count values less than current
        count_below = int(np.sum(window_data < current))
        # Rank as fraction (0 to 1)
        rank = count_below / (self._window - 1) if self._window > 1 else 0.5

        return {"quantile_rank": rank}

    def output_names(self) -> list[str]:
        return ["quantile_rank"]
