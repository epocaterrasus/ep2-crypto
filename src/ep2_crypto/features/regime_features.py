"""Regime feature computers: Efficiency Ratio, GARCH vol, HMM probs as model inputs.

These features provide regime context to the prediction model. They are not
the regime detector itself (Sprint 6), but lightweight features that capture
market state for use as model inputs.
All computations use only data at times <= current index (no look-ahead).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from ep2_crypto.features.base import FeatureComputer

if TYPE_CHECKING:
    from numpy.typing import NDArray


class ERFeatureComputer(FeatureComputer):
    """Kaufman Efficiency Ratio as a feature input.

    ER = |net_move| / sum(|individual_moves|) over a rolling window.
    ER near 1.0 = trending market (directional).
    ER near 0.0 = choppy/mean-reverting market.

    This is a fast O(1)-per-bar indicator that captures trend strength.
    """

    def __init__(self, window: int = 10) -> None:
        self._window = window

    @property
    def name(self) -> str:
        return "er_feature"

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
        nan_result = {
            "er_10": float("nan"),
            "er_20": float("nan"),
        }

        if idx < self.warmup_bars - 1:
            return nan_result

        result: dict[str, float] = {}

        for window, label in [(self._window, "10"), (self._window * 2, "20")]:
            if idx < window:
                result[f"er_{label}"] = float("nan")
                continue

            start = idx - window
            net_move = abs(closes[idx] - closes[start])
            individual_moves = float(np.sum(np.abs(np.diff(closes[start:idx + 1]))))

            if individual_moves > 0:
                result[f"er_{label}"] = net_move / individual_moves
            else:
                result[f"er_{label}"] = 0.0

        return result

    def output_names(self) -> list[str]:
        return ["er_10", "er_20"]


class GARCHFeatureComputer(FeatureComputer):
    """Simplified GARCH(1,1) conditional volatility as feature input.

    Uses a recursive EWMA-style update for conditional variance:
    sigma2_t = omega + alpha * r_{t-1}^2 + beta * sigma2_{t-1}

    With standard parameters: alpha=0.05, beta=0.94, omega=1e-6.
    This is not a full GARCH fit — it's a feature approximation that
    captures volatility clustering for the model.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        beta: float = 0.94,
        omega: float = 1e-6,
    ) -> None:
        self._alpha = alpha
        self._beta = beta
        self._omega = omega

    @property
    def name(self) -> str:
        return "garch_feature"

    @property
    def warmup_bars(self) -> int:
        return 20  # Need enough bars for variance to stabilize

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
            "garch_vol": float("nan"),
            "garch_vol_ratio": float("nan"),
        }

        if idx < self.warmup_bars - 1:
            return nan_result

        # Compute log returns from start
        log_prices = np.log(closes[:idx + 1])
        returns = np.diff(log_prices)

        if len(returns) < 2:
            return nan_result

        # Initialize conditional variance with sample variance
        sigma2 = float(np.var(returns[:self.warmup_bars]))
        if sigma2 <= 0:
            sigma2 = 1e-8

        # Recursive GARCH update
        for i in range(1, len(returns)):
            r2 = returns[i - 1] ** 2
            sigma2 = self._omega + self._alpha * r2 + self._beta * sigma2

        garch_vol = math.sqrt(max(sigma2, 0.0))

        # Ratio: current GARCH vol vs recent realized vol
        recent_returns = returns[-min(20, len(returns)):]
        realized_vol = float(np.std(recent_returns))
        vol_ratio = garch_vol / realized_vol if realized_vol > 1e-15 else 1.0

        return {
            "garch_vol": garch_vol,
            "garch_vol_ratio": vol_ratio,
        }

    def output_names(self) -> list[str]:
        return ["garch_vol", "garch_vol_ratio"]


class HMMFeatureComputer(FeatureComputer):
    """Simplified HMM regime probability proxy as feature input.

    Instead of fitting a full HMM (expensive, Sprint 6), this uses
    a simple 2-state volatility regime proxy based on whether current
    volatility is above/below its rolling median.

    Outputs a smoothed probability of being in the "high volatility" regime.
    """

    def __init__(self, vol_window: int = 20, smooth_window: int = 10) -> None:
        self._vol_window = vol_window
        self._smooth_window = smooth_window

    @property
    def name(self) -> str:
        return "hmm_feature"

    @property
    def warmup_bars(self) -> int:
        return self._vol_window + self._smooth_window + 1

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
            "hmm_high_vol_prob": float("nan"),
            "hmm_regime_change": float("nan"),
        }

        if idx < self.warmup_bars - 1:
            return nan_result

        # Compute rolling realized vol for the smoothing window
        log_prices = np.log(closes[:idx + 1])
        returns = np.diff(log_prices)

        if len(returns) < self._vol_window + self._smooth_window:
            return nan_result

        # Get rolling volatilities
        vol_series = np.zeros(self._smooth_window)
        for i in range(self._smooth_window):
            end = len(returns) - (self._smooth_window - 1 - i)
            start = end - self._vol_window
            vol_series[i] = float(np.std(returns[start:end]))

        current_vol = vol_series[-1]
        median_vol = float(np.median(vol_series))

        # Smooth probability: sigmoid-like mapping of current vs median
        if median_vol > 1e-15:
            ratio = current_vol / median_vol
            # Map ratio to probability: ratio > 1 -> high vol regime
            # Using logistic: p = 1 / (1 + exp(-k*(ratio - 1)))
            k = 5.0  # steepness
            prob = 1.0 / (1.0 + math.exp(-k * (ratio - 1.0)))
        else:
            prob = 0.5

        # Regime change: absolute difference in prob vs previous bar
        if idx >= self.warmup_bars:
            # Approximate previous prob by checking vol one bar earlier
            prev_vol_series = np.zeros(self._smooth_window)
            for i in range(self._smooth_window):
                end = len(returns) - 1 - (self._smooth_window - 1 - i)
                start = end - self._vol_window
                if start >= 0 and end > start:
                    prev_vol_series[i] = float(np.std(returns[start:end]))
            prev_vol = prev_vol_series[-1]
            prev_median = float(np.median(prev_vol_series))
            if prev_median > 1e-15:
                prev_ratio = prev_vol / prev_median
                prev_prob = 1.0 / (1.0 + math.exp(-k * (prev_ratio - 1.0)))
            else:
                prev_prob = 0.5
            regime_change = abs(prob - prev_prob)
        else:
            regime_change = 0.0

        return {
            "hmm_high_vol_prob": prob,
            "hmm_regime_change": regime_change,
        }

    def output_names(self) -> list[str]:
        return ["hmm_high_vol_prob", "hmm_regime_change"]
