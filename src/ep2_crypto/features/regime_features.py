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
        w1 = self._window
        w2 = self._window * 2
        nan_result = {
            f"er_{w1}": float("nan"),
            f"er_{w2}": float("nan"),
        }

        if idx < self.warmup_bars - 1:
            return nan_result

        result: dict[str, float] = {}

        for window in [w1, w2]:
            key = f"er_{window}"
            if idx < window:
                result[key] = float("nan")
                continue

            start = idx - window
            net_move = abs(closes[idx] - closes[start])
            individual_moves = float(np.sum(np.abs(np.diff(closes[start : idx + 1]))))

            if individual_moves > 0:
                result[key] = net_move / individual_moves
            else:
                result[key] = 0.0

        return result

    def output_names(self) -> list[str]:
        return [f"er_{self._window}", f"er_{self._window * 2}"]


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
        # State for O(1) incremental updates (reset on non-sequential calls)
        self._sigma2: float | None = None
        self._last_idx: int = -1

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

        if self._sigma2 is not None and self._last_idx == idx - 1:
            # Incremental O(1) update: one new return added since last call.
            # The GARCH loop uses r_{i-1} at step i, so the new step added
            # when idx increases by 1 uses r = log(closes[idx]) - log(closes[idx-1]).
            r = math.log(closes[idx]) - math.log(closes[idx - 1])
            sigma2 = self._omega + self._alpha * r * r + self._beta * self._sigma2
        else:
            # Full recomputation: initialization or non-sequential call.
            log_prices = np.log(closes[: idx + 1])
            returns = np.diff(log_prices)

            if len(returns) < 2:
                return nan_result

            sigma2 = float(np.var(returns[: self.warmup_bars]))
            if sigma2 <= 0:
                sigma2 = 1e-8

            # Iterate range(1, len(returns)+1) so the last step uses returns[idx-1],
            # matching the incremental path which also uses r = log(closes[idx]/closes[idx-1]).
            # Previously range(1, len(returns)) stopped at returns[idx-2], causing
            # cold-start vs incremental divergence of ~1-2% in garch_vol.
            for i in range(1, len(returns) + 1):
                r2 = returns[i - 1] ** 2
                sigma2 = self._omega + self._alpha * r2 + self._beta * sigma2

        self._sigma2 = sigma2
        self._last_idx = idx

        garch_vol = math.sqrt(max(sigma2, 0.0))

        # Ratio: current GARCH vol vs recent realized vol (O(1) window slice)
        start = max(0, idx - 20)
        recent_log = np.log(closes[start : idx + 1])
        recent_returns = np.diff(recent_log)
        realized_vol = float(np.std(recent_returns)) if len(recent_returns) > 1 else 1e-15
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
        # State for O(1) incremental updates: ring buffer of rolling vols
        self._vol_series: np.ndarray | None = None
        self._prev_prob: float = 0.5
        self._last_idx: int = -1

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

        k = 5.0

        if self._vol_series is not None and self._last_idx == idx - 1:
            # Incremental O(1) update: compute one new rolling vol, shift buffer.
            # New vol = std of the most recent vol_window returns.
            # log return requires two close prices, so returns end at closes[idx],
            # which means the last return index in the diff array is idx-1.
            start = idx - self._vol_window
            new_vol = float(np.std(np.diff(np.log(closes[start : idx + 1]))))
            # Shift left and append new vol (ring buffer)
            self._vol_series[:-1] = self._vol_series[1:]
            self._vol_series[-1] = new_vol
        else:
            # Full recomputation: initialization or non-sequential call
            returns = np.diff(np.log(closes[: idx + 1]))
            if len(returns) < self._vol_window + self._smooth_window:
                return nan_result
            vol_series = np.zeros(self._smooth_window)
            for i in range(self._smooth_window):
                end = len(returns) - (self._smooth_window - 1 - i)
                start = end - self._vol_window
                vol_series[i] = float(np.std(returns[start:end]))
            self._vol_series = vol_series
            # For prev_prob on full recompute, approximate with one-bar-back series
            prev_vol_series = np.zeros(self._smooth_window)
            for i in range(self._smooth_window):
                end = len(returns) - 1 - (self._smooth_window - 1 - i)
                start_p = end - self._vol_window
                if start_p >= 0 and end > start_p:
                    prev_vol_series[i] = float(np.std(returns[start_p:end]))

        vol_series = self._vol_series
        current_vol = float(vol_series[-1])
        median_vol = float(np.median(vol_series))

        if median_vol > 1e-15:
            ratio = current_vol / median_vol
            prob = 1.0 / (1.0 + math.exp(-k * (ratio - 1.0)))
        else:
            prob = 0.5

        # prev_prob: stored from previous bar (incremental) or computed above (full)
        if self._last_idx == idx - 1:
            prev_prob = self._prev_prob
        else:
            prev_median = float(np.median(prev_vol_series))
            if prev_median > 1e-15:
                prev_ratio = float(prev_vol_series[-1]) / prev_median
                prev_prob = 1.0 / (1.0 + math.exp(-k * (prev_ratio - 1.0)))
            else:
                prev_prob = 0.5

        regime_change = abs(prob - prev_prob) if idx >= self.warmup_bars else 0.0

        self._prev_prob = prob
        self._last_idx = idx

        return {
            "hmm_high_vol_prob": prob,
            "hmm_regime_change": regime_change,
        }

    def output_names(self) -> list[str]:
        return ["hmm_high_vol_prob", "hmm_regime_change"]
