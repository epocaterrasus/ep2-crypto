"""Volatility feature computers: realized, Parkinson, EWMA, vol-of-vol.

Multiple volatility estimators for regime detection and signal conditioning.
All computations use only data at times <= current index (no look-ahead).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from ep2_crypto.features.base import FeatureComputer

if TYPE_CHECKING:
    from numpy.typing import NDArray


class RealizedVolComputer(FeatureComputer):
    """Realized volatility: rolling standard deviation of log returns.

    Computed at 6-bar (30-min) and 12-bar (1-hour) windows for 5-min bars.
    NOT annualized — kept as raw per-bar vol for tree model consumption.
    """

    def __init__(self, short_window: int = 6, long_window: int = 12) -> None:
        self._short = short_window
        self._long = long_window

    @property
    def name(self) -> str:
        return "realized_vol"

    @property
    def warmup_bars(self) -> int:
        return self._long + 1  # Need +1 for log returns

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
            "realized_vol_short": float("nan"),
            "realized_vol_long": float("nan"),
        }

        if idx < self.warmup_bars - 1:
            return nan_result

        result: dict[str, float] = {}

        for window, label in [(self._short, "short"), (self._long, "long")]:
            start = idx - window
            log_returns = np.log(closes[start + 1:idx + 1] / closes[start:idx])
            result[f"realized_vol_{label}"] = float(np.std(log_returns, ddof=1))

        return result

    def output_names(self) -> list[str]:
        return ["realized_vol_short", "realized_vol_long"]


class ParkinsonVolComputer(FeatureComputer):
    """Parkinson volatility estimator using high-low range.

    More efficient than close-to-close: uses intrabar range information.
    sigma_P = sqrt(1/(4*n*ln2) * sum(ln(H/L)^2))

    Computed over 6-bar and 12-bar windows.
    """

    def __init__(self, short_window: int = 6, long_window: int = 12) -> None:
        self._short = short_window
        self._long = long_window

    @property
    def name(self) -> str:
        return "parkinson_vol"

    @property
    def warmup_bars(self) -> int:
        return self._long

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
            "parkinson_vol_short": float("nan"),
            "parkinson_vol_long": float("nan"),
        }

        if idx < self.warmup_bars - 1:
            return nan_result

        result: dict[str, float] = {}

        for window, label in [(self._short, "short"), (self._long, "long")]:
            start = idx - window + 1
            h = highs[start:idx + 1]
            lo = lows[start:idx + 1]

            # Guard against zero or negative lows
            valid = lo > 0
            if not np.all(valid):
                result[f"parkinson_vol_{label}"] = float("nan")
                continue

            log_hl = np.log(h / lo)
            n = len(log_hl)
            sigma_sq = float(np.sum(log_hl ** 2)) / (4.0 * n * math.log(2))
            result[f"parkinson_vol_{label}"] = math.sqrt(max(sigma_sq, 0.0))

        return result

    def output_names(self) -> list[str]:
        return ["parkinson_vol_short", "parkinson_vol_long"]


class EWMAVolComputer(FeatureComputer):
    """EWMA volatility with configurable decay factor.

    sigma_t^2 = lambda * sigma_{t-1}^2 + (1-lambda) * r_t^2

    Default lambda = 0.94 (RiskMetrics convention).
    Requires iterative computation from the start.
    """

    def __init__(self, decay: float = 0.94) -> None:
        self._decay = decay

    @property
    def name(self) -> str:
        return "ewma_vol"

    @property
    def warmup_bars(self) -> int:
        return 20  # Need enough bars for EWMA to stabilize

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
            return {"ewma_vol": float("nan")}

        # Compute log returns from start to idx
        log_returns = np.log(closes[1:idx + 1] / closes[:idx])

        # Initialize variance with first return squared
        var = float(log_returns[0] ** 2)

        # EWMA iteration
        lam = self._decay
        for i in range(1, len(log_returns)):
            var = lam * var + (1.0 - lam) * float(log_returns[i] ** 2)

        return {"ewma_vol": math.sqrt(max(var, 0.0))}

    def output_names(self) -> list[str]:
        return ["ewma_vol"]


class VolOfVolComputer(FeatureComputer):
    """Volatility of volatility: rolling std of realized volatility.

    Measures second-order uncertainty. High vol-of-vol suggests
    unstable market conditions.

    Uses rolling realized vol at 6-bar window, then std over 12 bars.
    """

    def __init__(self, inner_window: int = 6, outer_window: int = 12) -> None:
        self._inner = inner_window
        self._outer = outer_window

    @property
    def name(self) -> str:
        return "vol_of_vol"

    @property
    def warmup_bars(self) -> int:
        return self._inner + self._outer + 1

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
            return {"vol_of_vol": float("nan")}

        # Compute realized vol at each point in outer window
        vols = np.empty(self._outer, dtype=np.float64)
        for i in range(self._outer):
            bar_idx = idx - self._outer + 1 + i
            start = bar_idx - self._inner
            log_ret = np.log(closes[start + 1:bar_idx + 1] / closes[start:bar_idx])
            vols[i] = float(np.std(log_ret, ddof=1))

        return {"vol_of_vol": float(np.std(vols, ddof=1))}

    def output_names(self) -> list[str]:
        return ["vol_of_vol"]
