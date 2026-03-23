"""Volume feature computers: delta, VWAP deviation, rate of change.

Volume features capture buying/selling pressure and liquidity dynamics.
All computations use only data at times <= current index (no look-ahead).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ep2_crypto.features.base import FeatureComputer

if TYPE_CHECKING:
    from numpy.typing import NDArray


class VolumeDeltaComputer(FeatureComputer):
    """Volume delta: buy_volume - sell_volume.

    Computed at 1-bar and 5-bar windows using trade_sizes and trade_sides.
    Normalized by total volume to produce a ratio in [-1, 1].
    """

    @property
    def name(self) -> str:
        return "volume_delta"

    @property
    def warmup_bars(self) -> int:
        return 5

    def compute(
        self,
        idx: int,
        timestamps: NDArray[np.int64],
        opens: NDArray[np.float64],
        highs: NDArray[np.float64],
        lows: NDArray[np.float64],
        closes: NDArray[np.float64],
        volumes: NDArray[np.float64],
        *,
        trade_sizes: NDArray[np.float64] | None = None,
        trade_sides: NDArray[np.float64] | None = None,
        **kwargs: NDArray[np.float64] | None,
    ) -> dict[str, float]:
        nan_result = {
            "vol_delta_1bar": float("nan"),
            "vol_delta_5bar": float("nan"),
            "vol_delta_1bar_raw": float("nan"),
            "vol_delta_5bar_raw": float("nan"),
        }

        if idx < self.warmup_bars - 1:
            return nan_result

        if trade_sizes is None or trade_sides is None:
            return nan_result

        result: dict[str, float] = {}

        # 1-bar delta
        signed = float(trade_sizes[idx] * trade_sides[idx])
        total = float(abs(trade_sizes[idx]))
        result["vol_delta_1bar"] = signed / total if total > 0 else 0.0
        result["vol_delta_1bar_raw"] = signed

        # 5-bar delta
        start = idx - 4
        signed_sum = float(np.sum(trade_sizes[start:idx + 1] * trade_sides[start:idx + 1]))
        total_sum = float(np.sum(np.abs(trade_sizes[start:idx + 1])))
        result["vol_delta_5bar"] = signed_sum / total_sum if total_sum > 0 else 0.0
        result["vol_delta_5bar_raw"] = signed_sum

        return result

    def output_names(self) -> list[str]:
        return ["vol_delta_1bar", "vol_delta_5bar", "vol_delta_1bar_raw", "vol_delta_5bar_raw"]


class VWAPComputer(FeatureComputer):
    """VWAP deviation: (close - vwap) / vwap over rolling window.

    VWAP = sum(price * volume) / sum(volume) over window.
    Uses typical price = (high + low + close) / 3 for VWAP calculation.

    Default window is 12 bars (1 hour for 5-min bars).
    """

    def __init__(self, window: int = 12) -> None:
        self._window = window

    @property
    def name(self) -> str:
        return "vwap"

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
        nan_result = {
            "vwap": float("nan"),
            "vwap_deviation": float("nan"),
        }

        if idx < self.warmup_bars - 1:
            return nan_result

        start = idx - self._window + 1
        typical_price = (highs[start:idx + 1] + lows[start:idx + 1] + closes[start:idx + 1]) / 3.0
        vol_slice = volumes[start:idx + 1]
        total_vol = float(np.sum(vol_slice))

        if total_vol <= 0:
            return nan_result

        vwap = float(np.sum(typical_price * vol_slice)) / total_vol
        deviation = (closes[idx] - vwap) / vwap if vwap > 0 else 0.0

        return {
            "vwap": vwap,
            "vwap_deviation": float(deviation),
        }

    def output_names(self) -> list[str]:
        return ["vwap", "vwap_deviation"]


class VolumeROCComputer(FeatureComputer):
    """Volume Rate of Change at multiple lookback periods.

    ROC = (current_volume - past_volume) / past_volume

    Computed at 1, 3, and 6 bar lookbacks.
    """

    @property
    def name(self) -> str:
        return "volume_roc"

    @property
    def warmup_bars(self) -> int:
        return 7  # Need 6 bars of lookback + current

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
            "vol_roc_1": float("nan"),
            "vol_roc_3": float("nan"),
            "vol_roc_6": float("nan"),
        }

        if idx < self.warmup_bars - 1:
            return nan_result

        result: dict[str, float] = {}
        current_vol = float(volumes[idx])

        for period in [1, 3, 6]:
            past_vol = float(volumes[idx - period])
            if past_vol > 0:
                result[f"vol_roc_{period}"] = (current_vol - past_vol) / past_vol
            else:
                result[f"vol_roc_{period}"] = 0.0

        return result

    def output_names(self) -> list[str]:
        return ["vol_roc_1", "vol_roc_3", "vol_roc_6"]
