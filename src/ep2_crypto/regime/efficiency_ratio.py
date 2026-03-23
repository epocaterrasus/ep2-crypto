"""Kaufman Efficiency Ratio regime detector — fast layer.

ER = |net_move| / sum(|individual_moves|) over a rolling window.
- ER near 1.0 = perfectly trending (straight-line price movement)
- ER near 0.0 = choppy/range-bound (lots of movement, no net progress)

This is the fastest regime indicator: O(1) per bar with incremental updates.
It forms the "fast layer" of the hierarchical regime detection ensemble.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


class ERRegime(IntEnum):
    """Regime classification based on Efficiency Ratio."""

    CHOPPY = 0
    NEUTRAL = 1
    TRENDING = 2


@dataclass(frozen=True)
class ERResult:
    """Output from the Efficiency Ratio detector."""

    er_short: float  # ER at short window
    er_long: float  # ER at long window
    regime: ERRegime
    confidence: float  # 0-1, how strongly the ER supports the regime label


class EfficiencyRatioDetector:
    """Kaufman Efficiency Ratio with threshold-based regime classification.

    Computes ER at two timescales (short and long windows) and classifies
    the market regime as trending, choppy, or neutral based on configurable
    thresholds.

    Parameters
    ----------
    short_window : int
        Short ER lookback (default 20 bars = ~1.7 hours at 5-min).
    long_window : int
        Long ER lookback (default 100 bars = ~8.3 hours at 5-min).
    trending_threshold : float
        ER above this → trending regime (default 0.5).
    choppy_threshold : float
        ER below this → choppy regime (default 0.3).
    """

    def __init__(
        self,
        short_window: int = 20,
        long_window: int = 100,
        trending_threshold: float = 0.5,
        choppy_threshold: float = 0.3,
    ) -> None:
        if short_window < 2:
            msg = "short_window must be >= 2"
            raise ValueError(msg)
        if long_window <= short_window:
            msg = "long_window must be > short_window"
            raise ValueError(msg)
        if choppy_threshold >= trending_threshold:
            msg = "choppy_threshold must be < trending_threshold"
            raise ValueError(msg)

        self._short_window = short_window
        self._long_window = long_window
        self._trending_threshold = trending_threshold
        self._choppy_threshold = choppy_threshold

    @property
    def short_window(self) -> int:
        return self._short_window

    @property
    def long_window(self) -> int:
        return self._long_window

    @property
    def warmup_bars(self) -> int:
        """Minimum bars needed before producing valid output."""
        return self._long_window + 1

    @staticmethod
    def _compute_er(closes: NDArray[np.float64], idx: int, window: int) -> float:
        """Compute Efficiency Ratio at a single index for a given window.

        ER = |close[idx] - close[idx - window]| / sum(|diff(close[idx-window:idx+1])|)
        """
        if idx < window:
            return float("nan")

        start = idx - window
        net_move = abs(float(closes[idx] - closes[start]))
        # Sum of absolute bar-to-bar moves
        segment = closes[start : idx + 1]
        individual_moves = float(np.sum(np.abs(np.diff(segment))))

        if individual_moves < 1e-15:
            # No price movement at all — perfectly trending (or flat)
            return 1.0

        return net_move / individual_moves

    def update(
        self,
        idx: int,
        closes: NDArray[np.float64],
    ) -> ERResult:
        """Compute ER and regime classification for bar at idx.

        Parameters
        ----------
        idx : int
            Current bar index into the closes array.
        closes : NDArray[np.float64]
            Full close price array (only data up to idx is used).

        Returns
        -------
        ERResult with ER values, regime label, and confidence.
        """
        er_short = self._compute_er(closes, idx, self._short_window)
        er_long = self._compute_er(closes, idx, self._long_window)

        # Use short window ER for regime classification (more responsive)
        if np.isnan(er_short):
            return ERResult(
                er_short=er_short,
                er_long=er_long,
                regime=ERRegime.NEUTRAL,
                confidence=0.0,
            )

        # Classify regime based on short ER
        regime, confidence = self._classify(er_short)

        return ERResult(
            er_short=er_short,
            er_long=er_long,
            regime=regime,
            confidence=confidence,
        )

    def _classify(self, er: float) -> tuple[ERRegime, float]:
        """Classify regime from ER value and compute confidence.

        Confidence is how far the ER is from the neutral zone boundaries,
        normalized to [0, 1].
        """
        if er >= self._trending_threshold:
            # Distance from threshold to 1.0, normalized
            confidence = min(
                (er - self._trending_threshold) / (1.0 - self._trending_threshold),
                1.0,
            )
            return ERRegime.TRENDING, confidence

        if er <= self._choppy_threshold:
            # Distance from threshold to 0.0, normalized
            confidence = (
                min(
                    (self._choppy_threshold - er) / self._choppy_threshold,
                    1.0,
                )
                if self._choppy_threshold > 0
                else 1.0
            )
            return ERRegime.CHOPPY, confidence

        # Neutral zone — confidence based on distance from midpoint
        midpoint = (self._trending_threshold + self._choppy_threshold) / 2
        half_range = (self._trending_threshold - self._choppy_threshold) / 2
        confidence = 1.0 - abs(er - midpoint) / half_range if half_range > 0 else 0.5
        return ERRegime.NEUTRAL, confidence

    def compute_batch(
        self,
        closes: NDArray[np.float64],
    ) -> list[ERResult]:
        """Compute ER results for all bars in the series.

        Returns a list of ERResult, one per bar. Bars before warmup
        will have NaN ER values and NEUTRAL regime with 0 confidence.
        """
        results: list[ERResult] = []
        for idx in range(len(closes)):
            results.append(self.update(idx, closes))
        return results
