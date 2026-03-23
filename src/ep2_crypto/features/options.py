"""Options-derived features from Deribit IV surface data.

Features extracted from implied volatility surface:
- ATM IV (at-the-money implied volatility): level signal
- IV term structure slope: short-end vs long-end IV ratio (contango/backwardation)
- 25-delta risk reversal: market skew toward puts (bearish) or calls (bullish)
- IV z-score: deviation from rolling mean (spike detection)

All features use only past snapshots (no look-ahead bias).
Rolling statistics are computed on a fixed-size deque of past snapshots.

Research: Deribit IV predictive power for short-term direction — IV term structure
inversion (short > long) precedes mean-reversion moves. Negative RR (put skew)
signals institutional hedging demand.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass


@dataclass
class OptionsFeatures:
    """Options-derived features for model consumption.

    Attributes:
        atm_iv_7d: ATM implied volatility at ~7-day expiry. NaN if unavailable.
        iv_slope: (atm_iv_7d - atm_iv_30d) / atm_iv_30d. Positive = inverted
            term structure (short IV > long IV, often bearish).
            NaN if atm_iv_30d unavailable.
        rr_25d: 25-delta risk reversal (call IV - put IV). Negative = put skew
            (bearish hedging demand). NaN if unavailable.
        iv_zscore: Rolling z-score of atm_iv_7d vs past `window` snapshots.
            NaN if fewer than 2 past snapshots.
        iv_pct_change_1h: Percent change in ATM IV over last ~1 hour of snapshots.
            NaN if insufficient history.
    """

    atm_iv_7d: float
    iv_slope: float
    rr_25d: float
    iv_zscore: float
    iv_pct_change_1h: float


class OptionsFeatureComputer:
    """Computes rolling options features from a stream of IVSnapshot objects.

    Maintains a rolling window of past ATM IV values for z-score and rate-of-change
    computation. All outputs use only data from times <= current snapshot.

    Args:
        window: Number of past snapshots for rolling statistics (default 60 = 1h
            at 1-min poll interval).
        pct_change_lookback: Number of snapshots back for pct-change feature.
    """

    def __init__(self, window: int = 60, pct_change_lookback: int = 60) -> None:
        self._window = window
        self._pct_change_lookback = pct_change_lookback
        self._iv_history: deque[float] = deque(maxlen=window + pct_change_lookback)

    def update(
        self,
        atm_iv_7d: float,
        atm_iv_14d: float | None,
        atm_iv_30d: float | None,
        rr_25d: float | None,
    ) -> OptionsFeatures:
        """Ingest one IV snapshot and return computed features.

        Args:
            atm_iv_7d: ATM IV at ~7-day expiry (annualized, e.g., 0.65 = 65%).
            atm_iv_14d: ATM IV at ~14-day expiry. None if unavailable.
            atm_iv_30d: ATM IV at ~30-day expiry. None if unavailable.
            rr_25d: 25-delta risk reversal. None if unavailable.

        Returns:
            OptionsFeatures with all computed fields. Unavailable fields = NaN.
        """
        nan = float("nan")

        # IV term structure slope: (short_iv - long_iv) / long_iv
        iv_slope = nan
        if atm_iv_30d is not None and atm_iv_30d > 1e-8:
            iv_slope = (atm_iv_7d - atm_iv_30d) / atm_iv_30d

        # Rolling z-score of atm_iv_7d
        iv_zscore = nan
        if len(self._iv_history) >= 2:
            arr = list(self._iv_history)
            n = len(arr)
            mean = sum(arr) / n
            variance = sum((x - mean) ** 2 for x in arr) / (n - 1)
            std = math.sqrt(variance) if variance > 1e-12 else nan
            if not math.isnan(std):
                iv_zscore = (atm_iv_7d - mean) / std

        # Percent change over pct_change_lookback steps
        iv_pct_change = nan
        if len(self._iv_history) >= self._pct_change_lookback:
            past_iv = self._iv_history[-(self._pct_change_lookback)]
            if past_iv > 1e-8:
                iv_pct_change = (atm_iv_7d - past_iv) / past_iv

        # Append to history AFTER computing features (no look-ahead)
        self._iv_history.append(atm_iv_7d)

        return OptionsFeatures(
            atm_iv_7d=atm_iv_7d,
            iv_slope=iv_slope,
            rr_25d=rr_25d if rr_25d is not None else nan,
            iv_zscore=iv_zscore,
            iv_pct_change_1h=iv_pct_change,
        )

    def reset(self) -> None:
        """Clear rolling history."""
        self._iv_history.clear()

    @property
    def n_snapshots(self) -> int:
        """Number of snapshots in rolling history."""
        return len(self._iv_history)
