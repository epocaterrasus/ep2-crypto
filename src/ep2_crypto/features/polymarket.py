"""Polymarket market-implied probability features.

Uses resolved 5-minute BTC up/down markets from Polymarket as crowd-consensus
signals. Features are computed from the PREVIOUS window's data to prevent
look-ahead bias — the prior market resolved before the current bar opens.

Features produced:
  poly_yes_price_lag1      : yes_close_price of the previous 5-min market
                             (0.0 = crowd expected DOWN, 1.0 = expected UP,
                              0.5 = no consensus; for resolved markets this
                              converges toward 0 or 1)
  poly_volume_log_lag1     : log1p of previous market volume (USD)
                             (crowd activity / conviction level)
  poly_rolling_accuracy_20 : rolling accuracy of Polymarket predictions
                             over the last 20 resolved windows
  poly_market_exists       : 1.0 if Polymarket had a market this bar, else 0.0
"""

from __future__ import annotations

import numpy as np

from ep2_crypto.features.base import FeatureComputer

if False:  # TYPE_CHECKING
    from numpy.typing import NDArray


class PolymarketProbComputer(FeatureComputer):
    """Crowd-consensus features from Polymarket 5-min BTC markets.

    Reads optional ``poly_yes_prices``, ``poly_volumes``, and
    ``poly_resolved`` kwargs from the pipeline. Gracefully degrades to
    NaN when no Polymarket data is available (e.g., before Feb 2025).

    Warmup equals the rolling accuracy window (default 20 bars).
    """

    def __init__(self, accuracy_window: int = 20) -> None:
        self._accuracy_window = accuracy_window

    @property
    def name(self) -> str:
        return "polymarket_prob"

    @property
    def warmup_bars(self) -> int:
        return self._accuracy_window + 1

    def output_names(self) -> list[str]:
        return [
            "poly_yes_price_lag1",
            "poly_volume_log_lag1",
            "poly_rolling_accuracy_20",
            "poly_market_exists",
        ]

    def compute(
        self,
        idx: int,
        timestamps: "NDArray[np.int64]",
        opens: "NDArray[np.float64]",
        highs: "NDArray[np.float64]",
        lows: "NDArray[np.float64]",
        closes: "NDArray[np.float64]",
        volumes: "NDArray[np.float64]",
        *,
        poly_yes_prices: "NDArray[np.float64] | None" = None,
        poly_volumes: "NDArray[np.float64] | None" = None,
        poly_resolved: "NDArray[np.float64] | None" = None,
        **_kwargs: object,
    ) -> dict[str, float]:
        nan = float("nan")
        if poly_yes_prices is None or idx < 1:
            return {
                "poly_yes_price_lag1": nan,
                "poly_volume_log_lag1": nan,
                "poly_rolling_accuracy_20": nan,
                "poly_market_exists": 0.0,
            }

        # Lag-1: previous bar's market data (resolved before current bar opens)
        prev = idx - 1
        yes_lag1 = float(poly_yes_prices[prev])
        market_exists = 1.0 if np.isfinite(yes_lag1) else 0.0

        if not np.isfinite(yes_lag1):
            return {
                "poly_yes_price_lag1": nan,
                "poly_volume_log_lag1": nan,
                "poly_rolling_accuracy_20": nan,
                "poly_market_exists": 0.0,
            }

        # Volume (log-normalized)
        vol_lag1 = nan
        if poly_volumes is not None and np.isfinite(poly_volumes[prev]):
            vol_lag1 = float(np.log1p(max(0.0, poly_volumes[prev])))

        # Rolling accuracy: fraction of resolved markets where yes_price > 0.5
        # correctly predicted the BTC direction (close > open)
        accuracy = nan
        if poly_resolved is not None and idx >= self._accuracy_window:
            window_start = max(0, idx - self._accuracy_window)
            window_yes = poly_yes_prices[window_start:idx]
            window_res = poly_resolved[window_start:idx]
            window_closes = closes[window_start:idx]
            window_opens = opens[window_start:idx]

            correct = 0
            total = 0
            for i in range(len(window_yes)):
                if not np.isfinite(window_res[i]) or window_res[i] == 0:
                    continue
                predicted_up = window_yes[i] >= 0.5
                actually_up = window_closes[i] > window_opens[i]
                if predicted_up == actually_up:
                    correct += 1
                total += 1

            if total >= 5:  # need at least 5 resolved markets for accuracy
                accuracy = correct / total

        return {
            "poly_yes_price_lag1": yes_lag1,
            "poly_volume_log_lag1": vol_lag1,
            "poly_rolling_accuracy_20": accuracy,
            "poly_market_exists": market_exists,
        }
