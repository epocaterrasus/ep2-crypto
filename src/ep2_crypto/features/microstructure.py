"""Microstructure feature computers: OBI, OFI, microprice, TFI, spread, absorption.

These features capture 60-80% of achievable 5-min signal according to research.
All computations use only data at times <= current index (no look-ahead).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ep2_crypto.features.base import FeatureComputer

if TYPE_CHECKING:
    from numpy.typing import NDArray


class OBIComputer(FeatureComputer):
    """Order Book Imbalance at multiple depth levels.

    OBI = (bid_vol - ask_vol) / (bid_vol + ask_vol)
    Weighted variant uses inverse distance to mid as weights.

    Expects bid_sizes and ask_sizes as 2D arrays: (n_bars, n_levels).
    Expects bids and asks as 2D arrays: (n_bars, n_levels) for weighted OBI.
    """

    @property
    def name(self) -> str:
        return "obi"

    @property
    def warmup_bars(self) -> int:
        return 1

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
        bids: NDArray[np.float64] | None = None,
        asks: NDArray[np.float64] | None = None,
        bid_sizes: NDArray[np.float64] | None = None,
        ask_sizes: NDArray[np.float64] | None = None,
        **kwargs: NDArray[np.float64] | None,
    ) -> dict[str, float]:
        nan_result = {
            "obi_l3": float("nan"),
            "obi_l5": float("nan"),
            "obi_l3_weighted": float("nan"),
            "obi_l5_weighted": float("nan"),
        }

        if idx < self.warmup_bars - 1:
            return nan_result

        if bid_sizes is None or ask_sizes is None:
            return nan_result

        bs = bid_sizes[idx]
        as_ = ask_sizes[idx]

        n_levels = min(len(bs), len(as_))
        if n_levels < 3:
            return nan_result

        result: dict[str, float] = {}

        # Simple OBI at levels 1-3 and 1-5
        for depth, label in [(3, "l3"), (5, "l5")]:
            d = min(depth, n_levels)
            b_sum = float(np.sum(bs[:d]))
            a_sum = float(np.sum(as_[:d]))
            total = b_sum + a_sum
            result[f"obi_{label}"] = (b_sum - a_sum) / total if total > 0 else 0.0

        # Weighted OBI: weight by inverse distance to mid
        if bids is not None and asks is not None:
            bp = bids[idx]
            ap = asks[idx]
            mid = (bp[0] + ap[0]) / 2.0 if bp[0] > 0 and ap[0] > 0 else 0.0

            for depth, label in [(3, "l3"), (5, "l5")]:
                d = min(depth, n_levels)
                if mid > 0:
                    bid_dist = np.abs(bp[:d] - mid)
                    ask_dist = np.abs(ap[:d] - mid)
                    # Inverse distance weights (avoid div by zero)
                    bid_w = np.where(bid_dist > 0, 1.0 / bid_dist, 1.0)
                    ask_w = np.where(ask_dist > 0, 1.0 / ask_dist, 1.0)
                    weighted_bid = float(np.sum(bs[:d] * bid_w))
                    weighted_ask = float(np.sum(as_[:d] * ask_w))
                    total_w = weighted_bid + weighted_ask
                    result[f"obi_{label}_weighted"] = (
                        (weighted_bid - weighted_ask) / total_w if total_w > 0 else 0.0
                    )
                else:
                    result[f"obi_{label}_weighted"] = 0.0
        else:
            result["obi_l3_weighted"] = result["obi_l3"]
            result["obi_l5_weighted"] = result["obi_l5"]

        return result

    def output_names(self) -> list[str]:
        return ["obi_l3", "obi_l5", "obi_l3_weighted", "obi_l5_weighted"]


class OFIComputer(FeatureComputer):
    """Order Flow Imbalance (Cont-Stoikov-Talreja model).

    Tracks changes in top-of-book quantities between consecutive snapshots.
    Handles all 6 cases for bid/ask price movements.

    Requires 2D bids, asks, bid_sizes, ask_sizes arrays (n_bars, n_levels).
    """

    @property
    def name(self) -> str:
        return "ofi"

    @property
    def warmup_bars(self) -> int:
        return 2

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
        bids: NDArray[np.float64] | None = None,
        asks: NDArray[np.float64] | None = None,
        bid_sizes: NDArray[np.float64] | None = None,
        ask_sizes: NDArray[np.float64] | None = None,
        **kwargs: NDArray[np.float64] | None,
    ) -> dict[str, float]:
        nan_result = {
            "ofi_l1": float("nan"),
            "ofi_l3": float("nan"),
            "ofi_l5": float("nan"),
        }

        if idx < self.warmup_bars - 1:
            return nan_result

        if bids is None or asks is None or bid_sizes is None or ask_sizes is None:
            return nan_result

        n_levels = min(len(bids[idx]), len(asks[idx]))
        if n_levels < 1:
            return nan_result

        result: dict[str, float] = {}

        for depth, label in [(1, "l1"), (3, "l3"), (5, "l5")]:
            d = min(depth, n_levels)
            ofi_total = 0.0
            for lev in range(d):
                ofi_total += _compute_ofi_level(
                    bids[idx - 1, lev], bid_sizes[idx - 1, lev],
                    bids[idx, lev], bid_sizes[idx, lev],
                    asks[idx - 1, lev], ask_sizes[idx - 1, lev],
                    asks[idx, lev], ask_sizes[idx, lev],
                )
            result[f"ofi_{label}"] = ofi_total

        return result

    def output_names(self) -> list[str]:
        return ["ofi_l1", "ofi_l3", "ofi_l5"]


def _compute_ofi_level(
    prev_bid: float, prev_bid_sz: float,
    curr_bid: float, curr_bid_sz: float,
    prev_ask: float, prev_ask_sz: float,
    curr_ask: float, curr_ask_sz: float,
) -> float:
    """Compute OFI for a single level using Cont-Stoikov-Talreja cases.

    Bid side contribution (delta_b):
        - bid price increased: +curr_bid_sz (new orders at better price)
        - bid price same: +(curr_bid_sz - prev_bid_sz)
        - bid price decreased: -prev_bid_sz (orders removed)

    Ask side contribution (delta_a):
        - ask price decreased: -curr_ask_sz (new orders at better price)
        - ask price same: -(curr_ask_sz - prev_ask_sz)
        - ask price increased: +prev_ask_sz (orders removed)

    OFI = delta_b - delta_a
    """
    # Bid side
    if curr_bid > prev_bid:
        delta_b = curr_bid_sz
    elif curr_bid == prev_bid:
        delta_b = curr_bid_sz - prev_bid_sz
    else:
        delta_b = -prev_bid_sz

    # Ask side
    if curr_ask < prev_ask:
        delta_a = -curr_ask_sz
    elif curr_ask == prev_ask:
        delta_a = -(curr_ask_sz - prev_ask_sz)
    else:
        delta_a = prev_ask_sz

    return delta_b - delta_a


class MicropriceComputer(FeatureComputer):
    """Gatheral-Stoikov microprice: volume-weighted mid price.

    microprice = (ask_size * bid_price + bid_size * ask_price) / (bid_size + ask_size)

    This is a better estimate of fair value than simple mid price when
    there's an imbalance in the order book.

    Also computes microprice deviation from mid (the signal).
    """

    @property
    def name(self) -> str:
        return "microprice"

    @property
    def warmup_bars(self) -> int:
        return 1

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
        bids: NDArray[np.float64] | None = None,
        asks: NDArray[np.float64] | None = None,
        bid_sizes: NDArray[np.float64] | None = None,
        ask_sizes: NDArray[np.float64] | None = None,
        **kwargs: NDArray[np.float64] | None,
    ) -> dict[str, float]:
        nan_result = {
            "microprice": float("nan"),
            "microprice_mid_dev": float("nan"),
        }

        if idx < self.warmup_bars - 1:
            return nan_result

        if bids is None or asks is None or bid_sizes is None or ask_sizes is None:
            return nan_result

        best_bid = bids[idx, 0] if bids.ndim > 1 else bids[idx]
        best_ask = asks[idx, 0] if asks.ndim > 1 else asks[idx]
        best_bid_sz = bid_sizes[idx, 0] if bid_sizes.ndim > 1 else bid_sizes[idx]
        best_ask_sz = ask_sizes[idx, 0] if ask_sizes.ndim > 1 else ask_sizes[idx]

        total_sz = best_bid_sz + best_ask_sz
        if total_sz <= 0 or best_bid <= 0 or best_ask <= 0:
            return nan_result

        micro = (best_ask_sz * best_bid + best_bid_sz * best_ask) / total_sz
        mid = (best_bid + best_ask) / 2.0
        dev = (micro - mid) / mid if mid > 0 else 0.0

        return {
            "microprice": micro,
            "microprice_mid_dev": dev,
        }

    def output_names(self) -> list[str]:
        return ["microprice", "microprice_mid_dev"]


class TFIComputer(FeatureComputer):
    """Trade Flow Imbalance: (buy_vol - sell_vol) / total_vol over window.

    Uses trade_sizes (positive) and trade_sides (+1 buy, -1 sell).
    Computes TFI at 30-second and 5-minute windows.

    Also computes relative spread and absorption detection.
    """

    def __init__(self, bar_interval_s: int = 300) -> None:
        self._bar_interval_s = bar_interval_s

    @property
    def name(self) -> str:
        return "tfi"

    @property
    def warmup_bars(self) -> int:
        return 2

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
        bids: NDArray[np.float64] | None = None,
        asks: NDArray[np.float64] | None = None,
        bid_sizes: NDArray[np.float64] | None = None,
        ask_sizes: NDArray[np.float64] | None = None,
        trade_prices: NDArray[np.float64] | None = None,
        trade_sizes: NDArray[np.float64] | None = None,
        trade_sides: NDArray[np.float64] | None = None,
    ) -> dict[str, float]:
        nan_result = {
            "tfi_1bar": float("nan"),
            "tfi_6bar": float("nan"),
            "relative_spread": float("nan"),
            "absorption": float("nan"),
        }

        if idx < self.warmup_bars - 1:
            return nan_result

        result: dict[str, float] = {}

        # TFI from trade data (per-bar aggregated buy/sell volumes)
        if trade_sizes is not None and trade_sides is not None:
            # 1-bar TFI
            signed = trade_sizes[idx] * trade_sides[idx]
            total = abs(trade_sizes[idx])
            result["tfi_1bar"] = float(signed / total) if total > 0 else 0.0

            # 6-bar TFI (approx 30 min for 5-min bars, or 30s for 5s bars)
            lookback = min(6, idx + 1)
            start = idx - lookback + 1
            signed_sum = float(np.sum(trade_sizes[start:idx + 1] * trade_sides[start:idx + 1]))
            total_sum = float(np.sum(np.abs(trade_sizes[start:idx + 1])))
            result["tfi_6bar"] = signed_sum / total_sum if total_sum > 0 else 0.0
        else:
            result["tfi_1bar"] = float("nan")
            result["tfi_6bar"] = float("nan")

        # Relative spread
        if bids is not None and asks is not None:
            best_bid = bids[idx, 0] if bids.ndim > 1 else bids[idx]
            best_ask = asks[idx, 0] if asks.ndim > 1 else asks[idx]
            mid = (best_bid + best_ask) / 2.0
            result["relative_spread"] = (
                (best_ask - best_bid) / mid if mid > 0 else float("nan")
            )
        else:
            result["relative_spread"] = float("nan")

        # Absorption detection: high volume delta with low price change
        if trade_sizes is not None and trade_sides is not None and idx >= 1:
            signed_vol = float(trade_sizes[idx] * trade_sides[idx])
            price_change = abs(closes[idx] - closes[idx - 1])
            avg_price = (closes[idx] + closes[idx - 1]) / 2.0
            rel_price_change = price_change / avg_price if avg_price > 0 else 0.0
            abs_signed_vol = abs(signed_vol)

            # Absorption = high directional volume but small price move
            # Normalized: vol_intensity / (1 + price_change_bps * 100)
            # High value = absorption happening
            price_change_bps = rel_price_change * 10_000
            result["absorption"] = (
                abs_signed_vol / (1.0 + price_change_bps)
                if abs_signed_vol > 0
                else 0.0
            )
        else:
            result["absorption"] = float("nan")

        return result

    def output_names(self) -> list[str]:
        return ["tfi_1bar", "tfi_6bar", "relative_spread", "absorption"]


class KyleLambdaComputer(FeatureComputer):
    """Kyle's Lambda: rolling price impact coefficient.

    Lambda = Cov(delta_price, signed_volume) / Var(signed_volume)

    Estimated over a rolling window. High lambda means high price impact,
    which conditions how OBI/OFI signals should be interpreted.
    """

    def __init__(self, window: int = 20) -> None:
        self._window = window

    @property
    def name(self) -> str:
        return "kyle_lambda"

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
        *,
        trade_sizes: NDArray[np.float64] | None = None,
        trade_sides: NDArray[np.float64] | None = None,
        **kwargs: NDArray[np.float64] | None,
    ) -> dict[str, float]:
        if idx < self.warmup_bars - 1:
            return {"kyle_lambda": float("nan")}

        if trade_sizes is None or trade_sides is None:
            return {"kyle_lambda": float("nan")}

        start = idx - self._window + 1
        # Price changes
        delta_p = np.diff(closes[start:idx + 1])
        # Signed volumes (aligned with price changes)
        signed_v = (trade_sizes[start + 1:idx + 1] * trade_sides[start + 1:idx + 1])

        if len(delta_p) < 2 or len(signed_v) < 2:
            return {"kyle_lambda": float("nan")}

        var_v = float(np.var(signed_v))
        if var_v <= 0:
            return {"kyle_lambda": float("nan")}

        cov = float(np.cov(delta_p, signed_v)[0, 1])
        return {"kyle_lambda": cov / var_v}

    def output_names(self) -> list[str]:
        return ["kyle_lambda"]
