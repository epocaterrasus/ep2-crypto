"""Cross-market feature computers: NQ returns, ETH ratio, lead-lag, divergence.

Cross-market signals provide context from correlated assets. NQ (Nasdaq futures)
leads BTC during US hours. ETH/BTC ratio captures relative strength.
All computations use only data at times <= current index (no look-ahead).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ep2_crypto.features.base import FeatureComputer

if TYPE_CHECKING:
    from numpy.typing import NDArray

# US session: 16:00-00:00 UTC (NQ active hours)
_US_SESSION_START_HOUR = 16
_US_SESSION_END_HOUR = 24  # midnight


def _is_us_session(timestamp_ms: int) -> bool:
    """Check if timestamp falls within US session (16:00-00:00 UTC)."""
    hour = int((timestamp_ms // 3_600_000) % 24)
    return _US_SESSION_START_HOUR <= hour < _US_SESSION_END_HOUR


class NQReturnComputer(FeatureComputer):
    """NQ (Nasdaq) 5-min returns lagged 1-3 bars.

    NQ leads BTC during US hours. Returns are gated to US session only;
    outside US hours, values are zero (no signal, not NaN, to keep
    tree models happy).

    Expects nq_closes as a 1D array of NQ close prices per bar.
    """

    @property
    def name(self) -> str:
        return "nq_return"

    @property
    def warmup_bars(self) -> int:
        return 4  # Need 3 bars of lookback + current

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
        nq_closes: NDArray[np.float64] | None = None,
        **kwargs: NDArray[np.float64] | None,
    ) -> dict[str, float]:
        nan_result = {
            "nq_ret_lag1": float("nan"),
            "nq_ret_lag2": float("nan"),
            "nq_ret_lag3": float("nan"),
        }

        if idx < self.warmup_bars - 1:
            return nan_result

        if nq_closes is None:
            return nan_result

        # Gate to US session
        if not _is_us_session(int(timestamps[idx])):
            return {
                "nq_ret_lag1": 0.0,
                "nq_ret_lag2": 0.0,
                "nq_ret_lag3": 0.0,
            }

        result: dict[str, float] = {}
        for lag in [1, 2, 3]:
            prev_idx = idx - lag
            prev_prev = prev_idx - 1
            if prev_prev >= 0 and nq_closes[prev_prev] > 0:
                ret = (nq_closes[prev_idx] - nq_closes[prev_prev]) / nq_closes[prev_prev]
                result[f"nq_ret_lag{lag}"] = float(ret)
            else:
                result[f"nq_ret_lag{lag}"] = 0.0

        return result

    def output_names(self) -> list[str]:
        return ["nq_ret_lag1", "nq_ret_lag2", "nq_ret_lag3"]


class ETHRatioComputer(FeatureComputer):
    """ETH/BTC ratio and its momentum.

    Ratio = eth_close / btc_close.
    Ratio momentum = ROC of ratio at 1-bar and 6-bar lookback.

    When ETH strengthens vs BTC (ratio rising), it often signals
    risk-on sentiment in crypto markets.

    Expects eth_closes as a 1D array of ETH close prices per bar.
    """

    @property
    def name(self) -> str:
        return "eth_ratio"

    @property
    def warmup_bars(self) -> int:
        return 7  # Need 6-bar lookback for momentum

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
        eth_closes: NDArray[np.float64] | None = None,
        **kwargs: NDArray[np.float64] | None,
    ) -> dict[str, float]:
        nan_result = {
            "eth_btc_ratio": float("nan"),
            "eth_btc_ratio_roc1": float("nan"),
            "eth_btc_ratio_roc6": float("nan"),
        }

        if idx < self.warmup_bars - 1:
            return nan_result

        if eth_closes is None:
            return nan_result

        btc = closes[idx]
        eth = eth_closes[idx]
        if btc <= 0 or eth <= 0:
            return nan_result

        ratio = eth / btc
        result: dict[str, float] = {"eth_btc_ratio": ratio}

        # ROC of ratio at 1-bar
        prev_btc = closes[idx - 1]
        prev_eth = eth_closes[idx - 1]
        if prev_btc > 0 and prev_eth > 0:
            prev_ratio = prev_eth / prev_btc
            roc1 = (ratio - prev_ratio) / prev_ratio if prev_ratio > 0 else 0.0
            result["eth_btc_ratio_roc1"] = roc1
        else:
            result["eth_btc_ratio_roc1"] = 0.0

        # ROC of ratio at 6-bar
        prev6_btc = closes[idx - 6]
        prev6_eth = eth_closes[idx - 6]
        if prev6_btc > 0 and prev6_eth > 0:
            prev6_ratio = prev6_eth / prev6_btc
            roc6 = (ratio - prev6_ratio) / prev6_ratio if prev6_ratio > 0 else 0.0
            result["eth_btc_ratio_roc6"] = roc6
        else:
            result["eth_btc_ratio_roc6"] = 0.0

        return result

    def output_names(self) -> list[str]:
        return ["eth_btc_ratio", "eth_btc_ratio_roc1", "eth_btc_ratio_roc6"]


class LeadLagComputer(FeatureComputer):
    """Rolling lead-lag correlation between BTC returns and NQ returns at multiple lags.

    Correlation is computed over a rolling window. Positive correlation at lag k
    means NQ return at t-k predicts BTC return at t.

    Expects nq_closes as a 1D array.
    """

    def __init__(self, window: int = 20) -> None:
        self._window = window

    @property
    def name(self) -> str:
        return "lead_lag"

    @property
    def warmup_bars(self) -> int:
        return self._window + 3  # window + max lag

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
        nq_closes: NDArray[np.float64] | None = None,
        **kwargs: NDArray[np.float64] | None,
    ) -> dict[str, float]:
        nan_result = {
            "lead_lag_corr_1": float("nan"),
            "lead_lag_corr_2": float("nan"),
            "lead_lag_corr_3": float("nan"),
        }

        if idx < self.warmup_bars - 1:
            return nan_result

        if nq_closes is None:
            return nan_result

        # BTC returns over the window
        start = idx - self._window + 1
        btc_rets = np.diff(closes[start - 1:idx + 1]) / closes[start - 1:idx]

        result: dict[str, float] = {}
        for lag in [1, 2, 3]:
            # NQ returns lagged by `lag` bars
            nq_start = start - lag
            nq_end = idx - lag + 1
            nq_rets = np.diff(nq_closes[nq_start - 1:nq_end]) / nq_closes[nq_start - 1:nq_end - 1]

            min_len = min(len(btc_rets), len(nq_rets))
            if min_len < 5:
                result[f"lead_lag_corr_{lag}"] = 0.0
                continue

            # Align: use last min_len elements
            b = btc_rets[-min_len:]
            n = nq_rets[-min_len:]

            std_b = float(np.std(b))
            std_n = float(np.std(n))
            if std_b < 1e-15 or std_n < 1e-15:
                result[f"lead_lag_corr_{lag}"] = 0.0
            else:
                corr = float(np.corrcoef(b, n)[0, 1])
                result[f"lead_lag_corr_{lag}"] = corr if np.isfinite(corr) else 0.0

        return result

    def output_names(self) -> list[str]:
        return ["lead_lag_corr_1", "lead_lag_corr_2", "lead_lag_corr_3"]


class DivergenceComputer(FeatureComputer):
    """Divergence signals: BTC vs NQ/ETH moving in opposite directions.

    Measures sign disagreement and magnitude of divergence over a lookback.
    A positive divergence means BTC is outperforming the reference asset.

    Expects nq_closes and eth_closes as 1D arrays.
    """

    def __init__(self, lookback: int = 6) -> None:
        self._lookback = lookback

    @property
    def name(self) -> str:
        return "divergence"

    @property
    def warmup_bars(self) -> int:
        return self._lookback + 1

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
        nq_closes: NDArray[np.float64] | None = None,
        eth_closes: NDArray[np.float64] | None = None,
        **kwargs: NDArray[np.float64] | None,
    ) -> dict[str, float]:
        nan_result = {
            "div_btc_nq": float("nan"),
            "div_btc_eth": float("nan"),
        }

        if idx < self.warmup_bars - 1:
            return nan_result

        prev_idx = idx - self._lookback
        prev_close = closes[prev_idx]
        btc_ret = (closes[idx] - prev_close) / prev_close if prev_close > 0 else 0.0

        result: dict[str, float] = {}

        # BTC vs NQ divergence
        if nq_closes is not None and nq_closes[prev_idx] > 0:
            nq_ret = (nq_closes[idx] - nq_closes[prev_idx]) / nq_closes[prev_idx]
            result["div_btc_nq"] = btc_ret - nq_ret
        else:
            result["div_btc_nq"] = 0.0

        # BTC vs ETH divergence
        if eth_closes is not None and eth_closes[prev_idx] > 0:
            eth_ret = (eth_closes[idx] - eth_closes[prev_idx]) / eth_closes[prev_idx]
            result["div_btc_eth"] = btc_ret - eth_ret
        else:
            result["div_btc_eth"] = 0.0

        return result

    def output_names(self) -> list[str]:
        return ["div_btc_nq", "div_btc_eth"]


class CoinbasePremiumComputer(FeatureComputer):
    """Coinbase premium over Binance: indicator of US institutional flow.

    Premium = (coinbase_price - binance_price) / binance_price
    IC 0.03-0.07 at 30-min horizon (Augustin et al. 2022).

    Positive premium → US buyers dominant (bullish bias).
    Computes: raw premium, z-score over rolling window, and 1-bar delta.

    Expects coinbase_closes as a 1D array aligned with BTC bars.
    """

    def __init__(self, zscore_window: int = 288) -> None:
        self._zscore_window = zscore_window

    @property
    def name(self) -> str:
        return "coinbase_premium"

    @property
    def warmup_bars(self) -> int:
        return self._zscore_window + 1

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
        coinbase_closes: NDArray[np.float64] | None = None,
        **kwargs: NDArray[np.float64] | None,
    ) -> dict[str, float]:
        nan_result = {
            "coinbase_premium": float("nan"),
            "coinbase_premium_zscore": float("nan"),
            "coinbase_premium_delta": float("nan"),
        }

        if idx < self.warmup_bars - 1:
            return nan_result
        if coinbase_closes is None:
            return nan_result

        binance = closes[idx]
        coinbase = coinbase_closes[idx]
        if binance <= 0 or coinbase <= 0:
            return nan_result

        premium = (coinbase - binance) / binance

        # Rolling z-score
        start = idx - self._zscore_window + 1
        bin_window = closes[start:idx + 1]
        cb_window = coinbase_closes[start:idx + 1]
        valid = (bin_window > 0) & (cb_window > 0)
        if np.sum(valid) < 10:
            return nan_result

        premiums = (cb_window[valid] - bin_window[valid]) / bin_window[valid]
        mu = float(np.mean(premiums))
        sigma = float(np.std(premiums))
        zscore = (premium - mu) / sigma if sigma > 1e-12 else 0.0

        # 1-bar delta
        prev_bin = closes[idx - 1]
        prev_cb = coinbase_closes[idx - 1]
        if prev_bin > 0 and prev_cb > 0:
            prev_premium = (prev_cb - prev_bin) / prev_bin
            delta = premium - prev_premium
        else:
            delta = 0.0

        return {
            "coinbase_premium": premium,
            "coinbase_premium_zscore": zscore,
            "coinbase_premium_delta": delta,
        }

    def output_names(self) -> list[str]:
        return ["coinbase_premium", "coinbase_premium_zscore", "coinbase_premium_delta"]


class ETHOrderFlowComputer(FeatureComputer):
    """ETH net taker volume as a leading indicator for BTC.

    ETH order flow leads BTC by 1-5 min (54-57% accuracy per Alexander & Dakos 2020).
    Positive net taker = more aggressive buying in ETH → bullish BTC signal.

    Computes:
    - eth_net_taker: signed(trade_side) * volume per bar
    - eth_net_taker_1lag: lagged 1 bar
    - eth_net_taker_3lag: lagged 3 bars (15-min lead)
    - eth_net_taker_zscore: rolling z-score

    Expects eth_trade_sizes and eth_trade_sides arrays (same shape as closes).
    """

    def __init__(self, zscore_window: int = 60) -> None:
        self._zscore_window = zscore_window

    @property
    def name(self) -> str:
        return "eth_order_flow"

    @property
    def warmup_bars(self) -> int:
        return self._zscore_window + 3

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
        eth_trade_sizes: NDArray[np.float64] | None = None,
        eth_trade_sides: NDArray[np.float64] | None = None,
        **kwargs: NDArray[np.float64] | None,
    ) -> dict[str, float]:
        nan_result = {
            "eth_net_taker": float("nan"),
            "eth_net_taker_lag1": float("nan"),
            "eth_net_taker_lag3": float("nan"),
            "eth_net_taker_zscore": float("nan"),
        }

        if idx < self.warmup_bars - 1:
            return nan_result
        if eth_trade_sizes is None or eth_trade_sides is None:
            return nan_result

        def _net_taker(i: int) -> float:
            sz = eth_trade_sizes[i]
            side = eth_trade_sides[i]
            total = abs(float(sz))
            return float(sz * side) if total > 0 else 0.0

        net = _net_taker(idx)
        net_lag1 = _net_taker(idx - 1)
        net_lag3 = _net_taker(idx - 3)

        # Rolling z-score of net taker over window
        start = idx - self._zscore_window + 1
        raw = np.array([
            float(eth_trade_sizes[i] * eth_trade_sides[i])
            for i in range(start, idx + 1)
        ])
        mu = float(np.mean(raw))
        sigma = float(np.std(raw))
        zscore = (net - mu) / sigma if sigma > 1e-12 else 0.0

        return {
            "eth_net_taker": net,
            "eth_net_taker_lag1": net_lag1,
            "eth_net_taker_lag3": net_lag3,
            "eth_net_taker_zscore": zscore,
        }

    def output_names(self) -> list[str]:
        return [
            "eth_net_taker",
            "eth_net_taker_lag1",
            "eth_net_taker_lag3",
            "eth_net_taker_zscore",
        ]


class LongShortRatioComputer(FeatureComputer):
    """Binance Long/Short Ratio: contrarian signal at extremes.

    The ratio measures the fraction of traders with long vs short positions.
    At extremes (>2.5 or <0.5), the crowd is typically wrong → contrarian.

    Signals:
    - ratio > 2.5: extreme long crowding → bearish contrarian signal
    - ratio < 0.5: extreme short crowding → bullish contrarian signal
    - 0.5-2.5: neutral zone, no signal

    Computes: raw ratio, z-score vs 7-day history, and extremeness score.
    """

    def __init__(self, zscore_window: int = 2016) -> None:
        # 7 days × 288 bars/day
        self._zscore_window = zscore_window

    @property
    def name(self) -> str:
        return "long_short_ratio"

    @property
    def warmup_bars(self) -> int:
        return self._zscore_window + 1

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
        long_short_ratio: NDArray[np.float64] | None = None,
        **kwargs: NDArray[np.float64] | None,
    ) -> dict[str, float]:
        nan_result = {
            "ls_ratio": float("nan"),
            "ls_ratio_zscore": float("nan"),
            "ls_contrarian": float("nan"),
        }

        if idx < self.warmup_bars - 1:
            return nan_result
        if long_short_ratio is None:
            return nan_result

        ratio = float(long_short_ratio[idx])
        if ratio <= 0:
            return nan_result

        # Rolling z-score
        start = idx - self._zscore_window + 1
        window = long_short_ratio[start:idx + 1]
        valid = window > 0
        if np.sum(valid) < 10:
            return nan_result

        mu = float(np.mean(window[valid]))
        sigma = float(np.std(window[valid]))
        zscore = (ratio - mu) / sigma if sigma > 1e-12 else 0.0

        # Contrarian score: how far into extreme territory
        # >2.5 → negative score (bearish), <0.5 → positive score (bullish)
        if ratio > 2.5:
            contrarian = -(ratio - 2.5) / 2.5  # negative = expect down
        elif ratio < 0.5:
            contrarian = (0.5 - ratio) / 0.5   # positive = expect up
        else:
            contrarian = 0.0

        return {
            "ls_ratio": ratio,
            "ls_ratio_zscore": zscore,
            "ls_contrarian": contrarian,
        }

    def output_names(self) -> list[str]:
        return ["ls_ratio", "ls_ratio_zscore", "ls_contrarian"]


class CrossExchangeOFIComputer(FeatureComputer):
    """Cross-exchange OFI divergence: which exchange is leading price discovery.

    When OFI diverges across exchanges, the highest-volume exchange predicts
    price direction 58-63% of the time (Makarov & Schoar 2020).

    OFI divergence = OFI_binance - OFI_coinbase (or other exchange).
    Positive divergence = Binance buyers are more aggressive than Coinbase.
    Both normalized by their respective volume.

    Expects binance_ofi and coinbase_ofi as 1D arrays (pre-computed per bar).
    """

    def __init__(self, window: int = 12) -> None:
        self._window = window

    @property
    def name(self) -> str:
        return "cross_exchange_ofi"

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
        *,
        binance_ofi: NDArray[np.float64] | None = None,
        coinbase_ofi: NDArray[np.float64] | None = None,
        **kwargs: NDArray[np.float64] | None,
    ) -> dict[str, float]:
        nan_result = {
            "xex_ofi_divergence": float("nan"),
            "xex_ofi_divergence_ma": float("nan"),
        }

        if idx < self.warmup_bars - 1:
            return nan_result
        if binance_ofi is None or coinbase_ofi is None:
            return nan_result

        div = float(binance_ofi[idx]) - float(coinbase_ofi[idx])

        # Rolling mean divergence
        start = idx - self._window + 1
        diffs = binance_ofi[start:idx + 1] - coinbase_ofi[start:idx + 1]
        div_ma = float(np.mean(diffs))

        return {
            "xex_ofi_divergence": div,
            "xex_ofi_divergence_ma": div_ma,
        }

    def output_names(self) -> list[str]:
        return ["xex_ofi_divergence", "xex_ofi_divergence_ma"]
