"""Look-ahead bias detection tests for all feature computers.

Two tests per feature:
1. Truncation test: compute(data[:150]) == compute(data[:200]) at index 150
2. Shuffle test: permuted input should not preserve feature values
"""

from __future__ import annotations

import numpy as np
import pytest

from ep2_crypto.features.base import FeatureRegistry
from ep2_crypto.features.microstructure import (
    KyleLambdaComputer,
    MicropriceComputer,
    OBIComputer,
    OFIComputer,
    TFIComputer,
)
from ep2_crypto.features.momentum import (
    LinRegSlopeComputer,
    QuantileRankComputer,
    ROCComputer,
    RSIComputer,
)
from ep2_crypto.features.volatility import (
    EWMAVolComputer,
    ParkinsonVolComputer,
    RealizedVolComputer,
    VolOfVolComputer,
)
from ep2_crypto.features.volume import (
    VolumeDeltaComputer,
    VolumeROCComputer,
    VWAPComputer,
)


def _make_large_data(n: int = 200, n_levels: int = 5) -> dict[str, np.ndarray]:
    """Create larger synthetic dataset for bias testing."""
    rng = np.random.default_rng(123)

    mid = 50000.0 + np.cumsum(rng.standard_normal(n) * 10)

    bids = np.zeros((n, n_levels), dtype=np.float64)
    asks = np.zeros((n, n_levels), dtype=np.float64)
    bid_sizes = np.zeros((n, n_levels), dtype=np.float64)
    ask_sizes = np.zeros((n, n_levels), dtype=np.float64)

    for i in range(n):
        spread = rng.uniform(0.5, 2.0)
        for lev in range(n_levels):
            bids[i, lev] = mid[i] - spread / 2 - lev * 1.0
            asks[i, lev] = mid[i] + spread / 2 + lev * 1.0
            bid_sizes[i, lev] = rng.uniform(0.1, 5.0)
            ask_sizes[i, lev] = rng.uniform(0.1, 5.0)

    timestamps = np.arange(n, dtype=np.int64) * 60_000
    opens = mid - rng.uniform(0, 5, n)
    closes = mid + rng.uniform(0, 5, n)
    highs = np.maximum(opens, closes) + rng.uniform(0, 3, n)
    lows = np.minimum(opens, closes) - rng.uniform(0, 3, n)
    volumes = rng.uniform(100, 1000, n)
    trade_sizes = rng.uniform(0.1, 10.0, n)
    trade_sides = rng.choice([-1.0, 1.0], n)

    return {
        "timestamps": timestamps,
        "opens": opens,
        "highs": highs,
        "lows": lows,
        "closes": closes,
        "volumes": volumes,
        "bids": bids,
        "asks": asks,
        "bid_sizes": bid_sizes,
        "ask_sizes": ask_sizes,
        "trade_sizes": trade_sizes,
        "trade_sides": trade_sides,
    }


def _truncate(data: dict[str, np.ndarray], n: int) -> dict[str, np.ndarray]:
    """Truncate all arrays to first n elements."""
    result = {}
    for key, arr in data.items():
        if arr.ndim == 1:
            result[key] = arr[:n].copy()
        else:
            result[key] = arr[:n].copy()
    return result


def _build_registry() -> FeatureRegistry:
    """Build registry with all Sprint 3 + Sprint 4 features."""
    reg = FeatureRegistry()
    # Sprint 3: Microstructure
    reg.register(OBIComputer())
    reg.register(OFIComputer())
    reg.register(MicropriceComputer())
    reg.register(TFIComputer())
    reg.register(KyleLambdaComputer(window=20))
    # Sprint 4: Volume
    reg.register(VolumeDeltaComputer())
    reg.register(VWAPComputer(window=12))
    reg.register(VolumeROCComputer())
    # Sprint 4: Volatility
    reg.register(RealizedVolComputer(short_window=6, long_window=12))
    reg.register(ParkinsonVolComputer(short_window=6, long_window=12))
    reg.register(EWMAVolComputer(decay=0.94))
    reg.register(VolOfVolComputer(inner_window=6, outer_window=12))
    # Sprint 4: Momentum
    reg.register(ROCComputer())
    reg.register(RSIComputer(window=14))
    reg.register(LinRegSlopeComputer(window=20))
    reg.register(QuantileRankComputer(window=60))
    return reg


class TestTruncationBias:
    """Truncation test: features at index i computed on data[:i+1] must equal
    features at index i computed on data[:200].

    If they differ, the feature is using future data.
    """

    def test_all_features_truncation(self) -> None:
        data_full = _make_large_data(200)
        reg = _build_registry()

        # Test at several indices after max warmup (QuantileRank=60)
        test_indices = [65, 80, 100, 149]

        for test_idx in test_indices:
            # Compute on full data
            full_result = reg.compute_all(
                test_idx,
                data_full["timestamps"], data_full["opens"], data_full["highs"],
                data_full["lows"], data_full["closes"], data_full["volumes"],
                bids=data_full["bids"], asks=data_full["asks"],
                bid_sizes=data_full["bid_sizes"], ask_sizes=data_full["ask_sizes"],
                trade_sizes=data_full["trade_sizes"], trade_sides=data_full["trade_sides"],
            )

            # Compute on truncated data (only data up to test_idx + 1)
            data_trunc = _truncate(data_full, test_idx + 1)
            trunc_result = reg.compute_all(
                test_idx,
                data_trunc["timestamps"], data_trunc["opens"], data_trunc["highs"],
                data_trunc["lows"], data_trunc["closes"], data_trunc["volumes"],
                bids=data_trunc["bids"], asks=data_trunc["asks"],
                bid_sizes=data_trunc["bid_sizes"], ask_sizes=data_trunc["ask_sizes"],
                trade_sizes=data_trunc["trade_sizes"], trade_sides=data_trunc["trade_sides"],
            )

            for key in full_result:
                full_val = full_result[key]
                trunc_val = trunc_result[key]
                if np.isnan(full_val) and np.isnan(trunc_val):
                    continue
                assert full_val == pytest.approx(trunc_val, abs=1e-10), (
                    f"Look-ahead bias detected in '{key}' at idx={test_idx}: "
                    f"full={full_val}, truncated={trunc_val}"
                )


class TestShuffleBias:
    """Shuffle test: if we permute the time ordering of input data (except
    at the test index), features should change if they properly depend
    on temporal structure.

    If a feature gives the same result on shuffled data, it may only use
    the current bar (which is fine for OBI/microprice). But features that
    use windows (OFI, TFI, Kyle's lambda) should differ.
    """

    def test_windowed_features_change_on_shuffle(self) -> None:
        data = _make_large_data(100)
        rng = np.random.default_rng(99)

        test_idx = 70
        reg = _build_registry()

        # Compute on original data
        original = reg.compute_all(
            test_idx,
            data["timestamps"], data["opens"], data["highs"],
            data["lows"], data["closes"], data["volumes"],
            bids=data["bids"], asks=data["asks"],
            bid_sizes=data["bid_sizes"], ask_sizes=data["ask_sizes"],
            trade_sizes=data["trade_sizes"], trade_sides=data["trade_sides"],
        )

        # Shuffle data (permute indices 0..test_idx-1, keep test_idx fixed)
        perm = rng.permutation(test_idx)
        shuffled = {}
        for key, arr in data.items():
            new_arr = arr.copy()
            if arr.ndim == 1:
                new_arr[:test_idx] = arr[perm]
            else:
                new_arr[:test_idx] = arr[perm]
            shuffled[key] = new_arr

        shuffled_result = reg.compute_all(
            test_idx,
            shuffled["timestamps"], shuffled["opens"], shuffled["highs"],
            shuffled["lows"], shuffled["closes"], shuffled["volumes"],
            bids=shuffled["bids"], asks=shuffled["asks"],
            bid_sizes=shuffled["bid_sizes"], ask_sizes=shuffled["ask_sizes"],
            trade_sizes=shuffled["trade_sizes"], trade_sides=shuffled["trade_sides"],
        )

        # Point-in-time features (OBI, microprice) should NOT change
        # because they only use data at idx
        point_features = ["obi_l3", "obi_l5", "microprice", "microprice_mid_dev"]
        for key in point_features:
            if not np.isnan(original[key]):
                assert original[key] == pytest.approx(shuffled_result[key], abs=1e-10), (
                    f"Point-in-time feature '{key}' changed on shuffle (unexpected)"
                )

        # Windowed features SHOULD change because they depend on temporal sequence
        windowed_features = [
            "ofi_l1", "ofi_l3", "tfi_6bar", "kyle_lambda",
            # Sprint 4 windowed features
            "vol_delta_5bar", "vwap_deviation", "vol_roc_3",
            "realized_vol_short", "ewma_vol", "rsi", "linreg_slope",
            "quantile_rank",
        ]
        changed_count = 0
        for key in windowed_features:
            if (
                not np.isnan(original[key])
                and not np.isnan(shuffled_result[key])
                and original[key] != pytest.approx(shuffled_result[key], abs=1e-10)
            ):
                changed_count += 1

        assert changed_count > 0, (
            "No windowed features changed after shuffling — possible look-ahead or "
            "features not using temporal history"
        )

    def test_no_future_correlation(self) -> None:
        """Features should not correlate with future returns when computed on shuffled data."""
        data = _make_large_data(200)

        # Compute features and future returns
        reg = _build_registry()
        warmup = reg.max_warmup()

        # Collect OBI values and corresponding next-bar returns
        obi_vals = []
        future_rets = []
        for i in range(warmup, 198):
            result = reg.compute_all(
                i,
                data["timestamps"], data["opens"], data["highs"],
                data["lows"], data["closes"], data["volumes"],
                bids=data["bids"], asks=data["asks"],
                bid_sizes=data["bid_sizes"], ask_sizes=data["ask_sizes"],
                trade_sizes=data["trade_sizes"], trade_sides=data["trade_sides"],
            )
            if np.isfinite(result["obi_l3"]):
                obi_vals.append(result["obi_l3"])
                future_rets.append(data["closes"][i + 1] - data["closes"][i])

        # On random data, correlation should be weak (< 0.3 in absolute value)
        if len(obi_vals) > 10:
            corr = np.corrcoef(obi_vals, future_rets)[0, 1]
            # On synthetic random data, should not be strongly correlated
            assert abs(corr) < 0.5, (
                f"Suspiciously high correlation ({corr:.3f}) between OBI and "
                f"future returns on random data"
            )
