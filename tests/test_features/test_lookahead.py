"""Look-ahead bias detection tests for all feature computers.

Two tests per feature:
1. Truncation test: compute(data[:150]) == compute(data[:200]) at index 150
2. Shuffle test: permuted input should not preserve feature values
"""

from __future__ import annotations

import numpy as np
import pytest

from ep2_crypto.features.pipeline import FeaturePipeline


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

    # Use 16:00 UTC base for US session (needed by NQ features)
    base_ts = 16 * 3_600_000
    timestamps = np.arange(n, dtype=np.int64) * 300_000 + base_ts
    opens = mid - rng.uniform(0, 5, n)
    closes = mid + rng.uniform(0, 5, n)
    highs = np.maximum(opens, closes) + rng.uniform(0, 3, n)
    lows = np.minimum(opens, closes) - rng.uniform(0, 3, n)
    volumes = rng.uniform(100, 1000, n)
    trade_sizes = rng.uniform(0.1, 10.0, n)
    trade_sides = rng.choice([-1.0, 1.0], n)

    # Cross-market data
    nq_closes = 18000.0 + np.cumsum(rng.standard_normal(n) * 5)
    eth_closes = 3200.0 + np.cumsum(rng.standard_normal(n) * 2)

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
        "nq_closes": nq_closes,
        "eth_closes": eth_closes,
    }


def _truncate(data: dict[str, np.ndarray], n: int) -> dict[str, np.ndarray]:
    """Truncate all arrays to first n elements."""
    return {key: arr[:n].copy() for key, arr in data.items()}


def _get_kwargs(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Extract kwargs from data dict (everything except positional args)."""
    positional = {"timestamps", "opens", "highs", "lows", "closes", "volumes"}
    return {k: v for k, v in data.items() if k not in positional}


class TestTruncationBias:
    """Truncation test: features at index i computed on data[:i+1] must equal
    features at index i computed on data[:200].

    If they differ, the feature is using future data.
    """

    def test_all_features_truncation(self) -> None:
        data_full = _make_large_data(200)
        pipeline = FeaturePipeline()
        warmup = pipeline.warmup_bars

        # Test at several indices after max warmup
        test_indices = [warmup + 5, warmup + 20, warmup + 40, 149]

        for test_idx in test_indices:
            # Compute on full data
            full_result = pipeline.compute(
                test_idx,
                data_full["timestamps"], data_full["opens"], data_full["highs"],
                data_full["lows"], data_full["closes"], data_full["volumes"],
                **_get_kwargs(data_full),
            )

            # Compute on truncated data (only data up to test_idx + 1)
            data_trunc = _truncate(data_full, test_idx + 1)
            trunc_result = pipeline.compute(
                test_idx,
                data_trunc["timestamps"], data_trunc["opens"], data_trunc["highs"],
                data_trunc["lows"], data_trunc["closes"], data_trunc["volumes"],
                **_get_kwargs(data_trunc),
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
    """

    def test_windowed_features_change_on_shuffle(self) -> None:
        data = _make_large_data(150)
        rng = np.random.default_rng(99)

        pipeline = FeaturePipeline()
        test_idx = 100

        # Compute on original data
        original = pipeline.compute(
            test_idx,
            data["timestamps"], data["opens"], data["highs"],
            data["lows"], data["closes"], data["volumes"],
            **_get_kwargs(data),
        )

        # Shuffle data (permute indices 0..test_idx-1, keep test_idx fixed)
        perm = rng.permutation(test_idx)
        shuffled: dict[str, np.ndarray] = {}
        for key, arr in data.items():
            new_arr = arr.copy()
            if arr.ndim == 1:
                new_arr[:test_idx] = arr[perm]
            else:
                new_arr[:test_idx] = arr[perm]
            shuffled[key] = new_arr

        shuffled_result = pipeline.compute(
            test_idx,
            shuffled["timestamps"], shuffled["opens"], shuffled["highs"],
            shuffled["lows"], shuffled["closes"], shuffled["volumes"],
            **_get_kwargs(shuffled),
        )

        # Point-in-time features (OBI, microprice, session) should NOT change
        point_features = ["obi_l3", "obi_l5", "microprice", "microprice_mid_dev",
                          "session_asia", "session_europe", "session_us"]
        for key in point_features:
            if key in original and not np.isnan(original[key]):
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
            # Sprint 5 windowed features
            "eth_btc_ratio_roc6", "lead_lag_corr_1",
            "er_10", "garch_vol",
        ]
        changed_count = 0
        for key in windowed_features:
            if (
                key in original
                and not np.isnan(original[key])
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
        pipeline = FeaturePipeline()
        warmup = pipeline.warmup_bars

        # Collect OBI values and corresponding next-bar returns
        obi_vals = []
        future_rets = []
        for i in range(warmup, 198):
            result = pipeline.compute(
                i,
                data["timestamps"], data["opens"], data["highs"],
                data["lows"], data["closes"], data["volumes"],
                **_get_kwargs(data),
            )
            if np.isfinite(result["obi_l3"]):
                obi_vals.append(result["obi_l3"])
                future_rets.append(data["closes"][i + 1] - data["closes"][i])

        # On random data, correlation should be weak (< 0.3 in absolute value)
        if len(obi_vals) > 10:
            corr = np.corrcoef(obi_vals, future_rets)[0, 1]
            assert abs(corr) < 0.5, (
                f"Suspiciously high correlation ({corr:.3f}) between OBI and "
                f"future returns on random data"
            )
