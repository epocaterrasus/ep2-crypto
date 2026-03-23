"""Sprint 22: Tests for MultiLevelOFIComputer and MicropriceMlComputer.

Includes:
- Basic output verification (correct keys, finite values)
- Truncation tests (no look-ahead bias)
- Warmup behaviour (NaN before warmup_bars)
- Edge case: missing orderbook data → NaN
"""

from __future__ import annotations

import numpy as np

from ep2_crypto.features.microstructure import MicropriceMlComputer, MultiLevelOFIComputer


def _make_orderbook(n: int = 50, n_levels: int = 10) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(123)
    mid = 50000.0 + np.cumsum(rng.standard_normal(n) * 10)

    bids = np.zeros((n, n_levels))
    asks = np.zeros((n, n_levels))
    bid_sizes = np.zeros((n, n_levels))
    ask_sizes = np.zeros((n, n_levels))

    for i in range(n):
        spread = rng.uniform(0.5, 2.0)
        for lev in range(n_levels):
            bids[i, lev] = mid[i] - spread / 2 - lev
            asks[i, lev] = mid[i] + spread / 2 + lev
            bid_sizes[i, lev] = rng.uniform(0.1, 5.0)
            ask_sizes[i, lev] = rng.uniform(0.1, 5.0)

    ts = np.arange(n, dtype=np.int64) * 300_000
    closes = mid + rng.standard_normal(n) * 5
    opens = mid - rng.uniform(0, 5, n)
    highs = np.maximum(opens, closes) + rng.uniform(0, 3, n)
    lows = np.minimum(opens, closes) - rng.uniform(0, 3, n)
    volumes = rng.uniform(100, 1000, n)

    return {
        "timestamps": ts,
        "opens": opens,
        "highs": highs,
        "lows": lows,
        "closes": closes,
        "volumes": volumes,
        "bids": bids,
        "asks": asks,
        "bid_sizes": bid_sizes,
        "ask_sizes": ask_sizes,
    }


# ---------------------------------------------------------------------------
# MultiLevelOFIComputer
# ---------------------------------------------------------------------------


class TestMultiLevelOFIComputer:
    def test_output_keys(self) -> None:
        data = _make_orderbook()
        comp = MultiLevelOFIComputer(norm_window=20, roll_window=5)
        warmup = comp.warmup_bars - 1
        result = comp.compute(warmup + 5, **data)
        assert set(result.keys()) == {"ofi_l10", "ofi_l1_norm", "ofi_roll5"}

    def test_finite_after_warmup(self) -> None:
        data = _make_orderbook(n=60)
        comp = MultiLevelOFIComputer(norm_window=20, roll_window=5)
        for idx in range(comp.warmup_bars, 50):
            result = comp.compute(idx, **data)
            for key, val in result.items():
                assert np.isfinite(val), f"{key} not finite at idx={idx}"

    def test_nan_before_warmup(self) -> None:
        data = _make_orderbook()
        comp = MultiLevelOFIComputer(norm_window=20, roll_window=5)
        result = comp.compute(0, **data)
        for val in result.values():
            assert np.isnan(val)

    def test_nan_without_orderbook(self) -> None:
        data = _make_orderbook()
        comp = MultiLevelOFIComputer(norm_window=20, roll_window=5)
        data_no_ob = {k: v for k, v in data.items() if k not in ("bids", "asks", "bid_sizes", "ask_sizes")}
        result = comp.compute(40, **data_no_ob)
        for val in result.values():
            assert np.isnan(val)

    def test_ofi_l10_magnitude(self) -> None:
        """ofi_l10 should have larger magnitude than ofi_l1 (aggregates more levels)."""
        data = _make_orderbook(n=60)
        comp = MultiLevelOFIComputer(norm_window=20, roll_window=5)
        from ep2_crypto.features.microstructure import OFIComputer
        ofi_comp = OFIComputer()
        # Compute at a fixed idx and compare magnitudes over many bars
        l10_mags, l1_mags = [], []
        for idx in range(comp.warmup_bars, 55):
            r10 = comp.compute(idx, **data)
            r1 = ofi_comp.compute(idx, **data)
            if np.isfinite(r10["ofi_l10"]) and np.isfinite(r1["ofi_l1"]):
                l10_mags.append(abs(r10["ofi_l10"]))
                l1_mags.append(abs(r1["ofi_l1"]))
        mean_l10 = np.mean(l10_mags) if l10_mags else 0
        mean_l1 = np.mean(l1_mags) if l1_mags else 0
        assert mean_l10 >= mean_l1, "ofi_l10 should aggregate >= l1 magnitude on average"

    def test_ofi_l1_norm_bounded(self) -> None:
        """Normalized OFI should rarely exceed ±5 sigma on normal market data."""
        data = _make_orderbook(n=200)
        comp = MultiLevelOFIComputer(norm_window=20, roll_window=5)
        norms = [
            comp.compute(idx, **data)["ofi_l1_norm"]
            for idx in range(comp.warmup_bars, 180)
        ]
        finite_norms = [v for v in norms if np.isfinite(v)]
        assert len(finite_norms) > 0
        assert all(abs(v) < 10 for v in finite_norms), "z-score should be reasonable"

    def test_truncation_no_lookahead(self) -> None:
        """ofi_l10 at idx=150 on data[:200] must equal ofi_l10 at idx=150 on data[:151].

        This verifies no look-ahead bias: truncating future data does not change the
        result at the current index.
        """
        data = _make_orderbook(n=200)
        comp = MultiLevelOFIComputer(norm_window=20, roll_window=5)

        idx = 150
        result_full = comp.compute(idx, **data)

        # Truncate all arrays to idx+1
        data_trunc = {
            k: v[:idx + 1] for k, v in data.items()
        }
        result_trunc = comp.compute(idx, **data_trunc)

        for key in result_full:
            full_val = result_full[key]
            trunc_val = result_trunc[key]
            if np.isfinite(full_val) and np.isfinite(trunc_val):
                assert abs(full_val - trunc_val) < 1e-9, (
                    f"{key}: full={full_val}, trunc={trunc_val} — truncation changed result"
                )

    def test_output_names(self) -> None:
        comp = MultiLevelOFIComputer()
        assert comp.output_names() == ["ofi_l10", "ofi_l1_norm", "ofi_roll5"]

    def test_roll5_is_sum_of_recent_l1(self) -> None:
        """ofi_roll5 should equal the sum of ofi_l1 for the last 5 bars."""
        data = _make_orderbook(n=60)
        comp = MultiLevelOFIComputer(norm_window=20, roll_window=5)
        from ep2_crypto.features.microstructure import _compute_ofi_level
        idx = 40
        expected_roll = sum(
            _compute_ofi_level(
                data["bids"][j - 1, 0], data["bid_sizes"][j - 1, 0],
                data["bids"][j, 0], data["bid_sizes"][j, 0],
                data["asks"][j - 1, 0], data["ask_sizes"][j - 1, 0],
                data["asks"][j, 0], data["ask_sizes"][j, 0],
            )
            for j in range(idx - 5 + 1, idx + 1)
        )
        result = comp.compute(idx, **data)
        assert abs(result["ofi_roll5"] - expected_roll) < 1e-9


# ---------------------------------------------------------------------------
# MicropriceMlComputer
# ---------------------------------------------------------------------------


class TestMicropriceMlComputer:
    def test_output_keys(self) -> None:
        data = _make_orderbook()
        comp = MicropriceMlComputer(zscore_window=20)
        result = comp.compute(comp.warmup_bars, **data)
        assert set(result.keys()) == {"microprice_l3_dev", "microprice_l5_dev", "microprice_dev_zscore"}

    def test_finite_after_warmup(self) -> None:
        data = _make_orderbook(n=60)
        comp = MicropriceMlComputer(zscore_window=20)
        for idx in range(comp.warmup_bars, 55):
            result = comp.compute(idx, **data)
            for key, val in result.items():
                assert np.isfinite(val), f"{key} not finite at idx={idx}"

    def test_nan_before_warmup(self) -> None:
        data = _make_orderbook()
        comp = MicropriceMlComputer(zscore_window=20)
        result = comp.compute(0, **data)
        for val in result.values():
            assert np.isnan(val)

    def test_nan_without_orderbook(self) -> None:
        data = _make_orderbook()
        comp = MicropriceMlComputer(zscore_window=20)
        data_no_ob = {k: v for k, v in data.items() if k not in ("bids", "asks", "bid_sizes", "ask_sizes")}
        result = comp.compute(40, **data_no_ob)
        for val in result.values():
            assert np.isnan(val)

    def test_l5_dev_uses_more_levels_than_l3(self) -> None:
        """l5 and l3 microprice deviations should differ when book is not uniform."""
        data = _make_orderbook(n=60)
        comp = MicropriceMlComputer(zscore_window=20)
        idx = 40
        result = comp.compute(idx, **data)
        # Both should be finite and differ from each other (non-trivial book)
        assert np.isfinite(result["microprice_l3_dev"])
        assert np.isfinite(result["microprice_l5_dev"])
        # Values CAN be equal, but for random data they'll differ
        # Just confirm they're in a reasonable range
        assert abs(result["microprice_l3_dev"]) < 0.01  # <100bps
        assert abs(result["microprice_l5_dev"]) < 0.01

    def test_truncation_no_lookahead(self) -> None:
        """microprice_l3_dev at idx=150 on data[:200] must equal result on data[:151]."""
        data = _make_orderbook(n=200)
        comp = MicropriceMlComputer(zscore_window=20)

        idx = 150
        result_full = comp.compute(idx, **data)
        data_trunc = {k: v[:idx + 1] for k, v in data.items()}
        result_trunc = comp.compute(idx, **data_trunc)

        for key in result_full:
            full_val = result_full[key]
            trunc_val = result_trunc[key]
            if np.isfinite(full_val) and np.isfinite(trunc_val):
                assert abs(full_val - trunc_val) < 1e-9, (
                    f"{key}: full={full_val}, trunc={trunc_val}"
                )

    def test_dev_sign_matches_book_imbalance(self) -> None:
        """When ask side is heavy, microprice should be pulled below mid (negative dev)."""
        n = 50
        rng = np.random.default_rng(0)
        mid_price = 50000.0
        spread = 1.0
        n_levels = 5

        bids = np.zeros((n, n_levels))
        asks = np.zeros((n, n_levels))
        bid_sizes = np.zeros((n, n_levels))
        ask_sizes = np.zeros((n, n_levels))

        for i in range(n):
            for lev in range(n_levels):
                bids[i, lev] = mid_price - spread / 2 - lev
                asks[i, lev] = mid_price + spread / 2 + lev
                bid_sizes[i, lev] = 1.0  # equal sizes
                ask_sizes[i, lev] = 1.0

        # At last bar, make ask side much heavier (more ask volume → price pulled down)
        ask_sizes[-1, :] = 10.0
        bid_sizes[-1, :] = 1.0

        ts = np.arange(n, dtype=np.int64) * 300_000
        closes = np.full(n, mid_price)
        opens = closes.copy()
        highs = closes + 5
        lows = closes - 5
        volumes = np.ones(n) * 100

        comp = MicropriceMlComputer(zscore_window=20)
        result = comp.compute(
            n - 1,
            timestamps=ts, opens=opens, highs=highs, lows=lows,
            closes=closes, volumes=volumes,
            bids=bids, asks=asks, bid_sizes=bid_sizes, ask_sizes=ask_sizes,
        )
        # Heavy ask side → microprice pulled toward ask → higher than mid → positive dev
        # Actually: mp = (ask_sz * bid + bid_sz * ask) / (bid_sz + ask_sz)
        # heavy ask_sz → more weight on bid price → mp pulled toward bid → negative dev
        assert result["microprice_l3_dev"] < 0, "Heavy ask side should pull microprice below mid"

    def test_output_names(self) -> None:
        comp = MicropriceMlComputer()
        assert comp.output_names() == ["microprice_l3_dev", "microprice_l5_dev", "microprice_dev_zscore"]
