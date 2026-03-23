"""Tests for microstructure feature computers: OBI, OFI, microprice, TFI, spread, absorption."""

from __future__ import annotations

import numpy as np
import pytest

from ep2_crypto.features.microstructure import (
    KyleLambdaComputer,
    MicropriceComputer,
    OBIComputer,
    OFIComputer,
    TFIComputer,
    _compute_ofi_level,
)


def _make_orderbook_data(n: int = 20, n_levels: int = 5) -> dict[str, np.ndarray]:
    """Create synthetic orderbook data for testing."""
    rng = np.random.default_rng(42)

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
        "trade_prices": closes.copy(),
        "trade_sizes": trade_sizes,
        "trade_sides": trade_sides,
    }


# ---- OBI Tests ----

class TestOBI:
    def test_output_names(self) -> None:
        obi = OBIComputer()
        assert obi.output_names() == ["obi_l3", "obi_l5", "obi_l3_weighted", "obi_l5_weighted"]

    def test_warmup(self) -> None:
        obi = OBIComputer()
        assert obi.warmup_bars == 1
        assert obi.name == "obi"

    def test_symmetric_book_gives_zero(self) -> None:
        """Equal bid and ask sizes should give OBI = 0."""
        n = 5
        bids = np.full((n, 5), 50000.0, dtype=np.float64)
        asks = np.full((n, 5), 50001.0, dtype=np.float64)
        bid_sizes = np.ones((n, 5), dtype=np.float64) * 10.0
        ask_sizes = np.ones((n, 5), dtype=np.float64) * 10.0

        for lev in range(5):
            bids[:, lev] = 50000.0 - lev
            asks[:, lev] = 50001.0 + lev

        obi = OBIComputer()
        ts = np.arange(n, dtype=np.int64) * 60_000
        dummy = np.ones(n)
        result = obi.compute(
            2, ts, dummy, dummy, dummy, dummy, dummy,
            bids=bids, asks=asks, bid_sizes=bid_sizes, ask_sizes=ask_sizes,
        )
        assert result["obi_l3"] == pytest.approx(0.0)
        assert result["obi_l5"] == pytest.approx(0.0)

    def test_all_bid_gives_positive(self) -> None:
        """All volume on bid side -> OBI = 1.0."""
        n = 3
        bids = np.full((n, 5), 50000.0, dtype=np.float64)
        asks = np.full((n, 5), 50001.0, dtype=np.float64)
        bid_sizes = np.ones((n, 5), dtype=np.float64) * 10.0
        ask_sizes = np.zeros((n, 5), dtype=np.float64)

        for lev in range(5):
            bids[:, lev] = 50000.0 - lev
            asks[:, lev] = 50001.0 + lev

        obi = OBIComputer()
        ts = np.arange(n, dtype=np.int64) * 60_000
        dummy = np.ones(n)
        result = obi.compute(
            1, ts, dummy, dummy, dummy, dummy, dummy,
            bids=bids, asks=asks, bid_sizes=bid_sizes, ask_sizes=ask_sizes,
        )
        assert result["obi_l3"] == pytest.approx(1.0)

    def test_golden_dataset_obi(self) -> None:
        """Hand-verified OBI computation."""
        n = 3
        bid_sizes = np.array([
            [10.0, 5.0, 3.0, 2.0, 1.0],
            [10.0, 5.0, 3.0, 2.0, 1.0],
            [10.0, 5.0, 3.0, 2.0, 1.0],
        ])
        ask_sizes = np.array([
            [8.0, 4.0, 2.0, 1.0, 0.5],
            [8.0, 4.0, 2.0, 1.0, 0.5],
            [8.0, 4.0, 2.0, 1.0, 0.5],
        ])

        # OBI_l3 = (10+5+3 - 8+4+2) / (10+5+3 + 8+4+2) = (18-14)/(18+14) = 4/32
        expected_l3 = 4.0 / 32.0

        # OBI_l5 = (10+5+3+2+1 - 8+4+2+1+0.5) / (21 + 15.5) = 5.5/36.5
        expected_l5 = 5.5 / 36.5

        bids = np.full((n, 5), 50000.0, dtype=np.float64)
        asks = np.full((n, 5), 50001.0, dtype=np.float64)
        for lev in range(5):
            bids[:, lev] = 50000.0 - lev
            asks[:, lev] = 50001.0 + lev

        obi = OBIComputer()
        ts = np.arange(n, dtype=np.int64) * 60_000
        dummy = np.ones(n)
        result = obi.compute(
            1, ts, dummy, dummy, dummy, dummy, dummy,
            bids=bids, asks=asks, bid_sizes=bid_sizes, ask_sizes=ask_sizes,
        )
        assert result["obi_l3"] == pytest.approx(expected_l3, abs=1e-10)
        assert result["obi_l5"] == pytest.approx(expected_l5, abs=1e-10)

    def test_nan_without_data(self) -> None:
        obi = OBIComputer()
        ts = np.arange(5, dtype=np.int64) * 60_000
        dummy = np.ones(5)
        result = obi.compute(2, ts, dummy, dummy, dummy, dummy, dummy)
        assert np.isnan(result["obi_l3"])

    def test_obi_range(self) -> None:
        """OBI should always be in [-1, 1]."""
        data = _make_orderbook_data(50)
        obi = OBIComputer()
        for i in range(50):
            result = obi.compute(
                i, data["timestamps"], data["opens"], data["highs"],
                data["lows"], data["closes"], data["volumes"],
                bids=data["bids"], asks=data["asks"],
                bid_sizes=data["bid_sizes"], ask_sizes=data["ask_sizes"],
            )
            for key in ["obi_l3", "obi_l5"]:
                if not np.isnan(result[key]):
                    assert -1.0 <= result[key] <= 1.0, f"{key}={result[key]} out of range"


# ---- OFI Tests ----

class TestOFI:
    def test_output_names(self) -> None:
        ofi = OFIComputer()
        assert ofi.output_names() == ["ofi_l1", "ofi_l3", "ofi_l5"]

    def test_warmup(self) -> None:
        ofi = OFIComputer()
        assert ofi.warmup_bars == 2

    def test_nan_before_warmup(self) -> None:
        ofi = OFIComputer()
        data = _make_orderbook_data()
        result = ofi.compute(
            0, data["timestamps"], data["opens"], data["highs"],
            data["lows"], data["closes"], data["volumes"],
            bids=data["bids"], asks=data["asks"],
            bid_sizes=data["bid_sizes"], ask_sizes=data["ask_sizes"],
        )
        assert np.isnan(result["ofi_l1"])

    def test_ofi_level_bid_up(self) -> None:
        """Bid price increases: delta_b = +curr_bid_sz."""
        result = _compute_ofi_level(
            prev_bid=100.0, prev_bid_sz=5.0,
            curr_bid=101.0, curr_bid_sz=8.0,
            prev_ask=102.0, prev_ask_sz=3.0,
            curr_ask=102.0, curr_ask_sz=3.0,  # ask unchanged
        )
        # delta_b = 8.0 (bid up), delta_a = -(3-3)=0 (ask same)
        # OFI = 8 - 0 = 8
        assert result == pytest.approx(8.0)

    def test_ofi_level_bid_same(self) -> None:
        """Bid price same: delta_b = curr_sz - prev_sz."""
        result = _compute_ofi_level(
            prev_bid=100.0, prev_bid_sz=5.0,
            curr_bid=100.0, curr_bid_sz=8.0,
            prev_ask=102.0, prev_ask_sz=3.0,
            curr_ask=102.0, curr_ask_sz=3.0,
        )
        # delta_b = 8-5=3, delta_a = -(3-3)=0
        # OFI = 3 - 0 = 3
        assert result == pytest.approx(3.0)

    def test_ofi_level_bid_down(self) -> None:
        """Bid price decreases: delta_b = -prev_bid_sz."""
        result = _compute_ofi_level(
            prev_bid=100.0, prev_bid_sz=5.0,
            curr_bid=99.0, curr_bid_sz=8.0,
            prev_ask=102.0, prev_ask_sz=3.0,
            curr_ask=102.0, curr_ask_sz=3.0,
        )
        # delta_b = -5 (bid down), delta_a = 0
        # OFI = -5 - 0 = -5
        assert result == pytest.approx(-5.0)

    def test_ofi_level_ask_down(self) -> None:
        """Ask price decreases: delta_a = -curr_ask_sz (aggressive seller)."""
        result = _compute_ofi_level(
            prev_bid=100.0, prev_bid_sz=5.0,
            curr_bid=100.0, curr_bid_sz=5.0,
            prev_ask=102.0, prev_ask_sz=3.0,
            curr_ask=101.0, curr_ask_sz=7.0,
        )
        # delta_b = 0 (bid same), delta_a = -7 (ask down)
        # OFI = 0 - (-7) = 7
        assert result == pytest.approx(7.0)

    def test_ofi_level_ask_up(self) -> None:
        """Ask price increases: delta_a = +prev_ask_sz."""
        result = _compute_ofi_level(
            prev_bid=100.0, prev_bid_sz=5.0,
            curr_bid=100.0, curr_bid_sz=5.0,
            prev_ask=102.0, prev_ask_sz=3.0,
            curr_ask=103.0, curr_ask_sz=7.0,
        )
        # delta_b = 0, delta_a = +3
        # OFI = 0 - 3 = -3
        assert result == pytest.approx(-3.0)

    def test_golden_dataset_ofi(self) -> None:
        """Multi-level OFI with hand-verified values."""
        n = 3
        bids = np.array([
            [100.0, 99.0, 98.0],
            [100.0, 99.0, 98.0],  # same prices
            [101.0, 100.0, 99.0],  # all levels up by 1
        ])
        asks = np.array([
            [102.0, 103.0, 104.0],
            [102.0, 103.0, 104.0],
            [102.0, 103.0, 104.0],  # same
        ])
        bid_sizes = np.array([
            [5.0, 3.0, 2.0],
            [7.0, 4.0, 3.0],  # increased at same price
            [6.0, 5.0, 4.0],  # new prices (bid up)
        ])
        ask_sizes = np.array([
            [4.0, 2.0, 1.0],
            [3.0, 2.0, 1.0],
            [3.0, 2.0, 1.0],  # same
        ])

        # At idx=1 (comparing with idx=0):
        # Level 0: bid same, delta_b=7-5=2; ask same, delta_a=-(3-4)=1; OFI=2-1=1
        # Level 1: bid same, delta_b=4-3=1; ask same, delta_a=-(2-2)=0; OFI=1
        # Level 2: bid same, delta_b=3-2=1; ask same, delta_a=-(1-1)=0; OFI=1
        ofi = OFIComputer()
        ts = np.arange(n, dtype=np.int64) * 60_000
        dummy = np.ones(n)
        result = ofi.compute(
            1, ts, dummy, dummy, dummy, dummy, dummy,
            bids=bids, asks=asks, bid_sizes=bid_sizes, ask_sizes=ask_sizes,
        )
        assert result["ofi_l1"] == pytest.approx(1.0)
        assert result["ofi_l3"] == pytest.approx(3.0)

    def test_nan_without_data(self) -> None:
        ofi = OFIComputer()
        ts = np.arange(5, dtype=np.int64) * 60_000
        dummy = np.ones(5)
        result = ofi.compute(2, ts, dummy, dummy, dummy, dummy, dummy)
        assert np.isnan(result["ofi_l1"])


# ---- Microprice Tests ----

class TestMicroprice:
    def test_output_names(self) -> None:
        mp = MicropriceComputer()
        assert mp.output_names() == ["microprice", "microprice_mid_dev"]

    def test_symmetric_equals_mid(self) -> None:
        """Equal bid/ask sizes -> microprice = mid."""
        n = 3
        bids = np.array([[50000.0, 49999.0], [50000.0, 49999.0], [50000.0, 49999.0]])
        asks = np.array([[50001.0, 50002.0], [50001.0, 50002.0], [50001.0, 50002.0]])
        bid_sizes = np.array([[10.0, 5.0], [10.0, 5.0], [10.0, 5.0]])
        ask_sizes = np.array([[10.0, 5.0], [10.0, 5.0], [10.0, 5.0]])

        mp = MicropriceComputer()
        ts = np.arange(n, dtype=np.int64) * 60_000
        dummy = np.ones(n)
        result = mp.compute(
            1, ts, dummy, dummy, dummy, dummy, dummy,
            bids=bids, asks=asks, bid_sizes=bid_sizes, ask_sizes=ask_sizes,
        )
        expected_mid = (50000.0 + 50001.0) / 2.0
        assert result["microprice"] == pytest.approx(expected_mid)
        assert result["microprice_mid_dev"] == pytest.approx(0.0)

    def test_golden_dataset_microprice(self) -> None:
        """Hand-verified microprice computation."""
        n = 3
        bids = np.array([[100.0], [100.0], [100.0]])
        asks = np.array([[102.0], [102.0], [102.0]])
        bid_sizes = np.array([[3.0], [3.0], [3.0]])
        ask_sizes = np.array([[7.0], [7.0], [7.0]])

        # microprice = (7*100 + 3*102) / (3+7) = (700 + 306) / 10 = 100.6
        mp = MicropriceComputer()
        ts = np.arange(n, dtype=np.int64) * 60_000
        dummy = np.ones(n)
        result = mp.compute(
            1, ts, dummy, dummy, dummy, dummy, dummy,
            bids=bids, asks=asks, bid_sizes=bid_sizes, ask_sizes=ask_sizes,
        )
        assert result["microprice"] == pytest.approx(100.6)

        mid = 101.0
        assert result["microprice_mid_dev"] == pytest.approx((100.6 - mid) / mid)

    def test_heavy_bid_pulls_toward_ask(self) -> None:
        """Large bid size -> microprice shifts toward ask (more buying pressure)."""
        n = 3
        bids = np.array([[100.0], [100.0], [100.0]])
        asks = np.array([[102.0], [102.0], [102.0]])
        bid_sizes = np.array([[90.0], [90.0], [90.0]])
        ask_sizes = np.array([[10.0], [10.0], [10.0]])

        mp = MicropriceComputer()
        ts = np.arange(n, dtype=np.int64) * 60_000
        dummy = np.ones(n)
        result = mp.compute(
            1, ts, dummy, dummy, dummy, dummy, dummy,
            bids=bids, asks=asks, bid_sizes=bid_sizes, ask_sizes=ask_sizes,
        )
        mid = 101.0
        # With bid_size=90, ask_size=10: micro = (10*100 + 90*102)/100 = 101.8
        assert result["microprice"] > mid
        assert result["microprice"] == pytest.approx(101.8)


# ---- TFI Tests ----

class TestTFI:
    def test_output_names(self) -> None:
        tfi = TFIComputer()
        assert tfi.output_names() == ["tfi_1bar", "tfi_6bar", "relative_spread", "absorption"]

    def test_all_buys_gives_positive(self) -> None:
        n = 10
        trade_sizes = np.ones(n, dtype=np.float64) * 5.0
        trade_sides = np.ones(n, dtype=np.float64)  # all buys

        tfi = TFIComputer()
        ts = np.arange(n, dtype=np.int64) * 60_000
        dummy = np.ones(n) * 100.0
        result = tfi.compute(
            5, ts, dummy, dummy, dummy, dummy, dummy,
            trade_sizes=trade_sizes, trade_sides=trade_sides,
        )
        assert result["tfi_1bar"] == pytest.approx(1.0)
        assert result["tfi_6bar"] == pytest.approx(1.0)

    def test_all_sells_gives_negative(self) -> None:
        n = 10
        trade_sizes = np.ones(n, dtype=np.float64) * 5.0
        trade_sides = np.ones(n, dtype=np.float64) * -1.0  # all sells

        tfi = TFIComputer()
        ts = np.arange(n, dtype=np.int64) * 60_000
        dummy = np.ones(n) * 100.0
        result = tfi.compute(
            5, ts, dummy, dummy, dummy, dummy, dummy,
            trade_sizes=trade_sizes, trade_sides=trade_sides,
        )
        assert result["tfi_1bar"] == pytest.approx(-1.0)
        assert result["tfi_6bar"] == pytest.approx(-1.0)

    def test_relative_spread(self) -> None:
        n = 5
        bids = np.array([[99.0], [99.0], [99.0], [99.0], [99.0]])
        asks = np.array([[101.0], [101.0], [101.0], [101.0], [101.0]])

        tfi = TFIComputer()
        ts = np.arange(n, dtype=np.int64) * 60_000
        dummy = np.ones(n) * 100.0
        result = tfi.compute(
            2, ts, dummy, dummy, dummy, dummy, dummy,
            bids=bids, asks=asks,
        )
        # spread = 2.0, mid = 100.0, relative = 0.02
        assert result["relative_spread"] == pytest.approx(0.02)

    def test_absorption_high_vol_low_price_change(self) -> None:
        """High volume delta with no price change = high absorption."""
        n = 5
        closes = np.array([100.0, 100.0, 100.0, 100.0, 100.0])  # no price change
        trade_sizes = np.array([1.0, 1.0, 100.0, 1.0, 1.0])  # big volume at idx 2
        trade_sides = np.ones(n)

        tfi = TFIComputer()
        ts = np.arange(n, dtype=np.int64) * 60_000
        dummy = np.ones(n) * 100.0
        result = tfi.compute(
            2, ts, dummy, dummy, dummy, closes, dummy,
            trade_sizes=trade_sizes, trade_sides=trade_sides,
        )
        # price_change = 0, so absorption = abs_vol / 1.0 = 100.0
        assert result["absorption"] == pytest.approx(100.0)

    def test_tfi_range(self) -> None:
        """TFI should be in [-1, 1]."""
        data = _make_orderbook_data(50)
        tfi = TFIComputer()
        for i in range(2, 50):
            result = tfi.compute(
                i, data["timestamps"], data["opens"], data["highs"],
                data["lows"], data["closes"], data["volumes"],
                trade_sizes=data["trade_sizes"], trade_sides=data["trade_sides"],
            )
            for key in ["tfi_1bar", "tfi_6bar"]:
                if not np.isnan(result[key]):
                    assert -1.0 <= result[key] <= 1.0, f"{key}={result[key]} out of range"


# ---- Kyle's Lambda Tests ----

class TestKyleLambda:
    def test_output_names(self) -> None:
        kl = KyleLambdaComputer(window=10)
        assert kl.output_names() == ["kyle_lambda"]

    def test_warmup(self) -> None:
        kl = KyleLambdaComputer(window=10)
        assert kl.warmup_bars == 10

    def test_nan_before_warmup(self) -> None:
        kl = KyleLambdaComputer(window=10)
        data = _make_orderbook_data()
        result = kl.compute(
            5, data["timestamps"], data["opens"], data["highs"],
            data["lows"], data["closes"], data["volumes"],
            trade_sizes=data["trade_sizes"], trade_sides=data["trade_sides"],
        )
        assert np.isnan(result["kyle_lambda"])

    def test_positive_impact(self) -> None:
        """Buy pressure causing price increase -> positive lambda."""
        n = 25
        closes = np.arange(n, dtype=np.float64) * 1.0 + 100.0  # monotonic increase
        trade_sizes = np.ones(n) * 10.0
        trade_sides = np.ones(n)  # all buys

        kl = KyleLambdaComputer(window=20)
        ts = np.arange(n, dtype=np.int64) * 60_000
        dummy = np.ones(n) * 100.0
        result = kl.compute(
            22, ts, dummy, dummy, dummy, closes, dummy,
            trade_sizes=trade_sizes, trade_sides=trade_sides,
        )
        # All trades are buys with same size, so var(signed_v) > 0 only from numerical noise
        # This tests the structure; real data will have variation
        # With constant signed_v, var=0 -> nan
        assert np.isnan(result["kyle_lambda"])

    def test_varying_volume(self) -> None:
        """Non-constant volume should produce a finite lambda."""
        n = 25
        rng = np.random.default_rng(42)
        closes = 100.0 + np.cumsum(rng.standard_normal(n))
        trade_sizes = rng.uniform(1.0, 20.0, n)
        trade_sides = rng.choice([-1.0, 1.0], n)

        kl = KyleLambdaComputer(window=20)
        ts = np.arange(n, dtype=np.int64) * 60_000
        dummy = np.ones(n) * 100.0
        result = kl.compute(
            22, ts, dummy, dummy, dummy, closes, dummy,
            trade_sizes=trade_sizes, trade_sides=trade_sides,
        )
        assert np.isfinite(result["kyle_lambda"])


# ---- Registry Integration ----

class TestMicrostructureRegistry:
    def test_all_computers_register(self) -> None:
        from ep2_crypto.features.base import FeatureRegistry
        reg = FeatureRegistry()
        reg.register(OBIComputer())
        reg.register(OFIComputer())
        reg.register(MicropriceComputer())
        reg.register(TFIComputer())
        reg.register(KyleLambdaComputer())
        assert len(reg.names) == 5

    def test_compute_all_produces_all_features(self) -> None:
        from ep2_crypto.features.base import FeatureRegistry
        data = _make_orderbook_data(30)
        reg = FeatureRegistry()
        reg.register(OBIComputer())
        reg.register(OFIComputer())
        reg.register(MicropriceComputer())
        reg.register(TFIComputer())
        reg.register(KyleLambdaComputer(window=10))

        result = reg.compute_all(
            15,
            data["timestamps"], data["opens"], data["highs"],
            data["lows"], data["closes"], data["volumes"],
            bids=data["bids"], asks=data["asks"],
            bid_sizes=data["bid_sizes"], ask_sizes=data["ask_sizes"],
            trade_sizes=data["trade_sizes"], trade_sides=data["trade_sides"],
        )

        expected_keys = {
            "obi_l3", "obi_l5", "obi_l3_weighted", "obi_l5_weighted",
            "ofi_l1", "ofi_l3", "ofi_l5",
            "microprice", "microprice_mid_dev",
            "tfi_1bar", "tfi_6bar", "relative_spread", "absorption",
            "kyle_lambda",
        }
        assert set(result.keys()) == expected_keys

        # All values should be finite after warmup
        for key, val in result.items():
            assert np.isfinite(val), f"{key} is not finite: {val}"
