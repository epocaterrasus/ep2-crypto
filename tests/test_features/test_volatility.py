"""Tests for volatility feature computers: realized, Parkinson, EWMA, vol-of-vol."""

from __future__ import annotations

import math

import numpy as np
import pytest

from ep2_crypto.features.volatility import (
    EWMAVolComputer,
    ParkinsonVolComputer,
    RealizedVolComputer,
    VolOfVolComputer,
)


def _make_price_data(n: int = 50) -> dict[str, np.ndarray]:
    """Create synthetic OHLCV data for testing."""
    rng = np.random.default_rng(42)
    mid = 50000.0 + np.cumsum(rng.standard_normal(n) * 50)
    opens = mid - rng.uniform(0, 20, n)
    closes = mid + rng.uniform(0, 20, n)
    highs = np.maximum(opens, closes) + rng.uniform(5, 30, n)
    lows = np.minimum(opens, closes) - rng.uniform(5, 30, n)
    volumes = rng.uniform(100, 1000, n)

    return {
        "timestamps": np.arange(n, dtype=np.int64) * 60_000,
        "opens": opens,
        "highs": highs,
        "lows": lows,
        "closes": closes,
        "volumes": volumes,
    }


# ---- Realized Volatility Tests ----


class TestRealizedVol:
    def test_output_names(self) -> None:
        rv = RealizedVolComputer()
        assert rv.output_names() == ["realized_vol_short", "realized_vol_long"]

    def test_warmup(self) -> None:
        rv = RealizedVolComputer(short_window=6, long_window=12)
        assert rv.warmup_bars == 13
        assert rv.name == "realized_vol"

    def test_nan_before_warmup(self) -> None:
        rv = RealizedVolComputer()
        data = _make_price_data()
        result = rv.compute(
            5,
            data["timestamps"],
            data["opens"],
            data["highs"],
            data["lows"],
            data["closes"],
            data["volumes"],
        )
        assert np.isnan(result["realized_vol_short"])
        assert np.isnan(result["realized_vol_long"])

    def test_golden_dataset(self) -> None:
        """Hand-verified realized volatility."""
        closes = np.array(
            [
                100.0,
                101.0,
                99.0,
                102.0,
                98.0,
                103.0,
                100.0,
                101.0,
                99.0,
                102.0,
                98.0,
                103.0,
                100.0,
                101.0,
            ]
        )
        n = len(closes)

        rv = RealizedVolComputer(short_window=6, long_window=12)
        ts = np.arange(n, dtype=np.int64) * 60_000
        dummy = np.ones(n) * 100.0
        result = rv.compute(12, ts, dummy, dummy, dummy, closes, dummy)

        # Short window (6 bars): log returns from idx 6..12
        short_rets = np.log(closes[7:13] / closes[6:12])
        expected_short = float(np.std(short_rets, ddof=1))
        assert result["realized_vol_short"] == pytest.approx(expected_short, rel=1e-10)

        # Long window (12 bars): log returns from idx 0..12
        long_rets = np.log(closes[1:13] / closes[0:12])
        expected_long = float(np.std(long_rets, ddof=1))
        assert result["realized_vol_long"] == pytest.approx(expected_long, rel=1e-10)

    def test_strictly_positive(self) -> None:
        """Volatility should be strictly positive for varying prices."""
        data = _make_price_data()
        rv = RealizedVolComputer()
        for i in range(rv.warmup_bars - 1, 50):
            result = rv.compute(
                i,
                data["timestamps"],
                data["opens"],
                data["highs"],
                data["lows"],
                data["closes"],
                data["volumes"],
            )
            assert result["realized_vol_short"] > 0
            assert result["realized_vol_long"] > 0

    def test_constant_price_near_zero(self) -> None:
        """Constant prices -> near-zero volatility."""
        n = 20
        closes = np.full(n, 100.0)
        rv = RealizedVolComputer(short_window=6, long_window=12)
        ts = np.arange(n, dtype=np.int64) * 60_000
        dummy = np.ones(n) * 100.0

        # Can't compute log(100/100)=0 for all returns, std=0
        # But with ddof=1 this would also be 0
        result = rv.compute(15, ts, dummy, dummy, dummy, closes, dummy)
        assert result["realized_vol_short"] == pytest.approx(0.0)


# ---- Parkinson Volatility Tests ----


class TestParkinsonVol:
    def test_output_names(self) -> None:
        pv = ParkinsonVolComputer()
        assert pv.output_names() == ["parkinson_vol_short", "parkinson_vol_long"]

    def test_warmup(self) -> None:
        pv = ParkinsonVolComputer(short_window=6, long_window=12)
        assert pv.warmup_bars == 12
        assert pv.name == "parkinson_vol"

    def test_nan_before_warmup(self) -> None:
        pv = ParkinsonVolComputer()
        data = _make_price_data()
        result = pv.compute(
            5,
            data["timestamps"],
            data["opens"],
            data["highs"],
            data["lows"],
            data["closes"],
            data["volumes"],
        )
        assert np.isnan(result["parkinson_vol_short"])

    def test_golden_dataset(self) -> None:
        """Hand-verified Parkinson volatility."""
        n = 8
        highs = np.array([102.0, 104.0, 103.0, 105.0, 101.0, 106.0, 104.0, 103.0])
        lows = np.array([98.0, 100.0, 99.0, 101.0, 97.0, 102.0, 100.0, 99.0])

        # Window=6, idx=7: bars 2..7
        # log(H/L) for bars 2..7:
        h = highs[2:8]
        lo = lows[2:8]
        log_hl = np.log(h / lo)
        sigma_sq = float(np.sum(log_hl**2)) / (4.0 * 6 * math.log(2))
        expected = math.sqrt(sigma_sq)

        pv = ParkinsonVolComputer(short_window=6, long_window=6)
        ts = np.arange(n, dtype=np.int64) * 60_000
        dummy = np.ones(n) * 100.0
        closes = np.ones(n) * 100.0
        result = pv.compute(7, ts, dummy, highs, lows, closes, dummy)
        assert result["parkinson_vol_short"] == pytest.approx(expected, rel=1e-10)

    def test_strictly_positive(self) -> None:
        """Parkinson vol should be strictly positive when high != low."""
        data = _make_price_data()
        pv = ParkinsonVolComputer()
        for i in range(pv.warmup_bars - 1, 50):
            result = pv.compute(
                i,
                data["timestamps"],
                data["opens"],
                data["highs"],
                data["lows"],
                data["closes"],
                data["volumes"],
            )
            assert result["parkinson_vol_short"] > 0
            assert result["parkinson_vol_long"] > 0

    def test_wider_range_higher_vol(self) -> None:
        """Wider high-low range -> higher Parkinson vol."""
        n = 15
        # Narrow range
        highs_narrow = np.full(n, 101.0)
        lows_narrow = np.full(n, 99.0)
        # Wide range
        highs_wide = np.full(n, 110.0)
        lows_wide = np.full(n, 90.0)

        pv = ParkinsonVolComputer(short_window=6, long_window=12)
        ts = np.arange(n, dtype=np.int64) * 60_000
        dummy = np.ones(n) * 100.0

        narrow = pv.compute(13, ts, dummy, highs_narrow, lows_narrow, dummy, dummy)
        wide = pv.compute(13, ts, dummy, highs_wide, lows_wide, dummy, dummy)
        assert wide["parkinson_vol_short"] > narrow["parkinson_vol_short"]


# ---- EWMA Volatility Tests ----


class TestEWMAVol:
    def test_output_names(self) -> None:
        ev = EWMAVolComputer()
        assert ev.output_names() == ["ewma_vol"]

    def test_warmup(self) -> None:
        ev = EWMAVolComputer()
        assert ev.warmup_bars == 20
        assert ev.name == "ewma_vol"

    def test_nan_before_warmup(self) -> None:
        ev = EWMAVolComputer()
        data = _make_price_data()
        result = ev.compute(
            10,
            data["timestamps"],
            data["opens"],
            data["highs"],
            data["lows"],
            data["closes"],
            data["volumes"],
        )
        assert np.isnan(result["ewma_vol"])

    def test_strictly_positive(self) -> None:
        """EWMA vol should be positive for varying prices."""
        data = _make_price_data()
        ev = EWMAVolComputer()
        for i in range(ev.warmup_bars - 1, 50):
            result = ev.compute(
                i,
                data["timestamps"],
                data["opens"],
                data["highs"],
                data["lows"],
                data["closes"],
                data["volumes"],
            )
            assert result["ewma_vol"] > 0

    def test_high_decay_slower_reaction(self) -> None:
        """Higher decay (0.99) -> slower reaction to new data than lower decay (0.8)."""
        n = 30
        # Prices stable then shock
        closes = np.full(n, 100.0)
        closes[25:] = 110.0  # sudden jump

        ts = np.arange(n, dtype=np.int64) * 60_000
        dummy = np.ones(n) * 100.0

        ev_fast = EWMAVolComputer(decay=0.8)
        ev_slow = EWMAVolComputer(decay=0.99)

        fast = ev_fast.compute(28, ts, dummy, dummy, dummy, closes, dummy)
        slow = ev_slow.compute(28, ts, dummy, dummy, dummy, closes, dummy)

        # Fast decay should have higher vol right after shock
        assert fast["ewma_vol"] > slow["ewma_vol"]

    def test_golden_dataset(self) -> None:
        """Manually verify EWMA for a simple case."""
        closes = np.array([100.0, 101.0, 99.0, 102.0, 98.0] + [100.0] * 20)
        n = len(closes)

        ev = EWMAVolComputer(decay=0.94)
        ts = np.arange(n, dtype=np.int64) * 60_000
        dummy = np.ones(n) * 100.0

        # Compute manually
        log_rets = np.log(closes[1:] / closes[:-1])
        var = log_rets[0] ** 2
        for r in log_rets[1:23]:  # up to idx=23
            var = 0.94 * var + 0.06 * r**2
        expected = math.sqrt(var)

        result = ev.compute(23, ts, dummy, dummy, dummy, closes, dummy)
        assert result["ewma_vol"] == pytest.approx(expected, rel=1e-10)


# ---- Vol-of-Vol Tests ----


class TestVolOfVol:
    def test_output_names(self) -> None:
        vov = VolOfVolComputer()
        assert vov.output_names() == ["vol_of_vol"]

    def test_warmup(self) -> None:
        vov = VolOfVolComputer(inner_window=6, outer_window=12)
        assert vov.warmup_bars == 19
        assert vov.name == "vol_of_vol"

    def test_nan_before_warmup(self) -> None:
        vov = VolOfVolComputer()
        data = _make_price_data()
        result = vov.compute(
            10,
            data["timestamps"],
            data["opens"],
            data["highs"],
            data["lows"],
            data["closes"],
            data["volumes"],
        )
        assert np.isnan(result["vol_of_vol"])

    def test_non_negative(self) -> None:
        """Vol-of-vol should be non-negative."""
        data = _make_price_data()
        vov = VolOfVolComputer()
        for i in range(vov.warmup_bars - 1, 50):
            result = vov.compute(
                i,
                data["timestamps"],
                data["opens"],
                data["highs"],
                data["lows"],
                data["closes"],
                data["volumes"],
            )
            assert result["vol_of_vol"] >= 0

    def test_constant_vol_zero_vov(self) -> None:
        """When realized vol is constant, vol-of-vol should be ~0."""
        # Prices with constant percentage changes -> constant realized vol
        n = 50
        closes = np.empty(n)
        closes[0] = 100.0
        # Alternating +1%, -1% gives constant magnitude returns
        for i in range(1, n):
            if i % 2 == 1:
                closes[i] = closes[i - 1] * 1.01
            else:
                closes[i] = closes[i - 1] * 0.99

        vov = VolOfVolComputer(inner_window=6, outer_window=12)
        ts = np.arange(n, dtype=np.int64) * 60_000
        dummy = np.ones(n) * 100.0

        result = vov.compute(40, ts, dummy, dummy, dummy, closes, dummy)
        # With alternating returns, each inner window has the same vol
        assert result["vol_of_vol"] == pytest.approx(0.0, abs=1e-10)


# ---- Registry Integration ----


class TestVolatilityRegistry:
    def test_all_computers_register(self) -> None:
        from ep2_crypto.features.base import FeatureRegistry

        reg = FeatureRegistry()
        reg.register(RealizedVolComputer())
        reg.register(ParkinsonVolComputer())
        reg.register(EWMAVolComputer())
        reg.register(VolOfVolComputer())
        assert len(reg.names) == 4

    def test_compute_all_produces_all_features(self) -> None:
        from ep2_crypto.features.base import FeatureRegistry

        data = _make_price_data(50)
        reg = FeatureRegistry()
        reg.register(RealizedVolComputer())
        reg.register(ParkinsonVolComputer())
        reg.register(EWMAVolComputer())
        reg.register(VolOfVolComputer())

        result = reg.compute_all(
            40,
            data["timestamps"],
            data["opens"],
            data["highs"],
            data["lows"],
            data["closes"],
            data["volumes"],
        )

        expected_keys = {
            "realized_vol_short",
            "realized_vol_long",
            "parkinson_vol_short",
            "parkinson_vol_long",
            "ewma_vol",
            "vol_of_vol",
        }
        assert set(result.keys()) == expected_keys

        for key, val in result.items():
            assert np.isfinite(val), f"{key} is not finite: {val}"
