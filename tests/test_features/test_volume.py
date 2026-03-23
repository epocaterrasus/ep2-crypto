"""Tests for volume feature computers: delta, VWAP deviation, rate of change."""

from __future__ import annotations

import numpy as np
import pytest

from ep2_crypto.features.volume import (
    VolumeDeltaComputer,
    VolumeROCComputer,
    VWAPComputer,
)


def _make_volume_data(n: int = 30) -> dict[str, np.ndarray]:
    """Create synthetic OHLCV + trade data for testing."""
    rng = np.random.default_rng(42)

    mid = 50000.0 + np.cumsum(rng.standard_normal(n) * 10)
    opens = mid - rng.uniform(0, 5, n)
    closes = mid + rng.uniform(0, 5, n)
    highs = np.maximum(opens, closes) + rng.uniform(0, 3, n)
    lows = np.minimum(opens, closes) - rng.uniform(0, 3, n)
    volumes = rng.uniform(100, 1000, n)

    trade_sizes = rng.uniform(0.1, 10.0, n)
    trade_sides = rng.choice([-1.0, 1.0], n)

    return {
        "timestamps": np.arange(n, dtype=np.int64) * 60_000,
        "opens": opens,
        "highs": highs,
        "lows": lows,
        "closes": closes,
        "volumes": volumes,
        "trade_sizes": trade_sizes,
        "trade_sides": trade_sides,
    }


# ---- Volume Delta Tests ----

class TestVolumeDelta:
    def test_output_names(self) -> None:
        vd = VolumeDeltaComputer()
        assert vd.output_names() == [
            "vol_delta_1bar", "vol_delta_5bar", "vol_delta_1bar_raw", "vol_delta_5bar_raw",
        ]

    def test_warmup(self) -> None:
        vd = VolumeDeltaComputer()
        assert vd.warmup_bars == 5
        assert vd.name == "volume_delta"

    def test_nan_before_warmup(self) -> None:
        vd = VolumeDeltaComputer()
        data = _make_volume_data()
        result = vd.compute(
            2, data["timestamps"], data["opens"], data["highs"],
            data["lows"], data["closes"], data["volumes"],
            trade_sizes=data["trade_sizes"], trade_sides=data["trade_sides"],
        )
        assert np.isnan(result["vol_delta_1bar"])
        assert np.isnan(result["vol_delta_5bar"])

    def test_nan_without_trade_data(self) -> None:
        vd = VolumeDeltaComputer()
        data = _make_volume_data()
        result = vd.compute(
            10, data["timestamps"], data["opens"], data["highs"],
            data["lows"], data["closes"], data["volumes"],
        )
        assert np.isnan(result["vol_delta_1bar"])

    def test_all_buys_gives_positive(self) -> None:
        """All buy trades -> delta = 1.0."""
        n = 10
        trade_sizes = np.ones(n) * 5.0
        trade_sides = np.ones(n)  # all buys

        vd = VolumeDeltaComputer()
        ts = np.arange(n, dtype=np.int64) * 60_000
        dummy = np.ones(n) * 100.0
        result = vd.compute(
            6, ts, dummy, dummy, dummy, dummy, dummy,
            trade_sizes=trade_sizes, trade_sides=trade_sides,
        )
        assert result["vol_delta_1bar"] == pytest.approx(1.0)
        assert result["vol_delta_5bar"] == pytest.approx(1.0)
        assert result["vol_delta_1bar_raw"] == pytest.approx(5.0)
        assert result["vol_delta_5bar_raw"] == pytest.approx(25.0)

    def test_all_sells_gives_negative(self) -> None:
        """All sell trades -> delta = -1.0."""
        n = 10
        trade_sizes = np.ones(n) * 3.0
        trade_sides = -np.ones(n)

        vd = VolumeDeltaComputer()
        ts = np.arange(n, dtype=np.int64) * 60_000
        dummy = np.ones(n) * 100.0
        result = vd.compute(
            6, ts, dummy, dummy, dummy, dummy, dummy,
            trade_sizes=trade_sizes, trade_sides=trade_sides,
        )
        assert result["vol_delta_1bar"] == pytest.approx(-1.0)
        assert result["vol_delta_5bar"] == pytest.approx(-1.0)

    def test_golden_dataset_mixed(self) -> None:
        """Hand-verified volume delta with mixed buy/sell."""
        n = 10
        trade_sizes = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        trade_sides = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0])

        vd = VolumeDeltaComputer()
        ts = np.arange(n, dtype=np.int64) * 60_000
        dummy = np.ones(n) * 100.0
        result = vd.compute(
            6, ts, dummy, dummy, dummy, dummy, dummy,
            trade_sizes=trade_sizes, trade_sides=trade_sides,
        )
        # idx=6: trade_sizes[6]=7, trade_sides[6]=1 -> signed=7, total=7
        assert result["vol_delta_1bar"] == pytest.approx(1.0)
        assert result["vol_delta_1bar_raw"] == pytest.approx(7.0)

        # 5-bar: idx 2..6 -> sizes=[3,4,5,6,7], sides=[1,-1,1,-1,1]
        # signed = 3*1 + 4*(-1) + 5*1 + 6*(-1) + 7*1 = 3-4+5-6+7 = 5
        # total = 3+4+5+6+7 = 25
        assert result["vol_delta_5bar"] == pytest.approx(5.0 / 25.0)
        assert result["vol_delta_5bar_raw"] == pytest.approx(5.0)

    def test_delta_range(self) -> None:
        """Normalized delta should be in [-1, 1]."""
        data = _make_volume_data(50)
        vd = VolumeDeltaComputer()
        for i in range(vd.warmup_bars - 1, 50):
            result = vd.compute(
                i, data["timestamps"], data["opens"], data["highs"],
                data["lows"], data["closes"], data["volumes"],
                trade_sizes=data["trade_sizes"], trade_sides=data["trade_sides"],
            )
            for key in ["vol_delta_1bar", "vol_delta_5bar"]:
                if not np.isnan(result[key]):
                    assert -1.0 <= result[key] <= 1.0, f"{key}={result[key]} out of range"


# ---- VWAP Tests ----

class TestVWAP:
    def test_output_names(self) -> None:
        vwap = VWAPComputer()
        assert vwap.output_names() == ["vwap", "vwap_deviation"]

    def test_warmup(self) -> None:
        vwap = VWAPComputer(window=12)
        assert vwap.warmup_bars == 12
        assert vwap.name == "vwap"

    def test_nan_before_warmup(self) -> None:
        vwap = VWAPComputer(window=12)
        data = _make_volume_data()
        result = vwap.compute(
            5, data["timestamps"], data["opens"], data["highs"],
            data["lows"], data["closes"], data["volumes"],
        )
        assert np.isnan(result["vwap"])
        assert np.isnan(result["vwap_deviation"])

    def test_golden_dataset_vwap(self) -> None:
        """Hand-verified VWAP computation with small window."""
        n = 5
        highs = np.array([102.0, 104.0, 103.0, 105.0, 106.0])
        lows = np.array([98.0, 100.0, 99.0, 101.0, 102.0])
        closes = np.array([100.0, 102.0, 101.0, 103.0, 104.0])
        volumes = np.array([10.0, 20.0, 15.0, 25.0, 30.0])

        # window=3, idx=4: bars 2,3,4
        # typical = (H+L+C)/3 = [(103+99+101)/3, (105+101+103)/3, (106+102+104)/3]
        #         = [101.0, 103.0, 104.0]
        # VWAP = (101*15 + 103*25 + 104*30) / (15+25+30) = (1515+2575+3120)/70 = 7210/70
        expected_vwap = 7210.0 / 70.0
        expected_dev = (104.0 - expected_vwap) / expected_vwap

        vwap = VWAPComputer(window=3)
        ts = np.arange(n, dtype=np.int64) * 60_000
        opens = np.ones(n) * 100.0
        result = vwap.compute(4, ts, opens, highs, lows, closes, volumes)
        assert result["vwap"] == pytest.approx(expected_vwap)
        assert result["vwap_deviation"] == pytest.approx(expected_dev)

    def test_constant_price_zero_deviation(self) -> None:
        """Constant price -> VWAP = price, deviation = 0."""
        n = 15
        price = 100.0
        highs = np.full(n, price)
        lows = np.full(n, price)
        closes = np.full(n, price)
        volumes = np.ones(n) * 50.0

        vwap = VWAPComputer(window=12)
        ts = np.arange(n, dtype=np.int64) * 60_000
        opens = np.full(n, price)
        result = vwap.compute(13, ts, opens, highs, lows, closes, volumes)
        assert result["vwap"] == pytest.approx(price)
        assert result["vwap_deviation"] == pytest.approx(0.0)

    def test_vwap_strictly_positive(self) -> None:
        """VWAP should be positive for positive price data."""
        data = _make_volume_data(50)
        vwap = VWAPComputer(window=12)
        for i in range(vwap.warmup_bars - 1, 50):
            result = vwap.compute(
                i, data["timestamps"], data["opens"], data["highs"],
                data["lows"], data["closes"], data["volumes"],
            )
            if not np.isnan(result["vwap"]):
                assert result["vwap"] > 0, f"VWAP should be positive, got {result['vwap']}"

    def test_zero_volume_gives_nan(self) -> None:
        """Zero volume window -> NaN."""
        n = 15
        highs = np.ones(n) * 100.0
        lows = np.ones(n) * 100.0
        closes = np.ones(n) * 100.0
        volumes = np.zeros(n)

        vwap = VWAPComputer(window=12)
        ts = np.arange(n, dtype=np.int64) * 60_000
        opens = np.ones(n) * 100.0
        result = vwap.compute(13, ts, opens, highs, lows, closes, volumes)
        assert np.isnan(result["vwap"])


# ---- Volume ROC Tests ----

class TestVolumeROC:
    def test_output_names(self) -> None:
        vroc = VolumeROCComputer()
        assert vroc.output_names() == ["vol_roc_1", "vol_roc_3", "vol_roc_6"]

    def test_warmup(self) -> None:
        vroc = VolumeROCComputer()
        assert vroc.warmup_bars == 7
        assert vroc.name == "volume_roc"

    def test_nan_before_warmup(self) -> None:
        vroc = VolumeROCComputer()
        data = _make_volume_data()
        result = vroc.compute(
            3, data["timestamps"], data["opens"], data["highs"],
            data["lows"], data["closes"], data["volumes"],
        )
        assert np.isnan(result["vol_roc_1"])
        assert np.isnan(result["vol_roc_3"])
        assert np.isnan(result["vol_roc_6"])

    def test_golden_dataset_roc(self) -> None:
        """Hand-verified volume ROC."""
        n = 10
        volumes = np.array([100.0, 200.0, 150.0, 300.0, 250.0, 400.0, 500.0, 350.0, 600.0, 450.0])

        vroc = VolumeROCComputer()
        ts = np.arange(n, dtype=np.int64) * 60_000
        dummy = np.ones(n) * 100.0

        # At idx=7: vol=350
        # ROC_1 = (350-500)/500 = -0.3
        # ROC_3 = (350-vol[4])/vol[4] = (350-250)/250 = 0.4
        # ROC_6 = (350-vol[1])/vol[1] = (350-200)/200 = 0.75
        result = vroc.compute(7, ts, dummy, dummy, dummy, dummy, volumes)
        assert result["vol_roc_1"] == pytest.approx(-0.3)
        assert result["vol_roc_3"] == pytest.approx(0.4)
        assert result["vol_roc_6"] == pytest.approx(0.75)

    def test_constant_volume_zero_roc(self) -> None:
        """Constant volume -> ROC = 0."""
        n = 10
        volumes = np.ones(n) * 500.0

        vroc = VolumeROCComputer()
        ts = np.arange(n, dtype=np.int64) * 60_000
        dummy = np.ones(n) * 100.0
        result = vroc.compute(8, ts, dummy, dummy, dummy, dummy, volumes)
        assert result["vol_roc_1"] == pytest.approx(0.0)
        assert result["vol_roc_3"] == pytest.approx(0.0)
        assert result["vol_roc_6"] == pytest.approx(0.0)

    def test_doubling_volume(self) -> None:
        """Volume doubles -> ROC_1 = 1.0."""
        n = 10
        volumes = np.array([100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 200.0, 100.0, 100.0])

        vroc = VolumeROCComputer()
        ts = np.arange(n, dtype=np.int64) * 60_000
        dummy = np.ones(n) * 100.0
        result = vroc.compute(7, ts, dummy, dummy, dummy, dummy, volumes)
        assert result["vol_roc_1"] == pytest.approx(1.0)  # (200-100)/100


# ---- Registry Integration ----

class TestVolumeRegistry:
    def test_all_computers_register(self) -> None:
        from ep2_crypto.features.base import FeatureRegistry
        reg = FeatureRegistry()
        reg.register(VolumeDeltaComputer())
        reg.register(VWAPComputer())
        reg.register(VolumeROCComputer())
        assert len(reg.names) == 3

    def test_compute_all_produces_all_features(self) -> None:
        from ep2_crypto.features.base import FeatureRegistry
        data = _make_volume_data(30)
        reg = FeatureRegistry()
        reg.register(VolumeDeltaComputer())
        reg.register(VWAPComputer(window=12))
        reg.register(VolumeROCComputer())

        result = reg.compute_all(
            20,
            data["timestamps"], data["opens"], data["highs"],
            data["lows"], data["closes"], data["volumes"],
            trade_sizes=data["trade_sizes"], trade_sides=data["trade_sides"],
        )

        expected_keys = {
            "vol_delta_1bar", "vol_delta_5bar", "vol_delta_1bar_raw", "vol_delta_5bar_raw",
            "vwap", "vwap_deviation",
            "vol_roc_1", "vol_roc_3", "vol_roc_6",
        }
        assert set(result.keys()) == expected_keys

        for key, val in result.items():
            assert np.isfinite(val), f"{key} is not finite: {val}"
