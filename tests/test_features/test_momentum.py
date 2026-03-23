"""Tests for momentum feature computers: ROC, RSI, linreg slope, quantile rank."""

from __future__ import annotations

import numpy as np
import pytest

from ep2_crypto.features.momentum import (
    LinRegSlopeComputer,
    QuantileRankComputer,
    ROCComputer,
    RSIComputer,
)


def _make_price_data(n: int = 80) -> dict[str, np.ndarray]:
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


# ---- ROC Tests ----

class TestROC:
    def test_output_names(self) -> None:
        roc = ROCComputer()
        assert roc.output_names() == ["roc_1", "roc_3", "roc_6", "roc_12"]

    def test_warmup(self) -> None:
        roc = ROCComputer()
        assert roc.warmup_bars == 13
        assert roc.name == "roc"

    def test_nan_before_warmup(self) -> None:
        roc = ROCComputer()
        data = _make_price_data()
        result = roc.compute(
            5, data["timestamps"], data["opens"], data["highs"],
            data["lows"], data["closes"], data["volumes"],
        )
        assert np.isnan(result["roc_1"])

    def test_golden_dataset(self) -> None:
        """Hand-verified ROC computation."""
        n = 15
        closes = np.array([
            100.0, 102.0, 105.0, 103.0, 108.0, 110.0, 107.0,
            112.0, 115.0, 113.0, 118.0, 120.0, 117.0, 122.0, 125.0,
        ])

        roc = ROCComputer()
        ts = np.arange(n, dtype=np.int64) * 60_000
        dummy = np.ones(n) * 100.0

        # At idx=14: close=125
        # ROC_1 = (125 - closes[13]) / closes[13] = (125-122)/122
        # ROC_3 = (125 - closes[11]) / closes[11] = (125-120)/120
        # ROC_6 = (125 - closes[8]) / closes[8] = (125-115)/115
        # ROC_12 = (125 - closes[2]) / closes[2] = (125-105)/105
        result = roc.compute(14, ts, dummy, dummy, dummy, closes, dummy)
        assert result["roc_1"] == pytest.approx(3.0 / 122.0)
        assert result["roc_3"] == pytest.approx(5.0 / 120.0)
        assert result["roc_6"] == pytest.approx(10.0 / 115.0)
        assert result["roc_12"] == pytest.approx(20.0 / 105.0)

    def test_no_change_gives_zero(self) -> None:
        """Constant price -> ROC = 0."""
        n = 15
        closes = np.full(n, 100.0)

        roc = ROCComputer()
        ts = np.arange(n, dtype=np.int64) * 60_000
        dummy = np.ones(n) * 100.0
        result = roc.compute(14, ts, dummy, dummy, dummy, closes, dummy)
        assert result["roc_1"] == pytest.approx(0.0)
        assert result["roc_12"] == pytest.approx(0.0)

    def test_positive_trend(self) -> None:
        """Monotonically increasing price -> all ROC positive."""
        n = 15
        closes = np.arange(100.0, 115.0)

        roc = ROCComputer()
        ts = np.arange(n, dtype=np.int64) * 60_000
        dummy = np.ones(n) * 100.0
        result = roc.compute(14, ts, dummy, dummy, dummy, closes, dummy)
        for key in ["roc_1", "roc_3", "roc_6", "roc_12"]:
            assert result[key] > 0


# ---- RSI Tests ----

class TestRSI:
    def test_output_names(self) -> None:
        rsi = RSIComputer()
        assert rsi.output_names() == ["rsi"]

    def test_warmup(self) -> None:
        rsi = RSIComputer(window=14)
        assert rsi.warmup_bars == 15
        assert rsi.name == "rsi"

    def test_nan_before_warmup(self) -> None:
        rsi = RSIComputer()
        data = _make_price_data()
        result = rsi.compute(
            5, data["timestamps"], data["opens"], data["highs"],
            data["lows"], data["closes"], data["volumes"],
        )
        assert np.isnan(result["rsi"])

    def test_all_gains_gives_100(self) -> None:
        """Only gains -> RSI = 100."""
        n = 20
        closes = np.arange(100.0, 120.0)  # monotonic increase

        rsi = RSIComputer(window=14)
        ts = np.arange(n, dtype=np.int64) * 60_000
        dummy = np.ones(n) * 100.0
        result = rsi.compute(19, ts, dummy, dummy, dummy, closes, dummy)
        assert result["rsi"] == pytest.approx(100.0)

    def test_all_losses_gives_zero(self) -> None:
        """Only losses -> RSI = 0."""
        n = 20
        closes = np.arange(120.0, 100.0, -1.0)  # monotonic decrease

        rsi = RSIComputer(window=14)
        ts = np.arange(n, dtype=np.int64) * 60_000
        dummy = np.ones(n) * 100.0
        result = rsi.compute(19, ts, dummy, dummy, dummy, closes, dummy)
        assert result["rsi"] == pytest.approx(0.0)

    def test_bounded_0_100(self) -> None:
        """RSI should always be in [0, 100]."""
        data = _make_price_data()
        rsi = RSIComputer()
        for i in range(rsi.warmup_bars - 1, 80):
            result = rsi.compute(
                i, data["timestamps"], data["opens"], data["highs"],
                data["lows"], data["closes"], data["volumes"],
            )
            assert 0.0 <= result["rsi"] <= 100.0, f"RSI={result['rsi']} out of bounds at idx={i}"

    def test_equal_gains_losses_near_50(self) -> None:
        """Equal gains and losses -> RSI near 50."""
        n = 20
        # Alternating +1, -1
        closes = np.empty(n)
        closes[0] = 100.0
        for i in range(1, n):
            closes[i] = closes[i - 1] + (1.0 if i % 2 == 1 else -1.0)

        rsi = RSIComputer(window=14)
        ts = np.arange(n, dtype=np.int64) * 60_000
        dummy = np.ones(n) * 100.0
        result = rsi.compute(19, ts, dummy, dummy, dummy, closes, dummy)
        # Should be near 50 (gains and losses roughly equal)
        assert 40.0 <= result["rsi"] <= 60.0


# ---- LinReg Slope Tests ----

class TestLinRegSlope:
    def test_output_names(self) -> None:
        lrs = LinRegSlopeComputer()
        assert lrs.output_names() == ["linreg_slope"]

    def test_warmup(self) -> None:
        lrs = LinRegSlopeComputer(window=20)
        assert lrs.warmup_bars == 20
        assert lrs.name == "linreg_slope"

    def test_nan_before_warmup(self) -> None:
        lrs = LinRegSlopeComputer()
        data = _make_price_data()
        result = lrs.compute(
            10, data["timestamps"], data["opens"], data["highs"],
            data["lows"], data["closes"], data["volumes"],
        )
        assert np.isnan(result["linreg_slope"])

    def test_positive_trend(self) -> None:
        """Monotonically increasing price -> positive slope."""
        n = 25
        closes = np.arange(100.0, 125.0)

        lrs = LinRegSlopeComputer(window=20)
        ts = np.arange(n, dtype=np.int64) * 60_000
        dummy = np.ones(n) * 100.0
        result = lrs.compute(24, ts, dummy, dummy, dummy, closes, dummy)
        assert result["linreg_slope"] > 0

    def test_negative_trend(self) -> None:
        """Monotonically decreasing price -> negative slope."""
        n = 25
        closes = np.arange(125.0, 100.0, -1.0)

        lrs = LinRegSlopeComputer(window=20)
        ts = np.arange(n, dtype=np.int64) * 60_000
        dummy = np.ones(n) * 100.0
        result = lrs.compute(24, ts, dummy, dummy, dummy, closes, dummy)
        assert result["linreg_slope"] < 0

    def test_constant_price_zero_slope(self) -> None:
        """Constant price -> zero slope."""
        n = 25
        closes = np.full(n, 100.0)

        lrs = LinRegSlopeComputer(window=20)
        ts = np.arange(n, dtype=np.int64) * 60_000
        dummy = np.ones(n) * 100.0
        result = lrs.compute(24, ts, dummy, dummy, dummy, closes, dummy)
        assert result["linreg_slope"] == pytest.approx(0.0)

    def test_golden_dataset(self) -> None:
        """Verify linear prices give exact slope."""
        n = 25
        # Prices increase by exactly 2 per bar
        closes = 100.0 + np.arange(n) * 2.0

        lrs = LinRegSlopeComputer(window=20)
        ts = np.arange(n, dtype=np.int64) * 60_000
        dummy = np.ones(n) * 100.0
        result = lrs.compute(24, ts, dummy, dummy, dummy, closes, dummy)

        # Raw slope = 2.0 per bar, mean price of window = 100 + (5+24)/2 * 2 = 129.0
        # Normalized = 2.0 / 129.0
        mean_price = float(np.mean(closes[5:25]))
        assert result["linreg_slope"] == pytest.approx(2.0 / mean_price, rel=1e-10)


# ---- Quantile Rank Tests ----

class TestQuantileRank:
    def test_output_names(self) -> None:
        qr = QuantileRankComputer()
        assert qr.output_names() == ["quantile_rank"]

    def test_warmup(self) -> None:
        qr = QuantileRankComputer(window=60)
        assert qr.warmup_bars == 60
        assert qr.name == "quantile_rank"

    def test_nan_before_warmup(self) -> None:
        qr = QuantileRankComputer()
        data = _make_price_data()
        result = qr.compute(
            10, data["timestamps"], data["opens"], data["highs"],
            data["lows"], data["closes"], data["volumes"],
        )
        assert np.isnan(result["quantile_rank"])

    def test_highest_gives_one(self) -> None:
        """Highest price in window -> rank = 1.0."""
        n = 15
        closes = np.arange(1.0, 16.0)  # monotonic, current is highest

        qr = QuantileRankComputer(window=10)
        ts = np.arange(n, dtype=np.int64) * 60_000
        dummy = np.ones(n) * 100.0
        result = qr.compute(14, ts, dummy, dummy, dummy, closes, dummy)
        assert result["quantile_rank"] == pytest.approx(1.0)

    def test_lowest_gives_zero(self) -> None:
        """Lowest price in window -> rank = 0.0."""
        n = 15
        closes = np.arange(15.0, 0.0, -1.0)  # monotonic decrease, current is lowest

        qr = QuantileRankComputer(window=10)
        ts = np.arange(n, dtype=np.int64) * 60_000
        dummy = np.ones(n) * 100.0
        result = qr.compute(14, ts, dummy, dummy, dummy, closes, dummy)
        assert result["quantile_rank"] == pytest.approx(0.0)

    def test_bounded_0_1(self) -> None:
        """Quantile rank should be in [0, 1]."""
        data = _make_price_data()
        qr = QuantileRankComputer(window=60)
        for i in range(qr.warmup_bars - 1, 80):
            result = qr.compute(
                i, data["timestamps"], data["opens"], data["highs"],
                data["lows"], data["closes"], data["volumes"],
            )
            assert 0.0 <= result["quantile_rank"] <= 1.0

    def test_golden_dataset(self) -> None:
        """Hand-verified quantile rank."""
        n = 12
        closes = np.array([5.0, 3.0, 8.0, 1.0, 7.0, 2.0, 9.0, 4.0, 6.0, 10.0, 11.0, 7.5])

        # Window=10, idx=11: window = closes[2:12] = [8,1,7,2,9,4,6,10,11,7.5]
        # current = 7.5, count below: 1,2,4,6,7 = 5 values
        # rank = 5/9
        qr = QuantileRankComputer(window=10)
        ts = np.arange(n, dtype=np.int64) * 60_000
        dummy = np.ones(n) * 100.0
        result = qr.compute(11, ts, dummy, dummy, dummy, closes, dummy)
        assert result["quantile_rank"] == pytest.approx(5.0 / 9.0)


# ---- Registry Integration ----

class TestMomentumRegistry:
    def test_all_computers_register(self) -> None:
        from ep2_crypto.features.base import FeatureRegistry
        reg = FeatureRegistry()
        reg.register(ROCComputer())
        reg.register(RSIComputer())
        reg.register(LinRegSlopeComputer())
        reg.register(QuantileRankComputer())
        assert len(reg.names) == 4

    def test_compute_all_produces_all_features(self) -> None:
        from ep2_crypto.features.base import FeatureRegistry
        data = _make_price_data(80)
        reg = FeatureRegistry()
        reg.register(ROCComputer())
        reg.register(RSIComputer())
        reg.register(LinRegSlopeComputer())
        reg.register(QuantileRankComputer())

        result = reg.compute_all(
            70,
            data["timestamps"], data["opens"], data["highs"],
            data["lows"], data["closes"], data["volumes"],
        )

        expected_keys = {
            "roc_1", "roc_3", "roc_6", "roc_12",
            "rsi",
            "linreg_slope",
            "quantile_rank",
        }
        assert set(result.keys()) == expected_keys

        for key, val in result.items():
            assert np.isfinite(val), f"{key} is not finite: {val}"
