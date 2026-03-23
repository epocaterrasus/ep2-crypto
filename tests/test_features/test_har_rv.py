"""Tests for HAR-RV multi-scale realized volatility computer.

Golden dataset: hand-computed RV values for known price series.
Look-ahead bias: truncation test at multiple indices.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from ep2_crypto.features.volatility import HARRVComputer


def _make_prices(
    n: int = 400, seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    ts = np.arange(n, dtype=np.int64) * 300_000
    closes = 30_000.0 + np.cumsum(rng.normal(0, 20, n))
    closes = np.maximum(closes, 1.0)
    opens = closes - np.abs(rng.normal(0, 5, n))
    highs = closes + np.abs(rng.normal(0, 5, n))
    lows = opens - np.abs(rng.normal(0, 2, n))
    vols = np.abs(rng.normal(100, 20, n))
    return ts, opens, highs, lows, closes, vols


class TestHARRVComputer:
    def test_output_names(self) -> None:
        c = HARRVComputer()
        assert set(c.output_names()) == {
            "rv_1h",
            "rv_4h",
            "rv_1d",
            "har_ratio_1h_1d",
            "har_ratio_4h_1d",
        }

    def test_warmup_is_288_plus_one(self) -> None:
        c = HARRVComputer()
        assert c.warmup_bars == 289

    def test_warmup_returns_nan(self) -> None:
        c = HARRVComputer()
        n = 400
        ts, opens, highs, lows, closes, vols = _make_prices(n)
        result = c.compute(200, ts, opens, highs, lows, closes, vols)
        assert math.isnan(result["rv_1h"])  # idx=200 < warmup=288

    def test_returns_valid_after_warmup(self) -> None:
        c = HARRVComputer()
        n = 400
        ts, opens, highs, lows, closes, vols = _make_prices(n)
        result = c.compute(350, ts, opens, highs, lows, closes, vols)
        assert not math.isnan(result["rv_1h"])
        assert not math.isnan(result["rv_4h"])
        assert not math.isnan(result["rv_1d"])

    def test_golden_rv_1h_hand_computed(self) -> None:
        """RV_1h = sum of last 12 squared log returns."""
        c = HARRVComputer()
        n = 400
        ts, opens, highs, lows, closes, vols = _make_prices(n)
        idx = 350
        # Hand-compute: sum of squared log returns over last 12 bars
        window = 12
        log_rets = np.log(closes[idx - window + 1 : idx + 1] / closes[idx - window : idx])
        expected_rv_1h = float(np.sum(log_rets**2))
        result = c.compute(idx, ts, opens, highs, lows, closes, vols)
        assert result["rv_1h"] == pytest.approx(expected_rv_1h, rel=1e-10)

    def test_golden_rv_4h_hand_computed(self) -> None:
        """RV_4h = sum of last 48 squared log returns."""
        c = HARRVComputer()
        n = 400
        ts, opens, highs, lows, closes, vols = _make_prices(n)
        idx = 350
        window = 48
        log_rets = np.log(closes[idx - window + 1 : idx + 1] / closes[idx - window : idx])
        expected = float(np.sum(log_rets**2))
        result = c.compute(idx, ts, opens, highs, lows, closes, vols)
        assert result["rv_4h"] == pytest.approx(expected, rel=1e-10)

    def test_golden_rv_1d_hand_computed(self) -> None:
        """RV_1d = sum of last 288 squared log returns."""
        c = HARRVComputer()
        n = 400
        ts, opens, highs, lows, closes, vols = _make_prices(n)
        idx = 350
        window = 288
        log_rets = np.log(closes[idx - window + 1 : idx + 1] / closes[idx - window : idx])
        expected = float(np.sum(log_rets**2))
        result = c.compute(idx, ts, opens, highs, lows, closes, vols)
        assert result["rv_1d"] == pytest.approx(expected, rel=1e-10)

    def test_rv_ordering_1h_lt_4h_lt_1d(self) -> None:
        """Longer windows accumulate more variance: RV_1h ≤ RV_4h ≤ RV_1d."""
        c = HARRVComputer()
        n = 400
        ts, opens, highs, lows, closes, vols = _make_prices(n)
        result = c.compute(350, ts, opens, highs, lows, closes, vols)
        assert result["rv_1h"] <= result["rv_4h"] + 1e-12
        assert result["rv_4h"] <= result["rv_1d"] + 1e-12

    def test_ratio_1h_1d_is_rv1h_over_rv1d(self) -> None:
        """Ratio = RV_1h / RV_1d exactly."""
        c = HARRVComputer()
        n = 400
        ts, opens, highs, lows, closes, vols = _make_prices(n)
        result = c.compute(350, ts, opens, highs, lows, closes, vols)
        expected = result["rv_1h"] / result["rv_1d"]
        assert result["har_ratio_1h_1d"] == pytest.approx(expected, rel=1e-10)

    def test_ratio_4h_1d_is_rv4h_over_rv1d(self) -> None:
        """Ratio = RV_4h / RV_1d exactly."""
        c = HARRVComputer()
        n = 400
        ts, opens, highs, lows, closes, vols = _make_prices(n)
        result = c.compute(350, ts, opens, highs, lows, closes, vols)
        expected = result["rv_4h"] / result["rv_1d"]
        assert result["har_ratio_4h_1d"] == pytest.approx(expected, rel=1e-10)

    def test_constant_prices_give_zero_rv(self) -> None:
        """No price movement → RV = 0."""
        c = HARRVComputer()
        n = 400
        ts = np.arange(n, dtype=np.int64) * 300_000
        closes = np.ones(n) * 30_000.0
        opens = closes.copy()
        highs = closes.copy()
        lows = closes.copy()
        vols = np.ones(n)
        result = c.compute(350, ts, opens, highs, lows, closes, vols)
        assert result["rv_1h"] == pytest.approx(0.0, abs=1e-20)
        assert result["rv_1d"] == pytest.approx(0.0, abs=1e-20)

    def test_no_look_ahead_bias_rv_1h(self) -> None:
        """RV_1h at idx=350 is the same whether full or truncated array."""
        c = HARRVComputer()
        n = 400
        ts, opens, highs, lows, closes, vols = _make_prices(n)
        idx = 350
        full = c.compute(idx, ts, opens, highs, lows, closes, vols)
        trunc = c.compute(
            idx,
            ts[: idx + 1],
            opens[: idx + 1],
            highs[: idx + 1],
            lows[: idx + 1],
            closes[: idx + 1],
            vols[: idx + 1],
        )
        assert full["rv_1h"] == pytest.approx(trunc["rv_1h"], rel=1e-10)

    def test_no_look_ahead_bias_rv_1d(self) -> None:
        """RV_1d at idx=350 same for full vs truncated."""
        c = HARRVComputer()
        n = 400
        ts, opens, highs, lows, closes, vols = _make_prices(n)
        idx = 350
        full = c.compute(idx, ts, opens, highs, lows, closes, vols)
        trunc = c.compute(
            idx,
            ts[: idx + 1],
            opens[: idx + 1],
            highs[: idx + 1],
            lows[: idx + 1],
            closes[: idx + 1],
            vols[: idx + 1],
        )
        assert full["rv_1d"] == pytest.approx(trunc["rv_1d"], rel=1e-10)

    def test_ratios_positive(self) -> None:
        """All ratios must be positive."""
        c = HARRVComputer()
        n = 400
        ts, opens, highs, lows, closes, vols = _make_prices(n)
        result = c.compute(350, ts, opens, highs, lows, closes, vols)
        assert result["har_ratio_1h_1d"] > 0
        assert result["har_ratio_4h_1d"] > 0

    def test_high_vol_period_gives_higher_rv(self) -> None:
        """Doubling price moves roughly quadruples RV."""
        c = HARRVComputer()
        n = 400
        ts = np.arange(n, dtype=np.int64) * 300_000
        opens = highs = lows = np.ones(n) * 100.0
        rng = np.random.default_rng(99)

        # Low vol
        closes_low = 100.0 + np.cumsum(rng.normal(0, 0.1, n))
        closes_low = np.maximum(closes_low, 1.0)
        r_low = c.compute(350, ts, opens, highs, lows, closes_low, np.ones(n))

        # High vol (10× the step size)
        rng2 = np.random.default_rng(99)
        closes_high = 100.0 + np.cumsum(rng2.normal(0, 1.0, n))
        closes_high = np.maximum(closes_high, 1.0)
        r_high = c.compute(350, ts, opens, highs, lows, closes_high, np.ones(n))

        assert r_high["rv_1h"] > r_low["rv_1h"]
        assert r_high["rv_1d"] > r_low["rv_1d"]
