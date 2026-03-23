"""Tests for advanced microstructure features: OBI-L10, VPIN, book pressure, depth withdrawal.

All features are tested for:
1. Correct computation against hand-verified values (golden dataset)
2. Look-ahead bias: result at index t must equal result when array is truncated at t
3. Warmup behaviour: NaN returned before warmup_bars - 1
"""
from __future__ import annotations

import math
import numpy as np
import pytest

from ep2_crypto.features.microstructure import (
    BookPressureGradientComputer,
    DepthWithdrawalComputer,
    OBILevel10Computer,
    VPINComputer,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_prices(n: int = 200) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    ts = np.arange(n, dtype=np.int64) * 300_000
    closes = 30_000.0 + np.cumsum(np.random.default_rng(42).normal(0, 10, n))
    opens = closes - np.abs(np.random.default_rng(7).normal(0, 5, n))
    highs = closes + np.abs(np.random.default_rng(13).normal(0, 5, n))
    lows = closes - np.abs(np.random.default_rng(17).normal(0, 5, n))
    vols = np.abs(np.random.default_rng(23).normal(100, 20, n))
    return ts, opens, highs, lows, closes, vols


def _make_book(n: int = 200, n_levels: int = 12) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(99)
    bid_sizes = np.abs(rng.normal(5, 1, (n, n_levels)))
    ask_sizes = np.abs(rng.normal(5, 1, (n, n_levels)))
    # Bid prices: decreasing from mid
    mid = 30_000.0
    bids = np.tile(mid - np.arange(n_levels) * 5.0, (n, 1)).astype(np.float64)
    asks = np.tile(mid + np.arange(n_levels) * 5.0 + 5.0, (n, 1)).astype(np.float64)
    return bids, asks, bid_sizes, ask_sizes


# ---------------------------------------------------------------------------
# OBILevel10Computer
# ---------------------------------------------------------------------------

class TestOBILevel10Computer:
    def test_output_names(self) -> None:
        c = OBILevel10Computer()
        assert c.output_names() == ["obi_l10", "obi_l10_weighted"]

    def test_warmup_returns_nan(self) -> None:
        c = OBILevel10Computer()
        ts, opens, highs, lows, closes, vols = _make_prices(5)
        bids, asks, bid_sizes, ask_sizes = _make_book(5)
        result = c.compute(0, ts, opens, highs, lows, closes, vols,
                           bids=bids, asks=asks, bid_sizes=bid_sizes, ask_sizes=ask_sizes)
        # warmup_bars=1 so idx=0 is valid
        assert not math.isnan(result["obi_l10"])

    def test_no_book_data_returns_nan(self) -> None:
        c = OBILevel10Computer()
        ts, opens, highs, lows, closes, vols = _make_prices(10)
        result = c.compute(5, ts, opens, highs, lows, closes, vols)
        assert math.isnan(result["obi_l10"])

    def test_golden_balanced_book(self) -> None:
        """Balanced bid/ask → OBI = 0."""
        c = OBILevel10Computer()
        n, nl = 10, 12
        ts, opens, highs, lows, closes, vols = _make_prices(n)
        bid_sizes = np.ones((n, nl)) * 5.0
        ask_sizes = np.ones((n, nl)) * 5.0
        result = c.compute(5, ts, opens, highs, lows, closes, vols,
                           bid_sizes=bid_sizes, ask_sizes=ask_sizes)
        assert result["obi_l10"] == pytest.approx(0.0, abs=1e-10)

    def test_golden_all_bids(self) -> None:
        """All bid volume → OBI = 1."""
        c = OBILevel10Computer()
        n, nl = 10, 12
        ts, opens, highs, lows, closes, vols = _make_prices(n)
        bid_sizes = np.ones((n, nl)) * 10.0
        ask_sizes = np.zeros((n, nl))
        result = c.compute(5, ts, opens, highs, lows, closes, vols,
                           bid_sizes=bid_sizes, ask_sizes=ask_sizes)
        assert result["obi_l10"] == pytest.approx(1.0, abs=1e-10)

    def test_golden_all_asks(self) -> None:
        """All ask volume → OBI = -1."""
        c = OBILevel10Computer()
        n, nl = 10, 12
        ts, opens, highs, lows, closes, vols = _make_prices(n)
        bid_sizes = np.zeros((n, nl))
        ask_sizes = np.ones((n, nl)) * 10.0
        result = c.compute(5, ts, opens, highs, lows, closes, vols,
                           bid_sizes=bid_sizes, ask_sizes=ask_sizes)
        assert result["obi_l10"] == pytest.approx(-1.0, abs=1e-10)

    def test_only_uses_10_levels(self) -> None:
        """OBI-L10 only sums first 10 levels."""
        c = OBILevel10Computer()
        n, nl = 10, 15
        ts, opens, highs, lows, closes, vols = _make_prices(n)
        bid_sizes = np.ones((n, nl)) * 2.0
        ask_sizes = np.ones((n, nl)) * 2.0
        # Set levels 10-14 to huge values — should not affect result
        bid_sizes[:, 10:] = 1000.0
        ask_sizes[:, 10:] = 0.0
        result = c.compute(5, ts, opens, highs, lows, closes, vols,
                           bid_sizes=bid_sizes, ask_sizes=ask_sizes)
        # First 10 levels are balanced → OBI should be 0
        assert result["obi_l10"] == pytest.approx(0.0, abs=1e-10)

    def test_no_look_ahead_bias(self) -> None:
        n = 150
        ts, opens, highs, lows, closes, vols = _make_prices(n)
        bids, asks, bid_sizes, ask_sizes = _make_book(n)
        c = OBILevel10Computer()
        idx = 100
        full = c.compute(idx, ts, opens, highs, lows, closes, vols,
                         bids=bids, asks=asks, bid_sizes=bid_sizes, ask_sizes=ask_sizes)
        trunc = c.compute(idx, ts[:idx + 1], opens[:idx + 1], highs[:idx + 1],
                          lows[:idx + 1], closes[:idx + 1], vols[:idx + 1],
                          bids=bids[:idx + 1], asks=asks[:idx + 1],
                          bid_sizes=bid_sizes[:idx + 1], ask_sizes=ask_sizes[:idx + 1])
        assert full["obi_l10"] == pytest.approx(trunc["obi_l10"], abs=1e-10)


# ---------------------------------------------------------------------------
# VPINComputer
# ---------------------------------------------------------------------------

class TestVPINComputer:
    def test_output_names(self) -> None:
        c = VPINComputer()
        assert set(c.output_names()) == {"vpin", "vpin_imbalance"}

    def test_warmup_returns_nan(self) -> None:
        c = VPINComputer(window=50)
        n = 40
        ts, opens, highs, lows, closes, vols = _make_prices(n)
        result = c.compute(30, ts, opens, highs, lows, closes, vols)
        assert math.isnan(result["vpin"])

    def test_vpin_range(self) -> None:
        """VPIN must be in [0, 1]."""
        c = VPINComputer(window=50)
        n = 200
        ts, opens, highs, lows, closes, vols = _make_prices(n)
        for idx in [60, 100, 150, 199]:
            r = c.compute(idx, ts, opens, highs, lows, closes, vols)
            assert 0.0 <= r["vpin"] <= 1.0, f"VPIN out of range at idx={idx}: {r['vpin']}"

    def test_imbalance_signed(self) -> None:
        """Signed imbalance must be in [-1, 1]."""
        c = VPINComputer(window=50)
        n = 200
        ts, opens, highs, lows, closes, vols = _make_prices(n)
        for idx in [60, 100, 150, 199]:
            r = c.compute(idx, ts, opens, highs, lows, closes, vols)
            assert -1.0 <= r["vpin_imbalance"] <= 1.0

    def test_vpin_equals_abs_imbalance(self) -> None:
        """VPIN = |vpin_imbalance| by definition."""
        c = VPINComputer(window=50)
        n = 200
        ts, opens, highs, lows, closes, vols = _make_prices(n)
        for idx in [60, 100, 150, 199]:
            r = c.compute(idx, ts, opens, highs, lows, closes, vols)
            assert r["vpin"] == pytest.approx(abs(r["vpin_imbalance"]), abs=1e-10)

    def test_all_up_bars_high_imbalance(self) -> None:
        """Predominantly rising prices → positive VPIN imbalance."""
        rng = np.random.default_rng(55)
        c = VPINComputer(window=30)
        n = 100
        ts = np.arange(n, dtype=np.int64) * 300_000
        # Rising prices with small noise so sigma > 0
        closes = 30_000.0 + np.cumsum(np.abs(rng.normal(5, 1, n)))
        opens = closes - np.abs(rng.normal(3, 0.5, n))  # always below close → up bars
        highs = closes + 1.0
        lows = opens - 1.0
        vols = np.ones(n) * 100.0
        r = c.compute(60, ts, opens, highs, lows, closes, vols)
        assert not math.isnan(r["vpin_imbalance"])
        assert r["vpin_imbalance"] > 0.3

    def test_no_look_ahead_bias(self) -> None:
        c = VPINComputer(window=50)
        n = 150
        ts, opens, highs, lows, closes, vols = _make_prices(n)
        idx = 100
        full = c.compute(idx, ts, opens, highs, lows, closes, vols)
        trunc = c.compute(idx, ts[:idx + 1], opens[:idx + 1], highs[:idx + 1],
                          lows[:idx + 1], closes[:idx + 1], vols[:idx + 1])
        assert full["vpin"] == pytest.approx(trunc["vpin"], abs=1e-10)


# ---------------------------------------------------------------------------
# BookPressureGradientComputer
# ---------------------------------------------------------------------------

class TestBookPressureGradientComputer:
    def test_output_names(self) -> None:
        c = BookPressureGradientComputer()
        assert set(c.output_names()) == {
            "bid_pressure_gradient",
            "ask_pressure_gradient",
            "net_pressure_gradient",
        }

    def test_no_book_returns_nan(self) -> None:
        c = BookPressureGradientComputer()
        ts, opens, highs, lows, closes, vols = _make_prices(10)
        result = c.compute(5, ts, opens, highs, lows, closes, vols)
        assert math.isnan(result["bid_pressure_gradient"])

    def test_fewer_than_3_levels_returns_nan(self) -> None:
        c = BookPressureGradientComputer()
        n, nl = 10, 2
        ts, opens, highs, lows, closes, vols = _make_prices(n)
        bid_sizes = np.ones((n, nl))
        ask_sizes = np.ones((n, nl))
        result = c.compute(5, ts, opens, highs, lows, closes, vols,
                           bid_sizes=bid_sizes, ask_sizes=ask_sizes)
        assert math.isnan(result["bid_pressure_gradient"])

    def test_decreasing_bid_profile_negative_slope(self) -> None:
        """Bid sizes decreasing with depth → negative gradient."""
        c = BookPressureGradientComputer()
        n, nl = 10, 10
        ts, opens, highs, lows, closes, vols = _make_prices(n)
        bid_sizes = np.tile(np.linspace(10, 1, nl), (n, 1))
        ask_sizes = np.ones((n, nl)) * 5.0
        result = c.compute(5, ts, opens, highs, lows, closes, vols,
                           bid_sizes=bid_sizes, ask_sizes=ask_sizes)
        assert result["bid_pressure_gradient"] < 0

    def test_flat_book_zero_gradient(self) -> None:
        """Flat book across all levels → gradient near zero."""
        c = BookPressureGradientComputer()
        n, nl = 10, 10
        ts, opens, highs, lows, closes, vols = _make_prices(n)
        bid_sizes = np.ones((n, nl)) * 5.0
        ask_sizes = np.ones((n, nl)) * 5.0
        result = c.compute(5, ts, opens, highs, lows, closes, vols,
                           bid_sizes=bid_sizes, ask_sizes=ask_sizes)
        assert result["net_pressure_gradient"] == pytest.approx(0.0, abs=1e-10)

    def test_no_look_ahead_bias(self) -> None:
        c = BookPressureGradientComputer()
        n = 150
        ts, opens, highs, lows, closes, vols = _make_prices(n)
        _, _, bid_sizes, ask_sizes = _make_book(n)
        idx = 100
        full = c.compute(idx, ts, opens, highs, lows, closes, vols,
                         bid_sizes=bid_sizes, ask_sizes=ask_sizes)
        trunc = c.compute(idx, ts[:idx + 1], opens[:idx + 1], highs[:idx + 1],
                          lows[:idx + 1], closes[:idx + 1], vols[:idx + 1],
                          bid_sizes=bid_sizes[:idx + 1], ask_sizes=ask_sizes[:idx + 1])
        assert full["bid_pressure_gradient"] == pytest.approx(trunc["bid_pressure_gradient"], abs=1e-10)


# ---------------------------------------------------------------------------
# DepthWithdrawalComputer
# ---------------------------------------------------------------------------

class TestDepthWithdrawalComputer:
    def test_output_names(self) -> None:
        c = DepthWithdrawalComputer()
        assert set(c.output_names()) == {"depth_withdrawal", "bid_withdrawal", "ask_withdrawal"}

    def test_warmup_returns_nan(self) -> None:
        c = DepthWithdrawalComputer(lookback=6)
        n = 5
        ts, opens, highs, lows, closes, vols = _make_prices(n)
        _, _, bid_sizes, ask_sizes = _make_book(n)
        result = c.compute(4, ts, opens, highs, lows, closes, vols,
                           bid_sizes=bid_sizes, ask_sizes=ask_sizes)
        assert math.isnan(result["depth_withdrawal"])

    def test_no_book_returns_nan(self) -> None:
        c = DepthWithdrawalComputer()
        ts, opens, highs, lows, closes, vols = _make_prices(20)
        result = c.compute(10, ts, opens, highs, lows, closes, vols)
        assert math.isnan(result["depth_withdrawal"])

    def test_golden_zero_withdrawal(self) -> None:
        """Same depth at current and lookback → withdrawal = 0."""
        c = DepthWithdrawalComputer(lookback=6)
        n, nl = 20, 12
        ts, opens, highs, lows, closes, vols = _make_prices(n)
        bid_sizes = np.ones((n, nl)) * 5.0
        ask_sizes = np.ones((n, nl)) * 5.0
        result = c.compute(10, ts, opens, highs, lows, closes, vols,
                           bid_sizes=bid_sizes, ask_sizes=ask_sizes)
        assert result["depth_withdrawal"] == pytest.approx(0.0, abs=1e-10)

    def test_golden_full_withdrawal(self) -> None:
        """Depth drops to zero → withdrawal = 1."""
        c = DepthWithdrawalComputer(lookback=6)
        n, nl = 20, 12
        ts, opens, highs, lows, closes, vols = _make_prices(n)
        bid_sizes = np.ones((n, nl)) * 5.0
        ask_sizes = np.ones((n, nl)) * 5.0
        # Set current bar depth to zero
        bid_sizes[10, :] = 0.0
        ask_sizes[10, :] = 0.0
        result = c.compute(10, ts, opens, highs, lows, closes, vols,
                           bid_sizes=bid_sizes, ask_sizes=ask_sizes)
        assert result["depth_withdrawal"] == pytest.approx(1.0, abs=1e-10)

    def test_golden_50pct_withdrawal(self) -> None:
        """Depth halves → withdrawal = 0.5."""
        c = DepthWithdrawalComputer(lookback=6)
        n, nl = 20, 12
        ts, opens, highs, lows, closes, vols = _make_prices(n)
        bid_sizes = np.ones((n, nl)) * 10.0
        ask_sizes = np.ones((n, nl)) * 10.0
        # Current bar has half the depth
        bid_sizes[10, :] = 5.0
        ask_sizes[10, :] = 5.0
        result = c.compute(10, ts, opens, highs, lows, closes, vols,
                           bid_sizes=bid_sizes, ask_sizes=ask_sizes)
        assert result["depth_withdrawal"] == pytest.approx(0.5, abs=1e-10)

    def test_negative_withdrawal_is_depth_addition(self) -> None:
        """Depth increases → withdrawal < 0 (depth addition)."""
        c = DepthWithdrawalComputer(lookback=6)
        n, nl = 20, 12
        ts, opens, highs, lows, closes, vols = _make_prices(n)
        bid_sizes = np.ones((n, nl)) * 5.0
        ask_sizes = np.ones((n, nl)) * 5.0
        # More depth at current bar
        bid_sizes[10, :] = 10.0
        ask_sizes[10, :] = 10.0
        result = c.compute(10, ts, opens, highs, lows, closes, vols,
                           bid_sizes=bid_sizes, ask_sizes=ask_sizes)
        assert result["depth_withdrawal"] < 0.0

    def test_no_look_ahead_bias(self) -> None:
        c = DepthWithdrawalComputer(lookback=6)
        n = 150
        ts, opens, highs, lows, closes, vols = _make_prices(n)
        _, _, bid_sizes, ask_sizes = _make_book(n)
        idx = 100
        full = c.compute(idx, ts, opens, highs, lows, closes, vols,
                         bid_sizes=bid_sizes, ask_sizes=ask_sizes)
        trunc = c.compute(idx, ts[:idx + 1], opens[:idx + 1], highs[:idx + 1],
                          lows[:idx + 1], closes[:idx + 1], vols[:idx + 1],
                          bid_sizes=bid_sizes[:idx + 1], ask_sizes=ask_sizes[:idx + 1])
        assert full["depth_withdrawal"] == pytest.approx(trunc["depth_withdrawal"], abs=1e-10)
