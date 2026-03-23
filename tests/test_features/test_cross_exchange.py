"""Tests for cross-exchange signal features.

Covers: CoinbasePremiumComputer, ETHOrderFlowComputer,
        LongShortRatioComputer, CrossExchangeOFIComputer.
All tests check look-ahead bias and golden-dataset correctness.
"""
from __future__ import annotations

import math
import numpy as np
import pytest

from ep2_crypto.features.cross_market import (
    CoinbasePremiumComputer,
    CrossExchangeOFIComputer,
    ETHOrderFlowComputer,
    LongShortRatioComputer,
)


def _make_prices(n: int = 400, seed: int = 42) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    rng = np.random.default_rng(seed)
    ts = np.arange(n, dtype=np.int64) * 300_000
    closes = 30_000.0 + np.cumsum(rng.normal(0, 10, n))
    opens = closes - np.abs(rng.normal(0, 5, n))
    highs = closes + np.abs(rng.normal(0, 5, n))
    lows = opens - np.abs(rng.normal(0, 2, n))
    vols = np.abs(rng.normal(100, 20, n))
    return ts, opens, highs, lows, closes, vols


# ---------------------------------------------------------------------------
# CoinbasePremiumComputer
# ---------------------------------------------------------------------------

class TestCoinbasePremiumComputer:
    def test_output_names(self) -> None:
        c = CoinbasePremiumComputer()
        assert set(c.output_names()) == {
            "coinbase_premium", "coinbase_premium_zscore", "coinbase_premium_delta"
        }

    def test_warmup_returns_nan(self) -> None:
        c = CoinbasePremiumComputer(zscore_window=288)
        ts, opens, highs, lows, closes, vols = _make_prices(400)
        coinbase = closes * 1.001
        result = c.compute(10, ts, opens, highs, lows, closes, vols,
                           coinbase_closes=coinbase)
        assert math.isnan(result["coinbase_premium"])

    def test_no_coinbase_returns_nan(self) -> None:
        c = CoinbasePremiumComputer(zscore_window=50)
        ts, opens, highs, lows, closes, vols = _make_prices(200)
        result = c.compute(100, ts, opens, highs, lows, closes, vols)
        assert math.isnan(result["coinbase_premium"])

    def test_golden_zero_premium(self) -> None:
        """Coinbase = Binance → premium = 0."""
        c = CoinbasePremiumComputer(zscore_window=50)
        ts, opens, highs, lows, closes, vols = _make_prices(200)
        coinbase = closes.copy()
        result = c.compute(100, ts, opens, highs, lows, closes, vols,
                           coinbase_closes=coinbase)
        assert result["coinbase_premium"] == pytest.approx(0.0, abs=1e-12)

    def test_golden_positive_premium(self) -> None:
        """Coinbase 10 bps above Binance → premium ≈ 0.001."""
        c = CoinbasePremiumComputer(zscore_window=50)
        ts, opens, highs, lows, closes, vols = _make_prices(200)
        coinbase = closes * 1.001
        result = c.compute(100, ts, opens, highs, lows, closes, vols,
                           coinbase_closes=coinbase)
        assert result["coinbase_premium"] == pytest.approx(0.001, abs=1e-6)

    def test_delta_zero_when_constant_premium(self) -> None:
        """Constant premium → delta = 0."""
        c = CoinbasePremiumComputer(zscore_window=50)
        ts, opens, highs, lows, closes, vols = _make_prices(200)
        coinbase = closes * 1.001
        result = c.compute(100, ts, opens, highs, lows, closes, vols,
                           coinbase_closes=coinbase)
        assert result["coinbase_premium_delta"] == pytest.approx(0.0, abs=1e-10)

    def test_no_look_ahead_bias(self) -> None:
        c = CoinbasePremiumComputer(zscore_window=50)
        n = 200
        ts, opens, highs, lows, closes, vols = _make_prices(n)
        coinbase = closes * (1 + np.random.default_rng(5).normal(0, 0.0005, n))
        idx = 150
        full = c.compute(idx, ts, opens, highs, lows, closes, vols,
                         coinbase_closes=coinbase)
        trunc = c.compute(idx, ts[:idx + 1], opens[:idx + 1], highs[:idx + 1],
                          lows[:idx + 1], closes[:idx + 1], vols[:idx + 1],
                          coinbase_closes=coinbase[:idx + 1])
        assert full["coinbase_premium"] == pytest.approx(trunc["coinbase_premium"], abs=1e-12)
        assert full["coinbase_premium_zscore"] == pytest.approx(trunc["coinbase_premium_zscore"], abs=1e-8)


# ---------------------------------------------------------------------------
# ETHOrderFlowComputer
# ---------------------------------------------------------------------------

class TestETHOrderFlowComputer:
    def test_output_names(self) -> None:
        c = ETHOrderFlowComputer()
        assert set(c.output_names()) == {
            "eth_net_taker", "eth_net_taker_lag1",
            "eth_net_taker_lag3", "eth_net_taker_zscore"
        }

    def test_no_eth_data_returns_nan(self) -> None:
        c = ETHOrderFlowComputer(zscore_window=60)
        ts, opens, highs, lows, closes, vols = _make_prices(200)
        result = c.compute(100, ts, opens, highs, lows, closes, vols)
        assert math.isnan(result["eth_net_taker"])

    def test_warmup_returns_nan(self) -> None:
        c = ETHOrderFlowComputer(zscore_window=60)
        n = 200
        ts, opens, highs, lows, closes, vols = _make_prices(n)
        eth_sizes = np.ones(n)
        eth_sides = np.ones(n)
        result = c.compute(30, ts, opens, highs, lows, closes, vols,
                           eth_trade_sizes=eth_sizes, eth_trade_sides=eth_sides)
        assert math.isnan(result["eth_net_taker"])

    def test_golden_all_buys(self) -> None:
        """All ETH trades are buys → net_taker = volume."""
        c = ETHOrderFlowComputer(zscore_window=60)
        n = 200
        ts, opens, highs, lows, closes, vols = _make_prices(n)
        eth_sizes = np.ones(n) * 100.0
        eth_sides = np.ones(n)  # all buys
        result = c.compute(150, ts, opens, highs, lows, closes, vols,
                           eth_trade_sizes=eth_sizes, eth_trade_sides=eth_sides)
        assert result["eth_net_taker"] == pytest.approx(100.0, abs=1e-10)

    def test_golden_all_sells(self) -> None:
        """All ETH trades are sells → net_taker = -volume."""
        c = ETHOrderFlowComputer(zscore_window=60)
        n = 200
        ts, opens, highs, lows, closes, vols = _make_prices(n)
        eth_sizes = np.ones(n) * 100.0
        eth_sides = -np.ones(n)  # all sells
        result = c.compute(150, ts, opens, highs, lows, closes, vols,
                           eth_trade_sizes=eth_sizes, eth_trade_sides=eth_sides)
        assert result["eth_net_taker"] == pytest.approx(-100.0, abs=1e-10)

    def test_lags_use_past_only(self) -> None:
        """lag1 at t should equal eth_net_taker at t-1."""
        c = ETHOrderFlowComputer(zscore_window=60)
        n = 200
        ts, opens, highs, lows, closes, vols = _make_prices(n)
        rng = np.random.default_rng(7)
        eth_sizes = np.abs(rng.normal(50, 10, n))
        eth_sides = np.sign(rng.normal(0, 1, n))
        idx = 150
        result = c.compute(idx, ts, opens, highs, lows, closes, vols,
                           eth_trade_sizes=eth_sizes, eth_trade_sides=eth_sides)
        # Compute what lag1 should be
        expected_lag1 = float(eth_sizes[idx - 1] * eth_sides[idx - 1])
        assert result["eth_net_taker_lag1"] == pytest.approx(expected_lag1, abs=1e-10)

    def test_no_look_ahead_bias(self) -> None:
        c = ETHOrderFlowComputer(zscore_window=60)
        n = 200
        ts, opens, highs, lows, closes, vols = _make_prices(n)
        rng = np.random.default_rng(77)
        eth_sizes = np.abs(rng.normal(50, 10, n))
        eth_sides = np.sign(rng.normal(0, 1, n))
        idx = 150
        full = c.compute(idx, ts, opens, highs, lows, closes, vols,
                         eth_trade_sizes=eth_sizes, eth_trade_sides=eth_sides)
        trunc = c.compute(idx, ts[:idx + 1], opens[:idx + 1], highs[:idx + 1],
                          lows[:idx + 1], closes[:idx + 1], vols[:idx + 1],
                          eth_trade_sizes=eth_sizes[:idx + 1],
                          eth_trade_sides=eth_sides[:idx + 1])
        assert full["eth_net_taker"] == pytest.approx(trunc["eth_net_taker"], abs=1e-10)
        assert full["eth_net_taker_zscore"] == pytest.approx(trunc["eth_net_taker_zscore"], abs=1e-8)


# ---------------------------------------------------------------------------
# LongShortRatioComputer
# ---------------------------------------------------------------------------

class TestLongShortRatioComputer:
    def test_output_names(self) -> None:
        c = LongShortRatioComputer()
        assert set(c.output_names()) == {"ls_ratio", "ls_ratio_zscore", "ls_contrarian"}

    def test_no_data_returns_nan(self) -> None:
        c = LongShortRatioComputer(zscore_window=50)
        ts, opens, highs, lows, closes, vols = _make_prices(200)
        result = c.compute(100, ts, opens, highs, lows, closes, vols)
        assert math.isnan(result["ls_ratio"])

    def test_warmup_returns_nan(self) -> None:
        c = LongShortRatioComputer(zscore_window=100)
        n = 200
        ts, opens, highs, lows, closes, vols = _make_prices(n)
        ls = np.ones(n) * 1.5
        result = c.compute(50, ts, opens, highs, lows, closes, vols,
                           long_short_ratio=ls)
        assert math.isnan(result["ls_ratio"])

    def test_golden_ratio_passed_through(self) -> None:
        """Raw ratio is preserved exactly."""
        c = LongShortRatioComputer(zscore_window=50)
        n = 200
        ts, opens, highs, lows, closes, vols = _make_prices(n)
        ls = np.ones(n) * 1.5
        result = c.compute(150, ts, opens, highs, lows, closes, vols,
                           long_short_ratio=ls)
        assert result["ls_ratio"] == pytest.approx(1.5, abs=1e-10)

    def test_extreme_long_contrarian_negative(self) -> None:
        """Ratio > 2.5 → contrarian score is negative (expect down)."""
        c = LongShortRatioComputer(zscore_window=50)
        n = 200
        ts, opens, highs, lows, closes, vols = _make_prices(n)
        ls = np.ones(n) * 1.0
        ls[150] = 3.0  # extreme long
        result = c.compute(150, ts, opens, highs, lows, closes, vols,
                           long_short_ratio=ls)
        assert result["ls_contrarian"] < 0.0

    def test_extreme_short_contrarian_positive(self) -> None:
        """Ratio < 0.5 → contrarian score is positive (expect up)."""
        c = LongShortRatioComputer(zscore_window=50)
        n = 200
        ts, opens, highs, lows, closes, vols = _make_prices(n)
        ls = np.ones(n) * 1.0
        ls[150] = 0.2  # extreme short
        result = c.compute(150, ts, opens, highs, lows, closes, vols,
                           long_short_ratio=ls)
        assert result["ls_contrarian"] > 0.0

    def test_neutral_zone_zero_contrarian(self) -> None:
        """Ratio in [0.5, 2.5] → contrarian = 0."""
        c = LongShortRatioComputer(zscore_window=50)
        n = 200
        ts, opens, highs, lows, closes, vols = _make_prices(n)
        ls = np.ones(n) * 1.5
        result = c.compute(150, ts, opens, highs, lows, closes, vols,
                           long_short_ratio=ls)
        assert result["ls_contrarian"] == pytest.approx(0.0, abs=1e-10)

    def test_no_look_ahead_bias(self) -> None:
        c = LongShortRatioComputer(zscore_window=50)
        n = 200
        ts, opens, highs, lows, closes, vols = _make_prices(n)
        rng = np.random.default_rng(33)
        ls = np.abs(rng.normal(1.2, 0.3, n))
        idx = 150
        full = c.compute(idx, ts, opens, highs, lows, closes, vols,
                         long_short_ratio=ls)
        trunc = c.compute(idx, ts[:idx + 1], opens[:idx + 1], highs[:idx + 1],
                          lows[:idx + 1], closes[:idx + 1], vols[:idx + 1],
                          long_short_ratio=ls[:idx + 1])
        assert full["ls_ratio_zscore"] == pytest.approx(trunc["ls_ratio_zscore"], abs=1e-8)


# ---------------------------------------------------------------------------
# CrossExchangeOFIComputer
# ---------------------------------------------------------------------------

class TestCrossExchangeOFIComputer:
    def test_output_names(self) -> None:
        c = CrossExchangeOFIComputer()
        assert set(c.output_names()) == {"xex_ofi_divergence", "xex_ofi_divergence_ma"}

    def test_no_data_returns_nan(self) -> None:
        c = CrossExchangeOFIComputer(window=12)
        ts, opens, highs, lows, closes, vols = _make_prices(100)
        result = c.compute(50, ts, opens, highs, lows, closes, vols)
        assert math.isnan(result["xex_ofi_divergence"])

    def test_golden_zero_divergence(self) -> None:
        """Same OFI on both exchanges → divergence = 0."""
        c = CrossExchangeOFIComputer(window=12)
        n = 100
        ts, opens, highs, lows, closes, vols = _make_prices(n)
        ofi = np.random.default_rng(9).normal(0, 100, n)
        result = c.compute(50, ts, opens, highs, lows, closes, vols,
                           binance_ofi=ofi, coinbase_ofi=ofi)
        assert result["xex_ofi_divergence"] == pytest.approx(0.0, abs=1e-10)
        assert result["xex_ofi_divergence_ma"] == pytest.approx(0.0, abs=1e-10)

    def test_golden_specific_divergence(self) -> None:
        """Binance OFI = 200, Coinbase OFI = 50 → divergence = 150."""
        c = CrossExchangeOFIComputer(window=12)
        n = 100
        ts, opens, highs, lows, closes, vols = _make_prices(n)
        binance_ofi = np.zeros(n)
        binance_ofi[50] = 200.0
        coinbase_ofi = np.zeros(n)
        coinbase_ofi[50] = 50.0
        result = c.compute(50, ts, opens, highs, lows, closes, vols,
                           binance_ofi=binance_ofi, coinbase_ofi=coinbase_ofi)
        assert result["xex_ofi_divergence"] == pytest.approx(150.0, abs=1e-8)

    def test_ma_is_rolling_mean(self) -> None:
        """MA divergence = mean of last window divergences."""
        c = CrossExchangeOFIComputer(window=10)
        n = 100
        ts, opens, highs, lows, closes, vols = _make_prices(n)
        rng = np.random.default_rng(55)
        binance_ofi = rng.normal(0, 100, n)
        coinbase_ofi = rng.normal(0, 100, n)
        idx = 50
        result = c.compute(idx, ts, opens, highs, lows, closes, vols,
                           binance_ofi=binance_ofi, coinbase_ofi=coinbase_ofi)
        expected_ma = float(np.mean(binance_ofi[41:51] - coinbase_ofi[41:51]))
        assert result["xex_ofi_divergence_ma"] == pytest.approx(expected_ma, abs=1e-8)

    def test_no_look_ahead_bias(self) -> None:
        c = CrossExchangeOFIComputer(window=12)
        n = 100
        ts, opens, highs, lows, closes, vols = _make_prices(n)
        rng = np.random.default_rng(11)
        binance_ofi = rng.normal(0, 100, n)
        coinbase_ofi = rng.normal(0, 100, n)
        idx = 50
        full = c.compute(idx, ts, opens, highs, lows, closes, vols,
                         binance_ofi=binance_ofi, coinbase_ofi=coinbase_ofi)
        trunc = c.compute(idx, ts[:idx + 1], opens[:idx + 1], highs[:idx + 1],
                          lows[:idx + 1], closes[:idx + 1], vols[:idx + 1],
                          binance_ofi=binance_ofi[:idx + 1],
                          coinbase_ofi=coinbase_ofi[:idx + 1])
        assert full["xex_ofi_divergence"] == pytest.approx(trunc["xex_ofi_divergence"], abs=1e-10)
        assert full["xex_ofi_divergence_ma"] == pytest.approx(trunc["xex_ofi_divergence_ma"], abs=1e-10)
