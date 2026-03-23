"""Tests for PositionSizer: Kelly, ATR stops, max caps, rejections."""

from __future__ import annotations

import numpy as np
import pytest

from ep2_crypto.risk.position_sizer import PositionSizer


@pytest.fixture
def sizer() -> PositionSizer:
    return PositionSizer()


@pytest.fixture
def price_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic OHLC data for ATR computation."""
    rng = np.random.default_rng(42)
    n = 50
    base = 67000.0
    closes = base + np.cumsum(rng.normal(0, 50, n))
    highs = closes + rng.uniform(20, 100, n)
    lows = closes - rng.uniform(20, 100, n)
    return highs, lows, closes


class TestKellyFormula:
    def test_positive_edge(self, sizer: PositionSizer) -> None:
        # WR=0.55, payoff=1.0 -> Kelly = (0.55*1 - 0.45)/1 = 0.10
        result = sizer.compute(
            "long",
            0.80,
            50_000.0,
            67000.0,
            *_flat_prices(67000.0),
            current_idx=49,
            win_rate=0.55,
            payoff_ratio=1.0,
        )
        assert not result.rejected
        assert result.raw_kelly_fraction == pytest.approx(0.10, rel=0.01)

    def test_no_edge_rejected(self, sizer: PositionSizer) -> None:
        # WR=0.45, payoff=1.0 -> Kelly = (0.45 - 0.55)/1 = -0.10 -> 0
        result = sizer.compute(
            "long",
            0.80,
            50_000.0,
            67000.0,
            *_flat_prices(67000.0),
            current_idx=49,
            win_rate=0.45,
            payoff_ratio=1.0,
        )
        assert result.rejected
        assert "non-positive" in (result.rejection_reason or "")

    def test_quarter_kelly_applied(self, sizer: PositionSizer) -> None:
        result = sizer.compute(
            "long",
            1.0,
            50_000.0,
            67000.0,
            *_flat_prices(67000.0),
            current_idx=49,
            win_rate=0.55,
            payoff_ratio=1.0,
        )
        assert not result.rejected
        assert result.quarter_kelly_fraction == pytest.approx(0.025, rel=0.01)


class TestMaxCap:
    def test_never_exceeds_5_percent(self, sizer: PositionSizer) -> None:
        # High WR + high confidence should still cap at 5%
        result = sizer.compute(
            "long",
            1.0,
            50_000.0,
            67000.0,
            *_flat_prices(67000.0),
            current_idx=49,
            win_rate=0.80,
            payoff_ratio=2.0,
        )
        assert not result.rejected
        assert result.position_fraction <= 0.05 + 1e-9

    def test_notional_within_cap(self, sizer: PositionSizer) -> None:
        equity = 100_000.0
        result = sizer.compute(
            "long",
            1.0,
            equity,
            67000.0,
            *_flat_prices(67000.0),
            current_idx=49,
            win_rate=0.80,
            payoff_ratio=2.0,
        )
        assert not result.rejected
        assert result.notional_usd <= equity * 0.05 + 1.0


class TestDrawdownScaling:
    def test_drawdown_multiplier_reduces_size(self, sizer: PositionSizer) -> None:
        full = sizer.compute(
            "long",
            0.80,
            50_000.0,
            67000.0,
            *_flat_prices(67000.0),
            current_idx=49,
            drawdown_multiplier=1.0,
        )
        reduced = sizer.compute(
            "long",
            0.80,
            50_000.0,
            67000.0,
            *_flat_prices(67000.0),
            current_idx=49,
            drawdown_multiplier=0.50,
        )
        assert not full.rejected
        assert not reduced.rejected
        assert reduced.quantity_btc < full.quantity_btc
        assert reduced.quantity_btc == pytest.approx(full.quantity_btc * 0.50, rel=0.05)

    def test_zero_drawdown_multiplier_rejected(self, sizer: PositionSizer) -> None:
        result = sizer.compute(
            "long",
            0.80,
            50_000.0,
            67000.0,
            *_flat_prices(67000.0),
            current_idx=49,
            drawdown_multiplier=0.0,
        )
        assert result.rejected


class TestATRStop:
    def test_long_stop_below_price(self, sizer: PositionSizer) -> None:
        result = sizer.compute(
            "long",
            0.80,
            50_000.0,
            67000.0,
            *_volatile_prices(67000.0),
            current_idx=49,
        )
        assert not result.rejected
        assert result.stop_price < 67000.0
        assert result.stop_distance_usd > 0

    def test_short_stop_above_price(self, sizer: PositionSizer) -> None:
        result = sizer.compute(
            "short",
            0.80,
            50_000.0,
            67000.0,
            *_volatile_prices(67000.0),
            current_idx=49,
        )
        assert not result.rejected
        assert result.stop_price > 67000.0


class TestRejections:
    def test_invalid_direction(self, sizer: PositionSizer) -> None:
        result = sizer.compute(
            "sideways",
            0.80,
            50_000.0,
            67000.0,
            *_flat_prices(67000.0),
            current_idx=49,
        )
        assert result.rejected

    def test_zero_confidence(self, sizer: PositionSizer) -> None:
        result = sizer.compute(
            "long",
            0.0,
            50_000.0,
            67000.0,
            *_flat_prices(67000.0),
            current_idx=49,
        )
        assert result.rejected

    def test_zero_equity(self, sizer: PositionSizer) -> None:
        result = sizer.compute(
            "long",
            0.80,
            0.0,
            67000.0,
            *_flat_prices(67000.0),
            current_idx=49,
        )
        assert result.rejected


class TestWeekendFundingScaling:
    def test_time_multiplier_reduces_size(self, sizer: PositionSizer) -> None:
        full = sizer.compute(
            "long",
            0.80,
            50_000.0,
            67000.0,
            *_flat_prices(67000.0),
            current_idx=49,
            time_multiplier=1.0,
        )
        weekend = sizer.compute(
            "long",
            0.80,
            50_000.0,
            67000.0,
            *_flat_prices(67000.0),
            current_idx=49,
            time_multiplier=0.70,
        )
        assert not full.rejected
        assert not weekend.rejected
        assert weekend.quantity_btc < full.quantity_btc


class TestMaxHolding:
    def test_max_holding_bars_accessible(self, sizer: PositionSizer) -> None:
        assert sizer.max_holding_bars == 6


class TestValidation:
    def test_invalid_kelly_fraction(self) -> None:
        with pytest.raises(ValueError, match="kelly_fraction"):
            PositionSizer(kelly_fraction=0.0)

    def test_invalid_max_position_fraction(self) -> None:
        with pytest.raises(ValueError, match="max_position_fraction"):
            PositionSizer(max_position_fraction=1.5)

    def test_invalid_max_risk_per_trade(self) -> None:
        with pytest.raises(ValueError, match="max_risk_per_trade"):
            PositionSizer(max_risk_per_trade=0.06)

    def test_invalid_atr_multiplier(self) -> None:
        with pytest.raises(ValueError, match="atr_multiplier"):
            PositionSizer(atr_multiplier=-1.0)

    def test_invalid_atr_period(self) -> None:
        with pytest.raises(ValueError, match="atr_period"):
            PositionSizer(atr_period=1)


class TestNonPositivePrice:
    def test_zero_price_rejected(self, sizer: PositionSizer) -> None:
        result = sizer.compute(
            "long",
            0.80,
            50_000.0,
            0.0,
            *_flat_prices(67000.0),
            current_idx=49,
        )
        assert result.rejected

    def test_negative_price_rejected(self, sizer: PositionSizer) -> None:
        result = sizer.compute(
            "long",
            0.80,
            50_000.0,
            -100.0,
            *_flat_prices(67000.0),
            current_idx=49,
        )
        assert result.rejected


class TestRiskCapApplied:
    def test_risk_cap_reduces_quantity(self) -> None:
        """When risk per trade exceeds cap, quantity should be reduced."""
        sizer = PositionSizer(
            max_risk_per_trade=0.005,  # 0.5% very tight cap
            kelly_fraction=0.25,
        )
        result = sizer.compute(
            "long",
            1.0,
            100_000.0,
            67000.0,
            *_volatile_prices(67000.0),
            current_idx=49,
            win_rate=0.70,
            payoff_ratio=1.5,
        )
        if not result.rejected:
            max_risk_usd = 0.005 * 100_000.0
            assert result.risk_per_trade_usd <= max_risk_usd + 1.0

    def test_risk_cap_too_small_rejects(self) -> None:
        """Very small equity + tight risk cap -> rejection below min BTC."""
        sizer = PositionSizer(
            max_risk_per_trade=0.001,
            min_btc_quantity=0.01,
        )
        result = sizer.compute(
            "long",
            0.80,
            500.0,
            67000.0,  # tiny equity
            *_volatile_prices(67000.0),
            current_idx=49,
            win_rate=0.55,
            payoff_ratio=1.0,
        )
        # Should be rejected because risk budget too small for min BTC
        assert result.rejected


class TestPayoffRatioEdgeCases:
    def test_zero_payoff_rejected(self, sizer: PositionSizer) -> None:
        result = sizer.compute(
            "long",
            0.80,
            50_000.0,
            67000.0,
            *_flat_prices(67000.0),
            current_idx=49,
            win_rate=0.55,
            payoff_ratio=0.0,
        )
        assert result.rejected


# -- Helpers ------------------------------------------------------------------


def _flat_prices(price: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Constant price arrays (ATR will be near zero, fallback used)."""
    n = 50
    closes = np.full(n, price)
    highs = np.full(n, price + 10)
    lows = np.full(n, price - 10)
    return highs, lows, closes


def _volatile_prices(price: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Prices with realistic ATR for stop computation."""
    rng = np.random.default_rng(42)
    n = 50
    closes = price + np.cumsum(rng.normal(0, 50, n))
    highs = closes + rng.uniform(50, 200, n)
    lows = closes - rng.uniform(50, 200, n)
    return highs, lows, closes
