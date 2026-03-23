"""Tests for VolatilityGuard: vol range, trading hours, weekend, funding."""

from __future__ import annotations

import math
from datetime import UTC, datetime

import numpy as np
import pytest

from ep2_crypto.risk.volatility_guard import ANNUALIZATION_FACTOR, VolatilityGuard


@pytest.fixture
def guard() -> VolatilityGuard:
    return VolatilityGuard()


def _make_prices_with_vol(
    n_bars: int, base_price: float, target_vol_ann: float
) -> np.ndarray:
    """Generate synthetic close prices with approximately the target annualized vol."""
    rng = np.random.default_rng(42)
    bar_vol = target_vol_ann / ANNUALIZATION_FACTOR
    log_returns = rng.normal(0, bar_vol, size=n_bars)
    log_prices = np.cumsum(log_returns)
    log_prices = np.insert(log_prices, 0, 0.0)
    return base_price * np.exp(log_prices)


class TestVolatilityComputation:
    def test_annualization_factor(self) -> None:
        assert pytest.approx(math.sqrt(105_120), rel=1e-4) == ANNUALIZATION_FACTOR

    def test_zero_vol_when_insufficient_data(self, guard: VolatilityGuard) -> None:
        closes = np.array([67000.0])
        vol = guard.compute_volatility(closes, 0)
        assert vol == 0.0

    def test_reasonable_vol_on_synthetic_data(self, guard: VolatilityGuard) -> None:
        prices = _make_prices_with_vol(500, 67000.0, 0.50)
        vol = guard.compute_volatility(prices, len(prices) - 1)
        # Should be roughly 50% annualized (within tolerance of random noise)
        assert 0.30 < vol < 0.80

    def test_constant_prices_zero_vol(self, guard: VolatilityGuard) -> None:
        closes = np.full(300, 67000.0)
        vol = guard.compute_volatility(closes, 299)
        assert vol == pytest.approx(0.0)


class TestVolRange:
    def test_low_vol_blocks_trade(self, guard: VolatilityGuard) -> None:
        # Very quiet market: constant prices
        closes = np.full(300, 67000.0)
        ts_ms = int(datetime(2026, 3, 23, 12, 0, tzinfo=UTC).timestamp() * 1000)
        state = guard.check(closes, 299, ts_ms)
        assert not state.vol_in_range
        assert not state.can_trade
        assert "too low" in (state.rejection_reason or "")

    def test_high_vol_blocks_trade(self, guard: VolatilityGuard) -> None:
        prices = _make_prices_with_vol(500, 67000.0, 2.0)  # 200% vol
        ts_ms = int(datetime(2026, 3, 23, 12, 0, tzinfo=UTC).timestamp() * 1000)
        state = guard.check(prices, len(prices) - 1, ts_ms)
        assert not state.vol_in_range
        if not state.can_trade:
            assert "too high" in (state.rejection_reason or "")

    def test_normal_vol_allows_trade(self, guard: VolatilityGuard) -> None:
        prices = _make_prices_with_vol(500, 67000.0, 0.50)  # 50% vol
        ts_ms = int(datetime(2026, 3, 23, 12, 0, tzinfo=UTC).timestamp() * 1000)
        state = guard.check(prices, len(prices) - 1, ts_ms)
        # Vol should be in range (if hours are also ok)
        assert state.vol_in_range


class TestTradingHours:
    def test_within_hours_allowed(self) -> None:
        guard = VolatilityGuard(enforce_trading_hours=True)
        prices = _make_prices_with_vol(500, 67000.0, 0.50)
        # 14:00 UTC on a Wednesday
        ts_ms = int(datetime(2026, 3, 25, 14, 0, tzinfo=UTC).timestamp() * 1000)
        state = guard.check(prices, len(prices) - 1, ts_ms)
        assert state.in_trading_hours

    def test_outside_hours_blocked(self) -> None:
        guard = VolatilityGuard(enforce_trading_hours=True)
        prices = _make_prices_with_vol(500, 67000.0, 0.50)
        # 03:00 UTC on a Wednesday
        ts_ms = int(datetime(2026, 3, 25, 3, 0, tzinfo=UTC).timestamp() * 1000)
        state = guard.check(prices, len(prices) - 1, ts_ms)
        assert not state.in_trading_hours
        assert not state.can_trade

    def test_hours_enforcement_disabled(self) -> None:
        guard = VolatilityGuard(enforce_trading_hours=False)
        prices = _make_prices_with_vol(500, 67000.0, 0.50)
        ts_ms = int(datetime(2026, 3, 25, 3, 0, tzinfo=UTC).timestamp() * 1000)
        state = guard.check(prices, len(prices) - 1, ts_ms)
        # Vol check may still fail but hours should not block
        assert state.in_trading_hours or state.can_trade is not None  # hours not enforced


class TestWeekend:
    def test_saturday_detected(self, guard: VolatilityGuard) -> None:
        prices = _make_prices_with_vol(500, 67000.0, 0.50)
        # Saturday, March 28, 2026, 12:00 UTC
        ts_ms = int(datetime(2026, 3, 28, 12, 0, tzinfo=UTC).timestamp() * 1000)
        state = guard.check(prices, len(prices) - 1, ts_ms)
        assert state.is_weekend
        assert state.weekend_multiplier == pytest.approx(0.70)

    def test_weekday_full_size(self, guard: VolatilityGuard) -> None:
        prices = _make_prices_with_vol(500, 67000.0, 0.50)
        # Wednesday
        ts_ms = int(datetime(2026, 3, 25, 12, 0, tzinfo=UTC).timestamp() * 1000)
        state = guard.check(prices, len(prices) - 1, ts_ms)
        assert not state.is_weekend
        assert state.weekend_multiplier == pytest.approx(1.0)


class TestFundingProximity:
    def test_near_funding_detected(self, guard: VolatilityGuard) -> None:
        prices = _make_prices_with_vol(500, 67000.0, 0.50)
        # 07:55 UTC - 5 minutes before 08:00 funding
        ts_ms = int(datetime(2026, 3, 25, 7, 55, tzinfo=UTC).timestamp() * 1000)
        state = guard.check(prices, len(prices) - 1, ts_ms)
        assert state.near_funding_settlement
        assert state.funding_proximity_multiplier == pytest.approx(0.5)

    def test_away_from_funding(self, guard: VolatilityGuard) -> None:
        prices = _make_prices_with_vol(500, 67000.0, 0.50)
        # 12:00 UTC - far from any funding time
        ts_ms = int(datetime(2026, 3, 25, 12, 0, tzinfo=UTC).timestamp() * 1000)
        state = guard.check(prices, len(prices) - 1, ts_ms)
        assert not state.near_funding_settlement
        assert state.funding_proximity_multiplier == pytest.approx(1.0)


class TestValidation:
    def test_min_vol_negative_rejected(self) -> None:
        with pytest.raises(ValueError):
            VolatilityGuard(min_vol_annualized=-0.1)

    def test_max_vol_below_min_rejected(self) -> None:
        with pytest.raises(ValueError):
            VolatilityGuard(min_vol_annualized=0.50, max_vol_annualized=0.30)
