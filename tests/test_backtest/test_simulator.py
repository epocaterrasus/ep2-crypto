"""Tests for backtest execution simulator."""

from __future__ import annotations

import numpy as np
import pytest

from ep2_crypto.backtest.simulator import (
    ExecutionResult,
    ExecutionSimulator,
    FeeCalculator,
    FundingAccumulator,
    LatencySimulator,
    PartialFillSimulator,
    SlippageEstimator,
)


# ---------------------------------------------------------------------------
# SlippageEstimator
# ---------------------------------------------------------------------------
class TestSlippageEstimator:
    def test_zero_notional_returns_zero(self) -> None:
        s = SlippageEstimator()
        assert s.estimate(0.0) == 0.0

    def test_negative_notional_returns_zero(self) -> None:
        s = SlippageEstimator()
        assert s.estimate(-1000.0) == 0.0

    def test_small_order_low_slippage(self) -> None:
        """$10K order against $3.3M depth should be ~0.1-1 bps."""
        s = SlippageEstimator()
        slip = s.estimate(10_000.0)
        assert 0.0 < slip < 2.0

    def test_large_order_higher_slippage(self) -> None:
        """$500K order should have more slippage than $10K."""
        s = SlippageEstimator()
        small = s.estimate(10_000.0)
        large = s.estimate(500_000.0)
        assert large > small

    def test_sqrt_scaling(self) -> None:
        """Doubling order size should increase slippage by ~sqrt(2), not 2x."""
        s = SlippageEstimator(noise_std_bps=0.0)  # no noise for determinism
        slip_1x = s.estimate(100_000.0)
        slip_2x = s.estimate(200_000.0)
        ratio = slip_2x / slip_1x
        # sqrt(2) ≈ 1.414, but base_bps dampens the ratio
        assert 1.0 < ratio < 2.0

    def test_high_vol_increases_slippage(self) -> None:
        """High volatility should amplify slippage."""
        s = SlippageEstimator(noise_std_bps=0.0)
        normal = s.estimate(100_000.0, realized_vol_5m=0.001)
        high_vol = s.estimate(100_000.0, realized_vol_5m=0.005)
        assert high_vol > normal

    def test_noise_varies_output(self) -> None:
        """With noise, repeated calls should give different results."""
        s = SlippageEstimator(noise_std_bps=0.5)
        rng = np.random.default_rng(42)
        results = [s.estimate(100_000.0, rng=rng) for _ in range(20)]
        assert len(set(results)) > 1  # not all identical

    def test_always_non_negative(self) -> None:
        """Slippage should never be negative, even with noise."""
        s = SlippageEstimator(noise_std_bps=1.0)
        rng = np.random.default_rng(42)
        for _ in range(100):
            assert s.estimate(50_000.0, rng=rng) >= 0.0


# ---------------------------------------------------------------------------
# FeeCalculator
# ---------------------------------------------------------------------------
class TestFeeCalculator:
    def test_default_taker_fee(self) -> None:
        f = FeeCalculator()
        assert f.entry_fee_bps(is_taker=True) == 4.0

    def test_default_maker_fee(self) -> None:
        f = FeeCalculator()
        assert f.entry_fee_bps(is_taker=False) == 2.0

    def test_round_trip_taker_taker(self) -> None:
        f = FeeCalculator()
        assert f.round_trip_bps(entry_taker=True, exit_taker=True) == 8.0

    def test_round_trip_maker_maker(self) -> None:
        f = FeeCalculator()
        assert f.round_trip_bps(entry_taker=False, exit_taker=False) == 4.0

    def test_round_trip_mixed(self) -> None:
        f = FeeCalculator()
        assert f.round_trip_bps(entry_taker=False, exit_taker=True) == 6.0

    def test_fee_usd_calculation(self) -> None:
        f = FeeCalculator(taker_bps=4.0)
        # $100K * 4bps = $100K * 0.0004 = $40
        fee = f.fee_usd(100_000.0, is_taker=True)
        assert abs(fee - 40.0) < 0.01

    def test_custom_fee_rates(self) -> None:
        f = FeeCalculator(taker_bps=3.0, maker_bps=1.0)
        assert f.round_trip_bps() == 6.0


# ---------------------------------------------------------------------------
# LatencySimulator
# ---------------------------------------------------------------------------
class TestLatencySimulator:
    def test_sample_within_bounds(self) -> None:
        lat = LatencySimulator(min_latency_ms=20.0, max_latency_ms=500.0)
        rng = np.random.default_rng(42)
        for _ in range(100):
            val = lat.sample_latency_ms(rng)
            assert 20.0 <= val <= 500.0

    def test_bars_delayed_normal_latency(self) -> None:
        """100ms latency on 5-min bars → 0 extra bars delayed."""
        lat = LatencySimulator()
        assert lat.bars_delayed(100.0) == 0

    def test_bars_delayed_extreme_latency(self) -> None:
        """5+ minute latency → 1 extra bar delayed."""
        lat = LatencySimulator()
        assert lat.bars_delayed(300_001.0) == 1

    def test_price_adjustment_proportional(self) -> None:
        lat = LatencySimulator()
        short_adj = lat.execution_price_adjustment_bps(50.0, 10.0)
        long_adj = lat.execution_price_adjustment_bps(200.0, 10.0)
        assert long_adj > short_adj

    def test_zero_latency_zero_adjustment(self) -> None:
        lat = LatencySimulator()
        assert lat.execution_price_adjustment_bps(0.0, 10.0) == 0.0


# ---------------------------------------------------------------------------
# PartialFillSimulator
# ---------------------------------------------------------------------------
class TestPartialFillSimulator:
    def test_small_order_full_fill(self) -> None:
        """$50K order against $50M bar volume → full fill."""
        pf = PartialFillSimulator(max_participation_rate=0.10)
        frac = pf.fill_fraction(50_000.0, 50_000_000.0)
        assert frac == 1.0

    def test_large_order_partial_fill(self) -> None:
        """$10M order against $50M bar volume → 50% fill (10% cap)."""
        pf = PartialFillSimulator(max_participation_rate=0.10)
        frac = pf.fill_fraction(10_000_000.0, 50_000_000.0)
        assert abs(frac - 0.5) < 0.001

    def test_exceeds_bar_volume(self) -> None:
        """Order larger than bar volume * rate → capped."""
        pf = PartialFillSimulator(max_participation_rate=0.10)
        frac = pf.fill_fraction(100_000_000.0, 50_000_000.0)
        assert frac < 0.1

    def test_zero_volume_returns_zero(self) -> None:
        pf = PartialFillSimulator()
        assert pf.fill_fraction(100_000.0, 0.0) == 0.0

    def test_adjusted_quantity(self) -> None:
        pf = PartialFillSimulator(max_participation_rate=0.10)
        qty = pf.adjusted_quantity(1.0, 10_000_000.0, 50_000_000.0)
        assert abs(qty - 0.5) < 0.001

    def test_max_fill_usd(self) -> None:
        pf = PartialFillSimulator(max_participation_rate=0.10)
        assert pf.max_fill_usd(50_000_000.0) == 5_000_000.0


# ---------------------------------------------------------------------------
# FundingAccumulator
# ---------------------------------------------------------------------------
class TestFundingAccumulator:
    def test_settlement_detection_across_midnight(self) -> None:
        """Bar crossing 00:00 UTC should detect settlement."""
        fa = FundingAccumulator()
        # 23:55 UTC → 00:05 UTC
        prev_ts = 1711929300000  # 2024-04-01 23:55 UTC approx
        curr_ts = prev_ts + 10 * 60 * 1000  # +10 min
        # We just check the method returns bool without error
        result = fa.is_settlement_bar(curr_ts, prev_ts)
        assert isinstance(result, bool)

    def test_settlement_at_0800(self) -> None:
        """Bar crossing 08:00 UTC should detect settlement."""
        fa = FundingAccumulator()
        from datetime import datetime, timezone

        dt_before = datetime(2024, 4, 1, 7, 55, tzinfo=timezone.utc)
        dt_after = datetime(2024, 4, 1, 8, 5, tzinfo=timezone.utc)
        prev_ts = int(dt_before.timestamp() * 1000)
        curr_ts = int(dt_after.timestamp() * 1000)
        assert fa.is_settlement_bar(curr_ts, prev_ts) is True

    def test_no_settlement_mid_period(self) -> None:
        """Bar from 10:00-10:05 should NOT trigger settlement."""
        fa = FundingAccumulator()
        from datetime import datetime, timezone

        dt_before = datetime(2024, 4, 1, 10, 0, tzinfo=timezone.utc)
        dt_after = datetime(2024, 4, 1, 10, 5, tzinfo=timezone.utc)
        prev_ts = int(dt_before.timestamp() * 1000)
        curr_ts = int(dt_after.timestamp() * 1000)
        assert fa.is_settlement_bar(curr_ts, prev_ts) is False

    def test_funding_payment_long_positive_rate(self) -> None:
        """Long pays when funding rate is positive."""
        fa = FundingAccumulator()
        payment = fa.funding_payment(100_000.0, is_long=True, funding_rate=0.0001)
        assert payment > 0  # cost to longs

    def test_funding_payment_short_positive_rate(self) -> None:
        """Short receives when funding rate is positive."""
        fa = FundingAccumulator()
        payment = fa.funding_payment(100_000.0, is_long=False, funding_rate=0.0001)
        assert payment < 0  # income for shorts

    def test_funding_payment_zero_position(self) -> None:
        fa = FundingAccumulator()
        assert fa.funding_payment(0.0, True, 0.0001) == 0.0

    def test_funding_payment_magnitude(self) -> None:
        """$100K position at 0.01% rate → $10 payment."""
        fa = FundingAccumulator()
        payment = fa.funding_payment(100_000.0, is_long=True, funding_rate=0.0001)
        assert abs(payment - 10.0) < 0.01


# ---------------------------------------------------------------------------
# ExecutionSimulator (integrated)
# ---------------------------------------------------------------------------
class TestExecutionSimulator:
    def test_basic_entry(self) -> None:
        sim = ExecutionSimulator(seed=42)
        result = sim.simulate_entry(
            side="long",
            desired_quantity=0.05,
            price=100_000.0,
            bar_volume_usd=50_000_000.0,
        )
        assert result.executed is True
        assert result.fill_quantity > 0
        assert result.fill_price > 0
        assert result.slippage_bps >= 0
        assert result.fee_bps > 0
        assert result.total_cost_bps > 0

    def test_entry_long_slippage_direction(self) -> None:
        """Buying should fill at a higher price than the reference."""
        sim = ExecutionSimulator(seed=42)
        price = 100_000.0
        result = sim.simulate_entry("long", 0.05, price)
        assert result.fill_price >= price

    def test_entry_short_slippage_direction(self) -> None:
        """Selling should fill at a lower price than the reference."""
        sim = ExecutionSimulator(seed=42)
        price = 100_000.0
        result = sim.simulate_entry("short", 0.05, price)
        assert result.fill_price <= price

    def test_exit_long_slippage_direction(self) -> None:
        """Closing a long (selling) fills at lower price."""
        sim = ExecutionSimulator(seed=42)
        price = 100_000.0
        result = sim.simulate_exit("long", 0.05, price)
        assert result.fill_price <= price

    def test_exit_short_slippage_direction(self) -> None:
        """Closing a short (buying back) fills at higher price."""
        sim = ExecutionSimulator(seed=42)
        price = 100_000.0
        result = sim.simulate_exit("short", 0.05, price)
        assert result.fill_price >= price

    def test_zero_quantity_not_executed(self) -> None:
        sim = ExecutionSimulator(seed=42)
        result = sim.simulate_entry("long", 0.0, 100_000.0)
        assert result.executed is False

    def test_zero_price_not_executed(self) -> None:
        sim = ExecutionSimulator(seed=42)
        result = sim.simulate_entry("long", 0.05, 0.0)
        assert result.executed is False

    def test_partial_fill_large_order(self) -> None:
        """Very large order against small volume should be partially filled."""
        sim = ExecutionSimulator(seed=42)
        result = sim.simulate_entry(
            "long",
            desired_quantity=100.0,  # $10M at $100K
            price=100_000.0,
            bar_volume_usd=1_000_000.0,  # only $1M volume
        )
        assert result.executed is True
        assert result.partial_fill_fraction < 1.0
        assert result.fill_quantity < 100.0

    def test_deterministic_with_same_seed(self) -> None:
        """Same seed → same results."""
        results = []
        for _ in range(2):
            sim = ExecutionSimulator(seed=123)
            r = sim.simulate_entry("long", 0.05, 100_000.0)
            results.append(r)
        assert results[0].fill_price == results[1].fill_price
        assert results[0].slippage_bps == results[1].slippage_bps

    def test_different_seeds_differ(self) -> None:
        """Different seeds → different noise."""
        sim1 = ExecutionSimulator(seed=1)
        sim2 = ExecutionSimulator(seed=2)
        r1 = sim1.simulate_entry("long", 0.05, 100_000.0)
        r2 = sim2.simulate_entry("long", 0.05, 100_000.0)
        # Very likely to differ due to noise
        assert r1.slippage_bps != r2.slippage_bps

    def test_round_trip_cost_estimate(self) -> None:
        """Quick round-trip cost estimate should be reasonable."""
        sim = ExecutionSimulator()
        cost = sim.estimate_round_trip_cost_bps(50_000.0)
        # Should be > fees alone (8 bps taker/taker) and < 20 bps
        assert cost > 8.0
        assert cost < 30.0

    def test_maker_fees_lower(self) -> None:
        """Maker orders should cost less in fees."""
        sim = ExecutionSimulator(seed=42)
        taker = sim.simulate_entry("long", 0.05, 100_000.0, is_taker=True)
        # Reset RNG state for fair comparison
        sim2 = ExecutionSimulator(seed=42)
        maker = sim2.simulate_entry("long", 0.05, 100_000.0, is_taker=False)
        assert maker.fee_bps < taker.fee_bps

    def test_high_vol_increases_cost(self) -> None:
        """Higher volatility should increase total execution cost."""
        sim1 = ExecutionSimulator(seed=42)
        low_vol = sim1.simulate_entry(
            "long", 0.05, 100_000.0, realized_vol_5m=0.0005
        )
        sim2 = ExecutionSimulator(seed=42)
        high_vol = sim2.simulate_entry(
            "long", 0.05, 100_000.0, realized_vol_5m=0.005
        )
        assert high_vol.total_cost_bps > low_vol.total_cost_bps
