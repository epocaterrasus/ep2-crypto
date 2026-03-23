"""Execution simulator for backtesting.

Provides realistic trade execution modeling for the backtest engine:
  1. SlippageEstimator — sqrt-impact slippage (1-3 bps typical)
  2. FeeCalculator — taker/maker fees (configurable)
  3. LatencySimulator — delay between signal and execution (50-200ms)
  4. PartialFillSimulator — cap fills at fraction of bar volume
  5. FundingAccumulator — apply funding at 00/08/16 UTC settlement
  6. ExecutionSimulator — orchestrates all components

Each component uses its own seeded RNG for reproducibility.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import UTC

import numpy as np
import structlog

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BPS = 1e-4  # 1 basis point = 0.0001
BARS_PER_DAY = 288  # 5-min bars
BAR_DURATION_MS = 5 * 60 * 1000  # 5 minutes in milliseconds
FUNDING_SETTLEMENT_HOURS = (0, 8, 16)


# ---------------------------------------------------------------------------
# 1. Slippage Estimator
# ---------------------------------------------------------------------------
@dataclass
class SlippageEstimator:
    """Square-root impact slippage model.

    slippage_bps = base_bps + coefficient * sqrt(notional / depth) * vol_mult

    Typical range: 1-3 bps for $50K-$200K orders.
    """

    base_bps: float = 0.1
    impact_coefficient: float = 10.0
    impact_exponent: float = 0.5
    base_depth_usd: float = 3.3e6
    normal_vol_5m: float = 0.001
    noise_std_bps: float = 0.2

    def estimate(
        self,
        notional_usd: float,
        realized_vol_5m: float = 0.001,
        rng: np.random.Generator | None = None,
    ) -> float:
        """Estimate one-way slippage in bps.

        Args:
            notional_usd: Order size in USD.
            realized_vol_5m: Current 5-min realized volatility.
            rng: Random generator for noise (None = no noise).

        Returns:
            Slippage in basis points (always >= 0).
        """
        if notional_usd <= 0:
            return 0.0

        participation = notional_usd / self.base_depth_usd
        impact = self.impact_coefficient * (participation ** self.impact_exponent)

        vol_mult = 1.0
        if self.normal_vol_5m > 0:
            vol_ratio = realized_vol_5m / self.normal_vol_5m
            vol_mult = 1.0 + max(0.0, vol_ratio - 1.0)

        slippage = (self.base_bps + impact) * vol_mult

        if rng is not None and self.noise_std_bps > 0:
            noise = rng.normal(0.0, self.noise_std_bps)
            slippage += noise

        return max(0.0, slippage)


# ---------------------------------------------------------------------------
# 2. Fee Calculator
# ---------------------------------------------------------------------------
@dataclass
class FeeCalculator:
    """Exchange fee calculator with taker/maker distinction.

    Default: Binance VIP-0 USDT-M futures rates.
    """

    taker_bps: float = 4.0
    maker_bps: float = 2.0

    def entry_fee_bps(self, is_taker: bool = True) -> float:
        """Fee for opening a position, in bps."""
        return self.taker_bps if is_taker else self.maker_bps

    def exit_fee_bps(self, is_taker: bool = True) -> float:
        """Fee for closing a position, in bps."""
        return self.taker_bps if is_taker else self.maker_bps

    def round_trip_bps(
        self,
        entry_taker: bool = True,
        exit_taker: bool = True,
    ) -> float:
        """Total round-trip fees in bps."""
        return self.entry_fee_bps(entry_taker) + self.exit_fee_bps(exit_taker)

    def fee_usd(self, notional_usd: float, is_taker: bool = True) -> float:
        """Fee in USD for a single side."""
        rate = self.taker_bps if is_taker else self.maker_bps
        return notional_usd * rate * BPS


# ---------------------------------------------------------------------------
# 3. Latency Simulator
# ---------------------------------------------------------------------------
@dataclass
class LatencySimulator:
    """Models delay between signal generation and order execution.

    At 5-min bars, latency is expressed as probability of missing the
    intended bar open and executing at the next bar instead.

    With 50-200ms latency on a 5-min bar, the chance of missing is
    essentially zero — but we model it for correctness and for
    sub-bar execution timing.
    """

    mean_latency_ms: float = 100.0
    std_latency_ms: float = 30.0
    min_latency_ms: float = 20.0
    max_latency_ms: float = 500.0

    def sample_latency_ms(self, rng: np.random.Generator) -> float:
        """Sample a latency value in milliseconds."""
        lat = rng.normal(self.mean_latency_ms, self.std_latency_ms)
        return float(np.clip(lat, self.min_latency_ms, self.max_latency_ms))

    def bars_delayed(self, latency_ms: float) -> int:
        """How many additional bars of delay does this latency cause?

        For 5-min bars, any latency < 5min = 0 extra bars.
        Signal at bar t → always executes at bar t+1 open minimum.
        This adds extra delay on top of the mandatory 1-bar delay.
        """
        return int(latency_ms // BAR_DURATION_MS)

    def execution_price_adjustment_bps(
        self,
        latency_ms: float,
        bar_volatility_bps: float,
    ) -> float:
        """Estimate adverse price movement during latency period.

        During the latency window, price moves adversely (on average)
        proportional to sqrt(latency/bar_duration) * bar_vol.
        """
        if latency_ms <= 0 or bar_volatility_bps <= 0:
            return 0.0
        frac_of_bar = latency_ms / BAR_DURATION_MS
        return bar_volatility_bps * math.sqrt(frac_of_bar) * 0.5


# ---------------------------------------------------------------------------
# 4. Partial Fill Simulator
# ---------------------------------------------------------------------------
@dataclass
class PartialFillSimulator:
    """Caps order fills at a fraction of bar volume.

    Prevents unrealistic backtests where the strategy consumes
    more liquidity than available.
    """

    max_participation_rate: float = 0.10  # 10% of bar volume

    def max_fill_usd(self, bar_volume_usd: float) -> float:
        """Maximum fill size in USD for this bar."""
        return bar_volume_usd * self.max_participation_rate

    def fill_fraction(
        self,
        desired_notional_usd: float,
        bar_volume_usd: float,
    ) -> float:
        """What fraction of the desired order can be filled?

        Returns:
            Fraction between 0.0 and 1.0.
        """
        if desired_notional_usd <= 0 or bar_volume_usd <= 0:
            return 0.0
        max_fill = self.max_fill_usd(bar_volume_usd)
        return min(1.0, max_fill / desired_notional_usd)

    def adjusted_quantity(
        self,
        desired_quantity: float,
        desired_notional_usd: float,
        bar_volume_usd: float,
    ) -> float:
        """Adjust quantity for partial fill constraints."""
        frac = self.fill_fraction(desired_notional_usd, bar_volume_usd)
        return desired_quantity * frac


# ---------------------------------------------------------------------------
# 5. Funding Accumulator
# ---------------------------------------------------------------------------
@dataclass
class FundingAccumulator:
    """Tracks and applies funding rate payments at settlement times.

    Settlement occurs at 00:00, 08:00, 16:00 UTC on Binance.
    Longs pay shorts when funding > 0 (market overleveraged long).
    """

    settlement_hours: tuple[int, ...] = FUNDING_SETTLEMENT_HOURS

    def is_settlement_bar(
        self,
        bar_timestamp_ms: int,
        prev_bar_timestamp_ms: int,
    ) -> bool:
        """Check if a funding settlement occurs between prev_bar and current bar.

        A settlement at hour H occurs if prev_bar < H:00 <= current_bar.
        """
        from datetime import datetime

        prev_dt = datetime.fromtimestamp(prev_bar_timestamp_ms / 1000, tz=UTC)
        curr_dt = datetime.fromtimestamp(bar_timestamp_ms / 1000, tz=UTC)

        for day_offset in range(-1, 2):
            for hour in self.settlement_hours:
                settle_day = curr_dt.replace(
                    hour=hour, minute=0, second=0, microsecond=0
                )
                from datetime import timedelta

                settle_day = settle_day + timedelta(days=day_offset)
                if prev_dt < settle_day <= curr_dt:
                    return True
        return False

    def funding_payment(
        self,
        position_notional_usd: float,
        is_long: bool,
        funding_rate: float,
    ) -> float:
        """Calculate funding payment for one settlement.

        Args:
            position_notional_usd: Absolute notional of position.
            is_long: True if holding long.
            funding_rate: Current funding rate (e.g. 0.0001 = 0.01%).

        Returns:
            Payment in USD. Positive = cost to holder, negative = income.
        """
        if position_notional_usd <= 0:
            return 0.0
        # Longs pay when rate > 0, shorts receive
        sign = 1.0 if is_long else -1.0
        return position_notional_usd * funding_rate * sign


# ---------------------------------------------------------------------------
# 6. Execution Result
# ---------------------------------------------------------------------------
@dataclass
class ExecutionResult:
    """Result of simulating a trade execution."""

    executed: bool
    fill_price: float = 0.0
    fill_quantity: float = 0.0
    fill_notional_usd: float = 0.0
    slippage_bps: float = 0.0
    fee_bps: float = 0.0
    fee_usd: float = 0.0
    total_cost_bps: float = 0.0
    total_cost_usd: float = 0.0
    latency_ms: float = 0.0
    partial_fill_fraction: float = 1.0
    bars_delayed: int = 0


# ---------------------------------------------------------------------------
# 7. Execution Simulator (orchestrates all components)
# ---------------------------------------------------------------------------
@dataclass
class ExecutionSimulator:
    """Orchestrates slippage, fees, latency, partial fill for trade simulation.

    Usage:
        sim = ExecutionSimulator(seed=42)
        result = sim.simulate_entry(
            side="long",
            desired_quantity=0.05,
            price=100_000.0,
            bar_volume_usd=50_000_000.0,
        )
    """

    slippage: SlippageEstimator = field(default_factory=SlippageEstimator)
    fees: FeeCalculator = field(default_factory=FeeCalculator)
    latency: LatencySimulator = field(default_factory=LatencySimulator)
    partial_fill: PartialFillSimulator = field(default_factory=PartialFillSimulator)
    funding: FundingAccumulator = field(default_factory=FundingAccumulator)
    seed: int = 42

    def __post_init__(self) -> None:
        # Separate RNG streams for each component
        base_rng = np.random.SeedSequence(self.seed)
        child_seeds = base_rng.spawn(3)
        self._slippage_rng = np.random.default_rng(child_seeds[0])
        self._latency_rng = np.random.default_rng(child_seeds[1])
        self._misc_rng = np.random.default_rng(child_seeds[2])

    def simulate_entry(
        self,
        side: str,
        desired_quantity: float,
        price: float,
        bar_volume_usd: float = 50_000_000.0,
        realized_vol_5m: float = 0.001,
        is_taker: bool = True,
    ) -> ExecutionResult:
        """Simulate opening a position.

        Args:
            side: "long" or "short".
            desired_quantity: Desired position size in BTC.
            price: Execution price (next bar open).
            bar_volume_usd: Volume of the execution bar in USD.
            realized_vol_5m: Current 5-min realized volatility.
            is_taker: Whether this is a taker (market) order.

        Returns:
            ExecutionResult with fill details and costs.
        """
        if desired_quantity <= 0 or price <= 0:
            return ExecutionResult(executed=False)

        desired_notional = desired_quantity * price

        # 1. Partial fill check
        fill_frac = self.partial_fill.fill_fraction(desired_notional, bar_volume_usd)
        if fill_frac <= 0:
            return ExecutionResult(executed=False, partial_fill_fraction=0.0)

        actual_quantity = desired_quantity * fill_frac
        actual_notional = actual_quantity * price

        # 2. Slippage
        slip_bps = self.slippage.estimate(
            actual_notional, realized_vol_5m, self._slippage_rng
        )

        # 3. Latency
        lat_ms = self.latency.sample_latency_ms(self._latency_rng)
        extra_bars = self.latency.bars_delayed(lat_ms)

        # 4. Fees
        fee_bps = self.fees.entry_fee_bps(is_taker)
        fee_usd = self.fees.fee_usd(actual_notional, is_taker)

        # 5. Adjusted fill price (slippage moves price against us)
        slip_fraction = slip_bps * BPS
        if side == "long":
            fill_price = price * (1.0 + slip_fraction)
        else:
            fill_price = price * (1.0 - slip_fraction)

        total_cost_bps = slip_bps + fee_bps
        total_cost_usd = actual_notional * total_cost_bps * BPS

        return ExecutionResult(
            executed=True,
            fill_price=fill_price,
            fill_quantity=actual_quantity,
            fill_notional_usd=actual_notional,
            slippage_bps=slip_bps,
            fee_bps=fee_bps,
            fee_usd=fee_usd,
            total_cost_bps=total_cost_bps,
            total_cost_usd=total_cost_usd,
            latency_ms=lat_ms,
            partial_fill_fraction=fill_frac,
            bars_delayed=extra_bars,
        )

    def simulate_exit(
        self,
        side: str,
        quantity: float,
        price: float,
        bar_volume_usd: float = 50_000_000.0,
        realized_vol_5m: float = 0.001,
        is_taker: bool = True,
    ) -> ExecutionResult:
        """Simulate closing a position. Slippage works against the exit."""
        if quantity <= 0 or price <= 0:
            return ExecutionResult(executed=False)

        notional = quantity * price

        # Slippage
        slip_bps = self.slippage.estimate(
            notional, realized_vol_5m, self._slippage_rng
        )

        # Fees
        fee_bps = self.fees.exit_fee_bps(is_taker)
        fee_usd = self.fees.fee_usd(notional, is_taker)

        # Exit slippage: closing a long means selling (price moves down)
        slip_fraction = slip_bps * BPS
        if side == "long":
            fill_price = price * (1.0 - slip_fraction)
        else:
            fill_price = price * (1.0 + slip_fraction)

        total_cost_bps = slip_bps + fee_bps
        total_cost_usd = notional * total_cost_bps * BPS

        return ExecutionResult(
            executed=True,
            fill_price=fill_price,
            fill_quantity=quantity,
            fill_notional_usd=notional,
            slippage_bps=slip_bps,
            fee_bps=fee_bps,
            fee_usd=fee_usd,
            total_cost_bps=total_cost_bps,
            total_cost_usd=total_cost_usd,
            latency_ms=0.0,
            partial_fill_fraction=1.0,
            bars_delayed=0,
        )

    def estimate_round_trip_cost_bps(
        self,
        notional_usd: float,
        realized_vol_5m: float = 0.001,
    ) -> float:
        """Quick estimate of total round-trip cost in bps (no RNG noise)."""
        entry_slip = self.slippage.estimate(notional_usd, realized_vol_5m)
        exit_slip = self.slippage.estimate(notional_usd, realized_vol_5m)
        fees = self.fees.round_trip_bps()
        return entry_slip + exit_slip + fees
