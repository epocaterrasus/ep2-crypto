"""
Transaction Cost Engine for Crypto Backtesting.

Provides realistic, parameterized cost estimation for backtesting crypto
perpetual futures strategies. Every component is independently configurable
for sensitivity analysis.

Components:
    1. FeeModel - Exchange-specific, VIP-tier-aware fees
    2. SlippageModel - Order-size and volatility-dependent slippage
    3. MarketImpactModel - Square-root law permanent + temporary impact
    4. FundingRateModel - 8-hour funding accumulation
    5. SpreadModel - Time-of-day and volatility-aware spread costs
    6. OpportunityCostModel - Risk-free rate hurdle
    7. MakerTakerModel - Fill probability and rebate modeling
    8. TransactionCostEngine - Combines all components

References:
    - Almgren & Chriss (2000): Optimal Execution of Portfolio Transactions
    - Bouchaud et al.: The Square-Root Law of Market Impact
    - BitMEX Q3 2025 Derivatives Report: Funding Rate Analysis
    - Amberdata: The Rhythm of Liquidity
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums & Constants
# ---------------------------------------------------------------------------

class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    POST_ONLY = "post_only"


class Exchange(Enum):
    BINANCE = "binance"
    BYBIT = "bybit"


class VolatilityRegime(Enum):
    LOW = "low"
    NORMAL = "normal"
    ELEVATED = "elevated"
    HIGH = "high"
    EXTREME = "extreme"


# Basis point conversion
BPS = 1e-4  # 1 bps = 0.01% = 0.0001


# ---------------------------------------------------------------------------
# 1. Fee Model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VIPTier:
    """Fee rates for a specific VIP tier."""
    tier: int
    maker_rate: float  # as decimal (e.g., 0.0002 for 2 bps)
    taker_rate: float
    min_volume_30d: float  # minimum 30-day volume in USDT
    min_balance: float  # minimum balance requirement (BNB or USDT)


# Binance USDT-M Futures VIP tiers (as of Q1 2026)
BINANCE_VIP_TIERS = [
    VIPTier(0, 0.000200, 0.000500, 0, 0),
    VIPTier(1, 0.000160, 0.000400, 15_000_000, 25),  # 25 BNB
    VIPTier(2, 0.000140, 0.000350, 100_000_000, 100),
    VIPTier(3, 0.000120, 0.000320, 500_000_000, 250),
    VIPTier(4, 0.000100, 0.000300, 1_000_000_000, 500),
    VIPTier(5, 0.000080, 0.000270, 3_000_000_000, 1_000),
    VIPTier(6, 0.000060, 0.000250, 5_000_000_000, 1_750),
    VIPTier(7, 0.000040, 0.000220, 10_000_000_000, 3_000),
    VIPTier(8, 0.000020, 0.000200, 25_000_000_000, 4_500),
    VIPTier(9, 0.000000, 0.000170, 50_000_000_000, 5_500),
]

# Bybit Perpetual Futures VIP tiers (as of Q1 2026)
BYBIT_VIP_TIERS = [
    VIPTier(0, 0.000200, 0.000550, 0, 0),
    VIPTier(1, 0.000180, 0.000400, 10_000_000, 100_000),
    VIPTier(2, 0.000160, 0.000375, 25_000_000, 250_000),
    VIPTier(3, 0.000140, 0.000350, 50_000_000, 500_000),
    VIPTier(4, 0.000120, 0.000320, 100_000_000, 1_000_000),
    VIPTier(5, 0.000100, 0.000300, 250_000_000, 2_000_000),
]


@dataclass
class FeeModel:
    """
    Exchange-specific, VIP-tier-aware fee calculation.

    Supports dynamic tier inference from projected volume,
    BNB discount, and referral rebates.
    """
    exchange: Exchange = Exchange.BINANCE
    vip_tier: int = 0
    bnb_discount: bool = False  # 10% off on Binance
    referral_rebate_pct: float = 0.0  # 0-0.10 (10% max)

    def _get_tier_data(self) -> VIPTier:
        tiers = (
            BINANCE_VIP_TIERS
            if self.exchange == Exchange.BINANCE
            else BYBIT_VIP_TIERS
        )
        for tier in tiers:
            if tier.tier == self.vip_tier:
                return tier
        return tiers[0]

    @classmethod
    def from_projected_volume(
        cls,
        exchange: Exchange,
        monthly_volume: float,
        bnb_discount: bool = False,
        referral_rebate_pct: float = 0.0,
    ) -> FeeModel:
        """Infer VIP tier from projected 30-day trading volume."""
        tiers = (
            BINANCE_VIP_TIERS
            if exchange == Exchange.BINANCE
            else BYBIT_VIP_TIERS
        )
        best_tier = 0
        for tier in tiers:
            if monthly_volume >= tier.min_volume_30d:
                best_tier = tier.tier
        return cls(
            exchange=exchange,
            vip_tier=best_tier,
            bnb_discount=bnb_discount,
            referral_rebate_pct=referral_rebate_pct,
        )

    def get_fee_rate(self, order_type: OrderType) -> float:
        """
        Return the fee rate as a decimal.

        Returns:
            Fee rate (e.g., 0.0005 for 5 bps).
        """
        tier = self._get_tier_data()
        if order_type in (OrderType.LIMIT, OrderType.POST_ONLY):
            rate = tier.maker_rate
        else:
            rate = tier.taker_rate

        # BNB discount (Binance only, 10% off)
        if self.bnb_discount and self.exchange == Exchange.BINANCE:
            rate *= 0.90

        # Referral rebate
        rate *= (1.0 - self.referral_rebate_pct)

        return rate

    def calculate_fee(
        self,
        notional: float,
        order_type: OrderType,
    ) -> float:
        """Calculate fee in USD for a given notional value."""
        return notional * self.get_fee_rate(order_type)

    def round_trip_fee(
        self,
        notional: float,
        entry_type: OrderType = OrderType.MARKET,
        exit_type: OrderType = OrderType.MARKET,
    ) -> float:
        """Calculate total round-trip fee in USD."""
        return (
            self.calculate_fee(notional, entry_type)
            + self.calculate_fee(notional, exit_type)
        )

    def round_trip_fee_bps(
        self,
        entry_type: OrderType = OrderType.MARKET,
        exit_type: OrderType = OrderType.MARKET,
    ) -> float:
        """Calculate round-trip fee in basis points."""
        return (
            self.get_fee_rate(entry_type) + self.get_fee_rate(exit_type)
        ) / BPS


# ---------------------------------------------------------------------------
# 2. Slippage Model
# ---------------------------------------------------------------------------

# Typical BTCUSDT perp depth at 10 bps from mid, by hour (USD millions)
# Based on Amberdata research, BTC/FDUSD Binance 2025
HOURLY_DEPTH_PROFILE = {
    0: 3.45, 1: 3.40, 2: 3.30, 3: 3.30, 4: 3.30, 5: 3.35,
    6: 3.45, 7: 3.55, 8: 3.68, 9: 3.78, 10: 3.78, 11: 3.86,
    12: 3.78, 13: 3.68, 14: 3.61, 15: 3.55, 16: 3.45, 17: 3.30,
    18: 3.23, 19: 3.13, 20: 2.97, 21: 2.71, 22: 2.90, 23: 3.13,
}

# Mean depth for normalization
_MEAN_DEPTH = sum(HOURLY_DEPTH_PROFILE.values()) / 24


@dataclass
class SlippageModel:
    """
    Dynamic slippage model based on order size, book depth,
    volatility, and time-of-day.

    Formula:
        slippage = (spread/2 + impact) * volatility_multiplier * time_multiplier

    Where:
        impact = impact_coefficient * (order_size / book_depth) ^ impact_exponent
    """
    # Base parameters
    impact_coefficient: float = 1.0  # calibrated impact scaling
    impact_exponent: float = 0.5  # square-root by default
    base_depth_usd: float = 3.3e6  # typical depth at 10bps ($3.3M)

    # Volatility scaling
    normal_vol_5m: float = 0.001  # normal 5-min volatility (0.1%)
    vol_multiplier_scale: float = 1.0  # how much vol affects slippage

    # Time-of-day adjustment
    use_intraday_profile: bool = True

    def _volatility_multiplier(self, realized_vol_5m: float) -> float:
        """Scale slippage by current vs normal volatility."""
        vol_ratio = realized_vol_5m / self.normal_vol_5m if self.normal_vol_5m > 0 else 1.0
        return 1.0 + self.vol_multiplier_scale * max(0, vol_ratio - 1.0)

    def _time_of_day_multiplier(self, hour_utc: int) -> float:
        """Adjust slippage for time-of-day liquidity patterns."""
        if not self.use_intraday_profile:
            return 1.0
        depth_at_hour = HOURLY_DEPTH_PROFILE.get(hour_utc % 24, _MEAN_DEPTH)
        # Inverse relationship: less depth = more slippage
        return _MEAN_DEPTH / depth_at_hour

    def classify_volatility_regime(
        self, realized_vol: float
    ) -> VolatilityRegime:
        """Classify current volatility into a regime."""
        ratio = realized_vol / self.normal_vol_5m if self.normal_vol_5m > 0 else 1.0
        if ratio < 0.5:
            return VolatilityRegime.LOW
        if ratio < 1.5:
            return VolatilityRegime.NORMAL
        if ratio < 3.0:
            return VolatilityRegime.ELEVATED
        if ratio < 5.0:
            return VolatilityRegime.HIGH
        return VolatilityRegime.EXTREME

    def estimate_slippage_bps(
        self,
        order_size_usd: float,
        spread_bps: float = 0.2,
        realized_vol_5m: float = 0.001,
        hour_utc: int = 12,
    ) -> float:
        """
        Estimate slippage in basis points.

        Args:
            order_size_usd: Order notional in USD.
            spread_bps: Current bid-ask spread in bps.
            realized_vol_5m: Current 5-minute realized volatility.
            hour_utc: Current hour (0-23 UTC) for intraday adjustment.

        Returns:
            Estimated slippage in basis points (one-way).
        """
        half_spread = spread_bps / 2.0

        # Impact component: square-root scaling with depth
        # impact_bps = coefficient * sqrt(order_size / depth) * 10000
        # This models: if you consume X% of visible depth, slippage scales
        # as sqrt of that fraction, converted to basis points.
        # Calibration: $50K against $3.3M depth -> sqrt(0.015) * coeff
        #   = 0.123 * 1.0 = 0.123 -> ~1.23 bps (reasonable for $50K)
        time_mult = self._time_of_day_multiplier(hour_utc)
        effective_depth = self.base_depth_usd / time_mult  # less depth = more impact
        participation_rate = order_size_usd / effective_depth
        # Result is in "fraction" units; multiply by 1e4 to get bps
        impact = (
            self.impact_coefficient
            * (participation_rate ** self.impact_exponent)
            * 10.0  # scaling factor calibrated so $50K ~ 1-2 bps
        )

        # Volatility multiplier
        vol_mult = self._volatility_multiplier(realized_vol_5m)

        total_slippage = (half_spread + impact) * vol_mult
        return max(0.0, total_slippage)

    def estimate_slippage_usd(
        self,
        order_size_usd: float,
        spread_bps: float = 0.2,
        realized_vol_5m: float = 0.001,
        hour_utc: int = 12,
    ) -> float:
        """Estimate slippage in USD."""
        bps = self.estimate_slippage_bps(
            order_size_usd, spread_bps, realized_vol_5m, hour_utc
        )
        return order_size_usd * bps * BPS


# ---------------------------------------------------------------------------
# 3. Market Impact Model (Almgren-Chriss / Square-Root Law)
# ---------------------------------------------------------------------------

@dataclass
class MarketImpactModel:
    """
    Market impact using the universal square-root law.

    I(Q) = Y * sigma * sqrt(Q / V)

    Decomposes into permanent and temporary components:
        permanent = permanent_frac * I(Q)  (~2/3)
        temporary = (1 - permanent_frac) * I(Q)  (~1/3)

    References:
        - Bouchaud et al., "The Square-Root Law of Market Impact"
        - Almgren & Chriss (2000)
    """
    # Square-root law parameters
    y_prefactor: float = 1.0  # dimensionless, empirically ~1.0
    daily_volume_usd: float = 20e9  # avg daily volume ($20B for BTCUSDT perp)
    daily_volatility: float = 0.02  # daily vol (2% typical for BTC)

    # Permanent vs temporary split
    permanent_fraction: float = 0.667  # 2/3 permanent, 1/3 temporary

    # Temporary impact decay (exponential, in minutes)
    temporary_decay_halflife_min: float = 5.0

    def total_impact_fraction(self, order_size_usd: float) -> float:
        """
        Calculate total market impact as a fraction of price.

        Args:
            order_size_usd: Total order size in USD.

        Returns:
            Impact as a fraction (e.g., 0.0005 for 5 bps).
        """
        if self.daily_volume_usd <= 0:
            logger.warning("Daily volume is zero or negative; returning 0 impact")
            return 0.0
        participation = order_size_usd / self.daily_volume_usd
        return self.y_prefactor * self.daily_volatility * math.sqrt(participation)

    def total_impact_bps(self, order_size_usd: float) -> float:
        """Calculate total market impact in basis points."""
        return self.total_impact_fraction(order_size_usd) / BPS

    def permanent_impact_bps(self, order_size_usd: float) -> float:
        """Permanent component of impact (persists after execution)."""
        return self.total_impact_bps(order_size_usd) * self.permanent_fraction

    def temporary_impact_bps(self, order_size_usd: float) -> float:
        """Temporary component of impact (decays after execution)."""
        return self.total_impact_bps(order_size_usd) * (1 - self.permanent_fraction)

    def temporary_impact_at_time(
        self, order_size_usd: float, minutes_after: float
    ) -> float:
        """
        Temporary impact remaining after `minutes_after` minutes.

        Decays exponentially with configurable half-life.
        """
        temp = self.temporary_impact_bps(order_size_usd)
        decay = math.exp(
            -math.log(2) * minutes_after / self.temporary_decay_halflife_min
        )
        return temp * decay

    def impact_cost_usd(self, order_size_usd: float) -> float:
        """Total impact cost in USD."""
        return order_size_usd * self.total_impact_fraction(order_size_usd)

    def with_volatility(self, new_daily_vol: float) -> MarketImpactModel:
        """Return a copy with updated volatility."""
        return MarketImpactModel(
            y_prefactor=self.y_prefactor,
            daily_volume_usd=self.daily_volume_usd,
            daily_volatility=new_daily_vol,
            permanent_fraction=self.permanent_fraction,
            temporary_decay_halflife_min=self.temporary_decay_halflife_min,
        )


# ---------------------------------------------------------------------------
# 4. Funding Rate Model
# ---------------------------------------------------------------------------

@dataclass
class FundingRateModel:
    """
    Model funding rate costs for positions held across 8-hour boundaries.

    Uses empirical distribution parameters from BitMEX Q3 2025 report
    and Binance historical data.
    """
    # Empirical parameters (Binance BTC, per 8h period)
    mean_rate: float = 0.000057  # 0.0057% per 8h
    std_rate: float = 0.000039  # 0.0039% per 8h
    median_rate: float = 0.0001  # 0.01% per 8h (mode)

    # Clamp bounds (exchange-level)
    max_rate: float = 0.0005  # 0.05% per 8h
    min_rate: float = -0.0005  # -0.05% per 8h

    # Settlement times (hours UTC)
    settlement_hours: tuple[int, ...] = (0, 8, 16)

    # For simulation: use mean or sample from distribution
    use_stochastic: bool = False

    def funding_payments_in_period(
        self,
        entry_hour_utc: float,
        duration_hours: float,
    ) -> int:
        """
        Count how many funding settlements occur during a position hold.

        Args:
            entry_hour_utc: Entry time as fractional hour (e.g., 14.5 = 2:30 PM).
            duration_hours: How long the position is held.

        Returns:
            Number of funding payments incurred.
        """
        exit_hour = entry_hour_utc + duration_hours
        count = 0
        for day_offset in range(int(duration_hours / 24) + 2):
            for settlement_h in self.settlement_hours:
                settlement_time = day_offset * 24 + settlement_h
                # Position must be OPEN at the settlement time
                # (entry < settlement <= exit, approximately)
                if entry_hour_utc < settlement_time <= exit_hour:
                    count += 1
        return count

    def expected_funding_cost(
        self,
        position_value: float,
        is_long: bool,
        duration_hours: float,
        entry_hour_utc: float = 0.0,
    ) -> float:
        """
        Calculate expected funding cost in USD.

        Positive return = cost to the trader.
        Negative return = income (shorts receive when funding is positive).

        Args:
            position_value: Notional position value in USD.
            is_long: True for long, False for short.
            duration_hours: Expected hold duration in hours.
            entry_hour_utc: Entry time as fractional hour UTC.

        Returns:
            Expected funding cost in USD (positive = cost, negative = income).
        """
        n_payments = self.funding_payments_in_period(
            entry_hour_utc, duration_hours
        )
        if n_payments == 0:
            return 0.0

        if self.use_stochastic:
            # Sample from truncated normal
            rng = np.random.default_rng()
            rates = rng.normal(self.mean_rate, self.std_rate, size=n_payments)
            rates = np.clip(rates, self.min_rate, self.max_rate)
            total_rate = float(np.sum(rates))
        else:
            total_rate = n_payments * self.mean_rate

        # Longs pay positive funding, shorts receive it
        direction_sign = 1.0 if is_long else -1.0
        return position_value * total_rate * direction_sign

    def expected_funding_cost_bps(
        self,
        is_long: bool,
        duration_hours: float,
        entry_hour_utc: float = 0.0,
    ) -> float:
        """Expected funding cost in basis points."""
        cost_frac = self.expected_funding_cost(
            1.0, is_long, duration_hours, entry_hour_utc
        )
        return cost_frac / BPS

    def annualized_funding_rate(self) -> float:
        """Annualized funding rate based on mean per-period rate."""
        daily_rate = self.mean_rate * 3  # 3 settlements per day
        return daily_rate * 365


# ---------------------------------------------------------------------------
# 5. Spread Model
# ---------------------------------------------------------------------------

@dataclass
class SpreadModel:
    """
    Model bid-ask spread as a function of time-of-day, volatility,
    and market conditions.

    BTCUSDT perp on Binance: tick size = $0.10
    At BTC=$100K: min spread = 0.1 bps
    """
    base_spread_bps: float = 0.2  # typical median spread
    volatility_sensitivity: float = 0.5  # how much spread tracks vol (0-1)
    normal_vol_5m: float = 0.001  # baseline 5-min vol
    use_intraday_profile: bool = True

    # Spread multiplier by hour (relative to average)
    # Derived from inverse of depth profile
    _SPREAD_HOURLY: dict[int, float] = field(default_factory=lambda: {
        0: 1.00, 1: 1.00, 2: 1.02, 3: 1.02, 4: 1.02, 5: 1.00,
        6: 0.98, 7: 0.95, 8: 0.92, 9: 0.90, 10: 0.88, 11: 0.85,
        12: 0.87, 13: 0.88, 14: 0.90, 15: 0.92, 16: 0.95, 17: 1.00,
        18: 1.02, 19: 1.05, 20: 1.10, 21: 1.20, 22: 1.12, 23: 1.05,
    })

    def estimate_spread_bps(
        self,
        realized_vol_5m: float = 0.001,
        hour_utc: int = 12,
    ) -> float:
        """
        Estimate current bid-ask spread in basis points.

        Args:
            realized_vol_5m: Current 5-minute realized volatility.
            hour_utc: Current hour (0-23 UTC).

        Returns:
            Estimated spread in bps.
        """
        spread = self.base_spread_bps

        # Volatility adjustment
        if self.normal_vol_5m > 0:
            vol_ratio = realized_vol_5m / self.normal_vol_5m
            vol_adj = 1.0 + self.volatility_sensitivity * max(0, vol_ratio - 1.0)
            spread *= vol_adj

        # Time-of-day adjustment
        if self.use_intraday_profile:
            spread *= self._SPREAD_HOURLY.get(hour_utc % 24, 1.0)

        return max(0.1, spread)  # floor at 1 tick for BTC at $100K

    def half_spread_cost_bps(
        self,
        realized_vol_5m: float = 0.001,
        hour_utc: int = 12,
    ) -> float:
        """Half-spread cost (what you pay to cross) in bps."""
        return self.estimate_spread_bps(realized_vol_5m, hour_utc) / 2.0


# ---------------------------------------------------------------------------
# 6. Opportunity Cost Model
# ---------------------------------------------------------------------------

@dataclass
class OpportunityCostModel:
    """
    Model the opportunity cost of capital deployed to the exchange.

    Capital sitting on an exchange at 0% yield has an implicit cost
    equal to the risk-free rate it could earn elsewhere.
    """
    annual_risk_free_rate: float = 0.045  # 4.5% (stablecoin lending / T-bills)
    leverage: float = 1.0  # effective leverage (reduces capital needed)

    def daily_hurdle_rate(self) -> float:
        """Daily return needed to beat the risk-free rate."""
        return self.annual_risk_free_rate / 365

    def period_opportunity_cost_bps(self, holding_hours: float) -> float:
        """
        Opportunity cost for a given holding period in bps.

        This is the return you need to generate just to match
        the risk-free alternative.
        """
        daily_cost = self.annual_risk_free_rate / 365
        hourly_cost = daily_cost / 24
        # Adjust for leverage: only 1/leverage of capital is actual margin
        adjusted_cost = hourly_cost / self.leverage if self.leverage > 0 else hourly_cost
        return adjusted_cost * holding_hours / BPS

    def annual_opportunity_cost_usd(self, capital: float) -> float:
        """Annual opportunity cost in USD for deployed capital."""
        effective_capital = capital / self.leverage if self.leverage > 0 else capital
        return effective_capital * self.annual_risk_free_rate


# ---------------------------------------------------------------------------
# 7. Maker/Taker Fill Probability Model
# ---------------------------------------------------------------------------

@dataclass
class MakerTakerModel:
    """
    Model the probability of getting maker fills vs falling back to taker.

    Incorporates:
    - Fill probability as a function of patience/urgency
    - Adverse selection cost on maker fills
    - Expected blended fee rate
    """
    # Fill probability parameters
    maker_fill_prob_patient: float = 0.85  # mean-reversion, no urgency
    maker_fill_prob_moderate: float = 0.60  # some urgency
    maker_fill_prob_urgent: float = 0.10  # momentum, high urgency

    # Adverse selection on maker fills (bps)
    adverse_selection_bps: float = 1.0  # avg adverse move after maker fill

    def blended_fee_rate(
        self,
        fee_model: FeeModel,
        urgency: str = "moderate",
    ) -> float:
        """
        Calculate expected blended fee rate given fill probability.

        Args:
            fee_model: The exchange fee model.
            urgency: One of "patient", "moderate", "urgent".

        Returns:
            Expected fee rate as a decimal.
        """
        prob_map = {
            "patient": self.maker_fill_prob_patient,
            "moderate": self.maker_fill_prob_moderate,
            "urgent": self.maker_fill_prob_urgent,
        }
        p_maker = prob_map.get(urgency, self.maker_fill_prob_moderate)

        maker_rate = fee_model.get_fee_rate(OrderType.POST_ONLY)
        taker_rate = fee_model.get_fee_rate(OrderType.MARKET)

        return p_maker * maker_rate + (1 - p_maker) * taker_rate

    def effective_maker_cost_bps(
        self,
        fee_model: FeeModel,
    ) -> float:
        """
        True cost of a maker fill including adverse selection.

        Even with a lower/zero fee, maker fills have adverse selection.
        """
        maker_fee_bps = fee_model.get_fee_rate(OrderType.POST_ONLY) / BPS
        return maker_fee_bps + self.adverse_selection_bps

    def maker_vs_taker_advantage_bps(
        self,
        fee_model: FeeModel,
    ) -> float:
        """
        Net advantage of maker over taker (can be negative).

        If negative, you're better off with taker orders despite higher fees.
        """
        effective_maker = self.effective_maker_cost_bps(fee_model)
        taker_bps = fee_model.get_fee_rate(OrderType.MARKET) / BPS
        return taker_bps - effective_maker  # positive = maker is better


# ---------------------------------------------------------------------------
# 8. Transaction Cost Engine (Combines All Components)
# ---------------------------------------------------------------------------

@dataclass
class TradeDetails:
    """Input details for a single trade."""
    side: OrderSide
    notional_usd: float
    order_type: OrderType = OrderType.MARKET
    is_entry: bool = True
    holding_hours: float = 0.0  # expected hold duration (for funding/opp cost)
    hour_utc: int = 12
    realized_vol_5m: float = 0.001
    current_spread_bps: Optional[float] = None  # if None, model estimates it
    is_long_position: bool = True  # for funding direction


@dataclass
class CostBreakdown:
    """Detailed breakdown of all cost components."""
    fee_bps: float
    spread_bps: float
    slippage_bps: float
    market_impact_bps: float
    funding_bps: float
    opportunity_cost_bps: float
    total_bps: float

    fee_usd: float
    spread_usd: float
    slippage_usd: float
    market_impact_usd: float
    funding_usd: float
    opportunity_cost_usd: float
    total_usd: float

    # Metadata
    volatility_regime: VolatilityRegime = VolatilityRegime.NORMAL

    def to_dict(self) -> dict[str, float | str]:
        return {
            "fee_bps": self.fee_bps,
            "spread_bps": self.spread_bps,
            "slippage_bps": self.slippage_bps,
            "market_impact_bps": self.market_impact_bps,
            "funding_bps": self.funding_bps,
            "opportunity_cost_bps": self.opportunity_cost_bps,
            "total_bps": self.total_bps,
            "fee_usd": self.fee_usd,
            "spread_usd": self.spread_usd,
            "slippage_usd": self.slippage_usd,
            "market_impact_usd": self.market_impact_usd,
            "funding_usd": self.funding_usd,
            "opportunity_cost_usd": self.opportunity_cost_usd,
            "total_usd": self.total_usd,
            "volatility_regime": self.volatility_regime.value,
        }


@dataclass
class TransactionCostEngine:
    """
    Unified cost engine that combines all cost components.

    Usage:
        engine = TransactionCostEngine()  # conservative defaults
        cost = engine.estimate_trade_cost(trade_details)
        print(f"Total cost: {cost.total_bps:.1f} bps = ${cost.total_usd:.2f}")
    """
    fee_model: FeeModel = field(default_factory=FeeModel)
    slippage_model: SlippageModel = field(default_factory=SlippageModel)
    market_impact_model: MarketImpactModel = field(default_factory=MarketImpactModel)
    funding_model: FundingRateModel = field(default_factory=FundingRateModel)
    spread_model: SpreadModel = field(default_factory=SpreadModel)
    opportunity_cost_model: OpportunityCostModel = field(
        default_factory=OpportunityCostModel
    )
    maker_taker_model: MakerTakerModel = field(default_factory=MakerTakerModel)

    # Master switches for disabling components in sensitivity analysis
    include_fees: bool = True
    include_spread: bool = True
    include_slippage: bool = True
    include_market_impact: bool = True
    include_funding: bool = True
    include_opportunity_cost: bool = True

    def estimate_trade_cost(self, trade: TradeDetails) -> CostBreakdown:
        """
        Estimate all costs for a single trade.

        Args:
            trade: Details of the trade to cost.

        Returns:
            CostBreakdown with itemized costs in both bps and USD.
        """
        notional = trade.notional_usd

        # 1. Exchange fees
        fee_rate = 0.0
        if self.include_fees:
            fee_rate = self.fee_model.get_fee_rate(trade.order_type)
        fee_usd = notional * fee_rate
        fee_bps = fee_rate / BPS

        # 2. Spread cost
        spread_bps_val = 0.0
        if self.include_spread:
            if trade.current_spread_bps is not None:
                spread_bps_val = trade.current_spread_bps / 2.0
            else:
                spread_bps_val = self.spread_model.half_spread_cost_bps(
                    trade.realized_vol_5m, trade.hour_utc
                )
        spread_usd = notional * spread_bps_val * BPS

        # 3. Slippage
        slippage_bps_val = 0.0
        if self.include_slippage:
            effective_spread = (
                trade.current_spread_bps
                if trade.current_spread_bps is not None
                else self.spread_model.estimate_spread_bps(
                    trade.realized_vol_5m, trade.hour_utc
                )
            )
            slippage_bps_val = self.slippage_model.estimate_slippage_bps(
                trade.notional_usd,
                effective_spread,
                trade.realized_vol_5m,
                trade.hour_utc,
            )
        slippage_usd = notional * slippage_bps_val * BPS

        # 4. Market impact
        impact_bps_val = 0.0
        if self.include_market_impact:
            impact_bps_val = self.market_impact_model.total_impact_bps(
                trade.notional_usd
            )
        impact_usd = notional * impact_bps_val * BPS

        # 5. Funding (only for positions held across funding boundaries)
        funding_bps_val = 0.0
        funding_usd = 0.0
        if self.include_funding and trade.holding_hours > 0:
            funding_usd = self.funding_model.expected_funding_cost(
                notional,
                trade.is_long_position,
                trade.holding_hours,
                float(trade.hour_utc),
            )
            funding_bps_val = abs(funding_usd) / (notional * BPS) if notional > 0 else 0.0
            # Keep sign: positive = cost, negative = income
            if funding_usd < 0:
                funding_bps_val = -funding_bps_val

        # 6. Opportunity cost
        opp_bps_val = 0.0
        if self.include_opportunity_cost and trade.holding_hours > 0:
            opp_bps_val = self.opportunity_cost_model.period_opportunity_cost_bps(
                trade.holding_hours
            )
        opp_usd = notional * opp_bps_val * BPS

        # Volatility regime classification
        vol_regime = self.slippage_model.classify_volatility_regime(
            trade.realized_vol_5m
        )

        # Total
        total_bps = (
            fee_bps + spread_bps_val + slippage_bps_val
            + impact_bps_val + funding_bps_val + opp_bps_val
        )
        total_usd = (
            fee_usd + spread_usd + slippage_usd
            + impact_usd + funding_usd + opp_usd
        )

        return CostBreakdown(
            fee_bps=fee_bps,
            spread_bps=spread_bps_val,
            slippage_bps=slippage_bps_val,
            market_impact_bps=impact_bps_val,
            funding_bps=funding_bps_val,
            opportunity_cost_bps=opp_bps_val,
            total_bps=total_bps,
            fee_usd=fee_usd,
            spread_usd=spread_usd,
            slippage_usd=slippage_usd,
            market_impact_usd=impact_usd,
            funding_usd=funding_usd,
            opportunity_cost_usd=opp_usd,
            total_usd=total_usd,
            volatility_regime=vol_regime,
        )

    def estimate_round_trip_cost(
        self,
        notional_usd: float,
        entry_type: OrderType = OrderType.MARKET,
        exit_type: OrderType = OrderType.MARKET,
        holding_hours: float = 2.0,
        is_long: bool = True,
        hour_utc: int = 12,
        realized_vol_5m: float = 0.001,
    ) -> CostBreakdown:
        """
        Estimate total round-trip cost (entry + exit + holding costs).

        Convenience method that creates entry and exit trades and sums costs.
        """
        entry = self.estimate_trade_cost(TradeDetails(
            side=OrderSide.BUY if is_long else OrderSide.SELL,
            notional_usd=notional_usd,
            order_type=entry_type,
            is_entry=True,
            holding_hours=holding_hours,
            hour_utc=hour_utc,
            realized_vol_5m=realized_vol_5m,
            is_long_position=is_long,
        ))

        exit_trade = self.estimate_trade_cost(TradeDetails(
            side=OrderSide.SELL if is_long else OrderSide.BUY,
            notional_usd=notional_usd,
            order_type=exit_type,
            is_entry=False,
            holding_hours=0.0,  # funding already counted on entry
            hour_utc=(hour_utc + int(holding_hours)) % 24,
            realized_vol_5m=realized_vol_5m,
            is_long_position=is_long,
        ))

        return CostBreakdown(
            fee_bps=entry.fee_bps + exit_trade.fee_bps,
            spread_bps=entry.spread_bps + exit_trade.spread_bps,
            slippage_bps=entry.slippage_bps + exit_trade.slippage_bps,
            market_impact_bps=entry.market_impact_bps + exit_trade.market_impact_bps,
            funding_bps=entry.funding_bps,  # only counted once
            opportunity_cost_bps=entry.opportunity_cost_bps,
            total_bps=(
                entry.fee_bps + exit_trade.fee_bps
                + entry.spread_bps + exit_trade.spread_bps
                + entry.slippage_bps + exit_trade.slippage_bps
                + entry.market_impact_bps + exit_trade.market_impact_bps
                + entry.funding_bps + entry.opportunity_cost_bps
            ),
            fee_usd=entry.fee_usd + exit_trade.fee_usd,
            spread_usd=entry.spread_usd + exit_trade.spread_usd,
            slippage_usd=entry.slippage_usd + exit_trade.slippage_usd,
            market_impact_usd=entry.market_impact_usd + exit_trade.market_impact_usd,
            funding_usd=entry.funding_usd,
            opportunity_cost_usd=entry.opportunity_cost_usd,
            total_usd=(
                entry.fee_usd + exit_trade.fee_usd
                + entry.spread_usd + exit_trade.spread_usd
                + entry.slippage_usd + exit_trade.slippage_usd
                + entry.market_impact_usd + exit_trade.market_impact_usd
                + entry.funding_usd + entry.opportunity_cost_usd
            ),
            volatility_regime=entry.volatility_regime,
        )


# ---------------------------------------------------------------------------
# 9. Sensitivity Analysis Tools
# ---------------------------------------------------------------------------

@dataclass
class BreakEvenAnalysis:
    """
    Break-even analysis for strategy viability.

    Given strategy statistics, determine maximum sustainable cost
    and minimum win rate at a given cost level.
    """

    @staticmethod
    def break_even_cost_bps(
        win_rate: float,
        avg_win_bps: float,
        avg_loss_bps: float,
        avg_funding_cost_bps: float = 0.0,
    ) -> float:
        """
        Maximum round-trip cost the strategy can sustain.

        Formula:
            C_rt = W * avg_win - (1-W) * avg_loss - funding

        Args:
            win_rate: Fraction of winning trades (0-1).
            avg_win_bps: Average profit on winners (bps, before costs).
            avg_loss_bps: Average loss on losers (bps, positive number).
            avg_funding_cost_bps: Average funding cost per trade (bps).

        Returns:
            Break-even round-trip cost in basis points.
        """
        gross_edge = win_rate * avg_win_bps - (1 - win_rate) * avg_loss_bps
        return gross_edge - avg_funding_cost_bps

    @staticmethod
    def min_win_rate(
        avg_win_bps: float,
        avg_loss_bps: float,
        round_trip_cost_bps: float,
    ) -> float:
        """
        Minimum win rate to break even at a given cost level.

        Formula:
            W_min = (avg_loss + C_rt) / (avg_win + avg_loss)

        Returns:
            Minimum win rate as a fraction (0-1).
        """
        denominator = avg_win_bps + avg_loss_bps
        if denominator <= 0:
            return 1.0  # impossible to break even
        return (avg_loss_bps + round_trip_cost_bps) / denominator

    @staticmethod
    def net_profit_factor(
        win_rate: float,
        avg_win_bps: float,
        avg_loss_bps: float,
        round_trip_cost_bps: float,
    ) -> float:
        """
        Profit factor after costs.

        Formula:
            PF = W * (avg_win - C_rt) / ((1-W) * (avg_loss + C_rt))

        Returns:
            Net profit factor. > 1.0 is profitable.
        """
        gross_win = avg_win_bps - round_trip_cost_bps
        gross_loss = avg_loss_bps + round_trip_cost_bps
        numerator = win_rate * gross_win
        denominator = (1 - win_rate) * gross_loss
        if denominator <= 0:
            return float("inf") if numerator > 0 else 0.0
        return numerator / denominator


def cost_sensitivity_sweep(
    engine: TransactionCostEngine,
    notional_usd: float = 50_000,
    holding_hours: float = 2.0,
    cost_multipliers: Optional[list[float]] = None,
) -> list[dict[str, float]]:
    """
    Sweep cost assumptions from optimistic to pessimistic.

    Returns a list of cost estimates at different multiplier levels,
    useful for plotting the "cost frontier".

    Args:
        engine: The base cost engine.
        notional_usd: Trade size.
        holding_hours: Hold duration.
        cost_multipliers: List of multipliers (1.0 = base case).

    Returns:
        List of dicts with multiplier and total cost.
    """
    if cost_multipliers is None:
        cost_multipliers = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]

    results = []
    base = engine.estimate_round_trip_cost(
        notional_usd=notional_usd,
        holding_hours=holding_hours,
    )

    for mult in cost_multipliers:
        results.append({
            "multiplier": mult,
            "total_bps": base.total_bps * mult,
            "total_usd": base.total_usd * mult,
            "label": _multiplier_label(mult),
        })

    return results


def _multiplier_label(mult: float) -> str:
    if mult == 0.0:
        return "zero_cost"
    if mult <= 0.5:
        return "optimistic"
    if mult <= 1.0:
        return "base_case"
    if mult <= 2.0:
        return "conservative"
    return "stress_test"


# ---------------------------------------------------------------------------
# Preset Configurations
# ---------------------------------------------------------------------------

def conservative_engine(
    exchange: Exchange = Exchange.BINANCE,
) -> TransactionCostEngine:
    """
    Conservative cost engine -- use this as default for backtesting.

    Assumes:
    - VIP 0 (no volume discounts)
    - All taker orders
    - Normal market conditions
    - Includes all cost components
    """
    return TransactionCostEngine(
        fee_model=FeeModel(exchange=exchange, vip_tier=0),
        slippage_model=SlippageModel(),
        market_impact_model=MarketImpactModel(),
        funding_model=FundingRateModel(),
        spread_model=SpreadModel(),
        opportunity_cost_model=OpportunityCostModel(),
    )


def optimistic_engine(
    exchange: Exchange = Exchange.BINANCE,
    vip_tier: int = 3,
) -> TransactionCostEngine:
    """
    Optimistic cost engine -- for established traders with VIP status.

    Assumes:
    - VIP 3+ with BNB discount
    - Mix of maker/taker
    - Normal market conditions
    """
    return TransactionCostEngine(
        fee_model=FeeModel(
            exchange=exchange,
            vip_tier=vip_tier,
            bnb_discount=True,
        ),
        slippage_model=SlippageModel(impact_coefficient=0.7),
        market_impact_model=MarketImpactModel(),
        funding_model=FundingRateModel(),
        spread_model=SpreadModel(base_spread_bps=0.15),
        opportunity_cost_model=OpportunityCostModel(leverage=3.0),
    )


def stress_test_engine(
    exchange: Exchange = Exchange.BINANCE,
) -> TransactionCostEngine:
    """
    Stress test engine -- worst-case scenario.

    Assumes:
    - VIP 0
    - All taker
    - Elevated volatility (2x normal)
    - Wider spreads
    - Thin books
    """
    return TransactionCostEngine(
        fee_model=FeeModel(exchange=exchange, vip_tier=0),
        slippage_model=SlippageModel(
            impact_coefficient=2.0,
            base_depth_usd=1.5e6,  # half normal depth
        ),
        market_impact_model=MarketImpactModel(
            daily_volatility=0.04,  # 2x normal vol
        ),
        funding_model=FundingRateModel(
            mean_rate=0.0002,  # elevated funding
        ),
        spread_model=SpreadModel(
            base_spread_bps=0.5,  # wider spreads
        ),
        opportunity_cost_model=OpportunityCostModel(),
    )
