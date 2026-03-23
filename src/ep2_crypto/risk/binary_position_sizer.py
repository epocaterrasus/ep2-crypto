"""Position sizing for binary prediction markets (Polymarket).

Implements Kelly criterion for binary outcomes where:
- You buy shares at price p (market-implied probability)
- Your model estimates true probability q
- Payout: $1 if correct, $0 if wrong
- Fee: applied to profit only (Polymarket's fee structure)

Kelly fraction: f* = (q - p) / (1 - p)
Quarter-Kelly: f_practical = 0.25 * f*
"""

from __future__ import annotations

from dataclasses import dataclass

import structlog

logger = structlog.get_logger(__name__)

# Polymarket fee formula constants (as of 2026)
DEFAULT_FEE_RATE: float = 0.02
DEFAULT_FEE_EXPONENT: float = 2.0
MIN_BET_USD: float = 5.0
DEFAULT_MAX_BET_FRACTION: float = 0.05  # 5% of bankroll
DEFAULT_KELLY_FRACTION: float = 0.25  # Quarter-Kelly


@dataclass(frozen=True)
class BinarySizingResult:
    """Result of binary position sizing calculation."""

    shares: float  # Number of shares to buy
    cost_usd: float  # Total cost (shares * price)
    max_loss_usd: float  # = cost_usd (binary: lose entire premium)
    expected_value: float  # EV per share after fees
    kelly_fraction: float  # Raw Kelly fraction
    adjusted_kelly: float  # After quarter-Kelly + caps
    fee_per_share: float  # Estimated fee per share if winning
    rejected: bool = False
    rejection_reason: str | None = None


def polymarket_fee(
    price: float, fee_rate: float = DEFAULT_FEE_RATE, exponent: float = DEFAULT_FEE_EXPONENT
) -> float:
    """Calculate Polymarket fee rate for a given share price.

    Fee is charged on profits only. The dynamic scaling factor is
    (p * (1-p))^exponent, which peaks at p=0.50 and drops at extremes.

    Args:
        price: Share price (0-1), representing implied probability.
        fee_rate: Base fee rate (default 0.02 = 2%).
        exponent: Scaling exponent (default 2.0).

    Returns:
        Effective fee as a fraction of profit.
    """
    if price <= 0.0 or price >= 1.0:
        return 0.0
    return float(fee_rate * (price * (1.0 - price)) ** exponent)


def binary_kelly(
    model_prob: float,
    market_price: float,
    fee_rate: float = DEFAULT_FEE_RATE,
    fee_exponent: float = DEFAULT_FEE_EXPONENT,
) -> float:
    """Compute Kelly fraction for a binary outcome bet.

    Args:
        model_prob: Model's estimated probability of the outcome (q).
        market_price: Market price of the share (p).
        fee_rate: Polymarket fee rate.
        fee_exponent: Polymarket fee exponent.

    Returns:
        Optimal Kelly fraction of bankroll to bet. Negative means don't bet.
    """
    if market_price <= 0.0 or market_price >= 1.0:
        return 0.0
    if model_prob <= 0.0 or model_prob >= 1.0:
        return 0.0

    q = model_prob
    p = market_price
    r = polymarket_fee(p, fee_rate, fee_exponent)

    # Fee-adjusted Kelly for binary outcome:
    # Win: receive (1-p)(1-r) net profit per share
    # Lose: lose p per share
    # f* = [q*(1-p)*(1-r) - (1-q)*p] / [(1-p)*(1-r)]
    win_payoff = (1.0 - p) * (1.0 - r)
    lose_cost = p

    numerator = q * win_payoff - (1.0 - q) * lose_cost
    denominator = win_payoff

    if denominator <= 0.0:
        return 0.0

    return numerator / denominator


def expected_value_per_share(
    model_prob: float,
    market_price: float,
    fee_rate: float = DEFAULT_FEE_RATE,
    fee_exponent: float = DEFAULT_FEE_EXPONENT,
) -> float:
    """Expected value per share of a binary bet.

    Args:
        model_prob: Model's estimated probability (q).
        market_price: Share price (p).
        fee_rate: Fee rate.
        fee_exponent: Fee exponent.

    Returns:
        Expected profit per share (can be negative).
    """
    if market_price <= 0.0 or market_price >= 1.0:
        return 0.0

    q = model_prob
    p = market_price
    r = polymarket_fee(p, fee_rate, fee_exponent)

    return q * (1.0 - p) * (1.0 - r) - (1.0 - q) * p


class BinaryPositionSizer:
    """Position sizer for binary prediction market outcomes.

    Computes bet size using fee-adjusted Kelly criterion with:
    - Configurable Kelly fraction (default quarter-Kelly = 0.25)
    - Maximum bet cap (default 5% of bankroll)
    - Minimum bet enforcement ($5 Polymarket minimum)
    - Drawdown gate integration
    """

    def __init__(
        self,
        kelly_fraction: float = DEFAULT_KELLY_FRACTION,
        max_bet_fraction: float = DEFAULT_MAX_BET_FRACTION,
        min_bet_usd: float = MIN_BET_USD,
        fee_rate: float = DEFAULT_FEE_RATE,
        fee_exponent: float = DEFAULT_FEE_EXPONENT,
    ) -> None:
        if not 0.0 < kelly_fraction <= 1.0:
            raise ValueError(f"kelly_fraction must be in (0, 1], got {kelly_fraction}")
        if not 0.0 < max_bet_fraction <= 1.0:
            raise ValueError(f"max_bet_fraction must be in (0, 1], got {max_bet_fraction}")

        self.kelly_fraction = kelly_fraction
        self.max_bet_fraction = max_bet_fraction
        self.min_bet_usd = min_bet_usd
        self.fee_rate = fee_rate
        self.fee_exponent = fee_exponent

    def compute(
        self,
        model_prob: float,
        market_price: float,
        bankroll: float,
        drawdown_multiplier: float = 1.0,
    ) -> BinarySizingResult:
        """Compute position size for a binary bet.

        Args:
            model_prob: Model's probability estimate for this outcome (0-1).
            market_price: Current market price of the share (0-1).
            bankroll: Current bankroll in USD.
            drawdown_multiplier: From drawdown gate (0-1), reduces size.

        Returns:
            BinarySizingResult with shares, cost, and sizing details.
        """
        # Validate inputs
        if bankroll <= 0.0:
            return self._reject("bankroll_zero_or_negative")

        if market_price <= 0.0 or market_price >= 1.0:
            return self._reject(f"invalid_market_price_{market_price:.4f}")

        if model_prob <= 0.0 or model_prob >= 1.0:
            return self._reject(f"invalid_model_prob_{model_prob:.4f}")

        # No edge — model agrees with market or is worse
        if model_prob <= market_price:
            return self._reject("no_edge_model_prob_leq_market_price")

        # Compute raw Kelly
        raw_kelly = binary_kelly(model_prob, market_price, self.fee_rate, self.fee_exponent)

        if raw_kelly <= 0.0:
            return self._reject("negative_kelly_after_fees")

        # Apply fractional Kelly
        adjusted = raw_kelly * self.kelly_fraction

        # Apply drawdown gate
        adjusted *= max(0.0, min(1.0, drawdown_multiplier))

        # Cap at max bet fraction
        adjusted = min(adjusted, self.max_bet_fraction)

        # Convert to USD
        bet_usd = adjusted * bankroll

        # Enforce minimum
        if bet_usd < self.min_bet_usd:
            return self._reject(f"bet_below_minimum_{bet_usd:.2f}_lt_{self.min_bet_usd:.2f}")

        # Convert to shares
        shares = bet_usd / market_price

        # Compute EV and fee
        ev = expected_value_per_share(model_prob, market_price, self.fee_rate, self.fee_exponent)
        fee = polymarket_fee(market_price, self.fee_rate, self.fee_exponent)

        result = BinarySizingResult(
            shares=shares,
            cost_usd=bet_usd,
            max_loss_usd=bet_usd,
            expected_value=ev,
            kelly_fraction=raw_kelly,
            adjusted_kelly=adjusted,
            fee_per_share=fee,
        )

        logger.info(
            "binary_position_sized",
            model_prob=round(model_prob, 4),
            market_price=round(market_price, 4),
            raw_kelly=round(raw_kelly, 4),
            adjusted_kelly=round(adjusted, 4),
            shares=round(shares, 2),
            cost_usd=round(bet_usd, 2),
            ev_per_share=round(ev, 4),
        )

        return result

    def _reject(self, reason: str) -> BinarySizingResult:
        logger.debug("binary_sizing_rejected", reason=reason)
        return BinarySizingResult(
            shares=0.0,
            cost_usd=0.0,
            max_loss_usd=0.0,
            expected_value=0.0,
            kelly_fraction=0.0,
            adjusted_kelly=0.0,
            fee_per_share=0.0,
            rejected=True,
            rejection_reason=reason,
        )
