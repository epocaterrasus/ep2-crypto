"""Position sizing engine: Kelly criterion, ATR stops, max caps.

Computes position size in BTC given a signal, confidence, and risk state.
Uses quarter-Kelly with confidence scaling and drawdown gate multiplier.

Pipeline:
    1. Compute Kelly fraction: f = (WR * payoff - (1-WR)) / payoff
    2. Apply quarter-Kelly:    f_actual = f * 0.25
    3. Apply confidence scale: f_actual *= confidence
    4. Apply drawdown gate:    f_actual *= dd_multiplier
    5. Apply weekend/funding:  f_actual *= time_multiplier
    6. Apply max cap:          f_actual = min(f_actual, max_position_fraction)
    7. Convert to BTC qty:     qty = (f_actual * equity) / price
    8. Enforce minimum size:   qty = max(qty, min_btc_quantity)
    9. Compute ATR stop:       stop_distance = atr_multiplier * ATR(14)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import structlog

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = structlog.get_logger(__name__)


@dataclass
class SizingResult:
    """Complete output of position sizing computation."""

    # Position
    quantity_btc: float  # BTC amount to trade
    notional_usd: float  # Dollar value of position
    position_fraction: float  # Fraction of equity

    # Stop loss
    stop_distance_usd: float  # Absolute stop distance in USD
    stop_price: float  # Exact stop price
    risk_per_trade_usd: float  # Max loss if stopped out

    # Kelly internals (for logging/debugging)
    raw_kelly_fraction: float
    quarter_kelly_fraction: float
    confidence_scaled: float
    drawdown_scaled: float
    final_fraction: float

    # Rejection
    rejected: bool = False
    rejection_reason: str | None = None


class PositionSizer:
    """Computes position sizes using fractional Kelly criterion.

    CRITICAL: Monte Carlo simulation (RR-risk-capital-preservation-math.md) shows:
        - At 5% risk/trade over 18,250 trades/year: E[max_DD] = 66.7%
        - At 1% risk/trade: E[max_DD] = 18.4%
        - At 0.5% risk/trade: E[max_DD] = 9.6%
    Therefore max_risk_per_trade defaults to 1% and MUST NOT exceed 5%.

    Configuration:
        kelly_fraction: fraction of Kelly to use (default 0.25 = quarter-Kelly)
        max_position_fraction: absolute cap on position NOTIONAL (default 0.05 = 5% of equity)
        max_risk_per_trade: absolute cap on RISK per trade (default 0.01 = 1% of equity)
            Most important param. Caps: qty * stop_dist <= max_risk * equity
        min_btc_quantity: exchange minimum order size (default 0.001)
        atr_multiplier: ATR multiple for stop loss (default 3.0)
        atr_period: bars for ATR computation (default 14)
        max_holding_bars: force exit after this many bars (default 6 = 30min)
        default_win_rate: used when no historical data (default 0.52)
        default_payoff_ratio: avg win / avg loss (default 1.0)
    """

    def __init__(
        self,
        kelly_fraction: float = 0.25,
        max_position_fraction: float = 0.05,
        max_risk_per_trade: float = 0.01,
        min_btc_quantity: float = 0.001,
        atr_multiplier: float = 3.0,
        atr_period: int = 14,
        max_holding_bars: int = 6,
        default_win_rate: float = 0.52,
        default_payoff_ratio: float = 1.0,
    ) -> None:
        if not (0 < kelly_fraction <= 1):
            raise ValueError(f"kelly_fraction must be in (0, 1], got {kelly_fraction}")
        if not (0 < max_position_fraction <= 1):
            raise ValueError(
                f"max_position_fraction must be in (0, 1], got {max_position_fraction}"
            )
        if not (0 < max_risk_per_trade <= 0.05):
            raise ValueError(
                f"max_risk_per_trade must be in (0, 0.05], got {max_risk_per_trade}. "
                f"Monte Carlo shows >1% risk/trade leads to 66%+ drawdowns over a year."
            )
        if atr_multiplier <= 0:
            raise ValueError(f"atr_multiplier must be positive, got {atr_multiplier}")
        if atr_period < 2:
            raise ValueError(f"atr_period must be >= 2, got {atr_period}")

        self._kelly_frac = kelly_fraction
        self._max_frac = max_position_fraction
        self._max_risk = max_risk_per_trade
        self._min_btc = min_btc_quantity
        self._atr_mult = atr_multiplier
        self._atr_period = atr_period
        self._max_holding = max_holding_bars
        self._default_wr = default_win_rate
        self._default_payoff = default_payoff_ratio

    # -- Public API -----------------------------------------------------------

    @property
    def max_holding_bars(self) -> int:
        """Maximum bars to hold a position before force-exit."""
        return self._max_holding

    def compute(
        self,
        signal_direction: str,
        confidence: float,
        equity: float,
        current_price: float,
        highs: NDArray[np.float64],
        lows: NDArray[np.float64],
        closes: NDArray[np.float64],
        current_idx: int,
        drawdown_multiplier: float = 1.0,
        time_multiplier: float = 1.0,
        win_rate: float | None = None,
        payoff_ratio: float | None = None,
    ) -> SizingResult:
        """Compute position size for a signal.

        Args:
            signal_direction: "long" or "short"
            confidence: Model confidence score (0.0 to 1.0)
            equity: Current total equity in USD
            current_price: Current BTC price
            highs: Array of high prices for ATR computation
            lows: Array of low prices for ATR computation
            closes: Array of close prices for ATR computation
            current_idx: Index in the price arrays
            drawdown_multiplier: From drawdown gate (0.0 to 1.0)
            time_multiplier: Weekend/funding adjustment (0.0 to 1.0)
            win_rate: Historical win rate (None = use default)
            payoff_ratio: Historical avg_win/avg_loss (None = use default)

        Returns:
            SizingResult with quantity, stop price, and sizing breakdown.
        """
        if signal_direction not in ("long", "short"):
            return self._rejected(f"Invalid direction: {signal_direction}")
        if confidence <= 0:
            return self._rejected(f"Confidence too low: {confidence:.4f}")
        if equity <= 0:
            return self._rejected(f"Equity non-positive: {equity:.2f}")
        if current_price <= 0:
            return self._rejected(f"Price non-positive: {current_price}")
        if drawdown_multiplier <= 0:
            return self._rejected("Drawdown gate halted trading (multiplier = 0)")

        wr = win_rate if win_rate is not None else self._default_wr
        pr = payoff_ratio if payoff_ratio is not None else self._default_payoff

        # 1. Kelly fraction: f = (WR * payoff - (1 - WR)) / payoff
        kelly_f = self._kelly_formula(wr, pr)
        if kelly_f <= 0:
            return self._rejected(
                f"Kelly fraction non-positive ({kelly_f:.4f}): "
                f"WR={wr:.3f}, payoff={pr:.3f} implies no edge"
            )

        # 2. Quarter-Kelly
        quarter_kelly = kelly_f * self._kelly_frac

        # 3. Confidence scaling
        conf_scaled = quarter_kelly * confidence

        # 4. Drawdown gate
        dd_scaled = conf_scaled * drawdown_multiplier

        # 5. Time multiplier (weekend, funding proximity)
        time_scaled = dd_scaled * time_multiplier

        # 6. Max cap
        final_frac = min(time_scaled, self._max_frac)

        # 7. Convert to BTC quantity
        dollar_size = final_frac * equity
        quantity = dollar_size / current_price

        # 8. Minimum size check
        if quantity < self._min_btc:
            return self._rejected(
                f"Computed size {quantity:.6f} BTC below minimum {self._min_btc}"
            )

        # 9. ATR-based stop loss
        atr = self._compute_atr(highs, lows, closes, current_idx)
        stop_distance = self._atr_mult * atr

        if stop_distance <= 0:
            # Fallback: use 1% of price as stop
            stop_distance = current_price * 0.01
            logger.warning(
                "atr_fallback",
                reason="ATR was zero, using 1% of price",
                stop_distance=stop_distance,
            )

        if signal_direction == "long":
            stop_price = current_price - stop_distance
        else:
            stop_price = current_price + stop_distance

        risk_per_trade = quantity * stop_distance

        # 10. CRITICAL: Cap risk per trade (Monte Carlo shows >1% leads to 66%+ DD)
        # This is the MOST IMPORTANT constraint in the entire system.
        # Formula: risk = quantity * stop_distance, must be <= max_risk * equity
        max_risk_usd = self._max_risk * equity
        if risk_per_trade > max_risk_usd:
            # Reduce quantity to fit within risk budget
            quantity = max_risk_usd / stop_distance
            dollar_size = quantity * current_price
            final_frac = dollar_size / equity
            risk_per_trade = max_risk_usd
            logger.info(
                "risk_cap_applied",
                original_risk=risk_per_trade,
                capped_risk=max_risk_usd,
                max_risk_pct=self._max_risk * 100,
                new_quantity=quantity,
            )

            if quantity < self._min_btc:
                return self._rejected(
                    f"After risk cap, size {quantity:.6f} BTC below minimum {self._min_btc}. "
                    f"Risk budget too small for current volatility."
                )

        result = SizingResult(
            quantity_btc=quantity,
            notional_usd=dollar_size,
            position_fraction=final_frac,
            stop_distance_usd=stop_distance,
            stop_price=stop_price,
            risk_per_trade_usd=risk_per_trade,
            raw_kelly_fraction=kelly_f,
            quarter_kelly_fraction=quarter_kelly,
            confidence_scaled=conf_scaled,
            drawdown_scaled=dd_scaled,
            final_fraction=final_frac,
        )

        logger.info(
            "position_sized",
            direction=signal_direction,
            quantity_btc=round(quantity, 6),
            notional_usd=round(dollar_size, 2),
            fraction=round(final_frac, 4),
            stop_price=round(stop_price, 2),
            risk_usd=round(risk_per_trade, 2),
            kelly=round(kelly_f, 4),
            confidence=round(confidence, 4),
            dd_mult=round(drawdown_multiplier, 4),
        )

        return result

    # -- Private helpers -------------------------------------------------------

    def _kelly_formula(self, win_rate: float, payoff_ratio: float) -> float:
        """Kelly criterion: f = (WR * payoff - (1-WR)) / payoff.

        Returns the optimal fraction of capital to risk.
        If negative (no edge), returns 0.
        """
        if payoff_ratio <= 0:
            return 0.0
        f = (win_rate * payoff_ratio - (1.0 - win_rate)) / payoff_ratio
        return max(f, 0.0)

    def _compute_atr(
        self,
        highs: NDArray[np.float64],
        lows: NDArray[np.float64],
        closes: NDArray[np.float64],
        current_idx: int,
    ) -> float:
        """Average True Range over self._atr_period bars ending at current_idx.

        TR = max(H-L, |H-prev_close|, |L-prev_close|)
        ATR = mean(TR) over the window.
        """
        period = self._atr_period
        start = max(1, current_idx - period + 1)  # need previous close
        if start >= current_idx:
            return 0.0

        tr_values: list[float] = []
        for i in range(start, current_idx + 1):
            hl = highs[i] - lows[i]
            hc = abs(highs[i] - closes[i - 1])
            lc = abs(lows[i] - closes[i - 1])
            tr_values.append(max(hl, hc, lc))

        if not tr_values:
            return 0.0

        return float(np.mean(tr_values))

    def _rejected(self, reason: str) -> SizingResult:
        """Return a zero-size result with rejection reason."""
        logger.info("position_sizing_rejected", reason=reason)
        return SizingResult(
            quantity_btc=0.0,
            notional_usd=0.0,
            position_fraction=0.0,
            stop_distance_usd=0.0,
            stop_price=0.0,
            risk_per_trade_usd=0.0,
            raw_kelly_fraction=0.0,
            quarter_kelly_fraction=0.0,
            confidence_scaled=0.0,
            drawdown_scaled=0.0,
            final_fraction=0.0,
            rejected=True,
            rejection_reason=reason,
        )
