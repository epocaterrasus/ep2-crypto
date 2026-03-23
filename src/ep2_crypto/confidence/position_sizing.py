"""Confidence-aware position sizing using Quarter-Kelly criterion.

Computes the confidence-weighted Kelly fraction that feeds into the risk
module's PositionSizer for final enforcement (drawdown gate, weekend
multipliers, ATR stops, risk caps).

Pipeline:
    1. Compute raw Kelly fraction from win_rate and payoff ratio
    2. Optionally apply Bayesian adjustment (Beta posterior) for uncertainty
    3. Apply quarter-Kelly scaling: f_actual = 0.25 * kelly
    4. Scale by composite confidence: f_actual *= composite_confidence
    5. Cap at max_position_pct (5% of capital)
    6. Reject if composite_confidence < min_confidence threshold

The Bayesian Kelly uses a Beta(alpha, beta) prior updated with observed
wins/losses to produce a posterior mean win_rate, then penalizes by
posterior uncertainty (stddev). This prevents over-betting when the sample
size is small and win_rate estimates are unreliable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Config & result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ConfidencePositionConfig:
    """Configuration for confidence-aware position sizing."""

    kelly_fraction: float = 0.25
    """Fraction of full Kelly to use (quarter-Kelly default)."""

    max_position_pct: float = 0.05
    """Maximum position as fraction of equity (5%)."""

    min_confidence: float = 0.55
    """Minimum composite confidence to allow any trade."""

    bayesian: bool = True
    """Whether to use Bayesian Kelly with Beta posterior."""

    prior_alpha: float = 1.0
    """Beta prior alpha (successes). 1.0 = uniform prior."""

    prior_beta: float = 1.0
    """Beta prior beta (failures). 1.0 = uniform prior."""

    min_trades_for_kelly: int = 30
    """Minimum observed trades before using empirical Kelly.
    Below this threshold, a conservative default is returned."""

    def __post_init__(self) -> None:
        if not (0 < self.kelly_fraction <= 1):
            raise ValueError(f"kelly_fraction must be in (0, 1], got {self.kelly_fraction}")
        if not (0 < self.max_position_pct <= 1):
            raise ValueError(f"max_position_pct must be in (0, 1], got {self.max_position_pct}")
        if not (0 <= self.min_confidence <= 1):
            raise ValueError(f"min_confidence must be in [0, 1], got {self.min_confidence}")
        if self.prior_alpha <= 0 or self.prior_beta <= 0:
            raise ValueError(
                f"Beta prior params must be positive, got alpha={self.prior_alpha}, "
                f"beta={self.prior_beta}"
            )
        if self.min_trades_for_kelly < 1:
            raise ValueError(f"min_trades_for_kelly must be >= 1, got {self.min_trades_for_kelly}")


@dataclass
class KellyResult:
    """Output of Kelly fraction computation."""

    raw_kelly: float
    """Full Kelly fraction (uncapped, unscaled)."""

    quarter_kelly: float
    """Kelly fraction after fractional scaling (e.g. 0.25x)."""

    bayesian_kelly: float | None
    """Bayesian-adjusted Kelly fraction, or None if not using Bayesian."""

    uncertainty: float
    """Posterior standard deviation of win_rate estimate.
    Higher uncertainty => less confidence in Kelly estimate."""


@dataclass
class SizingResult:
    """Output of confidence-aware position sizing."""

    position_fraction: float
    """Final position size as fraction of equity."""

    position_usd: float
    """Position size in USD."""

    position_btc: float
    """Position size in BTC."""

    confidence_scaled_kelly: float
    """Kelly fraction after confidence scaling (before cap)."""

    capped: bool
    """Whether the position was capped at max_position_pct."""

    rejection_reason: str | None = None
    """If set, trade was rejected and sizes are zero."""


# ---------------------------------------------------------------------------
# Online trade statistics tracker
# ---------------------------------------------------------------------------


@dataclass
class _TradeStats:
    """Running statistics for win_rate and payoff ratio."""

    n_trades: int = 0
    n_wins: int = 0
    total_win_pnl: float = 0.0
    total_loss_pnl: float = 0.0
    n_wins_with_pnl: int = 0
    n_losses_with_pnl: int = 0


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class ConfidencePositionSizer:
    """Confidence-weighted position sizing using (Bayesian) Kelly criterion.

    This module computes how large a position should be given:
    - Historical win_rate and payoff ratio (Kelly input)
    - Composite model confidence (from meta-labeling, conformal, regime gates)
    - Optional Bayesian uncertainty adjustment

    The output feeds into the risk module's PositionSizer, which applies
    additional drawdown/weekend/regime multipliers and ATR stop enforcement.
    """

    def __init__(self, config: ConfidencePositionConfig | None = None) -> None:
        self._config = config or ConfidencePositionConfig()
        self._stats = _TradeStats()

        logger.info(
            "confidence_position_sizer_init",
            kelly_fraction=self._config.kelly_fraction,
            max_position_pct=self._config.max_position_pct,
            min_confidence=self._config.min_confidence,
            bayesian=self._config.bayesian,
        )

    # -- Public API -----------------------------------------------------------

    def compute_kelly(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        n_trades: int = 0,
    ) -> KellyResult:
        """Compute Kelly fraction from win/loss statistics.

        Args:
            win_rate: Historical win rate (0.0 to 1.0).
            avg_win: Average winning trade PnL (positive).
            avg_loss: Average losing trade PnL (positive magnitude).
            n_trades: Number of trades observed (for Bayesian adjustment).

        Returns:
            KellyResult with raw, quarter, and optionally Bayesian Kelly.
        """
        if avg_loss <= 0:
            logger.warning(
                "kelly_zero_avg_loss",
                avg_loss=avg_loss,
                msg="avg_loss must be positive; returning zero Kelly",
            )
            return KellyResult(
                raw_kelly=0.0,
                quarter_kelly=0.0,
                bayesian_kelly=0.0 if self._config.bayesian else None,
                uncertainty=1.0,
            )

        payoff_ratio = avg_win / avg_loss

        # Standard Kelly: f = (WR * payoff - (1 - WR)) / payoff
        raw_kelly = self._kelly_formula(win_rate, payoff_ratio)
        quarter_kelly = raw_kelly * self._config.kelly_fraction

        # Bayesian adjustment
        bayesian_kelly: float | None = None
        uncertainty: float = 0.0

        if self._config.bayesian:
            n_wins = round(win_rate * max(n_trades, 1))
            n_losses = max(n_trades, 1) - n_wins

            bayesian_kelly, uncertainty = self._bayesian_kelly(
                n_wins=n_wins,
                n_losses=n_losses,
                payoff_ratio=payoff_ratio,
            )
            bayesian_kelly *= self._config.kelly_fraction
        else:
            # Without Bayesian, uncertainty is based on sample size heuristic
            uncertainty = 1.0 / max(np.sqrt(n_trades), 1.0)

        logger.debug(
            "kelly_computed",
            win_rate=round(win_rate, 4),
            avg_win=round(avg_win, 4),
            avg_loss=round(avg_loss, 4),
            payoff_ratio=round(payoff_ratio, 4),
            raw_kelly=round(raw_kelly, 6),
            quarter_kelly=round(quarter_kelly, 6),
            bayesian_kelly=round(bayesian_kelly, 6) if bayesian_kelly is not None else None,
            uncertainty=round(uncertainty, 4),
            n_trades=n_trades,
        )

        return KellyResult(
            raw_kelly=raw_kelly,
            quarter_kelly=quarter_kelly,
            bayesian_kelly=bayesian_kelly,
            uncertainty=uncertainty,
        )

    def compute_size(
        self,
        kelly_result: KellyResult,
        composite_confidence: float,
        equity: float,
        price: float,
    ) -> SizingResult:
        """Apply confidence scaling and caps to produce a position size.

        Args:
            kelly_result: Output of compute_kelly().
            composite_confidence: Product of all gate confidences (0.0 to 1.0).
            equity: Current account equity in USD.
            price: Current BTC price in USD.

        Returns:
            SizingResult with position sizes or rejection reason.
        """
        # Validate inputs
        if equity <= 0:
            return self._rejected(f"Non-positive equity: {equity:.2f}")
        if price <= 0:
            return self._rejected(f"Non-positive price: {price:.2f}")

        # Confidence gate
        if composite_confidence < self._config.min_confidence:
            return self._rejected(
                f"Confidence {composite_confidence:.4f} below minimum "
                f"{self._config.min_confidence:.4f}"
            )

        # Select Kelly value: prefer Bayesian if available
        if kelly_result.bayesian_kelly is not None:
            kelly_value = kelly_result.bayesian_kelly
        else:
            kelly_value = kelly_result.quarter_kelly

        # Reject if Kelly is non-positive (no edge)
        if kelly_value <= 0:
            return self._rejected(
                f"Kelly fraction non-positive ({kelly_value:.6f}): no edge detected"
            )

        # Scale by composite confidence
        confidence_scaled = kelly_value * composite_confidence

        # Apply max cap
        capped = confidence_scaled > self._config.max_position_pct
        final_fraction = min(confidence_scaled, self._config.max_position_pct)

        # Convert to USD and BTC
        position_usd = final_fraction * equity
        position_btc = position_usd / price

        logger.info(
            "confidence_size_computed",
            kelly_value=round(kelly_value, 6),
            composite_confidence=round(composite_confidence, 4),
            confidence_scaled=round(confidence_scaled, 6),
            final_fraction=round(final_fraction, 6),
            position_usd=round(position_usd, 2),
            position_btc=round(position_btc, 6),
            capped=capped,
        )

        return SizingResult(
            position_fraction=final_fraction,
            position_usd=position_usd,
            position_btc=position_btc,
            confidence_scaled_kelly=confidence_scaled,
            capped=capped,
        )

    def update_stats(self, is_profitable: bool, pnl: float) -> None:
        """Update running trade statistics with a new trade result.

        Args:
            is_profitable: Whether the trade was profitable.
            pnl: Absolute PnL of the trade (positive for wins, negative for losses).
        """
        self._stats.n_trades += 1

        if is_profitable:
            self._stats.n_wins += 1
            self._stats.total_win_pnl += abs(pnl)
            self._stats.n_wins_with_pnl += 1
        else:
            self._stats.total_loss_pnl += abs(pnl)
            self._stats.n_losses_with_pnl += 1

        logger.debug(
            "trade_stats_updated",
            n_trades=self._stats.n_trades,
            n_wins=self._stats.n_wins,
            win_rate=round(self._stats.n_wins / self._stats.n_trades, 4),
            is_profitable=is_profitable,
            pnl=round(pnl, 4),
        )

    def get_stats(self) -> dict[str, float]:
        """Return current running trade statistics.

        Returns:
            Dictionary with win_rate, avg_win, avg_loss, n_trades.
        """
        n = self._stats.n_trades
        if n == 0:
            return {
                "win_rate": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "n_trades": 0.0,
            }

        win_rate = self._stats.n_wins / n
        avg_win = (
            self._stats.total_win_pnl / self._stats.n_wins_with_pnl
            if self._stats.n_wins_with_pnl > 0
            else 0.0
        )
        avg_loss = (
            self._stats.total_loss_pnl / self._stats.n_losses_with_pnl
            if self._stats.n_losses_with_pnl > 0
            else 0.0
        )

        return {
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "n_trades": float(n),
        }

    # -- Private helpers ------------------------------------------------------

    @staticmethod
    def _kelly_formula(win_rate: float, payoff_ratio: float) -> float:
        """Standard Kelly criterion: f = (WR * payoff - (1 - WR)) / payoff.

        Returns 0 if the formula yields a non-positive value (no edge).
        """
        if payoff_ratio <= 0:
            return 0.0
        f = (win_rate * payoff_ratio - (1.0 - win_rate)) / payoff_ratio
        return max(f, 0.0)

    def _bayesian_kelly(
        self,
        n_wins: int,
        n_losses: int,
        payoff_ratio: float,
    ) -> tuple[float, float]:
        """Compute Bayesian Kelly using Beta posterior for win_rate.

        Uses Beta(prior_alpha + n_wins, prior_beta + n_losses) posterior.
        The Kelly fraction is computed at the posterior mean win_rate,
        then penalized by one standard deviation of the posterior to
        account for estimation uncertainty.

        Returns:
            Tuple of (bayesian_kelly_fraction, posterior_stddev).
        """
        alpha_post = self._config.prior_alpha + n_wins
        beta_post = self._config.prior_beta + n_losses

        # Posterior mean and standard deviation
        posterior_mean = alpha_post / (alpha_post + beta_post)
        posterior_var = (alpha_post * beta_post) / (
            (alpha_post + beta_post) ** 2 * (alpha_post + beta_post + 1)
        )
        posterior_std = float(np.sqrt(posterior_var))

        # Conservative estimate: use mean - 1 stddev as effective win_rate
        conservative_wr = max(posterior_mean - posterior_std, 0.0)

        bayesian_f = self._kelly_formula(conservative_wr, payoff_ratio)

        logger.debug(
            "bayesian_kelly",
            alpha_post=round(alpha_post, 2),
            beta_post=round(beta_post, 2),
            posterior_mean=round(posterior_mean, 4),
            posterior_std=round(posterior_std, 4),
            conservative_wr=round(conservative_wr, 4),
            bayesian_f=round(bayesian_f, 6),
        )

        return bayesian_f, posterior_std

    @staticmethod
    def _rejected(reason: str) -> SizingResult:
        """Return a zero-size result with rejection reason."""
        logger.info("confidence_sizing_rejected", reason=reason)
        return SizingResult(
            position_fraction=0.0,
            position_usd=0.0,
            position_btc=0.0,
            confidence_scaled_kelly=0.0,
            capped=False,
            rejection_reason=reason,
        )
