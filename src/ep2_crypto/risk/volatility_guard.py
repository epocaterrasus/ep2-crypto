"""Volatility guard: block trades when market conditions are unsuitable.

Checks performed:
    1. Rolling annualized volatility must be in [min_vol, max_vol]
       - Below min: costs eat the signal (market too quiet)
       - Above max: regime too chaotic for model reliability
    2. Trading hours enforcement (configurable, default 08:00-21:00 UTC)
    3. Weekend sizing reduction factor
    4. Funding settlement proximity check (reduce around 00/08/16 UTC)

Volatility is computed from 5-min log returns over a 288-bar (24h) window,
annualized with sqrt(105120) = 324.22 per CLAUDE.md.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import numpy as np
import structlog

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = structlog.get_logger(__name__)

# Annualization factor for 5-min bars: sqrt(288 * 365) = sqrt(105120) ~ 324.22
ANNUALIZATION_FACTOR = math.sqrt(105_120)

# Funding settlement times in UTC hours
FUNDING_SETTLEMENT_HOURS = (0, 8, 16)

# Minutes before/after funding settlement to be cautious
FUNDING_PROXIMITY_MINUTES = 15


@dataclass
class VolatilityGuardState:
    """Snapshot of volatility guard status."""

    rolling_vol_annualized: float
    vol_in_range: bool
    in_trading_hours: bool
    is_weekend: bool
    near_funding_settlement: bool
    can_trade: bool
    rejection_reason: str | None
    weekend_multiplier: float
    funding_proximity_multiplier: float


class VolatilityGuard:
    """Guards against trading in unsuitable volatility and time conditions.

    Configuration:
        min_vol_annualized: minimum annualized vol to trade (default 0.15 = 15%)
        max_vol_annualized: maximum annualized vol to trade (default 1.50 = 150%)
        vol_window: number of 5-min bars for rolling vol (default 288 = 24h)
        trading_start_hour_utc: start of trading window (default 8)
        trading_end_hour_utc: end of trading window (default 21)
        weekend_reduction: sizing reduction on weekends (default 0.30 = 30% smaller)
        enforce_trading_hours: whether to enforce time-of-day limits (default True)
    """

    def __init__(
        self,
        min_vol_annualized: float = 0.15,
        max_vol_annualized: float = 1.50,
        vol_window: int = 288,
        trading_start_hour_utc: int = 8,
        trading_end_hour_utc: int = 21,
        weekend_reduction: float = 0.30,
        enforce_trading_hours: bool = True,
    ) -> None:
        if min_vol_annualized < 0:
            raise ValueError(f"min_vol must be >= 0, got {min_vol_annualized}")
        if max_vol_annualized <= min_vol_annualized:
            raise ValueError(
                f"max_vol ({max_vol_annualized}) must exceed min_vol ({min_vol_annualized})"
            )
        if vol_window < 10:
            raise ValueError(f"vol_window must be >= 10, got {vol_window}")
        if not (0 <= trading_start_hour_utc < 24):
            raise ValueError(f"Invalid trading_start_hour: {trading_start_hour_utc}")
        if not (0 < trading_end_hour_utc <= 24):
            raise ValueError(f"Invalid trading_end_hour: {trading_end_hour_utc}")
        if not (0 <= weekend_reduction <= 1):
            raise ValueError(f"weekend_reduction must be in [0, 1], got {weekend_reduction}")

        self._min_vol = min_vol_annualized
        self._max_vol = max_vol_annualized
        self._vol_window = vol_window
        self._trading_start = trading_start_hour_utc
        self._trading_end = trading_end_hour_utc
        self._weekend_reduction = weekend_reduction
        self._enforce_hours = enforce_trading_hours

        # Cache the last computed vol for reporting
        self._last_vol_annualized: float = 0.0

    # -- Public API -----------------------------------------------------------

    def check(
        self,
        closes: NDArray[np.float64],
        current_idx: int,
        timestamp_ms: int,
    ) -> VolatilityGuardState:
        """Run all guard checks and return the result.

        Args:
            closes: Array of close prices (full history up to current_idx).
            current_idx: Index of the current bar in the closes array.
            timestamp_ms: Current bar timestamp in milliseconds.

        Returns:
            VolatilityGuardState with all check results and combined verdict.
        """
        # 1. Compute rolling volatility
        vol_ann = self._compute_rolling_vol(closes, current_idx)
        self._last_vol_annualized = vol_ann

        vol_in_range = self._min_vol <= vol_ann <= self._max_vol

        # 2. Check trading hours
        dt = datetime.fromtimestamp(timestamp_ms / 1000, tz=UTC)
        in_hours = self._check_trading_hours(dt)

        # 3. Check weekend
        is_weekend = self._is_weekend(dt)
        weekend_mult = 1.0 - self._weekend_reduction if is_weekend else 1.0

        # 4. Check funding settlement proximity
        near_funding = self._near_funding_settlement(dt)
        funding_mult = 0.5 if near_funding else 1.0

        # Determine overall can_trade
        rejection_reason: str | None = None
        can_trade = True

        if not vol_in_range:
            can_trade = False
            if vol_ann < self._min_vol:
                rejection_reason = (
                    f"Volatility too low: {vol_ann:.4f} annualized "
                    f"< minimum {self._min_vol:.4f}"
                )
            else:
                rejection_reason = (
                    f"Volatility too high: {vol_ann:.4f} annualized "
                    f"> maximum {self._max_vol:.4f}"
                )

        if self._enforce_hours and not in_hours:
            can_trade = False
            rejection_reason = (
                f"Outside trading hours: {dt.hour:02d}:{dt.minute:02d} UTC "
                f"not in {self._trading_start:02d}:00-{self._trading_end:02d}:00"
            )

        return VolatilityGuardState(
            rolling_vol_annualized=vol_ann,
            vol_in_range=vol_in_range,
            in_trading_hours=in_hours,
            is_weekend=is_weekend,
            near_funding_settlement=near_funding,
            can_trade=can_trade,
            rejection_reason=rejection_reason,
            weekend_multiplier=weekend_mult,
            funding_proximity_multiplier=funding_mult,
        )

    def compute_volatility(
        self,
        closes: NDArray[np.float64],
        current_idx: int,
    ) -> float:
        """Compute rolling annualized volatility at the given index.

        Exposed publicly for use in position sizing (ATR-based stops etc).
        """
        return self._compute_rolling_vol(closes, current_idx)

    # -- Private helpers -------------------------------------------------------

    def _compute_rolling_vol(
        self,
        closes: NDArray[np.float64],
        current_idx: int,
    ) -> float:
        """Compute rolling annualized volatility from log returns.

        Uses self._vol_window bars ending at current_idx. Needs at least
        vol_window + 1 data points for vol_window returns.
        """
        start = max(0, current_idx - self._vol_window)
        if current_idx - start < 2:
            return 0.0

        window = closes[start : current_idx + 1]

        # Filter out zeros/NaN to avoid log domain errors
        valid = window[window > 0]
        if len(valid) < 2:
            return 0.0

        log_returns = np.diff(np.log(valid))
        if len(log_returns) == 0:
            return 0.0

        std_return = float(np.std(log_returns, ddof=1))
        vol_annualized = std_return * ANNUALIZATION_FACTOR
        return vol_annualized

    def _check_trading_hours(self, dt: datetime) -> bool:
        """Return True if current time is within trading hours."""
        hour = dt.hour
        if self._trading_start < self._trading_end:
            return self._trading_start <= hour < self._trading_end
        else:
            # Wraps midnight (e.g., 22:00 - 06:00)
            return hour >= self._trading_start or hour < self._trading_end

    def _is_weekend(self, dt: datetime) -> bool:
        """Saturday=5, Sunday=6."""
        return dt.weekday() >= 5

    def _near_funding_settlement(self, dt: datetime) -> bool:
        """Return True if within FUNDING_PROXIMITY_MINUTES of a settlement."""
        current_minutes = dt.hour * 60 + dt.minute
        for hour in FUNDING_SETTLEMENT_HOURS:
            settlement_minutes = hour * 60
            diff = abs(current_minutes - settlement_minutes)
            # Handle midnight wrap
            diff = min(diff, 24 * 60 - diff)
            if diff <= FUNDING_PROXIMITY_MINUTES:
                return True
        return False
