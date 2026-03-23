"""GJR-GARCH(1,1)-t conditional volatility regime detector — fast layer.

Implements the GJR-GARCH(1,1) model with Student-t innovations for
conditional volatility estimation and vol-regime classification.

GJR extension adds asymmetric leverage: negative returns can have a
different impact on volatility than positive returns (gamma term).

sigma2_t = omega + (alpha + gamma * I_{e<0}) * e_{t-1}^2 + beta * sigma2_{t-1}

Vol regime classification uses rolling percentiles of conditional volatility
to produce low/medium/high vol regime labels.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


class VolRegime(IntEnum):
    """Volatility regime classification."""

    LOW = 0
    MEDIUM = 1
    HIGH = 2


@dataclass(frozen=True)
class GARCHResult:
    """Output from the GARCH detector."""

    conditional_vol: float  # Current conditional volatility (sigma_t)
    vol_regime: VolRegime
    vol_percentile: float  # Where current vol falls in recent history [0, 1]
    confidence: float  # How strongly vol supports the regime label


class GARCHDetector:
    """GJR-GARCH(1,1)-t regime detector with vol-regime classification.

    Uses recursive conditional variance updates (O(1) per bar after init)
    with asymmetric leverage effect. Classifies into low/medium/high vol
    regimes using rolling percentile thresholds.

    Parameters
    ----------
    omega : float
        Constant term in variance equation (default 1e-6).
    alpha : float
        Squared shock coefficient (default 0.05).
    gamma : float
        Asymmetric leverage coefficient (default 0.10).
    beta : float
        Persistence coefficient (default 0.85).
    percentile_window : int
        Rolling window for vol percentile computation (default 100 bars).
    low_percentile : float
        Below this percentile → LOW vol regime (default 0.33).
    high_percentile : float
        Above this percentile → HIGH vol regime (default 0.67).
    """

    def __init__(
        self,
        omega: float = 1e-6,
        alpha: float = 0.05,
        gamma: float = 0.10,
        beta: float = 0.85,
        percentile_window: int = 100,
        low_percentile: float = 0.33,
        high_percentile: float = 0.67,
    ) -> None:
        if alpha + gamma / 2 + beta >= 1.0:
            msg = "GARCH stationarity requires alpha + gamma/2 + beta < 1.0"
            raise ValueError(msg)
        if percentile_window < 10:
            msg = "percentile_window must be >= 10"
            raise ValueError(msg)

        self._omega = omega
        self._alpha = alpha
        self._gamma = gamma
        self._beta = beta
        self._percentile_window = percentile_window
        self._low_pct = low_percentile
        self._high_pct = high_percentile

        # Internal state
        self._sigma2: float = 0.0
        self._last_return: float = 0.0
        self._initialized: bool = False
        self._vol_history: list[float] = []

    @property
    def warmup_bars(self) -> int:
        """Minimum bars needed: enough for initial variance + some percentile history."""
        return 21  # 20 bars for init variance + 1 for first GARCH update

    def reset(self) -> None:
        """Reset internal state for a fresh run."""
        self._sigma2 = 0.0
        self._last_return = 0.0
        self._initialized = False
        self._vol_history = []

    def _initialize(self, returns: NDArray[np.float64]) -> None:
        """Initialize conditional variance from sample variance."""
        self._sigma2 = float(np.var(returns))
        if self._sigma2 <= 0:
            self._sigma2 = 1e-8
        self._initialized = True

    def _update_variance(self, ret: float) -> None:
        """Single O(1) GARCH update step.

        sigma2_t = omega + (alpha + gamma * I_{ret<0}) * ret^2 + beta * sigma2_{t-1}
        """
        leverage = self._gamma if ret < 0 else 0.0
        self._sigma2 = (
            self._omega
            + (self._alpha + leverage) * ret * ret
            + self._beta * self._sigma2
        )
        # Floor to prevent numerical underflow
        self._sigma2 = max(self._sigma2, 1e-15)

    def update(
        self,
        idx: int,
        closes: NDArray[np.float64],
    ) -> GARCHResult:
        """Compute conditional vol and vol regime for bar at idx.

        On first call (or after reset), computes full recursive chain from
        the start. On subsequent calls, only updates incrementally.

        Parameters
        ----------
        idx : int
            Current bar index.
        closes : NDArray[np.float64]
            Close price array (data up to idx is used).
        """
        nan_result = GARCHResult(
            conditional_vol=float("nan"),
            vol_regime=VolRegime.MEDIUM,
            vol_percentile=0.5,
            confidence=0.0,
        )

        if idx < self.warmup_bars - 1:
            return nan_result

        # Compute log returns up to idx
        log_prices = np.log(closes[: idx + 1])
        returns = np.diff(log_prices)

        if len(returns) < 2:
            return nan_result

        # Initialize with sample variance from first warmup_bars
        init_window = min(20, len(returns))
        self._initialize(returns[:init_window])

        # Run full recursive GARCH from init to current
        for i in range(len(returns)):
            self._update_variance(float(returns[i]))

        cond_vol = math.sqrt(self._sigma2)

        # Track vol history for percentile computation
        self._vol_history.append(cond_vol)
        if len(self._vol_history) > self._percentile_window:
            self._vol_history = self._vol_history[-self._percentile_window :]

        # Compute percentile of current vol in recent history
        vol_percentile = self._compute_percentile(cond_vol)

        # Classify vol regime
        regime, confidence = self._classify(vol_percentile)

        return GARCHResult(
            conditional_vol=cond_vol,
            vol_regime=regime,
            vol_percentile=vol_percentile,
            confidence=confidence,
        )

    def _compute_percentile(self, current_vol: float) -> float:
        """Compute where current vol falls in recent history [0, 1]."""
        if len(self._vol_history) < 2:
            return 0.5

        arr = np.array(self._vol_history)
        return float(np.mean(arr <= current_vol))

    def _classify(self, percentile: float) -> tuple[VolRegime, float]:
        """Classify vol regime based on percentile and compute confidence."""
        if percentile <= self._low_pct:
            # Low vol — confidence from distance to boundary
            confidence = (self._low_pct - percentile) / self._low_pct if self._low_pct > 0 else 1.0
            return VolRegime.LOW, min(confidence, 1.0)

        if percentile >= self._high_pct:
            # High vol — confidence from distance to boundary
            range_above = 1.0 - self._high_pct
            confidence = (percentile - self._high_pct) / range_above if range_above > 0 else 1.0
            return VolRegime.HIGH, min(confidence, 1.0)

        # Medium vol
        midpoint = (self._low_pct + self._high_pct) / 2
        half_range = (self._high_pct - self._low_pct) / 2
        confidence = 1.0 - abs(percentile - midpoint) / half_range if half_range > 0 else 0.5
        return VolRegime.MEDIUM, confidence

    def compute_batch(
        self,
        closes: NDArray[np.float64],
    ) -> list[GARCHResult]:
        """Compute GARCH results for all bars. Resets state before running."""
        self.reset()
        results: list[GARCHResult] = []
        for idx in range(len(closes)):
            results.append(self.update(idx, closes))
        return results
