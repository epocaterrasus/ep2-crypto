"""Hierarchical regime detection ensemble.

Combines four layers into a unified regime output:
- Fast layer: Efficiency Ratio (trend vs chop)
- Fast layer: GJR-GARCH (volatility regime)
- Core layer: HMM (state identification via filtered probabilities)
- Core layer: BOCPD (change point early warning)

The ensemble uses a voting/weighting scheme to produce:
- A regime label (0 = low-vol/choppy, 1 = medium/neutral, 2 = high-vol/trending)
- Regime probabilities that sum to 1.0
- A change point alert flag
- An overall confidence score
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING

import numpy as np
import structlog

from ep2_crypto.regime.bocpd import BOCPDDetector, BOCPDResult
from ep2_crypto.regime.efficiency_ratio import (
    EfficiencyRatioDetector,
    ERResult,
)
from ep2_crypto.regime.garch import GARCHDetector, GARCHResult
from ep2_crypto.regime.hmm import HMMDetector, HMMResult

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = structlog.get_logger(__name__)


class MarketRegime(IntEnum):
    """Unified market regime classification."""

    LOW_VOL = 0  # Quiet, range-bound market
    NORMAL = 1  # Typical conditions
    HIGH_VOL = 2  # Volatile, potentially trending


@dataclass(frozen=True)
class RegimeResult:
    """Unified output from the hierarchical regime detector."""

    regime: MarketRegime
    regime_probabilities: tuple[float, float, float]  # P(low), P(normal), P(high)
    confidence: float  # Overall confidence in regime label [0, 1]
    changepoint_alert: bool  # True if BOCPD detects a regime transition
    changepoint_prob: float  # Raw BOCPD changepoint probability

    # Component outputs for transparency
    er_result: ERResult
    garch_result: GARCHResult
    hmm_result: HMMResult
    bocpd_result: BOCPDResult


class HierarchicalRegimeDetector:
    """Hierarchical ensemble combining ER, GARCH, HMM, and BOCPD.

    Parameters
    ----------
    er_detector : EfficiencyRatioDetector | None
        Custom ER detector. Uses defaults if None.
    garch_detector : GARCHDetector | None
        Custom GARCH detector. Uses defaults if None.
    hmm_detector : HMMDetector | None
        Custom HMM detector. Uses defaults if None.
    bocpd_detector : BOCPDDetector | None
        Custom BOCPD detector. Uses defaults if None.
    er_weight : float
        Weight for ER vote in ensemble (default 0.2).
    garch_weight : float
        Weight for GARCH vote in ensemble (default 0.3).
    hmm_weight : float
        Weight for HMM vote in ensemble (default 0.5).
    changepoint_confidence_discount : float
        Multiply confidence by this during transitions (default 0.5).
    """

    def __init__(
        self,
        er_detector: EfficiencyRatioDetector | None = None,
        garch_detector: GARCHDetector | None = None,
        hmm_detector: HMMDetector | None = None,
        bocpd_detector: BOCPDDetector | None = None,
        er_weight: float = 0.2,
        garch_weight: float = 0.3,
        hmm_weight: float = 0.5,
        changepoint_confidence_discount: float = 0.5,
    ) -> None:
        self._er = er_detector or EfficiencyRatioDetector()
        self._garch = garch_detector or GARCHDetector()
        self._hmm = hmm_detector or HMMDetector()
        self._bocpd = bocpd_detector or BOCPDDetector()

        # Normalize weights
        total_w = er_weight + garch_weight + hmm_weight
        self._er_weight = er_weight / total_w
        self._garch_weight = garch_weight / total_w
        self._hmm_weight = hmm_weight / total_w
        self._cp_discount = changepoint_confidence_discount

    @property
    def warmup_bars(self) -> int:
        """Maximum warmup across all components."""
        return max(
            self._er.warmup_bars,
            self._garch.warmup_bars,
            self._hmm.warmup_bars,
            self._bocpd.warmup_bars,
        )

    def update(
        self,
        idx: int,
        closes: NDArray[np.float64],
    ) -> RegimeResult:
        """Compute unified regime output for bar at idx.

        Runs all four detectors and combines their outputs into a single
        regime label with probabilities and confidence.
        """
        # Run all detectors
        er_result = self._er.update(idx, closes)
        garch_result = self._garch.update(idx, closes)
        hmm_result = self._hmm.update(idx, closes)
        bocpd_result = self._bocpd.update(idx, closes)

        # Convert each detector's output to a 3-class probability distribution
        er_probs = self._er_to_probs(er_result)
        garch_probs = self._garch_to_probs(garch_result)
        hmm_probs = self._hmm_to_probs(hmm_result)

        # Weighted combination
        combined = (
            self._er_weight * er_probs
            + self._garch_weight * garch_probs
            + self._hmm_weight * hmm_probs
        )

        # Normalize
        total = np.sum(combined)
        if total > 0:
            combined /= total

        regime = MarketRegime(int(np.argmax(combined)))

        # Confidence: max probability, discounted during transitions
        confidence = float(np.max(combined))
        if bocpd_result.is_changepoint:
            confidence *= self._cp_discount

        return RegimeResult(
            regime=regime,
            regime_probabilities=(
                float(combined[0]),
                float(combined[1]),
                float(combined[2]),
            ),
            confidence=confidence,
            changepoint_alert=bocpd_result.is_changepoint,
            changepoint_prob=bocpd_result.changepoint_prob,
            er_result=er_result,
            garch_result=garch_result,
            hmm_result=hmm_result,
            bocpd_result=bocpd_result,
        )

    @staticmethod
    def _er_to_probs(er: ERResult) -> NDArray[np.float64]:
        """Convert ER regime to a 3-class probability distribution.

        Maps: CHOPPY -> LOW_VOL, NEUTRAL -> NORMAL, TRENDING -> HIGH_VOL
        Uses confidence to spread probability mass.
        """
        probs = np.array([1.0 / 3, 1.0 / 3, 1.0 / 3])

        if np.isnan(er.er_short):
            return probs

        # Map ER regime to index
        regime_idx = int(er.regime)  # 0=CHOPPY, 1=NEUTRAL, 2=TRENDING
        conf = er.confidence

        # Allocate: conf portion to detected regime, rest uniform
        probs = np.full(3, (1.0 - conf) / 3)
        probs[regime_idx] += conf

        return probs

    @staticmethod
    def _garch_to_probs(garch: GARCHResult) -> NDArray[np.float64]:
        """Convert GARCH vol regime to a 3-class probability distribution."""
        probs = np.array([1.0 / 3, 1.0 / 3, 1.0 / 3])

        if np.isnan(garch.conditional_vol):
            return probs

        regime_idx = int(garch.vol_regime)  # 0=LOW, 1=MEDIUM, 2=HIGH
        conf = garch.confidence

        probs = np.full(3, (1.0 - conf) / 3)
        probs[regime_idx] += conf

        return probs

    @staticmethod
    def _hmm_to_probs(hmm: HMMResult) -> NDArray[np.float64]:
        """Convert HMM state probabilities to a 3-class distribution.

        For 2-state HMM: state 0 (low vol) -> split between LOW_VOL and NORMAL,
        state 1 (high vol) -> split between NORMAL and HIGH_VOL.
        For 3+ state HMM: direct mapping.
        """
        probs = np.array([1.0 / 3, 1.0 / 3, 1.0 / 3])

        if not hmm.is_fitted:
            return probs

        n = hmm.n_states
        state_probs = np.array(hmm.state_probabilities)

        if n == 2:
            # State 0 = low vol, State 1 = high vol
            # Map to 3 classes with overlap in NORMAL
            p_low = state_probs[0]
            p_high = state_probs[1]
            probs[0] = p_low * 0.7  # LOW_VOL
            probs[1] = p_low * 0.3 + p_high * 0.3  # NORMAL (shared)
            probs[2] = p_high * 0.7  # HIGH_VOL
        elif n == 3:
            # Direct 1:1 mapping (sorted by vol)
            probs = state_probs[:3].copy()
        else:
            # Merge into 3 bins
            bin_size = n / 3
            for i in range(3):
                start = int(i * bin_size)
                end = int((i + 1) * bin_size)
                probs[i] = float(np.sum(state_probs[start:end]))

        # Normalize
        total = np.sum(probs)
        if total > 0:
            probs /= total

        return probs

    def compute_batch(
        self,
        closes: NDArray[np.float64],
    ) -> list[RegimeResult]:
        """Compute regime results for all bars.

        Resets GARCH and BOCPD state. HMM auto-fits when enough data.
        """
        self._garch.reset()
        self._bocpd.reset()

        results: list[RegimeResult] = []
        for idx in range(len(closes)):
            results.append(self.update(idx, closes))
        return results
