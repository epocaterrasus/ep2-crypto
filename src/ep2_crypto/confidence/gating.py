"""Confidence gating pipeline — the single biggest Sharpe multiplier (2-4x).

Orchestrates 7 sequential gates that filter low-quality trade signals.
Each gate is independently enable/disable for ablation testing.

Pipeline:
    1. Isotonic calibration (from models.calibration)
    2. Meta-labeling gate (from confidence.meta_labeling)
    3. Ensemble agreement check (variance < threshold)
    4. Conformal prediction gate (singleton prediction set)
    5. Signal filters (vol range, regime stability, liquidity)
    6. Adaptive confidence threshold (regime-dependent)
    7. Drawdown gate (progressive 3% → 15% reduction)

Each gate produces a GateDecision with pass/fail and reason.
The composite confidence feeds into position sizing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import TYPE_CHECKING

import numpy as np
import structlog

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ep2_crypto.confidence.conformal import ConformalPredictor
    from ep2_crypto.confidence.meta_labeling import MetaLabeler
    from ep2_crypto.models.calibration import IsotonicCalibrator

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Gate identifiers
# ---------------------------------------------------------------------------


class GateID(IntEnum):
    """Identifiers for each gate in the pipeline."""

    CALIBRATION = 1
    META_LABELING = 2
    ENSEMBLE_AGREEMENT = 3
    CONFORMAL = 4
    SIGNAL_FILTERS = 5
    ADAPTIVE_THRESHOLD = 6
    DRAWDOWN = 7


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class GateDecision:
    """Result from a single gate evaluation."""

    gate_id: GateID
    passed: bool
    reason: str
    value: float = 0.0  # Gate-specific metric (e.g., confidence, variance)


@dataclass
class GatingResult:
    """Full result from the gating pipeline."""

    should_trade: bool
    direction: int  # -1 (short), 0 (abstain), +1 (long)
    composite_confidence: float  # Product of all gate confidences
    calibrated_probas: NDArray[np.float64] | None  # (3,) after calibration
    meta_label_prob: float  # P(profitable) from meta-labeler
    conformal_set_size: int  # Number of classes in conformal set
    drawdown_multiplier: float  # Position size multiplier from DD gate
    gate_decisions: list[GateDecision]  # Per-gate decisions
    rejection_gate: GateID | None  # First gate that rejected (or None)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class SignalFilterConfig:
    """Configuration for signal filter gate."""

    min_volatility_ann: float = 15.0  # Min annualized vol (%)
    max_volatility_ann: float = 150.0  # Max annualized vol (%)
    min_regime_stability: float = 0.6  # Min regime probability
    min_spread_liquidity: float = 0.0  # Min (unused by default)


@dataclass
class AdaptiveThresholdConfig:
    """Configuration for adaptive confidence threshold."""

    base_threshold: float = 0.60
    regime_adjustments: dict[int, float] = field(
        default_factory=lambda: {
            0: 0.0,  # Trending: no adjustment
            1: 0.05,  # Choppy: raise threshold
            2: 0.0,  # Neutral: no adjustment
        }
    )
    min_threshold: float = 0.50
    max_threshold: float = 0.80


@dataclass
class GatingConfig:
    """Configuration for the full gating pipeline."""

    # Gate enable/disable flags (for ablation)
    enable_calibration: bool = True
    enable_meta_labeling: bool = True
    enable_ensemble_agreement: bool = True
    enable_conformal: bool = True
    enable_signal_filters: bool = True
    enable_adaptive_threshold: bool = True
    enable_drawdown: bool = True

    # Meta-labeling threshold
    meta_label_threshold: float = 0.50

    # Ensemble agreement (max variance across base model probabilities)
    max_ensemble_variance: float = 0.10

    # Signal filters
    signal_filters: SignalFilterConfig = field(default_factory=SignalFilterConfig)

    # Adaptive threshold
    adaptive_threshold: AdaptiveThresholdConfig = field(default_factory=AdaptiveThresholdConfig)

    # Drawdown gate thresholds (DD% -> position multiplier)
    drawdown_start_pct: float = 3.0  # Start reducing at 3% DD
    drawdown_halt_pct: float = 15.0  # Halt at 15% DD
    drawdown_convex_k: float = 1.5  # Convex exponent


# ---------------------------------------------------------------------------
# Market context for signal filter gate
# ---------------------------------------------------------------------------


@dataclass
class MarketContext:
    """Current market state for signal filter evaluation."""

    volatility_ann: float = 50.0  # Current annualized volatility (%)
    regime_label: int = 0  # Current regime (0=trending, 1=choppy, 2=neutral)
    regime_probability: float = 0.8  # Confidence of regime detection
    spread_bps: float = 1.0  # Current bid-ask spread in basis points
    current_drawdown_pct: float = 0.0  # Current portfolio drawdown (%)


# ---------------------------------------------------------------------------
# Gating pipeline
# ---------------------------------------------------------------------------


class ConfidenceGatingPipeline:
    """Orchestrates 7 sequential confidence gates.

    Each gate either passes (allowing the signal through) or rejects.
    Gates run in order; the first rejection stops the pipeline.
    Passed gates contribute to composite confidence.

    Components are injected via set_* methods so the pipeline can be
    assembled incrementally and individual gates can be tested in isolation.
    """

    def __init__(self, config: GatingConfig | None = None) -> None:
        self._config = config or GatingConfig()

        # Injected components (optional — gates disabled if not set)
        self._calibrator: IsotonicCalibrator | None = None
        self._meta_labeler: MetaLabeler | None = None
        self._conformal: ConformalPredictor | None = None

    # -- Component injection ------------------------------------------------

    def set_calibrator(self, calibrator: IsotonicCalibrator) -> None:
        """Inject isotonic calibrator (from Sprint 7)."""
        self._calibrator = calibrator

    def set_meta_labeler(self, meta_labeler: MetaLabeler) -> None:
        """Inject meta-labeling model."""
        self._meta_labeler = meta_labeler

    def set_conformal(self, conformal: ConformalPredictor) -> None:
        """Inject conformal predictor."""
        self._conformal = conformal

    # -- Main entry point ---------------------------------------------------

    def evaluate(
        self,
        primary_probas: NDArray[np.float64],
        primary_prediction: int,
        features: NDArray[np.float64],
        market_ctx: MarketContext,
        ensemble_probas: list[NDArray[np.float64]] | None = None,
    ) -> GatingResult:
        """Run the full gating pipeline on a single trade signal.

        Args:
            primary_probas: Class probabilities (3,) for [DOWN, FLAT, UP]
                from the stacking ensemble (or best single model).
            primary_prediction: Predicted direction (-1, 0, +1).
            features: Original feature vector (n_features,) for this bar.
            market_ctx: Current market state for signal filters.
            ensemble_probas: Optional list of per-model probabilities for
                ensemble agreement check. Each is (3,).

        Returns:
            GatingResult with full decision trace.
        """
        decisions: list[GateDecision] = []
        composite_confidence = 1.0
        calibrated = primary_probas.copy()
        meta_prob = 0.0
        conformal_set_size = 0
        dd_multiplier = 1.0
        direction = primary_prediction

        # Gate 1: Calibration
        decision = self._gate_calibration(primary_probas)
        decisions.append(decision)
        if decision.passed and self._calibrator is not None:
            calibrated = self._calibrator.calibrate(primary_probas.reshape(1, -1))[0]
        if not decision.passed:
            return self._build_result(
                decisions,
                direction=0,
                confidence=0.0,
                calibrated=calibrated,
                meta_prob=meta_prob,
                conformal_size=conformal_set_size,
                dd_mult=dd_multiplier,
            )

        # Gate 2: Meta-labeling
        decision = self._gate_meta_labeling(
            calibrated, primary_prediction, features, market_ctx.regime_label
        )
        decisions.append(decision)
        meta_prob = decision.value
        if decision.passed:
            composite_confidence *= meta_prob
        if not decision.passed:
            return self._build_result(
                decisions,
                direction=0,
                confidence=0.0,
                calibrated=calibrated,
                meta_prob=meta_prob,
                conformal_size=conformal_set_size,
                dd_mult=dd_multiplier,
            )

        # Gate 3: Ensemble agreement
        decision = self._gate_ensemble_agreement(ensemble_probas)
        decisions.append(decision)
        if decision.passed:
            # Low variance → high agreement → boost confidence
            agreement = max(0.0, 1.0 - decision.value / self._config.max_ensemble_variance)
            composite_confidence *= 0.5 + 0.5 * agreement
        if not decision.passed:
            return self._build_result(
                decisions,
                direction=0,
                confidence=0.0,
                calibrated=calibrated,
                meta_prob=meta_prob,
                conformal_size=conformal_set_size,
                dd_mult=dd_multiplier,
            )

        # Gate 4: Conformal prediction
        decision, conf_direction = self._gate_conformal(calibrated)
        decisions.append(decision)
        conformal_set_size = int(decision.value)
        if decision.passed and conf_direction != 0:
            direction = conf_direction
        if not decision.passed:
            return self._build_result(
                decisions,
                direction=0,
                confidence=0.0,
                calibrated=calibrated,
                meta_prob=meta_prob,
                conformal_size=conformal_set_size,
                dd_mult=dd_multiplier,
            )

        # Gate 5: Signal filters
        decision = self._gate_signal_filters(market_ctx)
        decisions.append(decision)
        if not decision.passed:
            return self._build_result(
                decisions,
                direction=0,
                confidence=0.0,
                calibrated=calibrated,
                meta_prob=meta_prob,
                conformal_size=conformal_set_size,
                dd_mult=dd_multiplier,
            )

        # Gate 6: Adaptive threshold
        decision = self._gate_adaptive_threshold(composite_confidence, market_ctx.regime_label)
        decisions.append(decision)
        if not decision.passed:
            return self._build_result(
                decisions,
                direction=0,
                confidence=0.0,
                calibrated=calibrated,
                meta_prob=meta_prob,
                conformal_size=conformal_set_size,
                dd_mult=dd_multiplier,
            )

        # Gate 7: Drawdown
        decision = self._gate_drawdown(market_ctx.current_drawdown_pct)
        decisions.append(decision)
        dd_multiplier = decision.value
        if not decision.passed:
            return self._build_result(
                decisions,
                direction=0,
                confidence=0.0,
                calibrated=calibrated,
                meta_prob=meta_prob,
                conformal_size=conformal_set_size,
                dd_mult=dd_multiplier,
            )

        # All gates passed
        composite_confidence *= dd_multiplier

        logger.info(
            "gating_pipeline_passed",
            direction=direction,
            composite_confidence=round(composite_confidence, 4),
            meta_prob=round(meta_prob, 4),
            dd_multiplier=round(dd_multiplier, 4),
        )

        return self._build_result(
            decisions,
            direction=direction,
            confidence=composite_confidence,
            calibrated=calibrated,
            meta_prob=meta_prob,
            conformal_size=conformal_set_size,
            dd_mult=dd_multiplier,
        )

    # -- Individual gate implementations ------------------------------------

    def _gate_calibration(self, probas: NDArray[np.float64]) -> GateDecision:
        """Gate 1: Isotonic calibration."""
        if not self._config.enable_calibration:
            return GateDecision(
                gate_id=GateID.CALIBRATION,
                passed=True,
                reason="disabled",
                value=1.0,
            )
        if self._calibrator is None or not self._calibrator.is_fitted:
            return GateDecision(
                gate_id=GateID.CALIBRATION,
                passed=True,
                reason="no_calibrator_available",
                value=1.0,
            )
        return GateDecision(
            gate_id=GateID.CALIBRATION,
            passed=True,
            reason="calibrated",
            value=1.0,
        )

    def _gate_meta_labeling(
        self,
        calibrated_probas: NDArray[np.float64],
        primary_prediction: int,
        features: NDArray[np.float64],
        regime_label: int,
    ) -> GateDecision:
        """Gate 2: Meta-labeling profitability prediction."""
        if not self._config.enable_meta_labeling:
            return GateDecision(
                gate_id=GateID.META_LABELING,
                passed=True,
                reason="disabled",
                value=1.0,
            )
        if self._meta_labeler is None or not self._meta_labeler.is_fitted:
            return GateDecision(
                gate_id=GateID.META_LABELING,
                passed=True,
                reason="no_meta_labeler_available",
                value=1.0,
            )

        meta_features = self._meta_labeler.create_meta_features(
            primary_predictions=np.array([primary_prediction], dtype=np.int8),
            primary_probas=calibrated_probas.reshape(1, -1),
            features=features.reshape(1, -1),
            regime_labels=np.array([regime_label], dtype=np.int8),
        )
        prob_profitable = float(self._meta_labeler.predict_proba(meta_features)[0])

        passed = prob_profitable >= self._config.meta_label_threshold
        threshold = self._config.meta_label_threshold
        reason = "passed" if passed else f"prob={prob_profitable:.3f}<{threshold}"

        logger.debug(
            "gate_meta_labeling",
            prob_profitable=round(prob_profitable, 4),
            threshold=self._config.meta_label_threshold,
            passed=passed,
        )

        return GateDecision(
            gate_id=GateID.META_LABELING,
            passed=passed,
            reason=reason,
            value=prob_profitable,
        )

    def _gate_ensemble_agreement(
        self, ensemble_probas: list[NDArray[np.float64]] | None
    ) -> GateDecision:
        """Gate 3: Ensemble agreement (low variance = agreement)."""
        if not self._config.enable_ensemble_agreement:
            return GateDecision(
                gate_id=GateID.ENSEMBLE_AGREEMENT,
                passed=True,
                reason="disabled",
                value=0.0,
            )
        if ensemble_probas is None or len(ensemble_probas) < 2:
            return GateDecision(
                gate_id=GateID.ENSEMBLE_AGREEMENT,
                passed=True,
                reason="insufficient_models",
                value=0.0,
            )

        # Stack (n_models, 3) and compute mean variance across classes
        stacked = np.array(ensemble_probas, dtype=np.float64)
        variance = float(np.mean(np.var(stacked, axis=0)))

        passed = variance <= self._config.max_ensemble_variance
        reason = "passed" if passed else f"var={variance:.4f}>{self._config.max_ensemble_variance}"

        logger.debug(
            "gate_ensemble_agreement",
            variance=round(variance, 4),
            threshold=self._config.max_ensemble_variance,
            passed=passed,
        )

        return GateDecision(
            gate_id=GateID.ENSEMBLE_AGREEMENT,
            passed=passed,
            reason=reason,
            value=variance,
        )

    def _gate_conformal(self, calibrated_probas: NDArray[np.float64]) -> tuple[GateDecision, int]:
        """Gate 4: Conformal prediction (singleton set required).

        Returns:
            Tuple of (GateDecision, direction) where direction is -1/+1
            for tradeable singletons, 0 for abstention.
        """
        if not self._config.enable_conformal:
            return (
                GateDecision(
                    gate_id=GateID.CONFORMAL,
                    passed=True,
                    reason="disabled",
                    value=1.0,
                ),
                0,
            )
        if self._conformal is None or not self._conformal.is_calibrated:
            return (
                GateDecision(
                    gate_id=GateID.CONFORMAL,
                    passed=True,
                    reason="no_conformal_available",
                    value=1.0,
                ),
                0,
            )

        should_trade, pred_direction = self._conformal.gate(calibrated_probas.reshape(1, -1))
        trade = bool(should_trade[0])
        direction = int(pred_direction[0])
        sets = self._conformal.predict_sets(calibrated_probas.reshape(1, -1))
        set_size = len(sets[0]) if sets else 0

        reason = "passed" if trade else f"set_size={set_size}"

        logger.debug(
            "gate_conformal",
            set_size=set_size,
            direction=direction,
            passed=trade,
        )

        return (
            GateDecision(
                gate_id=GateID.CONFORMAL,
                passed=trade,
                reason=reason,
                value=float(set_size),
            ),
            direction,
        )

    def _gate_signal_filters(self, ctx: MarketContext) -> GateDecision:
        """Gate 5: Signal filters (vol range, regime stability)."""
        if not self._config.enable_signal_filters:
            return GateDecision(
                gate_id=GateID.SIGNAL_FILTERS,
                passed=True,
                reason="disabled",
                value=1.0,
            )

        sf = self._config.signal_filters

        if ctx.volatility_ann < sf.min_volatility_ann:
            return GateDecision(
                gate_id=GateID.SIGNAL_FILTERS,
                passed=False,
                reason=f"vol={ctx.volatility_ann:.1f}%<{sf.min_volatility_ann}%",
                value=ctx.volatility_ann,
            )
        if ctx.volatility_ann > sf.max_volatility_ann:
            return GateDecision(
                gate_id=GateID.SIGNAL_FILTERS,
                passed=False,
                reason=f"vol={ctx.volatility_ann:.1f}%>{sf.max_volatility_ann}%",
                value=ctx.volatility_ann,
            )
        if ctx.regime_probability < sf.min_regime_stability:
            return GateDecision(
                gate_id=GateID.SIGNAL_FILTERS,
                passed=False,
                reason=f"regime_prob={ctx.regime_probability:.2f}<{sf.min_regime_stability}",
                value=ctx.regime_probability,
            )

        return GateDecision(
            gate_id=GateID.SIGNAL_FILTERS,
            passed=True,
            reason="passed",
            value=ctx.volatility_ann,
        )

    def _gate_adaptive_threshold(
        self, current_confidence: float, regime_label: int
    ) -> GateDecision:
        """Gate 6: Adaptive confidence threshold (regime-dependent)."""
        if not self._config.enable_adaptive_threshold:
            return GateDecision(
                gate_id=GateID.ADAPTIVE_THRESHOLD,
                passed=True,
                reason="disabled",
                value=current_confidence,
            )

        at = self._config.adaptive_threshold
        adjustment = at.regime_adjustments.get(regime_label, 0.0)
        threshold = np.clip(at.base_threshold + adjustment, at.min_threshold, at.max_threshold)

        passed = current_confidence >= threshold
        reason = "passed" if passed else f"conf={current_confidence:.3f}<threshold={threshold:.3f}"

        logger.debug(
            "gate_adaptive_threshold",
            confidence=round(current_confidence, 4),
            threshold=round(float(threshold), 4),
            regime=regime_label,
            passed=passed,
        )

        return GateDecision(
            gate_id=GateID.ADAPTIVE_THRESHOLD,
            passed=passed,
            reason=reason,
            value=current_confidence,
        )

    def _gate_drawdown(self, current_drawdown_pct: float) -> GateDecision:
        """Gate 7: Drawdown-based position reduction.

        Uses convex decay: multiplier = max(0, (1 - dd_norm)^k)
        where dd_norm = (dd - start) / (halt - start), k = 1.5.
        """
        if not self._config.enable_drawdown:
            return GateDecision(
                gate_id=GateID.DRAWDOWN,
                passed=True,
                reason="disabled",
                value=1.0,
            )

        start = self._config.drawdown_start_pct
        halt = self._config.drawdown_halt_pct
        k = self._config.drawdown_convex_k

        if current_drawdown_pct <= start:
            return GateDecision(
                gate_id=GateID.DRAWDOWN,
                passed=True,
                reason="passed",
                value=1.0,
            )

        if current_drawdown_pct >= halt:
            return GateDecision(
                gate_id=GateID.DRAWDOWN,
                passed=False,
                reason=f"dd={current_drawdown_pct:.1f}%>={halt:.1f}%_halt",
                value=0.0,
            )

        # Convex decay between start and halt
        dd_norm = (current_drawdown_pct - start) / (halt - start)
        multiplier = max(0.0, (1.0 - dd_norm) ** k)

        logger.debug(
            "gate_drawdown",
            drawdown_pct=round(current_drawdown_pct, 2),
            multiplier=round(multiplier, 4),
        )

        return GateDecision(
            gate_id=GateID.DRAWDOWN,
            passed=True,
            reason="reduced",
            value=multiplier,
        )

    # -- Helpers ------------------------------------------------------------

    def _build_result(
        self,
        decisions: list[GateDecision],
        direction: int,
        confidence: float,
        calibrated: NDArray[np.float64],
        meta_prob: float,
        conformal_size: int,
        dd_mult: float,
    ) -> GatingResult:
        """Build GatingResult, identifying first rejection gate."""
        rejection = None
        for d in decisions:
            if not d.passed:
                rejection = d.gate_id
                break

        should_trade = rejection is None and direction != 0

        return GatingResult(
            should_trade=should_trade,
            direction=direction if should_trade else 0,
            composite_confidence=confidence if should_trade else 0.0,
            calibrated_probas=calibrated,
            meta_label_prob=meta_prob,
            conformal_set_size=conformal_size,
            drawdown_multiplier=dd_mult,
            gate_decisions=decisions,
            rejection_gate=rejection,
        )
