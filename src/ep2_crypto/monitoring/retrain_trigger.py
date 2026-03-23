"""Auto-retrain pipeline with validation gate.

Trigger conditions (any fires → retrain):
  1. PSI > 0.2 on any top-10 feature
  2. Rolling 7d Sharpe < 50% of baseline
  3. ADWIN fires on return distribution
  4. Scheduled interval (default 4h)

Validation gate (all must pass to swap):
  1. New model beats old on 24h holdout
  2. Top-10 feature overlap > 70%
  3. Calibration ECE < 0.05

If validation fails → keep old model + alert.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol

import structlog

logger = structlog.get_logger(__name__)


class RetrainReason(StrEnum):
    """Why a retrain was triggered."""

    FEATURE_DRIFT = "feature_drift"
    SHARPE_DECLINE = "sharpe_decline"
    ADWIN_CHANGE = "adwin_change"
    SCHEDULED = "scheduled"
    MANUAL = "manual"


class ValidationResult(StrEnum):
    """Result of model validation after retrain."""

    PASSED = "passed"
    FAILED_PERFORMANCE = "failed_performance"
    FAILED_FEATURE_OVERLAP = "failed_feature_overlap"
    FAILED_CALIBRATION = "failed_calibration"
    FAILED_MULTIPLE = "failed_multiple"


@dataclass(frozen=True)
class RetrainDecision:
    """Record of a retrain decision."""

    timestamp_ms: int
    reason: RetrainReason
    triggered: bool
    validation_result: ValidationResult | None = None
    model_swapped: bool = False
    old_sharpe: float | None = None
    new_sharpe: float | None = None
    feature_overlap: float | None = None
    calibration_ece: float | None = None
    details: dict[str, Any] = field(default_factory=dict)


class ModelTrainer(Protocol):
    """Protocol for model training — implemented by actual training pipeline."""

    def retrain(self, reason: RetrainReason) -> dict[str, Any]:
        """Run retrain and return metrics dict."""
        ...

    def get_holdout_sharpe(self) -> float:
        """Sharpe on 24h holdout for new model."""
        ...

    def get_current_sharpe(self) -> float:
        """Sharpe on 24h holdout for current model."""
        ...

    def get_feature_overlap(self) -> float:
        """Fraction of top-10 features shared between old and new."""
        ...

    def get_calibration_ece(self) -> float:
        """Expected Calibration Error of new model."""
        ...

    def swap_model(self) -> None:
        """Atomically swap new model into production."""
        ...

    def discard_new_model(self) -> None:
        """Discard the newly trained model."""
        ...


class RetrainTrigger:
    """Monitors conditions and triggers retrain when appropriate.

    Integrates with drift detector and alpha decay monitor to decide
    when to retrain, then validates the new model before swapping.
    """

    def __init__(
        self,
        schedule_interval_s: float = 4 * 3600,  # 4 hours
        psi_threshold: float = 0.2,
        sharpe_decline_ratio: float = 0.5,
        min_holdout_sharpe_improvement: float = 0.0,
        min_feature_overlap: float = 0.7,
        max_calibration_ece: float = 0.05,
        cooldown_s: float = 1800,  # 30 min between retrains
    ) -> None:
        self._schedule_interval_s = schedule_interval_s
        self._psi_threshold = psi_threshold
        self._sharpe_decline_ratio = sharpe_decline_ratio
        self._min_holdout_sharpe_improvement = min_holdout_sharpe_improvement
        self._min_feature_overlap = min_feature_overlap
        self._max_calibration_ece = max_calibration_ece
        self._cooldown_s = cooldown_s

        # None = never retrained; cooldown is inactive and scheduled trigger
        # uses object creation time as the reference.
        self._last_retrain_time: float | None = None
        self._created_at: float = time.monotonic()
        self._baseline_sharpe: float | None = None
        self._retrain_count = 0
        self._swap_count = 0
        self._history: list[RetrainDecision] = []

    def set_baseline_sharpe(self, sharpe: float) -> None:
        """Set the baseline Sharpe for decline comparison."""
        self._baseline_sharpe = sharpe
        logger.info("retrain_baseline_set", sharpe=round(sharpe, 4))

    def check_trigger(
        self,
        drifted_features: list[str] | None = None,
        current_sharpe: float | None = None,
        adwin_detected: bool = False,
        force: bool = False,
    ) -> RetrainReason | None:
        """Check if any retrain trigger fires.

        Returns the reason if triggered, None otherwise.

        Cooldown logic: only applies after the first retrain has completed.
        A freshly-created RetrainTrigger is never blocked by cooldown.
        """
        now = time.monotonic()

        # Cooldown check — only active after at least one retrain has run.
        if not force and self._last_retrain_time is not None and (now - self._last_retrain_time) < self._cooldown_s:
            return None

        # 1. Feature drift
        if drifted_features:
            logger.info(
                "retrain_trigger_drift",
                drifted_features=drifted_features,
            )
            return RetrainReason.FEATURE_DRIFT

        # 2. Sharpe decline
        if (
            self._baseline_sharpe is not None
            and self._baseline_sharpe > 0
            and current_sharpe is not None
            and current_sharpe < self._baseline_sharpe * self._sharpe_decline_ratio
        ):
            logger.info(
                "retrain_trigger_sharpe",
                current=round(current_sharpe, 4),
                baseline=round(self._baseline_sharpe, 4),
            )
            return RetrainReason.SHARPE_DECLINE

        # 3. ADWIN change detection
        if adwin_detected:
            logger.info("retrain_trigger_adwin")
            return RetrainReason.ADWIN_CHANGE

        # 4. Manual force (before scheduled so it gets correct reason)
        if force:
            return RetrainReason.MANUAL

        # 5. Scheduled — measure elapsed time from last retrain or object creation.
        reference_time = self._last_retrain_time if self._last_retrain_time is not None else self._created_at
        if (now - reference_time) >= self._schedule_interval_s:
            return RetrainReason.SCHEDULED

        return None

    def execute_retrain(
        self,
        reason: RetrainReason,
        trainer: ModelTrainer,
        timestamp_ms: int,
    ) -> RetrainDecision:
        """Execute the full retrain → validate → swap pipeline.

        Args:
            reason: Why we're retraining.
            trainer: Object implementing ModelTrainer protocol.
            timestamp_ms: Current timestamp for audit trail.

        Returns:
            RetrainDecision with full audit trail.
        """
        self._last_retrain_time = time.monotonic()
        self._retrain_count += 1

        logger.info("retrain_started", reason=reason.value, count=self._retrain_count)

        try:
            metrics = trainer.retrain(reason)
        except Exception:
            logger.exception("retrain_failed", reason=reason.value)
            decision = RetrainDecision(
                timestamp_ms=timestamp_ms,
                reason=reason,
                triggered=True,
                validation_result=ValidationResult.FAILED_PERFORMANCE,
                model_swapped=False,
                details={"error": "retrain_exception"},
            )
            self._history.append(decision)
            return decision

        # Validation gate
        new_sharpe = trainer.get_holdout_sharpe()
        old_sharpe = trainer.get_current_sharpe()
        feature_overlap = trainer.get_feature_overlap()
        calibration_ece = trainer.get_calibration_ece()

        failures: list[str] = []

        # Gate 1: Performance
        if new_sharpe < old_sharpe + self._min_holdout_sharpe_improvement:
            failures.append("performance")

        # Gate 2: Feature stability
        if feature_overlap < self._min_feature_overlap:
            failures.append("feature_overlap")

        # Gate 3: Calibration quality
        if calibration_ece > self._max_calibration_ece:
            failures.append("calibration")

        if not failures:
            trainer.swap_model()
            self._swap_count += 1
            if new_sharpe > 0:
                self._baseline_sharpe = new_sharpe
            validation = ValidationResult.PASSED
            swapped = True
            logger.info(
                "retrain_model_swapped",
                new_sharpe=round(new_sharpe, 4),
                old_sharpe=round(old_sharpe, 4),
                feature_overlap=round(feature_overlap, 4),
                ece=round(calibration_ece, 4),
            )
        else:
            trainer.discard_new_model()
            swapped = False
            if len(failures) > 1:
                validation = ValidationResult.FAILED_MULTIPLE
            elif "performance" in failures:
                validation = ValidationResult.FAILED_PERFORMANCE
            elif "feature_overlap" in failures:
                validation = ValidationResult.FAILED_FEATURE_OVERLAP
            else:
                validation = ValidationResult.FAILED_CALIBRATION
            logger.warning(
                "retrain_validation_failed",
                failures=failures,
                new_sharpe=round(new_sharpe, 4),
                old_sharpe=round(old_sharpe, 4),
                feature_overlap=round(feature_overlap, 4),
                ece=round(calibration_ece, 4),
            )

        decision = RetrainDecision(
            timestamp_ms=timestamp_ms,
            reason=reason,
            triggered=True,
            validation_result=validation,
            model_swapped=swapped,
            old_sharpe=old_sharpe,
            new_sharpe=new_sharpe,
            feature_overlap=feature_overlap,
            calibration_ece=calibration_ece,
            details={"metrics": metrics, "failures": failures},
        )
        self._history.append(decision)
        return decision

    @property
    def retrain_count(self) -> int:
        return self._retrain_count

    @property
    def swap_count(self) -> int:
        return self._swap_count

    @property
    def history(self) -> list[RetrainDecision]:
        return list(self._history)

    @property
    def baseline_sharpe(self) -> float | None:
        return self._baseline_sharpe

    def reset_cooldown(self) -> None:
        """Allow immediate retrain by clearing the last-retrain timestamp.

        This resets to the "never retrained" state: cooldown is inactive
        and the scheduled trigger measures from object creation time.
        """
        self._last_retrain_time = None
