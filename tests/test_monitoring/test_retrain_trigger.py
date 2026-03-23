"""Tests for retrain trigger pipeline."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from ep2_crypto.monitoring.retrain_trigger import (
    RetrainDecision,
    RetrainReason,
    RetrainTrigger,
    ValidationResult,
)


def make_trainer(
    holdout_sharpe: float = 1.5,
    current_sharpe: float = 1.0,
    feature_overlap: float = 0.8,
    calibration_ece: float = 0.03,
    retrain_raises: bool = False,
) -> MagicMock:
    """Create a mock ModelTrainer."""
    trainer = MagicMock()
    if retrain_raises:
        trainer.retrain.side_effect = RuntimeError("training failed")
    else:
        trainer.retrain.return_value = {"loss": 0.5}
    trainer.get_holdout_sharpe.return_value = holdout_sharpe
    trainer.get_current_sharpe.return_value = current_sharpe
    trainer.get_feature_overlap.return_value = feature_overlap
    trainer.get_calibration_ece.return_value = calibration_ece
    return trainer


class TestCheckTrigger:
    @pytest.fixture()
    def trigger(self) -> RetrainTrigger:
        return RetrainTrigger(
            schedule_interval_s=99999,  # Disable scheduled
            cooldown_s=0,  # No cooldown for tests
        )

    def test_no_trigger_when_all_clear(self, trigger: RetrainTrigger) -> None:
        result = trigger.check_trigger()
        assert result is None

    def test_trigger_on_feature_drift(self, trigger: RetrainTrigger) -> None:
        result = trigger.check_trigger(drifted_features=["obi", "rsi"])
        assert result == RetrainReason.FEATURE_DRIFT

    def test_trigger_on_sharpe_decline(self, trigger: RetrainTrigger) -> None:
        trigger.set_baseline_sharpe(2.0)
        result = trigger.check_trigger(current_sharpe=0.8)
        assert result == RetrainReason.SHARPE_DECLINE

    def test_no_trigger_sharpe_above_threshold(self, trigger: RetrainTrigger) -> None:
        trigger.set_baseline_sharpe(2.0)
        result = trigger.check_trigger(current_sharpe=1.5)
        assert result is None

    def test_trigger_on_adwin(self, trigger: RetrainTrigger) -> None:
        result = trigger.check_trigger(adwin_detected=True)
        assert result == RetrainReason.ADWIN_CHANGE

    def test_trigger_on_force(self, trigger: RetrainTrigger) -> None:
        result = trigger.check_trigger(force=True)
        assert result == RetrainReason.MANUAL

    def test_scheduled_trigger(self) -> None:
        trigger = RetrainTrigger(
            schedule_interval_s=0,  # Always due
            cooldown_s=0,
        )
        result = trigger.check_trigger()
        assert result == RetrainReason.SCHEDULED

    def test_cooldown_prevents_retrain(self) -> None:
        trigger = RetrainTrigger(
            schedule_interval_s=99999,
            cooldown_s=99999,
        )
        # Simulate a recent retrain
        trainer = make_trainer()
        trigger.execute_retrain(RetrainReason.MANUAL, trainer, timestamp_ms=1000)
        # Now drift trigger should be blocked by cooldown
        result = trigger.check_trigger(drifted_features=["obi"])
        assert result is None

    def test_force_ignores_cooldown(self) -> None:
        trigger = RetrainTrigger(
            schedule_interval_s=99999,
            cooldown_s=99999,
        )
        trainer = make_trainer()
        trigger.execute_retrain(RetrainReason.MANUAL, trainer, timestamp_ms=1000)
        # Force bypasses cooldown
        result = trigger.check_trigger(force=True)
        assert result == RetrainReason.MANUAL

    def test_drift_priority_over_sharpe(self, trigger: RetrainTrigger) -> None:
        trigger.set_baseline_sharpe(2.0)
        result = trigger.check_trigger(
            drifted_features=["obi"],
            current_sharpe=0.5,
        )
        # Drift should fire first (higher priority)
        assert result == RetrainReason.FEATURE_DRIFT


class TestExecuteRetrain:
    @pytest.fixture()
    def trigger(self) -> RetrainTrigger:
        return RetrainTrigger(cooldown_s=0)

    def test_successful_retrain_and_swap(self, trigger: RetrainTrigger) -> None:
        trainer = make_trainer(
            holdout_sharpe=1.5,
            current_sharpe=1.0,
            feature_overlap=0.85,
            calibration_ece=0.03,
        )
        decision = trigger.execute_retrain(
            RetrainReason.FEATURE_DRIFT, trainer, timestamp_ms=1000
        )
        assert decision.model_swapped
        assert decision.validation_result == ValidationResult.PASSED
        trainer.swap_model.assert_called_once()
        assert trigger.swap_count == 1

    def test_failed_performance_validation(self, trigger: RetrainTrigger) -> None:
        trainer = make_trainer(
            holdout_sharpe=0.8,  # Worse than current
            current_sharpe=1.0,
            feature_overlap=0.85,
            calibration_ece=0.03,
        )
        decision = trigger.execute_retrain(
            RetrainReason.SCHEDULED, trainer, timestamp_ms=2000
        )
        assert not decision.model_swapped
        assert decision.validation_result == ValidationResult.FAILED_PERFORMANCE
        trainer.discard_new_model.assert_called_once()

    def test_failed_feature_overlap(self, trigger: RetrainTrigger) -> None:
        trainer = make_trainer(
            holdout_sharpe=1.5,
            current_sharpe=1.0,
            feature_overlap=0.5,  # Below 0.7 threshold
            calibration_ece=0.03,
        )
        decision = trigger.execute_retrain(
            RetrainReason.SCHEDULED, trainer, timestamp_ms=3000
        )
        assert not decision.model_swapped
        assert decision.validation_result == ValidationResult.FAILED_FEATURE_OVERLAP

    def test_failed_calibration(self, trigger: RetrainTrigger) -> None:
        trainer = make_trainer(
            holdout_sharpe=1.5,
            current_sharpe=1.0,
            feature_overlap=0.85,
            calibration_ece=0.08,  # Above 0.05 threshold
        )
        decision = trigger.execute_retrain(
            RetrainReason.SCHEDULED, trainer, timestamp_ms=4000
        )
        assert not decision.model_swapped
        assert decision.validation_result == ValidationResult.FAILED_CALIBRATION

    def test_multiple_failures(self, trigger: RetrainTrigger) -> None:
        trainer = make_trainer(
            holdout_sharpe=0.5,
            current_sharpe=1.0,
            feature_overlap=0.3,
            calibration_ece=0.1,
        )
        decision = trigger.execute_retrain(
            RetrainReason.MANUAL, trainer, timestamp_ms=5000
        )
        assert not decision.model_swapped
        assert decision.validation_result == ValidationResult.FAILED_MULTIPLE

    def test_retrain_exception_handled(self, trigger: RetrainTrigger) -> None:
        trainer = make_trainer(retrain_raises=True)
        decision = trigger.execute_retrain(
            RetrainReason.MANUAL, trainer, timestamp_ms=6000
        )
        assert not decision.model_swapped
        assert decision.validation_result == ValidationResult.FAILED_PERFORMANCE
        assert "error" in decision.details

    def test_retrain_count_increments(self, trigger: RetrainTrigger) -> None:
        trainer = make_trainer()
        trigger.execute_retrain(RetrainReason.MANUAL, trainer, timestamp_ms=1000)
        trigger.execute_retrain(RetrainReason.MANUAL, trainer, timestamp_ms=2000)
        assert trigger.retrain_count == 2

    def test_baseline_updates_on_swap(self, trigger: RetrainTrigger) -> None:
        trigger.set_baseline_sharpe(1.0)
        trainer = make_trainer(holdout_sharpe=1.8, current_sharpe=1.0)
        trigger.execute_retrain(RetrainReason.MANUAL, trainer, timestamp_ms=1000)
        assert trigger.baseline_sharpe == 1.8

    def test_baseline_not_updated_on_failure(self, trigger: RetrainTrigger) -> None:
        trigger.set_baseline_sharpe(1.0)
        trainer = make_trainer(holdout_sharpe=0.5, current_sharpe=1.0)
        trigger.execute_retrain(RetrainReason.MANUAL, trainer, timestamp_ms=1000)
        assert trigger.baseline_sharpe == 1.0

    def test_history_recorded(self, trigger: RetrainTrigger) -> None:
        trainer = make_trainer()
        trigger.execute_retrain(RetrainReason.MANUAL, trainer, timestamp_ms=1000)
        trigger.execute_retrain(RetrainReason.SCHEDULED, trainer, timestamp_ms=2000)
        assert len(trigger.history) == 2
        assert trigger.history[0].reason == RetrainReason.MANUAL
        assert trigger.history[1].reason == RetrainReason.SCHEDULED

    def test_decision_fields_populated(self, trigger: RetrainTrigger) -> None:
        trainer = make_trainer(
            holdout_sharpe=1.5,
            current_sharpe=1.0,
            feature_overlap=0.85,
            calibration_ece=0.03,
        )
        decision = trigger.execute_retrain(
            RetrainReason.FEATURE_DRIFT, trainer, timestamp_ms=7000
        )
        assert decision.timestamp_ms == 7000
        assert decision.old_sharpe == 1.0
        assert decision.new_sharpe == 1.5
        assert decision.feature_overlap == 0.85
        assert decision.calibration_ece == 0.03
        assert decision.triggered


class TestResetCooldown:
    def test_reset_allows_immediate_retrain(self) -> None:
        trigger = RetrainTrigger(
            schedule_interval_s=99999,
            cooldown_s=99999,
        )
        trainer = make_trainer()
        trigger.execute_retrain(RetrainReason.MANUAL, trainer, timestamp_ms=1000)
        # Blocked by cooldown
        assert trigger.check_trigger(drifted_features=["obi"]) is None
        # Reset cooldown
        trigger.reset_cooldown()
        assert trigger.check_trigger(drifted_features=["obi"]) == RetrainReason.FEATURE_DRIFT
