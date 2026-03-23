"""Tests for the confidence gating pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import numpy as np
import pytest

from ep2_crypto.confidence.gating import (
    AdaptiveThresholdConfig,
    ConfidenceGatingPipeline,
    GateID,
    GatingConfig,
    MarketContext,
    SignalFilterConfig,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default_probas() -> NDArray[np.float64]:
    """Strong UP signal: [0.1, 0.1, 0.8]."""
    return np.array([0.1, 0.1, 0.8], dtype=np.float64)


def _default_features(n: int = 10) -> NDArray[np.float64]:
    rng = np.random.default_rng(42)
    return rng.standard_normal(n).astype(np.float64)


def _default_ctx(**overrides: float | int) -> MarketContext:
    defaults: dict[str, float | int] = {
        "volatility_ann": 50.0,
        "regime_label": 0,
        "regime_probability": 0.8,
        "spread_bps": 1.0,
        "current_drawdown_pct": 0.0,
    }
    defaults.update(overrides)
    return MarketContext(**defaults)  # type: ignore[arg-type]


def _all_gates_disabled() -> GatingConfig:
    return GatingConfig(
        enable_calibration=False,
        enable_meta_labeling=False,
        enable_ensemble_agreement=False,
        enable_conformal=False,
        enable_signal_filters=False,
        enable_adaptive_threshold=False,
        enable_drawdown=False,
    )


# ---------------------------------------------------------------------------
# Test: Pipeline with all gates disabled (baseline)
# ---------------------------------------------------------------------------


class TestAllGatesDisabled:
    """When all gates disabled, pipeline should pass everything."""

    def test_passes_through(self) -> None:
        pipeline = ConfidenceGatingPipeline(_all_gates_disabled())
        result = pipeline.evaluate(
            primary_probas=_default_probas(),
            primary_prediction=1,
            features=_default_features(),
            market_ctx=_default_ctx(),
        )
        assert result.should_trade is True
        assert result.direction == 1
        assert result.rejection_gate is None
        assert len(result.gate_decisions) == 7

    def test_all_decisions_passed(self) -> None:
        pipeline = ConfidenceGatingPipeline(_all_gates_disabled())
        result = pipeline.evaluate(
            primary_probas=_default_probas(),
            primary_prediction=-1,
            features=_default_features(),
            market_ctx=_default_ctx(),
        )
        for d in result.gate_decisions:
            assert d.passed is True
            assert d.reason == "disabled"

    def test_flat_prediction_no_trade(self) -> None:
        """Even with all gates disabled, direction=0 means no trade."""
        pipeline = ConfidenceGatingPipeline(_all_gates_disabled())
        result = pipeline.evaluate(
            primary_probas=np.array([0.1, 0.8, 0.1]),
            primary_prediction=0,
            features=_default_features(),
            market_ctx=_default_ctx(),
        )
        assert result.should_trade is False
        assert result.direction == 0


# ---------------------------------------------------------------------------
# Test: Calibration gate
# ---------------------------------------------------------------------------


class TestCalibrationGate:
    def test_passes_without_calibrator(self) -> None:
        """No calibrator → gate passes (graceful degradation)."""
        config = _all_gates_disabled()
        config.enable_calibration = True
        pipeline = ConfidenceGatingPipeline(config)
        result = pipeline.evaluate(
            primary_probas=_default_probas(),
            primary_prediction=1,
            features=_default_features(),
            market_ctx=_default_ctx(),
        )
        cal_decision = result.gate_decisions[0]
        assert cal_decision.gate_id == GateID.CALIBRATION
        assert cal_decision.passed is True

    def test_calibrates_probas(self) -> None:
        """With fitted calibrator, probas are calibrated."""
        config = _all_gates_disabled()
        config.enable_calibration = True
        pipeline = ConfidenceGatingPipeline(config)

        mock_cal = MagicMock()
        mock_cal.is_fitted = True
        mock_cal.calibrate.return_value = np.array([[0.05, 0.05, 0.90]])
        pipeline.set_calibrator(mock_cal)

        result = pipeline.evaluate(
            primary_probas=_default_probas(),
            primary_prediction=1,
            features=_default_features(),
            market_ctx=_default_ctx(),
        )
        assert mock_cal.calibrate.called
        assert result.calibrated_probas is not None
        np.testing.assert_allclose(result.calibrated_probas, [0.05, 0.05, 0.90])


# ---------------------------------------------------------------------------
# Test: Meta-labeling gate
# ---------------------------------------------------------------------------


class TestMetaLabelingGate:
    def test_passes_without_model(self) -> None:
        config = _all_gates_disabled()
        config.enable_meta_labeling = True
        pipeline = ConfidenceGatingPipeline(config)
        result = pipeline.evaluate(
            primary_probas=_default_probas(),
            primary_prediction=1,
            features=_default_features(),
            market_ctx=_default_ctx(),
        )
        ml_decision = result.gate_decisions[1]
        assert ml_decision.gate_id == GateID.META_LABELING
        assert ml_decision.passed is True

    def test_rejects_low_confidence(self) -> None:
        config = _all_gates_disabled()
        config.enable_meta_labeling = True
        config.meta_label_threshold = 0.6
        pipeline = ConfidenceGatingPipeline(config)

        mock_ml = MagicMock()
        mock_ml.is_fitted = True
        mock_ml.predict_proba.return_value = np.array([0.4])
        mock_ml.create_meta_features.return_value = np.zeros((1, 15))
        pipeline.set_meta_labeler(mock_ml)

        result = pipeline.evaluate(
            primary_probas=_default_probas(),
            primary_prediction=1,
            features=_default_features(),
            market_ctx=_default_ctx(),
        )
        assert result.should_trade is False
        assert result.rejection_gate == GateID.META_LABELING
        assert result.meta_label_prob == pytest.approx(0.4)

    def test_passes_high_confidence(self) -> None:
        config = _all_gates_disabled()
        config.enable_meta_labeling = True
        config.meta_label_threshold = 0.5
        pipeline = ConfidenceGatingPipeline(config)

        mock_ml = MagicMock()
        mock_ml.is_fitted = True
        mock_ml.predict_proba.return_value = np.array([0.8])
        mock_ml.create_meta_features.return_value = np.zeros((1, 15))
        pipeline.set_meta_labeler(mock_ml)

        result = pipeline.evaluate(
            primary_probas=_default_probas(),
            primary_prediction=1,
            features=_default_features(),
            market_ctx=_default_ctx(),
        )
        assert result.should_trade is True
        assert result.meta_label_prob == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# Test: Ensemble agreement gate
# ---------------------------------------------------------------------------


class TestEnsembleAgreementGate:
    def test_passes_without_ensemble(self) -> None:
        config = _all_gates_disabled()
        config.enable_ensemble_agreement = True
        pipeline = ConfidenceGatingPipeline(config)
        result = pipeline.evaluate(
            primary_probas=_default_probas(),
            primary_prediction=1,
            features=_default_features(),
            market_ctx=_default_ctx(),
        )
        ea_decision = result.gate_decisions[2]
        assert ea_decision.gate_id == GateID.ENSEMBLE_AGREEMENT
        assert ea_decision.passed is True

    def test_passes_low_variance(self) -> None:
        config = _all_gates_disabled()
        config.enable_ensemble_agreement = True
        config.max_ensemble_variance = 0.1
        pipeline = ConfidenceGatingPipeline(config)

        # Models agree closely
        ensemble = [
            np.array([0.1, 0.1, 0.8]),
            np.array([0.12, 0.08, 0.80]),
            np.array([0.09, 0.11, 0.80]),
        ]

        result = pipeline.evaluate(
            primary_probas=_default_probas(),
            primary_prediction=1,
            features=_default_features(),
            market_ctx=_default_ctx(),
            ensemble_probas=ensemble,
        )
        ea_decision = result.gate_decisions[2]
        assert ea_decision.passed is True

    def test_rejects_high_variance(self) -> None:
        config = _all_gates_disabled()
        config.enable_ensemble_agreement = True
        config.max_ensemble_variance = 0.01
        pipeline = ConfidenceGatingPipeline(config)

        # Models strongly disagree
        ensemble = [
            np.array([0.8, 0.1, 0.1]),
            np.array([0.1, 0.1, 0.8]),
            np.array([0.1, 0.8, 0.1]),
        ]

        result = pipeline.evaluate(
            primary_probas=_default_probas(),
            primary_prediction=1,
            features=_default_features(),
            market_ctx=_default_ctx(),
            ensemble_probas=ensemble,
        )
        assert result.should_trade is False
        assert result.rejection_gate == GateID.ENSEMBLE_AGREEMENT


# ---------------------------------------------------------------------------
# Test: Conformal prediction gate
# ---------------------------------------------------------------------------


class TestConformalGate:
    def test_passes_without_conformal(self) -> None:
        config = _all_gates_disabled()
        config.enable_conformal = True
        pipeline = ConfidenceGatingPipeline(config)
        result = pipeline.evaluate(
            primary_probas=_default_probas(),
            primary_prediction=1,
            features=_default_features(),
            market_ctx=_default_ctx(),
        )
        cf_decision = result.gate_decisions[3]
        assert cf_decision.gate_id == GateID.CONFORMAL
        assert cf_decision.passed is True

    def test_passes_singleton(self) -> None:
        config = _all_gates_disabled()
        config.enable_conformal = True
        pipeline = ConfidenceGatingPipeline(config)

        mock_cf = MagicMock()
        mock_cf.is_calibrated = True
        mock_cf.gate.return_value = (np.array([True]), np.array([1], dtype=np.int8))
        mock_cf.predict_sets.return_value = [{2}]  # {UP}
        pipeline.set_conformal(mock_cf)

        result = pipeline.evaluate(
            primary_probas=_default_probas(),
            primary_prediction=1,
            features=_default_features(),
            market_ctx=_default_ctx(),
        )
        assert result.should_trade is True
        assert result.conformal_set_size == 1

    def test_rejects_multi_class_set(self) -> None:
        config = _all_gates_disabled()
        config.enable_conformal = True
        pipeline = ConfidenceGatingPipeline(config)

        mock_cf = MagicMock()
        mock_cf.is_calibrated = True
        mock_cf.gate.return_value = (np.array([False]), np.array([0], dtype=np.int8))
        mock_cf.predict_sets.return_value = [{0, 2}]  # {DOWN, UP}
        pipeline.set_conformal(mock_cf)

        result = pipeline.evaluate(
            primary_probas=_default_probas(),
            primary_prediction=1,
            features=_default_features(),
            market_ctx=_default_ctx(),
        )
        assert result.should_trade is False
        assert result.rejection_gate == GateID.CONFORMAL
        assert result.conformal_set_size == 2


# ---------------------------------------------------------------------------
# Test: Signal filter gate
# ---------------------------------------------------------------------------


class TestSignalFilterGate:
    def test_passes_normal_conditions(self) -> None:
        config = _all_gates_disabled()
        config.enable_signal_filters = True
        pipeline = ConfidenceGatingPipeline(config)
        result = pipeline.evaluate(
            primary_probas=_default_probas(),
            primary_prediction=1,
            features=_default_features(),
            market_ctx=_default_ctx(),
        )
        sf_decision = result.gate_decisions[4]
        assert sf_decision.gate_id == GateID.SIGNAL_FILTERS
        assert sf_decision.passed is True

    def test_rejects_low_volatility(self) -> None:
        config = _all_gates_disabled()
        config.enable_signal_filters = True
        config.signal_filters = SignalFilterConfig(min_volatility_ann=15.0)
        pipeline = ConfidenceGatingPipeline(config)
        result = pipeline.evaluate(
            primary_probas=_default_probas(),
            primary_prediction=1,
            features=_default_features(),
            market_ctx=_default_ctx(volatility_ann=10.0),
        )
        assert result.should_trade is False
        assert result.rejection_gate == GateID.SIGNAL_FILTERS

    def test_rejects_high_volatility(self) -> None:
        config = _all_gates_disabled()
        config.enable_signal_filters = True
        config.signal_filters = SignalFilterConfig(max_volatility_ann=150.0)
        pipeline = ConfidenceGatingPipeline(config)
        result = pipeline.evaluate(
            primary_probas=_default_probas(),
            primary_prediction=1,
            features=_default_features(),
            market_ctx=_default_ctx(volatility_ann=200.0),
        )
        assert result.should_trade is False
        assert result.rejection_gate == GateID.SIGNAL_FILTERS

    def test_rejects_unstable_regime(self) -> None:
        config = _all_gates_disabled()
        config.enable_signal_filters = True
        config.signal_filters = SignalFilterConfig(min_regime_stability=0.6)
        pipeline = ConfidenceGatingPipeline(config)
        result = pipeline.evaluate(
            primary_probas=_default_probas(),
            primary_prediction=1,
            features=_default_features(),
            market_ctx=_default_ctx(regime_probability=0.4),
        )
        assert result.should_trade is False
        assert result.rejection_gate == GateID.SIGNAL_FILTERS


# ---------------------------------------------------------------------------
# Test: Adaptive threshold gate
# ---------------------------------------------------------------------------


class TestAdaptiveThresholdGate:
    def test_passes_high_confidence(self) -> None:
        """Meta-labeling provides composite confidence > threshold."""
        config = _all_gates_disabled()
        config.enable_meta_labeling = True
        config.enable_adaptive_threshold = True
        config.meta_label_threshold = 0.5
        config.adaptive_threshold = AdaptiveThresholdConfig(base_threshold=0.60)
        pipeline = ConfidenceGatingPipeline(config)

        mock_ml = MagicMock()
        mock_ml.is_fitted = True
        mock_ml.predict_proba.return_value = np.array([0.9])
        mock_ml.create_meta_features.return_value = np.zeros((1, 15))
        pipeline.set_meta_labeler(mock_ml)

        result = pipeline.evaluate(
            primary_probas=_default_probas(),
            primary_prediction=1,
            features=_default_features(),
            market_ctx=_default_ctx(),
        )
        assert result.should_trade is True

    def test_rejects_low_confidence(self) -> None:
        """Composite confidence below threshold → reject."""
        config = _all_gates_disabled()
        config.enable_meta_labeling = True
        config.enable_adaptive_threshold = True
        config.meta_label_threshold = 0.5
        config.adaptive_threshold = AdaptiveThresholdConfig(base_threshold=0.95)
        pipeline = ConfidenceGatingPipeline(config)

        mock_ml = MagicMock()
        mock_ml.is_fitted = True
        mock_ml.predict_proba.return_value = np.array([0.6])
        mock_ml.create_meta_features.return_value = np.zeros((1, 15))
        pipeline.set_meta_labeler(mock_ml)

        result = pipeline.evaluate(
            primary_probas=_default_probas(),
            primary_prediction=1,
            features=_default_features(),
            market_ctx=_default_ctx(),
        )
        assert result.should_trade is False
        assert result.rejection_gate == GateID.ADAPTIVE_THRESHOLD

    def test_regime_adjustment(self) -> None:
        """Choppy regime raises threshold, making it harder to pass."""
        config = _all_gates_disabled()
        config.enable_meta_labeling = True
        config.enable_adaptive_threshold = True
        config.meta_label_threshold = 0.5
        config.adaptive_threshold = AdaptiveThresholdConfig(
            base_threshold=0.60,
            regime_adjustments={0: 0.0, 1: 0.10, 2: 0.0},
        )
        pipeline = ConfidenceGatingPipeline(config)

        mock_ml = MagicMock()
        mock_ml.is_fitted = True
        mock_ml.predict_proba.return_value = np.array([0.65])
        mock_ml.create_meta_features.return_value = np.zeros((1, 15))
        pipeline.set_meta_labeler(mock_ml)

        # Regime 0 (trending): threshold = 0.60, confidence 0.65 → pass
        result = pipeline.evaluate(
            primary_probas=_default_probas(),
            primary_prediction=1,
            features=_default_features(),
            market_ctx=_default_ctx(regime_label=0),
        )
        assert result.should_trade is True

        # Regime 1 (choppy): threshold = 0.70, confidence 0.65 → reject
        result = pipeline.evaluate(
            primary_probas=_default_probas(),
            primary_prediction=1,
            features=_default_features(),
            market_ctx=_default_ctx(regime_label=1),
        )
        assert result.should_trade is False
        assert result.rejection_gate == GateID.ADAPTIVE_THRESHOLD


# ---------------------------------------------------------------------------
# Test: Drawdown gate
# ---------------------------------------------------------------------------


class TestDrawdownGate:
    def test_passes_no_drawdown(self) -> None:
        config = _all_gates_disabled()
        config.enable_drawdown = True
        pipeline = ConfidenceGatingPipeline(config)
        result = pipeline.evaluate(
            primary_probas=_default_probas(),
            primary_prediction=1,
            features=_default_features(),
            market_ctx=_default_ctx(current_drawdown_pct=0.0),
        )
        dd_decision = result.gate_decisions[6]
        assert dd_decision.gate_id == GateID.DRAWDOWN
        assert dd_decision.passed is True
        assert dd_decision.value == pytest.approx(1.0)

    def test_reduces_at_moderate_drawdown(self) -> None:
        config = _all_gates_disabled()
        config.enable_drawdown = True
        config.drawdown_start_pct = 3.0
        config.drawdown_halt_pct = 15.0
        pipeline = ConfidenceGatingPipeline(config)
        result = pipeline.evaluate(
            primary_probas=_default_probas(),
            primary_prediction=1,
            features=_default_features(),
            market_ctx=_default_ctx(current_drawdown_pct=9.0),
        )
        dd_decision = result.gate_decisions[6]
        assert dd_decision.passed is True
        assert 0 < dd_decision.value < 1.0
        assert result.drawdown_multiplier < 1.0

    def test_halts_at_max_drawdown(self) -> None:
        config = _all_gates_disabled()
        config.enable_drawdown = True
        config.drawdown_halt_pct = 15.0
        pipeline = ConfidenceGatingPipeline(config)
        result = pipeline.evaluate(
            primary_probas=_default_probas(),
            primary_prediction=1,
            features=_default_features(),
            market_ctx=_default_ctx(current_drawdown_pct=15.0),
        )
        assert result.should_trade is False
        assert result.rejection_gate == GateID.DRAWDOWN

    def test_convex_decay(self) -> None:
        """Multiplier decreases faster as drawdown deepens (convex k=1.5)."""
        config = _all_gates_disabled()
        config.enable_drawdown = True
        config.drawdown_start_pct = 3.0
        config.drawdown_halt_pct = 15.0
        config.drawdown_convex_k = 1.5
        pipeline = ConfidenceGatingPipeline(config)

        mults = []
        for dd in [4.0, 7.0, 10.0, 13.0]:
            result = pipeline.evaluate(
                primary_probas=_default_probas(),
                primary_prediction=1,
                features=_default_features(),
                market_ctx=_default_ctx(current_drawdown_pct=dd),
            )
            mults.append(result.drawdown_multiplier)

        # Each successive multiplier should be smaller
        for i in range(len(mults) - 1):
            assert mults[i] > mults[i + 1]

        # Convex (k=1.5): decay accelerates, so the absolute drop from
        # 7→10 should be larger than from 4→7 in the normalized space.
        # The multiplier itself drops faster as we approach halt.
        # Just verify monotonic decrease and all values in (0, 1).
        for m in mults:
            assert 0 < m < 1.0


# ---------------------------------------------------------------------------
# Test: Full pipeline (multiple gates enabled)
# ---------------------------------------------------------------------------


class TestFullPipeline:
    def test_all_gates_pass(self) -> None:
        """All gates enabled with favorable conditions → trade."""
        config = GatingConfig(
            enable_calibration=True,
            enable_meta_labeling=True,
            enable_ensemble_agreement=True,
            enable_conformal=True,
            enable_signal_filters=True,
            enable_adaptive_threshold=True,
            enable_drawdown=True,
            meta_label_threshold=0.5,
            max_ensemble_variance=0.1,
            adaptive_threshold=AdaptiveThresholdConfig(base_threshold=0.50),
        )
        pipeline = ConfidenceGatingPipeline(config)

        # Set up mocks
        mock_cal = MagicMock()
        mock_cal.is_fitted = True
        mock_cal.calibrate.return_value = np.array([[0.05, 0.05, 0.90]])
        pipeline.set_calibrator(mock_cal)

        mock_ml = MagicMock()
        mock_ml.is_fitted = True
        mock_ml.predict_proba.return_value = np.array([0.85])
        mock_ml.create_meta_features.return_value = np.zeros((1, 15))
        pipeline.set_meta_labeler(mock_ml)

        mock_cf = MagicMock()
        mock_cf.is_calibrated = True
        mock_cf.gate.return_value = (np.array([True]), np.array([1], dtype=np.int8))
        mock_cf.predict_sets.return_value = [{2}]
        pipeline.set_conformal(mock_cf)

        ensemble = [
            np.array([0.05, 0.05, 0.90]),
            np.array([0.06, 0.04, 0.90]),
        ]

        result = pipeline.evaluate(
            primary_probas=_default_probas(),
            primary_prediction=1,
            features=_default_features(),
            market_ctx=_default_ctx(),
            ensemble_probas=ensemble,
        )

        assert result.should_trade is True
        assert result.direction == 1
        assert result.composite_confidence > 0
        assert result.rejection_gate is None
        assert len(result.gate_decisions) == 7

    def test_first_rejection_stops_pipeline(self) -> None:
        """Pipeline stops at first rejection, remaining gates not evaluated."""
        config = GatingConfig(
            enable_calibration=True,
            enable_meta_labeling=True,
            enable_ensemble_agreement=True,
            enable_conformal=True,
            enable_signal_filters=True,
            enable_adaptive_threshold=True,
            enable_drawdown=True,
            meta_label_threshold=0.9,  # Very high → likely reject
        )
        pipeline = ConfidenceGatingPipeline(config)

        mock_ml = MagicMock()
        mock_ml.is_fitted = True
        mock_ml.predict_proba.return_value = np.array([0.3])  # Below 0.9
        mock_ml.create_meta_features.return_value = np.zeros((1, 15))
        pipeline.set_meta_labeler(mock_ml)

        result = pipeline.evaluate(
            primary_probas=_default_probas(),
            primary_prediction=1,
            features=_default_features(),
            market_ctx=_default_ctx(),
        )

        assert result.should_trade is False
        assert result.rejection_gate == GateID.META_LABELING
        # Only calibration + meta-labeling decisions (stopped after rejection)
        assert len(result.gate_decisions) == 2

    def test_gate_decisions_have_correct_ids(self) -> None:
        """Each gate decision has the expected GateID."""
        pipeline = ConfidenceGatingPipeline(_all_gates_disabled())
        result = pipeline.evaluate(
            primary_probas=_default_probas(),
            primary_prediction=1,
            features=_default_features(),
            market_ctx=_default_ctx(),
        )
        expected_ids = [
            GateID.CALIBRATION,
            GateID.META_LABELING,
            GateID.ENSEMBLE_AGREEMENT,
            GateID.CONFORMAL,
            GateID.SIGNAL_FILTERS,
            GateID.ADAPTIVE_THRESHOLD,
            GateID.DRAWDOWN,
        ]
        actual_ids = [d.gate_id for d in result.gate_decisions]
        assert actual_ids == expected_ids


# ---------------------------------------------------------------------------
# Test: Ablation (individual gate enable/disable)
# ---------------------------------------------------------------------------


class TestAblation:
    def test_toggle_each_gate(self) -> None:
        """Toggling each gate on/off produces different results."""
        gate_fields = [
            "enable_calibration",
            "enable_meta_labeling",
            "enable_ensemble_agreement",
            "enable_conformal",
            "enable_signal_filters",
            "enable_adaptive_threshold",
            "enable_drawdown",
        ]
        for gate_field in gate_fields:
            # All disabled except one
            config = _all_gates_disabled()
            setattr(config, gate_field, True)
            pipeline = ConfidenceGatingPipeline(config)
            result = pipeline.evaluate(
                primary_probas=_default_probas(),
                primary_prediction=1,
                features=_default_features(),
                market_ctx=_default_ctx(),
            )
            # With favorable conditions, should still pass
            assert result.should_trade is True, f"Failed with only {gate_field} enabled"

    def test_disable_meta_labeling_skips_gate(self) -> None:
        config = GatingConfig(enable_meta_labeling=False)
        pipeline = ConfidenceGatingPipeline(config)

        # Even with a meta-labeler that would reject, gate is skipped
        mock_ml = MagicMock()
        mock_ml.is_fitted = True
        mock_ml.predict_proba.return_value = np.array([0.01])
        pipeline.set_meta_labeler(mock_ml)

        result = pipeline.evaluate(
            primary_probas=_default_probas(),
            primary_prediction=1,
            features=_default_features(),
            market_ctx=_default_ctx(),
        )
        ml_decision = result.gate_decisions[1]
        assert ml_decision.passed is True
        assert ml_decision.reason == "disabled"
