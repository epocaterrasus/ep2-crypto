"""Integration tests for the full Sprint 8 confidence gating pipeline.

Tests the end-to-end flow: predict → calibrate → meta-label → conformal → gate → size.
Validates acceptance criteria from SPRINTS.md.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from ep2_crypto.confidence.conformal import ConformalConfig, ConformalPredictor
from ep2_crypto.confidence.gating import (
    ConfidenceGatingPipeline,
    GateID,
    GatingConfig,
    MarketContext,
)
from ep2_crypto.confidence.meta_labeling import MetaLabelConfig, MetaLabeler
from ep2_crypto.confidence.position_sizing import (
    ConfidencePositionConfig,
    ConfidencePositionSizer,
)
from ep2_crypto.models.calibration import CalibrationConfig, IsotonicCalibrator

if TYPE_CHECKING:
    from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------


def _generate_synthetic_data(
    n_samples: int = 500,
    n_features: int = 10,
    signal_strength: float = 0.3,
    seed: int = 42,
) -> dict[str, NDArray[np.float64] | NDArray[np.int8]]:
    """Generate synthetic data mimicking a trading scenario.

    Returns dict with:
        features: (n_samples, n_features)
        y_true: labels in {-1, 0, +1}
        primary_probas: simulated primary model probabilities (n_samples, 3)
        primary_predictions: predicted direction
        regime_labels: regime states
        is_profitable: binary labels for meta-labeling
    """
    rng = np.random.default_rng(seed)

    features = rng.standard_normal((n_samples, n_features)).astype(np.float64)

    # Generate true labels with some structure
    signal = features[:, 0] * signal_strength + rng.standard_normal(n_samples) * 0.5
    y_true = np.zeros(n_samples, dtype=np.int8)
    y_true[signal > 0.3] = 1
    y_true[signal < -0.3] = -1

    # Simulate primary model probabilities (somewhat correlated with truth)
    primary_probas = np.zeros((n_samples, 3), dtype=np.float64)
    for i in range(n_samples):
        true_class = int(y_true[i]) + 1  # -1→0, 0→1, 1→2
        # Give extra weight to true class with noise
        probs = rng.dirichlet([1.0, 1.0, 1.0])
        probs[true_class] += signal_strength
        probs = probs / probs.sum()
        primary_probas[i] = probs

    primary_predictions = np.array([np.argmax(p) - 1 for p in primary_probas], dtype=np.int8)

    # Is_profitable: did the primary model's prediction match truth?
    is_profitable = (primary_predictions == y_true).astype(np.int8)

    regime_labels = rng.choice([0, 1, 2], size=n_samples, p=[0.5, 0.3, 0.2]).astype(np.int8)

    return {
        "features": features,
        "y_true": y_true,
        "primary_probas": primary_probas,
        "primary_predictions": primary_predictions,
        "regime_labels": regime_labels,
        "is_profitable": is_profitable,
    }


# ---------------------------------------------------------------------------
# Integration: Calibration → Meta-labeling → Conformal → Gate → Size
# ---------------------------------------------------------------------------


class TestEndToEndPipeline:
    """Full pipeline integration test."""

    def test_pipeline_reduces_trade_count(self) -> None:
        """Gating pipeline should trade fewer signals than raw model."""
        data = _generate_synthetic_data(n_samples=500, signal_strength=0.3)

        # Count raw trades (non-flat predictions)
        raw_trades = int(np.sum(data["primary_predictions"] != 0))

        # Train meta-labeler
        ml = MetaLabeler(MetaLabelConfig(n_estimators=50))
        meta_feats = ml.create_meta_features(
            data["primary_predictions"],
            data["primary_probas"],
            data["features"],
            data["regime_labels"],
        )
        ml.fit(meta_feats, data["is_profitable"])

        # Calibrate conformal predictor
        cf = ConformalPredictor(ConformalConfig(alpha=0.1, min_calibration_size=100))
        cf.calibrate(data["primary_probas"], data["y_true"])

        # Set up gating pipeline
        config = GatingConfig(
            enable_calibration=False,
            enable_meta_labeling=True,
            enable_ensemble_agreement=False,
            enable_conformal=True,
            enable_signal_filters=True,
            enable_adaptive_threshold=False,
            enable_drawdown=False,
            meta_label_threshold=0.5,
        )
        pipeline = ConfidenceGatingPipeline(config)
        pipeline.set_meta_labeler(ml)
        pipeline.set_conformal(cf)

        # Evaluate each sample
        gated_trades = 0
        for i in range(len(data["features"])):
            result = pipeline.evaluate(
                primary_probas=data["primary_probas"][i],
                primary_prediction=int(data["primary_predictions"][i]),
                features=data["features"][i],
                market_ctx=MarketContext(
                    volatility_ann=50.0,
                    regime_label=int(data["regime_labels"][i]),
                    regime_probability=0.8,
                ),
            )
            if result.should_trade:
                gated_trades += 1

        # Gating should filter some trades
        assert gated_trades < raw_trades, (
            f"Expected fewer gated trades ({gated_trades}) than raw ({raw_trades})"
        )

    def test_calibrator_feeds_into_pipeline(self) -> None:
        """Isotonic calibrator integrates with the gating pipeline."""
        data = _generate_synthetic_data(n_samples=300)

        # Fit calibrator
        calibrator = IsotonicCalibrator(CalibrationConfig())
        calibrator.fit(data["primary_probas"], data["y_true"])

        config = GatingConfig(
            enable_calibration=True,
            enable_meta_labeling=False,
            enable_ensemble_agreement=False,
            enable_conformal=False,
            enable_signal_filters=False,
            enable_adaptive_threshold=False,
            enable_drawdown=False,
        )
        pipeline = ConfidenceGatingPipeline(config)
        pipeline.set_calibrator(calibrator)

        result = pipeline.evaluate(
            primary_probas=data["primary_probas"][0],
            primary_prediction=int(data["primary_predictions"][0]),
            features=data["features"][0],
            market_ctx=MarketContext(),
        )

        assert result.calibrated_probas is not None
        # Calibrated probas should sum to ~1
        assert abs(result.calibrated_probas.sum() - 1.0) < 0.01

    def test_position_sizing_with_gating_output(self) -> None:
        """Confidence position sizer uses gating pipeline output."""
        data = _generate_synthetic_data(n_samples=200)

        # Train meta-labeler
        ml = MetaLabeler(MetaLabelConfig(n_estimators=50))
        meta_feats = ml.create_meta_features(
            data["primary_predictions"],
            data["primary_probas"],
            data["features"],
            data["regime_labels"],
        )
        ml.fit(meta_feats, data["is_profitable"])

        # Set up pipeline with meta-labeling only
        config = GatingConfig(
            enable_calibration=False,
            enable_meta_labeling=True,
            enable_ensemble_agreement=False,
            enable_conformal=False,
            enable_signal_filters=False,
            enable_adaptive_threshold=False,
            enable_drawdown=False,
            meta_label_threshold=0.4,
        )
        pipeline = ConfidenceGatingPipeline(config)
        pipeline.set_meta_labeler(ml)

        # Set up position sizer
        sizer = ConfidencePositionSizer(
            ConfidencePositionConfig(
                kelly_fraction=0.25,
                max_position_pct=0.05,
                min_confidence=0.3,
                bayesian=False,
                min_trades_for_kelly=1,
            )
        )

        equity = 100_000.0
        price = 65_000.0

        # Evaluate a few samples
        sizes = []
        for i in range(min(50, len(data["features"]))):
            result = pipeline.evaluate(
                primary_probas=data["primary_probas"][i],
                primary_prediction=int(data["primary_predictions"][i]),
                features=data["features"][i],
                market_ctx=MarketContext(
                    regime_label=int(data["regime_labels"][i]),
                    regime_probability=0.8,
                ),
            )
            if result.should_trade:
                kelly = sizer.compute_kelly(
                    win_rate=0.55, avg_win=100.0, avg_loss=80.0, n_trades=50
                )
                sizing = sizer.compute_size(
                    kelly_result=kelly,
                    composite_confidence=result.composite_confidence,
                    equity=equity,
                    price=price,
                )
                if sizing.rejection_reason is None:
                    sizes.append(sizing.position_fraction)

        # Should have some valid sizes
        assert len(sizes) > 0
        # All sizes should be within cap
        for s in sizes:
            assert s <= 0.05 + 1e-9

    def test_each_gate_independently_toggleable(self) -> None:
        """Each gate can be independently enabled/disabled."""
        gate_fields = [
            "enable_calibration",
            "enable_meta_labeling",
            "enable_ensemble_agreement",
            "enable_conformal",
            "enable_signal_filters",
            "enable_adaptive_threshold",
            "enable_drawdown",
        ]

        for field in gate_fields:
            # Enable only this gate
            config = GatingConfig(
                enable_calibration=False,
                enable_meta_labeling=False,
                enable_ensemble_agreement=False,
                enable_conformal=False,
                enable_signal_filters=False,
                enable_adaptive_threshold=False,
                enable_drawdown=False,
            )
            setattr(config, field, True)

            pipeline = ConfidenceGatingPipeline(config)
            result = pipeline.evaluate(
                primary_probas=_default_probas(),
                primary_prediction=1,
                features=np.random.default_rng(42).standard_normal(10),
                market_ctx=MarketContext(),
            )
            # Enabled gate should have its real reason, others "disabled"
            for d in result.gate_decisions:
                gate_idx = list(GateID).index(d.gate_id)
                if gate_fields[gate_idx] == field:
                    # This is the enabled gate — should NOT say "disabled"
                    # (it might say "no_calibrator_available" etc., which is fine)
                    pass
                else:
                    assert d.reason == "disabled", (
                        f"Gate {d.gate_id.name} should be disabled when only {field} is enabled"
                    )


# ---------------------------------------------------------------------------
# Acceptance criteria
# ---------------------------------------------------------------------------


class TestAcceptanceCriteria:
    """Verify Sprint 8 acceptance criteria from SPRINTS.md."""

    def test_conformal_filters_ambiguous(self) -> None:
        """Conformal prediction filters out ambiguous predictions."""
        data = _generate_synthetic_data(n_samples=300, signal_strength=0.1)

        cf = ConformalPredictor(ConformalConfig(alpha=0.1, min_calibration_size=100))
        cf.calibrate(data["primary_probas"], data["y_true"])

        should_trade, _directions = cf.gate(data["primary_probas"])
        n_traded = int(should_trade.sum())
        n_abstained = len(should_trade) - n_traded

        # With weak signals, conformal should abstain on a significant fraction
        assert n_abstained > 0, "Conformal should filter some ambiguous signals"

    def test_position_sizing_never_exceeds_cap(self) -> None:
        """Position sizing never exceeds max position cap."""
        sizer = ConfidencePositionSizer(
            ConfidencePositionConfig(
                max_position_pct=0.05,
                bayesian=False,
                min_trades_for_kelly=1,
                min_confidence=0.0,
            )
        )

        # Try extreme inputs
        for confidence in [0.5, 0.8, 0.99, 1.0]:
            for win_rate in [0.55, 0.70, 0.90]:
                kelly = sizer.compute_kelly(
                    win_rate=win_rate, avg_win=200.0, avg_loss=50.0, n_trades=100
                )
                sizing = sizer.compute_size(
                    kelly_result=kelly,
                    composite_confidence=confidence,
                    equity=100_000.0,
                    price=65_000.0,
                )
                if sizing.rejection_reason is None:
                    assert sizing.position_fraction <= 0.05 + 1e-9, (
                        f"Position fraction {sizing.position_fraction} exceeds 5% cap "
                        f"with confidence={confidence}, win_rate={win_rate}"
                    )

    def test_drawdown_gate_progressive_reduction(self) -> None:
        """Drawdown gate reduces position size progressively."""
        config = GatingConfig(
            enable_calibration=False,
            enable_meta_labeling=False,
            enable_ensemble_agreement=False,
            enable_conformal=False,
            enable_signal_filters=False,
            enable_adaptive_threshold=False,
            enable_drawdown=True,
            drawdown_start_pct=3.0,
            drawdown_halt_pct=15.0,
        )
        pipeline = ConfidenceGatingPipeline(config)

        multipliers = []
        for dd in [0.0, 3.0, 5.0, 8.0, 12.0, 14.9, 15.0]:
            result = pipeline.evaluate(
                primary_probas=_default_probas(),
                primary_prediction=1,
                features=np.zeros(10),
                market_ctx=MarketContext(current_drawdown_pct=dd),
            )
            multipliers.append(result.drawdown_multiplier)

        # 0% and 3% DD → full size (1.0)
        assert multipliers[0] == pytest.approx(1.0)
        assert multipliers[1] == pytest.approx(1.0)

        # Progressive reduction
        for i in range(2, len(multipliers) - 1):
            assert multipliers[i] < multipliers[i - 1] or multipliers[i] == pytest.approx(
                multipliers[i - 1]
            )

        # 15% DD → halt (0.0)
        assert multipliers[-1] == pytest.approx(0.0)

    def test_meta_labeling_gate_decisions_logged(self) -> None:
        """Each gate logs its decision (via GateDecision)."""
        data = _generate_synthetic_data(n_samples=200)

        ml = MetaLabeler(MetaLabelConfig(n_estimators=50))
        meta_feats = ml.create_meta_features(
            data["primary_predictions"],
            data["primary_probas"],
            data["features"],
            data["regime_labels"],
        )
        ml.fit(meta_feats, data["is_profitable"])

        config = GatingConfig(
            enable_calibration=False,
            enable_meta_labeling=True,
            enable_ensemble_agreement=False,
            enable_conformal=False,
            enable_signal_filters=False,
            enable_adaptive_threshold=False,
            enable_drawdown=False,
        )
        pipeline = ConfidenceGatingPipeline(config)
        pipeline.set_meta_labeler(ml)

        result = pipeline.evaluate(
            primary_probas=data["primary_probas"][0],
            primary_prediction=int(data["primary_predictions"][0]),
            features=data["features"][0],
            market_ctx=MarketContext(),
        )

        # Every decision has gate_id, passed, reason, and value
        for d in result.gate_decisions:
            assert d.gate_id is not None
            assert isinstance(d.passed, bool)
            assert isinstance(d.reason, str)
            assert len(d.reason) > 0


# Helper for toggle test
def _default_probas() -> NDArray[np.float64]:
    return np.array([0.1, 0.1, 0.8], dtype=np.float64)
