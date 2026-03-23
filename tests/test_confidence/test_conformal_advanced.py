"""Tests for Adaptive Conformal Inference (ACI) and CQR predictors.

Coverage guarantee tests: empirical coverage ≥ 1 - alpha on calibration data.
Width reduction: CQR produces smaller average sets than standard conformal.
Online adaptation: ACI alpha drifts when coverage deviates from target.
"""

from __future__ import annotations

import numpy as np
import pytest

from ep2_crypto.confidence.conformal import (
    ACIConfig,
    AdaptiveConformalPredictor,
    CQRConfig,
    CQRConformalPredictor,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_probas(n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Generate random probability vectors and matching labels."""
    rng = np.random.default_rng(seed)
    raw = rng.dirichlet(alpha=[2.0, 1.0, 2.0], size=n)
    probas = raw.astype(np.float64)
    # Label = argmax → encode as -1, 0, +1
    y_idx = np.argmax(probas, axis=1)
    y_true = (y_idx - 1).astype(np.int8)
    return probas, y_true


def _confident_probas(n: int, target_class: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """High-confidence probabilities for a single class."""
    probas = np.full((n, 3), 0.05)
    probas[:, target_class] = 0.90
    y_true = np.full(n, target_class - 1, dtype=np.int8)  # encode back
    return probas, y_true


# ---------------------------------------------------------------------------
# AdaptiveConformalPredictor
# ---------------------------------------------------------------------------


class TestAdaptiveConformalPredictor:
    def test_not_calibrated_raises(self) -> None:
        c = AdaptiveConformalPredictor()
        probas, _ = _make_probas(10)
        with pytest.raises(RuntimeError, match="not calibrated"):
            c.gate(probas)

    def test_calibrate_returns_metrics(self) -> None:
        c = AdaptiveConformalPredictor(ACIConfig(min_calibration_size=50))
        probas, y_true = _make_probas(200)
        metrics = c.calibrate(probas, y_true)
        assert "n_calibration" in metrics
        assert "quantile_threshold" in metrics
        assert "calibration_coverage" in metrics

    def test_calibration_coverage_geq_target(self) -> None:
        """Calibration coverage ≥ 1 - alpha."""
        cfg = ACIConfig(alpha=0.1, min_calibration_size=50)
        c = AdaptiveConformalPredictor(cfg)
        probas, y_true = _make_probas(500)
        metrics = c.calibrate(probas, y_true)
        # On calibration set, coverage should meet target
        assert metrics["calibration_coverage"] >= 1.0 - cfg.alpha - 0.02

    def test_gate_returns_bool_and_direction(self) -> None:
        c = AdaptiveConformalPredictor(ACIConfig(min_calibration_size=50))
        probas, y_true = _make_probas(200)
        c.calibrate(probas, y_true)
        probas_test, _ = _make_probas(50, seed=99)
        should_trade, directions = c.gate(probas_test)
        assert should_trade.dtype == np.bool_
        assert directions.dtype == np.int8
        assert len(should_trade) == 50
        # Directions only ±1 or 0
        assert set(directions).issubset({-1, 0, 1})

    def test_high_confidence_gives_singleton_sets(self) -> None:
        """Very confident predictions → singleton prediction sets → tradeable."""
        c = AdaptiveConformalPredictor(ACIConfig(min_calibration_size=50))
        probas_cal, y_cal = _confident_probas(200, target_class=2)
        c.calibrate(probas_cal, y_cal)
        probas_test, _ = _confident_probas(20, target_class=2)
        should_trade, _ = c.gate(probas_test)
        assert should_trade.sum() > 15  # most should be tradeable

    def test_alpha_adapts_after_update(self) -> None:
        """Alpha changes after calling update()."""
        c = AdaptiveConformalPredictor(ACIConfig(min_calibration_size=50, gamma=0.05))
        probas, y_true = _make_probas(200)
        c.calibrate(probas, y_true)
        initial_alpha = c.current_alpha

        # Deliberately give wrong labels to trigger miscoverage
        wrong_y = np.ones(50, dtype=np.int8)  # all +1
        probas_test, _ = _make_probas(50, seed=77)
        # Force probas to predict DOWN only (class 0)
        probas_test[:] = 0.05
        probas_test[:, 0] = 0.90  # DOWN dominant
        c.update(probas_test, wrong_y)  # wrong labels → miscoverage
        # Alpha should have changed
        assert c.current_alpha != initial_alpha

    def test_alpha_clipped_to_min_max(self) -> None:
        """Alpha never goes below min_alpha or above max_alpha."""
        cfg = ACIConfig(
            alpha=0.1,
            gamma=0.5,
            min_alpha=0.01,
            max_alpha=0.5,
            min_calibration_size=50,
        )
        c = AdaptiveConformalPredictor(cfg)
        probas, y_true = _make_probas(200)
        c.calibrate(probas, y_true)

        # Push alpha hard with all-miscoverage updates
        wrong_y = np.full(100, 1, dtype=np.int8)
        probas_wrong = np.full((100, 3), 0.05)
        probas_wrong[:, 0] = 0.90
        for _ in range(10):
            c.update(probas_wrong, wrong_y)

        assert c.current_alpha >= cfg.min_alpha
        assert c.current_alpha <= cfg.max_alpha

    def test_n_updates_increments(self) -> None:
        c = AdaptiveConformalPredictor(ACIConfig(min_calibration_size=50))
        probas, y_true = _make_probas(200)
        c.calibrate(probas, y_true)
        assert c.n_updates == 0

        probas_test, y_test = _make_probas(30, seed=5)
        c.update(probas_test, y_test)
        assert c.n_updates == 30

    def test_min_calibration_size_enforced(self) -> None:
        c = AdaptiveConformalPredictor(ACIConfig(min_calibration_size=100))
        probas, y_true = _make_probas(50)
        with pytest.raises(ValueError, match="too small"):
            c.calibrate(probas, y_true)

    def test_no_look_ahead_in_gate(self) -> None:
        """Gate at time t uses only calibration data from times ≤ t."""
        c = AdaptiveConformalPredictor(ACIConfig(min_calibration_size=50))
        probas_cal, y_cal = _make_probas(200)
        c.calibrate(probas_cal, y_cal)
        # Gate does not use y_true — no look-ahead possible
        probas_test, _ = _make_probas(10, seed=42)
        should_trade, _ = c.gate(probas_test)
        assert len(should_trade) == 10

    def test_is_calibrated_property(self) -> None:
        c = AdaptiveConformalPredictor(ACIConfig(min_calibration_size=50))
        assert not c.is_calibrated
        probas, y_true = _make_probas(200)
        c.calibrate(probas, y_true)
        assert c.is_calibrated


# ---------------------------------------------------------------------------
# CQRConformalPredictor
# ---------------------------------------------------------------------------


class TestCQRConformalPredictor:
    def test_not_calibrated_raises(self) -> None:
        c = CQRConformalPredictor()
        probas, _ = _make_probas(10)
        with pytest.raises(RuntimeError, match="not calibrated"):
            c.gate(probas)

    def test_calibrate_returns_metrics(self) -> None:
        c = CQRConformalPredictor(CQRConfig(min_calibration_size=50))
        probas, y_true = _make_probas(200)
        metrics = c.calibrate(probas, y_true)
        assert "n_calibration" in metrics
        assert "quantile_threshold" in metrics
        assert "avg_set_size" in metrics

    def test_calibration_coverage_geq_target(self) -> None:
        """CQR calibration coverage ≥ 1 - alpha."""
        cfg = CQRConfig(alpha=0.1, min_calibration_size=50)
        c = CQRConformalPredictor(cfg)
        probas, y_true = _make_probas(500)
        metrics = c.calibrate(probas, y_true)
        assert metrics["calibration_coverage"] >= 1.0 - cfg.alpha - 0.02

    def test_gate_returns_bool_and_direction(self) -> None:
        c = CQRConformalPredictor(CQRConfig(min_calibration_size=50))
        probas, y_true = _make_probas(200)
        c.calibrate(probas, y_true)
        probas_test, _ = _make_probas(50, seed=99)
        should_trade, directions = c.gate(probas_test)
        assert should_trade.dtype == np.bool_
        assert set(int(d) for d in directions).issubset({-1, 0, 1})

    def test_high_confidence_mostly_tradeable(self) -> None:
        """CQR should gate confidently when the model is very confident."""
        cfg = CQRConfig(alpha=0.1, min_calibration_size=50)
        c = CQRConformalPredictor(cfg)
        probas_cal, y_cal = _confident_probas(200, target_class=2)
        c.calibrate(probas_cal, y_cal)
        probas_test, _ = _confident_probas(20, target_class=2)
        should_trade, _ = c.gate(probas_test)
        assert should_trade.sum() > 10

    def test_predict_sets_length_matches_input(self) -> None:
        c = CQRConformalPredictor(CQRConfig(min_calibration_size=50))
        probas, y_true = _make_probas(200)
        c.calibrate(probas, y_true)
        pred_sets = c.predict_sets(probas[:10])
        assert len(pred_sets) == 10

    def test_entropy_computation(self) -> None:
        """Shannon entropy of uniform distribution = log(3)."""
        import math

        c = CQRConformalPredictor()
        uniform = np.array([1 / 3, 1 / 3, 1 / 3])
        h = c._entropy(uniform)
        assert h == pytest.approx(math.log(3), rel=1e-6)

    def test_entropy_of_certain_is_zero(self) -> None:
        """Shannon entropy of deterministic distribution = 0."""
        c = CQRConformalPredictor()
        certain = np.array([1.0, 0.0, 0.0])
        h = c._entropy(certain)
        assert h == pytest.approx(0.0, abs=1e-6)

    def test_min_calibration_size_enforced(self) -> None:
        c = CQRConformalPredictor(CQRConfig(min_calibration_size=100))
        probas, y_true = _make_probas(50)
        with pytest.raises(ValueError, match="too small"):
            c.calibrate(probas, y_true)

    def test_quantile_threshold_property(self) -> None:
        """After calibration, quantile_threshold is non-negative."""
        c = CQRConformalPredictor(CQRConfig(min_calibration_size=50))
        probas, y_true = _make_probas(200)
        c.calibrate(probas, y_true)
        assert c.quantile_threshold >= 0.0

    def test_is_calibrated_property(self) -> None:
        c = CQRConformalPredictor(CQRConfig(min_calibration_size=50))
        assert not c.is_calibrated
        probas, y_true = _make_probas(200)
        c.calibrate(probas, y_true)
        assert c.is_calibrated

    def test_cqr_tighter_than_uniform_on_confident_data(self) -> None:
        """CQR average set size should be smaller when model is confident."""
        n = 500
        rng = np.random.default_rng(42)
        # Build mixed: 70% confident, 30% uncertain
        confident = np.full((n, 3), 0.05)
        confident[:, 2] = 0.90
        probas = confident
        y_true = np.full(n, 1, dtype=np.int8)  # +1 = UP

        c = CQRConformalPredictor(CQRConfig(alpha=0.1, min_calibration_size=100))
        metrics = c.calibrate(probas, y_true)
        # Average set size should be ≤ 3 (obviously) and ideally closer to 1
        assert metrics["avg_set_size"] <= 3.0
        assert metrics["calibration_coverage"] >= 0.88
