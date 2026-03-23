"""Tests for isotonic calibration module."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from ep2_crypto.models.calibration import CalibrationConfig, IsotonicCalibrator

if TYPE_CHECKING:
    from pathlib import Path


def _make_calibration_data(
    n_samples: int = 500,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic uncalibrated probabilities.

    Creates probabilities that are systematically overconfident
    (a common problem with tree models).
    """
    rng = np.random.default_rng(seed)

    y_true = rng.choice(
        np.array([-1, 0, 1], dtype=np.int8),
        size=n_samples,
        p=[0.3, 0.4, 0.3],
    )

    # Generate overconfident probabilities
    probas = rng.dirichlet([0.5, 0.5, 0.5], size=n_samples)

    # Add signal: push probability toward correct class
    for i in range(n_samples):
        correct = int(y_true[i]) + 1
        probas[i, correct] += rng.uniform(0.5, 1.5)

    # Renormalize
    probas = probas / probas.sum(axis=1, keepdims=True)
    return probas.astype(np.float64), y_true


@pytest.fixture
def cal_data() -> tuple[np.ndarray, np.ndarray]:
    return _make_calibration_data()


class TestCalibrationFit:
    def test_fit_returns_metrics(self, cal_data: tuple) -> None:
        probas, y_true = cal_data
        calibrator = IsotonicCalibrator()
        metrics = calibrator.fit(probas, y_true)
        assert "pre_calibration_ece" in metrics
        assert "post_calibration_ece" in metrics
        assert "ece_improvement" in metrics

    def test_calibration_improves_ece(self, cal_data: tuple) -> None:
        """Calibration should reduce ECE (or at worst not increase it much)."""
        probas, y_true = cal_data
        calibrator = IsotonicCalibrator()
        metrics = calibrator.fit(probas, y_true)
        # Post-calibration ECE should be at most equal to pre
        assert metrics["post_calibration_ece"] <= metrics["pre_calibration_ece"] + 0.02

    def test_is_fitted(self, cal_data: tuple) -> None:
        probas, y_true = cal_data
        calibrator = IsotonicCalibrator()
        assert not calibrator.is_fitted
        calibrator.fit(probas, y_true)
        assert calibrator.is_fitted


class TestCalibrate:
    def test_calibrated_sums_to_one(self, cal_data: tuple) -> None:
        probas, y_true = cal_data
        calibrator = IsotonicCalibrator()
        calibrator.fit(probas, y_true)
        calibrated = calibrator.calibrate(probas)
        np.testing.assert_allclose(
            calibrated.sum(axis=1),
            1.0,
            atol=1e-6,
        )

    def test_calibrated_shape(self, cal_data: tuple) -> None:
        probas, y_true = cal_data
        calibrator = IsotonicCalibrator()
        calibrator.fit(probas, y_true)
        calibrated = calibrator.calibrate(probas)
        assert calibrated.shape == probas.shape

    def test_calibrated_in_range(self, cal_data: tuple) -> None:
        """All calibrated probabilities should be in [0, 1]."""
        probas, y_true = cal_data
        calibrator = IsotonicCalibrator()
        calibrator.fit(probas, y_true)
        calibrated = calibrator.calibrate(probas)
        assert np.all(calibrated >= 0)
        assert np.all(calibrated <= 1)

    def test_calibrate_before_fit_raises(self) -> None:
        calibrator = IsotonicCalibrator()
        probas = np.random.default_rng(42).dirichlet([1, 1, 1], size=10)
        with pytest.raises(RuntimeError, match="not fitted"):
            calibrator.calibrate(probas)


class TestReliabilityCurve:
    def test_reliability_curve_shape(self, cal_data: tuple) -> None:
        probas, y_true = cal_data
        config = CalibrationConfig(n_bins=10)
        calibrator = IsotonicCalibrator(config)
        calibrator.fit(probas, y_true)

        mean_pred, frac_pos, counts = calibrator.reliability_curve(
            probas,
            y_true,
            class_idx=2,
        )
        assert mean_pred.shape == (10,)
        assert frac_pos.shape == (10,)
        assert counts.shape == (10,)

    def test_reliability_curve_values(self, cal_data: tuple) -> None:
        """Mean predicted and fraction positive should be in [0, 1]."""
        probas, y_true = cal_data
        calibrator = IsotonicCalibrator()
        calibrator.fit(probas, y_true)

        mean_pred, frac_pos, counts = calibrator.reliability_curve(
            probas,
            y_true,
            class_idx=2,
        )
        populated = counts > 0
        assert np.all(mean_pred[populated] >= 0)
        assert np.all(mean_pred[populated] <= 1)
        assert np.all(frac_pos[populated] >= 0)
        assert np.all(frac_pos[populated] <= 1)

    def test_calibrated_closer_to_diagonal(self, cal_data: tuple) -> None:
        """After calibration, reliability curve should be closer to y=x."""
        probas, y_true = cal_data
        calibrator = IsotonicCalibrator()
        calibrator.fit(probas, y_true)

        # Uncalibrated
        mp_raw, fp_raw, cnt_raw = calibrator.reliability_curve(
            probas,
            y_true,
            class_idx=2,
        )
        pop_raw = cnt_raw > 5
        raw_gap = np.mean(np.abs(mp_raw[pop_raw] - fp_raw[pop_raw])) if pop_raw.any() else 1.0

        # Calibrated
        cal_probas = calibrator.calibrate(probas)
        mp_cal, fp_cal, cnt_cal = calibrator.reliability_curve(
            cal_probas,
            y_true,
            class_idx=2,
        )
        pop_cal = cnt_cal > 5
        cal_gap = np.mean(np.abs(mp_cal[pop_cal] - fp_cal[pop_cal])) if pop_cal.any() else 1.0

        # Calibrated gap should be smaller or similar
        assert cal_gap <= raw_gap + 0.05


class TestECE:
    def test_perfect_calibration(self) -> None:
        """Truly calibrated predictions (confidence matches accuracy) have low ECE."""
        rng = np.random.default_rng(42)
        n = 600
        # Create calibrated predictions: confidence ~0.6, accuracy ~0.6
        y_true = rng.choice([0, 1, 2], size=n)
        probas = np.full((n, 3), 0.2 / 2, dtype=np.float64)
        for i in range(n):
            # Correct class gets 0.6 confidence, 60% of the time it IS correct
            if rng.random() < 0.6:
                probas[i, y_true[i]] = 0.6
            else:
                wrong = (y_true[i] + 1) % 3
                probas[i, wrong] = 0.6
            remainder = 1.0 - 0.6
            others = [j for j in range(3) if j != np.argmax(probas[i])]
            probas[i, others[0]] = remainder / 2
            probas[i, others[1]] = remainder / 2

        calibrator = IsotonicCalibrator()
        ece = calibrator._compute_ece(probas, y_true.astype(np.int32))
        # Well-calibrated → ECE should be low
        assert ece < 0.15


class TestCalibrationSaveLoad:
    def test_roundtrip(self, cal_data: tuple, tmp_path: Path) -> None:
        probas, y_true = cal_data
        calibrator = IsotonicCalibrator()
        calibrator.fit(probas, y_true)
        original = calibrator.calibrate(probas)

        save_path = tmp_path / "calibrator"
        calibrator.save(save_path)

        loaded = IsotonicCalibrator()
        loaded.load(save_path)
        loaded_output = loaded.calibrate(probas)

        np.testing.assert_allclose(original, loaded_output, atol=1e-10)

    def test_save_before_fit_raises(self, tmp_path: Path) -> None:
        calibrator = IsotonicCalibrator()
        with pytest.raises(RuntimeError, match="not fitted"):
            calibrator.save(tmp_path / "model")
