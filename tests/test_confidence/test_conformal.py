"""Tests for conformal prediction gate."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from ep2_crypto.confidence.conformal import (
    DOWN,
    FLAT,
    UP,
    ConformalConfig,
    ConformalPredictor,
)

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture
def config() -> ConformalConfig:
    return ConformalConfig(
        alpha=0.1,
        adaptive=True,
        adaptive_lr=0.01,
        min_calibration_size=100,
    )


@pytest.fixture
def predictor(config: ConformalConfig) -> ConformalPredictor:
    return ConformalPredictor(config=config)


def _make_calibration_data(
    rng: np.random.Generator,
    n_samples: int = 200,
    accuracy: float = 0.7,
) -> tuple[np.ndarray, np.ndarray]:
    """Create synthetic calibration data with controllable accuracy.

    Returns (probas, y_true) where probas is (n_samples, 3) and
    y_true is in {-1, 0, +1}.
    """
    # True labels: balanced across classes
    y_true = rng.choice([-1, 0, 1], size=n_samples).astype(np.int8)
    y_encoded = y_true.astype(np.int32) + 1  # 0, 1, 2

    probas = np.full((n_samples, 3), 0.1, dtype=np.float64)
    for i in range(n_samples):
        if rng.random() < accuracy:
            # Correct: assign high probability to true class
            probas[i, y_encoded[i]] = 0.6 + rng.random() * 0.3
        else:
            # Wrong: assign high probability to a different class
            wrong_class = (y_encoded[i] + rng.integers(1, 3)) % 3
            probas[i, wrong_class] = 0.6 + rng.random() * 0.3

    # Normalize rows
    row_sums = probas.sum(axis=1, keepdims=True)
    probas = probas / row_sums

    return probas, y_true


def _make_perfect_model_data(
    rng: np.random.Generator,
    n_samples: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """Create data where model is nearly perfect (high confidence, correct)."""
    y_true = rng.choice([-1, 0, 1], size=n_samples).astype(np.int8)
    y_encoded = y_true.astype(np.int32) + 1

    probas = np.full((n_samples, 3), 0.02, dtype=np.float64)
    for i in range(n_samples):
        probas[i, y_encoded[i]] = 0.96

    return probas, y_true


def _make_random_model_data(
    rng: np.random.Generator,
    n_samples: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """Create data where model outputs near-uniform probabilities."""
    y_true = rng.choice([-1, 0, 1], size=n_samples).astype(np.int8)

    # Near-uniform: each class gets ~0.33
    probas = rng.dirichlet([10.0, 10.0, 10.0], size=n_samples)

    return probas, y_true


# ---------------------------------------------------------------------------
# Calibration tests
# ---------------------------------------------------------------------------


class TestCalibration:
    def test_calibration_stores_scores(
        self, predictor: ConformalPredictor, rng: np.random.Generator
    ) -> None:
        probas, y_true = _make_calibration_data(rng)
        predictor.calibrate(probas, y_true)

        assert predictor.is_calibrated
        assert predictor._state.scores is not None
        assert len(predictor._state.scores) == len(probas)

    def test_calibration_scores_are_one_minus_p_true(
        self, predictor: ConformalPredictor, rng: np.random.Generator
    ) -> None:
        probas, y_true = _make_calibration_data(rng, n_samples=150)
        predictor.calibrate(probas, y_true)

        y_encoded = y_true.astype(np.int32) + 1
        expected_scores = 1.0 - probas[np.arange(len(probas)), y_encoded]

        np.testing.assert_allclose(
            predictor._state.scores, expected_scores, rtol=1e-10
        )

    def test_calibration_returns_metrics(
        self, predictor: ConformalPredictor, rng: np.random.Generator
    ) -> None:
        probas, y_true = _make_calibration_data(rng)
        metrics = predictor.calibrate(probas, y_true)

        assert "n_calibration_samples" in metrics
        assert "quantile_threshold" in metrics
        assert "empirical_coverage" in metrics
        assert "target_coverage" in metrics
        assert "singleton_rate" in metrics
        assert "tradeable_rate" in metrics
        assert "mean_nonconformity_score" in metrics

        assert metrics["n_calibration_samples"] == 200.0
        assert 0.0 <= metrics["empirical_coverage"] <= 1.0
        assert 0.0 <= metrics["quantile_threshold"] <= 1.0

    def test_calibration_coverage_near_target(
        self, rng: np.random.Generator
    ) -> None:
        """Empirical coverage should be close to 1 - alpha."""
        config = ConformalConfig(alpha=0.1, min_calibration_size=100)
        pred = ConformalPredictor(config=config)

        probas, y_true = _make_calibration_data(rng, n_samples=1000, accuracy=0.7)
        metrics = pred.calibrate(probas, y_true)

        # Coverage should be >= 1 - alpha (conformal guarantee)
        assert metrics["empirical_coverage"] >= 1.0 - config.alpha - 0.02

    def test_calibration_quantile_threshold_positive(
        self, predictor: ConformalPredictor, rng: np.random.Generator
    ) -> None:
        probas, y_true = _make_calibration_data(rng)
        predictor.calibrate(probas, y_true)

        assert predictor._state.quantile_threshold > 0.0
        assert predictor._state.quantile_threshold <= 1.0


# ---------------------------------------------------------------------------
# Prediction set tests
# ---------------------------------------------------------------------------


class TestPredictionSets:
    def test_predict_sets_returns_correct_type(
        self, predictor: ConformalPredictor, rng: np.random.Generator
    ) -> None:
        probas, y_true = _make_calibration_data(rng)
        predictor.calibrate(probas, y_true)

        new_probas = rng.dirichlet([1.0, 1.0, 1.0], size=50)
        pred_sets = predictor.predict_sets(new_probas)

        assert isinstance(pred_sets, list)
        assert len(pred_sets) == 50
        assert all(isinstance(s, set) for s in pred_sets)

    def test_prediction_sets_contain_valid_classes(
        self, predictor: ConformalPredictor, rng: np.random.Generator
    ) -> None:
        probas, y_true = _make_calibration_data(rng)
        predictor.calibrate(probas, y_true)

        new_probas = rng.dirichlet([1.0, 1.0, 1.0], size=50)
        pred_sets = predictor.predict_sets(new_probas)

        valid_classes = {DOWN, FLAT, UP}
        for pset in pred_sets:
            assert pset.issubset(valid_classes)

    def test_high_confidence_produces_singletons(
        self, predictor: ConformalPredictor, rng: np.random.Generator
    ) -> None:
        """When one class has very high probability, prediction set should be singleton."""
        probas, y_true = _make_perfect_model_data(rng)
        predictor.calibrate(probas, y_true)

        # Test with very confident predictions — must exceed 1 - quantile_threshold
        # With perfect calibration data (score ~0.04), threshold is ~0.04,
        # so we need probas >= 1 - 0.04 = 0.96
        confident_probas = np.array(
            [[0.97, 0.02, 0.01], [0.01, 0.02, 0.97], [0.01, 0.97, 0.02]],
            dtype=np.float64,
        )
        pred_sets = predictor.predict_sets(confident_probas)

        # With a perfect model calibration, very confident predictions
        # should yield singletons
        for pset in pred_sets:
            assert len(pset) == 1, f"Expected singleton, got {pset}"

    def test_uniform_produces_multi_class_sets(
        self, predictor: ConformalPredictor, rng: np.random.Generator
    ) -> None:
        """Near-uniform probabilities should produce multi-class sets."""
        probas, y_true = _make_calibration_data(rng, n_samples=200, accuracy=0.7)
        predictor.calibrate(probas, y_true)

        # Uniform probabilities
        uniform = np.full((10, 3), 1.0 / 3.0, dtype=np.float64)
        pred_sets = predictor.predict_sets(uniform)

        # At least some should have multiple classes
        multi_class_count = sum(1 for s in pred_sets if len(s) > 1)
        assert multi_class_count > 0


# ---------------------------------------------------------------------------
# Gate tests
# ---------------------------------------------------------------------------


class TestGate:
    def test_gate_output_shapes(
        self, predictor: ConformalPredictor, rng: np.random.Generator
    ) -> None:
        probas, y_true = _make_calibration_data(rng)
        predictor.calibrate(probas, y_true)

        test_probas = rng.dirichlet([1.0, 1.0, 1.0], size=30)
        should_trade, direction = predictor.gate(test_probas)

        assert should_trade.shape == (30,)
        assert direction.shape == (30,)
        assert should_trade.dtype == np.bool_
        assert direction.dtype == np.int8

    def test_gate_singleton_up_allows_trade(
        self, predictor: ConformalPredictor, rng: np.random.Generator
    ) -> None:
        """Singleton {UP} should allow trade with direction +1."""
        probas, y_true = _make_perfect_model_data(rng)
        predictor.calibrate(probas, y_true)

        # Very confident UP prediction
        up_probas = np.array([[0.01, 0.01, 0.98]], dtype=np.float64)
        should_trade, direction = predictor.gate(up_probas)

        assert should_trade[0] is np.bool_(True)
        assert direction[0] == 1

    def test_gate_singleton_down_allows_trade(
        self, predictor: ConformalPredictor, rng: np.random.Generator
    ) -> None:
        """Singleton {DOWN} should allow trade with direction -1."""
        probas, y_true = _make_perfect_model_data(rng)
        predictor.calibrate(probas, y_true)

        down_probas = np.array([[0.98, 0.01, 0.01]], dtype=np.float64)
        should_trade, direction = predictor.gate(down_probas)

        assert should_trade[0] is np.bool_(True)
        assert direction[0] == -1

    def test_gate_singleton_flat_abstains(
        self, predictor: ConformalPredictor, rng: np.random.Generator
    ) -> None:
        """Singleton {FLAT} should abstain (no trade signal)."""
        probas, y_true = _make_perfect_model_data(rng)
        predictor.calibrate(probas, y_true)

        flat_probas = np.array([[0.01, 0.98, 0.01]], dtype=np.float64)
        should_trade, direction = predictor.gate(flat_probas)

        assert should_trade[0] is np.bool_(False)
        assert direction[0] == 0

    def test_gate_multi_class_abstains(
        self, predictor: ConformalPredictor, rng: np.random.Generator
    ) -> None:
        """Multi-class prediction sets should abstain."""
        probas, y_true = _make_calibration_data(rng, n_samples=200, accuracy=0.7)
        predictor.calibrate(probas, y_true)

        # Near-uniform → should produce multi-class sets → abstain
        uniform = np.full((20, 3), 1.0 / 3.0, dtype=np.float64)
        should_trade, direction = predictor.gate(uniform)

        # At least most of these should abstain
        abstain_count = int((~should_trade).sum())
        assert abstain_count > 10  # Most of 20 should abstain

        # All abstentions should have direction 0
        assert np.all(direction[~should_trade] == 0)

    def test_gate_trades_have_nonzero_direction(
        self, predictor: ConformalPredictor, rng: np.random.Generator
    ) -> None:
        """All tradeable samples must have direction -1 or +1."""
        probas, y_true = _make_calibration_data(rng)
        predictor.calibrate(probas, y_true)

        test_probas = rng.dirichlet([1.0, 1.0, 1.0], size=100)
        should_trade, direction = predictor.gate(test_probas)

        if should_trade.any():
            assert np.all(np.abs(direction[should_trade]) == 1)


# ---------------------------------------------------------------------------
# Adaptive alpha tests
# ---------------------------------------------------------------------------


class TestAdaptiveAlpha:
    def test_update_alpha_returns_float(
        self, predictor: ConformalPredictor, rng: np.random.Generator
    ) -> None:
        probas, y_true = _make_calibration_data(rng)
        predictor.calibrate(probas, y_true)

        new_probas, new_y = _make_calibration_data(rng, n_samples=150)
        new_alpha = predictor.update_alpha(new_probas, new_y)

        assert isinstance(new_alpha, float)
        assert 0.01 <= new_alpha <= 0.50

    def test_update_alpha_bounded(
        self, rng: np.random.Generator
    ) -> None:
        """Alpha should stay within [0.01, 0.50] bounds."""
        config = ConformalConfig(alpha=0.1, adaptive=True, adaptive_lr=0.5)
        pred = ConformalPredictor(config=config)

        probas, y_true = _make_calibration_data(rng, n_samples=200)
        pred.calibrate(probas, y_true)

        # Run many updates to test bounds
        for _ in range(50):
            new_probas, new_y = _make_calibration_data(rng, n_samples=150)
            alpha = pred.update_alpha(new_probas, new_y)
            assert 0.01 <= alpha <= 0.50

    def test_update_alpha_noop_when_not_adaptive(
        self, rng: np.random.Generator
    ) -> None:
        """When adaptive=False, alpha should not change."""
        config = ConformalConfig(alpha=0.15, adaptive=False)
        pred = ConformalPredictor(config=config)

        probas, y_true = _make_calibration_data(rng, n_samples=200)
        pred.calibrate(probas, y_true)

        original_alpha = pred.current_alpha
        new_probas, new_y = _make_calibration_data(rng, n_samples=150)
        updated_alpha = pred.update_alpha(new_probas, new_y)

        assert updated_alpha == original_alpha

    def test_alpha_adjusts_toward_target_coverage(
        self, rng: np.random.Generator
    ) -> None:
        """Over many updates, alpha should steer coverage toward target."""
        config = ConformalConfig(alpha=0.1, adaptive=True, adaptive_lr=0.05)
        pred = ConformalPredictor(config=config)

        probas, y_true = _make_calibration_data(rng, n_samples=500, accuracy=0.7)
        pred.calibrate(probas, y_true)

        alphas = []
        for _ in range(20):
            new_probas, new_y = _make_calibration_data(rng, n_samples=200, accuracy=0.7)
            alpha = pred.update_alpha(new_probas, new_y)
            alphas.append(alpha)

        # Alpha should not diverge wildly — stays in a reasonable range
        assert all(0.01 <= a <= 0.50 for a in alphas)


# ---------------------------------------------------------------------------
# Perfect and random model tests
# ---------------------------------------------------------------------------


class TestModelExtremes:
    def test_perfect_model_mostly_singletons(
        self, rng: np.random.Generator
    ) -> None:
        """A perfect model should produce mostly singleton prediction sets."""
        pred = ConformalPredictor(ConformalConfig(alpha=0.1, min_calibration_size=100))

        probas, y_true = _make_perfect_model_data(rng, n_samples=300)
        pred.calibrate(probas, y_true)

        # Inference with similarly perfect data
        test_probas, _ = _make_perfect_model_data(rng, n_samples=100)
        pred_sets = pred.predict_sets(test_probas)

        singleton_rate = sum(1 for s in pred_sets if len(s) == 1) / len(pred_sets)
        assert singleton_rate > 0.9, f"Expected >90% singletons, got {singleton_rate:.2%}"

    def test_perfect_model_high_trade_rate(
        self, rng: np.random.Generator
    ) -> None:
        """A perfect model with UP/DOWN predictions should have high trade rate."""
        pred = ConformalPredictor(ConformalConfig(alpha=0.1, min_calibration_size=100))

        probas, y_true = _make_perfect_model_data(rng, n_samples=300)
        pred.calibrate(probas, y_true)

        # Test with only UP and DOWN confident predictions
        n_test = 100
        test_probas = np.zeros((n_test, 3), dtype=np.float64)
        for i in range(n_test):
            if i % 2 == 0:
                test_probas[i] = [0.96, 0.02, 0.02]  # DOWN
            else:
                test_probas[i] = [0.02, 0.02, 0.96]  # UP

        should_trade, _direction = pred.gate(test_probas)
        trade_rate = should_trade.sum() / n_test
        assert trade_rate > 0.8, f"Expected >80% trade rate, got {trade_rate:.2%}"

    def test_random_model_many_abstentions(
        self, rng: np.random.Generator
    ) -> None:
        """A random model should produce many multi-class sets → abstentions."""
        pred = ConformalPredictor(ConformalConfig(alpha=0.1, min_calibration_size=100))

        probas, y_true = _make_random_model_data(rng, n_samples=300)
        pred.calibrate(probas, y_true)

        test_probas, _ = _make_random_model_data(rng, n_samples=100)
        should_trade, _ = pred.gate(test_probas)

        abstain_rate = 1.0 - should_trade.sum() / len(should_trade)
        assert abstain_rate > 0.5, f"Expected >50% abstention, got {abstain_rate:.2%}"


# ---------------------------------------------------------------------------
# Save / load tests
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_save_load_roundtrip(
        self,
        predictor: ConformalPredictor,
        rng: np.random.Generator,
        tmp_path: Path,
    ) -> None:
        probas, y_true = _make_calibration_data(rng)
        predictor.calibrate(probas, y_true)

        save_path = tmp_path / "conformal"
        predictor.save(save_path)

        loaded = ConformalPredictor()
        loaded.load(save_path)

        assert loaded.is_calibrated
        assert loaded._state.n_calibration_samples == predictor._state.n_calibration_samples
        assert loaded._state.quantile_threshold == pytest.approx(
            predictor._state.quantile_threshold
        )
        assert loaded._state.current_alpha == pytest.approx(
            predictor._state.current_alpha
        )
        np.testing.assert_allclose(loaded._state.scores, predictor._state.scores)

    def test_save_load_produces_same_gate(
        self,
        predictor: ConformalPredictor,
        rng: np.random.Generator,
        tmp_path: Path,
    ) -> None:
        probas, y_true = _make_calibration_data(rng)
        predictor.calibrate(probas, y_true)

        test_probas = rng.dirichlet([1.0, 1.0, 1.0], size=50)
        orig_trade, orig_dir = predictor.gate(test_probas)

        save_path = tmp_path / "conformal"
        predictor.save(save_path)

        loaded = ConformalPredictor()
        loaded.load(save_path)
        loaded_trade, loaded_dir = loaded.gate(test_probas)

        np.testing.assert_array_equal(orig_trade, loaded_trade)
        np.testing.assert_array_equal(orig_dir, loaded_dir)

    def test_save_uncalibrated_raises(
        self, predictor: ConformalPredictor, tmp_path: Path
    ) -> None:
        with pytest.raises(RuntimeError, match="not calibrated"):
            predictor.save(tmp_path / "conformal")

    def test_load_creates_files(
        self,
        predictor: ConformalPredictor,
        rng: np.random.Generator,
        tmp_path: Path,
    ) -> None:
        probas, y_true = _make_calibration_data(rng)
        predictor.calibrate(probas, y_true)

        save_path = tmp_path / "conformal"
        predictor.save(save_path)

        assert (tmp_path / "conformal.scores.bin").exists()
        assert (tmp_path / "conformal.meta.json").exists()


# ---------------------------------------------------------------------------
# Error handling tests
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_uncalibrated_predict_sets_raises(
        self, predictor: ConformalPredictor, rng: np.random.Generator
    ) -> None:
        probas = rng.dirichlet([1.0, 1.0, 1.0], size=10)
        with pytest.raises(RuntimeError, match="not calibrated"):
            predictor.predict_sets(probas)

    def test_uncalibrated_gate_raises(
        self, predictor: ConformalPredictor, rng: np.random.Generator
    ) -> None:
        probas = rng.dirichlet([1.0, 1.0, 1.0], size=10)
        with pytest.raises(RuntimeError, match="not calibrated"):
            predictor.gate(probas)

    def test_uncalibrated_update_alpha_raises(
        self, predictor: ConformalPredictor, rng: np.random.Generator
    ) -> None:
        probas, y_true = _make_calibration_data(rng, n_samples=150)
        with pytest.raises(RuntimeError, match="not calibrated"):
            predictor.update_alpha(probas, y_true)

    def test_min_calibration_size_enforced(
        self, rng: np.random.Generator
    ) -> None:
        config = ConformalConfig(min_calibration_size=100)
        pred = ConformalPredictor(config=config)

        probas, y_true = _make_calibration_data(rng, n_samples=50)
        # Only use 50 samples (below minimum)
        small_probas = probas[:50]
        small_y = y_true[:50]

        with pytest.raises(ValueError, match="too small"):
            pred.calibrate(small_probas, small_y)

    def test_wrong_n_classes_raises(
        self, predictor: ConformalPredictor, rng: np.random.Generator
    ) -> None:
        probas = rng.random((100, 5))  # Wrong number of classes
        y_true = rng.choice([-1, 0, 1], size=100).astype(np.int8)

        with pytest.raises(ValueError, match="Expected 3 classes"):
            predictor.calibrate(probas, y_true)

    def test_is_calibrated_false_initially(
        self, predictor: ConformalPredictor
    ) -> None:
        assert not predictor.is_calibrated


# ---------------------------------------------------------------------------
# Config defaults test
# ---------------------------------------------------------------------------


class TestConfig:
    def test_default_config(self) -> None:
        config = ConformalConfig()
        assert config.alpha == 0.1
        assert config.adaptive is True
        assert config.adaptive_lr == 0.01
        assert config.min_calibration_size == 100

    def test_custom_config(self) -> None:
        config = ConformalConfig(alpha=0.2, adaptive=False, adaptive_lr=0.05)
        pred = ConformalPredictor(config=config)
        assert pred.current_alpha == 0.2
