"""Tests for stacking ensemble meta-learner."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from ep2_crypto.models.stacking import StackingEnsemble

if TYPE_CHECKING:
    from pathlib import Path


def _make_base_predictions(
    n_samples: int = 300,
    seed: int = 42,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Generate synthetic base model OOF predictions with signal."""
    rng = np.random.default_rng(seed)

    # True labels
    y_true = rng.choice(
        np.array([-1, 0, 1], dtype=np.int8),
        size=n_samples,
        p=[0.3, 0.4, 0.3],
    )

    base_probas = []
    for model_seed in [seed, seed + 1, seed + 2]:
        model_rng = np.random.default_rng(model_seed)
        proba = model_rng.dirichlet([1, 1, 1], size=n_samples)

        # Add signal: increase probability of correct class
        for i in range(n_samples):
            correct_class = int(y_true[i]) + 1  # -1→0, 0→1, 1→2
            proba[i, correct_class] += model_rng.uniform(0.3, 0.8)

        # Renormalize
        proba = proba / proba.sum(axis=1, keepdims=True)
        base_probas.append(proba.astype(np.float64))

    return base_probas, y_true


@pytest.fixture
def base_preds() -> tuple[list[np.ndarray], np.ndarray]:
    return _make_base_predictions()


class TestStackingTrain:
    def test_train_returns_metrics(self, base_preds: tuple) -> None:
        base_probas, y_true = base_preds
        ensemble = StackingEnsemble()
        metrics = ensemble.train(
            base_probas,
            y_true,
            base_model_names=["lgbm", "catboost", "gru"],
        )
        assert "meta_train_accuracy" in metrics
        assert metrics["meta_train_accuracy"] > 0.4
        assert metrics["n_base_models"] == 3.0
        assert metrics["n_meta_features"] == 9.0  # 3 models * 3 classes

    def test_is_fitted(self, base_preds: tuple) -> None:
        base_probas, y_true = base_preds
        ensemble = StackingEnsemble()
        assert not ensemble.is_fitted
        ensemble.train(base_probas, y_true)
        assert ensemble.is_fitted


class TestStackingPredict:
    def test_predict_shape(self, base_preds: tuple) -> None:
        base_probas, y_true = base_preds
        ensemble = StackingEnsemble()
        ensemble.train(base_probas, y_true)
        preds = ensemble.predict(base_probas)
        assert preds.shape == (len(y_true),)
        assert set(np.unique(preds)).issubset({-1, 0, 1})

    def test_predict_proba_shape(self, base_preds: tuple) -> None:
        base_probas, y_true = base_preds
        ensemble = StackingEnsemble()
        ensemble.train(base_probas, y_true)
        proba = ensemble.predict_proba(base_probas)
        assert proba.shape == (len(y_true), 3)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_predict_before_train_raises(self) -> None:
        ensemble = StackingEnsemble()
        probas = [np.random.default_rng(42).dirichlet([1, 1, 1], size=10)]
        with pytest.raises(RuntimeError, match="not fitted"):
            ensemble.predict(probas)

    def test_stacking_beats_random(self, base_preds: tuple) -> None:
        """Stacking should be better than random (33%)."""
        base_probas, y_true = base_preds
        ensemble = StackingEnsemble()
        ensemble.train(base_probas, y_true)
        preds = ensemble.predict(base_probas)
        accuracy = np.mean(preds == y_true)
        assert accuracy > 0.40  # Better than random


class TestSimpleAverage:
    def test_simple_average(self, base_preds: tuple) -> None:
        base_probas, y_true = base_preds
        ensemble = StackingEnsemble()
        preds = ensemble.simple_average_predict(base_probas)
        assert preds.shape == (len(y_true),)
        assert set(np.unique(preds)).issubset({-1, 0, 1})


class TestStackingImprovement:
    def test_stacking_at_least_as_good_as_average(
        self,
        base_preds: tuple,
    ) -> None:
        """Stacking meta-learner should match or beat simple average."""
        base_probas, y_true = base_preds
        ensemble = StackingEnsemble()
        ensemble.train(base_probas, y_true)

        stacking_preds = ensemble.predict(base_probas)
        average_preds = ensemble.simple_average_predict(base_probas)

        stacking_acc = np.mean(stacking_preds == y_true)
        average_acc = np.mean(average_preds == y_true)

        # Stacking should be at least as good (on training data, likely better)
        assert stacking_acc >= average_acc - 0.02


class TestWeightOptimization:
    def test_optimize_weights(self, base_preds: tuple) -> None:
        base_probas, y_true = base_preds
        ensemble = StackingEnsemble()
        weights = ensemble.optimize_weights(base_probas, y_true)
        assert weights.shape == (3,)
        assert np.isclose(weights.sum(), 1.0)
        assert np.all(weights >= 0)


class TestStackingSaveLoad:
    def test_roundtrip(
        self,
        base_preds: tuple,
        tmp_path: Path,
    ) -> None:
        base_probas, y_true = base_preds
        ensemble = StackingEnsemble()
        ensemble.train(
            base_probas,
            y_true,
            base_model_names=["lgbm", "catboost", "gru"],
        )

        original_preds = ensemble.predict(base_probas)
        original_proba = ensemble.predict_proba(base_probas)

        save_path = tmp_path / "stacking"
        ensemble.save(save_path)

        loaded = StackingEnsemble()
        loaded.load(save_path)
        loaded_preds = loaded.predict(base_probas)
        loaded_proba = loaded.predict_proba(base_probas)

        np.testing.assert_array_equal(original_preds, loaded_preds)
        np.testing.assert_allclose(original_proba, loaded_proba, atol=1e-10)

    def test_save_before_train_raises(self, tmp_path: Path) -> None:
        ensemble = StackingEnsemble()
        with pytest.raises(RuntimeError, match="not fitted"):
            ensemble.save(tmp_path / "model")


class TestOOFGeneration:
    def test_oof_features_shape(self) -> None:
        """OOF stacking produces correct meta-feature shape."""
        rng = np.random.default_rng(42)
        n = 100
        probas = [rng.dirichlet([1, 1, 1], size=n) for _ in range(3)]
        ensemble = StackingEnsemble()
        meta = ensemble.generate_oof_predictions(probas)
        assert meta.shape == (n, 9)  # 3 models * 3 classes
