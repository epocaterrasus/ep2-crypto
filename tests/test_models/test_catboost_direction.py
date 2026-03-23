"""Tests for CatBoost ternary direction classifier."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from ep2_crypto.models.catboost_direction import CatBoostConfig, CatBoostDirectionModel
from ep2_crypto.models.labeling import Direction, compute_class_weights

if TYPE_CHECKING:
    from pathlib import Path


def _make_synthetic_data(
    n_samples: int = 500,
    n_features: int = 20,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, size=(n_samples, n_features))
    score = x[:, 0] * 0.5 + x[:, 1] * 0.3 + x[:, 2] * 0.2
    labels = np.zeros(n_samples, dtype=np.int8)
    labels[score > 0.5] = Direction.UP
    labels[score < -0.5] = Direction.DOWN
    feature_names = [f"feature_{i}" for i in range(n_features)]
    return x, labels, feature_names


@pytest.fixture
def synthetic_data() -> tuple[np.ndarray, np.ndarray, list[str]]:
    return _make_synthetic_data()


@pytest.fixture
def train_val_split(
    synthetic_data: tuple,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    x, y, names = synthetic_data
    split = int(len(x) * 0.8)
    return x[:split], y[:split], x[split:], y[split:], names


class TestCatBoostTrainPredict:
    def test_train_returns_metrics(self, train_val_split: tuple) -> None:
        x_train, y_train, x_val, y_val, names = train_val_split
        model = CatBoostDirectionModel()
        metrics = model.train(
            x_train,
            y_train,
            x_val,
            y_val,
            feature_names=names,
        )
        assert "train_accuracy" in metrics
        assert "val_accuracy" in metrics
        assert metrics["train_accuracy"] > 0.4

    def test_predict_shape(self, train_val_split: tuple) -> None:
        x_train, y_train, x_val, _, names = train_val_split
        model = CatBoostDirectionModel()
        model.train(x_train, y_train, feature_names=names)
        preds = model.predict(x_val)
        assert preds.shape == (len(x_val),)
        assert set(np.unique(preds)).issubset({-1, 0, 1})

    def test_predict_proba_shape(self, train_val_split: tuple) -> None:
        x_train, y_train, x_val, _, names = train_val_split
        model = CatBoostDirectionModel()
        model.train(x_train, y_train, feature_names=names)
        proba = model.predict_proba(x_val)
        assert proba.shape == (len(x_val), 3)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_predict_before_train_raises(self) -> None:
        model = CatBoostDirectionModel()
        x = np.random.default_rng(42).normal(size=(10, 5))
        with pytest.raises(RuntimeError, match="not fitted"):
            model.predict(x)

    def test_is_fitted_property(self, train_val_split: tuple) -> None:
        x_train, y_train, _, _, names = train_val_split
        model = CatBoostDirectionModel()
        assert not model.is_fitted
        model.train(x_train, y_train, feature_names=names)
        assert model.is_fitted


class TestCatBoostEarlyStopping:
    def test_early_stopping(self, train_val_split: tuple) -> None:
        x_train, y_train, x_val, y_val, names = train_val_split
        config = CatBoostConfig(iterations=1000, early_stopping_rounds=10)
        model = CatBoostDirectionModel(config)
        metrics = model.train(
            x_train,
            y_train,
            x_val,
            y_val,
            feature_names=names,
        )
        assert metrics["best_iteration"] < 1000


class TestCatBoostFeatureImportance:
    def test_importance_tracked(self, train_val_split: tuple) -> None:
        x_train, y_train, _, _, names = train_val_split
        model = CatBoostDirectionModel()
        model.train(x_train, y_train, feature_names=names)
        importance = model.feature_importance
        assert importance is not None
        assert len(importance) == len(names)


class TestCatBoostSaveLoad:
    def test_roundtrip(
        self,
        train_val_split: tuple,
        tmp_path: Path,
    ) -> None:
        x_train, y_train, x_val, _, names = train_val_split
        model = CatBoostDirectionModel()
        model.train(x_train, y_train, feature_names=names)

        original_preds = model.predict(x_val)
        original_proba = model.predict_proba(x_val)

        save_path = tmp_path / "catboost_model"
        model.save(save_path)
        assert (tmp_path / "catboost_model.cbm").exists()
        assert (tmp_path / "catboost_model.meta.json").exists()

        loaded = CatBoostDirectionModel()
        loaded.load(save_path)
        loaded_preds = loaded.predict(x_val)
        loaded_proba = loaded.predict_proba(x_val)

        np.testing.assert_array_equal(original_preds, loaded_preds)
        np.testing.assert_allclose(original_proba, loaded_proba, atol=1e-10)

    def test_save_before_train_raises(self, tmp_path: Path) -> None:
        model = CatBoostDirectionModel()
        with pytest.raises(RuntimeError, match="not fitted"):
            model.save(tmp_path / "model")


class TestCatBoostClassWeights:
    def test_class_weights(self, train_val_split: tuple) -> None:
        x_train, y_train, x_val, y_val, names = train_val_split
        weights = compute_class_weights(y_train)
        model = CatBoostDirectionModel()
        metrics = model.train(
            x_train,
            y_train,
            x_val,
            y_val,
            feature_names=names,
            class_weights=weights,
        )
        assert metrics["train_accuracy"] > 0.3


class TestCatBoostLabelEncoding:
    def test_encode_decode_roundtrip(self) -> None:
        model = CatBoostDirectionModel()
        labels = np.array([-1, 0, 1, -1, 1, 0], dtype=np.int8)
        encoded = model._encode_labels(labels)
        decoded = model._decode_labels(encoded)
        np.testing.assert_array_equal(labels, decoded)
