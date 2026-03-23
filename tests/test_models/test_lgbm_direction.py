"""Tests for LightGBM ternary direction classifier."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from ep2_crypto.models.labeling import Direction, compute_class_weights
from ep2_crypto.models.lgbm_direction import LGBMConfig, LGBMDirectionModel

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_synthetic_data(
    n_samples: int = 500,
    n_features: int = 20,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Generate synthetic data with signal for ternary classification."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, size=(n_samples, n_features))

    # Create ternary target with some signal in first 3 features
    score = x[:, 0] * 0.5 + x[:, 1] * 0.3 + x[:, 2] * 0.2
    labels = np.zeros(n_samples, dtype=np.int8)
    labels[score > 0.5] = Direction.UP
    labels[score < -0.5] = Direction.DOWN
    # Rest stays FLAT

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


# ---------------------------------------------------------------------------
# Basic training and prediction
# ---------------------------------------------------------------------------


class TestLGBMTrainPredict:
    def test_train_returns_metrics(self, train_val_split: tuple) -> None:
        """Training returns accuracy metrics."""
        x_train, y_train, x_val, y_val, names = train_val_split
        model = LGBMDirectionModel()
        metrics = model.train(
            x_train,
            y_train,
            x_val,
            y_val,
            feature_names=names,
        )
        assert "train_accuracy" in metrics
        assert "val_accuracy" in metrics
        assert "best_iteration" in metrics
        assert metrics["train_accuracy"] > 0.4  # Better than random (0.33)

    def test_predict_shape(self, train_val_split: tuple) -> None:
        """Predict returns correct shape with valid labels."""
        x_train, y_train, x_val, _y_val, names = train_val_split
        model = LGBMDirectionModel()
        model.train(x_train, y_train, feature_names=names)
        preds = model.predict(x_val)
        assert preds.shape == (len(x_val),)
        assert set(np.unique(preds)).issubset({-1, 0, 1})

    def test_predict_proba_shape(self, train_val_split: tuple) -> None:
        """Predict proba returns (n_samples, 3) summing to 1."""
        x_train, y_train, x_val, _y_val, names = train_val_split
        model = LGBMDirectionModel()
        model.train(x_train, y_train, feature_names=names)
        proba = model.predict_proba(x_val)
        assert proba.shape == (len(x_val), 3)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_predict_before_train_raises(self) -> None:
        """Predicting before training raises RuntimeError."""
        model = LGBMDirectionModel()
        x = np.random.default_rng(42).normal(size=(10, 5))
        with pytest.raises(RuntimeError, match="not fitted"):
            model.predict(x)

    def test_predict_proba_before_train_raises(self) -> None:
        model = LGBMDirectionModel()
        x = np.random.default_rng(42).normal(size=(10, 5))
        with pytest.raises(RuntimeError, match="not fitted"):
            model.predict_proba(x)

    def test_is_fitted_property(self, train_val_split: tuple) -> None:
        x_train, y_train, _, _, names = train_val_split
        model = LGBMDirectionModel()
        assert not model.is_fitted
        model.train(x_train, y_train, feature_names=names)
        assert model.is_fitted


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------


class TestEarlyStopping:
    def test_early_stopping_with_validation(self, train_val_split: tuple) -> None:
        """Early stopping should stop before n_estimators if val plateaus."""
        x_train, y_train, x_val, y_val, names = train_val_split
        config = LGBMConfig(
            n_estimators=1000,
            early_stopping_rounds=10,
        )
        model = LGBMDirectionModel(config)
        metrics = model.train(
            x_train,
            y_train,
            x_val,
            y_val,
            feature_names=names,
        )
        # Should stop well before 1000
        assert metrics["best_iteration"] < 1000

    def test_training_without_validation(self, synthetic_data: tuple) -> None:
        """Training without validation data should complete without error."""
        x, y, names = synthetic_data
        model = LGBMDirectionModel()
        metrics = model.train(x, y, feature_names=names)
        assert "train_accuracy" in metrics
        assert "val_accuracy" not in metrics


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------


class TestFeatureImportance:
    def test_gain_importance(self, train_val_split: tuple) -> None:
        """Feature importance is tracked after training."""
        x_train, y_train, _, _, names = train_val_split
        model = LGBMDirectionModel()
        model.train(x_train, y_train, feature_names=names)
        importance = model.feature_importance
        assert importance is not None
        assert len(importance) == len(names)
        # Importances should be normalized (sum ~1)
        assert abs(sum(importance.values()) - 1.0) < 0.01

    def test_signal_features_important(self, train_val_split: tuple) -> None:
        """Features with signal should rank higher in importance."""
        x_train, y_train, _, _, names = train_val_split
        model = LGBMDirectionModel()
        model.train(x_train, y_train, feature_names=names)
        importance = model.feature_importance
        assert importance is not None

        # First 3 features have signal — their combined importance should be > noise
        signal_imp = sum(importance.get(f"feature_{i}", 0) for i in range(3))
        noise_imp = sum(importance.get(f"feature_{i}", 0) for i in range(17, 20))
        assert signal_imp > noise_imp

    def test_shap_importance(self, train_val_split: tuple) -> None:
        """SHAP importance computes without error."""
        x_train, y_train, x_val, _, names = train_val_split
        model = LGBMDirectionModel()
        model.train(x_train, y_train, feature_names=names)
        shap_imp = model.get_shap_importance(x_val[:50])
        assert len(shap_imp) == len(names)
        assert all(v >= 0 for v in shap_imp.values())


# ---------------------------------------------------------------------------
# Warm start
# ---------------------------------------------------------------------------


class TestWarmStart:
    def test_warm_start_improves(self) -> None:
        """Warm-start from previous model should not crash and can train."""
        x, y, names = _make_synthetic_data(n_samples=400, seed=42)
        split1 = 200
        split2 = 300

        # Train first fold
        model1 = LGBMDirectionModel(LGBMConfig(n_estimators=50))
        model1.train(x[:split1], y[:split1], feature_names=names)

        # Train second fold with warm-start
        model2 = LGBMDirectionModel(LGBMConfig(n_estimators=50))
        metrics = model2.train(
            x[split1:split2],
            y[split1:split2],
            feature_names=names,
            init_model=model1.raw_model,
        )
        assert metrics["train_accuracy"] > 0.3


# ---------------------------------------------------------------------------
# Save / Load round trip
# ---------------------------------------------------------------------------


class TestSaveLoad:
    def test_save_load_roundtrip(
        self,
        train_val_split: tuple,
        tmp_path: Path,
    ) -> None:
        """Saved and loaded model produces identical predictions."""
        x_train, y_train, x_val, _, names = train_val_split
        model = LGBMDirectionModel()
        model.train(x_train, y_train, feature_names=names)

        original_preds = model.predict(x_val)
        original_proba = model.predict_proba(x_val)

        save_path = tmp_path / "lgbm_model"
        model.save(save_path)

        # Verify files exist
        assert (tmp_path / "lgbm_model.txt").exists()
        assert (tmp_path / "lgbm_model.meta.json").exists()

        # Load and compare
        loaded = LGBMDirectionModel()
        loaded.load(save_path)
        loaded_preds = loaded.predict(x_val)
        loaded_proba = loaded.predict_proba(x_val)

        np.testing.assert_array_equal(original_preds, loaded_preds)
        np.testing.assert_allclose(original_proba, loaded_proba, atol=1e-10)

    def test_save_before_train_raises(self, tmp_path: Path) -> None:
        model = LGBMDirectionModel()
        with pytest.raises(RuntimeError, match="not fitted"):
            model.save(tmp_path / "model")

    def test_metadata_preserved(
        self,
        train_val_split: tuple,
        tmp_path: Path,
    ) -> None:
        """Feature names and importance survive save/load."""
        x_train, y_train, _, _, names = train_val_split
        model = LGBMDirectionModel()
        model.train(x_train, y_train, feature_names=names)

        save_path = tmp_path / "lgbm_model"
        model.save(save_path)

        loaded = LGBMDirectionModel()
        loaded.load(save_path)
        assert loaded.feature_names == names
        assert loaded.feature_importance is not None


# ---------------------------------------------------------------------------
# Class weights
# ---------------------------------------------------------------------------


class TestClassWeights:
    def test_class_weights_accepted(self, train_val_split: tuple) -> None:
        """Model trains successfully with class weights."""
        x_train, y_train, x_val, y_val, names = train_val_split
        weights = compute_class_weights(y_train)
        config = LGBMConfig(class_weight=weights)
        model = LGBMDirectionModel(config)
        metrics = model.train(
            x_train,
            y_train,
            x_val,
            y_val,
            feature_names=names,
        )
        assert metrics["train_accuracy"] > 0.3


# ---------------------------------------------------------------------------
# Label encoding
# ---------------------------------------------------------------------------


class TestLabelEncoding:
    def test_encode_decode_roundtrip(self) -> None:
        """Encode then decode returns original labels."""
        model = LGBMDirectionModel()
        labels = np.array([-1, 0, 1, -1, 1, 0], dtype=np.int8)
        encoded = model._encode_labels(labels)
        decoded = model._decode_labels(encoded)
        np.testing.assert_array_equal(labels, decoded)

    def test_encoded_values(self) -> None:
        """Encoded values are 0, 1, 2."""
        model = LGBMDirectionModel()
        labels = np.array([-1, 0, 1], dtype=np.int8)
        encoded = model._encode_labels(labels)
        np.testing.assert_array_equal(encoded, [0, 1, 2])
