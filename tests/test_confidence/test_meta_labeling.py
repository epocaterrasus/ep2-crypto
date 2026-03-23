"""Tests for the meta-labeling model.

Covers:
- Meta-feature creation (shape, concatenation correctness)
- Fit on synthetic data (metrics returned)
- predict_proba output shape and range
- Gate function (threshold filtering)
- Save/load round-trip (predictions match)
- Unfitted model error handling
- Edge cases: all-profitable and all-unprofitable labels
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from ep2_crypto.confidence.meta_labeling import MetaLabelConfig, MetaLabeler

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def rng() -> np.random.Generator:
    """Reproducible random generator."""
    return np.random.default_rng(42)


@pytest.fixture()
def n_samples() -> int:
    return 500


@pytest.fixture()
def n_features() -> int:
    return 10


@pytest.fixture()
def synthetic_data(
    rng: np.random.Generator,
    n_samples: int,
    n_features: int,
) -> dict[str, np.ndarray]:
    """Generate synthetic data for meta-labeling tests."""
    primary_predictions = rng.choice(np.array([-1, 0, 1], dtype=np.int8), size=n_samples)
    primary_probas = rng.dirichlet(alpha=[1, 1, 1], size=n_samples).astype(np.float64)
    features = rng.standard_normal((n_samples, n_features)).astype(np.float64)
    regime_labels = rng.choice(np.array([0, 1, 2], dtype=np.int8), size=n_samples)
    # Target: make profitable correlated with max proba being high
    max_proba = primary_probas.max(axis=1)
    is_profitable = (max_proba > 0.4).astype(np.int8)
    # Add noise
    flip_mask = rng.random(n_samples) < 0.2
    is_profitable[flip_mask] = 1 - is_profitable[flip_mask]

    return {
        "primary_predictions": primary_predictions,
        "primary_probas": primary_probas,
        "features": features,
        "regime_labels": regime_labels,
        "is_profitable": is_profitable.astype(np.int8),
    }


@pytest.fixture()
def fitted_labeler(
    synthetic_data: dict[str, np.ndarray],
) -> tuple[MetaLabeler, np.ndarray]:
    """Return a fitted MetaLabeler and its meta-features."""
    config = MetaLabelConfig(
        n_estimators=50,
        min_child_samples=10,
        num_leaves=8,
    )
    labeler = MetaLabeler(config=config)
    meta = labeler.create_meta_features(
        primary_predictions=synthetic_data["primary_predictions"],
        primary_probas=synthetic_data["primary_probas"],
        features=synthetic_data["features"],
        regime_labels=synthetic_data["regime_labels"],
    )
    labeler.fit(meta, synthetic_data["is_profitable"])
    return labeler, meta


# ---------------------------------------------------------------------------
# Meta-feature creation
# ---------------------------------------------------------------------------


class TestCreateMetaFeatures:
    """Tests for create_meta_features()."""

    def test_output_shape(
        self,
        synthetic_data: dict[str, np.ndarray],
        n_samples: int,
        n_features: int,
    ) -> None:
        labeler = MetaLabeler()
        meta = labeler.create_meta_features(
            primary_predictions=synthetic_data["primary_predictions"],
            primary_probas=synthetic_data["primary_probas"],
            features=synthetic_data["features"],
            regime_labels=synthetic_data["regime_labels"],
        )
        # 1 (pred) + 3 (probas) + n_features + 1 (regime)
        expected_cols = 1 + 3 + n_features + 1
        assert meta.shape == (n_samples, expected_cols)

    def test_concatenation_correctness(
        self,
        rng: np.random.Generator,
    ) -> None:
        """Verify columns are in the expected order."""
        n = 5
        preds = np.array([1, -1, 0, 1, -1], dtype=np.int8)
        probas = np.array(
            [
                [0.1, 0.2, 0.7],
                [0.8, 0.1, 0.1],
                [0.2, 0.6, 0.2],
                [0.1, 0.1, 0.8],
                [0.7, 0.2, 0.1],
            ],
            dtype=np.float64,
        )
        features = np.ones((n, 2), dtype=np.float64) * 99.0
        regimes = np.array([0, 1, 2, 0, 1], dtype=np.int8)

        labeler = MetaLabeler()
        meta = labeler.create_meta_features(preds, probas, features, regimes)

        # Column 0: primary predictions
        np.testing.assert_array_equal(meta[:, 0], preds.astype(np.float64))
        # Columns 1-3: probabilities
        np.testing.assert_array_almost_equal(meta[:, 1:4], probas)
        # Columns 4-5: original features (all 99)
        np.testing.assert_array_almost_equal(meta[:, 4:6], 99.0)
        # Column 6: regime
        np.testing.assert_array_equal(meta[:, 6], regimes.astype(np.float64))

    def test_dtype_is_float64(
        self,
        synthetic_data: dict[str, np.ndarray],
    ) -> None:
        labeler = MetaLabeler()
        meta = labeler.create_meta_features(
            primary_predictions=synthetic_data["primary_predictions"],
            primary_probas=synthetic_data["primary_probas"],
            features=synthetic_data["features"],
            regime_labels=synthetic_data["regime_labels"],
        )
        assert meta.dtype == np.float64

    def test_mismatched_samples_raises(self) -> None:
        labeler = MetaLabeler()
        with pytest.raises(ValueError, match="primary_probas has 3 samples"):
            labeler.create_meta_features(
                primary_predictions=np.array([1, -1], dtype=np.int8),
                primary_probas=np.ones((3, 3), dtype=np.float64),
                features=np.ones((2, 5), dtype=np.float64),
                regime_labels=np.array([0, 1], dtype=np.int8),
            )

    def test_wrong_proba_shape_raises(self) -> None:
        labeler = MetaLabeler()
        with pytest.raises(ValueError, match="must have shape"):
            labeler.create_meta_features(
                primary_predictions=np.array([1, -1], dtype=np.int8),
                primary_probas=np.ones((2, 2), dtype=np.float64),
                features=np.ones((2, 5), dtype=np.float64),
                regime_labels=np.array([0, 1], dtype=np.int8),
            )


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------


class TestFit:
    """Tests for fit()."""

    def test_returns_metrics(
        self,
        fitted_labeler: tuple[MetaLabeler, np.ndarray],
    ) -> None:
        labeler, _ = fitted_labeler
        # Re-fit to capture metrics (fitted_labeler already fitted)
        assert labeler.is_fitted

    def test_fit_metrics_keys(
        self,
        synthetic_data: dict[str, np.ndarray],
    ) -> None:
        config = MetaLabelConfig(n_estimators=20, min_child_samples=10)
        labeler = MetaLabeler(config=config)
        meta = labeler.create_meta_features(
            primary_predictions=synthetic_data["primary_predictions"],
            primary_probas=synthetic_data["primary_probas"],
            features=synthetic_data["features"],
            regime_labels=synthetic_data["regime_labels"],
        )
        metrics = labeler.fit(meta, synthetic_data["is_profitable"])

        assert "accuracy" in metrics
        assert "auc" in metrics
        assert "best_iteration" in metrics
        assert "n_samples" in metrics
        assert "positive_rate" in metrics

    def test_accuracy_reasonable(
        self,
        synthetic_data: dict[str, np.ndarray],
    ) -> None:
        """Accuracy should be above random (0.5) on correlated synthetic data."""
        config = MetaLabelConfig(n_estimators=50, min_child_samples=10)
        labeler = MetaLabeler(config=config)
        meta = labeler.create_meta_features(
            primary_predictions=synthetic_data["primary_predictions"],
            primary_probas=synthetic_data["primary_probas"],
            features=synthetic_data["features"],
            regime_labels=synthetic_data["regime_labels"],
        )
        metrics = labeler.fit(meta, synthetic_data["is_profitable"])
        # Train accuracy should be above chance
        assert metrics["accuracy"] > 0.5

    def test_auc_range(
        self,
        synthetic_data: dict[str, np.ndarray],
    ) -> None:
        config = MetaLabelConfig(n_estimators=30, min_child_samples=10)
        labeler = MetaLabeler(config=config)
        meta = labeler.create_meta_features(
            primary_predictions=synthetic_data["primary_predictions"],
            primary_probas=synthetic_data["primary_probas"],
            features=synthetic_data["features"],
            regime_labels=synthetic_data["regime_labels"],
        )
        metrics = labeler.fit(meta, synthetic_data["is_profitable"])
        assert 0.0 <= metrics["auc"] <= 1.0

    def test_fit_with_validation(
        self,
        synthetic_data: dict[str, np.ndarray],
    ) -> None:
        """Fit with validation set for early stopping."""
        config = MetaLabelConfig(n_estimators=100, min_child_samples=10)
        labeler = MetaLabeler(config=config)
        meta = labeler.create_meta_features(
            primary_predictions=synthetic_data["primary_predictions"],
            primary_probas=synthetic_data["primary_probas"],
            features=synthetic_data["features"],
            regime_labels=synthetic_data["regime_labels"],
        )
        y = synthetic_data["is_profitable"]

        # Split into train/val
        split = 400
        metrics = labeler.fit(
            meta[:split],
            y[:split],
            meta_features_val=meta[split:],
            is_profitable_val=y[split:],
        )
        assert labeler.is_fitted
        assert metrics["best_iteration"] > 0


# ---------------------------------------------------------------------------
# predict_proba
# ---------------------------------------------------------------------------


class TestPredictProba:
    """Tests for predict_proba()."""

    def test_output_shape(
        self,
        fitted_labeler: tuple[MetaLabeler, np.ndarray],
        n_samples: int,
    ) -> None:
        labeler, meta = fitted_labeler
        probas = labeler.predict_proba(meta)
        assert probas.shape == (n_samples,)

    def test_output_range(
        self,
        fitted_labeler: tuple[MetaLabeler, np.ndarray],
    ) -> None:
        labeler, meta = fitted_labeler
        probas = labeler.predict_proba(meta)
        assert np.all(probas >= 0.0)
        assert np.all(probas <= 1.0)

    def test_dtype_float64(
        self,
        fitted_labeler: tuple[MetaLabeler, np.ndarray],
    ) -> None:
        labeler, meta = fitted_labeler
        probas = labeler.predict_proba(meta)
        assert probas.dtype == np.float64


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------


class TestGate:
    """Tests for gate()."""

    def test_gate_returns_bool_array(
        self,
        fitted_labeler: tuple[MetaLabeler, np.ndarray],
        n_samples: int,
    ) -> None:
        labeler, meta = fitted_labeler
        mask = labeler.gate(meta, threshold=0.5)
        assert mask.shape == (n_samples,)
        assert mask.dtype == np.bool_

    def test_high_threshold_filters_more(
        self,
        fitted_labeler: tuple[MetaLabeler, np.ndarray],
    ) -> None:
        labeler, meta = fitted_labeler
        mask_low = labeler.gate(meta, threshold=0.3)
        mask_high = labeler.gate(meta, threshold=0.7)
        assert np.sum(mask_low) >= np.sum(mask_high)

    def test_zero_threshold_passes_all(
        self,
        fitted_labeler: tuple[MetaLabeler, np.ndarray],
        n_samples: int,
    ) -> None:
        labeler, meta = fitted_labeler
        mask = labeler.gate(meta, threshold=0.0)
        assert np.sum(mask) == n_samples

    def test_invalid_threshold_raises(
        self,
        fitted_labeler: tuple[MetaLabeler, np.ndarray],
    ) -> None:
        labeler, meta = fitted_labeler
        with pytest.raises(ValueError, match="threshold must be in"):
            labeler.gate(meta, threshold=1.5)
        with pytest.raises(ValueError, match="threshold must be in"):
            labeler.gate(meta, threshold=-0.1)


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------


class TestSaveLoad:
    """Tests for save() and load() round-trip."""

    def test_round_trip_predictions_match(
        self,
        fitted_labeler: tuple[MetaLabeler, np.ndarray],
        tmp_path: Path,
    ) -> None:
        labeler, meta = fitted_labeler
        probas_before = labeler.predict_proba(meta)

        save_path = tmp_path / "meta_model"
        labeler.save(save_path)

        loaded = MetaLabeler()
        loaded.load(save_path)

        probas_after = loaded.predict_proba(meta)
        np.testing.assert_array_almost_equal(probas_before, probas_after, decimal=6)

    def test_save_creates_files(
        self,
        fitted_labeler: tuple[MetaLabeler, np.ndarray],
        tmp_path: Path,
    ) -> None:
        labeler, _ = fitted_labeler
        save_path = tmp_path / "meta_model"
        labeler.save(save_path)

        assert (tmp_path / "meta_model.txt").exists()
        assert (tmp_path / "meta_model.meta.json").exists()

    def test_load_restores_metadata(
        self,
        fitted_labeler: tuple[MetaLabeler, np.ndarray],
        tmp_path: Path,
    ) -> None:
        labeler, _ = fitted_labeler
        save_path = tmp_path / "meta_model"
        labeler.save(save_path)

        loaded = MetaLabeler()
        loaded.load(save_path)

        assert loaded.is_fitted
        assert loaded._feature_names == labeler._feature_names
        assert loaded._best_iteration == labeler._best_iteration

    def test_load_nonexistent_raises(self, tmp_path: Path) -> None:
        labeler = MetaLabeler()
        with pytest.raises(FileNotFoundError):
            labeler.load(tmp_path / "nonexistent")


# ---------------------------------------------------------------------------
# Unfitted model errors
# ---------------------------------------------------------------------------


class TestUnfittedErrors:
    """Unfitted model should raise RuntimeError on predict/gate/save."""

    def test_predict_proba_raises(self) -> None:
        labeler = MetaLabeler()
        dummy = np.zeros((5, 10), dtype=np.float64)
        with pytest.raises(RuntimeError, match="not fitted"):
            labeler.predict_proba(dummy)

    def test_gate_raises(self) -> None:
        labeler = MetaLabeler()
        dummy = np.zeros((5, 10), dtype=np.float64)
        with pytest.raises(RuntimeError, match="not fitted"):
            labeler.gate(dummy)

    def test_save_raises(self, tmp_path: Path) -> None:
        labeler = MetaLabeler()
        with pytest.raises(RuntimeError, match="not fitted"):
            labeler.save(tmp_path / "model")

    def test_feature_importance_raises(self) -> None:
        labeler = MetaLabeler()
        with pytest.raises(RuntimeError, match="not fitted"):
            labeler.feature_importance()

    def test_is_fitted_false(self) -> None:
        labeler = MetaLabeler()
        assert labeler.is_fitted is False


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases: all-profitable, all-unprofitable."""

    def test_all_profitable(self, rng: np.random.Generator) -> None:
        """Model should handle all-positive labels without crashing."""
        n = 200
        config = MetaLabelConfig(n_estimators=20, min_child_samples=5)
        labeler = MetaLabeler(config=config)

        meta = labeler.create_meta_features(
            primary_predictions=rng.choice(np.array([-1, 0, 1], dtype=np.int8), size=n),
            primary_probas=rng.dirichlet([1, 1, 1], size=n).astype(np.float64),
            features=rng.standard_normal((n, 5)).astype(np.float64),
            regime_labels=np.zeros(n, dtype=np.int8),
        )

        y = np.ones(n, dtype=np.int8)
        metrics = labeler.fit(meta, y)

        assert metrics["positive_rate"] == 1.0
        assert metrics["auc"] == 0.5  # single class, AUC defaults to 0.5

        probas = labeler.predict_proba(meta)
        assert probas.shape == (n,)
        assert np.all(probas >= 0.0)
        assert np.all(probas <= 1.0)

    def test_all_unprofitable(self, rng: np.random.Generator) -> None:
        """Model should handle all-negative labels without crashing."""
        n = 200
        config = MetaLabelConfig(n_estimators=20, min_child_samples=5)
        labeler = MetaLabeler(config=config)

        meta = labeler.create_meta_features(
            primary_predictions=rng.choice(np.array([-1, 0, 1], dtype=np.int8), size=n),
            primary_probas=rng.dirichlet([1, 1, 1], size=n).astype(np.float64),
            features=rng.standard_normal((n, 5)).astype(np.float64),
            regime_labels=np.ones(n, dtype=np.int8),
        )

        y = np.zeros(n, dtype=np.int8)
        metrics = labeler.fit(meta, y)

        assert metrics["positive_rate"] == 0.0
        assert metrics["auc"] == 0.5

        probas = labeler.predict_proba(meta)
        assert probas.shape == (n,)


class TestFeatureImportance:
    """Tests for feature_importance()."""

    def test_returns_dict(
        self,
        fitted_labeler: tuple[MetaLabeler, np.ndarray],
    ) -> None:
        labeler, _ = fitted_labeler
        imp = labeler.feature_importance()
        assert isinstance(imp, dict)
        assert len(imp) > 0

    def test_values_sum_to_one(
        self,
        fitted_labeler: tuple[MetaLabeler, np.ndarray],
    ) -> None:
        labeler, _ = fitted_labeler
        imp = labeler.feature_importance()
        total = sum(imp.values())
        assert abs(total - 1.0) < 1e-6

    def test_includes_primary_features(
        self,
        fitted_labeler: tuple[MetaLabeler, np.ndarray],
    ) -> None:
        labeler, _ = fitted_labeler
        imp = labeler.feature_importance()
        assert "primary_pred" in imp
        assert "regime" in imp
