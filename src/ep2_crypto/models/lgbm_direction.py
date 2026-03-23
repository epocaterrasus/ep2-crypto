"""LightGBM ternary direction classifier.

Primary model in the stacking ensemble. Predicts ternary direction
(up/flat/down) on 5-minute BTC bars using gradient boosting with
leaf-wise growth.

Key features:
- Warm-start across walk-forward folds via init_model
- Early stopping on purged validation set
- SHAP feature importance tracking
- Save/load round-trip for model persistence
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import lightgbm as lgb
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class LGBMConfig:
    """LightGBM hyperparameters for ternary classification.

    Default values from CLAUDE.md / SPRINTS.md research recommendations.
    """

    num_leaves: int = 31
    max_depth: int = 5
    learning_rate: float = 0.05
    n_estimators: int = 500
    min_child_samples: int = 50
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0
    early_stopping_rounds: int = 50
    random_state: int = 42
    class_weight: dict[int, float] = field(default_factory=dict)


class LGBMDirectionModel:
    """LightGBM ternary direction classifier.

    Wraps LightGBM's multiclass classifier with:
    - Warm-start (init_model) across walk-forward folds
    - Early stopping on purged validation set
    - SHAP feature importance tracking
    - Serialization (save/load)

    Labels must be encoded as 0, 1, 2 (mapped from -1, 0, +1).
    """

    # Map from Direction(-1, 0, 1) to LightGBM class indices (0, 1, 2)
    LABEL_TO_CLASS: ClassVar[dict[int, int]] = {-1: 0, 0: 1, 1: 2}
    CLASS_TO_LABEL: ClassVar[dict[int, int]] = {0: -1, 1: 0, 2: 1}

    def __init__(self, config: LGBMConfig | None = None) -> None:
        self._config = config or LGBMConfig()
        self._model: lgb.LGBMClassifier | None = None
        self._feature_names: list[str] | None = None
        self._feature_importance: dict[str, float] | None = None
        self._best_iteration: int | None = None

    @property
    def is_fitted(self) -> bool:
        return self._model is not None and self._model.fitted_

    @property
    def feature_names(self) -> list[str] | None:
        return self._feature_names

    @property
    def feature_importance(self) -> dict[str, float] | None:
        return self._feature_importance

    @property
    def best_iteration(self) -> int | None:
        return self._best_iteration

    def _encode_labels(self, labels: NDArray[np.int8]) -> NDArray[np.int32]:
        """Convert Direction labels (-1, 0, +1) to class indices (0, 1, 2)."""
        encoded = np.empty(len(labels), dtype=np.int32)
        for i, label in enumerate(labels):
            encoded[i] = self.LABEL_TO_CLASS[int(label)]
        return encoded

    def _decode_labels(self, classes: NDArray[np.int32]) -> NDArray[np.int8]:
        """Convert class indices (0, 1, 2) back to Direction labels (-1, 0, +1)."""
        decoded = np.empty(len(classes), dtype=np.int8)
        for i, cls in enumerate(classes):
            decoded[i] = self.CLASS_TO_LABEL[int(cls)]
        return decoded

    def _build_sample_weights(
        self,
        labels: NDArray[np.int32],
    ) -> NDArray[np.float64] | None:
        """Build sample weights from class weight config."""
        if not self._config.class_weight:
            return None
        weights = np.ones(len(labels), dtype=np.float64)
        for i, label in enumerate(labels):
            direction = self.CLASS_TO_LABEL[int(label)]
            if direction in self._config.class_weight:
                weights[i] = self._config.class_weight[direction]
        return weights

    def _with_feature_names(self, x: NDArray[np.float64]) -> Any:
        """Wrap numpy array with feature names if available (avoids LightGBM warning)."""
        if self._feature_names is None:
            return x
        import pandas as pd

        return pd.DataFrame(x, columns=self._feature_names)

    def train(
        self,
        x_train: NDArray[np.float64],
        y_train: NDArray[np.int8],
        x_val: NDArray[np.float64] | None = None,
        y_val: NDArray[np.int8] | None = None,
        feature_names: list[str] | None = None,
        init_model: lgb.LGBMClassifier | None = None,
    ) -> dict[str, float]:
        """Train the LightGBM classifier.

        Args:
            x_train: Training features (n_samples, n_features).
            y_train: Training labels as Direction values (-1, 0, +1).
            x_val: Validation features (purged from training).
            y_val: Validation labels.
            feature_names: Feature names for importance tracking.
            init_model: Previous model for warm-start.

        Returns:
            Dict with training metrics (accuracy, best_iteration, etc.).
        """
        y_encoded = self._encode_labels(y_train)
        self._feature_names = feature_names

        # LightGBM's sklearn wrapper fits an internal LabelEncoder on y_train.
        # If a class is absent from the training window (flat labels are ~0.009%
        # of bars, so most 4032-bar windows have zero), the encoder only knows
        # the seen classes and crashes when the eval_set contains the missing
        # class. Fix: inject one zero-weight dummy row per missing class so the
        # encoder always sees all 3 classes without affecting training.
        missing = set([0, 1, 2]) - set(int(c) for c in np.unique(y_encoded))
        if missing:
            n_missing = len(missing)
            dummy_x = np.zeros((n_missing, x_train.shape[1]), dtype=np.float64)
            dummy_y = np.array(sorted(missing), dtype=np.int32)
            x_train = np.vstack([x_train, dummy_x])
            y_encoded = np.append(y_encoded, dummy_y)

        cfg = self._config
        model = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=3,
            num_leaves=cfg.num_leaves,
            max_depth=cfg.max_depth,
            learning_rate=cfg.learning_rate,
            n_estimators=cfg.n_estimators,
            min_child_samples=cfg.min_child_samples,
            subsample=cfg.subsample,
            colsample_bytree=cfg.colsample_bytree,
            reg_alpha=cfg.reg_alpha,
            reg_lambda=cfg.reg_lambda,
            random_state=cfg.random_state,
            verbose=-1,
            importance_type="gain",
        )

        sample_weight = self._build_sample_weights(y_encoded)
        # Zero-weight the dummy rows so they don't influence training
        if missing:
            n_missing = len(missing)
            if sample_weight is None:
                sample_weight = np.ones(len(y_encoded), dtype=np.float64)
            sample_weight[-n_missing:] = 0.0

        fit_kwargs: dict[str, Any] = {
            "X": x_train,
            "y": y_encoded,
            "sample_weight": sample_weight,
        }

        if init_model is not None:
            # Only warm-start if init_model was trained on the same set of classes.
            # When flat-labeled bars are absent from a fold window, LightGBM trains
            # with fewer classes, and its internal LabelEncoder conflicts with a
            # subsequent fold that has a different class count.
            current_classes = set(int(c) for c in np.unique(y_encoded))
            init_classes = set(int(c) for c in getattr(init_model, "classes_", []))
            if current_classes == init_classes:
                fit_kwargs["init_model"] = init_model.booster_

        callbacks: list[Any] = []
        eval_set = []
        eval_names = []

        if x_val is not None and y_val is not None:
            y_val_encoded = self._encode_labels(y_val)
            eval_set.append((x_val, y_val_encoded))
            eval_names.append("validation")
            callbacks.append(
                lgb.early_stopping(
                    stopping_rounds=cfg.early_stopping_rounds,
                    verbose=False,
                ),
            )
            callbacks.append(lgb.log_evaluation(period=0))

        if eval_set:
            fit_kwargs["eval_set"] = eval_set
            fit_kwargs["eval_names"] = eval_names

        if callbacks:
            fit_kwargs["callbacks"] = callbacks

        if feature_names is not None:
            fit_kwargs["feature_name"] = feature_names

        model.fit(**fit_kwargs)

        self._model = model
        best_iter = model.best_iteration_
        self._best_iteration = best_iter if best_iter > 0 else cfg.n_estimators

        # Track feature importance
        self._update_feature_importance()

        # Compute training metrics via predict_proba + argmax to avoid LabelEncoder
        # conflicts when some classes are absent from the training window.
        train_proba = model.predict_proba(self._with_feature_names(x_train))
        train_pred = np.argmax(train_proba, axis=1).astype(np.int32)
        # Remap internal indices back through the fitted LabelEncoder
        trained_classes = list(model.classes_)
        train_pred_mapped = np.array([trained_classes[i] for i in train_pred], dtype=np.int32)
        train_acc = float(np.mean(train_pred_mapped == y_encoded))

        metrics: dict[str, float] = {
            "train_accuracy": train_acc,
            "best_iteration": float(self._best_iteration),
        }

        if x_val is not None and y_val is not None:
            val_proba = model.predict_proba(self._with_feature_names(x_val))
            val_pred = np.argmax(val_proba, axis=1).astype(np.int32)
            val_pred_mapped = np.array([trained_classes[i] for i in val_pred], dtype=np.int32)
            val_acc = float(np.mean(val_pred_mapped == y_val_encoded))
            metrics["val_accuracy"] = val_acc

        return metrics

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.int8]:
        """Predict direction labels.

        Args:
            x: Features (n_samples, n_features).

        Returns:
            Direction labels (-1, 0, +1).
        """
        if self._model is None:
            msg = "Model not fitted. Call train() first."
            raise RuntimeError(msg)
        classes = self._model.predict(self._with_feature_names(x)).astype(np.int32)
        return self._decode_labels(classes)

    def predict_proba(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict class probabilities.

        Args:
            x: Features (n_samples, n_features).

        Returns:
            Probabilities (n_samples, 3) for [DOWN, FLAT, UP].
        """
        if self._model is None:
            msg = "Model not fitted. Call train() first."
            raise RuntimeError(msg)
        result: NDArray[np.float64] = self._model.predict_proba(self._with_feature_names(x)).astype(np.float64)
        return result

    def _update_feature_importance(self) -> None:
        """Update feature importance from the trained model."""
        if self._model is None:
            return

        importances = self._model.feature_importances_
        names = self._feature_names or [f"f{i}" for i in range(len(importances))]
        total = float(np.sum(importances))
        if total > 0:
            self._feature_importance = {
                name: float(imp / total) for name, imp in zip(names, importances, strict=True)
            }
        else:
            self._feature_importance = {name: 0.0 for name in names}

    def get_shap_importance(
        self,
        x: NDArray[np.float64],
    ) -> dict[str, float]:
        """Compute SHAP-based feature importance.

        Args:
            x: Sample of features for SHAP computation.

        Returns:
            Dict of feature name → mean absolute SHAP value.
        """
        if self._model is None:
            msg = "Model not fitted. Call train() first."
            raise RuntimeError(msg)

        import shap

        explainer = shap.TreeExplainer(self._model)
        shap_values = explainer.shap_values(x)

        # shap_values is a list of 3 arrays (one per class) or a 3D array
        if isinstance(shap_values, list):
            # Average absolute SHAP across all classes
            mean_abs = np.mean(
                [np.abs(sv) for sv in shap_values],
                axis=0,
            )
        else:
            mean_abs = np.mean(np.abs(shap_values), axis=2)

        # Average across samples
        mean_importance = np.mean(mean_abs, axis=0)

        names = self._feature_names or [f"f{i}" for i in range(len(mean_importance))]
        return dict(zip(names, mean_importance.tolist(), strict=True))

    def save(self, path: Path | str) -> None:
        """Save model to disk.

        Creates two files: {path}.txt (booster) and {path}.meta.json (metadata).
        """
        if self._model is None:
            msg = "Model not fitted. Cannot save."
            raise RuntimeError(msg)

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save booster
        self._model.booster_.save_model(str(path.with_suffix(".txt")))

        # Save metadata
        meta = {
            "feature_names": self._feature_names,
            "feature_importance": self._feature_importance,
            "best_iteration": self._best_iteration,
            "config": {
                "num_leaves": self._config.num_leaves,
                "max_depth": self._config.max_depth,
                "learning_rate": self._config.learning_rate,
                "n_estimators": self._config.n_estimators,
                "min_child_samples": self._config.min_child_samples,
            },
        }
        meta_path = path.with_suffix(".meta.json")
        meta_path.write_text(json.dumps(meta, indent=2))

    def load(self, path: Path | str) -> None:
        """Load model from disk.

        Args:
            path: Base path (will look for .txt and .meta.json).
        """
        path = Path(path)

        booster_path = path.with_suffix(".txt")
        booster = lgb.Booster(model_file=str(booster_path))

        # Load metadata first to know n_features
        meta_path = path.with_suffix(".meta.json")
        n_features = 1
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            self._feature_names = meta.get("feature_names")
            self._feature_importance = meta.get("feature_importance")
            self._best_iteration = meta.get("best_iteration")
            if self._feature_names:
                n_features = len(self._feature_names)

        # Reconstruct classifier wrapper using a minimal fit to set internal state,
        # then replace the booster with the loaded one.
        model = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=3,
            n_estimators=1,
            verbose=-1,
        )
        _dummy_x = np.zeros((3, n_features))
        _dummy_y = np.array([0, 1, 2])
        model.fit(_dummy_x, _dummy_y)
        model._Booster = booster
        model.n_features_in_ = n_features

        self._model = model

    @property
    def raw_model(self) -> lgb.LGBMClassifier | None:
        """Access the underlying LightGBM model (for warm-start)."""
        return self._model


class _LabelEncoder3:
    """Minimal label encoder for 3-class LightGBM reconstruction."""

    classes_ = np.array([0, 1, 2])

    def inverse_transform(self, y: NDArray[np.int64]) -> NDArray[np.int64]:
        return y
