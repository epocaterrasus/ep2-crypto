"""CatBoost ternary direction classifier.

Secondary model in the stacking ensemble, providing diversity through
ordered boosting which inherently prevents target leakage.

Uses the same ternary target and interface as LGBMDirectionModel.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import numpy as np
from catboost import CatBoostClassifier, Pool

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class CatBoostConfig:
    """CatBoost hyperparameters for ternary classification."""

    depth: int = 5
    learning_rate: float = 0.05
    iterations: int = 500
    l2_leaf_reg: float = 3.0
    subsample: float = 0.8
    early_stopping_rounds: int = 50
    random_seed: int = 42


class CatBoostDirectionModel:
    """CatBoost ternary direction classifier.

    Same interface as LGBMDirectionModel for drop-in replacement
    in the stacking ensemble. Uses ordered boosting for diversity.

    Labels: -1 (DOWN), 0 (FLAT), +1 (UP) mapped to classes 0, 1, 2.
    """

    LABEL_TO_CLASS: ClassVar[dict[int, int]] = {-1: 0, 0: 1, 1: 2}
    CLASS_TO_LABEL: ClassVar[dict[int, int]] = {0: -1, 1: 0, 2: 1}

    def __init__(self, config: CatBoostConfig | None = None) -> None:
        self._config = config or CatBoostConfig()
        self._model: CatBoostClassifier | None = None
        self._feature_names: list[str] | None = None
        self._feature_importance: dict[str, float] | None = None
        self._best_iteration: int | None = None

    @property
    def is_fitted(self) -> bool:
        return self._model is not None and self._model.is_fitted()

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
        encoded = np.empty(len(labels), dtype=np.int32)
        for i, label in enumerate(labels):
            encoded[i] = self.LABEL_TO_CLASS[int(label)]
        return encoded

    def _decode_labels(self, classes: NDArray[np.int32]) -> NDArray[np.int8]:
        decoded = np.empty(len(classes), dtype=np.int8)
        for i, cls in enumerate(classes):
            decoded[i] = self.CLASS_TO_LABEL[int(cls)]
        return decoded

    def _build_pool_with_all_classes(
        self,
        x: NDArray[np.float64],
        y_encoded: NDArray[np.int32],
        feature_names: list[str] | None,
        class_weights: dict[int, float] | None,
    ) -> Pool:
        """Build a CatBoost Pool that always contains all 3 classes (0, 1, 2).

        When a class is absent from the training window CatBoost will only
        output probabilities for the classes it saw, causing predict_proba to
        return (n, 2) instead of (n, 3) and breaking the stacking ensemble.

        Fix: append a single zero-weight dummy row for each missing class so
        that CatBoost registers all three classes during fit.
        """
        present = set(int(v) for v in y_encoded)
        missing = {0, 1, 2} - present

        x_aug = x
        y_aug = y_encoded
        weights: NDArray[np.float64] | None = None

        if missing:
            # Build per-row weights: 1.0 for real rows, 0.0 for dummy rows
            if class_weights:
                real_weights = np.array(
                    [class_weights.get(int(v), 1.0) for v in y_encoded],
                    dtype=np.float64,
                )
            else:
                real_weights = np.ones(len(y_encoded), dtype=np.float64)

            dummy_x = np.zeros((len(missing), x.shape[1]), dtype=np.float64)
            dummy_y = np.array(sorted(missing), dtype=np.int32)
            dummy_w = np.zeros(len(missing), dtype=np.float64)

            x_aug = np.concatenate([x, dummy_x], axis=0)
            y_aug = np.concatenate([y_encoded, dummy_y], axis=0)
            weights = np.concatenate([real_weights, dummy_w], axis=0)
        elif class_weights:
            weights = np.array(
                [class_weights.get(int(v), 1.0) for v in y_encoded],
                dtype=np.float64,
            )

        return Pool(
            x_aug,
            y_aug,
            weight=weights,
            feature_names=feature_names,
        )

    def train(
        self,
        x_train: NDArray[np.float64],
        y_train: NDArray[np.int8],
        x_val: NDArray[np.float64] | None = None,
        y_val: NDArray[np.int8] | None = None,
        feature_names: list[str] | None = None,
        class_weights: dict[int, float] | None = None,
    ) -> dict[str, float]:
        """Train the CatBoost classifier.

        Args:
            x_train: Training features.
            y_train: Training labels (-1, 0, +1).
            x_val: Validation features.
            y_val: Validation labels.
            feature_names: Feature names.
            class_weights: Weights per class label (-1, 0, +1).

        Returns:
            Dict with training metrics.
        """
        y_encoded = self._encode_labels(y_train)
        self._feature_names = feature_names

        cfg = self._config

        # Convert class weights from label space (-1,0,1) to class index space (0,1,2)
        cb_class_weights: dict[int, float] | None = None
        if class_weights:
            cb_class_weights = {self.LABEL_TO_CLASS[k]: v for k, v in class_weights.items()}

        model = CatBoostClassifier(
            depth=cfg.depth,
            learning_rate=cfg.learning_rate,
            iterations=cfg.iterations,
            l2_leaf_reg=cfg.l2_leaf_reg,
            random_seed=cfg.random_seed,
            loss_function="MultiClass",
            classes_count=3,  # Always declare 3 classes so predict_proba → (n,3)
            verbose=0,
            boosting_type="Ordered",
            bootstrap_type="Bayesian",
            bagging_temperature=1.0,
        )

        # Build Pool with zero-weight dummy rows for missing classes so CatBoost
        # always sees all 3 classes, keeping predict_proba output shape (n,3).
        train_pool = self._build_pool_with_all_classes(
            x_train, y_encoded, feature_names, cb_class_weights
        )

        eval_pool = None
        if x_val is not None and y_val is not None:
            y_val_encoded = self._encode_labels(y_val)
            eval_pool = Pool(
                x_val,
                y_val_encoded,
                feature_names=feature_names,
            )

        model.fit(
            train_pool,
            eval_set=eval_pool,
            early_stopping_rounds=cfg.early_stopping_rounds if eval_pool else None,
        )

        self._model = model
        best_iter = model.best_iteration_
        self._best_iteration = best_iter if best_iter is not None else cfg.iterations
        self._update_feature_importance()

        # Metrics computed on original (non-augmented) data
        train_pred = model.predict(x_train).astype(np.int32).ravel()
        train_acc = float(np.mean(train_pred == y_encoded))

        metrics: dict[str, float] = {
            "train_accuracy": train_acc,
            "best_iteration": float(self._best_iteration),
        }

        if x_val is not None and y_val is not None:
            val_pred = model.predict(x_val).astype(np.int32).ravel()
            val_acc = float(np.mean(val_pred == self._encode_labels(y_val)))
            metrics["val_accuracy"] = val_acc

        return metrics

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.int8]:
        """Predict direction labels."""
        if self._model is None:
            msg = "Model not fitted. Call train() first."
            raise RuntimeError(msg)
        classes = self._model.predict(x).astype(np.int32).ravel()
        return self._decode_labels(classes)

    def predict_proba(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict class probabilities [DOWN, FLAT, UP].

        Always returns (n_samples, 3) regardless of how many classes the model
        saw during training.  Missing class columns are padded with zero and the
        result is renormalized so rows still sum to 1.
        """
        if self._model is None:
            msg = "Model not fitted. Call train() first."
            raise RuntimeError(msg)
        raw: NDArray[np.float64] = self._model.predict_proba(x).astype(np.float64)
        n = len(x)
        if raw.shape[1] == 3:
            return raw

        # Pad missing class columns with 0 and renormalize
        full = np.zeros((n, 3), dtype=np.float64)
        model_classes = self._model.classes_
        for i, cls in enumerate(model_classes):
            col = int(cls)
            if 0 <= col < 3:
                full[:, col] = raw[:, i]

        row_sums = full.sum(axis=1, keepdims=True)
        row_sums = np.maximum(row_sums, 1e-10)
        result: NDArray[np.float64] = full / row_sums
        return result

    def _update_feature_importance(self) -> None:
        if self._model is None:
            return
        importances = self._model.get_feature_importance()
        names = self._feature_names or [f"f{i}" for i in range(len(importances))]
        total = float(np.sum(importances))
        if total > 0:
            self._feature_importance = {
                name: float(imp / total) for name, imp in zip(names, importances, strict=True)
            }
        else:
            self._feature_importance = {name: 0.0 for name in names}

    def save(self, path: Path | str) -> None:
        """Save model to disk."""
        if self._model is None:
            msg = "Model not fitted. Cannot save."
            raise RuntimeError(msg)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        self._model.save_model(str(path.with_suffix(".cbm")))

        meta = {
            "feature_names": self._feature_names,
            "feature_importance": self._feature_importance,
            "best_iteration": self._best_iteration,
        }
        meta_path = path.with_suffix(".meta.json")
        meta_path.write_text(json.dumps(meta, indent=2))

    def load(self, path: Path | str) -> None:
        """Load model from disk."""
        path = Path(path)
        model = CatBoostClassifier()
        model.load_model(str(path.with_suffix(".cbm")))
        self._model = model

        meta_path = path.with_suffix(".meta.json")
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            self._feature_names = meta.get("feature_names")
            self._feature_importance = meta.get("feature_importance")
            self._best_iteration = meta.get("best_iteration")

    @property
    def raw_model(self) -> CatBoostClassifier | None:
        return self._model
