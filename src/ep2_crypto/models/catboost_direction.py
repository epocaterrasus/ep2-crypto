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

        # Convert class weights from label space to CatBoost class index space
        cb_class_weights = None
        if class_weights:
            cb_class_weights = {self.LABEL_TO_CLASS[k]: v for k, v in class_weights.items()}

        model = CatBoostClassifier(
            depth=cfg.depth,
            learning_rate=cfg.learning_rate,
            iterations=cfg.iterations,
            l2_leaf_reg=cfg.l2_leaf_reg,
            random_seed=cfg.random_seed,
            loss_function="MultiClass",
            verbose=0,
            class_weights=cb_class_weights,
            boosting_type="Ordered",
            bootstrap_type="Bayesian",
            bagging_temperature=1.0,
        )

        train_pool = Pool(
            x_train,
            y_encoded,
            feature_names=feature_names,
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

        # Metrics
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
        """Predict class probabilities [DOWN, FLAT, UP] — always returns (n, 3).

        CatBoost only outputs columns for classes seen during training. When the
        training window has no flat-labeled bars (common since flat is rare), it
        returns (n, 2). We expand to (n, 3) by inserting zeros for missing classes.
        """
        if self._model is None:
            msg = "Model not fitted. Call train() first."
            raise RuntimeError(msg)
        raw: NDArray[np.float64] = self._model.predict_proba(x).astype(np.float64)
        trained_classes = list(self._model.classes_)
        if len(trained_classes) == 3:
            return raw
        # Fewer than 3 classes seen — pad output to full 3-column shape
        full = np.zeros((raw.shape[0], 3), dtype=np.float64)
        for col_idx, class_id in enumerate(trained_classes):
            full[:, int(class_id)] = raw[:, col_idx]
        return full

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
