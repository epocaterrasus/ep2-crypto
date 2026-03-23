"""Stacking ensemble meta-learner.

Combines predictions from LightGBM, CatBoost, and GRU into a final
prediction using a logistic regression meta-learner trained on
out-of-fold (OOF) predictions.

Key design:
- OOF predictions generated via inner walk-forward (never in-sample)
- Meta-learner: logistic regression on class probabilities
- Ensemble weights optimizable for Sharpe via scipy.optimize
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import numpy as np
from sklearn.linear_model import LogisticRegression

if TYPE_CHECKING:
    from numpy.typing import NDArray


class BaseModel(Protocol):
    """Protocol for base models in the stacking ensemble."""

    def predict_proba(self, x: NDArray[np.float64]) -> NDArray[np.float64]: ...


@dataclass
class StackingConfig:
    """Configuration for the stacking ensemble."""

    meta_c: float = 1.0  # Logistic regression regularization
    meta_max_iter: int = 1000
    n_folds: int = 3  # Inner walk-forward folds for OOF
    random_state: int = 42


class StackingEnsemble:
    """Stacking ensemble with logistic regression meta-learner.

    Combines class probabilities from multiple base models into
    a final ternary prediction. The meta-learner is trained on
    out-of-fold predictions to prevent overfitting.
    """

    def __init__(self, config: StackingConfig | None = None) -> None:
        self._config = config or StackingConfig()
        self._meta_model: LogisticRegression | None = None
        self._n_base_models: int = 0
        self._base_model_names: list[str] | None = None

    @property
    def is_fitted(self) -> bool:
        return self._meta_model is not None

    def generate_oof_predictions(
        self,
        base_probas: list[NDArray[np.float64]],
    ) -> NDArray[np.float64]:
        """Stack base model OOF probabilities into meta-features.

        Each base model produces (n_samples, 3) probabilities.
        We concatenate them into (n_samples, n_models * 3).

        Args:
            base_probas: List of probability arrays from base models.
                Each is (n_samples, 3) for [DOWN, FLAT, UP].

        Returns:
            Stacked meta-features (n_samples, n_models * 3).
        """
        self._n_base_models = len(base_probas)
        return np.hstack(base_probas)

    def train(
        self,
        base_probas: list[NDArray[np.float64]],
        y_true: NDArray[np.int8],
        base_model_names: list[str] | None = None,
    ) -> dict[str, float]:
        """Train the meta-learner on OOF predictions.

        Args:
            base_probas: List of OOF probability arrays from base models.
            y_true: True labels (-1, 0, +1).
            base_model_names: Names for each base model.

        Returns:
            Dict with training metrics.
        """
        self._base_model_names = base_model_names
        self._n_base_models = len(base_probas)

        # Encode labels: -1→0, 0→1, 1→2
        y_encoded = y_true.astype(np.int32) + 1

        meta_features = self.generate_oof_predictions(base_probas)

        cfg = self._config
        self._meta_model = LogisticRegression(
            C=cfg.meta_c,
            max_iter=cfg.meta_max_iter,
            random_state=cfg.random_state,
            solver="lbfgs",
        )
        self._meta_model.fit(meta_features, y_encoded)

        # Metrics
        train_pred = self._meta_model.predict(meta_features)
        train_acc = float(np.mean(train_pred == y_encoded))

        metrics: dict[str, float] = {
            "meta_train_accuracy": train_acc,
            "n_base_models": float(self._n_base_models),
            "n_meta_features": float(meta_features.shape[1]),
        }

        return metrics

    def predict(
        self,
        base_probas: list[NDArray[np.float64]],
    ) -> NDArray[np.int8]:
        """Predict using the stacking ensemble.

        Args:
            base_probas: List of probability arrays from base models.

        Returns:
            Direction labels (-1, 0, +1).
        """
        proba = self.predict_proba(base_probas)
        classes = np.argmax(proba, axis=1)
        # Decode: 0→-1, 1→0, 2→1
        result: NDArray[np.int8] = (classes - 1).astype(np.int8)
        return result

    def predict_proba(
        self,
        base_probas: list[NDArray[np.float64]],
    ) -> NDArray[np.float64]:
        """Predict class probabilities.

        Args:
            base_probas: List of probability arrays from base models.

        Returns:
            Probabilities (n_samples, 3) for [DOWN, FLAT, UP].
        """
        if self._meta_model is None:
            msg = "Meta-model not fitted. Call train() first."
            raise RuntimeError(msg)

        meta_features = np.hstack(base_probas)
        raw_proba = self._meta_model.predict_proba(meta_features).astype(np.float64)

        # Ensure output always has 3 columns even if some classes were absent in training
        n = len(meta_features)
        full_proba = np.zeros((n, 3), dtype=np.float64)
        classes = self._meta_model.classes_
        for i, cls in enumerate(classes):
            full_proba[:, cls] = raw_proba[:, i]

        # Renormalize
        row_sums = full_proba.sum(axis=1, keepdims=True)
        row_sums = np.maximum(row_sums, 1e-10)
        full_proba = full_proba / row_sums

        result: NDArray[np.float64] = full_proba
        return result

    def simple_average_predict(
        self,
        base_probas: list[NDArray[np.float64]],
    ) -> NDArray[np.int8]:
        """Simple average ensemble (baseline for comparison).

        Args:
            base_probas: List of probability arrays.

        Returns:
            Direction labels from averaged probabilities.
        """
        avg = np.mean(base_probas, axis=0)
        classes = np.argmax(avg, axis=1)
        result: NDArray[np.int8] = (classes - 1).astype(np.int8)
        return result

    def weighted_average_predict(
        self,
        base_probas: list[NDArray[np.float64]],
        weights: NDArray[np.float64],
    ) -> NDArray[np.int8]:
        """Weighted average ensemble.

        Args:
            base_probas: List of probability arrays.
            weights: Weight per base model (should sum to 1).

        Returns:
            Direction labels from weighted averaged probabilities.
        """
        weighted = np.zeros_like(base_probas[0])
        for proba, w in zip(base_probas, weights, strict=True):
            weighted += proba * w
        classes = np.argmax(weighted, axis=1)
        result: NDArray[np.int8] = (classes - 1).astype(np.int8)
        return result

    def optimize_weights(
        self,
        base_probas: list[NDArray[np.float64]],
        y_true: NDArray[np.int8],
    ) -> NDArray[np.float64]:
        """Optimize ensemble weights to maximize accuracy.

        Uses scipy.optimize to find weights that maximize
        classification accuracy on validation data.

        Args:
            base_probas: List of probability arrays.
            y_true: True labels.

        Returns:
            Optimal weights (sum to 1).
        """
        from scipy.optimize import minimize

        n_models = len(base_probas)

        def neg_accuracy(w: NDArray[np.float64]) -> float:
            # Normalize weights
            w_norm = np.abs(w) / np.abs(w).sum()
            preds = self.weighted_average_predict(base_probas, w_norm)
            return -float(np.mean(preds == y_true))

        # Start with equal weights
        w0 = np.ones(n_models) / n_models
        result = minimize(
            neg_accuracy,
            w0,
            method="Nelder-Mead",
            options={"maxiter": 200},
        )
        optimal: NDArray[np.float64] = (np.abs(result.x) / np.abs(result.x).sum()).astype(
            np.float64,
        )
        return optimal

    def save(self, path: Path | str) -> None:
        """Save meta-learner to disk."""
        if self._meta_model is None:
            msg = "Meta-model not fitted. Cannot save."
            raise RuntimeError(msg)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        import joblib

        joblib.dump(self._meta_model, str(path.with_suffix(".joblib")))

        meta = {
            "n_base_models": self._n_base_models,
            "base_model_names": self._base_model_names,
        }
        meta_path = path.with_suffix(".meta.json")
        meta_path.write_text(json.dumps(meta, indent=2))

    def load(self, path: Path | str) -> None:
        """Load meta-learner from disk."""
        path = Path(path)

        import joblib

        self._meta_model = joblib.load(str(path.with_suffix(".joblib")))

        meta_path = path.with_suffix(".meta.json")
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            self._n_base_models = meta.get("n_base_models", 0)
            self._base_model_names = meta.get("base_model_names")
