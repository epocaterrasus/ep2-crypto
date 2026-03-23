"""Meta-labeling model for trade profitability prediction.

Implements Lopez de Prado's meta-labeling technique: a secondary LightGBM
binary classifier that predicts whether the primary model's trade will be
profitable. The output P(profitable) feeds into the confidence gating
pipeline to filter low-quality trades.

Key design:
- Input: primary predictions, primary probabilities, original features, regime
- Target: binary (1 = primary model's direction was correct, 0 = incorrect)
- Output: P(profitable) used for gating and position sizing
- Trainable on walk-forward out-of-fold predictions
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import lightgbm as lgb
import numpy as np
import structlog

if TYPE_CHECKING:
    from typing import Any

    from numpy.typing import NDArray

logger = structlog.get_logger(__name__)


@dataclass
class MetaLabelConfig:
    """LightGBM hyperparameters for the meta-labeling binary classifier.

    Conservative defaults to avoid overfitting the meta-learner,
    which trains on OOF predictions with inherently noisy labels.
    """

    num_leaves: int = 15
    max_depth: int = 4
    learning_rate: float = 0.03
    n_estimators: int = 300
    min_child_samples: int = 100
    subsample: float = 0.7
    colsample_bytree: float = 0.7
    reg_alpha: float = 0.5
    reg_lambda: float = 2.0
    early_stopping_rounds: int = 30
    random_state: int = 42


# Feature group names for diagnostics
_PRIMARY_PRED_NAME = "primary_pred"
_PRIMARY_PROBA_PREFIX = "primary_proba_"
_REGIME_NAME = "regime"


class MetaLabeler:
    """Meta-labeling binary classifier for trade profitability prediction.

    A secondary LightGBM model that learns to predict whether the primary
    direction model's trades will be profitable. Used as the first stage
    in the confidence gating pipeline.

    Workflow:
        1. Primary model produces OOF predictions on walk-forward folds
        2. create_meta_features() builds the augmented feature matrix
        3. fit() trains the binary classifier on (meta_features, is_profitable)
        4. predict_proba() returns P(profitable) for new trades
        5. gate() filters trades below a confidence threshold
    """

    def __init__(self, config: MetaLabelConfig | None = None) -> None:
        self._config = config or MetaLabelConfig()
        self._model: lgb.LGBMClassifier | None = None
        self._feature_names: list[str] | None = None
        self._feature_importance_map: dict[str, float] | None = None
        self._n_original_features: int | None = None
        self._best_iteration: int | None = None

    @property
    def is_fitted(self) -> bool:
        """Whether the model has been trained."""
        return self._model is not None and self._model.fitted_

    def create_meta_features(
        self,
        primary_predictions: NDArray[np.int8],
        primary_probas: NDArray[np.float64],
        features: NDArray[np.float64],
        regime_labels: NDArray[np.int8],
    ) -> NDArray[np.float64]:
        """Build the meta-feature matrix for meta-labeling.

        Concatenates:
        - Primary model's predicted class (1 column)
        - Primary model's class probabilities (3 columns: down, flat, up)
        - Original feature matrix (n_features columns)
        - Regime label (1 column)

        Args:
            primary_predictions: Predicted direction classes, shape (n_samples,).
                Values in {-1, 0, 1}.
            primary_probas: Class probabilities from primary model,
                shape (n_samples, 3) for [down, flat, up].
            features: Original feature matrix, shape (n_samples, n_features).
            regime_labels: Regime state labels, shape (n_samples,).

        Returns:
            Meta-feature matrix, shape (n_samples, 5 + n_features).
        """
        n_samples = len(primary_predictions)

        if primary_probas.shape[0] != n_samples:
            msg = f"primary_probas has {primary_probas.shape[0]} samples, expected {n_samples}"
            raise ValueError(msg)
        if features.shape[0] != n_samples:
            msg = f"features has {features.shape[0]} samples, expected {n_samples}"
            raise ValueError(msg)
        if len(regime_labels) != n_samples:
            msg = f"regime_labels has {len(regime_labels)} samples, expected {n_samples}"
            raise ValueError(msg)
        if primary_probas.ndim != 2 or primary_probas.shape[1] != 3:
            msg = f"primary_probas must have shape (n_samples, 3), got {primary_probas.shape}"
            raise ValueError(msg)

        self._n_original_features = features.shape[1]

        # Build meta-feature names
        n_feat = features.shape[1]
        names: list[str] = [_PRIMARY_PRED_NAME]
        names.extend(f"{_PRIMARY_PROBA_PREFIX}{i}" for i in range(3))
        names.extend(f"feat_{i}" for i in range(n_feat))
        names.append(_REGIME_NAME)
        self._feature_names = names

        # Concatenate columns
        meta = np.column_stack(
            [
                primary_predictions.astype(np.float64).reshape(-1, 1),
                primary_probas.astype(np.float64),
                features.astype(np.float64),
                regime_labels.astype(np.float64).reshape(-1, 1),
            ]
        )

        logger.debug(
            "meta_features_created",
            n_samples=n_samples,
            n_meta_features=meta.shape[1],
            n_original_features=n_feat,
        )

        return meta

    def fit(
        self,
        meta_features: NDArray[np.float64],
        is_profitable: NDArray[np.int8],
        meta_features_val: NDArray[np.float64] | None = None,
        is_profitable_val: NDArray[np.int8] | None = None,
    ) -> dict[str, float]:
        """Train the meta-labeling binary classifier.

        Args:
            meta_features: Meta-feature matrix from create_meta_features(),
                shape (n_samples, n_meta_features).
            is_profitable: Binary labels (1 = profitable trade, 0 = not),
                shape (n_samples,).
            meta_features_val: Optional validation meta-features for early stopping.
            is_profitable_val: Optional validation labels.

        Returns:
            Dict with training metrics: accuracy, auc, best_iteration.
        """
        y = is_profitable.astype(np.int32)

        cfg = self._config
        model = lgb.LGBMClassifier(
            objective="binary",
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
            is_unbalance=True,
        )

        fit_kwargs: dict[str, Any] = {
            "X": meta_features,
            "y": y,
        }

        callbacks: list[Any] = []
        eval_set: list[tuple[NDArray[np.float64], NDArray[np.int32]]] = []
        eval_names: list[str] = []

        if meta_features_val is not None and is_profitable_val is not None:
            y_val = is_profitable_val.astype(np.int32)
            eval_set.append((meta_features_val, y_val))
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

        if self._feature_names is not None:
            fit_kwargs["feature_name"] = self._feature_names

        model.fit(**fit_kwargs)
        self._model = model

        best_iter = model.best_iteration_
        self._best_iteration = best_iter if best_iter > 0 else cfg.n_estimators

        # Track feature importance
        self._update_feature_importance()

        # Compute metrics
        train_proba = model.predict_proba(meta_features)[:, 1]
        train_pred = (train_proba >= 0.5).astype(np.int32)
        accuracy = float(np.mean(train_pred == y))

        # AUC: handle edge case where all labels are the same class
        unique_labels = np.unique(y)
        if len(unique_labels) < 2:
            auc = 0.5
            logger.warning(
                "meta_label_fit_single_class",
                unique_labels=unique_labels.tolist(),
                msg="All labels belong to one class, AUC set to 0.5",
            )
        else:
            auc = self._compute_auc(y, train_proba)

        metrics: dict[str, float] = {
            "accuracy": accuracy,
            "auc": auc,
            "best_iteration": float(self._best_iteration),
            "n_samples": float(len(y)),
            "positive_rate": float(np.mean(y)),
        }

        logger.info(
            "meta_labeler_fitted",
            accuracy=round(accuracy, 4),
            auc=round(auc, 4),
            best_iteration=self._best_iteration,
            n_samples=len(y),
        )

        return metrics

    def predict_proba(
        self,
        meta_features: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Predict P(profitable) for each sample.

        Args:
            meta_features: Meta-feature matrix from create_meta_features(),
                shape (n_samples, n_meta_features).

        Returns:
            Array of P(profitable) values, shape (n_samples,), in [0, 1].
        """
        if not self.is_fitted:
            msg = "Model not fitted. Call fit() first."
            raise RuntimeError(msg)

        assert self._model is not None  # for mypy, guarded by is_fitted
        probas: NDArray[np.float64] = self._model.predict_proba(meta_features)[:, 1].astype(
            np.float64
        )
        return probas

    def gate(
        self,
        meta_features: NDArray[np.float64],
        threshold: float = 0.5,
    ) -> NDArray[np.bool_]:
        """Gate trades based on meta-labeling confidence.

        Returns a boolean mask where True means "take this trade"
        (P(profitable) >= threshold).

        Args:
            meta_features: Meta-feature matrix, shape (n_samples, n_meta_features).
            threshold: Minimum P(profitable) to pass the gate.

        Returns:
            Boolean array, shape (n_samples,). True = trade approved.
        """
        if not 0.0 <= threshold <= 1.0:
            msg = f"threshold must be in [0, 1], got {threshold}"
            raise ValueError(msg)

        probas = self.predict_proba(meta_features)
        mask = probas >= threshold

        n_passed = int(np.sum(mask))
        n_total = len(mask)
        logger.debug(
            "meta_gate_applied",
            threshold=threshold,
            n_passed=n_passed,
            n_total=n_total,
            pass_rate=round(n_passed / max(n_total, 1), 4),
        )

        return mask

    def feature_importance(self) -> dict[str, float]:
        """Return normalized feature importance from the meta-labeling model.

        Returns:
            Dict of feature name to normalized importance (sums to 1.0).

        Raises:
            RuntimeError: If model is not fitted.
        """
        if not self.is_fitted:
            msg = "Model not fitted. Call fit() first."
            raise RuntimeError(msg)

        if self._feature_importance_map is None:
            msg = "Feature importance not available."
            raise RuntimeError(msg)

        return dict(self._feature_importance_map)

    def save(self, path: Path | str) -> None:
        """Save the meta-labeling model to disk.

        Creates two files:
        - {path}.txt: LightGBM booster model
        - {path}.meta.json: Metadata (feature names, importance, config)

        Args:
            path: Base path for saved files (extensions added automatically).
        """
        if not self.is_fitted:
            msg = "Model not fitted. Cannot save."
            raise RuntimeError(msg)

        assert self._model is not None  # guarded by is_fitted

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save booster
        booster_path = path.with_suffix(".txt")
        self._model.booster_.save_model(str(booster_path))

        # Save metadata
        meta = {
            "feature_names": self._feature_names,
            "feature_importance": self._feature_importance_map,
            "best_iteration": self._best_iteration,
            "n_original_features": self._n_original_features,
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

        logger.info(
            "meta_labeler_saved",
            path=str(path),
            n_features=len(self._feature_names) if self._feature_names else 0,
        )

    def load(self, path: Path | str) -> None:
        """Load a meta-labeling model from disk.

        Args:
            path: Base path (will look for .txt and .meta.json files).
        """
        path = Path(path)

        booster_path = path.with_suffix(".txt")
        if not booster_path.exists():
            msg = f"Booster file not found: {booster_path}"
            raise FileNotFoundError(msg)

        booster = lgb.Booster(model_file=str(booster_path))

        # Load metadata
        meta_path = path.with_suffix(".meta.json")
        n_features = 1
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            self._feature_names = meta.get("feature_names")
            self._feature_importance_map = meta.get("feature_importance")
            self._best_iteration = meta.get("best_iteration")
            self._n_original_features = meta.get("n_original_features")
            if self._feature_names:
                n_features = len(self._feature_names)

        # Reconstruct LGBMClassifier wrapper with dummy fit, then swap booster
        model = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=1,
            verbose=-1,
        )
        _dummy_x = np.zeros((2, n_features))
        _dummy_y = np.array([0, 1])
        model.fit(_dummy_x, _dummy_y)
        model._Booster = booster
        model.n_features_in_ = n_features

        self._model = model

        logger.info(
            "meta_labeler_loaded",
            path=str(path),
            n_features=n_features,
        )

    def _update_feature_importance(self) -> None:
        """Compute and store normalized feature importance."""
        if self._model is None:
            return

        importances = self._model.feature_importances_
        names = self._feature_names or [f"f{i}" for i in range(len(importances))]
        total = float(np.sum(importances))
        if total > 0:
            self._feature_importance_map = {
                name: float(imp / total) for name, imp in zip(names, importances, strict=True)
            }
        else:
            self._feature_importance_map = {name: 0.0 for name in names}

    @staticmethod
    def _compute_auc(
        y_true: NDArray[np.int32],
        y_score: NDArray[np.float64],
    ) -> float:
        """Compute ROC AUC without sklearn dependency.

        Uses the Wilcoxon-Mann-Whitney statistic: AUC = P(score(pos) > score(neg)).
        """
        pos_scores = y_score[y_true == 1]
        neg_scores = y_score[y_true == 0]

        if len(pos_scores) == 0 or len(neg_scores) == 0:
            return 0.5

        # Vectorized comparison
        n_pos = len(pos_scores)
        n_neg = len(neg_scores)
        # Count how many (pos, neg) pairs where pos > neg
        count = 0.0
        for ps in pos_scores:
            count += float(np.sum(ps > neg_scores)) + 0.5 * float(np.sum(ps == neg_scores))

        auc = count / (n_pos * n_neg)
        return auc
