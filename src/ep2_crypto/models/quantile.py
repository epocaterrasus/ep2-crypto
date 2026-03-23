"""LightGBM quantile regression for prediction intervals.

Trains 5 separate LightGBM models at quantiles (10th, 25th, 50th, 75th, 90th)
to produce prediction intervals. The interval width serves as a risk signal
for the confidence gating pipeline.

Wider intervals → more uncertainty → lower confidence → smaller position.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import lightgbm as lgb
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


DEFAULT_QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)


@dataclass
class QuantileConfig:
    """Configuration for quantile regression models."""

    quantiles: tuple[float, ...] = DEFAULT_QUANTILES
    num_leaves: int = 31
    max_depth: int = 5
    learning_rate: float = 0.05
    n_estimators: int = 300
    min_child_samples: int = 50
    early_stopping_rounds: int = 30
    random_state: int = 42


class QuantileModel:
    """LightGBM quantile regression ensemble.

    Trains one LightGBM regressor per quantile. Predicts the full
    distribution of forward returns, enabling:
    - Prediction interval construction
    - Uncertainty estimation (interval width)
    - Risk-adjusted position sizing
    """

    def __init__(self, config: QuantileConfig | None = None) -> None:
        self._config = config or QuantileConfig()
        self._models: dict[float, lgb.LGBMRegressor] = {}
        self._feature_names: list[str] | None = None

    @property
    def is_fitted(self) -> bool:
        return len(self._models) == len(self._config.quantiles)

    @property
    def quantiles(self) -> tuple[float, ...]:
        return self._config.quantiles

    def train(
        self,
        x_train: NDArray[np.float64],
        y_train: NDArray[np.float64],
        x_val: NDArray[np.float64] | None = None,
        y_val: NDArray[np.float64] | None = None,
        feature_names: list[str] | None = None,
    ) -> dict[str, float]:
        """Train quantile regression models.

        Args:
            x_train: Training features.
            y_train: Training targets (forward returns, continuous).
            x_val: Validation features.
            y_val: Validation targets.
            feature_names: Feature names.

        Returns:
            Dict with training metrics per quantile.
        """
        self._feature_names = feature_names
        cfg = self._config
        metrics: dict[str, float] = {}

        for q in cfg.quantiles:
            model = lgb.LGBMRegressor(
                objective="quantile",
                alpha=q,
                num_leaves=cfg.num_leaves,
                max_depth=cfg.max_depth,
                learning_rate=cfg.learning_rate,
                n_estimators=cfg.n_estimators,
                min_child_samples=cfg.min_child_samples,
                random_state=cfg.random_state,
                verbose=-1,
            )

            fit_kwargs: dict[str, Any] = {"X": x_train, "y": y_train}
            callbacks: list[Any] = []
            eval_set = []
            eval_names = []

            if x_val is not None and y_val is not None:
                eval_set.append((x_val, y_val))
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
            self._models[q] = model

            q_key = f"q{int(q * 100)}"
            metrics[f"{q_key}_best_iter"] = float(
                model.best_iteration_ if model.best_iteration_ > 0 else cfg.n_estimators,
            )

        return metrics

    def predict(
        self,
        x: NDArray[np.float64],
    ) -> dict[float, NDArray[np.float64]]:
        """Predict all quantiles.

        Args:
            x: Features (n_samples, n_features).

        Returns:
            Dict mapping quantile → predictions array.
        """
        if not self.is_fitted:
            msg = "Model not fitted. Call train() first."
            raise RuntimeError(msg)

        return {q: model.predict(x).astype(np.float64) for q, model in self._models.items()}

    def predict_intervals(
        self,
        x: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Predict median and 80% prediction interval (10th-90th).

        Args:
            x: Features (n_samples, n_features).

        Returns:
            Tuple of (median, lower_bound, upper_bound).
        """
        preds = self.predict(x)
        median = preds[0.50]
        lower = preds[min(self._config.quantiles)]
        upper = preds[max(self._config.quantiles)]
        return median, lower, upper

    def interval_width(
        self,
        x: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute prediction interval width (uncertainty measure).

        Width = q90 - q10. Wider → more uncertainty.

        Args:
            x: Features.

        Returns:
            Interval width array.
        """
        _, lower, upper = self.predict_intervals(x)
        return upper - lower

    def check_quantile_ordering(
        self,
        x: NDArray[np.float64],
    ) -> float:
        """Check that predicted quantiles are monotonically ordered.

        Returns the fraction of samples where ordering is violated.
        Crossing quantiles indicate poor calibration.
        """
        preds = self.predict(x)
        sorted_q = sorted(self._config.quantiles)
        n = len(x)
        violations = 0
        for i in range(n):
            for j in range(1, len(sorted_q)):
                if preds[sorted_q[j]][i] < preds[sorted_q[j - 1]][i]:
                    violations += 1
                    break
        return violations / n

    def save(self, path: Path | str) -> None:
        """Save all quantile models to disk."""
        if not self.is_fitted:
            msg = "Model not fitted. Cannot save."
            raise RuntimeError(msg)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        for q, model in self._models.items():
            q_path = path.parent / f"{path.stem}_q{int(q * 100)}.txt"
            model.booster_.save_model(str(q_path))

        meta = {
            "quantiles": list(self._config.quantiles),
            "feature_names": self._feature_names,
        }
        meta_path = path.with_suffix(".meta.json")
        meta_path.write_text(json.dumps(meta, indent=2))

    def load(self, path: Path | str) -> None:
        """Load all quantile models from disk."""
        path = Path(path)

        meta_path = path.with_suffix(".meta.json")
        meta = json.loads(meta_path.read_text())
        quantiles = tuple(meta["quantiles"])
        self._feature_names = meta.get("feature_names")
        self._config = QuantileConfig(quantiles=quantiles)

        n_features = len(self._feature_names) if self._feature_names else 1

        for q in quantiles:
            q_path = path.parent / f"{path.stem}_q{int(q * 100)}.txt"
            booster = lgb.Booster(model_file=str(q_path))
            model = lgb.LGBMRegressor(
                objective="quantile",
                alpha=q,
                n_estimators=1,
                verbose=-1,
            )
            _dummy_x = np.zeros((3, n_features))
            _dummy_y = np.array([0.0, 1.0, 2.0])
            model.fit(_dummy_x, _dummy_y)
            model._Booster = booster
            model.n_features_in_ = n_features
            self._models[q] = model
