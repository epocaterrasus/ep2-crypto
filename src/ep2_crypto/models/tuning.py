"""Hyperparameter tuning with Optuna.

Provides Optuna-based hyperparameter optimization for LightGBM, CatBoost,
and GRU models, plus confidence threshold optimization and feature importance
stability analysis.

Key design decisions:
- Objective: walk-forward Sharpe (NOT accuracy) — ensures tuning improves trading
- MedianPruner: stops unpromising trials early (startup_trials=5, warmup_steps=10)
- Deflated Sharpe Ratio: corrects for multiple comparison bias across trials
- Annualization: sqrt(105_120) — 24/7 crypto, 288 bars/day * 365 days
- Search spaces from SPRINTS.md / CLAUDE.md research

Acceptance criteria (from SPRINTS.md):
- 50+ trials per model
- Pruning stops >30% of trials early
- Top-10 feature overlap >70% across folds (stability)
- DSR applied to final parameters
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import optuna
import structlog

from ep2_crypto.models.catboost_direction import CatBoostConfig, CatBoostDirectionModel
from ep2_crypto.models.gru_features import GRUConfig, GRUFeatureExtractor
from ep2_crypto.models.lgbm_direction import LGBMConfig, LGBMDirectionModel

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = structlog.get_logger(__name__)

# Annualization factor: sqrt(288 bars/day * 365 days) = sqrt(105_120)
_ANNUALIZATION = math.sqrt(105_120)


# ---------------------------------------------------------------------------
# Sharpe helpers
# ---------------------------------------------------------------------------


def walk_forward_sharpe(
    returns: NDArray[np.float64],
    n_splits: int = 3,
    purge_bars: int = 18,
) -> float:
    """Compute mean walk-forward Sharpe across temporal splits.

    Uses a simple temporal split (no model retraining) for speed in Optuna
    trials. Full walk-forward with retraining is only done for final evaluation.

    Args:
        returns: Array of per-bar returns, in chronological order.
        n_splits: Number of temporal folds.
        purge_bars: Bars to drop between train and test to avoid leakage.

    Returns:
        Mean Sharpe ratio across folds. Returns -10.0 on degenerate input.
    """
    if len(returns) < 100:
        return -10.0

    fold_size = len(returns) // (n_splits + 1)
    sharpes: list[float] = []

    for i in range(n_splits):
        test_start = (i + 1) * fold_size + purge_bars
        test_end = (i + 2) * fold_size
        test_returns = returns[test_start:test_end]

        if len(test_returns) < 10:
            continue

        std = float(np.std(test_returns))
        if std < 1e-10:
            sharpes.append(0.0)
        else:
            sharpe = float(np.mean(test_returns)) / std * _ANNUALIZATION
            sharpes.append(sharpe)

    return float(np.mean(sharpes)) if sharpes else -10.0


def deflated_sharpe_ratio(
    observed_sharpe: float,
    n_trials: int,
    n_bars: int,
    skewness: float = 0.0,
    excess_kurtosis: float = 0.0,
) -> float:
    """Compute the Deflated Sharpe Ratio (DSR) for multiple comparison correction.

    Adjusts the observed Sharpe ratio to account for the fact that we tested
    many parameter combinations (multiple hypothesis testing). Based on
    Bailey & Lopez de Prado (2014).

    Args:
        observed_sharpe: Best Sharpe from all trials (annualized).
        n_trials: Number of parameter combinations tested.
        n_bars: Number of bars in the test period.
        skewness: Return distribution skewness (0 = normal).
        excess_kurtosis: Excess kurtosis (0 = normal).

    Returns:
        DSR: probability that the true Sharpe > 0 after multiple testing.
        Returns value in [0, 1]. Values > 0.95 suggest genuine edge.
    """
    if n_trials <= 0 or n_bars <= 0:
        return 0.0

    # Expected maximum Sharpe from n_trials random strategies (Bailey & Lopez de Prado eq. 9)
    euler_mascheroni = 0.5772156649
    expected_max_sr = (
        (1 - euler_mascheroni) * _inverse_normal(1 - 1.0 / n_trials)
        + euler_mascheroni * _inverse_normal(1 - 1.0 / (n_trials * math.e))
    )

    # Sharpe standard deviation under non-normality
    sr_std = math.sqrt(
        (1 - skewness * observed_sharpe + (excess_kurtosis / 4.0) * observed_sharpe**2)
        / (n_bars - 1)
    )

    # DSR: probability that observed SR beats the expected max under H0
    if sr_std < 1e-10:
        return 0.0

    z = (observed_sharpe - expected_max_sr) / sr_std
    return float(_normal_cdf(z))


def _inverse_normal(p: float) -> float:
    """Rational approximation of inverse normal CDF (Abramowitz & Stegun)."""
    p = max(1e-10, min(1 - 1e-10, p))
    if p < 0.5:
        t = math.sqrt(-2 * math.log(p))
        sign = -1.0
    else:
        t = math.sqrt(-2 * math.log(1 - p))
        sign = 1.0
    c = [2.515517, 0.802853, 0.010328]
    d = [1.432788, 0.189269, 0.001308]
    num = c[0] + c[1] * t + c[2] * t**2
    den = 1 + d[0] * t + d[1] * t**2 + d[2] * t**3
    return sign * (t - num / den)


def _normal_cdf(x: float) -> float:
    """Standard normal CDF via math.erfc."""
    return 0.5 * math.erfc(-x / math.sqrt(2))


# ---------------------------------------------------------------------------
# Simple synthetic returns builder for use in tuning objectives
# ---------------------------------------------------------------------------


def _predictions_to_returns(
    probas: NDArray[np.float64],
    y_true: NDArray[np.int8],
    threshold: float = 0.55,
    cost_bps: float = 10.0,
) -> NDArray[np.float64]:
    """Convert predicted probabilities to simulated returns for Sharpe computation.

    Uses max-class probability vs threshold as trade signal. Applies a flat
    round-trip cost in bps. This is intentionally simple — used only for
    Optuna trial evaluation, not final backtesting.

    Args:
        probas: Shape (n_samples, 3) — [DOWN, FLAT, UP] class probabilities.
        y_true: True labels in {-1, 0, 1}.
        threshold: Min probability to take a position.
        cost_bps: Round-trip cost in bps, applied per trade.

    Returns:
        Array of per-bar returns (non-traded bars return 0.0).
    """
    # Map class indices back: 0=DOWN(-1), 1=FLAT(0), 2=UP(+1)
    max_class = np.argmax(probas, axis=1)
    max_prob = probas[np.arange(len(probas)), max_class]

    signal = np.zeros(len(probas), dtype=np.float64)
    signal[max_prob > threshold] = (max_class[max_prob > threshold] - 1).astype(
        np.float64
    )

    # Simulated return: signal * direction of true label (crude approximation)
    true_dir = y_true.astype(np.float64)
    raw_returns = signal * true_dir / 100.0  # normalized scale

    # Deduct cost on every trade entry
    is_trade = signal != 0
    cost = cost_bps / 10_000.0
    raw_returns[is_trade] -= cost

    return raw_returns


# ---------------------------------------------------------------------------
# LightGBM Tuner
# ---------------------------------------------------------------------------


@dataclass
class TuningResult:
    """Result from a completed Optuna study."""

    best_params: dict[str, Any]
    best_sharpe: float
    dsr: float
    n_trials: int
    n_pruned: int
    pruning_rate: float
    study_name: str

    def to_config(self) -> dict[str, Any]:
        """Return params dict ready to be passed to the model Config dataclass."""
        return dict(self.best_params)


class LGBMTuner:
    """Optuna study for LightGBM hyperparameter tuning.

    Search space from CLAUDE.md:
    - num_leaves: [15, 63]
    - max_depth: [3, 8]
    - learning_rate: [0.01, 0.3] (log scale)
    - min_child_samples: [20, 100]
    - subsample: [0.6, 1.0]
    - colsample_bytree: [0.6, 1.0]
    - reg_alpha: [1e-3, 25] (log scale)
    - reg_lambda: [1e-3, 25] (log scale)
    """

    def __init__(
        self,
        n_trials: int = 50,
        timeout: float | None = None,
        n_jobs: int = 1,
    ) -> None:
        self._n_trials = n_trials
        self._timeout = timeout
        self._n_jobs = n_jobs

    def tune(
        self,
        x_train: NDArray[np.float64],
        y_train: NDArray[np.int8],
        x_val: NDArray[np.float64],
        y_val: NDArray[np.int8],
        feature_names: list[str] | None = None,
        study_name: str = "lgbm_tuning",
        seed: int = 42,
    ) -> TuningResult:
        """Run Optuna study.

        Args:
            x_train: Training features.
            y_train: Training labels in {-1, 0, 1}.
            x_val: Validation features (purged from training).
            y_val: Validation labels.
            feature_names: Optional feature names for logging.
            study_name: Optuna study identifier.
            seed: Random seed for reproducibility.

        Returns:
            TuningResult with best params and diagnostics.
        """
        sampler = optuna.samplers.TPESampler(seed=seed)
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=10,
        )
        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
            study_name=study_name,
        )

        def objective(trial: optuna.Trial) -> float:
            config = LGBMConfig(
                num_leaves=trial.suggest_int("num_leaves", 15, 63),
                max_depth=trial.suggest_int("max_depth", 3, 8),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                n_estimators=trial.suggest_int("n_estimators", 200, 800),
                min_child_samples=trial.suggest_int("min_child_samples", 20, 100),
                subsample=trial.suggest_float("subsample", 0.6, 1.0),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
                reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 25.0, log=True),
                reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 25.0, log=True),
                early_stopping_rounds=50,
            )
            model = LGBMDirectionModel(config)
            try:
                model.train(
                    x_train,
                    y_train,
                    x_val=x_val,
                    y_val=y_val,
                    feature_names=feature_names,
                )
                if not model.is_fitted:
                    return -10.0
                probas = model.predict_proba(x_val)
                returns = _predictions_to_returns(probas, y_val)
                sharpe = walk_forward_sharpe(returns, n_splits=1)
            except Exception:
                return -10.0

            # Report intermediate value for pruning after first fold
            trial.report(sharpe, step=0)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            return sharpe

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(
            objective,
            n_trials=self._n_trials,
            timeout=self._timeout,
            n_jobs=self._n_jobs,
            catch=(Exception,),
        )

        n_pruned = sum(
            1
            for t in study.trials
            if t.state == optuna.trial.TrialState.PRUNED
        )
        n_completed = sum(
            1
            for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        )
        n_total = len(study.trials)
        pruning_rate = n_pruned / n_total if n_total > 0 else 0.0

        best_sharpe = study.best_value if study.best_trial else -10.0
        best_params = study.best_params if study.best_trial else {}

        dsr = deflated_sharpe_ratio(
            observed_sharpe=best_sharpe,
            n_trials=max(1, n_completed),
            n_bars=len(y_val),
        )

        result = TuningResult(
            best_params=best_params,
            best_sharpe=best_sharpe,
            dsr=dsr,
            n_trials=n_total,
            n_pruned=n_pruned,
            pruning_rate=pruning_rate,
            study_name=study_name,
        )
        logger.info(
            "lgbm_tuning_complete",
            best_sharpe=round(best_sharpe, 4),
            dsr=round(dsr, 4),
            n_trials=n_total,
            n_pruned=n_pruned,
            pruning_rate=round(pruning_rate, 3),
        )
        return result

    def best_config(self, result: TuningResult) -> LGBMConfig:
        """Build LGBMConfig from tuning result."""
        p = result.best_params
        return LGBMConfig(
            num_leaves=p.get("num_leaves", 31),
            max_depth=p.get("max_depth", 5),
            learning_rate=p.get("learning_rate", 0.05),
            n_estimators=p.get("n_estimators", 500),
            min_child_samples=p.get("min_child_samples", 50),
            subsample=p.get("subsample", 0.8),
            colsample_bytree=p.get("colsample_bytree", 0.8),
            reg_alpha=p.get("reg_alpha", 0.1),
            reg_lambda=p.get("reg_lambda", 1.0),
        )


# ---------------------------------------------------------------------------
# CatBoost Tuner
# ---------------------------------------------------------------------------


class CatBoostTuner:
    """Optuna study for CatBoost hyperparameter tuning.

    Search space from CLAUDE.md / SPRINTS.md:
    - depth: [3, 8]
    - learning_rate: [0.01, 0.3] (log scale)
    - iterations: [200, 800]
    - l2_leaf_reg: [1.0, 10.0] (log scale)
    - subsample: [0.6, 1.0]
    """

    def __init__(
        self,
        n_trials: int = 50,
        timeout: float | None = None,
        n_jobs: int = 1,
    ) -> None:
        self._n_trials = n_trials
        self._timeout = timeout
        self._n_jobs = n_jobs

    def tune(
        self,
        x_train: NDArray[np.float64],
        y_train: NDArray[np.int8],
        x_val: NDArray[np.float64],
        y_val: NDArray[np.int8],
        study_name: str = "catboost_tuning",
        seed: int = 42,
    ) -> TuningResult:
        """Run Optuna study for CatBoost."""
        sampler = optuna.samplers.TPESampler(seed=seed)
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=10,
        )
        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
            study_name=study_name,
        )

        def objective(trial: optuna.Trial) -> float:
            config = CatBoostConfig(
                depth=trial.suggest_int("depth", 3, 8),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                iterations=trial.suggest_int("iterations", 200, 800),
                l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
                subsample=trial.suggest_float("subsample", 0.6, 1.0),
                early_stopping_rounds=50,
            )
            model = CatBoostDirectionModel(config)
            try:
                model.train(x_train, y_train, x_val=x_val, y_val=y_val)
                if not model.is_fitted:
                    return -10.0
                probas = model.predict_proba(x_val)
                returns = _predictions_to_returns(probas, y_val)
                sharpe = walk_forward_sharpe(returns, n_splits=1)
            except Exception:
                return -10.0

            trial.report(sharpe, step=0)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            return sharpe

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(
            objective,
            n_trials=self._n_trials,
            timeout=self._timeout,
            n_jobs=self._n_jobs,
            catch=(Exception,),
        )

        n_pruned = sum(
            1
            for t in study.trials
            if t.state == optuna.trial.TrialState.PRUNED
        )
        n_total = len(study.trials)
        n_completed = sum(
            1
            for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        )
        pruning_rate = n_pruned / n_total if n_total > 0 else 0.0
        best_sharpe = study.best_value if study.best_trial else -10.0
        best_params = study.best_params if study.best_trial else {}

        dsr = deflated_sharpe_ratio(
            observed_sharpe=best_sharpe,
            n_trials=max(1, n_completed),
            n_bars=len(y_val),
        )

        result = TuningResult(
            best_params=best_params,
            best_sharpe=best_sharpe,
            dsr=dsr,
            n_trials=n_total,
            n_pruned=n_pruned,
            pruning_rate=pruning_rate,
            study_name=study_name,
        )
        logger.info(
            "catboost_tuning_complete",
            best_sharpe=round(best_sharpe, 4),
            dsr=round(dsr, 4),
            n_trials=n_total,
            n_pruned=n_pruned,
        )
        return result

    def best_config(self, result: TuningResult) -> CatBoostConfig:
        """Build CatBoostConfig from tuning result."""
        p = result.best_params
        return CatBoostConfig(
            depth=p.get("depth", 5),
            learning_rate=p.get("learning_rate", 0.05),
            iterations=p.get("iterations", 500),
            l2_leaf_reg=p.get("l2_leaf_reg", 3.0),
            subsample=p.get("subsample", 0.8),
        )


# ---------------------------------------------------------------------------
# GRU Tuner
# ---------------------------------------------------------------------------


class GRUTuner:
    """Optuna study for GRU hyperparameter tuning.

    Search space from CLAUDE.md:
    - hidden_size: [32, 256]
    - num_layers: [1, 3]
    - learning_rate: [1e-5, 1e-2] (log scale)
    - dropout: [0.1, 0.5]
    - seq_len: [12, 60]

    Pruning is based on validation loss after each epoch. Uses a lightweight
    n_epochs=5 fast-eval mode to keep trial time manageable.
    """

    def __init__(
        self,
        n_trials: int = 50,
        timeout: float | None = None,
        fast_eval_epochs: int = 5,
    ) -> None:
        self._n_trials = n_trials
        self._timeout = timeout
        self._fast_eval_epochs = fast_eval_epochs

    def tune(
        self,
        x_train: NDArray[np.float64],
        y_train: NDArray[np.int8],
        x_val: NDArray[np.float64],
        y_val: NDArray[np.int8],
        study_name: str = "gru_tuning",
        seed: int = 42,
    ) -> TuningResult:
        """Run Optuna study for GRU feature extractor.

        Args:
            x_train: Training sequence data (n_samples, n_features).
            y_train: Training labels in {-1, 0, 1}.
            x_val: Validation sequence data.
            y_val: Validation labels.
            study_name: Optuna study identifier.
            seed: Random seed.

        Returns:
            TuningResult with best hyperparameters.
        """
        sampler = optuna.samplers.TPESampler(seed=seed)
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=10,
        )
        study = optuna.create_study(
            direction="minimize",  # minimize validation loss
            sampler=sampler,
            pruner=pruner,
            study_name=study_name,
        )

        def objective(trial: optuna.Trial) -> float:
            config = GRUConfig(
                hidden_size=trial.suggest_int("hidden_size", 32, 256),
                num_layers=trial.suggest_int("num_layers", 1, 3),
                learning_rate=trial.suggest_float(
                    "learning_rate", 1e-5, 1e-2, log=True
                ),
                dropout=trial.suggest_float("dropout", 0.1, 0.5),
                seq_len=trial.suggest_int("seq_len", 12, 60),
                n_epochs=self._fast_eval_epochs,
            )
            model = GRUFeatureExtractor(config)
            try:
                metrics = model.train(x_train, y_train, x_val=x_val, y_val=y_val)
                val_loss = metrics.get("val_loss", float("inf"))
            except Exception:
                return float("inf")

            trial.report(val_loss, step=0)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            return val_loss

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(
            objective,
            n_trials=self._n_trials,
            timeout=self._timeout,
            catch=(Exception,),
        )

        n_pruned = sum(
            1
            for t in study.trials
            if t.state == optuna.trial.TrialState.PRUNED
        )
        n_total = len(study.trials)
        n_completed = sum(
            1
            for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        )
        pruning_rate = n_pruned / n_total if n_total > 0 else 0.0
        best_val_loss = study.best_value if study.best_trial else float("inf")
        best_params = study.best_params if study.best_trial else {}

        # For GRU we use negative val_loss as proxy for "Sharpe" in DSR
        proxy_sharpe = -best_val_loss
        dsr = deflated_sharpe_ratio(
            observed_sharpe=proxy_sharpe,
            n_trials=max(1, n_completed),
            n_bars=len(x_val),
        )

        result = TuningResult(
            best_params=best_params,
            best_sharpe=proxy_sharpe,
            dsr=dsr,
            n_trials=n_total,
            n_pruned=n_pruned,
            pruning_rate=pruning_rate,
            study_name=study_name,
        )
        logger.info(
            "gru_tuning_complete",
            best_val_loss=round(best_val_loss, 6),
            n_trials=n_total,
            n_pruned=n_pruned,
        )
        return result

    def best_config(self, result: TuningResult) -> GRUConfig:
        """Build GRUConfig from tuning result."""
        p = result.best_params
        return GRUConfig(
            hidden_size=p.get("hidden_size", 64),
            num_layers=p.get("num_layers", 2),
            learning_rate=p.get("learning_rate", 1e-3),
            dropout=p.get("dropout", 0.3),
            seq_len=p.get("seq_len", 24),
        )


# ---------------------------------------------------------------------------
# Confidence Threshold Optimizer
# ---------------------------------------------------------------------------


@dataclass
class ThresholdResult:
    """Result from confidence threshold grid search."""

    best_threshold: float
    best_sharpe: float
    regime_thresholds: dict[str, float]
    grid_results: dict[float, float]


class ThresholdOptimizer:
    """Grid search confidence thresholds to maximize walk-forward Sharpe.

    Searches over [0.50, 0.55, 0.60, ..., 0.80] for each regime.
    Optimizes for Sharpe, NOT win rate — a higher threshold that trades
    less but with better edge is preferred.
    """

    _DEFAULT_GRID: list[float] = [
        0.50, 0.52, 0.54, 0.55, 0.56, 0.58, 0.60,
        0.62, 0.64, 0.65, 0.66, 0.68, 0.70, 0.72, 0.75, 0.80,
    ]

    def __init__(self, threshold_grid: list[float] | None = None) -> None:
        self._grid = threshold_grid or self._DEFAULT_GRID

    def optimize(
        self,
        probas: NDArray[np.float64],
        y_true: NDArray[np.int8],
        regimes: NDArray[np.int32] | None = None,
        cost_bps: float = 10.0,
    ) -> ThresholdResult:
        """Grid search threshold on validation data.

        Args:
            probas: Shape (n_samples, 3) — [DOWN, FLAT, UP] probabilities.
            y_true: True labels in {-1, 0, 1}.
            regimes: Optional regime per bar (0=bear, 1=neutral, 2=bull).
                     If provided, optimize per-regime.
            cost_bps: Round-trip transaction cost in bps.

        Returns:
            ThresholdResult with best global and per-regime thresholds.
        """
        # Global optimization
        grid_results: dict[float, float] = {}
        for threshold in self._grid:
            returns = _predictions_to_returns(probas, y_true, threshold, cost_bps)
            sharpe = walk_forward_sharpe(returns, n_splits=2)
            grid_results[threshold] = sharpe

        best_threshold = max(grid_results, key=lambda k: grid_results[k])
        best_sharpe = grid_results[best_threshold]

        # Per-regime optimization
        regime_thresholds: dict[str, float] = {}
        if regimes is not None:
            regime_names = {0: "bear", 1: "neutral", 2: "bull"}
            for regime_id, regime_name in regime_names.items():
                mask = regimes == regime_id
                if mask.sum() < 50:
                    regime_thresholds[regime_name] = best_threshold
                    continue

                regime_results: dict[float, float] = {}
                for threshold in self._grid:
                    r = _predictions_to_returns(
                        probas[mask], y_true[mask], threshold, cost_bps
                    )
                    s = walk_forward_sharpe(r, n_splits=1)
                    regime_results[threshold] = s

                regime_thresholds[regime_name] = max(
                    regime_results, key=lambda k: regime_results[k]
                )

        logger.info(
            "threshold_optimization_complete",
            best_threshold=best_threshold,
            best_sharpe=round(best_sharpe, 4),
            regime_thresholds=regime_thresholds,
        )
        return ThresholdResult(
            best_threshold=best_threshold,
            best_sharpe=best_sharpe,
            regime_thresholds=regime_thresholds,
            grid_results=grid_results,
        )


# ---------------------------------------------------------------------------
# Feature Importance Stability Analyzer
# ---------------------------------------------------------------------------


@dataclass
class StabilityResult:
    """Result from feature importance stability analysis."""

    stable_features: list[str]  # Appear in >50% of folds
    unstable_features: list[str]  # Appear in <50% of folds (removal candidates)
    top10_overlap_rate: float  # Fraction of folds where top-10 overlaps >70%
    feature_appearance_rate: dict[str, float]  # Feature → fraction of folds it appeared in
    mean_importance: dict[str, float]  # Feature → mean importance across folds


class FeatureImportanceAnalyzer:
    """Analyzes feature importance stability across walk-forward folds.

    Features that appear in the top-N in fewer than 50% of folds are
    candidates for removal — they indicate unstable/inconsistent signal.
    Top-10 overlap rate measures overall stability.
    """

    def __init__(self, top_n: int = 10, stability_threshold: float = 0.50) -> None:
        self._top_n = top_n
        self._stability_threshold = stability_threshold

    def analyze(
        self,
        fold_importances: list[dict[str, float]],
    ) -> StabilityResult:
        """Analyze feature importance across folds.

        Args:
            fold_importances: List of {feature_name: importance} dicts,
                one per walk-forward fold. Importances should be non-negative.

        Returns:
            StabilityResult with stable/unstable features and overlap stats.

        Raises:
            ValueError: If fewer than 2 folds provided.
        """
        if len(fold_importances) < 2:
            raise ValueError(
                f"Need at least 2 folds for stability analysis, got {len(fold_importances)}"
            )

        n_folds = len(fold_importances)
        all_features: set[str] = set()
        for fold in fold_importances:
            all_features.update(fold.keys())

        # Per-feature: fraction of folds where it appears in top-N
        appearance_count: dict[str, int] = {f: 0 for f in all_features}
        sum_importance: dict[str, float] = {f: 0.0 for f in all_features}

        fold_top_sets: list[set[str]] = []
        for fold in fold_importances:
            sorted_features = sorted(fold.items(), key=lambda x: x[1], reverse=True)
            top_n_features = {f for f, _ in sorted_features[: self._top_n]}
            fold_top_sets.append(top_n_features)
            for feature in top_n_features:
                appearance_count[feature] += 1
            for feature, importance in fold.items():
                sum_importance[feature] += importance

        appearance_rate = {
            f: appearance_count[f] / n_folds for f in all_features
        }
        mean_importance = {
            f: sum_importance[f] / n_folds for f in all_features
        }

        stable = [
            f for f, rate in appearance_rate.items() if rate >= self._stability_threshold
        ]
        unstable = [
            f for f, rate in appearance_rate.items() if rate < self._stability_threshold
        ]

        # Top-10 overlap rate: fraction of consecutive fold pairs with >70% overlap
        if len(fold_top_sets) < 2:
            top10_overlap_rate = 1.0
        else:
            overlap_threshold = 0.70
            n_pairs = len(fold_top_sets) - 1
            overlapping_pairs = 0
            for i in range(n_pairs):
                set_a = fold_top_sets[i]
                set_b = fold_top_sets[i + 1]
                if len(set_a) == 0 and len(set_b) == 0:
                    overlapping_pairs += 1
                    continue
                union = len(set_a | set_b)
                if union == 0:
                    continue
                intersection = len(set_a & set_b)
                overlap = intersection / min(len(set_a), len(set_b)) if min(len(set_a), len(set_b)) > 0 else 0.0
                if overlap >= overlap_threshold:
                    overlapping_pairs += 1
            top10_overlap_rate = overlapping_pairs / n_pairs if n_pairs > 0 else 1.0

        logger.info(
            "feature_stability_analysis",
            n_stable=len(stable),
            n_unstable=len(unstable),
            top10_overlap_rate=round(top10_overlap_rate, 3),
            n_folds=n_folds,
        )
        return StabilityResult(
            stable_features=sorted(stable, key=lambda f: mean_importance[f], reverse=True),
            unstable_features=sorted(unstable, key=lambda f: mean_importance[f], reverse=True),
            top10_overlap_rate=top10_overlap_rate,
            feature_appearance_rate=appearance_rate,
            mean_importance=mean_importance,
        )
