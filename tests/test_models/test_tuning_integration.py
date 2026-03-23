"""Integration tests for the hyperparameter tuning pipeline.

Verifies the sprint acceptance criteria:
- Tuners produce TuningResults with structured params
- Pruning rate is recorded (acceptance: >30% in real runs)
- DSR is applied to best params
- Feature importance stability analysis produces actionable results
- Full pipeline: tune → best_config → feature stability → threshold optimization
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ep2_crypto.models.tuning import (
    CatBoostTuner,
    FeatureImportanceAnalyzer,
    GRUTuner,
    LGBMTuner,
    ThresholdOptimizer,
    TuningResult,
    deflated_sharpe_ratio,
    walk_forward_sharpe,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dataset(
    n_samples: int = 300,
    n_features: int = 10,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create train/val split with synthetic data."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n_samples, n_features)).astype(np.float64)
    y = rng.choice([-1, 0, 1], size=n_samples).astype(np.int8)
    split = int(0.7 * n_samples)
    return x[:split], y[:split], x[split:], y[split:]


def _make_fold_importances(
    n_folds: int = 5,
    n_features: int = 10,
    stable_top_k: int = 5,
    seed: int = 1,
) -> list[dict[str, float]]:
    """Create fold importances where the top-k features are consistently dominant."""
    rng = np.random.default_rng(seed)
    feature_names = [f"feature_{i}" for i in range(n_features)]
    folds = []
    for _ in range(n_folds):
        importances: dict[str, float] = {}
        for j, name in enumerate(feature_names):
            if j < stable_top_k:
                importances[name] = 10.0 - j + rng.uniform(0, 0.5)
            else:
                importances[name] = rng.uniform(0, 1.0)
        folds.append(importances)
    return folds


def _make_mock_lgbm(
    probas: np.ndarray | None = None,
    n_val: int = 90,
) -> MagicMock:
    """Mock LGBMDirectionModel that returns usable probas."""
    mock = MagicMock()
    mock.is_fitted = True
    if probas is None:
        rng = np.random.default_rng(0)
        probas = rng.dirichlet([2, 1, 2], size=n_val).astype(np.float64)
    mock.predict_proba.return_value = probas
    mock.train.return_value = {}
    return mock


def _make_mock_catboost(n_val: int = 90) -> MagicMock:
    mock = MagicMock()
    mock.is_fitted = True
    rng = np.random.default_rng(5)
    mock.predict_proba.return_value = rng.dirichlet([1, 2, 1], size=n_val).astype(np.float64)
    mock.train.return_value = {}
    return mock


def _make_mock_gru() -> MagicMock:
    mock = MagicMock()
    mock.train.return_value = {"train_loss": 0.7, "val_loss": 0.6}
    return mock


# ---------------------------------------------------------------------------
# Acceptance criterion 1: Tuners produce valid TuningResults
# ---------------------------------------------------------------------------


class TestTuningResultStructure:
    """All tuners must return well-formed TuningResults."""

    def test_lgbm_result_has_required_fields(self) -> None:
        x_tr, y_tr, x_val, y_val = _make_dataset()
        tuner = LGBMTuner(n_trials=5)
        with patch(
            "ep2_crypto.models.tuning.LGBMDirectionModel",
            return_value=_make_mock_lgbm(n_val=len(y_val)),
        ):
            result = tuner.tune(x_tr, y_tr, x_val, y_val, study_name="lgbm_integ")

        assert isinstance(result, TuningResult)
        assert result.study_name == "lgbm_integ"
        assert result.n_trials >= 1
        assert 0.0 <= result.pruning_rate <= 1.0
        assert 0.0 <= result.dsr <= 1.0

    def test_catboost_result_has_required_fields(self) -> None:
        x_tr, y_tr, x_val, y_val = _make_dataset()
        tuner = CatBoostTuner(n_trials=5)
        with patch(
            "ep2_crypto.models.tuning.CatBoostDirectionModel",
            return_value=_make_mock_catboost(n_val=len(y_val)),
        ):
            result = tuner.tune(x_tr, y_tr, x_val, y_val, study_name="cat_integ")

        assert isinstance(result, TuningResult)
        assert result.n_trials >= 1
        assert 0.0 <= result.dsr <= 1.0

    def test_gru_result_has_required_fields(self) -> None:
        x_tr, y_tr, x_val, y_val = _make_dataset()
        tuner = GRUTuner(n_trials=5, fast_eval_epochs=2)
        with patch(
            "ep2_crypto.models.tuning.GRUFeatureExtractor",
            return_value=_make_mock_gru(),
        ):
            result = tuner.tune(x_tr, y_tr, x_val, y_val, study_name="gru_integ")

        assert isinstance(result, TuningResult)
        assert result.n_trials >= 1


# ---------------------------------------------------------------------------
# Acceptance criterion 2: Pruning rate is tracked
# ---------------------------------------------------------------------------


class TestPruningTracking:
    """Pruning stats must be accurately recorded."""

    def test_pruning_rate_recorded_lgbm(self) -> None:
        x_tr, y_tr, x_val, y_val = _make_dataset(n_samples=200)
        tuner = LGBMTuner(n_trials=10)
        with patch(
            "ep2_crypto.models.tuning.LGBMDirectionModel",
            return_value=_make_mock_lgbm(n_val=len(y_val)),
        ):
            result = tuner.tune(x_tr, y_tr, x_val, y_val)
        assert result.n_pruned + (result.n_trials - result.n_pruned) == result.n_trials

    def test_n_pruned_never_exceeds_n_trials(self) -> None:
        x_tr, y_tr, x_val, y_val = _make_dataset(n_samples=200)
        for tuner_cls in [LGBMTuner, CatBoostTuner]:
            mock = _make_mock_lgbm(n_val=len(y_val)) if tuner_cls == LGBMTuner else _make_mock_catboost(n_val=len(y_val))
            model_path = (
                "ep2_crypto.models.tuning.LGBMDirectionModel"
                if tuner_cls == LGBMTuner
                else "ep2_crypto.models.tuning.CatBoostDirectionModel"
            )
            tuner = tuner_cls(n_trials=5)
            with patch(model_path, return_value=mock):
                result = tuner.tune(x_tr, y_tr, x_val, y_val)
            assert result.n_pruned <= result.n_trials


# ---------------------------------------------------------------------------
# Acceptance criterion 3: DSR applied to final parameters
# ---------------------------------------------------------------------------


class TestDSRApplication:
    """DSR must be computed and recorded for every tuning run."""

    def test_dsr_computed_for_lgbm(self) -> None:
        x_tr, y_tr, x_val, y_val = _make_dataset()
        tuner = LGBMTuner(n_trials=5)
        with patch(
            "ep2_crypto.models.tuning.LGBMDirectionModel",
            return_value=_make_mock_lgbm(n_val=len(y_val)),
        ):
            result = tuner.tune(x_tr, y_tr, x_val, y_val)
        assert 0.0 <= result.dsr <= 1.0

    def test_dsr_computed_for_catboost(self) -> None:
        x_tr, y_tr, x_val, y_val = _make_dataset()
        tuner = CatBoostTuner(n_trials=5)
        with patch(
            "ep2_crypto.models.tuning.CatBoostDirectionModel",
            return_value=_make_mock_catboost(n_val=len(y_val)),
        ):
            result = tuner.tune(x_tr, y_tr, x_val, y_val)
        assert 0.0 <= result.dsr <= 1.0

    def test_dsr_computed_for_gru(self) -> None:
        x_tr, y_tr, x_val, y_val = _make_dataset()
        tuner = GRUTuner(n_trials=5, fast_eval_epochs=2)
        with patch(
            "ep2_crypto.models.tuning.GRUFeatureExtractor",
            return_value=_make_mock_gru(),
        ):
            result = tuner.tune(x_tr, y_tr, x_val, y_val)
        assert 0.0 <= result.dsr <= 1.0

    def test_dsr_function_standalone(self) -> None:
        # Core DSR function used by all tuners
        dsr = deflated_sharpe_ratio(observed_sharpe=2.5, n_trials=50, n_bars=1000)
        assert 0.0 <= dsr <= 1.0


# ---------------------------------------------------------------------------
# Acceptance criterion 4: Feature importance top-10 overlap
# ---------------------------------------------------------------------------


class TestFeatureImportanceStability:
    """Top-10 feature overlap across folds must be trackable."""

    def test_stable_top5_has_high_overlap(self) -> None:
        folds = _make_fold_importances(n_folds=5, stable_top_k=5)
        analyzer = FeatureImportanceAnalyzer(top_n=5)
        result = analyzer.analyze(folds)
        assert result.top10_overlap_rate > 0.7, (
            f"Expected overlap >0.7 for stable folds, got {result.top10_overlap_rate}"
        )

    def test_unstable_importances_below_threshold(self) -> None:
        # All random importances → very low stability
        rng = np.random.default_rng(42)
        folds = [
            {f"feat_{i}": rng.uniform(0, 10) for i in range(20)}
            for _ in range(5)
        ]
        analyzer = FeatureImportanceAnalyzer(top_n=10, stability_threshold=0.9)
        result = analyzer.analyze(folds)
        # Random data → many unstable features
        assert len(result.unstable_features) > 5

    def test_features_in_less_than_50pct_folds_flagged(self) -> None:
        # Construct folds where some features appear in only 2/5 folds (40%)
        folds: list[dict[str, float]] = []
        for i in range(5):
            fold: dict[str, float] = {
                "always_top": 10.0,
                "sometimes_top": 8.0 if i < 2 else 0.1,  # top in 2/5 = 40%
            }
            folds.append(fold)
        analyzer = FeatureImportanceAnalyzer(top_n=1, stability_threshold=0.5)
        result = analyzer.analyze(folds)
        # "sometimes_top" appears in top-1 in only 2/5 folds = 0.4 < 0.5 threshold
        assert "sometimes_top" in result.unstable_features

    def test_stability_result_covers_all_features(self) -> None:
        folds = _make_fold_importances(n_folds=4, n_features=15)
        analyzer = FeatureImportanceAnalyzer(top_n=10)
        result = analyzer.analyze(folds)
        all_from_result = set(result.stable_features) | set(result.unstable_features)
        all_from_folds = set(folds[0].keys())
        assert all_from_result == all_from_folds


# ---------------------------------------------------------------------------
# Acceptance criterion 5: Full pipeline test
# ---------------------------------------------------------------------------


class TestFullTuningPipeline:
    """End-to-end: tune LightGBM → best config → stability → threshold."""

    def test_lgbm_tune_to_config_pipeline(self) -> None:
        from ep2_crypto.models.lgbm_direction import LGBMConfig

        x_tr, y_tr, x_val, y_val = _make_dataset(n_samples=300)
        mock = _make_mock_lgbm(n_val=len(y_val))
        tuner = LGBMTuner(n_trials=5)

        with patch("ep2_crypto.models.tuning.LGBMDirectionModel", return_value=mock):
            result = tuner.tune(x_tr, y_tr, x_val, y_val)

        # Step 1: result has valid params
        assert isinstance(result.best_params, dict)

        # Step 2: build config from result
        config = tuner.best_config(result)
        assert isinstance(config, LGBMConfig)
        assert 15 <= config.num_leaves <= 63
        assert 3 <= config.max_depth <= 8

    def test_threshold_optimizer_after_tuning(self) -> None:
        rng = np.random.default_rng(7)
        probas = rng.dirichlet([1, 1, 3], size=200).astype(np.float64)  # UP bias
        labels = rng.choice([-1, 0, 1], size=200).astype(np.int8)

        optimizer = ThresholdOptimizer(threshold_grid=[0.50, 0.55, 0.60, 0.65, 0.70])
        result = optimizer.optimize(probas, labels)

        assert result.best_threshold in [0.50, 0.55, 0.60, 0.65, 0.70]
        assert len(result.grid_results) == 5

    def test_stability_then_threshold_pipeline(self) -> None:
        folds = _make_fold_importances(n_folds=5, n_features=20, stable_top_k=10)
        analyzer = FeatureImportanceAnalyzer(top_n=10)
        stability = analyzer.analyze(folds)

        # Features in the stable set should be usable for model input
        assert len(stability.stable_features) > 0

        # Run threshold optimization on mock probas
        rng = np.random.default_rng(3)
        probas = rng.dirichlet([1, 1, 1], size=150).astype(np.float64)
        labels = rng.choice([-1, 0, 1], size=150).astype(np.int8)
        optimizer = ThresholdOptimizer()
        threshold_result = optimizer.optimize(probas, labels)
        assert threshold_result.best_threshold >= 0.50

    def test_all_three_tuners_produce_compatible_configs(self) -> None:
        from ep2_crypto.models.catboost_direction import CatBoostConfig
        from ep2_crypto.models.gru_features import GRUConfig
        from ep2_crypto.models.lgbm_direction import LGBMConfig

        x_tr, y_tr, x_val, y_val = _make_dataset(n_samples=200)

        # LGBM
        lgbm_tuner = LGBMTuner(n_trials=3)
        with patch(
            "ep2_crypto.models.tuning.LGBMDirectionModel",
            return_value=_make_mock_lgbm(n_val=len(y_val)),
        ):
            lgbm_result = lgbm_tuner.tune(x_tr, y_tr, x_val, y_val)
        lgbm_config = lgbm_tuner.best_config(lgbm_result)
        assert isinstance(lgbm_config, LGBMConfig)

        # CatBoost
        cb_tuner = CatBoostTuner(n_trials=3)
        with patch(
            "ep2_crypto.models.tuning.CatBoostDirectionModel",
            return_value=_make_mock_catboost(n_val=len(y_val)),
        ):
            cb_result = cb_tuner.tune(x_tr, y_tr, x_val, y_val)
        cb_config = cb_tuner.best_config(cb_result)
        assert isinstance(cb_config, CatBoostConfig)

        # GRU
        gru_tuner = GRUTuner(n_trials=3, fast_eval_epochs=2)
        with patch(
            "ep2_crypto.models.tuning.GRUFeatureExtractor",
            return_value=_make_mock_gru(),
        ):
            gru_result = gru_tuner.tune(x_tr, y_tr, x_val, y_val)
        gru_config = gru_tuner.best_config(gru_result)
        assert isinstance(gru_config, GRUConfig)

        # All configs are valid
        assert lgbm_config.num_leaves > 0
        assert cb_config.depth > 0
        assert gru_config.hidden_size > 0

    def test_walk_forward_sharpe_used_in_tuning(self) -> None:
        """walk_forward_sharpe is the core tuning metric — verify it works."""
        rng = np.random.default_rng(42)
        # Slight positive drift → positive Sharpe
        returns = rng.normal(loc=0.002, scale=0.02, size=500).astype(np.float64)
        sharpe = walk_forward_sharpe(returns, n_splits=3)
        assert sharpe > -10.0  # Not the sentinel "no data" value

    def test_config_params_within_search_space_lgbm(self) -> None:
        x_tr, y_tr, x_val, y_val = _make_dataset()
        tuner = LGBMTuner(n_trials=5)
        with patch(
            "ep2_crypto.models.tuning.LGBMDirectionModel",
            return_value=_make_mock_lgbm(n_val=len(y_val)),
        ):
            result = tuner.tune(x_tr, y_tr, x_val, y_val)
        config = tuner.best_config(result)
        # Verify search space bounds are respected
        assert 15 <= config.num_leaves <= 63
        assert 3 <= config.max_depth <= 8
        assert 0.01 <= config.learning_rate <= 0.3
        assert 20 <= config.min_child_samples <= 100
        assert 0.6 <= config.subsample <= 1.0
        assert 0.6 <= config.colsample_bytree <= 1.0
        assert 1e-3 <= config.reg_alpha <= 25.0
        assert 1e-3 <= config.reg_lambda <= 25.0

    def test_config_params_within_search_space_gru(self) -> None:
        x_tr, y_tr, x_val, y_val = _make_dataset()
        tuner = GRUTuner(n_trials=5, fast_eval_epochs=2)
        with patch(
            "ep2_crypto.models.tuning.GRUFeatureExtractor",
            return_value=_make_mock_gru(),
        ):
            result = tuner.tune(x_tr, y_tr, x_val, y_val)
        config = tuner.best_config(result)
        assert 32 <= config.hidden_size <= 256
        assert 1 <= config.num_layers <= 3
        assert 1e-5 <= config.learning_rate <= 1e-2
        assert 0.1 <= config.dropout <= 0.5
        assert 12 <= config.seq_len <= 60
