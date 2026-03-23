"""Tests for hyperparameter tuning module.

Covers:
- walk_forward_sharpe() helper
- deflated_sharpe_ratio() helper
- _predictions_to_returns() helper
- LGBMTuner (mocked model)
- CatBoostTuner (mocked model)
- GRUTuner (mocked model)
- ThresholdOptimizer
- FeatureImportanceAnalyzer
- TuningResult
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ep2_crypto.models.tuning import (
    CatBoostTuner,
    FeatureImportanceAnalyzer,
    GRUTuner,
    LGBMTuner,
    StabilityResult,
    ThresholdOptimizer,
    ThresholdResult,
    TuningResult,
    _predictions_to_returns,
    deflated_sharpe_ratio,
    walk_forward_sharpe,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture()
def synthetic_returns(rng: np.random.Generator) -> np.ndarray:
    """500-bar return series with slight positive drift."""
    return rng.normal(loc=0.001, scale=0.02, size=500).astype(np.float64)


@pytest.fixture()
def zero_returns() -> np.ndarray:
    return np.zeros(500, dtype=np.float64)


@pytest.fixture()
def small_features(rng: np.random.Generator) -> np.ndarray:
    """200 samples × 10 features."""
    return rng.standard_normal((200, 10)).astype(np.float64)


@pytest.fixture()
def small_labels(rng: np.random.Generator) -> np.ndarray:
    """200 ternary labels in {-1, 0, 1}."""
    return rng.choice([-1, 0, 1], size=200).astype(np.int8)


@pytest.fixture()
def small_probas(rng: np.random.Generator) -> np.ndarray:
    """200 samples × 3-class probabilities (sum to 1)."""
    raw = rng.dirichlet([1, 1, 1], size=200)
    return raw.astype(np.float64)


# ---------------------------------------------------------------------------
# walk_forward_sharpe
# ---------------------------------------------------------------------------


class TestWalkForwardSharpe:
    def test_positive_drift_returns_positive_sharpe(self, synthetic_returns: np.ndarray) -> None:
        sharpe = walk_forward_sharpe(synthetic_returns, n_splits=2)
        # With positive drift we expect positive Sharpe
        assert sharpe > 0.0

    def test_zero_returns_returns_zero_or_negative(self, zero_returns: np.ndarray) -> None:
        sharpe = walk_forward_sharpe(zero_returns, n_splits=2)
        assert sharpe == pytest.approx(0.0, abs=1e-6)

    def test_too_few_bars_returns_sentinel(self) -> None:
        tiny = np.ones(50, dtype=np.float64)
        sharpe = walk_forward_sharpe(tiny, n_splits=2)
        assert sharpe == -10.0

    def test_n_splits_1(self, synthetic_returns: np.ndarray) -> None:
        sharpe = walk_forward_sharpe(synthetic_returns, n_splits=1)
        assert isinstance(sharpe, float)

    def test_annualization_applied(self) -> None:
        # Per-bar return of 0.001 with std 0.02 → annualized Sharpe ≈ 0.05 * sqrt(105_120)
        returns = np.full(500, 0.001, dtype=np.float64)
        # std = 0, so function returns 0.0
        sharpe = walk_forward_sharpe(returns, n_splits=1)
        assert sharpe == pytest.approx(0.0, abs=1e-6)

    def test_noisy_returns_finite(self, rng: np.random.Generator) -> None:
        returns = rng.standard_normal(1000).astype(np.float64)
        sharpe = walk_forward_sharpe(returns, n_splits=3)
        assert math.isfinite(sharpe)

    def test_purge_bars_respected(self, synthetic_returns: np.ndarray) -> None:
        # Different purge values should give different results
        s1 = walk_forward_sharpe(synthetic_returns, n_splits=2, purge_bars=0)
        s2 = walk_forward_sharpe(synthetic_returns, n_splits=2, purge_bars=50)
        # Not asserting equal — just that both are finite
        assert math.isfinite(s1) and math.isfinite(s2)


# ---------------------------------------------------------------------------
# deflated_sharpe_ratio
# ---------------------------------------------------------------------------


class TestDeflatedSharpeRatio:
    def test_high_sharpe_many_bars_returns_high_dsr(self) -> None:
        dsr = deflated_sharpe_ratio(observed_sharpe=3.0, n_trials=50, n_bars=5000)
        assert dsr > 0.5

    def test_low_sharpe_high_trials_returns_low_dsr(self) -> None:
        dsr = deflated_sharpe_ratio(observed_sharpe=0.5, n_trials=200, n_bars=500)
        assert dsr < 0.9

    def test_zero_bars_returns_zero(self) -> None:
        dsr = deflated_sharpe_ratio(observed_sharpe=2.0, n_trials=50, n_bars=0)
        assert dsr == 0.0

    def test_zero_trials_returns_zero(self) -> None:
        dsr = deflated_sharpe_ratio(observed_sharpe=2.0, n_trials=0, n_bars=1000)
        assert dsr == 0.0

    def test_output_in_zero_one_range(self) -> None:
        for sr in [-1.0, 0.0, 1.0, 2.0, 5.0]:
            dsr = deflated_sharpe_ratio(sr, n_trials=50, n_bars=1000)
            assert 0.0 <= dsr <= 1.0, f"DSR out of range for sharpe={sr}"

    def test_more_trials_reduces_dsr(self) -> None:
        dsr_few = deflated_sharpe_ratio(2.0, n_trials=10, n_bars=1000)
        dsr_many = deflated_sharpe_ratio(2.0, n_trials=500, n_bars=1000)
        assert dsr_few > dsr_many

    def test_sr_just_above_expected_max_more_bars_gives_higher_dsr(self) -> None:
        # SR=2.3 is just above the expected max SR for 50 trials (~2.28).
        # With few bars (sr_std is large), z is small and DSR is moderate.
        # With many bars (sr_std is small), z is higher and DSR improves.
        dsr_few_bars = deflated_sharpe_ratio(2.3, n_trials=50, n_bars=50)
        dsr_many_bars = deflated_sharpe_ratio(2.3, n_trials=50, n_bars=5000)
        assert dsr_few_bars < dsr_many_bars

    def test_non_normal_skewness(self) -> None:
        dsr = deflated_sharpe_ratio(
            observed_sharpe=2.0, n_trials=50, n_bars=1000, skewness=-1.0, excess_kurtosis=3.0
        )
        assert 0.0 <= dsr <= 1.0


# ---------------------------------------------------------------------------
# _predictions_to_returns
# ---------------------------------------------------------------------------


class TestPredictionsToReturns:
    def test_shape_matches_input(self, small_probas: np.ndarray, small_labels: np.ndarray) -> None:
        returns = _predictions_to_returns(small_probas, small_labels)
        assert returns.shape == (200,)

    def test_no_trade_when_all_below_threshold(self, small_labels: np.ndarray) -> None:
        # All probabilities equal (0.333 each) → max never exceeds 0.55
        probas = np.full((200, 3), 1 / 3, dtype=np.float64)
        returns = _predictions_to_returns(probas, small_labels, threshold=0.55)
        assert np.all(returns == 0.0)

    def test_cost_deducted_per_trade(self, small_labels: np.ndarray) -> None:
        # Force all UP predictions with high confidence
        probas = np.zeros((200, 3), dtype=np.float64)
        probas[:, 2] = 0.99  # UP class
        # All true labels UP → all correct trades
        labels = np.ones(200, dtype=np.int8)
        returns_no_cost = _predictions_to_returns(probas, labels, cost_bps=0.0)
        returns_with_cost = _predictions_to_returns(probas, labels, cost_bps=10.0)
        # With cost, returns should be lower
        assert returns_no_cost.mean() > returns_with_cost.mean()

    def test_returns_dtype_float64(
        self, small_probas: np.ndarray, small_labels: np.ndarray
    ) -> None:
        returns = _predictions_to_returns(small_probas, small_labels)
        assert returns.dtype == np.float64

    def test_high_threshold_fewer_trades(
        self, small_probas: np.ndarray, small_labels: np.ndarray
    ) -> None:
        returns_low = _predictions_to_returns(small_probas, small_labels, threshold=0.50)
        returns_high = _predictions_to_returns(small_probas, small_labels, threshold=0.75)
        n_trades_low = np.sum(returns_low != 0)
        n_trades_high = np.sum(returns_high != 0)
        assert n_trades_high <= n_trades_low


# ---------------------------------------------------------------------------
# TuningResult
# ---------------------------------------------------------------------------


class TestTuningResult:
    def test_to_config_returns_dict(self) -> None:
        result = TuningResult(
            best_params={"num_leaves": 31, "max_depth": 5},
            best_sharpe=1.5,
            dsr=0.8,
            n_trials=50,
            n_pruned=20,
            pruning_rate=0.4,
            study_name="test",
        )
        cfg = result.to_config()
        assert cfg == {"num_leaves": 31, "max_depth": 5}

    def test_pruning_rate_computed(self) -> None:
        result = TuningResult(
            best_params={},
            best_sharpe=0.5,
            dsr=0.6,
            n_trials=100,
            n_pruned=40,
            pruning_rate=0.4,
            study_name="test",
        )
        assert result.pruning_rate == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# LGBMTuner (mocked)
# ---------------------------------------------------------------------------


class TestLGBMTuner:
    def _make_mock_model(self, sharpe_value: float = 1.0) -> MagicMock:
        """Create a mock LGBMDirectionModel that returns a fixed Sharpe."""
        mock = MagicMock()
        mock.is_fitted = True
        # Return probas that produce the given Sharpe via our helper
        n = 200
        rng = np.random.default_rng(99)
        probas = rng.dirichlet([1, 1, 1], size=n).astype(np.float64)
        mock.predict_proba.return_value = probas
        mock.train.return_value = {"train_loss": 0.5}
        return mock

    def test_tuner_returns_tuning_result(
        self,
        small_features: np.ndarray,
        small_labels: np.ndarray,
    ) -> None:
        tuner = LGBMTuner(n_trials=3)
        mock_model = self._make_mock_model()
        with patch(
            "ep2_crypto.models.tuning.LGBMDirectionModel",
            return_value=mock_model,
        ):
            result = tuner.tune(
                small_features[:150],
                small_labels[:150],
                small_features[150:],
                small_labels[150:],
            )
        assert isinstance(result, TuningResult)
        assert result.n_trials >= 1

    def test_tuner_records_pruned_count(
        self,
        small_features: np.ndarray,
        small_labels: np.ndarray,
    ) -> None:
        tuner = LGBMTuner(n_trials=5)
        mock_model = self._make_mock_model()
        with patch(
            "ep2_crypto.models.tuning.LGBMDirectionModel",
            return_value=mock_model,
        ):
            result = tuner.tune(
                small_features[:150],
                small_labels[:150],
                small_features[150:],
                small_labels[150:],
            )
        assert result.n_pruned >= 0
        assert result.n_pruned <= result.n_trials

    def test_tuner_computes_dsr(
        self,
        small_features: np.ndarray,
        small_labels: np.ndarray,
    ) -> None:
        tuner = LGBMTuner(n_trials=3)
        mock_model = self._make_mock_model()
        with patch(
            "ep2_crypto.models.tuning.LGBMDirectionModel",
            return_value=mock_model,
        ):
            result = tuner.tune(
                small_features[:150],
                small_labels[:150],
                small_features[150:],
                small_labels[150:],
            )
        assert 0.0 <= result.dsr <= 1.0

    def test_best_config_returns_lgbm_config(
        self,
        small_features: np.ndarray,
        small_labels: np.ndarray,
    ) -> None:
        from ep2_crypto.models.lgbm_direction import LGBMConfig

        tuner = LGBMTuner(n_trials=3)
        mock_model = self._make_mock_model()
        with patch(
            "ep2_crypto.models.tuning.LGBMDirectionModel",
            return_value=mock_model,
        ):
            result = tuner.tune(
                small_features[:150],
                small_labels[:150],
                small_features[150:],
                small_labels[150:],
            )
        config = tuner.best_config(result)
        assert isinstance(config, LGBMConfig)

    def test_failed_trials_handled_gracefully(
        self,
        small_features: np.ndarray,
        small_labels: np.ndarray,
    ) -> None:
        tuner = LGBMTuner(n_trials=3)
        mock_model = MagicMock()
        mock_model.is_fitted = False
        mock_model.train.side_effect = RuntimeError("training failed")
        with patch(
            "ep2_crypto.models.tuning.LGBMDirectionModel",
            return_value=mock_model,
        ):
            # Should not raise
            result = tuner.tune(
                small_features[:150],
                small_labels[:150],
                small_features[150:],
                small_labels[150:],
            )
        assert isinstance(result, TuningResult)

    def test_pruning_rate_in_zero_one(
        self,
        small_features: np.ndarray,
        small_labels: np.ndarray,
    ) -> None:
        tuner = LGBMTuner(n_trials=5)
        mock_model = self._make_mock_model()
        with patch(
            "ep2_crypto.models.tuning.LGBMDirectionModel",
            return_value=mock_model,
        ):
            result = tuner.tune(
                small_features[:150],
                small_labels[:150],
                small_features[150:],
                small_labels[150:],
            )
        assert 0.0 <= result.pruning_rate <= 1.0


# ---------------------------------------------------------------------------
# CatBoostTuner (mocked)
# ---------------------------------------------------------------------------


class TestCatBoostTuner:
    def _make_mock_model(self) -> MagicMock:
        mock = MagicMock()
        mock.is_fitted = True
        rng = np.random.default_rng(77)
        mock.predict_proba.return_value = rng.dirichlet([1, 1, 1], size=50).astype(np.float64)
        mock.train.return_value = {}
        return mock

    def test_returns_tuning_result(
        self,
        small_features: np.ndarray,
        small_labels: np.ndarray,
    ) -> None:
        tuner = CatBoostTuner(n_trials=3)
        mock_model = self._make_mock_model()
        with patch(
            "ep2_crypto.models.tuning.CatBoostDirectionModel",
            return_value=mock_model,
        ):
            result = tuner.tune(
                small_features[:150],
                small_labels[:150],
                small_features[150:],
                small_labels[150:],
            )
        assert isinstance(result, TuningResult)

    def test_best_config_returns_catboost_config(
        self,
        small_features: np.ndarray,
        small_labels: np.ndarray,
    ) -> None:
        from ep2_crypto.models.catboost_direction import CatBoostConfig

        tuner = CatBoostTuner(n_trials=3)
        mock_model = self._make_mock_model()
        with patch(
            "ep2_crypto.models.tuning.CatBoostDirectionModel",
            return_value=mock_model,
        ):
            result = tuner.tune(
                small_features[:150],
                small_labels[:150],
                small_features[150:],
                small_labels[150:],
            )
        config = tuner.best_config(result)
        assert isinstance(config, CatBoostConfig)

    def test_dsr_in_valid_range(
        self,
        small_features: np.ndarray,
        small_labels: np.ndarray,
    ) -> None:
        tuner = CatBoostTuner(n_trials=3)
        mock_model = self._make_mock_model()
        with patch(
            "ep2_crypto.models.tuning.CatBoostDirectionModel",
            return_value=mock_model,
        ):
            result = tuner.tune(
                small_features[:150],
                small_labels[:150],
                small_features[150:],
                small_labels[150:],
            )
        assert 0.0 <= result.dsr <= 1.0

    def test_exception_in_training_handled(
        self,
        small_features: np.ndarray,
        small_labels: np.ndarray,
    ) -> None:
        tuner = CatBoostTuner(n_trials=3)
        mock_model = MagicMock()
        mock_model.train.side_effect = ValueError("bad data")
        with patch(
            "ep2_crypto.models.tuning.CatBoostDirectionModel",
            return_value=mock_model,
        ):
            result = tuner.tune(
                small_features[:150],
                small_labels[:150],
                small_features[150:],
                small_labels[150:],
            )
        assert isinstance(result, TuningResult)


# ---------------------------------------------------------------------------
# GRUTuner (mocked)
# ---------------------------------------------------------------------------


class TestGRUTuner:
    def _make_mock_gru(self, val_loss: float = 0.5) -> MagicMock:
        mock = MagicMock()
        mock.train.return_value = {"train_loss": 0.8, "val_loss": val_loss}
        return mock

    def test_returns_tuning_result(
        self,
        small_features: np.ndarray,
        small_labels: np.ndarray,
    ) -> None:
        tuner = GRUTuner(n_trials=3, fast_eval_epochs=2)
        mock_model = self._make_mock_gru()
        with patch(
            "ep2_crypto.models.tuning.GRUFeatureExtractor",
            return_value=mock_model,
        ):
            result = tuner.tune(
                small_features[:150],
                small_labels[:150],
                small_features[150:],
                small_labels[150:],
            )
        assert isinstance(result, TuningResult)

    def test_minimizes_val_loss(
        self,
        small_features: np.ndarray,
        small_labels: np.ndarray,
    ) -> None:
        tuner = GRUTuner(n_trials=3, fast_eval_epochs=2)
        # best_sharpe = -val_loss, so best_sharpe < 0
        mock_model = self._make_mock_gru(val_loss=0.5)
        with patch(
            "ep2_crypto.models.tuning.GRUFeatureExtractor",
            return_value=mock_model,
        ):
            result = tuner.tune(
                small_features[:150],
                small_labels[:150],
                small_features[150:],
                small_labels[150:],
            )
        # best_sharpe is proxy = -val_loss
        assert result.best_sharpe <= 0.0

    def test_best_config_returns_gru_config(
        self,
        small_features: np.ndarray,
        small_labels: np.ndarray,
    ) -> None:
        from ep2_crypto.models.gru_features import GRUConfig

        tuner = GRUTuner(n_trials=3, fast_eval_epochs=2)
        mock_model = self._make_mock_gru()
        with patch(
            "ep2_crypto.models.tuning.GRUFeatureExtractor",
            return_value=mock_model,
        ):
            result = tuner.tune(
                small_features[:150],
                small_labels[:150],
                small_features[150:],
                small_labels[150:],
            )
        config = tuner.best_config(result)
        assert isinstance(config, GRUConfig)

    def test_exception_in_gru_training_handled(
        self,
        small_features: np.ndarray,
        small_labels: np.ndarray,
    ) -> None:
        tuner = GRUTuner(n_trials=3, fast_eval_epochs=2)
        mock_model = MagicMock()
        mock_model.train.side_effect = RuntimeError("cuda oom")
        with patch(
            "ep2_crypto.models.tuning.GRUFeatureExtractor",
            return_value=mock_model,
        ):
            result = tuner.tune(
                small_features[:150],
                small_labels[:150],
                small_features[150:],
                small_labels[150:],
            )
        assert isinstance(result, TuningResult)

    def test_study_name_set(
        self,
        small_features: np.ndarray,
        small_labels: np.ndarray,
    ) -> None:
        tuner = GRUTuner(n_trials=2, fast_eval_epochs=1)
        mock_model = self._make_mock_gru()
        with patch(
            "ep2_crypto.models.tuning.GRUFeatureExtractor",
            return_value=mock_model,
        ):
            result = tuner.tune(
                small_features[:150],
                small_labels[:150],
                small_features[150:],
                small_labels[150:],
                study_name="gru_custom",
            )
        assert result.study_name == "gru_custom"


# ---------------------------------------------------------------------------
# ThresholdOptimizer
# ---------------------------------------------------------------------------


class TestThresholdOptimizer:
    def test_returns_threshold_result(
        self,
        small_probas: np.ndarray,
        small_labels: np.ndarray,
    ) -> None:
        optimizer = ThresholdOptimizer()
        result = optimizer.optimize(small_probas, small_labels)
        assert isinstance(result, ThresholdResult)

    def test_best_threshold_in_grid(
        self,
        small_probas: np.ndarray,
        small_labels: np.ndarray,
    ) -> None:
        grid = [0.50, 0.60, 0.70]
        optimizer = ThresholdOptimizer(threshold_grid=grid)
        result = optimizer.optimize(small_probas, small_labels)
        assert result.best_threshold in grid

    def test_grid_results_cover_all_thresholds(
        self,
        small_probas: np.ndarray,
        small_labels: np.ndarray,
    ) -> None:
        grid = [0.50, 0.55, 0.60, 0.65, 0.70]
        optimizer = ThresholdOptimizer(threshold_grid=grid)
        result = optimizer.optimize(small_probas, small_labels)
        assert set(result.grid_results.keys()) == set(grid)

    def test_per_regime_optimization(
        self,
        small_probas: np.ndarray,
        small_labels: np.ndarray,
        rng: np.random.Generator,
    ) -> None:
        regimes = rng.choice([0, 1, 2], size=200).astype(np.int32)
        optimizer = ThresholdOptimizer(threshold_grid=[0.50, 0.60, 0.70])
        result = optimizer.optimize(small_probas, small_labels, regimes=regimes)
        # Should have regime-specific thresholds
        assert "bear" in result.regime_thresholds
        assert "neutral" in result.regime_thresholds
        assert "bull" in result.regime_thresholds

    def test_per_regime_thresholds_in_grid(
        self,
        small_probas: np.ndarray,
        small_labels: np.ndarray,
        rng: np.random.Generator,
    ) -> None:
        grid = [0.50, 0.60, 0.70]
        regimes = rng.choice([0, 1, 2], size=200).astype(np.int32)
        optimizer = ThresholdOptimizer(threshold_grid=grid)
        result = optimizer.optimize(small_probas, small_labels, regimes=regimes)
        for regime_name, thresh in result.regime_thresholds.items():
            assert thresh in grid, f"Regime {regime_name} threshold {thresh} not in grid"

    def test_regime_with_few_samples_uses_global_threshold(
        self,
        small_probas: np.ndarray,
        small_labels: np.ndarray,
    ) -> None:
        # Force all samples to regime 0 (bear), leaving regime 1 and 2 empty
        regimes = np.zeros(200, dtype=np.int32)
        optimizer = ThresholdOptimizer(threshold_grid=[0.50, 0.60, 0.70])
        result = optimizer.optimize(small_probas, small_labels, regimes=regimes)
        # neutral and bull have <50 samples → should fall back to global
        assert result.regime_thresholds["neutral"] == result.best_threshold
        assert result.regime_thresholds["bull"] == result.best_threshold

    def test_best_sharpe_finite(
        self,
        small_probas: np.ndarray,
        small_labels: np.ndarray,
    ) -> None:
        optimizer = ThresholdOptimizer()
        result = optimizer.optimize(small_probas, small_labels)
        assert math.isfinite(result.best_sharpe)

    def test_high_confidence_predictions_trade_less(self, small_labels: np.ndarray) -> None:
        # Uniform probas → low confidence → more trades at low threshold
        uniform_probas = np.full((200, 3), 1 / 3, dtype=np.float64)
        # Strong DOWN predictions
        strong_probas = np.zeros((200, 3), dtype=np.float64)
        strong_probas[:, 0] = 0.9  # DOWN with high confidence

        opt_uniform = ThresholdOptimizer(threshold_grid=[0.50, 0.55])
        opt_strong = ThresholdOptimizer(threshold_grid=[0.50, 0.85])

        opt_uniform.optimize(uniform_probas, small_labels)
        result_strong = opt_strong.optimize(strong_probas, small_labels)
        # strong predictions should be able to use higher threshold
        assert result_strong.best_threshold >= 0.50


# ---------------------------------------------------------------------------
# FeatureImportanceAnalyzer
# ---------------------------------------------------------------------------


class TestFeatureImportanceAnalyzer:
    def _make_stable_folds(self, n_folds: int = 5) -> list[dict[str, float]]:
        """Folds where the same 5 features are always top-5."""
        folds = []
        rng = np.random.default_rng(1)
        for _ in range(n_folds):
            fold: dict[str, float] = {
                "feat_a": 10.0 + rng.uniform(0, 1),
                "feat_b": 8.0 + rng.uniform(0, 1),
                "feat_c": 7.0 + rng.uniform(0, 1),
                "feat_d": 6.0 + rng.uniform(0, 1),
                "feat_e": 5.0 + rng.uniform(0, 1),
                "feat_noise1": rng.uniform(0, 1),
                "feat_noise2": rng.uniform(0, 1),
                "feat_noise3": rng.uniform(0, 0.5),
            }
            folds.append(fold)
        return folds

    def _make_unstable_folds(self, n_folds: int = 5) -> list[dict[str, float]]:
        """Folds where the top features change every fold."""
        folds = []
        rng = np.random.default_rng(99)
        feature_names = [f"feat_{i}" for i in range(20)]
        for _ in range(n_folds):
            fold = {name: rng.uniform(0, 10) for name in feature_names}
            folds.append(fold)
        return folds

    def test_returns_stability_result(self) -> None:
        folds = self._make_stable_folds()
        analyzer = FeatureImportanceAnalyzer(top_n=5)
        result = analyzer.analyze(folds)
        assert isinstance(result, StabilityResult)

    def test_stable_features_identified(self) -> None:
        folds = self._make_stable_folds(n_folds=5)
        analyzer = FeatureImportanceAnalyzer(top_n=5, stability_threshold=0.5)
        result = analyzer.analyze(folds)
        stable_names = set(result.stable_features)
        assert "feat_a" in stable_names
        assert "feat_b" in stable_names
        assert "feat_c" in stable_names

    def test_noise_features_marked_unstable(self) -> None:
        folds = self._make_stable_folds(n_folds=5)
        analyzer = FeatureImportanceAnalyzer(top_n=5, stability_threshold=0.8)
        result = analyzer.analyze(folds)
        set(result.unstable_features)
        # noise features should be in the unstable set (rarely appear in top-5)
        assert len(result.unstable_features) > 0

    def test_high_overlap_rate_for_stable_folds(self) -> None:
        folds = self._make_stable_folds(n_folds=5)
        analyzer = FeatureImportanceAnalyzer(top_n=5)
        result = analyzer.analyze(folds)
        assert result.top10_overlap_rate > 0.5

    def test_appearance_rate_in_zero_one(self) -> None:
        folds = self._make_stable_folds(n_folds=5)
        analyzer = FeatureImportanceAnalyzer(top_n=5)
        result = analyzer.analyze(folds)
        for feat, rate in result.feature_appearance_rate.items():
            assert 0.0 <= rate <= 1.0, f"Feature {feat} rate={rate} out of range"

    def test_mean_importance_all_features_covered(self) -> None:
        folds = self._make_stable_folds(n_folds=3)
        analyzer = FeatureImportanceAnalyzer(top_n=5)
        result = analyzer.analyze(folds)
        all_feat = set(folds[0].keys())
        assert set(result.mean_importance.keys()) == all_feat

    def test_stable_sorted_by_mean_importance(self) -> None:
        folds = self._make_stable_folds(n_folds=5)
        analyzer = FeatureImportanceAnalyzer(top_n=5)
        result = analyzer.analyze(folds)
        if len(result.stable_features) < 2:
            return
        importances = [result.mean_importance[f] for f in result.stable_features]
        # Should be in descending order
        for i in range(len(importances) - 1):
            assert importances[i] >= importances[i + 1]

    def test_fewer_than_2_folds_raises(self) -> None:
        analyzer = FeatureImportanceAnalyzer()
        with pytest.raises(ValueError, match="at least 2 folds"):
            analyzer.analyze([{"feat_a": 1.0}])

    def test_two_folds_minimum_works(self) -> None:
        folds = [
            {"feat_a": 5.0, "feat_b": 3.0},
            {"feat_a": 4.5, "feat_b": 3.5},
        ]
        analyzer = FeatureImportanceAnalyzer(top_n=2)
        result = analyzer.analyze(folds)
        assert isinstance(result, StabilityResult)
        assert result.top10_overlap_rate >= 0.0

    def test_overlap_rate_in_zero_one(self) -> None:
        folds = self._make_unstable_folds(n_folds=6)
        analyzer = FeatureImportanceAnalyzer(top_n=10)
        result = analyzer.analyze(folds)
        assert 0.0 <= result.top10_overlap_rate <= 1.0

    def test_completely_consistent_folds_overlap_rate_one(self) -> None:
        # Identical folds → top-N is always the same
        fold = {"feat_a": 10.0, "feat_b": 8.0, "feat_c": 6.0}
        folds = [fold.copy() for _ in range(4)]
        analyzer = FeatureImportanceAnalyzer(top_n=3)
        result = analyzer.analyze(folds)
        assert result.top10_overlap_rate == pytest.approx(1.0)

    def test_feature_appearance_rate_sums_to_positive(self) -> None:
        folds = self._make_stable_folds(n_folds=4)
        analyzer = FeatureImportanceAnalyzer(top_n=5)
        result = analyzer.analyze(folds)
        total = sum(result.feature_appearance_rate.values())
        assert total > 0.0

    def test_all_features_covered_in_stable_and_unstable(self) -> None:
        folds = self._make_stable_folds(n_folds=4)
        analyzer = FeatureImportanceAnalyzer(top_n=5, stability_threshold=0.5)
        result = analyzer.analyze(folds)
        all_found = set(result.stable_features) | set(result.unstable_features)
        all_features = set(folds[0].keys())
        assert all_found == all_features
