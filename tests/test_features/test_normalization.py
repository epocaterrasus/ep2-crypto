"""Tests for normalization pipeline: raw, robust, rank-gaussian, dual pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from ep2_crypto.features.normalization import (
    DualNormalizationPipeline,
    RankGaussianTransformer,
    RawPassthrough,
    RobustScaler,
)


def _make_features(n: int = 200, n_features: int = 5) -> np.ndarray:
    """Create synthetic feature matrix with varying scales."""
    rng = np.random.default_rng(42)
    data = np.zeros((n, n_features))
    data[:, 0] = rng.standard_normal(n) * 100 + 50000  # BTC-scale
    data[:, 1] = rng.standard_normal(n) * 0.01         # Small-scale
    data[:, 2] = rng.uniform(0, 1, n)                   # Uniform [0,1]
    data[:, 3] = rng.exponential(5.0, n)                 # Skewed
    data[:, 4] = rng.standard_normal(n)                  # Standard normal
    return data


# ---- RawPassthrough Tests ----


class TestRawPassthrough:
    def test_transform_unchanged(self) -> None:
        raw = RawPassthrough()
        data = _make_features()
        result = raw.transform(data)
        np.testing.assert_array_equal(result, data)

    def test_fit_returns_self(self) -> None:
        raw = RawPassthrough()
        data = _make_features()
        assert raw.fit(data) is raw

    def test_fit_transform(self) -> None:
        raw = RawPassthrough()
        data = _make_features()
        result = raw.fit_transform(data)
        np.testing.assert_array_equal(result, data)

    def test_returns_copy(self) -> None:
        raw = RawPassthrough()
        data = _make_features()
        result = raw.transform(data)
        result[0, 0] = -999
        assert data[0, 0] != -999


# ---- RobustScaler Tests ----


class TestRobustScaler:
    def test_not_fitted_raises(self) -> None:
        scaler = RobustScaler()
        data = _make_features()
        with pytest.raises(RuntimeError, match="not fitted"):
            scaler.transform(data)

    def test_fit_marks_fitted(self) -> None:
        scaler = RobustScaler()
        assert not scaler.is_fitted
        scaler.fit(_make_features())
        assert scaler.is_fitted

    def test_median_centered(self) -> None:
        """After scaling, median of training data should be ~0."""
        scaler = RobustScaler()
        data = _make_features(1000)
        result = scaler.fit_transform(data)
        for j in range(data.shape[1]):
            assert abs(np.median(result[:, j])) < 0.01

    def test_iqr_normalized(self) -> None:
        """After scaling, IQR of training data should be ~1."""
        scaler = RobustScaler()
        data = _make_features(1000)
        result = scaler.fit_transform(data)
        for j in range(data.shape[1]):
            q75 = np.percentile(result[:, j], 75)
            q25 = np.percentile(result[:, j], 25)
            iqr = q75 - q25
            assert abs(iqr - 1.0) < 0.01

    def test_per_fold_fitting(self) -> None:
        """Fitting on different data should give different results."""
        data1 = _make_features(100)
        data2 = data1 * 2 + 1000  # Different scale and offset

        s1 = RobustScaler().fit(data1)
        s2 = RobustScaler().fit(data2)

        test = _make_features(10)
        r1 = s1.transform(test)
        r2 = s2.transform(test)
        assert not np.allclose(r1, r2)

    def test_1d_input(self) -> None:
        scaler = RobustScaler()
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = scaler.fit_transform(data)
        assert result.shape == (5, 1)
        assert abs(np.median(result)) < 0.01

    def test_zero_iqr_handled(self) -> None:
        """Constant features should not cause division by zero."""
        scaler = RobustScaler()
        data = np.ones((100, 2))
        result = scaler.fit_transform(data)
        assert np.all(np.isfinite(result))


# ---- RankGaussianTransformer Tests ----


class TestRankGaussian:
    def test_not_fitted_raises(self) -> None:
        rg = RankGaussianTransformer()
        with pytest.raises(RuntimeError, match="not fitted"):
            rg.transform(_make_features())

    def test_fit_marks_fitted(self) -> None:
        rg = RankGaussianTransformer()
        assert not rg.is_fitted
        rg.fit(_make_features())
        assert rg.is_fitted

    def test_output_approximately_normal(self) -> None:
        """Transformed data should be approximately standard normal."""
        rg = RankGaussianTransformer()
        data = _make_features(500)
        result = rg.fit_transform(data)
        for j in range(data.shape[1]):
            col = result[:, j]
            assert abs(np.mean(col)) < 0.2, f"Mean of col {j}: {np.mean(col)}"
            assert abs(np.std(col) - 1.0) < 0.3, f"Std of col {j}: {np.std(col)}"

    def test_bounded_output(self) -> None:
        """Output should not have extreme values (clamped probit)."""
        rg = RankGaussianTransformer()
        data = _make_features(200)
        result = rg.fit_transform(data)
        assert np.all(np.abs(result) < 4.0)  # Well within 4 sigma

    def test_skewed_data_normalized(self) -> None:
        """Exponentially distributed data should become approximately normal."""
        rg = RankGaussianTransformer()
        rng = np.random.default_rng(42)
        data = rng.exponential(5.0, (500, 1))
        result = rg.fit_transform(data)
        # Skewness should be reduced
        original_skew = float(np.mean(((data - np.mean(data)) / np.std(data)) ** 3))
        transformed_skew = float(np.mean(((result - np.mean(result)) / np.std(result)) ** 3))
        assert abs(transformed_skew) < abs(original_skew)

    def test_1d_input(self) -> None:
        rg = RankGaussianTransformer()
        data = np.arange(100, dtype=np.float64)
        result = rg.fit_transform(data)
        assert result.shape == (100, 1)

    def test_per_fold_independent(self) -> None:
        """Different training data should produce different transforms."""
        data1 = _make_features(100)
        data2 = _make_features(100) * 5

        rg1 = RankGaussianTransformer().fit(data1)
        rg2 = RankGaussianTransformer().fit(data2)

        test = _make_features(10)
        r1 = rg1.transform(test)
        r2 = rg2.transform(test)
        assert not np.allclose(r1, r2)


# ---- DualNormalizationPipeline Tests ----


class TestDualPipeline:
    def test_not_fitted_raises(self) -> None:
        pipeline = DualNormalizationPipeline()
        with pytest.raises(RuntimeError, match="not fitted"):
            pipeline.transform_neural(_make_features())

    def test_tree_unchanged(self) -> None:
        """Tree path should return data unchanged."""
        pipeline = DualNormalizationPipeline()
        data = _make_features()
        pipeline.fit(data)
        result = pipeline.transform_tree(data)
        np.testing.assert_array_equal(result, data)

    def test_neural_transformed(self) -> None:
        """Neural path should transform data (not unchanged)."""
        pipeline = DualNormalizationPipeline()
        data = _make_features()
        pipeline.fit(data)
        result = pipeline.transform_neural(data)
        assert not np.allclose(result, data)

    def test_neural_approximately_normal(self) -> None:
        """Neural path output should be approximately standard normal."""
        pipeline = DualNormalizationPipeline()
        data = _make_features(500)
        pipeline.fit(data)
        result = pipeline.transform_neural(data)
        for j in range(data.shape[1]):
            col = result[:, j]
            assert abs(np.mean(col)) < 0.2
            assert abs(np.std(col) - 1.0) < 0.3

    def test_is_fitted_property(self) -> None:
        pipeline = DualNormalizationPipeline()
        assert not pipeline.is_fitted
        pipeline.fit(_make_features())
        assert pipeline.is_fitted

    def test_per_fold_training(self) -> None:
        """Different folds should produce different neural transforms."""
        data1 = _make_features(100)
        data2 = _make_features(100) * 3 + 500

        p1 = DualNormalizationPipeline()
        p2 = DualNormalizationPipeline()
        p1.fit(data1)
        p2.fit(data2)

        test = _make_features(10)
        r1 = p1.transform_neural(test)
        r2 = p2.transform_neural(test)
        assert not np.allclose(r1, r2)
