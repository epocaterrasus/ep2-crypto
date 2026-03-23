"""Tests for LightGBM quantile regression model."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from ep2_crypto.models.quantile import QuantileConfig, QuantileModel

if TYPE_CHECKING:
    from pathlib import Path


def _make_regression_data(
    n_samples: int = 500,
    n_features: int = 10,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Generate synthetic regression data (forward returns)."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, size=(n_samples, n_features))
    # Target = linear combination + noise (simulating return prediction)
    y = x[:, 0] * 0.003 + x[:, 1] * 0.002 + rng.normal(0, 0.005, size=n_samples)
    names = [f"f{i}" for i in range(n_features)]
    return x, y, names


@pytest.fixture
def reg_data() -> tuple[np.ndarray, np.ndarray, list[str]]:
    return _make_regression_data()


@pytest.fixture
def train_val_reg(
    reg_data: tuple,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    x, y, names = reg_data
    split = int(len(x) * 0.8)
    return x[:split], y[:split], x[split:], y[split:], names


class TestQuantileTraining:
    def test_train_returns_metrics(self, train_val_reg: tuple) -> None:
        x_train, y_train, x_val, y_val, names = train_val_reg
        model = QuantileModel()
        metrics = model.train(
            x_train,
            y_train,
            x_val,
            y_val,
            feature_names=names,
        )
        assert "q10_best_iter" in metrics
        assert "q50_best_iter" in metrics
        assert "q90_best_iter" in metrics

    def test_is_fitted(self, reg_data: tuple) -> None:
        x, y, names = reg_data
        model = QuantileModel()
        assert not model.is_fitted
        model.train(x, y, feature_names=names)
        assert model.is_fitted


class TestQuantilePrediction:
    def test_predict_all_quantiles(self, reg_data: tuple) -> None:
        x, y, names = reg_data
        model = QuantileModel()
        model.train(x, y, feature_names=names)
        preds = model.predict(x[:50])
        assert len(preds) == 5
        for q in (0.10, 0.25, 0.50, 0.75, 0.90):
            assert q in preds
            assert preds[q].shape == (50,)

    def test_quantile_ordering(self, reg_data: tuple) -> None:
        """Lower quantiles should generally predict lower values."""
        x, y, names = reg_data
        model = QuantileModel()
        model.train(x, y, feature_names=names)
        preds = model.predict(x)
        # Check median of each quantile prediction
        q10_median = np.median(preds[0.10])
        q50_median = np.median(preds[0.50])
        q90_median = np.median(preds[0.90])
        assert q10_median < q50_median < q90_median

    def test_quantile_crossing_rate(self, reg_data: tuple) -> None:
        """Quantile crossing should be rare for well-trained model."""
        x, y, names = reg_data
        model = QuantileModel()
        model.train(x, y, feature_names=names)
        crossing_rate = model.check_quantile_ordering(x)
        # Allow up to 15% crossing (tree models can have some)
        assert crossing_rate < 0.15

    def test_predict_before_train_raises(self) -> None:
        model = QuantileModel()
        x = np.random.default_rng(42).normal(size=(10, 5))
        with pytest.raises(RuntimeError, match="not fitted"):
            model.predict(x)


class TestPredictionIntervals:
    def test_interval_shape(self, reg_data: tuple) -> None:
        x, y, names = reg_data
        model = QuantileModel()
        model.train(x, y, feature_names=names)
        median, lower, upper = model.predict_intervals(x[:50])
        assert median.shape == (50,)
        assert lower.shape == (50,)
        assert upper.shape == (50,)

    def test_interval_width_positive(self, reg_data: tuple) -> None:
        """Interval width should be positive (upper > lower on average)."""
        x, y, names = reg_data
        model = QuantileModel()
        model.train(x, y, feature_names=names)
        width = model.interval_width(x)
        assert np.mean(width) > 0

    def test_coverage(self, reg_data: tuple) -> None:
        """80% interval should contain ~80% of true values."""
        x, y, names = reg_data
        config = QuantileConfig(n_estimators=200)
        model = QuantileModel(config)
        model.train(x, y, feature_names=names)
        _, lower, upper = model.predict_intervals(x)
        in_interval = (y >= lower) & (y <= upper)
        coverage = np.mean(in_interval)
        # Should be near 80%, allow wide tolerance (60-95%)
        assert 0.60 < coverage < 0.95


class TestQuantileSaveLoad:
    def test_roundtrip(
        self,
        reg_data: tuple,
        tmp_path: Path,
    ) -> None:
        x, y, names = reg_data
        model = QuantileModel()
        model.train(x, y, feature_names=names)
        original = model.predict(x[:20])

        save_path = tmp_path / "quantile_model"
        model.save(save_path)

        loaded = QuantileModel()
        loaded.load(save_path)
        loaded_preds = loaded.predict(x[:20])

        for q in model.quantiles:
            np.testing.assert_allclose(
                original[q],
                loaded_preds[q],
                atol=1e-10,
            )

    def test_save_before_train_raises(self, tmp_path: Path) -> None:
        model = QuantileModel()
        with pytest.raises(RuntimeError, match="not fitted"):
            model.save(tmp_path / "model")


class TestCustomQuantiles:
    def test_custom_quantile_set(self, reg_data: tuple) -> None:
        x, y, names = reg_data
        config = QuantileConfig(quantiles=(0.05, 0.50, 0.95))
        model = QuantileModel(config)
        model.train(x, y, feature_names=names)
        preds = model.predict(x[:10])
        assert set(preds.keys()) == {0.05, 0.50, 0.95}
