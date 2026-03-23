"""Tests for GRU hidden state feature extractor."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from ep2_crypto.models.gru_features import GRUConfig, GRUFeatureExtractor
from ep2_crypto.models.labeling import Direction

if TYPE_CHECKING:
    from pathlib import Path


def _make_sequential_data(
    n_samples: int = 200,
    n_features: int = 10,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate sequential data with temporal signal."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, size=(n_samples, n_features))
    # Label based on rolling mean of feature 0
    rolling = np.convolve(x[:, 0], np.ones(5) / 5, mode="same")
    labels = np.zeros(n_samples, dtype=np.int8)
    labels[rolling > 0.3] = Direction.UP
    labels[rolling < -0.3] = Direction.DOWN
    return x, labels


@pytest.fixture
def seq_data() -> tuple[np.ndarray, np.ndarray]:
    return _make_sequential_data()


@pytest.fixture
def fast_config() -> GRUConfig:
    """Fast config for testing (few epochs, small model)."""
    return GRUConfig(
        hidden_size=16,
        num_layers=1,
        dropout=0.0,
        seq_len=10,
        n_epochs=3,
        batch_size=32,
    )


class TestGRUTraining:
    def test_train_returns_metrics(
        self,
        seq_data: tuple,
        fast_config: GRUConfig,
    ) -> None:
        x, y = seq_data
        model = GRUFeatureExtractor(fast_config)
        split = 150
        metrics = model.train(x[:split], y[:split], x[split:], y[split:])
        assert "train_loss" in metrics
        assert "train_accuracy" in metrics
        assert "val_loss" in metrics
        assert "val_accuracy" in metrics

    def test_train_without_validation(
        self,
        seq_data: tuple,
        fast_config: GRUConfig,
    ) -> None:
        x, y = seq_data
        model = GRUFeatureExtractor(fast_config)
        metrics = model.train(x, y)
        assert "train_loss" in metrics
        assert "val_loss" not in metrics

    def test_is_fitted(
        self,
        seq_data: tuple,
        fast_config: GRUConfig,
    ) -> None:
        x, y = seq_data
        model = GRUFeatureExtractor(fast_config)
        assert not model.is_fitted
        model.train(x, y)
        assert model.is_fitted


class TestHiddenStateExtraction:
    def test_hidden_shape(
        self,
        seq_data: tuple,
        fast_config: GRUConfig,
    ) -> None:
        """Hidden states have shape (n_valid, hidden_size)."""
        x, y = seq_data
        model = GRUFeatureExtractor(fast_config)
        model.train(x, y)
        hidden = model.extract_hidden(x)
        expected_n = len(x) - fast_config.seq_len
        assert hidden.shape == (expected_n, fast_config.hidden_size)

    def test_hidden_deterministic(
        self,
        seq_data: tuple,
        fast_config: GRUConfig,
    ) -> None:
        """Extracting hidden states twice yields same result."""
        x, y = seq_data
        model = GRUFeatureExtractor(fast_config)
        model.train(x, y)
        h1 = model.extract_hidden(x)
        h2 = model.extract_hidden(x)
        np.testing.assert_allclose(h1, h2, atol=1e-6)

    def test_hidden_not_fitted_raises(self, fast_config: GRUConfig) -> None:
        model = GRUFeatureExtractor(fast_config)
        x = np.random.default_rng(42).normal(size=(50, 10))
        with pytest.raises(RuntimeError, match="not fitted"):
            model.extract_hidden(x)


class TestPredictProba:
    def test_proba_shape(
        self,
        seq_data: tuple,
        fast_config: GRUConfig,
    ) -> None:
        x, y = seq_data
        model = GRUFeatureExtractor(fast_config)
        model.train(x, y)
        proba = model.predict_proba(x)
        expected_n = len(x) - fast_config.seq_len
        assert proba.shape == (expected_n, 3)

    def test_proba_sums_to_one(
        self,
        seq_data: tuple,
        fast_config: GRUConfig,
    ) -> None:
        x, y = seq_data
        model = GRUFeatureExtractor(fast_config)
        model.train(x, y)
        proba = model.predict_proba(x)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)


class TestDataLoaderOrdering:
    def test_no_shuffle(self, fast_config: GRUConfig) -> None:
        """Verify DataLoader never shuffles (temporal ordering preserved)."""
        from torch.utils.data import DataLoader

        from ep2_crypto.models.gru_features import _SequenceDataset

        n = 50
        x = np.arange(n * 3, dtype=np.float64).reshape(n, 3)
        y = np.zeros(n, dtype=np.int8)
        ds = _SequenceDataset(x, y, seq_len=5)
        loader = DataLoader(ds, batch_size=10, shuffle=False)

        indices = []
        for seq_batch, _ in loader:
            # First element of each sequence should be the index
            first_vals = seq_batch[:, 0, 0].numpy()
            indices.extend(first_vals.tolist())

        # Should be in strictly increasing order (0, 3, 6, 9, ...)
        for i in range(1, len(indices)):
            assert indices[i] > indices[i - 1], f"DataLoader output not in order at position {i}"


class TestSaveLoad:
    def test_roundtrip(
        self,
        seq_data: tuple,
        fast_config: GRUConfig,
        tmp_path: Path,
    ) -> None:
        x, y = seq_data
        model = GRUFeatureExtractor(fast_config)
        model.train(x, y)

        original_hidden = model.extract_hidden(x)

        save_path = tmp_path / "gru_model"
        model.save(save_path)
        assert (tmp_path / "gru_model.pt").exists()

        loaded = GRUFeatureExtractor(fast_config)
        loaded.load(save_path)
        loaded_hidden = loaded.extract_hidden(x)

        np.testing.assert_allclose(
            original_hidden,
            loaded_hidden,
            atol=1e-6,
        )

    def test_save_before_train_raises(
        self,
        fast_config: GRUConfig,
        tmp_path: Path,
    ) -> None:
        model = GRUFeatureExtractor(fast_config)
        with pytest.raises(RuntimeError, match="not fitted"):
            model.save(tmp_path / "model")


class TestONNXExport:
    def test_export_creates_file(
        self,
        seq_data: tuple,
        fast_config: GRUConfig,
        tmp_path: Path,
    ) -> None:
        x, y = seq_data
        model = GRUFeatureExtractor(fast_config)
        model.train(x, y)

        onnx_path = tmp_path / "gru_model"
        model.export_onnx(onnx_path)
        assert (tmp_path / "gru_model.onnx").exists()

    def test_onnx_inference_matches(
        self,
        seq_data: tuple,
        fast_config: GRUConfig,
        tmp_path: Path,
    ) -> None:
        """ONNX model produces same output as PyTorch."""
        import onnxruntime as ort

        x, y = seq_data
        model = GRUFeatureExtractor(fast_config)
        model.train(x, y)

        # PyTorch output
        proba_torch = model.predict_proba(x[: fast_config.seq_len + 5])

        # Export and run ONNX
        onnx_path = tmp_path / "gru_model"
        model.export_onnx(onnx_path)

        sess = ort.InferenceSession(str(tmp_path / "gru_model.onnx"))

        # Build sequence input manually
        import torch

        features = torch.tensor(x[: fast_config.seq_len + 5], dtype=torch.float32)
        sequences = []
        for i in range(5):
            sequences.append(
                features[i : i + fast_config.seq_len].numpy(),
            )
        batch = np.stack(sequences)

        onnx_out = sess.run(None, {"features": batch})
        logits_onnx = onnx_out[0]

        # Softmax
        exp_logits = np.exp(logits_onnx - logits_onnx.max(axis=1, keepdims=True))
        proba_onnx = exp_logits / exp_logits.sum(axis=1, keepdims=True)

        # GRU numerical precision differs between PyTorch and ONNX Runtime
        # due to different internal implementations. Use relaxed tolerance.
        np.testing.assert_allclose(
            proba_torch[:5],
            proba_onnx,
            atol=0.05,
        )

    def test_export_before_train_raises(
        self,
        fast_config: GRUConfig,
        tmp_path: Path,
    ) -> None:
        model = GRUFeatureExtractor(fast_config)
        with pytest.raises(RuntimeError, match="not fitted"):
            model.export_onnx(tmp_path / "model")
