"""Tests for multi-task GRU: forward pass, auxiliary losses, hidden state extraction.

Tests verify:
1. All three heads produce correct output shapes
2. Training with and without auxiliary tasks converges
3. Hidden state extraction returns correct shape
4. NEVER bidirectional (verified on model instantiation)
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ep2_crypto.models.gru_features import (
    MultiTaskGRUConfig,
    MultiTaskGRUFeatureExtractor,
    _MultiTaskGRUNet,
    _MultiTaskSequenceDataset,
)


def _make_data(
    n: int = 200,
    n_features: int = 10,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, (n, n_features)).astype(np.float64)
    y = rng.integers(-1, 2, n).astype(np.int8)
    vol = np.abs(rng.normal(0.001, 0.0005, n)).astype(np.float64)
    volume = np.abs(rng.normal(100, 20, n)).astype(np.float64)
    return x, y, vol, volume


class TestMultiTaskGRUNet:
    def test_not_bidirectional(self) -> None:
        model = _MultiTaskGRUNet(input_size=10, hidden_size=32)
        assert not model.gru.bidirectional

    def test_forward_output_shapes(self) -> None:
        model = _MultiTaskGRUNet(input_size=10, hidden_size=32, num_layers=2)
        batch = torch.randn(4, 12, 10)
        logits, vol_pred, volume_pred, hidden = model(batch)
        assert logits.shape == (4, 3)
        assert vol_pred.shape == (4,)
        assert volume_pred.shape == (4,)
        assert hidden.shape == (4, 32)

    def test_three_heads_exist(self) -> None:
        model = _MultiTaskGRUNet(input_size=8, hidden_size=16)
        assert hasattr(model, "cls_head")
        assert hasattr(model, "vol_head")
        assert hasattr(model, "volume_head")

    def test_hidden_size_matches_config(self) -> None:
        model = _MultiTaskGRUNet(input_size=8, hidden_size=64)
        x = torch.randn(2, 5, 8)
        _, _, _, h = model(x)
        assert h.shape[1] == 64

    def test_gradients_flow_to_all_heads(self) -> None:
        """Loss from all heads should produce gradients in all parameters."""
        model = _MultiTaskGRUNet(input_size=8, hidden_size=16)
        x = torch.randn(4, 6, 8)
        labels = torch.randint(0, 3, (4,))
        vol_t = torch.randn(4)
        volume_t = torch.randn(4)

        logits, vol_pred, volume_pred, _ = model(x)
        loss = (
            torch.nn.functional.cross_entropy(logits, labels)
            + 0.1 * torch.nn.functional.mse_loss(vol_pred, vol_t)
            + 0.1 * torch.nn.functional.mse_loss(volume_pred, volume_t)
        )
        loss.backward()

        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"


class TestMultiTaskSequenceDataset:
    def test_length(self) -> None:
        x = np.ones((100, 10))
        ds = _MultiTaskSequenceDataset(x, None, None, None, seq_len=12)
        assert len(ds) == 100 - 12

    def test_item_shape_with_all_targets(self) -> None:
        n, n_f, seq = 100, 8, 12
        x = np.random.randn(n, n_f).astype(np.float64)
        y = np.ones(n, dtype=np.int8)
        vol = np.ones(n, dtype=np.float64) * 0.01
        volume = np.ones(n, dtype=np.float64) * 100.0
        ds = _MultiTaskSequenceDataset(x, y, vol, volume, seq_len=seq)
        seq_t, label, vol_t, volume_t = ds[0]
        assert seq_t.shape == (seq, n_f)
        assert label.shape == ()
        assert vol_t.shape == ()
        assert volume_t.shape == ()

    def test_item_shape_without_aux_targets(self) -> None:
        n, n_f, seq = 50, 6, 10
        x = np.random.randn(n, n_f).astype(np.float64)
        y = np.zeros(n, dtype=np.int8)
        ds = _MultiTaskSequenceDataset(x, y, None, None, seq_len=seq)
        item = ds[0]
        assert len(item) == 2  # seq + label only


class TestMultiTaskGRUFeatureExtractor:
    def test_not_fitted_raises(self) -> None:
        ext = MultiTaskGRUFeatureExtractor()
        x = np.random.randn(100, 10).astype(np.float64)
        with pytest.raises(RuntimeError, match="not fitted"):
            ext.extract_hidden(x)

    def test_is_fitted_after_train(self) -> None:
        cfg = MultiTaskGRUConfig(n_epochs=2, hidden_size=16, seq_len=5)
        ext = MultiTaskGRUFeatureExtractor(cfg)
        x, y, _, _ = _make_data(50, 8)
        ext.train(x, y)
        assert ext.is_fitted

    def test_hidden_size_property(self) -> None:
        cfg = MultiTaskGRUConfig(hidden_size=32, seq_len=5, n_epochs=1)
        ext = MultiTaskGRUFeatureExtractor(cfg)
        assert ext.hidden_size == 32

    def test_train_without_auxiliary_targets(self) -> None:
        """Training with direction only should succeed."""
        cfg = MultiTaskGRUConfig(n_epochs=2, hidden_size=16, seq_len=5)
        ext = MultiTaskGRUFeatureExtractor(cfg)
        x, y, _, _ = _make_data(50, 8)
        metrics = ext.train(x, y)
        assert "train_loss" in metrics
        assert metrics["has_vol_task"] == 0.0
        assert metrics["has_volume_task"] == 0.0

    def test_train_with_all_auxiliary_targets(self) -> None:
        """Training with all three losses should succeed."""
        cfg = MultiTaskGRUConfig(
            n_epochs=2, hidden_size=16, seq_len=5, vol_weight=0.1, volume_weight=0.1
        )
        ext = MultiTaskGRUFeatureExtractor(cfg)
        x, y, vol, volume = _make_data(50, 8)
        metrics = ext.train(x, y, vol_targets=vol, volume_targets=volume)
        assert metrics["has_vol_task"] == 1.0
        assert metrics["has_volume_task"] == 1.0

    def test_train_with_vol_only(self) -> None:
        cfg = MultiTaskGRUConfig(n_epochs=2, hidden_size=16, seq_len=5)
        ext = MultiTaskGRUFeatureExtractor(cfg)
        x, y, vol, _ = _make_data(50, 8)
        metrics = ext.train(x, y, vol_targets=vol)
        assert metrics["has_vol_task"] == 1.0
        assert metrics["has_volume_task"] == 0.0

    def test_extract_hidden_shape(self) -> None:
        """extract_hidden returns (n_valid, hidden_size)."""
        cfg = MultiTaskGRUConfig(n_epochs=2, hidden_size=16, seq_len=5)
        ext = MultiTaskGRUFeatureExtractor(cfg)
        x, y, vol, volume = _make_data(60, 8)
        ext.train(x, y, vol_targets=vol, volume_targets=volume)
        hidden = ext.extract_hidden(x)
        # n_valid = n_samples - seq_len = 60 - 5 = 55
        assert hidden.shape == (55, 16)
        assert hidden.dtype == np.float64

    def test_hidden_states_differ_per_timestep(self) -> None:
        """Different time steps should yield different hidden states."""
        cfg = MultiTaskGRUConfig(n_epochs=3, hidden_size=16, seq_len=5)
        ext = MultiTaskGRUFeatureExtractor(cfg)
        x, y, vol, volume = _make_data(60, 8)
        ext.train(x, y, vol_targets=vol, volume_targets=volume)
        hidden = ext.extract_hidden(x)
        # Not all identical
        assert not np.allclose(hidden[0], hidden[10])

    def test_train_with_validation_set(self) -> None:
        cfg = MultiTaskGRUConfig(n_epochs=3, hidden_size=16, seq_len=5)
        ext = MultiTaskGRUFeatureExtractor(cfg)
        x, y, _, _ = _make_data(80, 8)
        x_train, y_train = x[:60], y[:60]
        x_val, y_val = x[60:], y[60:]
        metrics = ext.train(x_train, y_train, x_val=x_val, y_val=y_val)
        assert "best_val_loss" in metrics

    def test_auxiliary_task_changes_hidden_states(self) -> None:
        """Multi-task training should produce different hidden states than no-aux."""
        cfg = MultiTaskGRUConfig(n_epochs=5, hidden_size=16, seq_len=5, random_seed=0)
        x, y, vol, volume = _make_data(60, 8)

        ext_single = MultiTaskGRUFeatureExtractor(cfg)
        ext_single.train(x, y)
        h_single = ext_single.extract_hidden(x)

        cfg2 = MultiTaskGRUConfig(n_epochs=5, hidden_size=16, seq_len=5, random_seed=0)
        ext_multi = MultiTaskGRUFeatureExtractor(cfg2)
        ext_multi.train(x, y, vol_targets=vol, volume_targets=volume)
        h_multi = ext_multi.extract_hidden(x)

        # Hidden states should differ due to additional loss terms
        assert not np.allclose(h_single, h_multi, atol=1e-6)

    def test_train_returns_float_metrics(self) -> None:
        cfg = MultiTaskGRUConfig(n_epochs=2, hidden_size=8, seq_len=4)
        ext = MultiTaskGRUFeatureExtractor(cfg)
        x, y, vol, volume = _make_data(40, 6)
        metrics = ext.train(x, y, vol_targets=vol, volume_targets=volume)
        for v in metrics.values():
            assert isinstance(v, float)
