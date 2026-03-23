"""GRU hidden state extractor for the stacking ensemble.

Trains a 2-layer GRU on sequential feature data, then extracts the
final hidden state as a feature vector for tree-based models (LightGBM/CatBoost).
This captures temporal patterns that trees cannot model directly.

Key constraints:
- DataLoader NEVER shuffles (time series ordering preserved)
- AdamW optimizer with gradient clipping at 1.0
- Cosine annealing learning rate schedule
- ONNX export for fast inference
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class GRUConfig:
    """GRU model configuration."""

    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.3
    seq_len: int = 24  # 24 bars = 2 hours at 5-min
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    n_epochs: int = 30
    batch_size: int = 64
    grad_clip: float = 1.0
    random_seed: int = 42


class _SequenceDataset(Dataset):  # type: ignore[type-arg]
    """Dataset that creates sequential windows from feature arrays.

    CRITICAL: No shuffling. Time ordering is preserved.
    """

    def __init__(
        self,
        features: NDArray[np.float64],
        labels: NDArray[np.int8] | None,
        seq_len: int,
    ) -> None:
        self._features = torch.tensor(features, dtype=torch.float32)
        self._labels = (
            torch.tensor(labels.astype(np.int64), dtype=torch.long) if labels is not None else None
        )
        self._seq_len = seq_len

    def __len__(self) -> int:
        return max(0, len(self._features) - self._seq_len)

    def __getitem__(
        self,
        idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        seq = self._features[idx : idx + self._seq_len]
        if self._labels is not None:
            label = self._labels[idx + self._seq_len - 1]
            return seq, label
        return seq


class _GRUNet(nn.Module):
    """GRU network for ternary classification with hidden state extraction."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        n_classes: int = 3,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=False,  # NEVER bidirectional (uses future data)
        )
        self.classifier = nn.Linear(hidden_size, n_classes)

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning (logits, hidden_state).

        Args:
            x: (batch, seq_len, input_size)

        Returns:
            Tuple of (logits: (batch, n_classes), hidden: (batch, hidden_size))
        """
        # output: (batch, seq_len, hidden_size)
        # h_n: (num_layers, batch, hidden_size)
        _, h_n = self.gru(x)
        # Use the last layer's hidden state
        last_hidden = h_n[-1]  # (batch, hidden_size)
        logits = self.classifier(last_hidden)
        return logits, last_hidden


class GRUFeatureExtractor:
    """Trains a GRU and extracts hidden states as features.

    The hidden state of the trained GRU captures temporal patterns
    in the feature sequence. These hidden states become additional
    features for the tree-based models in the stacking ensemble.
    """

    def __init__(self, config: GRUConfig | None = None) -> None:
        self._config = config or GRUConfig()
        self._model: _GRUNet | None = None
        self._input_size: int | None = None
        self._device = torch.device("cpu")

    @property
    def is_fitted(self) -> bool:
        return self._model is not None

    @property
    def hidden_size(self) -> int:
        return self._config.hidden_size

    @property
    def seq_len(self) -> int:
        return self._config.seq_len

    def train(
        self,
        x_train: NDArray[np.float64],
        y_train: NDArray[np.int8],
        x_val: NDArray[np.float64] | None = None,
        y_val: NDArray[np.int8] | None = None,
    ) -> dict[str, float]:
        """Train the GRU model.

        Args:
            x_train: Training features (n_samples, n_features).
            y_train: Training labels (-1, 0, +1), mapped to (0, 1, 2).
            x_val: Validation features.
            y_val: Validation labels.

        Returns:
            Dict with training metrics.
        """
        torch.manual_seed(self._config.random_seed)
        cfg = self._config
        self._input_size = x_train.shape[1]

        # Encode labels: -1→0, 0→1, 1→2
        y_encoded = (y_train.astype(np.int64) + 1).astype(np.int8)

        train_ds = _SequenceDataset(x_train, y_encoded, cfg.seq_len)
        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.batch_size,
            shuffle=False,  # NEVER shuffle time series
            drop_last=False,
        )

        model = _GRUNet(
            input_size=self._input_size,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
        ).to(self._device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.n_epochs,
        )
        criterion = nn.CrossEntropyLoss()

        best_val_loss = float("inf")
        best_state = None

        for _epoch in range(cfg.n_epochs):
            model.train()
            epoch_loss = 0.0
            n_batches = 0

            for batch in train_loader:
                sequences, labels = batch
                sequences = sequences.to(self._device)
                labels = labels.to(self._device)

                optimizer.zero_grad()
                logits, _ = model(sequences)
                loss = criterion(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    cfg.grad_clip,
                )
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            scheduler.step()

            # Validation
            if x_val is not None and y_val is not None:
                val_loss = self._evaluate(
                    model,
                    x_val,
                    y_val,
                    criterion,
                )
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}

        # Restore best model if validation was used
        if best_state is not None:
            model.load_state_dict(best_state)

        self._model = model

        # Compute final metrics
        metrics: dict[str, float] = {}
        train_loss = epoch_loss / max(n_batches, 1)
        metrics["train_loss"] = train_loss
        metrics["train_accuracy"] = self._compute_accuracy(x_train, y_encoded)

        if x_val is not None and y_val is not None:
            y_val_encoded = (y_val.astype(np.int64) + 1).astype(np.int8)
            metrics["val_loss"] = best_val_loss
            metrics["val_accuracy"] = self._compute_accuracy(
                x_val,
                y_val_encoded,
            )

        return metrics

    def _evaluate(
        self,
        model: _GRUNet,
        x: NDArray[np.float64],
        y: NDArray[np.int8],
        criterion: nn.CrossEntropyLoss,
    ) -> float:
        """Compute validation loss."""
        y_encoded = (y.astype(np.int64) + 1).astype(np.int8)
        ds = _SequenceDataset(x, y_encoded, self._config.seq_len)
        loader = DataLoader(ds, batch_size=self._config.batch_size, shuffle=False)

        model.eval()
        total_loss = 0.0
        n_batches = 0
        with torch.no_grad():
            for batch in loader:
                sequences, labels = batch
                sequences = sequences.to(self._device)
                labels = labels.to(self._device)
                logits, _ = model(sequences)
                total_loss += criterion(logits, labels).item()
                n_batches += 1
        return total_loss / max(n_batches, 1)

    def _compute_accuracy(
        self,
        x: NDArray[np.float64],
        y_encoded: NDArray[np.int8],
    ) -> float:
        """Compute classification accuracy."""
        if self._model is None:
            return 0.0
        ds = _SequenceDataset(x, y_encoded, self._config.seq_len)
        loader = DataLoader(ds, batch_size=self._config.batch_size, shuffle=False)

        self._model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in loader:
                sequences, labels = batch
                sequences = sequences.to(self._device)
                labels = labels.to(self._device)
                logits, _ = self._model(sequences)
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += len(labels)
        return correct / max(total, 1)

    def extract_hidden(
        self,
        x: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Extract hidden state features for each valid position.

        Args:
            x: Features (n_samples, n_features). Must have >= seq_len rows.

        Returns:
            Hidden states (n_valid, hidden_size) where n_valid = n_samples - seq_len.
        """
        if self._model is None:
            msg = "Model not fitted. Call train() first."
            raise RuntimeError(msg)

        ds = _SequenceDataset(x, None, self._config.seq_len)
        loader = DataLoader(
            ds,
            batch_size=self._config.batch_size,
            shuffle=False,
        )

        self._model.eval()
        hidden_list: list[np.ndarray] = []
        with torch.no_grad():
            for batch in loader:
                sequences = batch.to(self._device)
                _, hidden = self._model(sequences)
                hidden_list.append(hidden.cpu().numpy())

        if not hidden_list:
            return np.empty(
                (0, self._config.hidden_size),
                dtype=np.float64,
            )
        return np.concatenate(hidden_list, axis=0).astype(np.float64)

    def predict_proba(
        self,
        x: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Predict class probabilities for the stacking ensemble.

        Args:
            x: Features (n_samples, n_features).

        Returns:
            Probabilities (n_valid, 3) for [DOWN, FLAT, UP].
        """
        if self._model is None:
            msg = "Model not fitted. Call train() first."
            raise RuntimeError(msg)

        ds = _SequenceDataset(x, None, self._config.seq_len)
        loader = DataLoader(
            ds,
            batch_size=self._config.batch_size,
            shuffle=False,
        )

        self._model.eval()
        proba_list: list[np.ndarray] = []
        with torch.no_grad():
            for batch in loader:
                sequences = batch.to(self._device)
                logits, _ = self._model(sequences)
                proba = torch.softmax(logits, dim=1)
                proba_list.append(proba.cpu().numpy())

        if not proba_list:
            return np.empty((0, 3), dtype=np.float64)
        return np.concatenate(proba_list, axis=0).astype(np.float64)

    def save(self, path: Path | str) -> None:
        """Save model weights to disk."""
        if self._model is None:
            msg = "Model not fitted. Cannot save."
            raise RuntimeError(msg)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self._model.state_dict(),
                "input_size": self._input_size,
                "config": {
                    "hidden_size": self._config.hidden_size,
                    "num_layers": self._config.num_layers,
                    "dropout": self._config.dropout,
                    "seq_len": self._config.seq_len,
                },
            },
            str(path.with_suffix(".pt")),
        )

    def load(self, path: Path | str) -> None:
        """Load model weights from disk."""
        path = Path(path)
        checkpoint = torch.load(
            str(path.with_suffix(".pt")),
            map_location=self._device,
            weights_only=True,
        )
        self._input_size = checkpoint["input_size"]
        cfg = checkpoint["config"]
        self._model = _GRUNet(
            input_size=self._input_size,
            hidden_size=cfg["hidden_size"],
            num_layers=cfg["num_layers"],
            dropout=cfg["dropout"],
        ).to(self._device)
        self._model.load_state_dict(checkpoint["state_dict"])

    def export_onnx(self, path: Path | str) -> None:
        """Export model to ONNX format for fast inference.

        Args:
            path: Output path (will use .onnx suffix).
        """
        if self._model is None or self._input_size is None:
            msg = "Model not fitted. Cannot export."
            raise RuntimeError(msg)

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        self._model.eval()
        dummy_input = torch.randn(
            1,
            self._config.seq_len,
            self._input_size,
        )
        onnx_path = str(path.with_suffix(".onnx"))

        torch.onnx.export(
            self._model,
            (dummy_input,),
            onnx_path,
            input_names=["features"],
            output_names=["logits", "hidden"],
            dynamic_axes={
                "features": {0: "batch"},
                "logits": {0: "batch"},
                "hidden": {0: "batch"},
            },
            opset_version=17,
        )
