"""Probability calibration via isotonic regression.

Calibrates model output probabilities so that when the model says
"70% probability of UP", the true frequency is actually ~70%.
Uses one-vs-rest isotonic regression for each class.

Key metrics:
- ECE (Expected Calibration Error): measures calibration quality
- Reliability diagram: visual check for calibration
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from sklearn.isotonic import IsotonicRegression

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class CalibrationConfig:
    """Calibration configuration."""

    n_bins: int = 10  # For ECE computation and reliability diagram
    clip_min: float = 1e-6  # Min probability after calibration
    clip_max: float = 1.0 - 1e-6  # Max probability after calibration


class IsotonicCalibrator:
    """Per-class isotonic regression calibrator.

    Fits one isotonic regression per class (one-vs-rest) on a held-out
    calibration set. At inference, calibrates each class probability
    independently and renormalizes.
    """

    def __init__(self, config: CalibrationConfig | None = None) -> None:
        self._config = config or CalibrationConfig()
        self._calibrators: list[IsotonicRegression] = []
        self._n_classes: int = 3

    @property
    def is_fitted(self) -> bool:
        return len(self._calibrators) == self._n_classes

    def fit(
        self,
        probas: NDArray[np.float64],
        y_true: NDArray[np.int8],
    ) -> dict[str, float]:
        """Fit isotonic calibrators on held-out data.

        Args:
            probas: Uncalibrated probabilities (n_samples, 3) for [DOWN, FLAT, UP].
            y_true: True labels (-1, 0, +1).

        Returns:
            Dict with pre/post calibration ECE.
        """
        # Encode: -1→0, 0→1, 1→2
        y_encoded = y_true.astype(np.int32) + 1

        pre_ece = self._compute_ece(probas, y_encoded)

        self._calibrators = []
        for c in range(self._n_classes):
            binary_y = (y_encoded == c).astype(np.float64)
            iso = IsotonicRegression(
                y_min=self._config.clip_min,
                y_max=self._config.clip_max,
                out_of_bounds="clip",
            )
            iso.fit(probas[:, c], binary_y)
            self._calibrators.append(iso)

        calibrated = self.calibrate(probas)
        post_ece = self._compute_ece(calibrated, y_encoded)

        return {
            "pre_calibration_ece": pre_ece,
            "post_calibration_ece": post_ece,
            "ece_improvement": pre_ece - post_ece,
        }

    def calibrate(
        self,
        probas: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Calibrate probabilities using fitted isotonic regressors.

        Args:
            probas: Uncalibrated probabilities (n_samples, 3).

        Returns:
            Calibrated probabilities (n_samples, 3), normalized to sum to 1.
        """
        if not self.is_fitted:
            msg = "Calibrator not fitted. Call fit() first."
            raise RuntimeError(msg)

        n = len(probas)
        calibrated = np.empty((n, self._n_classes), dtype=np.float64)

        for c in range(self._n_classes):
            calibrated[:, c] = self._calibrators[c].predict(probas[:, c])

        # Renormalize rows to sum to 1
        row_sums = calibrated.sum(axis=1, keepdims=True)
        row_sums = np.maximum(row_sums, 1e-10)  # Avoid division by zero
        calibrated = calibrated / row_sums

        result: NDArray[np.float64] = calibrated
        return result

    def reliability_curve(
        self,
        probas: NDArray[np.float64],
        y_true: NDArray[np.int8],
        class_idx: int = 2,  # Default: UP class
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]]:
        """Compute reliability diagram data for a specific class.

        Args:
            probas: Probabilities (n_samples, 3).
            y_true: True labels (-1, 0, +1).
            class_idx: Which class to plot (0=DOWN, 1=FLAT, 2=UP).

        Returns:
            Tuple of (mean_predicted_prob, fraction_of_positives, bin_counts)
            per bin.
        """
        y_encoded = y_true.astype(np.int32) + 1
        binary_y = (y_encoded == class_idx).astype(np.float64)
        class_probs = probas[:, class_idx]

        n_bins = self._config.n_bins
        bin_edges = np.linspace(0, 1, n_bins + 1)

        mean_predicted = np.zeros(n_bins, dtype=np.float64)
        fraction_positive = np.zeros(n_bins, dtype=np.float64)
        bin_counts = np.zeros(n_bins, dtype=np.int64)

        for b in range(n_bins):
            mask = (class_probs >= bin_edges[b]) & (class_probs < bin_edges[b + 1])
            if b == n_bins - 1:
                mask = (class_probs >= bin_edges[b]) & (class_probs <= bin_edges[b + 1])
            count = mask.sum()
            bin_counts[b] = count
            if count > 0:
                mean_predicted[b] = class_probs[mask].mean()
                fraction_positive[b] = binary_y[mask].mean()

        return mean_predicted, fraction_positive, bin_counts

    def _compute_ece(
        self,
        probas: NDArray[np.float64],
        y_encoded: NDArray[np.int32],
    ) -> float:
        """Compute Expected Calibration Error (ECE).

        ECE = sum(|accuracy_b - confidence_b| * n_b / N) over all bins.
        Uses the predicted class confidence and actual accuracy per bin.
        """
        # Get predicted class and its confidence
        pred_class = np.argmax(probas, axis=1)
        confidence = np.max(probas, axis=1)
        correct = (pred_class == y_encoded).astype(np.float64)

        n_bins = self._config.n_bins
        bin_edges = np.linspace(0, 1, n_bins + 1)
        n = len(probas)
        ece = 0.0

        for b in range(n_bins):
            mask = (confidence >= bin_edges[b]) & (confidence < bin_edges[b + 1])
            if b == n_bins - 1:
                mask = (confidence >= bin_edges[b]) & (confidence <= bin_edges[b + 1])
            count = mask.sum()
            if count > 0:
                avg_conf = confidence[mask].mean()
                avg_acc = correct[mask].mean()
                ece += abs(avg_acc - avg_conf) * count / n

        return float(ece)

    def save(self, path: Path | str) -> None:
        """Save calibrators to disk."""
        if not self.is_fitted:
            msg = "Calibrator not fitted. Cannot save."
            raise RuntimeError(msg)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        import joblib

        joblib.dump(self._calibrators, str(path.with_suffix(".joblib")))

        meta = {
            "n_classes": self._n_classes,
            "n_bins": self._config.n_bins,
        }
        meta_path = path.with_suffix(".meta.json")
        meta_path.write_text(json.dumps(meta, indent=2))

    def load(self, path: Path | str) -> None:
        """Load calibrators from disk."""
        path = Path(path)

        import joblib

        self._calibrators = joblib.load(str(path.with_suffix(".joblib")))

        meta_path = path.with_suffix(".meta.json")
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            self._n_classes = meta.get("n_classes", 3)
