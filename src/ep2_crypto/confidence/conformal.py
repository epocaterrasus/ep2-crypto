"""Conformal prediction gate for ambiguity detection.

Constructs prediction sets from nonconformity scores computed on
calibration data. At inference, if the prediction set contains more
than one class (or only FLAT), the model abstains from trading.

Key idea: nonconformity score = 1 - p_true_class. Low scores mean the
model was confident and correct. We use the calibration quantile of
these scores to decide which classes to include at inference.

Adaptive alpha tracks empirical coverage and adjusts to maintain the
target coverage rate over time.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import structlog

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = structlog.get_logger(__name__)

# Class index constants
DOWN = 0
FLAT = 1
UP = 2

# Tradeable singleton sets
_TRADEABLE_SETS: frozenset[frozenset[int]] = frozenset(
    {frozenset({DOWN}), frozenset({UP})}
)

# Direction mapping for tradeable singletons
_DIRECTION_MAP: dict[frozenset[int], int] = {
    frozenset({DOWN}): -1,
    frozenset({UP}): 1,
}


@dataclass
class ConformalConfig:
    """Configuration for conformal prediction gate."""

    alpha: float = 0.1  # Target miscoverage rate (1 - coverage = 0.1 → 90% coverage)
    adaptive: bool = True  # Whether to adapt alpha based on empirical coverage
    adaptive_lr: float = 0.01  # Learning rate for alpha adaptation
    min_calibration_size: int = 100  # Minimum samples for calibration


@dataclass
class _ConformalState:
    """Internal state for the conformal predictor."""

    scores: NDArray[np.float64] | None = None
    quantile_threshold: float = 0.0
    n_calibration_samples: int = 0
    empirical_coverage: float = 0.0
    current_alpha: float = 0.1
    # Track coverage history for adaptive alpha
    coverage_ema: float = field(default=0.0, init=False)


class ConformalPredictor:
    """Conformal prediction gate for ambiguity-aware trading.

    Uses nonconformity scores (1 - p_true_class) from calibration data
    to construct prediction sets at inference. Trading is only permitted
    when the prediction set is a singleton containing UP or DOWN.

    Adaptive alpha adjusts the miscoverage rate based on observed
    empirical coverage to maintain the target rate over time.
    """

    def __init__(self, config: ConformalConfig | None = None) -> None:
        self._config = config or ConformalConfig()
        self._state = _ConformalState(current_alpha=self._config.alpha)
        self._n_classes: int = 3

    @property
    def is_calibrated(self) -> bool:
        """Whether the predictor has been calibrated."""
        return self._state.scores is not None

    @property
    def current_alpha(self) -> float:
        """Current miscoverage rate (may differ from config if adaptive)."""
        return self._state.current_alpha

    def calibrate(
        self,
        probas: NDArray[np.float64],
        y_true: NDArray[np.int8],
    ) -> dict[str, float]:
        """Compute nonconformity scores on calibration data.

        Nonconformity score for sample i = 1 - probas[i, true_class].
        The quantile of these scores at level (1 - alpha) determines
        the threshold for prediction set construction.

        Args:
            probas: Calibrated probabilities (n_samples, 3) for [DOWN, FLAT, UP].
            y_true: True labels in {-1, 0, +1} encoding.

        Returns:
            Dict with calibration metrics.

        Raises:
            ValueError: If calibration set is too small.
        """
        n_samples = len(probas)
        if n_samples < self._config.min_calibration_size:
            msg = (
                f"Calibration set too small: {n_samples} < "
                f"{self._config.min_calibration_size} minimum"
            )
            raise ValueError(msg)

        if probas.shape[1] != self._n_classes:
            msg = f"Expected {self._n_classes} classes, got {probas.shape[1]}"
            raise ValueError(msg)

        # Encode: -1 → 0, 0 → 1, +1 → 2
        y_encoded = y_true.astype(np.int32) + 1

        # Nonconformity score = 1 - p(true class)
        scores = 1.0 - probas[np.arange(n_samples), y_encoded]

        # Compute quantile threshold at level ceil((n+1)(1-alpha)) / n
        # This is the standard conformal prediction quantile
        adjusted_level = np.ceil((n_samples + 1) * (1.0 - self._state.current_alpha)) / n_samples
        adjusted_level = min(adjusted_level, 1.0)
        quantile_threshold = float(np.quantile(scores, adjusted_level))

        self._state.scores = scores.astype(np.float64)
        self._state.quantile_threshold = quantile_threshold
        self._state.n_calibration_samples = n_samples
        self._state.coverage_ema = 1.0 - self._state.current_alpha

        # Compute metrics on calibration data
        pred_sets = self._build_prediction_sets(probas)
        n_singletons = sum(1 for s in pred_sets if len(s) == 1)
        n_tradeable = sum(1 for s in pred_sets if frozenset(s) in _TRADEABLE_SETS)
        n_empty = sum(1 for s in pred_sets if len(s) == 0)

        # Coverage: fraction of samples where true class is in the prediction set
        coverage = np.mean(
            [y_encoded[i] in pred_sets[i] for i in range(n_samples)]
        )
        self._state.empirical_coverage = float(coverage)

        metrics = {
            "n_calibration_samples": float(n_samples),
            "quantile_threshold": quantile_threshold,
            "empirical_coverage": float(coverage),
            "target_coverage": 1.0 - self._state.current_alpha,
            "singleton_rate": n_singletons / n_samples,
            "tradeable_rate": n_tradeable / n_samples,
            "empty_set_rate": n_empty / n_samples,
            "mean_nonconformity_score": float(scores.mean()),
            "median_nonconformity_score": float(np.median(scores)),
        }

        logger.info(
            "conformal_calibration_complete",
            n_samples=n_samples,
            quantile_threshold=round(quantile_threshold, 4),
            coverage=round(float(coverage), 4),
            tradeable_rate=round(metrics["tradeable_rate"], 4),
        )

        return metrics

    def predict_sets(
        self,
        probas: NDArray[np.float64],
    ) -> list[set[int]]:
        """Construct prediction sets for each sample.

        For each sample, include class c if:
            probas[i, c] >= 1 - quantile_threshold

        This is equivalent to: include class c if its nonconformity
        score (1 - p_c) <= quantile_threshold.

        Args:
            probas: Probabilities (n_samples, 3) for [DOWN, FLAT, UP].

        Returns:
            List of prediction sets. Each set contains class indices
            (0=DOWN, 1=FLAT, 2=UP).

        Raises:
            RuntimeError: If predictor is not calibrated.
        """
        self._check_calibrated()
        return self._build_prediction_sets(probas)

    def gate(
        self,
        probas: NDArray[np.float64],
    ) -> tuple[NDArray[np.bool_], NDArray[np.int8]]:
        """Apply conformal gating to determine tradeable samples.

        A sample is tradeable only if its prediction set is a singleton
        containing UP or DOWN (not FLAT alone).

        Args:
            probas: Probabilities (n_samples, 3) for [DOWN, FLAT, UP].

        Returns:
            Tuple of:
                should_trade: Boolean array (n_samples,). True if trade allowed.
                predicted_direction: Int8 array (n_samples,).
                    -1 for DOWN, +1 for UP, 0 for abstentions.

        Raises:
            RuntimeError: If predictor is not calibrated.
        """
        self._check_calibrated()

        pred_sets = self._build_prediction_sets(probas)
        n_samples = len(probas)

        should_trade = np.zeros(n_samples, dtype=np.bool_)
        predicted_direction = np.zeros(n_samples, dtype=np.int8)

        for i, pset in enumerate(pred_sets):
            frozen = frozenset(pset)
            if frozen in _TRADEABLE_SETS:
                should_trade[i] = True
                predicted_direction[i] = _DIRECTION_MAP[frozen]

        n_trades = int(should_trade.sum())
        n_abstain = n_samples - n_trades

        logger.debug(
            "conformal_gate_applied",
            n_samples=n_samples,
            n_trades=n_trades,
            n_abstain=n_abstain,
            trade_rate=round(n_trades / max(n_samples, 1), 4),
        )

        return should_trade, predicted_direction

    def update_alpha(
        self,
        probas: NDArray[np.float64],
        y_true: NDArray[np.int8],
    ) -> float:
        """Adaptively update alpha based on empirical coverage.

        If empirical coverage is below target → decrease alpha (widen sets).
        If empirical coverage is above target → increase alpha (tighten sets).

        Uses exponential moving average of coverage for stability.

        Args:
            probas: Probabilities (n_samples, 3) for [DOWN, FLAT, UP].
            y_true: True labels in {-1, 0, +1} encoding.

        Returns:
            Updated alpha value.

        Raises:
            RuntimeError: If predictor is not calibrated.
        """
        self._check_calibrated()

        if not self._config.adaptive:
            return self._state.current_alpha

        # Encode: -1 → 0, 0 → 1, +1 → 2
        y_encoded = y_true.astype(np.int32) + 1

        pred_sets = self._build_prediction_sets(probas)
        n_samples = len(probas)

        # Empirical coverage on this batch
        covered = np.mean(
            [y_encoded[i] in pred_sets[i] for i in range(n_samples)]
        )

        # EMA of coverage
        lr = self._config.adaptive_lr
        self._state.coverage_ema = (
            (1.0 - lr) * self._state.coverage_ema + lr * float(covered)
        )

        target_coverage = 1.0 - self._config.alpha
        coverage_gap = self._state.coverage_ema - target_coverage

        # Adjust alpha: if coverage too high, increase alpha (tighten);
        # if coverage too low, decrease alpha (widen)
        self._state.current_alpha = float(
            np.clip(
                self._state.current_alpha + lr * coverage_gap,
                0.01,  # Min alpha (99% coverage)
                0.50,  # Max alpha (50% coverage)
            )
        )

        # Recompute quantile threshold with new alpha
        scores = self._state.scores
        if scores is not None:
            n_cal = len(scores)
            adjusted_level = (
                np.ceil((n_cal + 1) * (1.0 - self._state.current_alpha)) / n_cal
            )
            adjusted_level = min(adjusted_level, 1.0)
            self._state.quantile_threshold = float(
                np.quantile(scores, adjusted_level)
            )

        self._state.empirical_coverage = float(covered)

        logger.debug(
            "conformal_alpha_updated",
            new_alpha=round(self._state.current_alpha, 4),
            empirical_coverage=round(float(covered), 4),
            coverage_ema=round(self._state.coverage_ema, 4),
            target_coverage=round(target_coverage, 4),
            quantile_threshold=round(self._state.quantile_threshold, 4),
        )

        return self._state.current_alpha

    def save(self, path: Path | str) -> None:
        """Save conformal predictor state to disk.

        Args:
            path: Base path for saving (extensions added automatically).

        Raises:
            RuntimeError: If predictor is not calibrated.
        """
        self._check_calibrated()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save scores as binary for efficiency
        scores_path = path.with_suffix(".scores.bin")
        if self._state.scores is not None:
            scores_path.write_bytes(self._state.scores.tobytes())

        # Save metadata as JSON
        meta = {
            "alpha": self._config.alpha,
            "adaptive": self._config.adaptive,
            "adaptive_lr": self._config.adaptive_lr,
            "min_calibration_size": self._config.min_calibration_size,
            "quantile_threshold": self._state.quantile_threshold,
            "n_calibration_samples": self._state.n_calibration_samples,
            "empirical_coverage": self._state.empirical_coverage,
            "current_alpha": self._state.current_alpha,
            "coverage_ema": self._state.coverage_ema,
            "n_classes": self._n_classes,
        }
        meta_path = path.with_suffix(".meta.json")
        meta_path.write_text(json.dumps(meta, indent=2))

        logger.info(
            "conformal_predictor_saved",
            path=str(path),
            n_calibration_samples=self._state.n_calibration_samples,
        )

    def load(self, path: Path | str) -> None:
        """Load conformal predictor state from disk.

        Args:
            path: Base path for loading (extensions added automatically).

        Raises:
            FileNotFoundError: If files don't exist.
        """
        path = Path(path)

        # Load metadata
        meta_path = path.with_suffix(".meta.json")
        meta = json.loads(meta_path.read_text())

        self._config = ConformalConfig(
            alpha=meta["alpha"],
            adaptive=meta["adaptive"],
            adaptive_lr=meta["adaptive_lr"],
            min_calibration_size=meta["min_calibration_size"],
        )
        self._n_classes = meta.get("n_classes", 3)

        # Load scores
        scores_path = path.with_suffix(".scores.bin")
        raw_bytes = scores_path.read_bytes()
        self._state.scores = np.frombuffer(raw_bytes, dtype=np.float64).copy()

        self._state.quantile_threshold = meta["quantile_threshold"]
        self._state.n_calibration_samples = meta["n_calibration_samples"]
        self._state.empirical_coverage = meta["empirical_coverage"]
        self._state.current_alpha = meta["current_alpha"]
        self._state.coverage_ema = meta.get("coverage_ema", 1.0 - meta["alpha"])

        logger.info(
            "conformal_predictor_loaded",
            path=str(path),
            n_calibration_samples=self._state.n_calibration_samples,
            current_alpha=round(self._state.current_alpha, 4),
        )

    def _build_prediction_sets(
        self,
        probas: NDArray[np.float64],
    ) -> list[set[int]]:
        """Build prediction sets from probabilities.

        Include class c if 1 - probas[i, c] <= quantile_threshold,
        i.e., probas[i, c] >= 1 - quantile_threshold.
        """
        threshold = 1.0 - self._state.quantile_threshold
        n_samples = len(probas)
        prediction_sets: list[set[int]] = []

        for i in range(n_samples):
            pset: set[int] = set()
            for c in range(self._n_classes):
                if probas[i, c] >= threshold:
                    pset.add(c)
            prediction_sets.append(pset)

        return prediction_sets

    def _check_calibrated(self) -> None:
        """Raise if not calibrated."""
        if not self.is_calibrated:
            msg = "ConformalPredictor not calibrated. Call calibrate() first."
            raise RuntimeError(msg)
