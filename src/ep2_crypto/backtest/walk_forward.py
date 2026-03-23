"""Purged walk-forward cross-validation with embargo.

Configuration (5-min BTC):
  - Training window: 14 days (4,032 bars) — sliding, NOT expanding
  - Test window: 1 day (288 bars)
  - Step size: 1 day (288 bars)
  - Purge: 18 bars (90 min) — label_horizon + feature_lookback
  - Embargo: 12 bars (60 min) — 2x label horizon
  - Window: Sliding (crypto non-stationarity makes old data harmful)

Nested inner loop for hyperparameter selection:
  - 3 inner folds within training set
  - 20% validation from END of training (with purge gap)

Walk-forward auditor for automated leak detection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import structlog

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = structlog.get_logger(__name__)

BARS_PER_DAY = 288


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class WalkForwardConfig:
    """Walk-forward validation parameters."""

    train_days: int = 14
    test_days: int = 1
    step_days: int = 1
    purge_bars: int = 18  # 90 min at 5-min bars
    embargo_bars: int = 12  # 60 min at 5-min bars
    expanding: bool = False  # sliding by default (NEVER expanding for crypto)

    # Inner loop (hyperparameter selection)
    inner_n_folds: int = 3
    inner_val_fraction: float = 0.20  # 20% of training for validation

    @property
    def train_bars(self) -> int:
        return self.train_days * BARS_PER_DAY

    @property
    def test_bars(self) -> int:
        return self.test_days * BARS_PER_DAY

    @property
    def step_bars(self) -> int:
        return self.step_days * BARS_PER_DAY


# ---------------------------------------------------------------------------
# Fold data structures
# ---------------------------------------------------------------------------
@dataclass
class Fold:
    """A single outer fold with train/test indices."""

    fold_idx: int
    train_start: int
    train_end: int  # exclusive
    test_start: int
    test_end: int  # exclusive

    @property
    def train_size(self) -> int:
        return self.train_end - self.train_start

    @property
    def test_size(self) -> int:
        return self.test_end - self.test_start


@dataclass
class InnerFold:
    """An inner fold for hyperparameter selection within training."""

    inner_idx: int
    train_start: int
    train_end: int
    val_start: int
    val_end: int


# ---------------------------------------------------------------------------
# Walk-Forward Validator
# ---------------------------------------------------------------------------
class WalkForwardValidator:
    """Generates purged walk-forward folds with embargo.

    Usage:
        wf = WalkForwardValidator(n_bars=50000)
        for fold in wf.folds():
            train_data = data[fold.train_start:fold.train_end]
            test_data = data[fold.test_start:fold.test_end]
            # train, predict, evaluate
    """

    def __init__(
        self,
        n_bars: int,
        config: WalkForwardConfig | None = None,
    ) -> None:
        self._n = n_bars
        self._config = config or WalkForwardConfig()
        self._folds: list[Fold] = []
        self._generate_folds()

    @property
    def n_folds(self) -> int:
        return len(self._folds)

    def folds(self) -> list[Fold]:
        """Return all outer folds."""
        return self._folds

    def _generate_folds(self) -> None:
        """Generate all walk-forward folds."""
        cfg = self._config
        min_required = cfg.train_bars + cfg.purge_bars + cfg.test_bars

        if self._n < min_required:
            logger.warning(
                "insufficient_data_for_walk_forward",
                n_bars=self._n,
                min_required=min_required,
            )
            return

        fold_idx = 0
        # First fold starts at the beginning
        train_start = 0

        while True:
            if cfg.expanding:
                train_end = train_start + cfg.train_bars + fold_idx * cfg.step_bars
            else:
                train_end = train_start + cfg.train_bars

            # Purge gap after training
            test_start = train_end + cfg.purge_bars

            # Test window
            test_end = test_start + cfg.test_bars

            # Embargo after test (for next fold's training start)
            if test_end > self._n:
                break

            self._folds.append(
                Fold(
                    fold_idx=fold_idx,
                    train_start=train_start,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end,
                )
            )

            fold_idx += 1

            # Slide training window forward
            if cfg.expanding:
                # For expanding: keep train_start fixed, advance test
                if test_end + cfg.embargo_bars + cfg.step_bars > self._n:
                    break
            else:
                # Enforce embargo: next train_end must be >= current test_end + embargo_bars.
                # This prevents the training window from including bars in the post-test
                # embargo zone of the previous fold.
                min_next_train_end = test_end + cfg.embargo_bars
                step_needed = min_next_train_end - (train_start + cfg.train_bars)
                train_start += max(cfg.step_bars, step_needed)

            # Check if next fold would exceed data
            next_test_end = (
                (train_start + cfg.train_bars + cfg.purge_bars + cfg.test_bars)
                if not cfg.expanding
                else (train_end + cfg.step_bars + cfg.purge_bars + cfg.test_bars)
            )
            if next_test_end > self._n:
                break

        logger.info(
            "walk_forward_folds_generated",
            n_folds=len(self._folds),
            n_bars=self._n,
            train_bars=cfg.train_bars,
            test_bars=cfg.test_bars,
            purge_bars=cfg.purge_bars,
            embargo_bars=cfg.embargo_bars,
        )

    def inner_folds(self, outer_fold: Fold) -> list[InnerFold]:
        """Generate inner folds for hyperparameter selection.

        Validation is carved from the END of the training set with a
        purge gap between inner train and inner validation.
        """
        cfg = self._config
        train_len = outer_fold.train_size
        val_size = max(1, int(train_len * cfg.inner_val_fraction))

        inner_folds: list[InnerFold] = []

        for i in range(cfg.inner_n_folds):
            # Each inner fold uses a different portion of the training data
            # Validation is at the END of the training portion
            inner_train_end = outer_fold.train_end - val_size * (cfg.inner_n_folds - i)
            inner_val_start = inner_train_end + cfg.purge_bars
            inner_val_end = inner_val_start + val_size

            if inner_val_end > outer_fold.train_end:
                inner_val_end = outer_fold.train_end

            if inner_train_end <= outer_fold.train_start:
                continue

            if inner_val_start >= inner_val_end:
                continue

            inner_folds.append(
                InnerFold(
                    inner_idx=i,
                    train_start=outer_fold.train_start,
                    train_end=inner_train_end,
                    val_start=inner_val_start,
                    val_end=inner_val_end,
                )
            )

        return inner_folds

    def concatenated_oos_indices(self) -> NDArray[np.int64]:
        """Return concatenated out-of-sample indices from all folds.

        These are the test indices across all folds — the "true" OOS.
        """
        indices: list[int] = []
        for fold in self._folds:
            indices.extend(range(fold.test_start, fold.test_end))
        return np.array(indices, dtype=np.int64)


# ---------------------------------------------------------------------------
# Walk-Forward Auditor
# ---------------------------------------------------------------------------
@dataclass
class AuditResult:
    """Result of walk-forward audit checks."""

    passed: bool
    checks: dict[str, bool] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


class WalkForwardAuditor:
    """Automated leak detection for walk-forward validation.

    Checks:
      1. No train/test index overlap
      2. Purge gap is sufficient (>= config)
      3. Temporal ordering (train before test)
      4. Consistent fold sizes
      5. No duplicate test indices across folds
      6. Embargo respected between folds
    """

    def __init__(self, config: WalkForwardConfig | None = None) -> None:
        self._config = config or WalkForwardConfig()

    def audit(self, folds: list[Fold]) -> AuditResult:
        """Run all audit checks on a list of folds.

        Returns:
            AuditResult with pass/fail and details.
        """
        checks: dict[str, bool] = {}
        warnings: list[str] = []
        errors: list[str] = []

        # 1. No overlap
        checks["no_train_test_overlap"] = self._check_no_overlap(folds, errors)

        # 2. Purge gap sufficient
        checks["purge_sufficient"] = self._check_purge(folds, errors)

        # 3. Temporal ordering
        checks["temporal_ordering"] = self._check_temporal_order(folds, errors)

        # 4. Consistent fold sizes
        checks["consistent_sizes"] = self._check_consistent_sizes(folds, warnings)

        # 5. No duplicate test indices
        checks["no_duplicate_test"] = self._check_no_duplicate_test(folds, errors)

        # 6. Embargo between consecutive folds
        checks["embargo_respected"] = self._check_embargo(folds, warnings)

        passed = all(checks.values()) and len(errors) == 0

        return AuditResult(
            passed=passed,
            checks=checks,
            warnings=warnings,
            errors=errors,
        )

    def _check_no_overlap(
        self,
        folds: list[Fold],
        errors: list[str],
    ) -> bool:
        """Check no training indices overlap with test indices."""
        ok = True
        for fold in folds:
            train_set = set(range(fold.train_start, fold.train_end))
            test_set = set(range(fold.test_start, fold.test_end))
            overlap = train_set & test_set
            if overlap:
                errors.append(f"Fold {fold.fold_idx}: {len(overlap)} overlapping indices")
                ok = False
        return ok

    def _check_purge(
        self,
        folds: list[Fold],
        errors: list[str],
    ) -> bool:
        """Check purge gap between train end and test start."""
        ok = True
        for fold in folds:
            gap = fold.test_start - fold.train_end
            if gap < self._config.purge_bars:
                errors.append(
                    f"Fold {fold.fold_idx}: purge gap {gap} < required {self._config.purge_bars}"
                )
                ok = False
        return ok

    def _check_temporal_order(
        self,
        folds: list[Fold],
        errors: list[str],
    ) -> bool:
        """Check train always comes before test."""
        ok = True
        for fold in folds:
            if fold.train_end > fold.test_start:
                errors.append(
                    f"Fold {fold.fold_idx}: train_end {fold.train_end}"
                    f" > test_start {fold.test_start}"
                )
                ok = False
        return ok

    def _check_consistent_sizes(
        self,
        folds: list[Fold],
        warnings: list[str],
    ) -> bool:
        """Check fold sizes are consistent."""
        if not folds:
            return True

        train_sizes = [f.train_size for f in folds]
        test_sizes = [f.test_size for f in folds]

        train_cv = np.std(train_sizes) / max(np.mean(train_sizes), 1)
        test_cv = np.std(test_sizes) / max(np.mean(test_sizes), 1)

        ok = True
        if train_cv > 0.1:
            warnings.append(f"Train size CV = {train_cv:.3f} (>0.1), sizes vary significantly")
            ok = False
        if test_cv > 0.1:
            warnings.append(f"Test size CV = {test_cv:.3f} (>0.1), sizes vary significantly")
            ok = False

        return ok

    def _check_no_duplicate_test(
        self,
        folds: list[Fold],
        errors: list[str],
    ) -> bool:
        """Check no test index appears in multiple folds."""
        seen: set[int] = set()
        for fold in folds:
            test_range = set(range(fold.test_start, fold.test_end))
            overlap = seen & test_range
            if overlap:
                errors.append(
                    f"Fold {fold.fold_idx}: {len(overlap)} test indices shared with earlier folds"
                )
                return False
            seen |= test_range
        return True

    def _check_embargo(
        self,
        folds: list[Fold],
        warnings: list[str],
    ) -> bool:
        """Check embargo between consecutive folds.

        The embargo requires that next_fold.train_end >= current.test_end + embargo_bars.
        This ensures the post-test embargo zone is excluded from the next fold's training.
        """
        ok = True
        for i in range(len(folds) - 1):
            current = folds[i]
            next_fold = folds[i + 1]
            required_train_end = current.test_end + self._config.embargo_bars
            if next_fold.train_end < required_train_end:
                warnings.append(
                    f"Folds {i}->{i + 1}: embargo violated — "
                    f"train_end {next_fold.train_end} < required {required_train_end} "
                    f"(test_end {current.test_end} + embargo {self._config.embargo_bars})"
                )
                ok = False
        return ok
