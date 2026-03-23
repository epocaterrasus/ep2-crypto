"""Tests for walk-forward validation."""

from __future__ import annotations

from ep2_crypto.backtest.walk_forward import (
    BARS_PER_DAY,
    Fold,
    WalkForwardAuditor,
    WalkForwardConfig,
    WalkForwardValidator,
)


# ---------------------------------------------------------------------------
# WalkForwardConfig
# ---------------------------------------------------------------------------
class TestWalkForwardConfig:
    def test_default_config(self) -> None:
        cfg = WalkForwardConfig()
        assert cfg.train_days == 14
        assert cfg.test_days == 1
        assert cfg.purge_bars == 18
        assert cfg.embargo_bars == 12
        assert cfg.expanding is False

    def test_bar_calculations(self) -> None:
        cfg = WalkForwardConfig()
        assert cfg.train_bars == 14 * 288  # 4032
        assert cfg.test_bars == 288
        assert cfg.step_bars == 288


# ---------------------------------------------------------------------------
# WalkForwardValidator — fold generation
# ---------------------------------------------------------------------------
class TestFoldGeneration:
    def test_generates_folds(self) -> None:
        """With enough data, should generate multiple folds."""
        n = 30 * BARS_PER_DAY  # 30 days
        wf = WalkForwardValidator(n)
        assert wf.n_folds > 0

    def test_insufficient_data_no_folds(self) -> None:
        """Too little data → 0 folds."""
        n = 100  # not enough for 14-day train + 1-day test
        wf = WalkForwardValidator(n)
        assert wf.n_folds == 0

    def test_fold_count_matches_data_length(self) -> None:
        """Number of folds should be approximately (n - train) / step."""
        n = 60 * BARS_PER_DAY  # 60 days
        cfg = WalkForwardConfig(train_days=14, test_days=1, step_days=1)
        wf = WalkForwardValidator(n, config=cfg)
        # Approximately 60 - 14 - purge_days ≈ 45 folds
        assert wf.n_folds >= 30
        assert wf.n_folds <= 50

    def test_sliding_window_train_sizes_consistent(self) -> None:
        """Sliding window → all train sizes should be equal."""
        n = 30 * BARS_PER_DAY
        wf = WalkForwardValidator(n)
        folds = wf.folds()
        if len(folds) > 1:
            sizes = [f.train_size for f in folds]
            assert min(sizes) == max(sizes)

    def test_test_sizes_consistent(self) -> None:
        """All test windows should be 1 day."""
        n = 30 * BARS_PER_DAY
        wf = WalkForwardValidator(n)
        folds = wf.folds()
        for fold in folds:
            assert fold.test_size == 288

    def test_no_train_test_overlap(self) -> None:
        """Train and test indices must never overlap."""
        n = 30 * BARS_PER_DAY
        wf = WalkForwardValidator(n)
        for fold in wf.folds():
            assert fold.train_end <= fold.test_start

    def test_purge_gap_respected(self) -> None:
        """Gap between train end and test start >= purge_bars."""
        n = 30 * BARS_PER_DAY
        cfg = WalkForwardConfig(purge_bars=18)
        wf = WalkForwardValidator(n, config=cfg)
        for fold in wf.folds():
            gap = fold.test_start - fold.train_end
            assert gap >= 18

    def test_temporal_ordering(self) -> None:
        """Train start < train end < test start < test end."""
        n = 30 * BARS_PER_DAY
        wf = WalkForwardValidator(n)
        for fold in wf.folds():
            assert fold.train_start < fold.train_end
            assert fold.train_end < fold.test_start
            assert fold.test_start < fold.test_end

    def test_folds_advance_monotonically(self) -> None:
        """Each fold's train start should advance by step_bars."""
        n = 30 * BARS_PER_DAY
        cfg = WalkForwardConfig(step_days=1)
        wf = WalkForwardValidator(n, config=cfg)
        folds = wf.folds()
        for i in range(1, len(folds)):
            assert folds[i].train_start > folds[i - 1].train_start

    def test_no_duplicate_test_indices(self) -> None:
        """Test indices should not overlap across folds."""
        n = 30 * BARS_PER_DAY
        wf = WalkForwardValidator(n)
        all_test = set()
        for fold in wf.folds():
            test_indices = set(range(fold.test_start, fold.test_end))
            overlap = all_test & test_indices
            assert len(overlap) == 0, f"Fold {fold.fold_idx} overlaps"
            all_test |= test_indices


# ---------------------------------------------------------------------------
# Inner folds
# ---------------------------------------------------------------------------
class TestInnerFolds:
    def test_inner_fold_generation(self) -> None:
        """Inner folds should be generated within training set."""
        n = 30 * BARS_PER_DAY
        wf = WalkForwardValidator(n)
        folds = wf.folds()
        if folds:
            inner = wf.inner_folds(folds[0])
            assert len(inner) > 0

    def test_inner_val_within_training(self) -> None:
        """Inner validation must be within outer training bounds."""
        n = 30 * BARS_PER_DAY
        wf = WalkForwardValidator(n)
        folds = wf.folds()
        if folds:
            for inner in wf.inner_folds(folds[0]):
                assert inner.train_start >= folds[0].train_start
                assert inner.val_end <= folds[0].train_end

    def test_inner_train_before_val(self) -> None:
        """Inner training must come before inner validation."""
        n = 30 * BARS_PER_DAY
        wf = WalkForwardValidator(n)
        folds = wf.folds()
        if folds:
            for inner in wf.inner_folds(folds[0]):
                assert inner.train_end < inner.val_start

    def test_inner_purge_respected(self) -> None:
        """Purge gap between inner train and inner val."""
        n = 30 * BARS_PER_DAY
        cfg = WalkForwardConfig(purge_bars=18)
        wf = WalkForwardValidator(n, config=cfg)
        folds = wf.folds()
        if folds:
            for inner in wf.inner_folds(folds[0]):
                gap = inner.val_start - inner.train_end
                assert gap >= cfg.purge_bars


# ---------------------------------------------------------------------------
# Concatenated OOS
# ---------------------------------------------------------------------------
class TestConcatenatedOOS:
    def test_oos_indices_contiguous(self) -> None:
        """OOS indices should cover test regions."""
        n = 30 * BARS_PER_DAY
        wf = WalkForwardValidator(n)
        oos = wf.concatenated_oos_indices()
        assert len(oos) > 0
        # All should be valid indices
        assert oos.min() >= 0
        assert oos.max() < n

    def test_oos_no_duplicates(self) -> None:
        n = 30 * BARS_PER_DAY
        wf = WalkForwardValidator(n)
        oos = wf.concatenated_oos_indices()
        assert len(oos) == len(set(oos))


# ---------------------------------------------------------------------------
# WalkForwardAuditor
# ---------------------------------------------------------------------------
class TestWalkForwardAuditor:
    def test_valid_folds_pass_audit(self) -> None:
        """Correctly generated folds should pass all checks."""
        n = 30 * BARS_PER_DAY
        cfg = WalkForwardConfig()
        wf = WalkForwardValidator(n, config=cfg)
        auditor = WalkForwardAuditor(config=cfg)
        result = auditor.audit(wf.folds())
        assert result.passed is True
        assert len(result.errors) == 0

    def test_overlap_detected(self) -> None:
        """Audit should detect train/test overlap."""
        bad_folds = [
            Fold(
                fold_idx=0, train_start=0, train_end=200, test_start=190, test_end=300
            ),  # overlap!
        ]
        auditor = WalkForwardAuditor()
        result = auditor.audit(bad_folds)
        assert result.checks["no_train_test_overlap"] is False
        assert len(result.errors) > 0

    def test_insufficient_purge_detected(self) -> None:
        """Audit should detect insufficient purge gap."""
        bad_folds = [
            Fold(
                fold_idx=0, train_start=0, train_end=4032, test_start=4035, test_end=4320
            ),  # only 3-bar gap
        ]
        auditor = WalkForwardAuditor(WalkForwardConfig(purge_bars=18))
        result = auditor.audit(bad_folds)
        assert result.checks["purge_sufficient"] is False

    def test_temporal_violation_detected(self) -> None:
        """Audit should detect train_end > test_start."""
        bad_folds = [
            Fold(fold_idx=0, train_start=0, train_end=5000, test_start=4000, test_end=5000),
        ]
        auditor = WalkForwardAuditor()
        result = auditor.audit(bad_folds)
        assert result.checks["temporal_ordering"] is False

    def test_duplicate_test_detected(self) -> None:
        """Audit should detect duplicate test indices across folds."""
        bad_folds = [
            Fold(fold_idx=0, train_start=0, train_end=4032, test_start=4050, test_end=4338),
            Fold(
                fold_idx=1, train_start=288, train_end=4320, test_start=4050, test_end=4338
            ),  # same test!
        ]
        auditor = WalkForwardAuditor()
        result = auditor.audit(bad_folds)
        assert result.checks["no_duplicate_test"] is False

    def test_empty_folds_pass(self) -> None:
        """Empty fold list should pass (nothing to check)."""
        auditor = WalkForwardAuditor()
        result = auditor.audit([])
        assert result.passed is True

    def test_all_checks_present(self) -> None:
        """All expected checks should be in the result."""
        n = 30 * BARS_PER_DAY
        cfg = WalkForwardConfig()
        wf = WalkForwardValidator(n, config=cfg)
        auditor = WalkForwardAuditor(config=cfg)
        result = auditor.audit(wf.folds())
        expected_checks = [
            "no_train_test_overlap",
            "purge_sufficient",
            "temporal_ordering",
            "consistent_sizes",
            "no_duplicate_test",
            "embargo_respected",
        ]
        for check in expected_checks:
            assert check in result.checks
