"""Tests for feature rotation tracker."""

from __future__ import annotations

import pytest

from ep2_crypto.monitoring.feature_rotation import (
    FeatureRotationTracker,
)


class TestFeatureRegistration:
    def test_register_single(self) -> None:
        tracker = FeatureRotationTracker()
        tracker.register_feature("obi")
        assert tracker.feature_count == 1

    def test_register_batch_with_importances(self) -> None:
        tracker = FeatureRotationTracker()
        tracker.register_features(importances={"obi": 0.15, "rsi": 0.10, "vol": 0.08})
        assert tracker.feature_count == 3

    def test_register_batch_with_names(self) -> None:
        tracker = FeatureRotationTracker()
        tracker.register_features(names=["obi", "rsi", "vol"])
        assert tracker.feature_count == 3

    def test_initial_weight_is_one(self) -> None:
        tracker = FeatureRotationTracker()
        tracker.register_feature("obi")
        assert tracker.get_weight("obi") == 1.0

    def test_unknown_feature_weight_is_one(self) -> None:
        tracker = FeatureRotationTracker()
        assert tracker.get_weight("nonexistent") == 1.0


class TestDailyPSI:
    @pytest.fixture()
    def tracker(self) -> FeatureRotationTracker:
        t = FeatureRotationTracker(
            drift_threshold=0.3,
            consecutive_days_to_flag=3,
            downweight_factor=0.5,
        )
        t.register_features(names=["obi", "rsi", "vol"])
        return t

    def test_no_flag_below_threshold(self, tracker: FeatureRotationTracker) -> None:
        for _ in range(10):
            flagged = tracker.record_daily_psi({"obi": 0.1, "rsi": 0.1, "vol": 0.1})
            assert flagged == []
        assert tracker.get_flagged_features() == []

    def test_flag_after_consecutive_drift(self, tracker: FeatureRotationTracker) -> None:
        # 3 consecutive days above threshold for obi
        for _ in range(3):
            flagged = tracker.record_daily_psi({"obi": 0.5, "rsi": 0.1, "vol": 0.1})
        assert "obi" in flagged
        assert "obi" in tracker.get_flagged_features()

    def test_downweight_on_flag(self, tracker: FeatureRotationTracker) -> None:
        for _ in range(3):
            tracker.record_daily_psi({"obi": 0.5, "rsi": 0.1, "vol": 0.1})
        assert tracker.get_weight("obi") == 0.5
        assert tracker.get_weight("rsi") == 1.0

    def test_recovery_on_return_to_normal(self, tracker: FeatureRotationTracker) -> None:
        # Flag it
        for _ in range(3):
            tracker.record_daily_psi({"obi": 0.5, "rsi": 0.1, "vol": 0.1})
        assert tracker.get_weight("obi") == 0.5
        # Recover
        tracker.record_daily_psi({"obi": 0.05, "rsi": 0.1, "vol": 0.1})
        assert tracker.get_weight("obi") == 1.0
        assert "obi" not in tracker.get_flagged_features()

    def test_interrupted_streak_resets(self, tracker: FeatureRotationTracker) -> None:
        tracker.record_daily_psi({"obi": 0.5, "rsi": 0.1, "vol": 0.1})
        tracker.record_daily_psi({"obi": 0.5, "rsi": 0.1, "vol": 0.1})
        # Break the streak
        tracker.record_daily_psi({"obi": 0.1, "rsi": 0.1, "vol": 0.1})
        tracker.record_daily_psi({"obi": 0.5, "rsi": 0.1, "vol": 0.1})
        assert "obi" not in tracker.get_flagged_features()

    def test_multiple_features_flagged(self, tracker: FeatureRotationTracker) -> None:
        for _ in range(3):
            tracker.record_daily_psi({"obi": 0.5, "rsi": 0.5, "vol": 0.1})
        assert set(tracker.get_flagged_features()) == {"obi", "rsi"}

    def test_auto_register_unknown_feature(self, tracker: FeatureRotationTracker) -> None:
        tracker.record_daily_psi({"new_feat": 0.1})
        assert tracker.feature_count == 4
        assert tracker.get_weight("new_feat") == 1.0

    def test_flagged_not_re_reported(self, tracker: FeatureRotationTracker) -> None:
        for _ in range(3):
            flagged = tracker.record_daily_psi({"obi": 0.5})
        assert "obi" in flagged
        # Already flagged — not newly flagged
        flagged = tracker.record_daily_psi({"obi": 0.5})
        assert "obi" not in flagged


class TestImportance:
    def test_record_importance(self) -> None:
        tracker = FeatureRotationTracker()
        tracker.register_feature("obi", initial_importance=0.15)
        tracker.record_importance({"obi": 0.12})
        state = tracker.get_feature_state("obi")
        assert state is not None
        assert len(state.importance_history) == 2


class TestWeights:
    def test_get_weights_all(self) -> None:
        tracker = FeatureRotationTracker(
            drift_threshold=0.3,
            consecutive_days_to_flag=2,
            downweight_factor=0.5,
        )
        tracker.register_features(names=["obi", "rsi"])
        for _ in range(2):
            tracker.record_daily_psi({"obi": 0.5, "rsi": 0.1})
        weights = tracker.get_weights()
        assert weights["obi"] == 0.5
        assert weights["rsi"] == 1.0

    def test_get_downweighted_features(self) -> None:
        tracker = FeatureRotationTracker(
            drift_threshold=0.3,
            consecutive_days_to_flag=2,
            downweight_factor=0.5,
        )
        tracker.register_features(names=["obi", "rsi", "vol"])
        for _ in range(2):
            tracker.record_daily_psi({"obi": 0.5, "rsi": 0.1, "vol": 0.5})
        dw = tracker.get_downweighted_features()
        assert "obi" in dw
        assert "vol" in dw
        assert "rsi" not in dw


class TestMonthlyReport:
    def test_report_generation(self) -> None:
        tracker = FeatureRotationTracker(
            drift_threshold=0.3,
            consecutive_days_to_flag=2,
            downweight_factor=0.5,
        )
        tracker.register_features(
            importances={"obi": 0.15, "rsi": 0.10, "vol": 0.08}
        )
        tracker.record_importance({"obi": 0.10, "rsi": 0.12, "vol": 0.08})
        for _ in range(2):
            tracker.record_daily_psi({"obi": 0.5, "rsi": 0.1, "vol": 0.1})
        report = tracker.generate_monthly_report()
        assert report.total_features == 3
        assert "obi" in report.flagged_features
        assert "obi" in report.downweighted_features
        assert "obi" in report.importance_changes
        assert report.importance_changes["obi"] == pytest.approx(-0.05)
        assert report.importance_changes["rsi"] == pytest.approx(0.02)


class TestReset:
    def test_reset_clears_all(self) -> None:
        tracker = FeatureRotationTracker()
        tracker.register_features(names=["obi", "rsi"])
        tracker.reset()
        assert tracker.feature_count == 0
