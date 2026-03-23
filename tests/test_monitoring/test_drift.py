"""Tests for feature drift detection (PSI)."""

from __future__ import annotations

import numpy as np
import pytest

from ep2_crypto.monitoring.drift import (
    DailyDriftSummary,
    FeatureDriftDetector,
    classify_psi,
    compute_psi,
)


class TestComputePSI:
    def test_identical_distributions_zero_psi(self) -> None:
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 1000)
        psi = compute_psi(data, data)
        assert psi < 0.01

    def test_similar_distributions_low_psi(self) -> None:
        rng = np.random.default_rng(42)
        ref = rng.normal(0, 1, 1000)
        cur = rng.normal(0.05, 1, 1000)
        psi = compute_psi(ref, cur)
        assert psi < 0.1

    def test_different_distributions_high_psi(self) -> None:
        rng = np.random.default_rng(42)
        ref = rng.normal(0, 1, 1000)
        cur = rng.normal(3, 1, 1000)
        psi = compute_psi(ref, cur)
        assert psi > 0.2

    def test_psi_non_negative(self) -> None:
        rng = np.random.default_rng(42)
        ref = rng.normal(0, 1, 500)
        cur = rng.normal(1, 2, 500)
        psi = compute_psi(ref, cur)
        assert psi >= 0.0

    def test_insufficient_data_returns_zero(self) -> None:
        ref = np.array([1.0, 2.0])
        cur = np.array([1.0, 2.0])
        psi = compute_psi(ref, cur, n_bins=10)
        assert psi == 0.0

    def test_custom_bins(self) -> None:
        rng = np.random.default_rng(42)
        ref = rng.normal(0, 1, 500)
        cur = rng.normal(2, 1, 500)
        psi_5 = compute_psi(ref, cur, n_bins=5)
        psi_20 = compute_psi(ref, cur, n_bins=20)
        # Both should detect drift, exact values may differ
        assert psi_5 > 0.1
        assert psi_20 > 0.1

    def test_constant_reference_returns_zero(self) -> None:
        ref = np.ones(100)
        cur = np.ones(100)
        psi = compute_psi(ref, cur)
        assert psi == 0.0


class TestClassifyPSI:
    def test_none_severity(self) -> None:
        assert classify_psi(0.05) == "none"

    def test_moderate_severity(self) -> None:
        assert classify_psi(0.15) == "moderate"

    def test_significant_severity(self) -> None:
        assert classify_psi(0.25) == "significant"

    def test_critical_severity(self) -> None:
        assert classify_psi(0.35) == "critical"

    def test_zero_psi(self) -> None:
        assert classify_psi(0.0) == "none"

    def test_boundary_at_0_1(self) -> None:
        assert classify_psi(0.1) == "moderate"

    def test_boundary_at_0_2(self) -> None:
        assert classify_psi(0.2) == "significant"

    def test_boundary_at_0_3(self) -> None:
        assert classify_psi(0.3) == "critical"


class TestFeatureDriftDetector:
    @pytest.fixture()
    def detector(self) -> FeatureDriftDetector:
        return FeatureDriftDetector(n_bins=10, alert_threshold=0.2, window_size=500)

    def test_set_reference(self, detector: FeatureDriftDetector) -> None:
        rng = np.random.default_rng(42)
        detector.set_reference("obi", rng.normal(0, 1, 500))
        assert "obi" in detector.feature_names

    def test_set_references_batch(self, detector: FeatureDriftDetector) -> None:
        rng = np.random.default_rng(42)
        detector.set_references_batch(
            {
                "obi": rng.normal(0, 1, 500),
                "rsi": rng.uniform(20, 80, 500),
            }
        )
        assert detector.feature_names == ["obi", "rsi"]

    def test_update_adds_to_buffer(self, detector: FeatureDriftDetector) -> None:
        rng = np.random.default_rng(42)
        detector.set_reference("obi", rng.normal(0, 1, 500))
        for _ in range(10):
            detector.update({"obi": float(rng.normal(0, 1))})
        report = detector.compute_drift("obi")
        assert report.current_bins == 10

    def test_no_drift_same_distribution(self, detector: FeatureDriftDetector) -> None:
        rng = np.random.default_rng(42)
        ref_data = rng.normal(0, 1, 500)
        detector.set_reference("obi", ref_data)
        for val in rng.normal(0, 1, 200):
            detector.update({"obi": float(val)})
        report = detector.compute_drift("obi")
        assert not report.is_drifted
        assert report.psi < 0.2

    def test_drift_detected_different_distribution(self, detector: FeatureDriftDetector) -> None:
        rng = np.random.default_rng(42)
        detector.set_reference("obi", rng.normal(0, 1, 500))
        for val in rng.normal(5, 1, 300):
            detector.update({"obi": float(val)})
        report = detector.compute_drift("obi")
        assert report.is_drifted
        assert report.psi > 0.2
        assert report.severity in ("significant", "critical")

    def test_unknown_feature_returns_safe_report(self, detector: FeatureDriftDetector) -> None:
        report = detector.compute_drift("nonexistent")
        assert report.psi == 0.0
        assert not report.is_drifted

    def test_compute_all_drift(self, detector: FeatureDriftDetector) -> None:
        rng = np.random.default_rng(42)
        detector.set_references_batch(
            {
                "obi": rng.normal(0, 1, 500),
                "rsi": rng.uniform(20, 80, 500),
                "vol": rng.lognormal(0, 0.5, 500),
            }
        )
        # Same distribution for all
        for _ in range(200):
            detector.update(
                {
                    "obi": float(rng.normal(0, 1)),
                    "rsi": float(rng.uniform(20, 80)),
                    "vol": float(rng.lognormal(0, 0.5)),
                }
            )
        reports = detector.compute_all_drift()
        assert len(reports) == 3
        assert all(not r.is_drifted for r in reports)

    def test_get_drifted_features(self, detector: FeatureDriftDetector) -> None:
        rng = np.random.default_rng(42)
        detector.set_references_batch(
            {
                "stable": rng.normal(0, 1, 500),
                "drifted": rng.normal(0, 1, 500),
            }
        )
        for _ in range(300):
            detector.update(
                {
                    "stable": float(rng.normal(0, 1)),
                    "drifted": float(rng.normal(5, 1)),
                }
            )
        detector.compute_all_drift()
        drifted = detector.get_drifted_features()
        assert "drifted" in drifted
        assert "stable" not in drifted

    def test_get_psi(self, detector: FeatureDriftDetector) -> None:
        rng = np.random.default_rng(42)
        detector.set_reference("obi", rng.normal(0, 1, 500))
        for val in rng.normal(0, 1, 200):
            detector.update({"obi": float(val)})
        detector.compute_drift("obi")
        psi = detector.get_psi("obi")
        assert psi is not None
        assert psi >= 0.0

    def test_get_psi_unknown_feature(self, detector: FeatureDriftDetector) -> None:
        assert detector.get_psi("nonexistent") is None

    def test_buffer_bounded(self, detector: FeatureDriftDetector) -> None:
        rng = np.random.default_rng(42)
        detector.set_reference("obi", rng.normal(0, 1, 500))
        # Add more than window_size
        for _ in range(600):
            detector.update({"obi": float(rng.normal(0, 1))})
        report = detector.compute_drift("obi")
        assert report.current_bins == 500  # Bounded at window_size

    def test_reset_buffer_single(self, detector: FeatureDriftDetector) -> None:
        rng = np.random.default_rng(42)
        detector.set_references_batch(
            {
                "obi": rng.normal(0, 1, 500),
                "rsi": rng.uniform(20, 80, 500),
            }
        )
        for _ in range(100):
            detector.update(
                {
                    "obi": float(rng.normal(0, 1)),
                    "rsi": float(rng.uniform(20, 80)),
                }
            )
        detector.reset_buffer("obi")
        obi_report = detector.compute_drift("obi")
        rsi_report = detector.compute_drift("rsi")
        assert obi_report.current_bins == 0
        assert rsi_report.current_bins == 100

    def test_reset_buffer_all(self, detector: FeatureDriftDetector) -> None:
        rng = np.random.default_rng(42)
        detector.set_references_batch(
            {
                "obi": rng.normal(0, 1, 500),
                "rsi": rng.uniform(20, 80, 500),
            }
        )
        for _ in range(100):
            detector.update(
                {
                    "obi": float(rng.normal(0, 1)),
                    "rsi": float(rng.uniform(20, 80)),
                }
            )
        detector.reset_buffer()
        for name in detector.feature_names:
            report = detector.compute_drift(name)
            assert report.current_bins == 0


class TestDailyDriftSummary:
    def test_generate_report_no_drift(self) -> None:
        rng = np.random.default_rng(42)
        detector = FeatureDriftDetector(n_bins=10, window_size=500)
        detector.set_references_batch(
            {
                "obi": rng.normal(0, 1, 500),
                "rsi": rng.uniform(20, 80, 500),
            }
        )
        for _ in range(200):
            detector.update(
                {
                    "obi": float(rng.normal(0, 1)),
                    "rsi": float(rng.uniform(20, 80)),
                }
            )
        summary = detector.generate_daily_report(timestamp_ms=1_000_000)
        assert summary.total_features == 2
        assert summary.drifted_features == 0
        assert not summary.any_alert
        assert summary.drift_ratio == 0.0

    def test_generate_report_with_drift(self) -> None:
        rng = np.random.default_rng(42)
        detector = FeatureDriftDetector(n_bins=10, window_size=500)
        detector.set_references_batch(
            {
                "stable": rng.normal(0, 1, 500),
                "drifted": rng.normal(0, 1, 500),
            }
        )
        for _ in range(300):
            detector.update(
                {
                    "stable": float(rng.normal(0, 1)),
                    "drifted": float(rng.normal(5, 1)),
                }
            )
        summary = detector.generate_daily_report(timestamp_ms=2_000_000)
        assert summary.drifted_features >= 1
        assert summary.any_alert
        assert summary.max_psi > 0.2
        assert summary.max_psi_feature == "drifted"

    def test_drift_ratio(self) -> None:
        summary = DailyDriftSummary(
            timestamp_ms=0,
            total_features=10,
            drifted_features=3,
            max_psi=0.5,
            max_psi_feature="test",
        )
        assert summary.drift_ratio == pytest.approx(0.3)

    def test_drift_ratio_zero_features(self) -> None:
        summary = DailyDriftSummary(
            timestamp_ms=0,
            total_features=0,
            drifted_features=0,
            max_psi=0.0,
            max_psi_feature="",
        )
        assert summary.drift_ratio == 0.0
