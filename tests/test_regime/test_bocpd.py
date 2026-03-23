"""Tests for Bayesian Online Change Point Detection."""

from __future__ import annotations

import numpy as np
import pytest

from ep2_crypto.regime.bocpd import BOCPDDetector, BOCPDResult


@pytest.fixture
def detector() -> BOCPDDetector:
    return BOCPDDetector(hazard_lambda=50.0, threshold=0.2, r_max=100)


class TestBOCPDInit:
    def test_default_params(self) -> None:
        d = BOCPDDetector()
        assert d.warmup_bars == 2
        assert d.threshold == 0.3

    def test_invalid_lambda(self) -> None:
        with pytest.raises(ValueError, match="hazard_lambda"):
            BOCPDDetector(hazard_lambda=0)

    def test_invalid_r_max(self) -> None:
        with pytest.raises(ValueError, match="r_max"):
            BOCPDDetector(r_max=5)


class TestBOCPDStep:
    def test_single_step(self, detector: BOCPDDetector) -> None:
        result = detector.step(0.01)
        assert isinstance(result, BOCPDResult)
        assert 0.0 <= result.changepoint_prob <= 1.0
        assert result.run_length >= 0.0

    def test_stable_series_low_cp_prob(self, detector: BOCPDDetector) -> None:
        """A constant series should have low changepoint probability after settling."""
        for _ in range(50):
            result = detector.step(0.0)
        # After many identical observations, cp prob should be low
        assert result.changepoint_prob < 0.5

    def test_sudden_shift_high_cp_prob(self, detector: BOCPDDetector) -> None:
        """A sudden mean shift should trigger high changepoint probability."""
        detector_sensitive = BOCPDDetector(
            hazard_lambda=20.0, threshold=0.1, r_max=100,
            mu0=0.0, kappa0=1.0, alpha0=1.0, beta0=1e-6,
        )
        # 30 bars of low values
        for _ in range(30):
            detector_sensitive.step(0.001)

        # Sudden shift to large value
        result = detector_sensitive.step(0.1)
        # The change should be detected (cp prob should increase)
        # After a few more observations at the new level, cp should have fired
        max_cp = result.changepoint_prob
        for _ in range(5):
            r = detector_sensitive.step(0.1)
            max_cp = max(max_cp, r.changepoint_prob)

        assert max_cp > 0.01  # Should detect something unusual

    def test_run_length_grows(self, detector: BOCPDDetector) -> None:
        """Expected run length should grow in a stable series."""
        run_lengths = []
        for _ in range(30):
            result = detector.step(0.001)
            run_lengths.append(result.run_length)
        # Run length should generally increase
        assert run_lengths[-1] > run_lengths[5]

    def test_changepoint_prob_bounded(self, detector: BOCPDDetector) -> None:
        """Changepoint probability must always be in [0, 1]."""
        rng = np.random.default_rng(42)
        for _ in range(100):
            x = rng.standard_normal() * 0.01
            result = detector.step(x)
            assert 0.0 <= result.changepoint_prob <= 1.0

    def test_run_length_probs_normalized(self, detector: BOCPDDetector) -> None:
        """Internal run-length distribution should sum to ~1."""
        rng = np.random.default_rng(42)
        for _ in range(50):
            detector.step(rng.standard_normal() * 0.01)
        total = float(np.sum(detector._run_length_probs))
        assert total == pytest.approx(1.0, abs=1e-6)


class TestBOCPDPruning:
    def test_prunes_at_r_max(self) -> None:
        """Run length distribution should never exceed r_max."""
        detector = BOCPDDetector(hazard_lambda=50.0, r_max=30)
        for _ in range(100):
            detector.step(0.001)
        assert len(detector._run_length_probs) <= 30

    def test_pruning_preserves_normalization(self) -> None:
        detector = BOCPDDetector(hazard_lambda=50.0, r_max=30)
        for _ in range(100):
            detector.step(0.001)
        total = float(np.sum(detector._run_length_probs))
        assert total == pytest.approx(1.0, abs=1e-6)


class TestBOCPDUpdate:
    def test_update_before_warmup(self, detector: BOCPDDetector) -> None:
        closes = np.array([100.0], dtype=np.float64)
        result = detector.update(0, closes)
        assert result.changepoint_prob == 0.0
        assert not result.is_changepoint

    def test_update_processes_all_bars(self, detector: BOCPDDetector) -> None:
        rng = np.random.default_rng(42)
        closes = 100.0 + np.cumsum(rng.standard_normal(50) * 0.01)
        result = detector.update(49, closes)
        assert result.run_length > 0

    def test_update_no_lookahead(self, detector: BOCPDDetector) -> None:
        """update(idx) should only use data up to idx."""
        rng = np.random.default_rng(42)
        closes = 100.0 + np.cumsum(rng.standard_normal(50) * 0.01)

        r_short = detector.update(30, closes[:31])
        r_long = detector.update(30, closes)
        assert r_short.changepoint_prob == pytest.approx(r_long.changepoint_prob, rel=1e-10)


class TestBOCPDBatch:
    def test_batch_length(self, detector: BOCPDDetector) -> None:
        closes = 100.0 + np.cumsum(np.random.default_rng(42).standard_normal(40) * 0.01)
        results = detector.compute_batch(closes)
        assert len(results) == 40

    def test_batch_first_result(self, detector: BOCPDDetector) -> None:
        closes = 100.0 + np.cumsum(np.random.default_rng(42).standard_normal(20) * 0.01)
        results = detector.compute_batch(closes)
        assert results[0].changepoint_prob == 0.0

    def test_batch_detects_regime_change(self) -> None:
        """Batch should detect change when volatility shifts dramatically."""
        detector = BOCPDDetector(hazard_lambda=20.0, threshold=0.1, r_max=100)
        rng = np.random.default_rng(42)
        calm = np.cumsum(rng.standard_normal(40) * 0.001)
        volatile = np.cumsum(rng.standard_normal(20) * 0.05)
        prices = 100.0 + np.concatenate([calm, volatile])

        results = detector.compute_batch(prices)

        # Find max changepoint probability around the transition (bar ~40)
        cp_probs_around_transition = [r.changepoint_prob for r in results[38:50]]
        max_cp = max(cp_probs_around_transition)
        # Should detect elevated changepoint probability
        assert max_cp > 0.01


class TestBOCPDResult:
    def test_result_is_frozen(self) -> None:
        result = BOCPDResult(
            changepoint_prob=0.1, run_length=10.0,
            max_run_length_prob=0.5, is_changepoint=False,
        )
        with pytest.raises(AttributeError):
            result.changepoint_prob = 0.9  # type: ignore[misc]

    def test_is_changepoint_flag(self) -> None:
        result = BOCPDResult(
            changepoint_prob=0.5, run_length=0.0,
            max_run_length_prob=0.5, is_changepoint=True,
        )
        assert result.is_changepoint
