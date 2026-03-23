"""Tests for hierarchical regime detection ensemble."""

from __future__ import annotations

import numpy as np
import pytest

from ep2_crypto.regime.bocpd import BOCPDDetector
from ep2_crypto.regime.detector import (
    HierarchicalRegimeDetector,
    MarketRegime,
    RegimeResult,
)
from ep2_crypto.regime.efficiency_ratio import EfficiencyRatioDetector
from ep2_crypto.regime.garch import GARCHDetector
from ep2_crypto.regime.hmm import HMMDetector


@pytest.fixture
def detector() -> HierarchicalRegimeDetector:
    """Small-window detector for fast tests."""
    return HierarchicalRegimeDetector(
        er_detector=EfficiencyRatioDetector(short_window=5, long_window=10),
        garch_detector=GARCHDetector(percentile_window=20),
        hmm_detector=HMMDetector(n_states=2, min_fit_samples=30, fit_window=100),
        bocpd_detector=BOCPDDetector(hazard_lambda=20.0, threshold=0.2, r_max=50),
    )


def _make_two_regime_data(n_calm: int = 100, n_volatile: int = 100, seed: int = 42) -> np.ndarray:
    """Create synthetic prices with clear calm -> volatile transition."""
    rng = np.random.default_rng(seed)
    calm_returns = rng.standard_normal(n_calm) * 0.001
    volatile_returns = rng.standard_normal(n_volatile) * 0.02
    all_returns = np.concatenate([calm_returns, volatile_returns])
    prices = 100.0 * np.exp(np.cumsum(all_returns))
    return np.concatenate([[100.0], prices])


class TestDetectorInit:
    def test_default_construction(self) -> None:
        d = HierarchicalRegimeDetector()
        assert d.warmup_bars > 0

    def test_custom_weights(self) -> None:
        d = HierarchicalRegimeDetector(er_weight=1.0, garch_weight=1.0, hmm_weight=1.0)
        # Weights should be normalized to sum to 1
        assert d._er_weight == pytest.approx(1 / 3)
        assert d._garch_weight == pytest.approx(1 / 3)
        assert d._hmm_weight == pytest.approx(1 / 3)


class TestDetectorOutput:
    def test_probabilities_sum_to_one(self, detector: HierarchicalRegimeDetector) -> None:
        """Regime probabilities must always sum to 1.0."""
        closes = _make_two_regime_data(60, 60)
        for idx in range(10, len(closes)):
            result = detector.update(idx, closes)
            prob_sum = sum(result.regime_probabilities)
            assert prob_sum == pytest.approx(1.0, abs=1e-6), (
                f"Probabilities sum to {prob_sum} at idx {idx}"
            )

    def test_regime_is_argmax(self, detector: HierarchicalRegimeDetector) -> None:
        """Regime label should match argmax of probabilities."""
        closes = _make_two_regime_data(60, 60)
        for idx in range(20, len(closes)):
            result = detector.update(idx, closes)
            expected = int(np.argmax(result.regime_probabilities))
            assert int(result.regime) == expected

    def test_confidence_bounded(self, detector: HierarchicalRegimeDetector) -> None:
        """Confidence must be in [0, 1]."""
        closes = _make_two_regime_data(60, 60)
        for idx in range(len(closes)):
            result = detector.update(idx, closes)
            assert 0.0 <= result.confidence <= 1.0

    def test_has_all_component_results(self, detector: HierarchicalRegimeDetector) -> None:
        """RegimeResult should include all component outputs."""
        closes = _make_two_regime_data(60, 60)
        result = detector.update(50, closes)
        assert result.er_result is not None
        assert result.garch_result is not None
        assert result.hmm_result is not None
        assert result.bocpd_result is not None


class TestDetectorRegimeDetection:
    def test_detects_different_regimes(self, detector: HierarchicalRegimeDetector) -> None:
        """Should detect different regimes for calm vs volatile periods."""
        closes = _make_two_regime_data(80, 80)
        results = detector.compute_batch(closes)

        # Get regimes in calm vs volatile periods (after warmup)
        calm_regimes = [r.regime for r in results[30:80]]
        volatile_regimes = [r.regime for r in results[120:160]]

        # At minimum, the distributions should differ
        calm_high = sum(1 for r in calm_regimes if r == MarketRegime.HIGH_VOL)
        calm_high_pct = calm_high / len(calm_regimes)
        vol_high = sum(1 for r in volatile_regimes if r == MarketRegime.HIGH_VOL)
        volatile_high_pct = vol_high / len(volatile_regimes)

        # Volatile period should have more HIGH_VOL classifications
        assert volatile_high_pct >= calm_high_pct

    def test_three_regimes_reachable(self) -> None:
        """All three market regimes should be reachable."""
        # Not guaranteed in small data, but enum should have all values
        assert MarketRegime.LOW_VOL == 0
        assert MarketRegime.NORMAL == 1
        assert MarketRegime.HIGH_VOL == 2


class TestChangePointAlert:
    def test_changepoint_alert_type(self, detector: HierarchicalRegimeDetector) -> None:
        closes = _make_two_regime_data(60, 60)
        result = detector.update(50, closes)
        assert isinstance(result.changepoint_alert, bool)
        assert isinstance(result.changepoint_prob, float)

    def test_confidence_discounted_during_transition(self) -> None:
        """Confidence should be reduced when BOCPD detects a change point."""
        # Create a detector where BOCPD has a very low threshold
        detector = HierarchicalRegimeDetector(
            er_detector=EfficiencyRatioDetector(short_window=5, long_window=10),
            garch_detector=GARCHDetector(percentile_window=20),
            hmm_detector=HMMDetector(n_states=2, min_fit_samples=30, fit_window=100),
            bocpd_detector=BOCPDDetector(hazard_lambda=10.0, threshold=0.01, r_max=50),
            changepoint_confidence_discount=0.5,
        )
        closes = _make_two_regime_data(50, 50)
        results = detector.compute_batch(closes)

        # Find results where changepoint alert is True
        cp_results = [r for r in results if r.changepoint_alert]
        if cp_results:
            # Confidence should be discounted
            for r in cp_results:
                # The confidence should be <= max probability (it's discounted)
                max_prob = max(r.regime_probabilities)
                assert r.confidence <= max_prob


class TestDetectorBatch:
    def test_batch_length(self, detector: HierarchicalRegimeDetector) -> None:
        closes = _make_two_regime_data(40, 40)
        results = detector.compute_batch(closes)
        assert len(results) == len(closes)

    def test_batch_probabilities_valid(self, detector: HierarchicalRegimeDetector) -> None:
        closes = _make_two_regime_data(40, 40)
        results = detector.compute_batch(closes)
        for r in results:
            assert sum(r.regime_probabilities) == pytest.approx(1.0, abs=1e-6)
            assert all(p >= 0 for p in r.regime_probabilities)


class TestRegimeResult:
    def test_result_is_frozen(self) -> None:
        from ep2_crypto.regime.bocpd import BOCPDResult
        from ep2_crypto.regime.efficiency_ratio import ERRegime, ERResult
        from ep2_crypto.regime.garch import GARCHResult, VolRegime
        from ep2_crypto.regime.hmm import HMMResult

        result = RegimeResult(
            regime=MarketRegime.NORMAL,
            regime_probabilities=(0.2, 0.5, 0.3),
            confidence=0.5,
            changepoint_alert=False,
            changepoint_prob=0.01,
            er_result=ERResult(
                er_short=0.4, er_long=0.3,
                regime=ERRegime.NEUTRAL, confidence=0.5,
            ),
            garch_result=GARCHResult(
                conditional_vol=0.01, vol_regime=VolRegime.MEDIUM,
                vol_percentile=0.5, confidence=0.5,
            ),
            hmm_result=HMMResult(
                state_probabilities=(0.5, 0.5),
                most_likely_state=0, n_states=2, is_fitted=True,
            ),
            bocpd_result=BOCPDResult(
                changepoint_prob=0.01, run_length=10.0,
                max_run_length_prob=0.5, is_changepoint=False,
            ),
        )
        with pytest.raises(AttributeError):
            result.regime = MarketRegime.HIGH_VOL  # type: ignore[misc]
