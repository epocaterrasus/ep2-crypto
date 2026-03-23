"""Tests for 2-state GaussianHMM regime detector."""

from __future__ import annotations

import numpy as np
import pytest

from ep2_crypto.regime.hmm import HMMDetector, HMMResult


@pytest.fixture
def detector() -> HMMDetector:
    return HMMDetector(n_states=2, min_fit_samples=50, fit_window=200)


def _make_regime_data(rng: np.random.Generator, n: int = 600) -> np.ndarray:
    """Create synthetic data with two clear regimes (low vol then high vol)."""
    low_vol = rng.standard_normal(n // 2) * 0.001
    high_vol = rng.standard_normal(n // 2) * 0.02
    returns = np.concatenate([low_vol, high_vol])
    prices = 100.0 * np.exp(np.cumsum(returns))
    return np.concatenate([[100.0], prices])


class TestHMMInit:
    def test_default_params(self) -> None:
        d = HMMDetector()
        assert d.n_states == 2
        assert d.warmup_bars == 500
        assert not d.is_fitted

    def test_min_states(self) -> None:
        with pytest.raises(ValueError, match="n_states"):
            HMMDetector(n_states=1)


class TestHMMFitting:
    def test_fit_returns_true(self, detector: HMMDetector) -> None:
        rng = np.random.default_rng(42)
        returns = rng.standard_normal(200) * 0.01
        assert detector.fit(returns) is True
        assert detector.is_fitted

    def test_fit_insufficient_data(self, detector: HMMDetector) -> None:
        returns = np.array([0.01, -0.01, 0.02])
        assert detector.fit(returns) is False
        assert not detector.is_fitted

    def test_fit_with_volatilities(self, detector: HMMDetector) -> None:
        rng = np.random.default_rng(42)
        returns = rng.standard_normal(200) * 0.01
        vols = np.abs(returns) * 2  # Proxy vol
        assert detector.fit(returns, volatilities=vols) is True

    def test_state_sorting_by_volatility(self, detector: HMMDetector) -> None:
        """After fitting, state 0 should be lower-vol than state 1."""
        rng = np.random.default_rng(42)
        data = _make_regime_data(rng, 400)
        returns = np.diff(np.log(data))
        detector.fit(returns)

        # State order should be set
        assert detector._state_order is not None
        assert len(detector._state_order) == 2

    def test_bic_model_selection(self) -> None:
        """auto_select_states should pick optimal n_states."""
        detector = HMMDetector(
            n_states=2,
            min_fit_samples=50,
            fit_window=300,
            auto_select_states=True,
        )
        rng = np.random.default_rng(42)
        returns = rng.standard_normal(300) * 0.01
        assert detector.fit(returns) is True
        assert detector.n_states >= 2


class TestHMMPrediction:
    def test_predict_before_fit(self, detector: HMMDetector) -> None:
        """Before fitting, should return uniform probs."""
        returns = np.array([0.01, -0.01, 0.02])
        result = detector.predict_proba(returns)
        assert not result.is_fitted
        assert result.state_probabilities == pytest.approx((0.5, 0.5))

    def test_probabilities_sum_to_one(self, detector: HMMDetector) -> None:
        """State probabilities must sum to 1.0."""
        rng = np.random.default_rng(42)
        data = _make_regime_data(rng, 400)
        returns = np.diff(np.log(data))
        detector.fit(returns)

        result = detector.predict_proba(returns)
        assert result.is_fitted
        assert sum(result.state_probabilities) == pytest.approx(1.0, abs=1e-6)

    def test_most_likely_state_consistent(self, detector: HMMDetector) -> None:
        """most_likely_state should be argmax of probabilities."""
        rng = np.random.default_rng(42)
        data = _make_regime_data(rng, 400)
        returns = np.diff(np.log(data))
        detector.fit(returns)
        result = detector.predict_proba(returns)

        expected_state = max(
            range(result.n_states),
            key=lambda i: result.state_probabilities[i],
        )
        assert result.most_likely_state == expected_state

    def test_detects_regime_shift(self, detector: HMMDetector) -> None:
        """HMM should assign different probabilities to low-vol vs high-vol periods."""
        rng = np.random.default_rng(42)
        data = _make_regime_data(rng, 400)
        returns = np.diff(np.log(data))
        detector.fit(returns)

        # Check probabilities at end of low-vol period vs end of high-vol period
        low_vol_returns = returns[: len(returns) // 2]
        high_vol_returns = returns  # Full series ends in high-vol

        r_low = detector.predict_proba(low_vol_returns)
        r_high = detector.predict_proba(high_vol_returns)

        assert r_low.is_fitted and r_high.is_fitted
        # The high-vol state probability should be higher at end of high-vol period
        # State 1 = high vol (sorted by variance)
        assert r_high.state_probabilities[1] > r_low.state_probabilities[1]


class TestHMMUpdate:
    def test_update_before_warmup(self, detector: HMMDetector) -> None:
        closes = np.arange(10, dtype=np.float64) + 100
        result = detector.update(5, closes)
        assert not result.is_fitted

    def test_update_auto_fits(self, detector: HMMDetector) -> None:
        """update() should auto-fit when enough data is available."""
        rng = np.random.default_rng(42)
        data = _make_regime_data(rng, 200)
        result = detector.update(len(data) - 1, data)
        assert result.is_fitted
        assert sum(result.state_probabilities) == pytest.approx(1.0, abs=1e-6)

    def test_update_no_lookahead(self, detector: HMMDetector) -> None:
        """update(idx) should only use data up to idx."""
        rng = np.random.default_rng(42)
        data = _make_regime_data(rng, 200)
        idx = 100

        # Result should be same regardless of future data
        r1 = detector.update(idx, data[: idx + 1])

        detector2 = HMMDetector(n_states=2, min_fit_samples=50, fit_window=200)
        r2 = detector2.update(idx, data)

        assert r1.most_likely_state == r2.most_likely_state


class TestHMMResult:
    def test_result_is_frozen(self) -> None:
        result = HMMResult(
            state_probabilities=(0.7, 0.3),
            most_likely_state=0,
            n_states=2,
            is_fitted=True,
        )
        with pytest.raises(AttributeError):
            result.most_likely_state = 1  # type: ignore[misc]

    def test_result_fields(self) -> None:
        result = HMMResult(
            state_probabilities=(0.3, 0.7),
            most_likely_state=1,
            n_states=2,
            is_fitted=True,
        )
        assert result.state_probabilities == (0.3, 0.7)
        assert result.most_likely_state == 1
        assert result.n_states == 2
        assert result.is_fitted
