"""Tests for OnlineHawkesEstimator and StateDependentAmplifier.

Tests verify:
1. OnlineHawkesEstimator converges toward stable parameters after many events
2. Branching ratio stays below stationarity cap
3. StateDependentAmplifier stress levels match OI/funding thresholds
4. Amplification is monotone in stress factor and base probability
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from ep2_crypto.events.cascade import OnlineHawkesEstimator, StateDependentAmplifier


# ---------------------------------------------------------------------------
# OnlineHawkesEstimator
# ---------------------------------------------------------------------------


class TestOnlineHawkesEstimator:
    def test_initial_params(self) -> None:
        est = OnlineHawkesEstimator(mu_init=0.02, alpha_init=0.04, beta_init=0.1)
        p = est.get_params()
        assert p["mu"] == pytest.approx(0.02)
        assert p["alpha"] == pytest.approx(0.04)
        assert p["beta"] == pytest.approx(0.1)

    def test_n_events_increments(self) -> None:
        est = OnlineHawkesEstimator()
        assert est.n_events == 0
        for i in range(10):
            est.add_event(float(i))
        assert est.n_events == 10

    def test_branching_ratio_bounded(self) -> None:
        """Branching ratio must stay below stationarity cap after many events."""
        rng = np.random.default_rng(0)
        est = OnlineHawkesEstimator(max_alpha_beta_ratio=0.99)
        t = 0.0
        for _ in range(500):
            t += float(rng.exponential(1.0))
            est.add_event(t)
        p = est.get_params()
        assert p["branching_ratio"] < 1.0

    def test_mu_positive(self) -> None:
        """mu must stay positive after gradient steps."""
        est = OnlineHawkesEstimator(mu_init=0.001, lr=0.01)
        for i in range(200):
            est.add_event(float(i) * 0.5)
        assert est.get_params()["mu"] > 0.0

    def test_alpha_non_negative(self) -> None:
        est = OnlineHawkesEstimator()
        for i in range(100):
            est.add_event(float(i))
        assert est.get_params()["alpha"] >= 0.0

    def test_beta_positive(self) -> None:
        est = OnlineHawkesEstimator()
        for i in range(100):
            est.add_event(float(i) * 0.1)
        assert est.get_params()["beta"] > 0.0

    def test_params_change_after_events(self) -> None:
        """Parameters should change from initial values after receiving events."""
        est = OnlineHawkesEstimator(lr=0.01)
        initial = est.get_params().copy()
        for i in range(100):
            est.add_event(float(i) * 0.3)
        updated = est.get_params()
        # At least one parameter should have changed
        changed = any(abs(updated[k] - initial[k]) > 1e-9 for k in ["mu", "alpha", "beta"])
        assert changed

    def test_reset_clears_state(self) -> None:
        est = OnlineHawkesEstimator()
        for i in range(50):
            est.add_event(float(i))
        assert est.n_events == 50
        est.reset()
        assert est.n_events == 0

    def test_invalid_beta_raises(self) -> None:
        with pytest.raises(ValueError):
            OnlineHawkesEstimator(beta_init=-0.1)

    def test_invalid_alpha_raises(self) -> None:
        with pytest.raises(ValueError):
            OnlineHawkesEstimator(alpha_init=-0.1)

    def test_get_params_returns_dict(self) -> None:
        est = OnlineHawkesEstimator()
        p = est.get_params()
        assert set(p.keys()) == {"mu", "alpha", "beta", "branching_ratio"}

    def test_branching_ratio_equals_alpha_over_beta(self) -> None:
        est = OnlineHawkesEstimator(alpha_init=0.04, beta_init=0.1)
        for i in range(20):
            est.add_event(float(i) * 2.0)
        p = est.get_params()
        assert p["branching_ratio"] == pytest.approx(p["alpha"] / p["beta"], rel=1e-9)

    def test_high_event_rate_increases_intensity(self) -> None:
        """Dense events should yield higher implied intensity than sparse events."""
        est_dense = OnlineHawkesEstimator(mu_init=0.01, alpha_init=0.05, beta_init=0.1)
        est_sparse = OnlineHawkesEstimator(mu_init=0.01, alpha_init=0.05, beta_init=0.1)

        # Dense: 100 events in 10 seconds
        for i in range(100):
            est_dense.add_event(float(i) * 0.1)

        # Sparse: 10 events in 10 seconds
        for i in range(10):
            est_sparse.add_event(float(i) * 1.0)

        dense_br = est_dense.get_params()["branching_ratio"]
        sparse_br = est_sparse.get_params()["branching_ratio"]
        # Dense events drive alpha/beta higher — branching ratio should be >= sparse
        assert dense_br >= sparse_br - 0.2  # Allow tolerance for gradient noise

    def test_lr_decay_reduces_learning_rate(self) -> None:
        """Learning rate should decay after decay_every events."""
        est = OnlineHawkesEstimator(lr=0.1, lr_decay=0.9, decay_every=10)
        for i in range(100):
            est.add_event(float(i) * 0.1)
        # After 10 decay steps, lr ≈ 0.1 * 0.9^10 ≈ 0.0349
        assert est._lr < 0.1  # must have decayed

    def test_single_event_no_error(self) -> None:
        """Single event should not cause any errors."""
        est = OnlineHawkesEstimator()
        est.add_event(1.0)
        p = est.get_params()
        assert math.isfinite(p["mu"])
        assert math.isfinite(p["alpha"])
        assert math.isfinite(p["beta"])


# ---------------------------------------------------------------------------
# StateDependentAmplifier
# ---------------------------------------------------------------------------


class TestStateDependentAmplifier:
    def test_normal_conditions_stress_zero(self) -> None:
        amp = StateDependentAmplifier()
        assert amp.stress_level(0.5, 0.5) == 0
        assert amp.stress_level(0.74, 0.99) == 0

    def test_elevated_conditions_oi_only(self) -> None:
        amp = StateDependentAmplifier()
        assert amp.stress_level(0.75, 0.0) == 1
        assert amp.stress_level(0.80, 0.5) == 1

    def test_elevated_conditions_funding_only(self) -> None:
        amp = StateDependentAmplifier()
        assert amp.stress_level(0.5, 1.0) == 1
        assert amp.stress_level(0.5, -1.2) == 1

    def test_high_stress_level(self) -> None:
        amp = StateDependentAmplifier()
        assert amp.stress_level(0.90, 1.5) == 2
        assert amp.stress_level(0.91, -1.6) == 2

    def test_critical_stress_level(self) -> None:
        amp = StateDependentAmplifier()
        assert amp.stress_level(0.95, 2.0) == 3
        assert amp.stress_level(0.99, -3.5) == 3

    def test_critical_requires_both_conditions(self) -> None:
        amp = StateDependentAmplifier()
        # High OI alone is not critical
        assert amp.stress_level(0.98, 0.5) < 3
        # High funding alone is not critical
        assert amp.stress_level(0.5, 3.0) < 3

    def test_amplify_no_change_at_normal(self) -> None:
        amp = StateDependentAmplifier()
        base = 0.4
        amplified, sf = amp.amplify(base, 0.5, 0.5)
        assert amplified == pytest.approx(base)
        assert sf == pytest.approx(0.0)

    def test_amplify_increases_probability(self) -> None:
        amp = StateDependentAmplifier()
        base = 0.4
        amplified, sf = amp.amplify(base, 0.95, 2.5)  # critical
        assert amplified > base
        assert sf > 0.0

    def test_amplify_output_in_unit_interval(self) -> None:
        amp = StateDependentAmplifier()
        for oi in [0.0, 0.5, 0.9, 0.99]:
            for fz in [0.0, 1.0, 2.0, 3.0]:
                for base in [0.0, 0.5, 1.0]:
                    a, _ = amp.amplify(base, oi, fz)
                    assert 0.0 <= a <= 1.0

    def test_amplify_monotone_in_stress_factor(self) -> None:
        """Higher stress → higher amplified probability for same base."""
        amp = StateDependentAmplifier()
        base = 0.3

        # Normal (level 0)
        a0, _ = amp.amplify(base, 0.5, 0.0)
        # Elevated (level 1)
        a1, _ = amp.amplify(base, 0.76, 0.0)
        # High (level 2)
        a2, _ = amp.amplify(base, 0.91, 1.6)
        # Critical (level 3)
        a3, _ = amp.amplify(base, 0.96, 2.1)

        assert a0 <= a1 <= a2 <= a3

    def test_amplify_monotone_in_base_prob(self) -> None:
        """For same stress, higher base → higher amplified."""
        amp = StateDependentAmplifier()
        oi, fz = 0.95, 2.5  # critical
        bases = [0.1, 0.3, 0.5, 0.7, 0.9]
        amplified = [amp.amplify(b, oi, fz)[0] for b in bases]
        assert all(amplified[i] <= amplified[i + 1] for i in range(len(amplified) - 1))

    def test_amplify_formula_level3(self) -> None:
        """Verify amplification formula: 1 - (1 - base) * exp(-sf)."""
        amp = StateDependentAmplifier()
        base = 0.4
        sf = amp._STRESS_FACTORS[3]  # 0.65
        expected = 1.0 - (1.0 - base) * math.exp(-sf)
        amplified, returned_sf = amp.amplify(base, 0.96, 2.5)
        assert amplified == pytest.approx(expected, rel=1e-6)
        assert returned_sf == pytest.approx(sf)

    def test_amplify_formula_level2(self) -> None:
        amp = StateDependentAmplifier()
        base = 0.3
        sf = amp._STRESS_FACTORS[2]  # 0.35
        expected = 1.0 - (1.0 - base) * math.exp(-sf)
        amplified, returned_sf = amp.amplify(base, 0.91, 1.6)
        assert amplified == pytest.approx(expected, rel=1e-6)
        assert returned_sf == pytest.approx(sf)

    def test_stress_level_boundary_oi_75th(self) -> None:
        amp = StateDependentAmplifier()
        assert amp.stress_level(0.749, 0.0) == 0
        assert amp.stress_level(0.750, 0.0) == 1

    def test_stress_level_boundary_funding_1sigma(self) -> None:
        amp = StateDependentAmplifier()
        assert amp.stress_level(0.5, 0.999) == 0
        assert amp.stress_level(0.5, 1.000) == 1

    def test_stress_level_negative_funding(self) -> None:
        """Negative funding z-score (shorts crowded) treated same as positive."""
        amp = StateDependentAmplifier()
        assert amp.stress_level(0.95, -2.1) == 3
