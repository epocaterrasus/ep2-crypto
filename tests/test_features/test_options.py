"""Tests for OptionsFeatureComputer.

Coverage:
- First snapshot returns NaN for rolling stats (insufficient history)
- IV slope formula: (short - long) / long
- IV z-score: deviation from rolling mean
- Percent change over lookback window
- No look-ahead: features at step t use only snapshots from steps < t
- Reset clears history
- Golden dataset: hand-computed values verified against implementation
"""

from __future__ import annotations

import math

import pytest

from ep2_crypto.features.options import OptionsFeatureComputer, OptionsFeatures


def _feed(computer: OptionsFeatureComputer, n: int, base_iv: float = 0.60) -> list[OptionsFeatures]:
    """Feed n identical snapshots and return all features."""
    results = []
    for _ in range(n):
        feat = computer.update(
            atm_iv_7d=base_iv,
            atm_iv_14d=base_iv * 0.97,
            atm_iv_30d=base_iv * 0.95,
            rr_25d=None,
        )
        results.append(feat)
    return results


class TestOptionsFeatureComputer:
    def test_first_snapshot_rolling_stats_nan(self) -> None:
        comp = OptionsFeatureComputer(window=10, pct_change_lookback=5)
        feat = comp.update(0.65, 0.63, 0.60, None)
        assert math.isnan(feat.iv_zscore)
        assert math.isnan(feat.iv_pct_change_1h)

    def test_atm_iv_passthrough(self) -> None:
        comp = OptionsFeatureComputer()
        feat = comp.update(0.72, None, None, None)
        assert feat.atm_iv_7d == pytest.approx(0.72)

    def test_iv_slope_formula(self) -> None:
        """iv_slope = (atm_iv_7d - atm_iv_30d) / atm_iv_30d."""
        comp = OptionsFeatureComputer()
        atm_7 = 0.70
        atm_30 = 0.60
        expected_slope = (atm_7 - atm_30) / atm_30  # 1/6 ≈ 0.1667
        feat = comp.update(atm_7, None, atm_30, None)
        assert feat.iv_slope == pytest.approx(expected_slope, rel=1e-6)

    def test_iv_slope_nan_when_no_30d(self) -> None:
        comp = OptionsFeatureComputer()
        feat = comp.update(0.65, None, None, None)
        assert math.isnan(feat.iv_slope)

    def test_rr_25d_passthrough(self) -> None:
        comp = OptionsFeatureComputer()
        feat = comp.update(0.65, 0.63, 0.60, -0.05)
        assert feat.rr_25d == pytest.approx(-0.05)

    def test_rr_25d_nan_when_none(self) -> None:
        comp = OptionsFeatureComputer()
        feat = comp.update(0.65, None, None, None)
        assert math.isnan(feat.rr_25d)

    def test_iv_zscore_near_zero_for_constant_iv(self) -> None:
        """With constant IV, z-score should be 0 (or NaN due to zero std)."""
        comp = OptionsFeatureComputer(window=10)
        feats = _feed(comp, 15, base_iv=0.60)
        # After enough constant values, std → 0, z-score → 0 or NaN
        last = feats[-1]
        assert math.isnan(last.iv_zscore) or abs(last.iv_zscore) < 1e-6

    def test_iv_zscore_large_for_spike(self) -> None:
        """After a period with slight variation, a large IV spike gives large z-score."""
        comp = OptionsFeatureComputer(window=20)
        # Feed 20 values with slight variation (so std > 0)
        for i in range(20):
            comp.update(0.60 + i * 0.001, None, None, None)
        # Spike: much larger than the stable range
        feat = comp.update(1.20, None, None, None)
        assert not math.isnan(feat.iv_zscore)
        assert feat.iv_zscore > 2.0  # Should be large positive

    def test_pct_change_correct_formula(self) -> None:
        """Golden test: pct_change = (current - past) / past."""
        comp = OptionsFeatureComputer(window=10, pct_change_lookback=5)
        # Feed 5 snapshots at 0.60
        for _ in range(5):
            comp.update(0.60, None, None, None)
        # 6th at 0.66 → pct_change = (0.66 - 0.60) / 0.60 = 0.10
        feat = comp.update(0.66, None, None, None)
        assert feat.iv_pct_change_1h == pytest.approx(0.10, rel=1e-6)

    def test_pct_change_nan_insufficient_history(self) -> None:
        comp = OptionsFeatureComputer(window=10, pct_change_lookback=10)
        for _ in range(9):  # 9 < lookback of 10
            comp.update(0.60, None, None, None)
        feat = comp.update(0.70, None, None, None)
        assert math.isnan(feat.iv_pct_change_1h)

    def test_n_snapshots_increments(self) -> None:
        comp = OptionsFeatureComputer(window=10)
        assert comp.n_snapshots == 0
        for i in range(5):
            comp.update(0.60, None, None, None)
            assert comp.n_snapshots == i + 1

    def test_n_snapshots_capped_at_window(self) -> None:
        window = 10
        lookback = 5
        comp = OptionsFeatureComputer(window=window, pct_change_lookback=lookback)
        # deque maxlen = window + lookback
        for _ in range(window + lookback + 20):
            comp.update(0.60, None, None, None)
        assert comp.n_snapshots <= window + lookback

    def test_reset_clears_history(self) -> None:
        comp = OptionsFeatureComputer(window=10)
        _feed(comp, 20)
        comp.reset()
        assert comp.n_snapshots == 0
        # After reset, z-score should be NaN again
        feat = comp.update(0.65, None, None, None)
        assert math.isnan(feat.iv_zscore)

    def test_no_look_ahead_zscore(self) -> None:
        """Feature at step t should not change if future data is appended."""
        comp = OptionsFeatureComputer(window=20)
        ivs = [0.60 + i * 0.01 for i in range(10)]

        # Compute features up to step 5
        comp1 = OptionsFeatureComputer(window=20)
        feats_short = [comp1.update(iv, None, None, None) for iv in ivs[:5]]
        feat_at_5_short = feats_short[-1]

        # Compute features up to step 10 (5 additional points)
        comp2 = OptionsFeatureComputer(window=20)
        feats_full = [comp2.update(iv, None, None, None) for iv in ivs]

        # Feature at step 5 should match between the two runs
        feat_at_5_full = feats_full[4]
        if not math.isnan(feat_at_5_short.iv_zscore):
            assert feat_at_5_short.iv_zscore == pytest.approx(
                feat_at_5_full.iv_zscore, rel=1e-9
            )

    def test_positive_rr_signals_call_skew(self) -> None:
        """Positive RR means calls more expensive than puts (bullish skew)."""
        comp = OptionsFeatureComputer()
        feat = comp.update(0.70, None, None, 0.05)
        assert feat.rr_25d > 0

    def test_negative_rr_signals_put_skew(self) -> None:
        """Negative RR means puts more expensive than calls (bearish hedging demand)."""
        comp = OptionsFeatureComputer()
        feat = comp.update(0.70, None, None, -0.08)
        assert feat.rr_25d < 0

    def test_golden_iv_zscore_hand_computed(self) -> None:
        """Hand-computed z-score after 4 snapshots: [0.60, 0.62, 0.64] then 0.70."""
        comp = OptionsFeatureComputer(window=10, pct_change_lookback=1)
        comp.update(0.60, None, None, None)
        comp.update(0.62, None, None, None)
        comp.update(0.64, None, None, None)
        feat = comp.update(0.70, None, None, None)

        # History before 0.70 is appended: [0.60, 0.62, 0.64]
        # mean = (0.60 + 0.62 + 0.64) / 3 = 0.62
        # var (ddof=1) = ((0.02^2 + 0^2 + 0.02^2)) / 2 = 0.0004/2 = 0.0002
        # std = sqrt(0.0002) ≈ 0.014142
        # zscore = (0.70 - 0.62) / 0.014142 ≈ 5.657
        mean = (0.60 + 0.62 + 0.64) / 3
        import math as _math

        std = _math.sqrt(((0.60 - mean) ** 2 + (0.62 - mean) ** 2 + (0.64 - mean) ** 2) / 2)
        expected_z = (0.70 - mean) / std
        assert feat.iv_zscore == pytest.approx(expected_z, rel=1e-6)
