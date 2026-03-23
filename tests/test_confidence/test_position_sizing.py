"""Tests for confidence-aware position sizing (Quarter-Kelly + Bayesian)."""

from __future__ import annotations

import pytest

from ep2_crypto.confidence.position_sizing import (
    ConfidencePositionConfig,
    ConfidencePositionSizer,
    KellyResult,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def default_sizer() -> ConfidencePositionSizer:
    """Sizer with default config (Bayesian enabled, quarter-Kelly)."""
    return ConfidencePositionSizer()


@pytest.fixture()
def non_bayesian_sizer() -> ConfidencePositionSizer:
    """Sizer with Bayesian disabled."""
    config = ConfidencePositionConfig(bayesian=False)
    return ConfidencePositionSizer(config)


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


class TestConfidencePositionConfig:
    def test_default_config(self) -> None:
        cfg = ConfidencePositionConfig()
        assert cfg.kelly_fraction == 0.25
        assert cfg.max_position_pct == 0.05
        assert cfg.min_confidence == 0.55
        assert cfg.bayesian is True
        assert cfg.prior_alpha == 1.0
        assert cfg.prior_beta == 1.0
        assert cfg.min_trades_for_kelly == 30

    def test_invalid_kelly_fraction(self) -> None:
        with pytest.raises(ValueError, match="kelly_fraction"):
            ConfidencePositionConfig(kelly_fraction=0.0)
        with pytest.raises(ValueError, match="kelly_fraction"):
            ConfidencePositionConfig(kelly_fraction=1.5)

    def test_invalid_max_position_pct(self) -> None:
        with pytest.raises(ValueError, match="max_position_pct"):
            ConfidencePositionConfig(max_position_pct=0.0)

    def test_invalid_min_confidence(self) -> None:
        with pytest.raises(ValueError, match="min_confidence"):
            ConfidencePositionConfig(min_confidence=-0.1)

    def test_invalid_prior_params(self) -> None:
        with pytest.raises(ValueError, match="Beta prior"):
            ConfidencePositionConfig(prior_alpha=0.0)
        with pytest.raises(ValueError, match="Beta prior"):
            ConfidencePositionConfig(prior_beta=-1.0)

    def test_invalid_min_trades(self) -> None:
        with pytest.raises(ValueError, match="min_trades_for_kelly"):
            ConfidencePositionConfig(min_trades_for_kelly=0)


# ---------------------------------------------------------------------------
# Kelly fraction computation
# ---------------------------------------------------------------------------


class TestKellyComputation:
    """Test raw Kelly fraction calculations with known inputs."""

    def test_kelly_basic(self, non_bayesian_sizer: ConfidencePositionSizer) -> None:
        """Kelly with 55% WR and 1:1 payoff = (0.55*1 - 0.45)/1 = 0.10."""
        result = non_bayesian_sizer.compute_kelly(
            win_rate=0.55, avg_win=100.0, avg_loss=100.0, n_trades=100
        )
        assert result.raw_kelly == pytest.approx(0.10, abs=1e-9)
        assert result.quarter_kelly == pytest.approx(0.025, abs=1e-9)
        assert result.bayesian_kelly is None

    def test_kelly_asymmetric_payoff(self, non_bayesian_sizer: ConfidencePositionSizer) -> None:
        """Kelly with 50% WR and 2:1 payoff = (0.5*2 - 0.5)/2 = 0.25."""
        result = non_bayesian_sizer.compute_kelly(
            win_rate=0.50, avg_win=200.0, avg_loss=100.0, n_trades=50
        )
        assert result.raw_kelly == pytest.approx(0.25, abs=1e-9)
        assert result.quarter_kelly == pytest.approx(0.0625, abs=1e-9)

    def test_kelly_losing_strategy(self, non_bayesian_sizer: ConfidencePositionSizer) -> None:
        """Kelly with 40% WR and 1:1 payoff = negative => 0."""
        result = non_bayesian_sizer.compute_kelly(
            win_rate=0.40, avg_win=100.0, avg_loss=100.0, n_trades=100
        )
        assert result.raw_kelly == 0.0
        assert result.quarter_kelly == 0.0

    def test_kelly_breakeven(self, non_bayesian_sizer: ConfidencePositionSizer) -> None:
        """Kelly with 50% WR and 1:1 payoff = 0 (no edge)."""
        result = non_bayesian_sizer.compute_kelly(
            win_rate=0.50, avg_win=100.0, avg_loss=100.0, n_trades=100
        )
        assert result.raw_kelly == 0.0
        assert result.quarter_kelly == 0.0

    def test_kelly_zero_avg_loss(self, non_bayesian_sizer: ConfidencePositionSizer) -> None:
        """Zero avg_loss should return zero Kelly safely."""
        result = non_bayesian_sizer.compute_kelly(
            win_rate=0.55, avg_win=100.0, avg_loss=0.0, n_trades=100
        )
        assert result.raw_kelly == 0.0
        assert result.quarter_kelly == 0.0

    def test_kelly_100_pct_win_rate(self, non_bayesian_sizer: ConfidencePositionSizer) -> None:
        """100% win rate: f = (1.0*1 - 0)/1 = 1.0."""
        result = non_bayesian_sizer.compute_kelly(
            win_rate=1.0, avg_win=100.0, avg_loss=100.0, n_trades=50
        )
        assert result.raw_kelly == pytest.approx(1.0, abs=1e-9)
        assert result.quarter_kelly == pytest.approx(0.25, abs=1e-9)

    def test_kelly_0_pct_win_rate(self, non_bayesian_sizer: ConfidencePositionSizer) -> None:
        """0% win rate: f = (0*1 - 1)/1 = -1 => clamped to 0."""
        result = non_bayesian_sizer.compute_kelly(
            win_rate=0.0, avg_win=100.0, avg_loss=100.0, n_trades=50
        )
        assert result.raw_kelly == 0.0
        assert result.quarter_kelly == 0.0


# ---------------------------------------------------------------------------
# Bayesian Kelly
# ---------------------------------------------------------------------------


class TestBayesianKelly:
    """Test Bayesian Kelly adjustments using Beta posterior."""

    def test_bayesian_more_conservative(self, default_sizer: ConfidencePositionSizer) -> None:
        """Bayesian Kelly should be <= standard quarter-Kelly due to
        penalization by posterior uncertainty."""
        result = default_sizer.compute_kelly(
            win_rate=0.55, avg_win=100.0, avg_loss=100.0, n_trades=100
        )
        assert result.bayesian_kelly is not None
        assert result.bayesian_kelly <= result.quarter_kelly

    def test_bayesian_wider_uncertainty_few_trades(
        self, default_sizer: ConfidencePositionSizer
    ) -> None:
        """With fewer trades, posterior uncertainty should be larger."""
        result_few = default_sizer.compute_kelly(
            win_rate=0.55, avg_win=100.0, avg_loss=100.0, n_trades=10
        )
        result_many = default_sizer.compute_kelly(
            win_rate=0.55, avg_win=100.0, avg_loss=100.0, n_trades=1000
        )
        assert result_few.uncertainty > result_many.uncertainty

    def test_bayesian_converges_with_many_trades(
        self, default_sizer: ConfidencePositionSizer
    ) -> None:
        """With very many trades, Bayesian Kelly should approach standard."""
        result = default_sizer.compute_kelly(
            win_rate=0.55, avg_win=100.0, avg_loss=100.0, n_trades=10000
        )
        assert result.bayesian_kelly is not None
        # With 10000 trades, posterior stddev is tiny; bayesian ≈ quarter
        assert result.bayesian_kelly == pytest.approx(result.quarter_kelly, abs=0.005)

    def test_bayesian_single_trade(self, default_sizer: ConfidencePositionSizer) -> None:
        """Single trade: extremely high uncertainty, very conservative."""
        result = default_sizer.compute_kelly(
            win_rate=1.0, avg_win=100.0, avg_loss=100.0, n_trades=1
        )
        assert result.bayesian_kelly is not None
        # With just 1 win and uniform prior, posterior is Beta(2,1)
        # mean = 2/3, std ≈ 0.236, conservative_wr ≈ 0.43 => kelly ≈ 0
        # Bayesian should be much smaller than quarter-Kelly of full win rate
        assert result.bayesian_kelly < result.quarter_kelly

    def test_bayesian_losing_strategy_stays_zero(
        self, default_sizer: ConfidencePositionSizer
    ) -> None:
        """Even with Bayesian, a losing strategy should yield 0."""
        result = default_sizer.compute_kelly(
            win_rate=0.40, avg_win=100.0, avg_loss=100.0, n_trades=100
        )
        assert result.bayesian_kelly is not None
        assert result.bayesian_kelly == 0.0


# ---------------------------------------------------------------------------
# Confidence scaling & sizing
# ---------------------------------------------------------------------------


class TestComputeSize:
    """Test confidence scaling and position sizing."""

    def _make_kelly_result(
        self,
        raw: float = 0.10,
        quarter: float = 0.025,
        bayesian: float | None = 0.020,
        uncertainty: float = 0.05,
    ) -> KellyResult:
        return KellyResult(
            raw_kelly=raw,
            quarter_kelly=quarter,
            bayesian_kelly=bayesian,
            uncertainty=uncertainty,
        )

    def test_basic_sizing(self, default_sizer: ConfidencePositionSizer) -> None:
        """Basic sizing: kelly * confidence * equity / price."""
        kelly = self._make_kelly_result(bayesian=0.020)
        result = default_sizer.compute_size(
            kelly_result=kelly,
            composite_confidence=0.70,
            equity=100_000.0,
            price=50_000.0,
        )
        # 0.020 * 0.70 = 0.014 fraction
        assert result.rejection_reason is None
        assert result.position_fraction == pytest.approx(0.014, abs=1e-9)
        assert result.position_usd == pytest.approx(1_400.0, abs=1e-6)
        assert result.position_btc == pytest.approx(0.028, abs=1e-6)
        assert result.capped is False

    def test_higher_confidence_larger_size(self, default_sizer: ConfidencePositionSizer) -> None:
        """Higher confidence should produce larger position."""
        kelly = self._make_kelly_result(bayesian=0.020)
        result_low = default_sizer.compute_size(
            kelly, composite_confidence=0.60, equity=100_000.0, price=50_000.0
        )
        result_high = default_sizer.compute_size(
            kelly, composite_confidence=0.90, equity=100_000.0, price=50_000.0
        )
        assert result_high.position_fraction > result_low.position_fraction
        assert result_high.position_usd > result_low.position_usd

    def test_max_cap_enforcement(self, default_sizer: ConfidencePositionSizer) -> None:
        """Position should never exceed max_position_pct (5%)."""
        # Large Kelly that would exceed 5%
        kelly = self._make_kelly_result(bayesian=0.10)
        result = default_sizer.compute_size(
            kelly, composite_confidence=0.90, equity=100_000.0, price=50_000.0
        )
        # 0.10 * 0.90 = 0.09, but capped at 0.05
        assert result.capped is True
        assert result.position_fraction == pytest.approx(0.05, abs=1e-9)
        assert result.position_usd == pytest.approx(5_000.0, abs=1e-6)
        assert result.confidence_scaled_kelly == pytest.approx(0.09, abs=1e-9)

    def test_min_confidence_rejection(self, default_sizer: ConfidencePositionSizer) -> None:
        """Below min_confidence (0.55), trade should be rejected."""
        kelly = self._make_kelly_result(bayesian=0.020)
        result = default_sizer.compute_size(
            kelly, composite_confidence=0.50, equity=100_000.0, price=50_000.0
        )
        assert result.rejection_reason is not None
        assert "below minimum" in result.rejection_reason.lower()
        assert result.position_fraction == 0.0
        assert result.position_usd == 0.0
        assert result.position_btc == 0.0

    def test_exact_min_confidence_rejected(self, default_sizer: ConfidencePositionSizer) -> None:
        """Confidence exactly at min threshold is below (strict less-than)."""
        kelly = self._make_kelly_result(bayesian=0.020)
        result = default_sizer.compute_size(
            kelly, composite_confidence=0.5499, equity=100_000.0, price=50_000.0
        )
        assert result.rejection_reason is not None

    def test_at_min_confidence_accepted(self, default_sizer: ConfidencePositionSizer) -> None:
        """Confidence exactly at min threshold should pass."""
        kelly = self._make_kelly_result(bayesian=0.020)
        result = default_sizer.compute_size(
            kelly, composite_confidence=0.55, equity=100_000.0, price=50_000.0
        )
        assert result.rejection_reason is None

    def test_zero_kelly_rejection(self, default_sizer: ConfidencePositionSizer) -> None:
        """Zero Kelly (no edge) should be rejected."""
        kelly = self._make_kelly_result(raw=0.0, quarter=0.0, bayesian=0.0)
        result = default_sizer.compute_size(
            kelly, composite_confidence=0.80, equity=100_000.0, price=50_000.0
        )
        assert result.rejection_reason is not None
        assert "non-positive" in result.rejection_reason.lower()

    def test_negative_equity_rejection(self, default_sizer: ConfidencePositionSizer) -> None:
        kelly = self._make_kelly_result(bayesian=0.020)
        result = default_sizer.compute_size(
            kelly, composite_confidence=0.70, equity=-1000.0, price=50_000.0
        )
        assert result.rejection_reason is not None
        assert "equity" in result.rejection_reason.lower()

    def test_zero_price_rejection(self, default_sizer: ConfidencePositionSizer) -> None:
        kelly = self._make_kelly_result(bayesian=0.020)
        result = default_sizer.compute_size(
            kelly, composite_confidence=0.70, equity=100_000.0, price=0.0
        )
        assert result.rejection_reason is not None
        assert "price" in result.rejection_reason.lower()

    def test_uses_quarter_kelly_when_no_bayesian(
        self, non_bayesian_sizer: ConfidencePositionSizer
    ) -> None:
        """When bayesian_kelly is None, uses quarter_kelly."""
        kelly = KellyResult(
            raw_kelly=0.10,
            quarter_kelly=0.025,
            bayesian_kelly=None,
            uncertainty=0.05,
        )
        result = non_bayesian_sizer.compute_size(
            kelly, composite_confidence=0.70, equity=100_000.0, price=50_000.0
        )
        # 0.025 * 0.70 = 0.0175
        assert result.position_fraction == pytest.approx(0.0175, abs=1e-9)


# ---------------------------------------------------------------------------
# Online stats update
# ---------------------------------------------------------------------------


class TestOnlineStats:
    def test_empty_stats(self, default_sizer: ConfidencePositionSizer) -> None:
        stats = default_sizer.get_stats()
        assert stats["n_trades"] == 0.0
        assert stats["win_rate"] == 0.0
        assert stats["avg_win"] == 0.0
        assert stats["avg_loss"] == 0.0

    def test_single_win(self, default_sizer: ConfidencePositionSizer) -> None:
        default_sizer.update_stats(is_profitable=True, pnl=50.0)
        stats = default_sizer.get_stats()
        assert stats["n_trades"] == 1.0
        assert stats["win_rate"] == 1.0
        assert stats["avg_win"] == 50.0
        assert stats["avg_loss"] == 0.0

    def test_single_loss(self, default_sizer: ConfidencePositionSizer) -> None:
        default_sizer.update_stats(is_profitable=False, pnl=-30.0)
        stats = default_sizer.get_stats()
        assert stats["n_trades"] == 1.0
        assert stats["win_rate"] == 0.0
        assert stats["avg_win"] == 0.0
        assert stats["avg_loss"] == 30.0  # stored as positive magnitude

    def test_mixed_trades(self, default_sizer: ConfidencePositionSizer) -> None:
        default_sizer.update_stats(is_profitable=True, pnl=100.0)
        default_sizer.update_stats(is_profitable=True, pnl=200.0)
        default_sizer.update_stats(is_profitable=False, pnl=-50.0)
        default_sizer.update_stats(is_profitable=False, pnl=-70.0)

        stats = default_sizer.get_stats()
        assert stats["n_trades"] == 4.0
        assert stats["win_rate"] == pytest.approx(0.5, abs=1e-9)
        assert stats["avg_win"] == pytest.approx(150.0, abs=1e-9)  # (100+200)/2
        assert stats["avg_loss"] == pytest.approx(60.0, abs=1e-9)  # (50+70)/2

    def test_running_win_rate_updates(self, default_sizer: ConfidencePositionSizer) -> None:
        """Win rate should update correctly as trades accumulate."""
        for _ in range(3):
            default_sizer.update_stats(is_profitable=True, pnl=10.0)
        assert default_sizer.get_stats()["win_rate"] == pytest.approx(1.0)

        for _ in range(7):
            default_sizer.update_stats(is_profitable=False, pnl=-10.0)
        assert default_sizer.get_stats()["win_rate"] == pytest.approx(0.3)

    def test_100_pct_win_rate(self, default_sizer: ConfidencePositionSizer) -> None:
        for _ in range(10):
            default_sizer.update_stats(is_profitable=True, pnl=25.0)
        stats = default_sizer.get_stats()
        assert stats["win_rate"] == 1.0
        assert stats["avg_win"] == 25.0
        assert stats["avg_loss"] == 0.0

    def test_0_pct_win_rate(self, default_sizer: ConfidencePositionSizer) -> None:
        for _ in range(10):
            default_sizer.update_stats(is_profitable=False, pnl=-25.0)
        stats = default_sizer.get_stats()
        assert stats["win_rate"] == 0.0
        assert stats["avg_win"] == 0.0
        assert stats["avg_loss"] == 25.0


# ---------------------------------------------------------------------------
# Integration: compute_kelly -> compute_size
# ---------------------------------------------------------------------------


class TestEndToEnd:
    """Integration tests combining Kelly computation and sizing."""

    def test_full_pipeline(self, default_sizer: ConfidencePositionSizer) -> None:
        """Full pipeline: compute Kelly, then compute size."""
        kelly = default_sizer.compute_kelly(
            win_rate=0.55, avg_win=120.0, avg_loss=100.0, n_trades=200
        )
        assert kelly.raw_kelly > 0
        assert kelly.bayesian_kelly is not None
        assert kelly.bayesian_kelly > 0

        result = default_sizer.compute_size(
            kelly_result=kelly,
            composite_confidence=0.70,
            equity=100_000.0,
            price=60_000.0,
        )
        assert result.rejection_reason is None
        assert result.position_fraction > 0
        assert result.position_usd > 0
        assert result.position_btc > 0

    def test_losing_strategy_pipeline(self, default_sizer: ConfidencePositionSizer) -> None:
        """Losing strategy should be rejected at sizing stage."""
        kelly = default_sizer.compute_kelly(
            win_rate=0.40, avg_win=100.0, avg_loss=100.0, n_trades=200
        )
        result = default_sizer.compute_size(
            kelly_result=kelly,
            composite_confidence=0.80,
            equity=100_000.0,
            price=60_000.0,
        )
        assert result.rejection_reason is not None
        assert result.position_fraction == 0.0

    def test_position_scales_linearly_with_equity(
        self, default_sizer: ConfidencePositionSizer
    ) -> None:
        """Doubling equity should double USD position (same fraction)."""
        kelly = default_sizer.compute_kelly(
            win_rate=0.55, avg_win=100.0, avg_loss=100.0, n_trades=200
        )
        r1 = default_sizer.compute_size(
            kelly, composite_confidence=0.70, equity=50_000.0, price=50_000.0
        )
        r2 = default_sizer.compute_size(
            kelly, composite_confidence=0.70, equity=100_000.0, price=50_000.0
        )
        assert r2.position_usd == pytest.approx(2 * r1.position_usd, rel=1e-9)
        assert r1.position_fraction == pytest.approx(r2.position_fraction, rel=1e-9)

    def test_custom_config(self) -> None:
        """Custom config with different parameters."""
        config = ConfidencePositionConfig(
            kelly_fraction=0.5,
            max_position_pct=0.10,
            min_confidence=0.60,
            bayesian=False,
        )
        sizer = ConfidencePositionSizer(config)
        kelly = sizer.compute_kelly(win_rate=0.55, avg_win=100.0, avg_loss=100.0, n_trades=100)
        # Half-Kelly: 0.10 * 0.5 = 0.05
        assert kelly.quarter_kelly == pytest.approx(0.05, abs=1e-9)
        assert kelly.bayesian_kelly is None

        result = sizer.compute_size(
            kelly, composite_confidence=0.80, equity=100_000.0, price=50_000.0
        )
        # 0.05 * 0.80 = 0.04 < 0.10 cap
        assert result.position_fraction == pytest.approx(0.04, abs=1e-9)
        assert result.capped is False
