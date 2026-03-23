"""Tests for BinaryPositionSizer: Kelly criterion for binary prediction markets."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ep2_crypto.risk.binary_position_sizer import (
    BinaryPositionSizer,
    binary_kelly,
    expected_value_per_share,
    polymarket_fee,
)

# ---------------------------------------------------------------------------
# polymarket_fee tests
# ---------------------------------------------------------------------------


class TestPolymarketFee:
    def test_fee_at_50c_is_maximum(self) -> None:
        """Fee is highest at p=0.50 where p*(1-p) is maximized."""
        fee_50 = polymarket_fee(0.50)
        fee_40 = polymarket_fee(0.40)
        fee_60 = polymarket_fee(0.60)
        assert fee_50 > fee_40
        assert fee_50 > fee_60
        # Symmetric around 0.5
        assert fee_40 == pytest.approx(fee_60)

    def test_fee_at_extremes_approaches_zero(self) -> None:
        assert polymarket_fee(0.01) < 0.001
        assert polymarket_fee(0.99) < 0.001

    def test_fee_at_boundaries(self) -> None:
        assert polymarket_fee(0.0) == 0.0
        assert polymarket_fee(1.0) == 0.0

    def test_fee_at_50c_value(self) -> None:
        # fee = 0.02 * (0.5 * 0.5)^2 = 0.02 * 0.0625 = 0.00125
        assert polymarket_fee(0.50) == pytest.approx(0.00125)

    def test_fee_always_positive(self) -> None:
        for p in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            assert polymarket_fee(p) > 0.0


# ---------------------------------------------------------------------------
# binary_kelly tests
# ---------------------------------------------------------------------------


class TestBinaryKelly:
    def test_no_edge_returns_zero(self) -> None:
        """When model prob equals market price, Kelly should be ~0."""
        # With fees, even equal prob gives slightly negative Kelly
        k = binary_kelly(0.50, 0.50)
        assert k <= 0.0

    def test_positive_edge(self) -> None:
        """Model probability > market price → positive Kelly."""
        k = binary_kelly(0.55, 0.50)
        assert k > 0.0

    def test_large_edge(self) -> None:
        """Larger edge → larger Kelly fraction."""
        k_small = binary_kelly(0.52, 0.50)
        k_large = binary_kelly(0.60, 0.50)
        assert k_large > k_small

    def test_golden_dataset_no_fees(self) -> None:
        """Verify Kelly formula without fees: f* = (q-p)/(1-p)."""
        # Use fee_rate=0 to isolate Kelly logic
        k = binary_kelly(0.54, 0.50, fee_rate=0.0)
        expected = (0.54 - 0.50) / (1.0 - 0.50)  # 0.08
        assert k == pytest.approx(expected, abs=1e-10)

    def test_golden_dataset_with_fees(self) -> None:
        """Verify fee-adjusted Kelly at 50c."""
        q, p = 0.54, 0.50
        r = polymarket_fee(p)  # 0.00125
        win_payoff = (1.0 - p) * (1.0 - r)
        numerator = q * win_payoff - (1.0 - q) * p
        denominator = win_payoff
        expected = numerator / denominator
        assert binary_kelly(q, p) == pytest.approx(expected, abs=1e-10)

    def test_boundary_model_prob(self) -> None:
        assert binary_kelly(0.0, 0.50) == 0.0
        assert binary_kelly(1.0, 0.50) == 0.0

    def test_boundary_market_price(self) -> None:
        assert binary_kelly(0.55, 0.0) == 0.0
        assert binary_kelly(0.55, 1.0) == 0.0

    def test_model_prob_less_than_market(self) -> None:
        """Negative edge → negative Kelly (don't bet)."""
        k = binary_kelly(0.45, 0.50)
        assert k < 0.0


# ---------------------------------------------------------------------------
# expected_value_per_share tests
# ---------------------------------------------------------------------------


class TestExpectedValue:
    def test_positive_edge_positive_ev(self) -> None:
        ev = expected_value_per_share(0.55, 0.50)
        assert ev > 0.0

    def test_no_edge_negative_ev(self) -> None:
        """At market price with fees, EV is slightly negative."""
        ev = expected_value_per_share(0.50, 0.50)
        assert ev < 0.0

    def test_golden_no_fees(self) -> None:
        # EV = q*(1-p) - (1-q)*p = q - p
        ev = expected_value_per_share(0.54, 0.50, fee_rate=0.0)
        assert ev == pytest.approx(0.04, abs=1e-10)

    def test_boundary_prices(self) -> None:
        assert expected_value_per_share(0.55, 0.0) == 0.0
        assert expected_value_per_share(0.55, 1.0) == 0.0


# ---------------------------------------------------------------------------
# BinaryPositionSizer construction
# ---------------------------------------------------------------------------


class TestBinaryPositionSizerInit:
    def test_default_params(self) -> None:
        sizer = BinaryPositionSizer()
        assert sizer.kelly_fraction == 0.25
        assert sizer.max_bet_fraction == 0.05
        assert sizer.min_bet_usd == 5.0

    def test_custom_params(self) -> None:
        sizer = BinaryPositionSizer(
            kelly_fraction=0.5,
            max_bet_fraction=0.10,
            min_bet_usd=10.0,
        )
        assert sizer.kelly_fraction == 0.5
        assert sizer.max_bet_fraction == 0.10
        assert sizer.min_bet_usd == 10.0

    def test_invalid_kelly_fraction(self) -> None:
        with pytest.raises(ValueError, match="kelly_fraction"):
            BinaryPositionSizer(kelly_fraction=0.0)
        with pytest.raises(ValueError, match="kelly_fraction"):
            BinaryPositionSizer(kelly_fraction=1.5)

    def test_invalid_max_bet_fraction(self) -> None:
        with pytest.raises(ValueError, match="max_bet_fraction"):
            BinaryPositionSizer(max_bet_fraction=0.0)


# ---------------------------------------------------------------------------
# BinaryPositionSizer.compute — golden dataset
# ---------------------------------------------------------------------------


class TestBinaryPositionSizerCompute:
    def setup_method(self) -> None:
        self.sizer = BinaryPositionSizer(
            kelly_fraction=0.25,
            max_bet_fraction=0.05,
            min_bet_usd=5.0,
            fee_rate=0.0,  # Zero fees for clean golden tests
        )

    def test_basic_bet(self) -> None:
        """54% model vs 50c market, $10K bankroll, zero fees."""
        result = self.sizer.compute(model_prob=0.54, market_price=0.50, bankroll=10_000.0)
        assert not result.rejected

        # Raw Kelly = (0.54-0.50)/(1-0.50) = 0.08
        assert result.kelly_fraction == pytest.approx(0.08, abs=1e-6)

        # Quarter-Kelly = 0.08 * 0.25 = 0.02
        assert result.adjusted_kelly == pytest.approx(0.02, abs=1e-6)

        # Cost = 0.02 * 10000 = $200
        assert result.cost_usd == pytest.approx(200.0, abs=0.01)

        # Shares = 200 / 0.50 = 400
        assert result.shares == pytest.approx(400.0, abs=0.01)

        # Max loss = cost
        assert result.max_loss_usd == result.cost_usd

    def test_large_edge_capped(self) -> None:
        """80% model vs 50c market → raw Kelly huge, capped at 5%."""
        result = self.sizer.compute(model_prob=0.80, market_price=0.50, bankroll=10_000.0)
        assert not result.rejected

        # Raw Kelly = (0.80-0.50)/(1-0.50) = 0.60
        assert result.kelly_fraction == pytest.approx(0.60, abs=1e-6)

        # Quarter-Kelly = 0.60 * 0.25 = 0.15 → capped at 0.05
        assert result.adjusted_kelly == pytest.approx(0.05, abs=1e-6)

        # Cost = 0.05 * 10000 = $500
        assert result.cost_usd == pytest.approx(500.0, abs=0.01)

    def test_no_edge_rejected(self) -> None:
        """Model prob <= market price → rejected."""
        result = self.sizer.compute(model_prob=0.50, market_price=0.50, bankroll=10_000.0)
        assert result.rejected
        assert "no_edge" in (result.rejection_reason or "")

    def test_below_minimum_rejected(self) -> None:
        """Very small bankroll → bet below $5 minimum."""
        result = self.sizer.compute(model_prob=0.52, market_price=0.50, bankroll=100.0)
        # Raw Kelly = 0.04, QK = 0.01, bet = $1 → below $5
        assert result.rejected
        assert "below_minimum" in (result.rejection_reason or "")

    def test_drawdown_multiplier(self) -> None:
        """Drawdown gate reduces position size."""
        result_full = self.sizer.compute(
            model_prob=0.60,
            market_price=0.50,
            bankroll=10_000.0,
            drawdown_multiplier=1.0,
        )
        result_half = self.sizer.compute(
            model_prob=0.60,
            market_price=0.50,
            bankroll=10_000.0,
            drawdown_multiplier=0.5,
        )
        assert not result_full.rejected
        assert not result_half.rejected
        assert result_half.cost_usd == pytest.approx(result_full.cost_usd * 0.5, abs=0.01)

    def test_drawdown_zero_rejects(self) -> None:
        """Drawdown multiplier 0 → bet is zero → rejected."""
        result = self.sizer.compute(
            model_prob=0.60,
            market_price=0.50,
            bankroll=10_000.0,
            drawdown_multiplier=0.0,
        )
        assert result.rejected

    def test_zero_bankroll_rejected(self) -> None:
        result = self.sizer.compute(model_prob=0.60, market_price=0.50, bankroll=0.0)
        assert result.rejected

    def test_invalid_market_price(self) -> None:
        result = self.sizer.compute(model_prob=0.60, market_price=0.0, bankroll=10_000.0)
        assert result.rejected

    def test_invalid_model_prob(self) -> None:
        result = self.sizer.compute(model_prob=0.0, market_price=0.50, bankroll=10_000.0)
        assert result.rejected


# ---------------------------------------------------------------------------
# With fees
# ---------------------------------------------------------------------------


class TestBinaryPositionSizerWithFees:
    def setup_method(self) -> None:
        self.sizer = BinaryPositionSizer(
            kelly_fraction=0.25,
            max_bet_fraction=0.05,
            min_bet_usd=5.0,
            # Default Polymarket fees
            fee_rate=0.02,
            fee_exponent=2.0,
        )

    def test_fees_reduce_kelly(self) -> None:
        """With fees, Kelly fraction should be smaller than without."""
        sizer_no_fee = BinaryPositionSizer(fee_rate=0.0)

        result_fee = self.sizer.compute(0.55, 0.50, 10_000.0)
        result_no_fee = sizer_no_fee.compute(0.55, 0.50, 10_000.0)

        assert result_fee.kelly_fraction < result_no_fee.kelly_fraction

    def test_fee_recorded_in_result(self) -> None:
        result = self.sizer.compute(0.55, 0.50, 10_000.0)
        assert result.fee_per_share > 0.0
        assert result.fee_per_share == pytest.approx(polymarket_fee(0.50), abs=1e-10)

    def test_breakeven_accuracy_at_50c(self) -> None:
        """At 50c, break-even is p / (p + (1-p)*(1-r)) ≈ 50.06%."""
        r = polymarket_fee(0.50)
        breakeven = 0.50 / (0.50 + 0.50 * (1.0 - r))
        # Slightly above 50%
        assert 0.500 < breakeven < 0.501

        # At break-even, EV should be ~0
        ev = expected_value_per_share(breakeven, 0.50)
        assert abs(ev) < 0.001


# ---------------------------------------------------------------------------
# Property-based tests (Hypothesis)
# ---------------------------------------------------------------------------


class TestBinaryPositionSizerProperties:
    @given(
        model_prob=st.floats(min_value=0.51, max_value=0.99),
        market_price=st.floats(min_value=0.05, max_value=0.95),
        bankroll=st.floats(min_value=100.0, max_value=1_000_000.0),
        drawdown_mult=st.floats(min_value=0.0, max_value=1.0),
    )
    @settings(max_examples=500)
    def test_bet_never_exceeds_max_cap(
        self,
        model_prob: float,
        market_price: float,
        bankroll: float,
        drawdown_mult: float,
    ) -> None:
        """CRITICAL: Bet size NEVER exceeds max_bet_fraction of bankroll."""
        sizer = BinaryPositionSizer(
            kelly_fraction=0.25,
            max_bet_fraction=0.05,
            min_bet_usd=1.0,  # Low minimum for property test
        )
        result = sizer.compute(model_prob, market_price, bankroll, drawdown_mult)
        if not result.rejected:
            assert result.cost_usd <= bankroll * 0.05 + 0.01  # Small epsilon for float

    @given(
        model_prob=st.floats(min_value=0.51, max_value=0.99),
        market_price=st.floats(min_value=0.05, max_value=0.95),
        bankroll=st.floats(min_value=100.0, max_value=1_000_000.0),
    )
    @settings(max_examples=500)
    def test_max_loss_equals_cost(
        self,
        model_prob: float,
        market_price: float,
        bankroll: float,
    ) -> None:
        """For binary bets, max loss always equals cost (lose the premium)."""
        sizer = BinaryPositionSizer(min_bet_usd=1.0)
        result = sizer.compute(model_prob, market_price, bankroll)
        if not result.rejected:
            assert result.max_loss_usd == pytest.approx(result.cost_usd, abs=0.01)

    @given(
        model_prob=st.floats(min_value=0.01, max_value=0.49),
        market_price=st.floats(min_value=0.50, max_value=0.95),
    )
    @settings(max_examples=200)
    def test_no_edge_always_rejected(
        self,
        model_prob: float,
        market_price: float,
    ) -> None:
        """When model_prob < market_price, always rejected."""
        sizer = BinaryPositionSizer(min_bet_usd=1.0)
        result = sizer.compute(model_prob, market_price, 10_000.0)
        assert result.rejected


# ---------------------------------------------------------------------------
# Parameterized golden dataset
# ---------------------------------------------------------------------------


GOLDEN_CASES = [
    # (model_prob, market_price, bankroll, fee_rate, expected_kelly, expected_cost_approx)
    (0.54, 0.50, 10_000, 0.0, 0.08, 200.0),  # Basic no-fee
    (0.60, 0.50, 10_000, 0.0, 0.20, 500.0),  # Larger edge, capped
    (0.55, 0.45, 10_000, 0.0, 0.1818, 454.55),  # Off-center price, QK=0.0455
    (0.52, 0.50, 10_000, 0.0, 0.04, 100.0),  # Thin edge
    (0.70, 0.30, 10_000, 0.0, 0.5714, 500.0),  # Deep value, capped
]


@pytest.mark.parametrize(
    "q, p, bankroll, fee, exp_kelly, exp_cost",
    GOLDEN_CASES,
    ids=[
        "basic_54_50",
        "strong_60_50",
        "offcenter_55_45",
        "thin_52_50",
        "deep_value_70_30",
    ],
)
def test_golden_kelly_values(
    q: float,
    p: float,
    bankroll: float,
    fee: float,
    exp_kelly: float,
    exp_cost: float,
) -> None:
    sizer = BinaryPositionSizer(
        kelly_fraction=0.25,
        max_bet_fraction=0.05,
        min_bet_usd=1.0,
        fee_rate=fee,
    )
    result = sizer.compute(q, p, bankroll)
    assert not result.rejected
    assert result.kelly_fraction == pytest.approx(exp_kelly, abs=0.001)
    assert result.cost_usd == pytest.approx(exp_cost, abs=1.0)
