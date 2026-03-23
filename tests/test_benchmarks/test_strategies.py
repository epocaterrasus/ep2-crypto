"""Tests for benchmark strategies.

Verifies each strategy produces valid positions and handles edge cases.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ep2_crypto.benchmarks.data import generate_synthetic_btc
from ep2_crypto.benchmarks.strategies import (
    BuyAndHold,
    CrossMarketLeadLag,
    FundingRateStrategy,
    MeanReversionRSI,
    MovingAverageCrossover,
    NaiveEnsemble,
    OracleStrategy,
    OrderBookImbalance,
    RandomEntry,
    SimpleMomentum,
    VolatilityBreakout,
)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return generate_synthetic_btc(n_bars=2000, seed=42)


class TestBuyAndHold:
    def test_always_long(self, sample_df: pd.DataFrame) -> None:
        strategy = BuyAndHold(leverage=1.0)
        positions = strategy.generate_positions(sample_df)
        assert (positions == 1.0).all()

    def test_leveraged(self, sample_df: pd.DataFrame) -> None:
        strategy = BuyAndHold(leverage=2.0)
        positions = strategy.generate_positions(sample_df)
        assert (positions == 2.0).all()

    def test_correct_length(self, sample_df: pd.DataFrame) -> None:
        positions = BuyAndHold().generate_positions(sample_df)
        assert len(positions) == len(sample_df)


class TestSimpleMomentum:
    def test_positions_in_valid_range(self, sample_df: pd.DataFrame) -> None:
        strategy = SimpleMomentum(lookback=5)
        positions = strategy.generate_positions(sample_df)
        assert positions.isin([-1.0, 0.0, 1.0]).all()

    def test_lookback_1(self, sample_df: pd.DataFrame) -> None:
        strategy = SimpleMomentum(lookback=1)
        positions = strategy.generate_positions(sample_df)
        assert len(positions) == len(sample_df)

    def test_invalid_lookback(self) -> None:
        with pytest.raises(ValueError):
            SimpleMomentum(lookback=0)


class TestMeanReversionRSI:
    def test_positions_valid(self, sample_df: pd.DataFrame) -> None:
        strategy = MeanReversionRSI()
        positions = strategy.generate_positions(sample_df)
        assert positions.isin([-1.0, 0.0, 1.0]).all()

    def test_rsi_computation(self) -> None:
        # Monotonically increasing prices should give RSI near 100
        close = pd.Series(range(100, 200), dtype=float)
        rsi = MeanReversionRSI._compute_rsi(close, 14)
        assert rsi.iloc[-1] > 90


class TestMovingAverageCrossover:
    def test_positions_valid(self, sample_df: pd.DataFrame) -> None:
        strategy = MovingAverageCrossover(10, 50, "ema")
        positions = strategy.generate_positions(sample_df)
        assert positions.isin([-1.0, 0.0, 1.0]).all()

    def test_sma_variant(self, sample_df: pd.DataFrame) -> None:
        strategy = MovingAverageCrossover(5, 20, "sma")
        positions = strategy.generate_positions(sample_df)
        assert len(positions) == len(sample_df)

    def test_invalid_periods(self) -> None:
        with pytest.raises(ValueError):
            MovingAverageCrossover(50, 10)


class TestRandomEntry:
    def test_reproducible(self, sample_df: pd.DataFrame) -> None:
        s1 = RandomEntry(seed=42).generate_positions(sample_df)
        s2 = RandomEntry(seed=42).generate_positions(sample_df)
        pd.testing.assert_series_equal(s1, s2)

    def test_different_seeds(self, sample_df: pd.DataFrame) -> None:
        s1 = RandomEntry(seed=42).generate_positions(sample_df)
        s2 = RandomEntry(seed=99).generate_positions(sample_df)
        # Very unlikely to be identical
        assert not s1.equals(s2)

    def test_positions_valid(self, sample_df: pd.DataFrame) -> None:
        positions = RandomEntry(seed=42).generate_positions(sample_df)
        assert positions.isin([-1.0, 0.0, 1.0]).all()


class TestVolatilityBreakout:
    def test_positions_valid(self, sample_df: pd.DataFrame) -> None:
        strategy = VolatilityBreakout(atr_multiplier=2.0, hold_bars=3)
        positions = strategy.generate_positions(sample_df)
        assert positions.isin([-1.0, 0.0, 1.0]).all()

    def test_sparse_trades(self, sample_df: pd.DataFrame) -> None:
        # High ATR multiplier should produce few trades
        strategy = VolatilityBreakout(atr_multiplier=5.0, hold_bars=1)
        positions = strategy.generate_positions(sample_df)
        # Most bars should be flat
        flat_pct = (positions == 0).mean()
        assert flat_pct > 0.5


class TestFundingRateStrategy:
    def test_positions_valid(self, sample_df: pd.DataFrame) -> None:
        strategy = FundingRateStrategy()
        positions = strategy.generate_positions(sample_df)
        assert positions.isin([-1.0, 0.0, 1.0]).all()

    def test_missing_column(self, sample_df: pd.DataFrame) -> None:
        df_no_funding = sample_df.drop(columns=["funding_rate"])
        with pytest.raises(ValueError, match="funding_rate"):
            FundingRateStrategy().generate_positions(df_no_funding)


class TestOrderBookImbalance:
    def test_positions_valid(self, sample_df: pd.DataFrame) -> None:
        strategy = OrderBookImbalance()
        positions = strategy.generate_positions(sample_df)
        assert positions.isin([-1.0, 0.0, 1.0]).all()


class TestCrossMarketLeadLag:
    def test_positions_valid(self, sample_df: pd.DataFrame) -> None:
        strategy = CrossMarketLeadLag(lag_bars=1)
        positions = strategy.generate_positions(sample_df)
        assert positions.isin([-1.0, 0.0, 1.0]).all()

    def test_invalid_lag(self) -> None:
        with pytest.raises(ValueError):
            CrossMarketLeadLag(lag_bars=0)


class TestNaiveEnsemble:
    def test_majority_vote(self, sample_df: pd.DataFrame) -> None:
        strategies = [
            SimpleMomentum(lookback=5),
            MeanReversionRSI(),
            MovingAverageCrossover(10, 50, "ema"),
        ]
        ensemble = NaiveEnsemble(strategies=strategies)
        positions = ensemble.generate_positions(sample_df)
        assert positions.isin([-1.0, 0.0, 1.0]).all()

    def test_empty_ensemble(self, sample_df: pd.DataFrame) -> None:
        with pytest.raises(ValueError):
            NaiveEnsemble(strategies=[]).generate_positions(sample_df)


class TestOracleStrategy:
    def test_perfect_foresight(self, sample_df: pd.DataFrame) -> None:
        oracle = OracleStrategy()
        positions = oracle.generate_positions(sample_df)
        current_ret = sample_df["close"].pct_change()
        # Oracle position[t] = sign(return[t]), so strategy_return = |return[t]|
        valid = ~current_ret.isna() & (current_ret != 0) & (positions != 0)
        agreement = (np.sign(current_ret[valid]) == positions[valid]).mean()
        assert agreement > 0.99  # Should be ~1.0

    def test_positions_valid(self, sample_df: pd.DataFrame) -> None:
        positions = OracleStrategy().generate_positions(sample_df)
        assert positions.isin([-1.0, 0.0, 1.0]).all()
