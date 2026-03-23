"""Tests for the database repository layer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ep2_crypto.config import DatabaseConfig
from ep2_crypto.db.connection import DatabaseConnection
from ep2_crypto.db.repository import Repository
from ep2_crypto.db.schema import create_tables


@pytest.fixture()
def repo() -> Repository:
    """Return a Repository backed by an in-memory SQLite database."""
    cfg = DatabaseConfig(backend="sqlite", sqlite_path=Path(":memory:"))
    db = DatabaseConnection(cfg)
    create_tables(db)
    return Repository(db)


class TestOHLCV:
    def test_insert_and_query(self, repo: Repository) -> None:
        repo.insert_ohlcv(1000, "BTC/USDT", "5m", 100.0, 101.0, 99.0, 100.5, 10.0)
        rows = repo.query_ohlcv("BTC/USDT", "5m", 0, 2000)
        assert len(rows) == 1
        assert rows[0][6] == 100.5  # close is column index 6

    def test_batch_insert(self, repo: Repository) -> None:
        batch = [
            (1000, "BTC/USDT", "5m", 100.0, 101.0, 99.0, 100.5, 10.0, None, None),
            (2000, "BTC/USDT", "5m", 100.5, 102.0, 100.0, 101.0, 15.0, None, None),
            (3000, "BTC/USDT", "5m", 101.0, 103.0, 100.5, 102.0, 12.0, None, None),
        ]
        count = repo.insert_ohlcv_batch(batch)
        assert count == 3
        rows = repo.query_ohlcv("BTC/USDT", "5m", 0, 4000)
        assert len(rows) == 3

    def test_upsert_behavior(self, repo: Repository) -> None:
        repo.insert_ohlcv(1000, "BTC/USDT", "5m", 100.0, 101.0, 99.0, 100.5, 10.0)
        repo.insert_ohlcv(1000, "BTC/USDT", "5m", 100.0, 101.0, 99.0, 105.0, 10.0)
        rows = repo.query_ohlcv("BTC/USDT", "5m", 0, 2000)
        assert len(rows) == 1
        assert rows[0][6] == 105.0  # close column (index 6)

    def test_query_latest(self, repo: Repository) -> None:
        batch = [
            (1000, "BTC/USDT", "5m", 100.0, 101.0, 99.0, 100.5, 10.0, None, None),
            (2000, "BTC/USDT", "5m", 100.5, 102.0, 100.0, 101.0, 15.0, None, None),
        ]
        repo.insert_ohlcv_batch(batch)
        rows = repo.query_latest_ohlcv("BTC/USDT", "5m", limit=1)
        assert len(rows) == 1
        assert rows[0][0] == 2000  # timestamp_ms column (index 0)

    def test_time_range_filtering(self, repo: Repository) -> None:
        batch = [
            (1000, "BTC/USDT", "5m", 100.0, 101.0, 99.0, 100.5, 10.0, None, None),
            (2000, "BTC/USDT", "5m", 100.5, 102.0, 100.0, 101.0, 15.0, None, None),
            (3000, "BTC/USDT", "5m", 101.0, 103.0, 100.5, 102.0, 12.0, None, None),
        ]
        repo.insert_ohlcv_batch(batch)
        rows = repo.query_ohlcv("BTC/USDT", "5m", 1500, 2500)
        assert len(rows) == 1
        assert rows[0][0] == 2000  # timestamp_ms


class TestOrderbook:
    def test_insert_and_query(self, repo: Repository) -> None:
        repo.insert_orderbook(
            1000,
            "BTC/USDT",
            "[100.0, 99.5]",
            "[1.0, 2.0]",
            "[100.5, 101.0]",
            "[1.5, 2.5]",
            100.25,
            0.5,
        )
        rows = repo.query_orderbook("BTC/USDT", 0, 2000)
        assert len(rows) == 1
        assert rows[0][6] == 100.25  # mid_price column (index 6)


class TestTrades:
    def test_insert_and_query(self, repo: Repository) -> None:
        repo.insert_trade(1000, "BTC/USDT", 100.0, 0.5, True, "t1")
        rows = repo.query_trades("BTC/USDT", 0, 2000)
        assert len(rows) == 1
        assert rows[0][4] == 1  # is_buyer_maker stored as int

    def test_batch_insert(self, repo: Repository) -> None:
        batch = [
            (1000, "BTC/USDT", 100.0, 0.5, 1, "t1"),
            (1001, "BTC/USDT", 100.1, 0.3, 0, "t2"),
        ]
        count = repo.insert_trades_batch(batch)
        assert count == 2


class TestFundingRate:
    def test_insert_and_query(self, repo: Repository) -> None:
        repo.insert_funding_rate(1000, "BTC/USDT", 0.0001, 100.0, 99.9)
        rows = repo.query_funding_rate("BTC/USDT", 0, 2000)
        assert len(rows) == 1
        assert rows[0][2] == pytest.approx(0.0001)  # funding_rate column (index 2)

    def test_batch_insert(self, repo: Repository) -> None:
        batch = [
            (1000, "BTC/USDT", 0.0001, 100.0, None),
            (2000, "BTC/USDT", 0.0002, 101.0, None),
        ]
        count = repo.insert_funding_rate_batch(batch)
        assert count == 2
        rows = repo.query_funding_rate("BTC/USDT", 0, 3000)
        assert len(rows) == 2


class TestOpenInterest:
    def test_insert_and_query(self, repo: Repository) -> None:
        repo.insert_open_interest(1000, "BTC/USDT", 50000.0, 5_000_000.0)
        rows = repo.query_open_interest("BTC/USDT", 0, 2000)
        assert len(rows) == 1
        assert rows[0][3] == 5_000_000.0  # oi_value_usd column (index 3)

    def test_batch_insert(self, repo: Repository) -> None:
        batch = [
            (1000, "BTC/USDT", 50000.0, None),
            (2000, "BTC/USDT", 51000.0, None),
        ]
        count = repo.insert_open_interest_batch(batch)
        assert count == 2


class TestLiquidation:
    def test_insert_and_query(self, repo: Repository) -> None:
        repo.insert_liquidation(1000, "BTC/USDT", "long", 99000.0, 1.5)
        rows = repo.query_liquidations("BTC/USDT", 0, 2000)
        assert len(rows) == 1
        assert rows[0][2] == "long"  # side column (index 2)


class TestCrossMarket:
    def test_insert_and_query(self, repo: Repository) -> None:
        repo.insert_cross_market(1000, "NQ", 18500.0, "yfinance")
        rows = repo.query_cross_market("NQ", "yfinance", 0, 2000)
        assert len(rows) == 1
        assert rows[0][2] == 18500.0  # price column (index 2)

    def test_batch_insert(self, repo: Repository) -> None:
        batch = [
            (1000, "NQ", 18500.0, "yfinance"),
            (2000, "NQ", 18600.0, "yfinance"),
        ]
        count = repo.insert_cross_market_batch(batch)
        assert count == 2


class TestWhale:
    def test_insert_and_query(self, repo: Repository) -> None:
        repo.insert_whale_tx(1000, "abc123", 50.0, 10.5, True)
        rows = repo.query_whale_txs(0, 2000)
        assert len(rows) == 1
        assert rows[0][2] == 50.0  # value_btc column (index 2)
        assert rows[0][4] == 1  # is_exchange_flow stored as int


class TestRegime:
    def test_insert_and_query(self, repo: Repository) -> None:
        repo.insert_regime(1000, "BTC/USDT", "trending", hmm_state=1, hmm_prob=0.85)
        rows = repo.query_regimes("BTC/USDT", 0, 2000)
        assert len(rows) == 1
        assert rows[0][2] == "trending"  # regime column (index 2)


class TestPrediction:
    def test_insert_and_query(self, repo: Repository) -> None:
        repo.insert_prediction(
            1000,
            "BTC/USDT",
            "up",
            0.72,
            calibrated_prob_up=0.65,
            model_version="v1.0",
        )
        rows = repo.query_predictions("BTC/USDT", 0, 2000)
        assert len(rows) == 1
        assert rows[0][2] == "up"   # direction column (index 2)
        assert rows[0][3] == pytest.approx(0.72)  # confidence column (index 3)


class TestFeatureSnapshot:
    def test_insert_and_query(self, repo: Repository) -> None:
        features = json.dumps({"obi": 0.3, "ofi": -0.1})
        repo.insert_feature_snapshot(1000, "BTC/USDT", features)
        rows = repo.query_feature_snapshots("BTC/USDT", 0, 2000)
        assert len(rows) == 1
        parsed = json.loads(rows[0][2])  # features_json column (index 2)
        assert parsed["obi"] == pytest.approx(0.3)


class TestCountRows:
    def test_count_empty_table(self, repo: Repository) -> None:
        assert repo.count_rows("ohlcv") == 0

    def test_count_after_insert(self, repo: Repository) -> None:
        repo.insert_ohlcv(1000, "BTC/USDT", "5m", 100.0, 101.0, 99.0, 100.5, 10.0)
        assert repo.count_rows("ohlcv") == 1

    def test_invalid_table_rejected(self, repo: Repository) -> None:
        with pytest.raises(ValueError, match="Unknown table"):
            repo.count_rows("drop_table_users")
