"""Tests for the database repository layer."""

import json

import pytest

from ep2_crypto.db.repository import Repository
from ep2_crypto.db.schema import create_tables


@pytest.fixture()
def repo():
    conn = create_tables(":memory:")
    return Repository(conn)


class TestOHLCV:
    def test_insert_and_query(self, repo):
        repo.insert_ohlcv(1000, "BTC/USDT", "5m", 100.0, 101.0, 99.0, 100.5, 10.0)
        rows = repo.query_ohlcv("BTC/USDT", "5m", 0, 2000)
        assert len(rows) == 1
        assert rows[0]["close"] == 100.5

    def test_batch_insert(self, repo):
        batch = [
            (1000, "BTC/USDT", "5m", 100.0, 101.0, 99.0, 100.5, 10.0, None, None),
            (2000, "BTC/USDT", "5m", 100.5, 102.0, 100.0, 101.0, 15.0, None, None),
            (3000, "BTC/USDT", "5m", 101.0, 103.0, 100.5, 102.0, 12.0, None, None),
        ]
        count = repo.insert_ohlcv_batch(batch)
        assert count == 3
        rows = repo.query_ohlcv("BTC/USDT", "5m", 0, 4000)
        assert len(rows) == 3

    def test_upsert_behavior(self, repo):
        repo.insert_ohlcv(1000, "BTC/USDT", "5m", 100.0, 101.0, 99.0, 100.5, 10.0)
        repo.insert_ohlcv(1000, "BTC/USDT", "5m", 100.0, 101.0, 99.0, 105.0, 10.0)
        rows = repo.query_ohlcv("BTC/USDT", "5m", 0, 2000)
        assert len(rows) == 1
        assert rows[0]["close"] == 105.0

    def test_query_latest(self, repo):
        batch = [
            (1000, "BTC/USDT", "5m", 100.0, 101.0, 99.0, 100.5, 10.0, None, None),
            (2000, "BTC/USDT", "5m", 100.5, 102.0, 100.0, 101.0, 15.0, None, None),
        ]
        repo.insert_ohlcv_batch(batch)
        rows = repo.query_latest_ohlcv("BTC/USDT", "5m", limit=1)
        assert len(rows) == 1
        assert rows[0]["timestamp_ms"] == 2000

    def test_time_range_filtering(self, repo):
        batch = [
            (1000, "BTC/USDT", "5m", 100.0, 101.0, 99.0, 100.5, 10.0, None, None),
            (2000, "BTC/USDT", "5m", 100.5, 102.0, 100.0, 101.0, 15.0, None, None),
            (3000, "BTC/USDT", "5m", 101.0, 103.0, 100.5, 102.0, 12.0, None, None),
        ]
        repo.insert_ohlcv_batch(batch)
        rows = repo.query_ohlcv("BTC/USDT", "5m", 1500, 2500)
        assert len(rows) == 1
        assert rows[0]["timestamp_ms"] == 2000


class TestOrderbook:
    def test_insert_and_query(self, repo):
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
        assert rows[0]["mid_price"] == 100.25


class TestTrades:
    def test_insert_and_query(self, repo):
        repo.insert_trade(1000, "BTC/USDT", 100.0, 0.5, True, "t1")
        rows = repo.query_trades("BTC/USDT", 0, 2000)
        assert len(rows) == 1
        assert rows[0]["is_buyer_maker"] == 1

    def test_batch_insert(self, repo):
        batch = [
            (1000, "BTC/USDT", 100.0, 0.5, 1, "t1"),
            (1001, "BTC/USDT", 100.1, 0.3, 0, "t2"),
        ]
        count = repo.insert_trades_batch(batch)
        assert count == 2


class TestFundingRate:
    def test_insert_and_query(self, repo):
        repo.insert_funding_rate(1000, "BTC/USDT", 0.0001, 100.0, 99.9)
        rows = repo.query_funding_rate("BTC/USDT", 0, 2000)
        assert len(rows) == 1
        assert rows[0]["funding_rate"] == 0.0001


class TestOpenInterest:
    def test_insert_and_query(self, repo):
        repo.insert_open_interest(1000, "BTC/USDT", 50000.0, 5_000_000.0)
        rows = repo.query_open_interest("BTC/USDT", 0, 2000)
        assert len(rows) == 1
        assert rows[0]["oi_value_usd"] == 5_000_000.0


class TestLiquidation:
    def test_insert_and_query(self, repo):
        repo.insert_liquidation(1000, "BTC/USDT", "long", 99000.0, 1.5)
        rows = repo.query_liquidations("BTC/USDT", 0, 2000)
        assert len(rows) == 1
        assert rows[0]["side"] == "long"


class TestCrossMarket:
    def test_insert_and_query(self, repo):
        repo.insert_cross_market(1000, "NQ", 18500.0, "yfinance")
        rows = repo.query_cross_market("NQ", "yfinance", 0, 2000)
        assert len(rows) == 1
        assert rows[0]["price"] == 18500.0


class TestWhale:
    def test_insert_and_query(self, repo):
        repo.insert_whale_tx(1000, "abc123", 50.0, 10.5, True)
        rows = repo.query_whale_txs(0, 2000)
        assert len(rows) == 1
        assert rows[0]["value_btc"] == 50.0
        assert rows[0]["is_exchange_flow"] == 1


class TestRegime:
    def test_insert_and_query(self, repo):
        repo.insert_regime(1000, "BTC/USDT", "trending", hmm_state=1, hmm_prob=0.85)
        rows = repo.query_regimes("BTC/USDT", 0, 2000)
        assert len(rows) == 1
        assert rows[0]["regime"] == "trending"


class TestPrediction:
    def test_insert_and_query(self, repo):
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
        assert rows[0]["direction"] == "up"
        assert rows[0]["confidence"] == 0.72


class TestFeatureSnapshot:
    def test_insert_and_query(self, repo):
        features = json.dumps({"obi": 0.3, "ofi": -0.1})
        repo.insert_feature_snapshot(1000, "BTC/USDT", features)
        rows = repo.query_feature_snapshots("BTC/USDT", 0, 2000)
        assert len(rows) == 1
        parsed = json.loads(rows[0]["features_json"])
        assert parsed["obi"] == 0.3


class TestCountRows:
    def test_count_empty_table(self, repo):
        assert repo.count_rows("ohlcv") == 0

    def test_count_after_insert(self, repo):
        repo.insert_ohlcv(1000, "BTC/USDT", "5m", 100.0, 101.0, 99.0, 100.5, 10.0)
        assert repo.count_rows("ohlcv") == 1

    def test_invalid_table_rejected(self, repo):
        with pytest.raises(ValueError, match="Unknown table"):
            repo.count_rows("drop_table_users")
