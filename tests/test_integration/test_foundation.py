"""Integration test for Sprint 1 foundation: config, logging, schema, repository."""

import json

import structlog

from ep2_crypto.config import AppConfig
from ep2_crypto.db.repository import Repository
from ep2_crypto.db.schema import create_tables
from ep2_crypto.logging import configure_logging


class TestFoundationIntegration:
    def test_full_lifecycle(self, tmp_path, capsys):
        # 1. Load config with defaults
        cfg = AppConfig()
        assert cfg.env == "dev"
        assert cfg.pipeline.bars_per_year == 105_120

        # 2. Configure logging
        configure_logging(level=cfg.log_level, json_output=True)
        logger = structlog.get_logger("integration_test")

        # 3. Create database with all tables
        db_path = tmp_path / "test.db"
        conn = create_tables(db_path)
        repo = Repository(conn)

        # Verify logging output from create_tables
        captured = capsys.readouterr()
        log_lines = [l for l in captured.err.strip().split("\n") if l]
        db_init_log = None
        for line in log_lines:
            data = json.loads(line)
            if data.get("event") == "database_initialized":
                db_init_log = data
                break
        assert db_init_log is not None
        assert db_init_log["tables"] == 11

        # 4. Insert sample OHLCV data
        ohlcv_rows = [
            (1000, "BTC/USDT", "5m", 67000.0, 67100.0, 66900.0, 67050.0, 15.5, 1039275.0, 120),
            (301000, "BTC/USDT", "5m", 67050.0, 67200.0, 67000.0, 67150.0, 12.3, 826245.0, 98),
            (601000, "BTC/USDT", "5m", 67150.0, 67300.0, 67100.0, 67250.0, 18.1, 1217225.0, 145),
        ]
        count = repo.insert_ohlcv_batch(ohlcv_rows)
        assert count == 3

        # 5. Insert order book snapshot
        repo.insert_orderbook(
            1000, "BTC/USDT",
            "[67000.0, 66999.5]", "[1.0, 2.5]",
            "[67001.0, 67001.5]", "[0.8, 1.2]",
            67000.5, 1.0,
        )

        # 6. Insert trades
        repo.insert_trades_batch([
            (1000, "BTC/USDT", 67000.5, 0.1, 0, "t1"),
            (1050, "BTC/USDT", 67001.0, 0.05, 1, "t2"),
        ])

        # 7. Insert funding rate
        repo.insert_funding_rate(1000, "BTC/USDT", 0.0001, 67000.0, 66998.0)

        # 8. Insert prediction
        repo.insert_prediction(
            1000, "BTC/USDT", "up", 0.68,
            calibrated_prob_up=0.62, model_version="v0.1",
        )

        # 9. Query and verify everything
        ohlcv = repo.query_ohlcv("BTC/USDT", "5m", 0, 700000)
        assert len(ohlcv) == 3
        assert ohlcv[0]["open"] == 67000.0

        ob = repo.query_orderbook("BTC/USDT", 0, 2000)
        assert len(ob) == 1
        assert ob[0]["spread"] == 1.0

        trades = repo.query_trades("BTC/USDT", 0, 2000)
        assert len(trades) == 2

        funding = repo.query_funding_rate("BTC/USDT", 0, 2000)
        assert len(funding) == 1

        preds = repo.query_predictions("BTC/USDT", 0, 2000)
        assert len(preds) == 1
        assert preds[0]["direction"] == "up"

        # 10. Verify row counts
        assert repo.count_rows("ohlcv") == 3
        assert repo.count_rows("orderbook_snapshot") == 1
        assert repo.count_rows("agg_trades") == 2
        assert repo.count_rows("funding_rate") == 1
        assert repo.count_rows("prediction") == 1

        # 11. Log completion
        logger.info("integration_test_complete", tables_verified=5, total_rows=8)

        conn.close()

    def test_config_propagates_to_db(self, tmp_path):
        cfg = AppConfig()
        assert cfg.database.wal_mode is True

        db_path = tmp_path / "config_test.db"
        conn = create_tables(db_path)

        journal = conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert journal == "wal"

        busy = conn.execute("PRAGMA busy_timeout").fetchone()[0]
        assert busy == cfg.database.busy_timeout_ms

        conn.close()
