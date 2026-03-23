"""Tests for configuration module."""

from pathlib import Path

import pytest

from ep2_crypto.config import (
    ApiConfig,
    AppConfig,
    DatabaseConfig,
    ExchangeConfig,
    MonitoringConfig,
    PipelineConfig,
)


class TestExchangeConfig:
    def test_defaults(self):
        cfg = ExchangeConfig()
        assert cfg.symbol == "BTC/USDT:USDT"
        assert cfg.orderbook_depth == 20
        assert cfg.binance_api_key.get_secret_value() == ""

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("EP2_EXCHANGE_SYMBOL", "ETH/USDT:USDT")
        cfg = ExchangeConfig()
        assert cfg.symbol == "ETH/USDT:USDT"

    def test_secret_str_hidden(self):
        cfg = ExchangeConfig(binance_api_key="my-secret-key")
        assert "my-secret-key" not in str(cfg.binance_api_key)
        assert cfg.binance_api_key.get_secret_value() == "my-secret-key"


class TestDatabaseConfig:
    def test_defaults(self):
        cfg = DatabaseConfig()
        assert cfg.backend == "sqlite"
        assert cfg.wal_mode is True
        assert cfg.sqlite_path == Path("data/ep2_crypto.db")

    def test_timescaledb_backend(self, monkeypatch):
        monkeypatch.setenv("EP2_DB_BACKEND", "timescaledb")
        cfg = DatabaseConfig()
        assert cfg.backend == "timescaledb"

    def test_invalid_backend_rejected(self):
        with pytest.raises(Exception):
            DatabaseConfig(backend="mysql")


class TestPipelineConfig:
    def test_critical_numbers(self):
        cfg = PipelineConfig()
        assert cfg.bars_per_year == 105_120
        assert cfg.bars_per_day == 288
        assert cfg.purge_bars == 18
        assert cfg.embargo_bars == 12
        assert cfg.training_window_days == 14
        assert cfg.max_features == 25

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("EP2_PIPELINE_CONFIDENCE_THRESHOLD", "0.70")
        cfg = PipelineConfig()
        assert cfg.confidence_threshold == 0.70


class TestMonitoringConfig:
    def test_risk_limits(self):
        cfg = MonitoringConfig()
        assert cfg.daily_loss_limit == 0.03
        assert cfg.weekly_loss_limit == 0.05
        assert cfg.max_drawdown_halt == 0.15
        assert cfg.max_trades_per_day == 30
        assert cfg.max_open_positions == 1

    def test_trading_hours(self):
        cfg = MonitoringConfig()
        assert cfg.trading_start_utc == 8
        assert cfg.trading_end_utc == 21


class TestApiConfig:
    def test_defaults(self):
        cfg = ApiConfig()
        assert cfg.port == 8000
        assert cfg.workers == 1


class TestAppConfig:
    def test_defaults(self):
        cfg = AppConfig()
        assert cfg.env == "dev"
        assert cfg.debug is False
        assert cfg.log_level == "INFO"

    def test_nested_configs_loaded(self):
        cfg = AppConfig()
        assert isinstance(cfg.exchange, ExchangeConfig)
        assert isinstance(cfg.database, DatabaseConfig)
        assert isinstance(cfg.pipeline, PipelineConfig)
        assert isinstance(cfg.monitoring, MonitoringConfig)
        assert isinstance(cfg.api, ApiConfig)

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("EP2_ENV", "prod")
        monkeypatch.setenv("EP2_DEBUG", "true")
        cfg = AppConfig()
        assert cfg.env == "prod"
        assert cfg.debug is True

    def test_invalid_log_level_rejected(self):
        with pytest.raises(Exception):
            AppConfig(log_level="VERBOSE")
