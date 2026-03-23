"""Configuration module using Pydantic Settings with environment variable loading."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class ExchangeConfig(BaseSettings):
    """Exchange connection settings."""

    model_config = SettingsConfigDict(env_prefix="EP2_EXCHANGE_")

    binance_api_key: SecretStr = SecretStr("")
    binance_api_secret: SecretStr = SecretStr("")
    bybit_api_key: SecretStr = SecretStr("")
    bybit_api_secret: SecretStr = SecretStr("")

    symbol: str = "BTC/USDT:USDT"
    orderbook_depth: int = 20
    ws_reconnect_delay_s: float = 1.0
    ws_max_reconnect_delay_s: float = 60.0
    ws_ping_interval_s: float = 20.0


class DatabaseConfig(BaseSettings):
    """Database connection settings."""

    model_config = SettingsConfigDict(env_prefix="EP2_DB_")

    backend: Literal["sqlite", "timescaledb"] = "sqlite"
    sqlite_path: Path = Path("data/ep2_crypto.db")
    timescaledb_url: SecretStr = SecretStr("")

    wal_mode: bool = True
    journal_size_limit: int = 67_108_864  # 64 MB
    cache_size: int = -64_000  # 64 MB (negative = KB)
    busy_timeout_ms: int = 5_000


class PipelineConfig(BaseSettings):
    """Feature pipeline and model settings."""

    model_config = SettingsConfigDict(env_prefix="EP2_PIPELINE_")

    bar_interval_m: int = 5
    bars_per_day: int = 288
    bars_per_year: int = 105_120
    training_window_days: int = 14
    test_window_days: int = 1
    purge_bars: int = 18
    embargo_bars: int = 12

    confidence_threshold: float = 0.60
    min_volatility_ann: float = 0.15
    max_volatility_ann: float = 1.50

    max_features: int = 25


class MonitoringConfig(BaseSettings):
    """Monitoring and risk management settings."""

    model_config = SettingsConfigDict(env_prefix="EP2_MONITOR_")

    daily_loss_limit: float = 0.03
    weekly_loss_limit: float = 0.05
    max_drawdown_halt: float = 0.15
    max_trades_per_day: int = 30
    max_open_positions: int = 1

    catastrophic_stop_atr: float = 3.0
    position_size_fraction: float = 0.05
    weekend_size_reduction: float = 0.30

    trading_start_utc: int = 8
    trading_end_utc: int = 21


class ApiConfig(BaseSettings):
    """API server settings."""

    model_config = SettingsConfigDict(env_prefix="EP2_API_")

    host: str = "0.0.0.0"  # noqa: S104
    port: int = 8000
    workers: int = 1
    log_level: str = "info"


class AppConfig(BaseSettings):
    """Top-level application configuration aggregating all sub-configs."""

    model_config = SettingsConfigDict(env_prefix="EP2_")

    env: Literal["dev", "staging", "prod"] = "dev"
    debug: bool = False
    data_dir: Path = Path("data")
    log_level: str = Field(default="INFO", pattern=r"^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")

    exchange: ExchangeConfig = Field(default_factory=ExchangeConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    api: ApiConfig = Field(default_factory=ApiConfig)
