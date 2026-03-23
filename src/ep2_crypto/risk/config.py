"""Risk management configuration using Pydantic for validation.

All risk parameters in one place. Loaded from environment variables
with EP2_RISK_ prefix. Validated on startup — reject invalid configs
immediately rather than discovering them mid-trade.

These defaults match CLAUDE.md Risk Parameters and REQUIREMENTS.md RR-1..RR-5.
"""

from __future__ import annotations

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class RiskConfig(BaseSettings):
    """All risk parameters with Pydantic validation.

    No hot-reload — require explicit restart to change risk params.
    """

    model_config = SettingsConfigDict(env_prefix="EP2_RISK_")

    # -- Kill switch thresholds ------------------------------------------------
    daily_loss_limit: float = Field(
        default=0.03, gt=0, le=1.0,
        description="Daily loss halt threshold as fraction of capital",
    )
    weekly_loss_limit: float = Field(
        default=0.05, gt=0, le=1.0,
        description="Weekly loss halt threshold as fraction of capital",
    )
    max_drawdown_halt: float = Field(
        default=0.15, gt=0, le=1.0,
        description="Peak-to-trough drawdown at which trading halts",
    )
    consecutive_loss_limit: int = Field(
        default=15, ge=1,
        description="Consecutive losing trades before halt",
    )

    # -- Position limits -------------------------------------------------------
    max_position_fraction: float = Field(
        default=0.05, gt=0, le=1.0,
        description="Max position NOTIONAL as fraction of equity (5%)",
    )
    max_trades_per_day: int = Field(
        default=30, ge=1,
        description="Maximum trades allowed per calendar day",
    )
    max_open_positions: int = Field(
        default=1, ge=1,
        description="Maximum simultaneous open positions",
    )

    # -- Sizing ----------------------------------------------------------------
    kelly_fraction: float = Field(
        default=0.25, gt=0, le=1.0,
        description="Fraction of Kelly to use (0.25 = quarter-Kelly)",
    )
    max_risk_per_trade: float = Field(
        default=0.01, gt=0, le=0.05,
        description="Max RISK per trade as fraction of equity. "
        "Monte Carlo: >1% leads to 66%+ DD over a year.",
    )
    min_btc_quantity: float = Field(
        default=0.001, gt=0,
        description="Binance minimum order size in BTC",
    )

    # -- Stop loss -------------------------------------------------------------
    atr_stop_multiplier: float = Field(
        default=3.0, gt=0,
        description="ATR multiple for catastrophic stop",
    )
    atr_period: int = Field(
        default=14, ge=2,
        description="Number of bars for ATR computation",
    )
    max_holding_bars: int = Field(
        default=6, ge=1,
        description="Force exit after this many bars (6 = 30min)",
    )

    # -- Volatility guard ------------------------------------------------------
    min_volatility_ann: float = Field(
        default=0.15, ge=0,
        description="Minimum annualized vol to trade (15%)",
    )
    max_volatility_ann: float = Field(
        default=1.50, gt=0,
        description="Maximum annualized vol to trade (150%)",
    )

    # -- Time guards -----------------------------------------------------------
    trading_start_hour_utc: int = Field(
        default=8, ge=0, lt=24,
        description="Start of trading window (UTC hour)",
    )
    trading_end_hour_utc: int = Field(
        default=21, ge=1, le=24,
        description="End of trading window (UTC hour)",
    )
    weekend_size_reduction: float = Field(
        default=0.30, ge=0, le=1.0,
        description="Position size reduction on weekends (30%)",
    )
    enforce_trading_hours: bool = Field(
        default=True,
        description="Whether to enforce time-of-day limits",
    )

    # -- Drawdown gate ---------------------------------------------------------
    drawdown_cooldown_bars: int = Field(
        default=5, ge=0,
        description="Min bars at each recovery phase before advancing",
    )

    @model_validator(mode="after")
    def _validate_vol_range(self) -> RiskConfig:
        if self.max_volatility_ann <= self.min_volatility_ann:
            msg = (
                f"max_volatility_ann ({self.max_volatility_ann}) must exceed "
                f"min_volatility_ann ({self.min_volatility_ann})"
            )
            raise ValueError(msg)
        return self
