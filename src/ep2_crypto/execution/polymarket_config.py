"""Polymarket CLOB adapter configuration.

All secrets (private key, funder address) come from environment variables only.
Never pass credentials directly in code or config files.
"""

from __future__ import annotations

import os

from pydantic import BaseModel, Field, field_validator, model_validator


class PolymarketConfig(BaseModel):
    """Configuration for the Polymarket CLOB execution adapter.

    Credentials are loaded from environment variables; all other parameters
    have sensible defaults for 5-minute BTC markets.
    """

    # CLOB API connection
    host: str = "https://clob.polymarket.com"
    chain_id: int = 137  # Polygon mainnet (137) or Amoy testnet (80002)

    # Credentials — loaded from env, never stored in config files
    private_key: str = Field(
        default_factory=lambda: os.environ.get("POLYMARKET_PRIVATE_KEY", ""),
        description="Ethereum private key hex string (from POLYMARKET_PRIVATE_KEY env var)",
    )
    funder_address: str = Field(
        default_factory=lambda: os.environ.get("POLYMARKET_FUNDER_ADDRESS", ""),
        description="USDC funder address (from POLYMARKET_FUNDER_ADDRESS env var)",
    )

    # Market discovery
    market_slug_pattern: str = "will-btc-be-higher-in-5-minutes"
    gamma_api_host: str = "https://gamma-api.polymarket.com"

    # Connection management
    heartbeat_interval_s: float = 5.0  # WebSocket heartbeat every 5 seconds
    reconnect_delay_s: float = 2.0
    max_reconnect_attempts: int = 5
    order_timeout_s: float = 30.0  # How long to wait for fill before cancelling

    # Order parameters
    tick_size: float = 0.01  # Minimum price increment (1 cent)
    min_order_size: float = 1.0  # Minimum shares per order
    neg_risk: bool = False  # Whether to use neg-risk collateral netting

    @field_validator("chain_id")
    @classmethod
    def validate_chain_id(cls, v: int) -> int:
        allowed = {137, 80002}  # Polygon mainnet, Amoy testnet
        if v not in allowed:
            raise ValueError(f"chain_id must be one of {allowed}, got {v}")
        return v

    @field_validator("heartbeat_interval_s")
    @classmethod
    def validate_heartbeat(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("heartbeat_interval_s must be positive")
        return v

    @model_validator(mode="after")
    def warn_missing_credentials(self) -> PolymarketConfig:
        """Credentials are optional at config time (e.g. for tests) but
        the adapter will refuse to connect without them."""
        return self

    @property
    def has_credentials(self) -> bool:
        """True if both private key and funder address are set."""
        return bool(self.private_key) and bool(self.funder_address)
