"""Execution layer: venue adapters and order routing.

Public API:
    VenueAdapter       — Abstract base class for all venue adapters
    VenueRegistry      — Registry for registering and retrieving adapters
    VenueType          — Supported venues (BINANCE_PERPS, POLYMARKET_BINARY)
    OrderRequest       — Input to place_order()
    OrderResult        — Output from place_order() / get_order()
    OrderSide          — BUY / SELL
    OrderType          — MARKET / LIMIT / LIMIT_POST_ONLY
    OrderStatus        — PENDING / OPEN / FILLED / CANCELLED / REJECTED / EXPIRED
    PositionInfo       — Current position on a venue
    OrderBookSnapshot  — Current order book state
    PolymarketAdapter  — Polymarket CLOB binary prediction market adapter
    PolymarketConfig   — Configuration for PolymarketAdapter
"""

from ep2_crypto.execution.polymarket import BinaryMarket, PolymarketAdapter, ResolutionEvent
from ep2_crypto.execution.polymarket_config import PolymarketConfig
from ep2_crypto.execution.registry import VenueRegistry
from ep2_crypto.execution.venue import (
    OrderBookLevel,
    OrderBookSnapshot,
    OrderRequest,
    OrderResult,
    OrderSide,
    OrderStatus,
    OrderType,
    PositionInfo,
    VenueAdapter,
    VenueType,
)

__all__ = [
    "BinaryMarket",
    "OrderBookLevel",
    "OrderBookSnapshot",
    "OrderRequest",
    "OrderResult",
    "OrderSide",
    "OrderStatus",
    "OrderType",
    "PolymarketAdapter",
    "PolymarketConfig",
    "PositionInfo",
    "ResolutionEvent",
    "VenueAdapter",
    "VenueRegistry",
    "VenueType",
]
