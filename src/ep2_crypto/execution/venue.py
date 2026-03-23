"""Venue adapter abstraction for multi-venue trade execution.

Defines the interface that all venue adapters (Binance, Polymarket, etc.)
must implement. The ML pipeline produces signals; the risk engine approves
trades; venue adapters execute them on the target venue.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class VenueType(StrEnum):
    """Supported trading venues."""

    BINANCE_PERPS = "binance_perps"
    POLYMARKET_BINARY = "polymarket_binary"
    PAPER = "paper"


class OrderSide(StrEnum):
    BUY = "buy"
    SELL = "sell"


class OrderType(StrEnum):
    MARKET = "market"
    LIMIT = "limit"
    LIMIT_POST_ONLY = "limit_post_only"


class OrderStatus(StrEnum):
    PENDING = "pending"
    OPEN = "open"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


@dataclass(frozen=True)
class OrderRequest:
    """Request to place an order on a venue."""

    side: OrderSide
    size: float  # Units (BTC for perps, shares for binary)
    order_type: OrderType = OrderType.MARKET
    price: float | None = None  # Required for limit orders
    time_in_force: str = "GTC"  # GTC, FOK, FAK, GTD
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class OrderResult:
    """Result of an order placement or query."""

    order_id: str
    status: OrderStatus
    fill_price: float = 0.0
    fill_quantity: float = 0.0
    fee: float = 0.0
    fee_currency: str = "USDC"
    venue: VenueType = VenueType.BINANCE_PERPS
    raw_response: dict[str, Any] = field(default_factory=dict)

    @property
    def is_filled(self) -> bool:
        return self.status == OrderStatus.FILLED

    @property
    def is_terminal(self) -> bool:
        return self.status in {
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED,
        }


@dataclass
class PositionInfo:
    """Current position on a venue."""

    venue: VenueType
    symbol: str
    side: OrderSide | None  # None if no position
    size: float = 0.0  # Units held
    entry_price: float = 0.0
    unrealized_pnl: float = 0.0
    cost_basis: float = 0.0  # Total cost to acquire

    @property
    def is_open(self) -> bool:
        return self.size > 0.0


@dataclass
class OrderBookLevel:
    """Single price level in an order book."""

    price: float
    size: float


@dataclass
class OrderBookSnapshot:
    """Snapshot of a venue's order book."""

    venue: VenueType
    symbol: str
    bids: list[OrderBookLevel]
    asks: list[OrderBookLevel]
    timestamp_ms: int = 0

    @property
    def best_bid(self) -> float | None:
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> float | None:
        return self.asks[0].price if self.asks else None

    @property
    def mid_price(self) -> float | None:
        if self.best_bid is not None and self.best_ask is not None:
            return (self.best_bid + self.best_ask) / 2.0
        return None

    @property
    def spread(self) -> float | None:
        if self.best_bid is not None and self.best_ask is not None:
            return self.best_ask - self.best_bid
        return None


class VenueAdapter(ABC):
    """Abstract base class for venue-specific trade execution.

    All venue adapters implement this interface so the trading system
    can switch venues without changing the core ML/risk pipeline.
    """

    @property
    @abstractmethod
    def venue_type(self) -> VenueType:
        """The type of venue this adapter connects to."""

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """Whether the adapter has an active connection to the venue."""

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to the venue (API auth, WebSocket, etc.)."""

    @abstractmethod
    async def disconnect(self) -> None:
        """Gracefully disconnect from the venue."""

    @abstractmethod
    async def place_order(self, request: OrderRequest) -> OrderResult:
        """Place an order on the venue.

        Args:
            request: Order parameters (side, size, type, price).

        Returns:
            OrderResult with fill details or rejection reason.
        """

    @abstractmethod
    async def cancel_order(self, order_id: str) -> OrderResult:
        """Cancel an open order.

        Args:
            order_id: The venue-specific order identifier.

        Returns:
            OrderResult with updated status.
        """

    @abstractmethod
    async def get_order(self, order_id: str) -> OrderResult:
        """Query the status of an order.

        Args:
            order_id: The venue-specific order identifier.

        Returns:
            OrderResult with current status and fill info.
        """

    @abstractmethod
    async def get_position(self) -> PositionInfo:
        """Get current position on this venue."""

    @abstractmethod
    async def get_orderbook(self) -> OrderBookSnapshot:
        """Get current order book snapshot."""

    @abstractmethod
    async def get_balance(self) -> float:
        """Get available balance in base currency (USD/USDC)."""

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the venue connection is healthy."""
