"""Live Binance USDT-M perpetual futures execution via ccxt.

LiveExchange implements VenueAdapter and provides the live trading path.
It is the only module that differs between paper and live modes.

Security:
  - API key and secret are loaded ONLY from environment variables
  - Credentials are NEVER logged
  - All order placement is gated through risk manager before reaching this layer

Usage:
    exchange = LiveExchange()  # Reads BINANCE_API_KEY + BINANCE_API_SECRET
    await exchange.connect()
    result = await exchange.place_order(request)
"""

from __future__ import annotations

import os
from typing import Any

import structlog

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

logger = structlog.get_logger(__name__)

SYMBOL = "BTC/USDT:USDT"  # Binance USDT-M perpetual


class LiveExchange(VenueAdapter):
    """Live Binance perpetuals adapter via ccxt.

    Reads credentials from environment variables:
      - BINANCE_API_KEY
      - BINANCE_API_SECRET

    Raises EnvironmentError on connect() if credentials are absent.
    """

    def __init__(
        self,
        symbol: str = SYMBOL,
        testnet: bool = False,
    ) -> None:
        self._symbol = symbol
        self._testnet = testnet
        self._exchange: Any = None  # ccxt.pro.binanceusdm instance
        self._connected = False

    # ------------------------------------------------------------------
    # VenueAdapter properties
    # ------------------------------------------------------------------

    @property
    def venue_type(self) -> VenueType:
        return VenueType.BINANCE_PERPS

    @property
    def is_connected(self) -> bool:
        return self._connected

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Connect to Binance and verify credentials."""
        try:
            import ccxt.pro as ccxtpro
        except ImportError as exc:
            raise ImportError(
                "ccxt[pro] is required for LiveExchange. Install with: uv add 'ccxt[pro]'"
            ) from exc

        api_key = os.environ.get("BINANCE_API_KEY", "")
        api_secret = os.environ.get("BINANCE_API_SECRET", "")
        if not api_key or not api_secret:
            raise OSError(
                "BINANCE_API_KEY and BINANCE_API_SECRET environment variables "
                "must be set before using LiveExchange."
            )

        options: dict[str, Any] = {"defaultType": "future"}
        if self._testnet:
            options["sandboxMode"] = True

        self._exchange = ccxtpro.binanceusdm(
            {
                "apiKey": api_key,
                "secret": api_secret,
                "options": options,
            }
        )
        # Verify credentials with a balance check (throws AuthenticationError if wrong)
        await self._exchange.fetch_balance()
        self._connected = True
        logger.info(
            "live_exchange_connected",
            symbol=self._symbol,
            testnet=self._testnet,
        )

    async def disconnect(self) -> None:
        if self._exchange is not None:
            await self._exchange.close()
        self._connected = False
        logger.info("live_exchange_disconnected")

    # ------------------------------------------------------------------
    # Order management
    # ------------------------------------------------------------------

    async def place_order(self, request: OrderRequest) -> OrderResult:
        """Place an order on Binance USDT-M futures."""
        self._require_connected()

        side_str = request.side.value  # "buy" or "sell"

        # Map our OrderType to ccxt order type
        order_type_str = self._map_order_type(request.order_type)
        params: dict[str, Any] = {}
        if request.order_type == OrderType.LIMIT_POST_ONLY:
            params["timeInForce"] = "GTX"  # Binance post-only

        try:
            raw = await self._exchange.create_order(
                symbol=self._symbol,
                type=order_type_str,
                side=side_str,
                amount=request.size,
                price=request.price,
                params=params,
            )
        except Exception as exc:
            logger.error(
                "live_order_failed",
                side=side_str,
                size=request.size,
                order_type=order_type_str,
                error=str(exc),
            )
            return OrderResult(
                order_id="",
                status=OrderStatus.REJECTED,
                raw_response={"error": str(exc)},
            )

        return self._parse_order_result(raw)

    async def cancel_order(self, order_id: str) -> OrderResult:
        self._require_connected()
        try:
            raw = await self._exchange.cancel_order(order_id, symbol=self._symbol)
            return self._parse_order_result(raw)
        except Exception as exc:
            logger.error("live_cancel_failed", order_id=order_id, error=str(exc))
            return OrderResult(
                order_id=order_id,
                status=OrderStatus.REJECTED,
                raw_response={"error": str(exc)},
            )

    async def get_order(self, order_id: str) -> OrderResult:
        self._require_connected()
        try:
            raw = await self._exchange.fetch_order(order_id, symbol=self._symbol)
            return self._parse_order_result(raw)
        except Exception as exc:
            logger.error("live_fetch_order_failed", order_id=order_id, error=str(exc))
            return OrderResult(
                order_id=order_id,
                status=OrderStatus.REJECTED,
                raw_response={"error": str(exc)},
            )

    async def get_position(self) -> PositionInfo:
        self._require_connected()
        try:
            positions = await self._exchange.fetch_positions([self._symbol])
            for pos in positions:
                if pos.get("symbol") == self._symbol and pos.get("contracts", 0) > 0:
                    side = OrderSide.BUY if pos.get("side", "long") == "long" else OrderSide.SELL
                    return PositionInfo(
                        venue=VenueType.BINANCE_PERPS,
                        symbol=self._symbol,
                        side=side,
                        size=float(pos.get("contracts", 0)),
                        entry_price=float(pos.get("entryPrice", 0)),
                        unrealized_pnl=float(pos.get("unrealizedPnl", 0)),
                        cost_basis=float(pos.get("initialMargin", 0)),
                    )
        except Exception as exc:
            logger.error("live_fetch_position_failed", error=str(exc))

        return PositionInfo(
            venue=VenueType.BINANCE_PERPS,
            symbol=self._symbol,
            side=None,
            size=0.0,
        )

    async def get_orderbook(self) -> OrderBookSnapshot:
        self._require_connected()
        try:
            raw = await self._exchange.fetch_order_book(self._symbol, limit=20)
            bids = [OrderBookLevel(price=p, size=s) for p, s in raw.get("bids", [])]
            asks = [OrderBookLevel(price=p, size=s) for p, s in raw.get("asks", [])]
            return OrderBookSnapshot(
                venue=VenueType.BINANCE_PERPS,
                symbol=self._symbol,
                bids=bids,
                asks=asks,
                timestamp_ms=raw.get("timestamp", 0) or 0,
            )
        except Exception as exc:
            logger.error("live_fetch_orderbook_failed", error=str(exc))
            return OrderBookSnapshot(
                venue=VenueType.BINANCE_PERPS,
                symbol=self._symbol,
                bids=[],
                asks=[],
            )

    async def get_balance(self) -> float:
        self._require_connected()
        try:
            balance = await self._exchange.fetch_balance()
            return float(balance.get("USDT", {}).get("free", 0.0))
        except Exception as exc:
            logger.error("live_fetch_balance_failed", error=str(exc))
            return 0.0

    async def health_check(self) -> bool:
        if not self._connected or self._exchange is None:
            return False
        try:
            await self._exchange.fetch_time()
            return True
        except Exception as exc:
            logger.warning("live_health_check_failed", error=str(exc))
            return False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _require_connected(self) -> None:
        if not self._connected:
            raise RuntimeError(
                "LiveExchange is not connected. Call await exchange.connect() first."
            )

    @staticmethod
    def _map_order_type(order_type: OrderType) -> str:
        mapping = {
            OrderType.MARKET: "market",
            OrderType.LIMIT: "limit",
            OrderType.LIMIT_POST_ONLY: "limit",
        }
        return mapping.get(order_type, "market")

    @staticmethod
    def _map_order_status(ccxt_status: str) -> OrderStatus:
        mapping = {
            "open": OrderStatus.OPEN,
            "closed": OrderStatus.FILLED,
            "canceled": OrderStatus.CANCELLED,
            "rejected": OrderStatus.REJECTED,
            "expired": OrderStatus.EXPIRED,
        }
        return mapping.get(ccxt_status, OrderStatus.PENDING)

    def _parse_order_result(self, raw: dict[str, Any]) -> OrderResult:
        ccxt_status = raw.get("status", "open")
        status = self._map_order_status(ccxt_status)
        fill_price = float(raw.get("average") or raw.get("price") or 0.0)
        fill_qty = float(raw.get("filled") or 0.0)
        fee_info = raw.get("fee") or {}
        fee = float(fee_info.get("cost") or 0.0)
        fee_currency = fee_info.get("currency") or "USDT"
        return OrderResult(
            order_id=str(raw.get("id", "")),
            status=status,
            fill_price=fill_price,
            fill_quantity=fill_qty,
            fee=fee,
            fee_currency=fee_currency,
            venue=VenueType.BINANCE_PERPS,
            raw_response=raw,
        )
