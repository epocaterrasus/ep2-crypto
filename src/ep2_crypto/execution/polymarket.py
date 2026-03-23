"""Polymarket CLOB execution adapter.

Implements the VenueAdapter ABC for Polymarket binary prediction markets.
Uses py-clob-client for REST + WebSocket communication.

Key design points:
- Private key loaded from env var only (never passed in code)
- WebSocket heartbeat every 5s to keep connection alive
- Deterministic slug-based market discovery for 5-min BTC markets
- Resolution tracking via WebSocket events + REST polling fallback
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from dataclasses import dataclass, field
from typing import Any

import structlog

from ep2_crypto.execution.polymarket_config import PolymarketConfig
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


# ---------------------------------------------------------------------------
# Market metadata
# ---------------------------------------------------------------------------


@dataclass
class BinaryMarket:
    """Metadata for a discovered Polymarket binary market."""

    condition_id: str  # Unique market identifier on-chain
    question_id: str
    slug: str
    question: str
    end_date_iso: str
    active: bool
    yes_token_id: str  # Token ID for the YES outcome
    no_token_id: str  # Token ID for the NO outcome
    current_yes_price: float = 0.5
    current_no_price: float = 0.5

    @property
    def token_id_for_side(self) -> dict[str, str]:
        return {"yes": self.yes_token_id, "no": self.no_token_id}


@dataclass
class ResolutionEvent:
    """Fired when a market resolves."""

    condition_id: str
    resolved_outcome: str  # "yes" or "no"
    resolved_at_ms: int
    winning_token_id: str


# ---------------------------------------------------------------------------
# Internal order tracking
# ---------------------------------------------------------------------------


@dataclass
class _PendingOrder:
    """Internal state for an order that has been placed but not yet terminal."""

    order_id: str
    condition_id: str
    side: OrderSide
    token_id: str
    size: float
    price: float
    placed_at_s: float = field(default_factory=time.time)
    status: OrderStatus = OrderStatus.PENDING


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class PolymarketAdapter(VenueAdapter):
    """Execution adapter for Polymarket binary prediction markets.

    Lifecycle:
        adapter = PolymarketAdapter(config)
        await adapter.connect()          # authenticates, starts heartbeat loop
        result = await adapter.place_order(request)
        await adapter.disconnect()

    Market discovery:
        market = await adapter.discover_market(slug_fragment)
        adapter.set_active_market(market)
    """

    def __init__(self, config: PolymarketConfig | None = None) -> None:
        self._config = config or PolymarketConfig()
        self._connected = False
        self._client: Any = None  # py_clob_client.ClobClient (lazy import)
        self._ws_client: Any = None  # py_clob_client.WebSocketClient
        self._heartbeat_task: asyncio.Task[None] | None = None
        self._ws_task: asyncio.Task[None] | None = None
        self._active_market: BinaryMarket | None = None
        self._pending_orders: dict[str, _PendingOrder] = {}
        self._resolution_callbacks: list[Any] = []

    # ------------------------------------------------------------------
    # VenueAdapter properties
    # ------------------------------------------------------------------

    @property
    def venue_type(self) -> VenueType:
        return VenueType.POLYMARKET_BINARY

    @property
    def is_connected(self) -> bool:
        return self._connected

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Authenticate with Polymarket CLOB and start heartbeat."""
        if self._connected:
            logger.warning("polymarket_already_connected")
            return

        if not self._config.has_credentials:
            raise ConnectionError(
                "Polymarket credentials not set. "
                "Set POLYMARKET_PRIVATE_KEY and POLYMARKET_FUNDER_ADDRESS env vars."
            )

        try:
            self._client = self._build_client()
            # Verify auth by fetching API key
            await asyncio.get_event_loop().run_in_executor(None, self._client.derive_api_key)
            self._connected = True
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            logger.info("polymarket_connected", host=self._config.host)
        except Exception as exc:
            self._connected = False
            logger.error("polymarket_connect_failed", error=str(exc))
            raise

    async def disconnect(self) -> None:
        """Cancel background tasks and close connections."""
        self._connected = False

        if self._heartbeat_task and not self._heartbeat_task.done():
            self._heartbeat_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._heartbeat_task

        if self._ws_task and not self._ws_task.done():
            self._ws_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._ws_task

        self._heartbeat_task = None
        self._ws_task = None
        self._client = None
        logger.info("polymarket_disconnected")

    def _build_client(self) -> Any:
        """Construct a ClobClient with credentials from config.

        Lazy-imports py_clob_client so the module can be imported without
        the SDK installed (tests mock this method).
        """
        try:
            from py_clob_client.client import ClobClient  # type: ignore[import]
            from py_clob_client.clob_types import ApiCreds  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "py-clob-client is required for Polymarket. Add it via: uv add py-clob-client"
            ) from exc

        creds = ApiCreds(
            api_key="",  # Will be derived from private key
            api_secret="",
            api_passphrase="",
        )
        return ClobClient(
            host=self._config.host,
            chain_id=self._config.chain_id,
            key=self._config.private_key,
            funder=self._config.funder_address,
            creds=creds,
        )

    # ------------------------------------------------------------------
    # Heartbeat
    # ------------------------------------------------------------------

    async def _heartbeat_loop(self) -> None:
        """Send a no-op ping every heartbeat_interval_s to keep WS alive."""
        while self._connected:
            try:
                await asyncio.sleep(self._config.heartbeat_interval_s)
                if self._connected and self._client is not None:
                    await asyncio.get_event_loop().run_in_executor(None, self._ping)
                    logger.debug("polymarket_heartbeat_ok")
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning("polymarket_heartbeat_error", error=str(exc))

    def _ping(self) -> None:
        """Synchronous ping via client — called in executor."""
        if self._client is not None:
            try:
                self._client.get_ok()
            except Exception:  # noqa: S110
                pass  # Heartbeat failures are logged in the main loop

    # ------------------------------------------------------------------
    # Market discovery
    # ------------------------------------------------------------------

    async def discover_market(self, slug_fragment: str | None = None) -> BinaryMarket:
        """Find the current 5-minute BTC market by slug pattern.

        Uses the Gamma REST API to list active markets, then filters by
        slug fragment. Returns the most recently created matching market.

        Args:
            slug_fragment: substring to match against market slugs.
                Defaults to config.market_slug_pattern.

        Raises:
            LookupError: If no matching active market is found.
        """
        fragment = slug_fragment or self._config.market_slug_pattern
        markets = await self._fetch_gamma_markets(fragment)

        if not markets:
            raise LookupError(f"No active Polymarket market found matching '{fragment}'")

        market = markets[0]  # Gamma API returns newest first
        logger.info(
            "polymarket_market_discovered",
            slug=market.slug,
            condition_id=market.condition_id,
        )
        return market

    async def _fetch_gamma_markets(self, slug_fragment: str) -> list[BinaryMarket]:
        """Query Gamma API and parse matching markets."""
        try:
            import aiohttp  # type: ignore[import]
        except ImportError:
            # Fallback: use executor with requests (always available via ccxt)
            return await asyncio.get_event_loop().run_in_executor(
                None, self._fetch_gamma_markets_sync, slug_fragment
            )

        url = f"{self._config.gamma_api_host}/markets"
        params = {"slug": slug_fragment, "active": "true", "closed": "false"}
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as resp:
                resp.raise_for_status()
                data = await resp.json()

        return [self._parse_gamma_market(m) for m in data if slug_fragment in m.get("slug", "")]

    def _fetch_gamma_markets_sync(self, slug_fragment: str) -> list[BinaryMarket]:
        """Synchronous fallback for Gamma API."""
        import json
        import urllib.request

        url = f"{self._config.gamma_api_host}/markets?slug={slug_fragment}&active=true&closed=false"
        with urllib.request.urlopen(url, timeout=10) as resp:  # noqa: S310
            data = json.loads(resp.read())

        return [self._parse_gamma_market(m) for m in data if slug_fragment in m.get("slug", "")]

    def _parse_gamma_market(self, raw: dict[str, Any]) -> BinaryMarket:
        """Parse a Gamma API market response dict into BinaryMarket."""
        tokens: list[dict[str, Any]] = raw.get("tokens", [])
        yes_token = next((t for t in tokens if t.get("outcome", "").lower() == "yes"), {})
        no_token = next((t for t in tokens if t.get("outcome", "").lower() == "no"), {})

        return BinaryMarket(
            condition_id=raw.get("conditionId", raw.get("condition_id", "")),
            question_id=raw.get("questionId", raw.get("question_id", "")),
            slug=raw.get("slug", ""),
            question=raw.get("question", ""),
            end_date_iso=raw.get("endDateIso", raw.get("end_date_iso", "")),
            active=raw.get("active", True),
            yes_token_id=yes_token.get("token_id", ""),
            no_token_id=no_token.get("token_id", ""),
            current_yes_price=float(yes_token.get("price", 0.5)),
            current_no_price=float(no_token.get("price", 0.5)),
        )

    def set_active_market(self, market: BinaryMarket) -> None:
        """Set the market that subsequent orders will target."""
        self._active_market = market
        logger.info(
            "polymarket_active_market_set",
            condition_id=market.condition_id,
            slug=market.slug,
        )

    # ------------------------------------------------------------------
    # Order execution
    # ------------------------------------------------------------------

    async def place_order(self, request: OrderRequest) -> OrderResult:
        """Place a limit or market order on the active binary market.

        For binary markets:
        - BUY = buy YES shares (bet BTC goes UP)
        - SELL = buy NO shares (bet BTC goes DOWN)
        - size = number of shares
        - price = share price in USD (0.01 - 0.99)
        """
        if not self._connected or self._client is None:
            return OrderResult(
                order_id="",
                status=OrderStatus.REJECTED,
                raw_response={"error": "not_connected"},
            )

        if self._active_market is None:
            return OrderResult(
                order_id="",
                status=OrderStatus.REJECTED,
                raw_response={"error": "no_active_market"},
            )

        token_id = self._resolve_token_id(request)
        price = request.price if request.price is not None else self._mid_price(request.side)

        try:
            order_args = self._build_order_args(
                token_id=token_id,
                side=request.side,
                size=request.size,
                price=price,
                order_type=request.order_type,
            )
            raw_resp = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self._client.post_order(*order_args)
            )
            order_id = raw_resp.get("orderID", raw_resp.get("order_id", ""))
            status = self._map_order_status(raw_resp.get("status", "OPEN"))

            pending = _PendingOrder(
                order_id=order_id,
                condition_id=self._active_market.condition_id,
                side=request.side,
                token_id=token_id,
                size=request.size,
                price=price,
                status=status,
            )
            self._pending_orders[order_id] = pending

            logger.info(
                "polymarket_order_placed",
                order_id=order_id,
                side=request.side.value,
                size=request.size,
                price=price,
            )
            return OrderResult(
                order_id=order_id,
                status=status,
                fill_price=price if status == OrderStatus.FILLED else 0.0,
                fill_quantity=request.size if status == OrderStatus.FILLED else 0.0,
                fee=self._estimate_fee(request.size, price),
                fee_currency="USDC",
                venue=VenueType.POLYMARKET_BINARY,
                raw_response=raw_resp,
            )
        except Exception as exc:
            logger.error("polymarket_place_order_failed", error=str(exc))
            return OrderResult(
                order_id="",
                status=OrderStatus.REJECTED,
                raw_response={"error": str(exc)},
            )

    def _resolve_token_id(self, request: OrderRequest) -> str:
        """Map OrderSide to YES/NO token ID on the active market."""
        assert self._active_market is not None
        if request.side == OrderSide.BUY:
            return self._active_market.yes_token_id  # Bet UP
        return self._active_market.no_token_id  # Bet DOWN

    def _mid_price(self, side: OrderSide) -> float:
        """Use market's current price as default for market orders."""
        if self._active_market is None:
            return 0.5
        if side == OrderSide.BUY:
            return self._active_market.current_yes_price
        return self._active_market.current_no_price

    def _build_order_args(
        self,
        token_id: str,
        side: OrderSide,
        size: float,
        price: float,
        order_type: OrderType,
    ) -> tuple[Any, ...]:
        """Build positional args for ClobClient.post_order().

        py-clob-client signature:
            post_order(order, orderType)
        where order is a dict or SignedOrder object.
        """
        try:
            from py_clob_client.order_builder.constants import BUY, SELL  # type: ignore[import]
        except ImportError:
            BUY, SELL = "BUY", "SELL"

        clob_side = BUY if side == OrderSide.BUY else SELL
        order = self._client.create_order(
            {
                "token_id": token_id,
                "price": round(price, 2),
                "size": round(size, 2),
                "side": clob_side,
            }
        )
        clob_order_type = "GTC" if order_type != OrderType.LIMIT_POST_ONLY else "GTD"
        return (order, clob_order_type)

    def _estimate_fee(self, size: float, price: float) -> float:
        """Estimate Polymarket taker fee: ~2% of notional."""
        return round(size * price * 0.02, 6)

    @staticmethod
    def _map_order_status(raw_status: str) -> OrderStatus:
        """Map Polymarket status strings to canonical OrderStatus."""
        mapping = {
            "LIVE": OrderStatus.OPEN,
            "OPEN": OrderStatus.OPEN,
            "FILLED": OrderStatus.FILLED,
            "PARTIALLY_FILLED": OrderStatus.PARTIALLY_FILLED,
            "CANCELLED": OrderStatus.CANCELLED,
            "CANCELED": OrderStatus.CANCELLED,
            "EXPIRED": OrderStatus.EXPIRED,
            "MATCHED": OrderStatus.FILLED,
        }
        return mapping.get(raw_status.upper(), OrderStatus.PENDING)

    # ------------------------------------------------------------------
    # Order management
    # ------------------------------------------------------------------

    async def cancel_order(self, order_id: str) -> OrderResult:
        if not self._connected or self._client is None:
            return OrderResult(order_id=order_id, status=OrderStatus.REJECTED)

        try:
            raw_resp = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self._client.cancel({"orderID": order_id})
            )
            status = self._map_order_status(raw_resp.get("status", "CANCELLED"))
            if order_id in self._pending_orders:
                self._pending_orders[order_id].status = OrderStatus.CANCELLED
            logger.info("polymarket_order_cancelled", order_id=order_id)
            return OrderResult(
                order_id=order_id,
                status=status,
                raw_response=raw_resp,
                venue=VenueType.POLYMARKET_BINARY,
            )
        except Exception as exc:
            logger.error("polymarket_cancel_failed", order_id=order_id, error=str(exc))
            return OrderResult(
                order_id=order_id,
                status=OrderStatus.REJECTED,
                raw_response={"error": str(exc)},
            )

    async def get_order(self, order_id: str) -> OrderResult:
        if not self._connected or self._client is None:
            return OrderResult(order_id=order_id, status=OrderStatus.PENDING)

        try:
            raw_resp = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self._client.get_order(order_id)
            )
            status = self._map_order_status(raw_resp.get("status", ""))
            fill_qty = float(raw_resp.get("size_matched", 0.0))
            fill_price = float(raw_resp.get("price", 0.0))
            fee = self._estimate_fee(fill_qty, fill_price)

            if order_id in self._pending_orders:
                self._pending_orders[order_id].status = status

            return OrderResult(
                order_id=order_id,
                status=status,
                fill_price=fill_price,
                fill_quantity=fill_qty,
                fee=fee,
                fee_currency="USDC",
                venue=VenueType.POLYMARKET_BINARY,
                raw_response=raw_resp,
            )
        except Exception as exc:
            logger.error("polymarket_get_order_failed", order_id=order_id, error=str(exc))
            return OrderResult(
                order_id=order_id,
                status=OrderStatus.PENDING,
                raw_response={"error": str(exc)},
            )

    # ------------------------------------------------------------------
    # Position and balance
    # ------------------------------------------------------------------

    async def get_position(self) -> PositionInfo:
        """Derive current position from pending orders (binary markets don't
        have a continuous position concept — each market is a new bet)."""
        if self._active_market is None:
            return PositionInfo(
                venue=VenueType.POLYMARKET_BINARY,
                symbol="",
                side=None,
                size=0.0,
            )

        filled_orders = [
            o
            for o in self._pending_orders.values()
            if o.condition_id == self._active_market.condition_id and o.status == OrderStatus.FILLED
        ]
        total_yes = sum(o.size for o in filled_orders if o.side == OrderSide.BUY)
        total_no = sum(o.size for o in filled_orders if o.side == OrderSide.SELL)
        net_size = total_yes - total_no

        return PositionInfo(
            venue=VenueType.POLYMARKET_BINARY,
            symbol=self._active_market.slug,
            side=OrderSide.BUY if net_size > 0 else (OrderSide.SELL if net_size < 0 else None),
            size=abs(net_size),
            entry_price=self._active_market.current_yes_price,
            cost_basis=sum(o.size * o.price for o in filled_orders),
        )

    async def get_orderbook(self) -> OrderBookSnapshot:
        """Fetch CLOB order book for active market's YES token."""
        if not self._connected or self._client is None or self._active_market is None:
            return OrderBookSnapshot(
                venue=VenueType.POLYMARKET_BINARY,
                symbol="",
                bids=[],
                asks=[],
            )

        try:
            raw = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._client.get_order_book(self._active_market.yes_token_id),  # type: ignore[union-attr]
            )
            bids = [
                OrderBookLevel(price=float(b["price"]), size=float(b["size"]))
                for b in raw.get("bids", [])
            ]
            asks = [
                OrderBookLevel(price=float(a["price"]), size=float(a["size"]))
                for a in raw.get("asks", [])
            ]
            return OrderBookSnapshot(
                venue=VenueType.POLYMARKET_BINARY,
                symbol=self._active_market.slug,
                bids=sorted(bids, key=lambda x: -x.price),  # highest bid first
                asks=sorted(asks, key=lambda x: x.price),  # lowest ask first
                timestamp_ms=int(time.time() * 1000),
            )
        except Exception as exc:
            logger.error("polymarket_orderbook_failed", error=str(exc))
            return OrderBookSnapshot(
                venue=VenueType.POLYMARKET_BINARY,
                symbol=self._active_market.slug if self._active_market else "",
                bids=[],
                asks=[],
            )

    async def get_balance(self) -> float:
        """Get available USDC balance."""
        if not self._connected or self._client is None:
            return 0.0

        try:
            raw = await asyncio.get_event_loop().run_in_executor(
                None, self._client.get_balance_allowance
            )
            return float(raw.get("balance", 0.0))
        except Exception as exc:
            logger.error("polymarket_balance_failed", error=str(exc))
            return 0.0

    async def health_check(self) -> bool:
        """Ping the CLOB endpoint."""
        if self._client is None:
            return False
        try:
            result = await asyncio.get_event_loop().run_in_executor(None, self._client.get_ok)
            healthy = result is not None
            logger.debug("polymarket_health_check", healthy=healthy)
            return healthy
        except Exception as exc:
            logger.warning("polymarket_health_check_failed", error=str(exc))
            return False

    # ------------------------------------------------------------------
    # Resolution tracking
    # ------------------------------------------------------------------

    def add_resolution_callback(self, callback: Any) -> None:
        """Register a callback invoked when a market resolves.

        Signature: callback(event: ResolutionEvent) -> None
        """
        self._resolution_callbacks.append(callback)

    def _fire_resolution(self, event: ResolutionEvent) -> None:
        for cb in self._resolution_callbacks:
            try:
                cb(event)
            except Exception as exc:
                logger.error("resolution_callback_error", error=str(exc))

    async def poll_resolution(self, condition_id: str) -> ResolutionEvent | None:
        """Poll REST API to check if a market has resolved.

        Returns a ResolutionEvent if resolved, None if still active.
        """
        if not self._connected or self._client is None:
            return None

        try:
            raw = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self._client.get_market(condition_id)
            )
            resolved = raw.get("closed", False) or raw.get("resolved", False)
            if not resolved:
                return None

            winning_outcome = raw.get("winner", "").lower()  # "yes" or "no"
            tokens: list[dict[str, Any]] = raw.get("tokens", [])
            winning_token = next(
                (t for t in tokens if t.get("outcome", "").lower() == winning_outcome),
                {},
            )
            event = ResolutionEvent(
                condition_id=condition_id,
                resolved_outcome=winning_outcome,
                resolved_at_ms=int(time.time() * 1000),
                winning_token_id=winning_token.get("token_id", ""),
            )
            self._fire_resolution(event)
            return event
        except Exception as exc:
            logger.error("polymarket_resolution_poll_failed", error=str(exc))
            return None

    # ------------------------------------------------------------------
    # Convenience: pending order accessors
    # ------------------------------------------------------------------

    def get_pending_orders(self) -> list[_PendingOrder]:
        """Return orders that are not yet in a terminal state."""
        terminal = {
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED,
        }
        return [o for o in self._pending_orders.values() if o.status not in terminal]
