"""Bybit derivatives data collectors for OI, funding rate, and liquidations.

OI and funding rate use REST polling. Liquidations use WebSocket stream.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import TYPE_CHECKING, Any

import structlog

from ep2_crypto.ingest.base import BaseCollector

if TYPE_CHECKING:
    from ep2_crypto.db.repository import Repository

logger = structlog.get_logger(__name__)


class BybitOICollector(BaseCollector):
    """Polls Bybit REST API for open interest data.

    Fetches open interest every poll_interval_s seconds and stores
    to the open_interest table.
    """

    def __init__(
        self,
        repository: Repository,
        *,
        symbol: str = "BTCUSDT",
        unified_symbol: str = "BTC/USDT:USDT",
        poll_interval_s: float = 300.0,
        exchange_class: Any = None,
        exchange_config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(name=f"bybit_oi_{symbol}", **kwargs)
        self._repository = repository
        self._symbol = symbol
        self._unified_symbol = unified_symbol
        self._poll_interval_s = poll_interval_s
        self._exchange: Any = None
        self._exchange_class = exchange_class
        self._exchange_config = exchange_config or {}
        self._log = logger.bind(collector=self.name, symbol=symbol)

    async def _connect(self) -> None:
        if self._exchange_class is not None:
            self._exchange = self._exchange_class(self._exchange_config)
        else:
            import ccxt

            self._exchange = ccxt.bybit(
                {
                    "enableRateLimit": True,
                    **self._exchange_config,
                }
            )
        self._log.info("exchange_connected")

    async def _disconnect(self) -> None:
        self._exchange = None

    async def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                await self._fetch_oi()
            except asyncio.CancelledError:
                raise
            except Exception:
                self._log.exception("fetch_oi_error")
                raise

            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=self._poll_interval_s,
                )
                return  # stop requested
            except TimeoutError:
                pass

    async def _fetch_oi(self) -> None:
        """Fetch open interest from exchange and store."""
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            self._exchange.fetch_open_interest,
            self._unified_symbol,
        )

        if not result:
            return

        self._record_message(time.time())

        timestamp_ms = int(result.get("timestamp", time.time() * 1000))
        oi_value = float(result.get("openInterestAmount", 0))
        raw_usd = result.get("openInterestValue")
        oi_value_usd = float(raw_usd) if raw_usd else None

        self._repository.insert_open_interest(
            timestamp_ms=timestamp_ms,
            symbol=self._unified_symbol,
            open_interest=oi_value,
            oi_value_usd=oi_value_usd,
        )

        self._log.debug("oi_stored", oi=oi_value, timestamp_ms=timestamp_ms)


class BybitFundingCollector(BaseCollector):
    """Polls Bybit REST API for funding rate data.

    Fetches funding rate every poll_interval_s seconds and stores
    to the funding_rate table.
    """

    def __init__(
        self,
        repository: Repository,
        *,
        symbol: str = "BTCUSDT",
        unified_symbol: str = "BTC/USDT:USDT",
        poll_interval_s: float = 300.0,
        exchange_class: Any = None,
        exchange_config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(name=f"bybit_funding_{symbol}", **kwargs)
        self._repository = repository
        self._symbol = symbol
        self._unified_symbol = unified_symbol
        self._poll_interval_s = poll_interval_s
        self._exchange: Any = None
        self._exchange_class = exchange_class
        self._exchange_config = exchange_config or {}
        self._log = logger.bind(collector=self.name, symbol=symbol)

    async def _connect(self) -> None:
        if self._exchange_class is not None:
            self._exchange = self._exchange_class(self._exchange_config)
        else:
            import ccxt

            self._exchange = ccxt.bybit(
                {
                    "enableRateLimit": True,
                    **self._exchange_config,
                }
            )
        self._log.info("exchange_connected")

    async def _disconnect(self) -> None:
        self._exchange = None

    async def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                await self._fetch_funding()
            except asyncio.CancelledError:
                raise
            except Exception:
                self._log.exception("fetch_funding_error")
                raise

            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=self._poll_interval_s,
                )
                return
            except TimeoutError:
                pass

    async def _fetch_funding(self) -> None:
        """Fetch funding rate from exchange and store."""
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            self._exchange.fetch_funding_rate,
            self._unified_symbol,
        )

        if not result:
            return

        self._record_message(time.time())

        timestamp_ms = int(result.get("timestamp", time.time() * 1000))
        funding_rate = float(result.get("fundingRate", 0))
        mark_price = float(result["markPrice"]) if result.get("markPrice") else None
        index_price = float(result["indexPrice"]) if result.get("indexPrice") else None

        self._repository.insert_funding_rate(
            timestamp_ms=timestamp_ms,
            symbol=self._unified_symbol,
            funding_rate=funding_rate,
            mark_price=mark_price,
            index_price=index_price,
        )

        self._log.debug(
            "funding_stored",
            funding_rate=funding_rate,
            timestamp_ms=timestamp_ms,
        )


class BybitLiquidationCollector(BaseCollector):
    """Collects liquidation events from Bybit via raw WebSocket.

    Connects to wss://stream.bybit.com/v5/public/linear and subscribes
    to allLiquidation.BTCUSDT. Sends heartbeat pings every 20s.
    """

    def __init__(
        self,
        repository: Repository,
        *,
        symbol: str = "BTCUSDT",
        unified_symbol: str = "BTC/USDT:USDT",
        ws_url: str = "wss://stream.bybit.com/v5/public/linear",
        ws_connector: Any = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(name=f"bybit_liquidation_{symbol}", **kwargs)
        self._repository = repository
        self._symbol = symbol
        self._unified_symbol = unified_symbol
        self._ws_url = ws_url
        self._ws: Any = None
        self._ws_connector = ws_connector
        self._log = logger.bind(collector=self.name, symbol=symbol)

    async def _connect(self) -> None:
        if self._ws_connector is not None:
            self._ws = await self._ws_connector(self._ws_url)
        else:
            import websockets

            self._ws = await websockets.connect(self._ws_url)

        # Subscribe to liquidation topic
        subscribe_msg = json.dumps(
            {
                "op": "subscribe",
                "args": [f"allLiquidation.{self._symbol}"],
            }
        )
        await self._ws.send(subscribe_msg)
        self._log.info("ws_connected_and_subscribed")

    async def _disconnect(self) -> None:
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                self._log.exception("ws_close_error")
            self._ws = None

    async def _run_loop(self) -> None:
        ping_interval = 20.0
        last_ping = time.monotonic()

        while not self._stop_event.is_set():
            # Send heartbeat if needed
            now = time.monotonic()
            if now - last_ping >= ping_interval:
                try:
                    await self._ws.send(json.dumps({"op": "ping"}))
                    last_ping = now
                except Exception:
                    self._log.exception("ping_error")
                    raise

            try:
                raw = await asyncio.wait_for(self._ws.recv(), timeout=5.0)
            except TimeoutError:
                continue
            except asyncio.CancelledError:
                raise
            except Exception:
                self._log.exception("ws_recv_error")
                raise

            self._process_message(raw)

    def _process_message(self, raw: str) -> None:
        """Parse and store liquidation events."""
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            self._log.warning("invalid_json", raw=raw[:200])
            return

        # Skip non-data messages (pong, subscription confirmation)
        if msg.get("op") in ("pong", "subscribe"):
            return

        topic = msg.get("topic", "")
        if not topic.startswith("allLiquidation"):
            return

        data = msg.get("data", {})
        if not data:
            return

        self._record_message(time.time())

        timestamp_ms = int(data.get("updatedTime", time.time() * 1000))
        side = str(data.get("side", "")).lower()
        # Bybit liquidation side: "Buy" means a short was liquidated (buy to close)
        # We store the position side that was liquidated
        position_side = "short" if side == "buy" else "long"

        self._repository.insert_liquidation(
            timestamp_ms=timestamp_ms,
            symbol=self._unified_symbol,
            side=position_side,
            price=float(data.get("price", 0)),
            quantity=float(data.get("size", 0)),
        )

        self._log.debug(
            "liquidation_stored",
            side=position_side,
            price=data.get("price"),
            size=data.get("size"),
        )
