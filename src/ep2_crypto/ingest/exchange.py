"""Binance exchange data collectors using ccxt pro WebSocket streams.

Collectors for klines (OHLCV), order book depth, and aggregated trades.
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


class BinanceKlineCollector(BaseCollector):
    """Collects 1-minute klines from Binance USD-M futures via ccxt pro.

    Uses watch_ohlcv for real-time candle updates. Each update stores
    the latest completed candle to the ohlcv table.
    """

    def __init__(
        self,
        repository: Repository,
        *,
        symbol: str = "BTC/USDT:USDT",
        timeframe: str = "1m",
        exchange_class: Any = None,
        exchange_config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(name=f"binance_kline_{symbol}_{timeframe}", **kwargs)
        self._repository = repository
        self._symbol = symbol
        self._timeframe = timeframe
        self._exchange: Any = None
        self._exchange_class = exchange_class
        self._exchange_config = exchange_config or {}
        self._last_candle_ts: int | None = None
        self._log = logger.bind(
            collector=self.name,
            symbol=symbol,
            timeframe=timeframe,
        )

    async def _connect(self) -> None:
        self._exchange = _create_exchange(self._exchange_class, self._exchange_config)
        self._log.info("exchange_connected")

    async def _disconnect(self) -> None:
        if self._exchange is not None:
            try:
                await self._exchange.close()
            except Exception:
                self._log.exception("exchange_close_error")
            self._exchange = None

    async def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                ohlcv_list = await asyncio.wait_for(
                    self._exchange.watch_ohlcv(self._symbol, self._timeframe),
                    timeout=60.0,
                )
            except TimeoutError:
                self._log.warning("watch_ohlcv_timeout")
                continue
            except asyncio.CancelledError:
                raise
            except Exception:
                self._log.exception("watch_ohlcv_error")
                raise

            if not ohlcv_list:
                continue

            self._record_message(time.time())
            self._process_candles(ohlcv_list)

    def _process_candles(self, ohlcv_list: list[list[Any]]) -> None:
        """Store completed candles (skip the in-progress latest candle)."""
        for candle in ohlcv_list:
            timestamp_ms = int(candle[0])

            # Skip if we've already processed this timestamp
            if self._last_candle_ts is not None and timestamp_ms <= self._last_candle_ts:
                continue

            # The last candle in the list is typically still forming
            # We store it anyway since watch_ohlcv updates it in place;
            # INSERT OR REPLACE handles the upsert
            self._repository.insert_ohlcv(
                timestamp_ms=timestamp_ms,
                symbol=self._symbol,
                interval=self._timeframe,
                open_=float(candle[1]),
                high=float(candle[2]),
                low=float(candle[3]),
                close=float(candle[4]),
                volume=float(candle[5]),
            )

            self._last_candle_ts = timestamp_ms

        self._log.debug(
            "candles_processed",
            count=len(ohlcv_list),
            latest_ts=self._last_candle_ts,
        )


def _create_exchange(exchange_class: Any, exchange_config: dict[str, Any]) -> Any:
    """Create exchange instance, using ccxt pro binanceusdm if no class provided."""
    if exchange_class is not None:
        return exchange_class(exchange_config)

    import ccxt.pro as ccxtpro

    return ccxtpro.binanceusdm(
        {
            "enableRateLimit": True,
            **exchange_config,
        }
    )


class BinanceDepthCollector(BaseCollector):
    """Collects order book depth snapshots from Binance via ccxt pro.

    Uses watch_order_book with configurable depth limit (default 20 levels).
    Stores top bid/ask prices and sizes as JSON-serialized arrays.
    """

    def __init__(
        self,
        repository: Repository,
        *,
        symbol: str = "BTC/USDT:USDT",
        depth_limit: int = 20,
        exchange_class: Any = None,
        exchange_config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(name=f"binance_depth_{symbol}", **kwargs)
        self._repository = repository
        self._symbol = symbol
        self._depth_limit = depth_limit
        self._exchange: Any = None
        self._exchange_class = exchange_class
        self._exchange_config = exchange_config or {}
        self._log = logger.bind(collector=self.name, symbol=symbol)

    async def _connect(self) -> None:
        self._exchange = _create_exchange(self._exchange_class, self._exchange_config)
        self._log.info("exchange_connected")

    async def _disconnect(self) -> None:
        if self._exchange is not None:
            try:
                await self._exchange.close()
            except Exception:
                self._log.exception("exchange_close_error")
            self._exchange = None

    async def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                orderbook = await asyncio.wait_for(
                    self._exchange.watch_order_book(
                        self._symbol,
                        limit=self._depth_limit,
                    ),
                    timeout=60.0,
                )
            except TimeoutError:
                self._log.warning("watch_orderbook_timeout")
                continue
            except asyncio.CancelledError:
                raise
            except Exception:
                self._log.exception("watch_orderbook_error")
                raise

            if not orderbook:
                continue

            self._record_message(time.time())
            self._process_orderbook(orderbook)

    def _process_orderbook(self, orderbook: dict[str, Any]) -> None:
        """Extract and store top N bid/ask levels."""
        bids = orderbook.get("bids", [])[: self._depth_limit]
        asks = orderbook.get("asks", [])[: self._depth_limit]

        if not bids or not asks:
            return

        bid_prices = json.dumps([float(b[0]) for b in bids])
        bid_sizes = json.dumps([float(b[1]) for b in bids])
        ask_prices = json.dumps([float(a[0]) for a in asks])
        ask_sizes = json.dumps([float(a[1]) for a in asks])

        best_bid = float(bids[0][0])
        best_ask = float(asks[0][0])
        mid_price = (best_bid + best_ask) / 2.0
        spread = best_ask - best_bid

        timestamp_ms = int(orderbook.get("timestamp", time.time() * 1000))

        self._repository.insert_orderbook(
            timestamp_ms=timestamp_ms,
            symbol=self._symbol,
            bid_prices=bid_prices,
            bid_sizes=bid_sizes,
            ask_prices=ask_prices,
            ask_sizes=ask_sizes,
            mid_price=mid_price,
            spread=spread,
        )

        self._log.debug(
            "orderbook_stored",
            mid_price=mid_price,
            spread=spread,
            depth=len(bids),
        )


class BinanceTradeCollector(BaseCollector):
    """Collects aggregated trades from Binance via ccxt pro.

    Uses watch_trades for real-time trade stream. Stores each trade
    to the agg_trades table.
    """

    def __init__(
        self,
        repository: Repository,
        *,
        symbol: str = "BTC/USDT:USDT",
        exchange_class: Any = None,
        exchange_config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(name=f"binance_trades_{symbol}", **kwargs)
        self._repository = repository
        self._symbol = symbol
        self._exchange: Any = None
        self._exchange_class = exchange_class
        self._exchange_config = exchange_config or {}
        self._last_trade_id: str | None = None
        self._log = logger.bind(collector=self.name, symbol=symbol)

    async def _connect(self) -> None:
        self._exchange = _create_exchange(self._exchange_class, self._exchange_config)
        self._log.info("exchange_connected")

    async def _disconnect(self) -> None:
        if self._exchange is not None:
            try:
                await self._exchange.close()
            except Exception:
                self._log.exception("exchange_close_error")
            self._exchange = None

    async def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                trades = await asyncio.wait_for(
                    self._exchange.watch_trades(self._symbol),
                    timeout=60.0,
                )
            except TimeoutError:
                self._log.warning("watch_trades_timeout")
                continue
            except asyncio.CancelledError:
                raise
            except Exception:
                self._log.exception("watch_trades_error")
                raise

            if not trades:
                continue

            self._record_message(time.time())
            self._process_trades(trades)

    def _process_trades(self, trades: list[dict[str, Any]]) -> None:
        """Store new trades, deduplicating by trade ID."""
        rows: list[tuple[Any, ...]] = []
        for trade in trades:
            trade_id = str(trade.get("id", ""))

            # Skip duplicates
            if self._last_trade_id is not None and trade_id == self._last_trade_id:
                continue

            rows.append(
                (
                    int(trade["timestamp"]),
                    self._symbol,
                    float(trade["price"]),
                    float(trade["amount"]),
                    int(trade.get("side", "") == "sell"),  # is_buyer_maker
                    trade_id,
                )
            )

        if rows:
            self._repository.insert_trades_batch(rows)
            self._last_trade_id = rows[-1][5]  # last trade_id

            self._log.debug("trades_stored", count=len(rows))
