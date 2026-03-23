"""Cross-market data collectors: NQ/DXY via Twelve Data or yfinance fallback.

Twelve Data ($29/mo) provides 1-min NQ/DXY data with ~1-min delay.
yfinance is the free fallback (15-min delayed, acceptable for regime context).

Priority: TwelveDataCollector if TWELVE_DATA_API_KEY is set, else YFinanceFallbackCollector.

Environment variables:
    TWELVE_DATA_API_KEY: API key for Twelve Data (optional; uses yfinance if absent)
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass

import structlog

from ep2_crypto.ingest.base import BaseCollector, CollectorState, HealthStatus

logger = structlog.get_logger(__name__)


@dataclass
class CrossMarketBar:
    """A single OHLCV bar for a cross-market instrument."""

    symbol: str  # e.g. "NQ1!", "DXY"
    timestamp_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    source: str  # "twelve_data" | "yfinance"


class TwelveDataCollector(BaseCollector):
    """REST polling collector for NQ and DXY via Twelve Data API.

    Polls every `poll_interval_s` seconds (default 60s for 1-min bars).
    Requires TWELVE_DATA_API_KEY environment variable.

    Twelve Data endpoint: https://api.twelvedata.com/time_series
    Parameters: symbol, interval=1min, outputsize=5, apikey

    Data is available for:
    - NQ (NASDAQ futures): symbol="NQ1!"
    - DXY (US Dollar index): symbol="DXY"
    """

    _TWELVE_DATA_BASE = "https://api.twelvedata.com"

    def __init__(
        self,
        symbols: list[str] | None = None,
        poll_interval_s: float = 60.0,
        *,
        api_key: str | None = None,
    ) -> None:
        super().__init__("TwelveDataCollector")
        self._symbols = symbols or ["NQ1!", "DXY"]
        self._poll_interval_s = poll_interval_s
        self._api_key = api_key or os.environ.get("TWELVE_DATA_API_KEY", "")
        self._bars: list[CrossMarketBar] = []

    async def _connect(self) -> None:
        if not self._api_key:
            msg = "TWELVE_DATA_API_KEY is not set"
            raise RuntimeError(msg)
        self._log.info("twelve_data_connected", symbols=self._symbols)

    async def _disconnect(self) -> None:
        pass  # REST polling — no persistent connection

    async def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            for symbol in self._symbols:
                bars = await self._fetch_bars(symbol)
                self._bars.extend(bars)
                for _ in bars:
                    self._record_message(time.time())
            # Wait for poll_interval or until stop
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=self._poll_interval_s,
                )
            except asyncio.TimeoutError:
                pass

    def get_latest_bars(self) -> list[CrossMarketBar]:
        return list(self._bars)

    async def _fetch_bars(self, symbol: str) -> list[CrossMarketBar]:
        """Fetch latest 5 bars for a symbol from Twelve Data REST API.

        Returns an empty list on error rather than raising, to allow the
        collector to continue polling other symbols.
        """
        params = urllib.parse.urlencode({
            "symbol": symbol,
            "interval": "1min",
            "outputsize": 5,
            "apikey": self._api_key,
            "format": "JSON",
        })
        url = f"{self._TWELVE_DATA_BASE}/time_series?{params}"

        try:
            loop = asyncio.get_event_loop()
            raw = await loop.run_in_executor(
                None, lambda: urllib.request.urlopen(url, timeout=10).read()
            )
            data = json.loads(raw)
        except Exception as exc:
            self._log.warning("twelve_data_fetch_failed", symbol=symbol, error=str(exc))
            return []

        if data.get("status") == "error":
            self._log.warning(
                "twelve_data_api_error",
                symbol=symbol,
                message=data.get("message", "unknown"),
            )
            return []

        values = data.get("values", [])
        bars: list[CrossMarketBar] = []
        for v in values:
            try:
                dt_str = v["datetime"]  # "2026-03-23 15:30:00"
                import datetime
                dt = datetime.datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
                ts_ms = int(dt.timestamp() * 1000)
                bars.append(CrossMarketBar(
                    symbol=symbol,
                    timestamp_ms=ts_ms,
                    open=float(v["open"]),
                    high=float(v["high"]),
                    low=float(v["low"]),
                    close=float(v["close"]),
                    volume=float(v.get("volume", 0)),
                    source="twelve_data",
                ))
            except (KeyError, ValueError) as exc:
                self._log.warning("twelve_data_parse_error", symbol=symbol, error=str(exc))
        return bars


class YFinanceFallbackCollector(BaseCollector):
    """yfinance-based collector for NQ and DXY (15-min delayed, free).

    Used when TWELVE_DATA_API_KEY is not set. Polls every 5 minutes
    matching the BTC bar cadence.

    yfinance symbols:
    - NQ futures: "NQ=F"
    - DXY index: "DX-Y.NYB"
    """

    _SYMBOL_MAP = {
        "NQ1!": "NQ=F",
        "DXY": "DX-Y.NYB",
        "NQ=F": "NQ=F",
        "DX-Y.NYB": "DX-Y.NYB",
    }

    def __init__(
        self,
        symbols: list[str] | None = None,
        poll_interval_s: float = 300.0,
    ) -> None:
        super().__init__("YFinanceFallbackCollector")
        self._symbols = symbols or ["NQ1!", "DXY"]
        self._poll_interval_s = poll_interval_s
        self._bars: list[CrossMarketBar] = []

    async def _connect(self) -> None:
        self._log.info("yfinance_collector_connected", symbols=self._symbols)

    async def _disconnect(self) -> None:
        pass  # REST polling — no persistent connection

    async def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            for symbol in self._symbols:
                bars = await self._fetch_bars(symbol)
                self._bars.extend(bars)
                for _ in bars:
                    self._record_message(time.time())
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=self._poll_interval_s,
                )
            except asyncio.TimeoutError:
                pass

    def get_latest_bars(self) -> list[CrossMarketBar]:
        return list(self._bars)

    async def _fetch_bars(self, symbol: str) -> list[CrossMarketBar]:
        """Fetch latest bars via yfinance. Runs in thread executor to avoid blocking."""
        yf_symbol = self._SYMBOL_MAP.get(symbol, symbol)
        loop = asyncio.get_event_loop()
        try:
            bars = await loop.run_in_executor(
                None, self._fetch_yfinance_sync, yf_symbol, symbol
            )
        except Exception as exc:
            self._log.warning("yfinance_fetch_failed", symbol=symbol, error=str(exc))
            return []
        return bars

    def _fetch_yfinance_sync(self, yf_symbol: str, orig_symbol: str) -> list[CrossMarketBar]:
        """Synchronous yfinance fetch (runs in thread executor)."""
        try:
            import yfinance as yf  # type: ignore[import-untyped]
        except ImportError:
            self._log.error("yfinance_not_installed")
            return []

        ticker = yf.Ticker(yf_symbol)
        hist = ticker.history(period="1d", interval="5m")
        if hist.empty:
            return []

        bars: list[CrossMarketBar] = []
        for ts, row in hist.iterrows():
            ts_ms = int(ts.timestamp() * 1000)
            bars.append(CrossMarketBar(
                symbol=orig_symbol,
                timestamp_ms=ts_ms,
                open=float(row["Open"]),
                high=float(row["High"]),
                low=float(row["Low"]),
                close=float(row["Close"]),
                volume=float(row.get("Volume", 0)),
                source="yfinance",
            ))
        return bars


def create_cross_market_collector(
    symbols: list[str] | None = None,
    poll_interval_s: float = 60.0,
) -> TwelveDataCollector | YFinanceFallbackCollector:
    """Factory: returns TwelveDataCollector if API key is set, else yfinance fallback."""
    api_key = os.environ.get("TWELVE_DATA_API_KEY", "")
    if api_key:
        logger.info("cross_market_using_twelve_data")
        return TwelveDataCollector(symbols=symbols, poll_interval_s=poll_interval_s, api_key=api_key)
    logger.info("cross_market_using_yfinance_fallback", reason="TWELVE_DATA_API_KEY not set")
    return YFinanceFallbackCollector(symbols=symbols, poll_interval_s=poll_interval_s)
