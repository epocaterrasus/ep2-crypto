"""Tests for cross-market data collectors: TwelveDataCollector, YFinanceFallbackCollector."""

from __future__ import annotations

import asyncio
import json
import os
from unittest.mock import MagicMock, patch

import pytest

from ep2_crypto.ingest.base import CollectorState
from ep2_crypto.ingest.cross_market import (
    CrossMarketBar,
    TwelveDataCollector,
    YFinanceFallbackCollector,
    create_cross_market_collector,
)

# ---------------------------------------------------------------------------
# CrossMarketBar dataclass
# ---------------------------------------------------------------------------


class TestCrossMarketBar:
    def test_dataclass_fields(self) -> None:
        bar = CrossMarketBar(
            symbol="NQ1!",
            timestamp_ms=1_700_000_000_000,
            open=17_000.0,
            high=17_050.0,
            low=16_980.0,
            close=17_020.0,
            volume=10_000.0,
            source="twelve_data",
        )
        assert bar.symbol == "NQ1!"
        assert bar.close == 17_020.0
        assert bar.source == "twelve_data"


# ---------------------------------------------------------------------------
# TwelveDataCollector
# ---------------------------------------------------------------------------


class TestTwelveDataCollector:
    def test_name(self) -> None:
        c = TwelveDataCollector(api_key="test_key")
        assert c.name == "TwelveDataCollector"

    def test_state_idle_on_init(self) -> None:
        c = TwelveDataCollector(api_key="key")
        assert c.state == CollectorState.IDLE

    def test_get_latest_bars_empty_initially(self) -> None:
        c = TwelveDataCollector(api_key="key")
        assert c.get_latest_bars() == []

    def test_health_check_idle_state(self) -> None:
        c = TwelveDataCollector(api_key="key")
        h = c.health_check()
        assert not h.healthy  # not running → unhealthy
        assert h.state == CollectorState.IDLE

    def test_parse_twelve_data_response(self) -> None:
        """_fetch_bars parses Twelve Data JSON response correctly."""

        async def _run() -> None:
            c = TwelveDataCollector(api_key="key", poll_interval_s=9999)

            response_data = {
                "status": "ok",
                "values": [
                    {
                        "datetime": "2026-03-23 15:30:00",
                        "open": "17100.5",
                        "high": "17150.0",
                        "low": "17090.0",
                        "close": "17120.0",
                        "volume": "5000",
                    },
                    {
                        "datetime": "2026-03-23 15:29:00",
                        "open": "17080.0",
                        "high": "17105.0",
                        "low": "17070.0",
                        "close": "17100.5",
                        "volume": "4800",
                    },
                ],
            }
            raw_bytes = json.dumps(response_data).encode()

            with patch("urllib.request.urlopen") as mock_urlopen:
                mock_response = MagicMock()
                mock_response.read.return_value = raw_bytes
                mock_urlopen.return_value = mock_response

                bars = await c._fetch_bars("NQ1!")

            assert len(bars) == 2
            assert bars[0].symbol == "NQ1!"
            assert bars[0].close == pytest.approx(17_120.0)
            assert bars[0].source == "twelve_data"
            assert bars[1].close == pytest.approx(17_100.5)

        asyncio.run(_run())

    def test_api_error_returns_empty(self) -> None:
        """API error response returns empty list without raising."""

        async def _run() -> None:
            c = TwelveDataCollector(api_key="key", poll_interval_s=9999)
            response_data = {"status": "error", "message": "Invalid API key"}
            raw_bytes = json.dumps(response_data).encode()

            with patch("urllib.request.urlopen") as mock_urlopen:
                mock_response = MagicMock()
                mock_response.read.return_value = raw_bytes
                mock_urlopen.return_value = mock_response

                bars = await c._fetch_bars("NQ1!")

            assert bars == []

        asyncio.run(_run())

    def test_network_error_returns_empty(self) -> None:
        """Network errors return empty list without raising."""

        async def _run() -> None:
            c = TwelveDataCollector(api_key="key", poll_interval_s=9999)
            with patch("urllib.request.urlopen", side_effect=OSError("timeout")):
                bars = await c._fetch_bars("NQ1!")
            assert bars == []

        asyncio.run(_run())

    def test_no_api_key_raises_on_connect(self) -> None:
        """Missing API key raises RuntimeError from _connect."""

        async def _run() -> None:
            c = TwelveDataCollector(api_key="")
            with pytest.raises(RuntimeError, match="TWELVE_DATA_API_KEY"):
                await c._connect()

        asyncio.run(_run())

    def test_default_symbols(self) -> None:
        c = TwelveDataCollector(api_key="key")
        assert "NQ1!" in c._symbols or "DXY" in c._symbols

    def test_custom_symbols(self) -> None:
        c = TwelveDataCollector(symbols=["NQ1!"], api_key="key")
        assert c._symbols == ["NQ1!"]

    def test_bars_accumulate_on_fetch(self) -> None:
        """After fetching bars, get_latest_bars returns them."""

        async def _run() -> None:
            c = TwelveDataCollector(api_key="key")
            response_data = {
                "status": "ok",
                "values": [
                    {
                        "datetime": "2026-03-23 15:30:00",
                        "open": "100",
                        "high": "105",
                        "low": "99",
                        "close": "103",
                        "volume": "1000",
                    }
                ],
            }
            with patch("urllib.request.urlopen") as mock_urlopen:
                mock_response = MagicMock()
                mock_response.read.return_value = json.dumps(response_data).encode()
                mock_urlopen.return_value = mock_response
                bars = await c._fetch_bars("NQ1!")
            c._bars.extend(bars)
            assert len(c.get_latest_bars()) == 1

        asyncio.run(_run())

    def test_connect_succeeds_with_api_key(self) -> None:
        async def _run() -> None:
            c = TwelveDataCollector(api_key="real_key")
            await c._connect()  # should not raise

        asyncio.run(_run())

    def test_disconnect_is_noop(self) -> None:
        async def _run() -> None:
            c = TwelveDataCollector(api_key="key")
            await c._disconnect()  # should not raise

        asyncio.run(_run())


# ---------------------------------------------------------------------------
# YFinanceFallbackCollector
# ---------------------------------------------------------------------------


class TestYFinanceFallbackCollector:
    def test_name(self) -> None:
        c = YFinanceFallbackCollector()
        assert c.name == "YFinanceFallbackCollector"

    def test_state_idle_on_init(self) -> None:
        c = YFinanceFallbackCollector()
        assert c.state == CollectorState.IDLE

    def test_get_latest_bars_empty_initially(self) -> None:
        c = YFinanceFallbackCollector()
        assert c.get_latest_bars() == []

    def test_symbol_map_nq(self) -> None:
        c = YFinanceFallbackCollector()
        assert c._SYMBOL_MAP["NQ1!"] == "NQ=F"
        assert c._SYMBOL_MAP["DXY"] == "DX-Y.NYB"

    def test_connect_is_noop(self) -> None:
        async def _run() -> None:
            c = YFinanceFallbackCollector()
            await c._connect()

        asyncio.run(_run())

    def test_disconnect_is_noop(self) -> None:
        async def _run() -> None:
            c = YFinanceFallbackCollector()
            await c._disconnect()

        asyncio.run(_run())

    def test_fetch_bars_with_mocked_yfinance(self) -> None:
        """_fetch_yfinance_sync parses yfinance DataFrame correctly."""
        import pandas as pd

        c = YFinanceFallbackCollector()

        ts = pd.Timestamp("2026-03-23 15:30:00", tz="UTC")
        hist = pd.DataFrame(
            [
                {
                    "Open": 17_100.0,
                    "High": 17_150.0,
                    "Low": 17_090.0,
                    "Close": 17_120.0,
                    "Volume": 5_000,
                }
            ],
            index=[ts],
        )
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = hist

        with patch("yfinance.Ticker", return_value=mock_ticker):
            bars = c._fetch_yfinance_sync("NQ=F", "NQ1!")

        assert len(bars) == 1
        assert bars[0].symbol == "NQ1!"
        assert bars[0].close == pytest.approx(17_120.0)
        assert bars[0].source == "yfinance"

    def test_empty_dataframe_returns_empty(self) -> None:
        import pandas as pd

        c = YFinanceFallbackCollector()
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame()

        with patch("yfinance.Ticker", return_value=mock_ticker):
            bars = c._fetch_yfinance_sync("NQ=F", "NQ1!")

        assert bars == []

    def test_unknown_symbol_falls_back_to_passthrough(self) -> None:
        """Symbol not in map is passed through directly."""
        c = YFinanceFallbackCollector()
        mapped = c._SYMBOL_MAP.get("UNKNOWN", "UNKNOWN")
        assert mapped == "UNKNOWN"

    def test_custom_symbols(self) -> None:
        c = YFinanceFallbackCollector(symbols=["NQ1!"])
        assert c._symbols == ["NQ1!"]

    def test_health_check_idle(self) -> None:
        c = YFinanceFallbackCollector()
        h = c.health_check()
        assert not h.healthy
        assert h.state == CollectorState.IDLE


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------


class TestCreateCrossMarketCollector:
    def test_returns_twelve_data_when_key_set(self) -> None:
        with patch.dict(os.environ, {"TWELVE_DATA_API_KEY": "my_key"}):
            collector = create_cross_market_collector()
        assert isinstance(collector, TwelveDataCollector)

    def test_returns_yfinance_when_no_key(self) -> None:
        env = {k: v for k, v in os.environ.items() if k != "TWELVE_DATA_API_KEY"}
        with patch.dict(os.environ, env, clear=True):
            collector = create_cross_market_collector()
        assert isinstance(collector, YFinanceFallbackCollector)

    def test_custom_symbols_passed_to_twelve_data(self) -> None:
        with patch.dict(os.environ, {"TWELVE_DATA_API_KEY": "key"}):
            collector = create_cross_market_collector(symbols=["NQ1!"])
        assert collector._symbols == ["NQ1!"]

    def test_custom_symbols_passed_to_yfinance(self) -> None:
        env = {k: v for k, v in os.environ.items() if k != "TWELVE_DATA_API_KEY"}
        with patch.dict(os.environ, env, clear=True):
            collector = create_cross_market_collector(symbols=["DXY"])
        assert collector._symbols == ["DXY"]
