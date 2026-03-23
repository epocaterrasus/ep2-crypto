"""Deribit options data collector: IV surface, 25-delta risk reversal, ATM IV.

Connects to Deribit public REST API to fetch BTC options implied volatility:
- ATM IV (at-the-money implied volatility) at the nearest expiry
- 25-delta risk reversal (call IV - put IV at 25-delta): skew signal
- Term structure: ATM IV across multiple expiries (7d, 14d, 30d)

No authentication required for public endpoints (read-only market data).

Usage:
    collector = DeribitCollector(on_snapshot=callback, poll_interval_s=60.0)
    async with collector:
        # callback receives IVSnapshot objects
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from typing import Any

import structlog

from ep2_crypto.ingest.base import BaseCollector

logger = structlog.get_logger(__name__)

_DERIBIT_BASE = "https://www.deribit.com/api/v2"


@dataclass
class IVSnapshot:
    """Implied volatility snapshot from Deribit options market.

    Attributes:
        timestamp_ms: Unix timestamp in milliseconds.
        atm_iv_7d: ATM implied volatility (~7-day expiry), annualized.
        atm_iv_14d: ATM IV (~14-day expiry), annualized. None if unavailable.
        atm_iv_30d: ATM IV (~30-day expiry), annualized. None if unavailable.
        rr_25d_7d: 25-delta risk reversal at ~7-day expiry (call_iv - put_iv).
        spot_price: BTC spot reference price used for ATM strike selection.
        n_calls: Number of call options in the surface.
        n_puts: Number of put options in the surface.
    """

    timestamp_ms: int
    atm_iv_7d: float
    atm_iv_14d: float | None
    atm_iv_30d: float | None
    rr_25d_7d: float | None
    spot_price: float
    n_calls: int
    n_puts: int


SnapshotCallback = Callable[[IVSnapshot], Coroutine[Any, Any, None]]


def _closest_strike(strikes: list[float], spot: float) -> float:
    """Return the strike closest to spot price."""
    return min(strikes, key=lambda s: abs(s - spot))


def _parse_iv_surface(
    instruments: list[dict[str, Any]],
    ticker_map: dict[str, dict[str, Any]],
    spot: float,
    target_days: int,
    tolerance_days: int = 5,
) -> tuple[float | None, float | None]:
    """Extract ATM IV and 25-delta RR from options surface at target expiry.

    Args:
        instruments: List of instrument dicts from Deribit API.
        ticker_map: Mapping instrument_name -> ticker data.
        spot: Current BTC spot price.
        target_days: Target days-to-expiry.
        tolerance_days: Accept expiries within +/- tolerance_days.

    Returns:
        Tuple of (atm_iv, rr_25d). Either may be None if insufficient data.
    """
    now_s = time.time()
    target_s = target_days * 86400.0
    tol_s = tolerance_days * 86400.0

    # Group instruments by expiry, filter to target window
    expiry_groups: dict[int, list[dict[str, Any]]] = {}
    for inst in instruments:
        exp_ts = inst.get("expiration_timestamp", 0) / 1000.0
        dte_s = exp_ts - now_s
        if abs(dte_s - target_s) <= tol_s and dte_s > 0:
            expiry_groups.setdefault(inst["expiration_timestamp"], []).append(inst)

    if not expiry_groups:
        return None, None

    # Pick closest expiry to target
    best_exp = min(
        expiry_groups.keys(),
        key=lambda e: abs(e / 1000.0 - now_s - target_s),
    )
    group = expiry_groups[best_exp]

    # Separate calls and puts, build strike -> iv maps
    call_ivs: dict[float, float] = {}
    put_ivs: dict[float, float] = {}

    for inst in group:
        name = inst["instrument_name"]
        ticker = ticker_map.get(name)
        if ticker is None:
            continue
        iv = ticker.get("mark_iv")
        if iv is None or iv <= 0:
            continue

        strike = float(inst["strike"])
        opt_type = inst.get("option_type", "")
        if opt_type == "call":
            call_ivs[strike] = iv / 100.0  # Deribit returns IV as percentage
        elif opt_type == "put":
            put_ivs[strike] = iv / 100.0

    # ATM IV: call IV at strike closest to spot
    atm_iv: float | None = None
    if call_ivs:
        atm_strike = _closest_strike(list(call_ivs.keys()), spot)
        atm_iv = call_ivs[atm_strike]

    # 25-delta RR: approximate by using OTM call and OTM put nearest to 25-delta
    # Simple approximation: call at 1.05*spot and put at 0.95*spot
    rr_25d: float | None = None
    if call_ivs and put_ivs:
        otm_call_strike = min(
            (s for s in call_ivs if s > spot),
            key=lambda s: abs(s - spot * 1.05),
            default=None,
        )
        otm_put_strike = min(
            (s for s in put_ivs if s < spot),
            key=lambda s: abs(s - spot * 0.95),
            default=None,
        )
        if otm_call_strike is not None and otm_put_strike is not None:
            rr_25d = call_ivs[otm_call_strike] - put_ivs[otm_put_strike]

    return atm_iv, rr_25d


class DeribitCollector(BaseCollector):
    """Polls Deribit REST API for BTC options IV surface at configurable interval.

    Emits IVSnapshot objects via an async callback for downstream feature computation.

    Requires aiohttp to be installed (included in project extras).

    Args:
        on_snapshot: Async callback receiving IVSnapshot on each poll.
        poll_interval_s: Seconds between polls (default 60s to avoid rate limits).
        currency: Options currency (default "BTC").
    """

    def __init__(
        self,
        on_snapshot: SnapshotCallback | None = None,
        poll_interval_s: float = 60.0,
        currency: str = "BTC",
    ) -> None:
        super().__init__(name="deribit_options")
        self._on_snapshot = on_snapshot
        self._poll_interval_s = poll_interval_s
        self._currency = currency
        self._session: Any = None  # aiohttp.ClientSession
        self._last_snapshot: IVSnapshot | None = None

    @property
    def last_snapshot(self) -> IVSnapshot | None:
        return self._last_snapshot

    async def _connect(self) -> None:
        try:
            import aiohttp
        except ImportError as exc:
            msg = "aiohttp is required for DeribitCollector. Install with: uv add aiohttp"
            raise RuntimeError(msg) from exc

        import aiohttp

        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30.0),
            headers={"User-Agent": "ep2-crypto/1.0"},
        )
        self._log.info("deribit_connected")

    async def _disconnect(self) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None
        self._log.info("deribit_disconnected")

    async def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                snapshot = await self._fetch_snapshot()
                if snapshot is not None:
                    self._last_snapshot = snapshot
                    self._record_message(snapshot.timestamp_ms / 1000.0)
                    if self._on_snapshot is not None:
                        await self._on_snapshot(snapshot)
            except Exception as exc:
                self._log.error("deribit_poll_error", error=str(exc))

            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=self._poll_interval_s,
                )
                break
            except TimeoutError:
                pass

    async def _fetch_snapshot(self) -> IVSnapshot | None:
        """Fetch current IV surface from Deribit REST API."""
        if self._session is None:
            return None

        # Step 1: Get all BTC option instruments
        instruments = await self._get_instruments()
        if not instruments:
            return None

        # Step 2: Get BTC spot from index price
        spot = await self._get_spot_price()
        if spot is None or spot <= 0:
            return None

        # Step 3: Fetch tickers for all instruments
        ticker_map = await self._get_tickers(instruments)

        # Step 4: Build IV snapshot
        now_ms = int(time.time() * 1000)

        atm_7d, rr_7d = _parse_iv_surface(instruments, ticker_map, spot, 7, 5)
        atm_14d, _ = _parse_iv_surface(instruments, ticker_map, spot, 14, 5)
        atm_30d, _ = _parse_iv_surface(instruments, ticker_map, spot, 30, 7)

        if atm_7d is None:
            self._log.warning("deribit_no_atm_iv_7d")
            return None

        n_calls = sum(1 for i in instruments if i.get("option_type") == "call")
        n_puts = sum(1 for i in instruments if i.get("option_type") == "put")

        return IVSnapshot(
            timestamp_ms=now_ms,
            atm_iv_7d=atm_7d,
            atm_iv_14d=atm_14d,
            atm_iv_30d=atm_30d,
            rr_25d_7d=rr_7d,
            spot_price=spot,
            n_calls=n_calls,
            n_puts=n_puts,
        )

    async def _get_instruments(self) -> list[dict[str, Any]]:
        url = f"{_DERIBIT_BASE}/public/get_instruments"
        params = {"currency": self._currency, "kind": "option", "expired": "false"}
        async with self._session.get(url, params=params) as resp:
            data = await resp.json()
        return data.get("result", [])

    async def _get_spot_price(self) -> float | None:
        url = f"{_DERIBIT_BASE}/public/get_index_price"
        params = {"index_name": f"{self._currency.lower()}_usd"}
        async with self._session.get(url, params=params) as resp:
            data = await resp.json()
        result = data.get("result", {})
        return result.get("index_price")

    async def _get_tickers(
        self,
        instruments: list[dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        """Fetch ticker data for all instruments (batched via summary endpoint)."""
        url = f"{_DERIBIT_BASE}/public/get_book_summary_by_currency"
        params = {"currency": self._currency, "kind": "option"}
        async with self._session.get(url, params=params) as resp:
            data = await resp.json()
        results = data.get("result", [])
        return {r["instrument_name"]: r for r in results}
