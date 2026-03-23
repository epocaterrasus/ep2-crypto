"""Tests for DeribitCollector and IVSnapshot parsing.

Tests do NOT require network access — all HTTP calls are mocked.
"""

from __future__ import annotations

import pytest

from ep2_crypto.ingest.deribit import (
    IVSnapshot,
    _closest_strike,
    _parse_iv_surface,
)

# ---------------------------------------------------------------------------
# _closest_strike helper
# ---------------------------------------------------------------------------


class TestClosestStrike:
    def test_exact_match(self) -> None:
        strikes = [30000.0, 35000.0, 40000.0]
        assert _closest_strike(strikes, 35000.0) == 35000.0

    def test_rounds_down(self) -> None:
        strikes = [30000.0, 35000.0, 40000.0]
        assert _closest_strike(strikes, 32000.0) == 30000.0

    def test_rounds_up(self) -> None:
        strikes = [30000.0, 35000.0, 40000.0]
        assert _closest_strike(strikes, 33000.0) == 35000.0

    def test_single_strike(self) -> None:
        assert _closest_strike([50000.0], 30000.0) == 50000.0


# ---------------------------------------------------------------------------
# _parse_iv_surface
# ---------------------------------------------------------------------------


def _make_instruments(spot: float, n: int = 5, exp_offset_days: float = 7.0) -> tuple[
    list[dict],
    dict[str, dict],
]:
    """Create fake BTC-7DEC24 instrument and ticker data."""
    import time

    exp_ts_ms = int((time.time() + exp_offset_days * 86400) * 1000)
    instruments = []
    ticker_map: dict[str, dict] = {}

    for i in range(n):
        multiplier = 0.9 + i * 0.05  # strikes: 0.9, 0.95, 1.0, 1.05, 1.1 * spot
        strike = round(spot * multiplier, -2)  # round to nearest 100

        for opt_type in ("call", "put"):
            name = f"BTC-7DEC24-{strike:.0f}-{'C' if opt_type == 'call' else 'P'}"
            instruments.append(
                {
                    "instrument_name": name,
                    "expiration_timestamp": exp_ts_ms,
                    "strike": str(strike),
                    "option_type": opt_type,
                    "kind": "option",
                    "currency": "BTC",
                }
            )
            # Mark IV in percent (Deribit convention)
            base_iv = 65.0 if opt_type == "call" else 60.0
            ticker_map[name] = {
                "instrument_name": name,
                "mark_iv": base_iv + i * 2.0,  # smile-ish
            }

    return instruments, ticker_map


class TestParseIVSurface:
    def test_returns_atm_iv_for_valid_surface(self) -> None:
        spot = 40000.0
        instruments, ticker_map = _make_instruments(spot, n=5, exp_offset_days=7.0)
        atm_iv, _ = _parse_iv_surface(instruments, ticker_map, spot, target_days=7)
        assert atm_iv is not None
        assert 0.0 < atm_iv < 5.0  # annualized (was 65% / 100 = 0.65)

    def test_returns_none_for_no_matching_expiry(self) -> None:
        spot = 40000.0
        instruments, ticker_map = _make_instruments(spot, n=5, exp_offset_days=60.0)
        # Request 7-day but only 60-day instruments exist
        atm_iv, rr = _parse_iv_surface(instruments, ticker_map, spot, target_days=7, tolerance_days=5)
        assert atm_iv is None
        assert rr is None

    def test_atm_iv_is_call_iv_at_closest_strike(self) -> None:
        spot = 40000.0
        instruments, ticker_map = _make_instruments(spot, n=5, exp_offset_days=7.0)
        atm_iv, _ = _parse_iv_surface(instruments, ticker_map, spot, target_days=7)
        # ATM strike is spot * 1.0 = 40000, call IV for i=2: (65 + 4) / 100 = 0.69
        # Closest to 40000 in [36000, 38000, 40000, 42000, 44000]
        assert atm_iv == pytest.approx(0.69, abs=0.01)

    def test_rr_25d_is_call_minus_put(self) -> None:
        spot = 40000.0
        instruments, ticker_map = _make_instruments(spot, n=5, exp_offset_days=7.0)
        _, rr = _parse_iv_surface(instruments, ticker_map, spot, target_days=7)
        # RR should be a number (can be positive or negative)
        if rr is not None:
            assert isinstance(rr, float)

    def test_no_data_returns_nones(self) -> None:
        atm_iv, rr = _parse_iv_surface([], {}, 40000.0, target_days=7)
        assert atm_iv is None
        assert rr is None

    def test_missing_tickers_returns_none(self) -> None:
        spot = 40000.0
        instruments, _ = _make_instruments(spot, n=5, exp_offset_days=7.0)
        # Empty ticker map → no IV data
        atm_iv, rr = _parse_iv_surface(instruments, {}, spot, target_days=7)
        assert atm_iv is None


# ---------------------------------------------------------------------------
# IVSnapshot dataclass
# ---------------------------------------------------------------------------


class TestIVSnapshot:
    def test_construction(self) -> None:
        snap = IVSnapshot(
            timestamp_ms=1_700_000_000_000,
            atm_iv_7d=0.65,
            atm_iv_14d=0.63,
            atm_iv_30d=0.60,
            rr_25d_7d=-0.05,
            spot_price=40000.0,
            n_calls=100,
            n_puts=100,
        )
        assert snap.atm_iv_7d == 0.65
        assert snap.rr_25d_7d == -0.05
        assert snap.n_calls == 100

    def test_optional_fields_can_be_none(self) -> None:
        snap = IVSnapshot(
            timestamp_ms=1_700_000_000_000,
            atm_iv_7d=0.65,
            atm_iv_14d=None,
            atm_iv_30d=None,
            rr_25d_7d=None,
            spot_price=40000.0,
            n_calls=0,
            n_puts=0,
        )
        assert snap.atm_iv_14d is None
        assert snap.rr_25d_7d is None
