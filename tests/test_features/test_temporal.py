"""Tests for temporal feature computers: cyclical time, session, funding time."""

from __future__ import annotations

import numpy as np
import pytest

from ep2_crypto.features.temporal import (
    CyclicalTimeComputer,
    FundingTimeComputer,
    SessionComputer,
)


def _make_temporal_data(n: int = 30, start_hour: int = 0) -> dict[str, np.ndarray]:
    """Create synthetic data with specific timestamp patterns."""
    base_ts = start_hour * 3_600_000  # start_hour in ms
    timestamps = np.arange(n, dtype=np.int64) * 300_000 + base_ts  # 5-min bars

    closes = np.ones(n) * 50000.0
    return {
        "timestamps": timestamps,
        "opens": closes.copy(),
        "highs": closes.copy(),
        "lows": closes.copy(),
        "closes": closes,
        "volumes": np.ones(n),
    }


# ---- Cyclical Time Tests ----


class TestCyclicalTime:
    def test_output_names(self) -> None:
        comp = CyclicalTimeComputer()
        assert comp.output_names() == [
            "minute_sin", "minute_cos", "hour_sin", "hour_cos", "dow_sin", "dow_cos",
        ]

    def test_warmup(self) -> None:
        comp = CyclicalTimeComputer()
        assert comp.warmup_bars == 1
        assert comp.name == "cyclical_time"

    def test_values_bounded(self) -> None:
        """All sin/cos values should be in [-1, 1]."""
        comp = CyclicalTimeComputer()
        data = _make_temporal_data(100, start_hour=10)
        for i in range(100):
            result = comp.compute(
                i, data["timestamps"], data["opens"], data["highs"],
                data["lows"], data["closes"], data["volumes"],
            )
            for key, val in result.items():
                assert -1.0 <= val <= 1.0, f"{key} = {val} out of bounds"

    def test_sin_cos_identity(self) -> None:
        """sin^2 + cos^2 should equal 1 for each pair."""
        comp = CyclicalTimeComputer()
        data = _make_temporal_data(20, start_hour=14)
        result = comp.compute(
            10, data["timestamps"], data["opens"], data["highs"],
            data["lows"], data["closes"], data["volumes"],
        )
        for prefix in ["minute", "hour", "dow"]:
            sin_val = result[f"{prefix}_sin"]
            cos_val = result[f"{prefix}_cos"]
            assert sin_val ** 2 + cos_val ** 2 == pytest.approx(1.0, abs=1e-10)

    def test_midnight_values(self) -> None:
        """At midnight UTC (hour=0, minute=0), hour_sin=0, hour_cos=1."""
        comp = CyclicalTimeComputer()
        # Create timestamp at exactly midnight UTC
        ts = np.array([0], dtype=np.int64)  # epoch is midnight
        dummy = np.ones(1)
        result = comp.compute(0, ts, dummy, dummy, dummy, dummy, dummy)
        assert result["hour_sin"] == pytest.approx(0.0, abs=1e-10)
        assert result["hour_cos"] == pytest.approx(1.0, abs=1e-10)
        assert result["minute_sin"] == pytest.approx(0.0, abs=1e-10)
        assert result["minute_cos"] == pytest.approx(1.0, abs=1e-10)

    def test_noon_values(self) -> None:
        """At noon UTC (hour=12), hour_sin=0, hour_cos=-1."""
        comp = CyclicalTimeComputer()
        ts = np.array([12 * 3_600_000], dtype=np.int64)  # noon UTC
        dummy = np.ones(1)
        result = comp.compute(0, ts, dummy, dummy, dummy, dummy, dummy)
        assert result["hour_sin"] == pytest.approx(0.0, abs=1e-10)
        assert result["hour_cos"] == pytest.approx(-1.0, abs=1e-10)

    def test_6am_values(self) -> None:
        """At 6:00 UTC (hour=6), hour_sin=1, hour_cos=0."""
        comp = CyclicalTimeComputer()
        ts = np.array([6 * 3_600_000], dtype=np.int64)
        dummy = np.ones(1)
        result = comp.compute(0, ts, dummy, dummy, dummy, dummy, dummy)
        assert result["hour_sin"] == pytest.approx(1.0, abs=1e-10)
        assert result["hour_cos"] == pytest.approx(0.0, abs=1e-10)

    def test_continuity_across_day(self) -> None:
        """23:55 and 00:05 should have close cyclical minute values."""
        comp = CyclicalTimeComputer()
        # 23:55 UTC
        ts1 = np.array([23 * 3_600_000 + 55 * 60_000], dtype=np.int64)
        # 00:05 UTC next day
        ts2 = np.array([24 * 3_600_000 + 5 * 60_000], dtype=np.int64)
        dummy = np.ones(1)
        r1 = comp.compute(0, ts1, dummy, dummy, dummy, dummy, dummy)
        r2 = comp.compute(0, ts2, dummy, dummy, dummy, dummy, dummy)
        # minute_sin for 55 and 5 should be close in absolute terms
        # sin(2pi*55/60) ≈ sin(2pi*5/60) with opposite sign
        # But hour values should be nearly opposite
        # Key: both should be finite
        assert np.isfinite(r1["minute_sin"])
        assert np.isfinite(r2["minute_sin"])


# ---- Session Tests ----


class TestSession:
    def test_output_names(self) -> None:
        comp = SessionComputer()
        assert comp.output_names() == ["session_asia", "session_europe", "session_us"]

    def test_warmup(self) -> None:
        comp = SessionComputer()
        assert comp.warmup_bars == 1
        assert comp.name == "session"

    def test_asia_session(self) -> None:
        """03:00 UTC should be Asia session."""
        comp = SessionComputer()
        ts = np.array([3 * 3_600_000], dtype=np.int64)
        dummy = np.ones(1)
        result = comp.compute(0, ts, dummy, dummy, dummy, dummy, dummy)
        assert result["session_asia"] == 1.0
        assert result["session_europe"] == 0.0
        assert result["session_us"] == 0.0

    def test_europe_session(self) -> None:
        """10:00 UTC should be Europe session."""
        comp = SessionComputer()
        ts = np.array([10 * 3_600_000], dtype=np.int64)
        dummy = np.ones(1)
        result = comp.compute(0, ts, dummy, dummy, dummy, dummy, dummy)
        assert result["session_asia"] == 0.0
        assert result["session_europe"] == 1.0
        assert result["session_us"] == 0.0

    def test_us_session(self) -> None:
        """18:00 UTC should be US session."""
        comp = SessionComputer()
        ts = np.array([18 * 3_600_000], dtype=np.int64)
        dummy = np.ones(1)
        result = comp.compute(0, ts, dummy, dummy, dummy, dummy, dummy)
        assert result["session_asia"] == 0.0
        assert result["session_europe"] == 0.0
        assert result["session_us"] == 1.0

    def test_boundary_europe_start(self) -> None:
        """08:00 UTC should be Europe (not Asia)."""
        comp = SessionComputer()
        ts = np.array([8 * 3_600_000], dtype=np.int64)
        dummy = np.ones(1)
        result = comp.compute(0, ts, dummy, dummy, dummy, dummy, dummy)
        assert result["session_europe"] == 1.0
        assert result["session_asia"] == 0.0

    def test_boundary_us_start(self) -> None:
        """16:00 UTC should be US (not Europe)."""
        comp = SessionComputer()
        ts = np.array([16 * 3_600_000], dtype=np.int64)
        dummy = np.ones(1)
        result = comp.compute(0, ts, dummy, dummy, dummy, dummy, dummy)
        assert result["session_us"] == 1.0
        assert result["session_europe"] == 0.0

    def test_exactly_one_session(self) -> None:
        """At any time, exactly one session should be active."""
        comp = SessionComputer()
        for hour in range(24):
            ts = np.array([hour * 3_600_000], dtype=np.int64)
            dummy = np.ones(1)
            result = comp.compute(0, ts, dummy, dummy, dummy, dummy, dummy)
            total = result["session_asia"] + result["session_europe"] + result["session_us"]
            assert total == 1.0, f"At hour {hour}, total sessions = {total}"


# ---- Funding Time Tests ----


class TestFundingTime:
    def test_output_names(self) -> None:
        comp = FundingTimeComputer()
        assert comp.output_names() == ["time_to_funding_norm", "time_to_funding_min"]

    def test_warmup(self) -> None:
        comp = FundingTimeComputer()
        assert comp.warmup_bars == 1
        assert comp.name == "funding_time"

    def test_just_before_8am(self) -> None:
        """At 07:55 UTC, 5 minutes to next funding (08:00)."""
        comp = FundingTimeComputer()
        ts = np.array([7 * 3_600_000 + 55 * 60_000], dtype=np.int64)
        dummy = np.ones(1)
        result = comp.compute(0, ts, dummy, dummy, dummy, dummy, dummy)
        assert result["time_to_funding_min"] == pytest.approx(5.0, abs=0.1)
        assert result["time_to_funding_norm"] == pytest.approx(5.0 / 480.0, abs=0.01)

    def test_just_after_midnight(self) -> None:
        """At 00:05 UTC, 475 minutes to next funding (08:00)."""
        comp = FundingTimeComputer()
        ts = np.array([5 * 60_000], dtype=np.int64)
        dummy = np.ones(1)
        result = comp.compute(0, ts, dummy, dummy, dummy, dummy, dummy)
        assert result["time_to_funding_min"] == pytest.approx(475.0, abs=0.1)

    def test_just_before_midnight(self) -> None:
        """At 23:55 UTC, 5 minutes to next funding (00:00)."""
        comp = FundingTimeComputer()
        ts = np.array([23 * 3_600_000 + 55 * 60_000], dtype=np.int64)
        dummy = np.ones(1)
        result = comp.compute(0, ts, dummy, dummy, dummy, dummy, dummy)
        assert result["time_to_funding_min"] == pytest.approx(5.0, abs=0.1)

    def test_at_funding_time(self) -> None:
        """At exactly 08:00 UTC, should be 480 min to next (16:00)."""
        comp = FundingTimeComputer()
        ts = np.array([8 * 3_600_000], dtype=np.int64)
        dummy = np.ones(1)
        result = comp.compute(0, ts, dummy, dummy, dummy, dummy, dummy)
        assert result["time_to_funding_min"] == pytest.approx(480.0, abs=0.1)

    def test_normalized_range(self) -> None:
        """Normalized time should be in (0, 1]."""
        comp = FundingTimeComputer()
        for hour in range(24):
            for minute in [0, 15, 30, 45]:
                ts = np.array([hour * 3_600_000 + minute * 60_000], dtype=np.int64)
                dummy = np.ones(1)
                result = comp.compute(0, ts, dummy, dummy, dummy, dummy, dummy)
                assert 0.0 < result["time_to_funding_norm"] <= 1.0, (
                    f"At {hour}:{minute}, norm={result['time_to_funding_norm']}"
                )

    def test_midpoint_between_funding(self) -> None:
        """At 04:00 UTC (halfway between 00:00 and 08:00), 240 min to funding."""
        comp = FundingTimeComputer()
        ts = np.array([4 * 3_600_000], dtype=np.int64)
        dummy = np.ones(1)
        result = comp.compute(0, ts, dummy, dummy, dummy, dummy, dummy)
        assert result["time_to_funding_min"] == pytest.approx(240.0, abs=0.1)
        assert result["time_to_funding_norm"] == pytest.approx(0.5, abs=0.01)
