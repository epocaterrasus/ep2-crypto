"""Tests for the live prediction loop."""

from __future__ import annotations

import asyncio

# Add scripts to path for import
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

from live import LivePredictionLoop


class TestLivePredictionLoop:
    def test_init_defaults(self) -> None:
        loop = LivePredictionLoop()
        assert not loop.running
        assert loop.bar_count == 0
        assert loop.degraded_sources == set()

    def test_init_collect_only(self) -> None:
        loop = LivePredictionLoop(collect_only=True)
        assert not loop.running

    def test_stop(self) -> None:
        loop = LivePredictionLoop()
        loop._running = True
        loop.stop()
        assert not loop.running

    def test_next_bar_close_time(self) -> None:
        close_time = LivePredictionLoop._next_bar_close_time()
        now = time.time()
        # Should be in the future
        assert close_time > now
        # Should be within 5 minutes
        assert close_time - now <= 300
        # Should be aligned to 5-minute boundary
        assert int(close_time) % 300 == 0


class TestGracefulDegradation:
    def test_mark_source_degraded(self) -> None:
        loop = LivePredictionLoop()
        loop.mark_source_degraded("binance_ws")
        assert "binance_ws" in loop.degraded_sources

    def test_mark_source_recovered(self) -> None:
        loop = LivePredictionLoop()
        loop.mark_source_degraded("binance_ws")
        loop.mark_source_recovered("binance_ws")
        assert "binance_ws" not in loop.degraded_sources

    def test_recover_unknown_source_no_error(self) -> None:
        loop = LivePredictionLoop()
        loop.mark_source_recovered("nonexistent")
        assert loop.degraded_sources == set()

    def test_multiple_degraded_sources(self) -> None:
        loop = LivePredictionLoop()
        loop.mark_source_degraded("binance_ws")
        loop.mark_source_degraded("bybit_oi")
        assert loop.degraded_sources == {"binance_ws", "bybit_oi"}


class TestBarProcessing:
    @pytest.mark.asyncio()
    async def test_on_bar_close_collect_only(self) -> None:
        loop = LivePredictionLoop(collect_only=True)
        loop._running = True
        await loop._on_bar_close()
        assert loop.bar_count == 1
        # No API state update in collect-only mode
        assert loop.api_state == {}

    @pytest.mark.asyncio()
    async def test_on_bar_close_full_mode(self) -> None:
        loop = LivePredictionLoop(collect_only=False)
        loop._running = True
        await loop._on_bar_close()
        assert loop.bar_count == 1
        assert "direction" in loop.api_state
        assert "last_prediction_ms" in loop.api_state

    @pytest.mark.asyncio()
    async def test_multiple_bars(self) -> None:
        loop = LivePredictionLoop(collect_only=False)
        loop._running = True
        for _ in range(5):
            await loop._on_bar_close()
        assert loop.bar_count == 5

    @pytest.mark.asyncio()
    async def test_bar_processing_under_1s(self) -> None:
        loop = LivePredictionLoop(collect_only=False)
        loop._running = True
        start = time.perf_counter()
        await loop._on_bar_close()
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert elapsed_ms < 1000  # < 1 second requirement


class TestFeatureComputation:
    @pytest.mark.asyncio()
    async def test_compute_features_returns_dict(self) -> None:
        loop = LivePredictionLoop()
        features = await loop._compute_features(timestamp_ms=1000)
        assert isinstance(features, dict)

    @pytest.mark.asyncio()
    async def test_compute_features_with_degraded_source(self) -> None:
        loop = LivePredictionLoop()
        loop.mark_source_degraded("binance_ws")
        features = await loop._compute_features(timestamp_ms=1000)
        # Should still return features (graceful degradation)
        assert isinstance(features, dict)


class TestInference:
    @pytest.mark.asyncio()
    async def test_run_inference_returns_prediction(self) -> None:
        loop = LivePredictionLoop()
        prediction = await loop._run_inference({"timestamp_ms": 1000.0})
        assert "direction" in prediction
        assert "confidence" in prediction
        assert "regime" in prediction

    @pytest.mark.asyncio()
    async def test_prediction_direction_valid(self) -> None:
        loop = LivePredictionLoop()
        prediction = await loop._run_inference({})
        assert prediction["direction"] in ("long", "short", "flat")


class TestAPIState:
    def test_update_api_state(self) -> None:
        loop = LivePredictionLoop()
        prediction = {
            "direction": "long",
            "magnitude": 0.002,
            "confidence": 0.72,
            "regime": "trending",
        }
        loop._update_api_state(prediction, timestamp_ms=5000)
        state = loop.api_state
        assert state["direction"] == "long"
        assert state["magnitude"] == 0.002
        assert state["confidence"] == 0.72
        assert state["last_prediction_ms"] == 5000


class TestRetrainCheck:
    @pytest.mark.asyncio()
    async def test_retrain_not_due(self) -> None:
        loop = LivePredictionLoop(retrain_interval_s=99999)
        loop._last_retrain_time = time.monotonic()
        await loop._check_retrain()
        # No exception = success

    @pytest.mark.asyncio()
    async def test_retrain_due_resets_timer(self) -> None:
        loop = LivePredictionLoop(retrain_interval_s=0)
        old_time = loop._last_retrain_time
        await loop._check_retrain()
        assert loop._last_retrain_time > old_time


class TestShutdown:
    @pytest.mark.asyncio()
    async def test_shutdown(self) -> None:
        loop = LivePredictionLoop()
        loop._running = True
        loop._bar_count = 42
        await loop._shutdown()
        assert not loop.running

    @pytest.mark.asyncio()
    async def test_start_and_stop(self) -> None:
        loop = LivePredictionLoop()

        async def stop_after_delay() -> None:
            await asyncio.sleep(0.1)
            loop.stop()

        # Run loop briefly then stop
        await asyncio.gather(
            loop.start(),
            stop_after_delay(),
        )
        assert not loop.running
