"""Tests for FastAPI prediction server."""

from __future__ import annotations

import time

import pytest
from httpx import ASGITransport, AsyncClient

from ep2_crypto.api.server import AppState, create_app


@pytest.fixture()
def state() -> AppState:
    return AppState(
        direction="long",
        magnitude=0.002,
        confidence=0.72,
        regime="trending",
        regime_confidence=0.85,
        regime_duration_bars=42,
        model_version="v1.2.0",
        last_prediction_ms=int(time.time() * 1000),
        accuracy=0.54,
        rolling_sharpe=1.8,
        trade_count=150,
        completed_trades=145,
        cumulative_pnl=1250.0,
        win_rate=0.54,
        alpha_decay_level="NORMAL",
        dependency_checks=[
            {"name": "binance_ws", "status": "ok", "latency_ms": 12.5},
            {"name": "model_loaded", "status": "ok", "latency_ms": 0.1},
            {"name": "database", "status": "ok", "latency_ms": 2.0},
        ],
    )


@pytest.fixture()
def app(state: AppState):  # type: ignore[no-untyped-def]
    return create_app(state)


@pytest.fixture()
async def client(app):  # type: ignore[no-untyped-def]
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


class TestPredict:
    @pytest.mark.asyncio()
    async def test_predict_returns_prediction(self, client: AsyncClient) -> None:
        resp = await client.get("/predict")
        assert resp.status_code == 200
        data = resp.json()
        assert data["direction"] == "long"
        assert data["magnitude"] == 0.002
        assert data["confidence"] == 0.72
        assert data["regime"] == "trending"
        assert data["model_version"] == "v1.2.0"

    @pytest.mark.asyncio()
    async def test_predict_includes_timestamp(self, client: AsyncClient) -> None:
        resp = await client.get("/predict")
        data = resp.json()
        assert "timestamp_ms" in data
        assert data["timestamp_ms"] > 0

    @pytest.mark.asyncio()
    async def test_predict_staleness_fresh(self, client: AsyncClient) -> None:
        resp = await client.get("/predict")
        data = resp.json()
        assert data["staleness_s"] < 5
        assert not data["stale_warning"]

    @pytest.mark.asyncio()
    async def test_predict_staleness_stale(self) -> None:
        stale_state = AppState(
            last_prediction_ms=int((time.time() - 700) * 1000),  # 700s ago
        )
        app = create_app(stale_state)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get("/predict")
        data = resp.json()
        assert data["stale_warning"]
        assert data["staleness_s"] > 600

    @pytest.mark.asyncio()
    async def test_predict_response_time(self, client: AsyncClient) -> None:
        start = time.perf_counter()
        await client.get("/predict")
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert elapsed_ms < 100  # < 100ms requirement


class TestHealth:
    @pytest.mark.asyncio()
    async def test_health_all_ok(self, client: AsyncClient) -> None:
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert len(data["checks"]) == 3

    @pytest.mark.asyncio()
    async def test_health_degraded(self) -> None:
        state = AppState(dependency_checks=[
            {"name": "binance_ws", "status": "ok"},
            {"name": "model_loaded", "status": "degraded", "message": "stale model"},
        ])
        app = create_app(state)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get("/health")
        data = resp.json()
        assert data["status"] == "degraded"

    @pytest.mark.asyncio()
    async def test_health_unhealthy(self) -> None:
        state = AppState(dependency_checks=[
            {"name": "binance_ws", "status": "down", "message": "connection lost"},
            {"name": "model_loaded", "status": "ok"},
        ])
        app = create_app(state)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get("/health")
        data = resp.json()
        assert data["status"] == "unhealthy"

    @pytest.mark.asyncio()
    async def test_health_no_checks_degraded(self) -> None:
        state = AppState(dependency_checks=[])
        app = create_app(state)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get("/health")
        data = resp.json()
        assert data["status"] == "degraded"

    @pytest.mark.asyncio()
    async def test_health_includes_uptime(self, client: AsyncClient) -> None:
        resp = await client.get("/health")
        data = resp.json()
        assert data["uptime_s"] >= 0

    @pytest.mark.asyncio()
    async def test_health_checks_all_deps(self, client: AsyncClient) -> None:
        resp = await client.get("/health")
        data = resp.json()
        names = {c["name"] for c in data["checks"]}
        assert "binance_ws" in names
        assert "model_loaded" in names
        assert "database" in names


class TestMetrics:
    @pytest.mark.asyncio()
    async def test_metrics_returns_data(self, client: AsyncClient) -> None:
        resp = await client.get("/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert data["accuracy"] == 0.54
        assert data["rolling_sharpe"] == 1.8
        assert data["trade_count"] == 150
        assert data["completed_trades"] == 145
        assert data["cumulative_pnl"] == 1250.0
        assert data["win_rate"] == 0.54
        assert data["alpha_decay_level"] == "NORMAL"

    @pytest.mark.asyncio()
    async def test_metrics_includes_timestamp(self, client: AsyncClient) -> None:
        resp = await client.get("/metrics")
        data = resp.json()
        assert data["timestamp_ms"] > 0

    @pytest.mark.asyncio()
    async def test_metrics_empty_state(self) -> None:
        state = AppState()
        app = create_app(state)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get("/metrics")
        data = resp.json()
        assert data["accuracy"] is None
        assert data["trade_count"] == 0


class TestRegime:
    @pytest.mark.asyncio()
    async def test_regime_returns_data(self, client: AsyncClient) -> None:
        resp = await client.get("/regime")
        assert resp.status_code == 200
        data = resp.json()
        assert data["current_regime"] == "trending"
        assert data["regime_confidence"] == 0.85
        assert data["regime_duration_bars"] == 42

    @pytest.mark.asyncio()
    async def test_regime_staleness(self, client: AsyncClient) -> None:
        resp = await client.get("/regime")
        data = resp.json()
        assert data["staleness_s"] < 5
        assert not data["stale_warning"]


class TestResponseTime:
    @pytest.mark.asyncio()
    async def test_all_endpoints_under_100ms(self, client: AsyncClient) -> None:
        for endpoint in ["/predict", "/health", "/metrics", "/regime"]:
            start = time.perf_counter()
            resp = await client.get(endpoint)
            elapsed_ms = (time.perf_counter() - start) * 1000
            assert resp.status_code == 200
            assert elapsed_ms < 100, f"{endpoint} took {elapsed_ms:.1f}ms"
