"""FastAPI prediction server.

Endpoints:
  GET /predict  — Current prediction (direction, magnitude, confidence, regime)
  GET /health   — Checks ALL dependencies (not just "ok")
  GET /metrics  — Live accuracy, rolling Sharpe, trade count
  GET /regime   — Current regime + transition context

All responses include timestamps and staleness warnings.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import structlog
from fastapi import FastAPI
from pydantic import BaseModel

logger = structlog.get_logger(__name__)

STALENESS_WARN_S = 600  # 10 minutes


# ── Response Models ──────────────────────────────────────────────────────────

class PredictionResponse(BaseModel):
    """Response for GET /predict."""

    direction: str  # "long", "short", "flat"
    magnitude: float
    confidence: float
    regime: str
    timestamp_ms: int
    staleness_s: float
    stale_warning: bool
    model_version: str


class HealthCheck(BaseModel):
    """Individual dependency health."""

    name: str
    status: str  # "ok", "degraded", "down"
    latency_ms: float | None = None
    message: str | None = None


class HealthResponse(BaseModel):
    """Response for GET /health."""

    status: str  # "healthy", "degraded", "unhealthy"
    timestamp_ms: int
    checks: list[HealthCheck]
    uptime_s: float


class MetricsResponse(BaseModel):
    """Response for GET /metrics."""

    accuracy: float | None
    rolling_sharpe: float | None
    trade_count: int
    completed_trades: int
    cumulative_pnl: float
    win_rate: float | None
    alpha_decay_level: str
    timestamp_ms: int


class RegimeResponse(BaseModel):
    """Response for GET /regime."""

    current_regime: str
    regime_confidence: float
    regime_duration_bars: int
    timestamp_ms: int
    staleness_s: float
    stale_warning: bool


# ── Application State ────────────────────────────────────────────────────────

@dataclass
class AppState:
    """Mutable application state shared across the API.

    Set by the live prediction loop; read by API endpoints.
    """

    # Prediction state
    direction: str = "flat"
    magnitude: float = 0.0
    confidence: float = 0.0
    regime: str = "unknown"
    regime_confidence: float = 0.0
    regime_duration_bars: int = 0
    model_version: str = "none"
    last_prediction_ms: int = 0

    # Metrics state
    accuracy: float | None = None
    rolling_sharpe: float | None = None
    trade_count: int = 0
    completed_trades: int = 0
    cumulative_pnl: float = 0.0
    win_rate: float | None = None
    alpha_decay_level: str = "NORMAL"

    # Health state
    dependency_checks: list[dict[str, Any]] = field(default_factory=list)

    # Server metadata
    start_time: float = field(default_factory=time.time)


def _now_ms() -> int:
    return int(time.time() * 1000)


def _staleness(last_ms: int) -> tuple[float, bool]:
    if last_ms == 0:
        return 0.0, True
    staleness_s = (time.time() * 1000 - last_ms) / 1000
    return staleness_s, staleness_s > STALENESS_WARN_S


# ── App Factory ──────────────────────────────────────────────────────────────

def create_app(state: AppState | None = None) -> FastAPI:
    """Create the FastAPI application.

    Args:
        state: Shared mutable state. If None, creates a default.
    """
    app_state = state or AppState()

    app = FastAPI(
        title="ep2-crypto Prediction API",
        version="0.1.0",
    )

    # Store state on app for access in endpoints
    app.state.ep2 = app_state  # type: ignore[attr-defined]

    @app.get("/predict", response_model=PredictionResponse)
    async def predict() -> PredictionResponse:
        s: AppState = app.state.ep2  # type: ignore[attr-defined]
        staleness_s, stale = _staleness(s.last_prediction_ms)
        return PredictionResponse(
            direction=s.direction,
            magnitude=s.magnitude,
            confidence=s.confidence,
            regime=s.regime,
            timestamp_ms=s.last_prediction_ms,
            staleness_s=round(staleness_s, 1),
            stale_warning=stale,
            model_version=s.model_version,
        )

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        s: AppState = app.state.ep2  # type: ignore[attr-defined]
        checks = []
        for dep in s.dependency_checks:
            checks.append(HealthCheck(
                name=dep.get("name", "unknown"),
                status=dep.get("status", "down"),
                latency_ms=dep.get("latency_ms"),
                message=dep.get("message"),
            ))

        # If no dependency checks configured, report degraded
        if not checks:
            overall = "degraded"
        elif all(c.status == "ok" for c in checks):
            overall = "healthy"
        elif any(c.status == "down" for c in checks):
            overall = "unhealthy"
        else:
            overall = "degraded"

        uptime_s = time.time() - s.start_time

        return HealthResponse(
            status=overall,
            timestamp_ms=_now_ms(),
            checks=checks,
            uptime_s=round(uptime_s, 1),
        )

    @app.get("/metrics", response_model=MetricsResponse)
    async def metrics() -> MetricsResponse:
        s: AppState = app.state.ep2  # type: ignore[attr-defined]
        return MetricsResponse(
            accuracy=s.accuracy,
            rolling_sharpe=s.rolling_sharpe,
            trade_count=s.trade_count,
            completed_trades=s.completed_trades,
            cumulative_pnl=s.cumulative_pnl,
            win_rate=s.win_rate,
            alpha_decay_level=s.alpha_decay_level,
            timestamp_ms=_now_ms(),
        )

    @app.get("/regime", response_model=RegimeResponse)
    async def regime() -> RegimeResponse:
        s: AppState = app.state.ep2  # type: ignore[attr-defined]
        staleness_s, stale = _staleness(s.last_prediction_ms)
        return RegimeResponse(
            current_regime=s.regime,
            regime_confidence=s.regime_confidence,
            regime_duration_bars=s.regime_duration_bars,
            timestamp_ms=s.last_prediction_ms,
            staleness_s=round(staleness_s, 1),
            stale_warning=stale,
        )

    return app


# Default app instance for uvicorn
app = create_app()
