# Development Progress

> This file is the single source of truth for what's done and what's next.
> Updated at the end of every session. Read this FIRST in every new session.

## Current Sprint: Sprint 2 — Data Ingestion
**Started**: Not yet
**Target**: WebSocket and REST collectors for Binance and Bybit exchange data

---

## Sprint 2 Tickets

### S2-T1: Base collector interface + orchestrator skeleton [x]
**Create**: Abstract base collector (async context manager) + orchestrator shell
**Files**:
- `src/ep2_crypto/ingest/base.py` — BaseCollector ABC with start/stop/health_check
- `src/ep2_crypto/ingest/orchestrator.py` — Task manager skeleton (add_collector, run, shutdown)
- `tests/test_ingest/test_orchestrator.py` — Lifecycle tests
**Verify**: `uv run pytest tests/test_ingest/test_orchestrator.py -v`
**Research**: `RR-production-realtime-python-architecture.md` (async patterns)
**Notes**: Async context manager pattern. Exponential backoff reconnection. SIGTERM/SIGINT graceful shutdown.

### S2-T2: Binance WebSocket kline collector [x]
**Create**: Binance 1m kline collector via ccxt pro
**Files**:
- `src/ep2_crypto/ingest/exchange.py` — BinanceKlineCollector
- `tests/test_ingest/test_exchange.py` — Mock WS tests for klines
**Verify**: `uv run pytest tests/test_ingest/test_exchange.py -v -k kline`
**Research**: `RR-api-ccxt-exchange-integration.md`
**Notes**: ccxt pro watch_ohlcv. Store to ohlcv table via repository. Handle reconnection.

### S2-T3: Binance WebSocket depth + aggTrades collector [x]
**Create**: Order book depth@100ms (top 20) and aggregated trades collectors
**Files**:
- `src/ep2_crypto/ingest/exchange.py` — BinanceDepthCollector, BinanceTradeCollector
- `tests/test_ingest/test_exchange.py` — Mock WS tests for depth + trades
**Verify**: `uv run pytest tests/test_ingest/test_exchange.py -v`
**Research**: `RR-api-ccxt-exchange-integration.md`
**Notes**: ccxt pro watch_order_book (limit=20), watch_trades. JSON-serialize bid/ask arrays for orderbook_snapshot.

### S2-T4: Bybit derivatives (OI, funding rate via REST) [x]
**Create**: Bybit OI + funding rate polling collectors
**Files**:
- `src/ep2_crypto/ingest/derivatives.py` — BybitOICollector, BybitFundingCollector
- `tests/test_ingest/test_derivatives.py` — Mock REST tests
**Verify**: `uv run pytest tests/test_ingest/test_derivatives.py -v -k "oi or funding"`
**Research**: `RR-api-ccxt-exchange-integration.md`
**Notes**: REST poll every 5 minutes. Store to open_interest and funding_rate tables.

### S2-T5: Bybit liquidation WebSocket stream [x]
**Create**: Bybit liquidation stream via raw websockets
**Files**:
- `src/ep2_crypto/ingest/derivatives.py` — BybitLiquidationCollector
- `tests/test_ingest/test_derivatives.py` — Mock WS tests for liquidations
**Verify**: `uv run pytest tests/test_ingest/test_derivatives.py -v -k liquidation`
**Research**: `RR-api-ccxt-exchange-integration.md`, `RR-cascade-liquidation-detection-system.md`
**Notes**: Raw WS to `wss://stream.bybit.com/v5/public/linear`, topic `allLiquidation.BTCUSDT`. Heartbeat every 20s.

### S2-T6: Reconnection + error handling + health checks [x]
**Create**: Robust reconnection, health check endpoints, graceful shutdown
**Files**:
- Update `src/ep2_crypto/ingest/orchestrator.py` — reconnection logic, health aggregation
- Update `src/ep2_crypto/ingest/base.py` — health_check method
- `tests/test_ingest/test_orchestrator.py` — Reconnection + shutdown tests
**Verify**: `uv run pytest tests/test_ingest/ -v`
**Research**: `RR-production-realtime-python-architecture.md`
**Notes**: Exponential backoff (1s -> 60s max). No duplicate records (upsert). SIGTERM/SIGINT handling.

### S2-T7: Sprint 2 integration test [x]
**Create**: End-to-end test with mock exchange running for simulated period
**Files**:
- `tests/test_integration/test_ingestion.py`
**Verify**: `uv run pytest tests/ -v --tb=short`
**Research**: N/A
**Notes**: Mock WS server, verify data in SQLite, verify no duplicates, verify health checks.

---

## Session Log

### Session 1 (2026-03-23)
- **What happened**: Research phase (40 agents, 3 rounds). Created PLAN.md, RESEARCH_SYNTHESIS.md, BACKTESTING_PLAN.md.
- **Infrastructure**: Created CLAUDE.md, SPRINTS.md, REQUIREMENTS.md, DECISIONS.md, SESSION_PROTOCOL.md, pyproject.toml.
- **Research files**: Renamed all 48 to RR-{category}-{topic}.md format. Created research/INDEX.md.
- **Memory**: Set up project memory with context + key decisions.
- **Next session**: Start S1-T1 (project structure and package init)

### Session 2 (2026-03-23)
- **What happened**: Completed ALL 6 Sprint 1 tickets. 100 tests passing.
- **S1-T1**: Created directory tree + `__init__.py` for 9 subpackages + 7 test subdirs
- **S1-T2**: Config module with 5 Pydantic Settings classes (Exchange, DB, Pipeline, Monitoring, API, App). 15 tests.
- **S1-T3**: structlog JSON logging with callsite info, level filtering, third-party silencing. 5 tests.
- **S1-T4**: SQLite schema with 11 tables, 9 indexes, WAL mode + PRAGMA tuning. 9 tests.
- **S1-T5**: Repository with parameterized CRUD for all 11 tables, batch inserts, upsert. 19 tests.
- **S1-T6**: Integration test verifying full lifecycle (config -> logging -> schema -> repository). 2 tests.
- **Sprint 1 acceptance criteria**: All passed. Committed.
- **Next session**: Start S2-T1 (base collector interface + orchestrator skeleton)

### Session 3 (2026-03-23)
- **What happened**: Completed ALL 7 Sprint 2 tickets. 151 tests passing (51 new).
- **S2-T1**: BaseCollector ABC with async context manager, exponential backoff reconnection, health status. Orchestrator with add/remove/run/shutdown, SIGTERM/SIGINT, health aggregation. 20 tests.
- **S2-T2**: BinanceKlineCollector via ccxt pro watch_ohlcv, stores to ohlcv table with dedup. 7 tests.
- **S2-T3**: BinanceDepthCollector (watch_order_book, JSON-serialized bid/ask arrays) + BinanceTradeCollector (watch_trades, batch insert with dedup). 7 tests.
- **S2-T4**: BybitOICollector + BybitFundingCollector (REST polling with configurable interval, run_in_executor for sync ccxt). 8 tests.
- **S2-T5**: BybitLiquidationCollector (raw WebSocket to Bybit v5, subscribe/ping/parse). 5 tests.
- **S2-T6**: Already covered in T1 — reconnection, health checks, error handling baked into BaseCollector.
- **S2-T7**: Integration test with 6 collectors running via orchestrator, verifying data in DB, health checks, clean shutdown, no duplicates. 4 tests.
- **Sprint 2 acceptance criteria**: All passed. Committed.
- **Next session**: Start Sprint 3 (Feature Engineering)

---

## Sprint Completion Protocol

When ALL tickets in a sprint are `[x]`:

1. Run the sprint's acceptance criteria from SPRINTS.md
2. Commit all code: `git add -A && git commit -m "Sprint N complete: <summary>"`
3. Update this section with the completion date
4. Decompose the NEXT sprint into tickets (copy from SPRINTS.md, add verification commands)
5. Generate the **handover prompt** for the user:

```
SPRINT N COMPLETE. Handover prompt for next session:

---
Read PROGRESS.md and start Sprint {N+1}. The current ticket is S{N+1}-T1.
PROGRESS.md has the full ticket breakdown. Run the verification command
after each ticket. Update PROGRESS.md before ending the session.
---
```

The user pastes this prompt to start the next session.

## Sprint Completion Log

| Sprint | Status | Completed |
|--------|--------|-----------|
| Sprint 1: Foundation | Complete | 2026-03-23 |
| Sprint 2: Data Ingestion | Complete | 2026-03-23 |
| Sprint 3-14 | Not started | — |

---

## Future Sprint Ticket Decomposition

Sprints 3-14 will be decomposed into tickets when they become the current sprint.
See SPRINTS.md for high-level sprint definitions.
