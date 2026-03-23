# Development Progress

> This file is the single source of truth for what's done and what's next.
> Updated at the end of every session. Read this FIRST in every new session.

## Current Sprint: Sprint 6 — Regime Detection
**Started**: Not yet
**Target**: Hierarchical regime detection ensemble (ER, GARCH, HMM, BOCPD)

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

## Sprint 3 Tickets

### S3-T1: Feature base interface and registry [x]
**Create**: Feature computation interface + registry pattern
**Files**:
- `src/ep2_crypto/features/base.py` — FeatureComputer ABC with compute(), warmup_bars, name
- `tests/test_features/test_base.py` — Interface tests
**Verify**: `uv run pytest tests/test_features/test_base.py -v`
**Research**: N/A
**Notes**: Each feature returns a dict[str, float]. Registry allows selecting features by name.

### S3-T2: Order Book Imbalance (OBI) [x]
**Create**: Weighted OBI at multiple levels
**Files**:
- `src/ep2_crypto/features/microstructure.py` — OBI computation (levels 1-3, 1-5)
- `tests/test_features/test_microstructure.py` — Golden dataset tests for OBI
**Verify**: `uv run pytest tests/test_features/test_microstructure.py -v -k obi`
**Research**: `RR-ofi-microprice-implementation.md`
**Notes**: OBI = (bid_vol - ask_vol) / (bid_vol + ask_vol). Weighted by inverse distance to mid.

### S3-T3: Order Flow Imbalance (OFI) + Microprice [x]
**Create**: Multi-level OFI (Cont-Stoikov-Talreja) and Gatheral-Stoikov microprice
**Files**:
- `src/ep2_crypto/features/microstructure.py` — OFI and microprice
- `tests/test_features/test_microstructure.py` — Golden dataset tests
**Verify**: `uv run pytest tests/test_features/test_microstructure.py -v -k "ofi or microprice"`
**Research**: `RR-ofi-microprice-implementation.md`
**Notes**: OFI tracks changes in top-of-book quantities. Microprice = weighted mid.

### S3-T4: Trade Flow Imbalance (TFI) + spread + absorption [x]
**Create**: TFI at 30s and 5min windows, relative spread, absorption detection
**Files**:
- `src/ep2_crypto/features/microstructure.py` — TFI, spread, absorption, Kyle's lambda
- `tests/test_features/test_microstructure.py` — Tests for TFI and spread
**Verify**: `uv run pytest tests/test_features/test_microstructure.py -v`
**Research**: `RR-ofi-microprice-implementation.md`
**Notes**: TFI = (buy_vol - sell_vol) / total_vol over window. Absorption = high volume delta with low price change.

### S3-T5: Look-ahead bias tests [x]
**Create**: Shuffle test + truncation test for all microstructure features
**Files**:
- `tests/test_features/test_lookahead.py` — Bias detection tests
**Verify**: `uv run pytest tests/test_features/test_lookahead.py -v`
**Research**: N/A
**Notes**: Truncation: compute(data[:150]) == compute(data[:200]) at index 150. Shuffle: permuted input should not correlate with target.

### S3-T6: Sprint 3 integration test [x]
**Create**: End-to-end feature computation from mock market data
**Files**:
- `tests/test_features/test_feature_integration.py`
**Verify**: `uv run pytest tests/test_features/ -v`
**Research**: N/A
**Notes**: Verify feature shape, no NaN/Inf after warmup, compute time < 1ms/bar.

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

### Session 4 (2026-03-23)
- **What happened**: Completed ALL 6 Sprint 3 tickets. 57 new tests (208 total).
- **S3-T1**: FeatureComputer ABC (compute/warmup_bars/name) + FeatureRegistry with register/get/select/compute_all. 15 tests.
- **S3-T2**: OBIComputer — weighted OBI at levels 1-3 and 1-5, inverse-distance weighting. 7 tests with golden dataset.
- **S3-T3**: OFIComputer (Cont-Stoikov-Talreja 6 cases, multi-level) + MicropriceComputer (Gatheral-Stoikov weighted mid). 14 tests.
- **S3-T4**: TFIComputer (1-bar/6-bar windows), relative spread, absorption detection, KyleLambdaComputer (rolling Cov/Var). 11 tests.
- **S3-T5**: Truncation test (no look-ahead across 4 indices) + shuffle test (point-in-time vs windowed) + no-future-correlation test. 3 tests.
- **S3-T6**: Integration test: consistent shape, no NaN/Inf after warmup, <1ms/bar compute, 14 features, value range sanity. 5 tests.
- **Sprint 3 acceptance criteria**: All passed. Committed.
- **Next session**: Start Sprint 4 (Volume, Volatility, Momentum features)

### Session 5 (2026-03-23)
- **What happened**: Completed ALL 5 Sprint 4 tickets. 77 new tests (285 total).
- **S4-T1**: VolumeDeltaComputer (1-bar/5-bar normalized + raw), VWAPComputer (rolling typical price), VolumeROCComputer (1/3/6 bar). 23 tests.
- **S4-T2**: RealizedVolComputer (6/12-bar log return std), ParkinsonVolComputer (high-low range), EWMAVolComputer (decay=0.94), VolOfVolComputer (rolling std of realized vol). 25 tests.
- **S4-T3**: ROCComputer (1/3/6/12 bars), RSIComputer (Wilder's smoothing, bounded [0,100]), LinRegSlopeComputer (normalized by price), QuantileRankComputer (rolling rank [0,1]). 29 tests.
- **S4-T4**: Extended look-ahead bias tests to cover all 16 feature computers (truncation + shuffle + no-future-correlation). 3 tests.
- **S4-T5**: Updated integration test: 36 total features, all value range checks, <1ms/bar compute time. 5 tests.
- **Sprint 4 acceptance criteria**: All passed. Committed.
- **Next session**: Start Sprint 5 (Cross-market, Regime, Temporal features)

### Session 6 (2026-03-23)
- **What happened**: Completed ALL 6 Sprint 5 tickets. 131 new tests (245 feature tests total).
- **S5-T1**: NQReturnComputer (lagged 1-3 bars, US session gated), ETHRatioComputer (ratio + ROC), LeadLagComputer (rolling correlation at 1-3 lags), DivergenceComputer (BTC vs NQ/ETH). 31 tests.
- **S5-T2**: CyclicalTimeComputer (sin/cos for minute/hour/dow), SessionComputer (Asia/Europe/US one-hot), FundingTimeComputer (time-to-funding normalized). 24 tests.
- **S5-T3**: ERFeatureComputer (Kaufman ER at 10/20 bars), GARCHFeatureComputer (recursive GARCH(1,1) vol + ratio), HMMFeatureComputer (2-state vol regime proxy + regime change). 21 tests.
- **S5-T4**: RawPassthrough, RobustScaler (median/IQR), RankGaussianTransformer (probit), DualNormalizationPipeline. 24 tests.
- **S5-T5**: FeaturePipeline with build_default_registry (26 computers, 64 features), kwarg filtering via inspect, compute_batch with forward-fill. 11 tests.
- **S5-T6**: Updated look-ahead (truncation + shuffle + future correlation) and integration tests to cover all 64 features across Sprints 3-5. 8 tests.
- **Sprint 5 acceptance criteria**: All passed. Committed.
- **Next session**: Start Sprint 6 (Regime Detection)

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
| Sprint 3: Feature Engineering (Microstructure) | Complete | 2026-03-23 |
| Sprint 4: Features - Vol/Momentum | Complete | 2026-03-23 |
| Sprint 5: Features - Cross-market | Complete | 2026-03-23 |
| Sprint 6: Regime Detection | Not started | — |
| Sprint 7: Models + Stacking | Not started | — |
| Sprint 8: Confidence Gating | Not started | — |
| **Sprint 9: Risk Management** | **Not started** | — |
| Sprint 10: Backtesting Framework | Not started | — |
| Sprint 11: Hyperparameter Tuning | Not started | — |
| Sprint 12: Macro Events + Cascade | Not started | — |
| Sprint 13: API + Live + Monitoring | Not started | — |
| Sprint 14: Paper Trading | Not started | — |
| Sprint 15: Validation + Ablation | Not started | — |

---

## Sprint 4 Tickets

### S4-T1: Volume features (delta, VWAP, rate of change) [x]
**Create**: Volume delta, VWAP deviation, volume rate of change
**Files**:
- `src/ep2_crypto/features/volume.py` — VolumeDeltaComputer, VWAPComputer, VolumeROCComputer
- `tests/test_features/test_volume.py` — Golden dataset tests
**Verify**: `uv run pytest tests/test_features/test_volume.py -v`
**Notes**: Volume delta = buy_vol - sell_vol (from trade_sides). VWAP deviation = (close - vwap) / vwap. Volume ROC at 1, 3, 6 bars.

### S4-T2: Volatility features (realized, Parkinson, EWMA, vol-of-vol) [x]
**Create**: Multiple volatility estimators
**Files**:
- `src/ep2_crypto/features/volatility.py` — RealizedVolComputer, ParkinsonVolComputer, EWMAVolComputer
- `tests/test_features/test_volatility.py` — Golden dataset tests
**Verify**: `uv run pytest tests/test_features/test_volatility.py -v`
**Notes**: Parkinson uses high-low range. EWMA with configurable decay (0.94 default). Vol-of-vol = rolling std of volatility.

### S4-T3: Momentum features (ROC, RSI, linreg slope, quantile rank) [x]
**Create**: Momentum and mean-reversion indicators
**Files**:
- `src/ep2_crypto/features/momentum.py` — ROCComputer, RSIComputer, LinRegSlopeComputer, QuantileRankComputer
- `tests/test_features/test_momentum.py` — Golden dataset tests
**Verify**: `uv run pytest tests/test_features/test_momentum.py -v`
**Notes**: ROC at 1/3/6/12 bars. RSI(14) bounded [0,100]. LinReg slope over 20 bars. Quantile rank over 60 bars.

### S4-T4: Look-ahead bias tests for Sprint 4 features [x]
**Create**: Truncation + shuffle tests for volume, volatility, momentum
**Files**:
- Update `tests/test_features/test_lookahead.py` — Add Sprint 4 features to bias detection
**Verify**: `uv run pytest tests/test_features/test_lookahead.py -v`
**Notes**: Extend existing bias test infrastructure to cover new feature computers.

### S4-T5: Sprint 4 integration test [x]
**Create**: Combined Sprint 3+4 feature pipeline test
**Files**:
- Update `tests/test_features/test_feature_integration.py` — Add Sprint 4 features
**Verify**: `uv run pytest tests/test_features/ -v`
**Notes**: Verify total feature count, no NaN/Inf, <1ms/bar, feature correlations documented.

---

## Sprint 5 Tickets

### S5-T1: Cross-market features (NQ, ETH, lead-lag) [x]
**Create**: Cross-market signal features
**Files**:
- `src/ep2_crypto/features/cross_market.py` — NQReturnComputer, ETHRatioComputer, LeadLagComputer, DivergenceComputer
- `tests/test_features/test_cross_market.py` — Golden dataset tests
**Verify**: `uv run pytest tests/test_features/test_cross_market.py -v`
**Notes**: NQ 5-min returns lagged 1-3 bars (US hours only). ETH/BTC ratio momentum. Rolling lead-lag correlation. Divergence signals.

### S5-T2: Temporal features (cyclical encoding, session, funding) [x]
**Create**: Time-based features
**Files**:
- `src/ep2_crypto/features/temporal.py` — CyclicalTimeComputer, SessionComputer, FundingTimeComputer
- `tests/test_features/test_temporal.py` — Golden dataset tests
**Verify**: `uv run pytest tests/test_features/test_temporal.py -v`
**Notes**: Sin/cos for minute, hour, day-of-week. Session indicator (Asia/Europe/US). Time-to-funding in minutes.

### S5-T3: Regime features (ER, GARCH vol, HMM probs as inputs) [x]
**Create**: Regime context features for model input
**Files**:
- `src/ep2_crypto/features/regime_features.py` — ERFeatureComputer, GARCHFeatureComputer, HMMFeatureComputer
- `tests/test_features/test_regime_features.py` — Golden dataset tests
**Verify**: `uv run pytest tests/test_features/test_regime_features.py -v`
**Notes**: Efficiency Ratio (Kaufman) as feature. GARCH conditional vol. HMM probabilities.

### S5-T4: Normalization pipeline (dual: raw for trees, robust for neural) [x]
**Create**: Feature normalization with per-fold fitting
**Files**:
- `src/ep2_crypto/features/normalization.py` — RawPassthrough, RobustScaler, RankGaussianTransformer
- `tests/test_features/test_normalization.py` — Tests for per-fold fitting
**Verify**: `uv run pytest tests/test_features/test_normalization.py -v`
**Notes**: Raw pass-through for tree models. Robust scaling + rank-to-gaussian for neural. Per-fold fitting only.

### S5-T5: Feature pipeline (combine all modules) [x]
**Create**: Unified feature pipeline
**Files**:
- `src/ep2_crypto/features/pipeline.py` — FeaturePipeline combining all modules
- `tests/test_features/test_pipeline.py` — End-to-end pipeline tests
**Verify**: `uv run pytest tests/test_features/test_pipeline.py -v`
**Notes**: Handle NaN filling. Configurable feature selection. Consistent output shape.

### S5-T6: Look-ahead + integration tests for Sprint 5 [x]
**Create**: Bias detection and integration for all Sprint 5 features
**Files**:
- Update `tests/test_features/test_lookahead.py` — Add Sprint 5 features
- Update `tests/test_features/test_feature_integration.py` — Full pipeline test
**Verify**: `uv run pytest tests/test_features/ -v`
**Notes**: Total feature count 40-50. Pipeline produces consistent output.

---

## Future Sprint Ticket Decomposition

Sprints 6-14 will be decomposed into tickets when they become the current sprint.
See SPRINTS.md for high-level sprint definitions.
