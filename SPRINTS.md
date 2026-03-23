# Sprint Planning: ep2-crypto

## Overview

15 sprints covering the full development lifecycle. Each sprint builds on the previous.
Total estimated timeline: 44-58 days of development.

**Status Legend**: [ ] Not started | [~] In progress | [x] Complete

---

## Sprint 1: Foundation (2-3 days) [x]

**Status**: Complete (2026-03-23) — 100 tests passing

### Objectives
Set up the development environment, database schema, configuration management, and logging infrastructure.

### Deliverables
- [x] `pyproject.toml` with all dependencies (exists, needs update)
- [x] `uv.lock` (exists)
- [ ] `src/ep2_crypto/config.py` - Pydantic Settings with env-var configuration
- [ ] `src/ep2_crypto/db/schema.py` - SQLite schema for all data types
- [ ] `src/ep2_crypto/db/repository.py` - Parameterized query layer
- [ ] `src/ep2_crypto/logging.py` - structlog JSON configuration
- [ ] `tests/test_db/test_schema.py` - Schema creation tests
- [ ] `tests/test_db/test_repository.py` - Query parameterization tests

### Acceptance Criteria
- `uv run python -c "import ep2_crypto"` succeeds
- Database creates all tables on first run
- `grep -r "f'" src/ep2_crypto/db/` returns zero results (no f-string SQL)
- structlog outputs JSON format to stderr
- Config loads from environment variables with sensible defaults
- All tests pass: `uv run pytest tests/test_db/`

### Key Decisions
- SQLite for development, TimescaleDB schema-compatible for production migration
- Pydantic Settings for config (not raw os.environ)
- All SQL via parameterized queries through repository pattern

### Dependencies
None (first sprint)

---

## Sprint 2: Data Ingestion - Exchange & Derivatives (3-4 days) [x]

### Objectives
Build WebSocket and REST collectors for Binance and Bybit exchange data.

### Deliverables
- [ ] `src/ep2_crypto/ingest/base.py` - Base collector interface (async context manager)
- [ ] `src/ep2_crypto/ingest/exchange.py` - Binance WS: klines (1m), depth@100ms (L2 top 20), aggTrades
- [ ] `src/ep2_crypto/ingest/derivatives.py` - Bybit: OI (5-min poll), funding rate (5-min poll), liquidations (WS)
- [ ] `src/ep2_crypto/ingest/orchestrator.py` - Async task management, reconnection, health checks
- [ ] `tests/test_ingest/test_exchange.py` - Mock WebSocket tests
- [ ] `tests/test_ingest/test_derivatives.py` - Mock REST/WS tests
- [ ] `tests/test_ingest/test_orchestrator.py` - Lifecycle tests

### Acceptance Criteria
- Collectors run for 10 minutes without crashes
- Data stored in SQLite with correct schema
- Liquidation events appear during volatile periods
- OI and funding rate update every 5 minutes
- Automatic reconnection after simulated disconnect (test with mock)
- No duplicate records in database (upsert or dedup logic)
- Graceful shutdown on SIGTERM/SIGINT

### Key Technical Details
- ccxt pro `watch_order_book('BTC/USDT:USDT', limit=20)` for depth
- ccxt pro `watch_trades('BTC/USDT:USDT')` for aggTrades
- Bybit liquidation: `allLiquidation.BTCUSDT` topic via `wss://stream.bybit.com/v5/public/linear`
- Bybit WS heartbeat: send `ping` every 20 seconds
- Bybit order book depths: [1, 50, 200, 1000] only - slice from 50

### Dependencies
Sprint 1 (database, config, logging)

---

## Sprint 3: Feature Engineering - Microstructure (3-4 days) [x]

### Objectives
Implement the highest-priority features: order book imbalance, order flow imbalance, microprice, and trade flow features. These capture 60-80% of achievable 5-min signal.

### Deliverables
- [ ] `src/ep2_crypto/features/base.py` - Feature interface and registry
- [ ] `src/ep2_crypto/features/microstructure.py`:
  - Order Book Imbalance (OBI): weighted, levels 1-3 and 1-5
  - Order Flow Imbalance (OFI): Cont-Stoikov-Talreja model, multi-level
  - Microprice (Gatheral-Stoikov): volume-weighted mid
  - Trade Flow Imbalance (TFI): 30s and 5min windows
  - Bid-ask spread (relative to mid)
  - Absorption detection (large volume delta without price change)
  - Kyle's Lambda (rolling, as conditioning variable)
- [ ] `tests/test_features/test_microstructure.py` - Golden dataset tests
- [ ] `tests/test_features/test_lookahead.py` - Look-ahead bias detection (shuffle test)

### Acceptance Criteria
- OBI correlates with short-term price direction (sanity check on historical data)
- All features use only data at time <= t (verified by truncation test)
- Golden dataset tests pass against hand-verified values
- No NaN/Inf in output after warmup period (configurable warmup)
- Features compute in < 1ms per bar (numpy-native, no pandas on hot path)

### Key Research Reference
- `research/microstructure_ofi_microprice.md` - Implementation details for OFI and microprice

### Dependencies
Sprint 1 (database), Sprint 2 (data available for testing)

---

## Sprint 4: Feature Engineering - Volume, Volatility, Momentum (2-3 days) [x]

### Objectives
Build Tier 1 and Tier 2 features for volume, volatility, and momentum.

### Deliverables
- [ ] `src/ep2_crypto/features/volume.py`:
  - Volume delta (buy - sell), 1-bar and 5-bar
  - VWAP deviation (1hr rolling)
  - Volume rate of change
  - Volume profile (price levels with high volume)
- [ ] `src/ep2_crypto/features/volatility.py`:
  - Realized volatility (5-min, 30-min rolling)
  - Parkinson volatility (30-min, uses high-low)
  - EWMA volatility (configurable decay)
  - Volatility of volatility (vol-of-vol)
- [ ] `src/ep2_crypto/features/momentum.py`:
  - Rate of Change at 1, 3, 6, 12 bars
  - RSI(14) with regime filter
  - Linear regression slope(20)
  - Price quantile rank(60)
- [ ] Tests for each module with golden datasets

### Acceptance Criteria
- All features produce stable, non-degenerate output on test data
- Volatility measures are strictly positive
- RSI bounded [0, 100]
- No look-ahead bias (truncation test passes)
- Feature correlations documented (identify redundant features)

### Dependencies
Sprint 1, Sprint 3 (feature base interface)

---

## Sprint 5: Feature Engineering - Cross-market, Regime, Temporal (2-3 days) [x]

### Objectives
Build context features: cross-market signals, temporal encoding, and regime-input features.

### Deliverables
- [ ] `src/ep2_crypto/features/cross_market.py`:
  - NQ 5-min returns (lagged 1-3 bars, US hours only)
  - ETH/BTC ratio momentum
  - Lead-lag rolling correlation at multiple lags
  - Divergence signals (BTC vs current leader)
  - Coinbase premium (BTC price Coinbase vs Binance)
  - Polymarket BTC-related market probabilities (30-min poll, regime feature)
  - Deribit options IV term structure and put/call ratio (5-min poll)
- [ ] `src/ep2_crypto/features/temporal.py`:
  - Cyclical encoding: sin/cos for minute, hour, day-of-week
  - Session indicator (Asia/Europe/US)
  - Time-to-funding (minutes until next 8h settlement)
- [ ] `src/ep2_crypto/features/regime_features.py`:
  - Efficiency Ratio (Kaufman) as feature input
  - GARCH conditional volatility as feature input
  - HMM state probabilities as feature input
- [ ] `src/ep2_crypto/features/normalization.py`:
  - Dual pipeline: raw pass-through for tree models, robust scaling + rank-to-gaussian for neural
  - Per-fold fitting (never global)
- [ ] `src/ep2_crypto/features/pipeline.py`:
  - Combine all feature modules into single output
  - Handle NaN filling (forward-fill prices, zero counts)
  - Configurable feature selection
- [ ] Tests for pipeline end-to-end

### Acceptance Criteria
- Cross-market features properly handle 15-min delay (NQ bars aligned to actual arrival time)
- Temporal features are cyclical (sin/cos, not raw integers)
- Normalization fitted on training data only (test with mock fold)
- Pipeline produces consistent output shape regardless of missing data sources
- Total feature count: 25-35 before selection

### Key Technical Detail
- NQ data arrives 15-min delayed. Timeline must present bars in arrival order, not timestamp order
- Cross-market features gated to US session hours only (16:00-00:00 UTC)

### Dependencies
Sprint 2 (cross-market data), Sprint 3-4 (feature interfaces)

---

## Sprint 6: Regime Detection (3-4 days) [x]

### Objectives
Build the hierarchical regime detection ensemble: fast indicators per bar, core HMM, slow Hurst, and meta-learner.

### Deliverables
- [ ] `src/ep2_crypto/regime/efficiency_ratio.py`:
  - Kaufman Efficiency Ratio (every bar, O(1))
  - Threshold-based fast regime estimate
- [ ] `src/ep2_crypto/regime/garch.py`:
  - GJR-GARCH(1,1)-t conditional volatility (every bar)
  - Vol regime classification (low/medium/high)
- [ ] `src/ep2_crypto/regime/hmm.py`:
  - 2-state GaussianHMM (refit weekly on 7-day sliding window)
  - Filtered probabilities (not Viterbi - use forward algorithm for online)
  - Model selection via BIC (n=2,3,4,5)
  - Pseudo-incremental: `init_params=""` with warm start
- [ ] `src/ep2_crypto/regime/bocpd.py`:
  - Bayesian Online Change Point Detection
  - Early warning for regime transitions (2-10 bars faster than HMM)
- [ ] `src/ep2_crypto/regime/detector.py`:
  - Hierarchical ensemble combining all layers
  - LightGBM meta-learner on regime features (weekly refit)
  - Output: regime label + probabilities + change point alert

### Acceptance Criteria
- 2-3 distinct regimes visible when plotted against price history
- Regime labels are semantically stable across refits (sorted by volatility)
- BOCPD fires before HMM detects transition (measure lead time)
- Regime probabilities sum to 1.0
- Efficiency Ratio + GARCH update in O(1) per bar
- All regime components tested independently and as ensemble

### Key Research Reference
- RESEARCH_SYNTHESIS.md section "Regime Detection: Hierarchical Ensemble"

### Dependencies
Sprint 1 (database), Sprint 4 (volatility features used as input)

---

## Sprint 7: Models - LightGBM + CatBoost + GRU + Stacking (4-5 days) [x]

### Objectives
Build the core prediction models and stacking ensemble.

### Deliverables
- [ ] `src/ep2_crypto/models/lgbm_direction.py`:
  - LightGBM ternary classifier (up/flat/down)
  - Warm-start across folds: `init_model=previous_model`
  - Early stopping with purge gap from validation
  - SHAP feature importance tracking
- [ ] `src/ep2_crypto/models/catboost_direction.py`:
  - CatBoost ternary classifier (ordered boosting for diversity)
  - Same target and validation as LightGBM
- [ ] `src/ep2_crypto/models/gru_features.py`:
  - GRU(input_size=N, hidden_size=64, num_layers=2, dropout=0.3)
  - Sequence length: 24 bars (2 hours)
  - Hidden state extraction as features for stacking
  - AdamW optimizer, gradient clipping at 1.0, cosine annealing LR
  - ONNX export for fast inference
- [ ] `src/ep2_crypto/models/quantile.py`:
  - LightGBM quantile regression (5 quantiles: 10th, 25th, 50th, 75th, 90th)
  - Risk assessment: prediction interval width
- [ ] `src/ep2_crypto/models/stacking.py`:
  - Logistic regression meta-learner on OOF predictions
  - OOF generated via inner walk-forward (never in-sample)
  - Ensemble weights optimized for Sharpe via scipy.optimize
- [ ] `src/ep2_crypto/models/calibration.py`:
  - Isotonic regression calibration on held-out set
  - Calibration curve evaluation (reliability diagram)
- [ ] Tests for each model and stacking pipeline

### Acceptance Criteria
- Each model trains successfully on 14-day window of test data
- Walk-forward validation shows >52% directional accuracy (above random)
- Stacking improves over best single model by measurable margin
- Feature importance tracked and differs meaningfully by regime
- GRU DataLoader never shuffles
- All models serializable (save/load round-trip test)
- Calibration curve shows improved reliability after isotonic regression

### Key Parameters
- LightGBM: `num_leaves=31, max_depth=5, learning_rate=0.05, min_child_samples=50`
- GRU: `hidden_size=64, num_layers=2, dropout=0.3, seq_len=24`
- Training window: 14 days (4,032 bars), test: 1 day (288 bars)

### Dependencies
Sprint 3-5 (features), Sprint 6 (regime as input feature)

---

## Sprint 8: Confidence Gating Pipeline (3-4 days) [x]

### Objectives
Build the full confidence gating pipeline - the single biggest Sharpe multiplier (2-4x improvement).

### Deliverables
- [ ] `src/ep2_crypto/confidence/meta_labeling.py`:
  - Secondary model predicts "will THIS trade be profitable?"
  - Lopez de Prado's technique (see `research/meta_labeling_deep_research.md`)
  - Input: primary model prediction + features + regime
  - Output: probability of profitable trade
- [ ] `src/ep2_crypto/confidence/conformal.py`:
  - Conformal prediction with adaptive alpha
  - Prediction set must be singleton {UP} or {DOWN} to trade
  - If set is {UP, DOWN} or {UP, FLAT, DOWN}: abstain
- [ ] `src/ep2_crypto/confidence/gating.py`:
  - Full pipeline orchestration:
    1. Isotonic regression calibration (from Sprint 7)
    2. Meta-labeling gate
    3. Ensemble agreement check (variance < threshold)
    4. Conformal prediction gate
    5. Signal filters (vol range, regime stability, liquidity)
    6. Adaptive confidence threshold (regime-dependent, weekly update)
    7. Drawdown gate (progressive 3% -> 15% reduction)
  - Each gate logs its decision with structlog
- [ ] `src/ep2_crypto/confidence/position_sizing.py`:
  - Quarter-Kelly: `size = 0.25 * kelly_fraction * composite_confidence`
  - Kelly with uncertain probabilities (Bayesian Kelly)
  - Max position cap: 5% of capital
- [ ] Tests for each gate independently and full pipeline

### Acceptance Criteria
- Meta-labeling improves Sharpe by >50% (on backtest data)
- Conformal prediction filters out ambiguous predictions
- Full gating pipeline trades fewer but higher-quality signals
- Each gate can be independently enabled/disabled for ablation
- Position sizing never exceeds max cap
- Drawdown gate reduces size progressively as drawdown increases

### Key Research Reference
- `research/meta_labeling_deep_research.md`
- RESEARCH_SYNTHESIS.md section "Confidence Gating Pipeline"

### Dependencies
Sprint 7 (models produce predictions to gate)

---

## Sprint 9: Risk Management Engine (5-7 days) [x]

### Objectives
Build the MOST CRITICAL module in the entire system: capital preservation. This MUST exist before backtesting so backtests reflect real trading constraints. Without this, backtest results are fantasy. **Backed by 23 dedicated research reports totaling 3MB.**

### Why This Sprint Is The Most Important
- A bad model loses slowly; bad risk management loses everything instantly
- Exchange insolvency (FTX) is the #1 existential risk — mitigated by capital allocation
- Human override of automated halts is the #2 risk — mitigated by interference protection
- Position sizing bugs (decimal errors) are the #3 risk — mitigated by 3-layer hard limits
- Every position entry MUST atomically place an exchange-side stop-market order

### Design Principles (from research)
- **Risk engine wraps exchange API** — trading engine NEVER has direct exchange access
- **Fail-safe, not fail-open** — if anything goes wrong, STOP TRADING
- **Separate process authority** — risk engine can kill trading engine, not vice versa
- **Exchange state is truth** — local state is a cache; reconcile every 5 minutes
- **Position sizing chain**: `min(quarter_kelly, max_cap, heat_constrained, budget_constrained) × weekend_mult × eod_mult × drawdown_mult`

### Deliverables

#### Core Risk Components
- [ ] `src/ep2_crypto/risk/__init__.py`
- [ ] `src/ep2_crypto/risk/position_tracker.py`:
  - Real-time position state (entry price, size, unrealized PnL, duration, MAE)
  - Mark-to-market on every bar
  - Margin utilization tracking (never >30% of capital)
  - Liquidation price calculation (Binance isolated margin formula)
  - Auto-close at 85% margin ratio (BEFORE exchange liquidates you)
  - Maximum position cap enforcement (5% of capital, adjustable by leverage)
  - Maximum 1 open position at a time
  - Position reconciliation with exchange every 5 minutes
- [ ] `src/ep2_crypto/risk/kill_switches.py`:
  - DailyLossLimit: halt when daily loss > 2% of capital (configurable)
  - WeeklyLossLimit: halt when weekly loss > 5%
  - MaxDrawdownHalt: halt when peak-to-trough > 15%
  - ConsecutiveLossHalt: halt after 10 consecutive losses (expected once per 2 months at 55% WR)
  - EmergencyKillSwitch: close all positions immediately (manual trigger)
  - ALL kill switches persist to SQLite (survive restart)
  - ALL require manual reset with reason string (no auto-resume)
  - State machine: ARMED → TRIGGERED → RESET (with audit trail)
  - Cascading hierarchy: per-trade → daily → weekly → max DD
  - State exposed via health endpoint data
- [ ] `src/ep2_crypto/risk/drawdown_gate.py`:
  - **Convex reduction** (k=1.5), NOT step function:
    - Formula: `multiplier = max(0, (1 - dd/max_dd)^1.5)`
    - At 3% DD: ~71% size
    - At 5% DD: ~54% size
    - At 8% DD: ~32% size
    - At 12% DD: ~9% size
    - At 15% DD: 0% (halt)
  - **Duration-based reduction** (independent of depth):
    - 3 days underwater: 80% size
    - 7 days: 40% size
    - 14 days: halt
  - Final multiplier = min(depth_mult, duration_mult)
  - Graduated re-entry protocol: 10% → 25% → 50% → 75% → 100% over 5 phases
  - Each phase requires profitability to advance; failure drops back to evaluation
  - Cooldown period: minimum 5 bars at reduced size before any increase
  - Bayesian edge assessment at each recovery phase gate
- [ ] `src/ep2_crypto/risk/volatility_guard.py`:
  - Minimum volatility: 15% annualized (below = no trade, costs eat signal)
  - Maximum volatility: 150% annualized (above = reduce size or abstain)
  - Graduated vol response: 2x baseline → reduce 50%; 3x → halt entries; 5x → close all
  - Trading hours: 08:00-21:00 UTC (configurable)
  - Weekend sizing: -30% reduction (Friday 20:00 to Monday 04:00 UTC)
  - Funding settlement proximity: reduce/skip entry within 30 min of settlement if adverse funding > 5 bps
  - End-of-day entry blocking: no new positions in last 15 minutes of trading window
- [ ] `src/ep2_crypto/risk/position_sizer.py`:
  - **Full sizing chain**: `min(quarter_kelly, max_cap, heat_constrained, budget_constrained) × weekend × eod × drawdown × regime`
  - Quarter-Kelly with Bayesian uncertainty: integrate over posterior distribution of win rate
  - ATR-based stop loss: 2.0 ATR default, confidence-weighted (1.5-3.5 ATR range)
  - Multi-level stop system:
    - Level 1: Model signal reversal (primary exit)
    - Level 2: Time stop — 6 bars max holding period
    - Level 3: ATR stop — confidence-adjusted
    - Level 4: Catastrophic stop — 3% max per trade, non-negotiable
  - Portfolio heat tracking: total open risk never > 2% of capital
  - Daily risk budget: 3% total, tracked and depleted through the day
  - Minimum trade size: enforce exchange minimums ($5 notional on Binance)
  - Exchange-side stop placement: ATOMIC with position entry (non-negotiable)
- [ ] `src/ep2_crypto/risk/risk_manager.py`:
  - RiskManager orchestrates ALL components
  - **Pre-trade checks** (`approve_trade`):
    1. Check all kill switches → reject if any triggered
    2. Check volatility guard → reject if out of range
    3. Check drawdown gate → get multiplier
    4. Check daily risk budget → reject if exhausted
    5. Check position limit → reject if position open
    6. Compute position size → full chain
    7. Return: `TradeDecision(approved, quantity, stop_price, reason)`
  - **Post-trade monitoring** (`on_bar`):
    1. Mark position to market
    2. Update equity, drawdown, daily PnL
    3. Check all kill switches (may trigger)
    4. Check max holding period → force close at 6 bars
    5. Check stop loss → bar.low vs stop for longs, bar.high for shorts
    6. Return: list of `RiskAction` (close, reduce, alert)
  - `get_risk_state() -> RiskState` — complete state for health/monitoring/dashboard
  - `trigger_emergency(reason)` — immediate halt
  - `reset_kill_switch(name, reason)` — requires reason string, logged
  - All decisions logged via structlog with full context
  - State persistence to SQLite on every state change

#### Risk Configuration
- [ ] `src/ep2_crypto/risk/config.py`:
  - `RiskConfig` Pydantic model with ALL risk parameters
  - Validation: reject invalid configs (e.g., max_position > 100%)
  - No hot-reload — require explicit restart to change risk params
  - Default values from research: 2% daily, 5% weekly, 15% max DD, 0.25% risk/trade

#### Monitoring Integration
- [ ] `src/ep2_crypto/monitoring/risk_exporter.py`:
  - Prometheus metrics for all risk state (49 metric names from research)
  - `PrometheusRiskExporter` reads `RiskState` and pushes to Prometheus
  - Traffic light system: GREEN/YELLOW/RED/EMERGENCY
  - Daily loss budget gauge, drawdown gauge, kill switch status panel

#### Testing (95%+ coverage required)
- [ ] `tests/test_risk/test_kill_switches.py`:
  - Each switch triggers at exact threshold (boundary tests: ±epsilon)
  - Persistence: trigger → kill process → restart → verify still triggered
  - Manual reset requires reason string
  - Cascading: multiple switches can trigger simultaneously
- [ ] `tests/test_risk/test_drawdown_gate.py`:
  - Convex reduction matches formula at every DD level
  - Duration-based reduction independent of depth
  - V-shaped recovery with cooldown (no premature recovery)
  - Graduated re-entry: must be profitable to advance
- [ ] `tests/test_risk/test_position_sizer.py`:
  - Kelly golden dataset (known WR/payoff → expected fraction)
  - Full sizing chain: verify every multiplier applied
  - Stop distance: 2.0 ATR default, confidence-adjusted
  - Never exceeds max cap (property-based test with random inputs)
- [ ] `tests/test_risk/test_risk_manager.py`:
  - Full orchestration: normal day, bad day, crash, slow bleed, recovery
  - approve_trade returns reason for every rejection
  - on_bar detects stop loss on bar high/low (not just close)
  - Emergency trigger closes everything immediately
- [ ] `tests/test_risk/test_risk_properties.py`:
  - Hypothesis property-based tests:
    - Position size NEVER exceeds max cap (500+ random inputs)
    - After kill switch, ZERO trades approved until reset
    - Total exposure NEVER exceeds portfolio heat limit
- [ ] `tests/test_risk/test_risk_persistence.py`:
  - Kill switch state survives file-backed SQLite close/reopen
  - Position tracker state survives crash simulation
  - Daily counters reset at midnight (intentional, documented)
- [ ] `tests/test_risk/test_risk_golden.py`:
  - 20+ parameterized golden dataset cases
  - Known (equity, price, confidence, WR, DD) → expected (approved, size, reason)
  - These NEVER change — regression protection
- [ ] `tests/test_risk/test_risk_stress.py`:
  - March 2020 COVID replay: system survives with <15% DD
  - Flash crash 20% in 30 min: stop fires within first bar
  - 10 consecutive losses in 1 hour: kill switch triggers
  - Slow bleed 14 days: duration gate activates

### Acceptance Criteria
- Daily loss limit triggers at exactly 2% (not 1.99%, not 2.01%)
- Drawdown gate uses convex formula (k=1.5), verified at 5 DD levels
- Kill switch state persists to disk and survives process restart (file-backed test)
- Kill switch requires explicit `reset(reason)` call (no auto-resume)
- Volatility guard blocks trades when vol < 15% AND > 150%
- Position sizer never exceeds 5% of capital (property-based test, 500+ random inputs)
- Maximum holding period force-exits at exactly 6 bars
- Exchange-side stop order is part of position entry (not a separate step)
- All risk decisions logged with full context (structlog JSON)
- Risk module has 95%+ line coverage
- All 4 stress scenarios pass
- All 20+ golden dataset cases pass

### Key Research Reference (23 reports)
**Core architecture**: `RR-risk-engine-architecture.md`, `RR-risk-implementation-guide.md`
**Position sizing**: `RR-risk-position-sizing-methods.md` (Kelly, optimal f, ATR, Bayesian)
**Kill switches**: `RR-risk-kill-switch-design.md` (thresholds, recovery, testing, fire drills)
**Drawdown**: `RR-risk-drawdown-management.md` (convex reduction, duration, recovery protocol, Bayesian edge test)
**Stop losses**: `RR-risk-stop-loss-strategies.md` (multi-level system, confidence-weighted, cost interaction)
**Margin/liquidation**: `RR-risk-margin-liquidation.md` (Binance formulas, auto-close at 85%)
**Tail risk**: `RR-risk-tail-risk-black-swan.md` (survive everything analysis, compound scenarios)
**Worst case**: `RR-risk-worst-case-scenarios.md` (risk catalog, reverse stress testing)
**Testing**: `RR-risk-testing-framework.md` (property-based, chaos engineering, golden dataset)
**Money management**: `RR-risk-money-management.md` (account stages, variance drain, compounding)
**Capital math**: `RR-risk-capital-preservation-math.md` (ruin probability, recovery math, Bayesian Kelly)
**Monitoring**: `RR-risk-monitoring-dashboard.md` (49 Prometheus metrics, Grafana panels, alert rules)
**Hedge fund practices**: `RR-risk-hedge-fund-practices.md`, `RR-risk-institutional-deep-research.md`
**Funding rate**: `RR-risk-funding-rate.md` (settlement timing, cost modeling)
**Exchange ops**: `RR-risk-exchange-operational.md` (API errors, reconciliation, key security)
**Regime risk**: `RR-risk-regime-conditional.md` (vol targeting, regime-dependent sizing/stops)
**Risk-adjusted optimization**: `RR-risk-adjusted-return-optimization.md` (CVaR, multi-objective)
**Portfolio heat**: `RR-risk-portfolio-heat-exposure.md` (daily budget, exposure limits)
**Multi-signal allocation**: `RR-risk-parity-multi-signal.md` (60/25/15 allocation)
**Backtesting integration**: `RR-risk-backtesting-integration.md` (why backtests without risk lie)

### Dependencies
Sprint 1 (database for state persistence), Sprint 4 (volatility features for vol guard)

---

## Sprint 10: Backtesting Framework (4-5 days) [x]

### Objectives
Build the complete backtesting infrastructure: walk-forward engine, execution simulator, statistical validation, and benchmarks.

### Deliverables
- [ ] `src/ep2_crypto/backtest/walk_forward.py`:
  - Purged walk-forward with embargo
  - 14-day sliding window, 1-day test, 18-bar purge, 12-bar embargo
  - Nested inner loop for hyperparameter selection
  - Walk-forward auditor (automated leak detection)
- [ ] `src/ep2_crypto/backtest/engine.py`:
  - Event-driven backtest engine (Numba-compiled inner loop)
  - Next-bar-open execution (never fill at current bar close)
  - Full state persistence (positions, margin, funding, liquidation levels)
- [ ] `src/ep2_crypto/backtest/simulator.py`:
  - Slippage model: sqrt-impact, 1-3 bps
  - Transaction fees: 4 bps taker / 2 bps maker (configurable)
  - Latency simulation: 50-200ms delay
  - Partial fill: cap at 10% of bar volume
  - Funding rate application at 00:00, 08:00, 16:00 UTC
- [ ] `src/ep2_crypto/backtest/metrics.py`:
  - Sharpe (Lo-corrected, annualized with sqrt(105,120))
  - Sortino, Calmar, max drawdown + duration, CVaR(5%)
  - Profit factor, win rate, expectancy per trade
  - Rolling 30-day Sharpe
  - Regime-specific breakdowns
- [ ] Update `src/ep2_crypto/benchmarks/` (existing code):
  - Verify benchmark strategies are compatible with new backtest engine
  - Add funding rate carry benchmark
- [ ] Statistical validation module:
  - Probabilistic Sharpe Ratio (PSR)
  - Deflated Sharpe Ratio (DSR)
  - Permutation test (5K permutations)
  - Block bootstrap CI (10K iterations)
  - Walk-forward stability (CV of fold Sharpes < 0.5)
- [ ] `scripts/backtest.py` - CLI entry point

### Acceptance Criteria
- Backtest runs on 30+ days of data without errors
- Results always include transaction costs (no zero-cost mode)
- System beats all benchmarks (buy-and-hold, momentum, random, funding carry)
- Statistical validation produces GENUINE_EDGE / INCONCLUSIVE / LIKELY_NOISE verdict
- Monte Carlo bootstrap produces 95% CI on Sharpe
- Cost sensitivity analysis at 0, 4, 8, 12, 16, 20 bps shows break-even > 15 bps
- Regime-specific metrics show which regimes are profitable

### Key Research Reference
- `BACKTESTING_PLAN.md` - Complete backtesting methodology
- `research/statistical_validation_tests.py` - Reference implementation
- `research/monte_carlo_validation.md`
- `research/transaction_cost_modeling.md`

### Dependencies
Sprint 7 (models), Sprint 8 (confidence gating), **Sprint 9 (risk engine — backtests MUST include risk constraints)**

---

## Sprint 11: Hyperparameter Tuning (2-3 days) [x]

### Objectives
Optimize model and pipeline hyperparameters using Optuna with walk-forward Sharpe as objective.

### Deliverables
- [ ] `src/ep2_crypto/models/tuning.py`:
  - Optuna study for LightGBM (max_depth, learning_rate, num_leaves, reg_alpha, reg_lambda, etc.)
  - Optuna study for CatBoost (depth, learning_rate, l2_leaf_reg, etc.)
  - Optuna study for GRU (hidden_size, num_layers, lr, dropout, seq_len)
  - MedianPruner(n_startup_trials=5, n_warmup_steps=10)
  - XGBoostPruningCallback equivalent for LightGBM
  - Objective: walk-forward Sharpe (NOT accuracy)
  - Deflated Sharpe Ratio to account for multiple trials
- [ ] Confidence threshold optimization:
  - Grid search over threshold values per regime
  - Optimize for Sharpe, not win rate
- [ ] Feature importance stability check:
  - Top features should be stable across walk-forward folds
  - Features that appear in <50% of folds are candidates for removal

### Acceptance Criteria
- Optuna completes 50+ trials per model
- Best hyperparameters improve Sharpe over defaults
- Pruning stops >30% of trials early (efficiency check)
- Feature importance stable across folds (top 10 overlap > 70%)
- DSR applied to final selected parameters
- Tuned parameters documented in config

### Key Search Spaces
- LightGBM: `num_leaves[15-63], max_depth[3-8], lr[0.01-0.3], min_child[20-100], subsample[0.6-1.0], colsample[0.6-1.0], reg_alpha[1e-3-25], reg_lambda[1e-3-25]`
- GRU: `hidden[32-256], layers[1-3], lr[1e-5-1e-2], dropout[0.1-0.5], seq_len[12-60]`

### Dependencies
Sprint 9 (backtest framework for Sharpe evaluation)

---

## Sprint 12: Event-Driven Macro Module + Cascade Detector (3-4 days) [x]

### Objectives
Build independent alpha sources: macro event trading (NQ lead-lag) and liquidation cascade detection.

### Deliverables
- [ ] `src/ep2_crypto/events/macro.py`:
  - Pre-scheduled macro calendar (CPI, FOMC, NFP dates)
  - NQ monitoring at T+0 of macro event
  - Trade logic: enter BTC in NQ direction at T+2min, exit at T+5-10min
  - VIX gate: skip when VIX > 35
  - EWMA BTC-NQ correlation (trailing 6h) for signal weighting
- [ ] `src/ep2_crypto/events/cascade.py`:
  - Multi-factor cascade score: OI percentile, funding z-score, liquidation burst rate, book depth, price velocity
  - Hawkes process for liquidation intensity modeling
  - Output: cascade probability -> position size reduction
  - Liquidation heatmap estimation from volume profile + leverage assumptions
- [ ] `src/ep2_crypto/ingest/macro.py`:
  - Economic calendar data source
  - NQ real-time monitoring integration
- [ ] Tests for both modules

### Acceptance Criteria
- Macro module fires on historical CPI/FOMC dates (backtest validation)
- NQ lead-lag shows ~62-68% accuracy on historical macro events
- Cascade detector fires before or during known cascade events (FTX, Luna, etc.)
- Both modules operate independently from ML model
- VIX gate correctly suppresses signals during extreme volatility
- Hawkes process parameters estimated on historical liquidation data

### Key Research Reference
- RESEARCH_SYNTHESIS.md section "Event-Driven Macro Module"
- RESEARCH_SYNTHESIS.md section "Liquidation Cascade Detector"

### Dependencies
Sprint 2 (derivatives data for cascade), Sprint 5 (cross-market data for NQ)

---

## Sprint 13: API + Live Prediction + Monitoring (3-4 days) [x]

**Status**: Complete (2026-03-23) — 239 tests passing

### Objectives
Build the production API, live prediction loop, and monitoring infrastructure.

### Deliverables
- [ ] `src/ep2_crypto/api/server.py`:
  - `GET /predict` - current prediction (direction, magnitude, confidence, regime)
  - `GET /health` - checks all streams, models loaded, last prediction timestamp
  - `GET /metrics` - live accuracy, rolling Sharpe, trade count
  - `GET /regime` - current regime + transition probabilities
  - All responses include timestamps and staleness warnings
- [ ] `scripts/live.py`:
  - Async event loop: all collectors + feature pipeline + model inference
  - Triggers on each 5-min candle close
  - Stores prediction + actual outcome for live accuracy
  - Auto-retrain models every 2-4 hours (warm-start)
  - Graceful degradation: if one source fails, lower confidence
- [ ] `src/ep2_crypto/monitoring/alpha_decay.py`:
  - CUSUM on returns (fast, 2-5 bars)
  - Rolling Sharpe decline (medium, 7-14 days)
  - ADWIN adaptive windowing
  - SPRT on win rate (50 trades)
  - Response protocol: Warning -> Caution -> Stop -> Emergency
- [ ] `src/ep2_crypto/monitoring/drift.py`:
  - Feature PSI (Population Stability Index) per feature
  - Alert at PSI > 0.2
  - Daily drift report
- [ ] `src/ep2_crypto/monitoring/kill_switch.py`:
  - Daily loss limit (2-3%)
  - Weekly loss limit (5%)
  - Max drawdown (15%)
  - Consecutive loss limit
  - Automated trading halt with manual reset required

#### Feedback Loop Infrastructure (critical for operational edge)
- [ ] `src/ep2_crypto/monitoring/performance_logger.py`:
  - Log every trade: prediction, confidence, actual direction, PnL, slippage (expected vs actual), latency (bar close to order fill), regime, features used
  - Log every bar: model state, risk engine state, feature values snapshot
  - Store in TimescaleDB (or SQLite with time-series indices for dev)
  - This is the "flight recorder" — everything else depends on it
- [ ] `src/ep2_crypto/monitoring/slippage_tracker.py`:
  - Track expected slippage (from model) vs actual slippage (from fill price)
  - Aggregate stats: mean, p50, p95, p99 slippage by order size, time of day, volatility regime
  - Feed slippage stats back to position sizer (adaptive cost model)
  - Alert when actual slippage > 2x expected for 10+ consecutive trades
- [ ] `src/ep2_crypto/monitoring/retrain_trigger.py`:
  - Auto-retrain pipeline with validation gate:
    1. Trigger conditions: PSI > 0.2 on any top-10 feature, OR rolling 7d Sharpe < 50% of baseline, OR ADWIN fires
    2. Retrain: new walk-forward fold on last 14 days, warm-start LightGBM/CatBoost, GRU fine-tune (2 epochs)
    3. Recalibrate: isotonic regression + conformal thresholds
    4. Validate: new model must beat old model on last 24h holdout AND top-10 feature overlap > 70% AND calibration ECE < 0.05
    5. Deploy: atomic model swap if validation passes; keep old model + alert if fails
  - Configurable schedule: every 4h (default) or drift-triggered
  - Full audit trail of every retrain decision
- [ ] `src/ep2_crypto/monitoring/feature_rotation.py`:
  - Track per-feature PSI daily
  - If any feature has PSI > 0.3 for 7 consecutive days: flag for review
  - Auto-downweight (not remove) drifted features in next retrain cycle
  - Monthly feature importance report: which features gained/lost importance
- [ ] `src/ep2_crypto/monitoring/alerts.py`:
  - Telegram bot integration (python-telegram-bot library)
  - Alert tiers:
    - INFO: daily summary (PnL, Sharpe, trade count, regime)
    - WARNING: alpha decay warning, feature drift, slippage anomaly
    - CRITICAL: kill switch triggered, model validation failed, API down
    - EMERGENCY: exchange error, position stuck, catastrophic loss
  - Rate limiting: max 1 INFO/hour, max 5 WARNING/hour, unlimited CRITICAL/EMERGENCY
  - Optional Slack webhook as secondary channel
- [ ] `src/ep2_crypto/execution/quality_tracker.py`:
  - A/B test framework for execution strategies:
    - Strategy A: market orders (baseline)
    - Strategy B: limit IOC 1-2 ticks from mid
  - Track fill rate, slippage, and total cost per strategy
  - Auto-switch to best strategy after 100+ trades per arm
  - Log execution quality metrics for optimization

### Acceptance Criteria
- API responds in < 100ms
- Health endpoint checks ALL dependencies (not just "ok")
- Predictions emit within 1 second of candle close
- Alpha decay detection fires on simulated decay scenario
- Kill switches halt trading at configured thresholds
- System survives 24h continuous run without memory leaks
- Performance logger captures every trade with all fields (no missing data)
- Slippage tracker alerts when actual > 2x expected
- Auto-retrain validates new model before deploy (never blind swap)
- Telegram alerts fire within 30 seconds of trigger event
- Feature rotation correctly identifies drifted features on synthetic test

### Dependencies
Sprint 7-8 (models + gating), Sprint 9 (risk engine)

---

## Sprint 14: Paper Trading (2-3 days setup + 14-30 days running) [x]

### Objectives
Deploy paper trading with real market data, simulated execution, and full monitoring.

### Deliverables
- [ ] Paper trading exchange simulator:
  - `ExchangeInterface` with `PaperExchange` and `LiveExchange` implementations
  - 100% code path parity - only execution layer swaps
  - Fill simulation walks actual orderbook
- [ ] Deployment scripts:
  - Docker compose for paper trading setup
  - Monitoring dashboard (Prometheus + Grafana or simple web UI)
- [ ] Daily report generation:
  - PnL, Sharpe, drawdown, trade count, regime breakdown
  - Feature drift report
  - Alpha decay indicators

### Acceptance Criteria (Go/No-Go for live trading)
- Calendar days >= 14
- Total trades >= 200
- Win rate > 51%
- Sharpe (annualized) > 1.0
- Max drawdown < 8%
- Profit factor > 1.2
- Avg trade PnL > 2 bps after fees
- Consecutive losses < 15
- Profitable in >= 2 of 3 regimes
- Model confidence correlates with actual accuracy (r > 0.1)
- No single day > 30% of total PnL

### Dependencies
Sprint 12 (live prediction system)

---

## Sprint 15: Validation + Ablation Study (3-4 days) [x]

### Objectives
Final validation: ablation study proving each component adds value, stress testing, and documentation.

### Deliverables
- [ ] Ablation study script:
  - Full system vs without microstructure features
  - Full system vs without cross-market (NQ) features
  - Full system vs without regime detection
  - Full system vs without confidence gating
  - Full system vs without macro event module
  - Each component must show positive marginal contribution
- [ ] Historical stress test replay:
  - March 2020 COVID crash
  - May 2021 China ban
  - November 2022 FTX collapse
  - March 2024 ATH correction
- [ ] Synthetic stress tests:
  - 48h zero volatility
  - 10 flash crashes in 1 week
  - BTC-NQ correlation drops to 0
  - Funding rate at +0.3% for 2 weeks
- [ ] Final performance report:
  - Concatenated OOS Sharpe with CI
  - DSR accounting for all trials
  - Regime-specific performance
  - Cost sensitivity analysis
  - Monte Carlo ruin probability

### Acceptance Criteria
- Every component shows positive marginal contribution (or is removed)
- Full system Sharpe > any single-component system
- System survives all stress scenarios within kill switch parameters
- 95% CI lower bound on Sharpe > 0
- P(ruin at 20% drawdown) < 5%
- DSR > 0.95 (Sharpe genuine after multiple testing)

### Key Research Reference
- `BACKTESTING_PLAN.md` - Complete validation methodology
- `research/monte_carlo_validation.md`
- `research/STRESS_TESTING.md`

### Dependencies
Sprint 9-13 (everything)

---

## Sprint 16: Alpha Enhancement — Mega-Research Integration (5-7 days) [ ]

### Objectives
Integrate the highest-ROI findings from the 20-agent deep research investigation (2026-03-23). This sprint focuses on features and upgrades that were validated by academic literature and have concrete expected Sharpe improvements. Only implement what has published evidence; skip speculative additions.

### Key Research Reference
- `research/RR-mega-research-20-agents-consolidated.md` — Complete findings from 20 parallel research agents

### Deliverables

#### T1: Advanced Order Flow Features (Sharpe +0.2-0.5)
Extend `src/ep2_crypto/features/microstructure.py`:
- [ ] **Multi-level OBI**: Compute OBI at levels 1, 5, and 10 separately (uses existing depth@100ms data). Divergence between shallow and deep OBI signals institutional vs retail pressure.
- [ ] **VPIN** (Volume-Synchronized Probability of Informed Trading): Implement via Bulk Volume Classification on aggTrades. Volume bucket = 1/50th daily avg volume. Output: VPIN score 0-1. Use as regime filter (tradeable zone: 0.3-0.6).
- [ ] **Book pressure gradient**: `gradient_asymmetry = bid_gradient / ask_gradient` where gradient = cumulative depth change rate across levels. Predicts 1-bar direction at 54-56%.
- [ ] **Depth withdrawal ratio**: `(depth_N_bars_ago - current_depth) / depth_N_bars_ago` without corresponding trades. Market maker pull-away signal, ~58% accuracy at 1-min.
- [ ] Tests with golden datasets for each new feature
- References: Cont, Kukanov & Stoikov (2014); Easley, Lopez de Prado & O'Hara (2012); Kolm et al. (2023)

#### T2: Cross-Exchange Signals (Sharpe +0.1-0.3)
Add new data source and extend `src/ep2_crypto/features/cross_market.py`:
- [ ] **Coinbase WebSocket integration** via ccxt: Add BTC/USD ticker stream from Coinbase
- [ ] **Coinbase premium features**: `premium_raw`, `premium_zscore_60`, `premium_delta_6` (change over 30 min). IC 0.03-0.07 at 30-min horizon. Only meaningful during US hours (14:00-21:00 UTC).
- [ ] **Cross-exchange OFI divergence**: When Binance OFI and Coinbase OFI disagree, the exchange with highest volume "wins" 58-63% of the time. Feature: `ofi_consensus` (binary: agree/disagree).
- [ ] **ETH order flow features**: Add ETH/USDT perpetual aggTrades from Binance WS. Compute ETH net taker volume and ETH OFI. ETH order flow leads BTC by 1-5 min with 54-57% accuracy.
- [ ] **ETH/BTC ratio rate-of-change** at 5-min and 15-min lookback. Asymmetric: stronger for downside moves (55-58% accuracy on drops).
- [ ] **Binance Long/Short Ratio**: Poll `/futures/data/topLongShortPositionRatio` every 5 min. Contrarian at extremes (>2.5 or <0.5).
- [ ] Tests for each new cross-exchange feature
- References: Makarov & Schoar (2020); Alexander & Heck (2020); Augustin et al. (2022)

#### T3: Twelve Data Upgrade for NQ/DXY (Sharpe +0.1-0.2)
- [ ] **Replace yfinance with Twelve Data** ($29/mo) for NQ, DXY, Gold intraday data
- [ ] 1-min delay vs current 15-min delay — critical because NQ leads BTC by only 5-15 min
- [ ] Update `src/ep2_crypto/ingest/cross_market.py` with Twelve Data WebSocket/REST
- [ ] Fallback to yfinance if Twelve Data unavailable

#### T4: HAR-RV Multi-Scale Volatility (Sharpe +0.1-0.2)
Extend `src/ep2_crypto/features/volatility.py`:
- [ ] **HAR-RV components**: Realized volatility at 1h (12 bars), 4h (48 bars), 1d (288 bars), 1w (2016 bars)
- [ ] **Ratio features**: `RV_12/RV_288`, `RV_48/RV_288` — capture volatility term structure shifts that precede regime changes
- [ ] Outperforms single-scale GARCH by 5-10% RMSE for volatility forecasting
- [ ] Tests with known volatility scenarios
- Reference: Corsi (2009) HAR-RV model

#### T5: Confidence Gating Upgrades (Sharpe improvement on existing pipeline)
Upgrade `src/ep2_crypto/confidence/conformal.py`:
- [ ] **Adaptive Conformal Inference (ACI)**: Adjust alpha_t online based on recent coverage. Reduces interval width by 20-30% while maintaining coverage guarantee. Handles crypto non-stationarity.
- [ ] **Conformalized Quantile Regression (CQR)**: Produces intervals that are narrower in low-vol and wider in high-vol regimes. Replace standard conformal with CQR.
- [ ] **Weighted conformal scores**: Recent observations get higher weight — critical for non-stationary crypto.
- [ ] Calibration tests: verify 90% coverage maintained across regimes
- References: Gibbs & Candes (2024); Romano et al. CQR; Barber et al. (2023-2024)

#### T6: Multi-Task GRU Training (Sharpe +0.1-0.2)
Upgrade `src/ep2_crypto/models/gru_features.py`:
- [ ] Add **auxiliary prediction heads** during training: next-bar volatility, next-bar volume
- [ ] Use **Uncertain Weighting** (Kendall et al.) to auto-balance task losses
- [ ] Discard auxiliary heads at inference — only keep enriched hidden states
- [ ] Hidden states fed to LightGBM encode richer market state information
- [ ] Compare hidden state quality: single-task vs multi-task via downstream LightGBM accuracy
- Reference: Zhang & Zhong (2024), Sawhney et al. (2024)

#### T7: Enhanced Liquidation Cascade Detection (Sharpe +0.3-0.5 as risk filter)
Upgrade `src/ep2_crypto/events/cascade.py`:
- [ ] **Bybit allLiquidation.BTCUSDT** (upgraded Feb 2025): Reports ALL liquidations at 500ms, not just 1/sec. Add to ingest layer.
- [ ] **Hawkes process with online branching ratio estimation**: Normal: 0.3-0.5, pre-cascade: 0.7-0.85, cascade: >0.9. Use recursive intensity: `R(t_n) = exp(-beta*(t_n - t_{n-1})) * (1 + R(t_{n-1}))`.
- [ ] **State-dependent amplifier**: Inverse of order book depth — thinner book = more excitation per liquidation.
- [ ] **Cascade probability score**: logistic combination of branching ratio, OI percentile, funding z-score, book thinning rate, price velocity.
- [ ] Action: cascade_probability > 0.7 → reduce position to 25% or halt.
- [ ] Optional: **Coinglass API** ($79/mo) for pre-computed liquidation heatmaps and aggregated multi-exchange data.
- [ ] Tests: replay known cascade events (FTX Nov 2022, Oct 2025)
- References: Atak (2020); Ali (2025 SSRN)

#### T8: Deribit Options Data (Sharpe +0.05-0.15)
Add new data source:
- [ ] **Deribit WebSocket integration** via ccxt: IV surface, options OI, block trades
- [ ] **25-delta risk reversal rate of change** (1h rolling): Directional signal, ~52-53% at 5-min but better as regime context
- [ ] **ATM IV rate of change** (1h rolling): Volatility regime signal
- [ ] **Put/call OI ratio**: Low cost aggregation from Deribit OI
- [ ] **Max pain distance**: Calendar feature for large quarterly/monthly expirations (62% convergence for quarterly)
- [ ] **Deribit quarterly basis** as CME proxy: Basis momentum IC 0.02-0.05
- References: Alexander, Choi, Park, Sohn (2020); Cao, Chen, Griffin (2023); Foley, Karlsen, Putnins (2023)

### Acceptance Criteria
- All new features pass look-ahead bias tests (truncation + shuffle)
- Each feature has golden dataset tests with hand-verified values
- Feature count stays within 30-40 total (prune weakest existing features if needed)
- Ablation study: each T1-T8 ticket shows positive marginal Sharpe contribution
- Walk-forward backtest with all enhancements shows Sharpe improvement > 0.3 over Sprint 15 baseline
- No new external API dependency causes system failure (graceful degradation if Twelve Data, Deribit, or Coinglass are unavailable)
- Total monthly data cost increase: <$120 (Twelve Data $29 + optional Coinglass $79)
- All new features compute in < 5ms per bar total (maintain <100ms feature budget)

### What Was Investigated But NOT Included (Confirmed Skip)
These were researched by the 20 agents and determined to be not useful at 5-min horizon:
- Stablecoin mint/burn (30min-24h latency)
- Whale wallet tracking (10+ min blockchain latency, 40-70% false positives)
- ETF flows (T+1 daily data)
- Google Trends / Wikipedia (daily granularity)
- Social sentiment NLP (0 predictive power at 5-min)
- Graph Neural Networks (blockchain latency blocker)
- Reinforcement Learning (supervised wins; 60-85% sim-to-real gap)
- Prediction markets (lag spot at 5-min)
- News NLP (+1-2% doesn't justify complexity)
- Transformer/foundation models (LightGBM wins on tabular)
- Dark web, job postings, app downloads, energy prices, Lightning Network

### Dependencies
Sprint 15 (Validation — need baseline metrics to measure improvement against)

---

## Sprint 17: Venue Abstraction + Polymarket Execution (3-4 days) [x]

### Objectives
Build a venue-agnostic execution layer so the ML pipeline (features → models → gating → risk) can trade on **both** Binance perpetual futures and Polymarket's 5-minute BTC prediction markets. The core ML stack stays untouched — only the execution, position sizing, and backtest layers gain a new adapter.

### Why Polymarket
- 5-minute "Bitcoin Up or Down" markets resolve via Chainlink Data Streams — a natural fit for our directional model
- **Maker fee = 0%**, hold-to-resolution exit = 0% — break-even accuracy is ~50.5% (vs ~52% on Binance perps)
- Binary payout: known max loss = premium paid, no liquidation risk, no margin
- Quarter-Kelly on binary outcomes: `f* = 0.25 * (q - p) / (1 - p)` where q = model probability, p = market price
- ROI potential is high but absolute scale is limited (~$200-$2K per trade due to thin orderbooks)

### Design Principles
- **Venue as a swappable adapter** — ML pipeline produces `GatingResult` (direction + confidence), risk manager produces `TradeDecision`, venue adapter executes
- **No changes to existing Binance-oriented code** — new modules only, existing interfaces preserved
- **Binary-aware position sizing** — separate `BinaryPositionSizer` for Polymarket's payout structure
- **Hold-to-resolution as primary strategy** — avoids spread cost on exit; model predicts 5-min direction which maps directly to market resolution

### Deliverables

#### T1: VenueAdapter ABC + venue registry
- [ ] `src/ep2_crypto/execution/venue.py`:
  - `VenueAdapter` abstract base class: `place_order()`, `cancel_order()`, `get_position()`, `get_orderbook()`, `get_balance()`
  - `VenueType` enum: `BINANCE_PERPS`, `POLYMARKET_BINARY`
  - `OrderResult` dataclass: fill_price, fill_qty, fees, status
  - `VenueConfig` Pydantic model: venue-specific configuration
- [ ] `src/ep2_crypto/execution/registry.py`:
  - `VenueRegistry`: register/get adapters by type
- [ ] Tests for ABC contract enforcement

#### T2: PolymarketAdapter implementation
- [ ] `src/ep2_crypto/execution/polymarket.py`:
  - Market discovery: deterministic slug `btc-updown-5m-{floor(now/300)*300}` via Gamma API
  - Order placement via `py-clob-client`: limit (maker, post-only) and market (FOK) orders
  - WebSocket orderbook subscription with 5s heartbeat (10s required, 5s safety margin)
  - Position tracking: current shares held, unrealized value
  - Hold-to-resolution: detect window end, await resolution, claim payout
  - Token approval management (3 contracts)
  - Connection health monitoring + auto-reconnect
- [ ] `src/ep2_crypto/execution/polymarket_config.py`:
  - `PolymarketConfig` Pydantic model: private_key (from env), funder_address, chain_id, fee_rate, min_order_size
  - **Private key via env var only** — never in code or config files
- [ ] Tests with mocked py-clob-client

#### T3: BinaryPositionSizer
- [ ] `src/ep2_crypto/risk/binary_position_sizer.py`:
  - Kelly for binary outcomes: `f* = (q - p) / (1 - p)` where q = model confidence, p = market price
  - Fee-adjusted Kelly: accounts for Polymarket's profit-only fee structure
  - Quarter-Kelly default (configurable fraction)
  - Max bet cap: configurable % of bankroll (default 5%)
  - Min bet size: $5 (Polymarket minimum)
  - Drawdown gate integration: multiply by drawdown_multiplier from existing gate
  - Output: `BinarySizingResult(shares, cost_usd, max_loss_usd, expected_value, kelly_fraction)`
- [ ] Tests with golden dataset: known (q, p, bankroll, fee) → expected (shares, cost)
- [ ] Property-based test: bet size NEVER exceeds max cap (500+ random inputs)

#### T4: PolymarketRiskAdapter
- [ ] `src/ep2_crypto/risk/polymarket_risk.py`:
  - Wraps existing `RiskManager` for binary context
  - Maps `GatingResult` → `SignalInput` with binary-appropriate fields
  - Translates `TradeDecision` → Polymarket order parameters
  - Kill switches reused as-is (daily loss, weekly loss, max DD, consecutive losses)
  - Drawdown gate reused as-is (convex reduction applied to bet size)
  - **No margin tracking** — max loss is known upfront (= cost of shares)
  - **No stop-loss** — binary resolution handles exit
  - **No holding period limit** — market resolves in 5 min max
  - PnL tracking: record each resolution outcome for kill switch thresholds
- [ ] Tests: normal trade, kill switch trigger, drawdown gate reduction, consecutive losses

#### T5: PolymarketBacktester
- [ ] `src/ep2_crypto/backtest/polymarket_backtest.py`:
  - Simulates binary payoff: buy at market price p, receive $1 if correct direction, $0 if wrong
  - Fee model: Polymarket's `fee = C * p * feeRate * (p*(1-p))^exponent` for takers; 0 for makers
  - Spread simulation: configurable bid-ask spread (default 3-5 cents)
  - Per-window P&L: `pnl = (1 - p - fee) if correct, else (-p)` per share
  - Uses existing model predictions as signal source
  - Outputs: accuracy, Sharpe, profit factor, max DD, avg trade PnL, total P&L
  - Comparison report: Polymarket P&L vs Binance perps P&L on same signals
- [ ] Tests with synthetic signals at known accuracy levels

#### T6: Integration + live loop hookup
- [ ] Update `scripts/live.py` to support venue selection via config
- [ ] `src/ep2_crypto/execution/__init__.py` — clean exports
- [ ] Integration test: full pipeline mock (gating → risk → polymarket adapter → resolution)
- [ ] Add `py-clob-client` to `pyproject.toml` dependencies

### Key Technical Details
- **py-clob-client SDK**: `pip install py-clob-client` (Python 3.9+)
- **Market slug**: `btc-updown-5m-{floor(unix_ts/300)*300}` — deterministic, no search needed
- **Token IDs**: `clobTokenIds[0]` = YES/Up, `clobTokenIds[1]` = NO/Down
- **Heartbeat**: send `"PING"` every 5s on WebSocket, expect `"PONG"`
- **Resolution**: Chainlink Data Streams BTC/USD — this is truth, not exchange price
- **Order types**: GTC (maker), FOK (taker), with `post_only=True` for guaranteed maker
- **Auth**: derive API creds from Polygon wallet private key, no manual registration
- **Min order**: ~5 shares ($2.50 at 50c)
- **Known SDK issues**: neg_risk flag bugs, tick size caching, signature edge cases — code defensively

### Risk Parameters (Polymarket-specific)
| Parameter | Value |
|-----------|-------|
| Max bet per window | 5% of bankroll |
| Quarter-Kelly fraction | 0.25 |
| Min bet size | $5 |
| Daily loss limit | 2% of bankroll |
| Weekly loss limit | 5% of bankroll |
| Max drawdown halt | 15% of bankroll |
| Consecutive loss halt | 10 |
| Strategy | Maker entry + hold-to-resolution (0% fees) |
| Fallback | FOK taker if maker not filled within 60s of window open |

### Acceptance Criteria
- VenueAdapter ABC enforced — both Polymarket and (future) Binance adapters implement same interface
- Polymarket adapter discovers current 5-min market within 1 second
- Orders placed and tracked through resolution
- Binary position sizer never exceeds max bet cap (property-based test, 500+ inputs)
- Kill switches halt trading at exact thresholds (reusing existing tests)
- Backtest on historical BTC 5-min directions shows Sharpe > 1.0 with maker strategy at 54% accuracy
- Full pipeline integration test passes: signal → risk → order → resolution → PnL update
- No changes to any existing module (features, models, gating, risk core, backtest core)
- Private key loaded from environment variable only, never hardcoded

### Dependencies
Sprint 8 (confidence gating), Sprint 9 (risk engine)

---

## Sprint 18: Historical Data Backfill (1-2 days) [~]

### Objectives
Collect all available historical data needed to train, validate, and backtest the full system. BTC data goes back to Binance Futures launch (Sep 2019) for maximum regime coverage. Polymarket data covers everything since 5-min markets launched (~Feb 2026), plus synthetic outcomes derived from BTC OHLCV for pre-Polymarket backtesting.

### Why Full History
- 2 years of data only covers 1-2 regimes. 6+ years covers COVID crash (Mar 2020), China ban (May 2021), FTX collapse (Nov 2022), 2024 ATH, and 2025-26 conditions.
- More data = more walk-forward folds = higher confidence that the model generalizes.
- Synthetic Polymarket outcomes from OHLCV let us estimate binary accuracy going back to 2019.

### Deliverables

#### T1: BTC Historical Data Backfill
- [ ] `scripts/collect_history.py`:
  - Binance Futures REST: 1m OHLCV klines BTCUSDT (Sep 2019 → now, ~3.1M candles)
  - Binance Futures REST: funding rate history (every 8h)
  - Bybit REST: open interest 5-min history
  - yfinance: NQ, Gold, DXY, ETH daily OHLCV (full history)
  - Storage: SQLite (dev) or TimescaleDB (prod) via `EP2_DB_URL` env var
  - CLI: `--start 2019-09-01 --end today`, `--resume` flag to continue from last stored timestamp
  - Rate limiting, exponential backoff, checkpoint every 10K rows
  - Upsert (INSERT OR REPLACE) for idempotent re-runs
  - Progress bar (tqdm) + summary at end
- [ ] Tests for data integrity (no gaps > 5 min, correct OHLC ranges)

#### T2: Polymarket Historical Data Backfill
- [ ] `scripts/collect_polymarket_history.py`:
  - Gamma API: all resolved 5-min BTC Up/Down markets (Feb 2026 → now, ~45 days)
  - Store: window_ts, slug, outcome (up/down), yes/no prices, volume, condition_id
  - `--derive-from-btc` flag: generate synthetic outcomes from ohlcv_1m for pre-Polymarket dates (2019-2026)
  - CLI: `--start 2026-02-01`, `--resume`, `--derive-from-btc --start 2019-09-01`
  - Rate limiting (Gamma API: 300 req/10s)
- [ ] Tests for synthetic derivation correctness

### Acceptance Criteria
- OHLCV table: 3M+ rows covering Sep 2019 → present with no gaps > 10 minutes
- Funding rate: 9K+ rows
- Open interest: 600K+ rows
- Cross-market: 4 symbols × 2400+ days
- Polymarket real: all available resolved 5-min windows
- Polymarket synthetic: 6+ years of derived outcomes matching BTC direction
- `--resume` works correctly after interruption (verified by stopping mid-run and restarting)
- Same scripts work against both SQLite (dev) and TimescaleDB (prod) via connection string

### Dependencies
Sprint 1 (database schema). No other dependencies — can run in parallel with everything.

---

## Sprint 19: Production Deployment (2-3 days) [ ]

### Objectives
Deploy the complete system to Hetzner VPS with Docker Compose, TimescaleDB, monitoring (Prometheus + Grafana), Telegram alerts, and Doppler secrets management. The system should run continuously, auto-restart on failure, and be deployable with a single command.

### Infrastructure Architecture
```
Hetzner VPS
├── Docker Compose
│   ├── ep2-crypto (main service)
│   │   ├── Live data collectors (Binance WS — public, no API key)
│   │   ├── Feature pipeline (25-35 features)
│   │   ├── Model inference (ONNX Runtime)
│   │   ├── Confidence gating (7-gate pipeline)
│   │   ├── Risk engine (PolymarketRiskAdapter)
│   │   ├── Polymarket execution (py-clob-client)
│   │   └── FastAPI (health, metrics, predictions)
│   ├── TimescaleDB (replaces SQLite for production)
│   ├── Prometheus (scrapes /metrics every 15s)
│   └── Grafana (dashboards: PnL, accuracy, risk state, alpha decay)
├── Doppler (prd config → secrets injected at runtime)
└── Systemd (ensures docker-compose auto-starts on boot)
```

### Secrets (Doppler project: ep2-crypto, config: prd)
- `EP2_POLYMARKET_PRIVATE_KEY` — wallet private key (already stored)
- `EP2_POLYMARKET_WALLET_ADDRESS` — wallet address (already stored)
- `TELEGRAM_BOT_TOKEN` — alert bot (already stored)
- `TELEGRAM_CHAT_ID` — destination chat for alerts (to be configured)
- `EP2_DB_URL` — TimescaleDB connection string
- Future: `BINANCE_API_KEY`, `BYBIT_API_KEY` (if trading on Binance)

### Deliverables

#### T1: Docker containerization
- [ ] `docker/Dockerfile`:
  - Python 3.12-slim base, uv for deps
  - `uv sync --frozen --extra polymarket`
  - ONNX Runtime for inference
  - Doppler CLI installed
  - Healthcheck: `curl localhost:8000/health`
  - CMD: `doppler run -- uv run python scripts/live.py --venue polymarket_binary`
- [ ] `docker/.dockerignore`: exclude .venv, __pycache__, .git, data/, research/, *.db
- [ ] `.env.example`: documents required DOPPLER_TOKEN

#### T2: Docker Compose stack
- [ ] `docker/docker-compose.yml`:
  - ep2-crypto service (main app, port 8000)
  - TimescaleDB (timescale/timescaledb:latest-pg16, port 5432 localhost only)
  - Prometheus (port 9090 localhost only)
  - Grafana (port 3000)
  - Named volumes for persistence (pgdata, prometheus_data, grafana_data)
  - Health checks on all services
  - `restart: unless-stopped` on all services
- [ ] `docker/prometheus.yml`: scrape config for ep2-crypto /metrics

#### T3: Deploy script
- [ ] `scripts/deploy.sh`:
  - rsync code to Hetzner (exclude .venv, .git, data/)
  - docker compose build + up -d
  - Health check with 60s timeout
  - Rollback on failure (show logs)
  - Usage: `./scripts/deploy.sh`
- [ ] `scripts/setup_server.sh`:
  - One-time server setup: install Docker, Doppler CLI, create deploy user
  - Systemd service to auto-start docker-compose on boot
  - UFW firewall: allow 22, 8000, 3000

#### T4: Telegram alert configuration
- [ ] Configure `TELEGRAM_CHAT_ID` in Doppler
- [ ] Verify alerts fire: test INFO (daily summary), WARNING (alpha decay), CRITICAL (kill switch)
- [ ] Rate limiting: max 1 INFO/hour, 5 WARNING/hour, unlimited CRITICAL

#### T5: Grafana dashboards
- [ ] `docker/grafana/dashboards/`:
  - Trading dashboard: PnL (daily, cumulative), trade count, win rate, Sharpe rolling 30d
  - Risk dashboard: drawdown, kill switch status, daily loss budget, exposure
  - Model dashboard: confidence distribution, feature drift PSI, prediction accuracy
  - System dashboard: latency, memory, API response times

#### T6: Data migration (SQLite → TimescaleDB)
- [ ] Migration script: export SQLite historical data → import into TimescaleDB
- [ ] Verify: `scripts/collect_history.py --resume` works against TimescaleDB
- [ ] Create TimescaleDB hypertables with time-based partitioning on timestamp columns

### Acceptance Criteria
- `docker compose up -d` brings up all 4 services within 2 minutes
- Health endpoint at :8000/health returns OK with all dependency checks passing
- Prometheus scrapes metrics successfully (verify at :9090/targets)
- Grafana dashboards load with real data (verify at :3000)
- Telegram bot sends test alert within 30 seconds of trigger
- System survives reboot (systemd restarts docker-compose)
- `scripts/deploy.sh` deploys new version in < 5 minutes
- No secrets in Docker images, compose files, or git history
- ep2-crypto container restarts automatically after crash
- TimescaleDB data survives container restart (persistent volume)
- Backfill scripts work against TimescaleDB with same CLI interface

### Dependencies
Sprint 13 (API + monitoring), Sprint 17 (Polymarket execution), Sprint 18 (historical data)

---

## Sprint Dependencies Graph

```
Sprint 1 (Foundation)
  |
  +---> Sprint 2 (Exchange Data) ----+
  |                                   |
  +---> Sprint 3 (Microstructure) ---+---> Sprint 5 (Cross-mkt, Regime, Temporal)
  |                                   |         |
  +---> Sprint 4 (Vol, Mom, Volume) -+         |
                                               |
                                    Sprint 6 (Regime Detection)
                                               |
                                    Sprint 7 (Models)
                                               |
                                    Sprint 8 (Confidence Gating)
                                               |
                              +----------------+----------------+
                              |                |                 |
                    Sprint 9 (Risk Mgmt)  Sprint 12 (Macro)  Sprint 17 (Polymarket)
                              |                                  |
                    Sprint 10 (Backtesting)            Sprint 18 (Data Backfill) ←── parallelizable
                              |                                  |
                    Sprint 11 (Tuning)                 Sprint 19 (Deployment)
                              |                                  |
                    Sprint 13 (API + Live) ─────────────────────→+
                              |
                    Sprint 14 (Paper Trading)
                              |
                    Sprint 15 (Validation)
                              |
                    Sprint 16 (Alpha Enhancement)
```
