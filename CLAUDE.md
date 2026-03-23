# ep2-crypto: 5-Minute BTC Prediction Engine

## Project Overview

ep2-crypto is a multi-edge Bitcoin price prediction system targeting 5-minute directional forecasts on BTC/USDT perpetual futures. It combines order flow microstructure, cross-market lead-lag, regime-adaptive modeling, and confidence gating to achieve a tradeable edge.

**Realistic expectations** (from 500K+ words of research):
- Directional accuracy: 52-56% at 5-min horizon
- Backtest-to-live Sharpe degradation: 50-70% (need Sharpe >2.0 in backtest to survive)
- Round-trip costs: 8-12 bps minimum
- Alpha half-life: 1-6 months for ML patterns
- Optimal feature count: 18-25 (not 200+)

## Tech Stack

| Component | Choice | Why |
|-----------|--------|-----|
| Language | Python 3.12+ | ML ecosystem, async support |
| Package manager | uv | Fast, deterministic (never pip/poetry) |
| Primary model | LightGBM | Outperforms XGBoost by 0.5-1.5% on 5-min crypto |
| Secondary model | CatBoost | Ordered boosting prevents leakage, adds diversity |
| Sequence model | GRU (PyTorch) | Hidden state extracted as features for LightGBM |
| Ensemble | Stacking (logistic regression meta-learner) | 2-5% over best single model |
| Regime detection | Hierarchical (HMM + BOCPD + GARCH + Efficiency Ratio) | Faster detection than HMM alone |
| Exchange data | ccxt (WebSocket + REST) | Binance WS for klines/depth/trades, Bybit for derivatives |
| Cross-market | yfinance, Twelve Data | NQ, Gold, DXY (15-min delayed, acceptable) |
| On-chain | mempool.space WS | Whale detection only (on-chain operates at 4h+, not 5-min) |
| Logging | structlog | Structured JSON logs, never print() |
| API | FastAPI + uvicorn | Prediction endpoints, health, metrics |
| Database | SQLite (dev) -> TimescaleDB (prod) | Zero setup for dev |
| Inference | ONNX Runtime | 2-5x speedup over native |
| Tuning | Optuna | MedianPruner, Sharpe objective |
| Testing | pytest + pytest-cov | Required for every feature computation |
| Linting | ruff + mypy (strict) | Type safety, code quality |

## Key Architecture Decisions

These decisions are the result of extensive research (49 files in `research/`). Do NOT change them without reviewing the corresponding ADR in `DECISIONS.md`.

### Model Architecture
- **LightGBM over XGBoost** (ADR-001): Faster training, better accuracy on crypto tick data
- **Stacking ensemble over per-regime models** (ADR-006): Per-regime models fragment scarce data. Single ensemble with regime as input feature is more robust
- **Ternary target (up/flat/down) with adaptive threshold** (ADR-002): Acknowledges untradeable low-volatility periods. Threshold adapts per regime
- **Triple barrier labeling over fixed threshold** (ADR-008): Accounts for time decay and stop-loss, produces more realistic labels

### Data & Features
- **Dual normalization pipeline** (ADR in RESEARCH_SYNTHESIS.md): Raw features for tree models (normalization harms trees), robust scaling + rank-to-gaussian for neural models
- **On-chain as regime context only, not primary signal** (ADR-004): On-chain signals operate at 4h-24h, not 5-min
- **Skip real-time sentiment, use funding rate** (ADR-005): NLP sentiment too slow for 5-min. Funding rate is the only useful "sentiment" at this timeframe
- **Feature priority**: Microstructure (OBI, OFI, microprice) captures 60-80% of achievable 5-min signal

### Validation & Risk
- **Purged walk-forward validation**: 14-day sliding window, 18-bar purge (90 min), 12-bar embargo (60 min). Sliding, NOT expanding (crypto non-stationarity)
- **Confidence gating pipeline**: isotonic calibration -> meta-labeling -> conformal prediction -> adaptive threshold -> drawdown gate. This is the single biggest Sharpe multiplier (2-4x)
- **Quarter-Kelly position sizing**: Position size = 0.25 * Kelly fraction * confidence score
- **Annualization**: sqrt(105,120) for 24/7 crypto. 288 bars/day * 365 days = 105,120 bars/year. NEVER use sqrt(252)

### Infrastructure
- **asyncio single-process architecture** (ADR-010): All collectors and processing in one event loop
- **Event-driven backtest engine with Numba** (ADR-009): 100x faster than pure Python, realistic fills
- **Numpy ring buffers** for data structures: 10-20x less memory than pandas DataFrames on hot path

## Code Conventions

### Mandatory Rules
1. **Named exports only** - never use `export default` or bare module-level assignments
2. **structlog for all logging** - never use `print()` for any output
3. **Parameterized SQL queries only** - never use f-strings or string interpolation for SQL
4. **All features must use only past data** - no look-ahead bias. Every feature computation at time t may only use data from times <= t
5. **Type hints on all functions** - mypy strict mode is enforced
6. **Tests required for every feature computation** - including golden dataset tests against hand-verified values
7. **No `any` types** - be explicit with all type annotations
8. **No TODO/FIXME/HACK without linked issue** - if you must leave one, create a GitHub issue first
9. **Secrets via environment variables** - never commit API keys, never use .env files in repo

### Error Handling
- Always use try/catch with meaningful error messages at system boundaries
- Never swallow errors silently - log with context via structlog
- Never fire-and-forget async operations - track completion
- Health endpoints must check actual dependencies, not just return "ok"

### Feature Engineering Rules
- Every feature must be tested for look-ahead bias (shuffle test + truncation test)
- Rolling calculations use only past data (no centered windows)
- NaN handling: forward-fill for price data, zero for counts, drop for model input
- All normalization fitted on training data only, per fold

### Commit Hygiene
- Meaningful commit messages explaining WHY, not just WHAT
- Pass lint + typecheck + test before committing
- One logical change per commit, max 5 commits per feature
- Build one feature, verify, commit - never batch 5 features

## Project Structure

```
ep2-crypto/
├── CLAUDE.md                    # THIS FILE - project context for AI sessions
├── PLAN.md                      # Original development plan (Phase 0-11)
├── RESEARCH_SYNTHESIS.md        # Distilled findings from 49 research files
├── BACKTESTING_PLAN.md          # Comprehensive backtesting methodology
├── SPRINTS.md                   # Sprint planning with acceptance criteria
├── REQUIREMENTS.md              # Functional and non-functional requirements
├── DECISIONS.md                 # Architecture Decision Records (ADR-001 to ADR-010)
├── pyproject.toml               # Dependencies and tool configuration
├── uv.lock                      # Locked dependencies
├── research/                    # 49 research files (read-only reference)
│   ├── meta_labeling_deep_research.md
│   ├── microstructure_ofi_microprice.md
│   ├── monte_carlo_validation.md
│   ├── retraining_strategies.md
│   ├── statistical_validation_tests.py
│   ├── STRESS_TESTING.md
│   ├── transaction_cost_modeling.md
│   └── ... (42 more research files)
├── src/
│   └── ep2_crypto/
│       ├── __init__.py
│       ├── config.py                  # Pydantic Settings, env-var based
│       ├── db/
│       │   ├── schema.py              # SQLite/TimescaleDB schema
│       │   └── repository.py          # Parameterized query layer
│       ├── ingest/
│       │   ├── base.py                # Base collector interface
│       │   ├── exchange.py            # Binance WS: klines, depth@100ms, aggTrades
│       │   ├── derivatives.py         # Bybit: OI, funding, liquidations
│       │   ├── onchain.py             # mempool.space: whale detection
│       │   ├── cross_market.py        # NQ, Gold, DXY, ETH
│       │   ├── macro.py               # Economic calendar, event monitoring
│       │   └── orchestrator.py        # Async task management, reconnection
│       ├── features/
│       │   ├── base.py                # Feature pipeline interface
│       │   ├── microstructure.py      # OBI, OFI, microprice, spread, TFI
│       │   ├── volume.py              # Volume delta, VWAP deviation, rate-of-change
│       │   ├── volatility.py          # Realized, Parkinson, EWMA, vol-of-vol
│       │   ├── momentum.py            # ROC, RSI, linreg slope, quantile rank
│       │   ├── cross_market.py        # Lead-lag, correlation, divergence
│       │   ├── temporal.py            # Cyclical encoding, session, time-to-funding
│       │   ├── regime_features.py     # ER, GARCH vol, HMM probs as features
│       │   ├── normalization.py       # Dual pipeline (raw for trees, robust for neural)
│       │   └── pipeline.py            # Combines all feature sets
│       ├── regime/
│       │   ├── detector.py            # Hierarchical regime detection
│       │   ├── hmm.py                 # 2-state GaussianHMM
│       │   ├── bocpd.py               # Bayesian Online Change Point Detection
│       │   ├── garch.py               # GJR-GARCH(1,1)-t conditional volatility
│       │   └── efficiency_ratio.py    # Kaufman Efficiency Ratio
│       ├── models/
│       │   ├── lgbm_direction.py      # LightGBM ternary classifier
│       │   ├── catboost_direction.py  # CatBoost ternary classifier
│       │   ├── gru_features.py        # GRU hidden state extractor
│       │   ├── quantile.py            # LightGBM quantile regression (5 quantiles)
│       │   ├── stacking.py            # Logistic regression meta-learner
│       │   ├── calibration.py         # Isotonic regression calibration
│       │   └── tuning.py              # Optuna hyperparameter optimization
│       ├── confidence/
│       │   ├── meta_labeling.py       # Meta-labeling gate
│       │   ├── conformal.py           # Conformal prediction gate
│       │   ├── gating.py              # Full confidence gating pipeline
│       │   └── position_sizing.py     # Quarter-Kelly sizing
│       ├── backtest/
│       │   ├── walk_forward.py        # Purged walk-forward with embargo
│       │   ├── engine.py              # Event-driven backtest engine (Numba)
│       │   ├── cost_engine.py         # Transaction cost modeling (EXISTS)
│       │   ├── metrics.py             # Sharpe, drawdown, profit factor, etc.
│       │   └── simulator.py           # Execution simulator with slippage
│       ├── benchmarks/               # ALREADY EXISTS with code
│       │   ├── __init__.py
│       │   ├── data.py
│       │   ├── engine.py
│       │   ├── metrics.py
│       │   ├── statistics.py
│       │   └── strategies.py
│       ├── risk/                      # SPRINT 9 — THE MOST CRITICAL MODULE
│       │   ├── config.py              # RiskConfig Pydantic model (all params)
│       │   ├── position_tracker.py    # Position state, MTM, margin, liquidation, reconciliation
│       │   ├── kill_switches.py       # 5 switches: daily/weekly/DD/consecutive/emergency (persist to disk)
│       │   ├── drawdown_gate.py       # Convex reduction (k=1.5) + duration-based + graduated recovery
│       │   ├── volatility_guard.py    # Min/max vol, trading hours, weekend, funding proximity
│       │   ├── position_sizer.py      # Full chain: Kelly × cap × heat × budget × weekend × DD × regime
│       │   └── risk_manager.py        # Orchestrator: approve_trade(), on_bar(), emergency, reset
│       ├── events/
│       │   ├── macro.py               # Event-driven macro module (CPI/FOMC)
│       │   └── cascade.py             # Liquidation cascade detector (Hawkes)
│       ├── monitoring/
│       │   ├── alpha_decay.py         # CUSUM, rolling Sharpe, ADWIN
│       │   └── drift.py               # Feature drift detection (PSI)
│       └── api/
│           └── server.py              # FastAPI endpoints
├── scripts/
│   ├── collect_history.py             # Backfill historical data
│   ├── train.py                       # Full training pipeline
│   ├── backtest.py                    # Run backtests
│   └── live.py                        # Live prediction loop
├── tests/
│   ├── test_features/                 # Feature computation tests
│   ├── test_models/                   # Model training/inference tests
│   ├── test_ingest/                   # Data collection tests
│   ├── test_backtest/                 # Backtesting validation tests
│   └── test_confidence/               # Gating pipeline tests
└── docker/
    ├── Dockerfile
    └── docker-compose.yml
```

## How to Run

### Setup
```bash
# Install dependencies
uv sync --all-extras

# Verify installation
uv run python -c "import ep2_crypto"
```

### Tests
```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=ep2_crypto --cov-report=term-missing

# Run specific test module
uv run pytest tests/test_features/
```

### Linting and Type Checking
```bash
# Lint
uv run ruff check src/ tests/

# Format
uv run ruff format src/ tests/

# Type check
uv run mypy src/ep2_crypto/
```

### Data Collection
```bash
# Backfill historical data (60 days OHLCV, OI, funding)
uv run python scripts/collect_history.py

# Start live data collection
uv run python scripts/live.py --collect-only
```

### Training
```bash
# Full training pipeline (walk-forward)
uv run python scripts/train.py

# Hyperparameter tuning
uv run python scripts/train.py --tune --trials 100
```

### Backtesting
```bash
# Run full backtest with statistical validation
uv run python scripts/backtest.py

# Run with specific date range
uv run python scripts/backtest.py --start 2025-01-01 --end 2025-03-01
```

### API Server
```bash
# Start prediction API
uv run uvicorn ep2_crypto.api.server:app --host 0.0.0.0 --port 8000
```

## Sprint Process

Development follows 14 sprints defined in `SPRINTS.md`. Each sprint has:
- Clear objectives and deliverables
- Acceptance criteria that must pass before moving on
- Key files to create/modify
- Dependencies on previous sprints

**Current sprint progress**: See `SPRINTS.md` for status.

## Key Research References

When implementing a module, consult the relevant research files in `research/`:

| Module | Key Research Files |
|--------|-------------------|
| Microstructure features | `microstructure_ofi_microprice.md` |
| Meta-labeling | `meta_labeling_deep_research.md` |
| Monte Carlo validation | `monte_carlo_validation.md` |
| Stress testing | `STRESS_TESTING.md` |
| Transaction costs | `transaction_cost_modeling.md` |
| Model retraining | `retraining_strategies.md` |
| Statistical validation | `statistical_validation_tests.py` (executable code) |
| Full synthesis | `RESEARCH_SYNTHESIS.md` (root) |
| Backtesting methodology | `BACKTESTING_PLAN.md` (root) |
| Architecture decisions | `DECISIONS.md` (root) |

## Risk Parameters (Starting Configuration)

| Parameter | Value |
|-----------|-------|
| Position size per trade | 4-5% of capital (quarter-Kelly capped) |
| Daily loss limit | 2-3% of capital |
| Weekly loss limit | 5% of capital |
| Max drawdown halt | 15% of capital |
| Catastrophic stop per trade | 3 ATR |
| Max open positions | 1 |
| Max trades per day | 30 |
| Trading hours | 08:00-21:00 UTC (Europe + US sessions) |
| Weekend position sizing | -30% reduction |
| Confidence threshold | 0.60 (starting, adaptive) |
| Min bar volatility | 15% annualized |
| Max bar volatility | 150% annualized |

## Critical Numbers

| Parameter | Value |
|-----------|-------|
| Annualization factor | sqrt(105,120) = 324.22 (NOT sqrt(252)) |
| Bars per year | 288/day * 365 = 105,120 |
| Training window | 14 days (4,032 bars) sliding |
| Test window | 1 day (288 bars) |
| Purge size | 18 bars (90 min) |
| Embargo size | 12 bars (60 min) |
| Target backtest Sharpe | > 2.0 (to survive 50-70% degradation) |
| Break-even cost target | Strategy must survive > 15 bps round-trip |
| Expected live Sharpe | 0.6-1.0 (after degradation) |
| Round-trip cost assumption | 8-12 bps (conservative) |
| **Max risk per trade** | **1% of equity (CRITICAL — Monte Carlo shows 5% risk → 66% DD)** |
| Max position notional | 5% of equity (separate from risk cap) |
| Max risk per trade at 0.5% | E[max_DD] = 9.6% over 1 year (safest) |
| Max risk per trade at 1.0% | E[max_DD] = 18.4% over 1 year (default) |
| Max risk per trade at 5.0% | E[max_DD] = 66.7% over 1 year (ACCOUNT DEATH) |

## Anti-Patterns to Avoid

1. Never random-split time series data
2. Never use `bidirectional=True` on GRU (uses future data)
3. Never use centered rolling windows for features (look-ahead bias)
4. Never trust >70% directional accuracy without checking for look-ahead bias
5. Never use SMOTE or other oversampling on time series
6. Never interpolate missing data (forward-fill or drop only)
7. Never use global normalization (fit on all data) - always per-fold on training data
8. Never use expanding window for training - crypto non-stationarity makes old data harmful
9. Never skip transaction costs in backtest evaluation
10. Never commit without lint + typecheck + test passing
11. **NEVER risk more than 1% of equity per trade** — Monte Carlo proves 5% risk leads to 66% drawdowns over 18K trades/year. The `max_risk_per_trade` parameter in PositionSizer enforces this.

## Research Reference

All research is in `research/` with `research/INDEX.md` as the master lookup.
Files follow `RR-{category}-{topic}.md` naming. Key files per area:

| Need to implement... | Read this first |
|---------------------|-----------------|
| WebSocket data ingestion | `RR-api-ccxt-exchange-integration.md` |
| Order book features (OBI, OFI) | `RR-ofi-microprice-implementation.md` |
| Liquidation cascade detection | `RR-cascade-liquidation-detection-system.md` |
| Cross-market NQ lead-lag | `RR-crossmarket-lead-lag-correlations.md` |
| Regime detection (HMM etc) | `RR-regime-detection-methods.md` |
| LightGBM training | `RR-lightgbm-crypto-tuning.md` |
| GRU/TCN architecture | `RR-gru-tcn-training-architecture.md` |
| Meta-labeling | `RR-metalabeling-implementation-guide.md` |
| Walk-forward validation | `RR-walkforward-validation-methods.md` |
| Backtesting framework | `RR-backtest-framework-design.md` |
| Statistical validation | `RR-stats-edge-vs-luck-tests.md` |
| Transaction costs | `RR-costs-transaction-cost-modeling.md` |
| Paper trading | `RR-papertrade-system-architecture.md` |
| Alpha decay monitoring | `RR-decay-edge-disappearance-detection.md` |
| **Risk engine architecture** | **`RR-risk-engine-architecture.md`** + `RR-risk-implementation-guide.md` |
| **Position sizing (Kelly, ATR)** | **`RR-risk-position-sizing-methods.md`** |
| **Kill switches** | **`RR-risk-kill-switch-design.md`** |
| **Drawdown management** | **`RR-risk-drawdown-management.md`** |
| **Stop loss strategies** | **`RR-risk-stop-loss-strategies.md`** |
| **Margin & liquidation** | **`RR-risk-margin-liquidation.md`** |
| **Tail risk protection** | **`RR-risk-tail-risk-black-swan.md`** + `RR-risk-worst-case-scenarios.md` |
| **Risk testing** | **`RR-risk-testing-framework.md`** |
| **Risk monitoring dashboard** | **`RR-risk-monitoring-dashboard.md`** (49 Prometheus metrics, Grafana panels) |
| **Money management** | **`RR-risk-money-management.md`** |

## Pre-Code Checklists

### Before writing ANY feature computation:
- [ ] Does it use ONLY data at times <= t? (no look-ahead)
- [ ] Is it tested with a truncation test? (compute on data[:150] matches data[:200] at index 150)
- [ ] Does it have a unit test with known input/output?
- [ ] Is it added to the golden dataset for regression testing?
- [ ] Is normalization rolling (not global)?
- [ ] For tree model path: raw value (no normalization)
- [ ] For neural model path: robust scaling or rank-to-gaussian

### Before writing ANY model training code:
- [ ] Is the train/val/test split temporal (never random)?
- [ ] Is early stopping validation carved from training set (not test set)?
- [ ] Is there a purge gap (18 bars) between train and test?
- [ ] Is there an embargo (12 bars) after test?
- [ ] Are sample weights applied (time decay + regime balance)?
- [ ] Is the target ternary with adaptive threshold?

### Before any backtest evaluation:
- [ ] Are transaction costs included (8-12 bps round-trip)?
- [ ] Is execution at next-bar-open (never current-bar-close)?
- [ ] Is latency simulated (200ms minimum)?
- [ ] Are funding rate costs modeled for positions held across settlement?
- [ ] Does the backtest period include at least one bull AND one bear market?

## Session Management

**CRITICAL: Read `PROGRESS.md` at the start of EVERY session.** It has:
- Current sprint and ticket
- What was done in previous sessions
- The EXACT next step to take
- Any handoff notes from the previous session

Full protocol is in `SESSION_PROTOCOL.md`. Key rules:
1. One ticket per session is fine — quality over speed
2. Never read research files unless actively implementing from them (they're 10-50KB)
3. Update PROGRESS.md at end of session with what was done + exact next step
4. If context is getting large, wrap up — don't push through

### End-of-Session Handover

At the END of every session, you MUST:
1. Update PROGRESS.md (what was done, what wasn't, exact next step)
2. Commit code if any was written
3. **Generate a handover prompt** for the user to paste in the next session:

```
Handover prompt for next session:
---
Read PROGRESS.md and continue from ticket S{X}-T{Y}. [1-sentence description of what's next].
Run verification after each ticket. Update PROGRESS.md before ending.
---
```

### Sprint Completion Handover

When a sprint is FULLY complete (all tickets `[x]`, acceptance criteria pass):
1. Commit with message `"Sprint N complete: <summary>"`
2. Decompose the next sprint into tickets in PROGRESS.md
3. **Generate the sprint handover prompt**:

```
Sprint N COMPLETE. Start next session with:
---
Read PROGRESS.md and start Sprint {N+1}. The tickets are decomposed and ready.
Current ticket: S{N+1}-T1. Run verification after each ticket.
Update PROGRESS.md before ending.
---
```

## Sprint Progress Tracking

Current sprint status tracked in `SPRINTS.md` (high-level) and `PROGRESS.md` (ticket-level).
- `[ ]` = not started, `[~]` = in progress, `[x]` = complete
- Each ticket has verification commands — run them before marking complete

## Data Sources (Complete)

Primary: Binance WS (klines, depth, aggTrades), Bybit (OI, funding, liquidations), ccxt (ETH)
Cross-market: yfinance (NQ, Gold, DXY), Coinbase (premium), Deribit (options IV)
Alternative: mempool.space (whales), Alternative.me (Fear&Greed), **Polymarket** (BTC prediction markets)
Context only: on-chain metrics (4h+ timeframe, NOT 5-min signal)
