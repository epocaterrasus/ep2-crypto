# Sprint Planning: ep2-crypto

## Overview

14 sprints covering the full development lifecycle. Each sprint builds on the previous.
Total estimated timeline: 40-52 days of development.

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

## Sprint 3: Feature Engineering - Microstructure (3-4 days) [ ]

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

## Sprint 4: Feature Engineering - Volume, Volatility, Momentum (2-3 days) [ ]

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

## Sprint 5: Feature Engineering - Cross-market, Regime, Temporal (2-3 days) [ ]

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

## Sprint 6: Regime Detection (3-4 days) [ ]

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

## Sprint 7: Models - LightGBM + CatBoost + GRU + Stacking (4-5 days) [ ]

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

## Sprint 8: Confidence Gating Pipeline (3-4 days) [ ]

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

## Sprint 9: Backtesting Framework (4-5 days) [ ]

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
Sprint 7 (models), Sprint 8 (confidence gating)

---

## Sprint 10: Hyperparameter Tuning (2-3 days) [ ]

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

## Sprint 11: Event-Driven Macro Module + Cascade Detector (3-4 days) [ ]

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

## Sprint 12: API + Live Prediction + Monitoring (3-4 days) [ ]

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

### Acceptance Criteria
- API responds in < 100ms
- Health endpoint checks ALL dependencies (not just "ok")
- Predictions emit within 1 second of candle close
- Alpha decay detection fires on simulated decay scenario
- Kill switches halt trading at configured thresholds
- System survives 24h continuous run without memory leaks

### Dependencies
Sprint 7-8 (models + gating), Sprint 9 (metrics)

---

## Sprint 13: Paper Trading (2-3 days setup + 14-30 days running) [ ]

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

## Sprint 14: Validation + Ablation Study (3-4 days) [ ]

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
                              |                                 |
                    Sprint 9 (Backtesting)           Sprint 11 (Macro + Cascade)
                              |
                    Sprint 10 (Tuning)
                              |
                    Sprint 12 (API + Live)
                              |
                    Sprint 13 (Paper Trading)
                              |
                    Sprint 14 (Validation)
```
