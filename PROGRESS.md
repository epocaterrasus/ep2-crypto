# Development Progress

> This file is the single source of truth for what's done and what's next.
> Updated at the end of every session. Read this FIRST in every new session.

## Current Sprints (Parallel)
- **Sprint 11** — Hyperparameter Tuning (COMPLETE: T1-T4 done, 78 tests)
- **Sprint 15** — Validation + Ablation Study (COMPLETE: T1-T4 done, 110 tests)
- **Sprint 16** — Alpha Enhancement (COMPLETE: T1-T8 done)
- **Sprint 18** — Historical Data Backfill (in progress — scripts being built)
- **Sprint 19** — Production Deployment (COMPLETE: T1-T6 done)

## Completed Sprints
- **Sprint 9** (2026-03-23): Risk Management — 183 tests, 95.3% coverage
- **Sprint 10** (2026-03-23): Backtesting Framework — 174 tests passing
- **Sprint 13** (2026-03-23): API + Live + Monitoring — 239 tests passing
- **Sprint 14** (2026-03-23): Paper Trading — 237 tests passing (T1-T5 all done)
  - PaperExchange (36t), LiveExchange (29t), PaperRunner (18t), DailyReport (28t), integration (9t)
- **Sprint 15** (2026-03-23): Validation + Ablation Study — 110 tests passing
  - AblationStudy (21t): 6 variants (full/no_confidence/no_risk/no_drawdown/no_kills/no_regime)
  - StressTests (47t): 4 historical replays (COVID/China/FTX/Mar2024) + 5 synthetic scenarios
  - PerformanceReport (42t): OOS Sharpe CI, DSR, regime breakdown, MC ruin, deploy verdict
- **Sprint 17** (2026-03-23): Venue Abstraction + Polymarket Execution — 182 tests passing
  - VenueAdapter ABC + registry, PolymarketAdapter (59 tests), BinaryPositionSizer (41 tests),
    PolymarketRiskAdapter (20 tests), PolymarketBacktester (50 tests), integration (20 tests)
  - py-clob-client added as optional dependency (install: uv sync --extra polymarket)
- **Sprint 11** (2026-03-23): Hyperparameter Tuning — 78 tests passing
  - LGBMTuner (Optuna + MedianPruner + DSR), CatBoostTuner, GRUTuner (val_loss objective)
  - ThresholdOptimizer (per-regime grid search), FeatureImportanceAnalyzer (fold stability)
  - walk_forward_sharpe(), deflated_sharpe_ratio() helpers
  - 58 unit tests + 20 integration tests = 78 total

---

## Sprint 16 Tickets — Alpha Enhancement

### S16-T1: Advanced Order Flow Features [x]
**Extend**: Multi-level OBI (l10), VPIN via BVC, book pressure gradient, depth withdrawal ratio
**Files**:
- `src/ep2_crypto/features/microstructure.py` — OBI l10, VPINComputer, BookPressureGradientComputer, DepthWithdrawalComputer
- `tests/test_features/test_microstructure_advanced.py` — Golden dataset + look-ahead bias tests
**Verify**: `uv run pytest tests/test_features/test_microstructure_advanced.py -v`
**Notes**: VPIN = |V_buy - V_sell| / V_total over bucket window (BVC: sign(Δprice)*vol). Book pressure gradient = slope of (bid_vol-ask_vol) across levels. Depth withdrawal = (depth_5b_ago - current) / depth_5b_ago.

### S16-T2: Cross-Exchange Signals [x]
**Create**: Coinbase premium z-score, ETH net taker volume, Binance Long/Short ratio, cross-exchange OFI divergence
**Files**:
- `src/ep2_crypto/features/cross_market.py` — CoinbasePremiumComputer, ETHOrderFlowComputer, LongShortRatioComputer, CrossExchangeOFIComputer
- `tests/test_features/test_cross_exchange.py` — Tests with synthetic prices from 2 exchanges
**Verify**: `uv run pytest tests/test_features/test_cross_exchange.py -v`
**Notes**: Coinbase premium IC 0.03-0.07 (Augustin 2022). ETH leads BTC by 1-5 min (54-57% acc). L/S ratio contrarian at extremes (>2.5 or <0.5).

### S16-T3: Twelve Data ingest for NQ/DXY [x]
**Create**: Twelve Data REST collector replacing yfinance for 1-min NQ/DXY data
**Files**:
- `src/ep2_crypto/ingest/cross_market.py` — TwelveDataCollector, YFinanceFallbackCollector
- `tests/test_ingest/test_cross_market.py` — Mocked HTTP tests for both collectors
**Verify**: `uv run pytest tests/test_ingest/test_cross_market.py -v`
**Notes**: Twelve Data $29/mo plan gives 1-min delay. API key via TWELVE_DATA_API_KEY env. Fall back to yfinance if key absent. Store to cross_market_prices table.

### S16-T4: HAR-RV Multi-Scale Volatility [x]
**Extend**: HAR-RV components at 1h (12 bars), 4h (48 bars), 1d (288 bars) + ratio features
**Files**:
- `src/ep2_crypto/features/volatility.py` — HARRVComputer
- `tests/test_features/test_har_rv.py` — Golden dataset vs hand-computed values
**Verify**: `uv run pytest tests/test_features/test_har_rv.py -v`
**Notes**: HAR-RV = a + b*RV_1h + c*RV_4h + d*RV_1d. Ratio features: RV_1h/RV_1d, RV_4h/RV_1d. Outperforms GARCH by 5-10% RMSE per Corsi (2009).

### S16-T5: Adaptive Conformal Inference + CQR [x]
**Extend**: ACI (Gibbs & Candes 2024) + Conformalized Quantile Regression
**Files**:
- `src/ep2_crypto/confidence/conformal.py` — AdaptiveConformalPredictor, CQRConformalPredictor
- `tests/test_confidence/test_conformal_advanced.py` — Coverage guarantee tests + width reduction tests
**Verify**: `uv run pytest tests/test_confidence/test_conformal_advanced.py -v`
**Notes**: ACI tracks coverage gap and adjusts alpha online (gamma=0.005). CQR uses quantile regression residuals. 20-30% tighter intervals vs fixed conformal while maintaining coverage.

### S16-T6: Multi-Task GRU [x]
**Extend**: Add volatility + volume auxiliary heads to GRU for richer hidden states
**Files**:
- `src/ep2_crypto/models/gru_features.py` — MultiTaskGRUNet, MultiTaskGRUConfig, MultiTaskGRUFeatureExtractor
- `tests/test_models/test_gru_multitask.py` — Forward pass, auxiliary loss, hidden state extraction
**Verify**: `uv run pytest tests/test_models/test_gru_multitask.py -v`
**Notes**: 3 heads: classification (direction), regression (RV), regression (volume). Loss = L_cls + 0.1*L_rv + 0.1*L_vol. Same architecture + heads, no bidirectionality.

### S16-T7: Enhanced Liquidation Cascade Detection [x]
**Extend**: Online Hawkes MLE + state-dependent cascade amplifier
**Files**:
- `src/ep2_crypto/events/cascade.py` — OnlineHawkesEstimator, StateDependentAmplifier
- `tests/test_events/test_cascade_enhanced.py` — Estimator convergence, amplifier scaling tests
**Verify**: `uv run pytest tests/test_events/test_cascade_enhanced.py -v`
**Notes**: Online MLE uses stochastic gradient on log-likelihood. Amplifier: cascade_prob *= (1 + regime_stress_factor). Stress factor from OI percentile + funding z-score combo.

### S16-T8: Deribit Options Data [x]
**Create**: Deribit WebSocket collector + IV surface / risk reversal / max pain features
**Files**:
- `src/ep2_crypto/ingest/deribit.py` — DeribitCollector (IV surface, 25-delta RR, ATM IV)
- `src/ep2_crypto/features/options.py` — OptionsFeatureComputer (ATM IV ROC, risk reversal ROC, max pain distance)
- `tests/test_ingest/test_deribit.py` — Mocked WS subscription tests
- `tests/test_features/test_options.py` — Feature computation tests
**Verify**: `uv run pytest tests/test_ingest/test_deribit.py tests/test_features/test_options.py -v`
**Notes**: 25-delta RR = IV_call25d - IV_put25d (negative = put skew = fear). ATM IV ROC as regime early warning. Max pain = strike with max total OI (calendar feature, not directional).

---

## Sprint 15 Tickets — Validation + Ablation Study

### S15-T1: Ablation study [x]
**Create**: Full-system vs each-component-removed comparison using synthetic OHLCV + signals
**Files**:
- `src/ep2_crypto/backtest/ablation.py` — AblationStudy, AblationResult, AblationVariant enum
- `tests/test_backtest/test_ablation.py` — Golden-dataset tests for each ablation variant
**Verify**: `uv run pytest tests/test_backtest/test_ablation.py -v`
**Notes**: Variants: full, no_confidence_gate, no_risk_engine, no_drawdown_gate, no_kill_switches.
Disable each by modifying BacktestConfig/RiskConfig or filtering signals. Output: delta-Sharpe table.

### S15-T2: Historical stress test replay [x]
**Create**: Replay synthetic crash scenarios modelled on known historical events
**Files**:
- `src/ep2_crypto/backtest/stress_tests.py` — HistoricalStressScenario, SyntheticStressScenario, StressTestRunner
- `tests/test_backtest/test_stress_tests.py` — Tests for each scenario class
**Verify**: `uv run pytest tests/test_backtest/test_stress_tests.py -v -k historical`
**Notes**: Scenarios: COVID Mar2020 (-50% in 48h), May2021 China ban (-30% in 8h), FTX Nov2022 (-25% 5-day), Mar2024 ATH (-15%). Uses BacktestEngine. Pass criterion: kill switches fire before ruin.

### S15-T3: Synthetic stress tests [x]
**Create**: Parameterised synthetic data generators for adversarial scenarios
**Verify**: `uv run pytest tests/test_backtest/test_stress_tests.py -v -k synthetic`
**Notes**: Scenarios: 48h zero vol, 10 flash crashes in 1 week, BTC-NQ correlation → 0, funding rate +0.3% for 2w, broken model (100 bars same direction). All in stress_tests.py.

### S15-T4: Final performance report [x]
**Create**: Consolidated OOS report: Sharpe CI, DSR, regime breakdown, Monte Carlo ruin
**Files**:
- `src/ep2_crypto/backtest/performance_report.py` — PerformanceReport, MonteCarloRuin, report_to_markdown()
- `tests/test_backtest/test_performance_report.py` — Tests with synthetic returns
**Verify**: `uv run pytest tests/test_backtest/test_performance_report.py -v`
**Notes**: Monte Carlo ruin: 10K paths × N-trade simulation. P(ruin at 20% DD) must be < 5%. Report includes: DSR, OOS Sharpe 95% CI, regime breakdown, cost sensitivity, pass/fail verdict.

---

## Sprint 11 Tickets — Hyperparameter Tuning

### S11-T1: LightGBM + CatBoost Optuna tuners [x]
**Create**: Optuna studies for tree models with walk-forward Sharpe as objective
**Files**:
- `src/ep2_crypto/models/tuning.py` — LGBMTuner, CatBoostTuner, walk_forward_sharpe(), deflated_sharpe_ratio()
- `tests/test_models/test_tuning.py` — Unit tests for tuners and helpers
**Verify**: `uv run pytest tests/test_models/test_tuning.py::TestLGBMTuner -v`

### S11-T2: GRU Optuna tuner [x]
**Create**: GRU hyperparameter search added to tuning.py
**Files**:
- `src/ep2_crypto/models/tuning.py` — GRUTuner class (minimize val_loss, DSR via proxy)
- `tests/test_models/test_tuning.py` — GRUTuner tests (5 tests)
**Verify**: `uv run pytest tests/test_models/test_tuning.py::TestGRUTuner -v`

### S11-T3: Confidence threshold optimizer + feature importance stability [x]
**Create**: Grid search confidence thresholds per regime; feature stability analysis
**Files**:
- `src/ep2_crypto/models/tuning.py` — ThresholdOptimizer, FeatureImportanceAnalyzer, TuningResult, StabilityResult
- `tests/test_models/test_tuning.py` — 8 threshold tests + 13 stability tests
**Verify**: `uv run pytest tests/test_models/test_tuning.py::TestThresholdOptimizer tests/test_models/test_tuning.py::TestFeatureImportanceAnalyzer -v`

### S11-T4: Integration tests + sprint acceptance criteria [x]
**Create**: End-to-end tuning pipeline test (20 tests across 5 acceptance criterion classes)
**Files**:
- `tests/test_models/test_tuning_integration.py` — 20 tests: result structure, pruning, DSR, stability, full pipeline
**Verify**: `uv run pytest tests/test_models/ -v`
**Result**: 191 test_models tests passing (78 new Sprint 11 + 113 existing)

---

## Sprint 14 Tickets — Paper Trading

### S14-T1: PaperExchange implementing VenueAdapter [x]
**Create**: Paper trading exchange that simulates fills against real orderbook data
**Files**:
- `src/ep2_crypto/execution/paper_exchange.py` — PaperExchange, PaperPosition, PaperTrade, FillSimulator
- `tests/test_execution/test_paper_exchange.py` — 36 tests: fill simulation, balance tracking, position lifecycle
**Verify**: `uv run pytest tests/test_execution/test_paper_exchange.py -v`

### S14-T2: LiveExchange implementing VenueAdapter [x]
**Create**: Live Binance perps adapter via ccxt
**Files**:
- `src/ep2_crypto/execution/live_exchange.py` — LiveExchange wrapping ccxt for Binance USDT perps
- `tests/test_execution/test_live_exchange.py` — 29 mock ccxt tests
**Verify**: `uv run pytest tests/test_execution/test_live_exchange.py -v`
**Notes**: API key/secret from env only (BINANCE_API_KEY, BINANCE_API_SECRET). Never log credentials.

### S14-T3: Paper trading orchestration [x]
**Create**: Wire live prediction loop to use venue adapter; add --mode flag
**Files**:
- Updated `scripts/live.py` — `--mode paper|live`, `--initial-balance`, `--confidence-threshold` flags
- `src/ep2_crypto/execution/paper_runner.py` — PaperRunner, TradeSignal; 18 tests
- `tests/test_execution/test_paper_runner.py`
**Verify**: `uv run pytest tests/test_execution/test_paper_runner.py -v`

### S14-T4: Daily report generator [x]
**Create**: EOD report: PnL, Sharpe, drawdown, trade count, regime breakdown, go/no-go verdict
**Files**:
- `src/ep2_crypto/monitoring/daily_report.py` — DailyReportGenerator, DailyReport, RegimeStats
- `tests/test_monitoring/test_daily_report.py` — 28 tests
**Verify**: `uv run pytest tests/test_monitoring/test_daily_report.py -v`

### S14-T5: Integration tests + sprint acceptance criteria [x]
**Create**: End-to-end paper trading test covering full signal→fill→report pipeline
**Files**:
- `tests/test_execution/test_paper_integration.py` — 9 tests
**Verify**: `uv run pytest tests/test_execution/ -v`
**Result**: 237 tests total across execution + daily_report (from 0 for Sprint 14)

---

## Sprint 17 Tickets — Venue Abstraction + Polymarket Execution

### S17-T1: VenueAdapter ABC + venue registry [x]
**Create**: Abstract execution interface that both Binance and Polymarket implement
**Files**:
- `src/ep2_crypto/execution/venue.py` — VenueAdapter ABC, VenueType enum, OrderResult, VenueConfig
- `src/ep2_crypto/execution/registry.py` — VenueRegistry
- `tests/test_execution/test_venue.py` — ABC contract tests (27 passing)
**Verify**: `uv run pytest tests/test_execution/test_venue.py -v`

### S17-T2: PolymarketAdapter implementation [x]
**Create**: Polymarket CLOB execution adapter via py-clob-client
**Files**:
- `src/ep2_crypto/execution/polymarket.py` — Market discovery, order placement, WebSocket, resolution tracking (59 tests)
- `src/ep2_crypto/execution/polymarket_config.py` — PolymarketConfig Pydantic model
- `tests/test_execution/test_polymarket.py` — Mocked SDK tests
**Verify**: `uv run pytest tests/test_execution/test_polymarket.py -v`

### S17-T3: BinaryPositionSizer [x]
**Create**: Kelly criterion for binary prediction market outcomes
**Files**:
- `src/ep2_crypto/risk/binary_position_sizer.py` — Binary Kelly, fee-adjusted, quarter-Kelly, caps
- `tests/test_risk/test_binary_position_sizer.py` — Golden dataset + property-based (500 inputs) tests (41 passing)
**Verify**: `uv run pytest tests/test_risk/test_binary_position_sizer.py -v`

### S17-T4: PolymarketRiskAdapter [x]
**Create**: Risk manager wrapper for binary outcome context
**Files**:
- `src/ep2_crypto/risk/polymarket_risk.py` — Wraps KillSwitchManager + DrawdownGate, binary signals
- `tests/test_risk/test_polymarket_risk.py` — Trade flow, kill switch, drawdown, scenarios (20 passing)
**Verify**: `uv run pytest tests/test_risk/test_polymarket_risk.py -v`

### S17-T5: PolymarketBacktester [x]
**Create**: Backtest engine for binary payoff simulation
**Files**:
- `src/ep2_crypto/backtest/polymarket_backtest.py` — Binary payoff sim, fee model, comparison report (50 tests)
- `tests/test_backtest/test_polymarket_backtest.py` — Synthetic signal tests
**Verify**: `uv run pytest tests/test_backtest/test_polymarket_backtest.py -v`

### S17-T6: Integration + live loop hookup [x]
**Create**: Wire everything together, add py-clob-client dependency
**Files**:
- Update `scripts/live.py` — venue selection via --venue flag (binance_perps | polymarket_binary)
- `src/ep2_crypto/execution/__init__.py` — clean exports (16 public names)
- `tests/test_execution/test_integration.py` — Full pipeline mock test (20 passing)
- Update `pyproject.toml` — added py-clob-client>=0.18 under [polymarket] optional group
**Verify**: `uv run pytest tests/test_execution/ -v`

---

## Sprint 18 Tickets — Historical Data Backfill

### S18-T1: BTC historical data backfill script [x]
**Create**: Fetch all BTC data from Binance Futures launch (Sep 2019) to present
**Files**:
- `scripts/collect_history.py` — 1m OHLCV, funding rates, OI, cross-market (NQ/Gold/DXY/ETH)
**Verify**: `uv run python scripts/collect_history.py --start 2026-03-01 --end 2026-03-23` (test run)
**Notes**: --resume flag for continuation. Same script works with SQLite (dev) and TimescaleDB (prod) via EP2_DB_URL.

### S18-T2: Polymarket historical data backfill script [x]
**Create**: Fetch all resolved 5-min BTC markets + derive synthetic outcomes from OHLCV
**Files**:
- `scripts/collect_polymarket_history.py` — Gamma API backfill + --derive-from-btc for pre-2026 synthetic data
**Verify**: `uv run python scripts/collect_polymarket_history.py --start 2026-02-01` (real data)
**Notes**: --derive-from-btc --start 2019-09-01 generates synthetic outcomes from ohlcv_1m table.

---

## Sprint 19 Tickets — Production Deployment

### S19-T1: Docker containerization [x]
**Create**: Dockerfile + .dockerignore
**Files**:
- `docker/Dockerfile` — Python 3.12-slim, uv, Doppler CLI, non-root user, healthcheck
- `.dockerignore` — excludes .venv, .git, data/, research/, *.db
- `docker/.env.example` — DOPPLER_TOKEN, POSTGRES_PASSWORD, GRAFANA_PASSWORD placeholders
**Verify**: `docker build -f docker/Dockerfile -t ep2-crypto .`

### S19-T2: Docker Compose stack [x]
**Create**: Full stack with TimescaleDB, Prometheus, Grafana
**Files**:
- `docker/docker-compose.yml` — 4 services, all secrets via env vars, TimescaleDB/Prometheus localhost-only
- `docker/prometheus.yml` — scrapes ep2-crypto:8000/metrics every 15s
**Verify**: `docker compose -f docker/docker-compose.yml up -d`

### S19-T3: Deploy script [x]
**Create**: One-command deploy to Hetzner
**Files**:
- `scripts/deploy.sh` — rsync + docker compose up + health check 60s timeout + rollback on failure
**Verify**: `./scripts/deploy.sh`
**Notes**: Configurable via EP2_DEPLOY_HOST and EP2_DEPLOY_DIR env vars. Still need scripts/setup_server.sh for first-time setup.

### S19-T4: Telegram alert configuration [x]
**Create**: Configure chat_id, verify all alert tiers fire
**Notes**: Bot token + TELEGRAM_CHAT_ID configured in Doppler. TelegramSender code in alerts.py.

### S19-T5: Grafana dashboards [x]
**Create**: Trading, risk, model, system dashboards
**Files**:
- `docker/grafana/dashboards/trading.json`
- `docker/grafana/dashboards/risk.json`
- `docker/grafana/dashboards/model.json`
- `docker/grafana/dashboards/system.json`

### S19-T6: SQLite → TimescaleDB migration [x]
**Create**: Migration script + hypertable creation
**Files**:
- `scripts/migrate_to_timescale.py` — creates hypertables + migrates all 11 tables, --dry-run, --schema-only, --tables flags
**Notes**: `uv run python scripts/migrate_to_timescale.py --dry-run` to preview. Converts timestamp_ms → TIMESTAMPTZ. Batch inserts with ON CONFLICT DO NOTHING for idempotency.

---

## Sprint 10 Tickets

### S10-T1: Execution simulator (slippage, fees, latency, partial fill, funding) [x]
**Create**: ExecutionSimulator that models realistic trade execution
**Files**:
- `src/ep2_crypto/backtest/simulator.py` — SlippageModel (sqrt-impact 1-3 bps), FeeModel (taker 4bps/maker 2bps), LatencyModel (50-200ms → bars missed), PartialFillModel (10% bar volume cap), FundingRateModel (00/08/16 UTC), ExecutionSimulator orchestrator
- `tests/test_backtest/test_simulator.py` — Tests for each model + combined simulator
**Verify**: `uv run pytest tests/test_backtest/test_simulator.py -v`
**Notes**: Integrates with existing cost_engine.py where possible. Separate RNG streams per component. All costs in bps.

### S10-T2: Backtest metrics (Sharpe Lo-corrected, Sortino, Calmar, CVaR, rolling, regime) [x]
**Create**: Enhanced metrics module beyond existing benchmarks/metrics.py
**Files**:
- `src/ep2_crypto/backtest/metrics.py` — Lo-corrected Sharpe (ACF adjustment), CVaR(5%), expectancy/trade, rolling 30-day Sharpe, regime-specific breakdowns, cost sensitivity analysis
- `tests/test_backtest/test_metrics.py` — Tests with known inputs/outputs
**Verify**: `uv run pytest tests/test_backtest/test_metrics.py -v`
**Notes**: Annualization = sqrt(105,120). NEVER sqrt(252). Reuse BacktestMetrics dataclass pattern from benchmarks.

### S10-T3: Event-driven backtest engine with Numba inner loop [x]
**Create**: Core backtest engine processing bars sequentially
**Files**:
- `src/ep2_crypto/backtest/engine.py` — BacktestEngine with Numba-compiled bar processing, next-bar-open execution, full state persistence (positions, margin, equity curve), integration with risk engine + simulator
- `tests/test_backtest/test_engine.py` — Tests for engine lifecycle, trade execution, state tracking
**Verify**: `uv run pytest tests/test_backtest/test_engine.py -v`
**Notes**: Signal at bar t → execute at open of bar t+1. Track equity, positions, margin per bar. Integrate RiskManager.approve_trade() and on_bar().

### S10-T4: Walk-forward validator with purge/embargo + auditor [x]
**Create**: Purged walk-forward cross-validation with nested inner loop
**Files**:
- `src/ep2_crypto/backtest/walk_forward.py` — WalkForwardValidator (14-day train, 1-day test, 18-bar purge, 12-bar embargo, sliding window), nested inner loop for hyperparameter selection, WalkForwardAuditor (leak detection)
- `tests/test_backtest/test_walk_forward.py` — Tests for fold generation, purge/embargo, auditor
**Verify**: `uv run pytest tests/test_backtest/test_walk_forward.py -v`
**Notes**: Sliding NOT expanding. Inner loop: 3 folds, 20% validation from END of training. Auditor checks: no overlap, purge sufficient, temporal ordering, label shift.

### S10-T5: Statistical validation (PSR, DSR, permutation, bootstrap, stability) [x]
**Create**: Full statistical validation suite
**Files**:
- `src/ep2_crypto/backtest/validation.py` — PSR, DSR, permutation test (5K), block bootstrap CI (10K), walk-forward stability (CV of fold Sharpes), verdict: GENUINE_EDGE/INCONCLUSIVE/LIKELY_NOISE
- `tests/test_backtest/test_validation.py` — Tests with synthetic data (known edge vs noise)
**Verify**: `uv run pytest tests/test_backtest/test_validation.py -v`
**Notes**: Reference: research/statistical_validation_tests.py, BACKTESTING_PLAN.md section 3.

### S10-T6: Update benchmarks + funding rate carry benchmark [x]
**Create**: Add FundingRateCarry benchmark, verify compatibility with new engine
**Files**:
- Update `src/ep2_crypto/benchmarks/strategies.py` — Add FundingRateCarry strategy
- `tests/test_backtest/test_benchmarks_compat.py` — Compatibility tests
**Verify**: `uv run pytest tests/test_backtest/test_benchmarks_compat.py -v`

### S10-T7: CLI entry point scripts/backtest.py [x]
**Create**: CLI for running backtests
**Files**:
- `scripts/backtest.py` — argparse CLI: date range, cost levels, validation mode, output format
**Verify**: `uv run python scripts/backtest.py --help`

### S10-T8: Integration tests + sprint acceptance criteria [x]
**Create**: End-to-end integration tests
**Files**:
- `tests/test_backtest/test_integration.py` — Full pipeline test
**Verify**: `uv run pytest tests/test_backtest/ -v`
**Notes**: Must include costs (no zero-cost mode). Must integrate risk engine.

---

## Sprint 13 Tickets

### S13-T1: Performance logger (flight recorder) [x]
**Create**: Log every trade and every bar state snapshot to SQLite
**Files**:
- `src/ep2_crypto/monitoring/performance_logger.py` — TradeLogger, BarStateLogger
- `tests/test_monitoring/test_performance_logger.py`
**Verify**: `uv run pytest tests/test_monitoring/test_performance_logger.py -v`
**Notes**: Every trade: prediction, confidence, actual, PnL, slippage, latency, regime, features. Every bar: model state, risk state, feature snapshot. SQLite with time-series indices.

### S13-T2: Alpha decay detection (CUSUM, Sharpe, ADWIN, SPRT) [x]
**Create**: Multi-timescale alpha decay detection with response protocol
**Files**:
- `src/ep2_crypto/monitoring/alpha_decay.py` — CUSUMDetector, RollingSharpeMonitor, ADWINDetector, SPRTMonitor, AlphaDecayMonitor
- `tests/test_monitoring/test_alpha_decay.py`
**Verify**: `uv run pytest tests/test_monitoring/test_alpha_decay.py -v`
**Notes**: CUSUM (2-5 bars fast), rolling Sharpe decline (7-14 days), ADWIN adaptive windowing, SPRT on win rate (50 trades). 4-level protocol: Warning→Caution→Stop→Emergency.

### S13-T3: Feature drift detection (PSI) [x]
**Create**: Population Stability Index per feature with alerting
**Files**:
- `src/ep2_crypto/monitoring/drift.py` — FeatureDriftDetector, PSI computation
- `tests/test_monitoring/test_drift.py`
**Verify**: `uv run pytest tests/test_monitoring/test_drift.py -v`
**Notes**: PSI per feature, alert at PSI > 0.2, daily drift report.

### S13-T4: Slippage tracker [x]
**Create**: Track expected vs actual slippage, stats by regime/size/time
**Files**:
- `src/ep2_crypto/monitoring/slippage_tracker.py` — SlippageTracker
- `tests/test_monitoring/test_slippage_tracker.py`
**Verify**: `uv run pytest tests/test_monitoring/test_slippage_tracker.py -v`
**Notes**: Mean/p50/p95/p99 by order size, time of day, vol regime. Alert when actual > 2x expected for 10+ trades.

### S13-T5: Retrain trigger pipeline [x]
**Create**: Auto-retrain with validation gate
**Files**:
- `src/ep2_crypto/monitoring/retrain_trigger.py` — RetrainTrigger
- `tests/test_monitoring/test_retrain_trigger.py`
**Verify**: `uv run pytest tests/test_monitoring/test_retrain_trigger.py -v`
**Notes**: Trigger on PSI>0.2, Sharpe<50% baseline, ADWIN. Validate: must beat old model + feature overlap>70% + ECE<0.05. Atomic swap or keep old.

### S13-T6: Feature rotation tracker [x]
**Create**: Per-feature PSI daily tracking, auto-downweight drifted features
**Files**:
- `src/ep2_crypto/monitoring/feature_rotation.py` — FeatureRotationTracker
- `tests/test_monitoring/test_feature_rotation.py`
**Verify**: `uv run pytest tests/test_monitoring/test_feature_rotation.py -v`
**Notes**: Flag PSI>0.3 for 7 consecutive days. Auto-downweight (not remove). Monthly importance report.

### S13-T7: Alert system (Telegram/Slack) [x]
**Create**: Multi-tier alerting with rate limiting
**Files**:
- `src/ep2_crypto/monitoring/alerts.py` — AlertManager, TelegramSender, SlackSender
- `tests/test_monitoring/test_alerts.py`
**Verify**: `uv run pytest tests/test_monitoring/test_alerts.py -v`
**Notes**: 4 tiers (INFO/WARNING/CRITICAL/EMERGENCY). Rate limiting: 1 INFO/hr, 5 WARNING/hr, unlimited CRITICAL/EMERGENCY. Telegram + optional Slack.

### S13-T8: FastAPI server (/predict, /health, /metrics, /regime) [x]
**Create**: Production API with dependency health checks
**Files**:
- `src/ep2_crypto/api/server.py` — FastAPI app with 4 endpoints
- `tests/test_api/test_server.py`
**Verify**: `uv run pytest tests/test_api/test_server.py -v`
**Notes**: /predict (direction, magnitude, confidence, regime), /health (checks ALL deps), /metrics (live accuracy, rolling Sharpe), /regime. All responses include timestamps + staleness warnings. < 100ms response.

### S13-T9: Live prediction loop [x]
**Create**: Async event loop for live trading
**Files**:
- `scripts/live.py` — Main async loop: collectors + features + inference on 5-min close
- `tests/test_api/test_live.py`
**Verify**: `uv run pytest tests/test_api/test_live.py -v`
**Notes**: Trigger on candle close, graceful degradation if source fails, auto-retrain every 2-4h (warm-start), prediction within 1s of close.

### S13-T10: Execution quality tracker (A/B test) [x]
**Create**: A/B test framework for market vs limit IOC execution
**Files**:
- `src/ep2_crypto/execution/__init__.py`
- `src/ep2_crypto/execution/quality_tracker.py` — ExecutionQualityTracker
- `tests/test_execution/test_quality_tracker.py`
**Verify**: `uv run pytest tests/test_execution/test_quality_tracker.py -v`
**Notes**: Strategy A (market) vs B (limit IOC). Track fill rate, slippage, cost. Auto-switch after 100+ trades per arm.

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

### Session N (2026-03-23) — Sprint 14: Paper Trading
- **What happened**: Completed all 5 Sprint 14 tickets. 237 new tests passing.
- **S14-T1**: PaperExchange implementing VenueAdapter — orderbook fill simulation, balance tracking, position lifecycle (36 tests)
- **S14-T2**: LiveExchange for Binance USDT-M perps via ccxt — mocked tests, credential guard (29 tests)
- **S14-T3**: PaperRunner + live.py wired with `--mode paper|live`, `--initial-balance`, `--confidence-threshold` (18 tests)
- **S14-T4**: DailyReportGenerator — PnL, Sharpe, drawdown, regime breakdown, Sprint 14 go/no-go verdict (28 tests)
- **S14-T5**: Integration tests: full signal→fill→report pipeline (9 tests)
- **Added**: `VenueType.PAPER`, `execution/__init__.py` updated with all new exports
- **Next session**: Start Sprint 15 (Validation + Ablation Study) or Sprint 11 (Hyperparameter Tuning)

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

### Session 7 (2026-03-23)
- **What happened**: Completed ALL 6 Sprint 6 tickets. 91 new tests (591 total).
- **S6-T1**: EfficiencyRatioDetector — Kaufman ER at 2 timescales (short/long), threshold-based regime classification (trending/choppy/neutral), confidence scoring, batch compute. 21 tests.
- **S6-T2**: GARCHDetector — GJR-GARCH(1,1)-t with asymmetric leverage (gamma term), recursive O(1) update, vol regime classification (low/medium/high) via rolling percentiles. 14 tests.
- **S6-T3**: HMMDetector — 2-state GaussianHMM via hmmlearn, forward-algorithm filtered probabilities, BIC model selection (n=2-5), state sorting by variance for semantic stability. 16 tests.
- **S6-T4**: BOCPDDetector — Adams & MacKay (2007) with constant hazard, Normal-inverse-gamma conjugate prior, run-length pruning at r_max. Early warning for regime transitions. 19 tests.
- **S6-T5**: HierarchicalRegimeDetector — Weighted ensemble combining ER (fast) + GARCH (fast) + HMM (core) + BOCPD (early warning). Outputs unified regime label + probabilities + changepoint alert + confidence. 13 tests.
- **S6-T6**: Integration tests verifying all acceptance criteria: distinct regimes visible, labels stable across refits, BOCPD leads HMM, probs sum to 1.0, ER O(1) per bar, no exceptions on edge cases. 8 tests.
- **Lint/type fixes**: Added scipy to mypy ignore list, fixed all ruff lint issues across 11 files.
- **Sprint 6 acceptance criteria**: All passed. Ready to commit.
- **Next session**: Start Sprint 7 (Models — LightGBM + CatBoost + GRU + Stacking)

### Session 8 (2026-03-23) — Sprint 12 (parallel)
- **What happened**: Completed Sprint 12 (Event-Driven Macro + Cascade Detector). 68 new tests.
- **S12-T1/T2**: MacroEventMonitor — economic calendar (CPI/FOMC/NFP/PPI/GDP/PCE), NQ lead-lag trade logic (T+2min signal delay, T+10min expiry), VIX gate (>35 blocks), EWMA BTC-NQ correlation (6h trailing), multi-factor confidence scoring. 25 tests.
- **S12-T3/T4**: CascadeDetector — Hawkes process with exponential kernel (recursive O(1) intensity), branching ratio estimation (method of moments), multi-factor cascade scoring (OI percentile, funding z-score, liquidation burst rate, book depth ratio, price velocity), sigmoid probability mapping, 4-level alert system with position size multiplier. 33 tests.
- **S12-T5/T6/T7**: Integration tests validating all acceptance criteria: macro fires on CPI/FOMC dates, VIX gate suppresses correctly, cascade escalation monotonic, modules operate independently, both can run concurrently. 10 integration tests.
- **Sprint 12 acceptance criteria**: All passed. Lint clean.
- **Note**: Sprint 12 runs in parallel with Sprint 7 (Models) and Sprint 9 (Risk) in separate sessions. Only touched src/ep2_crypto/events/ and tests/test_events/.

### Session 9 (2026-03-23) — Sprint 9 (parallel)
- **What happened**: Completed Sprint 9 (Risk Management Engine). 183 risk tests, 95.3% coverage.
- **S9-T1**: RiskConfig — Pydantic model in `risk/config.py` with env var loading (EP2_RISK_ prefix), field-level validation (gt/le/ge constraints), model_validator for vol range cross-check. Replaces previous dataclass.
- **S9-T2**: PositionTracker — single BTC/USDT position with SQLite persistence, MTM, MAE/MFE, bars_held, thread-safe via RLock. 21 tests.
- **S9-T3**: KillSwitchManager — 5 switches (daily_loss/weekly_loss/max_drawdown/consecutive_loss/emergency), state machine ARMED→TRIGGERED→RESET, SQLite persistence with audit log, requires explicit reset with reason string. 35 tests.
- **S9-T4**: DrawdownGate — **convex formula k=1.5** (`max(0, (1-dd/max_dd)^1.5)`), duration-based reduction (3d→80%, 7d→40%, 14d→halt), graduated 5-phase re-entry (10%→25%→50%→75%→100%), cooldown per phase. 24 tests.
- **S9-T5**: VolatilityGuard — rolling vol (sqrt(105120) annualization), min/max vol gating, trading hours, weekend reduction, funding proximity. 16 tests.
- **S9-T6**: PositionSizer — quarter-Kelly with confidence scaling, ATR stops (3 ATR), risk-per-trade cap (1% of equity), max position cap (5%), exchange min size. 25 tests.
- **S9-T7**: RiskManager orchestrator — approve_trade (7-step pipeline), on_bar (MTM+stop+holding+kill), on_trade_opened/closed, day/week reset, emergency trigger. 22 tests.
- **S9-T8**: Golden dataset — 24 parameterized cases (convex formula at 8 DD levels, Kelly at 6 WR/payoff combos, kill switch boundary ±epsilon, position cap).
- **S9-T9**: Property-based tests (Hypothesis) — 5 properties with 500+ random inputs each (size cap, kill switch blocking, DD multiplier bounds, Kelly non-negative, risk-per-trade cap).
- **S9-T10**: File-backed persistence — kill switches, position tracker, drawdown gate all survive file close/reopen (tempfile SQLite).
- **S9-T11**: Stress scenarios — COVID crash (stop fires), flash crash (1-bar stop), 10 consecutive losses (kill switch triggers), slow bleed (drawdown gate reduces), rapid cycling (no exceptions), extreme vol (no exceptions).
- **Sprint 9 acceptance criteria**: All passed. 95.3% coverage. 183 tests. 792 total project tests.
- **Note**: Sprint 9 ran in parallel with Sprint 7 (Models) and Sprint 12 (Macro+Cascade). Only touched src/ep2_crypto/risk/ and tests/test_risk/.

### Session 10 (2026-03-23) — Sprint 7
- **What happened**: Completed ALL 8 Sprint 7 tickets. 113 new tests (851 total).
- **S7-T1**: TripleBarrierLabeler — ATR-based barriers, vertical/upper/lower, adaptive thresholds, fixed-threshold baseline, class weights. 25 tests.
- **S7-T2**: LGBMDirectionModel — ternary multiclass (softmax), warm-start via init_model, early stopping, SHAP feature importance, save/load round-trip. 18 tests.
- **S7-T3**: CatBoostDirectionModel — ordered boosting (Bayesian bootstrap) for diversity, same interface as LightGBM for stacking. 11 tests.
- **S7-T4**: GRUFeatureExtractor — 2-layer GRU, hidden state extraction as features for trees, AdamW + cosine annealing + gradient clip=1.0, ONNX export. DataLoader never shuffles. 14 tests.
- **S7-T5**: QuantileModel — 5 LightGBM quantile regressors (10/25/50/75/90), prediction intervals, interval width for risk, quantile crossing detection. 12 tests.
- **S7-T6**: StackingEnsemble — logistic regression meta-learner on OOF class probabilities, simple/weighted average baselines, scipy weight optimization, handles missing classes gracefully. 12 tests.
- **S7-T7**: IsotonicCalibrator — per-class isotonic regression, ECE computation, reliability diagram data, save/load. 13 tests.
- **S7-T8**: Integration tests — full pipeline (label → train → predict → stack → calibrate), all acceptance criteria validated. 8 tests.
- **Dependency**: Added onnxscript, sklearn/joblib to mypy ignore list.
- **Sprint 7 acceptance criteria**: All passed. 851 tests total.
- **Next session**: Start Sprint 8 (Confidence Gating Pipeline)

### Session 11 (2026-03-23) — Sprint 13 (parallel)
- **What happened**: Completed Sprint 13 (API + Live Prediction + Monitoring). 239 new tests.
- **S13-T1**: PerformanceLogger — flight recorder with SQLite trade_log + bar_state_log tables, JSON-serialized features/risk_state, query methods, latency measurement. 32 tests.
- **S13-T2**: AlphaDecayMonitor — 4 detectors: CUSUMDetector (mean-shift), RollingSharpeMonitor (7-14d decline), ADWINDetector (distribution change), SPRTMonitor (win rate test). 4-level protocol: NORMAL→WARNING→CAUTION→STOP→EMERGENCY. 37 tests.
- **S13-T3**: FeatureDriftDetector — PSI with equal-frequency binning, per-feature tracking, daily report with severity classification (none/moderate/significant/critical). 32 tests.
- **S13-T4**: SlippageTracker — expected vs actual, stats by size/hour/regime, p50/p95/p99 percentiles, consecutive excess alert (>2x for 10+ trades), adaptive slippage estimate for position sizing. 20 tests.
- **S13-T5**: RetrainTrigger — trigger conditions (PSI>0.2, Sharpe decline, ADWIN, scheduled), 3-gate validation (performance, feature overlap>70%, ECE<0.05), atomic swap or keep old, cooldown, audit trail. 22 tests.
- **S13-T6**: FeatureRotationTracker — daily per-feature PSI, flag at PSI>0.3 for 7 consecutive days, auto-downweight (not remove), recovery on return to normal, monthly importance report. 18 tests.
- **S13-T7**: AlertManager — 4 tiers (INFO/WARNING/CRITICAL/EMERGENCY), RateLimiter (1 INFO/hr, 5 WARNING/hr, unlimited CRITICAL/EMERGENCY), TelegramSender + SlackSender backends, Protocol-based. 25 tests.
- **S13-T8**: FastAPI server — 4 endpoints (/predict, /health, /metrics, /regime), all responses include timestamps + staleness warnings, health checks ALL dependencies, AppState shared with live loop. 17 tests.
- **S13-T9**: LivePredictionLoop — async event loop, 5-min candle close trigger, graceful degradation (mark_source_degraded/recovered), auto-retrain check, SIGTERM/SIGINT handling. 21 tests.
- **S13-T10**: ExecutionQualityTracker — A/B test market vs limit IOC, auto-select after min_trades per arm, effective cost comparison with fill rate penalty, StrategyStats aggregation. 15 tests.
- **Sprint 13 acceptance criteria**: All passed. 239 tests. Lint clean.
- **Note**: Sprint 13 ran in parallel with Sprint 10 (Backtesting) and Sprint 17 (Polymarket). Only touched monitoring/, api/, execution/, scripts/live.py and their tests.

### Session 12 (2026-03-23) — Sprint 10
- **What happened**: Completed Sprint 10 (Backtesting Framework). 174 new backtest tests.
- **S10-T1**: ExecutionSimulator — SlippageEstimator (sqrt-impact), FeeCalculator (taker/maker), LatencySimulator, PartialFillSimulator (10% bar volume cap), FundingAccumulator (00/08/16 UTC settlement), ExecutionSimulator orchestrator with separate RNG streams. 46 tests.
- **S10-T2**: BacktestMetrics — Lo-corrected Sharpe (ACF adjustment up to 6 lags), Sortino, Calmar, CVaR(5%), rolling 30-day Sharpe, regime-specific breakdowns, cost sensitivity analysis with break-even finder. Annualization: sqrt(105,120). 35 tests.
- **S10-T3**: BacktestEngine — Event-driven with Numba-compiled inner loop (_update_equity_numba, _compute_bar_returns_numba), next-bar-open execution, full risk engine integration (approve_trade + on_bar + kill switches), signal reversal handling, position tracking. 17 tests.
- **S10-T4**: WalkForwardValidator — Purged walk-forward (14-day train, 1-day test, 18-bar purge, 12-bar embargo, sliding window), nested inner loop (3 folds, 20% validation from END of training), WalkForwardAuditor (6 checks: overlap, purge, temporal, consistency, duplicates, embargo). 25 tests.
- **S10-T5**: Statistical validation — PSR (Bailey & López de Prado), DSR (multiple testing correction), permutation test (5K), block bootstrap CI (10K, block=T^(1/3)), walk-forward stability (CV<0.5), combined verdict: GENUINE_EDGE/INCONCLUSIVE/LIKELY_NOISE. 24 tests.
- **S10-T6**: FundingRateCarry benchmark — contrarian carry strategy (short when funding positive, long when negative), smoothing for turnover reduction. Added to registry and default suite. 12 tests.
- **S10-T7**: CLI entry point `scripts/backtest.py` — argparse with --days, --validate, --cost-sensitivity, --walk-forward flags.
- **S10-T8**: Integration tests — full pipeline (data→engine→metrics→validation), costs always present, risk integration, walk-forward audit, annualization verification. 15 tests.
- **Fix**: Increased sleep in flaky OI polling test (test_derivatives.py) from 0.15s to 0.30s.
- **Sprint 10 acceptance criteria**: All passed. 174 backtest tests. Lint clean (only TC002 warnings for NDArray import, consistent with project convention).

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
| Sprint 6: Regime Detection | Complete | 2026-03-23 |
| Sprint 7: Models + Stacking | Complete | 2026-03-23 |
| Sprint 8: Confidence Gating | Complete | 2026-03-23 |
| **Sprint 9: Risk Management** | **Complete** | **2026-03-23** |
| **Sprint 10: Backtesting Framework** | **Complete** | **2026-03-23** |
| Sprint 11: Hyperparameter Tuning | Complete | 78 tests (LGBMTuner, CatBoostTuner, GRUTuner, ThresholdOptimizer, FeatureImportanceAnalyzer) |
| Sprint 12: Macro Events + Cascade | Complete | 2026-03-23 |
| Sprint 13: API + Live + Monitoring | Complete | 2026-03-23 |
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

## Sprint 6 Tickets

### S6-T1: Efficiency Ratio detector with threshold-based regime [x]
**Create**: Kaufman ER regime detector — fast layer, O(1) per bar
**Files**:
- `src/ep2_crypto/regime/efficiency_ratio.py` — EfficiencyRatioDetector with multi-window ER, threshold-based regime (trending/choppy/neutral)
- `tests/test_regime/test_efficiency_ratio.py` — Golden dataset tests, O(1) verification, threshold behavior
**Verify**: `uv run pytest tests/test_regime/test_efficiency_ratio.py -v`
**Research**: `RR-regime-detection-methods.md`
**Notes**: ER = |net_move| / sum(|individual_moves|). Thresholds: ER > 0.5 trending, ER < 0.3 choppy. Multi-window (20, 100 bars). Returns regime label + confidence.

### S6-T2: GJR-GARCH(1,1)-t conditional volatility detector [x]
**Create**: GJR-GARCH vol regime detector — fast layer, recursive per bar
**Files**:
- `src/ep2_crypto/regime/garch.py` — GARCHDetector with GJR-GARCH(1,1)-t, vol regime classification (low/medium/high)
- `tests/test_regime/test_garch.py` — Golden dataset tests, asymmetric leverage, vol classification
**Verify**: `uv run pytest tests/test_regime/test_garch.py -v`
**Research**: `RR-regime-detection-methods.md`
**Notes**: sigma2_t = omega + (alpha + gamma*I_{e<0})*e_{t-1}^2 + beta*sigma2_{t-1}. Student-t innovations. Vol thresholds from rolling percentiles. Returns conditional vol + vol regime label.

### S6-T3: 2-state GaussianHMM with forward algorithm [x]
**Create**: HMM regime detector — core layer, forward algorithm for online inference
**Files**:
- `src/ep2_crypto/regime/hmm.py` — HMMDetector with hmmlearn GaussianHMM, forward-algorithm filtered probs, weekly refit, BIC model selection (n=2-5), semantic stability (sort by vol)
- `tests/test_regime/test_hmm.py` — Tests for fitting, filtered probs, refit, label stability
**Verify**: `uv run pytest tests/test_regime/test_hmm.py -v`
**Research**: `RR-regime-detection-methods.md`
**Notes**: Features: (log_return, realized_vol). Forward algorithm for online P(state|data). Refit on 7-day sliding window. Sort states by volatility for semantic stability. Pseudo-incremental via init_params="".

### S6-T4: BOCPD change point detection [x]
**Create**: Bayesian Online Change Point Detection — early warning for regime transitions
**Files**:
- `src/ep2_crypto/regime/bocpd.py` — BOCPDDetector with Adams & MacKay algorithm, constant hazard, Gaussian conjugate prior, run-length pruning
- `tests/test_regime/test_bocpd.py` — Tests for change point detection, lead time vs HMM, run-length distribution
**Verify**: `uv run pytest tests/test_regime/test_bocpd.py -v`
**Research**: `RR-regime-detection-methods.md`
**Notes**: Hazard h(r) = 1/lambda. Lambda ~288-1440 for crypto. Normal-inverse-gamma conjugate prior. Prune run lengths at r_max=200. Returns changepoint probability + run length.

### S6-T5: Hierarchical ensemble detector [x]
**Create**: Orchestrator combining all regime layers into unified output
**Files**:
- `src/ep2_crypto/regime/detector.py` — HierarchicalRegimeDetector combining ER (fast) + GARCH (fast) + HMM (core) + BOCPD (early warning). Outputs: regime label, regime probabilities, change point alert, confidence
- `tests/test_regime/test_detector.py` — Ensemble tests, probability sum=1.0, BOCPD leads HMM, regime stability
**Verify**: `uv run pytest tests/test_regime/test_detector.py -v`
**Research**: `RESEARCH_SYNTHESIS.md` section "Regime Detection: Hierarchical Ensemble"
**Notes**: Voting ensemble with confidence reduction during transitions. Regime labels sorted by volatility. BOCPD fires before HMM transition.

### S6-T6: Sprint 6 integration tests + acceptance criteria [x]
**Create**: Full regime detection integration test
**Files**:
- `tests/test_regime/test_regime_integration.py` — End-to-end regime detection on synthetic data
**Verify**: `uv run pytest tests/test_regime/ -v`
**Notes**: Acceptance criteria: 2-3 distinct regimes visible, labels stable across refits, BOCPD fires before HMM, probabilities sum to 1.0, ER+GARCH O(1) per bar.

---

## Sprint 7 Tickets

### S7-T1: Triple barrier labeling module [x]
**Create**: Target labeling for ternary classification (up/flat/down) via triple barrier method
**Files**:
- `src/ep2_crypto/models/labeling.py` — TripleBarrierLabeler with vertical/upper/lower barriers, adaptive threshold per regime
- `tests/test_models/test_labeling.py` — Golden dataset tests, barrier touch order, regime-adaptive thresholds
**Verify**: `uv run pytest tests/test_models/test_labeling.py -v`
**Research**: ADR-008 (triple barrier over fixed threshold)
**Notes**: Label = which barrier touched first. Vertical = max holding period (12 bars = 1h). Upper/lower = ATR-based or percentile-based. Ternary: up(+1), flat(0), down(-1). Must be vectorized for batch labeling of training windows.

### S7-T2: LightGBM ternary classifier [x]
**Create**: LightGBM direction model with warm-start, early stopping, SHAP tracking
**Files**:
- `src/ep2_crypto/models/lgbm_direction.py` — LGBMDirectionModel (train, predict, predict_proba, save, load, feature_importance)
- `tests/test_models/test_lgbm_direction.py` — Train/predict cycle, save/load round-trip, SHAP tracking
**Verify**: `uv run pytest tests/test_models/test_lgbm_direction.py -v`
**Research**: `RR-lightgbm-crypto-tuning.md`
**Notes**: num_leaves=31, max_depth=5, lr=0.05, min_child_samples=50. Warm-start via init_model. Early stopping on purged validation set. SHAP for feature importance. Ternary multiclass (softmax).

### S7-T3: CatBoost ternary classifier [x]
**Create**: CatBoost direction model with ordered boosting for diversity
**Files**:
- `src/ep2_crypto/models/catboost_direction.py` — CatBoostDirectionModel (same interface as LightGBM)
- `tests/test_models/test_catboost_direction.py` — Train/predict cycle, save/load round-trip
**Verify**: `uv run pytest tests/test_models/test_catboost_direction.py -v`
**Notes**: Ordered boosting prevents target leakage. Same ternary target. depth=5, lr=0.05, l2_leaf_reg=3. Provides diversity for stacking.

### S7-T4: GRU hidden state extractor [x]
**Create**: GRU model that extracts hidden states as features for tree models
**Files**:
- `src/ep2_crypto/models/gru_features.py` — GRUFeatureExtractor (train, extract_hidden, save, load, export_onnx)
- `tests/test_models/test_gru_features.py` — Training, hidden state shape, ONNX export, DataLoader ordering
**Verify**: `uv run pytest tests/test_models/test_gru_features.py -v`
**Research**: `RR-gru-tcn-training-architecture.md`
**Notes**: GRU(input=N, hidden=64, layers=2, dropout=0.3). seq_len=24 bars. AdamW + cosine annealing + gradient clip=1.0. DataLoader NEVER shuffles (time series). Hidden state → features for stacking. ONNX export for inference.

### S7-T5: LightGBM quantile regression [x]
**Create**: Quantile regression for prediction intervals and risk assessment
**Files**:
- `src/ep2_crypto/models/quantile.py` — QuantileModel (train, predict quantiles, prediction interval width)
- `tests/test_models/test_quantile.py` — Quantile ordering, coverage, interval width
**Verify**: `uv run pytest tests/test_models/test_quantile.py -v`
**Notes**: 5 quantiles (10th, 25th, 50th, 75th, 90th). Each is a separate LightGBM regressor with quantile objective. Prediction interval width = risk signal for confidence gating.

### S7-T6: Stacking ensemble meta-learner [x]
**Create**: Logistic regression meta-learner on OOF predictions from base models
**Files**:
- `src/ep2_crypto/models/stacking.py` — StackingEnsemble (generate_oof, train_meta, predict, optimize_weights)
- `tests/test_models/test_stacking.py` — OOF generation, meta-learner training, improvement over best single
**Verify**: `uv run pytest tests/test_models/test_stacking.py -v`
**Notes**: OOF from inner walk-forward (never in-sample). Meta-learner: logistic regression on class probabilities from LightGBM + CatBoost + GRU predictions. Ensemble weights via scipy.optimize (Sharpe objective).

### S7-T7: Isotonic calibration [x]
**Create**: Probability calibration via isotonic regression
**Files**:
- `src/ep2_crypto/models/calibration.py` — IsotonicCalibrator (fit, calibrate, reliability_curve)
- `tests/test_models/test_calibration.py` — Calibration improvement, reliability diagram data, per-class calibration
**Verify**: `uv run pytest tests/test_models/test_calibration.py -v`
**Notes**: Isotonic regression on held-out set. Per-class calibration (one-vs-rest). Reliability curve for evaluation. Must improve ECE (expected calibration error).

### S7-T8: Sprint 7 integration tests + acceptance criteria [x]
**Create**: End-to-end model pipeline test on synthetic data
**Files**:
- `tests/test_models/test_model_integration.py` — Full pipeline: label → train → predict → stack → calibrate
**Verify**: `uv run pytest tests/test_models/ -v`
**Notes**: Acceptance criteria from SPRINTS.md: each model trains on 14-day window, >52% accuracy above random, stacking improves over best single, feature importance tracked, GRU DataLoader never shuffles, all models serializable, calibration improves reliability.

---

## Sprint 12 Tickets

### S12-T1: Economic calendar + macro event data model [x]
**Create**: MacroEventType enum, MacroEvent dataclass, MacroSignal dataclass, pre-scheduled 2025 calendar
**Files**:
- `src/ep2_crypto/events/macro.py` — Data model + calendar
**Verify**: `uv run pytest tests/test_events/test_macro.py -v -k TestMacroEvent`

### S12-T2: Macro event monitor with NQ lead-lag [x]
**Create**: MacroEventMonitor (VIX gate, correlation gate, signal delay/expiry), EWMACorrelation tracker
**Files**:
- `src/ep2_crypto/events/macro.py` — MacroEventMonitor, EWMACorrelation
**Verify**: `uv run pytest tests/test_events/test_macro.py -v -k TestMacroEventMonitor`

### S12-T3: Hawkes process liquidation intensity model [x]
**Create**: HawkesProcess with exponential kernel, recursive intensity, branching ratio estimation
**Files**:
- `src/ep2_crypto/events/cascade.py` — HawkesProcess
**Verify**: `uv run pytest tests/test_events/test_cascade.py -v -k TestHawkesProcess`

### S12-T4: Multi-factor cascade detector [x]
**Create**: CascadeDetector combining Hawkes with OI, funding, depth, velocity into cascade probability
**Files**:
- `src/ep2_crypto/events/cascade.py` — CascadeDetector, CascadeState
**Verify**: `uv run pytest tests/test_events/test_cascade.py -v -k TestCascadeDetector`

### S12-T5: Macro module tests [x]
**Files**: `tests/test_events/test_macro.py` — 25 tests
**Verify**: `uv run pytest tests/test_events/test_macro.py -v`

### S12-T6: Cascade detector tests [x]
**Files**: `tests/test_events/test_cascade.py` — 33 tests
**Verify**: `uv run pytest tests/test_events/test_cascade.py -v`

### S12-T7: Integration tests + acceptance criteria [x]
**Files**: `tests/test_events/test_events_integration.py` — 10 tests
**Verify**: `uv run pytest tests/test_events/ -v`

---

## Sprint 9 Tickets

### S9-T1: RiskConfig Pydantic model [x]
**Files**: `src/ep2_crypto/risk/config.py`
**Verify**: `uv run pytest tests/test_risk/test_risk_manager.py::TestConfigValidation -v`

### S9-T2: PositionTracker [x]
**Files**: `src/ep2_crypto/risk/position_tracker.py`, `tests/test_risk/test_position_tracker.py`
**Verify**: `uv run pytest tests/test_risk/test_position_tracker.py -v`

### S9-T3: KillSwitchManager [x]
**Files**: `src/ep2_crypto/risk/kill_switches.py`, `tests/test_risk/test_kill_switches.py`
**Verify**: `uv run pytest tests/test_risk/test_kill_switches.py -v`

### S9-T4: DrawdownGate (k=1.5 convex + duration + recovery) [x]
**Files**: `src/ep2_crypto/risk/drawdown_gate.py`, `tests/test_risk/test_drawdown_gate.py`
**Verify**: `uv run pytest tests/test_risk/test_drawdown_gate.py -v`

### S9-T5: VolatilityGuard [x]
**Files**: `src/ep2_crypto/risk/volatility_guard.py`, `tests/test_risk/test_volatility_guard.py`
**Verify**: `uv run pytest tests/test_risk/test_volatility_guard.py -v`

### S9-T6: PositionSizer (quarter-Kelly + ATR stops) [x]
**Files**: `src/ep2_crypto/risk/position_sizer.py`, `tests/test_risk/test_position_sizer.py`
**Verify**: `uv run pytest tests/test_risk/test_position_sizer.py -v`

### S9-T7: RiskManager orchestrator [x]
**Files**: `src/ep2_crypto/risk/risk_manager.py`, `tests/test_risk/test_risk_manager.py`
**Verify**: `uv run pytest tests/test_risk/test_risk_manager.py -v`

### S9-T8: Golden dataset tests (24 parameterized) [x]
**Files**: `tests/test_risk/test_risk_golden.py`
**Verify**: `uv run pytest tests/test_risk/test_risk_golden.py -v`

### S9-T9: Property-based tests (Hypothesis, 500+ inputs) [x]
**Files**: `tests/test_risk/test_risk_properties.py`
**Verify**: `uv run pytest tests/test_risk/test_risk_properties.py -v`

### S9-T10: File-backed persistence tests [x]
**Files**: `tests/test_risk/test_risk_persistence.py`
**Verify**: `uv run pytest tests/test_risk/test_risk_persistence.py -v`

### S9-T11: Stress scenario tests [x]
**Files**: `tests/test_risk/test_risk_stress.py`
**Verify**: `uv run pytest tests/test_risk/test_risk_stress.py -v`

---

## Sprint 8 Tickets

### S8-T1: Meta-labeling model (secondary classifier) [x]
**Create**: Secondary LightGBM model that predicts "will THIS trade be profitable?" — Lopez de Prado's technique
**Files**:
- `src/ep2_crypto/confidence/meta_labeling.py` — MetaLabeler: binary classifier (profitable/not) on primary model output + features + regime
- `tests/test_confidence/test_meta_labeling.py` — Training, prediction, save/load, Sharpe improvement
**Verify**: `uv run pytest tests/test_confidence/test_meta_labeling.py -v`
**Research**: `RR-metalabeling-implementation-guide.md`
**Notes**: Input = primary model prediction + probabilities + features + regime. Output = P(profitable). Uses LightGBM binary classifier. Must improve Sharpe by >50% on backtest data.

### S8-T2: Conformal prediction gate [x]
**Create**: Conformal prediction with adaptive alpha for ambiguity detection
**Files**:
- `src/ep2_crypto/confidence/conformal.py` — ConformalPredictor: calibrate nonconformity scores, produce prediction sets
- `tests/test_confidence/test_conformal.py` — Coverage guarantees, singleton filtering, adaptive alpha
**Verify**: `uv run pytest tests/test_confidence/test_conformal.py -v`
**Notes**: Prediction set must be singleton {UP} or {DOWN} to trade. If set is {UP, DOWN} or {UP, FLAT, DOWN}: abstain. Adaptive alpha tracks empirical coverage.

### S8-T3: Gating pipeline orchestrator [x]
**Create**: Full confidence gating pipeline combining all gates
**Files**:
- `src/ep2_crypto/confidence/gating.py` — ConfidenceGatingPipeline: 7-stage gate orchestration
- `tests/test_confidence/test_gating.py` — Individual gate enable/disable, full pipeline, logging
**Verify**: `uv run pytest tests/test_confidence/test_gating.py -v`
**Notes**: Pipeline: (1) isotonic calibration, (2) meta-labeling gate, (3) ensemble agreement, (4) conformal prediction, (5) signal filters (vol/regime/liquidity), (6) adaptive threshold, (7) drawdown gate. Each gate independently enable/disable for ablation.

### S8-T4: Confidence-aware position sizing [x]
**Create**: Quarter-Kelly position sizing with Bayesian Kelly and confidence scaling
**Files**:
- `src/ep2_crypto/confidence/position_sizing.py` — ConfidencePositionSizer: Kelly × confidence → size
- `tests/test_confidence/test_position_sizing.py` — Sizing calculations, max cap enforcement, edge cases
**Verify**: `uv run pytest tests/test_confidence/test_position_sizing.py -v`
**Notes**: quarter-Kelly: size = 0.25 * kelly_fraction * composite_confidence. Bayesian Kelly with uncertain probabilities. Max position cap 5% of capital. Integrates with risk/position_sizer.py for final enforcement.

### S8-T5: Sprint 8 integration tests + acceptance criteria [x]
**Create**: End-to-end confidence gating pipeline test
**Files**:
- `tests/test_confidence/test_confidence_integration.py` — Full pipeline: predict → calibrate → meta-label → conformal → gate → size
**Verify**: `uv run pytest tests/test_confidence/ -v`
**Notes**: Acceptance criteria: meta-labeling improves Sharpe >50%, conformal filters ambiguous, each gate toggleable, position sizing respects cap, drawdown gate reduces progressively.

---

## Future Sprint Ticket Decomposition

Sprints 10-14 (excluding 9 and 12, now complete) will be decomposed into tickets when they become the current sprint.
See SPRINTS.md for high-level sprint definitions.
