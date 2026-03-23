# Requirements: ep2-crypto

## 1. Functional Requirements

### FR-1: Data Ingestion
- **FR-1.1**: Ingest BTC/USDT 1-minute OHLCV candles from Binance via WebSocket in real-time
- **FR-1.2**: Ingest L2 order book snapshots (top 20 levels) from Binance at 100ms intervals
- **FR-1.3**: Ingest aggTrade stream from Binance for trade flow analysis
- **FR-1.4**: Poll open interest from Bybit every 5 minutes via REST
- **FR-1.5**: Poll funding rate from Bybit every 5 minutes via REST
- **FR-1.6**: Stream liquidation events from Bybit WebSocket (`allLiquidation.BTCUSDT`)
- **FR-1.7**: Stream ETH/USDT 1-minute OHLCV from Binance via WebSocket
- **FR-1.8**: Poll NQ futures, Gold, DXY 5-minute bars from yfinance every 5 minutes
- **FR-1.9**: Stream mempool data from mempool.space WebSocket for whale detection (>10 BTC)
- **FR-1.10**: Maintain economic calendar with CPI, FOMC, NFP dates for macro event module
- **FR-1.11**: Automatically reconnect all data streams after disconnection
- **FR-1.12**: Backfill 60 days of historical OHLCV, OI, and funding rate data

### FR-2: Feature Engineering
- **FR-2.1**: Compute Order Book Imbalance (OBI) at levels 1-3 and 1-5, weighted by price proximity
- **FR-2.2**: Compute Order Flow Imbalance (OFI) using Cont-Stoikov-Talreja model
- **FR-2.3**: Compute microprice (Gatheral-Stoikov volume-weighted mid)
- **FR-2.4**: Compute trade flow imbalance at 30-second and 5-minute windows
- **FR-2.5**: Compute volume delta, VWAP deviation, volume rate of change
- **FR-2.6**: Compute realized volatility (5-min, 30-min), Parkinson volatility, EWMA volatility, vol-of-vol
- **FR-2.7**: Compute Rate of Change at 1, 3, 6, 12 bar lookbacks
- **FR-2.8**: Compute RSI(14), linear regression slope(20), price quantile rank(60)
- **FR-2.9**: Encode temporal features: cyclical hour/minute/day, session indicator, time-to-funding
- **FR-2.10**: Compute cross-market lead-lag correlation with NQ (lagged 1-3 bars, US hours only)
- **FR-2.11**: Compute regime features: Efficiency Ratio, GARCH vol, HMM probabilities
- **FR-2.12**: Apply dual normalization: raw for tree models, robust scaling for neural models
- **FR-2.13**: All features must use only data at time <= t (strict no-look-ahead)
- **FR-2.14**: Pipeline must handle missing data sources gracefully (fill or skip with lower confidence)

### FR-3: Regime Detection
- **FR-3.1**: Compute Efficiency Ratio every bar (O(1) complexity)
- **FR-3.2**: Compute GJR-GARCH(1,1)-t conditional volatility every bar
- **FR-3.3**: Run 2-state GaussianHMM with weekly refit on 7-day sliding window
- **FR-3.4**: Run BOCPD for change point detection (2-10 bars faster than HMM)
- **FR-3.5**: Combine regime signals via LightGBM meta-learner (weekly refit)
- **FR-3.6**: Output regime label, regime probabilities (summing to 1.0), and change point alerts

### FR-4: Prediction Models
- **FR-4.1**: Train LightGBM ternary classifier (up/flat/down) with adaptive threshold
- **FR-4.2**: Train CatBoost ternary classifier for ensemble diversity
- **FR-4.3**: Train GRU sequence model and extract hidden state as features
- **FR-4.4**: Train LightGBM quantile regression (5 quantiles: 10th, 25th, 50th, 75th, 90th)
- **FR-4.5**: Combine via stacking meta-learner (logistic regression on OOF predictions)
- **FR-4.6**: Calibrate predictions via isotonic regression
- **FR-4.7**: Retrain models every 2-4 hours with warm-start on 14-day sliding window
- **FR-4.8**: Export GRU to ONNX for fast inference

### FR-5: Confidence Gating
- **FR-5.1**: Apply isotonic calibration to raw predictions
- **FR-5.2**: Meta-labeling gate: predict whether trade will be profitable
- **FR-5.3**: Ensemble agreement check: variance of base model predictions < threshold
- **FR-5.4**: Conformal prediction: only trade when prediction set is singleton {UP} or {DOWN}
- **FR-5.5**: Signal filters: volatility range, regime stability, liquidity checks
- **FR-5.6**: Adaptive confidence threshold: regime-dependent, updated weekly
- **FR-5.7**: Drawdown gate: progressive position reduction at 3%, 5%, 10%, 15% drawdown
- **FR-5.8**: Position sizing: quarter-Kelly * composite confidence, capped at 5% of capital

### FR-6: Event-Driven Modules
- **FR-6.1**: Macro module: monitor NQ at T+0 of scheduled macro events, enter BTC in NQ direction at T+2min
- **FR-6.2**: Macro module: VIX gate (skip when VIX > 35)
- **FR-6.3**: Cascade detector: multi-factor cascade score from OI, funding, liquidation rate, book depth
- **FR-6.4**: Cascade detector: Hawkes process for liquidation intensity
- **FR-6.5**: Cascade output reduces position size proportional to cascade probability

### FR-7: Backtesting
- **FR-7.1**: Purged walk-forward validation with 14-day train, 1-day test, 18-bar purge, 12-bar embargo
- **FR-7.2**: Event-driven backtest engine with Numba-compiled inner loop
- **FR-7.3**: Next-bar-open execution (never fill at current bar close)
- **FR-7.4**: Transaction cost model: slippage, maker/taker fees, latency, partial fills
- **FR-7.5**: Compute all metrics: Sharpe, Sortino, Calmar, max drawdown, profit factor, CVaR
- **FR-7.6**: Statistical validation: PSR, DSR, permutation test, bootstrap CI
- **FR-7.7**: Benchmark comparison: buy-and-hold, momentum, random, funding carry
- **FR-7.8**: Cost sensitivity analysis at multiple cost levels (0-20 bps)

### FR-8: API & Live System
- **FR-8.1**: `GET /predict` returns direction, magnitude, confidence, regime with timestamps
- **FR-8.2**: `GET /health` checks all data streams, model status, last prediction time
- **FR-8.3**: `GET /metrics` returns live rolling Sharpe, accuracy, trade count
- **FR-8.4**: `GET /regime` returns current regime and transition probabilities
- **FR-8.5**: Predictions emit within 1 second of 5-min candle close
- **FR-8.6**: Auto-retrain on schedule without service interruption

### FR-9: Monitoring & Safety
- **FR-9.1**: Alpha decay detection via CUSUM, rolling Sharpe, ADWIN, SPRT
- **FR-9.2**: Feature drift detection via PSI, alert at PSI > 0.2
- **FR-9.3**: Kill switches: daily loss limit, weekly loss limit, max drawdown, consecutive losses
- **FR-9.4**: Graduated response: Warning -> Caution (reduce size) -> Stop (halt) -> Emergency (close all)

---

## 2. Non-Functional Requirements

### NFR-1: Latency
- **NFR-1.1**: Feature computation: < 1ms per bar for any single feature
- **NFR-1.2**: Full feature pipeline: < 10ms per bar
- **NFR-1.3**: Model inference (all models + stacking): < 50ms
- **NFR-1.4**: End-to-end (candle close to prediction): < 1 second
- **NFR-1.5**: API response time: < 100ms for all endpoints
- **NFR-1.6**: Order placement latency: < 200ms from signal to exchange

### NFR-2: Reliability
- **NFR-2.1**: System uptime: > 99.5% (< 44 hours downtime per year)
- **NFR-2.2**: Automatic reconnection for all WebSocket streams within 30 seconds
- **NFR-2.3**: Graceful degradation: continue with reduced confidence if one data source fails
- **NFR-2.4**: No data loss during brief disconnections (buffer and replay)
- **NFR-2.5**: Model loading fallback: if retrain fails, continue with previous model

### NFR-3: Resource Usage
- **NFR-3.1**: Memory: < 2GB RAM during normal operation
- **NFR-3.2**: CPU: single-process asyncio, no multiprocessing required for live
- **NFR-3.3**: Disk: < 10GB for 60 days of historical data in SQLite
- **NFR-3.4**: Monthly infrastructure cost: $40-100 (VPS + data feeds)
- **NFR-3.5**: No GPU required for inference (ONNX Runtime CPU)

### NFR-4: Maintainability
- **NFR-4.1**: Test coverage > 80% for feature computations
- **NFR-4.2**: Test coverage > 70% overall
- **NFR-4.3**: All public functions have type hints
- **NFR-4.4**: mypy strict mode passes with zero errors
- **NFR-4.5**: ruff lint passes with zero warnings
- **NFR-4.6**: Every module has docstrings explaining purpose and key decisions

---

## 3. Data Requirements

### DR-1: Data Sources

| Source | Data Type | Frequency | Latency | Retention |
|--------|-----------|-----------|---------|-----------|
| Binance WS | 1m OHLCV (BTC/USDT perps) | Every minute | < 100ms | 90 days |
| Binance WS | L2 order book (top 20) | 100ms | < 100ms | 7 days (snapshots at 5-min) |
| Binance WS | aggTrades | Real-time | < 100ms | 7 days |
| Binance WS | 1m OHLCV (ETH/USDT) | Every minute | < 100ms | 90 days |
| Bybit REST | Open interest | 5 minutes | < 1s | 90 days |
| Bybit REST | Funding rate | 5 minutes | < 1s | 90 days |
| Bybit WS | Liquidation events | Real-time | < 500ms | 90 days |
| yfinance REST | NQ, Gold, DXY 5-min bars | 5 minutes | 15-min delayed | 60 days |
| mempool.space WS | Whale transactions (>10 BTC) | Real-time | < 5s | 30 days |
| Alternative.me REST | Fear & Greed Index | 30 minutes | Minutes | 90 days |

### DR-2: Data Quality
- **DR-2.1**: No crossed order books (bid >= ask indicates stale data, discard)
- **DR-2.2**: OHLCV candles validated: open/close within high/low range, volume >= 0
- **DR-2.3**: Cross-source price divergence < 0.5% (Binance vs Bybit)
- **DR-2.4**: Gap detection: alert if > 2 consecutive missing candles
- **DR-2.5**: Outlier detection: flag prices > 5 standard deviations from rolling mean

### DR-3: Data Retention
- **DR-3.1**: Raw tick data (order book, trades): 7 days
- **DR-3.2**: Aggregated features: 90 days
- **DR-3.3**: Model predictions and outcomes: indefinite
- **DR-3.4**: Trade logs: indefinite
- **DR-3.5**: Backtest results: indefinite

---

## 4. Model Requirements

### MR-1: Accuracy
- **MR-1.1**: Directional accuracy > 52% on out-of-sample data (above random baseline)
- **MR-1.2**: Stacking ensemble must outperform best single model
- **MR-1.3**: Backtest Sharpe > 2.0 (to survive 50-70% degradation)
- **MR-1.4**: Live Sharpe target: > 0.8 after degradation

### MR-2: Calibration
- **MR-2.1**: Predicted probabilities must be well-calibrated (isotonic regression)
- **MR-2.2**: Reliability diagram error < 5% per confidence bin
- **MR-2.3**: Confidence must correlate with actual accuracy (positive rank correlation)

### MR-3: Retraining
- **MR-3.1**: Warm-start retrain every 2-4 hours
- **MR-3.2**: Full retrain on 14-day sliding window daily
- **MR-3.3**: Retrain must not cause prediction discontinuity > 10% change in signal distribution
- **MR-3.4**: Model regression test: < 30% prediction direction flip after retrain

### MR-4: Validation
- **MR-4.1**: Probabilistic Sharpe Ratio (PSR) > 0.95
- **MR-4.2**: Deflated Sharpe Ratio (DSR) > 0.95 (accounting for all Optuna trials)
- **MR-4.3**: Permutation test p < 0.05 (timing matters)
- **MR-4.4**: Walk-forward stability: CV of fold Sharpes < 0.5
- **MR-4.5**: Survive all historical stress scenarios within kill switch limits

---

## 5. Risk Requirements

### RR-1: Position Limits
- **RR-1.1**: Maximum position size: 5% of capital per trade (quarter-Kelly capped)
- **RR-1.2**: Maximum open positions: 1
- **RR-1.3**: Maximum trades per day: 30
- **RR-1.4**: Weekend position sizing: -30% reduction

### RR-2: Loss Limits
- **RR-2.1**: Catastrophic stop per trade: 3 ATR
- **RR-2.2**: Daily loss limit: 2-3% of capital
- **RR-2.3**: Weekly loss limit: 5% of capital
- **RR-2.4**: Maximum drawdown halt: 15% of capital
- **RR-2.5**: Consecutive loss halt: 15 consecutive losses triggers review

### RR-3: Drawdown Management
- **RR-3.1**: Progressive position reduction:
  - 3% drawdown: reduce size by 25%
  - 5% drawdown: reduce size by 50%
  - 10% drawdown: reduce size by 75%
  - 15% drawdown: halt trading completely
- **RR-3.2**: Graduated re-entry: restore position size over 5 profitable trades after drawdown recovery
- **RR-3.3**: Time-based exit: maximum holding period of 6 bars (30 minutes)

### RR-4: Kill Switches
- **RR-4.1**: All kill switches require manual reset (not auto-resume)
- **RR-4.2**: Kill switch state persisted to disk (survives restart)
- **RR-4.3**: Health endpoint reports kill switch status
- **RR-4.4**: Emergency kill switch closes all open positions immediately

### RR-5: Volatility Guards
- **RR-5.1**: Minimum annualized volatility: 15% (below = no signal, market too quiet)
- **RR-5.2**: Maximum annualized volatility: 150% (above = reduce size or abstain)
- **RR-5.3**: Trading hours: 08:00-21:00 UTC (Europe + US sessions for liquidity)

---

## 6. Testing Requirements

### TR-1: Unit Tests
- **TR-1.1**: Every feature computation has a golden dataset test (hand-verified values)
- **TR-1.2**: Every feature passes look-ahead bias detection (shuffle test + truncation test)
- **TR-1.3**: Every model can save/load round-trip without prediction change
- **TR-1.4**: Every kill switch triggers at correct threshold

### TR-2: Integration Tests
- **TR-2.1**: End-to-end pipeline: raw data -> features -> prediction (on recorded data)
- **TR-2.2**: Walk-forward auditor passes (no data leakage detected)
- **TR-2.3**: Paper trading system matches backtest results within expected degradation range

### TR-3: Validation Tests
- **TR-3.1**: Ablation study: every component shows positive marginal contribution
- **TR-3.2**: Monte Carlo: 95% CI lower bound on Sharpe > 0
- **TR-3.3**: P(ruin at 20% drawdown) < 5%
- **TR-3.4**: >50% of parameter perturbations remain profitable
- **TR-3.5**: System survives all historical stress scenarios

### TR-4: Coverage
- **TR-4.1**: Feature modules: > 90% line coverage
- **TR-4.2**: Model modules: > 80% line coverage
- **TR-4.3**: Overall project: > 70% line coverage
- **TR-4.4**: Zero untested public functions
