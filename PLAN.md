# ep2-crypto: 5-Minute BTC Prediction Engine

## Vision
A multi-edge Bitcoin price prediction system that combines on-chain signals, order flow microstructure, cross-market lead-lag, and regime-adaptive modeling — designed to be fundamentally different from the standard LSTM+RSI tutorial approach.

---

## Phase 0: Documentation Discovery (Completed)

### Verified APIs & Libraries

**Exchange Data (ccxt pro)**
- `watch_order_book(symbol, limit)` — real-time L2 order book (Binance supports limit=5,10,20,50)
- `watch_trades(symbol)` — aggTrade stream (sub-100ms latency)
- `watch_ohlcv(symbol, timeframe)` — kline streaming
- `fetch_open_interest(symbol)` / `fetch_funding_rate(symbol)` — Bybit derivatives (REST)
- Liquidations: NOT in ccxt — must use Bybit WebSocket directly (`allLiquidation.BTCUSDT`)

**On-Chain Data (free, no auth)**
- mempool.space: REST + WebSocket (`wss://mempool.space/api/v1/ws`), `track-mempool` for streaming all new txs
- Blockchair: SQL-like filtering `?q=output_total(100000000..)` for whale detection (1,440 req/day free)
- blockchain.info: `/unconfirmed-transactions`, `/balance`, `/q/*` stats endpoints
- Alternative.me: Fear & Greed Index (`GET https://api.alternative.me/fng/`)
- Exchange inflow/outflow: NO free API exists — monitor known exchange addresses as workaround

**Cross-Market Data**
- ETH: ccxt/Binance WebSocket (free, real-time, sub-100ms) — clear winner
- NQ/Gold/DXY historical: yfinance (`NQ=F`, `GC=F`, `DX-Y.NYB`) — 5-min bars, 15-min delayed, 60 days depth
- NQ/Gold near-real-time polling: Twelve Data (800 req/day free, `XAU/USD`)
- DXY: not available as native instrument anywhere free — must compute from forex components

**ML Stack**
- XGBoost: `XGBClassifier`, incremental learning via `xgb_model=`, feature importance via `get_score(importance_type='gain')`
- PyTorch GRU: `nn.GRU(input_size, hidden_size, num_layers, batch_first=True)`, sequence length 12-60 steps
- HMM: `hmmlearn.GaussianHMM(n_components=3)`, no native online learning (use `init_params=""` workaround)
- Optuna: `XGBoostPruningCallback` for XGB, `trial.report()`+`trial.should_prune()` for PyTorch
- Walk-forward: `sklearn.model_selection.TimeSeriesSplit(max_train_size=N)` for sliding window
- Feature engineering: `pandas-ta` (or `pandas-ta-classic` as fallback)

### Anti-Patterns to Avoid
- Never random-split time series data
- Never use `bidirectional=True` on GRU (uses future data)
- Never run tsfresh in live prediction loop (offline feature discovery only)
- Never trust >70% directional accuracy without checking for look-ahead bias
- ccxt `binanceusdm` has intermittent WebSocket issues in v4.4.x — test your version
- Bybit order book via ccxt only supports depths [1, 50, 200, 1000] — slice from 50 for top-10
- hmmlearn is in maintenance mode — plan for potential migration

---

## Project Structure

```
ep2-crypto/
├── pyproject.toml
├── PLAN.md
├── src/
│   └── ep2_crypto/
│       ├── __init__.py
│       ├── config.py                  # Settings, API keys, constants
│       ├── db/
│       │   ├── __init__.py
│       │   ├── schema.py              # SQLite/TimescaleDB schema
│       │   └── repository.py          # Parameterized query layer
│       ├── ingest/
│       │   ├── __init__.py
│       │   ├── base.py                # Base collector interface
│       │   ├── exchange.py            # Binance/Bybit klines, depth, trades
│       │   ├── derivatives.py         # OI, funding rate, liquidations
│       │   ├── onchain.py             # Mempool, whale txs, exchange addresses
│       │   ├── cross_market.py        # NQ, gold, DXY, ETH
│       │   └── orchestrator.py        # Manages all collectors
│       ├── features/
│       │   ├── __init__.py
│       │   ├── base.py                # Feature pipeline interface
│       │   ├── microstructure.py      # Order book imbalance, trade flow, absorption
│       │   ├── onchain.py             # Whale flow, mempool pressure, exchange balance
│       │   ├── cross_market.py        # Lead-lag, correlation, divergence signals
│       │   ├── technical.py           # TA indicators (pandas-ta), volume features
│       │   ├── temporal.py            # Cyclical time encoding (sin/cos)
│       │   └── pipeline.py            # Combines all feature sets
│       ├── regime/
│       │   ├── __init__.py
│       │   ├── detector.py            # HMM regime classification
│       │   └── labels.py              # Regime labeling utilities
│       ├── models/
│       │   ├── __init__.py
│       │   ├── xgb_direction.py       # XGBoost direction classifier
│       │   ├── gru_magnitude.py       # GRU return magnitude predictor
│       │   ├── ensemble.py            # Regime-routed ensemble
│       │   └── tuning.py              # Optuna hyperparameter optimization
│       ├── backtest/
│       │   ├── __init__.py
│       │   ├── walk_forward.py        # Walk-forward validation engine
│       │   ├── metrics.py             # Sharpe, drawdown, profit factor, accuracy
│       │   └── simulator.py           # Execution simulator with slippage/fees
│       └── api/
│           ├── __init__.py
│           └── server.py              # FastAPI prediction endpoint
├── scripts/
│   ├── collect_history.py             # Backfill historical data
│   ├── train.py                       # Full training pipeline
│   ├── backtest.py                    # Run backtests
│   └── live.py                        # Live prediction loop
├── tests/
│   ├── test_features/
│   ├── test_models/
│   ├── test_ingest/
│   └── test_backtest/
└── docker/
    ├── Dockerfile
    └── docker-compose.yml
```

---

## Phase 1: Foundation — Project Setup & Data Storage

### What to implement
1. Initialize Python project with uv, configure `pyproject.toml` with all dependencies
2. Set up SQLite database with schema for: OHLCV candles, order book snapshots, trades, on-chain events, cross-market prices, regime labels
3. Build parameterized query repository layer (no string interpolation)
4. Set up structlog logging configuration
5. Create `config.py` with environment-variable-based settings (no .env files committed)

### Dependencies
```toml
[project]
requires-python = ">=3.12"
dependencies = [
    "ccxt>=4.4",
    "websockets>=13.0",
    "pandas>=2.2",
    "numpy>=2.0",
    "xgboost>=2.1",
    "torch>=2.5",
    "hmmlearn>=0.3",
    "optuna>=4.0",
    "pandas-ta>=0.3",
    "scikit-learn>=1.5",
    "fastapi>=0.115",
    "uvicorn>=0.32",
    "structlog>=24.0",
    "httpx>=0.27",
    "aiosqlite>=0.20",
]
```

### Verification
- [ ] `uv run python -c "import ep2_crypto"` works
- [ ] Database creates tables on first run
- [ ] All SQL uses parameterized queries (grep for f-strings in db/)
- [ ] structlog outputs JSON logs

---

## Phase 2: Data Ingestion — Exchange & Derivatives

### What to implement
1. **Exchange collector** (`ingest/exchange.py`):
   - ccxt pro WebSocket: `watch_ohlcv('BTC/USDT:USDT', '1m')` for 1-min klines
   - ccxt pro WebSocket: `watch_order_book('BTC/USDT:USDT', limit=20)` for depth snapshots
   - ccxt pro WebSocket: `watch_trades('BTC/USDT:USDT')` for aggTrades
   - Store each to SQLite at configured intervals

2. **Derivatives collector** (`ingest/derivatives.py`):
   - ccxt REST: `bybit.fetch_open_interest('BTC/USDT:USDT')` — poll every 5 min
   - ccxt REST: `bybit.fetch_funding_rate('BTC/USDT:USDT')` — poll every 5 min
   - Direct Bybit WebSocket: `allLiquidation.BTCUSDT` stream — real-time
   - Connection: `wss://stream.bybit.com/v5/public/linear`, subscribe `{"op":"subscribe","args":["allLiquidation.BTCUSDT"]}`

3. **Orchestrator** (`ingest/orchestrator.py`):
   - Manages all collectors as async tasks
   - Handles reconnection, error logging, graceful shutdown
   - Health checks for each stream

### Documentation references
- ccxt pro: `watch_order_book`, `watch_trades`, `watch_ohlcv` — batch_first pattern
- Bybit liquidation: `allLiquidation` topic (NOT deprecated `liquidation` topic)
- Bybit WS heartbeat: send `ping` every 20 seconds

### Verification
- [ ] Run collectors for 10 minutes, verify data in SQLite
- [ ] Liquidation events appear when market is volatile
- [ ] OI and funding rate update every 5 minutes
- [ ] Reconnection works after simulated disconnect
- [ ] No duplicate records in database

---

## Phase 3: Data Ingestion — On-Chain & Cross-Market

### What to implement
1. **On-chain collector** (`ingest/onchain.py`):
   - mempool.space WebSocket (`wss://mempool.space/api/v1/ws`):
     - Subscribe: `{"action":"want","data":["blocks","stats"]}`
     - Subscribe: `{"track-mempool": true}` for all new txs
     - Client-side filter: flag transactions with output sum > 10 BTC as whale txs
   - Blockchair REST (polling every 60s):
     - `GET https://api.blockchair.com/bitcoin/mempool/transactions?q=output_total(1000000000..)&s=output_total(desc)`
     - Whale transactions > 10 BTC from mempool
   - Known exchange address monitoring:
     - Maintain a list of top 20 exchange cold/hot wallet addresses
     - Poll via mempool.space `/address/:address/txs` every 2 min
   - Alternative.me Fear & Greed: `GET https://api.alternative.me/fng/` — poll every 30 min

2. **Cross-market collector** (`ingest/cross_market.py`):
   - ETH: ccxt pro `watch_ohlcv('ETH/USDT:USDT', '1m')` — real-time
   - NQ futures: yfinance `yf.download('NQ=F', interval='5m', period='5d')` — poll every 5 min
   - Gold: yfinance `yf.download('GC=F', interval='5m', period='5d')` — poll every 5 min
   - DXY: yfinance `yf.download('DX-Y.NYB', interval='5m', period='5d')` — poll every 5 min
   - Note: yfinance data is ~15 min delayed — acceptable for lead-lag detection (lag is minutes-to-hours)

### Rate limit awareness
- Blockchair: max 30 req/min, 1,440/day — budget carefully
- mempool.space: no published limit, be respectful (1 WebSocket connection is fine)
- yfinance: no official limit, keep under 100 req/session

### Verification
- [ ] Whale transactions detected and stored (check against whale-alert Twitter for validation)
- [ ] Mempool pressure metrics (size, fee rates) updating in real-time
- [ ] ETH price streaming alongside BTC
- [ ] NQ/Gold/DXY data updating every 5 min (with 15-min delay noted in metadata)
- [ ] Fear & Greed index stored with timestamps

---

## Phase 4: Feature Engineering — Microstructure Edge

### What to implement
1. **Order book features** (`features/microstructure.py`):
   - Bid-ask spread: `(best_ask - best_bid) / mid_price`
   - Order book imbalance (top 5 levels): `(bid_vol - ask_vol) / (bid_vol + ask_vol)`
   - Depth imbalance: weighted by price proximity to mid
   - Book pressure: ratio of volume within 0.1% of mid on each side
   - Spread velocity: rate of change of spread over last N snapshots

2. **Trade flow features** (`features/microstructure.py`):
   - Aggressive buy/sell ratio: classify trades by taker side from aggTrade `m` field (buyer is maker = sell aggression)
   - Trade intensity: trades per second, rolling 1-min and 5-min
   - Volume delta: cumulative (aggressive_buy_vol - aggressive_sell_vol)
   - Large trade detection: flag trades > 2 std devs above mean size
   - Trade clustering: count of trades in rapid succession (< 100ms apart)

3. **Absorption detection** (`features/microstructure.py`):
   - Price doesn't move despite large aggressive orders hitting one side
   - Measure: large volume delta without proportional price change
   - Signal: potential hidden liquidity / institutional activity

### Documentation references
- Binance aggTrade `m` field: `true` = buyer is market maker (i.e., trade was a sell aggression)
- Order book from `watch_order_book`: `ob['bids']` = [[price, amount], ...], `ob['asks']` = [[price, amount], ...]

### Verification
- [ ] Order book imbalance correlates with short-term price direction (basic sanity check)
- [ ] Volume delta calculation matches manual spot-check
- [ ] Features update in real-time as new data arrives
- [ ] No look-ahead bias in feature calculations (features only use data available at calculation time)

---

## Phase 5: Feature Engineering — On-Chain & Cross-Market Edges

### What to implement
1. **On-chain features** (`features/onchain.py`):
   - Whale flow score: net BTC moved to/from exchange addresses in rolling 30-min window
   - Mempool pressure: mempool size rate of change, fee spike detection
   - Large tx momentum: count and total value of >10 BTC transactions in last 15 min
   - Exchange balance delta: change in known exchange address balances (approximation)
   - Fear & Greed momentum: rate of change of index (not raw value — everyone has raw value)

2. **Cross-market features** (`features/cross_market.py`):
   - Lead-lag correlation: rolling Pearson correlation between BTC returns and NQ/Gold/DXY returns at various lags (0, 1, 2, 5, 10 min)
   - Dynamic lead-lag detection: which asset is currently leading? Use Granger causality test on rolling window
   - Divergence signals: BTC moving opposite to its current leader = mean reversion signal
   - ETH/BTC ratio momentum: rate of change of ETH/BTC price ratio (risk appetite proxy)
   - Cross-market volatility spillover: when NQ vol spikes, does BTC vol follow? Rolling correlation of volatility

3. **Temporal features** (`features/temporal.py`):
   - Cyclical encoding: `sin(2π × minute/60)`, `cos(2π × minute/60)` for minute-of-hour
   - Same for hour-of-day, day-of-week
   - Session indicator: Asia (00-08 UTC), Europe (08-16 UTC), US (16-00 UTC)
   - Time-to-funding: minutes until next Bybit funding rate settlement (every 8h)

4. **Technical features** (`features/technical.py`):
   - Using pandas-ta: RSI(14), MACD(12,26,9), Bollinger(20,2), ATR(14), VWAP
   - Volume: OBV, volume-weighted momentum, volume profile
   - Returns: 1m, 5m, 15m, 30m log returns and volatility

5. **Feature pipeline** (`features/pipeline.py`):
   - Combines all feature sets into single DataFrame
   - Handles NaN filling (forward-fill for price, zero for counts)
   - Feature normalization (rolling z-score, not global — prevents look-ahead)
   - Outputs: feature matrix aligned to 5-min prediction intervals

### Verification
- [ ] Lead-lag correlation shows NQ leading BTC during US session (known relationship)
- [ ] Whale flow features are non-zero during known large BTC movements
- [ ] Feature pipeline produces consistent output shape
- [ ] No NaN/Inf values in output
- [ ] Rolling z-score uses only past data (no future leakage)

---

## Phase 6: Regime Detection

### What to implement
1. **HMM regime detector** (`regime/detector.py`):
   - Input features: log returns, realized volatility (12-period rolling std), volume change rate
   - Fit `GaussianHMM(n_components=3, covariance_type='full', n_iter=1000)` for 3 regimes
   - Label regimes post-hoc by characteristics:
     - Trending: high absolute returns, moderate volatility
     - Mean-reverting: low returns, low volatility
     - Volatile/chaotic: high volatility, mixed returns
   - Model selection: fit n=2,3,4,5 and select by BIC
   - Pseudo-incremental update: refit every 4 hours on sliding 7-day window with `init_params=""`

2. **Regime labels** (`regime/labels.py`):
   - Map HMM state indices to semantic labels (states are arbitrary numbers)
   - Sort by volatility: lowest vol state = mean-reverting, highest = volatile
   - Provide regime probability distribution (not just hard label)
   - Regime transition probability: extract from `model.transmat_` for "regime about to change" signal

### Verification
- [ ] 3 distinct regimes visible when plotted against price history
- [ ] Regime labels are stable (same market conditions → same regime across refits)
- [ ] Regime transitions align with known market events (e.g., CPI releases, ETF approvals)
- [ ] Regime probabilities sum to 1.0

---

## Phase 7: Models — Direction & Magnitude

### What to implement
1. **XGBoost direction classifier** (`models/xgb_direction.py`):
   - Target: 3-class classification (up > +0.1%, down < -0.1%, flat)
   - One model per regime (3 regimes × 1 XGBoost = 3 models)
   - Features: full feature pipeline output
   - Training: sliding window walk-forward, retrain every 24h on 14-day window
   - Incremental update: `model.fit(X_new, y_new, xgb_model=model.get_booster())`
   - Feature importance tracking: log top 20 features per regime per retrain

2. **GRU magnitude predictor** (`models/gru_magnitude.py`):
   - Target: predicted 5-min return magnitude (regression)
   - One model per regime (3 regimes × 1 GRU = 3 models)
   - Architecture: `GRU(input_size=N, hidden_size=64, num_layers=2, batch_first=True)` → `Linear(64, 1)`
   - Sequence length: 24 steps (2 hours of 5-min candles) — tune with Optuna
   - Training: sliding window, retrain every 24h, gradient clipping at 1.0
   - Never shuffle DataLoader

3. **Regime-routed ensemble** (`models/ensemble.py`):
   - Input: current regime (from Phase 6), feature matrix
   - Route to regime-specific XGBoost + GRU pair
   - XGBoost outputs: direction (up/down/flat) + confidence (probability)
   - GRU outputs: predicted return magnitude
   - Combined signal: `direction × magnitude × confidence`
   - Confidence threshold: only emit signal when XGBoost confidence > 0.6
   - Disagreement handling: if XGBoost says up but GRU predicts negative magnitude → no signal

### Verification
- [ ] Each regime-specific model trained on filtered data (only samples from that regime)
- [ ] Walk-forward validation shows >52% directional accuracy (above random baseline)
- [ ] Feature importance differs meaningfully across regimes
- [ ] Ensemble disagreement rate is tracked and logged
- [ ] No look-ahead bias: verify test predictions only use training data available before test timestamp

---

## Phase 8: Hyperparameter Tuning

### What to implement
1. **Optuna integration** (`models/tuning.py`):
   - XGBoost search space:
     ```python
     max_depth: suggest_int(3, 12)
     learning_rate: suggest_float(0.01, 0.3, log=True)
     min_child_weight: suggest_int(1, 250)
     subsample: suggest_float(0.5, 1.0)
     colsample_bytree: suggest_float(0.5, 1.0)
     reg_lambda: suggest_float(0.001, 25.0, log=True)
     reg_alpha: suggest_float(0.001, 25.0, log=True)
     ```
   - GRU search space:
     ```python
     hidden_size: suggest_int(32, 256)
     num_layers: suggest_int(1, 3)
     lr: suggest_float(1e-5, 1e-2, log=True)
     dropout: suggest_float(0.1, 0.5)
     seq_len: suggest_int(12, 60)
     ```
   - HMM: tune `n_components` (2-5) by BIC
   - Use `MedianPruner(n_startup_trials=5, n_warmup_steps=10)` for both
   - Use `XGBoostPruningCallback` for XGBoost specifically
   - Objective: maximize walk-forward Sharpe ratio (not just accuracy)

2. **Feature selection via tsfresh** (offline, one-time):
   - Run `extract_features` with `EfficientFCParameters` on historical data
   - Identify top statistical features (autocorrelation lags, entropy, etc.)
   - Implement top 10-15 discovered features manually in the live feature pipeline

### Verification
- [ ] Optuna study completes 50+ trials per model
- [ ] Best hyperparameters improve Sharpe over defaults
- [ ] Pruning correctly stops poor trials early
- [ ] tsfresh features that made the cut are documented

---

## Phase 9: Backtesting Framework

### What to implement
1. **Walk-forward engine** (`backtest/walk_forward.py`):
   - Custom implementation (not just sklearn TimeSeriesSplit):
     ```python
     walk_forward_validate(X, y, model, train_size=4032, test_size=288, step_size=288, gap=1)
     # 4032 = 14 days of 5-min candles, 288 = 1 day, gap=1 to prevent leakage
     ```
   - Sliding window (not expanding) — crypto regimes shift too fast
   - Re-detect regime at each step
   - Retrain regime-specific models at each step

2. **Execution simulator** (`backtest/simulator.py`):
   - Slippage model: 0.01-0.05% per trade (configurable)
   - Transaction fees: 0.04% maker / 0.06% taker (Binance futures)
   - Latency simulation: 50-200ms delay between signal and execution
   - Position sizing: Kelly criterion or fixed fractional
   - No trading during regime transitions (confidence < threshold)

3. **Metrics** (`backtest/metrics.py`):
   - Directional accuracy (per regime)
   - Sharpe ratio (annualized)
   - Max drawdown
   - Profit factor
   - Win rate, average win/loss ratio
   - Calmar ratio
   - Trade count per day
   - Regime accuracy (predicted vs actual)

### Verification
- [ ] Backtest on 30+ days of historical data
- [ ] Results include transaction costs (no "zero-cost" delusion)
- [ ] Sharpe > 0 after costs (basic sanity)
- [ ] Compare against baselines: buy-and-hold, momentum, random
- [ ] Metrics breakdown per regime shows which regimes are profitable

---

## Phase 10: API & Live Prediction

### What to implement
1. **FastAPI server** (`api/server.py`):
   - `GET /predict` — returns current prediction (direction, magnitude, confidence, regime)
   - `GET /health` — checks all data streams are alive, models are loaded, last prediction timestamp
   - `GET /metrics` — live accuracy tracking, rolling Sharpe
   - `GET /regime` — current regime + transition probabilities
   - All responses include timestamps and staleness warnings

2. **Live prediction loop** (`scripts/live.py`):
   - Async event loop running all collectors
   - Feature pipeline triggers on each new 5-min candle close
   - Regime detection updates
   - Model inference → prediction
   - Store prediction + actual outcome for live accuracy tracking
   - Auto-retrain models every 24h on latest data

### Verification
- [ ] API responds in < 100ms
- [ ] Health endpoint actually checks dependencies (not just returns "ok")
- [ ] Predictions emit within 1 second of candle close
- [ ] Live accuracy tracking matches backtest expectations (within reason)
- [ ] Graceful degradation: if one data source fails, system still produces predictions (with lower confidence)

---

## Phase 11: Historical Backfill & Full Validation

### What to implement
1. **Historical data collection** (`scripts/collect_history.py`):
   - Binance REST: fetch 30-60 days of 1-min OHLCV via `exchange.fetch_ohlcv()`
   - Bybit REST: backfill OI history (`fetch_open_interest_history`) and funding rate history
   - yfinance: backfill NQ, Gold, DXY 5-min data (limited to 60 days)
   - Note: order book and trade-level data cannot be backfilled — microstructure features start from collection date

2. **Full system validation**:
   - Run complete pipeline: ingest → features → regime → models → backtest
   - Compare regime detection against known market events
   - Validate that each edge contributes independently (ablation study):
     - Model with all features vs model without microstructure
     - Model with all features vs model without on-chain
     - Model with all features vs model without cross-market
     - Model with all features vs model without regime routing

### Verification
- [ ] Each edge shows positive marginal contribution in ablation study
- [ ] Full system Sharpe > any single-edge system
- [ ] No edge has negative contribution (remove if so)
- [ ] System handles missing data gracefully (e.g., no on-chain data for first 30 days)

---

## Execution Order & Dependencies

```
Phase 1 (Foundation)
  ↓
Phase 2 (Exchange Ingestion) ──────┐
  ↓                                 │
Phase 3 (On-Chain + Cross-Market) ─┤
  ↓                                 │
Phase 4 (Microstructure Features) ──┤── can partially parallelize
  ↓                                 │
Phase 5 (On-Chain + Cross Features)┘
  ↓
Phase 6 (Regime Detection)
  ↓
Phase 7 (Models)
  ↓
Phase 8 (Tuning)
  ↓
Phase 9 (Backtesting)
  ↓
Phase 10 (API + Live)
  ↓
Phase 11 (Validation + Ablation)
```

Phases 2-5 can overlap: start collecting data in Phase 2-3 while building features in Phase 4-5. Models (Phase 7) need data from all feature phases.

---

## Key Design Decisions

1. **Sliding window, not expanding** — crypto non-stationarity means old data hurts more than helps
2. **Per-regime models** — a trending market model is useless in a ranging market
3. **Confidence gating** — no signal is better than a low-confidence signal
4. **Magnitude + direction separation** — XGBoost excels at classification, GRU at regression
5. **15-min delay on cross-market is acceptable** — the lead-lag relationship operates on minutes-to-hours, not seconds
6. **No exchange flow data** — no free API exists; we approximate via address monitoring
7. **SQLite for dev** — zero setup, switch to TimescaleDB when moving to production
8. **Feature normalization via rolling z-score** — prevents global normalization look-ahead bias
