# Research Synthesis: Key Findings & Plan Revisions

> 20 research agents, 21 reports, ~500K words of analysis. This document distills the critical findings that change how we build the system.

---

## Critical Reality Checks

### The Hard Numbers
- **Realistic directional accuracy**: 52-56% at 5-min (not 60%+)
- **Backtest-to-live degradation**: 50-70% (a Sharpe 2.0 backtest → Sharpe 0.6-1.0 live)
- **Round-trip costs**: 8-12 bps minimum (taker fee + slippage)
- **Minimum gross return per trade**: >15 bps to be profitable after costs
- **Alpha half-life in crypto**: 1-6 months for ML-based patterns
- **Optimal feature count**: 18-25 (not 200+)
- **Monthly infra cost**: $40-100 (minimum viable)

### What We Got Wrong in V1 Plan
1. **XGBoost → LightGBM** — LightGBM outperforms XGBoost on 5-min crypto by 0.5-1.5%
2. **LSTM → GRU + TCN** — GRU trains 20-40% faster, comparable accuracy. TCN is parallelizable and competitive.
3. **3 regime-specific models → hierarchical ensemble** — regime detection should be a feature, not model routing (too few samples per regime)
4. **On-chain as primary signal → on-chain as regime context only** — academic evidence shows on-chain signals operate at 4h-24h, NOT 5-min
5. **Sentiment analysis → skip for 5-min** — too slow. Funding rate is the only useful "sentiment" at this timeframe
6. **Most TA indicators → noise at 5-min** — MACD, Bollinger, Ichimoku, etc. are useless. Only RSI (with regime filter), short ROC, and ATR survive

---

## 5 Highest-Impact Changes to the Plan

### 1. Core Model: LightGBM + Stacking Ensemble (not XGBoost per-regime)

**Old plan**: 3 XGBoost classifiers (one per regime) + 3 GRU regressors
**New plan**:
- LightGBM (primary classifier) + CatBoost (secondary) + GRU (temporal features)
- Stacking meta-learner (logistic regression) combines them
- Single model set, not per-regime (regime becomes a feature input)
- Add LightGBM quantile regression (5 quantiles) for risk assessment

**Why**: Per-regime models fragment already-scarce data. A single ensemble with regime features is more robust. Stacking is the #1 gain after feature engineering (Kaggle consensus + financial ML literature).

### 2. Feature Priority: Microstructure First, Everything Else Second

**Research consensus**: OFI + OBI + microprice capture 60-80% of achievable 5-min signal.

**Tier 1 (must-have, ~15 features)**:
- Order book imbalance (weighted, levels 1-3 and 1-5)
- Trade flow imbalance (30s and 5min windows)
- Bid-ask spread (relative)
- Volume delta (1-bar, 5-bar)
- Volume rate of change
- VWAP deviation (1hr)
- Realized vol (5-min, 30-min)
- Parkinson vol (30-min)
- ROC (1, 3, 6, 12 bars)

**Tier 2 (strong additions, ~10 features)**:
- Book depth ratio, pressure gradient
- Trade arrival rate
- EWMA volatility, vol-of-vol
- RSI(14), linear regression slope(20)
- Price quantile rank(60)

**Tier 3 (context, ~5 features)**:
- Hour-of-day (sin/cos)
- Funding rate
- Spread × OBI interaction
- NQ 5-min returns (lagged 1-3 bars) — during US hours only
- VIX level (regime classifier)

### 3. Confidence Gating Pipeline (2-4x Sharpe improvement)

The full gating pipeline is the single biggest Sharpe multiplier:

```
Raw Prediction → Calibration (isotonic) → Meta-Labeling Gate → Ensemble Agreement Check → Conformal Prediction Gate → Signal Filters (vol, regime, liquidity) → Adaptive Threshold → Drawdown Gate → Position Sizing (quarter-Kelly) → EXECUTE or ABSTAIN
```

Key layers:
- **Meta-labeling**: Secondary model predicts "will THIS trade be profitable?" — Lopez de Prado's technique, 50-100% Sharpe improvement
- **Conformal prediction**: Prediction set must be singleton {UP} or {DOWN}, not {UP, DOWN}
- **Adaptive threshold**: Regime-dependent, updated weekly via rolling Sharpe optimization
- **Drawdown gate**: Progressive position reduction 3%→15%, graduated re-entry

### 4. Regime Detection: Hierarchical Ensemble (not just HMM)

**Old plan**: Single 3-state HMM
**New plan**: 4-layer hierarchy

| Layer | Method | Update Frequency | Purpose |
|-------|--------|-----------------|---------|
| Fast (every bar) | Efficiency Ratio + GARCH vol | O(1) | Quick regime estimate |
| Core (every bar) | 2-state HMM (filtered probs) + BOCPD | O(K²) | State identification + change detection |
| Slow (hourly) | Rolling Hurst (24h DFA) | O(window) | Trending vs mean-reverting context |
| Meta (weekly refit) | LightGBM on all above | Batch | Optimal regime label |

**Why**: BOCPD detects changes 2-10 bars faster than HMM alone. Efficiency Ratio is the highest signal-to-complexity feature for trend vs range. Hurst provides macro context.

### 5. Event-Driven Macro Module (Highest Single Edge)

**New finding**: NQ reaction to macro events (CPI, FOMC) leads BTC by 2-5 minutes with ~62-68% accuracy.

```
T+0s:     CPI/FOMC released
T+0-60s:  NQ/ES reacts
T+2-5m:   BTC follows NQ direction (~65% of the time)
```

**Implementation**:
- Pre-scheduled macro calendar
- At T+0: monitor NQ 5-min return direction
- At T+2min: enter BTC in NQ's direction
- At T+5-10min: exit (mean reversion risk)
- VIX-gated: skip when VIX > 35

This is orthogonal to the ML model and can run as an independent alpha source.

---

## Revised Architecture

```
┌─────────────────────────────────────────────────────┐
│                   DATA LAYER                         │
├─────────────────────────────────────────────────────┤
│ Binance WS: klines, depth@100ms, aggTrades          │
│ Bybit WS: allLiquidation, funding (REST)            │
│ Cross-market: NQ (yfinance/TwelveData), ETH (ccxt)  │
│ On-chain: mempool.space WS (whale detection only)    │
│ Macro: economic calendar (scheduled events)          │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│               FEATURE ENGINE                         │
├─────────────────────────────────────────────────────┤
│ Microstructure: OBI, OFI, microprice, spread, TFI   │
│ Volume: delta, VWAP dev, rate-of-change             │
│ Volatility: realized, Parkinson, EWMA, vol-of-vol   │
│ Momentum: ROC(1,3,6,12), RSI, linreg slope          │
│ Context: funding rate, hour encoding, NQ returns     │
│ Regime: Efficiency Ratio, GARCH vol, HMM probs      │
│                                                      │
│ Normalization: dual pipeline                         │
│   Tree path: raw features (no normalization)         │
│   Neural path: robust scaling + rank-to-gaussian     │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│                 MODEL LAYER                          │
├─────────────────────────────────────────────────────┤
│ LightGBM classifier (direction: up/flat/down)       │
│ CatBoost classifier (ordered boosting, diversity)   │
│ GRU → hidden state as features for LightGBM         │
│ LightGBM quantile regression (5 quantiles: risk)    │
│ Stacking meta-learner (logistic regression)          │
│                                                      │
│ Target: ternary with adaptive threshold              │
│ Validation: purged walk-forward (sliding window)     │
│ Retraining: warm-start every 2-4 hours               │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│              CONFIDENCE GATING                       │
├─────────────────────────────────────────────────────┤
│ 1. Isotonic regression calibration                   │
│ 2. Meta-labeling gate (is this trade profitable?)    │
│ 3. Ensemble agreement check (variance < threshold)   │
│ 4. Conformal prediction (singleton prediction set)   │
│ 5. Signal filters (vol range, regime stability)      │
│ 6. Adaptive confidence threshold (regime-dependent)  │
│ 7. Drawdown gate (progressive 3%→15%)                │
│ 8. Position sizing (quarter-Kelly × confidence)      │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│               EXECUTION                              │
├─────────────────────────────────────────────────────┤
│ Aggressive limit order (1-2 ticks inside)            │
│ 10s timeout → convert to market order                │
│ 5-30s entry delay for better fills                   │
│ Time-based exit (max 6 bars = 30 min)                │
│ Catastrophic stop at 3 ATR                           │
│ Signal decay model (half-life ~3 min)                │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│         EVENT-DRIVEN MACRO MODULE                    │
├─────────────────────────────────────────────────────┤
│ Independent alpha source (not ML-based)              │
│ CPI/FOMC calendar → watch NQ for 60-120s            │
│ Enter BTC in NQ direction at T+2min                  │
│ Exit at T+5-10min                                    │
│ VIX-gated (skip when VIX > 35)                       │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│           LIQUIDATION CASCADE DETECTOR               │
├─────────────────────────────────────────────────────┤
│ Independent risk module (not for direction)           │
│ Monitors: OI percentile, funding z-score,            │
│   liquidation burst rate, book depth, price velocity │
│ Hawkes process for cascade intensity                 │
│ Output: cascade probability → reduce position size   │
└─────────────────────────────────────────────────────┘
```

---

## Revised Tech Stack

| Component | V1 Plan | V2 (Post-Research) | Why Changed |
|-----------|---------|-------------------|-------------|
| Primary model | XGBoost | **LightGBM** | Faster, +0.5-1.5% accuracy on crypto |
| Secondary model | — | **CatBoost** | Ordered boosting prevents leakage, diversity |
| Sequence model | GRU | **GRU + TCN** | TCN is parallelizable, competitive |
| Ensemble | Simple combination | **Stacking** | 2-5% over best single model |
| Regime detection | 3-state HMM | **Hierarchical** (HMM + BOCPD + GARCH + ER) | Faster detection, more robust |
| Feature library | pandas-ta | **numpy-native** | 10-50x faster on hot path |
| Normalization | Rolling z-score | **Dual pipeline** (raw for trees, robust for neural) | Trees harmed by normalization |
| Target | Binary (up/down) | **Ternary** (up/flat/down) with adaptive threshold | Acknowledges untradeable periods |
| Calibration | — | **Isotonic + Venn-Abers** | Required for confidence gating |
| Position sizing | Fixed fractional | **Quarter-Kelly × confidence** | Research-backed optimal sizing |
| Sentiment | Twitter NLP + Fear&Greed | **Funding rate only** | All other sentiment too slow for 5-min |
| On-chain | Primary signal | **Regime context only** | Operates at 4h-24h, not 5-min |
| Data structures | pandas DataFrames | **Numpy ring buffers** | 10-20x less memory |
| Tick storage | SQLite | **SQLite dev → TimescaleDB prod** | (unchanged, confirmed by research) |
| Model inference | Native Python | **ONNX Runtime** | 2-5x speedup |

---

## Revised Phase Plan

### Phase 1: Foundation (unchanged)
Project setup, uv, SQLite, structlog, config via Pydantic Settings + Doppler.

### Phase 2: Data Ingestion — Exchange & Derivatives
- Add Binance `!forceOrder@arr` for liquidation stream (not just Bybit)
- Add Binance `depth@100ms` (not just depth20)
- Collect aggTrades for trade flow features

### Phase 3: Data Ingestion — Cross-Market & Events
- **Downgrade on-chain**: Mempool monitoring for whale detection only (>10 BTC to known exchange)
- **Add macro calendar**: Pre-scheduled CPI/FOMC/NFP dates with NQ monitoring
- **Add VIX**: yfinance or Twelve Data for VIX level (regime classifier)
- **ETH**: ccxt WebSocket (unchanged)

### Phase 4: Feature Engineering — Microstructure (HIGHEST PRIORITY)
- OBI (weighted, multi-level)
- OFI (Cont-Stoikov-Talreja model, multi-level)
- Microprice (Gatheral-Stoikov)
- Trade flow imbalance
- Absorption detection
- Kyle's Lambda (rolling, as conditioning variable)

### Phase 5: Feature Engineering — Volume, Volatility, Momentum
- Volume delta (buy-sell), VWAP deviation, rate-of-change
- Realized vol, Parkinson, EWMA, vol-of-vol
- ROC at multiple lookbacks (1, 3, 6, 12 bars)
- RSI(14), linear regression slope(20), price quantile rank
- Dual normalization pipeline (raw for trees, robust scaling for neural)

### Phase 6: Regime Detection — Hierarchical Ensemble
- Efficiency Ratio (fast, every bar)
- GJR-GARCH(1,1)-t conditional vol (fast, every bar)
- 2-state GaussianHMM (core, every bar, refit weekly)
- BOCPD change point detection (early warning)
- Rolling Hurst (slow, hourly)
- LightGBM meta-learner on regime features

### Phase 7: Models — Stacking Ensemble
- LightGBM classifier (ternary target, adaptive threshold)
- CatBoost classifier (diversity through ordered boosting)
- GRU feature extractor → hidden state feeds into LightGBM
- LightGBM quantile regression (10th, 25th, 50th, 75th, 90th)
- Stacking meta-learner (logistic regression on OOF predictions)
- Isotonic regression calibration on held-out calibration set

### Phase 8: Confidence Gating Pipeline
- Meta-labeling (primary ensemble → meta-model → trade gate)
- Ensemble agreement scoring
- Conformal prediction (adaptive α)
- Signal filters (vol range, regime stability, liquidity, news events)
- Adaptive confidence threshold (regime-dependent, weekly update)
- Drawdown gate (progressive reduction)
- Quarter-Kelly position sizing × composite confidence

### Phase 9: Hyperparameter Tuning
- Optuna with MedianPruner
- Objective: walk-forward Sharpe (not accuracy)
- Deflated Sharpe Ratio to account for multiple trials
- Feature importance stability check across folds

### Phase 10: Backtesting Framework
- Walk-forward with purging and embargo (gap = max feature lookback)
- Sliding window (not expanding) — 30-90 day training window
- Execution simulator: slippage model (sqrt-impact), maker/taker fees, 200ms latency
- Kelly criterion with uncertain probabilities for position sizing
- Monte Carlo bootstrap (10K simulations, 95% CI on Sharpe)
- Regime-specific metrics (must be profitable or flat in all regimes)
- Time-of-day analysis (restrict to 08:00-21:00 UTC)
- **Target backtest Sharpe > 2.0** (to survive 50-70% degradation)

### Phase 11: Event-Driven Macro Module
- Economic calendar integration
- NQ real-time monitoring via yfinance/Twelve Data
- EWMA BTC-NQ correlation (trailing 6h)
- VIX-dependent NQ signal weighting
- Independent execution from ML model

### Phase 12: Liquidation Cascade Detector
- Multi-factor cascade score (OI, funding, liq rate, book depth, price velocity)
- Hawkes process for liquidation intensity
- Output: cascade probability → reduce position size
- Liquidation heatmap estimation from volume profile + leverage assumptions

### Phase 13: API, Live Prediction & Monitoring
- FastAPI endpoints (predict, health, metrics, regime)
- Prometheus metrics + Grafana dashboards
- Alpha decay monitor (CUSUM, rolling Sharpe, hit rate)
- Feature drift detection (PSI > 0.2 alert)
- Automated kill switches (daily loss, max drawdown, consecutive losses)

### Phase 14: Paper Trading & Validation (4-8 weeks)
- Go/no-go criteria:
  - Paper Sharpe > 1.0 after costs
  - Max drawdown < 15%
  - Win rate > 50%
  - No single day > 30% of total profit
  - Performance stable across all weeks
  - Model confidence correlates with actual accuracy

### Phase 15: Ablation Study
- Full system vs without microstructure features
- Full system vs without cross-market (NQ) features
- Full system vs without regime detection
- Full system vs without confidence gating
- Full system vs without macro event module
- Each component must show positive marginal contribution or be removed

---

## Risk Parameters (Starting Config)

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

---

## Research Files Index

All 21 research reports are in [research/](research/). Key files by topic:

| Topic | Key Finding |
|-------|-------------|
| Order book microstructure | OFI + OBI + microprice = 60-80% of signal |
| Liquidation cascades | Hawkes process model, specific funding thresholds |
| Whale detection | On-chain useful at 4h+, mempool monitoring for 5-min |
| Cross-market lead-lag | NQ leads BTC by 2-5 min during macro events (65% accuracy) |
| Regime detection | Hierarchical ensemble (BOCPD + HMM + GARCH + ER) |
| Model architectures | LightGBM > XGBoost, skip RL/TabNet, stacking is #1 ensemble |
| Feature engineering | 18-25 optimal features, most TA indicators are noise |
| Overfitting prevention | Deflated Sharpe, meta-labeling, purged walk-forward |
| System architecture | asyncio, numpy ring buffers, ONNX Runtime |
| Backtesting pitfalls | Expect 50-70% degradation, need Sharpe > 2.0 in backtest |
| Sentiment | Skip real-time NLP, funding rate is the only useful signal |
| Production systems | Start simple, $40-100/mo, Hetzner for hosting |
| Lopez de Prado (AFML) | Triple barrier, meta-labeling, fractional diff, CPCV |
| Data normalization | Dual pipeline, ternary target, never SMOTE, never interpolate |
| Alternative data | Options GEX marginal, stablecoin flows daily signal |
| Confidence gating | Full pipeline = 2-4x Sharpe improvement |
| Execution | Aggressive limits with market fallback, 5-30s entry delay |
