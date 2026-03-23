# Comprehensive Backtesting Plan

> Synthesized from 40+ research agents, 4MB of analysis, covering every aspect of validating a 5-min BTC prediction system.

---

## 1. Backtesting Framework Architecture

### Engine: Custom Event-Driven + Vectorized Hybrid

| Phase | Engine | Purpose | Speed |
|-------|--------|---------|-------|
| Signal research | Vectorized (numpy) | Fast iteration on features | 0.1s/year |
| Strategy validation | Event-driven (Numba-compiled) | Realistic fills, path-dependent | <3s/year |
| Production backtest | Event-driven + L2 replay | Gold standard | ~30s/year |

### Core Design Principles
- **Next-bar-open execution** — never fill at current bar close
- **Numba-compiled inner loop** — 100x faster than pure Python
- **Separate RNG streams** — fill simulation, slippage, strategy each get independent seeded RNG
- **Full state persistence** — positions, margin, funding, liquidation levels tracked every bar
- **Parquet output** — equity curves, trade logs, feature snapshots stored for analysis

### Order Fill Models

| Order Type | Simulation Method | Key Parameters |
|------------|-------------------|----------------|
| Market | Walk L2 book, take max 25% per level | slippage_bps=1-3, taker_fee=4bps |
| Limit | Probability model (70% at touch, 95% through) | adverse_selection=2bps, queue_position |
| Stop | Fill between stop and bar low/high (gap risk) | gap_fill=midpoint of stop and bar extreme |
| Partial | Cap at 10% of bar volume | participation_rate=0.10 |

### Multi-Asset Timeline Merger
NQ data arrives 15-min delayed. The backtest timeline presents bars in the order your system would ACTUALLY receive them:
```
14:00 BTC bar close → process immediately
14:05 BTC bar close → process immediately
14:05 NQ 13:45-14:00 bar arrives (15-min delay) → now available
```

---

## 2. Walk-Forward Validation Protocol

### Configuration (5-min BTC)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Training window | 14 days (4,032 bars) | Recency premium in crypto; >21 days dilutes signal |
| Test window | 1 day (288 bars) | Sufficient for daily evaluation |
| Step size | 1 day (288 bars) | Daily retraining cadence |
| Purge size | 18 bars (90 min) | label_horizon(6) + feature_lookback(12) |
| Embargo size | 12 bars (60 min) | 2x label horizon |
| Window scheme | Sliding (NOT expanding) | Crypto non-stationarity makes old data harmful |

### Nested Walk-Forward for Hyperparameters
- **Outer loop**: strategy evaluation (unbiased OOS performance)
- **Inner loop**: hyperparameter selection (3 inner folds, 20% validation ratio)
- Inner validation carved from END of outer training set with purge gap
- Best params from inner loop → retrain on full outer training set → predict outer test

### Critical Anti-Leakage Rules
1. Early stopping validation set must come from TRAINING data (with purge gap), never test data
2. Feature selection happens WITHIN each fold's training set only
3. Normalization (StandardScaler) fit on training data only per fold
4. GRU scaler fit on training data only per fold
5. All features use only data at times ≤ t (verified by shuffle test)

### Walk-Forward Auditor (run before every backtest)
Automated checks for: train/test overlap, purge sufficiency, temporal ordering, fold sizes, label distribution shift, early stopping leakage warnings.

---

## 3. Statistical Validation Suite

### Required Tests (Must Pass All Before Deployment)

| Test | What It Proves | Threshold | Min Sample |
|------|---------------|-----------|------------|
| **Concatenated OOS Sharpe** | Strategy has edge after costs | CI lower bound > 0 | 50K bars |
| **Probabilistic Sharpe Ratio (PSR)** | Sharpe exceeds benchmark accounting for non-normality | PSR > 0.95 | 50K bars |
| **Deflated Sharpe Ratio (DSR)** | Sharpe survives multiple testing correction | DSR > 0.95 | 50K bars, after N trials |
| **Permutation test** | Timing matters (not just position bias) | p < 0.05 | 5K permutations |
| **Binomial win rate test** | Win rate exceeds break-even after costs | p < 0.01 (Wilson CI) | 1K trades |
| **Walk-forward stability** | Sharpe consistent across folds | CV of fold Sharpes < 0.5 | 20+ folds |

### Supplementary Tests (Strongly Recommended)

| Test | What It Proves | When to Use |
|------|---------------|-------------|
| **Hansen's SPA** | Best strategy beats benchmark vs data mining | After comparing multiple strategies |
| **PBO (CSCV)** | Probability of backtest overfitting | After extensive parameter search |
| **Bootstrap CI (block)** | Confidence intervals on all metrics | Always |
| **Power analysis** | Have enough data to detect the edge | Before committing to backtest |

### Full Validation Pipeline
Run all tests and produce verdict: **GENUINE_EDGE** / **INCONCLUSIVE** / **LIKELY_NOISE**
- GENUINE_EDGE: majority of tests pass
- LIKELY_NOISE: majority fail
- Requires 900-line `statistical_validation_tests.py` module (implemented)

---

## 4. Monte Carlo Robustness Testing

### Required Monte Carlo Analyses

| Analysis | Method | Iterations | What It Reveals |
|----------|--------|------------|-----------------|
| **Bootstrap equity curves** | Block bootstrap (block=T^(1/3)) | 10K | 95% CI on Sharpe, max DD, return |
| **Return shuffling** | Permute trade order | 5K | Whether timing adds value |
| **Parameter sensitivity** | Perturb params ±10-20% | 1K | Fraction remaining profitable |
| **Drawdown distribution** | Bootstrap max drawdowns | 10K | P(ruin) at various DD thresholds |
| **Ruin probability** | Simulate N-trade paths | 10K | P(losing X% of capital) |

### Stress Testing Scenarios

**Historical replays** (must survive all):
- March 2020 COVID crash (BTC -50% in 2 days)
- May 2021 China ban (-30% in hours)
- November 2022 FTX collapse (-25% over 5 days)
- March 2024 new ATH euphoria then -15% correction

**Synthetic stress tests**:
- 48h of zero volatility (costs accumulate, no alpha)
- 10 flash crashes in 1 week
- BTC-NQ correlation drops to 0 overnight
- Funding rate at +0.3% for 2 weeks
- WebSocket drops for 5 minutes mid-position
- Model predicts same direction 100 bars straight (broken model detection)

### Pass/Fail Criteria
- 95% CI lower bound on Sharpe > 0
- P(ruin at 20% drawdown) < 5%
- >50% of parameter perturbations remain profitable
- System survives all historical stress scenarios without exceeding kill switches

---

## 5. Transaction Cost Model

### Cost Components (Binance BTCUSDT Perps)

| Component | Taker-Only | Optimized (Hybrid) |
|-----------|-----------|-------------------|
| Exchange fee | 4.0 bps | 2.4 bps (60% maker) |
| Slippage | 1-3 bps | 0.5-1.5 bps |
| Market impact | 0.1-0.5 bps | 0.1-0.3 bps |
| Latency cost | 0.1-0.5 bps | 0.1-0.3 bps |
| **Round trip** | **10-16 bps** | **6-9 bps** |

### Cost Sensitivity Analysis
Run backtest at costs = 0, 4, 8, 12, 16, 20 bps round-trip.
- **Break-even cost**: the cost level where Sharpe = 0
- **Target**: break-even cost > 15 bps (healthy margin above realistic costs)

### Funding Rate Model
- Apply at 00:00, 08:00, 16:00 UTC exactly
- Use historical funding rate series, not a constant
- Track cumulative funding as a separate P&L component
- For 5-min system: usually close before funding, but model explicitly

---

## 6. Benchmark Strategies (Must Beat All)

| Benchmark | Expected Sharpe (5-min BTC) | Why It's Hard |
|-----------|---------------------------|---------------|
| **Buy and Hold** | 0.8-1.2 (bull), negative (bear) | Free beta |
| **5-bar Momentum** | 0.3-0.8 (before costs), often negative after | Same signal you're likely using |
| **RSI Mean Reversion** | 0-0.5 | Simple, low turnover |
| **Random Entry** | ~0 | True null hypothesis |
| **Funding Rate Carry** | 1.0-2.0 | Surprisingly strong baseline |
| **Oracle (perfect foresight)** | Upper bound after costs | Maximum achievable |

Your ML system must show **statistically significant outperformance** over the hardest benchmark (5-bar momentum) using a paired t-test on per-trade returns.

---

## 7. Performance Metrics Dashboard

### Minimal Dashboard (7 Metrics)

| Metric | Formula | Annualization | Good Threshold |
|--------|---------|---------------|----------------|
| **Sharpe (Lo-corrected)** | μ/σ × √105,120 with ACF correction | √105,120 ≈ 324.2 | > 1.5 backtest, > 0.8 live |
| **Sortino** | μ/DD × √105,120 | Same | > 2.0 |
| **Max Drawdown + Duration** | Peak-to-trough of equity curve | N/A | < 10% backtest |
| **CVaR (5%)** | Mean of worst 5% of returns | Per-bar | < 0.5% per bar |
| **Expectancy/trade** | WR×avg_win + (1-WR)×avg_loss | Per-trade | > 3 bps after costs |
| **Profit Factor** | Σ(wins) / |Σ(losses)| | Per-trade | > 1.3 |
| **Rolling Sharpe (30d)** | Sharpe on trailing 30 days | Annualized | Stable, > 0.5 |

### Key: Annualization for 24/7 Crypto
- **bars_per_year** = 288 × 365 = **105,120**
- **sqrt(bars_per_year)** = **324.22**
- Do NOT use 252 trading days — crypto trades 24/7/365

---

## 8. Performance Attribution

### PnL Decomposition (daily review)
- **Signal attribution**: waterfall chart — which signals (OBI, OFI, NQ lead-lag, etc.) contributed how much PnL
- **Regime decomposition**: PnL by regime (trending/ranging/volatile)
- **Time-of-day**: hourly PnL heatmap (identify profitable sessions)
- **Confidence gating value**: PnL saved by abstaining vs PnL from taken trades
- **Alpha vs beta**: regression vs buy-and-hold — what fraction of return is skill vs market exposure

### Drawdown Diagnosis
When drawdowns occur, automatic diagnosis:
1. Hit rate decline? → model accuracy degradation
2. Feature drift (PSI > 0.2)? → market microstructure changed
3. Regime changed? → model suited for old regime, not new
4. Many small losses vs few large? → systematic failure vs tail event

### Marginal Contribution (ablation)
Leave-one-out analysis: remove each component and measure impact
- Without microstructure features → PnL impact
- Without cross-market features → PnL impact
- Without regime detection → Sharpe impact
- Without confidence gating → Sharpe impact
Each component must show positive marginal contribution or be removed.

---

## 9. Alpha Decay Detection (Live Monitoring)

### Detection Methods
| Method | Speed | False Positive Rate | Use For |
|--------|-------|-------------------|---------|
| CUSUM on returns | Fast (2-5 bars) | ~15% | Early warning |
| Rolling Sharpe decline | Medium (7-14 days) | ~5% | Confirmation |
| Feature PSI drift | Medium (daily) | ~10% | Root cause |
| ADWIN (River library) | Adaptive | ~8% | Online detection |
| SPRT on win rate | Fast (50 trades) | Controlled | Statistical certainty |

### Response Protocol
| Alert Level | Trigger | Action |
|-------------|---------|--------|
| **Warning** | 1 detector fires | Log, continue monitoring |
| **Caution** | 2 detectors fire | Reduce position size by 50% |
| **Stop** | 3+ detectors fire OR rolling Sharpe < -0.5 | Halt trading, trigger retrain |
| **Emergency** | Max drawdown hit (10-15%) | Close all positions, manual review |

---

## 10. Paper Trading Protocol

### Architecture
- Exchange Simulator pattern: `ExchangeInterface` implemented by both `PaperExchange` and `LiveExchange`
- 100% code path parity — only execution layer swaps
- Real market data feed for paper trading (never synthetic)
- Fill simulation walks the actual orderbook

### Go/No-Go Criteria (before ANY live capital)

| Criterion | Threshold |
|-----------|-----------|
| Calendar days | ≥ 14 |
| Total trades | ≥ 200 |
| Win rate | > 51% |
| Sharpe (annualized) | > 1.0 |
| Max drawdown | < 8% |
| Profit factor | > 1.2 |
| Avg trade PnL | > 2 bps after fees |
| Consecutive losses | < 15 |
| Profitable regimes | ≥ 2 of 3 |
| Model confidence correlation | > 0.1 |
| No single day > 30% of total PnL | Required |

### Graduated Rollout to Live

| Stage | Capital | Duration | Key Criterion |
|-------|---------|----------|---------------|
| 1. Minimum Viable | 5% | 7-14 days | Integration works, zero critical errors |
| 2. Validation | 15% | 7-21 days | Performance within expected range |
| 3. Scale Test | 35% | 7-14 days | Slippage doesn't increase with size |
| 4. Near Full | 65% | 7-14 days | Slippage stable |
| 5. Full Production | 100% | Ongoing | Continuous monitoring with kill switches |

**Total timeline: ~8-13 weeks from paper start to full production**

---

## 11. Data Pipeline Validation

### Priority Testing Order
1. **Look-ahead bias tests** — shuffle test + truncation test + future data injection test
2. **Candle + order book validation** — Pandera schemas, structural integrity checks
3. **Historical data validation** — gaps, duplicates, outliers, impossible values
4. **Feature unit tests + golden datasets** — verify computations against hand-verified outputs
5. **Data freshness monitoring** — per-source staleness tracking with configurable thresholds
6. **Feature drift monitoring (PSI)** — Population Stability Index per feature, alert at PSI > 0.2
7. **Integration tests** — end-to-end replay of recorded market data
8. **Model regression tests** — <30% prediction direction flip after retraining

### Automated Checks (run on every backtest and in production)
- No NaN/Inf in feature output after warmup period
- All features use only past data (verified by truncation test)
- Cross-source price divergence < 0.5% (Binance vs Bybit)
- Order book: bids < asks, sorted correctly, no negative quantities
- No crossed books (bid ≥ ask = stale data)

---

## 12. Model Training Specifics

### LightGBM (Primary)
- Objective: `multiclass` (ternary: up/flat/down) with adaptive threshold
- Key params: `num_leaves=31, max_depth=5, learning_rate=0.05, min_child_samples=50, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1`
- Early stopping: patience=50, on validation carved from training with purge gap
- Warm-start across folds: `init_model=previous_model`, add 50 new trees
- Quantile regression: 5 separate models (α=0.1, 0.25, 0.5, 0.75, 0.9)
- SHAP for feature importance (TreeSHAP, exact, fast)

### GRU (Temporal Features)
- Architecture: `GRU(input_size=N, hidden_size=64, num_layers=2, dropout=0.3, batch_first=True)` → `Linear(64, 1)`
- Sequence length: 24 bars (2 hours of 5-min candles)
- Training: AdamW(lr=1e-3, weight_decay=1e-2), gradient clipping at 1.0, cosine annealing LR
- Fine-tuning: lr=1e-4, 15 epochs (vs 50 initial), replay buffer for catastrophic forgetting prevention
- Hidden state extraction: final hidden state as features for LightGBM
- ONNX export for 2-5x faster inference

### Stacking Meta-Learner
- Logistic regression on OOF predictions from LightGBM + CatBoost + GRU
- OOF predictions generated via inner walk-forward (never in-sample)
- Calibrated via isotonic regression
- Ensemble weights optimized for Sharpe (not accuracy) via `scipy.optimize`

---

## 13. Backtesting Execution Timeline

### Phase A: Data Collection (Weeks 1-2)
- Backfill 60 days of 1-min OHLCV from Binance REST
- Backfill OI, funding rate history from Bybit
- Backfill NQ/Gold/DXY 5-min from yfinance
- Start collecting live order book, aggTrades, liquidations

### Phase B: Feature Engineering + Validation (Weeks 2-4)
- Implement all Tier 1 features (OBI, OFI, microprice, volume delta, ROC, vol)
- Run look-ahead bias tests on every feature
- Create golden datasets for regression testing
- Validate stationarity of all features (ADF + KPSS)

### Phase C: Model Training + Walk-Forward (Weeks 4-6)
- Train LightGBM, CatBoost, GRU with nested walk-forward
- Run full statistical validation suite (DSR, PSR, permutation, bootstrap)
- Ablation study: marginal contribution of each feature group
- Monte Carlo robustness (10K bootstrap, parameter perturbation)
- Historical stress test replay

### Phase D: Confidence Gating (Weeks 6-7)
- Implement isotonic calibration
- Train meta-labeling model
- Implement conformal prediction gate
- Optimize confidence threshold for Sharpe (not accuracy)
- Verify gating improves Sharpe by >50%

### Phase E: Paper Trading (Weeks 7-10)
- Deploy on Hetzner/AWS VPS with real data feeds
- Shadow mode: real signals, simulated execution
- 14-30 days minimum
- Evaluate go/no-go criteria

### Phase F: Graduated Live Rollout (Weeks 10-18)
- 5% → 15% → 35% → 65% → 100% capital
- 7-14 days per stage
- Continuous monitoring and alpha decay detection

---

## Key Numbers to Remember

| Parameter | Value |
|-----------|-------|
| Annualization factor | √105,120 ≈ 324.22 (NOT √252) |
| Minimum backtest Sharpe | > 2.0 (survives 50-70% degradation to ~0.8+ live) |
| Round-trip cost assumption | 8-12 bps (conservative) |
| Break-even cost threshold | Strategy must survive > 15 bps |
| Minimum data for backtest | 6 months (50K bars) for mean-based, 1 year for tail-based |
| Purge size | label_horizon + feature_lookback = 18 bars (90 min) |
| Training window | 14 days sliding |
| Expected backtest→live degradation | 50-70% |
| Paper trading minimum | 14 days, 200 trades |
| Full production timeline | 8-13 weeks from paper start |

---

## Research Files Index

All research reports are in [research/](research/). 40+ files covering:
- Backtesting framework architecture (event-driven, fill models, L2 replay)
- Statistical significance (DSR, PBO, SPA, permutation, bootstrap, power)
- Walk-forward optimization (purging, embargo, nested, regime-aware)
- Monte Carlo validation (bootstrap, stress testing, ruin probability)
- Performance attribution (signal decomposition, regime PnL, alpha/beta)
- Risk metrics (17 metrics with exact formulas and code)
- Transaction cost modeling (fee, slippage, impact, cost sensitivity)
- Benchmark strategies (8 benchmarks with implementations)
- Stress testing (6 historical scenarios + 6 synthetic)
- Alpha decay detection (CUSUM, ADWIN, SPRT, drift)
- LightGBM + GRU/TCN training recipes
- Paper trading infrastructure (exchange simulator, shadow mode, A/B testing)
- Data pipeline testing (schemas, look-ahead detection, PSI drift)
- Multi-timeframe analysis (alignment, partial candles, wavelet features)
- Meta-labeling implementation
- OFI + microprice implementation
- Cascade detection (Hawkes process)
- Model retraining strategies
- Optuna advanced optimization
