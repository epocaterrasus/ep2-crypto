# Architecture Decision Records (ADRs)

## ADR-001: LightGBM over XGBoost

**Date**: 2026-03-23
**Status**: Accepted

### Context
The original plan specified XGBoost as the primary classifier. Research across multiple reports consistently showed LightGBM outperforms XGBoost on 5-min crypto prediction tasks.

### Decision
Use LightGBM as the primary gradient boosting classifier.

### Consequences
- **Positive**: 0.5-1.5% accuracy improvement on 5-min crypto data
- **Positive**: Faster training (leaf-wise growth vs level-wise)
- **Positive**: Native categorical feature support
- **Positive**: Lower memory footprint
- **Negative**: Slightly different API from XGBoost (minor migration effort)
- **Negative**: Can overfit more easily with small data (mitigated by `min_child_samples=50`)

### Alternatives Considered
- **XGBoost**: Slower, slightly lower accuracy on this specific task. Well-documented but no advantage here.
- **TabNet**: Self-attention mechanism promising but inconsistent results, harder to tune, 2-3x slower inference.
- **Random Forest**: Baseline model, consistently outperformed by boosting.

### Research Source
- `RESEARCH_SYNTHESIS.md` section "Core Model: LightGBM + Stacking Ensemble"
- Multiple research files confirming LightGBM > XGBoost on crypto tick data

---

## ADR-002: Ternary Target over Binary

**Date**: 2026-03-23
**Status**: Accepted

### Context
The original plan used binary classification (up/down). Research showed this forces the model to make predictions during low-volatility periods where no tradeable edge exists.

### Decision
Use ternary classification: UP (return > +threshold), DOWN (return < -threshold), FLAT (within threshold). The threshold is adaptive per regime.

### Consequences
- **Positive**: Acknowledges untradeable low-volatility periods
- **Positive**: Reduces false signals during quiet markets
- **Positive**: Adaptive threshold adjusts to current market conditions
- **Negative**: Three-class problem is harder than binary
- **Negative**: Class imbalance (FLAT may dominate in low-vol regimes)
- **Negative**: Threshold selection adds a hyperparameter

### Alternatives Considered
- **Binary (up/down)**: Simpler but forces predictions during untradeable periods
- **Regression (continuous returns)**: Harder to calibrate, confidence gating less natural
- **Binary + volatility filter**: Two-step approach, less elegant than integrated ternary

### Research Source
- `RESEARCH_SYNTHESIS.md` section "Revised Tech Stack" (Target row)
- Research on Lopez de Prado's triple barrier method

---

## ADR-003: Sliding Window over Expanding Window

**Date**: 2026-03-23
**Status**: Accepted

### Context
Standard walk-forward validation uses expanding windows (all historical data). Crypto markets exhibit strong non-stationarity where patterns from months ago may be counterproductive.

### Decision
Use 14-day sliding windows for training. Old data is dropped, not accumulated.

### Consequences
- **Positive**: Model adapts to current market regime faster
- **Positive**: Training time stays constant (not growing)
- **Positive**: Reduces impact of regime changes from distant past
- **Negative**: Less training data per fold (4,032 bars vs potentially 100K+)
- **Negative**: May miss rare but important patterns (e.g., flash crash behavior)

### Alternatives Considered
- **Expanding window**: Standard approach, but crypto non-stationarity means old data hurts
- **Weighted expanding**: Decay weight on old samples. More complex, marginal benefit over sliding
- **21-day window**: Research showed >21 days dilutes signal in crypto
- **7-day window**: Too little data, model underfits

### Research Source
- `BACKTESTING_PLAN.md` section "Walk-Forward Validation Protocol"
- Research consensus: "Recency premium in crypto; >21 days dilutes signal"

---

## ADR-004: On-Chain as Regime Context Only

**Date**: 2026-03-23
**Status**: Accepted

### Context
The original plan used on-chain data (whale flows, mempool, exchange balances) as primary prediction signals. Research revealed these signals operate at 4h-24h timescales, not 5-min.

### Decision
Downgrade on-chain data to regime context only. Use mempool whale detection (>10 BTC to known exchange addresses) as a background signal, not for direct 5-min prediction.

### Consequences
- **Positive**: Removes noise from feature set at 5-min timescale
- **Positive**: Reduces data source dependencies (fewer failure points)
- **Positive**: Simpler pipeline
- **Negative**: Loses potential long-horizon signals (acceptable for 5-min system)
- **Negative**: May miss large exchange inflow/outflow signals

### Alternatives Considered
- **On-chain as primary signal**: Research shows these are 4h+ signals, not 5-min
- **Skip on-chain entirely**: Loses useful regime context (e.g., whale activity as volatility predictor)
- **Multi-horizon model**: Could use on-chain at 4h and microstructure at 5-min, but adds massive complexity

### Research Source
- `RESEARCH_SYNTHESIS.md`: "On-chain signals operate at 4h-24h, NOT 5-min"
- Research files on whale detection and exchange flows

---

## ADR-005: Skip Real-Time Sentiment, Use Funding Rate

**Date**: 2026-03-23
**Status**: Accepted

### Context
The original plan included Twitter/social sentiment NLP analysis. Research showed that by the time sentiment signals are processed and available, the price has already moved at the 5-min timescale.

### Decision
Skip all NLP-based sentiment analysis. Use Bybit funding rate as the only "sentiment" proxy, as it is quantitative, real-time, and directly actionable.

### Consequences
- **Positive**: Eliminates slow, noisy, and expensive NLP pipeline
- **Positive**: Funding rate is free, real-time, and quantitative
- **Positive**: Reduces infrastructure complexity significantly
- **Negative**: Misses potential early signals from viral social media events
- **Negative**: Funding rate is lagging indicator of sentiment, not leading

### Alternatives Considered
- **Twitter NLP (FinBERT/GPT)**: Too slow for 5-min (30-60s processing time)
- **Fear & Greed Index**: Daily resolution, useless for 5-min. Kept as regime context only.
- **News API sentiment**: Same latency problem as social media
- **Telegram/Discord scraping**: Legal concerns, unreliable, high noise

### Research Source
- `RESEARCH_SYNTHESIS.md`: "Sentiment analysis → skip for 5-min — too slow"
- Research files on sentiment analysis latency

---

## ADR-006: Stacking Ensemble over Per-Regime Models

**Date**: 2026-03-23
**Status**: Accepted

### Context
The original plan trained separate model sets per regime (3 regimes x 2 models = 6 models). Research showed this fragments already-scarce data, leading to undertrained models.

### Decision
Use a single stacking ensemble (LightGBM + CatBoost + GRU) with regime as an input feature, not a routing mechanism. Stacking meta-learner (logistic regression) combines base model predictions.

### Consequences
- **Positive**: All training data available to all models (no fragmentation)
- **Positive**: Stacking is the #1 ensemble gain after feature engineering (Kaggle consensus)
- **Positive**: 2-5% improvement over best single model
- **Positive**: Models learn regime-dependent patterns via regime input features
- **Negative**: Single model set may be less specialized per regime
- **Negative**: Stacking adds complexity (OOF predictions required)

### Alternatives Considered
- **Per-regime models**: Fragments data (e.g., volatile regime may have only 15% of samples)
- **Simple averaging**: Worse than stacking, no learned combination weights
- **Bayesian Model Averaging**: Theoretically superior but harder to implement, marginal gain
- **Mixture of Experts**: More complex, similar to per-regime but with soft routing. Future consideration.

### Research Source
- `RESEARCH_SYNTHESIS.md`: "Per-regime models fragment already-scarce data"
- Kaggle consensus on stacking as top ensemble method

---

## ADR-007: Meta-Labeling for Confidence Gating

**Date**: 2026-03-23
**Status**: Accepted

### Context
A raw ML prediction with 53% accuracy is not profitable after transaction costs. We need a way to select only the highest-conviction trades. Lopez de Prado's meta-labeling technique trains a secondary model to predict whether a primary model's trade will be profitable.

### Decision
Implement meta-labeling as a core gate in the confidence pipeline. A secondary LightGBM model takes the primary prediction + features + regime and outputs probability of profitable trade.

### Consequences
- **Positive**: 50-100% Sharpe improvement (research consensus)
- **Positive**: Dramatically reduces trade count (fewer but better trades)
- **Positive**: Naturally adapts to changing market conditions
- **Positive**: Provides interpretable confidence score
- **Negative**: Requires careful implementation to avoid data leakage
- **Negative**: Reduces number of trades (may miss some profitable opportunities)
- **Negative**: Two-model system is harder to debug

### Alternatives Considered
- **Fixed confidence threshold**: Simple but not adaptive
- **Ensemble agreement only**: Weaker signal than meta-labeling
- **Manual signal filters**: Not learnable, requires constant tuning
- **No gating**: 53% accuracy is unprofitable after costs

### Research Source
- `research/meta_labeling_deep_research.md` - Full implementation guide
- Lopez de Prado, "Advances in Financial Machine Learning" (AFML)

---

## ADR-008: Triple Barrier Labeling over Fixed Threshold

**Date**: 2026-03-23
**Status**: Accepted

### Context
Simple fixed-threshold labeling (if return > X bps, label "up") ignores the path of the price. A trade that hits a stop loss before eventually being correct should not be labeled as a win.

### Decision
Use triple barrier labeling (Lopez de Prado): vertical barrier (time limit), upper barrier (take profit), lower barrier (stop loss). The label is determined by which barrier is touched first.

### Consequences
- **Positive**: More realistic labels that account for path dependency
- **Positive**: Naturally handles holding period (vertical barrier)
- **Positive**: Stop loss incorporated into label definition
- **Positive**: Better alignment between backtest labels and live trading outcomes
- **Negative**: More complex to implement than fixed threshold
- **Negative**: Labels depend on barrier parameters (additional hyperparameters)
- **Negative**: Computing labels is slower (must simulate forward from each point)

### Alternatives Considered
- **Fixed threshold**: Simple but ignores path. A trade that drops 2% before recovering is labeled "win"
- **Forward return only**: Standard in academia but doesn't match trading reality
- **Realized P&L labeling**: Requires a trading strategy to exist before labeling, circular

### Research Source
- Lopez de Prado, "Advances in Financial Machine Learning" Chapter 3
- `RESEARCH_SYNTHESIS.md` references to AFML techniques

---

## ADR-009: Event-Driven Backtest Engine with Numba

**Date**: 2026-03-23
**Status**: Accepted

### Context
Vectorized backtesting is fast but cannot model path-dependent behavior (partial fills, stop losses, latency). Pure Python event-driven backtesting is realistic but 100x too slow for large-scale walk-forward validation.

### Decision
Build a custom event-driven backtest engine with Numba-compiled inner loop. Use vectorized backtesting for initial signal research (fast iteration), event-driven for final validation (realistic fills).

### Consequences
- **Positive**: 100x speedup over pure Python event-driven
- **Positive**: Realistic execution modeling (slippage, partial fills, latency)
- **Positive**: Path-dependent behavior correctly modeled
- **Positive**: < 3 seconds per year of 5-min bars (fast enough for walk-forward)
- **Negative**: Numba compilation adds startup time
- **Negative**: Numba code is harder to debug (limited Python features)
- **Negative**: Requires careful data type management (Numba type restrictions)

### Alternatives Considered
- **Pure vectorized (numpy)**: Fast (0.1s/year) but cannot model execution realistically
- **Pure Python event-driven**: Realistic but ~300s/year (too slow for walk-forward)
- **Existing frameworks (Backtrader, Zipline)**: Not designed for 5-min crypto, limited customization
- **C++ engine**: Maximum performance but high development cost, harder to iterate

### Research Source
- `BACKTESTING_PLAN.md` section "Engine: Custom Event-Driven + Vectorized Hybrid"

---

## ADR-010: asyncio Single-Process Architecture

**Date**: 2026-03-23
**Status**: Accepted

### Context
The system must manage multiple WebSocket connections, REST polling, feature computation, model inference, and API serving concurrently. Options are multiprocessing, threading, or asyncio.

### Decision
Use asyncio in a single process for all concurrent operations. Model training (CPU-intensive) is the only operation that may optionally use a separate process.

### Consequences
- **Positive**: No inter-process communication overhead
- **Positive**: Simple shared state (no serialization needed)
- **Positive**: Lower memory footprint than multiprocessing
- **Positive**: Natural fit for I/O-bound workloads (WebSocket, REST polling)
- **Positive**: ccxt pro is natively async
- **Negative**: CPU-bound operations block the event loop (mitigated by running in executor)
- **Negative**: Single point of failure (one crash kills everything)
- **Negative**: Cannot utilize multiple CPU cores for parallel computation

### Alternatives Considered
- **Multiprocessing**: Better CPU utilization but complex IPC, higher memory, harder to share state
- **Threading**: GIL limits CPU parallelism, harder to reason about concurrency
- **Actor model (Ray)**: Overkill for this scale, adds infrastructure dependency
- **Microservices**: Each component as separate service. Too much overhead for single-developer project

### Research Source
- `RESEARCH_SYNTHESIS.md` section on system architecture
- Research files on production system design
