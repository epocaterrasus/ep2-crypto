# Research Index

> 48 research reports + 1 Python test suite for the ep2-crypto project.
> Naming convention: `RR-{category}-{topic}.md`

---

## Quick Lookup by Sprint

### Sprint 1: Data Infrastructure & APIs
| File | Topic |
|------|-------|
| RR-api-ccxt-exchange-integration.md | ccxt library API signatures and exchange integration |
| RR-api-onchain-mempool-sources.md | On-chain and mempool data sources (Mempool.space, Blockchain.com, Glassnode, Whale Alert) |
| RR-crossmarket-data-sources-api.md | Cross-market data APIs for NQ futures, DXY, gold, ETH |
| RR-pipeline-data-validation-testing.md | Data pipeline validation with Great Expectations |

### Sprint 2: Feature Engineering & Signals
| File | Topic |
|------|-------|
| RR-features-crypto-feature-engineering.md | Feature engineering for short-term crypto prediction |
| RR-normalization-stationarity-preprocessing.md | Data normalization, stationarity, and preprocessing |
| RR-multitf-multi-timeframe-analysis.md | Multi-timeframe feature construction (1m, 5m, 15m, 1h) |
| RR-microstructure-order-book-analysis.md | Order book microstructure analysis techniques |
| RR-ofi-microprice-implementation.md | OFI and Microprice implementation (Cont-Kukanov-Stoikov) |
| RR-ofi-microprice-deep-research.md | Deep research on OFI from L2 snapshots and aggTrades |
| RR-crossmarket-lead-lag-correlations.md | Cross-market correlations and lead-lag with BTC |
| RR-cascade-prediction-mechanics.md | Liquidation cascade mechanics for prediction |
| RR-cascade-liquidation-detection-system.md | Real-time liquidation cascade detection (Hawkes process) |
| RR-whale-large-transaction-detection.md | Whale/large transaction detection for BTC prediction |
| RR-sentiment-crypto-analysis.md | Sentiment analysis for short-term crypto prediction |
| RR-altdata-unconventional-sources.md | Alternative/unconventional data sources (options gamma, derivatives) |

### Sprint 3: Model Training & Architecture
| File | Topic |
|------|-------|
| RR-models-architecture-comparison.md | ML architecture comparison for 5-min crypto prediction |
| RR-models-btc-prediction-repos-survey.md | Survey of open-source BTC prediction repos |
| RR-models-ml-stack-xgboost-catboost.md | ML stack: XGBoost, CatBoost API signatures and patterns |
| RR-lightgbm-crypto-tuning.md | LightGBM tuning specifically for crypto |
| RR-gru-tcn-training-architecture.md | GRU and TCN training for 5-min crypto prediction |
| RR-metalabeling-implementation-guide.md | Lopez de Prado meta-labeling implementation |
| RR-metalabeling-deep-research.md | Deep research on meta-labeling theory and integration |
| RR-afml-lopez-de-prado-techniques.md | Advances in Financial ML (AFML) techniques |
| RR-regime-detection-methods.md | Market regime detection (HMM and alternatives) |
| RR-confidence-calibration-signal-gating.md | Model confidence calibration and signal gating |
| RR-optuna-advanced-hyperparameter-optimization.md | Advanced Optuna multi-objective optimization |

### Sprint 4: Backtesting & Validation
| File | Topic |
|------|-------|
| RR-backtest-framework-design.md | Production-grade backtesting framework (event-driven vs vectorized) |
| RR-backtest-pitfalls-best-practices.md | Backtesting pitfalls and best practices for crypto |
| RR-overfitting-prevention-validation.md | Overfitting prevention and robust validation |
| RR-walkforward-validation-methods.md | Walk-forward optimization and validation |
| RR-montecarlo-bootstrap-validation.md | Monte Carlo bootstrap resampling for strategy validation |
| RR-montecarlo-validation-methods.md | 12 Monte Carlo validation methods with Python |
| RR-stats-edge-vs-luck-tests.md | Statistical tests: genuine edge vs luck |
| RR-stats-validation-tests.py | Python test suite for statistical validation |
| RR-costs-transaction-cost-modeling.md | Transaction cost modeling for backtesting |
| RR-costs-modeling-guide.md | Transaction cost modeling guide (fees, slippage, impact) |
| RR-benchmarks-null-hypothesis-strategies.md | Benchmark strategies as null hypotheses |
| RR-metrics-risk-adjusted-performance.md | Risk-adjusted performance metrics beyond Sharpe |
| RR-attribution-pnl-decomposition.md | PnL decomposition and signal attribution |

### Sprint 5: Production & Live Trading
| File | Topic |
|------|-------|
| RR-production-realtime-python-architecture.md | Real-time Python system architecture |
| RR-architecture-institutional-quant-systems.md | How institutional quant funds build trading systems |
| RR-execution-trade-optimization.md | Trade execution optimization (market vs limit orders) |
| RR-papertrade-system-architecture.md | Paper trading system architecture |
| RR-retrain-strategies-live-system.md | Model retraining strategies for live systems |
| RR-retrain-live-retraining-guide.md | Live retraining guide for LightGBM + CatBoost + GRU |
| RR-decay-edge-disappearance-detection.md | Edge decay detection (CUSUM test) |
| RR-stress-scenario-design.md | Stress test scenario design (historical replays) |
| RR-stress-testing-framework.md | Stress testing framework with CI/CD automation |

---

## Full Index by Category

### afml
| File | Description | Sprint |
|------|-------------|--------|
| RR-afml-lopez-de-prado-techniques.md | Marcos Lopez de Prado's AFML techniques for crypto trading | 3 |

### altdata
| File | Description | Sprint |
|------|-------------|--------|
| RR-altdata-unconventional-sources.md | Alternative data sources: options gamma, derivatives, unconventional signals | 2 |

### api
| File | Description | Sprint |
|------|-------------|--------|
| RR-api-ccxt-exchange-integration.md | ccxt library documentation, API signatures, and code examples for exchange integration | 1 |
| RR-api-onchain-mempool-sources.md | On-chain/mempool APIs: Mempool.space, Blockchain.com, Glassnode alternatives, Whale Alert | 1 |

### architecture
| File | Description | Sprint |
|------|-------------|--------|
| RR-architecture-institutional-quant-systems.md | How prop shops and quant funds build crypto trading systems | 5 |

### attribution
| File | Description | Sprint |
|------|-------------|--------|
| RR-attribution-pnl-decomposition.md | PnL decomposition to identify which signals contribute to performance | 4 |

### backtest
| File | Description | Sprint |
|------|-------------|--------|
| RR-backtest-framework-design.md | Event-driven vs vectorized backtesting framework design for 5-min BTC | 4 |
| RR-backtest-pitfalls-best-practices.md | Slippage modeling, look-ahead bias, survivorship bias, and other pitfalls | 4 |

### benchmarks
| File | Description | Sprint |
|------|-------------|--------|
| RR-benchmarks-null-hypothesis-strategies.md | Buy-and-hold and other benchmark strategies as null hypotheses | 4 |

### cascade
| File | Description | Sprint |
|------|-------------|--------|
| RR-cascade-liquidation-detection-system.md | Real-time liquidation cascade detection using Hawkes processes | 2 |
| RR-cascade-prediction-mechanics.md | How liquidation cascades work mechanically in crypto futures | 2 |

### confidence
| File | Description | Sprint |
|------|-------------|--------|
| RR-confidence-calibration-signal-gating.md | Model confidence calibration and when-to-trade signal gating | 3 |

### costs
| File | Description | Sprint |
|------|-------------|--------|
| RR-costs-transaction-cost-modeling.md | Binance/Bybit fee models, slippage, and market impact modeling | 4 |
| RR-costs-modeling-guide.md | Comprehensive transaction cost modeling guide for crypto backtesting | 4 |

### crossmarket
| File | Description | Sprint |
|------|-------------|--------|
| RR-crossmarket-data-sources-api.md | API endpoints for NQ futures, DXY, gold, ETH real-time and historical data | 1 |
| RR-crossmarket-lead-lag-correlations.md | BTC-Nasdaq, BTC-DXY, BTC-Gold lead-lag relationships | 2 |

### decay
| File | Description | Sprint |
|------|-------------|--------|
| RR-decay-edge-disappearance-detection.md | CUSUM test and other methods for detecting strategy edge decay | 5 |

### execution
| File | Description | Sprint |
|------|-------------|--------|
| RR-execution-trade-optimization.md | Market vs limit orders, TWAP/VWAP, execution quality measurement | 5 |

### features
| File | Description | Sprint |
|------|-------------|--------|
| RR-features-crypto-feature-engineering.md | Feature importance studies and predictive features for short-term crypto | 2 |

### gru-tcn
| File | Description | Sprint |
|------|-------------|--------|
| RR-gru-tcn-training-architecture.md | GRU and TCN architecture search, training, and hyperparameters | 3 |

### lightgbm
| File | Description | Sprint |
|------|-------------|--------|
| RR-lightgbm-crypto-tuning.md | LightGBM objective functions, feature interaction, and crypto-specific tuning | 3 |

### metalabeling
| File | Description | Sprint |
|------|-------------|--------|
| RR-metalabeling-implementation-guide.md | Step-by-step meta-labeling implementation for 5-min BTC system | 3 |
| RR-metalabeling-deep-research.md | Meta-labeling theory, pitfalls, and integration with ep2-crypto | 3 |

### metrics
| File | Description | Sprint |
|------|-------------|--------|
| RR-metrics-risk-adjusted-performance.md | Sortino, Calmar, Omega, and other risk-adjusted metrics with Python code | 4 |

### microstructure
| File | Description | Sprint |
|------|-------------|--------|
| RR-microstructure-order-book-analysis.md | Order book imbalance, depth analysis, and microstructure features | 2 |

### models
| File | Description | Sprint |
|------|-------------|--------|
| RR-models-architecture-comparison.md | XGBoost vs LightGBM vs CatBoost vs LSTM vs Transformer comparison | 3 |
| RR-models-btc-prediction-repos-survey.md | Survey of GitHub repos for short-term BTC price prediction | 3 |
| RR-models-ml-stack-xgboost-catboost.md | XGBoost and CatBoost API signatures, best practices, code patterns | 3 |

### montecarlo
| File | Description | Sprint |
|------|-------------|--------|
| RR-montecarlo-bootstrap-validation.md | Bootstrap resampling and Monte Carlo methods for strategy stress-testing | 4 |
| RR-montecarlo-validation-methods.md | 12 Monte Carlo validation methods with full Python implementations | 4 |

### multitf
| File | Description | Sprint |
|------|-------------|--------|
| RR-multitf-multi-timeframe-analysis.md | Combining 1-min, 5-min, 15-min, 1-hour signals for prediction | 2 |

### normalization
| File | Description | Sprint |
|------|-------------|--------|
| RR-normalization-stationarity-preprocessing.md | Stationarity tests, differencing, and normalization for financial time series | 2 |

### ofi-microprice
| File | Description | Sprint |
|------|-------------|--------|
| RR-ofi-microprice-implementation.md | OFI and Microprice implementation from L2 order book data | 2 |
| RR-ofi-microprice-deep-research.md | Deep dive into OFI from L2 snapshots and aggTrades | 2 |

### optuna
| File | Description | Sprint |
|------|-------------|--------|
| RR-optuna-advanced-hyperparameter-optimization.md | Multi-objective Optuna optimization for Sharpe + max drawdown | 3 |

### overfitting
| File | Description | Sprint |
|------|-------------|--------|
| RR-overfitting-prevention-validation.md | Common overfitting traps in financial ML and prevention techniques | 4 |

### papertrade
| File | Description | Sprint |
|------|-------------|--------|
| RR-papertrade-system-architecture.md | Paper trading architecture for pre-live model validation | 5 |

### pipeline
| File | Description | Sprint |
|------|-------------|--------|
| RR-pipeline-data-validation-testing.md | Data pipeline testing with Great Expectations framework | 1 |

### production
| File | Description | Sprint |
|------|-------------|--------|
| RR-production-realtime-python-architecture.md | Real-time Python architecture: async patterns, performance, failure handling | 5 |

### regime
| File | Description | Sprint |
|------|-------------|--------|
| RR-regime-detection-methods.md | HMM and alternative approaches for crypto market regime detection | 3 |

### retrain
| File | Description | Sprint |
|------|-------------|--------|
| RR-retrain-strategies-live-system.md | Fixed schedule, triggered, and online retraining strategies | 5 |
| RR-retrain-live-retraining-guide.md | Live retraining guide for LightGBM + CatBoost + GRU ensemble | 5 |

### sentiment
| File | Description | Sprint |
|------|-------------|--------|
| RR-sentiment-crypto-analysis.md | Twitter/X sentiment, news sentiment, and their predictive value for crypto | 2 |

### stats
| File | Description | Sprint |
|------|-------------|--------|
| RR-stats-edge-vs-luck-tests.md | t-tests and statistical methods to validate genuine trading edge | 4 |
| RR-stats-validation-tests.py | Python implementation of statistical validation test suite | 4 |

### stress
| File | Description | Sprint |
|------|-------------|--------|
| RR-stress-scenario-design.md | Historical stress scenarios for replay testing | 5 |
| RR-stress-testing-framework.md | Stress testing framework with scenario catalog and CI/CD automation | 5 |

### walkforward
| File | Description | Sprint |
|------|-------------|--------|
| RR-walkforward-validation-methods.md | Walk-forward vs expanding window vs anchored validation for financial ML | 4 |

### whale
| File | Description | Sprint |
|------|-------------|--------|
| RR-whale-large-transaction-detection.md | Detecting and using whale/large transaction activity for BTC prediction | 2 |
