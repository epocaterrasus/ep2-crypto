# Mega-Research: 20-Agent Deep Investigation (2026-03-23)

## Summary

20 parallel research agents investigated correlations, data sources, ML techniques, and improvements
for the ep2-crypto 5-minute BTC prediction system. This document consolidates actionable findings.

**Output files with full details**: `/private/tmp/claude-501/-Users-edgarpocaterra-ep2-crypto/3b4ced9a-4d14-4a6e-b98d-074663043293/tasks/`

---

## TOP 5 ACTIONABLE IMPROVEMENTS (by Sharpe impact)

### 1. Confidence Gating Upgrades — Sharpe x1.5-3.0 (Sprint 8)
- Upgrade conformal prediction to **ACI + CQR** (Adaptive Conformal Inference + Conformalized Quantile Regression)
- Reduces prediction interval width by 20-30% while maintaining coverage
- Regime-adaptive intervals without explicit regime detection
- References: Gibbs & Candes (2024), Romano et al. CQR+, Kivaranovic et al. (2024)

### 2. Advanced Order Flow Features — Sharpe +0.2-0.5 (Sprint 5/6 extension)
Extensions of existing microstructure.py:
- **Multi-level OBI** at levels 1, 5, 10 — same depth@100ms data, +1-2% accuracy
- **VPIN** (Volume-Synchronized Probability of Informed Trading) via BVC classification — uses existing aggTrades, +1-2% as regime filter. Threshold: VPIN 0.3-0.6 = tradeable zone
- **Book pressure gradient** — bid vs ask depth slope across levels, +1% accuracy
- **Depth withdrawal ratio** — `(depth_5min_ago - current_depth) / depth_5min_ago`, market maker pull-away signal, ~58% accuracy at 1-min
- **Spoofing detection (simplified)** — phantom liquidity score from consecutive depth snapshots
- References: Cont, Kukanov & Stoikov (2014), Kolm et al. (2023), Easley, Lopez de Prado & O'Hara (2012)

### 3. Cross-Exchange Signals — Sharpe +0.1-0.3 (Sprint 5 or new sprint)
- **Coinbase premium** (IC 0.03-0.07): Add Coinbase WS via ccxt, compute premium z-score, delta
- **Cross-exchange OFI divergence** (IC 0.04-0.08): When OFI diverges across exchanges, the largest-volume exchange "wins" 58-63% of the time
- **ETH order flow** leading BTC by 1-5 min (54-57% accuracy): ETH net taker volume from Binance aggTrades
- **ETH/BTC ratio ROC** — early warning of risk-off, 55-58% accuracy on drops
- **Binance Long/Short Ratio** — free, already in API, contrarian at extremes
- References: Makarov & Schoar (2020), Alexander & Heck (2020), Augustin et al. (2022)

### 4. Regime Detection Upgrades — Sharpe +0.3-0.7 (Sprint 6 enhancements)
- **HAR-RV** (Heterogeneous Autoregressive Realized Volatility): Multi-scale components at 1h/4h/1d. Ratio features (RV_12bar/RV_288bar) capture vol term structure shifts that precede regime changes. Outperforms single-scale GARCH by 5-10% RMSE
- **Regime-conditional position sizing** (+0.2-0.4 Sharpe alone): Not trading during regime transitions captures 60-70% of total regime benefit
- **Multi-task GRU training**: Add volatility + volume prediction heads as auxiliary tasks. Hidden states encode richer market state. +0.1-0.2 Sharpe. Same architecture, just add heads during training
- **Optimal detection lag**: 10-15 bars (50-75 min) balances false positives vs detection speed
- References: Corsi (2009) HAR-RV, Nystrup et al. (2020), Zhang & Zhong (2024)

### 5. Liquidation Cascade Detection — Sharpe +0.3-0.5 as risk filter (Sprint 9)
- **Hawkes process** with branching ratio: normal 0.3-0.5, pre-cascade >0.8, cascade >0.9
- **Bybit upgraded Feb 2025**: `allLiquidation.BTCUSDT` now reports ALL liquidations (not just 1/sec)
- **Cascade probability score** = logistic(branching_ratio, OI_percentile, funding_zscore, book_thinning, price_velocity)
- When cascade probability >0.7, reduce position to 25% or halt
- BTC endogeneity ~80% (80% of price moves triggered by prior market activity)
- References: Atak (2020), Ali (2025 SSRN), Binance/Bybit liquidation stream docs

---

## VALIDATED ARCHITECTURE DECISIONS (no changes needed)

| Decision | Validation Source |
|----------|------------------|
| ADR-001: LightGBM > XGBoost | Transformers, RL, ensembles all confirm trees win on tabular crypto 5-min |
| ADR-004: On-chain = regime context only | Stablecoins, whales, mempool, GNNs all operate at 10min-24h |
| ADR-005: Skip NLP sentiment, use funding rate | Social sentiment has 0 predictive power at 5-min; funding rate is fastest sentiment |
| ADR-006: Stacking > per-regime models | Regime as feature > separate models; confirmed by ensemble literature |
| ADR-008: Triple barrier labeling | Not challenged by any research line |
| 5-min timeframe | Confirmed as sweet spot between signal decay and transaction costs |
| 52-56% accuracy target | R² of 0.03-0.05 at 5-min, consistent with this range |
| GRU hidden state extraction for LightGBM | Better than end-to-end transformers on this data volume |

---

## KEY NUMBERS FROM RESEARCH

| Metric | Value | Source |
|--------|-------|--------|
| OFI R² at 5-min | 0.03-0.05 (vs 0.65 at 1-sec) | Cont et al. (2014) adapted |
| Confidence gating Sharpe multiplier | 1.5-3.0x | De Prado (2024), multiple |
| Regime awareness Sharpe delta | +0.3-0.7 | Nystrup (2020), Zhang (2024) |
| Backtest-to-live degradation (supervised) | 50-70% | Multiple sources |
| Backtest-to-live degradation (RL) | 60-85% (worse) | Kumar et al. (2025) |
| NQ→BTC lead time (US hours) | 5-15 min | Jiang, Nie, Ruan (2023) |
| ETH→BTC lead time | 1-5 min | Alexander & Dakos (2020) |
| Alpha half-life (ML features crypto) | 6-18 months | Liu & Tsyvinski (2021) |
| Optimal regime detection lag | 10-15 bars (50-75 min) | Empirical consensus |
| Hawkes branching ratio cascade threshold | >0.8 | Academic literature |
| BTC Hawkes endogeneity | ~80% | Atak (2020) |
| Funding rate reversal at >0.10% per 8h | ~65% within 48h | CryptoQuant data |
| Coinbase premium IC | 0.03-0.07 at 30-min | Augustin et al. (2022) |
| DeepLOB 5-min accuracy on BTC | 56-59% (3-class) | Lucchese et al. (2024) |
| Transformer vs LightGBM at 5-min | LightGBM wins (53.9% vs 52.8%) | Zhang et al. (2024) |

---

## NEW DATA SOURCES RECOMMENDED

| Source | Cost | Signal | Priority | Sprint |
|--------|------|--------|----------|--------|
| Coinbase WebSocket (ccxt) | Free | Premium, US institutional flow | High | 5+ |
| Binance Long/Short Ratio | Free | Positioning extremes | High | 5+ |
| Bybit allLiquidation WS | Free | Full liquidation feed for Hawkes | High | 9 |
| ETH aggTrades (Binance WS) | Free | ETH order flow leading BTC | High | 5+ |
| Deribit WebSocket | Free | IV surface, skew, basis | Medium | 10+ |
| Twelve Data ($29/mo upgrade) | $29/mo | NQ/DXY 1-min delay vs 15-min | Medium-High | 5+ |
| Coinglass API | $79/mo | Liquidation heatmap, aggregated | Medium | 9 |
| Whale Alert API | $10/mo | Large USDT mints for regime | Low | 10+ |

---

## SIGNALS NOT USEFUL FOR 5-MIN (confirmed skip)

| Signal | Why Not | Useful Horizon |
|--------|---------|---------------|
| Stablecoin mint/burn | 30min-24h latency | 4h+ regime |
| Whale wallet tracking | 10+ min blockchain latency, 40-70% false positives | 4h-24h |
| ETF flows | T+1 daily data | Daily regime |
| Google Trends / Wikipedia | Daily granularity | Weekly |
| BTC dominance | Coincident, not leading | 4h+ |
| DeFi TVL | Consequence of price | Daily |
| Lightning Network | Zero predictive power | Months |
| Graph Neural Networks | Blockchain latency blocker | 1h+ (valuable there) |
| Reinforcement Learning | Supervised wins; 60-85% sim-to-real gap | Post-MVP execution timing |
| Prediction markets (Polymarket) | Lag spot at 5-min | 12-48h for regulatory events |
| News NLP | +1-2% accuracy doesn't justify complexity; NQ lead-lag subsumes value | 1h+ |
| Dark web activity | Inaccessible, irrelevant | N/A |
| Job postings, app downloads | Monthly+ timeframes | Quarters |
| Energy prices / mining cost | Too slow, weak correlation | Quarterly |

---

## FEATURE PRIORITY RANKING (for future sprints)

### Already Implemented (Sprint 3-5)
1. OBI, OFI, microprice, TFI, Kyle's Lambda
2. Volume delta, VWAP deviation, volume ROC
3. Realized vol, Parkinson, EWMA, vol-of-vol
4. ROC, RSI, linreg slope, quantile rank
5. NQ returns, ETH/BTC ratio, lead-lag correlation
6. Cyclical encoding, session, time-to-funding
7. Efficiency Ratio, GARCH vol, HMM probs

### High Priority Additions
8. Multi-level OBI (levels 1, 5, 10) — extend microstructure.py
9. VPIN via BVC — new feature using existing aggTrades
10. Book pressure gradient — from existing depth data
11. Coinbase premium (raw + z-score + delta) — add Coinbase WS
12. ETH net taker volume (order flow) — add ETH aggTrades
13. HAR-RV multi-scale components (1h/4h/1d ratios) — extend volatility.py
14. Depth withdrawal ratio — from existing depth data
15. Binance Long/Short Ratio — free API endpoint

### Medium Priority Additions
16. Cross-exchange OFI divergence — requires multi-venue depth
17. ETH/BTC ratio rate-of-change (15-min) — from existing ETH data
18. Deribit 25-delta risk reversal rate of change — add Deribit WS
19. ATM IV rate of change — from Deribit
20. Max pain distance (calendar feature) — from Deribit OI
21. Hawkes process branching ratio — from liquidation streams
22. USDT peg deviation — from Binance spot

### Low Priority (regime context only)
23. Polymarket event proximity score — poll every 15-60 min
24. Macro event binary flags — calendar lookup
25. Social volume spike (Santiment) — volatility predictor only
26. USDT mint/burn alerts — Whale Alert, regime bias

---

## ENSEMBLE & MODEL IMPROVEMENTS (for Sprint 7-8+)

### High ROI Upgrades
1. **ACI + CQR conformal prediction** — 20-30% tighter intervals, better calibration
2. **Multi-task GRU** — add vol + volume heads, richer hidden states
3. **Regime-conditional meta-learner weights** — upgrade from regime-as-feature
4. **Online weight adaptation** — after system is live, OEWA with regime priors

### Medium ROI
5. **BMA for position sizing calibration** — complement isotonic
6. **Decision Transformer** for execution timing (post-MVP, needs paper trading logs)

### Not Worth Pursuing Now
7. Sparse MoE — needs more data than 14-day windows
8. NAS — compute cost > marginal gain
9. Knowledge distillation — inference is already fast enough
10. Pure RL — supervised wins at 5-min

---

## ACADEMIC REFERENCES (most cited across agents)

1. Cont, Kukanov & Stoikov (2014) — "The Price Impact of Order Book Events" — OFI foundation
2. Makarov & Schoar (2020) — "Trading and Arbitrage in Cryptocurrency Markets" — cross-exchange dynamics
3. Alexander & Heck (2020) — "Price Discovery in Bitcoin" — CME vs Binance leadership
4. Easley, Lopez de Prado & O'Hara (2012) — "Flow Toxicity" — VPIN
5. Nystrup et al. (2020) — HMM for dynamic allocation with online adaptation
6. Corsi (2009) — HAR-RV model for multi-scale volatility
7. Gibbs & Candes (2024) — Adaptive Conformal Inference
8. Zhang et al. (2019) — DeepLOB architecture
9. Bouri et al. (2020) — Macro-crypto correlations
10. Liu & Tsyvinski (2021) — "Risks and Returns of Cryptocurrency" — alpha decay
