# Hedge Fund Infrastructure: 20-Agent Deep Investigation (2026-03-23)

## Summary

20 parallel research agents investigated infrastructure, operations, legal, and business aspects
of running ep2-crypto as a professional crypto hedge fund. This document consolidates actionable findings.

**Output files with full details**: `/private/tmp/claude-501/-Users-edgarpocaterra-ep2-crypto/3b4ced9a-4d14-4a6e-b98d-074663043293/tasks/`

---

## INFRASTRUCTURE COST SUMMARY

### Minimum Viable Setup (Paper Trading → Live)

| Component | Monthly Cost | Provider |
|-----------|-------------|----------|
| Trading server (Singapore) | $62 | AWS c6i.large ap-southeast-1 |
| Monitoring server | $40-60 | Hetzner 4-core/16GB |
| Database (TimescaleDB) | $5 | Hetzner CX22 self-hosted |
| Standby server (DR) | $50 | Hetzner AX41 |
| Backup storage | $2 | Cloudflare R2 free tier |
| Monitoring stack | $0 | Prometheus + Grafana + Loki (self-hosted) |
| Alerting | $5 | Telegram bot + Twilio |
| Secrets management | $0-18 | Doppler free/pro |
| Market data (Twelve Data) | $29 | NQ/DXY 1-min delay |
| **Total infrastructure** | **$193-271/mo** | |

### Scaling Infrastructure

| AUM | Monthly Infra | Monthly Compliance | Total Monthly |
|-----|--------------|-------------------|---------------|
| $10K-100K | $200 | $0 | $200 |
| $100K-1M | $300 | $2K | $2.3K |
| $1M-10M | $500 | $8-15K | $8.5-15.5K |
| $10M-50M | $1K | $15-30K | $16-31K |

---

## TOP 10 ACTIONABLE FINDINGS

### 1. For 5-min bars, a $50/mo VPS is sufficient
Infrastructure barely matters at this timeframe. Total bar processing: <20ms. Budget: 299,980ms headroom. Invest in reliability (monitoring, DR), not speed.

### 2. Exchange-native stop losses are non-negotiable
If your server dies, the exchange-side stop still protects you. Every position entry MUST atomically place a stop-market order on the exchange. This is the #1 DR measure.

### 3. Numba + ONNX + Ring Buffers = 10-50x performance for free
- Numba for rolling features: 10-50x over pure Python
- ONNX Runtime for inference: 3-5x over native LightGBM
- NumPy ring buffers: constant memory, 100x faster append than pandas

### 4. Prometheus + Grafana + Loki + Tempo for $55/mo
Full observability stack self-hosted. 50 specific metrics defined. 4 dashboards designed. Telegram for alerts. Skip Datadog ($300-500/mo).

### 5. Max 40% of capital on any single exchange
Post-FTX rule. Capital split: Binance 35-40%, Bybit 25-30%, OKX 15-20%, cold wallet 10-15%.

### 6. Strategy capacity ceiling: $3-5M single venue, $15-25M multi-venue
After that, your own market impact eats your alpha. Square-root impact model confirmed.

### 7. Fund setup costs $350K-800K first year
Legal + compliance + admin + audit + insurance. Break-even AUM: $15-25M at 2/20 fees. Below $5M is economically unviable as a business.

### 8. First hire should be a quant developer, not researcher
The founder is the researcher. The bottleneck is turning research into production. Don't hire until management fees cover salaries (~$3M AUM).

### 9. Shadow mode → Paper → 25% capital → Full deployment
Never blue/green for trading. 4-stage deployment with auto-rollback triggers (8 consecutive losses, 5% deployment drawdown, prediction staleness).

### 10. Start live track record with personal capital immediately
Even $50K of real money traded live for 6+ months is worth more than any amount of backtesting to investors. The clock starts the day you go live.

---

## VALIDATED ARCHITECTURE DECISIONS

| Decision | Validation |
|----------|-----------|
| Python single-process asyncio | Sufficient for 5-min bars; GIL not a bottleneck |
| systemd over Kubernetes | K8s is massive overkill for single-strategy |
| SQLite→TimescaleDB migration | ~42 MB/year compressed per symbol; Hetzner $5/mo |
| Doppler for secrets | Per CLAUDE.md; upgrade to Vault at $500K+ capital |
| GitHub Actions for CI/CD | 10-gate pipeline sufficient; self-hosted runner for paper trade smoke tests |
| MLflow for model registry | Self-hosted; skip feature stores (Feast overkill) |
| CPU for inference, GPU for training | ONNX on CPU beats GPU at batch_size=1 |

---

## LEGAL/BUSINESS PATH

### Phase 1: Prove the Strategy ($0-6 months)
- Trade personal capital ($50K-200K)
- Single exchange (Binance), no legal entity needed
- Total cost: ~$200/mo infrastructure
- Goal: 6+ month live track record with Sharpe >1.0

### Phase 2: Friends & Family Fund ($6-12 months)
- Delaware LP + LLC: $50-70K setup
- 3-5 investors, <$5M AUM
- Self-administer or basic fund admin ($25-40K/yr)
- Fee structure: 1.5/20 with high-water mark
- Total ongoing: $25-40K/yr

### Phase 3: Institutional Fund ($12-24 months)
- Cayman LP + CIMA registration: $150-300K setup
- Fund administrator (NAV Consulting): $40-75K/yr
- Annual audit (Cohen & Co): $30-60K/yr
- Target: $10-25M AUM
- Fee structure: 1.5/20 with hurdle at risk-free rate
- Total ongoing: $150-300K/yr

### Phase 4: Scale ($24-36 months)
- Approach seeders (Brevan Howard Digital, Galaxy): $10-50M
- Multi-strategy (add macro, cascade)
- Prime broker (FalconX, Copper ClearLoop)
- Target: $25-50M AUM — self-sustaining

---

## KEY NUMBERS

| Metric | Value |
|--------|-------|
| Bar processing cycle (all optimizations) | <20ms |
| Numba feature speedup | 10-50x |
| ONNX inference speedup | 3-5x |
| Infrastructure cost (live single-symbol) | $200-300/mo |
| Monitoring stack cost | $55-75/mo |
| DR setup cost | ~$105/mo |
| Exchange fee (Binance Regular taker RT) | 10 bps |
| Exchange fee (Binance VIP1 taker RT) | 8 bps |
| Exchange fee (Binance maker RT) | 3.2-4 bps |
| Strategy AUM ceiling (5-min BTC) | $3-5M single venue |
| Strategy AUM ceiling (multi-venue) | $15-25M |
| Fund setup cost (Cayman LP) | $150-300K |
| Fund ongoing cost (sub-$50M) | $150-300K/yr |
| Break-even AUM (2/20 fees) | $15-25M |
| Minimum track record for institutional | 12-18 months live |
| Timeline to self-sustaining fund | 24-36 months |
| First hire AUM threshold | ~$3M |
| Max single exchange exposure | 40% |

---

## 20 RESEARCH TOPICS COVERED

### Infrastructure (7)
1. Low-latency infra — AWS $62/mo in Singapore is the sweet spot
2. Exchange connectivity — Binance VIP tiers, multi-exchange routing
3. Database — TimescaleDB, complete schema, 42 MB/yr per symbol
4. Monitoring — Prometheus+Grafana, 50 metrics, 4 dashboards
5. Disaster recovery — Hot standby, exchange-side stops, Litestream
6. Python performance — Numba, ONNX, ring buffers, asyncio tuning
7. Security — API key management, server hardening, incident response

### DevOps & MLOps (3)
8. CI/CD — 10-gate pipeline, shadow deployment, auto-rollback
9. MLOps — MLflow registry, DVC for data, 6-stage validation pipeline
10. Backtest-production parity — 10 friction sources, calibration techniques

### Trading Operations (3)
11. Execution algorithms — TWAP, slippage models, pre-trade cost gate, TCA
12. Order management — State machine, reconciliation, audit trail
13. Capital allocation — Multi-strategy Kelly, risk parity, AUM scaling

### Business & Legal (5)
14. Hedge fund structure — Delaware LP / Cayman, $50-300K setup
15. Regulatory compliance — MiCA, SEC/CFTC, offshore jurisdictions
16. Tax & accounting — Section 475(f), crypto tax software, 7-year retention
17. Investor relations — LP reporting, NAV methodology, fee structures
18. Fundraising — Seeders, timeline, pitch deck structure

### People (2)
19. Institutional risk management — 3LOD, VaR/CVaR, stress testing, LP reports
20. Team building — Hiring sequence by AUM, compensation, key person risk
