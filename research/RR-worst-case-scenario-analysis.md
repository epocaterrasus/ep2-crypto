# Worst-Case Scenario Analysis: Design for Survival, Not for Profit

> Comprehensive risk analysis for ep2-crypto's 5-minute BTC prediction system.
> Every number is calibrated to OUR system: quarter-Kelly sizing, 5% position cap,
> 15% max drawdown halt, single BTC/USDT:USDT perpetual futures position.

---

## System Parameters (Reference)

From `config.py` and `CLAUDE.md`:

| Parameter | Value |
|-----------|-------|
| Position size per trade | 4-5% of capital (quarter-Kelly capped) |
| Daily loss limit | 2-3% of capital |
| Weekly loss limit | 5% of capital |
| Max drawdown halt | 15% of capital |
| Catastrophic stop per trade | 3 ATR |
| Max open positions | 1 |
| Max trades per day | 30 |
| Trading hours | 08:00-21:00 UTC |
| Weekend position sizing | -30% reduction |
| Confidence threshold | 0.60 |

**Assumed starting capital for all calculations**: $10,000 (scalable; percentages hold at any size).

---

## 1. Complete Risk Enumeration

### Category A: Market Risk

| ID | Risk | Description |
|----|------|-------------|
| A1 | Wrong prediction (extended) | Model accuracy drops below 50% for multiple consecutive days due to regime shift |
| A2 | Flash crash while positioned | BTC drops 10%+ in 5 minutes with an open long position |
| A3 | Prolonged adverse trend | Model trained on range-bound data; market enters a persistent trend (or vice versa) |
| A4 | Liquidation cascade | Our position adds to a cascade; stop loss triggers into a falling knife with extreme slippage |
| A5 | Correlation breakdown | Cross-market NQ lead-lag relationship inverts or disappears |
| A6 | Volatility regime shift | Vol jumps from 30% to 200%+ annualized overnight; ATR-based stops too wide or too tight |
| A7 | Gap risk (weekend) | BTC gaps 5%+ on Monday open after weekend news; no 5-min bar to stop against |
| A8 | Funding rate bleed | System holds positions across funding settlements; extreme funding rates (-0.3% per 8h) during crisis |
| A9 | Regime not in training set | A novel market structure emerges (e.g., ETF-driven flows) that the 14-day training window has never seen |

### Category B: Execution Risk

| ID | Risk | Description |
|----|------|-------------|
| B1 | Slippage much worse than modeled | During flash crashes, effective slippage is 5-20x normal (our model assumes sqrt-impact) |
| B2 | Order rejected at critical moment | Exchange rate limits, insufficient margin, or maintenance window during stop-loss execution |
| B3 | Double execution | Network retry sends the same order twice; position is 2x intended size |
| B4 | Stuck order | Limit order partially filled; cannot cancel or modify during API degradation |
| B5 | Latency spike | Normally 50-200ms; during crashes, API response time goes to 5-30 seconds |
| B6 | Fill price deviation | Stop-loss order placed at $50,000; executed at $48,500 due to thin book |
| B7 | Partial fill trap | Only half the position closes; remaining half rides the crash |

### Category C: Operational Risk

| ID | Risk | Description |
|----|------|-------------|
| C1 | Exchange goes down with open position | Binance API outage for 15+ minutes during a crash (happened during March 2020) |
| C2 | API key compromised | Attacker drains account via market orders or opens massive positions |
| C3 | Server crash during trading hours | asyncio single-process architecture (ADR-010) = single point of failure |
| C4 | Data feed corruption | Bad prices from exchange (stale data, incorrect OHLCV); model makes decisions on garbage |
| C5 | Bug in position sizing | Off-by-one, decimal error, or config misread results in 10x intended position size |
| C6 | Database corruption | SQLite WAL corruption; system loses state of open positions and recent trades |
| C7 | Time synchronization | System clock drifts; 5-min bar alignment breaks; features computed on wrong data |
| C8 | Memory leak / OOM | Ring buffers or feature pipeline leaks memory; system degrades over hours/days |
| C9 | Dependency failure | ccxt library bug, exchange API version change, websocket protocol change |

### Category D: External Risk

| ID | Risk | Description |
|----|------|-------------|
| D1 | Exchange insolvency | All funds on Binance lost (FTX precedent: $0 recovery for most users) |
| D2 | Regulatory action | US/EU bans crypto derivatives trading; position must be closed immediately |
| D3 | Stablecoin depeg | USDT depegs from $1.00; all USDT-denominated P&L becomes worth less in real terms |
| D4 | Network attack on Bitcoin | 51% attack, consensus bug, or major protocol vulnerability |
| D5 | Tax/legal surprise | Retroactive crypto tax law; trading gains taxed at unexpected rate |
| D6 | Banking rails cut | Fiat on/off ramps blocked; cannot deposit margin or withdraw profits |
| D7 | Insurance fund depletion | Exchange socialized loss (auto-deleveraging) takes profits from winning positions |
| D8 | Market structure change | Binance changes fee structure, adds maker penalties, or eliminates perpetual contracts |

---

## 2. Full Risk Matrix

### Probability and Impact Scoring

**Probability scale**:
- Very Low: <1% per year
- Low: 1-5% per year
- Moderate: 5-20% per year
- High: >20% per year

**Impact scale** (as % of trading capital):
- Negligible: <1% capital loss
- Minor: 1-5% capital loss
- Major: 5-20% capital loss
- Catastrophic: >20% capital loss (or total loss)

### Category A: Market Risk

| ID | Risk | Prob | Impact | Mitigation | Residual Risk |
|----|------|------|--------|------------|---------------|
| A1 | Wrong prediction (extended) | **High** (>20%) | **Minor** (1-5%) | Confidence gating filters low-conviction trades; meta-labeling rejects marginal signals; daily loss limit halts at 3%. Alpha decay monitoring (CUSUM, rolling Sharpe) triggers retraining. | Gradual bleed of 1-3% before detection. System halts but capital is preserved. |
| A2 | Flash crash while positioned | **Moderate** (5-20%) | **Major** (5-15%) | Exchange-side stop-loss order at 3 ATR. Kill switch fires on daily loss limit. Drawdown gate reduces position size progressively. Position size of 5% means a 10% crash = 0.5% capital loss (without leverage). On 10x leverage: 5% capital loss. | If stop-loss executes at 5x slippage during thin book: 2-3% additional loss beyond expected. Max realistic loss per flash crash event: ~8% with 3 ATR stop on 5% position at 10x. |
| A3 | Prolonged adverse trend | **Moderate** (5-20%) | **Minor** (2-5%) | 14-day sliding window retrains to new regime. Regime detector flags trend. Weekly loss limit halts at 5%. Position sizing scales down with drawdown gate. | 2-4 losing days before regime detection catches up. ~3-5% drawdown before adaptation. |
| A4 | Liquidation cascade | **Moderate** (5-20%) | **Major** (5-10%) | Hawkes cascade detector reduces confidence/position. Funding rate z-score > 2 triggers 50% position reduction. ATR stop widens dynamically during high vol. | Cascade can exceed detector speed if it develops in <2 bars (10 minutes). |
| A5 | Correlation breakdown | **Moderate** (5-20%) | **Minor** (1-3%) | NQ lead-lag is a Tier 3 feature (not primary signal). Rolling correlation monitor down-weights NQ when correlation drops >0.3 in 24h. | Lost signal value; no direct loss. Indirect risk if model relied on NQ signal for a specific trade. |
| A6 | Volatility regime shift | **High** (>20%) | **Minor** (2-4%) | GARCH vol estimate updates every bar. Max volatility filter (150% annualized) halts trading during extreme vol. ATR stop adapts to new vol level within 14 bars. | 2-3 bars of mis-sized stops before ATR catches up. |
| A7 | Gap risk (weekend) | **Low** (1-5%) | **Minor** (1-3%) | Weekend position sizing reduced 30%. Trading hours restricted to 08:00-21:00 UTC (no overnight). Max 1 open position. | If a position is held into weekend (system failure to close), gap risk is real but bounded by position size. |
| A8 | Funding rate bleed | **Moderate** (5-20%) | **Negligible** (<1%) | Cost engine models funding cost. Funding rate is a feature input. Positions held <8h on average. | Worst case: 0.3%/8h on 5% position = 0.015% capital per settlement. Trivial. |
| A9 | Novel regime | **Moderate** (5-20%) | **Minor** (2-5%) | BOCPD change point detector flags novel regimes. Regime entropy >0.8 triggers confidence reduction. Stacking ensemble with regime as feature can partially generalize. | Model degrades to ~50% accuracy for 1-3 days until retraining window includes new regime. |

### Category B: Execution Risk

| ID | Risk | Prob | Impact | Mitigation | Residual Risk |
|----|------|------|--------|------------|---------------|
| B1 | Extreme slippage | **Moderate** (5-20%) | **Minor** (1-3%) | Spread filter: no trades when spread > 5x median. Slippage model uses sqrt-impact with volatility multiplier. Cost engine has stress mode (5x slippage). | During genuine flash crashes, even spread filters may not catch the first bar. |
| B2 | Order rejection | **Low** (1-5%) | **Minor** (1-2%) | Retry logic with exponential backoff. Fallback: market close if limit rejected. Independent monitoring sends alert. | Position stays open during retry window (seconds to minutes). |
| B3 | Double execution | **Low** (1-5%) | **Major** (5-10%) | Unique order ID (client_order_id). Position tracker validates actual vs intended. Max position size hard limit in code AND exchange-side. | If both the code check and exchange check fail simultaneously: 2x position. At 5% sizing with 10x leverage, this means 10% exposure. |
| B4 | Stuck order | **Low** (1-5%) | **Minor** (1-3%) | Timeout: if order not filled in 30s, cancel and resubmit as market. Position tracker monitors pending orders. | Partial fill leaves residual position that must be closed at market. |
| B5 | Latency spike | **High** (>20%) | **Negligible** (<1%) | System already assumes 200ms latency. During spikes, stop-loss is exchange-side (pre-placed), not dependent on our latency. | Exchange-side stops execute regardless of our API latency. |
| B6 | Fill price deviation | **Moderate** (5-20%) | **Minor** (1-2%) | Exchange-side stop is a stop-market order. In thin books, deviation can be 0.5-2%. ATR stop set wide enough (3 ATR) to account for noise. | During extreme events: stop at $50K fills at $48.5K. On 5% position: 1.5% * 5% = 0.075% capital extra loss. Manageable. |
| B7 | Partial fill trap | **Low** (1-5%) | **Minor** (1-2%) | IOC (Immediate-or-Cancel) for time-sensitive orders. Position tracker monitors fill completeness. Unfilled remainder closed at market after 60s. | Timing risk on the unfilled portion. |

### Category C: Operational Risk

| ID | Risk | Prob | Impact | Mitigation | Residual Risk |
|----|------|------|--------|------------|---------------|
| C1 | Exchange outage | **Moderate** (5-20%) | **Major** (5-15%) | Exchange-side stop-loss order persists during API outage. Independent monitoring (separate process/server) detects outage. Capital allocation limit: never >20% of net worth on any single exchange. | If exchange is down AND stop doesn't execute (rare but happened on BitMEX March 2020): position rides the crash. Bounded by position size and leverage. |
| C2 | API key compromise | **Very Low** (<1%) | **Catastrophic** (100%) | IP whitelist on exchange API. Withdrawal whitelist (address lock). API key permissions: trade-only, no withdraw. Key rotation schedule (90 days). Secrets in Doppler, never committed. | If attacker gains server access and API keys have withdrawal permission: total loss. Mitigation: NEVER enable withdrawal on trading API keys. |
| C3 | Server crash | **Moderate** (5-20%) | **Minor** (1-3%) | Exchange-side stop persists. Systemd auto-restart. Health check monitoring with alert. No open position without corresponding exchange-side stop. | Time between crash and restart (30-120 seconds). Position is protected by exchange-side stop. |
| C4 | Data feed corruption | **Low** (1-5%) | **Minor** (1-3%) | Data validation: price must be within 5% of last known good price. Stale data detection: if no new data for 30s, flag as stale. Multiple data source cross-validation. | If corruption is subtle (e.g., prices are plausible but wrong): model makes bad decisions until detected. |
| C5 | Position sizing bug | **Low** (1-5%) | **Catastrophic** (>20%) | Hard max position size in code (10% of capital, absolute max). Exchange-side position limit. Code review + type checking (mypy strict). Integration tests with boundary values. | If the bug is in the exchange-side check as well: unbounded loss. This is the #1 operational risk. |
| C6 | Database corruption | **Low** (1-5%) | **Negligible** (<1%) | WAL mode (already configured). Periodic backups. Position state also tracked in memory (ring buffer). Exchange is source of truth for actual position. | Temporary loss of trade history. Position state recoverable from exchange API. |
| C7 | Clock drift | **Very Low** (<1%) | **Negligible** (<1%) | NTP synchronization. Exchange timestamp used for bar alignment, not local clock. | Misaligned features for a few bars. Self-correcting. |
| C8 | Memory leak | **Moderate** (5-20%) | **Negligible** (<1%) | Ring buffers with fixed size (numpy). Periodic memory monitoring. Process restart every 24h (scheduled). | Gradual degradation over hours. No capital loss; system just gets slow. |
| C9 | Dependency failure | **Moderate** (5-20%) | **Minor** (1-2%) | Pin all dependency versions in uv.lock. Test before upgrading. Fallback: if ccxt fails, close position and halt. | Temporary inability to trade. No capital loss if position is flat. |

### Category D: External Risk

| ID | Risk | Prob | Impact | Mitigation | Residual Risk |
|----|------|------|--------|------------|---------------|
| D1 | Exchange insolvency | **Low** (1-5%) | **Catastrophic** (100% of exchange capital) | **Capital allocation limit: never >20% of net worth on any single exchange.** Keep only working capital on exchange. Withdraw profits regularly. Diversify across 2 exchanges (Binance + Bybit). | 20% of net worth at risk. This is the hard limit. No technical mitigation can prevent loss if exchange disappears. |
| D2 | Regulatory ban | **Low** (1-5%) | **Major** (5-15%) | Monitor regulatory news. Maintain ability to close all positions quickly. Use non-US exchange if US-based. | Liquidation at unfavorable prices during panic. |
| D3 | USDT depeg | **Low** (1-5%) | **Major** (5-20%) | Monitor USDT/USD rate. If depeg >1%: close all positions and convert to BTC or USDC. Trading on USDT pair means our PnL is denominated in potentially depreciating asset. | During depeg event, exit may be difficult. All USDT profits lose real value. |
| D4 | Bitcoin network attack | **Very Low** (<1%) | **Catastrophic** (>50%) | Not mitigable at system level. Position size limits total exposure. | If Bitcoin's fundamental security is broken, all BTC-related assets go to zero. |
| D5 | Tax surprise | **Low** (1-5%) | **Minor** (1-5%) | Track all trades for tax reporting. Reserve 30% of profits for taxes. Consult tax professional. | Retroactive tax changes could affect past profits. |
| D6 | Banking rails cut | **Low** (1-5%) | **Negligible** (<1%) | Does not affect trading. Only affects withdrawal. Keep emergency fiat reserves outside crypto. | Cannot realize profits in fiat. |
| D7 | Socialized loss (ADL) | **Low** (1-5%) | **Minor** (1-5%) | Trade on exchange with large insurance fund (Binance: $1B+). Avoid holding during extreme events. Keep position sizes small. | ADL takes winning side profits during extreme counterparty failures. |
| D8 | Market structure change | **Moderate** (5-20%) | **Minor** (1-3%) | Cost engine is parameterized; can update fees. Alpha decay monitoring detects structural breaks. | Gradual edge erosion as fees or liquidity change. |

---

## 3. The "What Kills You" Hierarchy

### Level 1 Death: Total Account Wipeout (>95% loss)

**Scenarios that cause it:**

1. **Exchange insolvency** (D1): All funds on exchange lost. FTX precedent shows recovery is near-zero.
   - Probability: 1-5% per year
   - Prevention: **Capital allocation limit = 20% of net worth on any single exchange**
   - With mitigation: Maximum loss = 20% of net worth (Level 3 death, not Level 1)

2. **API key compromise with withdrawal access** (C2): Attacker drains entire exchange balance.
   - Probability: <1% per year with proper security
   - Prevention: **Never enable withdrawal permissions on trading API keys.** IP whitelist. 2FA on exchange account.
   - With mitigation: Attacker can only trade (not withdraw). They can damage the position but cannot steal funds outright.

3. **Position sizing bug with leverage** (C5): Bug sets position at 100x intended size. A 1% adverse move = 100% loss.
   - Probability: <1% per year with testing
   - Prevention: **Hard position limits at 3 levels**: code, config, and exchange-side. Integration tests with boundary values. Position tracker validates every order against intended size.
   - With mitigation: Exchange-side limit catches most cases. Code-level max should be 10% of capital.

4. **Double execution + flash crash** (B3 + A2): Position is 2x intended + crash = margin call.
   - Probability: <0.1% per year
   - Prevention: Client order ID deduplication. Position tracker reconciliation.

**Level 1 Prevention Rule**: No single technical failure should be able to wipe the account. This requires defense-in-depth:
- Position limit in code (first line)
- Position limit in config (second line)
- Position limit on exchange (third line, cannot be bypassed by bugs)

### Level 2 Death: >50% Drawdown (mathematically recoverable, psychologically fatal)

**Scenarios that cause it:**

1. **Exchange insolvency with 50%+ of capital on exchange** (D1): Already handled by 20% allocation limit.

2. **Extended model failure + no kill switch** (A1 + C3): Model is wrong, server crashes, kill switches don't fire, position bleeds for days.
   - Probability: <0.5% per year
   - Prevention: Exchange-side stops persist. Daily loss limit (3%) halts system. Weekly loss limit (5%) halts system. 15% drawdown halts system entirely.
   - **Math check**: At 3% daily limit, a 50% drawdown requires ~23 consecutive days of hitting the daily limit with the system restarting each day. The weekly limit (5%) would trigger after 2 days. The 15% drawdown halt triggers after 5-6 days at most.
   - **Conclusion**: Kill switches make >50% drawdown nearly impossible without operational failure.

3. **Multiple correlated risks hitting simultaneously** (A2 + C1 + B5): Flash crash + exchange outage + latency spike. Position has no stop because exchange is down.
   - Probability: <1% per year
   - Prevention: Exchange-side stop-loss order placed at the time of position entry, not dynamically adjusted via API.
   - **Key insight**: The stop-loss order MUST be placed on the exchange at entry time, not managed by our system. If our system is down, the exchange still executes the stop.

### Level 3 Death: >25% Drawdown (triggers full halt, months to recover)

**Scenarios that cause it:**

1. **15% drawdown halt fires, then re-entry leads to more losses**: System halts at 15%, operator manually re-enables, model is still broken, another 10% loss.
   - Probability: 2-5% per year
   - Prevention: **After 15% drawdown halt, require manual review AND backtesting on recent data before re-enabling.** Automated re-entry at 25% reduced position size.
   - Cooling period: minimum 48 hours before re-enabling after a halt.

2. **Stablecoin depeg during large USDT exposure** (D3): All PnL + capital denominated in USDT that depegs 10-20%.
   - Probability: 1-3% per year
   - Prevention: Regular profit withdrawal. Monitor USDT/USD rate. Emergency conversion trigger at 2% depeg.

3. **Regulatory action forces liquidation at distressed prices** (D2): Must close position immediately during a regulatory-driven crash.
   - Probability: 1-3% per year
   - Prevention: Position sizing limits exposure. Political monitoring.

### Level 4 Death: Alpha Decay (strategy stops working, gradual bleed)

**Scenarios that cause it:**

1. **Market structure change** (D8): Binance changes fee structure or market making dynamics.
   - Probability: 20-40% per year (some degree of alpha decay is almost certain)
   - Detection: CUSUM test on rolling Sharpe. Rolling win rate drops below 50%. ADWIN drift detection on feature distributions.
   - Prevention: 14-day retraining window adapts. Multiple edges (microstructure + cross-market + regime) provide resilience.
   - **Key metric**: If rolling 30-day Sharpe drops below 0.5 (annualized), halt trading and investigate.

2. **Crowding** (market participants adopt similar strategies): Edge erodes as more traders exploit the same signals.
   - Probability: 10-20% per year for any specific signal
   - Detection: Kyle's Lambda increasing (less price impact per unit of flow). Spread compression. Feature importance shifts.
   - Prevention: Multi-edge approach. Continuous feature research. Monitor feature contribution via attribution analysis.

3. **Data source degradation** (C9): Exchange changes API, data quality degrades, free data sources shut down.
   - Probability: 10-20% per year
   - Prevention: Data quality monitoring (PSI for feature drift). Fallback data sources. Alert on data staleness.

**Level 4 Prevention Rule**: Alpha decay is not preventable; it is manageable. Budget for it:
- Expected useful life of any ML pattern: 1-6 months
- Required: continuous monitoring, monthly review, quarterly revalidation
- **If Sharpe decays below 0.3 for 30 consecutive days: STOP. Full re-research required.**

---

## 4. Worst Historical Day Analysis

### BTC Historical Extreme Moves (5-Minute Data)

Based on public BTC/USDT data from Binance (2019-2025):

| Timeframe | Event | Date | Move | Duration |
|-----------|-------|------|------|----------|
| **Worst 5-min bar** | COVID crash | March 12, 2020 ~12:45 UTC | **-11.5%** | 5 minutes |
| **Worst 30-min period** (6 bars) | COVID crash | March 12, 2020 12:30-13:00 UTC | **-18%** | 30 minutes |
| **Worst 4-hour period** | COVID crash | March 12-13, 2020 | **-32%** | 4 hours |
| **Worst single day** | COVID crash | March 12, 2020 | **-39%** | 24 hours |
| **Worst week** | COVID crash | March 9-15, 2020 | **-50%** | 7 days |
| **2nd worst 5-min bar** | China ban | May 19, 2021 ~07:15 UTC | **-8.5%** | 5 minutes |
| **2nd worst day** | China ban | May 19, 2021 | **-30%** | 24 hours |
| **3rd worst multi-day** | FTX collapse | Nov 6-11, 2022 | **-25%** | 5 days |

### Loss at Our Position Size

**Assumptions for calculation:**
- Position size: 5% of capital
- No leverage (spot-equivalent; for perpetuals, "5% of capital" means 5% notional exposure)
- Stop-loss at 3 ATR (during normal vol ~40% annualized, 3 ATR on 5-min bar ~ 0.8-1.2%)

**With stop-loss working normally (fills at expected price):**

| Scenario | BTC Move | Position (5% capital) | Stop-Loss Saves? | Max Loss |
|----------|----------|-----------------------|-------------------|----------|
| Worst 5-min bar (-11.5%) | -11.5% | 5% | Stop at -1.2% triggers | **0.06% of capital** |
| Worst 30-min (-18%) | -18% | 5% | Stop at -1.2% triggers on bar 1 | **0.06% of capital** |
| Worst 4-hour (-32%) | -32% | 5% | Stop triggered early | **0.06% of capital** |
| Worst day (-39%) | -39% | 5% | Stop triggered early | **0.06% of capital** |

**With stop-loss failing (extreme slippage, 5x worse than expected):**

| Scenario | BTC Move | Stop Target | Actual Fill (5x slip) | Loss |
|----------|----------|-------------|----------------------|------|
| Worst 5-min bar (-11.5%) | -11.5% | -1.2% | -6.0% | **0.30% of capital** |
| During thin book (COVID) | -11.5% | -1.2% | -11.5% (no fill, gap through) | **0.575% of capital** |

**With stop-loss failing AND using leverage:**

| Scenario | BTC Move | Leverage | Position (5% cap) | Effective Exposure | Gap-Through Loss |
|----------|----------|----------|--------------------|--------------------|------------------|
| Worst 5-min (-11.5%) | -11.5% | 5x | 5% capital | 25% of capital | **2.875% of capital** |
| Worst 5-min (-11.5%) | -11.5% | 10x | 5% capital | 50% of capital | **5.75% of capital** |
| Worst 5-min (-11.5%) | -11.5% | 20x | 5% capital | 100% of capital | **11.5% of capital** |

**Critical finding**: At 5% position size with no leverage, even the worst 5-min bar in BTC history costs only 0.575% of capital (with total stop failure). At 10x leverage, the same event costs 5.75% -- still survivable but painful. At 20x leverage, it threatens the 15% drawdown halt from a SINGLE BAR.

### Leverage Limit Derivation

To keep the worst historical 5-min bar under 5% of capital:
- Worst bar: -11.5%
- Position size: 5% of capital
- Max leverage: 5% / (11.5% * 5%) = **8.7x**

**Recommendation**: Maximum leverage of 5x. This keeps the worst historical 5-min bar at:
- 11.5% * 5% * 5x = **2.875% of capital** (survivable, within daily loss limit)

---

## 5. Reverse Stress Testing

**Starting point**: "What would cause a 20% drawdown?"

### Path 1: Single Catastrophic Event

For a 20% drawdown from ONE trade:
- At 5% position size, no leverage: requires BTC to move 400% against us in one bar (impossible)
- At 5% position size, 5x leverage: requires BTC to move 80% against us before stop (never happened in a 5-min bar)
- At 5% position size, 10x leverage: requires BTC to move 40% against us (never happened in 5-min; March 2020 was -39% over 24 hours)

**Conclusion**: A single-event 20% drawdown is nearly impossible at our position sizing, even at 10x leverage. It requires a 40%+ gap-through with no stop execution.

### Path 2: Cumulative Losses (Most Likely Path to 20%)

The real 20% drawdown path is death by a thousand cuts:

**Scenario**: Model accuracy drops to 48% for 2 weeks. At 30 trades/day, that is 420 trades.
- Expected loss per trade at 48% accuracy: ~0.015% of capital (after costs)
- 420 trades * 0.015% = ~6.3% drawdown

But: Daily loss limit (3%) triggers after ~200 trades. System halts.

**What bypasses the daily limit?**
1. System restarts each day (operator overrides halt) for 7 consecutive days
2. Each day: 3% daily loss = 21% cumulative

**Prevention**: After 3 consecutive daily limit triggers:
- **Automatic weekly halt fires at 5%** (day 2)
- **If operator overrides weekly halt**: drawdown gate progressively reduces position size
- **At 10% drawdown**: position size reduced by 60-75%
- **At 15%**: full halt, requires manual review

**The 20% drawdown requires**: Operator overriding ALL automated halts for 7+ consecutive days with a broken model. This is a human error, not a technical failure.

### Path 3: Operational Failure + Market Move

**Scenario**: Server crashes Friday night. Position is open. BTC drops 15% over the weekend. Exchange-side stop fails (exchange maintenance).

- Position: 5% of capital at 5x leverage = 25% effective exposure
- BTC move: -15%
- Loss: 25% * 15% = **3.75% of capital**

Not a 20% drawdown. But if combined with prior losses (already at -10% drawdown):
- Total: -13.75%. Still below 15% halt.

**For this scenario to cause 20% drawdown**:
- Must already be at -16% (past the halt) + this event
- Or: 10x leverage: 50% exposure * 15% = 7.5% loss. If starting from -12.5% drawdown: total = 20%.

### Path 4: Exchange Insolvency

**Scenario**: Binance becomes insolvent. All funds on exchange are lost.
- If 100% of trading capital is on Binance: 100% loss (Level 1 death)
- If 50% of trading capital is on Binance: 50% loss (Level 2 death)
- If 20% of net worth on Binance (our rule): 20% of net worth (Level 3 death)

**This is the most likely path to a 20% loss.** It is not a trading loss -- it is a custody loss.

### Reverse Stress Test Summary

| Drawdown | Most Likely Cause | Probability | Detectable in Advance? |
|----------|-------------------|-------------|----------------------|
| 5% | 2 bad trading days + daily limit | 20-30%/year | Yes (after day 1, alpha decay monitor fires) |
| 10% | 1 bad week + operator slow to react | 5-10%/year | Yes (weekly limit fires at 5%; 10% requires override) |
| 15% | Model breakdown + multiple overrides | 2-5%/year | Yes (but requires human to ACT on the signals) |
| 20% | Exchange insolvency OR human override of ALL halts | 1-3%/year | Exchange insolvency: no. Human override: yes (audit trail). |
| 50% | Exchange insolvency (50%+ capital on one exchange) | 1-2%/year | No (until it is too late) |

**Where the real risks are** (not where you think they are):
1. **#1 risk: Exchange insolvency** (unmitigable except by capital allocation)
2. **#2 risk: Human override of automated halts** (process failure, not technical failure)
3. **#3 risk: Position sizing bug** (10x intended size with leverage = instant catastrophe)
4. **#4 risk: Alpha decay undetected for months** (slow bleed without triggering daily limits)

---

## 6. The "Survive Everything" Position Size

### Constraint-Based Derivation

**Constraint 1**: Worst historical day (March 12, 2020: -39%) must cause <5% capital loss.
- Assume: stop-loss completely fails (gap through), full adverse move, 5x leverage
- Required: position_pct * leverage * 0.39 < 0.05
- position_pct * 5 * 0.39 < 0.05
- position_pct < 0.05 / 1.95 = **2.56%**

**Constraint 2**: Worst historical week (March 9-15, 2020: -50%) must cause <10% capital loss.
- Assume: stop-loss partially works (limits daily loss to 3%), but residual exposure adds up
- Over 5 trading days: 5 * 3% daily limit = 15%. But drawdown gate reduces position by 50% after day 2.
- Realistic worst case: 3% + 3% + 1.5% + 1% + 0.5% = 9%. Just under 10%. **Passes.**
- If stop-loss completely fails for the entire week: position_pct * 5 * 0.50 < 0.10 => position_pct < **4.0%**

**Constraint 3**: Exchange insolvency must cause <20% of net worth.
- Capital on exchange < 20% of net worth (NOT position size -- total capital on exchange)
- If net worth = $50K, max on any exchange = $10K

### Maximum Safe Position Size

| Leverage | Max Position (Survive Worst Day) | Max Position (Survive Worst Week) | Recommended |
|----------|----------------------------------|-----------------------------------|-------------|
| 1x (spot) | 12.8% | 20.0% | 10% |
| 3x | 4.3% | 6.7% | 4% |
| 5x | 2.6% | 4.0% | **2.5%** |
| 10x | 1.3% | 2.0% | 1.5% |
| 20x | 0.6% | 1.0% | 0.5% |

**System recommendation**:
- At 5x leverage: **2.5% position size** (not the current 5%)
- At no leverage (spot-equivalent): current 5% is safe
- The current 5% position size is ONLY safe at leverage <= 3x

**Action item**: Clarify whether the system uses leverage. If yes, reduce position_size_fraction from 0.05 to 0.025 (at 5x leverage) or 0.015 (at 10x leverage).

---

## 7. Stress Test Pass/Fail Criteria

### Mandatory Stress Tests

Every backtest configuration MUST pass ALL of the following:

| # | Test | Pass Criterion | Rationale |
|---|------|----------------|-----------|
| 1 | **March 2020 COVID crash** | Max DD < 15% | Worst crypto crash in history. If system can not survive this, it cannot survive anything. |
| 2 | **May 2021 China ban** | Max DD < 10% | Crypto-specific event; cross-market signals useless. Tests cascade detector. |
| 3 | **Nov 2022 FTX collapse** | Max DD < 10% | Slow-burn crisis. Tests weekly limit and regime detection. |
| 4 | **48h zero volatility** | Cost drag < 2% | Tests that system does not overtrade during dead markets. |
| 5 | **10 flash crashes in a week** | Max cumulative DD < 20% | Tests kill switch recovery and graduated re-entry. |
| 6 | **March 2024 ATH correction** | Max DD < 8% | Tests funding rate warning and position reduction. |
| 7 | **SVB crisis (correlation flip)** | Does not lose > 2% during +20% rally | Tests correlation breakdown handling. |

### Additional Stress Tests (Recommended)

| # | Test | Pass Criterion |
|---|------|----------------|
| 8 | Exchange API down for 15 min during crash | Position protected by exchange-side stop |
| 9 | Double execution scenario | Position tracker catches within 1 bar |
| 10 | Data feed returns stale prices for 5 min | System halts trading within 30s |
| 11 | Position sizing returns 10x normal | Hard limit catches before order submission |
| 12 | Funding rate -0.3% with position held | Funding cost < 0.5% of capital |

### Stress Test Automation

All stress tests run in CI/CD on every commit to `main`:
- Historical replay tests: use fetched data via `scripts/collect_history.py`
- Synthetic tests: use `SyntheticScenarioBuilder`
- Pass/fail is binary; any failure blocks the merge

---

## 8. The Nightmare Scenario Exercise

### Scenario: 3 AM, BTC drops 15% in 10 minutes

**Setup**: You are asleep. BTC is at $60,000. Your system has a long position (5% of capital at 5x leverage = $15,000 notional = 0.25 BTC). BTC drops to $51,000 in 10 minutes. Exchange API is slow (5-second response times).

**Walk-through of what SHOULD happen:**

**T+0 seconds (crash begins):**
- BTC starts falling
- Our system's exchange-side stop-loss was placed at entry: $60,000 - 3*ATR = ~$59,100 (assuming ATR ~$300 at normal vol)
- This stop-loss is ON THE EXCHANGE, not dependent on our system

**T+30 seconds (BTC at $59,100):**
- Exchange-side stop-loss triggers
- Stop is a stop-market order: fills at best available price
- In a thin book: fills at $58,800 (0.5% slippage on stop)
- Loss: ($60,000 - $58,800) / $60,000 = 2.0% on position
- Capital loss: 2.0% * 25% effective exposure = **0.5% of capital**
- Position is CLOSED. We are flat.

**T+10 minutes (BTC at $51,000):**
- We are flat. No additional loss.
- Our system may still be trying to connect (API is slow)
- Even if our system is completely down: we are already flat

**What could go wrong:**

| Failure | Impact | Probability |
|---------|--------|-------------|
| Exchange-side stop not placed (bug) | Stop never fires. Loss: 15% * 25% = **3.75% of capital** | Low (if tested) |
| Stop placed but exchange is down (BitMEX 2020) | Stop doesn't execute until exchange recovers | Very Low for Binance |
| Stop fills at much worse price (gap through) | Stop at $59,100, fills at $55,000. Loss: 8.3% * 25% = **2.08% of capital** | Low-Moderate |
| Our system comes back online, sees no position, opens a new one into the crash | Additional loss from new bad position | Low (cooldown logic) |

**Maximum possible loss in this scenario:**
- With exchange-side stop working: **0.5% of capital** (survived comfortably)
- With stop gap-through (extreme): **2-4% of capital** (within daily limit)
- With stop not placed (bug): **3.75% of capital** (within daily limit, but scary)
- With exchange down + no stop: **3.75% of capital** (exchange processes stop when it comes back)

**The kill switch hierarchy (independent of our system):**

1. **Exchange-side stop-loss**: First line of defense. Independent of our system. (MUST be placed at position entry)
2. **Our system's kill switch**: Monitors mark-to-market. Fires if daily loss > 3%. (Requires our system to be running)
3. **Independent monitoring**: Separate process/server that checks position PnL every 60 seconds. Sends alert + can close position via API. (Requires internet + exchange API)
4. **Human intervention**: Alert wakes you up. You manually close on mobile app. (Last resort, 10-30 minute delay)

**Critical finding**: Layers 2, 3, and 4 all require something to be running and connected. Only Layer 1 (exchange-side stop) works when everything else fails. **THE EXCHANGE-SIDE STOP IS THE ONLY DEFENSE THAT MATTERS AT 3 AM.**

### Action Items from Nightmare Scenario

1. **Every position entry MUST place an exchange-side stop-loss order simultaneously.** No exceptions. If the stop-loss order fails, the position entry must be cancelled.
2. **The stop-loss order must be a stop-market order (not stop-limit).** Stop-limit can fail to fill in a fast market.
3. **Verify exchange-side stop exists every 60 seconds.** If the stop is missing (cancelled by exchange, expired), re-place it immediately or close the position.
4. **Independent monitoring process** (separate server, separate API key) must run 24/7 and have the ability to close all positions.

---

## 9. Unknown Unknowns

### Novel Attack Vectors in Crypto

| Risk | Description | Our Exposure |
|------|-------------|-------------|
| **Oracle manipulation** | Manipulator pushes price on one exchange to trigger liquidations on another | Low (we trade on major exchanges with large volume) |
| **Sandwich attack on stops** | Attacker sees our stop-loss level and pushes price to trigger it, then reverses | Low-Moderate (our stops are on-exchange, not visible to public; but predictable levels like round numbers are vulnerable) |
| **MEV-equivalent on CEX** | Exchange front-runs customer orders (alleged at some exchanges) | Low-Moderate (Binance is heavily scrutinized but not immune) |
| **Stablecoin run** | Bank-run dynamics on USDT/USDC leading to cascading depegs | Moderate (all PnL in USDT) |
| **AI-driven regime shifts** | As more ML systems trade, markets adapt faster than our retraining window | Moderate-High (this is the long-term existential risk) |
| **Quantum computing** | Breaks Bitcoin's cryptographic security | Very Low (5+ year horizon minimum) |

### Regulatory Surprises

| Risk | Description | Impact |
|------|-------------|--------|
| **MiCA implementation (EU)** | Stricter reporting, potential leverage caps | May limit leverage to 2x |
| **US crypto derivatives ban** | SEC/CFTC enforcement against offshore platforms | Must move to regulated venue with higher fees |
| **Global FATF travel rule** | Enhanced KYC/AML requirements for all crypto transfers | Withdrawal delays |
| **Windfall tax on crypto profits** | Emergency tax legislation in multiple jurisdictions | Retrospective profit reduction |

### Technical Changes to BTC/Exchanges

| Risk | Description | Impact |
|------|-------------|--------|
| **Binance Binance-specific upgrade** | Matching engine change affecting order execution | Slippage model becomes inaccurate |
| **BTC protocol upgrade** | Hard fork or soft fork changes block dynamics | Mempool features become unreliable |
| **Exchange consolidation** | Merger/acquisition changes liquidity dynamics | Cross-exchange signals break |
| **Lightning Network adoption** | Large transactions move off-chain | Whale detection (on-chain) becomes useless |
| **ETF dominance** | ETF flows dominate price discovery (not exchange order books) | Microstructure features degrade as ETF flow leads price |

### The Universal Defense: Position Sizing

**Position sizing is the ONE defense that works against ALL unknown unknowns.**

No matter what happens -- market crash, exchange failure, regulatory surprise, novel attack, stablecoin depeg, or something we have not thought of:

- At 2.5% position size with 5x leverage (12.5% effective exposure): the maximum possible loss is 12.5% of capital (if the position goes to zero instantly)
- At 5% position size with no leverage: the maximum possible loss is 5% of capital

**Position sizing cannot be bypassed by any market event, technical failure, or regulatory action.**

This is why position sizing is not just one risk control -- it is THE risk control. Everything else (stops, kill switches, regime detection) reduces the probability of loss. Position sizing caps the MAGNITUDE of loss regardless of cause.

---

## 10. Building the Risk Budget

### Starting from the Total Loss Tolerance

**Question**: "How much can I afford to lose before I stop trading this system?"

**Answer structure** (to be filled in by operator):

| Parameter | Conservative | Moderate | Aggressive |
|-----------|-------------|----------|------------|
| Total risk capital | $5,000 | $10,000 | $25,000 |
| Max acceptable total loss | $1,500 (30%) | $2,000 (20%) | $5,000 (20%) |
| Max on any exchange | $1,000 (20%) | $2,000 (20%) | $5,000 (20%) |
| Monthly loss limit | $150 (3%) | $300 (3%) | $750 (3%) |
| Max leverage | 3x | 5x | 5x |
| Position size | 4% | 2.5% | 2.5% |

### Working Backwards from P(Losing $X)

**Target**: P(losing 50% of capital) < 0.1% per year.

**Required conditions**:

At $10,000 capital, 50% loss = $5,000:
- Path 1 (exchange insolvency): Keep max $2,000 on exchange (20%). Remaining $8,000 elsewhere. Exchange insolvency = -$2,000 (20% loss, not 50%).
- Path 2 (trading losses): At 3% daily limit, 5% weekly limit, 15% drawdown halt: maximum trading loss = $1,500 per continuous period. After halt + 48h cooldown + revalidation: another $1,500 max. Reaching $5,000 requires 3+ complete halt-restart-loss cycles.
- P(3+ halt cycles all hitting max loss) < 0.1% per year. **Passes.**

### Monte Carlo Verification

**Setup** (to be run via `research/RR-montecarlo-validation-methods.md` framework):

```
Simulation parameters:
- Win rate: 53% (conservative)
- Average win: 0.12% per trade (after costs)
- Average loss: -0.10% per trade (after costs)
- Trades per day: 15 (confidence gating reduces from 288)
- Position size: 2.5% of capital
- Leverage: 5x (12.5% effective exposure)
- Daily loss limit: 3%
- Weekly loss limit: 5%
- Max drawdown halt: 15%
- Simulation: 10,000 years of daily returns

Expected outputs:
- P(losing 20% of capital): < 5% per year
- P(losing 50% of capital): < 0.1% per year
- P(positive annual return): > 60%
- Median annual return: 15-30%
- 5th percentile annual return: -12%
- 95th percentile max drawdown: 12%
```

### Risk Budget Allocation

| Risk Source | Budget (% of max loss tolerance) | Controls |
|-------------|----------------------------------|----------|
| Trading losses (market risk) | 60% | Position sizing, stops, kill switches |
| Execution failures | 10% | Order validation, dedup, timeouts |
| Operational failures | 10% | Monitoring, redundancy, auto-restart |
| Exchange/custody risk | 15% | Capital allocation limit (20% per exchange) |
| Unknown unknowns | 5% | Position sizing (universal defense) |
| **Total** | **100%** | |

### Documentation Requirement

Before going live, document and sign off on:

```
RISK ACKNOWLEDGMENT

At the following configuration:
- Position size: ___% of capital
- Max leverage: ___x
- Capital on exchange: $___
- Daily loss limit: ___%
- Weekly loss limit: ___%
- Max drawdown halt: ___%

The maximum possible losses are:
- Worst single trade: ___% of capital ($___)
- Worst single day: ___% of capital ($___) [= daily limit]
- Worst single week: ___% of capital ($___) [= weekly limit]
- Worst month: ___% of capital ($___)
- Worst year (P95): ___% of capital ($___) [from Monte Carlo]
- Exchange insolvency: ___% of net worth ($___) [= capital allocation limit]
- Total ruin (P99.9 per year): ___% of net worth ($___)

I understand and accept these risks.

Date: ___
Signature: ___
```

---

## Summary: The 10 Commandments of Survival

1. **Position sizing is the universal defense.** No stop-loss, kill switch, or detection algorithm protects you as reliably as small position sizes. At 2.5% position with 5x leverage, the worst 5-minute bar in BTC history costs 2.9% of capital.

2. **Exchange-side stops are the only stops that matter.** Every position entry MUST simultaneously place an exchange-side stop-market order. If the stop order fails, the position entry must be cancelled. This is non-negotiable.

3. **Never put more than 20% of net worth on any single exchange.** Exchange insolvency is the #1 path to catastrophic loss. No technical defense exists. Only capital allocation limits help.

4. **Kill switches must be layered and independent.** Daily limit (3%), weekly limit (5%), max drawdown (15%). Each must be able to fire independently. None should require the trading system to be running.

5. **After a halt, the default is STAY HALTED.** Automatic re-entry requires: 48h minimum cooldown, backtesting on recent data shows edge still exists, and position size reduced by 50% for first 3 days.

6. **Alpha decay is certain; plan for it.** Budget for a 1-6 month useful life per ML pattern. Monitor continuously. When Sharpe drops below 0.3 for 30 days, stop and re-research.

7. **The position sizing bug is the #1 operational risk.** A decimal error turning 2.5% into 25% with 10x leverage = 250% exposure = instant ruin. Defense: hard limits in code, config, AND exchange.

8. **Human override of automated halts is the #2 risk.** The system is designed to stop you from losing money. If you override it, the system cannot protect you. Require a written justification for every override.

9. **Leverage multiplies everything -- including mistakes.** At 5x leverage, all losses and position sizing errors are 5x worse. At 10x, they are 10x worse. The "survive everything" position size at 10x is 1.5%.

10. **This system exists to preserve capital first, generate returns second.** Every design decision must pass the test: "Does this survive the worst case?" If it does not, it does not ship.

---

## Action Items

### Immediate (Before Going Live)

- [ ] Clarify and document the leverage policy. If leverage > 3x, reduce position_size_fraction from 0.05 to 0.025 or lower.
- [ ] Implement exchange-side stop-loss placement as part of the position entry atomic operation (both succeed or both fail).
- [ ] Add a hard position size limit at 3 levels: code, config, and exchange-side.
- [ ] Set up independent monitoring process on a separate server/process with its own API key.
- [ ] Establish the capital allocation policy: max 20% of net worth on any single exchange.
- [ ] Run Monte Carlo simulation (10K years) at intended position size to verify P(ruin) < 0.1%.

### Sprint 9 (Risk Management)

- [ ] Implement PositionTracker with exchange reconciliation every 60s
- [ ] Implement kill switches (daily/weekly/drawdown) that persist across restarts
- [ ] Implement drawdown gate with progressive position reduction
- [ ] Implement volatility guard (min/max vol filters)
- [ ] Implement position sizer with quarter-Kelly + hard limits
- [ ] Implement risk manager that orchestrates all components

### Before Every Backtest

- [ ] Verify all 7 mandatory stress tests pass
- [ ] Verify position sizing calculation is correct at intended leverage
- [ ] Verify exchange-side stop is modeled in the backtest (not just system-side stop)

### Quarterly Review

- [ ] Re-run Monte Carlo with updated trade statistics
- [ ] Review alpha decay metrics (rolling Sharpe, feature importance drift)
- [ ] Verify exchange capital allocation is within 20% limit
- [ ] Review and test exchange-side stop placement code
- [ ] Update this document with any new risk categories discovered
