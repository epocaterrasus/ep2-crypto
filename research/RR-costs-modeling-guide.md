# Transaction Cost Modeling for Crypto Backtesting

**The #1 reason strategies fail from backtest to live.**

Every basis point of unmodeled cost is a basis point of fantasy alpha. This document provides exact models, calibration methods, Python implementations, and sensitivity analysis for every cost component in crypto perpetual futures trading.

---

## Table of Contents

1. [Fee Model](#1-fee-model)
2. [Slippage Model (Empirical)](#2-slippage-model)
3. [Market Impact Model](#3-market-impact-model)
4. [Funding Rate Model](#4-funding-rate-model)
5. [Spread Model](#5-spread-model)
6. [Borrowing Costs](#6-borrowing-costs)
7. [Opportunity Cost of Capital](#7-opportunity-cost-of-capital)
8. [Rebate Modeling](#8-rebate-modeling)
9. [Cost-Aware Backtest Engine](#9-cost-aware-backtest-engine)
10. [Cost Sensitivity Analysis](#10-cost-sensitivity-analysis)
11. [Break-Even Analysis](#11-break-even-analysis)
12. [Intraday Cost Patterns](#12-intraday-cost-patterns)

---

## 1. Fee Model

### Exchange Fee Structures (Current as of Q1 2026)

#### Binance USDT-M Futures

| VIP Tier | 30-Day Volume (USDT) | BNB Balance | Maker | Taker |
|----------|---------------------|-------------|-------|-------|
| VIP 0 | < 15M | < 25 BNB | 0.0200% | 0.0500% |
| VIP 1 | >= 15M | >= 25 BNB | 0.0160% | 0.0400% |
| VIP 2 | >= 100M | >= 100 BNB | 0.0140% | 0.0350% |
| VIP 3 | >= 500M | >= 250 BNB | 0.0120% | 0.0320% |
| VIP 4 | >= 1B | >= 500 BNB | 0.0100% | 0.0300% |
| VIP 5 | >= 3B | >= 1,000 BNB | 0.0080% | 0.0270% |
| VIP 6 | >= 5B | >= 1,750 BNB | 0.0060% | 0.0250% |
| VIP 7 | >= 10B | >= 3,000 BNB | 0.0040% | 0.0220% |
| VIP 8 | >= 25B | >= 4,500 BNB | 0.0020% | 0.0200% |
| VIP 9 | >= 50B | >= 5,500 BNB | 0.0000% | 0.0170% |

**BNB Discount:** 10% off when paying fees with BNB (applies to USDT-M and COIN-M futures).
- VIP 0 with BNB: 0.0180% maker / 0.0450% taker
- This stacks with VIP tier discounts

**Referral Rebate:** Up to 20% commission kickback (10% to referrer, 10% to referee) on trading fees.

#### Bybit Perpetual Futures

| VIP Tier | 30-Day Volume (USDT) | Asset Balance | Maker | Taker |
|----------|---------------------|---------------|-------|-------|
| VIP 0 | < 10M | < $100K | 0.0200% | 0.0550% |
| VIP 1 | >= 10M | >= $100K | 0.0180% | 0.0400% |
| VIP 2 | >= 25M | >= $250K | 0.0160% | 0.0375% |
| VIP 3 | >= 50M | >= $500K | 0.0140% | 0.0350% |
| VIP 4 | >= 100M | >= $1M | 0.0120% | 0.0320% |
| VIP 5 | >= 250M | >= $2M | 0.0100% | 0.0300% |
| Supreme VIP | Market Maker | Negotiated | 0.0000% | 0.0300% |

### Dynamic Fee Model

Fees should be modeled dynamically based on projected monthly volume, because your VIP tier changes as you trade more.

**Key insight:** A strategy that projects $50M/month in volume should NOT use VIP 0 fees in its backtest -- it would reach VIP 2-3 within the first month. Conversely, don't assume VIP 5+ unless you have the capital to maintain the BNB/balance requirements.

### Calibration Method

1. Estimate monthly notional volume from strategy signal frequency and position size
2. Map to the appropriate VIP tier
3. Determine maker vs taker ratio based on order type (see Section 8)
4. Apply BNB discount if holding BNB is part of the strategy
5. Factor referral rebates if applicable

### Fee Model Parameters

```
Conservative (new account, all taker):
  Entry: 0.0500%, Exit: 0.0500%, Round-trip: 0.1000% = 10.0 bps

Moderate (VIP 1, mixed maker/taker):
  Entry: 0.0280% (50/50 maker/taker), Exit: 0.0280%, Round-trip: 0.0560% = 5.6 bps

Optimistic (VIP 3+, mostly maker with BNB):
  Entry: 0.0108%, Exit: 0.0108%, Round-trip: 0.0216% = 2.16 bps

Professional (VIP 7+, post-only orders):
  Entry: 0.0036%, Exit: 0.0036%, Round-trip: 0.0072% = 0.72 bps
```

---

## 2. Slippage Model (Empirical)

### Order Book Depth Reality for BTCUSDT Perpetual

Based on empirical data from Binance BTCUSDT perpetual (2024-2026):

**Typical Market Depth at 10 bps from Mid-Price:**
- Peak (11:00 UTC, EU/Asia overlap): ~$3.86M per side
- Trough (21:00 UTC, US close): ~$2.71M per side
- Average: ~$3.3M per side
- Peak-to-trough ratio: 1.42x

**Estimated Market Depth at Various Levels (Binance BTCUSDT Perp, normal conditions):**

| Distance from Mid | Typical Bid Depth | Typical Ask Depth |
|-------------------|-------------------|-------------------|
| 1 bps (0.01%) | $0.5-1.0M | $0.5-1.0M |
| 5 bps (0.05%) | $2.0-3.5M | $2.0-3.5M |
| 10 bps (0.10%) | $3.0-5.0M | $3.0-5.0M |
| 50 bps (0.50%) | $15-25M | $15-25M |
| 100 bps (1.00%) | $30-60M | $30-60M |

### Expected Slippage by Order Size

Using empirical order book depth estimates:

| Order Size | Normal Market | Low Vol | High Vol | Flash Crash |
|------------|--------------|---------|----------|-------------|
| $1K | 0.0 bps | 0.0 bps | 0.5 bps | 1-2 bps |
| $10K | 0.1 bps | 0.1 bps | 0.5-1 bps | 2-5 bps |
| $50K | 0.5 bps | 0.3 bps | 1-3 bps | 5-15 bps |
| $100K | 1.0 bps | 0.5 bps | 2-5 bps | 10-30 bps |
| $500K | 3-5 bps | 2-3 bps | 5-15 bps | 30-100 bps |
| $1M | 5-10 bps | 3-5 bps | 10-30 bps | 50-200+ bps |

### Session-Dependent Slippage

Empirical findings from Amberdata research:

- **European session (08:00-16:00 UTC):** Best liquidity, $3.61M avg depth at 10bps. Slippage ~baseline.
- **Asian session (00:00-08:00 UTC):** Good liquidity, $3.58M avg depth. Slippage ~1.01x baseline.
- **US session (16:00-24:00 UTC):** Worst liquidity, $3.32M avg depth (9% below EU). Slippage ~1.09x baseline.
- **Danger zone (21:00 UTC):** US close, 25% depth reduction. Slippage ~1.42x baseline.
- **Peak (11:00 UTC):** EU/Asia overlap. Slippage ~0.85x baseline.

### Dynamic Slippage Formula

```
slippage_bps = base_spread_bps/2
             + impact_bps * (order_size / depth_at_level)^0.5
             + volatility_multiplier * (realized_vol / normal_vol)
```

Where:
- `base_spread_bps`: Current bid-ask spread in basis points
- `impact_bps`: Calibrated impact coefficient (~1.0 for BTCUSDT)
- `order_size`: Notional order value in USD
- `depth_at_level`: Available liquidity within expected price impact range
- `realized_vol`: Current 5-min realized volatility
- `normal_vol`: Long-run average 5-min volatility

### Volatility Regime Multipliers

| Regime | Realized Vol / Normal Vol | Slippage Multiplier |
|--------|--------------------------|-------------------|
| Low volatility | < 0.5 | 0.7x |
| Normal | 0.5 - 1.5 | 1.0x |
| Elevated | 1.5 - 3.0 | 1.5x |
| High volatility | 3.0 - 5.0 | 2.5x |
| Extreme / flash crash | > 5.0 | 5.0-10.0x |

---

## 3. Market Impact Model

### Almgren-Chriss Decomposition

Market impact decomposes into two components:

**Permanent Impact (g):** The price change that persists after your order is fully executed. Caused by information revelation -- the market updates its estimate of fair value.

```
g(v) = theta * v
```

Where `v` is the trading rate (shares/time) and `theta` is the permanent impact coefficient.

**Temporary Impact (h):** The additional cost of immediacy that decays after execution. Caused by liquidity consumption.

```
h(v) = epsilon * sign(v) + eta * v
```

Where `epsilon` is a fixed cost (half-spread) and `eta` is the temporary impact coefficient.

### The Square-Root Law

The universal law of market impact, validated across equities, futures, options, and crypto:

```
I(Q) = Y * sigma * sqrt(Q / V)
```

Where:
- `I(Q)` = expected price impact as a fraction of price
- `Y` = dimensionless prefactor (~1.0, remarkably universal)
- `sigma` = daily price volatility (standard deviation of daily returns)
- `Q` = total order size (in currency units)
- `V` = average daily traded volume (in currency units)

**Empirical validation:** The exponent delta in `I(Q) ~ Q^delta` is consistently measured at delta ~ 0.5 across all asset classes, with error bars below 0.01. This is one of the most robust empirical laws in finance.

### Calibration for BTCUSDT Perpetual

**Parameters (typical values, 2025-2026):**
- Daily volume `V`: $15-25 billion (Binance BTCUSDT perp alone)
- Daily volatility `sigma`: 1.5-3.0% (varies by regime)
- Prefactor `Y`: ~1.0 (universal)

**Example calculations:**

For a $100K order on a day with sigma=2%, V=$20B:
```
I = 1.0 * 0.02 * sqrt(100,000 / 20,000,000,000)
  = 0.02 * sqrt(5e-6)
  = 0.02 * 0.002236
  = 0.0000447 = 0.45 bps
```

For a $1M order:
```
I = 1.0 * 0.02 * sqrt(1,000,000 / 20,000,000,000)
  = 0.02 * sqrt(5e-5)
  = 0.02 * 0.00707
  = 0.000141 = 1.41 bps
```

For a $10M order:
```
I = 1.0 * 0.02 * sqrt(10,000,000 / 20,000,000,000)
  = 0.02 * sqrt(5e-4)
  = 0.02 * 0.02236
  = 0.000447 = 4.47 bps
```

### Permanent vs Temporary Decomposition

Empirical research suggests approximately:
- **Permanent impact:** ~2/3 of total impact (persists indefinitely)
- **Temporary impact:** ~1/3 of total impact (decays over minutes)

This means: even after your order is done, 2/3 of the price move you caused is permanent. If you're a momentum trader, this works in your favor. If you're mean-reverting, it's an additional cost.

### How Impact Changes with Volatility

Impact scales linearly with volatility (the sigma term in the square-root law). During a 2x vol spike:
- All impact numbers double
- A $100K order that normally costs 0.45 bps now costs 0.90 bps
- This is why volatile periods are doubly expensive: wider spreads AND higher impact

### Calibration from Historical Data

To calibrate Y and confirm the square-root exponent:

1. Collect trade data with known order sizes and resulting price changes
2. Compute `I_observed = |price_after - price_before| / price_before` for each metaorder
3. Regress `log(I_observed)` on `log(Q/V)` -- slope should be ~0.5
4. The intercept gives `log(Y * sigma)`
5. Since sigma is observable, extract Y

In practice, for BTCUSDT on major exchanges, Y ~ 0.8-1.2 with most estimates clustered around 1.0.

---

## 4. Funding Rate Model

### Mechanism

Perpetual futures have no expiry, so funding rates anchor the perp price to spot:
- **Positive funding:** Longs pay shorts (perp premium over spot)
- **Negative funding:** Shorts pay longs (perp discount to spot)
- **Settlement:** Every 8 hours (00:00, 08:00, 16:00 UTC on Binance/Bybit)

### Funding Rate Formula

```
Funding Rate = clamp(Premium Index + Interest Rate Component, -0.05%, +0.05%)

Interest Rate Component = 0.01% / 8h  (i.e., 0.03% daily, ~10.95% annualized)

Premium Index = (TWAP(perp_price - index_price) / index_price)
```

The clamp at +/-0.05% per period means the maximum annualized funding cost is:
```
max_annual = 0.05% * 3 * 365 = 54.75%
```

But the 0.01% baseline means that under normal conditions, funding gravitates to ~0.01%/8h.

### Historical Distribution (Empirical, BTC Perpetuals)

**Q3 2025 Cross-Exchange Data:**

| Exchange | Mean (/8h) | Std Dev | At 0.01% Exactly | Max Observed |
|----------|-----------|---------|------------------|-------------|
| BitMEX | 0.0081% | 0.0049% | 78.19% | ~0.05% |
| Binance | 0.0057% | 0.0039% | 30.70% | ~0.05% |
| Hyperliquid | 0.0120% | 0.0097% | N/A | 0.0672% |

**Key Statistics for Backtesting (Binance BTC):**
- Mean: 0.0057% per 8h = 0.0171% per day = 6.24% annualized
- Median: 0.01% per 8h (mode, due to the 0.01% baseline)
- 5th percentile: ~-0.01% per 8h
- 95th percentile: ~0.03% per 8h
- 99th percentile: ~0.05% per 8h (clamped)
- Positive 92%+ of the time (structural long bias in crypto)

**Historical Evolution:**
- Pre-2021: Funding rates frequently exceeded 0.20%/8h during bull markets (200%+ annualized)
- 2021 bull: Peaks of 0.15-0.20%/8h during euphoric phases
- 2022-2023 bear: Often negative, down to -0.15%/8h during March 2020 crash
- 2024-2026: Much tighter, institutional capital compresses extremes. Ethena alone has $7.83B in arbitrage capital enforcing convergence

### Funding Rate Regime Classification

| Regime | Rate Range (/8h) | Annualized | Interpretation |
|--------|-----------------|------------|----------------|
| Bearish extreme | < -0.05% | < -54.75% | Panic/capitulation |
| Bearish | -0.05% to -0.01% | -54% to -11% | Short crowding |
| Neutral | -0.01% to 0.02% | -11% to 22% | Normal market |
| Bullish | 0.02% to 0.05% | 22% to 55% | Long crowding |
| Bullish extreme | > 0.05% (clamped) | 55%+ | Euphoria, top signal |

### Modeling Funding Costs in Backtests

For any position held across a funding timestamp:

```
funding_cost = position_value * funding_rate * (1 if long else -1)
```

**Critical nuances:**
1. You only pay/receive if you hold at the exact funding timestamp
2. Closing 1 second before funding = zero cost
3. The funding rate is KNOWN 8 hours in advance (it's computed from the prior period)
4. For backtesting: accumulate funding at each 8h mark for all open positions
5. Funding is a TRANSFER between longs and shorts -- it's not paid to the exchange

### Dynamic Funding Cost Estimation

For position duration `d` hours:
```
expected_funding_payments = floor(d / 8)
expected_funding_cost = expected_funding_payments * E[funding_rate] * position_value

# Using Binance mean:
expected_funding_cost = floor(d/8) * 0.0057% * position_value  (if long)
expected_funding_cost = -floor(d/8) * 0.0057% * position_value  (if short, you RECEIVE)
```

**For a strategy with average hold time of 2 hours:** Zero funding cost (never crosses an 8h boundary).
**For a strategy with average hold time of 24 hours:** ~3 funding payments = 0.017% cost if long.
**For a strategy holding 7 days:** ~21 payments = 0.12% cost if long.

---

## 5. Spread Model

### What is the Spread Cost?

When you cross the spread to execute a market order, you immediately lose half the spread. This is the most basic and unavoidable transaction cost.

```
spread_cost = spread / 2 = (best_ask - best_bid) / (2 * mid_price)
```

### BTCUSDT Perpetual Spread Distribution (Binance)

**Normal Market Conditions (2024-2026):**
- Minimum tick size: $0.10
- At BTC = $100,000: minimum spread = 0.10 / 100,000 = 0.1 bps
- Typical quoted spread: 0.1-0.3 bps (1-3 ticks)
- Median spread: ~0.2 bps
- Mean spread: ~0.3 bps (skewed by occasional widenings)

**By Volatility Regime:**

| Condition | Typical Spread | Effective Half-Spread Cost |
|-----------|---------------|---------------------------|
| Low vol, peak hours | 0.1-0.2 bps | 0.05-0.1 bps |
| Normal | 0.2-0.5 bps | 0.1-0.25 bps |
| Elevated vol | 0.5-2.0 bps | 0.25-1.0 bps |
| High vol event | 2-10 bps | 1-5 bps |
| Flash crash | 10-100+ bps | 5-50+ bps |

### Spread as a Function of Time-of-Day

Based on the liquidity cycle (depth as a proxy for spread tightness):

```
spread_multiplier(hour_utc) = {
    0-7 (Asia):     1.00  (good liquidity, tight spreads)
    8-11 (EU open): 0.90  (best liquidity, tightest spreads)
    11-15 (EU/US):  0.85  (peak overlap, tightest)
    16-20 (US):     1.05  (slightly wider)
    21-23 (thin):   1.30  (widest, danger zone)
}
```

### Spread as a Function of Volatility

Empirically, spread scales roughly linearly with short-term realized volatility:

```
spread = base_spread * (1 + k * (realized_vol / normal_vol - 1))
```

Where `k ~ 0.5-1.0` (spread is sticky, doesn't fully track vol).

### Spread During Flash Crashes

During the March 2020 COVID crash and similar events:
- BTCUSDT spreads widened to 50-200 bps
- Order book depth collapsed by 80-95%
- Recovery to normal spreads took 5-30 minutes
- Market maker algorithms shut down or widen dramatically

**For backtesting:** If your strategy trades during high-vol events, you MUST model spread widening. Using normal spreads during crashes will massively overestimate performance.

### Order Book State Variables for Spread Prediction

The spread can be modeled as:
```
log(spread) = beta_0
            + beta_1 * log(realized_vol)
            + beta_2 * book_imbalance
            + beta_3 * trade_intensity
            + beta_4 * time_of_day_factor
            + epsilon
```

Typical R-squared: 0.3-0.5 (spreads are partially predictable).

---

## 6. Borrowing Costs

### Perpetual Futures: No Direct Borrowing

Unlike spot margin trading, perpetual futures do NOT have explicit borrowing costs. The "cost of shorting" in perps is embedded in:

1. **Funding rate:** If funding is positive and you're short, you RECEIVE funding (it's income, not cost)
2. **Margin interest:** Some exchanges charge interest on the margin used, but for USDT-M perps on Binance/Bybit, margin in USDT is typically not interest-bearing
3. **No stock-borrow-like fee:** Unlike equities, there's no "hard-to-borrow" fee for crypto shorts

### Cross-Margin vs Isolated Margin Implications

**Isolated Margin:**
- Margin locked per position
- Funding only affects that position's margin
- Liquidation risk limited to allocated margin
- Capital inefficiency: unused margin earns nothing

**Cross-Margin:**
- Shared balance across positions
- More capital efficient (unrealized PnL from one position can support another)
- Higher liquidation risk (entire balance at risk)
- For cost modeling: same fees, but less capital locked up (affects opportunity cost calculation)

### Spot Margin Short Selling Costs

If shorting via spot margin instead of perps:
- Binance: 0.02-0.06% daily interest on borrowed BTC (~7-22% annualized)
- Bybit: Similar ranges, tiered by VIP
- These are SIGNIFICANTLY more expensive than perp funding for most holding periods

**Conclusion for backtesting:** Use perps for short exposure. The only "borrowing cost" is the funding rate, which is often negative (i.e., shorts get PAID) in crypto.

### Hidden Costs of Short Positions

1. **Liquidation risk asymmetry:** Shorts have unlimited upside risk (price can go to infinity). Required margin is higher for shorts in some exchange modes.
2. **Funding rate regime risk:** A short held through a bull run will pay increasingly positive funding
3. **ADL (Auto-Deleveraging) risk:** In extreme moves, your profitable short may be forcibly closed

---

## 7. Opportunity Cost of Capital

### The Problem

Capital sitting in an exchange wallet earns 0% (or near-zero). If that capital could earn a risk-free-ish return elsewhere, that's a real cost of running the strategy.

### Risk-Free Alternatives (2025-2026 Rates)

| Instrument | Typical APY | Risk Level | Liquidity |
|------------|------------|------------|-----------|
| USDT/USDC in Aave/Compound | 4-7% | Smart contract risk | Minutes |
| ETH staking (native) | 3-4% | Slashing risk | Days |
| Liquid staking (stETH) | 3-4% | Smart contract risk | Minutes |
| T-bills (via RWA tokens) | 4-5% | Minimal | Days |
| Exchange earn products | 2-5% | Counterparty risk | Variable |
| Cash/T-bills (TradFi) | 4.5-5.0% | Minimal | Days |

### Modeling Opportunity Cost

For a backtest with average capital deployed `C_deployed` and total capital `C_total`:

```
opportunity_cost_daily = (C_total - C_deployed) * risk_free_rate / 365
```

But even deployed capital has opportunity cost -- it could be earning yield elsewhere:

```
total_opportunity_cost = C_total * risk_free_rate * (days_in_backtest / 365)
```

**For a strategy using $100K in capital over 1 year at 5% opportunity cost:**
```
annual_opportunity_cost = $100,000 * 0.05 = $5,000
```

This means the strategy needs to generate AT LEAST $5,000 in absolute returns just to break even vs. doing nothing.

### Practical Adjustments

1. **Margin efficiency:** If using 5x leverage, you only need $20K in margin for $100K notional. The other $80K could be earning yield.
2. **Exchange earn while trading:** Some exchanges let you earn interest on margin balances (Bybit's earn vault). This reduces but doesn't eliminate opportunity cost.
3. **Cross-margin benefits:** Cross-margin lets you use unrealized PnL as margin, reducing total capital needed.

### For the Backtest

```
hurdle_rate_daily = risk_free_rate / 365  # What you must beat daily
strategy_return - hurdle_rate_daily = true_alpha
```

If your strategy makes 15% annualized but the risk-free rate is 5%, your true alpha is 10%.

---

## 8. Rebate Modeling

### Maker Rebate Opportunity

Some exchanges offer negative maker fees (rebates) at higher VIP tiers:
- Binance VIP 9: 0.0000% maker (break-even, not a rebate)
- Some exchanges offer true negative maker fees (-0.005% to -0.025%)
- Bybit Supreme VIP: 0.0000% maker

### Post-Only Order Economics

Post-only orders guarantee maker execution (order is rejected if it would cross the spread). The fee savings vs taker:

```
savings_per_trade = taker_fee - maker_fee
```

At VIP 0 Binance: 0.0500% - 0.0200% = 0.0300% = 3 bps per trade
Round-trip with both sides maker: 0.0400% vs 0.1000% = 6 bps saved

### The Fill Probability Problem

The critical issue: post-only orders may not fill. The market must come to you.

**Fill Probability Factors:**
1. **Queue position:** Earlier in queue = higher fill probability
2. **Market direction:** If price is moving toward your order, fill probability increases. But this means adverse selection -- you fill when it's bad for you.
3. **Time in queue:** Longer wait = higher fill probability, but higher risk of adverse price movement
4. **Volatility:** Higher vol = more fills, but also more adverse selection

### Adverse Selection Cost

The cruel reality of maker orders: most fills happen when the price moves THROUGH your limit price. This means:

```
actual_maker_cost = -maker_rebate + adverse_selection_cost
```

If adverse selection exceeds the rebate, you're WORSE off than paying taker fees.

**Empirical estimates for BTCUSDT:**
- Average adverse selection on maker fills: 0.5-2.0 bps
- Maker rebate at VIP 0: 0 bps (just lower fee)
- Maker fee savings vs taker: 3 bps
- Net benefit: 1-2.5 bps (still worth it, usually)

### Modeling Maker Fill Probability

```
P(fill | limit_buy_at_bid) = f(time, volatility, queue_position, order_flow)
```

Simplified model:
```
P(fill within T seconds) = 1 - exp(-lambda * T)

Where lambda = (trade_intensity * P(price_crosses_level))
```

For BTCUSDT with typical trade intensity of 10-50 trades/second:
- P(fill within 1 second at best bid): ~5-15%
- P(fill within 10 seconds): ~30-60%
- P(fill within 60 seconds): ~70-95%
- P(fill within 300 seconds): ~95-99%

### Hybrid Strategy for Backtesting

Model a realistic mix of maker and taker fills:

```
For each trade signal:
  if urgency == "high" (momentum signal, decaying alpha):
    use taker fees, assume immediate fill
  elif urgency == "medium":
    try maker, fall back to taker after timeout
    P(maker_fill) = 0.6, P(taker_fallback) = 0.4
    effective_fee = 0.6 * maker_fee + 0.4 * taker_fee
  elif urgency == "low" (mean reversion, patient):
    use maker with high probability
    P(maker_fill) = 0.85, P(taker_fallback) = 0.15
    effective_fee = 0.85 * maker_fee + 0.15 * taker_fee
```

---

## 9. Cost-Aware Backtest Engine

### Complete Implementation

See `src/ep2_crypto/backtest/cost_engine.py` for the full implementation.

The cost engine is designed as a composable system with these components:

1. **FeeModel** -- exchange-specific, VIP-tier-aware fee calculation
2. **SlippageModel** -- order-size and volatility-dependent slippage
3. **MarketImpactModel** -- square-root law permanent + temporary impact
4. **FundingRateModel** -- 8-hour funding accumulation
5. **SpreadModel** -- time-of-day and volatility-aware spread costs
6. **OpportunityCostModel** -- risk-free rate hurdle
7. **TransactionCostEngine** -- combines all components

Each component is independently parameterized for easy sensitivity analysis.

---

## 10. Cost Sensitivity Analysis

### The Cost Frontier

For any strategy, there exists a "cost frontier" -- the relationship between assumed costs and realized performance. As costs increase, performance degrades. The key question: at what cost level does performance go to zero?

### Standard Sensitivity Sweep

Test strategy performance at these cost levels:

| Scenario | Round-Trip Cost | Components |
|----------|----------------|------------|
| Fantasy | 0 bps | No costs (NEVER use this) |
| Ultra-optimistic | 2 bps | VIP 9, maker only, no slippage |
| Optimistic | 5 bps | VIP 3, mostly maker, minimal slippage |
| Moderate | 10 bps | VIP 1, mixed, normal slippage |
| Conservative | 15 bps | VIP 0, taker, normal slippage + spread |
| Pessimistic | 20 bps | VIP 0, taker, high vol slippage |
| Stress test | 30 bps | Worst case, elevated vol, thin books |

### Interpretation Framework

```
If strategy is profitable at 20 bps: ROBUST strategy, likely works in live
If profitable at 10-15 bps but not 20: VIABLE but sensitive to execution quality
If profitable at 5-10 bps but not 15: FRAGILE, needs excellent execution, likely VIP tier
If only profitable below 5 bps: PROBABLY WON'T WORK live -- the alpha is a cost artifact
```

### The "Half-Alpha" Rule

A practical rule of thumb: if your strategy's gross alpha (before costs) is X bps per trade, costs should be no more than X/2 for it to be robust.

- Strategy with 20 bps gross alpha: viable up to ~10 bps costs
- Strategy with 8 bps gross alpha: viable up to ~4 bps costs (fragile)
- Strategy with 3 bps gross alpha: effectively zero alpha after any realistic cost

---

## 11. Break-Even Analysis

### Closed-Form Break-Even Formula

Given:
- `W` = win rate (fraction of winning trades)
- `avg_win` = average profit on winning trades (in bps, BEFORE costs)
- `avg_loss` = average loss on losing trades (in bps, BEFORE costs, as positive number)
- `C` = total cost per trade (bps, one-way -- half of round-trip)

**Expected PnL per trade (in bps):**
```
E[PnL] = W * (avg_win - C) - (1-W) * (avg_loss + C)
       = W * avg_win - (1-W) * avg_loss - C
```

Note: cost `C` is paid on EVERY trade (wins and losses), so it's subtracted once (not multiplied by W).

Wait -- costs apply to both entry AND exit, so for round-trip cost `C_rt`:
```
E[PnL] = W * avg_win - (1-W) * avg_loss - C_rt
```

**Break-even cost (round-trip):**
```
C_rt_breakeven = W * avg_win - (1-W) * avg_loss
```

**Example:**
- Win rate: 55%
- Average win: 30 bps
- Average loss: 20 bps

```
C_rt_breakeven = 0.55 * 30 - 0.45 * 20
               = 16.5 - 9.0
               = 7.5 bps round-trip
```

So this strategy breaks even at 7.5 bps round-trip cost. Anything above that, it loses money.

### Generalized Break-Even with Funding

For strategies that hold across funding periods:

```
C_rt_breakeven = W * avg_win - (1-W) * avg_loss - avg_funding_cost_per_trade
```

If funding averages +0.5 bps per trade (for longs):
```
C_rt_breakeven = 7.5 - 0.5 = 7.0 bps
```

### Break-Even Surfaces

For a strategy with known win rate, the break-even cost is a function of the win/loss ratio:

```
At W=0.50: C_breakeven = 0.50 * avg_win - 0.50 * avg_loss = 0.50 * (avg_win - avg_loss)
At W=0.55: C_breakeven = 0.55 * avg_win - 0.45 * avg_loss
At W=0.60: C_breakeven = 0.60 * avg_win - 0.40 * avg_loss
```

**Minimum win rate to break even at cost C:**
```
W_min = (avg_loss + C_rt) / (avg_win + avg_loss)
```

**Example:** At 10 bps cost, 30 bps avg win, 20 bps avg loss:
```
W_min = (20 + 10) / (30 + 20) = 30/50 = 60%
```

You need 60% win rate just to break even. At 55% win rate, you're losing money.

### Profit Factor Adjustment

```
Gross Profit Factor = (W * avg_win) / ((1-W) * avg_loss)
Net Profit Factor = (W * (avg_win - C_rt)) / ((1-W) * (avg_loss + C_rt))
```

A gross profit factor of 1.5 with 10 bps costs might become a net profit factor of 0.9.

---

## 12. Intraday Cost Patterns

### Time-of-Day Effects

Based on empirical research across multiple studies:

**Liquidity Cycle (UTC):**
```
Hour  | Depth Multiplier | Spread Multiplier | Slippage Multiplier
00:00 | 1.05             | 1.00              | 0.95
01:00 | 1.03             | 1.00              | 0.97
02:00 | 1.00             | 1.02              | 1.00
03:00 | 1.00             | 1.02              | 1.00
04:00 | 1.00             | 1.02              | 1.00
05:00 | 1.02             | 1.00              | 0.98
06:00 | 1.05             | 0.98              | 0.95
07:00 | 1.08             | 0.95              | 0.93
08:00 | 1.12             | 0.92              | 0.89    <- EU open
09:00 | 1.15             | 0.90              | 0.87
10:00 | 1.15             | 0.88              | 0.87
11:00 | 1.17             | 0.85              | 0.85    <- PEAK
12:00 | 1.15             | 0.87              | 0.87
13:00 | 1.12             | 0.88              | 0.89
14:00 | 1.10             | 0.90              | 0.91    <- US open
15:00 | 1.08             | 0.92              | 0.93
16:00 | 1.05             | 0.95              | 0.95
17:00 | 1.00             | 1.00              | 1.00
18:00 | 0.98             | 1.02              | 1.02
19:00 | 0.95             | 1.05              | 1.05
20:00 | 0.90             | 1.10              | 1.10
21:00 | 0.82             | 1.20              | 1.22    <- TROUGH
22:00 | 0.88             | 1.12              | 1.14
23:00 | 0.95             | 1.05              | 1.05
```

### Day-of-Week Effects

```
Monday:    +11.5% bid imbalance at 10:00 UTC (strongest bid bias)
Tuesday:   Normal
Wednesday: Slightly elevated spreads (FOMC often on Wed)
Thursday:  Normal
Friday:    Slightly reduced depth (TradFi close)
Saturday:  -11.9% ask imbalance at 01:00 UTC (weakest liquidity)
Sunday:    Reduced depth, wider spreads (~10-15% wider)
Weekend overall: ~15-20% less depth, ~10-20% wider spreads
```

### Around Macro Events

**CPI/FOMC/NFP releases:**
- 30 min before: Spreads widen 2-5x as market makers reduce exposure
- At release: Spreads can blow out 10-50x for seconds
- 5 min after: Rapid recovery to 2-3x normal
- 30 min after: Near-normal conditions

**Crypto-specific events (ETF approvals, exchange hacks, etc.):**
- No warning period
- Instant spread widening
- Recovery time: minutes to hours depending on severity

### Funding Rate Settlement Effects

Around 8-hour funding timestamps (00:00, 08:00, 16:00 UTC):
- 5 min before: Increased activity as positions adjust to avoid/capture funding
- At settlement: Brief volume spike
- 5 min after: Normal
- **Cost implication:** Slightly wider spreads and more slippage in the +/- 5 min window

### Modeling Intraday Costs in Backtests

For each simulated trade, adjust costs by:

```
effective_cost = base_cost * time_of_day_multiplier * day_of_week_multiplier * event_multiplier
```

Where event_multiplier requires a calendar of known macro events.

---

## Summary: Total Cost Estimation

For a single trade, the total cost is:

```
total_cost = fee_cost + spread_cost + slippage_cost + market_impact_cost + funding_cost

Where:
  fee_cost         = notional * fee_rate  (2-10 bps depending on tier/type)
  spread_cost      = notional * spread/2  (0.1-5 bps depending on conditions)
  slippage_cost    = f(order_size, depth, volatility)  (0-15 bps)
  market_impact    = Y * sigma * sqrt(Q/V) * notional  (0-5 bps for typical sizes)
  funding_cost     = position_value * funding_rate * num_settlements  (0-2 bps per 8h)
```

### Realistic Total Cost Estimates (Round-Trip)

| Strategy Type | Trade Size | Frequency | Estimated RT Cost |
|--------------|------------|-----------|-------------------|
| HFT/scalping | $1-10K | 100+/day | 3-5 bps (maker, minimal slippage) |
| Intraday momentum | $10-50K | 10-30/day | 8-15 bps |
| Swing (4-24h) | $50-200K | 2-5/day | 10-20 bps + funding |
| Position (days-weeks) | $100K-1M | 0.5-2/day | 12-25 bps + funding |
| Large institutional | $1M+ | 1-3/day | 15-40 bps + funding |

---

## Sources

- [Binance Futures Fee Structure](https://www.binance.com/en/support/faq/detail/360033544231)
- [Bybit Trading Fee Structure](https://www.bybit.com/en/help-center/article/Trading-Fee-Structure)
- [Binance Fees Breakdown 2026](https://www.bitdegree.org/crypto/tutorials/binance-fees)
- [Bybit Futures Fees](https://tradersunion.com/brokers/crypto/view/bybit/futures-fees/)
- [The Rhythm of Liquidity: Temporal Patterns in Market Depth (Amberdata)](https://blog.amberdata.io/the-rhythm-of-liquidity-temporal-patterns-in-market-depth)
- [High-frequency dynamics of Bitcoin futures (ScienceDirect)](https://www.sciencedirect.com/science/article/pii/S2214845025001188)
- [Fragmentation, Price Formation and Cross-Impact in Bitcoin Markets](https://www.tandfonline.com/doi/full/10.1080/1350486X.2022.2080083)
- [Almgren-Chriss Framework (Anboto Labs)](https://medium.com/@anboto_labs/deep-dive-into-is-the-almgren-chriss-framework-be45a1bde831)
- [Solving the Almgren-Chriss Model](https://dm13450.github.io/2024/06/06/Solving-the-Almgren-Chris-Model.html)
- [The Square-Root Law of Market Impact (Bouchaud)](https://bouchaud.substack.com/p/the-square-root-law-of-market-impact)
- [The two square root laws of market impact](https://arxiv.org/pdf/2311.18283)
- [Square-Root Law Baruch/Gatheral](https://mfe.baruch.cuny.edu/wp-content/uploads/2012/09/Chicago2016OptimalExecution.pdf)
- [BTC Funding Rate (CoinGlass)](https://www.coinglass.com/FundingRate/BTC)
- [BitMEX Q3 2025 Derivatives Report - Funding Rate Analysis](https://www.bitmex.com/blog/2025q3-derivatives-report)
- [Bitcoin Funding Rates (MacroMicro)](https://en.macromicro.me/charts/49213/bitcoin-perpetual-futures-funding-rate)
- [Fees, Rebates, and Maker/Taker Math (Axon Trade)](https://axon.trade/fees-rebates-and-maker-taker-math)
- [The Market Maker's Dilemma: Fill Probability vs Post-Fill Returns](https://arxiv.org/html/2502.18625v2)
- [Maker & Taker Fees: Market Structure Analysis (Deribit)](https://insights.deribit.com/market-research/maker-taker-fees-on-crypto-exchanges-a-market-structure-analysis/)
- [Realistic Backtesting Methodology (HyperQuant)](https://www.hyper-quant.tech/research/realistic-backtesting-methodology)
- [TCA in Crypto Trading (Anboto Labs)](https://medium.com/@anboto_labs/slippage-benchmarks-and-beyond-transaction-cost-analysis-tca-in-crypto-trading-2f0b0186980e)
- [Crypto Intraday Session Patterns](https://link.springer.com/article/10.1007/s11156-024-01304-1)
- [Crypto Staking Guide 2026](https://www.spotedcrypto.com/crypto-staking-guide-2026-real-yield/)
- [Crypto Lending Statistics 2026](https://coinlaw.io/crypto-lending-and-borrowing-statistics/)
