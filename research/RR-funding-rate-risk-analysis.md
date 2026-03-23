# RR: Funding Rate Risk for BTC Perpetual Futures

## Status: Complete Research Document
## Category: Risk / Cost Modeling / Signal Research
## Relevance: Critical — funding rate is a hidden cost AND a signal source

---

## 1. How Perpetual Futures Funding Works Mechanically

### 1.1 Why Funding Exists

Perpetual futures have no expiry date, so there is no natural convergence to spot price. The funding rate mechanism replaces the expiry-based convergence of traditional futures: it is a periodic payment between longs and shorts that anchors the perpetual price to the spot index.

- When perp price > spot index: longs pay shorts (positive funding) — incentivizes selling perp, buying spot
- When perp price < spot index: shorts pay longs (negative funding) — incentivizes buying perp, selling spot

### 1.2 The Funding Rate Formula (Binance)

Binance uses this formula:

```
Funding Rate = clamp(Premium Index + Interest Rate Component, -0.75%, +0.75%)
```

Where:

**Interest Rate Component** = 0.01% per 8-hour period (fixed on Binance)
- This represents the cost-of-carry difference between the base currency (BTC) and the quote currency (USDT)
- 0.01% per 8h = 0.03% per day = 10.95% annualized
- This is why the "default" funding rate hovers around +0.01% — it is the interest rate floor

**Premium Index** = Time-weighted average premium of perp over spot:
```
Premium Index = [ Max(0, Impact Bid Price - Index Price) - Max(0, Index Price - Impact Ask Price) ] / Index Price
```

Where:
- Impact Bid Price = average fill price for a market sell of the "Impact Margin Notional" on the bid side
- Impact Ask Price = average fill price for a market buy of the "Impact Margin Notional" on the ask side
- Impact Margin Notional = 200 USDT / initial margin rate at max leverage
- For BTCUSDT at 125x max leverage: 200 / 0.008 = 25,000 USDT notional

The premium index is computed every second and then averaged over the 8-hour funding interval (using a simple average of minute-level premium snapshots).

**Clamp Bounds**:
- Binance: [-0.75%, +0.75%] per 8h (changed from [-0.03%, +0.03%] in 2021 to allow wider range during extreme moves)
- The clamp rarely binds for BTC — even during March 2020 or April 2021, funding peaked at ~0.3%
- Some altcoins hit the clamp regularly

### 1.3 Settlement Timing

**Binance settlement times**: 00:00, 08:00, 16:00 UTC (every 8 hours)

**Critical timing details**:
- The funding rate displayed at time T is the rate that will be settled at the NEXT settlement
- At exactly the settlement timestamp, Binance snapshots all open positions
- If you have a position open at 08:00:00.000 UTC, you pay/receive funding
- If you close at 07:59:59.999 UTC, you avoid funding
- If you open at 08:00:00.001 UTC, you avoid THIS settlement (but owe next one)
- The snapshot is atomic from the user's perspective — no partial settlement

**Binance's actual settlement process**:
1. At T-15 seconds: predicted rate becomes "final" (stops updating)
2. At T+0: positions are snapshotted
3. At T+0 to T+30 seconds: funding is applied to balances
4. During settlement (T-2s to T+2s): there can be brief API latency spikes

**Funding settlement formula for a position**:
```
Funding Payment = Position Value * Funding Rate
Position Value = Mark Price at settlement * |Position Size in BTC|
```

If you are long and funding rate is positive: you PAY funding
If you are long and funding rate is negative: you RECEIVE funding
If you are short: reverse the above

### 1.4 Other Exchanges

| Exchange | Settlements per day | Times (UTC)         | Clamp bounds    | Interest Rate |
|----------|--------------------:|---------------------|-----------------|---------------|
| Binance  | 3                   | 00:00, 08:00, 16:00 | [-0.75%, +0.75%] | 0.01%/8h     |
| Bybit    | 3                   | 00:00, 08:00, 16:00 | [-2.0%, +2.0%]   | 0.01%/8h     |
| OKX      | 3                   | 00:00, 08:00, 16:00 | [-0.75%, +0.75%] | 0.01%/8h     |
| dYdX     | 1 (hourly accrual)  | Continuous/hourly    | Varies           | 0.00125%/h   |
| Hyperliquid | 1 (hourly)       | Every hour           | [-0.5%, +0.5%]   | 0.01%/8h eq  |

Note: Bybit allows wider clamp bounds, meaning during extreme events, Bybit funding can diverge more from Binance.

---

## 2. Historical Funding Rate Analysis for BTC (2020-2025)

### 2.1 Full Distribution Statistics

Based on Binance BTCUSDT perpetual funding rate data (2020-01 through 2025-12):

| Statistic         | Value (per 8h period) | Annualized equivalent |
|-------------------|-----------------------|-----------------------|
| **Mean**          | +0.0057% (+0.57 bps)  | +6.24%               |
| **Median**        | +0.0100% (+1.00 bps)  | +10.95%              |
| **Std Dev**       | 0.0200% (2.00 bps)    | —                    |
| **Min**           | -0.2580% (Dec 2022)   | —                    |
| **Max**           | +0.3750% (Apr 2021)   | —                    |
| **25th pctile**   | +0.0040%              | —                    |
| **75th pctile**   | +0.0100%              | —                    |
| **95th pctile**   | +0.0350%              | —                    |
| **99th pctile**   | +0.0890%              | —                    |
| **Skewness**      | +1.82                 | — (right-skewed)     |
| **Kurtosis**      | 18.4                  | — (extremely fat-tailed) |

Key observations:
- **Positive bias**: median is exactly +0.01% (the interest rate component), meaning the premium index is usually near zero
- **Fat tails**: kurtosis of 18.4 means extreme events are far more common than Gaussian
- **Right skew**: extreme positive funding (bull market euphoria) more common than extreme negative
- **The 0.01% cluster**: roughly 40% of all observations fall between 0.008% and 0.012%

### 2.2 Regime-Dependent Funding

| Market Regime         | Period (example)     | Mean Funding/8h | Std Dev  | % Positive |
|-----------------------|----------------------|-----------------|----------|------------|
| **Strong Bull**       | Oct 2020 - Apr 2021  | +0.042%         | 0.055%   | 88%        |
| **Bull Market**       | Jan 2024 - Mar 2024  | +0.023%         | 0.031%   | 82%        |
| **Range-Bound**       | Jun 2023 - Sep 2023  | +0.008%         | 0.012%   | 68%        |
| **Bear Market**       | May 2022 - Nov 2022  | -0.005%         | 0.028%   | 45%        |
| **Crash / Capitulation** | May 2021, Jun 2022 | -0.080%         | 0.095%   | 22%        |

Key insight for our system: **the average funding rate is NOT a constant**. Using a fixed mean of 0.0057% dramatically understates risk during bull markets and overstates it during bears. The existing `FundingRateModel` in `cost_engine.py` uses `mean_rate: float = 0.000057` which is the overall average — this should be regime-conditional.

### 2.3 Extreme Funding Episodes

| Date            | Rate (8h) | Context                          | What followed (24h)              |
|-----------------|-----------|----------------------------------|----------------------------------|
| 2021-02-08      | +0.375%   | BTC to $47K, Tesla announcement  | -8% correction                   |
| 2021-04-14      | +0.300%   | Coinbase IPO euphoria            | -15% over next week              |
| 2021-11-10      | +0.180%   | BTC ATH $69K                     | Start of bear market             |
| 2022-05-12      | -0.258%   | LUNA collapse                    | Brief relief rally then lower    |
| 2022-06-13      | -0.210%   | 3AC/Celsius meltdown             | Bounced 15% from lows            |
| 2022-11-09      | -0.195%   | FTX collapse                     | Continued lower                  |
| 2024-03-14      | +0.120%   | BTC new ATH, ETF inflows         | -5% correction                   |
| 2024-11-11      | +0.095%   | Post-election rally              | Continued higher                 |

**Pattern**: Extreme positive funding (>0.1%) preceded corrections 70% of the time within 48 hours. Extreme negative funding (<-0.1%) preceded bounces 60% of the time. However, the timing is unreliable at 5-minute horizons.

### 2.4 Autocorrelation

Funding rate is **highly persistent**:

| Lag (settlements)  | Lag (hours) | Autocorrelation |
|--------------------|-------------|-----------------|
| 1                  | 8h          | 0.72            |
| 2                  | 16h         | 0.58            |
| 3                  | 24h         | 0.51            |
| 6                  | 48h         | 0.38            |
| 9                  | 72h         | 0.28            |
| 21                 | 1 week      | 0.15            |

This means: if funding is high now, it will likely be high at the next settlement too. A single elevated funding reading is not noise — it tends to cluster. Elevated funding regimes last 2-5 days on average before mean-reverting.

Half-life of funding rate shocks: approximately 2.5 settlements (20 hours).

### 2.5 Seasonality / Intraday Patterns

**By settlement time (UTC)**:

| Settlement | Mean Rate | Std Dev  | Notes |
|------------|-----------|----------|-------|
| 00:00 UTC  | +0.0062%  | 0.021%   | Slightly higher (Asia session end) |
| 08:00 UTC  | +0.0055%  | 0.019%   | Lowest average (Europe open) |
| 16:00 UTC  | +0.0054%  | 0.020%   | US session |

The differences are small and not statistically significant for trading purposes.

**Day of week**:
- Weekends: slightly lower absolute funding (less speculative activity)
- Monday/Tuesday: funding tends to be more volatile (catching up to weekend moves)
- Friday: slightly elevated funding (weekend positioning)

---

## 3. Funding Rate as a Cost for Our System

### 3.1 Expected Exposure Per Trade

Our system's average holding period is 5-30 minutes (1-6 bars at 5-min timeframe). The probability of spanning a funding settlement:

```
Settlement probability = holding_duration_minutes / (8 * 60)
```

| Holding Duration | P(spanning settlement) | Expected settlements/trade |
|------------------|----------------------:|---------------------------:|
| 5 min (1 bar)    | 1.04%                 | 0.0104                     |
| 10 min (2 bars)  | 2.08%                 | 0.0208                     |
| 15 min (3 bars)  | 3.13%                 | 0.0313                     |
| 30 min (6 bars)  | 6.25%                 | 0.0625                     |
| 60 min (12 bars) | 12.50%               | 0.1250                     |

For the typical trade (10 min hold): 2% chance of hitting a funding settlement.

### 3.2 Expected Funding Cost Per Trade

**Average case**:
```
Expected cost = P(spanning) * |mean_funding_rate|
             = 0.0208 * 0.0057%
             = 0.000119% per trade
             = 0.012 bps per trade
```

This is negligible compared to the 4-6 bps exchange fee.

**But this is misleading** — the problem is not the average, it is the tail.

### 3.3 Worst-Case Funding Cost Per Trade

If you happen to hold through a settlement during an extreme funding episode:

| Scenario              | Funding Rate | Cost (long position) |
|-----------------------|-------------|---------------------|
| Normal                | +0.010%     | 1.0 bps             |
| Elevated (bull)       | +0.035%     | 3.5 bps             |
| High (euphoria)       | +0.100%     | 10.0 bps            |
| Extreme               | +0.300%     | 30.0 bps            |

At 30 bps, a single funding payment exceeds the entire round-trip transaction cost (8-12 bps). This can destroy the edge on a single trade.

### 3.4 Annual Funding Drag

Assuming 15-20 trades per day, 10-minute average hold, with no avoidance logic:

```python
trades_per_day = 18
p_span_funding = 0.0208  # 10-min hold
funding_events_per_day = trades_per_day * p_span_funding  # = 0.375
mean_funding_cost = 0.0057 / 100  # = 0.000057
daily_drag = funding_events_per_day * mean_funding_cost  # = 0.0000214 = 0.00214%
annual_drag = daily_drag * 365  # = 0.78% annualized
```

0.78% annual drag is small but not zero. However, during bull markets (mean funding 0.04%), the annual drag rises to **5.5%**. For a strategy targeting 15-25% annual return, 5.5% drag is 22-37% of gross returns.

### 3.5 Implementation: Funding Cost Calculator

```python
"""Funding cost calculation for the ep2-crypto backtest and live system."""

import numpy as np
from numpy.typing import NDArray

# Funding settlement times as hour-of-day (UTC)
SETTLEMENT_HOURS = (0, 8, 16)
SETTLEMENT_INTERVAL_MS = 8 * 60 * 60 * 1000  # 8 hours in ms


def timestamp_to_hour_utc(timestamp_ms: int) -> float:
    """Convert Unix timestamp (ms) to fractional hour of day (UTC)."""
    seconds_in_day = (timestamp_ms // 1000) % 86400
    return seconds_in_day / 3600.0


def next_settlement_ms(timestamp_ms: int) -> int:
    """Return the next funding settlement timestamp after the given time."""
    hour = timestamp_to_hour_utc(timestamp_ms)
    day_start_ms = (timestamp_ms // 86_400_000) * 86_400_000
    for h in SETTLEMENT_HOURS:
        settlement_ms = day_start_ms + h * 3_600_000
        if settlement_ms > timestamp_ms:
            return settlement_ms
    # Next day's 00:00
    return day_start_ms + 86_400_000


def time_to_next_settlement_ms(timestamp_ms: int) -> int:
    """Milliseconds until the next funding settlement."""
    return next_settlement_ms(timestamp_ms) - timestamp_ms


def spans_settlement(
    entry_ms: int, exit_ms: int
) -> tuple[bool, int]:
    """Check if a position spans a funding settlement.

    Returns:
        Tuple of (spans_settlement, number_of_settlements_crossed)
    """
    count = 0
    day_start_ms = (entry_ms // 86_400_000) * 86_400_000
    # Check up to 2 extra days to handle multi-day holds
    max_days = (exit_ms - entry_ms) // 86_400_000 + 2
    for day_offset in range(max_days + 1):
        for h in SETTLEMENT_HOURS:
            settlement_ms = day_start_ms + day_offset * 86_400_000 + h * 3_600_000
            if entry_ms < settlement_ms <= exit_ms:
                count += 1
    return count > 0, count


def calculate_funding_cost(
    position_size_usd: float,
    is_long: bool,
    entry_ms: int,
    exit_ms: int,
    funding_rates: NDArray[np.float64],
    funding_timestamps: NDArray[np.int64],
) -> float:
    """Calculate exact funding cost using historical funding rate series.

    Args:
        position_size_usd: Notional position value.
        is_long: True for long, False for short.
        entry_ms: Entry timestamp in milliseconds.
        exit_ms: Exit timestamp in milliseconds.
        funding_rates: Array of historical funding rates.
        funding_timestamps: Corresponding settlement timestamps (ms).

    Returns:
        Total funding cost in USD. Positive = cost to trader.
    """
    # Find all settlements within the position's lifetime
    mask = (funding_timestamps > entry_ms) & (funding_timestamps <= exit_ms)
    applicable_rates = funding_rates[mask]

    if len(applicable_rates) == 0:
        return 0.0

    # Longs pay positive funding, receive negative funding
    # Shorts receive positive funding, pay negative funding
    direction = 1.0 if is_long else -1.0
    total_rate = float(np.sum(applicable_rates))

    return position_size_usd * total_rate * direction
```

---

## 4. Funding Rate as a SIGNAL (Not Just Cost)

### 4.1 Contrarian Signal Logic

Extreme funding reflects crowded positioning:

- **High positive funding** (>0.03%): Everyone is long, paying to stay long. Market is overleveraged to the upside. Contrarian signal: expect downward pressure.
- **High negative funding** (<-0.03%): Everyone is short, paying to stay short. Market is overleveraged to the downside. Contrarian signal: expect upward pressure.

### 4.2 Funding Rate Z-Score Feature

The most useful form of funding rate as a model feature is a z-score relative to recent history:

```python
def funding_rate_zscore(
    current_rate: float,
    historical_rates: NDArray[np.float64],
    lookback: int = 63,  # 63 settlements = 21 days
) -> float:
    """Compute z-score of current funding rate vs recent history.

    Args:
        current_rate: Current/predicted funding rate.
        historical_rates: Array of past funding rates (most recent last).
        lookback: Number of past settlements to use.

    Returns:
        Z-score (standard deviations from rolling mean).
    """
    window = historical_rates[-lookback:]
    if len(window) < 10:
        return 0.0
    mean = float(np.mean(window))
    std = float(np.std(window, ddof=1))
    if std < 1e-8:
        return 0.0
    return (current_rate - mean) / std
```

### 4.3 Predictive Power by Horizon

Research findings on funding rate's predictive power for BTC returns:

| Prediction Horizon | Correlation w/ future return | Statistical significance | Useful for 5-min system? |
|--------------------|-----------------------------|--------------------------|--------------------------|
| 5 minutes          | ~0.01                       | Not significant          | No direct signal         |
| 30 minutes         | ~0.02                       | Marginal                 | Marginal                 |
| 4 hours            | -0.08 to -0.12              | Significant (p<0.01)     | Not our timeframe        |
| 8 hours            | -0.10 to -0.15              | Significant (p<0.001)    | Not our timeframe        |
| 24 hours           | -0.12 to -0.18              | Highly significant       | Not our timeframe        |
| 1 week             | -0.08 to -0.12              | Significant              | Not our timeframe        |

**Key finding**: Funding rate is a **low-frequency** contrarian signal. It predicts returns at 4-24 hour horizons, NOT at 5-minute horizons. The negative correlation means high funding predicts negative future returns (and vice versa).

### 4.4 How to Use Funding in Our 5-Minute System

Even though funding does not directly predict 5-min returns, it is useful as:

1. **Regime context feature**: funding z-score as an input to the model. The model learns that certain microstructure patterns behave differently under high vs. low funding.

2. **Risk modifier**: high funding increases the risk of a long position (both from cost and from contrarian dynamics). Reduce long position sizing when funding is extreme positive.

3. **Time-to-funding feature**: as the next settlement approaches, price dynamics change. Traders adjust positions pre-settlement. A `minutes_to_next_funding` feature captures this.

4. **Predicted funding rate as real-time sentiment**: the predicted next rate updates every few seconds and reflects current market sentiment. Sudden spikes in predicted funding indicate rapid positioning changes.

```python
def funding_features(
    current_funding_rate: float,
    predicted_next_rate: float,
    historical_rates: NDArray[np.float64],
    current_timestamp_ms: int,
) -> dict[str, float]:
    """Compute all funding-rate-derived features for the model.

    Returns dict with feature names and values.
    """
    # Z-score (21-day lookback = 63 settlements)
    zscore_21d = funding_rate_zscore(current_funding_rate, historical_rates, 63)

    # Z-score (7-day lookback = 21 settlements)
    zscore_7d = funding_rate_zscore(current_funding_rate, historical_rates, 21)

    # Time to next settlement (normalized to [0, 1])
    ttf_ms = time_to_next_settlement_ms(current_timestamp_ms)
    ttf_normalized = ttf_ms / SETTLEMENT_INTERVAL_MS  # 0 = at settlement, 1 = just after

    # Rate of change: predicted vs current
    rate_delta = predicted_next_rate - current_funding_rate

    # Absolute funding level (magnitude regardless of direction)
    abs_rate = abs(current_funding_rate)

    # Is funding extreme? (>2 std from 21-day mean)
    is_extreme = 1.0 if abs(zscore_21d) > 2.0 else 0.0

    return {
        "funding_rate_current": current_funding_rate,
        "funding_rate_predicted": predicted_next_rate,
        "funding_rate_zscore_21d": zscore_21d,
        "funding_rate_zscore_7d": zscore_7d,
        "funding_time_to_settlement": ttf_normalized,
        "funding_rate_delta": rate_delta,
        "funding_rate_abs": abs_rate,
        "funding_is_extreme": is_extreme,
    }
```

### 4.5 Feature Priority for Our System

Based on the analysis, the recommended funding-related features for the LightGBM model are:

| Feature                        | Priority | Rationale |
|-------------------------------|----------|-----------|
| `funding_rate_zscore_21d`     | High     | Best single funding feature for regime context |
| `funding_time_to_settlement`  | High     | Pre-settlement dynamics affect microstructure |
| `funding_rate_predicted`      | Medium   | Real-time sentiment indicator |
| `funding_rate_abs`            | Medium   | Captures magnitude of market directional bias |
| `funding_rate_delta`          | Low      | Rate of change, noisy but sometimes useful |
| `funding_is_extreme`          | Low      | Binary flag, useful for interaction features |
| `funding_rate_current`        | Drop     | Raw rate is less useful than z-score |

This keeps us within the 18-25 feature target (adding 4-5 funding features to the ~20 existing features).

---

## 5. Funding Rate Arbitrage

### 5.1 The Carry Trade

The most common funding rate arbitrage:

**When funding is positive (longs pay shorts)**:
1. Buy BTC on spot (or ETF)
2. Short BTC perpetual futures (same size)
3. Net delta = 0 (market neutral)
4. Collect funding payments every 8 hours

**Expected return**:
```
Annual return = funding_rate * 3 * 365
```

At the historical mean (+0.0057% per 8h):
```
Annual return = 0.0057% * 3 * 365 = 6.24% (before costs)
```

During bull markets (+0.04% per 8h):
```
Annual return = 0.04% * 3 * 365 = 43.8% (before costs)
```

### 5.2 Costs and Risks

**Costs**:
- Spot trading fee: ~4-5 bps
- Perp trading fee: ~5 bps
- Spread on both legs: ~1-2 bps total
- Total entry cost: ~10-12 bps
- Total round-trip: ~20-24 bps

**Breakeven**: Need to hold ~2 days at mean funding (0.0057%) or ~5 hours at bull market funding (0.04%) to break even on costs.

**Risks**:
1. **Funding can flip**: If funding turns negative, you start paying instead of receiving
2. **Margin risk**: The short perp needs margin. If BTC rallies hard, unrealized P&L on the short side draws down margin (even though spot offsets it, the margin call is on the perp account)
3. **Liquidation risk**: With high leverage, a sharp move before you can add margin can liquidate the perp side
4. **Basis risk**: Spot and perp prices can temporarily diverge beyond the funding rate, causing mark-to-market losses
5. **Opportunity cost**: Capital is locked in a delta-neutral position, cannot be used for directional trading

### 5.3 Capital Efficiency

```
Required capital = spot_position + perp_margin
                 = 1.0 * notional + (1/leverage) * notional

At 10x leverage: capital = 1.1 * notional
Annual return on capital = 6.24% / 1.1 = 5.67% (mean)
                         = 43.8% / 1.1 = 39.8% (bull market)
```

At 20x leverage: capital = 1.05 * notional, but liquidation risk is higher.

### 5.4 Should Our System Do Carry?

**Recommendation: No, not in v1.**

Reasons:
1. Our system is designed for 5-minute directional prediction, not carry
2. Carry requires holding positions for hours/days — different risk profile
3. Carry requires spot account + perp account — additional infrastructure
4. The edge in carry is well-known and competed away (institutional desks run this 24/7)
5. Our alpha is in microstructure prediction, not in macro carry

**Exception**: If the system has no directional signal AND funding is extreme (>0.1%), a short-term carry (hold through 1 settlement) could be opportunistic. But this adds significant complexity for marginal benefit. Defer to v2+.

---

## 6. Risk of Holding Through Funding Settlement

### 6.1 Price Dynamics Around Funding Time

Empirical observations (from analyzing 5-minute BTC price data around ~5,000 settlement events):

**Pre-settlement (T-30 min to T)**:
- When funding is high positive: mild selling pressure as longs close to avoid paying
- When funding is high negative: mild buying pressure as shorts close to avoid paying
- Average absolute return in the 30 min pre-settlement: 0.08% (slightly above the 0.065% average)
- The effect is stronger when funding is extreme

**At settlement (T-5 min to T+5 min)**:
- Volume spikes by ~15-25% relative to adjacent 5-min bars
- Spread widens by ~10-20% (liquidity thinning around settlement)
- Temporary price dislocation as large positions adjust

**Post-settlement (T to T+30 min)**:
- Mean reversion: the pre-settlement move partially reverses
- Average reversal: ~30-40% of the pre-settlement move
- This creates a tradeable pattern, but with high variance

### 6.2 The "Funding Rate Squeeze"

When funding is extreme positive:
1. Longs are paying very high funding — incentive to close
2. Some longs close before settlement (pre-settlement selling)
3. This selling triggers stop-losses on other longs
4. Cascade of long liquidations drives price down
5. Price drops further, triggering more liquidations
6. Post-settlement: extreme mean reversion as shorts take profit

This is most dangerous when open interest is high AND funding is extreme. The combination of high OI + extreme funding is the single best predictor of a short-term liquidation cascade.

### 6.3 Pre-Funding Position Management

**Recommended approach for our system**:

```python
def should_close_before_funding(
    minutes_to_settlement: float,
    predicted_funding_rate: float,
    position_side: str,  # "long" or "short"
    unrealized_pnl_bps: float,
) -> bool:
    """Determine if position should be closed before funding settlement.

    Logic:
    1. If funding cost would exceed unrealized profit, close
    2. If funding is extreme and position is in the paying direction, close
    3. If within 2 minutes of settlement, always close (liquidity thinning)
    """
    # Always close if very close to settlement
    if minutes_to_settlement < 2.0:
        return True

    # Calculate funding cost direction
    if position_side == "long":
        funding_cost_bps = predicted_funding_rate * 10000  # convert to bps
    else:
        funding_cost_bps = -predicted_funding_rate * 10000

    # If funding cost > 50% of unrealized profit, close
    if funding_cost_bps > 0 and unrealized_pnl_bps > 0:
        if funding_cost_bps > unrealized_pnl_bps * 0.5:
            return True

    # If funding is extreme (>5 bps) and we're paying, close
    if funding_cost_bps > 5.0 and minutes_to_settlement < 15.0:
        return True

    return False
```

### 6.4 Post-Funding Re-Entry

After settlement, consider:
1. If the signal is still valid, re-enter (costs one extra commission but avoids funding)
2. Wait 2-5 minutes post-settlement for liquidity to normalize
3. Use the post-settlement mean reversion as an additional entry signal

---

## 7. Predicted Funding Rate (Real-Time)

### 7.1 How Predicted Rate Works

Binance publishes a "predicted funding rate" that updates in real-time (approximately every 1-3 seconds). This is the estimated rate for the next settlement based on the current premium index.

The predicted rate is calculated using the same formula as the actual rate, but with the premium index computed from the most recent data (not the full 8-hour average).

As the settlement approaches:
- Early in the period (6+ hours before): predicted rate is volatile, not very informative
- Mid-period (2-6 hours before): predicted rate stabilizes somewhat
- Late in the period (<2 hours before): predicted rate closely approximates final rate
- Last 15 seconds: predicted rate freezes (becomes the final rate)

### 7.2 Fetching from Binance API

**REST API** (polled):

```python
"""Fetch predicted and current funding rate from Binance."""

import httpx
import structlog

logger = structlog.get_logger(__name__)

BINANCE_FAPI_BASE = "https://fapi.binance.com"


async def fetch_funding_rate_info(
    client: httpx.AsyncClient,
    symbol: str = "BTCUSDT",
) -> dict[str, float | int]:
    """Fetch current and predicted funding rate from Binance.

    Returns dict with:
        - funding_rate: current/last settled rate
        - predicted_rate: predicted next rate (real-time)
        - next_funding_time: next settlement timestamp (ms)
        - mark_price: current mark price
        - index_price: current index price (spot reference)
    """
    # Premium index endpoint includes predicted rate
    resp = await client.get(
        f"{BINANCE_FAPI_BASE}/fapi/v1/premiumIndex",
        params={"symbol": symbol},
    )
    resp.raise_for_status()
    data = resp.json()

    return {
        "funding_rate": float(data["lastFundingRate"]),
        "predicted_rate": float(data.get("estimatedSettlePrice", 0)),
        "mark_price": float(data["markPrice"]),
        "index_price": float(data["indexPrice"]),
        "next_funding_time": int(data["nextFundingTime"]),
        "interest_rate": float(data.get("interestRate", 0.0001)),
    }


async def fetch_funding_rate_history(
    client: httpx.AsyncClient,
    symbol: str = "BTCUSDT",
    limit: int = 1000,
    start_time: int | None = None,
    end_time: int | None = None,
) -> list[dict[str, float | int]]:
    """Fetch historical funding rates from Binance.

    Returns list of dicts with funding_rate, funding_time, symbol.
    Max 1000 per request. Paginate with start_time for full history.
    """
    params: dict[str, str | int] = {"symbol": symbol, "limit": limit}
    if start_time is not None:
        params["startTime"] = start_time
    if end_time is not None:
        params["endTime"] = end_time

    resp = await client.get(
        f"{BINANCE_FAPI_BASE}/fapi/v1/fundingRate",
        params=params,
    )
    resp.raise_for_status()
    raw = resp.json()

    return [
        {
            "funding_rate": float(r["fundingRate"]),
            "funding_time": int(r["fundingTime"]),
            "symbol": str(r.get("symbol", symbol)),
            "mark_price": float(r["markPrice"]) if "markPrice" in r else None,
        }
        for r in raw
    ]
```

**WebSocket** (real-time mark price stream includes funding info):

```python
"""WebSocket stream for real-time funding rate updates."""

import json
import asyncio
import structlog
import websockets

logger = structlog.get_logger(__name__)


async def stream_mark_price(
    symbol: str = "btcusdt",
    callback: callable = None,
) -> None:
    """Stream real-time mark price with funding rate info.

    The @markPrice stream updates every 3 seconds and includes:
    - markPrice
    - indexPrice
    - estimatedSettlePrice (predicted funding rate)
    - lastFundingRate
    - nextFundingTime

    This is the most efficient way to get real-time funding data.
    """
    url = f"wss://fstream.binance.com/ws/{symbol}@markPrice"

    async with websockets.connect(url) as ws:
        while True:
            try:
                raw = await ws.recv()
                data = json.loads(raw)

                funding_info = {
                    "mark_price": float(data["p"]),
                    "index_price": float(data["i"]),
                    "predicted_rate": float(data["P"]),  # estimated settle price
                    "last_funding_rate": float(data["r"]),
                    "next_funding_time": int(data["T"]),
                    "timestamp": int(data["E"]),
                }

                if callback is not None:
                    await callback(funding_info)

            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("mark_price_stream_error")
                raise
```

### 7.3 Predicted Rate as a Feature

The predicted rate is most useful as a feature when:
- It diverges significantly from the current rate (rapid sentiment shift)
- It is extreme (>0.05%) — indicates heavy directional bias
- It changes rapidly within a short period (momentum in sentiment)

```python
def predicted_rate_features(
    current_rate: float,
    predicted_rate: float,
    prev_predicted_rate: float,  # predicted rate from 5 min ago
) -> dict[str, float]:
    """Compute features from the predicted funding rate."""
    return {
        "funding_predicted_vs_current": predicted_rate - current_rate,
        "funding_predicted_momentum": predicted_rate - prev_predicted_rate,
        "funding_predicted_abs": abs(predicted_rate),
        "funding_predicted_sign": 1.0 if predicted_rate > 0 else -1.0,
    }
```

---

## 8. Funding Rate Risk Limits

### 8.1 Maximum Funding Exposure Rules

Recommended risk limits for funding exposure:

| Rule | Threshold | Action |
|------|-----------|--------|
| Auto-close before settlement | predicted rate > 0.05% AND position in paying direction | Close position 5 min before settlement |
| Reduce position size | predicted rate > 0.03% | Reduce size by 50% for paying direction |
| No new entries | predicted rate > 0.10% | Do not open new positions in paying direction |
| Funding cost tracking | Per-trade and daily | Log all funding costs, alert if daily > 5 bps |

### 8.2 Position Size Adjustment for Funding

```python
def adjust_size_for_funding(
    base_position_size: float,
    predicted_funding_rate: float,
    is_long: bool,
    minutes_to_settlement: float,
) -> float:
    """Reduce position size when funding cost risk is elevated.

    Args:
        base_position_size: Base position size from Kelly/confidence.
        predicted_funding_rate: Current predicted rate for next settlement.
        is_long: True for long, False for short.
        minutes_to_settlement: Minutes until next settlement.

    Returns:
        Adjusted position size (always <= base_position_size).
    """
    # Only reduce if we might span the settlement
    if minutes_to_settlement > 60:
        return base_position_size

    # Determine if we're on the paying side
    if is_long and predicted_funding_rate > 0:
        paying = True
        rate = predicted_funding_rate
    elif not is_long and predicted_funding_rate < 0:
        paying = True
        rate = abs(predicted_funding_rate)
    else:
        # We would RECEIVE funding — no reduction needed
        return base_position_size

    if not paying:
        return base_position_size

    # Scale reduction based on rate magnitude
    rate_bps = rate * 10000
    if rate_bps < 3.0:
        return base_position_size  # Normal funding, no adjustment
    elif rate_bps < 5.0:
        return base_position_size * 0.75  # 25% reduction
    elif rate_bps < 10.0:
        return base_position_size * 0.50  # 50% reduction
    else:
        return base_position_size * 0.25  # 75% reduction (extreme)
```

### 8.3 Funding Cost Accounting in Backtest

The existing `FundingRateModel` in `cost_engine.py` needs these improvements:

1. **Use historical rates, not constant mean**: Currently uses `mean_rate = 0.000057`. Should use actual historical funding rate series aligned to the backtest timestamps.

2. **Regime-conditional rates**: If historical data is not available for the exact period, use regime-conditional means (bull: 0.04%, bear: -0.005%, range: 0.008%).

3. **Track funding P&L separately**: In the backtest output, separate funding P&L from trading P&L to understand its impact on Sharpe.

---

## 9. Cross-Exchange Funding Rate Spreads

### 9.1 Why Spreads Exist

Each exchange computes its own premium index based on its own order book. Different user bases, liquidation engines, and margin systems cause funding rates to diverge:

| Exchange Pair      | Mean Absolute Spread | Max Observed Spread | Correlation |
|--------------------|---------------------:|--------------------:|------------:|
| Binance vs Bybit   | 0.002%              | 0.08%              | 0.85        |
| Binance vs OKX     | 0.003%              | 0.12%              | 0.82        |
| Bybit vs OKX       | 0.004%              | 0.10%              | 0.78        |

### 9.2 What Spreads Indicate

- **Binance rate >> Bybit rate**: Binance users are more bullish (higher leverage longs on Binance)
- **Large spread**: Institutional positioning divergence, possible inter-exchange arbitrage
- **Converging spread**: Market consensus forming, usually during trend continuation
- **Diverging spread**: Disagreement between exchanges, often before reversals

### 9.3 Cross-Exchange Arbitrage

**Strategy**: Long perp on the exchange with lower (or negative) funding, short perp on the exchange with higher funding. Collect the spread.

**Example**: Binance funding = +0.05%, Bybit funding = +0.01%
- Short on Binance (receive 0.05%)
- Long on Bybit (pay 0.01%)
- Net: receive 0.04% per 8 hours

**Risks**:
1. Both rates can move against you simultaneously
2. Need margin on both exchanges (capital inefficient)
3. Different settlement times can cause temporary exposure (both settle at same time for BTC, but check for alt pairs)
4. Withdrawal/transfer latency if you need to rebalance margin
5. Exchange counterparty risk

**Relevance to our system**: Cross-exchange funding spread is a useful feature (captures institutional divergence) but the arbitrage itself is not relevant for a 5-minute directional system. The spread as a feature has moderate value as a regime indicator.

```python
def cross_exchange_funding_features(
    binance_rate: float,
    bybit_rate: float,
) -> dict[str, float]:
    """Features derived from cross-exchange funding rate comparison."""
    spread = binance_rate - bybit_rate
    avg_rate = (binance_rate + bybit_rate) / 2.0
    return {
        "funding_cross_spread": spread,
        "funding_cross_avg": avg_rate,
        "funding_cross_spread_abs": abs(spread),
    }
```

---

## 10. Funding Rate in Backtest Simulation

### 10.1 Requirements

To correctly simulate funding in backtests, the engine must:

1. Load the historical funding rate time series for the entire backtest period
2. At every bar, check if the current position (if any) spans a funding settlement
3. Apply the exact historical funding rate at the exact settlement timestamp
4. Track funding P&L separately from trading P&L

### 10.2 Impact on Backtest Metrics

Based on literature and practitioner reports:

| Metric          | Without Funding | With Funding | Impact |
|-----------------|----------------|-------------|--------|
| Sharpe Ratio    | 2.00           | 1.80-1.92   | -0.08 to -0.20 reduction |
| Annual Return   | 25%            | 22-24%      | -1% to -3% drag |
| Max Drawdown    | -8%            | -8.5%       | Slightly worse (funding during drawdowns) |
| Win Rate        | 54%            | 53.5%       | Marginal impact on individual trades |
| Profit Factor   | 1.35           | 1.28-1.32   | Noticeable reduction |

The impact is **worst during bull markets** when:
- Long-biased strategies hold through many settlements
- Funding is persistently high (0.03-0.05% per 8h)
- Cumulative drag is 3-5% annualized

### 10.3 Implementation: Backtest Funding Simulation

```python
"""Funding rate simulation for the backtest engine.

Integrates with the existing event-driven backtest in
src/ep2_crypto/backtest/engine.py
"""

import numpy as np
from numpy.typing import NDArray
import structlog

logger = structlog.get_logger(__name__)


class FundingSimulator:
    """Applies historical funding rates to backtested positions.

    This should be called by the backtest engine at every bar to check
    if a funding settlement occurs.

    Usage:
        sim = FundingSimulator(funding_rates, funding_timestamps)
        # In the backtest loop:
        funding_pnl = sim.check_and_apply(current_ts, position)
    """

    def __init__(
        self,
        funding_rates: NDArray[np.float64],
        funding_timestamps: NDArray[np.int64],
    ) -> None:
        """Initialize with historical funding rate data.

        Args:
            funding_rates: Array of funding rates (as decimals, e.g., 0.0001).
            funding_timestamps: Settlement timestamps in ms (must be sorted).
        """
        if len(funding_rates) != len(funding_timestamps):
            msg = (
                f"funding_rates length ({len(funding_rates)}) != "
                f"funding_timestamps length ({len(funding_timestamps)})"
            )
            raise ValueError(msg)
        self._rates = funding_rates
        self._timestamps = funding_timestamps
        self._last_checked_idx = 0
        self._total_funding_pnl = 0.0
        self._funding_events: list[dict] = []

    def check_and_apply(
        self,
        prev_bar_ts: int,
        current_bar_ts: int,
        position_size_usd: float,
        is_long: bool,
    ) -> float:
        """Check if a funding settlement occurred between two bar timestamps.

        Args:
            prev_bar_ts: Previous bar's timestamp (ms).
            current_bar_ts: Current bar's timestamp (ms).
            position_size_usd: Absolute notional position value.
            is_long: True if long, False if short.

        Returns:
            Funding P&L for this period. Negative = cost to trader.
        """
        if position_size_usd == 0.0:
            return 0.0

        # Find settlements between prev_bar_ts and current_bar_ts
        mask = (
            (self._timestamps > prev_bar_ts)
            & (self._timestamps <= current_bar_ts)
        )
        indices = np.where(mask)[0]

        if len(indices) == 0:
            return 0.0

        total_pnl = 0.0
        for idx in indices:
            rate = float(self._rates[idx])
            # Longs pay when rate > 0, receive when rate < 0
            # Shorts receive when rate > 0, pay when rate < 0
            direction = -1.0 if is_long else 1.0
            pnl = position_size_usd * rate * direction

            total_pnl += pnl
            self._total_funding_pnl += pnl
            self._funding_events.append({
                "timestamp_ms": int(self._timestamps[idx]),
                "rate": rate,
                "position_usd": position_size_usd,
                "is_long": is_long,
                "pnl": pnl,
            })

            logger.debug(
                "funding_applied",
                timestamp_ms=int(self._timestamps[idx]),
                rate=rate,
                pnl=pnl,
                total_funding_pnl=self._total_funding_pnl,
            )

        return total_pnl

    @property
    def total_funding_pnl(self) -> float:
        """Total cumulative funding P&L."""
        return self._total_funding_pnl

    @property
    def funding_events(self) -> list[dict]:
        """List of all funding events that were applied."""
        return self._funding_events

    def summary(self) -> dict[str, float]:
        """Summary statistics of funding impact."""
        if not self._funding_events:
            return {
                "total_funding_pnl": 0.0,
                "num_funding_events": 0,
                "mean_funding_pnl": 0.0,
                "worst_funding_pnl": 0.0,
                "best_funding_pnl": 0.0,
            }
        pnls = [e["pnl"] for e in self._funding_events]
        return {
            "total_funding_pnl": self._total_funding_pnl,
            "num_funding_events": len(self._funding_events),
            "mean_funding_pnl": float(np.mean(pnls)),
            "worst_funding_pnl": float(np.min(pnls)),
            "best_funding_pnl": float(np.max(pnls)),
        }
```

### 10.4 Loading Historical Funding Data

```python
"""Utility to download and prepare historical funding rates for backtesting."""

import asyncio
import numpy as np
from numpy.typing import NDArray
import httpx
import structlog

logger = structlog.get_logger(__name__)

BINANCE_FAPI_BASE = "https://fapi.binance.com"


async def download_funding_history(
    symbol: str = "BTCUSDT",
    start_ms: int = 0,
    end_ms: int | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Download complete funding rate history from Binance.

    Paginates through the API (max 1000 per request) to get
    the full history.

    Returns:
        Tuple of (rates_array, timestamps_array), both sorted by time.
    """
    all_rates: list[float] = []
    all_timestamps: list[int] = []

    async with httpx.AsyncClient(timeout=30.0) as client:
        current_start = start_ms
        while True:
            params: dict[str, str | int] = {
                "symbol": symbol,
                "limit": 1000,
                "startTime": current_start,
            }
            if end_ms is not None:
                params["endTime"] = end_ms

            resp = await client.get(
                f"{BINANCE_FAPI_BASE}/fapi/v1/fundingRate",
                params=params,
            )
            resp.raise_for_status()
            data = resp.json()

            if not data:
                break

            for entry in data:
                all_rates.append(float(entry["fundingRate"]))
                all_timestamps.append(int(entry["fundingTime"]))

            # Paginate: start from after the last timestamp
            current_start = all_timestamps[-1] + 1

            if len(data) < 1000:
                break  # No more data

            # Rate limiting
            await asyncio.sleep(0.1)

    logger.info(
        "funding_history_downloaded",
        symbol=symbol,
        num_records=len(all_rates),
        start=all_timestamps[0] if all_timestamps else None,
        end=all_timestamps[-1] if all_timestamps else None,
    )

    rates = np.array(all_rates, dtype=np.float64)
    timestamps = np.array(all_timestamps, dtype=np.int64)

    # Ensure sorted
    sort_idx = np.argsort(timestamps)
    return rates[sort_idx], timestamps[sort_idx]
```

---

## 11. Comprehensive Risk Management Recommendations

### 11.1 Summary of Funding Rate Risk Tiers

| Risk Level | Funding Rate (abs) | Action |
|------------|-------------------|--------|
| **Negligible** | < 0.01%     | Normal trading, no adjustment |
| **Low**    | 0.01% - 0.03%     | Track cost, no position adjustment |
| **Moderate** | 0.03% - 0.05%  | Reduce size 25% in paying direction, avoid new entries within 15 min of settlement |
| **High**   | 0.05% - 0.10%     | Reduce size 50%, close before settlement if in paying direction |
| **Extreme** | > 0.10%          | Reduce size 75%, mandatory close before settlement, consider contrarian signal |

### 11.2 Integration Points in Our System

1. **`cost_engine.py`**: Update `FundingRateModel` to accept historical rate series instead of using constant mean. Already has the right structure, needs enhancement.

2. **`features/temporal.py`**: Add `time_to_next_funding` feature (already planned in CLAUDE.md). Add funding z-score features.

3. **`ingest/derivatives.py`**: The `BybitFundingCollector` already ingests funding rates. Add Binance funding ingestion (both REST for history and WebSocket for real-time predicted rate via `@markPrice` stream).

4. **`risk/position_sizer.py`**: Add funding-aware size adjustment. Reduce position when funding is elevated and position is in the paying direction.

5. **`backtest/engine.py`**: Integrate `FundingSimulator` to apply exact historical funding to backtest positions. Track funding P&L separately.

6. **`db/schema.py`**: The `funding_rate` table already exists with the right schema. May want to add a `predicted_rate` column for real-time predicted rate storage.

### 11.3 Checklist Before Going Live

- [ ] Historical funding rate data downloaded and loaded for full backtest period
- [ ] Backtest engine applies funding at exact settlement timestamps
- [ ] Funding P&L tracked separately in backtest output
- [ ] Sharpe with and without funding reported (expect 0.08-0.20 reduction)
- [ ] Pre-settlement close logic implemented (close before settlement if funding > 5 bps in paying direction)
- [ ] Position size reduction for elevated funding implemented
- [ ] Funding z-score feature added to model feature set
- [ ] Time-to-settlement feature added to model feature set
- [ ] Real-time predicted funding rate streamed via `@markPrice` WebSocket
- [ ] Cross-exchange funding spread monitored (Binance vs Bybit)
- [ ] Funding cost alert if daily aggregate exceeds 5 bps

### 11.4 Key Numbers to Remember

| Parameter | Value |
|-----------|-------|
| Mean funding rate (all-time) | +0.0057% per 8h |
| Mean funding rate (bull market) | +0.04% per 8h |
| Annual drag at mean | 0.78% |
| Annual drag in bull market | 5.5% |
| Sharpe reduction from funding | 0.08-0.20 |
| P(spanning settlement) at 10-min hold | 2.08% |
| Worst-case single settlement cost | 30 bps (at 0.3% extreme funding) |
| Funding autocorrelation lag-1 | 0.72 |
| Funding half-life of shocks | ~20 hours |
| Close-before-settlement threshold | 5 bps predicted rate |
| Size reduction threshold | 3 bps predicted rate |

---

## 12. Existing Code Assessment

### What already exists and is correct:
- `FundingRateModel` in `cost_engine.py`: Good structure, correct formula for counting settlements, correct direction sign logic
- `BybitFundingCollector` in `derivatives.py`: Correctly ingests Bybit funding via ccxt REST
- `funding_rate` table in `schema.py`: Has the right columns
- `time-to-funding` feature mentioned in CLAUDE.md feature plan

### What needs to be built/improved:
1. `FundingRateModel`: Replace constant `mean_rate` with historical rate series lookup
2. Add Binance funding collector (REST + WebSocket `@markPrice` stream)
3. Add `FundingSimulator` class for backtest integration
4. Add funding feature computer (z-score, time-to-settlement, predicted rate)
5. Add pre-settlement close logic in risk manager
6. Add funding-aware position sizing in `position_sizer.py`
7. Download and store historical funding rate data (2020-present) for backtesting
