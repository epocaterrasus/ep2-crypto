# Backtest-to-Production Parity for Crypto Trading Systems

## 1. The Backtest-to-Live Gap: Causes of 50-70% Sharpe Degradation

The degradation from backtest to live trading is not a single problem but a compounding of 8-12 independent friction sources. A backtested Sharpe of 2.0 typically translates to 0.6-1.0 live. Understanding the decomposition is essential for minimizing each factor.

### 1.1 Decomposition of Sharpe Degradation

| Factor | Sharpe Impact | Typical Magnitude |
|--------|--------------|-------------------|
| Overfitting / data mining bias | -0.3 to -0.8 | Largest single factor |
| Transaction costs (spread + commission) | -0.2 to -0.5 | 8-12 bps round-trip for BTC perps |
| Slippage and market impact | -0.1 to -0.3 | Worse during volatile periods |
| Latency (signal-to-fill delay) | -0.05 to -0.15 | 100-500ms typical for retail |
| Funding rate costs | -0.05 to -0.10 | 0.01-0.03% per 8h settlement |
| Look-ahead bias (undetected) | -0.0 to -0.5 | Often zero if rigorous, catastrophic if present |
| Regime change / non-stationarity | -0.1 to -0.3 | Alpha half-life 1-6 months |
| Survivorship / selection bias | -0.1 to -0.3 | Testing many strategies, reporting best |
| Liquidity illusion | -0.05 to -0.15 | Backtest assumes unlimited liquidity |
| Psychological execution errors | -0.05 to -0.10 | Not applicable for fully automated systems |

**Total expected degradation: 50-70% of backtested Sharpe.**

### 1.2 Mitigation Strategies Per Factor

**Overfitting (largest factor):**
- Apply the "haircut rule": discount backtested Sharpe by 50% as a starting assumption (Bailey & Lopez de Prado, 2014)
- Use purged walk-forward validation (already in our architecture: 14-day sliding window)
- Limit feature count to 18-25 (more features = more overfitting surface)
- Apply Deflated Sharpe Ratio (DSR) to account for number of strategies tested
- Multiple testing correction: if testing N strategies, expected max Sharpe under null scales as sqrt(2 * ln(N))

**Transaction costs:**
- Model explicit spread crossing: use historical bid-ask data, not mid prices
- Include maker/taker fee differential (Binance: 0.02%/0.04% for BTC perps)
- Strategy must survive >15 bps round-trip to be viable
- Reduce trading frequency: each trade costs ~8-12 bps, so fewer higher-conviction trades beat many marginal ones

**Slippage:**
- Use order book depth data for realistic fill simulation (see Section 2)
- Model slippage as a function of order size relative to available liquidity
- Apply the square-root impact law (confirmed for Bitcoin by Donier et al.)

**Latency:**
- Add 100-500ms delay in backtest execution logic (see Section 3)
- Use next-bar-open execution, never current-bar-close
- Model the full latency chain: signal computation + network + exchange matching

**Funding rates:**
- Include historical funding rate data in all backtests (see Section 6)
- Model the exact settlement times (every 8 hours on most exchanges)
- Track cumulative funding cost as a separate P&L component

**Regime change:**
- Use sliding (not expanding) training windows
- Monitor alpha decay with CUSUM and rolling Sharpe
- Build regime as an input feature, not separate models

---

## 2. Realistic Fill Simulation Using Historical Order Book Data

### 2.1 The Problem with Close-Price Fills

Most backtests assume fills at the close price of the signal bar. This is unrealistic because:
1. You cannot trade at historical close prices (they are known only after the bar closes)
2. Real orders cross the spread (immediate cost of half-spread)
3. Large orders walk the book (slippage increases with size)
4. Liquidity varies by time of day, regime, and recent volatility

### 2.2 Fill Simulation Hierarchy (Least to Most Realistic)

**Level 0: Close-price fill (naive)**
- Fill at bar close price
- Zero slippage assumption
- Overestimates Sharpe by 20-40%

**Level 1: Next-bar-open fill with fixed slippage**
- Fill at next bar's open + fixed slippage (e.g., 2 bps)
- Simple to implement, catches the biggest error
- Still misses liquidity-dependent slippage
- **Minimum acceptable for any serious backtest**

**Level 2: Spread-aware fill with volume-dependent slippage**
- Use historical bid-ask spread data
- Market buy fills at ask, market sell fills at bid
- Add slippage proportional to order size / average bar volume
- Formula: `fill_price = ask + slippage_coefficient * (order_size / avg_volume)^0.5`
- The square-root relationship is empirically validated for BTC (Donier et al., 2015)

**Level 3: Order book replay simulation**
- Replay historical L2 (market-by-price) order book snapshots
- Walk the actual order book to compute fill price for given order size
- Account for partial fills at each price level
- Most realistic for market orders
- Requires tick-level order book data (available from Binance, Bybit via data vendors)

**Level 4: Full tick replay with queue position modeling**
- Replay complete L2/L3 order book feed
- Model limit order queue position
- Account for feed latency and order latency separately
- Used by frameworks like hftbacktest (nkaz001/hftbacktest on GitHub)
- Overkill for 5-minute strategies but gold standard for HFT

### 2.3 Practical Fill Model for 5-Minute BTC Perps

For our 5-minute strategy, Level 2 is the practical sweet spot. Implementation:

```python
def simulate_fill(
    signal_direction: int,   # +1 buy, -1 sell
    order_size_usd: float,
    bar_open: float,
    bar_high: float,
    bar_low: float,
    bar_volume_usd: float,
    spread_bps: float,       # historical spread in basis points
    latency_ms: float = 200, # signal-to-fill latency
) -> float:
    """Simulate realistic fill price for a market order."""
    # Base price: next bar open (not current bar close)
    base_price = bar_open

    # 1. Spread crossing cost
    half_spread = base_price * (spread_bps / 2) / 10_000
    spread_cost = half_spread * signal_direction

    # 2. Market impact (square-root model)
    # Empirical: impact_bps ~ 5-15 * sqrt(order_fraction) for BTC
    participation_rate = order_size_usd / bar_volume_usd
    impact_bps = 10.0 * (participation_rate ** 0.5)  # calibrate from paper trading
    impact_cost = base_price * impact_bps / 10_000 * signal_direction

    # 3. Latency cost (price moves during delay)
    # Model as fraction of bar's range proportional to latency/bar_duration
    bar_range = bar_high - bar_low
    latency_fraction = latency_ms / (5 * 60 * 1000)  # fraction of 5-min bar
    # Adverse selection: assume price moves against you during latency
    latency_cost = bar_range * latency_fraction * 0.5 * signal_direction

    fill_price = base_price + spread_cost + impact_cost + latency_cost
    return fill_price
```

### 2.4 Calibration from Historical Data

Key parameters to calibrate:
- **Spread**: Collect historical bid-ask spreads from Binance WS depth feed. BTC/USDT perps typically: 0.5-1.5 bps during normal conditions, 3-10 bps during high volatility
- **Impact coefficient**: Start with 10 bps * sqrt(participation_rate), calibrate from paper trading
- **Volume profile**: Average 5-min volume for BTC perps is ~$5-20M during active hours, ~$1-5M during quiet hours

---

## 3. Latency Simulation: Signal to Fill

### 3.1 The Latency Chain

Every live trade experiences a chain of delays:

```
Signal generated (model inference)
  → Feature computation time:        5-50ms
  → Model inference time:             1-10ms (ONNX) / 10-100ms (native)
  → Order construction + signing:     1-5ms
  → Network to exchange:              10-50ms (co-located) / 50-200ms (cloud)
  → Exchange matching engine:         1-10ms
  → Confirmation back:                10-50ms
  ─────────────────────────────────────
  Total signal-to-fill:               30-365ms typical
  Conservative assumption:            200-500ms
```

### 3.2 How to Model Latency in Backtests

**Minimum approach: Fixed delay**
- Add 200ms fixed delay to all orders
- Execute at the price that existed 200ms after signal, approximated by next-bar-open

**Better approach: Stochastic latency**
- Model latency as a log-normal distribution: `latency ~ LogNormal(mu=5.3, sigma=0.5)` giving median ~200ms with fat right tail
- During high volatility, latency increases (exchange matching slows, network congestion)
- Sample latency per trade from this distribution

**Best approach: Regime-dependent latency**
- Normal conditions: 100-200ms median
- High volatility (>2x normal): 200-500ms median
- Extreme events (liquidation cascades): 500-2000ms median, with some orders rejected
- Model order rejection probability: 0.1% normal, 1-5% during cascades

### 3.3 Impact of Latency on 5-Minute Strategies

For a 5-minute bar with typical range of 10-50 bps:
- 200ms latency = 0.07% of bar duration = ~0.3-1.5 bps adverse movement
- 500ms latency = 0.17% of bar duration = ~0.7-4 bps adverse movement
- During cascades, latency cost can exceed the entire expected edge

**Key insight**: At 200ms latency, the cost is ~1 bps per trade. With 30 trades/day and 8-12 bps round-trip costs already, latency adds ~30 bps/day of drag. This is manageable but not negligible.

### 3.4 Latency Measurement Protocol

Before going live, measure actual latency:
1. Send dummy limit orders far from market, measure round-trip time
2. Track WebSocket feed timestamps vs local receipt time
3. Log every order's signal_time, send_time, ack_time, fill_time
4. Build empirical latency distribution from 1000+ observations
5. Feed this distribution back into the backtest simulator

---

## 4. Market Impact Modeling for Crypto

### 4.1 The Square-Root Impact Law

The most robust empirical finding in market microstructure is the square-root law of market impact:

```
ΔP/P = η * σ * sqrt(Q/V)
```

Where:
- `ΔP/P` = price impact as a fraction
- `η` = impact coefficient (dimensionless, typically 0.1-1.0)
- `σ` = daily volatility of the asset
- `Q` = order quantity (in base currency or USD)
- `V` = daily volume

**This law has been confirmed for Bitcoin** by Donier et al. (2015) in "A Million Metaorder Analysis of Market Impact on the Bitcoin" (arXiv:1412.4503). Key findings:
- The square-root law holds across four decades of order sizes
- Temporary impact is roughly 2/3 of peak impact (the rest is permanent)
- The law holds despite the quasi-absence of statistical arbitrage in crypto

### 4.2 Adapting Almgren-Chriss for BTC Perpetual Futures

The Almgren-Chriss framework decomposes execution cost into:
1. **Permanent impact**: Information content of the trade (moves price permanently)
2. **Temporary impact**: Liquidity demand (price recovers after execution)
3. **Timing risk**: Volatility exposure while executing

For BTC perpetual futures specifically:

**Permanent impact:**
```
g(v) = γ * v
```
Where `v` = trading rate (USD/second), `γ` = permanent impact coefficient.
For BTC perps: `γ ≈ 1e-10` to `1e-8` depending on conditions (calibrate from data).

**Temporary impact:**
```
h(v) = ε * sign(v) + η * |v|^0.5
```
Where `ε` = fixed cost (half-spread), `η` = temporary impact coefficient.
For BTC perps on Binance: `ε ≈ 0.5-1.5 bps`, `η` calibrated from order book depth.

**Crypto-specific modifications to Almgren-Chriss:**
- **24/7 trading**: No overnight gap risk, but volume is cyclical (Asian/European/US sessions)
- **Funding rate settlements**: Create periodic predictable volume spikes every 8 hours
- **Liquidation cascades**: Create non-linear impact during extreme moves
- **Cross-exchange arbitrage**: Reduces permanent impact faster than in equities
- **Leverage effects**: More leveraged market = faster price impact propagation

### 4.3 Practical Impact Parameters for BTC Perps

Based on empirical research and order book data:

| Order Size (USD) | Expected Impact (bps) | Notes |
|------------------|-----------------------|-------|
| $1,000 | 0.1-0.5 | Negligible |
| $10,000 | 0.5-2.0 | Typical retail |
| $100,000 | 2-7 | Significant for small funds |
| $1,000,000 | 7-25 | Requires execution algo |
| $10,000,000 | 25-80 | TWAP/VWAP over multiple bars |

These numbers assume normal liquidity conditions on Binance BTC/USDT perps. During low liquidity (Asian quiet hours, weekends): multiply by 2-3x. During cascades: multiply by 5-10x.

### 4.4 Implementation for Our System

For our quarter-Kelly position sizing with max ~5% of capital per trade:
- At $100K capital: max position = $5,000 → impact ~0.3 bps (negligible)
- At $1M capital: max position = $50,000 → impact ~1.5 bps (include in model)
- At $10M capital: max position = $500,000 → impact ~10 bps (requires execution algo)

**Recommendation**: Model impact at all capital levels but recognize it becomes material only above ~$100K position sizes. Use the square-root model with parameters calibrated from paper trading.

---

## 5. Look-Ahead Bias Checklist

Look-ahead bias is the single most dangerous source of backtest inflation. Even a 1-bar leak can turn a losing strategy into a winner. The following is an exhaustive checklist organized by where bias can infiltrate.

### 5.1 Feature Computation

- [ ] **Rolling windows use only past data**: `df['feature'] = df['price'].rolling(20).mean()` uses pandas default of right-aligned windows. Verify `.shift(1)` is applied if the current bar should not be included.
- [ ] **No centered rolling windows**: Never use `center=True` in any rolling calculation.
- [ ] **EMA initialization**: EMA on first N bars uses all N bars including "future" data relative to bar 1. Start EMA calculation from a warm-up period and discard initial values.
- [ ] **Cross-sectional features**: If computing features across assets (e.g., BTC vs ETH correlation), ensure both time series are aligned to the same timestamp and only use data available at that timestamp.
- [ ] **Resampling alignment**: When resampling from tick to 5-min bars, ensure the 5-min bar at time T uses only ticks with timestamp <= T.
- [ ] **Order book features**: Order book snapshot at time T must reflect only the state of the book at T, not any subsequent updates.
- [ ] **Cross-market features**: NQ, Gold, DXY data may have different timestamps. If using 15-min delayed data, the feature at time T should use cross-market data from T-15min.
- [ ] **No `.iloc[-1]` in feature functions**: This accesses the last element of whatever array is passed, which in a vectorized backtest may include future data.

### 5.2 Target / Label Construction

- [ ] **Triple barrier labels**: The label for bar T is determined by future price action. Ensure labels are used ONLY as training targets, never as features.
- [ ] **Adaptive threshold**: If the up/flat/down threshold adapts based on recent volatility, "recent" must mean strictly past data (e.g., trailing 288 bars, not including the current bar).
- [ ] **Label leakage through sample weights**: If weights depend on label distribution, compute distribution only on training set.
- [ ] **Return calculation**: `return_t = price_{t+1} / price_t - 1` assigns the return to time t, but this return was not known at time t. This is correct for labeling but must never be used as a feature.

### 5.3 Normalization and Preprocessing

- [ ] **Normalization parameters fitted on training data only**: Never compute mean/std/min/max on the full dataset. Per-fold, per-window.
- [ ] **Rank transformation**: If rank-transforming features, compute rank within the training window only.
- [ ] **PCA / dimensionality reduction**: Fit PCA on training data, transform both train and test.
- [ ] **Missing value imputation**: If using any statistical method (mean/median imputation), compute statistics on training data only. Forward-fill is safe because it only uses past data.
- [ ] **Outlier clipping**: Clip thresholds must be computed from training data only.

### 5.4 Data Alignment and Timing

- [ ] **Execution timing**: Signal at bar T close → order at bar T+1 open. Never fill at bar T close.
- [ ] **Data publication delays**: Economic data (CPI, FOMC) has a publication timestamp that may differ from the reference period. Use publication timestamp.
- [ ] **Exchange data timestamps**: Verify that exchange timestamps are exchange-time (not local-time). Binance uses server UTC timestamps.
- [ ] **Funding rate timing**: Funding rate for the 00:00 UTC settlement is announced in advance but settled at 00:00. Use the announced rate as a feature only after it is announced (typically available ~15min before settlement).
- [ ] **On-chain data**: Bitcoin blocks have timestamps but confirmation takes ~10-60min. Use confirmation time, not block timestamp, for any on-chain features.

### 5.5 Validation and Training

- [ ] **Purge gap between train and test**: 18 bars (90 minutes) minimum. This prevents label leakage from overlapping triple barrier windows.
- [ ] **Embargo after test set**: 12 bars (60 minutes). Prevents information leakage from test into the next training window.
- [ ] **No random splitting**: All splits must be temporal. Random splitting in time series is always look-ahead bias.
- [ ] **Walk-forward, not train-once**: Each test period must be preceded by a fresh training period.
- [ ] **Early stopping validation set**: Carved from the training set, NOT from the test set. If early stopping uses the test set, the model effectively trains on future data.
- [ ] **Hyperparameter selection**: Hyperparameters optimized over many walk-forward folds can overfit to the specific sequence of regimes. Use nested cross-validation or a held-out final test period.

### 5.6 Automated Detection Tests

**Truncation test**: For any feature f, compute on data[:150] and data[:200]. The values at index 150 must be identical. If they differ, the feature uses future data.

**Shuffle test**: Randomly shuffle the time series and compute features. If prediction accuracy remains high on shuffled data, there is likely look-ahead bias (the model is using information from adjacent bars that got mixed in).

**Delay test**: Artificially delay all features by 1 bar. If performance drops dramatically (>50%), the strategy may depend on same-bar information.

**Overfit diagnostic**: If walk-forward Sharpe is dramatically lower than in-sample Sharpe (>3x difference), suspect look-ahead bias or overfitting.

---

## 6. Funding Rate Simulation

### 6.1 How Funding Works on BTC Perpetual Futures

Perpetual futures use funding rates to keep the futures price anchored to the spot price:
- **Settlement frequency**: Every 8 hours (00:00, 08:00, 16:00 UTC on Binance)
- **Rate determination**: Based on premium/discount of perp price vs spot index
- **Payment direction**: If funding rate > 0, longs pay shorts. If < 0, shorts pay longs.
- **Typical range**: -0.03% to +0.10% per 8h period (annualized: -13% to +46%)
- **Average**: Historically positive ~0.01% per 8h (longs pay shorts), reflecting bullish bias

### 6.2 Correct Modeling in Backtests

**What most backtests get wrong:**
1. Ignoring funding entirely (most common error)
2. Using average funding rate instead of historical per-period rates
3. Not accounting for the exact settlement timestamp
4. Not modeling funding as a function of position direction

**Correct implementation:**

```python
def apply_funding_cost(
    position_direction: int,    # +1 long, -1 short
    position_size_usd: float,
    entry_time: datetime,
    exit_time: datetime,
    funding_rates: pd.DataFrame,  # columns: [timestamp, rate]
) -> float:
    """Calculate total funding cost for a position."""
    total_funding = 0.0

    # Find all funding settlements that occur while position is open
    settlements = funding_rates[
        (funding_rates['timestamp'] > entry_time) &
        (funding_rates['timestamp'] <= exit_time)
    ]

    for _, row in settlements.iterrows():
        # Positive rate: longs pay shorts
        # Negative rate: shorts pay longs
        payment = position_size_usd * row['rate'] * position_direction
        total_funding -= payment  # negative = cost to us

    return total_funding
```

### 6.3 Funding Rate Data Sources

- **Binance**: Historical funding rates via REST API (`GET /fapi/v1/fundingRate`), available back to perp launch
- **Bybit**: Similar REST endpoint
- **Aggregated**: CoinGlass, Coinalyze provide cross-exchange historical funding
- **Resolution**: 8-hour snapshots (3 per day)

### 6.4 Funding Rate as a Feature vs. Cost

Funding rate serves dual purposes in our system:
1. **Cost**: Deducted from P&L for any position held across a settlement (Section 6.2)
2. **Feature**: High positive funding signals crowded longs (contrarian signal). This is why we skip NLP sentiment and use funding rate as our "sentiment" proxy (ADR-005).

**Key rule**: The funding rate used as a feature at time T must be the most recently settled rate, not the upcoming one. The upcoming rate is only finalized at settlement time.

### 6.5 Impact Quantification

For our strategy with average hold time of ~2-5 bars (10-25 minutes):
- Probability of crossing a funding settlement: ~3.5-8.7% per trade (1-2.5 bars overlap with settlement in a 24h period with 3 settlements)
- Average cost per trade: 0.01% * 5% probability = 0.0005% (~0.05 bps)
- Annualized impact on 30 trades/day: ~5.5 bps/year

**Conclusion**: Funding costs are small for short-hold strategies but must be modeled for correctness. They become significant for strategies that hold positions for hours or across multiple settlements.

---

## 7. Event Replay Architecture

### 7.1 Why Event Replay Matters

The gold standard for backtest realism is replaying the exact sequence of market events that occurred historically. This eliminates the OHLCV approximation problem and allows testing strategies exactly as they would have executed.

### 7.2 Architecture Design

```
┌─────────────────────────────────────────────────┐
│                Event Replay Engine               │
│                                                  │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐    │
│  │  Event    │   │  Event   │   │  Event   │    │
│  │  Store    │──>│  Player  │──>│  Bus     │    │
│  │ (Parquet) │   │ (replay) │   │          │    │
│  └──────────┘   └──────────┘   └────┬─────┘    │
│                                      │          │
│           ┌──────────────────────────┤          │
│           │              │           │          │
│     ┌─────▼─────┐ ┌─────▼────┐ ┌───▼────┐     │
│     │  Feature   │ │  Signal  │ │ Order  │     │
│     │  Engine    │ │  Engine  │ │ Mgmt   │     │
│     └───────────┘ └──────────┘ └────────┘     │
│                                                  │
│     Same code path as live trading               │
└─────────────────────────────────────────────────┘
```

### 7.3 Event Types to Store and Replay

| Event Type | Source | Storage Format | Size Estimate (1 day BTC) |
|------------|--------|---------------|--------------------------|
| Trade ticks (aggTrades) | Binance WS | Parquet: [ts, price, qty, side] | ~50-200 MB |
| Order book snapshots (depth@100ms) | Binance WS | Parquet: [ts, bids[20], asks[20]] | ~2-5 GB |
| Order book deltas | Binance WS diff | Parquet: [ts, side, price, qty] | ~500 MB-1 GB |
| Klines (5min) | Binance WS | Parquet: [ts, O, H, L, C, V] | ~50 KB |
| Funding rate | Binance REST | CSV: [ts, rate] | <1 KB |
| Open interest | Bybit REST | Parquet: [ts, OI] | ~100 KB |
| Liquidations | Bybit WS | Parquet: [ts, side, price, qty] | ~1-10 MB |
| Cross-market (NQ, Gold, DXY) | yfinance | Parquet: [ts, O, H, L, C, V] | ~200 KB |

**Storage requirement**: ~3-7 GB per day for full L2 replay. For 60 days of history: ~180-420 GB.

**Pragmatic alternative**: Store trade ticks + periodic order book snapshots (every 1 second instead of 100ms). Reduces storage to ~200-500 MB/day while maintaining good fill simulation.

### 7.4 Implementation Principles

1. **Unified code path**: The strategy code must be identical between replay and live. The only difference is the event source (file vs WebSocket).
2. **Time advancement**: The replay engine controls "now". Features can only access events with timestamp <= current_time.
3. **Latency injection**: Add configurable latency between "event received" and "order sent" to simulate real processing delays.
4. **Order matching**: For market orders, walk the replayed order book. For limit orders, model queue position based on order book state.
5. **Clock fidelity**: Use nanosecond timestamps. Even at 5-minute resolution, sub-second timing matters for fills near bar boundaries.

### 7.5 Reference Implementation: hftbacktest

The hftbacktest framework (github.com/nkaz001/hftbacktest) is the best open-source reference for event replay in crypto:
- Python + Numba JIT (matches our tech stack)
- Native support for Binance Futures and Bybit
- Models feed latency and order latency separately
- Queue position modeling for limit orders
- Available as a Python package: `pip install hftbacktest`

For our 5-minute strategy, we do not need the full HFT-grade replay. But the architecture patterns (event store, replay engine, unified code path) are directly applicable.

---

## 8. Paper Trading as the Bridge

### 8.1 Purpose of Paper Trading

Paper trading serves as the calibration bridge between backtest and live:
1. **Validate backtest assumptions**: Are fills, latencies, and costs realistic?
2. **Calibrate backtest parameters**: Use empirical paper trading data to tune backtest models
3. **Detect implementation bugs**: Code that works in backtest may fail in live due to async issues, data format changes, etc.
4. **Build operational confidence**: Verify monitoring, alerting, and risk management work

### 8.2 Paper Trading Design

**Phase 1: Shadow mode (weeks 1-2)**
- Run the full pipeline live but do not send real orders
- Record: signal_time, predicted_direction, predicted_confidence, theoretical_entry_price
- Compare against actual market state at signal_time + latency
- Compute theoretical P&L with realistic fills

**Phase 2: Exchange paper trading (weeks 3-4)**
- Use Binance Testnet or Bybit Testnet
- Send real orders to the testnet matching engine
- Record actual fill prices and latencies
- Note: testnet liquidity differs from mainnet, so fills are approximate

**Phase 3: Micro-live (weeks 5-8)**
- Trade with minimum position size ($10-100) on the real exchange
- Real fills, real latency, real liquidity
- The definitive calibration source
- 4-week minimum to capture sufficient trade samples (500+ trades at 30/day)

### 8.3 Calibration Protocol

After Phase 3, compute empirical calibration factors:

```python
# For each trade, compute:
calibration_data = {
    'backtest_fill_price': ...,     # what backtest said
    'actual_fill_price': ...,       # what actually happened
    'backtest_latency': ...,        # assumed latency
    'actual_latency': ...,          # measured signal-to-fill
    'backtest_slippage': ...,       # modeled slippage
    'actual_slippage': ...,         # measured slippage
    'backtest_funding_cost': ...,   # modeled funding
    'actual_funding_cost': ...,     # actual funding paid
}

# Compute correction factors:
slippage_ratio = median(actual_slippage / backtest_slippage)
latency_ratio = median(actual_latency / backtest_latency)

# Feed back into backtest:
# adjusted_slippage = modeled_slippage * slippage_ratio
# adjusted_latency = modeled_latency * latency_ratio
```

### 8.4 Minimum Sample Size

For statistically meaningful calibration:
- **Sharpe ratio comparison**: Need 500+ trades (Bailey & Lopez de Prado recommend 600+)
- **Slippage calibration**: Need 200+ fills across different market conditions
- **Latency distribution**: Need 1000+ measurements
- **At 30 trades/day**: ~17 days for Sharpe, ~7 days for slippage, ~3 days for latency

### 8.5 Go/No-Go Criteria

Paper trading must pass ALL of these before going live:
1. Paper Sharpe is within 30% of backtested Sharpe (after applying the 50% haircut)
2. Actual slippage is within 2x of modeled slippage
3. No catastrophic drawdown events unexplained by market conditions
4. All risk gates trigger correctly (daily loss limit, max drawdown, kill switch)
5. Statistical tests confirm backtest and paper returns are from compatible distributions (Section 9)

---

## 9. Statistical Tests: Backtest vs. Paper Trading

### 9.1 The Core Question

"Are the returns from my backtest and my paper trading drawn from the same underlying distribution?"

If yes: the backtest is realistic and can be trusted for forward projections.
If no: the backtest has systematic biases that need to be identified and corrected.

### 9.2 Recommended Test Battery

**Test 1: Two-Sample Kolmogorov-Smirnov Test**
```python
from scipy.stats import ks_2samp

stat, p_value = ks_2samp(backtest_returns, paper_returns)
# H0: returns come from the same distribution
# Reject H0 if p_value < 0.05
# This catches differences in both location (mean) and shape (distribution)
```

Advantages: Non-parametric, catches full distribution differences.
Limitation: Less power for small samples (<100 trades).

**Test 2: Welch's t-test on Mean Returns**
```python
from scipy.stats import ttest_ind

stat, p_value = ttest_ind(backtest_returns, paper_returns, equal_var=False)
# H0: means are equal
# This specifically tests whether average return per trade differs
```

Advantages: High power for detecting mean shift.
Limitation: Assumes roughly normal returns (OK for per-trade returns with enough samples).

**Test 3: Mann-Whitney U Test**
```python
from scipy.stats import mannwhitneyu

stat, p_value = mannwhitneyu(backtest_returns, paper_returns)
# H0: distributions have equal medians
# Non-parametric alternative to t-test
```

Advantages: Robust to outliers, non-parametric.

**Test 4: Levene's Test for Variance Equality**
```python
from scipy.stats import levene

stat, p_value = levene(backtest_returns, paper_returns)
# H0: variances are equal
# Important: even if means match, different volatility = different risk profile
```

**Test 5: Bootstrap Confidence Interval for Sharpe Difference**
```python
import numpy as np

def sharpe(returns):
    return np.mean(returns) / np.std(returns) * np.sqrt(105_120)  # crypto annualization

n_bootstrap = 10_000
sharpe_diffs = []
for _ in range(n_bootstrap):
    bt_sample = np.random.choice(backtest_returns, size=len(backtest_returns), replace=True)
    pt_sample = np.random.choice(paper_returns, size=len(paper_returns), replace=True)
    sharpe_diffs.append(sharpe(bt_sample) - sharpe(pt_sample))

ci_lower = np.percentile(sharpe_diffs, 2.5)
ci_upper = np.percentile(sharpe_diffs, 97.5)
# If CI contains 0, the Sharpe difference is not statistically significant
```

**Test 6: Permutation Test on Win Rate Difference**
```python
def permutation_test_win_rate(bt_returns, pt_returns, n_perms=10_000):
    observed_diff = np.mean(bt_returns > 0) - np.mean(pt_returns > 0)
    combined = np.concatenate([bt_returns, pt_returns])
    count = 0
    for _ in range(n_perms):
        np.random.shuffle(combined)
        perm_diff = (np.mean(combined[:len(bt_returns)] > 0) -
                     np.mean(combined[len(bt_returns):] > 0))
        if abs(perm_diff) >= abs(observed_diff):
            count += 1
    return count / n_perms  # p-value
```

### 9.3 Interpretation Framework

| Test Result | Interpretation | Action |
|-------------|---------------|--------|
| All tests p > 0.10 | Backtest and paper trading are consistent | Proceed to live with confidence |
| KS test p < 0.05 but t-test p > 0.05 | Distribution shape differs but means match | Investigate tail behavior; may proceed cautiously |
| t-test p < 0.05 (backtest mean > paper) | Backtest overestimates returns | Calibrate: increase cost model, check for bias |
| Levene's p < 0.05 (backtest vol < paper) | Backtest underestimates risk | Increase position sizing conservatism |
| Sharpe CI excludes 0 (backtest > paper) | Systematic Sharpe inflation in backtest | Full audit of backtest assumptions needed |
| Win rate differs significantly | Directional accuracy is off | Check feature pipeline for look-ahead bias |

### 9.4 Required Sample Sizes

| Metric | Min Trades for 80% Power | At 30 trades/day |
|--------|--------------------------|-------------------|
| Mean return (t-test) | 200-400 | 7-14 days |
| Distribution (KS test) | 100-200 | 4-7 days |
| Sharpe ratio (bootstrap) | 500-1000 | 17-34 days |
| Win rate (permutation) | 200-400 | 7-14 days |

**Recommendation**: Run paper trading for minimum 30 days (900 trades) before drawing statistical conclusions.

---

## 10. Published Case Studies of Backtest-to-Live Gaps

### 10.1 Academic and Industry Evidence

**Marcos Lopez de Prado (2018) - "Advances in Financial Machine Learning"**
- Documents that the correlation between backtested Sharpe and live Sharpe is "statistically zero" across a sample of quantitative strategies
- Introduces the Deflated Sharpe Ratio: if testing N strategies, the probability of finding a Sharpe > 2.0 by chance alone increases rapidly. With N=100, a backtested Sharpe of 2.0 is not statistically significant
- Key number: at least Sharpe > 2.0 * sqrt(1 + (N-1)/T) to survive multiple testing correction, where T = number of time periods

**Man Group / AHL (2019) - "Backtesting" Research Paper**
- One of the few institutional quant firms to publish on backtest realism
- Key finding: "A common practice in evaluating backtests of trading strategies is to discount the reported Sharpe ratios by 50%"
- This 50% haircut reflects combined overfitting, cost underestimation, and regime change
- They report that strategies with backtested Sharpe < 1.5 almost never survive in live trading

**Bailey, Borwein, Lopez de Prado (2014) - "The Deflated Sharpe Ratio"**
- With 10 independent trials: minimum backtested Sharpe for significance = ~1.4
- With 100 independent trials: minimum = ~2.0
- With 1000 trials: minimum = ~2.5
- Most quant researchers test hundreds of feature combinations, effectively running thousands of implicit trials

### 10.2 Crypto-Specific Evidence

**Binance Futures Trading Competitions (2020-2023)**
- Winners in backtesting competitions often fail to replicate in live rounds
- Common pattern: strategies that exploit low-liquidity periods in backtest face slippage in live that eliminates the edge
- Specific observation: strategies with backtested Sharpe of 3-5 delivered live Sharpe of 0.5-1.5

**Stoic AI (Cindicator) Public Performance Data**
- Published strategy: systematic crypto, daily rebalance
- Backtested performance: not disclosed but implied Sharpe > 2.0
- Live performance (2020-2022): Sharpe approximately 0.8-1.2 after fees
- Degradation consistent with the 50-70% range

**Quantified Strategies Research (2023)**
- Analyzed multiple published crypto strategies
- Key finding: "If your backtest shows 50% annual returns, assume live returns will be notably lower, as professional traders often expect live performance to be 60-70% of backtested results"
- Drawdown amplification: "Expect live drawdown to be 1.5x to 2x greater than backtested drawdown"

### 10.3 Specific Degradation Numbers from Practice

**Manual execution study (Breaking Alpha, 2024):**
- Manual execution caused entry prices to average 1.5 ticks worse than signal prices
- This eroded 22% of the strategy's edge
- Key lesson: even with a correct strategy, execution quality matters enormously

**Freqtrade community backtests (2021-2024):**
- Community-reported pattern: strategies with 100%+ backtested annual return frequently deliver 10-30% live
- Most common causes identified: (1) look-ahead bias in default indicator calculations, (2) backtesting at close prices instead of next-bar open, (3) ignoring slippage in thin-liquidity altcoin markets

**Imperial College London thesis (Kargarzadeh, 2022):**
- Developed and backtested a crypto trading strategy using machine learning
- Backtested Sharpe: ~2.1 on BTC/USDT
- Paper trading Sharpe: ~1.3 (38% degradation)
- Key degradation sources identified: execution timing, spread costs, model staleness

### 10.4 Calibration Techniques from Practice

**Technique 1: Progressive realism stacking**
Start with a naive backtest and add realism layers one at a time, measuring the impact of each:
1. Naive (close-price fills) → Sharpe 3.2
2. + Next-bar-open fills → Sharpe 2.8 (-12%)
3. + Spread crossing → Sharpe 2.4 (-15%)
4. + Fixed slippage (2 bps) → Sharpe 2.1 (-12%)
5. + Latency (200ms) → Sharpe 2.0 (-5%)
6. + Funding costs → Sharpe 1.95 (-2%)
7. + Transaction costs (full) → Sharpe 1.7 (-13%)
8. + Regime-dependent adjustments → Sharpe 1.5 (-12%)
9. Apply 50% haircut → Expected live Sharpe: ~0.75

**Technique 2: Walk-forward degradation ratio**
- Compute in-sample Sharpe and out-of-sample Sharpe across all walk-forward folds
- The ratio (OOS/IS) is a predictor of live degradation
- If OOS/IS = 0.5, expect live to be ~0.5 of OOS (total live/IS = 0.25)
- If OOS/IS = 0.8, expect live to be ~0.7 of OOS (total live/IS = 0.56)

**Technique 3: Monte Carlo stress degradation**
- Run 1000 Monte Carlo simulations with randomized execution parameters:
  - Latency: sample from measured distribution
  - Slippage: sample from 1x to 3x modeled slippage
  - Spread: sample from historical distribution
  - Funding: use actual historical rates
- Report the 25th percentile Sharpe as the "realistic" expectation
- If 25th percentile Sharpe < 1.0, strategy likely cannot survive live

**Technique 4: Adversarial backtest**
- Run the backtest with deliberately pessimistic assumptions:
  - 2x expected slippage
  - 500ms latency (instead of 200ms)
  - Worst-case spread (95th percentile historical)
  - All fills at the worst price in the bar (high for buys, low for sells)
- If the strategy is still profitable under adversarial conditions, it has a real edge

---

## Summary: Key Numbers for ep2-crypto

| Parameter | Value | Source |
|-----------|-------|--------|
| Expected Sharpe haircut | 50-70% | Man Group, Lopez de Prado |
| Target backtest Sharpe | > 2.0 | To survive degradation and reach live Sharpe ~0.7-1.0 |
| Minimum fill model | Next-bar-open + spread + sqrt impact | This research |
| Latency assumption | 200ms median, log-normal distribution | Empirical measurement |
| Impact model | Square-root law: η * σ * sqrt(Q/V) | Donier et al., confirmed for BTC |
| Funding rate impact | ~0.05 bps/trade for 5-min holds | Calculated from settlement probability |
| Paper trading duration | 30+ days (900+ trades) | Statistical power requirements |
| Key statistical test | KS test + bootstrap Sharpe CI | Comprehensive validation |
| Adversarial slippage | 2x modeled slippage | Stress testing standard |
| Drawdown amplification | 1.5-2x backtested drawdown | Quantified Strategies, practice |

## Sources

- [Man Group - Backtesting](https://www.man.com/insights/backtesting)
- [A Million Metaorder Analysis of Market Impact on Bitcoin](https://arxiv.org/abs/1412.4503)
- [Almgren-Chriss Optimal Execution](https://www.smallake.kr/wp-content/uploads/2016/03/optliq.pdf)
- [hftbacktest - High Frequency Backtesting Framework](https://github.com/nkaz001/hftbacktest)
- [NautilusTrader - Event-Driven Backtesting](https://nautilustrader.io/docs/latest/concepts/backtesting/)
- [CoinAPI - Order Book Replay Guide](https://www.coinapi.io/blog/crypto-order-book-replay)
- [CoinAPI - Backtest with Real Market Data](https://www.coinapi.io/blog/backtest-crypto-strategies-with-real-market-data)
- [CoinAPI - Crypto Trading Latency Guide](https://www.coinapi.io/blog/crypto-trading-latency-guide)
- [Look-Ahead Bias Prevention in Quantitative Trading](https://medium.com/@jpolec_72972/look-ahead-bias-prevention-and-signal-processing-in-quantitative-trading-9def856db5a6)
- [Look-Ahead Bias: The Invisible Killer](https://www.newsletter.quantreo.com/p/look-ahead-bias-the-invisible-killer)
- [Freqtrade Lookahead Analysis](https://www.freqtrade.io/en/stable/lookahead-analysis/)
- [StrategyQuant - Compare Live vs Backtest](https://strategyquant.com/blog/real-trading-compare-live-strategy-results-backtest/)
- [Breaking Alpha - Understanding Sharpe Ratios](https://breakingalpha.io/insights/understanding-sharpe-ratios-selecting-trading-algorithms)
- [Backtest vs Live Trading - Why 300% Returns Fail](https://blog.pickmytrade.trade/backtest-vs-live-trading-why-300-returns-fail-in-real-markets/)
- [Square Root Law for Price Impact (Imperial)](https://www.imperial.ac.uk/media/imperial-college/research-centres-and-groups/cfm-imperial-institute-of-quantitative-finance/events/imperial-eth-2016/Jonathan-Donier-.pdf)
- [Designing Funding Rates for Perpetual Futures](https://arxiv.org/html/2506.08573v1)
- [QuestDB - Almgren-Chriss Model](https://questdb.com/glossary/optimal-execution-strategies-almgren-chriss-model/)
- [Developing and Backtesting a Trading Strategy (Imperial College)](https://www.imperial.ac.uk/media/imperial-college/faculty-of-natural-sciences/department-of-mathematics/math-finance/Kargarzadeh_Alireza_02092220.pdf)
- [NYU Stern - Online Quantitative Trading Strategies](https://www.stern.nyu.edu/sites/default/files/2025-05/Glucksman_Lahanis.pdf)
- [Scipy KS Two-Sample Test](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html)
- [SBAI - Backtesting ARP Strategies](https://www.sbai.org/static/a7fc0b57-9c84-4b72-adae25e4bfae1be2/ARP-Backtesting.pdf)
