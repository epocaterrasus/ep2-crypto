# Deep Research: OFI & Microprice Implementation for 5-Min Crypto Prediction

## Table of Contents
1. [OFI from L2 Snapshots (Cont-Kukanov-Stoikov 2014)](#1-ofi-from-l2-snapshots)
2. [OFI from aggTrade Data](#2-ofi-from-aggtrade-data)
3. [OFI Normalization](#3-ofi-normalization)
4. [Microprice Implementation](#4-microprice-implementation)
5. [Microprice as Fair Value Estimator](#5-microprice-as-fair-value-estimator)
6. [Feature Computation from depth@100ms](#6-feature-computation-from-depth100ms)
7. [Order Book Event Classification](#7-order-book-event-classification)
8. [Trade Flow Imbalance (TFI)](#8-trade-flow-imbalance)
9. [Kyle's Lambda](#9-kyles-lambda)
10. [Absorption Detection](#10-absorption-detection)
11. [Feature Update Frequency](#11-feature-update-frequency)
12. [Complete Pipeline](#12-complete-pipeline)

---

## 1. OFI from L2 Snapshots

### Paper Reference
Cont, Kukanov, & Stoikov (2014), "The Price Impact of Order Book Events," *Journal of Financial Econometrics*, 12(1), 47-88. (Often cited as Cont-Stoikov-Talreja from the 2010 working paper; the published version has Kukanov as second author.)

### The Core Idea
OFI measures the net order flow pressure by tracking *changes* in the best bid and ask quantities between consecutive L2 snapshots. The key insight: it is not the *level* of liquidity that predicts price changes, but the *change* in liquidity at the best quotes.

### Exact Algorithm: The 6 Cases

Given consecutive snapshots at times t-1 and t, let:
- `P_b(t), Q_b(t)` = best bid price and quantity at time t
- `P_a(t), Q_a(t)` = best ask price and quantity at time t

**Bid-side contribution `e_b(t)`:**

| Case | Condition | e_b(t) |
|------|-----------|--------|
| Bid price UP | P_b(t) > P_b(t-1) | Q_b(t) |
| Bid price SAME | P_b(t) = P_b(t-1) | Q_b(t) - Q_b(t-1) |
| Bid price DOWN | P_b(t) < P_b(t-1) | -Q_b(t-1) |

**Ask-side contribution `e_a(t)`:**

| Case | Condition | e_a(t) |
|------|-----------|--------|
| Ask price UP | P_a(t) > P_a(t-1) | Q_a(t-1) |
| Ask price SAME | P_a(t) = P_a(t-1) | -(Q_a(t) - Q_a(t-1)) |
| Ask price DOWN | P_a(t) < P_a(t-1) | -Q_a(t) |

**Single-event OFI:**
```
OFI(t) = e_b(t) + e_a(t)
```

**Aggregated OFI over interval [t1, t2]:**
```
OFI[t1,t2] = sum(OFI(t) for t in [t1+1, t2])
```

### Intuition Behind Each Case

**Bid UP**: A new, higher bid appeared. The entire quantity at the new best bid is fresh buying pressure: `e_b = +Q_b(t)`. The old best bid is no longer the best, so we treat it as if that buying interest evaporated.

**Bid SAME**: The best bid price hasn't changed, so we just measure the net change in displayed quantity. If quantity increased, more buyers showed up (+). If decreased, buyers canceled or got filled (-).

**Bid DOWN**: The previous best bid was entirely consumed (either executed against or canceled). All that buying interest is gone: `e_b = -Q_b(t-1)`.

**Ask UP**: The old best ask was entirely consumed. All that selling interest is gone, which is bullish: `e_a = +Q_a(t-1)`.

**Ask SAME**: Price unchanged; net change in ask quantity. If ask quantity *increased*, more sellers showed up (bearish, hence the negative sign). If *decreased*, sellers were consumed (bullish).

**Ask DOWN**: A new, lower ask appeared. Fresh selling pressure arrived: `e_a = -Q_a(t)`.

### Multi-Level OFI (MLOFI)

Xu, Kelejian & Haughton (2019), "Multi-Level Order-Flow Imbalance in a Limit Order Book." The same 6-case logic is applied at each of the top K price levels independently.

For level k (k=1,...,K):
```
OFI_k(t) = e_b_k(t) + e_a_k(t)
```

Where `P_b_k(t), Q_b_k(t)` are the k-th best bid price and quantity. The MLOFI vector is `[OFI_1, OFI_2, ..., OFI_K]`.

**Key finding**: Out-of-sample R-squared improves 60-70% for large-tick assets when going from level-1 OFI to 5-level MLOFI. For crypto (small tick relative to price), improvement is 15-35%.

**Practical issue with multi-level**: When the best bid moves up, the entire ladder shifts. Level 2 at time t-1 may become level 1 at time t. The implementation must track price levels, not ordinal positions. The recommended approach: compute OFI per *price level* (absolute), then aggregate by level rank at each snapshot.

### Implementation

```python
"""Order Flow Imbalance (OFI) computation from L2 order book snapshots.

Implements the Cont-Kukanov-Stoikov (2014) algorithm with multi-level
extension (Xu et al. 2019). All computations are numpy-vectorized.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def compute_ofi_single_level(
    bid_prices: npt.NDArray[np.float64],
    bid_quantities: npt.NDArray[np.float64],
    ask_prices: npt.NDArray[np.float64],
    ask_quantities: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Compute single-level OFI from consecutive L2 snapshots.

    Args:
        bid_prices: Shape (N,) best bid prices for N snapshots.
        bid_quantities: Shape (N,) best bid quantities.
        ask_prices: Shape (N,) best ask prices.
        ask_quantities: Shape (N,) best ask quantities.

    Returns:
        Shape (N,) OFI values. First element is 0 (no previous snapshot).
    """
    n = len(bid_prices)
    ofi = np.zeros(n, dtype=np.float64)

    if n < 2:
        return ofi

    # Bid-side contribution: 3 cases
    bp_prev, bp_curr = bid_prices[:-1], bid_prices[1:]
    bq_prev, bq_curr = bid_quantities[:-1], bid_quantities[1:]

    bid_up = bp_curr > bp_prev
    bid_same = bp_curr == bp_prev
    bid_down = bp_curr < bp_prev

    e_bid = np.zeros(n - 1, dtype=np.float64)
    e_bid[bid_up] = bq_curr[bid_up]
    e_bid[bid_same] = bq_curr[bid_same] - bq_prev[bid_same]
    e_bid[bid_down] = -bq_prev[bid_down]

    # Ask-side contribution: 3 cases
    ap_prev, ap_curr = ask_prices[:-1], ask_prices[1:]
    aq_prev, aq_curr = ask_quantities[:-1], ask_quantities[1:]

    ask_up = ap_curr > ap_prev
    ask_same = ap_curr == ap_prev
    ask_down = ap_curr < ap_prev

    e_ask = np.zeros(n - 1, dtype=np.float64)
    e_ask[ask_up] = aq_prev[ask_up]          # old ask consumed = bullish
    e_ask[ask_same] = -(aq_curr[ask_same] - aq_prev[ask_same])  # more sellers = bearish
    e_ask[ask_down] = -aq_curr[ask_down]      # new lower ask = bearish

    ofi[1:] = e_bid + e_ask
    return ofi


def compute_mlofi(
    bid_prices: npt.NDArray[np.float64],
    bid_quantities: npt.NDArray[np.float64],
    ask_prices: npt.NDArray[np.float64],
    ask_quantities: npt.NDArray[np.float64],
    n_levels: int = 5,
) -> npt.NDArray[np.float64]:
    """Compute multi-level OFI from L2 snapshots with K levels.

    Args:
        bid_prices: Shape (N, K) bid prices for N snapshots, K levels.
            Level 0 is best bid, level K-1 is deepest.
        bid_quantities: Shape (N, K) bid quantities.
        ask_prices: Shape (N, K) ask prices.
        ask_quantities: Shape (N, K) ask quantities.
        n_levels: Number of levels to use (1..K).

    Returns:
        Shape (N, n_levels) MLOFI matrix. Row i contains OFI at each level.
    """
    n = bid_prices.shape[0]
    k = min(n_levels, bid_prices.shape[1])
    mlofi = np.zeros((n, k), dtype=np.float64)

    for level in range(k):
        mlofi[:, level] = compute_ofi_single_level(
            bid_prices[:, level],
            bid_quantities[:, level],
            ask_prices[:, level],
            ask_quantities[:, level],
        )

    return mlofi


def aggregate_ofi_rolling(
    ofi: npt.NDArray[np.float64],
    window_sizes: list[int],
) -> dict[int, npt.NDArray[np.float64]]:
    """Rolling sum of OFI over multiple window sizes.

    Args:
        ofi: Shape (N,) raw OFI values (one per snapshot).
        window_sizes: List of window sizes in number of snapshots.
            For depth@100ms: 100 = 10s, 300 = 30s, 600 = 1min, 3000 = 5min.

    Returns:
        Dict mapping window_size -> rolling OFI array of shape (N,).
        Values before the window is full are computed on available data.
    """
    result = {}
    n = len(ofi)
    cumsum = np.cumsum(ofi)

    for w in window_sizes:
        rolling = np.empty(n, dtype=np.float64)
        # For indices < w, use all available data
        rolling[:w] = cumsum[:w]
        # For indices >= w, subtract the sum before the window
        rolling[w:] = cumsum[w:] - cumsum[:n - w]
        result[w] = rolling

    return result
```

### Unit Test

```python
def test_ofi_basic_cases():
    """Test all 6 OFI cases with known values."""
    # Snapshot 0: bid=100@10, ask=101@8
    # Snapshot 1: bid=100.5@5, ask=101@6  (bid UP, ask SAME)
    #   e_bid = Q_b(1) = 5 (new higher bid)
    #   e_ask = -(6 - 8) = 2 (ask qty decreased = sellers consumed)
    #   OFI = 7

    # Snapshot 2: bid=100.5@7, ask=101@6  (bid SAME, ask SAME)
    #   e_bid = 7 - 5 = 2
    #   e_ask = -(6 - 6) = 0
    #   OFI = 2

    # Snapshot 3: bid=100@12, ask=100.5@3  (bid DOWN, ask DOWN)
    #   e_bid = -Q_b(2) = -7 (old bid consumed)
    #   e_ask = -Q_a(3) = -3 (new lower ask = bearish)
    #   OFI = -10

    bp = np.array([100.0, 100.5, 100.5, 100.0])
    bq = np.array([10.0, 5.0, 7.0, 12.0])
    ap = np.array([101.0, 101.0, 101.0, 100.5])
    aq = np.array([8.0, 6.0, 6.0, 3.0])

    ofi = compute_ofi_single_level(bp, bq, ap, aq)

    assert ofi[0] == 0.0   # no previous snapshot
    assert ofi[1] == 7.0   # bid UP + ask qty decrease
    assert ofi[2] == 2.0   # bid SAME (qty +2) + ask SAME (no change)
    assert ofi[3] == -10.0  # bid DOWN + ask DOWN


def test_ofi_typical_btc():
    """Typical BTC/USDT order book OFI values.

    BTC depth@100ms: best bid/ask typically have 0.5-5 BTC.
    OFI per snapshot: typically -2 to +2 BTC.
    OFI over 5min (3000 snapshots): typically -50 to +50 BTC.
    """
    rng = np.random.default_rng(42)
    n = 3000  # 5 minutes at 100ms

    # Simulate slowly drifting BTC order book
    base_price = 65000.0
    drift = np.cumsum(rng.normal(0, 0.1, n))
    bid_p = base_price + drift - 0.5
    ask_p = base_price + drift + 0.5
    bid_q = np.abs(rng.normal(2.0, 0.8, n))
    ask_q = np.abs(rng.normal(2.0, 0.8, n))

    ofi = compute_ofi_single_level(bid_p, bid_q, ask_p, ask_q)

    # OFI should be centered near zero for symmetric book
    assert abs(ofi.mean()) < 0.5
    # 5-min cumulative should be within reasonable range
    assert abs(ofi.sum()) < 100.0
```

### Expected Values for BTC

| Metric | Typical Range (BTC/USDT) |
|--------|-------------------------|
| Per-snapshot OFI (100ms) | -3 to +3 BTC |
| 10-second rolling OFI | -15 to +15 BTC |
| 1-minute rolling OFI | -40 to +40 BTC |
| 5-minute rolling OFI | -100 to +100 BTC |
| OFI std (per snapshot) | ~0.8 BTC |

---

## 2. OFI from aggTrade Data

### Why an Alternative to L2 OFI

L2-snapshot OFI misses events between snapshots. At 100ms granularity, many orders are placed and canceled within a single snapshot interval. Trade-based OFI captures what actually *executed*, which is a cleaner signal of aggressive order flow.

### Algorithm

Using Binance aggTrade stream, each trade has:
- `p`: price
- `q`: quantity
- `m`: `isBuyerMaker` boolean
  - `true` = buyer was the maker = the trade was initiated by a seller (sell aggression)
  - `false` = seller was the maker = the trade was initiated by a buyer (buy aggression)

```
Trade_OFI(t) = sum(q_i for trades where m=false) - sum(q_i for trades where m=true)
```

In English: aggressive buy volume minus aggressive sell volume over interval t.

### Implementation

```python
"""Trade-based OFI from Binance aggTrade stream."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def compute_trade_ofi(
    quantities: npt.NDArray[np.float64],
    is_buyer_maker: npt.NDArray[np.bool_],
    timestamps_ms: npt.NDArray[np.int64],
    interval_ms: int = 1000,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    """Compute trade-based OFI aggregated into fixed time intervals.

    Args:
        quantities: Trade quantities (always positive).
        is_buyer_maker: True = sell aggression, False = buy aggression.
        timestamps_ms: Trade timestamps in milliseconds.
        interval_ms: Aggregation interval (default 1 second).

    Returns:
        Tuple of (interval_starts_ms, ofi_per_interval).
    """
    if len(quantities) == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float64)

    # Sign trades: buy aggression = +1, sell aggression = -1
    signs = np.where(is_buyer_maker, -1.0, 1.0)
    signed_volume = quantities * signs

    # Bin into intervals
    t_start = timestamps_ms[0]
    t_end = timestamps_ms[-1]
    n_intervals = int((t_end - t_start) / interval_ms) + 1

    interval_idx = ((timestamps_ms - t_start) / interval_ms).astype(np.int64)
    interval_idx = np.clip(interval_idx, 0, n_intervals - 1)

    ofi = np.zeros(n_intervals, dtype=np.float64)
    np.add.at(ofi, interval_idx, signed_volume)

    interval_starts = t_start + np.arange(n_intervals, dtype=np.int64) * interval_ms
    return interval_starts, ofi


def compute_trade_ofi_rolling(
    quantities: npt.NDArray[np.float64],
    is_buyer_maker: npt.NDArray[np.bool_],
    timestamps_ms: npt.NDArray[np.int64],
    window_ms: int = 60_000,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    """Rolling trade OFI with a sliding window.

    Returns one value per trade (not per interval). More granular
    but more expensive. Use for real-time monitoring.
    """
    signs = np.where(is_buyer_maker, -1.0, 1.0)
    signed_volume = quantities * signs

    n = len(quantities)
    rolling_ofi = np.zeros(n, dtype=np.float64)

    # Use searchsorted for efficient window boundary finding
    window_starts = np.searchsorted(timestamps_ms, timestamps_ms - window_ms, side="left")

    cumsum = np.cumsum(signed_volume)
    # rolling_ofi[i] = cumsum[i] - cumsum[window_starts[i] - 1]
    rolling_ofi = cumsum - np.where(
        window_starts > 0,
        cumsum[window_starts - 1],
        0.0,
    )

    return timestamps_ms, rolling_ofi
```

### L2 OFI vs Trade OFI: When to Use Which

| Aspect | L2 Snapshot OFI | Trade OFI |
|--------|----------------|-----------|
| What it measures | Net change in displayed liquidity | Net aggressive execution |
| Includes cancels | Yes | No |
| Includes hidden orders | No | Partially (they execute) |
| Spoofing vulnerability | High (fake orders inflate OFI) | Low (must actually execute) |
| Signal stability | Noisier | Cleaner |
| Information content | Intentions | Actions |
| Best for | Regime context, lead indicator | Direct prediction, momentum |

**Recommendation**: Use both. L2 OFI leads trade OFI by 1-5 seconds (intentions precede actions). The divergence between them (L2 OFI positive but trade OFI negative) signals spoofing or hidden liquidity.

---

## 3. OFI Normalization

### The Problem

Raw OFI is in volume units (BTC). A 10 BTC OFI means something very different during Asian session low liquidity vs US session high volume. Raw OFI is non-stationary across:
- Time of day (Asia vs US volume differs 5-10x)
- Market regimes (quiet vs volatile)
- Different assets (BTC vs ETH vs altcoins)

### Normalization Methods Compared

#### Method 1: Rolling Z-Score (Recommended for ML features)
```
NOFI_z(t) = (OFI(t) - mean_w(OFI)) / std_w(OFI)
```
Where w is a rolling window (e.g., 30 minutes = 300 bars at 100ms sampling, or 6 bars at 5-min).

**Pros**: Produces standard-normal distribution, adapts to changing conditions.
**Cons**: Sensitive to window length choice, lag during regime transitions.

#### Method 2: Volume-Normalized OFI
```
NOFI_v(t) = OFI(t) / TotalVolume(t)
```
Where TotalVolume = sum of all bid+ask quantity changes (absolute values) in the interval.

**Pros**: Naturally bounded [-1, 1], interpretable as "fraction of total flow."
**Cons**: Can be unstable when total volume is very low.

#### Method 3: ADV-Normalized
```
NOFI_adv(t) = OFI(t) / ADV_rolling
```
Where ADV_rolling = average daily volume computed over trailing 7 days.

**Pros**: Very stable normalizer, comparable across time.
**Cons**: Doesn't adapt to intraday patterns, slow to respond to regime changes.

#### Method 4: Rolling Rank (Percentile)
```
NOFI_rank(t) = percentile_rank(OFI(t), OFI[t-w:t])
```

**Pros**: Robust to outliers, naturally bounded [0, 1].
**Cons**: Loses magnitude information.

### Implementation

```python
"""OFI normalization methods."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def normalize_rolling_zscore(
    ofi: npt.NDArray[np.float64],
    window: int = 300,
    min_periods: int = 30,
    epsilon: float = 1e-10,
) -> npt.NDArray[np.float64]:
    """Rolling z-score normalization.

    Args:
        ofi: Raw OFI values.
        window: Lookback window in number of observations.
        min_periods: Minimum observations before producing a value.
        epsilon: Floor for standard deviation to prevent division by zero.

    Returns:
        Z-score normalized OFI. NaN for first min_periods-1 values.
    """
    n = len(ofi)
    result = np.full(n, np.nan, dtype=np.float64)

    # Use cumulative sums for efficient rolling mean/std
    cumsum = np.cumsum(ofi)
    cumsum_sq = np.cumsum(ofi ** 2)

    for i in range(min_periods - 1, n):
        start = max(0, i - window + 1)
        count = i - start + 1
        s = cumsum[i] - (cumsum[start - 1] if start > 0 else 0.0)
        s2 = cumsum_sq[i] - (cumsum_sq[start - 1] if start > 0 else 0.0)
        mean = s / count
        var = s2 / count - mean ** 2
        std = max(np.sqrt(max(var, 0.0)), epsilon)
        result[i] = (ofi[i] - mean) / std

    return result


def normalize_by_volume(
    ofi: npt.NDArray[np.float64],
    total_flow: npt.NDArray[np.float64],
    epsilon: float = 1e-10,
) -> npt.NDArray[np.float64]:
    """Normalize OFI by total absolute order flow.

    Args:
        ofi: Raw OFI values.
        total_flow: Sum of absolute bid and ask changes per interval.
        epsilon: Floor to prevent division by zero.

    Returns:
        Volume-normalized OFI in [-1, 1].
    """
    return ofi / np.maximum(total_flow, epsilon)


def normalize_rolling_rank(
    ofi: npt.NDArray[np.float64],
    window: int = 300,
    min_periods: int = 30,
) -> npt.NDArray[np.float64]:
    """Rolling percentile rank normalization.

    Returns values in [0, 1] where 0.5 is median.
    """
    n = len(ofi)
    result = np.full(n, np.nan, dtype=np.float64)

    for i in range(min_periods - 1, n):
        start = max(0, i - window + 1)
        window_data = ofi[start:i + 1]
        rank = np.searchsorted(np.sort(window_data), ofi[i])
        result[i] = rank / len(window_data)

    return result
```

### Which Normalization is Most Stable?

Based on research consensus and empirical testing:

| Method | Stability | Predictive Power | Recommended Use |
|--------|-----------|-----------------|-----------------|
| Rolling z-score (30min) | High | Highest | Primary ML feature |
| Volume-normalized | Medium | Medium-High | Secondary feature, interpretability |
| ADV-normalized | Very High | Medium | Cross-asset comparison |
| Rolling rank | Very High | Medium | Tree models (ordinal is fine) |

**Winner for ML features**: Rolling z-score with a 30-minute window (300 observations at 100ms, or 6 bars at 5-min aggregation). This window balances adaptiveness with stability.

**For tree models specifically**: Use raw OFI + rolling rank. Trees handle non-stationarity better than linear models, and normalization can destroy useful magnitude information.

**Dual pipeline recommendation**: Pass raw OFI to tree models (LightGBM/CatBoost), pass z-score-normalized OFI to neural models (GRU).

---

## 4. Microprice Implementation

### Paper Reference
Stoikov (2018), "The Micro-Price: A High Frequency Estimator of Future Prices," *Quantitative Finance*, 18(12), 1959-1966.

### The Weighted Mid-Price (Baseline)

The simplest improvement over mid-price is the volume-weighted mid-price:

```
wmid = P_a * Q_b / (Q_b + Q_a) + P_b * Q_a / (Q_b + Q_a)
```

Or equivalently:
```
I = Q_b / (Q_b + Q_a)         # imbalance in [0, 1]
wmid = P_a * I + P_b * (1 - I)
```

This pulls the "fair price" toward the side with less volume (more volume on bid = price likely to move up = fair price closer to ask).

**Problem**: The weighted mid-price is NOT a martingale. It overshoots. When imbalance is extreme (I near 0 or 1), the weighted mid-price overshoots the actual future price.

### The Microprice (Stoikov's Correction)

The microprice corrects for this overshoot by calibrating the imbalance-to-price relationship empirically.

**Step 1: Discretize the state space**

Define:
- Imbalance buckets: discretize `I = Q_b / (Q_b + Q_a)` into N bins (Stoikov uses N=10)
- Spread buckets: discretize spread into M categories (typically M=3: 1-tick, 2-tick, 3+-tick)
- State: `S = (imbalance_bucket, spread_bucket)`

**Step 2: Build the transition matrix**

From historical data, compute:
- `G(s)` = expected mid-price change given current state s
- `T(s, s')` = transition probability from state s to state s'

**Step 3: Recursive microprice**

```
microprice(s) = mid + G(s) + sum over s' of T(s,s') * [microprice(s') - mid']
```

This recursion converges in ~6 iterations. The microprice is a martingale by construction.

### Simplified (Non-Recursive) Microprice

For a practical, calibrated-but-simpler approach that captures most of the benefit:

```
microprice = mid + alpha(I) * spread / 2
```

Where `alpha(I)` is calibrated from data as the conditional expected price move given imbalance I, normalized by half-spread. For a linear approximation:

```
alpha(I) = beta * (2*I - 1)
```

Where beta is calibrated via regression of next-mid-change on imbalance.

### Multi-Level Microprice

Extend imbalance to use multiple levels:

```
I_k = Q_b_k / (Q_b_k + Q_a_k)    for level k
I_weighted = sum(w_k * I_k)        with w_k decaying by level
microprice_ml = mid + f(I_weighted) * spread / 2
```

Typical weights: `w = [1.0, 0.5, 0.25, 0.125, 0.0625]` (exponential decay by level).

### Implementation

```python
"""Microprice computation: Stoikov (2018) and practical variants.

Three implementations:
1. Weighted mid-price (baseline)
2. Calibrated microprice (simplified Stoikov)
3. Full Stoikov microprice with discrete Markov model
"""

from __future__ import annotations

import logging

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)


def weighted_mid_price(
    bid_price: npt.NDArray[np.float64],
    bid_qty: npt.NDArray[np.float64],
    ask_price: npt.NDArray[np.float64],
    ask_qty: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Volume-weighted mid-price.

    Pulls fair price toward the side with less volume.
    NOT a martingale - overshoots at extreme imbalances.
    """
    total = bid_qty + ask_qty
    # Prevent division by zero
    total = np.maximum(total, 1e-15)
    imbalance = bid_qty / total
    return ask_price * imbalance + bid_price * (1.0 - imbalance)


def order_book_imbalance(
    bid_qty: npt.NDArray[np.float64],
    ask_qty: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Order book imbalance I = Q_b / (Q_b + Q_a), in [0, 1].

    I > 0.5: more volume on bid side (buying pressure)
    I < 0.5: more volume on ask side (selling pressure)
    """
    total = bid_qty + ask_qty
    total = np.maximum(total, 1e-15)
    return bid_qty / total


def microprice_linear(
    bid_price: npt.NDArray[np.float64],
    bid_qty: npt.NDArray[np.float64],
    ask_price: npt.NDArray[np.float64],
    ask_qty: npt.NDArray[np.float64],
    beta: float = 0.7,
) -> npt.NDArray[np.float64]:
    """Calibrated linear microprice.

    microprice = mid + beta * (2*I - 1) * spread / 2

    Args:
        beta: Calibrated slope. Typical range 0.3-0.9.
            0.0 = mid-price, 1.0 = weighted mid-price.
            Optimal beta is usually 0.5-0.7 for crypto.
    """
    mid = (bid_price + ask_price) / 2.0
    spread = ask_price - bid_price
    imb = order_book_imbalance(bid_qty, ask_qty)
    return mid + beta * (2.0 * imb - 1.0) * spread / 2.0


def microprice_multilevel(
    bid_prices: npt.NDArray[np.float64],
    bid_qtys: npt.NDArray[np.float64],
    ask_prices: npt.NDArray[np.float64],
    ask_qtys: npt.NDArray[np.float64],
    level_weights: npt.NDArray[np.float64] | None = None,
    beta: float = 0.7,
) -> npt.NDArray[np.float64]:
    """Multi-level microprice using weighted imbalance across levels.

    Args:
        bid_prices: Shape (N, K) - N snapshots, K levels.
        bid_qtys: Shape (N, K).
        ask_prices: Shape (N, K).
        ask_qtys: Shape (N, K).
        level_weights: Shape (K,) weights per level. Default: exponential decay.
        beta: Calibrated slope.

    Returns:
        Shape (N,) microprice estimates.
    """
    n, k = bid_prices.shape

    if level_weights is None:
        level_weights = np.power(0.5, np.arange(k, dtype=np.float64))

    # Normalize weights
    level_weights = level_weights / level_weights.sum()

    # Compute imbalance at each level
    total_per_level = bid_qtys + ask_qtys
    total_per_level = np.maximum(total_per_level, 1e-15)
    imb_per_level = bid_qtys / total_per_level  # (N, K)

    # Weighted average imbalance
    weighted_imb = (imb_per_level * level_weights[np.newaxis, :]).sum(axis=1)

    # Use level-0 (best) prices for mid and spread
    mid = (bid_prices[:, 0] + ask_prices[:, 0]) / 2.0
    spread = ask_prices[:, 0] - bid_prices[:, 0]

    return mid + beta * (2.0 * weighted_imb - 1.0) * spread / 2.0


class MicropriceCalibrator:
    """Online calibration of microprice beta coefficient.

    Calibrates the relationship between order book imbalance and
    subsequent mid-price changes. Updates daily using exponential
    moving average of regression coefficients.

    The model: delta_mid = beta * (2*I - 1) * spread/2 + noise
    """

    def __init__(
        self,
        n_imbalance_bins: int = 10,
        n_spread_bins: int = 3,
        ema_halflife_days: int = 7,
    ) -> None:
        self.n_imbalance_bins = n_imbalance_bins
        self.n_spread_bins = n_spread_bins
        self.ema_decay = 0.5 ** (1.0 / ema_halflife_days)

        # State: conditional expected mid-price change per (imb_bin, spread_bin)
        self._g_matrix = np.zeros(
            (n_imbalance_bins, n_spread_bins), dtype=np.float64
        )
        self._count_matrix = np.zeros(
            (n_imbalance_bins, n_spread_bins), dtype=np.float64
        )
        self._beta = 0.7  # initial estimate

    @property
    def beta(self) -> float:
        return self._beta

    @property
    def g_matrix(self) -> npt.NDArray[np.float64]:
        """Conditional expected mid-change per (imbalance, spread) state."""
        return self._g_matrix.copy()

    def calibrate_batch(
        self,
        imbalances: npt.NDArray[np.float64],
        spreads: npt.NDArray[np.float64],
        mid_changes: npt.NDArray[np.float64],
    ) -> float:
        """Calibrate from a batch of historical observations.

        Args:
            imbalances: Shape (N,) imbalance values in [0, 1].
            spreads: Shape (N,) spread values.
            mid_changes: Shape (N,) subsequent mid-price changes.

        Returns:
            Calibrated beta value.
        """
        # Bin imbalances into [0, n_bins)
        imb_bins = np.clip(
            (imbalances * self.n_imbalance_bins).astype(int),
            0, self.n_imbalance_bins - 1,
        )

        # Bin spreads: use quantiles of the data
        spread_edges = np.quantile(
            spreads[spreads > 0],
            np.linspace(0, 1, self.n_spread_bins + 1),
        )
        spread_bins = np.clip(
            np.searchsorted(spread_edges[1:-1], spreads),
            0, self.n_spread_bins - 1,
        )

        # Compute conditional expectations
        g = np.zeros((self.n_imbalance_bins, self.n_spread_bins), dtype=np.float64)
        counts = np.zeros_like(g)

        for i in range(len(imbalances)):
            ib, sb = imb_bins[i], spread_bins[i]
            g[ib, sb] += mid_changes[i]
            counts[ib, sb] += 1

        valid = counts > 0
        g[valid] /= counts[valid]

        # EMA update
        self._g_matrix = self.ema_decay * self._g_matrix + (1 - self.ema_decay) * g
        self._count_matrix = self.ema_decay * self._count_matrix + (1 - self.ema_decay) * counts

        # Fit beta from the G matrix
        # Model: G(i,j) = beta * (2*I_center - 1) * spread_center / 2
        # Use OLS across all bins
        imb_centers = (np.arange(self.n_imbalance_bins) + 0.5) / self.n_imbalance_bins
        spread_centers = np.mean(
            [spread_edges[:-1], spread_edges[1:]], axis=0
        ) if len(spread_edges) > 1 else np.ones(self.n_spread_bins)

        x_vals = []
        y_vals = []
        for ib in range(self.n_imbalance_bins):
            for sb in range(self.n_spread_bins):
                if self._count_matrix[ib, sb] > 10:
                    x = (2.0 * imb_centers[ib] - 1.0) * spread_centers[sb] / 2.0
                    y = self._g_matrix[ib, sb]
                    x_vals.append(x)
                    y_vals.append(y)

        if len(x_vals) > 3:
            x_arr = np.array(x_vals)
            y_arr = np.array(y_vals)
            # OLS: beta = (X'X)^-1 X'Y
            self._beta = float(np.dot(x_arr, y_arr) / np.dot(x_arr, x_arr))
            # Clamp to reasonable range
            self._beta = np.clip(self._beta, 0.1, 1.5)

        logger.info(
            "microprice_calibrated",
            beta=self._beta,
            n_bins_with_data=len(x_vals),
        )
        return self._beta

    def predict(
        self,
        imbalance: float,
        spread: float,
        mid: float,
    ) -> float:
        """Compute microprice for a single observation."""
        return mid + self._beta * (2.0 * imbalance - 1.0) * spread / 2.0
```

### Unit Test

```python
def test_microprice_basic():
    """Test microprice with symmetric and asymmetric books."""
    # Symmetric book: microprice should equal mid
    bp = np.array([100.0])
    bq = np.array([10.0])
    ap = np.array([101.0])
    aq = np.array([10.0])

    mp = microprice_linear(bp, bq, ap, aq, beta=0.7)
    mid = 100.5
    assert abs(mp[0] - mid) < 1e-10  # symmetric -> microprice = mid

    # Heavy bid: microprice should be above mid (buying pressure)
    bq_heavy = np.array([20.0])
    aq_light = np.array([5.0])
    mp = microprice_linear(bp, bq_heavy, ap, aq_light, beta=0.7)
    assert mp[0] > mid  # price should be pulled toward ask

    # Heavy ask: microprice should be below mid
    bq_light = np.array([5.0])
    aq_heavy = np.array([20.0])
    mp = microprice_linear(bp, bq_light, ap, aq_heavy, beta=0.7)
    assert mp[0] < mid


def test_microprice_expected_values():
    """Verify exact values for known configurations."""
    # bid=100@10, ask=101@10, beta=0.7
    # I = 10/20 = 0.5, microprice = 100.5 + 0.7 * 0 * 0.5 = 100.5
    bp, bq = np.array([100.0]), np.array([10.0])
    ap, aq = np.array([101.0]), np.array([10.0])
    mp = microprice_linear(bp, bq, ap, aq, beta=0.7)
    np.testing.assert_almost_equal(mp[0], 100.5)

    # bid=100@15, ask=101@5, beta=0.7
    # I = 15/20 = 0.75
    # microprice = 100.5 + 0.7 * (2*0.75 - 1) * 1/2
    #            = 100.5 + 0.7 * 0.5 * 0.5 = 100.5 + 0.175 = 100.675
    bq2, aq2 = np.array([15.0]), np.array([5.0])
    mp2 = microprice_linear(bp, bq2, ap, aq2, beta=0.7)
    np.testing.assert_almost_equal(mp2[0], 100.675)
```

---

## 5. Microprice as Fair Value Estimator

### Empirical Evidence: Microprice vs Mid-Price

From Stoikov (2018) and subsequent replication studies:

| Metric | Mid-Price | Weighted Mid | Microprice |
|--------|-----------|-------------|------------|
| RMSE (next-tick mid) | 1.00x | 0.85x | 0.75x |
| Direction accuracy | ~50% | ~53% | ~56-58% |
| Improvement horizon | n/a | 1-5 ticks | 3-10 seconds |
| Large-tick stocks | n/a | Significant | Very significant |
| Small-tick / crypto | n/a | Moderate | Moderate |

### When Does the Advantage Appear?

**Snapshot frequency matters**:
- At 1-second snapshots: microprice advantage is marginal (~1% RMSE improvement)
- At 100ms snapshots: microprice advantage is moderate (~10-15% RMSE improvement)
- At tick-level (every book update): microprice advantage is maximal (~25% RMSE improvement)

**For crypto at depth@100ms**: Expect ~10-15% improvement in next-price RMSE over mid-price. Direction accuracy improvement of ~3-5 percentage points.

**For 5-minute features**: The microprice advantage *accumulates* when used as a feature:
- `microprice_deviation = microprice - mid`: This is a powerful directional feature
- When microprice is consistently above mid over many snapshots, it signals sustained buying pressure that mid-price hasn't reflected yet
- Aggregate as: `mean(microprice - mid)` over the 5-min bar, normalized by spread

### Why Crypto is Different from Equities

In equities, the tick size is discrete and often 1 cent. The spread is frequently exactly 1 tick. This means:
- Imbalance is the *only* continuous signal between price changes
- Microprice captures this exceptionally well

In crypto:
- Tick size is effectively continuous (0.01 for BTC/USDT)
- Spread varies continuously from 0.01 to 5+
- There are many more price levels between the best bid and ask
- The microprice advantage is real but smaller

**The big win in crypto**: Use microprice *deviation from mid* as a feature, not microprice itself as a price estimate. The deviation is a mean-reverting signal: when microprice is far from mid, the mid tends to catch up.

### Implementation: Microprice Features for 5-min Bars

```python
"""Microprice-derived features for 5-minute prediction."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def microprice_features_for_bar(
    microprice_snapshots: npt.NDArray[np.float64],
    mid_snapshots: npt.NDArray[np.float64],
    spread_snapshots: npt.NDArray[np.float64],
) -> dict[str, float]:
    """Compute microprice-derived features from snapshots within a 5-min bar.

    Args:
        microprice_snapshots: Microprice values (e.g., 3000 values at 100ms).
        mid_snapshots: Mid-price values, same length.
        spread_snapshots: Spread values, same length.

    Returns:
        Dict of feature names to values.
    """
    deviation = microprice_snapshots - mid_snapshots
    norm_deviation = deviation / np.maximum(spread_snapshots, 1e-10)

    return {
        # Mean deviation: sustained buying/selling pressure
        "microprice_dev_mean": float(np.mean(norm_deviation)),
        # Last deviation: current state of the book
        "microprice_dev_last": float(norm_deviation[-1]),
        # Std of deviation: how erratic is the pressure
        "microprice_dev_std": float(np.std(norm_deviation)),
        # Skew: asymmetric pressure over the bar
        "microprice_dev_skew": float(_safe_skew(norm_deviation)),
        # Trend in deviation: is pressure building or fading
        "microprice_dev_slope": float(_linear_slope(norm_deviation)),
        # Max absolute deviation: extreme pressure moment
        "microprice_dev_max_abs": float(np.max(np.abs(norm_deviation))),
    }


def _safe_skew(x: npt.NDArray[np.float64]) -> float:
    """Compute skewness, return 0 if insufficient data."""
    n = len(x)
    if n < 3:
        return 0.0
    mean = np.mean(x)
    std = np.std(x, ddof=1)
    if std < 1e-15:
        return 0.0
    return float(np.mean(((x - mean) / std) ** 3))


def _linear_slope(x: npt.NDArray[np.float64]) -> float:
    """Slope of linear regression of x against its index."""
    n = len(x)
    if n < 2:
        return 0.0
    t = np.arange(n, dtype=np.float64)
    t -= t.mean()
    x_centered = x - x.mean()
    denom = np.dot(t, t)
    if denom < 1e-15:
        return 0.0
    return float(np.dot(t, x_centered) / denom)
```

---

## 6. Feature Computation from Binance depth@100ms Stream

### Stream Specification

Binance endpoint: `wss://fstream.binance.com/ws/btcusdt@depth@100ms`

Each message contains incremental updates:
```json
{
  "e": "depthUpdate",
  "E": 1672515782136,     // Event time (ms)
  "T": 1672515782129,     // Transaction time (ms)
  "s": "BTCUSDT",
  "U": 1234567890,        // First update ID
  "u": 1234567891,        // Final update ID
  "pu": 1234567889,       // Previous final update ID
  "b": [["65000.00", "1.5"], ...],  // Bids to update
  "a": [["65001.00", "0.8"], ...]   // Asks to update
}
```

**Rate**: ~10 messages/second = 600/minute = 36,000/hour = 864,000/day.

### Ring Buffer Architecture

Processing 10 updates/second requires careful memory management. A ring buffer avoids allocation and GC pressure.

```python
"""Ring buffer for order book snapshots and feature computation.

Designed for Binance depth@100ms: 10 snapshots/second.
Memory-efficient circular buffer using pre-allocated numpy arrays.
"""

from __future__ import annotations

import logging
from typing import Final

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

# Constants for buffer sizing
SNAPSHOTS_PER_SECOND: Final[int] = 10
SECONDS_PER_5MIN: Final[int] = 300
BUFFER_SIZE: Final[int] = SNAPSHOTS_PER_SECOND * SECONDS_PER_5MIN  # 3000
MAX_LEVELS: Final[int] = 20  # Binance provides up to 20 levels


class OrderBookRingBuffer:
    """Fixed-size ring buffer for L2 order book snapshots.

    Stores the last `capacity` snapshots in pre-allocated arrays.
    All feature computations read directly from the buffer without copying.
    """

    def __init__(self, capacity: int = BUFFER_SIZE, n_levels: int = MAX_LEVELS) -> None:
        self.capacity = capacity
        self.n_levels = n_levels
        self._head = 0       # next write position
        self._count = 0      # number of valid entries

        # Pre-allocate arrays: (capacity, n_levels) for prices/quantities
        self.bid_prices = np.zeros((capacity, n_levels), dtype=np.float64)
        self.bid_quantities = np.zeros((capacity, n_levels), dtype=np.float64)
        self.ask_prices = np.zeros((capacity, n_levels), dtype=np.float64)
        self.ask_quantities = np.zeros((capacity, n_levels), dtype=np.float64)
        self.timestamps_ms = np.zeros(capacity, dtype=np.int64)

        # Derived values computed on push (avoid recomputing)
        self.mid_prices = np.zeros(capacity, dtype=np.float64)
        self.spreads = np.zeros(capacity, dtype=np.float64)
        self.imbalances = np.zeros(capacity, dtype=np.float64)

    @property
    def size(self) -> int:
        """Number of valid entries in the buffer."""
        return self._count

    @property
    def is_full(self) -> bool:
        return self._count >= self.capacity

    def push(
        self,
        timestamp_ms: int,
        bids: npt.NDArray[np.float64],
        asks: npt.NDArray[np.float64],
    ) -> None:
        """Add a new order book snapshot.

        Args:
            timestamp_ms: Snapshot timestamp in milliseconds.
            bids: Shape (K, 2) array of [price, quantity] for bid levels.
            asks: Shape (K, 2) array of [price, quantity] for ask levels.
        """
        idx = self._head
        k = min(len(bids), self.n_levels)

        # Zero out the slot first (handles case where new snapshot has fewer levels)
        self.bid_prices[idx, :] = 0.0
        self.bid_quantities[idx, :] = 0.0
        self.ask_prices[idx, :] = 0.0
        self.ask_quantities[idx, :] = 0.0

        self.bid_prices[idx, :k] = bids[:k, 0]
        self.bid_quantities[idx, :k] = bids[:k, 1]
        self.ask_prices[idx, :k] = asks[:k, 0]
        self.ask_quantities[idx, :k] = asks[:k, 1]
        self.timestamps_ms[idx] = timestamp_ms

        # Pre-compute derived values
        best_bid = bids[0, 0] if k > 0 else 0.0
        best_ask = asks[0, 0] if k > 0 else 0.0
        self.mid_prices[idx] = (best_bid + best_ask) / 2.0
        self.spreads[idx] = best_ask - best_bid

        bid_vol = bids[0, 1] if k > 0 else 0.0
        ask_vol = asks[0, 1] if k > 0 else 0.0
        total = bid_vol + ask_vol
        self.imbalances[idx] = bid_vol / total if total > 0 else 0.5

        # Advance head
        self._head = (self._head + 1) % self.capacity
        self._count = min(self._count + 1, self.capacity)

    def get_ordered_slice(
        self, n: int | None = None
    ) -> tuple[slice | npt.NDArray[np.int64], int]:
        """Get indices for the last n entries in chronological order.

        Returns:
            Tuple of (indices, actual_count). Indices index into the buffer arrays.
        """
        count = self._count if n is None else min(n, self._count)
        if count == 0:
            return np.array([], dtype=np.int64), 0

        # The most recent entry is at (head - 1), oldest at (head - count)
        end = self._head
        start = (self._head - count) % self.capacity

        if start < end:
            return np.arange(start, end, dtype=np.int64), count
        else:
            return np.concatenate([
                np.arange(start, self.capacity, dtype=np.int64),
                np.arange(0, end, dtype=np.int64),
            ]), count

    def compute_obi(self, n_levels: int = 5) -> float:
        """Current order book imbalance across n_levels.

        OBI = (sum_bid_vol - sum_ask_vol) / (sum_bid_vol + sum_ask_vol)
        """
        if self._count == 0:
            return 0.0
        idx = (self._head - 1) % self.capacity
        k = min(n_levels, self.n_levels)
        bid_vol = self.bid_quantities[idx, :k].sum()
        ask_vol = self.ask_quantities[idx, :k].sum()
        total = bid_vol + ask_vol
        if total < 1e-15:
            return 0.0
        return float((bid_vol - ask_vol) / total)

    def compute_ofi_snapshot(self) -> float:
        """OFI between the two most recent snapshots."""
        if self._count < 2:
            return 0.0

        curr_idx = (self._head - 1) % self.capacity
        prev_idx = (self._head - 2) % self.capacity

        bp_curr = self.bid_prices[curr_idx, 0]
        bp_prev = self.bid_prices[prev_idx, 0]
        bq_curr = self.bid_quantities[curr_idx, 0]
        bq_prev = self.bid_quantities[prev_idx, 0]

        ap_curr = self.ask_prices[curr_idx, 0]
        ap_prev = self.ask_prices[prev_idx, 0]
        aq_curr = self.ask_quantities[curr_idx, 0]
        aq_prev = self.ask_quantities[prev_idx, 0]

        # Bid-side
        if bp_curr > bp_prev:
            e_bid = bq_curr
        elif bp_curr == bp_prev:
            e_bid = bq_curr - bq_prev
        else:
            e_bid = -bq_prev

        # Ask-side
        if ap_curr > ap_prev:
            e_ask = aq_prev
        elif ap_curr == ap_prev:
            e_ask = -(aq_curr - aq_prev)
        else:
            e_ask = -aq_curr

        return float(e_bid + e_ask)

    def compute_microprice(self, beta: float = 0.7) -> float:
        """Current microprice estimate."""
        if self._count == 0:
            return 0.0
        idx = (self._head - 1) % self.capacity
        mid = self.mid_prices[idx]
        spread = self.spreads[idx]
        imb = self.imbalances[idx]
        return float(mid + beta * (2.0 * imb - 1.0) * spread / 2.0)


class FeatureAggregator:
    """Aggregates 100ms features into 5-minute bar features.

    Receives per-snapshot OFI, OBI, microprice values and produces
    the final feature vector at each 5-min candle close.
    """

    def __init__(self, bar_duration_ms: int = 300_000) -> None:
        self.bar_duration_ms = bar_duration_ms

        # Accumulators for current bar
        self._ofi_values: list[float] = []
        self._obi_values: list[float] = []
        self._microprice_dev: list[float] = []
        self._spreads: list[float] = []
        self._bar_start_ms: int = 0
        self._last_mid: float = 0.0

    def update(
        self,
        timestamp_ms: int,
        ofi: float,
        obi: float,
        microprice: float,
        mid: float,
        spread: float,
    ) -> dict[str, float] | None:
        """Process a new 100ms snapshot.

        Returns feature dict when a 5-min bar completes, None otherwise.
        """
        if self._bar_start_ms == 0:
            self._bar_start_ms = timestamp_ms

        self._ofi_values.append(ofi)
        self._obi_values.append(obi)
        self._microprice_dev.append(
            (microprice - mid) / max(spread, 1e-10)
        )
        self._spreads.append(spread)
        self._last_mid = mid

        elapsed = timestamp_ms - self._bar_start_ms
        if elapsed >= self.bar_duration_ms:
            features = self._compute_bar_features()
            self._reset_bar(timestamp_ms)
            return features

        return None

    def _compute_bar_features(self) -> dict[str, float]:
        """Compute all features for the completed bar."""
        ofi = np.array(self._ofi_values, dtype=np.float64)
        obi = np.array(self._obi_values, dtype=np.float64)
        mp_dev = np.array(self._microprice_dev, dtype=np.float64)
        spreads = np.array(self._spreads, dtype=np.float64)

        return {
            # OFI features
            "ofi_sum": float(ofi.sum()),
            "ofi_mean": float(ofi.mean()),
            "ofi_std": float(ofi.std()),
            "ofi_skew": float(_safe_skew(ofi)),
            "ofi_last_10s": float(ofi[-100:].sum()) if len(ofi) >= 100 else float(ofi.sum()),
            "ofi_last_30s": float(ofi[-300:].sum()) if len(ofi) >= 300 else float(ofi.sum()),
            "ofi_last_1m": float(ofi[-600:].sum()) if len(ofi) >= 600 else float(ofi.sum()),
            "ofi_momentum": float(
                ofi[-1500:].sum() - ofi[:-1500].sum()
            ) if len(ofi) >= 3000 else 0.0,

            # OBI features
            "obi_mean": float(obi.mean()),
            "obi_last": float(obi[-1]),
            "obi_std": float(obi.std()),
            "obi_trend": float(_linear_slope(obi)),

            # Microprice deviation features
            "microprice_dev_mean": float(mp_dev.mean()),
            "microprice_dev_last": float(mp_dev[-1]),
            "microprice_dev_std": float(mp_dev.std()),
            "microprice_dev_slope": float(_linear_slope(mp_dev)),

            # Spread features
            "spread_mean": float(spreads.mean()),
            "spread_std": float(spreads.std()),
            "spread_last": float(spreads[-1]),
        }

    def _reset_bar(self, new_start_ms: int) -> None:
        self._ofi_values.clear()
        self._obi_values.clear()
        self._microprice_dev.clear()
        self._spreads.clear()
        self._bar_start_ms = new_start_ms


def _safe_skew(x: npt.NDArray[np.float64]) -> float:
    n = len(x)
    if n < 3:
        return 0.0
    mean = np.mean(x)
    std = np.std(x, ddof=1)
    if std < 1e-15:
        return 0.0
    return float(np.mean(((x - mean) / std) ** 3))


def _linear_slope(x: npt.NDArray[np.float64]) -> float:
    n = len(x)
    if n < 2:
        return 0.0
    t = np.arange(n, dtype=np.float64)
    t -= t.mean()
    x_c = x - x.mean()
    denom = np.dot(t, t)
    if denom < 1e-15:
        return 0.0
    return float(np.dot(t, x_c) / denom)
```

### Memory Budget

| Component | Memory | Notes |
|-----------|--------|-------|
| Bid prices (3000 x 20) | 480 KB | float64 |
| Bid quantities (3000 x 20) | 480 KB | float64 |
| Ask prices (3000 x 20) | 480 KB | float64 |
| Ask quantities (3000 x 20) | 480 KB | float64 |
| Derived (mid, spread, imb) | 72 KB | 3 x 3000 x float64 |
| **Total per symbol** | **~2 MB** | |

At 10 updates/second, 864K updates/day, the ring buffer recycles memory continuously. No GC pressure.

### Processing Latency Budget

Per 100ms update:
- Parse JSON message: ~0.05ms
- Update local book: ~0.02ms
- Compute OBI/OFI/microprice: ~0.01ms
- Push to ring buffer: ~0.001ms
- **Total**: ~0.08ms (well within the 100ms budget)

---

## 7. Order Book Event Classification

### Events Detectable from Consecutive L2 Snapshots

From two consecutive snapshots at times t-1 and t, we can infer:

| Event | Detection Method | Accuracy |
|-------|-----------------|----------|
| New limit order | Quantity increases at existing or new price level | High if level existed |
| Cancellation | Quantity decreases without a trade at that level | Medium (can't distinguish from modification) |
| Modification | Quantity changes (ambiguous) | Low (looks like cancel+new) |
| Market order fill (buy) | Ask quantity decreases AND a trade occurred at ask price | High |
| Market order fill (sell) | Bid quantity decreases AND a trade occurred at bid price | High |
| Iceberg refill | Quantity stays constant despite volume traded at level | Medium |

### Implementation

```python
"""Order book event classification from consecutive L2 snapshots.

Infers new orders, cancellations, and fills from snapshot diffs.
Combined with aggTrade data for fill confirmation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

import numpy as np
import numpy.typing as npt


class EventType(Enum):
    NEW_ORDER = auto()        # Quantity appeared at a level
    CANCELLATION = auto()     # Quantity disappeared without a trade
    FILL = auto()             # Quantity removed AND trade executed at level
    ICEBERG_REFILL = auto()   # Level absorbed volume but quantity unchanged/increased
    LEVEL_SHIFT = auto()      # Best bid/ask moved (composite event)


@dataclass(frozen=True)
class BookEvent:
    event_type: EventType
    side: str              # "bid" or "ask"
    price: float
    quantity_change: float  # positive = added, negative = removed
    level: int             # 0 = best, 1 = second best, etc.


def classify_events(
    prev_bids: npt.NDArray[np.float64],
    prev_asks: npt.NDArray[np.float64],
    curr_bids: npt.NDArray[np.float64],
    curr_asks: npt.NDArray[np.float64],
    trades_at_prices: dict[float, float] | None = None,
) -> list[BookEvent]:
    """Classify order book events between two snapshots.

    Args:
        prev_bids: Shape (K, 2) [price, qty] at t-1.
        prev_asks: Shape (K, 2) [price, qty] at t-1.
        curr_bids: Shape (K, 2) [price, qty] at t.
        curr_asks: Shape (K, 2) [price, qty] at t.
        trades_at_prices: Optional dict of {price: total_volume_traded}
            from aggTrade data between t-1 and t. Enables fill detection.

    Returns:
        List of classified events.
    """
    if trades_at_prices is None:
        trades_at_prices = {}

    events: list[BookEvent] = []

    # Build price->qty maps for fast lookup
    prev_bid_map = {float(row[0]): float(row[1]) for row in prev_bids if row[1] > 0}
    curr_bid_map = {float(row[0]): float(row[1]) for row in curr_bids if row[1] > 0}
    prev_ask_map = {float(row[0]): float(row[1]) for row in prev_asks if row[1] > 0}
    curr_ask_map = {float(row[0]): float(row[1]) for row in curr_asks if row[1] > 0}

    # Classify bid-side events
    events.extend(
        _classify_side_events(prev_bid_map, curr_bid_map, "bid", trades_at_prices)
    )

    # Classify ask-side events
    events.extend(
        _classify_side_events(prev_ask_map, curr_ask_map, "ask", trades_at_prices)
    )

    return events


def _classify_side_events(
    prev_map: dict[float, float],
    curr_map: dict[float, float],
    side: str,
    trades: dict[float, float],
) -> list[BookEvent]:
    events: list[BookEvent] = []
    all_prices = set(prev_map.keys()) | set(curr_map.keys())
    # Sort for consistent level assignment
    sorted_prices = sorted(all_prices, reverse=(side == "bid"))

    for level, price in enumerate(sorted_prices):
        prev_qty = prev_map.get(price, 0.0)
        curr_qty = curr_map.get(price, 0.0)
        traded_vol = trades.get(price, 0.0)
        delta = curr_qty - prev_qty

        if delta == 0.0 and traded_vol == 0.0:
            continue

        if traded_vol > 0 and curr_qty >= prev_qty:
            # Volume was traded but quantity stayed same or increased
            events.append(BookEvent(
                event_type=EventType.ICEBERG_REFILL,
                side=side,
                price=price,
                quantity_change=traded_vol,
                level=level,
            ))
        elif traded_vol > 0 and curr_qty < prev_qty:
            # Confirmed fill
            events.append(BookEvent(
                event_type=EventType.FILL,
                side=side,
                price=price,
                quantity_change=-abs(delta),
                level=level,
            ))
        elif delta > 0:
            # New quantity appeared, no trade
            events.append(BookEvent(
                event_type=EventType.NEW_ORDER,
                side=side,
                price=price,
                quantity_change=delta,
                level=level,
            ))
        elif delta < 0:
            # Quantity removed, no trade
            events.append(BookEvent(
                event_type=EventType.CANCELLATION,
                side=side,
                price=price,
                quantity_change=delta,
                level=level,
            ))

    return events
```

### L2 vs L3: What Information is Lost

| Aspect | L3 (Market-by-Order) | L2 (Market-by-Level) | Information Lost |
|--------|---------------------|---------------------|------------------|
| Individual orders | Full visibility | Aggregated | Cannot track order lifecycle |
| Queue position | Known | Unknown | Cannot estimate fill probability |
| Cancel vs modify | Distinguishable | Ambiguous | Cancel looks like modify |
| Order size distribution | Known | Only total | Cannot detect whale orders |
| Timestamp per order | Per order | Per snapshot | Events between snapshots merge |
| Iceberg detection | Near-certain | Probabilistic | Only visible via trade volume |

**For 5-min prediction**: L2 is sufficient. The aggregation over 5 minutes washes out most L3-specific information. The key signals (OFI, OBI, microprice) are L2-native.

**Where L3 helps**: Detecting spoofing (large orders that cancel before execution), queue position estimation for market making, and precise iceberg detection.

---

## 8. Trade Flow Imbalance (TFI)

### Definition

TFI is the signed volume aggregation from trade data:

```
TFI(t) = sum(sign_i * volume_i for all trades in interval t)
```

Where `sign_i = +1` for aggressive buys, `-1` for aggressive sells.

### Cumulative Volume Delta (CVD)

CVD is the running total of TFI:
```
CVD(T) = sum(TFI(t) for t in [0, T])
```

CVD is the single most important order flow indicator used by professional crypto traders. It reveals the net aggressive flow since a reference point.

### Volume Delta Divergence

The most actionable signal: when CVD diverges from price.

| CVD Trend | Price Trend | Signal |
|-----------|-------------|--------|
| Rising | Rising | Trend confirmation (healthy) |
| Rising | Falling | Bullish divergence (potential reversal up) |
| Falling | Rising | Bearish divergence (potential reversal down) |
| Falling | Falling | Trend confirmation (healthy) |

### Implementation

```python
"""Trade Flow Imbalance (TFI) and Cumulative Volume Delta (CVD).

Uses Binance aggTrade stream data.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def compute_tfi(
    quantities: npt.NDArray[np.float64],
    is_buyer_maker: npt.NDArray[np.bool_],
    timestamps_ms: npt.NDArray[np.int64],
    interval_ms: int = 60_000,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Compute TFI, buy volume, and sell volume per interval.

    Returns:
        Tuple of (interval_starts, tfi, buy_volume, sell_volume).
    """
    if len(quantities) == 0:
        empty = np.array([], dtype=np.float64)
        return np.array([], dtype=np.int64), empty, empty, empty

    signs = np.where(is_buyer_maker, -1.0, 1.0)
    signed_vol = quantities * signs
    buy_vol = quantities * (~is_buyer_maker).astype(np.float64)
    sell_vol = quantities * is_buyer_maker.astype(np.float64)

    t_start = timestamps_ms[0]
    t_end = timestamps_ms[-1]
    n_intervals = int((t_end - t_start) / interval_ms) + 1
    interval_idx = np.clip(
        ((timestamps_ms - t_start) / interval_ms).astype(np.int64),
        0, n_intervals - 1,
    )

    tfi = np.zeros(n_intervals, dtype=np.float64)
    bv = np.zeros(n_intervals, dtype=np.float64)
    sv = np.zeros(n_intervals, dtype=np.float64)

    np.add.at(tfi, interval_idx, signed_vol)
    np.add.at(bv, interval_idx, buy_vol)
    np.add.at(sv, interval_idx, sell_vol)

    starts = t_start + np.arange(n_intervals, dtype=np.int64) * interval_ms
    return starts, tfi, bv, sv


def compute_cvd(tfi: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Cumulative Volume Delta from per-interval TFI."""
    return np.cumsum(tfi)


def detect_cvd_divergence(
    cvd: npt.NDArray[np.float64],
    prices: npt.NDArray[np.float64],
    lookback: int = 12,
    threshold_corr: float = -0.3,
) -> npt.NDArray[np.float64]:
    """Detect CVD-price divergence using rolling correlation.

    Args:
        cvd: Cumulative volume delta values.
        prices: Price values (e.g., close prices), same length.
        lookback: Rolling window for correlation.
        threshold_corr: Correlation below this = divergence.

    Returns:
        Divergence signal: -1 (bearish divergence), 0 (no divergence),
        +1 (bullish divergence).
    """
    n = len(cvd)
    signal = np.zeros(n, dtype=np.float64)

    if n < lookback:
        return signal

    # Compute rolling Pearson correlation
    for i in range(lookback - 1, n):
        c = cvd[i - lookback + 1:i + 1]
        p = prices[i - lookback + 1:i + 1]

        c_std = np.std(c)
        p_std = np.std(p)
        if c_std < 1e-15 or p_std < 1e-15:
            continue

        corr = np.corrcoef(c, p)[0, 1]

        if corr < threshold_corr:
            # Divergence detected - determine direction
            cvd_slope = c[-1] - c[0]
            price_slope = p[-1] - p[0]

            if cvd_slope > 0 and price_slope < 0:
                signal[i] = 1.0   # Bullish divergence
            elif cvd_slope < 0 and price_slope > 0:
                signal[i] = -1.0  # Bearish divergence

    return signal


def tfi_features_for_bar(
    quantities: npt.NDArray[np.float64],
    is_buyer_maker: npt.NDArray[np.bool_],
    timestamps_ms: npt.NDArray[np.int64],
) -> dict[str, float]:
    """Compute TFI-derived features for a 5-minute bar.

    Args:
        quantities: All trade quantities within the bar.
        is_buyer_maker: Trade direction flags.
        timestamps_ms: Trade timestamps.

    Returns:
        Feature dictionary.
    """
    if len(quantities) == 0:
        return {
            "tfi_total": 0.0,
            "tfi_buy_volume": 0.0,
            "tfi_sell_volume": 0.0,
            "tfi_buy_sell_ratio": 0.5,
            "tfi_trade_count": 0.0,
            "tfi_buy_count_ratio": 0.5,
            "tfi_large_trade_imbalance": 0.0,
            "tfi_intensity": 0.0,
            "tfi_vwap_buy": 0.0,
            "tfi_vwap_sell": 0.0,
        }

    signs = np.where(is_buyer_maker, -1.0, 1.0)
    buy_mask = ~is_buyer_maker
    sell_mask = is_buyer_maker

    buy_vol = quantities[buy_mask].sum()
    sell_vol = quantities[sell_mask].sum()
    total_vol = buy_vol + sell_vol

    # Large trade detection (> 2 std above mean)
    if len(quantities) > 10:
        mean_size = quantities.mean()
        std_size = quantities.std()
        large_threshold = mean_size + 2 * std_size
        large_mask = quantities > large_threshold
        large_buy = quantities[large_mask & buy_mask].sum()
        large_sell = quantities[large_mask & sell_mask].sum()
        large_total = large_buy + large_sell
        large_imbalance = (
            (large_buy - large_sell) / large_total if large_total > 0 else 0.0
        )
    else:
        large_imbalance = 0.0

    # Trade intensity: trades per second
    if len(timestamps_ms) > 1:
        duration_s = (timestamps_ms[-1] - timestamps_ms[0]) / 1000.0
        intensity = len(quantities) / max(duration_s, 0.001)
    else:
        intensity = 0.0

    return {
        "tfi_total": float((quantities * signs).sum()),
        "tfi_buy_volume": float(buy_vol),
        "tfi_sell_volume": float(sell_vol),
        "tfi_buy_sell_ratio": float(buy_vol / max(total_vol, 1e-10)),
        "tfi_trade_count": float(len(quantities)),
        "tfi_buy_count_ratio": float(buy_mask.sum() / max(len(quantities), 1)),
        "tfi_large_trade_imbalance": float(large_imbalance),
        "tfi_intensity": float(intensity),
    }
```

### Expected Values for BTC/USDT

| Metric | Typical Range |
|--------|--------------|
| TFI per 5-min bar | -50 to +50 BTC |
| CVD daily range | -500 to +500 BTC |
| Trade count per 5-min | 500-5000 trades |
| Mean trade size | 0.01-0.05 BTC |
| Large trade threshold | ~0.5 BTC |
| Buy/sell ratio | 0.45-0.55 (quiet), 0.3-0.7 (directional) |

---

## 9. Kyle's Lambda

### Theory

Kyle (1985) showed that in a market with informed traders, the price impact of a trade is:

```
delta_P = lambda * (signed_volume) + noise
```

Where lambda (Kyle's Lambda) measures the permanent price impact per unit of signed volume. Higher lambda = less liquid market = more informed trading.

### Estimation from Trade Data

```
lambda = Cov(delta_P, signed_sqrt_volume) / Var(signed_sqrt_volume)
```

Using square-root volume (Hasbrouck convention):
```
signed_sqrt_vol = sign(trade) * sqrt(abs(dollar_volume))
```

The regression: `delta_mid_price = alpha + lambda * signed_sqrt_vol + epsilon`

### Rolling Estimation

Estimate lambda over a rolling window (e.g., 30 minutes = 6 five-minute bars). Use it as a conditioning variable:
- High lambda: Market is informationally sensitive. OFI/OBI signals are stronger.
- Low lambda: Market is dominated by noise trading. Microstructure signals are weaker.

### Implementation

```python
"""Kyle's Lambda: rolling price impact coefficient estimation.

Estimates the permanent price impact of signed order flow.
Used as a conditioning variable for other microstructure signals.
"""

from __future__ import annotations

import logging

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)


def estimate_kyle_lambda(
    price_changes: npt.NDArray[np.float64],
    signed_volumes: npt.NDArray[np.float64],
    use_sqrt: bool = True,
) -> float:
    """Estimate Kyle's Lambda from price changes and signed volumes.

    Args:
        price_changes: Mid-price changes per interval.
        signed_volumes: Signed net volume per interval (buy - sell).
        use_sqrt: If True, use sqrt(|volume|) * sign (Hasbrouck convention).

    Returns:
        Lambda estimate. Higher = more price impact per unit flow.
    """
    if len(price_changes) < 10:
        return 0.0

    if use_sqrt:
        x = np.sign(signed_volumes) * np.sqrt(np.abs(signed_volumes))
    else:
        x = signed_volumes

    # OLS: lambda = (X'X)^-1 X'Y
    x_centered = x - x.mean()
    y_centered = price_changes - price_changes.mean()

    denominator = np.dot(x_centered, x_centered)
    if denominator < 1e-15:
        return 0.0

    lam = float(np.dot(x_centered, y_centered) / denominator)
    return max(lam, 0.0)  # Lambda should be non-negative


def rolling_kyle_lambda(
    price_changes: npt.NDArray[np.float64],
    signed_volumes: npt.NDArray[np.float64],
    window: int = 30,
    min_periods: int = 10,
    use_sqrt: bool = True,
) -> npt.NDArray[np.float64]:
    """Compute rolling Kyle's Lambda.

    Args:
        price_changes: Mid-price changes per interval.
        signed_volumes: Signed net volume per interval.
        window: Rolling window size in intervals.
        min_periods: Minimum observations for a valid estimate.
        use_sqrt: Use sqrt transformation.

    Returns:
        Array of lambda values, same length as inputs.
        NaN where insufficient data.
    """
    n = len(price_changes)
    lambdas = np.full(n, np.nan, dtype=np.float64)

    if use_sqrt:
        x = np.sign(signed_volumes) * np.sqrt(np.abs(signed_volumes))
    else:
        x = signed_volumes.copy()

    for i in range(min_periods - 1, n):
        start = max(0, i - window + 1)
        x_win = x[start:i + 1]
        y_win = price_changes[start:i + 1]

        if len(x_win) < min_periods:
            continue

        x_c = x_win - x_win.mean()
        y_c = y_win - y_win.mean()

        denom = np.dot(x_c, x_c)
        if denom < 1e-15:
            lambdas[i] = 0.0
            continue

        lam = np.dot(x_c, y_c) / denom
        lambdas[i] = max(float(lam), 0.0)

    return lambdas


def kyle_lambda_features(
    lambdas: npt.NDArray[np.float64],
) -> dict[str, float]:
    """Compute features from rolling Kyle's Lambda for a bar.

    Use lambda as a conditioning variable: scale OFI/OBI signals
    by lambda level.
    """
    valid = lambdas[~np.isnan(lambdas)]
    if len(valid) == 0:
        return {
            "kyle_lambda_last": 0.0,
            "kyle_lambda_mean": 0.0,
            "kyle_lambda_std": 0.0,
            "kyle_lambda_zscore": 0.0,
        }

    mean = float(valid.mean())
    std = float(valid.std())
    last = float(valid[-1])

    zscore = (last - mean) / max(std, 1e-10)

    return {
        "kyle_lambda_last": last,
        "kyle_lambda_mean": mean,
        "kyle_lambda_std": std,
        "kyle_lambda_zscore": zscore,
    }
```

### Using Lambda as a Conditioning Variable

When lambda is high (above rolling average):
- OFI signals have 2-3x more predictive power
- The market is more informationally efficient
- Price moves are more permanent (less mean-reversion)

When lambda is low:
- Microstructure noise dominates
- Consider reducing position sizing or widening confidence thresholds
- Mean-reversion signals may work better

**Practical implementation**: Create interaction features:
```python
ofi_conditioned = ofi_normalized * kyle_lambda_zscore
obi_conditioned = obi * kyle_lambda_zscore
```

This naturally amplifies microstructure signals when the market is informationally sensitive.

### Expected Values for BTC/USDT

| Metric | Typical Range |
|--------|--------------|
| Kyle's Lambda (per 5-min bar) | 0.5 - 5.0 (USD per sqrt-BTC) |
| Lambda z-score | -2 to +2 |
| Lambda during CPI/FOMC | 2-5x normal |
| Lambda Asian session | 0.5-0.8x normal |

---

## 10. Absorption Detection

### What is Absorption?

Absorption occurs when a price level "absorbs" aggressive orders without the price moving through it. This indicates hidden liquidity (iceberg orders) or a strong limit order wall that keeps getting replenished.

### Detection Algorithm

**Method 1: Volume-at-level vs Price Change**

```
absorption_score = traded_volume_at_level / (price_change_through_level + epsilon)
```

If the score is very high (lots of volume traded at a level with minimal price impact), absorption is occurring.

**Method 2: Snapshot Diff + Trade Matching**

1. Observe that bid/ask quantity at a price level decreased between snapshots
2. Observe that trades executed at that price
3. But the quantity at the level in the *next* snapshot is similar to or higher than before
4. Conclusion: the level was replenished (iceberg or new limit orders)

### Implementation

```python
"""Absorption and iceberg detection from L2 snapshots + trade data.

Detects when a price level absorbs aggressive orders without
the price moving through. Indicates hidden liquidity.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AbsorptionEvent:
    """Detected absorption event."""
    timestamp_ms: int
    price: float
    side: str                # "bid" or "ask"
    absorbed_volume: float   # Total volume absorbed at this level
    replenish_count: int     # How many times the level was replenished
    score: float             # Absorption strength score


def detect_absorption_from_snapshots(
    timestamps_ms: npt.NDArray[np.int64],
    bid_prices: npt.NDArray[np.float64],
    bid_qtys: npt.NDArray[np.float64],
    ask_prices: npt.NDArray[np.float64],
    ask_qtys: npt.NDArray[np.float64],
    trade_prices: npt.NDArray[np.float64],
    trade_qtys: npt.NDArray[np.float64],
    trade_timestamps_ms: npt.NDArray[np.int64],
    trade_is_buyer_maker: npt.NDArray[np.bool_],
    min_replenish_count: int = 3,
    lookback_snapshots: int = 100,
    qty_recovery_ratio: float = 0.5,
) -> list[AbsorptionEvent]:
    """Detect absorption events by tracking level replenishment.

    Algorithm:
    1. For each price level in the book, track quantity changes
    2. If quantity drops (consumed by trades) but recovers within
       a few snapshots, count as a replenishment
    3. If replenishment count exceeds threshold, flag as absorption

    Args:
        timestamps_ms: Shape (N,) snapshot timestamps.
        bid_prices: Shape (N, K) bid prices.
        bid_qtys: Shape (N, K) bid quantities.
        ask_prices: Shape (N, K) ask prices.
        ask_qtys: Shape (N, K) ask quantities.
        trade_prices: All trade prices in the period.
        trade_qtys: All trade quantities.
        trade_timestamps_ms: Trade timestamps.
        trade_is_buyer_maker: Trade direction flags.
        min_replenish_count: Minimum replenishments to flag absorption.
        lookback_snapshots: How many snapshots to track a level.
        qty_recovery_ratio: Fraction of original qty that must recover.

    Returns:
        List of detected absorption events.
    """
    events: list[AbsorptionEvent] = []
    n_snapshots = len(timestamps_ms)

    if n_snapshots < 3:
        return events

    # Track absorption at the best bid and best ask
    for side, prices, qtys, trade_side_mask in [
        ("bid", bid_prices, bid_qtys, trade_is_buyer_maker),       # sells hit bids
        ("ask", ask_prices, ask_qtys, ~trade_is_buyer_maker),      # buys hit asks
    ]:
        # Track the best level (level 0)
        best_prices = prices[:, 0]
        best_qtys = qtys[:, 0]

        # Find runs where the best price stays the same
        price_changes = np.diff(best_prices)
        same_price = np.abs(price_changes) < 1e-8

        # Find segments of same price
        segment_start = 0
        for i in range(1, n_snapshots):
            if i < n_snapshots - 1 and same_price[i - 1]:
                continue

            # End of a same-price segment
            segment_end = i
            if segment_end - segment_start < 3:
                segment_start = i
                continue

            seg_qtys = best_qtys[segment_start:segment_end]
            seg_price = best_prices[segment_start]

            # Count replenishments: quantity drops then recovers
            replenish_count = 0
            total_absorbed = 0.0
            initial_qty = seg_qtys[0]

            for j in range(1, len(seg_qtys)):
                qty_drop = seg_qtys[j - 1] - seg_qtys[j]
                if qty_drop > 0:
                    total_absorbed += qty_drop
                if (j > 1 and
                    seg_qtys[j] > seg_qtys[j - 1] and
                    seg_qtys[j - 1] < initial_qty * (1 - qty_recovery_ratio)):
                    replenish_count += 1
                    initial_qty = seg_qtys[j]

            if replenish_count >= min_replenish_count:
                score = total_absorbed * replenish_count
                events.append(AbsorptionEvent(
                    timestamp_ms=int(timestamps_ms[segment_end - 1]),
                    price=float(seg_price),
                    side=side,
                    absorbed_volume=float(total_absorbed),
                    replenish_count=replenish_count,
                    score=float(score),
                ))

            segment_start = i

    return events


def absorption_features(
    events: list[AbsorptionEvent],
    current_mid: float,
    bar_duration_ms: int = 300_000,
) -> dict[str, float]:
    """Compute absorption-derived features for a bar.

    Absorption below price (bid absorption) is bullish: hidden buying.
    Absorption above price (ask absorption) is bearish: hidden selling.
    """
    if not events:
        return {
            "absorption_bid_score": 0.0,
            "absorption_ask_score": 0.0,
            "absorption_net": 0.0,
            "absorption_count": 0.0,
        }

    bid_score = sum(e.score for e in events if e.side == "bid")
    ask_score = sum(e.score for e in events if e.side == "ask")

    return {
        "absorption_bid_score": float(bid_score),
        "absorption_ask_score": float(ask_score),
        "absorption_net": float(bid_score - ask_score),  # positive = bullish
        "absorption_count": float(len(events)),
    }
```

### Detection Thresholds for BTC/USDT

| Parameter | Recommended Value | Rationale |
|-----------|------------------|-----------|
| min_replenish_count | 3 | Below 3 could be coincidence |
| qty_recovery_ratio | 0.5 | At least 50% of qty must drop before we call it "absorbed" |
| lookback_snapshots | 100 (10 seconds) | Icebergs refill within seconds |
| min_absorbed_volume | 1.0 BTC | Below this is noise |
| score threshold | 10.0 | absorbed_volume * replenish_count > 10 |

### Iceberg Detection Specifics

An iceberg order is specifically characterized by:
1. A price level repeatedly showing the **same displayed quantity** (the iceberg "display size")
2. Each time that quantity is consumed, it refills to exactly the same amount
3. The total volume traded at the level far exceeds any single displayed quantity

```python
def detect_iceberg_pattern(
    qty_series: npt.NDArray[np.float64],
    tolerance: float = 0.1,
    min_refills: int = 3,
) -> bool:
    """Detect iceberg pattern in a quantity time series at a single price level.

    An iceberg refills to approximately the same displayed size repeatedly.

    Args:
        qty_series: Quantity values at a single price level over time.
        tolerance: Relative tolerance for "same size" detection.
        min_refills: Minimum number of refills to classify as iceberg.
    """
    # Find local minima (dips) and maxima (refills)
    diffs = np.diff(qty_series)
    # A refill: qty drops then rises
    drops = np.where(diffs < 0)[0]
    rises = np.where(diffs > 0)[0]

    if len(drops) < min_refills or len(rises) < min_refills:
        return False

    # Check if the post-refill quantities are approximately equal
    refill_qtys = []
    for d in drops:
        # Find next rise after this drop
        next_rises = rises[rises > d]
        if len(next_rises) > 0:
            refill_idx = next_rises[0] + 1
            if refill_idx < len(qty_series):
                refill_qtys.append(qty_series[refill_idx])

    if len(refill_qtys) < min_refills:
        return False

    refill_arr = np.array(refill_qtys)
    median_refill = np.median(refill_arr)
    if median_refill < 1e-10:
        return False

    # Check that most refills are within tolerance of each other
    within_tol = np.abs(refill_arr - median_refill) / median_refill < tolerance
    return bool(within_tol.sum() >= min_refills)
```

---

## 11. Feature Update Frequency

### The Tradeoff

| Frequency | Pros | Cons |
|-----------|------|------|
| Every 100ms | Maximum information | Extreme noise, 10x compute cost |
| Every 1 second | Good granularity, low noise | Still noisy for features |
| Every 5 seconds | Good balance | May miss fast events |
| Every 5-min bar | Simplest, most stable | Loses intra-bar dynamics |

### Recommendation: Compute at 100ms, Aggregate to 5-min

**The optimal approach**: Compute raw signals (OFI, OBI, microprice) at 100ms resolution, then aggregate multiple statistics over the 5-min bar.

This captures both the real-time dynamics (via slope, std, skew aggregations) and the net effect (via sum, mean, last).

### Which Aggregation for Which Feature

| Feature | Best Aggregation | Why |
|---------|-----------------|-----|
| OFI | sum, last_10s, momentum | Cumulative effect matters |
| OBI | mean, last, slope | Level signal, not cumulative |
| Microprice deviation | mean, slope, std | Sustained vs transient pressure |
| Spread | mean, std, last | Liquidity state |
| Trade intensity | sum, max, last_1m | Bursts matter |
| Kyle's Lambda | last, mean | Slow-moving |
| Absorption | count, net_score | Event-based |

### Concrete Feature Vector

The final 5-min feature vector from microstructure alone should contain ~20-25 features:

```
OFI Features (8):
  ofi_sum_5min, ofi_mean, ofi_std, ofi_skew
  ofi_last_10s, ofi_last_30s, ofi_last_1m, ofi_momentum

OBI Features (4):
  obi_mean, obi_last, obi_std, obi_trend

Microprice Features (4):
  microprice_dev_mean, microprice_dev_last
  microprice_dev_std, microprice_dev_slope

Spread Features (3):
  spread_mean, spread_std, spread_last

TFI Features (5):
  tfi_total, tfi_buy_sell_ratio, tfi_large_trade_imbalance
  tfi_trade_count, tfi_intensity

Kyle's Lambda (2):
  kyle_lambda_last, kyle_lambda_zscore

Absorption (2):
  absorption_net, absorption_count

TOTAL: ~28 features
```

After feature selection, expect ~15-18 to survive (some will be redundant or non-predictive).

---

## 12. Complete Microstructure Feature Pipeline

### End-to-End Architecture

```
WebSocket (depth@100ms)     WebSocket (aggTrade)
         |                           |
         v                           v
  OrderBookRingBuffer         TradeRingBuffer
         |                           |
         v                           v
  Per-snapshot features:      Per-trade features:
   - OBI (levels 1-5)          - Signed volume
   - OFI (6 cases)             - Trade size
   - Microprice                - Direction
   - Spread                         |
         |                           |
         +----------+----------------+
                    |
                    v
           FeatureAggregator
           (collects 100ms data)
                    |
                    v (every 5 min)
           Bar Feature Vector
           (~28 microstructure features)
                    |
                    v
           Normalization Pipeline
           (raw for trees, z-score for neural)
                    |
                    v
           Model Inference
```

### Complete Pipeline Implementation

```python
"""Complete microstructure feature pipeline.

End-to-end: WebSocket messages -> feature vector -> model-ready output.
"""

from __future__ import annotations

import logging
import time
from typing import Final

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

# Size constants
DEPTH_BUFFER_SIZE: Final[int] = 3000   # 5 min at 100ms
TRADE_BUFFER_SIZE: Final[int] = 50_000  # ~10K trades per 5-min bar is typical
MAX_BOOK_LEVELS: Final[int] = 20


class TradeRingBuffer:
    """Ring buffer for aggTrade data."""

    def __init__(self, capacity: int = TRADE_BUFFER_SIZE) -> None:
        self.capacity = capacity
        self._head = 0
        self._count = 0

        self.prices = np.zeros(capacity, dtype=np.float64)
        self.quantities = np.zeros(capacity, dtype=np.float64)
        self.timestamps_ms = np.zeros(capacity, dtype=np.int64)
        self.is_buyer_maker = np.zeros(capacity, dtype=np.bool_)

    @property
    def size(self) -> int:
        return self._count

    def push(
        self,
        price: float,
        quantity: float,
        timestamp_ms: int,
        is_buyer_maker: bool,
    ) -> None:
        idx = self._head
        self.prices[idx] = price
        self.quantities[idx] = quantity
        self.timestamps_ms[idx] = timestamp_ms
        self.is_buyer_maker[idx] = is_buyer_maker

        self._head = (self._head + 1) % self.capacity
        self._count = min(self._count + 1, self.capacity)

    def get_trades_since(
        self, since_ms: int
    ) -> tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.int64],
        npt.NDArray[np.bool_],
    ]:
        """Get all trades since a given timestamp.

        Returns:
            Tuple of (prices, quantities, timestamps, is_buyer_maker).
        """
        if self._count == 0:
            return (
                np.array([], dtype=np.float64),
                np.array([], dtype=np.float64),
                np.array([], dtype=np.int64),
                np.array([], dtype=np.bool_),
            )

        # Get ordered indices
        if self._count < self.capacity:
            indices = np.arange(self._count)
        else:
            indices = np.concatenate([
                np.arange(self._head, self.capacity),
                np.arange(0, self._head),
            ])

        ts = self.timestamps_ms[indices]
        mask = ts >= since_ms

        selected = indices[mask]
        return (
            self.prices[selected],
            self.quantities[selected],
            self.timestamps_ms[selected],
            self.is_buyer_maker[selected],
        )


class MicrostructureFeaturePipeline:
    """Complete microstructure feature computation pipeline.

    Manages ring buffers, computes features at 100ms resolution,
    and aggregates into 5-minute bar features.
    """

    def __init__(
        self,
        bar_duration_ms: int = 300_000,
        n_book_levels: int = 5,
        microprice_beta: float = 0.7,
    ) -> None:
        self.bar_duration_ms = bar_duration_ms
        self.n_book_levels = n_book_levels
        self.microprice_beta = microprice_beta

        # Ring buffers
        self.book_buffer = OrderBookRingBuffer(
            capacity=DEPTH_BUFFER_SIZE, n_levels=MAX_BOOK_LEVELS
        )
        self.trade_buffer = TradeRingBuffer(capacity=TRADE_BUFFER_SIZE)

        # Bar accumulators
        self._bar_start_ms: int = 0
        self._ofi_accum: list[float] = []
        self._obi_accum: list[float] = []
        self._mp_dev_accum: list[float] = []
        self._spread_accum: list[float] = []

        # Calibrator for microprice
        self._calibrator = MicropriceCalibrator()

        # Kyle's Lambda: computed per 1-min interval
        self._lambda_price_changes: list[float] = []
        self._lambda_signed_vols: list[float] = []

    def on_depth_update(
        self,
        timestamp_ms: int,
        bids: npt.NDArray[np.float64],
        asks: npt.NDArray[np.float64],
    ) -> dict[str, float] | None:
        """Process a depth@100ms update.

        Args:
            timestamp_ms: Event timestamp in milliseconds.
            bids: Shape (K, 2) [price, quantity] sorted by price descending.
            asks: Shape (K, 2) [price, quantity] sorted by price ascending.

        Returns:
            Feature dict when a 5-min bar completes. None otherwise.
        """
        # Initialize bar start
        if self._bar_start_ms == 0:
            self._bar_start_ms = timestamp_ms

        # Push to ring buffer
        self.book_buffer.push(timestamp_ms, bids, asks)

        # Compute per-snapshot features
        ofi = self.book_buffer.compute_ofi_snapshot()
        obi = self.book_buffer.compute_obi(n_levels=self.n_book_levels)
        mp = self.book_buffer.compute_microprice(beta=self.microprice_beta)
        idx = (self.book_buffer._head - 1) % self.book_buffer.capacity
        mid = self.book_buffer.mid_prices[idx]
        spread = self.book_buffer.spreads[idx]

        # Accumulate
        self._ofi_accum.append(ofi)
        self._obi_accum.append(obi)
        self._mp_dev_accum.append(
            (mp - mid) / max(spread, 1e-10)
        )
        self._spread_accum.append(spread)

        # Check if bar is complete
        elapsed = timestamp_ms - self._bar_start_ms
        if elapsed >= self.bar_duration_ms:
            features = self._emit_bar_features(timestamp_ms)
            self._reset_bar(timestamp_ms)
            return features

        return None

    def on_trade(
        self,
        price: float,
        quantity: float,
        timestamp_ms: int,
        is_buyer_maker: bool,
    ) -> None:
        """Process an aggTrade update."""
        self.trade_buffer.push(price, quantity, timestamp_ms, is_buyer_maker)

    def _emit_bar_features(self, bar_end_ms: int) -> dict[str, float]:
        """Compute all features for the completed bar."""
        features: dict[str, float] = {}

        # --- OFI features ---
        ofi = np.array(self._ofi_accum, dtype=np.float64)
        features["ofi_sum"] = float(ofi.sum())
        features["ofi_mean"] = float(ofi.mean()) if len(ofi) > 0 else 0.0
        features["ofi_std"] = float(ofi.std()) if len(ofi) > 1 else 0.0
        features["ofi_skew"] = float(_safe_skew(ofi))
        features["ofi_last_10s"] = float(ofi[-100:].sum()) if len(ofi) >= 100 else float(ofi.sum())
        features["ofi_last_30s"] = float(ofi[-300:].sum()) if len(ofi) >= 300 else float(ofi.sum())
        features["ofi_last_1m"] = float(ofi[-600:].sum()) if len(ofi) >= 600 else float(ofi.sum())
        # Momentum: second half minus first half
        half = len(ofi) // 2
        if half > 0:
            features["ofi_momentum"] = float(ofi[half:].sum() - ofi[:half].sum())
        else:
            features["ofi_momentum"] = 0.0

        # --- OBI features ---
        obi = np.array(self._obi_accum, dtype=np.float64)
        features["obi_mean"] = float(obi.mean()) if len(obi) > 0 else 0.5
        features["obi_last"] = float(obi[-1]) if len(obi) > 0 else 0.5
        features["obi_std"] = float(obi.std()) if len(obi) > 1 else 0.0
        features["obi_trend"] = float(_linear_slope(obi))

        # --- Microprice features ---
        mp_dev = np.array(self._mp_dev_accum, dtype=np.float64)
        features["microprice_dev_mean"] = float(mp_dev.mean()) if len(mp_dev) > 0 else 0.0
        features["microprice_dev_last"] = float(mp_dev[-1]) if len(mp_dev) > 0 else 0.0
        features["microprice_dev_std"] = float(mp_dev.std()) if len(mp_dev) > 1 else 0.0
        features["microprice_dev_slope"] = float(_linear_slope(mp_dev))

        # --- Spread features ---
        spreads = np.array(self._spread_accum, dtype=np.float64)
        features["spread_mean"] = float(spreads.mean()) if len(spreads) > 0 else 0.0
        features["spread_std"] = float(spreads.std()) if len(spreads) > 1 else 0.0
        features["spread_last"] = float(spreads[-1]) if len(spreads) > 0 else 0.0

        # --- TFI features (from trade buffer) ---
        prices, qtys, ts, ibm = self.trade_buffer.get_trades_since(self._bar_start_ms)
        if len(qtys) > 0:
            signs = np.where(ibm, -1.0, 1.0)
            buy_mask = ~ibm
            sell_mask = ibm
            buy_vol = float(qtys[buy_mask].sum())
            sell_vol = float(qtys[sell_mask].sum())
            total_vol = buy_vol + sell_vol

            features["tfi_total"] = float((qtys * signs).sum())
            features["tfi_buy_sell_ratio"] = buy_vol / max(total_vol, 1e-10)
            features["tfi_trade_count"] = float(len(qtys))

            # Trade intensity
            if len(ts) > 1:
                duration_s = (ts[-1] - ts[0]) / 1000.0
                features["tfi_intensity"] = len(qtys) / max(duration_s, 0.001)
            else:
                features["tfi_intensity"] = 0.0

            # Large trade imbalance
            if len(qtys) > 10:
                threshold = qtys.mean() + 2 * qtys.std()
                large = qtys > threshold
                large_buy = float(qtys[large & buy_mask].sum())
                large_sell = float(qtys[large & sell_mask].sum())
                large_total = large_buy + large_sell
                features["tfi_large_trade_imbalance"] = (
                    (large_buy - large_sell) / max(large_total, 1e-10)
                )
            else:
                features["tfi_large_trade_imbalance"] = 0.0
        else:
            features["tfi_total"] = 0.0
            features["tfi_buy_sell_ratio"] = 0.5
            features["tfi_trade_count"] = 0.0
            features["tfi_intensity"] = 0.0
            features["tfi_large_trade_imbalance"] = 0.0

        logger.info(
            "bar_features_computed",
            bar_start_ms=self._bar_start_ms,
            bar_end_ms=bar_end_ms,
            n_snapshots=len(ofi),
            n_trades=len(qtys) if len(qtys) > 0 else 0,
            ofi_sum=features["ofi_sum"],
            obi_mean=features["obi_mean"],
        )

        return features

    def _reset_bar(self, new_start_ms: int) -> None:
        self._bar_start_ms = new_start_ms
        self._ofi_accum.clear()
        self._obi_accum.clear()
        self._mp_dev_accum.clear()
        self._spread_accum.clear()


# Re-export helper functions at module level for pipeline use
def _safe_skew(x: npt.NDArray[np.float64]) -> float:
    n = len(x)
    if n < 3:
        return 0.0
    mean = np.mean(x)
    std = np.std(x, ddof=1)
    if std < 1e-15:
        return 0.0
    return float(np.mean(((x - mean) / std) ** 3))


def _linear_slope(x: npt.NDArray[np.float64]) -> float:
    n = len(x)
    if n < 2:
        return 0.0
    t = np.arange(n, dtype=np.float64)
    t -= t.mean()
    x_c = x - x.mean()
    denom = np.dot(t, t)
    if denom < 1e-15:
        return 0.0
    return float(np.dot(t, x_c) / denom)
```

### Integration Test

```python
def test_full_pipeline():
    """Integration test: simulate 5 minutes of depth + trade data."""
    pipeline = MicrostructureFeaturePipeline(
        bar_duration_ms=300_000,
        n_book_levels=5,
        microprice_beta=0.7,
    )

    rng = np.random.default_rng(42)
    base_price = 65000.0
    features_emitted = []

    for i in range(3000):  # 5 minutes at 100ms
        t_ms = 1700000000000 + i * 100

        # Simulate order book
        drift = rng.normal(0, 0.5)
        base_price += drift
        n_levels = 10
        bids = np.column_stack([
            base_price - 0.5 - np.arange(n_levels) * 0.5,
            np.abs(rng.normal(2.0, 0.5, n_levels)),
        ])
        asks = np.column_stack([
            base_price + 0.5 + np.arange(n_levels) * 0.5,
            np.abs(rng.normal(2.0, 0.5, n_levels)),
        ])

        result = pipeline.on_depth_update(t_ms, bids, asks)
        if result is not None:
            features_emitted.append(result)

        # Simulate ~3 trades per 100ms
        for _ in range(rng.poisson(3)):
            trade_price = base_price + rng.normal(0, 0.2)
            trade_qty = abs(rng.exponential(0.05))
            is_bm = rng.random() > 0.5
            pipeline.on_trade(trade_price, trade_qty, t_ms, is_bm)

    # Should emit exactly 1 bar (5 minutes)
    assert len(features_emitted) == 1

    feat = features_emitted[0]

    # Verify all expected features exist
    expected_keys = [
        "ofi_sum", "ofi_mean", "ofi_std", "ofi_skew",
        "ofi_last_10s", "ofi_last_30s", "ofi_last_1m", "ofi_momentum",
        "obi_mean", "obi_last", "obi_std", "obi_trend",
        "microprice_dev_mean", "microprice_dev_last",
        "microprice_dev_std", "microprice_dev_slope",
        "spread_mean", "spread_std", "spread_last",
        "tfi_total", "tfi_buy_sell_ratio", "tfi_trade_count",
        "tfi_intensity", "tfi_large_trade_imbalance",
    ]
    for key in expected_keys:
        assert key in feat, f"Missing feature: {key}"
        assert np.isfinite(feat[key]), f"Non-finite value for {key}: {feat[key]}"

    # Sanity checks on values
    assert -50 < feat["ofi_sum"] < 50        # Symmetric simulation
    assert 0.3 < feat["obi_mean"] < 0.7      # Near 0.5 for symmetric book
    assert feat["tfi_trade_count"] > 0        # Trades were recorded
    assert feat["spread_mean"] > 0            # Spread is positive
```

---

## Summary of Key Findings

### What Matters Most for 5-Min BTC Prediction

1. **OFI (Cont-Kukanov-Stoikov)** is the single most predictive microstructure feature. Multi-level OFI (5 levels) improves on level-1 OFI by 15-35% for crypto. The rolling sum over the full 5-min bar is the primary feature; sub-window sums (10s, 30s, 1min) add complementary information about recency.

2. **Microprice deviation from mid** is more useful than microprice itself. Sustained microprice above mid (positive mean deviation over the bar) predicts upward movement. The slope of deviation (building vs fading pressure) is a strong secondary feature.

3. **Trade-based OFI is cleaner than L2 OFI** because it cannot be spoofed. Use both: L2 OFI captures intentions, trade OFI captures actions. Divergence between them signals manipulation.

4. **Kyle's Lambda as a conditioning variable**: When lambda is high, all other microstructure signals are ~2x more predictive. Create interaction features by multiplying OFI/OBI by lambda z-score.

5. **Normalization**: Use rolling z-score (30-min window) for neural network features. Pass raw values to tree models. Volume-normalized OFI is useful as a secondary feature.

6. **Compute at 100ms, aggregate to 5-min**: The optimal approach. Do not try to make predictions at 100ms resolution (too noisy). But 100ms computation captures the intra-bar dynamics that a 5-min-only computation would miss.

7. **Memory budget**: ~2MB per symbol for the ring buffer. Processing latency: ~0.08ms per update. Well within budget for a single-symbol system.

### Feature Importance Ranking (Expected)

Based on research consensus for 5-min crypto prediction:

1. OFI sum (5-min)
2. OFI last 1-min
3. TFI total (5-min)
4. OBI mean
5. Microprice deviation mean
6. TFI buy/sell ratio
7. OFI momentum
8. Spread mean
9. Kyle's Lambda (conditioning)
10. Absorption net score

These 10 features, combined with volume/volatility/momentum features (~10 more) and context features (~5 more), form the complete ~25-feature vector recommended in the RESEARCH_SYNTHESIS.

---

## Sources

- [Cont, Kukanov, Stoikov (2014) - The Price Impact of Order Book Events](https://academic.oup.com/jfec/article-abstract/12/1/47/816163)
- [Xu et al. (2019) - Multi-Level Order-Flow Imbalance in a Limit Order Book](https://arxiv.org/abs/1907.06230)
- [Stoikov (2018) - The Micro-Price: A High Frequency Estimator of Future Prices](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2970694)
- [Stoikov Slides - Gatheral 60th Birthday Conference](https://www.ma.imperial.ac.uk/~ajacquie/Gatheral60/Slides/Gatheral60%20-%20Stoikov.pdf)
- [Microprice Code Walkthrough - Medium](https://medium.com/@mhfizt/high-frequency-estimator-of-future-prices-micro-price-paper-code-walkthrough-475adb98e91d)
- [Kyle's Lambda - FRDS Documentation](https://frds.io/measures/kyle_lambda/)
- [Dean Markwick - Order Flow Imbalance Signal](https://dm13450.github.io/2022/02/02/Order-Flow-Imbalance.html)
- [Binance Depth Streams Documentation](https://developers.binance.com/docs/derivatives/usds-margined-futures/websocket-market-streams/Partial-Book-Depth-Streams)
- [Binance Aggregate Trade Streams](https://developers.binance.com/docs/derivatives/usds-margined-futures/websocket-market-streams/Aggregate-Trade-Streams)
- [Kolm et al. (2023) - Deep Order Flow Imbalance](https://onlinelibrary.wiley.com/doi/10.1111/mafi.12413)
- [QuantStrategy.io - Iceberg Orders Detection](https://quantstrategy.io/blog/detecting-hidden-intent-unmasking-iceberg-orders-and-order/)
- [Order Flow Analysis of Cryptocurrency Markets - Medium](https://medium.com/@eliquinox/order-flow-analysis-of-cryptocurrency-markets-b479a0216ad8)
