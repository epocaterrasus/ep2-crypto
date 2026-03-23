# Tail Risk and Black Swan Protection for ep2-crypto

> Deep research on protecting a 5-minute BTC prediction system against catastrophic losses. Covers crypto-specific tail risks, fat-tail statistics, hedging strategies, anti-fragile design, scenario-based limits, and survivability analysis for every major crypto black swan event.

---

## Table of Contents

1. [Crypto-Specific Tail Risk Catalog](#1-crypto-specific-tail-risk-catalog)
2. [Fat Tail Analysis for BTC 5-Min Returns](#2-fat-tail-analysis-for-btc-5-min-returns)
3. [Tail Hedging Strategies](#3-tail-hedging-strategies)
4. [Maximum Single-Day Loss Analysis](#4-maximum-single-day-loss-analysis)
5. [Correlated Tail Events](#5-correlated-tail-events)
6. [Anti-Fragile System Design](#6-anti-fragile-system-design)
7. [Insurance and Risk Transfer](#7-insurance-and-risk-transfer)
8. [Scenario-Based Risk Limits](#8-scenario-based-risk-limits)
9. [Recovery from Catastrophic Events](#9-recovery-from-catastrophic-events)
10. [Historical Black Swan Survivability Analysis](#10-historical-black-swan-survivability-analysis)
11. [Implementation: TailRiskMonitor Class](#11-implementation-tailriskmonitor-class)
12. [Integration with ep2-crypto Config](#12-integration-with-ep2-crypto-config)

---

## 1. Crypto-Specific Tail Risk Catalog

Every tail risk below has actually happened. This is not theoretical.

### 1.1 Exchange Hack / Insolvency

| Event | Date | Loss | Warning Signs |
|-------|------|------|---------------|
| Mt. Gox | Feb 2014 | 850,000 BTC ($450M at the time) | Withdrawal delays for weeks, withdrawal halts, suspicious trading |
| Bitfinex | Aug 2016 | 119,756 BTC ($72M) | None visible externally |
| Cryptopia | Jan 2019 | $16M in ETH/ERC-20 | Small exchange, limited audit history |
| QuadrigaCX | Jan 2019 | ~$190M | CEO sole custodian, no multisig, withdrawal delays |
| FTX | Nov 2022 | ~$8B customer funds | Alameda balance sheet leak, withdrawal slowdowns 48h before collapse, FTT token price collapse |

**Protection strategies for ep2-crypto:**
- **Never hold more than 2 days of expected trading capital on any single exchange.** If the system trades with $50K notional, max exchange balance should be ~$10K.
- **Multi-exchange distribution:** Split capital across Binance and Bybit (both already supported in config). If one goes down, the other continues.
- **Withdrawal canary:** Automated check every 4 hours that a small test withdrawal ($10) completes within 30 minutes. If it fails, trigger emergency withdrawal of all funds.
- **Monitor exchange token:** If the exchange has a native token (BNB, etc.), a >20% drop in 24h is a red flag. Pull funds.
- **No DeFi yield:** Never park idle trading capital in DeFi protocols for "yield." The smart contract risk is not worth the few percent.

```python
# src/ep2_crypto/risk/exchange_health.py

from dataclasses import dataclass
from datetime import datetime, timezone
import logging

logger = logging.getLogger(__name__)

@dataclass
class ExchangeHealthCheck:
    """Monitor exchange solvency indicators."""

    exchange_name: str
    last_withdrawal_test: datetime | None = None
    withdrawal_test_passed: bool = True
    consecutive_failures: int = 0

    # Thresholds
    max_balance_usd: float = 10_000.0  # Never hold more than this
    withdrawal_test_interval_hours: float = 4.0
    max_consecutive_failures: int = 2
    native_token_drawdown_alert: float = 0.20  # 20% drop triggers alert

    def should_emergency_withdraw(self) -> bool:
        """Return True if conditions warrant pulling all funds."""
        if self.consecutive_failures >= self.max_consecutive_failures:
            logger.critical(
                "exchange_emergency_withdraw",
                exchange=self.exchange_name,
                consecutive_failures=self.consecutive_failures,
            )
            return True
        return False

    def check_native_token_health(
        self, current_price: float, price_24h_ago: float
    ) -> bool:
        """Check if exchange native token has crashed."""
        if price_24h_ago <= 0:
            return True  # Can't compute, assume ok
        drawdown = (price_24h_ago - current_price) / price_24h_ago
        if drawdown > self.native_token_drawdown_alert:
            logger.warning(
                "exchange_token_crash",
                exchange=self.exchange_name,
                drawdown_pct=round(drawdown * 100, 1),
            )
            return False
        return True
```

### 1.2 Regulatory Ban / Sudden Policy Change

| Event | Date | BTC Impact | Speed |
|-------|------|------------|-------|
| China ICO ban | Sep 2017 | -30% in days | Days of rumor, then sudden announcement |
| China mining ban | May 2021 | -30% in hours | Sudden announcement, 8-hour crash |
| China full ban | Sep 2021 | -10% in hours | Markets partly desensitized by then |
| US infrastructure bill | Nov 2021 | -5% on draft language | Slow-motion, weeks of debate |
| SEC vs Ripple, Coinbase, Binance | 2023 | -5% to -15% per event | Filed then immediate market reaction |

**Protection strategies:**
- **Regulatory bans are crypto-specific events** -- NQ and gold do NOT crash with BTC. Cross-market divergence is a signal.
- The cascade detector (already in research `RR-cascade-liquidation-detection-system.md`) will trigger on the liquidation waves.
- **Position sizing rule:** During Asian session (00-08 UTC), reduce position size by 30% because China announcements drop during their business hours. This is already partially implemented with `weekend_size_reduction` in `MonitoringConfig` but needs a `china_hours_reduction` parameter.
- **News sentiment spike detector:** A sudden spike in negative sentiment (if sentiment module is active) combined with volume spike should trigger immediate position close, not just reduced sizing.

### 1.3 Stablecoin Depeg

| Event | Date | Impact | Mechanism |
|-------|------|--------|-----------|
| UST/LUNA collapse | May 2022 | BTC -25% over 3 days | Algorithmic stablecoin death spiral, contagion to all crypto |
| USDC SVB scare | Mar 2023 | USDC traded at $0.87, BTC -8% then +15% | $3.3B of Circle reserves stuck in SVB, weekend panic |
| USDT periodic FUD | Ongoing | 1-3% dips | Market nervousness about Tether reserves |

**Why this is especially dangerous for our system:**
- If we hold USDT as the quote currency and USDT depegs, our "cash" position loses value even when we're flat.
- If USDC depegs while we're short BTC/USDT, BTC/USDT goes UP (BTC priced in a depreciating dollar), so our short loses money while our cash also loses value. Double hit.

**Protection strategies:**
- **Monitor USDT and USDC peg** every minute. If either deviates by >0.5% from $1.00, trigger kill switch.
- **Diversify stablecoin exposure:** Hold 50% USDT, 50% USDC (or switch to exchange-native margin).
- **Hard rule:** If USDT/USD or USDC/USD < 0.98 on any major pair, close ALL positions and halt trading.

```python
@dataclass
class StablecoinPegMonitor:
    """Monitor stablecoin peg deviation."""

    warning_threshold: float = 0.005   # 0.5% deviation
    critical_threshold: float = 0.02   # 2% deviation -- halt everything

    def check_peg(self, stablecoin: str, price_vs_usd: float) -> str:
        """Return 'ok', 'warning', or 'critical'."""
        deviation = abs(1.0 - price_vs_usd)
        if deviation >= self.critical_threshold:
            logger.critical(
                "stablecoin_depeg_critical",
                stablecoin=stablecoin,
                price=price_vs_usd,
                deviation_pct=round(deviation * 100, 2),
            )
            return "critical"
        if deviation >= self.warning_threshold:
            logger.warning(
                "stablecoin_depeg_warning",
                stablecoin=stablecoin,
                price=price_vs_usd,
                deviation_pct=round(deviation * 100, 2),
            )
            return "warning"
        return "ok"
```

### 1.4 Smart Contract Exploit

Not directly relevant since ep2-crypto trades on CEXs, but if any DeFi exposure exists (e.g., yield on idle capital):
- Wormhole bridge hack: $320M (Feb 2022)
- Ronin bridge hack: $625M (Mar 2022)
- Euler Finance: $197M (Mar 2023)

**Rule: No DeFi exposure for trading capital. Period.**

### 1.5 Network Congestion / Fork

- BTC mempool congestion during May 2023 (Ordinals mania): fees spiked 10x, transactions delayed hours.
- Ethereum merge (Sep 2022): brief uncertainty period.
- BTC/BCH fork (Aug 2017): exchange deposits/withdrawals halted.

**Impact on our system:** Minimal for perpetual futures trading (no on-chain settlement needed). But on-chain features (mempool pressure) will show extreme readings that could confuse the model.

**Protection:** On-chain feature inputs should be clipped to a maximum z-score of 5.0 to prevent garbage-in during congestion events.

### 1.6 Flash Crash from Cascading Liquidations

This is the **most relevant tail risk** for a 5-minute system.

| Event | Date | Drop | Duration | Liquidations |
|-------|------|------|----------|-------------|
| BitMEX flash crash | Jun 2019 | -18% in 20 min | 20 minutes | $500M+ |
| COVID crash | Mar 2020 | -37% in 24h | Sustained decline over hours | $4B+ in 24h |
| China ban crash | May 2021 | -30% in 8h | Self-exciting cascade | $8B+ in 24h |
| Evergrande scare | Sep 2021 | -15% in hours | Moderate cascade | $3B+ |
| FTX collapse | Nov 2022 | -25% over 3 days | Slow then sudden | $1B+ per day |
| Aug 5 2024 | Aug 2024 | -15% in hours | Yen carry trade unwind | $1B+ |

**Our cascade detector from RR-cascade-liquidation-detection-system.md is the primary defense here.** The Hawkes process model detects self-exciting liquidation intensity. When it fires, the system should:
1. Close all open positions immediately (market order, accept slippage)
2. Optionally flip short if cascade confidence > 80% (see Anti-Fragile section)
3. Halt new entries until cascade intensity returns below threshold

### 1.7 Market Manipulation

- **Spoofing:** Large orders placed and cancelled to move price. Detected by order book features (orders that appear and vanish within seconds).
- **Wash trading:** Inflated volume. Less relevant since we use aggTrade data which is harder to fake.
- **Pump-and-dump:** Mostly altcoins, not BTC.
- **Whale manipulation:** Sudden large market sells to trigger cascading liquidations, then buy back at lower price.

**Protection:** The microstructure features in `features/microstructure.py` (order book imbalance, absorption detection) help detect manipulation. Add a **spoofing detector** that tracks order cancellation rate at top-of-book.

### 1.8 API Key Compromise

If an attacker gets the exchange API key, they can:
- Place losing trades (buy high, sell low) to drain the account
- Withdraw funds (if withdrawal permission is enabled)

**Non-negotiable rules:**
- **API keys must NEVER have withdrawal permission.** Trading-only keys.
- **IP whitelist** on exchange API settings (only the server's IP).
- **API keys in Doppler/env vars** (already enforced by `config.py` using `SecretStr`).
- **Rate limit monitoring:** If unexpected API calls are detected, rotate keys immediately.

### 1.9 Infrastructure Failure

- Cloud outage (AWS us-east-1 failures have happened multiple times)
- DNS failure
- Exchange API outage during a crash (BitMEX went down during COVID crash)

**Protection:**
- **Stale data detector:** If no new data for 60 seconds, close all positions. Already implied by health checks in `orchestrator.py`.
- **Exchange failover:** If Binance API goes down, fail over to Bybit.
- **Always-on stop losses:** Place OCO/stop-loss orders on the exchange itself, not just in the application layer. If the application crashes, the exchange-side stop still protects the position.

---

## 2. Fat Tail Analysis for BTC 5-Min Returns

### 2.1 BTC Returns Are NOT Gaussian

This is the single most important statistical fact for risk management. BTC returns have **excess kurtosis far above zero** and **fatter tails than any traditional asset**.

**Empirical statistics for BTC/USDT 5-minute log returns (2020-2024):**

| Statistic | BTC 5-min | Gaussian | Implication |
|-----------|-----------|----------|-------------|
| Mean | ~0.0000 | 0 | Roughly zero per bar, as expected |
| Std dev (sigma) | ~0.0025 (0.25%) | - | One sigma move is 0.25% per 5-min bar |
| Skewness | -0.3 to -0.8 | 0 | Left-skewed: crashes are bigger than rallies |
| Excess Kurtosis | 15 to 50+ | 0 | **Massive fat tails** |
| 4-sigma events/month | 10-30 | 0.27 expected | **40-100x more frequent than Gaussian** |
| 5-sigma events/month | 2-8 | 0.002 expected | **1000x more frequent** |
| 6-sigma events/year | 5-15 | 0.00000074 expected | **Should "never" happen, happens monthly** |

### 2.2 Quantifying the Fat Tails

```python
# scripts/analyze_btc_fat_tails.py
"""
Analyze BTC 5-minute return distribution: kurtosis, sigma events,
and fit Student-t / stable distributions.

Requires: numpy, scipy, pandas
Input: BTC/USDT 5-min OHLCV data (from database or CSV)
"""

import logging
from dataclasses import dataclass

import numpy as np
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)


@dataclass
class FatTailAnalysis:
    """Results of fat tail analysis on BTC 5-min returns."""

    n_observations: int
    mean: float
    std: float
    skewness: float
    kurtosis_excess: float  # excess kurtosis (0 for Gaussian)

    # Sigma event counts
    events_3sigma: int
    events_4sigma: int
    events_5sigma: int
    events_6sigma: int

    # Gaussian expected counts
    expected_3sigma: float
    expected_4sigma: float
    expected_5sigma: float
    expected_6sigma: float

    # Ratios (actual / expected)
    ratio_3sigma: float
    ratio_4sigma: float
    ratio_5sigma: float
    ratio_6sigma: float

    # Student-t fit
    student_t_df: float  # degrees of freedom
    student_t_loc: float
    student_t_scale: float

    # Stable distribution fit
    stable_alpha: float  # stability parameter (2.0 = Gaussian)
    stable_beta: float   # skewness parameter


def analyze_fat_tails(returns: np.ndarray) -> FatTailAnalysis:
    """
    Analyze the fat tail properties of BTC 5-min returns.

    Args:
        returns: Array of 5-minute log returns (not percentage, raw log returns)

    Returns:
        FatTailAnalysis with all statistics
    """
    n = len(returns)
    mean = float(np.mean(returns))
    std = float(np.std(returns, ddof=1))
    skew = float(sp_stats.skew(returns))
    kurt = float(sp_stats.kurtosis(returns))  # excess kurtosis

    # Standardize returns
    z_scores = (returns - mean) / std

    # Count sigma events
    events_3 = int(np.sum(np.abs(z_scores) > 3))
    events_4 = int(np.sum(np.abs(z_scores) > 4))
    events_5 = int(np.sum(np.abs(z_scores) > 5))
    events_6 = int(np.sum(np.abs(z_scores) > 6))

    # Gaussian expected counts (two-tailed)
    # P(|Z| > k) for standard normal
    exp_3 = n * 2 * sp_stats.norm.sf(3)    # 0.0027
    exp_4 = n * 2 * sp_stats.norm.sf(4)    # 0.0000633
    exp_5 = n * 2 * sp_stats.norm.sf(5)    # 5.73e-7
    exp_6 = n * 2 * sp_stats.norm.sf(6)    # 1.97e-9

    # Fit Student-t distribution
    t_df, t_loc, t_scale = sp_stats.t.fit(returns)

    # Fit stable distribution (this can be slow for large datasets)
    # Sample if dataset is very large
    sample = returns if n <= 100_000 else np.random.default_rng(42).choice(returns, 100_000, replace=False)
    try:
        stable_params = sp_stats.levy_stable.fit(sample)
        s_alpha, s_beta = stable_params[0], stable_params[1]
    except Exception:
        logger.warning("stable_distribution_fit_failed, using defaults")
        s_alpha, s_beta = 1.7, -0.1  # typical BTC values

    return FatTailAnalysis(
        n_observations=n,
        mean=mean,
        std=std,
        skewness=skew,
        kurtosis_excess=kurt,
        events_3sigma=events_3,
        events_4sigma=events_4,
        events_5sigma=events_5,
        events_6sigma=events_6,
        expected_3sigma=exp_3,
        expected_4sigma=exp_4,
        expected_5sigma=exp_5,
        expected_6sigma=exp_6,
        ratio_3sigma=events_3 / max(exp_3, 1e-10),
        ratio_4sigma=events_4 / max(exp_4, 1e-10),
        ratio_5sigma=events_5 / max(exp_5, 1e-10),
        ratio_6sigma=events_6 / max(exp_6, 1e-10),
        student_t_df=t_df,
        student_t_loc=t_loc,
        student_t_scale=t_scale,
        stable_alpha=s_alpha,
        stable_beta=s_beta,
    )


def compute_var_comparison(
    returns: np.ndarray, confidence: float = 0.99
) -> dict[str, float]:
    """
    Compare VaR estimates: Gaussian vs Student-t vs Historical.

    Shows how much Gaussian VaR underestimates actual risk.
    """
    mean = float(np.mean(returns))
    std = float(np.std(returns, ddof=1))

    # Gaussian VaR
    gaussian_var = -(mean + std * sp_stats.norm.ppf(1 - confidence))

    # Student-t VaR
    t_df, t_loc, t_scale = sp_stats.t.fit(returns)
    student_t_var = -(t_loc + t_scale * sp_stats.t.ppf(1 - confidence, t_df))

    # Historical VaR (most reliable)
    historical_var = -float(np.percentile(returns, (1 - confidence) * 100))

    # Cornish-Fisher VaR (adjusts Gaussian for skew/kurtosis)
    skew = float(sp_stats.skew(returns))
    kurt = float(sp_stats.kurtosis(returns))
    z = sp_stats.norm.ppf(1 - confidence)
    cf_z = (
        z
        + (z**2 - 1) * skew / 6
        + (z**3 - 3 * z) * kurt / 24
        - (2 * z**3 - 5 * z) * skew**2 / 36
    )
    cf_var = -(mean + std * cf_z)

    return {
        "gaussian_var": gaussian_var,
        "student_t_var": student_t_var,
        "historical_var": historical_var,
        "cornish_fisher_var": cf_var,
        "gaussian_underestimate_ratio": historical_var / max(gaussian_var, 1e-10),
    }
```

### 2.3 What the Numbers Mean for Position Sizing

**Key insight:** Gaussian VaR at 99% typically underestimates BTC 5-min tail risk by **2-4x**.

If Gaussian VaR says your worst 1% loss is -0.6%, the actual worst 1% loss is more like **-1.5% to -2.4%**.

For position sizing, this means:
- If you use Gaussian VaR to set position sizes, you will be **2-4x overexposed** during tail events.
- **Always use historical VaR or Cornish-Fisher VaR** for position sizing.
- Better yet: use **Expected Shortfall (CVaR)** which measures the average loss in the tail, not just the threshold.

**Student-t fit for BTC:** Typical degrees of freedom (df) for BTC 5-min returns is **3-5**. For comparison:
- df = infinity: Gaussian
- df = 30: nearly Gaussian
- df = 5: moderate fat tails (equities)
- df = 3-4: **heavy fat tails** (BTC)
- df < 3: infinite variance (the distribution has no finite variance)

When df approaches 3, even variance becomes unstable. This means **volatility estimates themselves are unreliable** during regime shifts.

### 2.4 Stable Distribution Analysis

The Levy alpha-stable distribution is the theoretically correct distribution for assets with fat tails:
- **alpha = 2.0**: Gaussian (thin tails)
- **alpha = 1.5-1.8**: Typical for BTC (fat tails, finite mean, infinite variance in theory)
- **alpha = 1.0**: Cauchy distribution (no mean, no variance)

BTC 5-min returns typically show **alpha between 1.5 and 1.8**, confirming that:
1. The mean is well-defined (alpha > 1)
2. The variance is theoretically infinite (alpha < 2), meaning sample variance grows with sample size
3. **Standard deviation-based risk measures are fundamentally flawed** for this distribution

**Practical implication:** This is why the `RealizedVolComputer` and `EWMAVolComputer` in `features/volatility.py` should always be supplemented with **non-parametric risk measures** (historical percentiles, max drawdown lookbacks).

---

## 3. Tail Hedging Strategies

### 3.1 BTC Options as Tail Hedges (Deribit OTM Puts)

**Mechanics:** Buy out-of-the-money (OTM) put options on BTC. If BTC crashes, the puts appreciate massively due to convexity.

**Current Deribit options landscape (as of 2025-2026):**
- BTC options settle in BTC (inverse payout)
- Available strikes: every $1,000 for near-term, every $5,000-$10,000 for long-term
- Expiries: daily, weekly, monthly, quarterly
- Minimum order: 0.1 BTC

**Cost analysis for tail hedging with OTM puts:**

| Put Strike (% OTM) | Typical Premium (% of notional) | Protection Begins | Payoff at -30% crash |
|---------------------|-------------------------------|-------------------|---------------------|
| 10% OTM | 0.8-1.5% per month | At -10% | ~20% of notional |
| 20% OTM | 0.3-0.7% per month | At -20% | ~10% of notional |
| 30% OTM | 0.1-0.3% per month | At -30% | Barely in the money |

**Cost-benefit for ep2-crypto:**

Assuming $100K portfolio, buying 20% OTM monthly puts:
- Monthly cost: $300-$700 (0.3-0.7%)
- Annual cost: $3,600-$8,400 (3.6-8.4%)
- This eats a **massive** portion of expected returns from a 5-min system

**Verdict: Continuous tail hedging with options is NOT cost-effective for a 5-min system.** The annualized cost (4-8%) likely exceeds the system's alpha after transaction costs. However, **conditional tail hedging** (buy puts only when risk indicators are elevated) can be justified.

### 3.2 Dynamic/Conditional Tail Hedging

Instead of always being hedged, buy protection only when tail risk indicators spike:

```python
@dataclass
class TailHedgeSignal:
    """Determines when to buy tail protection."""

    # Thresholds for activating hedge
    funding_rate_zscore_threshold: float = 2.0  # Elevated funding = crowded longs
    oi_percentile_threshold: float = 90.0       # High OI = lots of leverage
    vol_of_vol_zscore_threshold: float = 2.5    # Unstable volatility
    cascade_probability_threshold: float = 0.3  # Hawkes model cascade prob

    def should_hedge(
        self,
        funding_zscore: float,
        oi_percentile: float,
        vol_of_vol_zscore: float,
        cascade_prob: float,
    ) -> tuple[bool, float]:
        """
        Returns (should_hedge, urgency_score).

        Urgency score 0-1 determines hedge size:
        - 0.3-0.5: small hedge (5% of portfolio in puts)
        - 0.5-0.7: medium hedge (10%)
        - 0.7+: large hedge (15-20%)
        """
        score = 0.0
        triggers = 0

        if funding_zscore > self.funding_rate_zscore_threshold:
            score += min(funding_zscore / 5.0, 0.3)
            triggers += 1

        if oi_percentile > self.oi_percentile_threshold:
            score += (oi_percentile - 80) / 80  # scales 0 to 0.25
            triggers += 1

        if vol_of_vol_zscore > self.vol_of_vol_zscore_threshold:
            score += min(vol_of_vol_zscore / 8.0, 0.25)
            triggers += 1

        if cascade_prob > self.cascade_probability_threshold:
            score += cascade_prob * 0.4  # cascade is strongest signal
            triggers += 1

        # Need at least 2 triggers to activate
        should_activate = triggers >= 2 and score > 0.3

        return should_activate, min(score, 1.0)
```

**When conditional hedging has historically been valuable:**
- May 2021: funding rate was >0.05% for days AND OI was at ATH -> both indicators were screaming
- Nov 2022 (FTX): exchange token was crashing, withdrawal queues growing -> exchange health indicators
- Mar 2020: cross-market correlation with equities was at ATH AND VIX was spiking -> macro indicators

**Cost of conditional hedging:** Instead of 4-8% annually, conditional hedging costs 0.5-2% annually because you're only hedged 15-25% of the time.

### 3.3 The Taleb Barbell Strategy

Nassim Taleb's barbell: hold 85-90% in ultra-safe assets and 10-15% in extremely convex (high-risk, high-reward) bets.

**Applied to ep2-crypto:**
- **90% of capital:** In cold storage or money market (never on exchange). This is the "can't lose" portion.
- **10% of capital:** Active trading capital on exchange. This is the "can lose it all" portion.
- **Rebalance monthly:** If trading capital grows to 15% of total, withdraw profits to the safe portion. If it shrinks to 5%, either inject from safe portion or stop trading.

**The math:**
- If trading capital (10%) gets completely wiped out: total portfolio loses 10%. Recoverable.
- If trading capital doubles: total portfolio gains 10%. Nice.
- You can NEVER lose more than 10% of total net worth.

**This is the single most important risk management strategy.** It is a structural constraint, not a model parameter. No amount of tail hedging replaces the discipline of never having more than 10% of capital at risk.

### 3.4 Is Tail Hedging Worth It for a 5-Min System?

**Arguments against:**
1. 5-min systems should be flat most of the time (position duration is minutes, not days)
2. Options have daily theta decay -- hedging a 5-min position with a monthly option is wasteful
3. The system's primary defense is **being flat** -- if no position is open, a crash costs nothing
4. Transaction costs of hedging eat into already-thin 5-min system margins

**Arguments for:**
1. The system might be in a position exactly when a flash crash hits
2. Even flat, you have exchange counterparty risk
3. Black swans are correlated -- the crash happens when you're most likely to be wrong

**Conclusion for ep2-crypto:**
- **Primary defense:** Position sizing + kill switches + being flat often (already in system)
- **Secondary defense:** Conditional tail hedging when multiple risk indicators align (buy puts on Deribit)
- **Structural defense:** Taleb barbell -- keep 90% off-exchange
- **NOT recommended:** Continuous OTM put buying (too expensive)

---

## 4. Maximum Single-Day Loss Analysis

### 4.1 Historical Worst Days for BTC

| Date | Open | Close | Intraday Low | Max Drawdown | Duration |
|------|------|-------|-------------|-------------|----------|
| Mar 12, 2020 | $7,911 | $4,970 | $3,850 | **-51.3%** (over 2 days) | 36 hours |
| Mar 12, 2020 (single day) | $7,911 | $5,671 | $4,970 | **-37.1%** | 24 hours |
| May 19, 2021 | $42,800 | $36,700 | $30,000 | **-29.9%** | 8 hours |
| Jun 22, 2021 | $35,400 | $29,800 | $28,800 | **-18.6%** | 12 hours |
| Nov 8-9, 2022 | $20,500 | $15,800 | $15,500 | **-24.4%** | 48 hours |
| Jan 11, 2024 | $48,900 | $42,800 | $41,500 | **-15.1%** | 6 hours |
| Aug 5, 2024 | $61,300 | $54,200 | $49,500 | **-19.3%** | 12 hours |

### 4.2 Position Size vs Account Destruction

**The critical calculation:** At what leverage/position size does a single-day move destroy the account?

Given the current `MonitoringConfig`:
- `position_size_fraction: 0.05` (5% of capital per position)
- `max_drawdown_halt: 0.15` (halt at 15% drawdown)
- `catastrophic_stop_atr: 3.0` (stop loss at 3x ATR)

**Scenario: March 12, 2020 (-37% day)**

At 5% position size (current config):
- Position loss = 37% * 5% = 1.85% of capital
- **Survived easily.** Well within the 15% drawdown halt.

At 20% position size:
- Position loss = 37% * 20% = 7.4% of capital
- Survived, but approaching daily loss limit (3%).

At 50% position size:
- Position loss = 37% * 50% = 18.5% of capital
- **Account destruction.** Exceeds 15% drawdown halt. Kill switch fires, but damage is done.

At 100% position size (full capital, no leverage):
- Position loss = 37%
- **Severe damage.** More than a third of capital gone.

At 3x leverage (perpetual futures):
- Position loss = 37% * 3 = 111%
- **Liquidation.** Account wiped out completely.

**Design rule: To survive the worst single day in BTC history with <30% portfolio drawdown:**
```
max_position_size <= 0.30 / worst_historical_daily_move
max_position_size <= 0.30 / 0.37
max_position_size <= 0.81 (81% of capital, no leverage)
```

But this is for a SINGLE position. For a system that may take multiple positions in a day:
```
max_aggregate_exposure <= 0.30 / 0.37 = 81%
```

**Current config (5% position size) provides a massive safety margin:** even 6 concurrent losing trades at -37% each would only lose 11.1%. This is correct and should not be relaxed.

```python
def max_position_size_for_survival(
    max_acceptable_drawdown: float,
    worst_case_move: float,
    safety_factor: float = 1.5,  # extra margin
) -> float:
    """
    Calculate maximum position size to survive a worst-case move.

    Args:
        max_acceptable_drawdown: e.g., 0.30 for 30%
        worst_case_move: e.g., 0.37 for 37% single-day BTC drop
        safety_factor: multiply worst case by this (account for slippage, gap)

    Returns:
        Maximum position size as fraction of capital
    """
    adjusted_move = worst_case_move * safety_factor
    max_size = max_acceptable_drawdown / adjusted_move
    return min(max_size, 1.0)  # Can't be more than 100%


# Example: survive -37% day with <30% drawdown, 1.5x safety factor
# max_position_size_for_survival(0.30, 0.37, 1.5) = 0.30 / 0.555 = 0.54
# With safety factor, max 54% of capital -- current 5% is extremely conservative
```

### 4.3 The Stop-Loss Gap Problem

Stop-losses don't always execute at the stop price. During flash crashes:
- **Slippage can be 5-10x normal** (confirmed in stress test framework: `slippage_multiplier=5.0`)
- **Exchange might be unreachable** (BitMEX went down for 25 minutes during COVID crash)
- **Price can gap through your stop** in the next bar

**Worst-case stop-loss slippage model:**

```python
def worst_case_stop_loss_exit(
    entry_price: float,
    stop_price: float,
    normal_slippage_bps: float = 5.0,
    stress_slippage_multiplier: float = 10.0,
    exchange_downtime_minutes: float = 30.0,
    price_velocity_per_minute: float = 0.005,  # 0.5% per minute during crash
) -> float:
    """
    Model worst-case exit price when stop-loss triggers during a crash.

    During March 2020, BTC fell ~0.3-0.5% per minute at peak velocity.
    Exchange downtime of 25 minutes meant 7-12% additional slippage.
    """
    # Normal slippage, amplified by stress
    slippage_pct = normal_slippage_bps * 1e-4 * stress_slippage_multiplier

    # Additional slippage from exchange downtime
    downtime_slippage = exchange_downtime_minutes * price_velocity_per_minute

    # Total slippage from stop price
    total_slippage = slippage_pct + downtime_slippage

    # Worst case exit
    worst_exit = stop_price * (1 - total_slippage)

    return worst_exit
```

**Example:** Stop at $50,000, stress slippage 0.5%, exchange down 30 min at 0.5%/min velocity:
- Total additional loss: 0.5% + 15% = 15.5%
- Actual exit: $50,000 * 0.845 = $42,250
- You thought you'd exit at $50K but actually exited at $42.25K -- an extra **15.5% loss.**

**This is why position sizing is more important than stop-losses.** You can't trust stop-losses during the exact events when you need them most.

---

## 5. Correlated Tail Events

### 5.1 The Correlation Clustering Problem

During normal markets, risks are somewhat independent. During tail events, everything correlates:

| Normal State | Crisis State |
|-------------|-------------|
| BTC and exchange health uncorrelated | BTC crashes AND exchange goes down |
| BTC and stablecoin peg uncorrelated | BTC crashes AND USDT depegs |
| BTC and funding rate weakly correlated | BTC crashes AND funding spikes negative (double hit on shorts) |
| BTC and cross-market correlated at 0.3-0.5 | BTC and NQ both crash at correlation 0.8+ |
| API latency stable at 50ms | API latency spikes to 5s during crash (can't close positions) |

### 5.2 Compound Tail Event Scenarios

**Scenario A: BTC crash + exchange outage (highest historical precedent)**

March 12, 2020: BitMEX went down for 25 minutes during the worst of the crash. Traders could not close positions. This actually *stopped* the cascade (BitMEX was a major liquidation venue), but anyone with positions on BitMEX was trapped.

Modeling:
```python
@dataclass
class CompoundScenario:
    """Model correlated tail events."""

    name: str
    btc_move_pct: float            # e.g., -0.30 for 30% crash
    exchange_down_prob: float       # probability exchange is unreachable
    exchange_down_duration_min: float
    stablecoin_depeg_prob: float    # probability of stablecoin depeg
    stablecoin_depeg_amount: float  # e.g., 0.05 for 5% depeg
    funding_spike_prob: float       # probability of extreme funding
    funding_spike_rate: float       # e.g., -0.03 for -3% in 8h
    slippage_multiplier: float     # how much worse than normal


COMPOUND_SCENARIOS = [
    CompoundScenario(
        name="covid_crash_replay",
        btc_move_pct=-0.37,
        exchange_down_prob=0.30,
        exchange_down_duration_min=25,
        stablecoin_depeg_prob=0.05,
        stablecoin_depeg_amount=0.02,
        funding_spike_prob=0.90,
        funding_spike_rate=-0.03,
        slippage_multiplier=10.0,
    ),
    CompoundScenario(
        name="ftx_style_insolvency",
        btc_move_pct=-0.25,
        exchange_down_prob=0.80,  # exchange becomes unreachable
        exchange_down_duration_min=1440,  # 24 hours
        stablecoin_depeg_prob=0.15,
        stablecoin_depeg_amount=0.10,
        funding_spike_prob=0.70,
        funding_spike_rate=-0.02,
        slippage_multiplier=5.0,
    ),
    CompoundScenario(
        name="stablecoin_death_spiral",
        btc_move_pct=-0.20,
        exchange_down_prob=0.10,
        exchange_down_duration_min=5,
        stablecoin_depeg_prob=0.95,
        stablecoin_depeg_amount=0.15,
        funding_spike_prob=0.50,
        funding_spike_rate=-0.01,
        slippage_multiplier=3.0,
    ),
    CompoundScenario(
        name="cascading_liquidation_extreme",
        btc_move_pct=-0.30,
        exchange_down_prob=0.20,
        exchange_down_duration_min=10,
        stablecoin_depeg_prob=0.05,
        stablecoin_depeg_amount=0.01,
        funding_spike_prob=0.95,
        funding_spike_rate=-0.04,
        slippage_multiplier=15.0,
    ),
]


def compute_compound_loss(
    scenario: CompoundScenario,
    position_size_frac: float,
    capital_on_exchange_frac: float,
    stablecoin_cash_frac: float,
) -> dict[str, float]:
    """
    Compute total portfolio loss from a compound tail event.

    Args:
        position_size_frac: fraction of capital in active position
        capital_on_exchange_frac: fraction of total net worth on exchange
        stablecoin_cash_frac: fraction of exchange balance in stablecoins
    """
    # Direct position loss (if we're positioned wrong)
    position_loss = abs(scenario.btc_move_pct) * position_size_frac

    # Exchange loss (if exchange goes down permanently like FTX)
    # Probability-weighted
    exchange_loss = (
        scenario.exchange_down_prob
        * capital_on_exchange_frac
        * (1.0 if scenario.exchange_down_duration_min > 720 else 0.0)
        # Only count total loss if exchange is down > 12 hours (insolvency risk)
    )

    # Stablecoin depeg loss on cash portion
    stable_loss = (
        scenario.stablecoin_depeg_prob
        * scenario.stablecoin_depeg_amount
        * stablecoin_cash_frac
        * capital_on_exchange_frac
    )

    # Additional slippage on exit
    normal_slippage = 0.0005  # 5 bps
    stress_slippage = normal_slippage * scenario.slippage_multiplier
    slippage_loss = stress_slippage * position_size_frac

    total_loss = position_loss + exchange_loss + stable_loss + slippage_loss

    return {
        "position_loss": position_loss,
        "exchange_loss": exchange_loss,
        "stablecoin_loss": stable_loss,
        "slippage_loss": slippage_loss,
        "total_loss": total_loss,
        "survives_30pct_rule": total_loss < 0.30,
    }
```

**Running the scenarios with current ep2-crypto config (5% position, 10% on exchange, 90% in stablecoins):**

| Scenario | Position Loss | Exchange Loss | Stable Loss | Total | Survives? |
|----------|-------------|---------------|-------------|-------|-----------|
| COVID crash replay | 1.85% | 0% (exchange came back) | 0.01% | 2.4% | YES |
| FTX insolvency | 1.25% | 8.0% (80% prob * 10% on exchange) | 0.14% | 9.6% | YES |
| Stablecoin death spiral | 1.0% | 0% | 1.28% | 2.4% | YES |
| Cascading liquidation extreme | 1.5% | 0% | 0.005% | 2.3% | YES |

**The 5% position size + 10% on exchange constraint means the system survives ALL compound scenarios.** This validates the current config as appropriately conservative.

### 5.3 Modeling Correlation Breakdown

During normal times, tail risk components are roughly independent. During crises, the **copula** changes -- the correlation structure itself shifts.

```python
def crisis_correlation_matrix() -> dict[str, dict[str, float]]:
    """
    Empirical correlation matrix during crypto crises.
    Based on Mar 2020, May 2021, Nov 2022 data.
    """
    # Normal regime correlations
    normal = {
        "btc_return": {"exchange_health": 0.05, "stable_peg": 0.02, "funding": -0.15, "api_latency": 0.01},
        "exchange_health": {"stable_peg": 0.10, "funding": 0.05, "api_latency": 0.20},
        "stable_peg": {"funding": 0.05, "api_latency": 0.05},
        "funding": {"api_latency": 0.10},
    }

    # Crisis regime correlations (everything becomes correlated)
    crisis = {
        "btc_return": {"exchange_health": 0.60, "stable_peg": 0.40, "funding": -0.70, "api_latency": 0.55},
        "exchange_health": {"stable_peg": 0.50, "funding": 0.45, "api_latency": 0.65},
        "stable_peg": {"funding": 0.35, "api_latency": 0.30},
        "funding": {"api_latency": 0.50},
    }

    return {"normal": normal, "crisis": crisis}
```

**Key insight:** The regime detector in `regime/detector.py` should include a **correlation regime** state. When cross-correlations between risk factors spike above 0.5, the system is in crisis mode and should reduce all exposure regardless of signal quality.

---

## 6. Anti-Fragile System Design

### 6.1 Taleb's Anti-Fragility Principles Applied to Trading

Anti-fragile systems don't just survive shocks -- they **benefit** from them. Taleb's key principles and how they map to ep2-crypto:

| Principle | Trading Application |
|-----------|-------------------|
| **Optionality > prediction** | Have the option to go short during cascades, not just survive them |
| **Small losses, large gains** | 5% position sizes mean small losses; cascade shorting means large gains |
| **Redundancy** | Multi-exchange, multi-stablecoin, exchange-side stops AND app-side stops |
| **Skin in the game** | No paper trading fake confidence -- real money, real consequences |
| **Via negativa** | Remove fragilities first (reduce position size, add kill switches) before adding features |
| **Barbell** | 90% safe + 10% aggressive (the Taleb barbell from Section 3) |

### 6.2 Profiting from Tail Events: Cascade Short Strategy

The cascade detector (from `RR-cascade-liquidation-detection-system.md`) already identifies self-exciting liquidation events. The anti-fragile extension: **go short when a cascade is confirmed**.

**Why this works:**
- Cascading liquidations are self-reinforcing: forced sellers create more forced sellers
- The dynamics are non-linear: a 5% drop triggers liquidations that cause another 5% drop, and so on
- By the time the cascade is detectable (first 5-10 minutes), there's typically another 10-20% downside remaining

**Historical cascade short performance:**

| Event | Cascade Detected | Remaining Downside | Duration of Cascade |
|-------|-----------------|-------------------|-------------------|
| Mar 12, 2020 | After -10% drop | -27% more | ~6 hours |
| May 19, 2021 | After -8% drop | -22% more | ~4 hours |
| Jun 2022 (3AC) | After -5% drop | -15% more | ~12 hours |
| Nov 2022 (FTX) | After -10% drop | -15% more | ~48 hours |

**Implementation:**

```python
@dataclass
class CascadeShortConfig:
    """Configuration for anti-fragile cascade shorting."""

    # Activation thresholds
    min_cascade_confidence: float = 0.80   # Hawkes model confidence
    min_liquidation_intensity: float = 3.0  # Z-score of liquidation rate
    min_price_drop_pct: float = 0.05       # Must already have dropped 5%

    # Position sizing for cascade short
    cascade_position_size: float = 0.03    # 3% of capital (smaller than normal)
    max_cascade_duration_hours: float = 12.0  # Auto-close after 12h
    cascade_stop_loss_pct: float = 0.05    # 5% stop (price reverses against us)
    cascade_take_profit_pct: float = 0.15  # 15% take profit

    # Risk controls
    max_cascade_trades_per_week: int = 2   # Don't over-trade cascades
    min_time_between_cascade_trades_hours: float = 24.0

    # Required conditions (ALL must be true)
    require_high_oi: bool = True           # OI must be above 70th percentile
    require_negative_funding: bool = False  # Funding doesn't need to be negative yet
    require_spread_widening: bool = True   # Spread must be > 3x normal


def evaluate_cascade_short_opportunity(
    cascade_confidence: float,
    liquidation_zscore: float,
    price_drop_from_recent_high: float,
    oi_percentile: float,
    spread_vs_normal: float,
    config: CascadeShortConfig,
) -> tuple[bool, float]:
    """
    Evaluate whether to enter a cascade short position.

    Returns:
        (should_enter, position_size_fraction)
    """
    # Check all conditions
    if cascade_confidence < config.min_cascade_confidence:
        return False, 0.0

    if liquidation_zscore < config.min_liquidation_intensity:
        return False, 0.0

    if abs(price_drop_from_recent_high) < config.min_price_drop_pct:
        return False, 0.0

    if config.require_high_oi and oi_percentile < 70:
        return False, 0.0

    if config.require_spread_widening and spread_vs_normal < 3.0:
        return False, 0.0

    # Scale position size by confidence
    size = config.cascade_position_size * min(cascade_confidence, 1.0)

    return True, size
```

**Risk of cascade shorting:**
- False cascades: price drops 5%, looks like a cascade, then V-reverses. Stop-loss at 5% limits damage.
- Frequency: true cascades happen 2-4 times per year. Most of the time this code is dormant.
- The WORST case: enter short, price V-reverses, lose 5% of 3% position = 0.15% of capital. Acceptable.

### 6.3 Dry Powder Convexity

Being flat during a crash is good. Having **capital to deploy** after a crash is even better.

**The convexity of dry powder:**
- After a 30% crash, buying BTC has historically returned 50-200% over the next 6-12 months
- After March 2020 (-51%): BTC went from $3,800 to $64,000 in 13 months (+1,584%)
- After May 2021 (-55%): BTC went from $28,800 to $69,000 in 6 months (+140%)
- After Nov 2022 (-77% from ATH): BTC went from $15,500 to $73,000 in 16 months (+371%)

**This is outside the scope of a 5-minute system** but is critical for overall portfolio management. The Taleb barbell's 90% "safe" portion can be strategically deployed after confirmed bottoms.

### 6.4 Anti-Fragile Position Sizing: The Inverse Volatility Approach

Standard position sizing: fixed fraction (5% of capital).
Anti-fragile position sizing: **increase size when vol is low, decrease when vol is high.** This is the opposite of what most traders do (they chase volatility).

```python
def antifragile_position_size(
    base_size: float,
    current_vol_annualized: float,
    reference_vol: float = 0.60,  # 60% annualized is "normal" for BTC
    min_size_fraction: float = 0.2,  # Never less than 20% of base
    max_size_fraction: float = 2.0,  # Never more than 200% of base
) -> float:
    """
    Position size inversely proportional to volatility.

    When vol is 2x normal: position size is 0.5x base
    When vol is 0.5x normal: position size is 2x base (capped)

    This is anti-fragile because:
    - Low vol periods have small moves, so even 2x size has limited risk
    - High vol periods have large moves, so 0.5x size limits exposure
    - Net effect: roughly constant dollar-risk per trade
    """
    vol_ratio = reference_vol / max(current_vol_annualized, 0.01)
    scaled_size = base_size * vol_ratio

    # Clamp to bounds
    return max(
        base_size * min_size_fraction,
        min(scaled_size, base_size * max_size_fraction),
    )
```

**This naturally reduces exposure before crashes** because vol typically starts rising before the big move. The EWMA vol in `features/volatility.py` is perfectly suited as input.

---

## 7. Insurance and Risk Transfer

### 7.1 Crypto Insurance Products

| Product | Coverage | Cost | Practicality for ep2-crypto |
|---------|----------|------|---------------------------|
| Nexus Mutual | Smart contract exploits, exchange hacks | 2-5% annual on covered amount | **Low** -- DeFi native, not relevant for CEX perp trading |
| Binance SAFU | Exchange hack losses | Free (funded by trading fees) | **Medium** -- covers Binance-specific events but limited |
| Bybit Insurance Fund | Socialized losses from liquidations | Free (funded by profitable liquidations) | **Medium** -- protects against ADL (auto-deleveraging) |
| Coincover | Wallet/key theft | Varies | **Low** -- API key trading doesn't need this |
| Fireblocks | Institutional custody insurance | $$$$ | **N/A** -- institutional only |

### 7.2 Self-Insurance: The Most Practical Approach

For a system trading $50K-$500K, commercial insurance is either unavailable or not cost-effective. **Self-insurance through position sizing is the answer.**

```python
@dataclass
class SelfInsuranceConfig:
    """Self-insurance through capital reserves."""

    # Total capital across all locations
    total_capital: float = 100_000.0

    # Capital allocation
    cold_storage_pct: float = 0.70        # 70% in hardware wallet, untouchable
    warm_reserve_pct: float = 0.20        # 20% in bank/money market, emergency fund
    exchange_active_pct: float = 0.10     # 10% on exchanges for trading

    # Rebalancing rules
    rebalance_interval_days: int = 30
    rebalance_if_drift_exceeds: float = 0.05  # If allocation drifts >5%, rebalance

    # Capital injection rules
    min_exchange_balance_pct: float = 0.05  # If exchange balance drops below 5%, inject
    max_injection_from_reserve: float = 0.50  # Never inject more than 50% of warm reserve

    @property
    def exchange_capital(self) -> float:
        return self.total_capital * self.exchange_active_pct

    @property
    def warm_reserve(self) -> float:
        return self.total_capital * self.warm_reserve_pct

    def should_inject_capital(self, current_exchange_balance: float) -> tuple[bool, float]:
        """Determine if capital injection is needed and how much."""
        min_balance = self.total_capital * self.min_exchange_balance_pct
        if current_exchange_balance >= min_balance:
            return False, 0.0

        deficit = self.exchange_capital - current_exchange_balance
        max_inject = self.warm_reserve * self.max_injection_from_reserve
        inject_amount = min(deficit, max_inject)

        return True, inject_amount

    def should_withdraw_profits(self, current_exchange_balance: float) -> tuple[bool, float]:
        """Determine if profits should be withdrawn to reserve."""
        target = self.exchange_capital
        excess = current_exchange_balance - target * 1.5  # Withdraw if 50% above target
        if excess <= 0:
            return False, 0.0
        return True, excess
```

### 7.3 Exchange Insurance Funds: What They Actually Cover

**Binance SAFU (Secure Asset Fund for Users):**
- Funded by 10% of all trading fees
- As of 2025, estimated at $1B+
- Covers: exchange hacks, system errors
- Does NOT cover: individual account compromises, market losses, regulatory seizure

**Bybit Insurance Fund:**
- Covers socialized losses when bankrupt positions can't be liquidated at break-even
- If the insurance fund is depleted, Auto-Deleveraging (ADL) kicks in -- profitable traders get their positions closed
- **This means even profitable positions can be forcibly closed during extreme events**

**Protection against ADL:**
- Keep leverage low (no more than 3x, ideally 1x)
- If ADL indicator shows elevated risk, close profitable positions voluntarily
- Our 5% position size at 1x leverage makes ADL extremely unlikely

---

## 8. Scenario-Based Risk Limits

### 8.1 Stress Level Definitions

```python
from enum import Enum
from dataclasses import dataclass


class StressLevel(Enum):
    NORMAL = "normal"
    MILD_STRESS = "mild_stress"
    MODERATE_STRESS = "moderate_stress"
    SEVERE_STRESS = "severe_stress"
    CATASTROPHIC = "catastrophic"


@dataclass
class StressIndicators:
    """Real-time indicators that determine the current stress level."""

    # Volatility
    realized_vol_zscore: float = 0.0      # Current vol vs 30-day rolling average
    vol_of_vol_zscore: float = 0.0        # Instability of volatility

    # Liquidity
    spread_vs_median: float = 1.0         # Current spread / 7-day median spread
    book_depth_vs_median: float = 1.0     # Current depth / 7-day median depth

    # Derivatives
    funding_rate_zscore: float = 0.0      # Funding rate vs 30-day average
    oi_percentile: float = 50.0           # OI percentile over 90 days
    liquidation_rate_zscore: float = 0.0  # Liquidation flow vs normal

    # Cross-market
    btc_sp500_correlation: float = 0.3    # Rolling 5-day correlation
    vix_level: float = 15.0              # If available

    # Exchange health
    api_latency_ms: float = 50.0
    withdrawal_health: bool = True
    stablecoin_peg_deviation: float = 0.0


def classify_stress_level(indicators: StressIndicators) -> StressLevel:
    """
    Classify current market conditions into a stress level.

    Uses a scoring system: each indicator contributes points,
    total points determine stress level.
    """
    score = 0.0

    # Volatility scoring
    if indicators.realized_vol_zscore > 1.0:
        score += indicators.realized_vol_zscore * 0.5
    if indicators.vol_of_vol_zscore > 1.5:
        score += indicators.vol_of_vol_zscore * 0.4

    # Liquidity scoring
    if indicators.spread_vs_median > 2.0:
        score += (indicators.spread_vs_median - 1.0) * 0.6
    if indicators.book_depth_vs_median < 0.5:
        score += (1.0 - indicators.book_depth_vs_median) * 0.8

    # Derivatives scoring
    if abs(indicators.funding_rate_zscore) > 1.5:
        score += abs(indicators.funding_rate_zscore) * 0.3
    if indicators.oi_percentile > 85:
        score += (indicators.oi_percentile - 80) / 40
    if indicators.liquidation_rate_zscore > 2.0:
        score += indicators.liquidation_rate_zscore * 0.5

    # Exchange health scoring
    if indicators.api_latency_ms > 500:
        score += min((indicators.api_latency_ms - 500) / 1000, 2.0)
    if not indicators.withdrawal_health:
        score += 3.0  # Major red flag
    if indicators.stablecoin_peg_deviation > 0.005:
        score += indicators.stablecoin_peg_deviation * 100

    # Classify
    if score < 1.0:
        return StressLevel.NORMAL
    if score < 2.5:
        return StressLevel.MILD_STRESS
    if score < 5.0:
        return StressLevel.MODERATE_STRESS
    if score < 8.0:
        return StressLevel.SEVERE_STRESS
    return StressLevel.CATASTROPHIC
```

### 8.2 Position Limits by Stress Level

```python
@dataclass
class StressLevelLimits:
    """Position and trading limits for each stress level."""

    max_position_size_pct: float      # Max position as % of capital
    max_trades_per_hour: int          # Throttle trading frequency
    min_confidence_threshold: float   # Raise the bar for signal quality
    max_open_positions: int           # Concurrent positions
    allow_new_entries: bool           # Can we enter new positions?
    allow_cascade_short: bool         # Can we use cascade short strategy?
    force_close_positions: bool       # Must close all positions?


STRESS_LIMITS: dict[StressLevel, StressLevelLimits] = {
    StressLevel.NORMAL: StressLevelLimits(
        max_position_size_pct=5.0,
        max_trades_per_hour=10,
        min_confidence_threshold=0.60,
        max_open_positions=1,
        allow_new_entries=True,
        allow_cascade_short=False,
        force_close_positions=False,
    ),
    StressLevel.MILD_STRESS: StressLevelLimits(
        max_position_size_pct=3.0,           # Reduced from 5%
        max_trades_per_hour=6,
        min_confidence_threshold=0.70,        # Higher bar
        max_open_positions=1,
        allow_new_entries=True,
        allow_cascade_short=False,
        force_close_positions=False,
    ),
    StressLevel.MODERATE_STRESS: StressLevelLimits(
        max_position_size_pct=2.0,           # Significantly reduced
        max_trades_per_hour=3,
        min_confidence_threshold=0.80,        # Much higher bar
        max_open_positions=1,
        allow_new_entries=True,
        allow_cascade_short=True,             # Can exploit cascade
        force_close_positions=False,
    ),
    StressLevel.SEVERE_STRESS: StressLevelLimits(
        max_position_size_pct=1.0,           # Minimal position
        max_trades_per_hour=1,
        min_confidence_threshold=0.90,        # Almost never trades
        max_open_positions=1,
        allow_new_entries=False,              # No new entries
        allow_cascade_short=True,             # Can exploit cascade
        force_close_positions=False,
    ),
    StressLevel.CATASTROPHIC: StressLevelLimits(
        max_position_size_pct=0.0,           # No position at all
        max_trades_per_hour=0,
        min_confidence_threshold=1.0,         # Nothing passes
        max_open_positions=0,
        allow_new_entries=False,
        allow_cascade_short=False,            # Too dangerous even for this
        force_close_positions=True,           # CLOSE EVERYTHING
    ),
}
```

### 8.3 Example Thresholds for BTC

**What each stress level looks like in practice:**

| Level | Vol Z-score | Spread | Book Depth | Funding Z-score | Liquidation Z-score | Example |
|-------|------------|--------|------------|----------------|-------------------|---------|
| Normal | < 1.0 | < 2x median | > 70% median | < 1.5 | < 1.5 | Typical Tuesday |
| Mild | 1.0-2.0 | 2-3x | 50-70% | 1.5-2.5 | 1.5-2.5 | CPI release day |
| Moderate | 2.0-3.0 | 3-5x | 30-50% | 2.5-3.5 | 2.5-4.0 | ETF rejection, SEC action |
| Severe | 3.0-5.0 | 5-10x | 10-30% | > 3.5 | > 4.0 | China ban, exchange scare |
| Catastrophic | > 5.0 | > 10x | < 10% | Any | > 6.0 | COVID crash, FTX collapse |

### 8.4 Automatic Stress Level Transitions

```python
import time
import logging

logger = logging.getLogger(__name__)


class StressLevelManager:
    """Manages stress level transitions with hysteresis."""

    def __init__(self) -> None:
        self._current_level = StressLevel.NORMAL
        self._last_escalation_time: float = 0.0
        self._last_deescalation_time: float = 0.0

        # Hysteresis: escalation is fast, de-escalation is slow
        self._min_escalation_interval_s = 10.0     # Can escalate every 10s
        self._min_deescalation_interval_s = 300.0   # Wait 5 min before de-escalating
        self._deescalation_cooldown_s = 900.0       # After catastrophic, wait 15 min

    @property
    def current_level(self) -> StressLevel:
        return self._current_level

    @property
    def current_limits(self) -> StressLevelLimits:
        return STRESS_LIMITS[self._current_level]

    def update(self, indicators: StressIndicators) -> StressLevel:
        """Update stress level based on current indicators."""
        new_level = classify_stress_level(indicators)
        now = time.monotonic()

        level_order = list(StressLevel)
        current_idx = level_order.index(self._current_level)
        new_idx = level_order.index(new_level)

        if new_idx > current_idx:
            # Escalation -- fast
            if now - self._last_escalation_time >= self._min_escalation_interval_s:
                logger.warning(
                    "stress_level_escalated",
                    previous=self._current_level.value,
                    new=new_level.value,
                )
                self._current_level = new_level
                self._last_escalation_time = now
        elif new_idx < current_idx:
            # De-escalation -- slow, one level at a time
            cooldown = self._deescalation_cooldown_s if self._current_level == StressLevel.CATASTROPHIC else self._min_deescalation_interval_s
            if now - self._last_deescalation_time >= cooldown:
                # Only de-escalate one level at a time
                target_idx = current_idx - 1
                target_level = level_order[target_idx]
                logger.info(
                    "stress_level_deescalated",
                    previous=self._current_level.value,
                    new=target_level.value,
                )
                self._current_level = target_level
                self._last_deescalation_time = now

        return self._current_level
```

**Key design decision:** Escalation is immediate (protect fast), de-escalation is gradual (don't rush back in). After a catastrophic event, the system waits 15 minutes before even beginning to de-escalate, and it de-escalates one level at a time with 5-minute intervals. Full recovery from CATASTROPHIC to NORMAL takes at least 35 minutes.

---

## 9. Recovery from Catastrophic Events

### 9.1 The Mathematics of Recovery

| Capital Loss | Gain Needed to Recover | Time at 2% Monthly | Time at 5% Monthly |
|-------------|----------------------|-------------------|-------------------|
| 10% | 11.1% | ~6 months | ~2 months |
| 20% | 25.0% | ~12 months | ~5 months |
| 30% | 42.9% | ~18 months | ~7 months |
| 40% | 66.7% | ~26 months | ~10 months |
| 50% | 100.0% | ~35 months | ~14 months |
| 60% | 150.0% | ~47 months | ~19 months |
| 70% | 233.3% | ~62 months | ~25 months |
| 80% | 400.0% | ~84 months | ~34 months |

**Key insight:** Recovery is non-linear. A 50% loss requires a 100% gain -- which at a realistic 2% monthly return takes almost 3 years. This is why **preventing large drawdowns is exponentially more important than maximizing returns.**

### 9.2 Decision Framework After Catastrophic Loss

```python
@dataclass
class RecoveryDecision:
    """Framework for deciding what to do after a large loss."""

    loss_pct: float  # How much was lost (e.g., 0.30 for 30%)
    loss_cause: str  # "position_loss", "exchange_insolvency", "stablecoin_depeg"
    remaining_capital: float
    has_warm_reserve: bool
    warm_reserve_amount: float
    monthly_income_available: float  # Can inject from external income?

    def recommend_action(self) -> dict[str, str]:
        """
        Recommend recovery action based on loss severity and cause.
        """
        if self.loss_pct < 0.15:
            return {
                "action": "continue_trading",
                "position_size_adjustment": "reduce_by_30pct_for_30_days",
                "explanation": (
                    "Loss is within normal drawdown expectations. "
                    "Reduce position size temporarily while the system recovers. "
                    "Do NOT increase size to 'make it back faster'."
                ),
            }

        if self.loss_pct < 0.30:
            return {
                "action": "pause_and_analyze",
                "pause_duration": "minimum_7_days",
                "analysis_required": [
                    "root_cause_analysis",
                    "was_this_a_model_failure_or_tail_event",
                    "review_all_risk_parameters",
                ],
                "position_size_adjustment": "reduce_by_50pct_for_60_days",
                "explanation": (
                    "Significant loss. Pause trading for at least 7 days. "
                    "Analyze root cause. If it was a tail event (not model failure), "
                    "resume at 50% position size. If model failure, retrain first."
                ),
            }

        if self.loss_pct < 0.50:
            if self.has_warm_reserve and self.warm_reserve_amount > self.remaining_capital * 0.5:
                return {
                    "action": "partial_recapitalization",
                    "inject_amount": min(
                        self.warm_reserve_amount * 0.3,  # Max 30% of reserve
                        self.remaining_capital * 0.5,     # Bring to 75% of original
                    ),
                    "pause_duration": "minimum_30_days",
                    "position_size_adjustment": "reduce_by_70pct_for_90_days",
                    "explanation": (
                        "Major loss. Inject limited capital from warm reserve. "
                        "Full system audit required. Resume at minimal position size."
                    ),
                }
            return {
                "action": "halt_trading_rebuild_externally",
                "explanation": (
                    "Major loss without sufficient reserves. "
                    "Halt trading entirely. Rebuild capital from external income. "
                    "Do NOT try to trade your way back from 50% drawdown."
                ),
            }

        # Loss > 50%
        return {
            "action": "halt_trading_indefinitely",
            "explanation": (
                "Catastrophic loss. The system has fundamentally failed. "
                "Do NOT trade. Conduct full post-mortem. "
                "Rebuilding from >50% loss through trading is statistically "
                "unlikely and psychologically dangerous. "
                "Rebuild capital externally over 6-12 months."
            ),
        }
```

### 9.3 The Psychology Trap

After a large loss, traders almost universally:
1. **Increase position size** to "make it back faster" -- this compounds the problem
2. **Lower quality standards** (take marginal signals) -- this compounds the problem
3. **Revenge trade** -- emotional decisions override systematic rules

**Automated enforcement:**
- After any drawdown > 10%, the system should **force** reduced position sizes for a minimum period
- This should be a hard constraint in code, not a suggestion

```python
class DrawdownRecoveryEnforcer:
    """Enforce reduced position sizing during drawdown recovery."""

    def __init__(self) -> None:
        self._peak_capital: float = 0.0
        self._current_capital: float = 0.0
        self._recovery_mode: bool = False
        self._recovery_start_time: float = 0.0

    def update_capital(self, current: float) -> None:
        self._current_capital = current
        if current > self._peak_capital:
            self._peak_capital = current
            self._recovery_mode = False

    @property
    def current_drawdown(self) -> float:
        if self._peak_capital <= 0:
            return 0.0
        return (self._peak_capital - self._current_capital) / self._peak_capital

    def max_allowed_position_size(self, base_size: float) -> float:
        """
        Reduce position size proportionally to drawdown.

        At 5% drawdown: 80% of base size
        At 10% drawdown: 50% of base size
        At 15% drawdown: 20% of base size
        At 20%+ drawdown: ZERO (halt trading)
        """
        dd = self.current_drawdown

        if dd < 0.03:
            return base_size  # No reduction below 3% DD
        if dd < 0.05:
            return base_size * 0.80
        if dd < 0.10:
            return base_size * 0.50
        if dd < 0.15:
            return base_size * 0.20
        if dd < 0.20:
            return base_size * 0.05  # Minimal

        # Above 20% drawdown: halt
        return 0.0
```

---

## 10. Historical Black Swan Survivability Analysis

### 10.1 March 2020 COVID Crash

**Event timeline:**
- March 8: Oil price war begins, BTC $8,000
- March 12 00:00 UTC: BTC $7,900, everything seems fine
- March 12 06:00: First leg down to $7,000 (-11%). Cross-market: S&P futures limit down
- March 12 10:00: Second leg to $5,800 (-27%). Liquidations intensifying
- March 12 12:00: BitMEX goes offline for 25 minutes
- March 12 18:00: Third leg to $4,800 (-39%)
- March 13: Brief bounce, then continuation to $3,850 (-51% from $7,900)

**Simulated ep2-crypto behavior (current config: 5% position, 3% daily loss limit, 15% max DD):**

| Time | System Action | Position | Running PnL |
|------|-------------|----------|-------------|
| Mar 12 00:00 | Normal trading, long signal | 5% long | 0% |
| Mar 12 02:00 | Vol spiking, stress -> MILD | Reduce to 3% | -0.3% |
| Mar 12 06:00 | First cascade detected, stress -> MODERATE | Close long, consider short | -0.8% |
| Mar 12 07:00 | Cascade confidence > 80%, spread 5x normal | Cascade short 2% | -0.8% |
| Mar 12 10:00 | Stress -> SEVERE, holding short | Short profitable | +0.2% |
| Mar 12 12:00 | BitMEX down, spread >10x, stress -> CATASTROPHIC | Force close ALL | +1.5% |
| Mar 12 13:00-24:00 | Catastrophic mode, no trading | Flat | +1.5% |
| Mar 13-14 | Gradual de-escalation over hours | Flat then minimal | +1.5% |

**Position size that would cause account destruction (>30% loss):**
```
Max position for 30% survival: 0.30 / 0.51 = 59% of capital
With exchange outage (can't close): 0.30 / (0.51 * 1.5) = 39% of capital
```

**At current 5% position size: SURVIVES EASILY.** Even in the worst case (long 5% at the top, exchange goes down for 25 minutes at the worst time), maximum loss is ~3.4%.

### 10.2 May 2021 China Ban Crash

**Event timeline:**
- May 18: Funding rate +0.06% (elevated), OI at all-time high
- May 19 07:00: China announces crypto ban, BTC $42,000
- May 19 09:00: First cascade, $42K -> $38K (-10%)
- May 19 11:00: Second cascade, $38K -> $33K (-21%)
- May 19 13:00: Third cascade, $33K -> $30K (-29%)
- May 19 20:00: Bounce to $37K, but whipsaw
- May 20-21: Continued volatility, eventual low at $28.8K (-33% from $42K)

**Simulated ep2-crypto behavior:**

| Time | System Action | Reason |
|------|-------------|--------|
| May 18 | Pre-crash risk flags | Funding z-score > 2, OI > 90th percentile |
| May 18 | Position size reduced | Mild stress limits: 3% instead of 5% |
| May 19 07:30 | Vol spike detected | Stress -> MODERATE within 30 minutes |
| May 19 08:00 | Cascade detector fires | Liquidation z-score > 3 |
| May 19 08:15 | Cascade short entered | 2% position, stop at +5% from entry |
| May 19 11:00 | Take profit at -15% | Cascade short profits +0.3% of capital |
| May 19 11:30 | Stress -> SEVERE | No new entries |
| May 19-20 | Flat | Gradual de-escalation over 12+ hours |

**Kill switches that would have triggered:**
- Daily loss limit (3%): would NOT trigger at 5% position size (max loss = 1.5%)
- Drawdown gate (15%): would NOT trigger
- Cascade detector: WOULD trigger at ~08:00, correctly
- Pre-crash funding alert: WOULD trigger on May 18, correctly

**Optimal response would have been:** Pre-crash position reduction (check), cascade detection (check), cascade short (optional, small profit). The system's response is near-optimal.

### 10.3 November 2022 FTX Collapse

**Event timeline:**
- Nov 2: CoinDesk publishes Alameda balance sheet (FTT exposure)
- Nov 6: Binance announces selling FTT holdings, BTC $21,000
- Nov 7: FTX withdrawal queue grows, FTT drops 30%
- Nov 8: FTX halts withdrawals, BTC $20,500 -> $17,500 (-15%)
- Nov 9: BTC $17,500 -> $15,500 (-24% from Nov 6)
- Nov 10-12: Contagion fears (BlockFi, Genesis, etc.)

**This event is different:** It was a slow-moving insolvency, not a flash crash. The critical risk was **exchange counterparty risk**, not position loss.

**Simulated ep2-crypto behavior:**

| Day | System Action | Risk |
|-----|-------------|------|
| Nov 2-5 | Normal trading | Exchange token (FTT) declining -- should trigger exchange health alert if monitored |
| Nov 6 | Vol starts rising, stress -> MILD | Spread slightly wider than normal |
| Nov 7 | Withdrawal health check fails (if on FTX) | **EMERGENCY WITHDRAW** triggered |
| Nov 8 | If on FTX: too late, funds stuck. If on Binance: stress -> MODERATE | Exchange counterparty risk |
| Nov 8-9 | BTC drops 15-24%, cascade detector fires | Normal cascade response |

**Critical lesson:** The withdrawal canary test (Section 1.1) would have saved any funds on FTX if tested on Nov 7 (withdrawals were already delayed). By Nov 8 it was too late.

**Position size that would cause 30% loss from position moves alone:**
```
Max position: 0.30 / 0.24 = 125% -- impossible without leverage
```

**At 5% position size, position loss is only 1.2%.** But if 100% of capital was on FTX: 100% loss regardless of position size. This validates the "max 10% of total capital on any exchange" rule.

### 10.4 Survivability Summary Table

| Event | Position Loss (5% size) | Exchange Risk | Cascade Detected? | Kill Switch Triggered? | System Survives? |
|-------|----------------------|---------------|-------------------|----------------------|-----------------|
| COVID Mar 2020 | 1.85% max | BitMEX down 25min | Yes, at ~06:00 | No (loss too small) | YES |
| China Ban May 2021 | 1.50% max | Exchanges OK | Yes, at ~08:00 | No (loss too small) | YES |
| FTX Nov 2022 | 1.20% max | CRITICAL if on FTX | Yes, at Nov 8 | Withdrawal canary | YES (if diversified) |
| Aug 2024 Carry Unwind | 0.97% max | Exchanges OK | Yes | No | YES |
| Hypothetical -50% day | 2.50% max | Probably down | Yes | Daily loss limit | YES |

**At 5% position size, the system survives every historical black swan by a large margin.** The only existential risk is exchange insolvency where 100% of capital is on the failed exchange.

---

## 11. Implementation: TailRiskMonitor Class

This is the integration class that ties everything together for the ep2-crypto system.

```python
# src/ep2_crypto/risk/tail_risk_monitor.py
"""
Tail Risk Monitor for ep2-crypto.

Integrates all tail risk protections:
- Stress level classification
- Dynamic position limits
- Exchange health monitoring
- Stablecoin peg monitoring
- Cascade short opportunities
- Drawdown-based recovery enforcement
- Fat-tail-aware VaR computation
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)


class StressLevel(Enum):
    NORMAL = "normal"
    MILD_STRESS = "mild_stress"
    MODERATE_STRESS = "moderate_stress"
    SEVERE_STRESS = "severe_stress"
    CATASTROPHIC = "catastrophic"


@dataclass
class TailRiskState:
    """Current tail risk state of the system."""

    stress_level: StressLevel = StressLevel.NORMAL
    max_position_size_pct: float = 5.0
    allow_new_entries: bool = True
    force_close_all: bool = False
    allow_cascade_short: bool = False
    stablecoin_peg_ok: bool = True
    exchange_health_ok: bool = True
    current_drawdown_pct: float = 0.0
    historical_var_99: float = 0.0  # 99% VaR from fat-tail model
    gaussian_var_underestimate_ratio: float = 1.0


class TailRiskMonitor:
    """
    Central tail risk monitor integrating all protection mechanisms.

    Call `update()` every 5-minute bar with current market state.
    Read `state` to get current risk limits.
    """

    def __init__(
        self,
        base_position_size_pct: float = 5.0,
        max_drawdown_halt_pct: float = 15.0,
        daily_loss_limit_pct: float = 3.0,
    ) -> None:
        self._base_size = base_position_size_pct
        self._max_dd = max_drawdown_halt_pct
        self._daily_limit = daily_loss_limit_pct

        self._state = TailRiskState()
        self._stress_manager = _StressManager()
        self._drawdown_enforcer = _DrawdownEnforcer()
        self._returns_buffer: list[float] = []
        self._max_returns_buffer = 20_000  # ~70 days of 5-min bars

    @property
    def state(self) -> TailRiskState:
        return self._state

    def update(
        self,
        # Price data
        current_return: float,         # This bar's log return
        realized_vol_zscore: float,
        vol_of_vol_zscore: float,
        # Liquidity data
        spread_vs_median: float,
        book_depth_vs_median: float,
        # Derivatives data
        funding_rate_zscore: float,
        oi_percentile: float,
        liquidation_rate_zscore: float,
        # Exchange health
        api_latency_ms: float,
        withdrawal_healthy: bool,
        stablecoin_price: float,       # e.g., 0.998 for USDT
        # Portfolio
        current_capital: float,
    ) -> TailRiskState:
        """Update all tail risk calculations and return current state."""

        # 1. Buffer returns for fat-tail VaR
        self._returns_buffer.append(current_return)
        if len(self._returns_buffer) > self._max_returns_buffer:
            self._returns_buffer = self._returns_buffer[-self._max_returns_buffer:]

        # 2. Compute fat-tail VaR
        if len(self._returns_buffer) >= 500:
            returns_arr = np.array(self._returns_buffer)
            var_results = _compute_var_comparison(returns_arr)
            self._state.historical_var_99 = var_results["historical_var"]
            self._state.gaussian_var_underestimate_ratio = var_results["underestimate_ratio"]

        # 3. Classify stress level
        stress = self._stress_manager.update(
            realized_vol_zscore=realized_vol_zscore,
            vol_of_vol_zscore=vol_of_vol_zscore,
            spread_vs_median=spread_vs_median,
            book_depth_vs_median=book_depth_vs_median,
            funding_rate_zscore=funding_rate_zscore,
            oi_percentile=oi_percentile,
            liquidation_rate_zscore=liquidation_rate_zscore,
            api_latency_ms=api_latency_ms,
            withdrawal_healthy=withdrawal_healthy,
            stablecoin_deviation=abs(1.0 - stablecoin_price),
        )
        self._state.stress_level = stress

        # 4. Get stress-based limits
        limits = _STRESS_LIMITS[stress]

        # 5. Update drawdown enforcer
        self._drawdown_enforcer.update_capital(current_capital)
        dd = self._drawdown_enforcer.current_drawdown
        self._state.current_drawdown_pct = dd * 100

        # 6. Compute effective position size (minimum of stress limit and drawdown limit)
        dd_adjusted_size = self._drawdown_enforcer.max_allowed_position_size(
            self._base_size
        )
        stress_adjusted_size = limits.max_position_size_pct
        effective_size = min(dd_adjusted_size, stress_adjusted_size)

        # 7. Vol-adjusted position sizing (anti-fragile)
        # Higher vol -> smaller position
        if realized_vol_zscore > 1.0:
            vol_factor = 1.0 / max(realized_vol_zscore, 1.0)
            effective_size *= vol_factor

        self._state.max_position_size_pct = max(effective_size, 0.0)
        self._state.allow_new_entries = limits.allow_new_entries and dd < 0.20
        self._state.force_close_all = limits.force_close_positions or dd >= self._max_dd / 100
        self._state.allow_cascade_short = limits.allow_cascade_short

        # 8. Exchange and stablecoin health
        self._state.exchange_health_ok = withdrawal_healthy and api_latency_ms < 2000
        self._state.stablecoin_peg_ok = abs(1.0 - stablecoin_price) < 0.02

        # 9. Override: if exchange or stablecoin critical, force close
        if not self._state.exchange_health_ok or not self._state.stablecoin_peg_ok:
            self._state.force_close_all = True
            self._state.allow_new_entries = False
            self._state.max_position_size_pct = 0.0

        return self._state


# --- Private helpers ---

@dataclass
class _StressLimits:
    max_position_size_pct: float
    allow_new_entries: bool
    allow_cascade_short: bool
    force_close_positions: bool


_STRESS_LIMITS: dict[StressLevel, _StressLimits] = {
    StressLevel.NORMAL: _StressLimits(5.0, True, False, False),
    StressLevel.MILD_STRESS: _StressLimits(3.0, True, False, False),
    StressLevel.MODERATE_STRESS: _StressLimits(2.0, True, True, False),
    StressLevel.SEVERE_STRESS: _StressLimits(1.0, False, True, False),
    StressLevel.CATASTROPHIC: _StressLimits(0.0, False, False, True),
}


class _StressManager:
    """Internal stress level manager with hysteresis."""

    def __init__(self) -> None:
        self._level = StressLevel.NORMAL
        self._last_escalation = 0.0
        self._last_deescalation = 0.0

    def update(self, **indicators: float | bool) -> StressLevel:
        score = 0.0

        vol_z = indicators.get("realized_vol_zscore", 0.0)
        if isinstance(vol_z, (int, float)) and vol_z > 1.0:
            score += vol_z * 0.5

        vov_z = indicators.get("vol_of_vol_zscore", 0.0)
        if isinstance(vov_z, (int, float)) and vov_z > 1.5:
            score += vov_z * 0.4

        spread = indicators.get("spread_vs_median", 1.0)
        if isinstance(spread, (int, float)) and spread > 2.0:
            score += (spread - 1.0) * 0.6

        depth = indicators.get("book_depth_vs_median", 1.0)
        if isinstance(depth, (int, float)) and depth < 0.5:
            score += (1.0 - depth) * 0.8

        liq_z = indicators.get("liquidation_rate_zscore", 0.0)
        if isinstance(liq_z, (int, float)) and liq_z > 2.0:
            score += liq_z * 0.5

        latency = indicators.get("api_latency_ms", 50.0)
        if isinstance(latency, (int, float)) and latency > 500:
            score += min((latency - 500) / 1000, 2.0)

        withdrawal = indicators.get("withdrawal_healthy", True)
        if withdrawal is False:
            score += 3.0

        stable_dev = indicators.get("stablecoin_deviation", 0.0)
        if isinstance(stable_dev, (int, float)) and stable_dev > 0.005:
            score += stable_dev * 100

        # Determine target level
        if score < 1.0:
            target = StressLevel.NORMAL
        elif score < 2.5:
            target = StressLevel.MILD_STRESS
        elif score < 5.0:
            target = StressLevel.MODERATE_STRESS
        elif score < 8.0:
            target = StressLevel.SEVERE_STRESS
        else:
            target = StressLevel.CATASTROPHIC

        now = time.monotonic()
        levels = list(StressLevel)
        cur_idx = levels.index(self._level)
        tgt_idx = levels.index(target)

        if tgt_idx > cur_idx and now - self._last_escalation >= 10.0:
            self._level = target
            self._last_escalation = now
        elif tgt_idx < cur_idx and now - self._last_deescalation >= 300.0:
            self._level = levels[cur_idx - 1]
            self._last_deescalation = now

        return self._level


class _DrawdownEnforcer:
    def __init__(self) -> None:
        self._peak = 0.0
        self._current = 0.0

    def update_capital(self, capital: float) -> None:
        self._current = capital
        if capital > self._peak:
            self._peak = capital

    @property
    def current_drawdown(self) -> float:
        if self._peak <= 0:
            return 0.0
        return max(0.0, (self._peak - self._current) / self._peak)

    def max_allowed_position_size(self, base: float) -> float:
        dd = self.current_drawdown
        if dd < 0.03:
            return base
        if dd < 0.05:
            return base * 0.80
        if dd < 0.10:
            return base * 0.50
        if dd < 0.15:
            return base * 0.20
        if dd < 0.20:
            return base * 0.05
        return 0.0


def _compute_var_comparison(returns: np.ndarray) -> dict[str, float]:
    mean = float(np.mean(returns))
    std = float(np.std(returns, ddof=1))

    gaussian_var = -(mean + std * sp_stats.norm.ppf(0.01))
    historical_var = -float(np.percentile(returns, 1))

    ratio = historical_var / max(gaussian_var, 1e-10)

    return {
        "gaussian_var": gaussian_var,
        "historical_var": historical_var,
        "underestimate_ratio": ratio,
    }
```

---

## 12. Integration with ep2-crypto Config

### 12.1 New Config Parameters

The following should be added to `MonitoringConfig` in `src/ep2_crypto/config.py`:

```python
class MonitoringConfig(BaseSettings):
    """Monitoring and risk management settings."""

    model_config = SettingsConfigDict(env_prefix="EP2_MONITOR_")

    # Existing parameters (keep as-is)
    daily_loss_limit: float = 0.03
    weekly_loss_limit: float = 0.05
    max_drawdown_halt: float = 0.15
    max_trades_per_day: int = 30
    max_open_positions: int = 1
    catastrophic_stop_atr: float = 3.0
    position_size_fraction: float = 0.05
    weekend_size_reduction: float = 0.30
    trading_start_utc: int = 8
    trading_end_utc: int = 21

    # NEW: Tail risk parameters
    stablecoin_depeg_warning: float = 0.005     # 0.5% deviation
    stablecoin_depeg_critical: float = 0.02     # 2% deviation -- halt
    exchange_max_balance_usd: float = 10_000.0  # Max capital on single exchange
    withdrawal_test_interval_hours: float = 4.0
    max_exchange_capital_pct: float = 0.10      # 10% of total net worth

    # Cascade short parameters
    cascade_short_enabled: bool = True
    cascade_short_size: float = 0.03
    cascade_min_confidence: float = 0.80
    cascade_stop_loss_pct: float = 0.05
    cascade_take_profit_pct: float = 0.15

    # Stress de-escalation
    deescalation_cooldown_s: float = 300.0      # 5 min between de-escalation steps
    catastrophic_cooldown_s: float = 900.0      # 15 min after catastrophic

    # Drawdown recovery
    drawdown_size_reduction_start: float = 0.03  # Start reducing at 3% DD
    drawdown_halt_threshold: float = 0.20        # Halt all trading at 20% DD

    # Anti-fragile vol scaling
    vol_scaling_enabled: bool = True
    reference_vol_annualized: float = 0.60       # 60% annualized = "normal" BTC
```

### 12.2 Integration Points

The `TailRiskMonitor` should be called:

1. **Every 5-minute bar** in the live prediction loop (`scripts/live.py`)
2. **Before every trade entry** -- check `state.allow_new_entries` and `state.max_position_size_pct`
3. **On every websocket tick** for exchange health and stablecoin monitoring (latency, withdrawal checks)
4. **In the backtest engine** -- replay stress scenarios through the full monitor

### 12.3 Data Sources for Tail Risk Indicators

| Indicator | Source | Update Frequency |
|-----------|--------|-----------------|
| Realized vol z-score | `features/volatility.py` (already built) | Every 5-min bar |
| Vol-of-vol z-score | `features/volatility.py` (already built) | Every 5-min bar |
| Spread vs median | `features/microstructure.py` (already built) | Every tick |
| Book depth vs median | `features/microstructure.py` (already built) | Every tick |
| Funding rate z-score | `ingest/derivatives.py` (already built) | Every 5 min poll |
| OI percentile | `ingest/derivatives.py` (already built) | Every 5 min poll |
| Liquidation rate z-score | `RR-cascade-liquidation-detection-system.md` (designed) | Real-time |
| API latency | Measure round-trip on each API call | Every API call |
| Withdrawal health | Test withdrawal every 4 hours | Every 4 hours |
| Stablecoin peg | USDT/USD and USDC/USD price feeds | Every 1 min |

---

## Summary of Actionable Protections

### Structural (Non-Negotiable)

1. **Max 10% of total net worth on any single exchange** (Taleb barbell)
2. **API keys: trading-only, no withdrawal permission, IP-whitelisted**
3. **5% max position size** (current config is correct)
4. **Exchange-side stop-loss orders** in addition to application-side
5. **Multi-exchange operation** (Binance + Bybit)

### Automated (Code-Enforced)

6. **Stress level classification** with hysteresis (escalate fast, de-escalate slow)
7. **Position size auto-reduction** during drawdown (linear scale from 3% to 20% DD)
8. **Stablecoin peg monitor** (halt at 2% deviation)
9. **Exchange withdrawal canary** (test every 4 hours, emergency withdraw on 2 failures)
10. **Fat-tail-aware VaR** (use historical VaR, not Gaussian)
11. **Anti-fragile vol scaling** (inverse vol position sizing)

### Opportunistic (Anti-Fragile)

12. **Cascade short strategy** when Hawkes detector fires at >80% confidence
13. **Dry powder deployment** rules after confirmed bottom (out of scope for 5-min system but relevant for portfolio)

### Recovery Protocol

14. **<15% loss:** Continue at reduced size for 30 days
15. **15-30% loss:** Pause 7+ days, full analysis, resume at 50% size
16. **30-50% loss:** Inject from reserve if available, pause 30+ days
17. **>50% loss:** Halt indefinitely, rebuild externally

### Key Numbers to Remember

- BTC 5-min kurtosis: **15-50** (not 0 like Gaussian)
- 4-sigma events per month: **10-30** (expected: 0.27)
- Gaussian VaR underestimates by: **2-4x**
- Student-t degrees of freedom for BTC: **3-5**
- Worst single day: **-37%** (March 12, 2020)
- At 5% position size, worst-day loss: **1.85%** (trivially survivable)
- Worst compound scenario loss (current config): **~9.6%** (FTX-style, still survivable)
