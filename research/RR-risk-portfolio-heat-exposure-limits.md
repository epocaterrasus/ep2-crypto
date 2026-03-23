# RR: Portfolio Heat, Exposure Limits, and Risk Budgeting

**For**: ep2-crypto 5-minute BTC perpetual futures system
**Capital range**: $10K-$100K
**Scope**: Risk framework for Sprint 9 (risk management engine) and beyond

---

## 1. Portfolio Heat

### Concept

Portfolio heat measures the total capital at risk across all open positions at any moment. Unlike position limits (which count positions) or gross exposure (which measures notional), heat measures **how much you actually lose if every stop is hit simultaneously**.

```
Portfolio Heat = SUM( position_size_i × distance_to_stop_i ) / total_capital
```

For a single-position BTC system:

```
Heat = (position_notional × stop_distance_pct) / total_capital
```

Where:
- `position_notional` = position size in USD
- `stop_distance_pct` = (entry_price - stop_price) / entry_price (absolute value)
- `total_capital` = total account equity

### Why Heat Matters More Than Position Limits

Position limits say "max 1 position." Heat says "max X% at risk." The difference:

| Scenario | Position Count | Gross Exposure | Heat |
|----------|---------------|----------------|------|
| 1 BTC long, stop 0.5% away | 1 | $60K | 0.5% of capital |
| 1 BTC long, stop 3% away | 1 | $60K | 3.0% of capital |
| 1 BTC long, no stop | 1 | $60K | **100% of capital** |

Same position count, same gross exposure, wildly different risk. Heat captures what matters: the actual loss if things go wrong.

### Heat Limit Formula

```python
max_heat_pct = 0.02  # Maximum 2% of capital at risk

# Before entering a trade:
proposed_heat = (position_notional * stop_distance_pct) / total_capital

# Check against existing heat
total_heat = current_heat + proposed_heat
if total_heat > max_heat_pct:
    # Reduce position size to fit within heat budget
    available_heat = max_heat_pct - current_heat
    max_notional = (available_heat * total_capital) / stop_distance_pct
```

### Recommended Heat Limits for ep2-crypto

| Capital Level | Conservative | Moderate | Aggressive |
|--------------|-------------|----------|------------|
| $10K | 1.0% ($100) | 2.0% ($200) | 3.0% ($300) |
| $50K | 1.5% ($750) | 2.5% ($1,250) | 4.0% ($2,000) |
| $100K | 2.0% ($2,000) | 3.0% ($3,000) | 5.0% ($5,000) |

**Recommendation for ep2-crypto: 2% maximum heat.** Rationale:
- With a 3-ATR catastrophic stop on 5-min BTC, stop distance is typically 0.3-1.5%
- At 2% heat and 1% stop distance: max position = 2x capital (200% notional)
- At 2% heat and 0.5% stop distance: max position = 4x capital (400% notional)
- The position sizer (quarter-Kelly capped at 5% of capital) will bind before heat does in most cases
- Heat becomes the **binding constraint during volatile markets** when stops widen

### Interaction with Existing Risk Parameters

The current config has `position_size_fraction: float = 0.05` (5% of capital per trade). With heat:

```python
# The effective position size is the MINIMUM of:
# 1. Quarter-Kelly sizing
# 2. Max position cap (5% of capital)
# 3. Heat-constrained size

def compute_position_size(
    capital: float,
    kelly_fraction: float,
    confidence: float,
    stop_distance_pct: float,
    current_heat: float,
    max_heat: float = 0.02,
    max_position_frac: float = 0.05,
) -> float:
    """Compute position size respecting all constraints."""
    # Quarter-Kelly
    kelly_size = 0.25 * kelly_fraction * confidence * capital

    # Max position cap
    cap_size = max_position_frac * capital

    # Heat constraint
    available_heat = max(0.0, max_heat - current_heat)
    heat_size = (available_heat * capital) / max(stop_distance_pct, 1e-8)

    return min(kelly_size, cap_size, heat_size)
```

### Implementation for `risk/position_tracker.py`

```python
from dataclasses import dataclass, field
from typing import Optional
import structlog

logger = structlog.get_logger()


@dataclass
class PositionState:
    """Tracks a single open position with heat calculation."""
    symbol: str
    side: str  # "long" or "short"
    entry_price: float
    size: float  # in base currency (BTC)
    notional: float  # size * entry_price
    stop_price: float
    entry_time: float  # unix timestamp
    unrealized_pnl: float = 0.0
    bars_held: int = 0
    max_favorable: float = 0.0  # Maximum favorable excursion
    max_adverse: float = 0.0    # Maximum adverse excursion

    @property
    def stop_distance_pct(self) -> float:
        """Distance from entry to stop as a percentage."""
        return abs(self.entry_price - self.stop_price) / self.entry_price

    @property
    def heat(self) -> float:
        """Risk in USD if stop is hit."""
        return self.notional * self.stop_distance_pct


@dataclass
class PortfolioHeatTracker:
    """Tracks total portfolio heat across all positions."""
    capital: float
    max_heat_pct: float = 0.02  # 2% default
    positions: dict[str, PositionState] = field(default_factory=dict)

    @property
    def total_heat_usd(self) -> float:
        return sum(p.heat for p in self.positions.values())

    @property
    def total_heat_pct(self) -> float:
        return self.total_heat_usd / self.capital if self.capital > 0 else 0.0

    @property
    def available_heat_usd(self) -> float:
        return max(0.0, (self.max_heat_pct * self.capital) - self.total_heat_usd)

    def max_notional_for_stop(self, stop_distance_pct: float) -> float:
        """Maximum position notional given current heat and proposed stop."""
        if stop_distance_pct <= 0:
            return 0.0
        return self.available_heat_usd / stop_distance_pct

    def can_add_position(self, notional: float, stop_distance_pct: float) -> bool:
        proposed_heat = notional * stop_distance_pct
        return (self.total_heat_usd + proposed_heat) <= (self.max_heat_pct * self.capital)
```

### Monitoring

```python
# Metrics to track and log on every bar:
heat_metrics = {
    "portfolio_heat_pct": tracker.total_heat_pct,
    "portfolio_heat_usd": tracker.total_heat_usd,
    "available_heat_usd": tracker.available_heat_usd,
    "heat_utilization": tracker.total_heat_pct / tracker.max_heat_pct,
}

# Alerts:
# - heat_utilization > 0.80: WARNING - approaching heat limit
# - heat_utilization > 0.95: CRITICAL - near heat ceiling
# - heat_pct > max_heat_pct: ERROR - heat breach (should never happen)
```

---

## 2. Gross and Net Exposure Limits

### Definitions

```
Gross Exposure = SUM( |position_notional_i| ) / total_capital
Net Exposure   = SUM( signed_position_notional_i ) / total_capital
```

For a single long BTC position of $5K with $50K capital:
- Gross = $5K / $50K = 10%
- Net = +$5K / $50K = +10% (positive = net long)

For a long BTC $5K + short ETH $3K:
- Gross = ($5K + $3K) / $50K = 16%
- Net = ($5K - $3K) / $50K = +4%

### For Single-Asset BTC System

When trading only BTC with max 1 position: **gross = |net|**. The exposure limit simplifies to a maximum notional cap.

```
Max Position Notional = max_gross_exposure * total_capital
```

### Recommended Exposure Limits

| Parameter | Conservative | Moderate | For ep2-crypto |
|-----------|-------------|----------|----------------|
| Max gross exposure | 100% | 150% | **100%** |
| Max net exposure | 100% | 100% | **100%** |
| Max per-asset gross | 100% | 100% | **100%** (BTC only) |

**Rationale for 100% max gross**: With a $50K account, max position notional = $50K (roughly 0.8 BTC at $60K). This means:
- Maximum leverage used: ~1x
- With 5% position sizing: typical position = $2.5K (0.04 BTC), well within limits
- Even at maximum quarter-Kelly sizing with high confidence, you would not exceed 100%
- The 100% gross limit is a **circuit breaker**, not the normal operating range

### If Expanding to Multi-Asset (ETH, SOL)

```python
# Multi-asset exposure framework
@dataclass
class ExposureLimits:
    max_gross_exposure: float = 1.0      # 100% of capital
    max_net_exposure: float = 0.80       # 80% directional tilt
    max_per_asset: dict[str, float] = field(default_factory=lambda: {
        "BTC": 1.0,    # Primary: up to 100%
        "ETH": 0.50,   # Secondary: up to 50%
        "SOL": 0.25,   # Tertiary: up to 25%
    })
    max_correlated_exposure: float = 1.20  # Max combined exposure for corr > 0.7

    def check_exposure(
        self,
        positions: dict[str, float],  # symbol -> signed notional
        capital: float,
    ) -> tuple[bool, str]:
        gross = sum(abs(v) for v in positions.values()) / capital
        net = sum(positions.values()) / capital

        if gross > self.max_gross_exposure:
            return False, f"Gross exposure {gross:.1%} exceeds {self.max_gross_exposure:.1%}"
        if abs(net) > self.max_net_exposure:
            return False, f"Net exposure {net:.1%} exceeds {self.max_net_exposure:.1%}"

        for symbol, notional in positions.items():
            limit = self.max_per_asset.get(symbol, 0.25)
            if abs(notional) / capital > limit:
                return False, f"{symbol} exposure {abs(notional)/capital:.1%} exceeds {limit:.1%}"

        return True, "OK"
```

### Crypto Correlation Reality

BTC-ETH rolling 30-day correlation (historical data):
- Bull markets: 0.85-0.95
- Bear markets: 0.90-0.98 (correlation increases in crashes)
- Range-bound: 0.70-0.85
- Flash events: approaches 1.0

**Implication**: Adding ETH to a BTC portfolio provides almost no diversification benefit during the times you need it most (drawdowns). The correlated exposure limit (`max_correlated_exposure`) accounts for this:

```python
def correlated_exposure(
    positions: dict[str, float],
    correlations: dict[tuple[str, str], float],
    capital: float,
) -> float:
    """
    Compute effective exposure accounting for correlations.

    Effective exposure > gross exposure when correlations are high.
    This prevents the illusion of diversification.

    Formula: sqrt( SUM_i SUM_j |w_i| * |w_j| * rho_ij )
    where w_i = position_i / capital, rho_ij = correlation
    """
    symbols = list(positions.keys())
    n = len(symbols)
    weights = [abs(positions[s]) / capital for s in symbols]

    effective = 0.0
    for i in range(n):
        for j in range(n):
            pair = (symbols[i], symbols[j]) if symbols[i] < symbols[j] else (symbols[j], symbols[i])
            rho = correlations.get(pair, 1.0 if i == j else 0.85)  # default high for crypto
            effective += weights[i] * weights[j] * rho

    return effective ** 0.5
```

---

## 3. Sector and Asset Concentration Limits

### BTC-Only System (Current)

By definition, concentration is 100%. This is **acceptable** because:
1. BTC is the most liquid crypto asset (deepest order books, tightest spreads)
2. The system is designed around BTC-specific microstructure signals
3. Diversifying into altcoins with 0.85+ correlation adds complexity without reducing risk

### If Expanding: Recommended Allocation Framework

```python
CONCENTRATION_LIMITS = {
    # Tier 1: Primary assets (deep liquidity, established microstructure)
    "tier_1": {
        "assets": ["BTC"],
        "max_allocation": 1.0,  # Up to 100% of risk budget
        "min_allocation": 0.50,  # At least 50% when multi-asset
    },
    # Tier 2: Secondary (good liquidity, correlated to BTC)
    "tier_2": {
        "assets": ["ETH"],
        "max_allocation": 0.40,
        "min_allocation": 0.0,
    },
    # Tier 3: Tertiary (lower liquidity, higher idiosyncratic risk)
    "tier_3": {
        "assets": ["SOL", "AVAX", "DOGE"],
        "max_allocation": 0.15,  # Per asset
        "max_tier_total": 0.25,  # All tier 3 combined
        "min_allocation": 0.0,
    },
}
```

### The Illusion of Diversification in Crypto

Why "diversifying" from BTC to ETH+SOL is mostly theater:

1. **Correlation regime-dependency**: During calm markets (when you least need protection), crypto correlations drop to 0.70-0.80, giving the appearance of diversification. During crashes (when you need protection most), correlations spike to 0.95+.

2. **Common factor exposure**: All crypto assets share exposure to:
   - Tether/stablecoin risk
   - Exchange solvency risk
   - Regulatory headlines
   - Macro risk-off flows
   - DeFi contagion

3. **Quantifying the illusion**:
   ```
   Portfolio volatility = sqrt(w_BTC^2 * vol_BTC^2 + w_ETH^2 * vol_ETH^2 + 2 * w_BTC * w_ETH * rho * vol_BTC * vol_ETH)

   Example with rho = 0.85:
   - 100% BTC: vol = vol_BTC (say 60% annualized)
   - 70% BTC + 30% ETH: vol = sqrt(0.49*0.36 + 0.09*0.81 + 2*0.7*0.3*0.85*0.6*0.9)
                             = sqrt(0.1764 + 0.0729 + 0.3213)
                             = sqrt(0.5706)
                             = 75.5% (HIGHER due to ETH's higher vol)
   ```

4. **Recommendation**: Stay BTC-only until the system is proven profitable. Adding assets should be motivated by uncorrelated alpha signals, not portfolio theory applied to highly correlated assets.

### Concentration Monitoring

```python
@dataclass
class ConcentrationMetrics:
    herfindahl_index: float     # HHI: sum of squared weights. 1.0 = fully concentrated
    max_single_asset_pct: float
    num_active_assets: int
    effective_num_assets: float  # 1/HHI: 1.0 means one asset, 3.0 means ~3 equal-weighted

def compute_concentration(positions: dict[str, float], capital: float) -> ConcentrationMetrics:
    if not positions:
        return ConcentrationMetrics(0.0, 0.0, 0, 0.0)

    weights = {s: abs(v) / capital for s, v in positions.items()}
    total_weight = sum(weights.values())
    if total_weight == 0:
        return ConcentrationMetrics(0.0, 0.0, 0, 0.0)

    normalized = {s: w / total_weight for s, w in weights.items()}
    hhi = sum(w ** 2 for w in normalized.values())

    return ConcentrationMetrics(
        herfindahl_index=hhi,
        max_single_asset_pct=max(weights.values()),
        num_active_assets=sum(1 for w in weights.values() if w > 0.01),
        effective_num_assets=1.0 / hhi if hhi > 0 else 0.0,
    )
```

---

## 4. Daily Risk Budget Allocation

### Concept

The daily risk budget limits the total risk you can take in a single day. This prevents a single bad day from causing outsized damage and ensures you have "ammunition" for later opportunities.

```
Daily Risk Budget = max_daily_loss * total_capital
```

With the existing config: `daily_loss_limit: float = 0.03` (3%), the daily risk budget for $50K capital = $1,500.

### Budget Consumption Tracking

```python
from dataclasses import dataclass
from datetime import datetime, timezone
import structlog

logger = structlog.get_logger()


@dataclass
class DailyRiskBudget:
    """Tracks daily risk budget consumption."""
    capital: float
    max_daily_loss_pct: float = 0.03  # 3%
    budget_reset_hour_utc: int = 0    # Reset at midnight UTC

    # State
    realized_loss_today: float = 0.0
    unrealized_loss_today: float = 0.0
    trades_today: int = 0
    wins_today: int = 0
    losses_today: int = 0
    _last_reset: datetime = datetime.now(timezone.utc)

    @property
    def total_budget_usd(self) -> float:
        return self.max_daily_loss_pct * self.capital

    @property
    def consumed_budget_usd(self) -> float:
        """Total consumed = realized losses + worst unrealized."""
        return self.realized_loss_today + max(0.0, self.unrealized_loss_today)

    @property
    def consumed_budget_pct(self) -> float:
        budget = self.total_budget_usd
        return self.consumed_budget_usd / budget if budget > 0 else 1.0

    @property
    def remaining_budget_usd(self) -> float:
        return max(0.0, self.total_budget_usd - self.consumed_budget_usd)

    @property
    def remaining_budget_pct(self) -> float:
        return 1.0 - self.consumed_budget_pct

    def position_size_multiplier(self) -> float:
        """
        Reduce position size as daily budget is consumed.

        Budget consumed | Size multiplier
        0-25%           | 1.00 (full size)
        25-50%          | 0.75
        50-75%          | 0.50
        75-90%          | 0.25
        >90%            | 0.00 (halt)
        """
        consumed = self.consumed_budget_pct
        if consumed >= 0.90:
            return 0.0
        elif consumed >= 0.75:
            return 0.25
        elif consumed >= 0.50:
            return 0.50
        elif consumed >= 0.25:
            return 0.75
        else:
            return 1.0

    def record_trade_pnl(self, pnl: float) -> None:
        self.trades_today += 1
        if pnl >= 0:
            self.wins_today += 1
        else:
            self.losses_today += 1
            self.realized_loss_today += abs(pnl)

        logger.info(
            "daily_risk_budget_update",
            pnl=pnl,
            realized_loss=self.realized_loss_today,
            consumed_pct=self.consumed_budget_pct,
            remaining_usd=self.remaining_budget_usd,
            trades_today=self.trades_today,
        )

    def update_unrealized(self, unrealized_pnl: float) -> None:
        """Update unrealized loss for open positions."""
        self.unrealized_loss_today = max(0.0, -unrealized_pnl)

    def check_reset(self, now: datetime) -> bool:
        """Check if budget should reset (new day)."""
        if now.date() > self._last_reset.date():
            self.realized_loss_today = 0.0
            self.unrealized_loss_today = 0.0
            self.trades_today = 0
            self.wins_today = 0
            self.losses_today = 0
            self._last_reset = now
            logger.info("daily_risk_budget_reset", date=now.date().isoformat())
            return True
        return False
```

### Session-Based Allocation (Asia/Europe/US)

Crypto trades 24/7, but liquidity and opportunity distribution is not uniform. The system's trading hours are 08:00-21:00 UTC (Europe + US sessions).

```python
SESSION_BUDGET_WEIGHTS = {
    # Based on historical BTC volume distribution and signal quality
    "asia":   {"hours": (0, 8),   "budget_weight": 0.0},   # Not traded
    "europe": {"hours": (8, 16),  "budget_weight": 0.45},  # 45% of daily budget
    "us":     {"hours": (16, 21), "budget_weight": 0.55},  # 55% (NQ lead-lag active)
    "night":  {"hours": (21, 24), "budget_weight": 0.0},   # Not traded
}

# Effective budget available at any time:
# If entering US session and only 20% of budget consumed during Europe: full US allocation
# If entering US session and 80% consumed: only 20% left regardless of US allocation
# The session weights are TARGETS, not guarantees

def session_budget_remaining(
    current_hour_utc: int,
    consumed_pct: float,
    total_budget: float,
) -> float:
    """
    Calculate remaining budget considering session timing.

    Early in the day: conservative (leave budget for US session)
    Late in the day: use what's left
    """
    if current_hour_utc < 8 or current_hour_utc >= 21:
        return 0.0  # Outside trading hours

    remaining_total = total_budget * (1.0 - consumed_pct)

    if current_hour_utc < 16:
        # Europe session: only use Europe's allocation
        europe_budget = total_budget * 0.45
        europe_remaining = max(0.0, europe_budget - (consumed_pct * total_budget))
        return min(remaining_total, europe_remaining)
    else:
        # US session: use whatever's left
        return remaining_total
```

### End-of-Day Risk Reduction

The current config has `trading_end_utc: int = 21`. Risk reduction before day-end:

```python
def eod_size_multiplier(current_hour_utc: int, current_minute: int) -> float:
    """
    Reduce position sizing as end of trading hours approaches.

    20:00 - 20:30: full size
    20:30 - 20:45: 50% size
    20:45 - 21:00: new entries blocked, existing positions may be reduced
    """
    if current_hour_utc < 20:
        return 1.0
    if current_hour_utc >= 21:
        return 0.0  # No new trades

    minutes_remaining = (21 * 60) - (current_hour_utc * 60 + current_minute)
    if minutes_remaining > 30:
        return 1.0
    elif minutes_remaining > 15:
        return 0.50
    else:
        return 0.0  # Block new entries in last 15 minutes
```

---

## 5. Signal-Level Risk Budgeting

### Concept

When multiple signal sources can generate trades, each source gets a fraction of the total risk budget. This prevents any single signal module from monopolizing risk capacity.

### Budget Allocation for ep2-crypto

```python
SIGNAL_RISK_BUDGET = {
    "ml_model": {
        "budget_share": 0.60,      # 60% of total risk budget
        "description": "LightGBM/CatBoost stacking ensemble",
        "max_concurrent": 1,       # Max positions from this signal
        "min_confidence": 0.60,    # Minimum confidence to use budget
    },
    "macro_event": {
        "budget_share": 0.25,      # 25% of total risk budget
        "description": "CPI/FOMC NQ lead-lag trades",
        "max_concurrent": 1,
        "min_confidence": 0.55,    # Lower threshold (event-driven)
    },
    "cascade_detection": {
        "budget_share": 0.15,      # 15% of total risk budget
        "description": "Liquidation cascade short trades",
        "max_concurrent": 1,
        "min_confidence": 0.70,    # Higher threshold (rare events)
    },
}
```

### Enforcement When Signals Overlap

When multiple signals fire simultaneously, the system must handle conflicts and prevent over-allocation.

```python
from dataclasses import dataclass
from enum import Enum


class SignalDirection(Enum):
    LONG = 1
    SHORT = -1
    FLAT = 0


@dataclass
class SignalRequest:
    source: str
    direction: SignalDirection
    confidence: float
    requested_size_pct: float  # % of capital
    stop_distance_pct: float
    timestamp: float


class SignalBudgetAllocator:
    """Allocates risk budget across competing signal sources."""

    def __init__(
        self,
        capital: float,
        max_heat_pct: float = 0.02,
        signal_budgets: dict[str, dict] | None = None,
    ):
        self.capital = capital
        self.max_heat_pct = max_heat_pct
        self.signal_budgets = signal_budgets or SIGNAL_RISK_BUDGET
        self.consumed: dict[str, float] = {k: 0.0 for k in self.signal_budgets}

    def allocate(self, requests: list[SignalRequest]) -> list[tuple[SignalRequest, float]]:
        """
        Given simultaneous signal requests, allocate sizes.

        Rules:
        1. Each signal can only use its budget share
        2. Agreeing signals (same direction) can combine up to total heat limit
        3. Conflicting signals (opposite direction) cancel each other
        4. Higher confidence gets priority when budget is scarce
        """
        if not requests:
            return []

        # Check for conflicts
        directions = {r.source: r.direction for r in requests}
        unique_dirs = set(d for d in directions.values() if d != SignalDirection.FLAT)

        if len(unique_dirs) > 1:
            # Conflicting signals: use highest confidence, skip the rest
            best = max(requests, key=lambda r: r.confidence)
            return self._allocate_single(best)

        # Agreeing signals: allocate from each budget
        results = []
        total_allocated_heat = 0.0

        # Sort by confidence descending
        sorted_requests = sorted(requests, key=lambda r: r.confidence, reverse=True)

        for req in sorted_requests:
            budget_info = self.signal_budgets.get(req.source)
            if budget_info is None:
                continue

            # Check minimum confidence
            if req.confidence < budget_info["min_confidence"]:
                results.append((req, 0.0))
                continue

            # Check source budget
            source_budget_heat = budget_info["budget_share"] * self.max_heat_pct
            source_remaining = source_budget_heat - self.consumed[req.source]

            # Check total heat
            total_remaining = self.max_heat_pct - total_allocated_heat

            # Available heat is the minimum of source budget and total remaining
            available_heat = min(source_remaining, total_remaining)

            if available_heat <= 0:
                results.append((req, 0.0))
                continue

            # Convert heat to position size
            proposed_heat = req.requested_size_pct * req.stop_distance_pct
            actual_heat = min(proposed_heat, available_heat)
            actual_size_pct = actual_heat / max(req.stop_distance_pct, 1e-8)

            self.consumed[req.source] += actual_heat
            total_allocated_heat += actual_heat
            results.append((req, actual_size_pct))

        return results

    def _allocate_single(self, req: SignalRequest) -> list[tuple[SignalRequest, float]]:
        budget_info = self.signal_budgets.get(req.source, {"budget_share": 0.0, "min_confidence": 1.0})
        if req.confidence < budget_info["min_confidence"]:
            return [(req, 0.0)]
        source_budget_heat = budget_info["budget_share"] * self.max_heat_pct
        max_heat = min(source_budget_heat, self.max_heat_pct)
        proposed_heat = req.requested_size_pct * req.stop_distance_pct
        actual_heat = min(proposed_heat, max_heat)
        actual_size_pct = actual_heat / max(req.stop_distance_pct, 1e-8)
        return [(req, actual_size_pct)]
```

### Signal Correlation and Total Risk

When signals are correlated (they tend to fire at the same time in the same direction), the real diversification benefit is lower than the budget shares suggest.

```
Effective combined risk = sqrt(w_ml^2 + w_macro^2 + w_cascade^2
                              + 2*w_ml*w_macro*rho_ml_macro
                              + 2*w_ml*w_cascade*rho_ml_cascade
                              + 2*w_macro*w_cascade*rho_macro_cascade)
```

Estimated signal correlations for ep2-crypto:
- ML model vs Macro event: ~0.30 (ML incorporates some NQ features, but macro is event-driven)
- ML model vs Cascade: ~0.40 (both respond to volatility, but cascade is derivatives-specific)
- Macro vs Cascade: ~0.20 (largely independent trigger conditions)

With these correlations, the effective combined risk when all three fire simultaneously:
```
= sqrt(0.60^2 + 0.25^2 + 0.15^2 + 2*0.60*0.25*0.30 + 2*0.60*0.15*0.40 + 2*0.25*0.15*0.20)
= sqrt(0.36 + 0.0625 + 0.0225 + 0.09 + 0.072 + 0.015)
= sqrt(0.622)
= 0.789
```

So the effective combined allocation is ~79% of the simple sum (100%), meaning roughly 21% diversification benefit. This is meaningful but not dramatic -- the signals do share some common factors.

---

## 6. Maximum Concurrent Positions

### Recommendation: 1 Position at a Time

For a 5-minute BTC perpetual system at $10K-$100K capital: **one position at a time is strongly recommended**. This is already set in the config (`max_open_positions: int = 1`).

### Arguments For Single Position

1. **Simplicity**: One position means no correlation management, no margin cross-calculation, no need to handle partial liquidations
2. **Clean PnL attribution**: Every trade's result is unambiguous. With multiple positions, PnL attribution becomes complex
3. **Margin efficiency**: With one position, 100% of allocated margin goes to one trade. With 2 positions, margin is split, and cross-margin risk increases
4. **No self-interference**: Two BTC positions cannot be uncorrelated -- they are literally the same asset
5. **Backtest fidelity**: Single-position backtests are much more realistic and less prone to hidden biases

### Arguments Against (Why You Might Want >1)

1. **Scaling in**: Add to a winning position. Counter: at 5-min timeframe, by the time you confirm the move, it is often over
2. **Hedging**: Hold a core position + hedge. Counter: for a 5-min system, the "core position" concept does not apply
3. **Multi-signal**: ML says long, macro event says short. Counter: conflicting signals should abstain, not hedge

### If Allowing 2 Positions

If the system is later extended to allow 2 concurrent positions:

```python
@dataclass
class MultiPositionPolicy:
    max_positions: int = 1  # Start with 1, may increase later

    # If max_positions > 1:
    same_direction_only: bool = True   # Both positions must be same direction
    max_combined_notional_pct: float = 0.08  # 8% combined (not 5% + 5%)
    max_combined_heat_pct: float = 0.02      # Total heat unchanged
    min_entry_separation_bars: int = 3       # At least 15 min between entries
    require_different_signals: bool = True    # Cannot be both from ML model

    # Opposite-direction positions (hedging) - NOT recommended for this system
    allow_opposite: bool = False
```

**Verdict**: Keep `max_open_positions = 1`. The complexity cost of multiple positions far exceeds the potential benefit at this timeframe and capital level.

---

## 7. Time-in-Market Limits

### Concept

Time-in-market (TIM) measures the percentage of bars where the system has an open position. A high TIM suggests the system is always trading, which means:
- High transaction costs (constant entry/exit)
- Essentially buy-and-hold with extra steps
- No selectivity advantage

### Target for ep2-crypto

```
Target TIM: 20-40%
Maximum TIM: 50%
Alert threshold: 45%
```

Rationale:
- At 288 bars/day and 13 active hours (08:00-21:00 UTC = 156 bars), the system has 156 potential trading bars
- At 40% TIM: ~62 bars with position = ~5 hours in position per day
- With hold period of 1-6 bars (5-30 min) and 30 max trades: 30 * 3-bar avg = 90 bars = 58% of active period
- The confidence gating should filter out many signals, keeping TIM at 20-40%

### Implementation

```python
from collections import deque
from dataclasses import dataclass
import structlog

logger = structlog.get_logger()


@dataclass
class TimeInMarketTracker:
    """Tracks the fraction of time the system has an open position."""
    window_bars: int = 288  # 1 day rolling window
    max_tim_pct: float = 0.50
    alert_tim_pct: float = 0.45

    def __post_init__(self) -> None:
        self._position_history: deque[bool] = deque(maxlen=self.window_bars)

    def update(self, has_position: bool) -> None:
        self._position_history.append(has_position)

    @property
    def current_tim(self) -> float:
        if not self._position_history:
            return 0.0
        return sum(self._position_history) / len(self._position_history)

    def should_block_entry(self) -> bool:
        """Block new entries if TIM exceeds maximum."""
        return self.current_tim >= self.max_tim_pct

    def check_alerts(self) -> str | None:
        tim = self.current_tim
        if tim >= self.max_tim_pct:
            logger.warning(
                "time_in_market_exceeded",
                tim_pct=tim,
                max_pct=self.max_tim_pct,
            )
            return "TIM_EXCEEDED"
        elif tim >= self.alert_tim_pct:
            logger.warning(
                "time_in_market_high",
                tim_pct=tim,
                alert_pct=self.alert_tim_pct,
            )
            return "TIM_HIGH"
        return None
```

### Monitoring Metrics

```python
tim_metrics = {
    "time_in_market_1d": tracker.current_tim,                    # Rolling 1-day
    "time_in_market_7d": tracker_7d.current_tim,                 # Rolling 7-day
    "avg_hold_duration_bars": avg_hold_bars,                     # Average bars held
    "trades_per_day": trades_today,
    "bars_per_trade": avg_bars_between_trades,                   # Selectivity measure
    "tim_vs_buy_and_hold": tracker.current_tim,                  # If > 80%: essentially buy-and-hold
}
```

### What High TIM Means

| TIM | Interpretation | Action |
|-----|---------------|--------|
| < 15% | Very selective, may be missing opportunities | Review confidence threshold (too high?) |
| 15-25% | Highly selective, good for 5-min system | Ideal range |
| 25-40% | Active but selective | Acceptable |
| 40-50% | Approaching always-in territory | Review signal quality, consider raising confidence threshold |
| > 50% | Essentially always in market | **Problem**: you are paying costs for buy-and-hold returns. Increase confidence threshold or add more gates |

---

## 8. Overnight and Weekend Risk

### Crypto 24/7 Reality

Unlike equities, crypto has no close. But liquidity is not uniform:

```
Liquidity profile (approximate, BTC/USDT perps):
- 00:00-04:00 UTC (Asia late): Low-medium
- 04:00-08:00 UTC (Asia AM/Europe pre): Low
- 08:00-12:00 UTC (Europe AM): High
- 12:00-16:00 UTC (Europe PM/US pre): High
- 16:00-20:00 UTC (US AM): Very High
- 20:00-00:00 UTC (US PM): Medium-High

Weekend liquidity: ~40-60% of weekday levels
```

### Weekend Risk Management

The current config has `weekend_size_reduction: float = 0.30` (30% reduction). This should be applied via the position sizer.

```python
from datetime import datetime, timezone


def weekend_risk_multiplier(
    dt: datetime,
    base_reduction: float = 0.30,
) -> float:
    """
    Compute position size multiplier for weekend risk.

    Friday 20:00 UTC to Monday 04:00 UTC: reduced sizing
    The reduction tapers in (Friday evening) and out (Sunday night).
    """
    weekday = dt.weekday()  # 0=Monday, 6=Sunday
    hour = dt.hour

    # Full weekdays
    if weekday in (0, 1, 2, 3):  # Mon-Thu
        if weekday == 0 and hour < 4:
            return 1.0 - (base_reduction * 0.5)  # Monday early AM: partial reduction
        return 1.0

    # Friday
    if weekday == 4:
        if hour < 20:
            return 1.0  # Full size during Friday trading
        else:
            return 1.0 - base_reduction  # Reduced after 20:00

    # Saturday
    if weekday == 5:
        return 1.0 - base_reduction  # Full reduction

    # Sunday
    if weekday == 6:
        if hour >= 20:
            return 1.0 - (base_reduction * 0.5)  # Tapering back
        return 1.0 - base_reduction

    return 1.0
```

### Friday-to-Monday Specific Risks

1. **Wider spreads**: BTC/USDT spread widens from ~1-2 bps weekday to 3-8 bps weekend on thinner exchanges. On Binance perps, effect is smaller but still measurable at 1.5-3x typical spread.

2. **Thinner order books**: Top-of-book liquidity drops 40-60%. A $50K market order that moves price 1 bp on weekdays may move 2-3 bps on weekends.

3. **Funding rate settlement timing**: Funding settles every 8 hours (00:00, 08:00, 16:00 UTC) regardless of day. Weekend funding rates can be extreme due to lower participation.

4. **Headline risk**: Regulatory announcements, exchange outages, and protocol exploits often happen when traditional markets are closed and crypto desks are understaffed.

### Funding Rate and Weekend Positions

```python
FUNDING_SETTLEMENT_HOURS_UTC = [0, 8, 16]  # Every 8 hours

def funding_cost_for_position(
    position_notional: float,
    funding_rate: float,
    side: str,  # "long" or "short"
) -> float:
    """
    Calculate funding payment for a position.

    If funding_rate > 0: longs pay shorts
    If funding_rate < 0: shorts pay longs

    Returns: cost (negative = you receive payment)
    """
    if side == "long":
        return position_notional * funding_rate  # Positive = cost
    else:
        return -position_notional * funding_rate  # Negative = income

def hours_to_next_funding(current_hour_utc: int, current_minute: int) -> float:
    """Hours until next funding settlement."""
    current_time_hours = current_hour_utc + current_minute / 60.0
    for settlement in FUNDING_SETTLEMENT_HOURS_UTC + [24]:
        if settlement > current_time_hours:
            return settlement - current_time_hours
    return 24 - current_time_hours  # Next day 00:00

# Risk consideration: if holding through funding settlement with high funding rate,
# the cost can exceed the expected trade profit. For a 5-min system with 1-6 bar holds,
# most trades should NOT cross funding settlement. But the position sizer should check:

def should_avoid_funding_settlement(
    expected_hold_bars: int,
    current_hour_utc: int,
    current_minute: int,
    funding_rate: float,
    side: str,
    bar_interval_minutes: int = 5,
) -> bool:
    """
    Check if entering now would likely cross a funding settlement
    in the wrong direction (paying, not receiving).
    """
    hours_to_funding = hours_to_next_funding(current_hour_utc, current_minute)
    hold_hours = (expected_hold_bars * bar_interval_minutes) / 60.0

    crosses_funding = hold_hours >= hours_to_funding

    if not crosses_funding:
        return False

    # Would we pay or receive?
    pays = (side == "long" and funding_rate > 0) or (side == "short" and funding_rate < 0)

    # Only avoid if payment exceeds threshold (e.g., 5 bps)
    return pays and abs(funding_rate) > 0.0005
```

### Recommendations for ep2-crypto

1. **Trading hours (keep current)**: 08:00-21:00 UTC. This covers the two highest-liquidity sessions and avoids the thin 00:00-08:00 UTC window.

2. **Weekend reduction (keep current)**: 30% position size reduction Friday 20:00 to Monday 04:00 UTC.

3. **Consider full weekend close**: For early deployment, consider closing all positions at Friday 20:00 UTC and resuming Monday 08:00 UTC. The risk-reward of weekend trades is unfavorable:
   - Wider spreads increase costs 50-200%
   - Thinner books increase slippage
   - Higher funding rates possible
   - But the same signal quality cannot be assumed (features trained on weekday data)

4. **Funding awareness**: Before entering a trade within 30 minutes of funding settlement, check the rate. Skip entry if the funding cost would exceed 5 bps and the position is on the paying side.

---

## 9. Cascade Risk from Own Trades (Market Impact)

### At What Size Do We Move the Market?

For BTC/USDT perpetuals on Binance, order book depth and market impact depend on market conditions. Based on publicly observable data:

```
Typical Binance BTC/USDT Perp Order Book (calm market):
- Top-of-book (best bid/ask): ~$200K-$500K per side
- Within 10 bps: ~$2M-$5M per side
- Within 50 bps: ~$10M-$25M per side
- Within 100 bps: ~$20M-$50M per side

Thin market (3 AM UTC, weekend, post-cascade):
- All numbers above: divide by 2-4x
```

### Market Impact by Position Size

```python
def estimate_market_impact_bps(
    order_size_usd: float,
    book_depth_usd_at_10bps: float = 3_000_000,  # $3M within 10 bps (conservative)
    impact_exponent: float = 0.5,  # Square-root impact model
) -> float:
    """
    Estimate market impact in bps using square-root model.

    Impact = sigma * sqrt(V/ADV) * sign(V)

    Simplified for order book impact:
    Impact_bps = 10 * (order_size / book_depth_at_10bps) ^ 0.5

    The square-root law is well-established for equity markets
    and approximately holds for crypto (Kyle, 1985; Almgren-Chriss, 2001).
    """
    if order_size_usd <= 0:
        return 0.0

    participation_rate = order_size_usd / book_depth_usd_at_10bps
    return 10.0 * (participation_rate ** impact_exponent)
```

| Order Size | Normal Market Impact | Thin Market Impact | Assessment |
|-----------|---------------------|-------------------|------------|
| $10K | 0.6 bps | 1.2 bps | Negligible |
| $50K | 1.3 bps | 2.6 bps | Minimal |
| $100K | 1.8 bps | 3.7 bps | Noticeable |
| $250K | 2.9 bps | 5.8 bps | Significant |
| $500K | 4.1 bps | 8.2 bps | Material |
| $1M | 5.8 bps | 11.5 bps | Large |

### Feedback Loop Risk

At $100K position size, the market impact (~2-4 bps) is within the system's cost model (8-12 bps round-trip assumption). But the risk is a feedback loop:

1. Our system enters a trade (moves price 2 bps)
2. Other algorithms detect the same microstructure signal (because our trade changed the microstructure)
3. They pile on (moves price further)
4. Our stop is tighter, so we get stopped out
5. Our exit creates more microstructure signals in the opposite direction

This feedback is negligible at $10K-$50K but becomes a real concern above $250K.

### Maximum Position as % of Market Depth

```python
MAX_PARTICIPATION_RATES = {
    # Max order size as percentage of visible book depth at 10 bps
    "conservative": 0.01,  # 1% of book depth -> impact ~1 bps
    "moderate": 0.03,      # 3% -> impact ~1.7 bps
    "aggressive": 0.05,    # 5% -> impact ~2.2 bps
}

# For ep2-crypto at $50K capital, 5% position = $2,500
# $2,500 / $3,000,000 = 0.08% participation rate -> essentially zero impact

# Market impact becomes a concern at:
# - $100K capital, 50% position = $50K -> 1.7% participation -> ~1.3 bps impact
# - $500K capital, 10% position = $50K -> same
# - $1M capital, 5% position = $50K -> same

# Rule of thumb: worry about impact when single-trade notional exceeds $100K
```

### Implementation for Cost Engine

```python
def adjusted_cost_with_impact(
    notional: float,
    base_fee_bps: float = 4.0,     # Taker fee
    base_slippage_bps: float = 1.0,
    book_depth_usd: float = 3_000_000,
) -> float:
    """Total execution cost including market impact."""
    impact_bps = estimate_market_impact_bps(notional, book_depth_usd)
    total_bps = base_fee_bps + base_slippage_bps + impact_bps
    return total_bps

# At $10K capital: total cost ~5.6 bps (4 + 1 + 0.6)
# At $50K capital: total cost ~6.3 bps (4 + 1 + 1.3)
# At $100K capital: total cost ~6.8 bps (4 + 1 + 1.8)
# All well within the 8-12 bps budget
```

### Monitoring

```python
impact_metrics = {
    "order_size_usd": order_size,
    "estimated_impact_bps": impact_bps,
    "book_depth_at_10bps": observed_depth,
    "participation_rate": order_size / observed_depth,
    "total_execution_cost_bps": total_cost,
}

# Alert if:
# - participation_rate > 0.03 (3%): WARNING
# - participation_rate > 0.05 (5%): REDUCE SIZE
# - participation_rate > 0.10 (10%): CRITICAL - split order or reject
```

---

## 10. Risk Limits from Regulatory Frameworks

### MiCA (Markets in Crypto-Assets Regulation) - EU

MiCA took full effect in December 2024. Key requirements for algorithmic crypto trading firms operating in the EU:

1. **Capital adequacy**: CASPs (Crypto-Asset Service Providers) must maintain minimum own funds (EUR 50K-150K depending on services). For algorithmic trading specifically, this may increase.

2. **Risk management**: Article 67 requires CASPs to have "effective risk management procedures" including:
   - Internal risk limits documented and reviewed quarterly
   - Stress testing regime
   - Business continuity plans
   - Pre-trade risk controls for algorithmic systems

3. **Algorithmic trading requirements** (Article 76):
   - Effective systems and risk controls
   - Kill switch capability (already planned)
   - Testing and monitoring of algorithms
   - Record-keeping of all orders, trades, and algorithm parameters
   - Annual self-assessment

4. **Documentation requirements**:
   - Risk limit framework document
   - Algorithm description document
   - Incident reporting procedures
   - Record retention: 5 years minimum

### CFTC Position Limits (CME Bitcoin Futures)

If trading CME Bitcoin Futures (not perpetuals on Binance):

```
CME Bitcoin Futures (BTC) position limits:
- Spot month: 2,000 contracts
- Single month: 2,000 contracts
- All months combined: 2,000 contracts
- 1 contract = 5 BTC

At $60K/BTC: 2,000 * 5 * $60K = $600M notional limit
```

This is irrelevant for our capital range ($10K-$100K), but worth knowing if scaling up to trade CME products.

For Binance/Bybit perpetuals: **no regulatory position limits** for most jurisdictions, but exchange-imposed limits exist:
- Binance: position limits per account tier ($500K-$50M+ depending on VIP level)
- Bybit: similar tiered limits

At $10K-$100K capital, exchange limits are not binding.

### Internal Risk Limits (Self-Imposed)

These are more important than external regulatory limits at our scale. The complete framework:

```python
@dataclass
class RiskLimitFramework:
    """
    Comprehensive risk limit framework.

    Internal documentation requirement: review quarterly.
    All limits must be justified, tested, and monitored.
    """

    # === Position Limits ===
    max_position_pct: float = 0.05       # 5% of capital per trade
    max_open_positions: int = 1
    max_leverage: float = 1.0            # No leverage beyond 1x notional

    # === Loss Limits ===
    max_daily_loss_pct: float = 0.03     # 3% daily
    max_weekly_loss_pct: float = 0.05    # 5% weekly
    max_drawdown_halt_pct: float = 0.15  # 15% drawdown -> halt
    max_consecutive_losses: int = 15

    # === Heat & Exposure ===
    max_portfolio_heat_pct: float = 0.02  # 2% of capital at risk
    max_gross_exposure_pct: float = 1.0   # 100% of capital
    max_net_exposure_pct: float = 1.0     # 100% directional

    # === Trade Limits ===
    max_trades_per_day: int = 30
    max_hold_bars: int = 6               # 30 minutes
    min_bar_volatility_ann: float = 0.15
    max_bar_volatility_ann: float = 1.50

    # === Time Limits ===
    trading_start_utc: int = 8
    trading_end_utc: int = 21
    max_time_in_market_pct: float = 0.50
    weekend_size_reduction: float = 0.30

    # === Signal Budget ===
    ml_signal_budget_pct: float = 0.60
    macro_signal_budget_pct: float = 0.25
    cascade_signal_budget_pct: float = 0.15

    # === Market Impact ===
    max_participation_rate: float = 0.03  # 3% of visible book depth
    max_order_size_usd: float = 100_000   # Hard cap regardless of capital

    # === Stop Loss ===
    catastrophic_stop_atr: float = 3.0
    trailing_stop_activation_pct: float = 0.005  # Activate after 0.5% favorable move
    trailing_stop_distance_pct: float = 0.003    # Trail at 0.3%

    # === Kill Switch Thresholds ===
    kill_switch_daily_loss: float = 0.03
    kill_switch_weekly_loss: float = 0.05
    kill_switch_drawdown: float = 0.15
    kill_switch_consecutive: int = 15
    kill_switch_requires_manual_reset: bool = True
```

### Documentation Requirements (Best Practice)

Even without MiCA requirements, maintaining these documents is essential for a disciplined operation:

1. **Risk Limit Framework Document** (this research file forms the basis)
   - All limits with rationale
   - Review frequency: quarterly
   - Change process: requires backtest evidence

2. **Algorithm Description Document**
   - Signal generation logic
   - Position sizing methodology
   - Risk management procedures
   - Kill switch behavior

3. **Incident Log**
   - Every kill switch activation
   - Every manual override
   - Every limit breach (should be zero, but track near-misses)
   - Every emergency close

4. **Performance and Risk Report** (generated weekly)
   - PnL summary
   - Risk utilization (how close to limits)
   - Kill switch activation count
   - Maximum drawdown this period
   - Sharpe ratio rolling

---

## Summary: Complete Risk Parameter Set for ep2-crypto

### Tier 1: Must Have Before Any Live Trading

| Parameter | Value | Implemented In |
|-----------|-------|----------------|
| Max position % of capital | 5% | `risk/position_sizer.py` |
| Max portfolio heat | 2% | `risk/position_tracker.py` |
| Daily loss limit | 3% | `risk/kill_switches.py` |
| Weekly loss limit | 5% | `risk/kill_switches.py` |
| Max drawdown halt | 15% | `risk/kill_switches.py` |
| Consecutive loss halt | 15 | `risk/kill_switches.py` |
| Catastrophic stop | 3 ATR | `risk/position_sizer.py` |
| Max open positions | 1 | `risk/position_tracker.py` |
| Max trades/day | 30 | `risk/kill_switches.py` |
| Trading hours | 08-21 UTC | `risk/volatility_guard.py` |
| Min volatility | 15% ann | `risk/volatility_guard.py` |
| Max volatility | 150% ann | `risk/volatility_guard.py` |
| Weekend reduction | 30% | `risk/position_sizer.py` |
| Max hold period | 6 bars | `risk/position_sizer.py` |
| Kill switch manual reset | Required | `risk/kill_switches.py` |

### Tier 2: Should Have Before Paper Trading

| Parameter | Value | Implemented In |
|-----------|-------|----------------|
| Daily risk budget tracking | 3% budget | `risk/risk_manager.py` |
| Budget consumption sizing | Taper at 25/50/75/90% | `risk/risk_manager.py` |
| Signal-level budgets | 60/25/15% | `risk/risk_manager.py` |
| Time-in-market limit | 50% max | `risk/risk_manager.py` |
| Drawdown progressive reduction | Taper 3/5/10/15% | `risk/drawdown_gate.py` |
| Funding rate awareness | Skip if >5 bps cost | `risk/position_sizer.py` |
| EOD entry blocking | Last 15 min | `risk/volatility_guard.py` |

### Tier 3: Nice to Have / For Scaling

| Parameter | Value | Implemented In |
|-----------|-------|----------------|
| Market impact estimation | sqrt model | `backtest/cost_engine.py` |
| Max participation rate | 3% of book | `risk/risk_manager.py` |
| Exposure limits (multi-asset) | 100% gross | `risk/risk_manager.py` |
| Concentration metrics | HHI tracking | `monitoring/` |
| Session-based budget allocation | 45/55% EU/US | `risk/risk_manager.py` |
| Correlated exposure calculation | Matrix approach | `risk/risk_manager.py` |

### Integration with Existing Config

The existing `MonitoringConfig` in `config.py` covers most Tier 1 parameters. The additions needed:

```python
class RiskConfig(BaseSettings):
    """Extended risk configuration for Sprint 9."""

    model_config = SettingsConfigDict(env_prefix="EP2_RISK_")

    # Heat limits (new)
    max_portfolio_heat_pct: float = 0.02
    heat_warning_threshold: float = 0.80  # Warn at 80% heat utilization

    # Daily budget (new)
    budget_taper_thresholds: list[float] = [0.25, 0.50, 0.75, 0.90]
    budget_taper_multipliers: list[float] = [0.75, 0.50, 0.25, 0.00]

    # Time in market (new)
    max_time_in_market_pct: float = 0.50
    time_in_market_alert_pct: float = 0.45

    # Signal budgets (new)
    ml_signal_budget: float = 0.60
    macro_signal_budget: float = 0.25
    cascade_signal_budget: float = 0.15

    # Market impact (new)
    max_participation_rate: float = 0.03
    assumed_book_depth_usd: float = 3_000_000

    # Funding rate (new)
    max_funding_cost_bps: float = 5.0  # Skip entry if funding cost > 5 bps

    # EOD (new)
    eod_block_minutes: int = 15  # Block new entries this many minutes before close
    eod_reduce_minutes: int = 30  # Reduce size this many minutes before close
```

---

## Key Formulas Reference

```
Portfolio Heat = SUM(position_notional_i * stop_distance_pct_i) / capital
Gross Exposure = SUM(|position_notional_i|) / capital
Net Exposure = SUM(signed_position_notional_i) / capital
Daily Budget Consumed = (realized_losses + max(0, unrealized_loss)) / (max_daily_loss * capital)
Time in Market = bars_with_position / total_bars (rolling window)
Market Impact (bps) = 10 * (order_size / book_depth_at_10bps) ^ 0.5
Effective Correlated Exposure = sqrt(SUM_ij(|w_i| * |w_j| * rho_ij))
Heat-Constrained Size = (max_heat - current_heat) * capital / stop_distance_pct
Position Size = min(quarter_kelly, max_cap, heat_constrained, budget_constrained) * weekend_mult * eod_mult * drawdown_mult
```
