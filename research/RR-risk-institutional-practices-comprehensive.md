# Institutional-Grade Risk Management for Crypto Hedge Funds

> Comprehensive framework covering three lines of defense, pre-trade checks, position limits, counterparty risk, operational risk, stress testing, LP reporting, VaR/CVaR, liquidity risk, and regulatory risk. Adapted for a small BTC perpetual futures fund.

---

## Table of Contents

1. [Three Lines of Defense Model](#1-three-lines-of-defense-model)
2. [Pre-Trade Risk Checks](#2-pre-trade-risk-checks)
3. [Real-Time Position Limits and Margin Monitoring](#3-real-time-position-limits-and-margin-monitoring)
4. [Counterparty Risk Management](#4-counterparty-risk-management)
5. [Operational Risk](#5-operational-risk)
6. [Stress Testing Requirements](#6-stress-testing-requirements)
7. [Risk Reporting for LPs](#7-risk-reporting-for-lps)
8. [VaR and CVaR for Crypto Portfolios](#8-var-and-cvar-for-crypto-portfolios)
9. [Liquidity Risk](#9-liquidity-risk)
10. [Regulatory Risk](#10-regulatory-risk)

---

## 1. Three Lines of Defense Model

### 1.1 The IIA Framework Adapted for Crypto

The Three Lines of Defense (3LOD) model, created by the Institute of Internal Auditors and updated in 2020 as the "Three Lines Model," assigns risk ownership at three levels. For a small crypto fund, the adaptation maps software components to organizational roles.

### 1.2 First Line: Strategy-Level Risk (Trading Engine)

**Owner:** The trading logic / signal generation code.

**Responsibilities:**
- Position sizing per trade (quarter-Kelly, capped at 5% of NAV)
- Stop-loss placement and enforcement (3x ATR catastrophic stop)
- Entry/exit criteria based on model confidence
- Signal quality gating (minimum confidence threshold 0.60)
- Individual trade risk budget (max 1% of equity at risk per trade)

**Implementation principle:** The first line PROPOSES trades. It never executes them directly. Every proposed order is a request that must be approved by the second line.

```python
# First Line: trade proposal (NOT execution)
@dataclass
class TradeProposal:
    """First line proposes, second line approves."""
    symbol: str
    direction: str          # "long" or "short"
    quantity_btc: float
    proposed_stop: float
    confidence: float
    signal_source: str
    risk_per_trade_pct: float  # Must be <= 1%
    estimated_slippage_bps: float
    timestamp: float
```

### 1.3 Second Line: Portfolio-Level Risk (Risk Engine)

**Owner:** Independent risk engine process.

**Responsibilities:**
- Aggregate exposure limits (gross, net, directional)
- VaR/CVaR computation and limit enforcement
- Drawdown monitoring and progressive position reduction
- Correlation monitoring (are all signals correlated during stress?)
- Regime-based parameter adjustment
- Kill switch enforcement (daily/weekly/drawdown/consecutive loss)
- Every order passes through this gate -- no exceptions

**Key design constraint:** The risk engine is the PARENT process. The trading engine is the CHILD. The trading engine has no direct access to the exchange API. All orders route through `RiskGatedBroker`.

```python
# Second Line: order approval gate
class PreTradeRiskGate:
    """Every order must pass ALL checks. One failure = rejection."""

    def evaluate(self, proposal: TradeProposal) -> TradeDecision:
        checks = [
            self._check_kill_switches(),
            self._check_volatility_guard(proposal),
            self._check_drawdown_gate(),
            self._check_var_limit(proposal),
            self._check_exposure_limit(proposal),
            self._check_position_concentration(proposal),
            self._check_order_rate_limit(),
            self._check_liquidity_adequacy(proposal),
            self._check_margin_adequacy(proposal),
            self._check_fat_finger_guard(proposal),
        ]
        rejections = [c for c in checks if not c.passed]
        if rejections:
            return TradeDecision(approved=False, reasons=rejections)
        return TradeDecision(approved=True, sizing=self._compute_final_size(proposal))
```

### 1.4 Third Line: Operational Oversight (Watchdog)

**Owner:** Fully independent process with kill authority.

**Responsibilities:**
- Exchange connectivity monitoring (heartbeat, latency, spread sanity)
- Infrastructure reliability (CPU, memory, disk, network)
- API key security and rotation tracking
- Dead man's switch (if watchdog stops, trading halts -- fail-safe)
- System health checks and alerting (Telegram, SMS, email)
- Disaster recovery verification

**Critical pattern -- Dead Man's Switch:**
- Trading engine sends heartbeats every 10 seconds
- If watchdog misses 3 consecutive heartbeats (30s), it assumes crash
- Watchdog closes all positions via market order and sends alert
- If the WATCHDOG crashes, trading engine detects no heartbeat response and self-halts
- Default state is SAFE (no positions). Active effort required to be in the market.

```
Architecture (correct):

  [Third Line: Watchdog Process]
    |-- Heartbeat monitor (10s interval)
    |-- Exchange health check (5s interval)
    |-- System resource monitor (30s interval)
    |-- Authority: KILL everything, cannot be overridden
    |
    v
  [Second Line: Risk Engine Process]
    |-- VaR/CVaR computation
    |-- Drawdown tracking
    |-- Kill switch management
    |-- Order approval gate
    |-- Authority: reject/reduce orders, force exits
    |
    v
  [First Line: Trading Engine Process]
    |-- Signal generation
    |-- Position sizing proposal
    |-- Authority: propose orders ONLY
    |-- Cannot reach exchange directly
    |
    Exchange API (only reachable via Risk Engine)
```

### 1.5 Authority Hierarchy (Immutable)

| Level | Can Override Level Below | Can Be Overridden By | Runtime Modifiable |
|-------|------------------------|---------------------|--------------------|
| Third Line (Watchdog) | Yes -- can kill everything | Nothing | No -- requires code change + restart |
| Second Line (Risk Engine) | Yes -- can reject/reduce | Only Watchdog | No -- frozen config at startup |
| First Line (Trading) | No override capability | Risk Engine, Watchdog | Model parameters only |

---

## 2. Pre-Trade Risk Checks

### 2.1 The Institutional Standard: 10 Checks Before Every Order

Every institutional fund runs a checklist before placing any order. For crypto, the following 10 checks are the minimum. Every check must PASS. One failure rejects the order.

### 2.2 Check 1: Kill Switch Status

```python
def check_kill_switches(self) -> CheckResult:
    """Are any kill switches active?"""
    # Daily loss limit (3% of equity)
    # Weekly loss limit (5% of equity)
    # Max drawdown halt (15% peak-to-trough)
    # Consecutive loss limit (15 trades)
    # Max trades per day (30)
    # Manual halt (operator override)
    status = self.kill_switch_manager.get_status()
    if status.any_active:
        return CheckResult(
            passed=False,
            reason=f"Kill switch active: {status.active_switches}",
        )
    return CheckResult(passed=True)
```

### 2.3 Check 2: Volatility Guard

```python
def check_volatility_guard(self, current_vol_ann: float) -> CheckResult:
    """Is volatility within tradeable range?"""
    # Too low: no edge to exploit (< 15% annualized)
    # Too high: risk of gaps/slippage (> 150% annualized)
    if current_vol_ann < self.config.min_volatility_ann:
        return CheckResult(passed=False, reason="Volatility too low for edge")
    if current_vol_ann > self.config.max_volatility_ann:
        return CheckResult(passed=False, reason="Volatility too high -- gap risk")
    return CheckResult(passed=True)
```

### 2.4 Check 3: Drawdown Gate

```python
def check_drawdown_gate(self) -> CheckResult:
    """What is the current drawdown phase?"""
    # Phase 0 (0-3% DD): normal sizing
    # Phase 1 (3-5% DD): reduce to 75%
    # Phase 2 (5-8% DD): reduce to 50%
    # Phase 3 (8-12% DD): reduce to 25%
    # Phase 4 (12-15% DD): reduce to 10%
    # Phase 5 (>15% DD): HALT -- no new trades
    dd_state = self.drawdown_gate.get_state()
    if dd_state.phase >= 5:
        return CheckResult(passed=False, reason="Drawdown halt: >15%")
    return CheckResult(passed=True, size_multiplier=dd_state.size_fraction)
```

### 2.5 Check 4: VaR Limit

```python
def check_var_limit(self, proposed_order: TradeProposal) -> CheckResult:
    """Would this order push portfolio VaR beyond limit?"""
    current_var = self.compute_portfolio_var(confidence=0.95)
    incremental_var = self.compute_incremental_var(proposed_order)
    projected_var = current_var + incremental_var

    var_limit = self.equity * self.config.max_daily_var_pct  # 2%

    if projected_var > var_limit:
        return CheckResult(
            passed=False,
            reason=f"VaR breach: projected {projected_var:.0f} > limit {var_limit:.0f}",
        )
    return CheckResult(passed=True)
```

### 2.6 Check 5: Exposure Limits

```python
def check_exposure_limits(self, proposed_order: TradeProposal) -> CheckResult:
    """Gross and net exposure within bounds?"""
    current_gross = self.position_tracker.gross_exposure()
    new_gross = current_gross + proposed_order.notional_usd

    max_gross = self.equity * 0.40  # 40% of NAV

    if new_gross > max_gross:
        return CheckResult(passed=False, reason="Gross exposure limit breach")

    # Single position concentration
    if proposed_order.notional_usd > self.equity * 0.20:
        return CheckResult(passed=False, reason="Single position > 20% of NAV")

    return CheckResult(passed=True)
```

### 2.7 Check 6: Order Rate Limit

```python
def check_order_rate(self) -> CheckResult:
    """Prevent runaway order generation (code bug protection)."""
    orders_last_hour = self.count_orders_in_window(3600)
    orders_last_minute = self.count_orders_in_window(60)

    if orders_last_hour > 20:
        return CheckResult(passed=False, reason="Rate limit: >20 orders/hour")
    if orders_last_minute > 3:
        return CheckResult(passed=False, reason="Rate limit: >3 orders/minute")

    return CheckResult(passed=True)
```

### 2.8 Check 7: Liquidity Adequacy

```python
def check_liquidity(self, proposed_order: TradeProposal) -> CheckResult:
    """Is there enough order book depth for this trade?"""
    visible_depth = self.get_order_book_depth_usd()

    # Never use more than 10% of visible depth
    max_order = visible_depth * 0.10

    if proposed_order.notional_usd > max_order:
        return CheckResult(
            passed=False,
            reason=f"Order {proposed_order.notional_usd:.0f} > 10% of depth {visible_depth:.0f}",
        )

    # Check bid-ask spread -- wide spreads signal thin liquidity
    spread_bps = self.get_current_spread_bps()
    if spread_bps > 20:  # 0.2% spread = something wrong
        return CheckResult(passed=False, reason=f"Spread too wide: {spread_bps} bps")

    return CheckResult(passed=True)
```

### 2.9 Check 8: Margin Adequacy

```python
def check_margin(self, proposed_order: TradeProposal) -> CheckResult:
    """Is there sufficient margin for this trade + buffer?"""
    required_margin = proposed_order.notional_usd / self.leverage
    current_available = self.get_available_margin()

    # Require 2x the margin needed (50% buffer for adverse moves)
    if current_available < required_margin * 2.0:
        return CheckResult(
            passed=False,
            reason=f"Insufficient margin: need {required_margin*2:.0f}, have {current_available:.0f}",
        )
    return CheckResult(passed=True)
```

### 2.10 Check 9: Fat Finger Guard

```python
def check_fat_finger(self, proposed_order: TradeProposal) -> CheckResult:
    """Reject obviously wrong orders (code bug, data error)."""
    # Order size sanity
    if proposed_order.notional_usd > 10_000:  # Hard cap for $50K account
        return CheckResult(passed=False, reason="Fat finger: notional > $10K")

    # Price sanity -- is the proposed stop reasonable?
    current_price = self.get_current_price()
    stop_distance_pct = abs(proposed_order.proposed_stop - current_price) / current_price

    if stop_distance_pct > 0.10:  # 10% stop distance = suspicious
        return CheckResult(passed=False, reason=f"Fat finger: stop {stop_distance_pct:.1%} from price")

    if stop_distance_pct < 0.001:  # 0.1% stop = almost certainly wrong
        return CheckResult(passed=False, reason=f"Fat finger: stop too tight {stop_distance_pct:.3%}")

    # Direction sanity -- does the signal match the order?
    if proposed_order.confidence < 0.50:
        return CheckResult(passed=False, reason="Confidence below minimum threshold")

    return CheckResult(passed=True)
```

### 2.11 Check 10: Trading Hours / Calendar

```python
def check_trading_hours(self, timestamp: float) -> CheckResult:
    """Are we within approved trading hours?"""
    from datetime import datetime, timezone
    dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)

    # Trading window: 08:00-21:00 UTC
    if not (self.config.trading_start_hour_utc <= dt.hour < self.config.trading_end_hour_utc):
        return CheckResult(passed=False, reason="Outside trading hours")

    # Funding rate proximity -- avoid trading 5 min before/after funding
    minutes_to_funding = self.minutes_to_next_funding(timestamp)
    if minutes_to_funding < 5 or minutes_to_funding > (480 - 5):
        return CheckResult(passed=False, reason="Too close to funding settlement")

    return CheckResult(passed=True)
```

### 2.12 Pre-Trade Check Summary

| # | Check | Threshold | Failure Action |
|---|-------|-----------|----------------|
| 1 | Kill switches | Any active | Reject |
| 2 | Volatility guard | 15%-150% annualized | Reject |
| 3 | Drawdown gate | >15% DD | Reject; 3-15% reduces size |
| 4 | VaR limit | >2% daily 95% VaR | Reject |
| 5 | Exposure limits | >40% gross, >20% single | Reject |
| 6 | Order rate | >20/hour, >3/minute | Reject |
| 7 | Liquidity | >10% of depth, >20bps spread | Reject |
| 8 | Margin | <2x required margin | Reject |
| 9 | Fat finger | Size/price/stop sanity | Reject |
| 10 | Trading hours | 08-21 UTC, not near funding | Reject |

---

## 3. Real-Time Position Limits and Margin Monitoring

### 3.1 Position Limit Framework

```
POSITION LIMIT HIERARCHY (for $50K equity)
============================================

Hard limits (immutable at runtime):
  Max single position notional:     $10,000 (20% of NAV)
  Max gross exposure:               $20,000 (40% of NAV)
  Max open positions:               1
  Max leverage:                     3x
  Max position duration:            6 bars (30 minutes)

Soft limits (adjusted by drawdown gate):
  Normal sizing:                    quarter-Kelly, capped at 5% of NAV
  Phase 1 (3-5% DD):              75% of normal
  Phase 2 (5-8% DD):              50% of normal
  Phase 3 (8-12% DD):             25% of normal
  Phase 4 (12-15% DD):            10% of normal
  Phase 5 (>15% DD):              0% (halt)
```

### 3.2 Real-Time Margin Monitoring

Margin must be checked continuously, not just at trade entry.

```python
class MarginMonitor:
    """Continuous margin monitoring -- checked every bar (5 min)."""

    # Margin zones
    HEALTHY_MARGIN_RATIO = 3.0      # Available / required > 3x
    WARNING_MARGIN_RATIO = 2.0      # Reduce position
    CRITICAL_MARGIN_RATIO = 1.5     # Close position
    LIQUIDATION_BUFFER_RATIO = 1.2  # Emergency -- should never reach here

    def on_bar(self, current_price: float) -> MarginAction:
        """Called every 5-min bar."""
        available = self.get_available_margin()
        required = self.get_maintenance_margin()

        if required == 0:
            return MarginAction.NONE

        ratio = available / required

        if ratio < self.LIQUIDATION_BUFFER_RATIO:
            return MarginAction.EMERGENCY_CLOSE  # Market order NOW
        elif ratio < self.CRITICAL_MARGIN_RATIO:
            return MarginAction.CLOSE_POSITION   # Close within 1 bar
        elif ratio < self.WARNING_MARGIN_RATIO:
            return MarginAction.REDUCE_POSITION  # Reduce by 50%
        else:
            return MarginAction.NONE

    def compute_margin_metrics(self) -> dict:
        """Metrics for monitoring dashboard."""
        return {
            "available_margin_usd": self.get_available_margin(),
            "maintenance_margin_usd": self.get_maintenance_margin(),
            "margin_ratio": self.get_available_margin() / max(self.get_maintenance_margin(), 1),
            "unrealized_pnl_usd": self.get_unrealized_pnl(),
            "liquidation_price": self.compute_liquidation_price(),
            "distance_to_liquidation_pct": self.distance_to_liquidation(),
        }
```

### 3.3 Mark-to-Market Frequency

| Component | Update Frequency | Data Source |
|-----------|-----------------|-------------|
| Position P&L | Every bar (5 min) | Last trade price |
| Margin ratio | Every bar (5 min) | Exchange margin API |
| Unrealized P&L | Every bar (5 min) | Mark-to-market |
| Liquidation price | On position change | Exchange margin params |
| Funding cost accrual | Every 8 hours | Exchange funding rate |
| NAV calculation | Every bar (5 min) | Cash + unrealized P&L |

---

## 4. Counterparty Risk Management

### 4.1 The Post-FTX Reality

After FTX's collapse ($8B+ in customer losses), counterparty risk is the defining risk for any crypto fund. The core lesson: **your exchange is not a bank -- it has no deposit insurance, no regulatory backstop, and historically has commingled customer funds.**

Key statistics (2025-2026):
- 64% of market participants remain concerned about proof-of-reserves quality
- 47% cite counterparty risk as their top concern
- Over 55% of institutional hedge funds now invest directly in digital assets
- Post-FTX, many funds cap single-exchange exposure at 25-30% of NAV

### 4.2 Exchange Exposure Limits

```
EXCHANGE EXPOSURE FRAMEWORK
=============================

Rule 1: Never have >50% of total capital on ANY single exchange
  For $50K fund:
    Exchange A (primary, e.g., Binance): max $25K
    Exchange B (secondary, e.g., Bybit): max $15K
    Cold storage (not on exchange): min $10K (20%)

Rule 2: Only deposit what you need for trading margin + buffer
  If max position is $10K at 3x leverage:
    Required margin: ~$3,500
    Buffer (2x): $7,000
    Keep on exchange: $7,000-10,000
    Rest: cold storage or second exchange

Rule 3: Daily sweep excess funds off-exchange
  End of each trading day:
    If exchange_balance > required_margin * 3:
        withdraw(exchange_balance - required_margin * 2)

Rule 4: Evaluate exchange health monthly (see scorecard below)
```

### 4.3 Exchange Risk Scorecard

Rate each exchange monthly on a 1-5 scale:

```
EXCHANGE RISK SCORECARD
========================
Exchange: _____________     Date: _____________

Category                               Score (1-5)   Weight
-------------------------------------------------------
1. Proof of Reserves published?        [   ]         20%
   - Merkle tree proof available?
   - Third-party audit?
   - Frequency of updates?

2. Regulatory status                   [   ]         20%
   - Licensed in major jurisdiction?
   - Compliance with local regulations?
   - Any enforcement actions?

3. Financial transparency              [   ]         15%
   - Published financial statements?
   - Revenue sources known?
   - Insurance fund size disclosed?

4. Operational history                 [   ]         15%
   - Years in operation?
   - Major incidents/hacks?
   - Incident response quality?

5. Withdrawal reliability              [   ]         15%
   - Withdrawal delays reported?
   - Maximum withdrawal tested?
   - 24h withdrawal processing?

6. Technology and security             [   ]         15%
   - Cold storage %?
   - 2FA enforcement?
   - Bug bounty program?
   - SOC 2 / ISO 27001 certification?

COMPOSITE SCORE: _____ / 5.0

Action thresholds:
  4.0-5.0: Full allocation allowed
  3.0-3.9: Reduced allocation (max 30% of NAV)
  2.0-2.9: Minimal allocation (max 10% of NAV)
  < 2.0:   No allocation -- withdraw immediately
```

### 4.4 Counterparty Risk Monitoring Signals

Monitor these early warning indicators continuously:

```python
class CounterpartyMonitor:
    """Monitor exchange health signals."""

    WARNING_SIGNALS = {
        # Direct signals
        "withdrawal_delays": "Withdrawals taking >2 hours (normally <30 min)",
        "api_degradation": "API latency >5x normal or frequent errors",
        "spread_widening": "Spreads >3x normal on the exchange specifically",
        "insurance_fund_decline": "Insurance fund declining >10% in a week",

        # Indirect signals
        "exchange_token_decline": "Exchange token (BNB, etc.) dropping >20% in a week",
        "social_media_panic": "Surge in withdrawal-related posts",
        "competitor_issues": "Another exchange failing (contagion risk)",
        "regulatory_action": "New enforcement action or investigation announced",
        "key_personnel_departure": "CRO, CFO, or CEO sudden departure",
    }

    def daily_check(self) -> CounterpartyStatus:
        """Run daily counterparty health check."""
        signals = []

        # Check withdrawal speed (test withdrawal)
        withdrawal_time = self.test_small_withdrawal()
        if withdrawal_time > timedelta(hours=2):
            signals.append("withdrawal_delays")

        # Check API health
        latency = self.measure_api_latency()
        if latency > self.normal_latency * 5:
            signals.append("api_degradation")

        # Check insurance fund
        fund_change = self.get_insurance_fund_weekly_change()
        if fund_change < -0.10:
            signals.append("insurance_fund_decline")

        if len(signals) >= 2:
            return CounterpartyStatus.WITHDRAW_EXCESS
        elif len(signals) >= 1:
            return CounterpartyStatus.ELEVATED_RISK
        return CounterpartyStatus.NORMAL
```

### 4.5 Off-Exchange Custody (Post-FTX Best Practice)

Leading institutional crypto funds now use off-exchange settlement:
- **Copper ClearLoop:** Assets sit off-exchange in segregated custody; trades settle on-exchange but capital is not at exchange credit risk
- **Fireblocks:** Multi-party computation (MPC) custody with exchange settlement
- **Smaller funds:** At minimum, keep trading capital only (not full AUM) on exchange

For ep2-crypto ($50K fund), the practical approach:
1. Keep 40-50% of capital on primary exchange for trading margin
2. Keep 20-30% on secondary exchange (backup + diversification)
3. Keep 20-30% in self-custody (hardware wallet)
4. Daily sweep: if exchange balance exceeds 2x required margin, withdraw excess

---

## 5. Operational Risk

### 5.1 Taxonomy of Operational Risks

```
OPERATIONAL RISK CATEGORIES
=============================

A. Human Error
   - Fat finger trades (wrong size, wrong direction)
   - Deploying untested code to production
   - Misconfiguring risk parameters
   - Forgetting to restart after maintenance
   - Accidentally revoking active API keys

B. Code Bugs
   - Look-ahead bias in features (most common)
   - Integer overflow in position sizing
   - Race condition in order submission
   - Wrong sign convention (buy vs sell)
   - Timezone errors (UTC vs local)
   - Float precision errors in price comparison

C. Infrastructure Failures
   - Server crash mid-position
   - Network partition (connected to exchange, can't reach monitoring)
   - Database corruption
   - Clock drift (affects time-based features)
   - Disk full (can't write logs/state)

D. External Failures
   - Exchange API outage
   - Exchange matching engine lag
   - Data feed corruption (wrong prices)
   - DNS resolution failure
   - Cloud provider outage

E. Security Incidents
   - API key compromise
   - Phishing attack
   - Malware on trading machine
   - Supply chain attack (compromised dependency)
```

### 5.2 Prevention Framework

```python
# Prevention: code-level safeguards

class OperationalSafeguards:
    """Embedded in the trading system codebase."""

    # 1. Fat Finger Prevention
    MAX_ORDER_NOTIONAL_USD = 10_000  # Hard-coded, not configurable
    MAX_ORDER_QUANTITY_BTC = 0.50     # Even at $100K BTC = $50K
    MIN_ORDER_QUANTITY_BTC = 0.001

    # 2. Direction Sanity
    def validate_order_direction(self, signal: str, order_side: str) -> bool:
        """Signal and order must agree."""
        valid_combos = {
            ("long", "buy"), ("long", "BUY"),
            ("short", "sell"), ("short", "SELL"),
            ("close_long", "sell"), ("close_long", "SELL"),
            ("close_short", "buy"), ("close_short", "BUY"),
        }
        return (signal, order_side) in valid_combos

    # 3. Price Sanity
    def validate_price(self, price: float, reference_price: float) -> bool:
        """Order price must be within 5% of reference."""
        deviation = abs(price - reference_price) / reference_price
        return deviation < 0.05

    # 4. State Consistency
    def validate_state_consistency(self) -> bool:
        """Cross-check internal state with exchange state."""
        local_position = self.position_tracker.current_position()
        exchange_position = self.exchange.get_position()

        if abs(local_position.quantity - exchange_position.quantity) > 0.0001:
            logger.critical(
                "Position mismatch",
                local=local_position.quantity,
                exchange=exchange_position.quantity,
            )
            return False
        return True

    # 5. Deployment Safeguard
    def pre_deployment_check(self) -> bool:
        """Must pass before any code deployment."""
        checks = [
            self.run_all_tests(),          # pytest exit code 0
            self.run_type_check(),         # mypy exit code 0
            self.run_lint(),               # ruff exit code 0
            self.verify_config_valid(),    # Pydantic validation passes
            self.verify_db_connectivity(), # Can connect to database
            self.verify_exchange_api(),    # API key works, read-only test
        ]
        return all(checks)
```

### 5.3 Detection Framework

```python
class OperationalAnomalyDetector:
    """Detect operational issues in real-time."""

    def on_bar(self) -> list[OperationalAlert]:
        alerts = []

        # 1. Position reconciliation (every bar)
        if not self.reconcile_positions():
            alerts.append(OperationalAlert(
                severity="CRITICAL",
                message="Position mismatch between local and exchange",
                action="HALT_TRADING",
            ))

        # 2. P&L sanity (every bar)
        bar_pnl = self.compute_bar_pnl()
        if abs(bar_pnl) > self.equity * 0.02:  # 2% in 5 minutes = suspicious
            alerts.append(OperationalAlert(
                severity="HIGH",
                message=f"Abnormal bar P&L: {bar_pnl:.2f}",
                action="INVESTIGATE",
            ))

        # 3. Order fill rate (rolling 1 hour)
        fill_rate = self.compute_fill_rate(window=3600)
        if fill_rate < 0.50:  # <50% fills = something wrong
            alerts.append(OperationalAlert(
                severity="MEDIUM",
                message=f"Low fill rate: {fill_rate:.0%}",
                action="CHECK_EXCHANGE",
            ))

        # 4. Latency monitoring
        api_latency_ms = self.measure_api_latency()
        if api_latency_ms > 5000:
            alerts.append(OperationalAlert(
                severity="HIGH",
                message=f"API latency {api_latency_ms}ms",
                action="CONSIDER_HALT",
            ))

        # 5. Clock drift
        exchange_time = self.get_exchange_server_time()
        local_time = time.time()
        drift_ms = abs(exchange_time - local_time) * 1000
        if drift_ms > 1000:  # >1 second drift
            alerts.append(OperationalAlert(
                severity="MEDIUM",
                message=f"Clock drift: {drift_ms:.0f}ms",
                action="SYNC_CLOCK",
            ))

        return alerts
```

### 5.4 Recovery Procedures

```
OPERATIONAL RECOVERY PLAYBOOK
================================

Scenario: Server crash mid-position
  1. Watchdog detects missing heartbeat (30s)
  2. Watchdog closes all positions via exchange API
  3. Watchdog sends alert (Telegram + SMS)
  4. On restart: reconcile local state with exchange
  5. Resume trading only after operator confirmation

Scenario: Exchange API outage
  1. Detect: 3 consecutive failed API calls
  2. Switch to backup exchange if position is on backup
  3. If position is on affected exchange: WAIT (cannot close via API)
  4. Use exchange web UI or mobile app as manual fallback
  5. Log incident; review after resolution

Scenario: Position mismatch detected
  1. HALT all new orders immediately
  2. Query exchange for authoritative position state
  3. Update local state to match exchange
  4. Investigate: was there a missed fill? A phantom order?
  5. Resume only after root cause identified

Scenario: Suspected API key compromise
  1. Revoke ALL keys immediately on exchange
  2. Check trade history for unauthorized trades
  3. Check withdrawal history for unauthorized transfers
  4. Generate new keys with fresh IP whitelist
  5. Rotate all secrets in Doppler
  6. Review system logs for intrusion
  7. Resume only after full security audit
```

---

## 6. Stress Testing Requirements

### 6.1 Scenario Catalog

```
STRESS TEST SCENARIOS
======================

HISTORICAL REPLAYS (must-have):
ID    Event                    BTC Move       Timeframe    Data Required
H-1   COVID crash (Mar 2020)   -52%           36 hours     1-min OHLCV, depth, funding
H-2   China ban (May 2021)     -53%           2 weeks      5-min OHLCV, OI, funding
H-3   Terra/Luna (May 2022)    -40%           1 week       1-min OHLCV, liquidations
H-4   FTX collapse (Nov 2022)  -25%           3 days       5-min OHLCV, funding, OI
H-5   Leverage flush (Apr '21) -27%           1 day        1-min OHLCV, $10B liqs

SYNTHETIC SCENARIOS (hypothetical):
ID    Scenario                    Parameters
S-1   Flash crash                 -20% in 5 min, depth drops 95%
S-2   Exchange outage             Primary down 2 hours mid-position
S-3   Funding rate spike          -0.5% per 8h for 3 settlements
S-4   Liquidity crisis            Order book depth drops 80% for 4 hours
S-5   Correlation breakdown       All assets correlate at 0.95
S-6   Extended grind              -30% over 60 days (no single large drop)
S-7   Black swan                  -70% in 1 week (worse than any historical)
S-8   Whipsaw                     +15%, -25%, +10% in 24 hours
S-9   Gap after weekend           Opens -10% from Friday close
S-10  Cascading liquidations      $5B liquidations in 1 hour

OPERATIONAL STRESS:
ID    Scenario                    Tests
O-1   API failover                Primary exchange down, switch to backup
O-2   Data feed corruption        Bad price data for 10 bars
O-3   Model degradation           Model accuracy drops to 50% for 1 week
O-4   System restart mid-trade    Recovery and state reconciliation
```

### 6.2 Stress Test Schedule

| Test Type | Frequency | Trigger | Pass Criteria |
|-----------|-----------|---------|---------------|
| Historical replays (H-1 to H-5) | Monthly | Always | Max DD < scenario-specific limit |
| Synthetic scenarios (S-1 to S-10) | Quarterly | Or when new risk factors emerge | Survive without forced liquidation |
| Reverse stress test | Monthly | Or when position size changes | Destruction scenario is implausible |
| Liquidity stress (S-4) | Weekly | Or when depth changes >30% | Can exit in <5 minutes with <50bps slip |
| Operational stress (O-1 to O-4) | Monthly | Or after any code change | System recovers within 5 minutes |
| Full suite | Before any model deployment | Mandatory | All scenarios pass |
| Ad-hoc | Immediately | Market conditions change rapidly | Case-by-case |

### 6.3 Stress Test Implementation

```python
class StressTestRunner:
    """Run portfolio through crisis scenarios."""

    HISTORICAL_SCENARIOS = {
        "H-1_covid": {
            "start": "2020-03-11", "end": "2020-03-15",
            "description": "BTC -52% in 36 hours",
            "max_acceptable_loss_pct": 15.0,
            "slippage_multiplier": 5.0,    # 5x normal slippage
            "depth_multiplier": 0.20,       # 80% depth reduction
        },
        "H-3_terra_luna": {
            "start": "2022-05-07", "end": "2022-05-13",
            "description": "BTC -40%, cascading liquidations",
            "max_acceptable_loss_pct": 12.0,
            "slippage_multiplier": 3.0,
            "depth_multiplier": 0.30,
        },
        "H-4_ftx": {
            "start": "2022-11-06", "end": "2022-11-11",
            "description": "BTC -25%, exchange counterparty risk",
            "max_acceptable_loss_pct": 10.0,
            "slippage_multiplier": 2.0,
            "depth_multiplier": 0.40,
        },
    }

    def run_all_scenarios(self) -> StressTestReport:
        results = []
        for name, scenario in self.HISTORICAL_SCENARIOS.items():
            result = self._run_single_scenario(name, scenario)
            results.append(result)

        return StressTestReport(
            date=datetime.utcnow(),
            results=results,
            all_passed=all(r.passed for r in results),
        )

    def reverse_stress_test(self, target_loss_pct: float = 0.50) -> list[dict]:
        """What would cause a 50% loss?"""
        scenarios = []

        # Price move needed
        position = self.get_current_position()
        if position.notional > 0:
            btc_move = -target_loss_pct * self.equity / position.notional
            scenarios.append({
                "type": "price_move",
                "btc_move_pct": btc_move * 100,
                "plausible": abs(btc_move) < 0.30,  # BTC has moved 30% in a day
                "historical_precedent": "March 2020" if abs(btc_move) < 0.55 else "None",
            })

        # Liquidity failure
        scenarios.append({
            "type": "liquidity_failure",
            "description": "Exchange halt during max position, forced liquidation at -20% slippage",
            "plausible": True,  # BitMEX 2020
        })

        # Exchange insolvency
        scenarios.append({
            "type": "exchange_failure",
            "description": "Exchange becomes insolvent, funds frozen/lost",
            "loss": self.get_exchange_balance(),
            "plausible": True,  # FTX 2022
        })

        return scenarios
```

### 6.4 Pass/Fail Criteria

| Scenario | Max Acceptable DD | Max Acceptable Loss (on $50K) | Recovery Time |
|----------|-------------------|-------------------------------|---------------|
| COVID crash | 15% | $7,500 | 2 weeks |
| Terra/Luna | 12% | $6,000 | 1 week |
| FTX collapse | 10% | $5,000 | 3 days |
| Flash crash | 8% | $4,000 | 1 day |
| Black swan (-70%) | 15% (kill switch halts) | $7,500 | N/A (halted) |

If any scenario exceeds its limit, position sizing must be reduced until it passes.

---

## 7. Risk Reporting for LPs

### 7.1 What LPs Expect

Institutional limited partners expect standardized risk reporting. Even for a small fund, maintaining this discipline is essential for attracting future allocators and for self-governance.

### 7.2 Monthly Risk Report Template

```
═══════════════════════════════════════════════════════════
       [FUND NAME] — MONTHLY RISK REPORT
       Period: [MONTH YEAR]
       Report Date: [DATE]
       Prepared By: [RISK OFFICER / CRO]
═══════════════════════════════════════════════════════════

1. EXECUTIVE SUMMARY
─────────────────────
NAV (start of month):          $XX,XXX
NAV (end of month):            $XX,XXX
Monthly Return (net):          X.XX%
YTD Return (net):              X.XX%
Monthly Sharpe (annualized):   X.XX
YTD Sharpe (annualized):       X.XX

Key Events This Month:
• [Brief description of significant events]

2. PERFORMANCE METRICS
─────────────────────
                        Month       3-Month     6-Month     YTD
Gross Return            X.XX%       X.XX%       X.XX%       X.XX%
Net Return              X.XX%       X.XX%       X.XX%       X.XX%
Sharpe Ratio            X.XX        X.XX        X.XX        X.XX
Sortino Ratio           X.XX        X.XX        X.XX        X.XX
Calmar Ratio            X.XX        X.XX        X.XX        X.XX
Win Rate                XX.X%       XX.X%       XX.X%       XX.X%
Profit Factor           X.XX        X.XX        X.XX        X.XX
Max Drawdown            X.XX%       X.XX%       X.XX%       X.XX%
Drawdown Duration       X days      X days      X days      X days
# of Trades             XXX         XXX         XXX         XXX
Avg Trade Duration      X.X bars    X.X bars    X.X bars    X.X bars

3. RISK METRICS
─────────────────────
                        Current     Month Avg   Limit       Status
Daily VaR (95%)         $XXX        $XXX        $X,XXX      [OK/BREACH]
Daily VaR (99%)         $XXX        $XXX        N/A         [INFO]
CVaR / ES (95%)         $XXX        $XXX        $X,XXX      [OK/BREACH]
Stressed VaR (99%)      $XXX        N/A         $X,XXX      [OK/BREACH]
Max Daily Loss          $XXX        N/A         $X,XXX      [OK/BREACH]
Max Weekly Loss         $XXX        N/A         $X,XXX      [OK/BREACH]
Peak-to-Trough DD       X.XX%       N/A         15%         [OK/BREACH]
Gross Exposure          XX.X%       XX.X%       40%         [OK/BREACH]

4. RISK LIMIT COMPLIANCE
─────────────────────
Limit                           # Times Approached   # Times Breached
Daily Loss Limit (3%)           X                    X
Weekly Loss Limit (5%)          X                    X
Drawdown Halt (15%)             X                    X
VaR Limit (2% daily 95%)       X                    X
Order Rate Limit (20/hr)        X                    X
Volatility Guard (15-150%)      X                    X

Kill Switch Activations This Month: X
  Details: [Date, type, duration, reason]

5. EXPOSURE ANALYSIS
─────────────────────
Average Gross Exposure:         XX.X% of NAV
Average Net Exposure:           XX.X% of NAV (long/short bias)
Max Gross Exposure (date):      XX.X% on [date]
Average Leverage Used:          X.Xx
Instrument:                     BTCUSDT Perpetual (100%)
Exchange Distribution:          Binance XX%, Bybit XX%

6. COUNTERPARTY EXPOSURE
─────────────────────
Exchange        Balance     % of NAV    Risk Score    Status
Binance         $XX,XXX     XX%         X.X/5.0       [OK/WATCH]
Bybit           $XX,XXX     XX%         X.X/5.0       [OK/WATCH]
Cold Storage    $XX,XXX     XX%         N/A           [OK]

7. TRADING COSTS
─────────────────────
Total Commissions:              $XXX
Total Funding Costs:            $XXX
Estimated Slippage:             $XXX
Total Trading Costs:            $XXX (XX bps round-trip avg)
Cost as % of Gross P&L:        XX.X%

8. STRESS TEST RESULTS
─────────────────────
Scenario            Last Run    Result      Max DD      Status
COVID March 2020    [date]      [P/F]       X.X%        [PASS/FAIL]
Terra/Luna 2022     [date]      [P/F]       X.X%        [PASS/FAIL]
FTX Collapse 2022   [date]      [P/F]       X.X%        [PASS/FAIL]
Flash Crash -20%    [date]      [P/F]       X.X%        [PASS/FAIL]
Black Swan -70%     [date]      [P/F]       X.X%        [PASS/FAIL]

Reverse Stress Test:
  "What BTC move would cause 50% loss?"    Answer: X.X%
  "Is this plausible?"                      Answer: [Yes/No]

9. MODEL HEALTH
─────────────────────
Rolling Accuracy (30-day):      XX.X% (target: 52-56%)
CUSUM Alpha Decay Signal:       [Normal/Warning/Alarm]
Calibration (are 60% signals winning 60%?): [Good/Degraded/Failed]
Feature Drift (PSI):            [Normal/Elevated/High]
Days Since Last Retrain:        XX (threshold: 14)

10. OPERATIONAL METRICS
─────────────────────
System Uptime:                  XX.XX%
API Connection Drops:           X
Average Order Latency:          XXXms
Position Reconciliation Errors: X
Alert Events (non-risk):        X

11. REGULATORY & COMPLIANCE
─────────────────────
Regulatory Changes This Month:  [None / Description]
Compliance Actions Required:    [None / Description]
License/Registration Status:    [Current / Pending / N/A]

═══════════════════════════════════════════════════════════
NEXT REVIEW: [Date]
RISK OFFICER SIGN-OFF: _________________ Date: _________
═══════════════════════════════════════════════════════════
```

### 7.3 Quarterly LP Letter: Additional Sections

Beyond the monthly report, quarterly letters should add:

```
QUARTERLY SUPPLEMENT
════════════════════

A. STRATEGY COMMENTARY
   - Market environment description
   - How strategy performed relative to expectations
   - Attribution: which signals contributed most to P&L
   - Any parameter changes made and rationale

B. RISK FRAMEWORK CHANGES
   - Risk parameter adjustments (with before/after)
   - New risk checks added
   - Kelly criterion recalibration results
   - Changes to exchange exposure allocation

C. COMPETITIVE LANDSCAPE
   - BTC perp market structure changes (spreads, depth, funding)
   - New competitor activity (MEV, new bots)
   - Alpha half-life assessment

D. FORWARD-LOOKING RISK ASSESSMENT
   - Upcoming macro events (FOMC, CPI, halving, etc.)
   - Known regulatory changes ahead
   - Expected volatility regime for next quarter
   - Planned model retraining schedule

E. KEY METRICS (Form PF aligned for SEC compliance)
   - Investment exposures by asset class
   - Borrowing and counterparty exposure
   - Currency exposure
   - Turnover
   - Portfolio liquidity profile
   - Financing and investor liquidity
```

### 7.4 Risk Report Delivery Schedule

| Report | Frequency | Delivery | Recipients |
|--------|-----------|----------|------------|
| Daily risk summary | Daily | Automated email/Telegram | Operator |
| Weekly risk review | Weekly | PDF + meeting | Operator + advisor |
| Monthly risk report | Monthly | PDF, T+15 calendar days | LPs, operator |
| Quarterly LP letter | Quarterly | PDF, T+45 calendar days | LPs, board |
| Annual risk review | Annually | PDF + presentation | LPs, board, auditor |
| Ad-hoc incident report | As needed | Within 24 hours | LPs, board |

---

## 8. VaR and CVaR for Crypto Portfolios

### 8.1 Why Standard VaR Fails for Crypto

BTC return characteristics that break standard VaR:

| Property | Normal Distribution | BTC Returns |
|----------|-------------------|-------------|
| Kurtosis | 3.0 | 5-12 (fat tails) |
| Skewness | 0.0 | -0.5 to -2.0 (left skew) |
| Volatility clustering | No | Extreme (GARCH effects) |
| Serial correlation | No | Present at 5-min level |
| Regime changes | No | Frequent (bull/bear/chop) |

Consequence: Parametric VaR (assumes normality) will **underestimate tail risk by 30-50%** for crypto.

### 8.2 Recommended Method: GARCH-EVT-CVaR

The academic consensus for crypto risk measurement is a three-stage pipeline:

**Stage 1: GARCH for Volatility Dynamics**
```python
# GJR-GARCH(1,1) with skewed Student's t innovations
# Captures: volatility clustering, leverage effect, fat tails

from arch import arch_model

def fit_garch_model(returns: np.ndarray) -> arch_model:
    """Fit GJR-GARCH(1,1) with skewed-t distribution."""
    model = arch_model(
        returns,
        vol="GARCH",
        p=1, o=1, q=1,  # GJR-GARCH(1,1)
        dist="skewt",    # Skewed Student's t for crypto
    )
    result = model.fit(disp="off")
    return result

# Output: conditional volatility sigma_t for each time step
# This captures the time-varying nature of crypto vol
```

**Stage 2: EVT (Extreme Value Theory) for Tails**
```python
from scipy.stats import genpareto

def fit_evt_tail(
    standardized_residuals: np.ndarray,
    threshold_quantile: float = 0.95,
) -> dict:
    """Fit Generalized Pareto Distribution to tail losses.

    EVT models the tail separately from the body of the distribution.
    This is critical for crypto because:
    1. The body is somewhat normal-ish
    2. The tails are MUCH heavier than normal
    3. Modeling them separately gives better tail risk estimates
    """
    # Extract tail exceedances
    threshold = np.quantile(standardized_residuals, threshold_quantile)
    exceedances = standardized_residuals[standardized_residuals > threshold] - threshold

    # Fit GPD to exceedances
    shape, loc, scale = genpareto.fit(exceedances, floc=0)

    return {
        "shape": shape,    # xi > 0 means heavy tails (crypto: typically 0.1-0.4)
        "scale": scale,    # sigma
        "threshold": threshold,
        "n_exceedances": len(exceedances),
        "n_total": len(standardized_residuals),
    }
```

**Stage 3: VaR and CVaR Computation**
```python
def compute_var_cvar_garch_evt(
    garch_result,
    evt_params: dict,
    portfolio_value: float,
    confidence: float = 0.95,
    horizon_bars: int = 1,  # 1 bar = 5 minutes
) -> dict:
    """Compute VaR and CVaR using GARCH-EVT model.

    For crypto 5-min horizon:
    - Use conditional volatility from GARCH (time-varying)
    - Use EVT tail model for tail quantiles
    - Scale by sqrt(horizon) for multi-bar horizons
    """
    # Current conditional volatility from GARCH
    sigma_t = garch_result.conditional_volatility[-1] / 100  # Convert from %

    # EVT parameters
    xi = evt_params["shape"]
    beta = evt_params["scale"]
    u = evt_params["threshold"]
    n_exceed = evt_params["n_exceedances"]
    n_total = evt_params["n_total"]

    # Tail probability
    p = 1 - confidence  # e.g., 0.05 for 95%
    exceed_prob = n_exceed / n_total

    # GPD quantile for VaR
    var_standardized = u + (beta / xi) * ((p / exceed_prob) ** (-xi) - 1)

    # Scale by current conditional volatility
    var_dollar = var_standardized * sigma_t * portfolio_value * np.sqrt(horizon_bars)

    # CVaR (Expected Shortfall) from GPD
    # ES = VaR / (1 - xi) + (beta - xi * u) / (1 - xi)
    if xi < 1:
        es_standardized = var_standardized / (1 - xi) + (beta - xi * u) / (1 - xi)
        cvar_dollar = es_standardized * sigma_t * portfolio_value * np.sqrt(horizon_bars)
    else:
        cvar_dollar = float("inf")  # xi >= 1 means infinite expected shortfall

    return {
        "var_95_dollar": var_dollar,
        "cvar_95_dollar": cvar_dollar,
        "var_95_pct": var_dollar / portfolio_value,
        "cvar_95_pct": cvar_dollar / portfolio_value,
        "conditional_vol_ann": sigma_t * np.sqrt(105_120),  # Crypto annualization
        "garch_regime": "high_vol" if sigma_t * np.sqrt(105_120) > 0.80 else "normal",
        "evt_tail_index": xi,
    }
```

### 8.3 Alternative Methods Comparison

| Method | Pros | Cons | Accuracy for Crypto |
|--------|------|------|-------------------|
| Historical VaR | No distributional assumptions | Only as good as history window | Medium -- misses new regimes |
| Parametric (Normal) | Fast, simple | Assumes normality | Poor -- underestimates 30-50% |
| Parametric (Student's t) | Captures some fat tails | Fixed degrees of freedom | Fair |
| Monte Carlo (Normal) | Flexible | Still assumes normality | Poor |
| Monte Carlo (Student's t, df=4) | Fat tails, fast | Doesn't capture vol clustering | Good |
| GARCH-Normal | Captures vol clustering | Misses extreme tails | Fair-Good |
| **GARCH-EVT** | **Vol clustering + extreme tails** | **More complex, needs calibration** | **Best** |
| GARCH-EVT-Copula | Adds multi-asset dependency | Overkill for single-asset | Best for multi-asset |

**Recommendation for ep2-crypto:** Use GARCH-EVT for risk monitoring. Use Historical VaR as a sanity check. Use Monte Carlo (Student's t, df=4) for stress testing.

### 8.4 VaR/CVaR Limits Framework

```
RISK LIMIT HIERARCHY
=====================

Level 1 (Green):   Daily VaR (95%) < 2% of NAV
                   -> Normal trading

Level 2 (Yellow):  Daily VaR (95%) between 2-3% of NAV
                   -> Reduce position size by 50%

Level 3 (Red):     Daily VaR (95%) > 3% of NAV
                   -> Close to flat, wait for vol to decline

Level 4 (Black):   CVaR (95%) > 5% of NAV
                   -> Close ALL positions immediately

Additionally:
  Stressed VaR (99%) should not exceed 10% of NAV
  If it does, the system is over-leveraged for crisis conditions
```

### 8.5 VaR Backtesting (Kupiec Test)

VaR must be validated -- does the model actually work?

```python
def backtest_var(
    daily_returns: np.ndarray,
    var_forecasts: np.ndarray,
    confidence: float = 0.95,
) -> dict:
    """Kupiec proportion of failures test.

    If VaR is calibrated correctly, the number of exceedances
    should match the expected frequency.

    For 95% VaR: expect ~5% exceedances.
    If significantly more: VaR model is inadequate.
    If significantly fewer: VaR model is too conservative (capital inefficient).
    """
    n = len(daily_returns)
    exceedances = np.sum(daily_returns < -var_forecasts)
    exceedance_rate = exceedances / n
    expected_rate = 1 - confidence

    # Kupiec likelihood ratio test
    from scipy.stats import chi2

    if exceedances == 0 or exceedances == n:
        lr_stat = float("inf")
    else:
        lr_stat = -2 * (
            np.log((1 - expected_rate) ** (n - exceedances) * expected_rate ** exceedances)
            - np.log((1 - exceedance_rate) ** (n - exceedances) * exceedance_rate ** exceedances)
        )

    p_value = 1 - chi2.cdf(lr_stat, df=1)

    return {
        "n_observations": n,
        "n_exceedances": int(exceedances),
        "exceedance_rate": exceedance_rate,
        "expected_rate": expected_rate,
        "kupiec_lr_stat": lr_stat,
        "kupiec_p_value": p_value,
        "var_model_adequate": p_value > 0.05,  # Fail to reject at 5%
    }
```

---

## 9. Liquidity Risk

### 9.1 BTC Perpetual Futures Liquidity Profile

BTC perpetual futures on Binance are the most liquid crypto derivative, but liquidity is still fragile:

| Condition | Top-10 Bid Depth | Spread | Slippage (for $50K order) |
|-----------|------------------|--------|---------------------------|
| Normal | $5-20M | 0.01-0.05% | < 0.01% |
| Moderate stress | $2-5M | 0.05-0.15% | 0.01-0.05% |
| High stress (May 2022) | $0.5-2M | 0.10-0.50% | 0.1-0.5% |
| Cascade (March 2020) | < $0.5M | > 0.50% | 1-5%+ |

Key liquidity characteristics:
- Depth declines 4-5% during volatility spikes
- Liquidity is time-of-day dependent: deepest 08:00-21:00 UTC, thinnest 02:00-06:00 UTC
- Weekend depth is 30-50% lower than weekday
- Liquidation cascades can temporarily remove 95% of visible depth

### 9.2 Liquidity Risk Metrics

```python
class LiquidityRiskMetrics:
    """Compute and monitor liquidity risk metrics."""

    def compute_metrics(self) -> dict:
        """Full liquidity assessment -- run every bar."""
        order_book = self.get_order_book()

        return {
            # 1. Spread metrics
            "bid_ask_spread_bps": self.compute_spread(order_book),
            "weighted_mid_price": self.compute_weighted_mid(order_book),

            # 2. Depth metrics
            "bid_depth_usd_10bps": self.depth_within_bps(order_book, "bid", 10),
            "ask_depth_usd_10bps": self.depth_within_bps(order_book, "ask", 10),
            "bid_depth_usd_50bps": self.depth_within_bps(order_book, "bid", 50),
            "total_depth_ratio": self.current_depth / self.avg_depth_30d,

            # 3. Impact metrics
            "estimated_slippage_bps": self.estimate_slippage(
                self.max_position_size, order_book
            ),

            # 4. Resilience metrics
            "depth_recovery_time_sec": self.measure_depth_recovery(),

            # 5. Concentration metrics
            "order_book_imbalance": self.compute_obi(order_book),
        }

    def compute_liquidation_risk(self) -> dict:
        """How vulnerable are we to cascading liquidations?"""
        open_interest = self.get_open_interest()
        funding_rate = self.get_funding_rate()
        long_short_ratio = self.get_long_short_ratio()

        # High OI + extreme funding + skewed L/S = cascade risk
        cascade_risk_score = 0

        if open_interest > self.oi_90th_percentile:
            cascade_risk_score += 1
        if abs(funding_rate) > 0.001:  # 0.1% per 8h = elevated
            cascade_risk_score += 1
        if long_short_ratio > 2.0 or long_short_ratio < 0.5:
            cascade_risk_score += 1

        return {
            "open_interest_usd": open_interest,
            "funding_rate_8h": funding_rate,
            "long_short_ratio": long_short_ratio,
            "cascade_risk_score": cascade_risk_score,  # 0-3
            "cascade_risk_level": ["low", "medium", "high", "extreme"][cascade_risk_score],
        }
```

### 9.3 Position Sizing Adjusted for Liquidity

```python
def liquidity_adjusted_position_size(
    kelly_position_usd: float,
    current_depth_usd: float,
    avg_depth_30d_usd: float,
    max_depth_usage_pct: float = 0.10,
) -> float:
    """Reduce position size when liquidity is poor.

    Three constraints applied:
    1. Never use more than 10% of visible depth
    2. Reduce proportionally when depth < 50% of normal
    3. Kelly position is the upper bound
    """
    # Constraint 1: liquidity cap
    liquidity_cap = current_depth_usd * max_depth_usage_pct

    # Constraint 2: proportional reduction if depth is low
    depth_ratio = current_depth_usd / avg_depth_30d_usd
    if depth_ratio < 0.50:
        liquidity_adjusted = kelly_position_usd * depth_ratio
    else:
        liquidity_adjusted = kelly_position_usd

    # Final: minimum of all constraints
    return min(kelly_position_usd, liquidity_cap, liquidity_adjusted)
```

### 9.4 Liquidity Stress Testing

```
LIQUIDITY STRESS SCENARIOS
============================

Scenario 1: Normal Exit
  Conditions: Depth at 30-day average
  Test: Can we exit full position with <10bps slippage?
  Pass criteria: Yes, within 1 bar

Scenario 2: Stressed Exit
  Conditions: Depth at 20% of normal (80% reduction)
  Test: Can we exit full position with <50bps slippage?
  Pass criteria: Yes, within 5 bars

Scenario 3: Cascade Exit
  Conditions: Depth at 5% of normal (95% reduction)
  Test: Can we exit full position with <200bps slippage?
  Pass criteria: Yes, but accept significant slippage

Scenario 4: Extended Illiquidity
  Conditions: Depth at 30% of normal for 4 hours
  Test: Can we TWAP exit over 4 hours?
  Pass criteria: Average slippage <30bps

Scenario 5: Exchange Specific
  Conditions: Primary exchange depth drops to zero (outage)
  Test: Can we hedge on backup exchange?
  Pass criteria: Hedge placed within 2 minutes
```

---

## 10. Regulatory Risk

### 10.1 Current Regulatory Landscape (2025-2026)

```
REGULATORY FRAMEWORK STATUS
=============================

EU -- MiCA (Markets in Crypto-Assets):
  Status: Fully effective December 30, 2024
  Impact: Mandatory asset segregation, complaint handling,
          transparent risk disclosures
  Compliance deadline: Some providers have until June 30, 2026
  Key stats:
    - 55% of crypto hedge funds reviewing MiCA impact
    - 30% intend to increase EU exposure under clearer framework
    - 35% of startups estimate >$500K annual compliance cost
    - 412M in fines issued for violations in 2024
    - 58+ CASPs had licenses revoked by early 2025

Basel Committee -- Bank Crypto Exposure:
  Status: Capital standards effective January 1, 2025
  Impact: Group 2 cryptoassets carry 1,250% risk-weight
  Bank disclosure requirements: effective 2026
  Review of prudential rules: ongoing

US -- SEC / CFTC:
  Status: Evolving; Form PF updates effective June 12, 2025
  Form PF requires: quarterly/annual reporting on exposures,
    counterparty risk, risk metrics, liquidity, leverage
  Impact on crypto hedge funds:
    - More granular disclosure requirements
    - Counterparty exposure reporting
    - Portfolio liquidity reporting

Global Trends:
  - Tax reporting frameworks expanding (OECD CARF)
  - Travel Rule enforcement increasing
  - Stablecoin regulation tightening globally
  - DeFi regulation still developing
```

### 10.2 Regulatory Risk Monitoring Framework

```python
class RegulatoryRiskMonitor:
    """Track and assess regulatory changes affecting operations."""

    # Jurisdictions to monitor based on fund domicile and exchange locations
    MONITORED_JURISDICTIONS = [
        "US_SEC", "US_CFTC", "US_FINCEN",
        "EU_MICA", "EU_ESMA",
        "UK_FCA",
        "CAYMAN_CIMA",  # Common fund domicile
        "SINGAPORE_MAS",
        "HONG_KONG_SFC",
    ]

    # Exchange-specific regulatory risks
    EXCHANGE_REGULATORY_RISKS = {
        "binance": {
            "jurisdiction": "multiple",
            "known_issues": [
                "US DOJ settlement 2023 ($4.3B)",
                "Restricted in multiple jurisdictions",
            ],
            "risk_level": "elevated",
        },
        "bybit": {
            "jurisdiction": "Dubai/UAE",
            "known_issues": [],
            "risk_level": "moderate",
        },
    }

    def monthly_regulatory_review(self) -> RegulatoryReport:
        """Monthly check on regulatory changes."""
        changes = []

        for jurisdiction in self.MONITORED_JURISDICTIONS:
            recent = self.check_regulatory_updates(jurisdiction)
            if recent:
                changes.extend(recent)

        return RegulatoryReport(
            date=datetime.utcnow(),
            changes=changes,
            action_required=[c for c in changes if c.impact == "high"],
            upcoming_deadlines=self.get_upcoming_deadlines(),
        )
```

### 10.3 Regulatory Risk Mitigation

```
REGULATORY RISK MITIGATION CHECKLIST
======================================

1. FUND STRUCTURE
   [ ] Establish in regulatory-friendly jurisdiction (Cayman, BVI, Delaware)
   [ ] Separate management company from fund entity
   [ ] Obtain necessary registrations (SEC if >$150M AUM for Form PF)
   [ ] Engage crypto-experienced legal counsel

2. COMPLIANCE INFRASTRUCTURE
   [ ] AML/KYC program documented and implemented
   [ ] Transaction monitoring for suspicious activity
   [ ] Sanctions screening for counterparties
   [ ] Record retention policy (minimum 5 years)
   [ ] Compliance calendar with regulatory deadlines

3. OPERATIONAL COMPLIANCE
   [ ] Trade allocation and best execution policy
   [ ] Conflicts of interest policy
   [ ] Personal trading policy
   [ ] Valuation policy (mark-to-market methodology)
   [ ] Business continuity plan
   [ ] Cybersecurity policy and incident response plan

4. REPORTING COMPLIANCE
   [ ] Form PF (if SEC-registered, quarterly/annual)
   [ ] Form ADV (if SEC-registered, annual)
   [ ] MiCA reporting (if EU operations)
   [ ] Tax reporting (fund-level and investor K-1s)
   [ ] FATCA/CRS reporting (if applicable)

5. EXCHANGE COMPLIANCE
   [ ] Only use exchanges licensed in their operating jurisdictions
   [ ] Maintain records of exchange due diligence
   [ ] Monitor exchange regulatory status changes
   [ ] Have backup exchanges in case primary loses license
   [ ] IP whitelist API keys to known infrastructure
```

### 10.4 Scenario Planning for Regulatory Events

```
REGULATORY SCENARIO PLAYBOOK
================================

Scenario R-1: Primary Exchange Banned in Fund's Jurisdiction
  Preparation:
    - Backup exchange always funded and configured
    - Strategy tested on backup exchange
    - Max 70% of capital on any single exchange
  Response:
    - Withdraw funds within 24 hours of announcement
    - Migrate to backup exchange
    - Notify LPs within 48 hours

Scenario R-2: Crypto Derivatives Regulated as Securities
  Preparation:
    - Monitor SEC/CFTC proposals
    - Legal counsel on retainer for rapid assessment
    - Fund structure allows quick jurisdiction change
  Response:
    - Legal assessment within 72 hours
    - Comply or migrate jurisdiction
    - LP communication within 1 week

Scenario R-3: Tax Treatment Changes (e.g., unrealized gains tax)
  Preparation:
    - Maintain detailed trade records for any tax methodology
    - Tax advisor on retainer
  Response:
    - Quantify impact within 1 week
    - Adjust strategy if economics change materially
    - LP communication with impact analysis

Scenario R-4: KYC/AML Enforcement Freezes Exchange Account
  Preparation:
    - Maintain clean transaction history
    - Document fund of funds trail
    - Keep backup exchange accounts
  Response:
    - Engage legal counsel immediately
    - Provide requested documentation
    - Use backup exchange for ongoing operations
    - LP notification if material impact

Scenario R-5: Mandatory Licensing for Algorithmic Trading
  Preparation:
    - Monitor ESMA and SEC algo trading rules
    - Maintain audit trail of all algorithmic decisions
    - Document risk controls per MiFID II / SEC standards
  Response:
    - Apply for license within deadline
    - Implement required pre/post-trade controls
    - Engage compliance consultant for gap analysis
```

---

## Summary: Implementation Priority

For ep2-crypto, implement these components in order of criticality:

| Priority | Component | Why First |
|----------|-----------|-----------|
| 1 | Three Lines of Defense architecture | Foundation -- everything depends on this |
| 2 | Pre-trade risk checks (all 10) | Prevents entering bad trades |
| 3 | Kill switches and drawdown gate | Limits losses when things go wrong |
| 4 | VaR/CVaR (GARCH-EVT) | Quantifies risk for position sizing |
| 5 | Margin monitoring | Prevents liquidation |
| 6 | Counterparty risk (exchange limits) | Post-FTX survival requirement |
| 7 | Stress testing suite | Validates all the above works |
| 8 | Operational safeguards | Prevents code/human errors |
| 9 | Liquidity monitoring | Adjusts execution for market conditions |
| 10 | LP reporting | Governance and transparency |
| 11 | Regulatory monitoring | Long-term compliance |

The existing ep2-crypto risk engine (`src/ep2_crypto/risk/`) already implements components 2-3 (kill switches, drawdown gate, volatility guard, position sizer, position tracker). Components 4-11 need to be added.

---

## Sources

- [Industry Guide to Crypto Hedge Funds (2025 Edition)](https://www.cryptoinsightsgroup.com/resources/industry-guide-to-crypto-hedge-funds-2025-edition)
- [What a 30% Crypto Drawdown Reveals About the Future of Digital-Asset Hedge Funds](https://www.hedgeco.net/news/02/2026/what-a-30-crypto-drawdown-reveals-about-the-future-of-digital-asset-hedge-funds.html)
- [Counterparty Risk in Crypto: Understanding the Potential Threats](https://www.merklescience.com/counterparty-risk-in-crypto-understanding-the-potential-threats)
- [Understanding Crypto Custodian Risk: How to Measure and Price Counterparty Exposure](https://www.agioratings.io/insights/crypto-custodian-risk-how-to-measure-and-price-counterparty-exposure)
- [Post-FTX Crypto Exchange Risk Management and Corporate Governance Reform](https://www.ainvest.com/news/post-ftx-crypto-exchange-risk-management-corporate-governance-reform-lessons-collapse-path-2509/)
- [Counterparty risk the top concern for crypto derivatives market](https://www.acuiti.io/counterparty-risk-the-top-concern-for-crypto-derivatives-market/)
- [Ultimate Guide to Hedge Fund Investor Reporting](https://chartergroupadmin.com/index.php/2025/02/24/ultimate-guide-to-hedge-fund-investor-reporting/)
- [Form PF; Reporting Requirements for All Filers and Large Hedge Fund Advisers](https://www.federalregister.gov/documents/2025/04/11/2025-05267/form-pf-reporting-requirements-for-all-filers-and-large-hedge-fund-advisers)
- [Regulatory Reporting Requirements for Hedge Funds](https://www.arcesium.com/blog/hedge-funds-regulatory-reporting-requirements-explained)
- [Optimization of Financial Asset Portfolio Using GARCH-EVT-Copula-CVaR Model](https://www.scirp.org/journal/paperinformation?paperid=144934)
- [Risk management and volatility prediction in crypto markets using AI-driven systems](https://semantjournals.org/index.php/AJTA/article/view/2968)
- [Crypto Regulation in 2026: What Changed and What's Ahead](https://sumsub.com/blog/global-crypto-regulations/)
- [MiCA Regulation Guide 2026: EU Crypto-Asset Framework Explained](https://complyfactor.com/mica-regulation-guide-2026-eu-crypto-asset-framework-explained/)
- [Global Crypto Policy Review Outlook 2025/26 Report](https://www.trmlabs.com/reports-and-whitepapers/global-crypto-policy-review-outlook-2025-26)
- [Bitcoin's Order Book Fragility and Whale-Driven Volatility](https://www.ainvest.com/news/bitcoin-order-book-fragility-whale-driven-volatility-high-leverage-environment-2512/)
- [The Rhythm of Liquidity: Temporal Patterns in Market Depth](https://blog.amberdata.io/the-rhythm-of-liquidity-temporal-patterns-in-market-depth)
- [Crypto Alpha From Volatility and Inefficiency (Hedge Fund Journal)](https://thehedgefundjournal.com/amphibian-quant-crypto-alpha-volatility-inefficiency/)
- [Liquibit's Market Neutral Crypto Strategy (Hedge Fund Journal)](https://thehedgefundjournal.com/liquibit-market-neutral-crypto-strategy-traditional-trading/)
- [Launching Your Crypto Hedge Fund: A Comprehensive 2025 Blueprint](https://cryptoresearch.report/crypto-research/launching-your-crypto-hedge-fund-a-comprehensive-2025-blueprint/)
- IIA Three Lines Model (2020)
- Basel Committee on Banking Supervision -- Prudential Treatment of Cryptoasset Exposures (2024)
- FIA Best Practices for Automated Trading Risk Controls (2024)
