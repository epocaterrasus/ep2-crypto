# Institutional Risk Management: Deep Research for Solo Crypto Trading

> Lessons from the best quant funds, the worst blowups, and how to implement institutional-grade risk management for a $10K-$100K solo BTC trading system.

---

## Table of Contents

1. [Risk Management at Top Quant Firms](#1-risk-management-at-top-quant-firms)
2. [Crypto Fund Failures: Lessons from Blowups](#2-crypto-fund-failures-lessons-from-blowups)
3. [Three Lines of Defense Model](#3-three-lines-of-defense-model)
4. [Independent Risk Monitoring](#4-independent-risk-monitoring)
5. [Risk Committee Decisions for Solo Traders](#5-risk-committee-decisions-for-solo-traders)
6. [Value at Risk (VaR) in Practice](#6-value-at-risk-var-in-practice)
7. [Stress Testing at Institutional Level](#7-stress-testing-at-institutional-level)
8. [Operational Risk Management](#8-operational-risk-management)
9. [Liquidity Risk Management](#9-liquidity-risk-management)
10. [Risk Culture and Psychology](#10-risk-culture-and-psychology)

---

## 1. Risk Management at Top Quant Firms

### 1.1 Renaissance Technologies (Medallion Fund)

**What's known publicly:**

Renaissance Technologies, founded by Jim Simons, operates the Medallion Fund -- widely considered the most successful hedge fund in history. While deeply secretive, several risk management principles have been publicly confirmed:

**Position Sizing via Kelly Criterion:**
Elwyn Berlekamp (a co-inventor of the Kelly criterion's application to investing) rewrote Medallion's algorithms to use Kelly-optimal sizing. The formula:

```
Kelly % = W - (1 - W) / R

Where:
  W = win rate (probability of winning trade)
  R = win/loss ratio (average win / average loss)
```

In practice, Renaissance reportedly uses fractional Kelly (typically half-Kelly or less) to reduce variance. With thousands of positions, the law of large numbers works in their favor.

**Leverage and Diversification:**
- Average leverage: 12.5x, sometimes reaching 20x when data confidence is high
- At peak, $60 billion in positions on $5 billion capital (via basket options)
- Diversification across 4,000 long and 4,000 short equity positions simultaneously
- Market-neutral strategy: long/short balanced to minimize directional exposure
- With 12.5x leverage, even a 0.015% daily edge compounds to ~60% annual returns

**Trust in Systems Over Humans:**
During the August 2007 "quant quake," Medallion lost $1 billion in three days. Renaissance's response: they did NOT intervene. They trusted their automated systems. The fund ended 2007 up 85.9%. This is perhaps the most important lesson: systems that work should not be overridden by human panic.

**Capacity Limits:**
Medallion is capped at $10-15 billion AUM. Profits are distributed to employees every six months. This prevents the fund from growing too large and eroding its edge -- a form of risk management that most funds ignore.

**Key Takeaway for ep2-crypto:** Position sizing should be mathematically derived (Kelly or fractional Kelly), not gut-based. Diversification is the primary risk reducer at RenTech -- for a single-instrument system (BTC perps), we must compensate with smaller position sizes and tighter drawdown limits.

### 1.2 Two Sigma

**Risk Management Structure:**
- Centralized risk management function separate from portfolio management
- During the 2008 financial crisis, Two Sigma's funds finished with high single-digit returns while most hedge funds lost heavily
- Investors specifically praised Two Sigma's "risk management capabilities" as the key differentiator
- Technology-first approach: AI, machine learning, and distributed computing underpin risk models
- Collaborative culture across disciplines prevents siloed risk assessment

**Key Takeaway for ep2-crypto:** The risk function must be separate from the trading function. Even in a solo operation, the risk checking code must be an independent process that the trading logic cannot bypass.

### 1.3 Citadel

**Multi-Manager Pod Structure:**
- Each "pod" (team) operates as an independent unit with strict drawdown thresholds
- Capital allocation decided by CIO office based on Sharpe ratios, drawdowns, correlations
- Factor sensitivities monitored centrally and daily: interest rates, credit spreads, currency risk, sector concentration
- If a strategy breaches risk limits, capital is resized immediately -- no negotiation
- Capital moves from underperforming pods to outperforming pods in real-time

**Risk Budget System:**
Each pod operates within a defined "risk budget" expressed as:
- Maximum VaR allocation
- Maximum drawdown threshold (typically 3-5% of allocated capital before review, 7-10% before forced reduction)
- Correlation limits to other pods (to prevent hidden concentration)
- Gross and net exposure limits

**Key Takeaway for ep2-crypto:** The "risk budget" concept is directly applicable. Set a VaR limit and a max drawdown threshold. If breached, the system reduces exposure automatically -- no human override allowed.

### 1.4 D.E. Shaw

**Risk Allocation Methodology:**
- Centralized and collaborative -- unlike Citadel's pod autonomy
- Research-intensive: risk models are developed collaboratively across disciplines
- Known for attracting mathematicians, physicists, and computer scientists to build risk models
- The centralized approach means risk assessment considers cross-strategy correlations holistically

**Key Takeaway for ep2-crypto:** Even with a single strategy, consider how different signal components correlate with each other during stress. If all your signals become correlated in a crash, your effective diversification drops to zero.

### 1.5 Common Themes Across All Four Firms

| Practice | RenTech | Two Sigma | Citadel | DE Shaw |
|----------|---------|-----------|---------|---------|
| Risk separate from trading | Yes | Yes | Yes | Yes |
| Automated risk limits | Yes | Yes | Yes | Yes |
| Position sizing formulas | Kelly | Proprietary | VaR-based | Proprietary |
| Drawdown triggers | System | System | System | System |
| Human override allowed | No (2007 proof) | Limited | No (auto-resize) | Limited |
| Leverage control | 12-20x w/ diversification | Conservative | Pod-level limits | Conservative |

---

## 2. Crypto Fund Failures: Lessons from Blowups

### 2.1 Alameda Research / FTX -- The Complete Risk Management Failure

**What happened:**
- Alameda Research, Sam Bankman-Fried's trading firm, was given a $65 billion negative balance exemption on FTX
- Alameda was exempt from FTX's liquidation protocols -- the very risk controls that protected every other user
- At least $10 billion in FTX customer funds were used by Alameda for trading, venture investments, real estate, and political donations
- No formalities for intercompany transactions -- assets and liabilities moved between entities without documentation
- Neither FTX nor Alameda ever published audited balance sheets
- By mid-2022, Alameda owed FTX over $10 billion, undisclosed to anyone

**Root Cause Analysis -- Risk Management Failures:**

1. **No independence between risk and trading:** The person making trading decisions (SBF) also controlled the exchange's risk parameters. The fox was guarding the henhouse.
2. **Exemption from own risk controls:** The single most damning fact. Alameda was exempt from liquidation -- the core risk mechanism. Any system that allows exemptions from its own risk rules is not a risk system at all.
3. **No transparency or audit trail:** Zero documentation of intercompany transfers. No audited financials. Risk management requires visibility.
4. **Commingled funds:** Customer assets and trading capital were not separated. The most basic custody risk control was violated.
5. **Concentration risk:** Alameda held massive positions in illiquid FTT tokens (FTX's own token), creating a circular dependency.

**Specific Losses:**
- Alameda recorded up to $12 billion in losses in under two months surrounding the Terra/Luna collapse (May 2022)
- Total customer losses from FTX: approximately $8 billion

**Key Takeaway for ep2-crypto:** Risk rules must be immutable. The trading system cannot grant itself exemptions. Separation of concerns between trading logic and risk enforcement is not optional -- it is THE lesson of FTX.

### 2.2 Three Arrows Capital (3AC) -- Over-Leverage Without Risk Management

**What happened:**
- Founded in 2012 by Kyle Davies and Su Zhu, managing ~$10 billion at peak
- Invested $200 million in LUNA tokens (February 2022), which went to near-zero by May 2022
- Used extreme leverage: borrowed $189 million in USDC/USDT using ETH as collateral
- Could not repay loans or meet margin calls
- Owed over $3.5 billion to more than 20 counterparties
- Ordered to liquidate June 27, 2022

**Root Cause Analysis:**

1. **No position limits relative to NAV:** $200M in a single token (LUNA) when they had no mechanism to cut the position if it declined
2. **Leverage without stress testing:** They never asked "what happens if LUNA goes to zero?" which is the literal definition of reverse stress testing
3. **Counterparty opacity:** They misled counterparties about their exposure levels
4. **Correlation blindness:** Their positions were all correlated to crypto sentiment -- when one failed, everything failed simultaneously
5. **No drawdown triggers:** There was no automatic mechanism to reduce exposure when losses mounted

**The Math of Their Failure:**
```
Starting capital: ~$10B
LUNA position: $200M (2% of NAV -- seems reasonable)
But: LUNA went to $0 (-100%)
Plus: All other crypto positions declined 50-70%
Plus: Leverage amplified all losses
Plus: Margin calls forced sales at worst prices

Result: Total loss exceeding available capital
```

**Key Takeaway for ep2-crypto:** Even a "small" 2% allocation can be lethal when combined with leverage and correlated positions. Stress test everything with the assumption that your worst-case scenario is not bad enough. 3AC's real problem was not the LUNA position -- it was the leverage on top of correlated positions with no circuit breakers.

### 2.3 Jump Crypto / Tai Mo Shan -- Hidden Exposure and Market Manipulation

**What happened:**
- In May 2021, when UST first depegged, Jump secretly bought ~$20 million in UST to stabilize the price -- this created a false sense of security
- In February 2022, Jump led a $1 billion funding round for LUNA Foundation Guard
- Jump suggested creating a Bitcoin reserve pool to defend UST's peg
- Three months later, Terra collapsed entirely
- Jump's subsidiary Tai Mo Shan settled with SEC for $123 million for "misleading investors"

**Root Cause Analysis:**

1. **Conflict of interest:** Jump was simultaneously a market maker, investor, and secret price supporter for the same asset
2. **Propping up a failing system:** By secretly stabilizing UST in 2021, they prevented the market from pricing in the fundamental weakness, making the eventual collapse worse
3. **Doubling down:** Instead of cutting exposure after the 2021 depegging scare, they increased their bet by leading a $1B investment round
4. **Sunk cost fallacy:** The more they invested, the more incentive they had to keep the system alive, regardless of fundamentals

**Key Takeaway for ep2-crypto:** Never add to a losing position hoping to "fix" the situation. If a position triggers a risk alert, reduce -- never increase. The system should enforce this mechanically.

### 2.4 Wintermute -- The $160M Operational Risk Failure

**What happened (September 2022):**
- Wintermute, a major crypto market maker, lost ~$160 million across 90 assets
- The attack vector: a vanity wallet address generated using the Profanity tool
- This wallet had admin permissions to their vault (could execute withdrawals)
- The Profanity tool had a known vulnerability that allowed private key recovery
- The attacker recovered the private key and simply asked the vault to send them $160M

**Root Cause Analysis:**

1. **Using a known-vulnerable tool for critical infrastructure:** Profanity's vulnerability was publicly known before the hack
2. **Excessive permissions on a single key:** One compromised key could drain $160M
3. **No multi-signature requirement:** A single key should never authorize large transfers
4. **DeFi-specific operational risk:** Smart contract interactions create attack surfaces that don't exist in CeFi

**Impact:**
- Wintermute remained solvent ($320M equity remaining)
- Several token pairs experienced temporary liquidity shortages
- Post-hack: restructured hot wallet management, implemented multi-signature custody

**Key Takeaway for ep2-crypto:** Operational security is risk management. API keys with excessive permissions, single points of failure in key management, and failure to rotate credentials are all operational risks that can cause losses larger than any trading loss.

### 2.5 Summary Table: Crypto Fund Failures

| Fund | Loss | Primary Failure | Risk Rule Violated |
|------|------|----------------|-------------------|
| Alameda/FTX | $8B+ (customer) | Exempted self from risk controls | Independence of risk function |
| 3AC | $10B+ (total) | Leverage without stress testing | Position limits, drawdown triggers |
| Jump Crypto | $123M+ (settlement) | Doubling down on failing investment | Stop-loss discipline, conflict management |
| Wintermute | $160M | Operational security failure | Key management, least-privilege access |

---

## 3. Three Lines of Defense Model

### 3.1 The Framework

The Three Lines of Defense (3LOD) was created by the Institute of Internal Auditors. In institutional finance, it prevents risks from "falling through the cracks" by assigning clear ownership at each level.

**First Line: Strategy-Level Risk (Owned by the trader/PM)**
- Position sizing per trade
- Stop-loss placement and enforcement
- Entry/exit criteria
- Signal quality gating
- Individual trade risk (e.g., "risk no more than 1% of NAV per trade")

**Second Line: Portfolio-Level Risk (Owned by risk management function)**
- Aggregate exposure limits (gross, net, directional)
- Correlation monitoring across positions
- VaR computation and limit enforcement
- Drawdown monitoring and trigger enforcement
- Regime-based parameter adjustment

**Third Line: Operational Risk (Owned by independent audit/oversight)**
- Exchange counterparty risk assessment
- Infrastructure reliability monitoring
- API key security and rotation
- Cybersecurity controls
- System health checks and alerting
- Disaster recovery and business continuity

### 3.2 Implementation for a Solo Trader

For a solo trader, you are all three lines. The solution: encode each line as a separate software component with different authority levels.

```
Architecture:

[Third Line: Operational Monitor]
  |
  |-- Monitors: exchange connectivity, API health, system resources
  |-- Authority: can halt ALL trading, cannot be overridden
  |-- Runs as: separate process, watchdog timer
  |
  v
[Second Line: Risk Engine]
  |
  |-- Monitors: portfolio exposure, VaR, drawdown, correlations
  |-- Authority: can reduce positions, reject new orders
  |-- Runs as: independent process wrapping the trading engine
  |
  v
[First Line: Trading Engine]
  |
  |-- Executes: signal generation, position sizing, order placement
  |-- Authority: can propose trades, cannot bypass risk limits
  |-- Runs as: child process of the risk engine
```

### 3.3 Concrete Implementation

```python
# First Line: Strategy-level risk checks (within trading engine)
class StrategyRiskChecks:
    """Embedded in the trading engine. These are advisory."""

    def check_position_size(self, signal_confidence: float, nav: float) -> float:
        """Kelly-based position sizing with fractional Kelly."""
        # Use quarter-Kelly for crypto volatility
        kelly_fraction = 0.25
        win_rate = self.model.estimated_win_rate
        win_loss_ratio = self.model.estimated_payoff_ratio

        full_kelly = win_rate - (1 - win_rate) / win_loss_ratio
        position_pct = max(0, full_kelly * kelly_fraction)

        # Hard cap: never more than 5% of NAV in a single trade
        position_pct = min(position_pct, 0.05)

        return position_pct * nav

    def check_stop_loss(self, entry_price: float, direction: str) -> float:
        """ATR-based stop loss."""
        atr = self.compute_atr(period=14)
        multiplier = 2.0  # 2x ATR stop

        if direction == "long":
            return entry_price - (atr * multiplier)
        else:
            return entry_price + (atr * multiplier)


# Second Line: Portfolio-level risk (independent process)
class PortfolioRiskEngine:
    """Runs as a separate process. Wraps the trading engine."""

    # Hard limits -- coded as constants, not config
    MAX_GROSS_EXPOSURE_PCT = 0.30       # 30% of NAV
    MAX_DAILY_VAR_PCT = 0.02            # 2% daily VaR limit
    MAX_DRAWDOWN_SOFT_PCT = 0.05        # 5% drawdown -> reduce size by 50%
    MAX_DRAWDOWN_HARD_PCT = 0.10        # 10% drawdown -> halt trading
    MAX_DAILY_LOSS_PCT = 0.03           # 3% daily loss -> halt for day
    MAX_CORRELATION_TO_BTC_SPOT = 0.95  # alert if strategy too correlated

    def approve_order(self, proposed_order: Order) -> OrderDecision:
        """Every order must pass through this gate."""
        checks = [
            self._check_gross_exposure(proposed_order),
            self._check_var_limit(proposed_order),
            self._check_drawdown_state(),
            self._check_daily_loss(),
            self._check_position_concentration(),
        ]

        rejections = [c for c in checks if not c.approved]
        if rejections:
            logger.warning(
                "Order rejected by risk engine",
                reasons=[r.reason for r in rejections],
                order=proposed_order,
            )
            return OrderDecision(approved=False, reasons=rejections)

        return OrderDecision(approved=True)

    def _check_drawdown_state(self) -> CheckResult:
        """Drawdown-based position reduction."""
        current_dd = self.compute_current_drawdown()

        if current_dd >= self.MAX_DRAWDOWN_HARD_PCT:
            self.halt_trading("Hard drawdown limit breached")
            return CheckResult(approved=False, reason="HARD_DRAWDOWN_BREACH")

        if current_dd >= self.MAX_DRAWDOWN_SOFT_PCT:
            self.reduce_position_size_by(0.50)  # Cut size in half
            return CheckResult(
                approved=True,
                warning="SOFT_DRAWDOWN: position size reduced 50%"
            )

        return CheckResult(approved=True)


# Third Line: Operational watchdog (fully independent process)
class OperationalWatchdog:
    """Separate process with kill authority. Uses watchdog pattern."""

    HEARTBEAT_TIMEOUT_SECONDS = 30
    MAX_API_LATENCY_MS = 5000
    MAX_SPREAD_BPS = 50  # 0.5% spread = something is wrong

    def run(self):
        """Main watchdog loop. If this crashes, trading halts (fail-safe)."""
        while True:
            try:
                self._check_heartbeat()
                self._check_exchange_connectivity()
                self._check_spread_sanity()
                self._check_system_resources()
                self._check_api_key_expiry()
            except WatchdogAlert as e:
                self._emergency_halt(reason=str(e))

            time.sleep(5)

    def _emergency_halt(self, reason: str):
        """Kill the trading process. Close all positions."""
        logger.critical("EMERGENCY HALT", reason=reason)
        self.trading_process.terminate()
        self._close_all_positions_market_order()
        self._send_alert(reason)  # SMS, email, Telegram
```

### 3.4 Key Principle: Fail-Safe Design

The operational watchdog (third line) should use a **dead man's switch** pattern:
- The trading engine must send heartbeats to the watchdog every N seconds
- If the watchdog stops receiving heartbeats, it assumes the trading engine is hung/crashed
- The watchdog closes all positions and sends alerts
- If the WATCHDOG itself crashes, the trading engine should detect this and halt itself

This is the same pattern used by nuclear reactors and commercial aviation. The default state is SAFE (no positions), and active effort is required to maintain an UNSAFE state (holding positions).

---

## 4. Independent Risk Monitoring

### 4.1 Why Risk Must Be Separate from Trading

Every major fund blowup shares one trait: the person controlling risk was the same person controlling trading.

- **Alameda/FTX:** SBF controlled both the exchange risk parameters and the trading firm's decisions
- **3AC:** Kyle Davies and Su Zhu were both PMs and risk managers
- **LTCM (1998):** The partners who built the risk models also made the trading decisions
- **Barings Bank (1995):** Nick Leeson was both front-office trader and back-office settlement

The institutional solution: the Chief Risk Officer (CRO) reports to the board, NOT to the Chief Investment Officer (CIO). They have veto power over trades.

### 4.2 For Solo Trading: Risk Engine Wraps Trading Engine

The critical design pattern: **the risk engine is the parent process, the trading engine is the child.**

```
WRONG architecture (trading wraps risk):
  Trading Engine
    |-- generates signal
    |-- calls risk.check() <-- can be skipped!
    |-- places order

RIGHT architecture (risk wraps trading):
  Risk Engine (parent process)
    |-- spawns Trading Engine (child process)
    |-- intercepts ALL orders from child
    |-- applies risk checks
    |-- forwards approved orders to exchange
    |-- Trading Engine CANNOT reach exchange directly
```

### 4.3 Implementation: Risk Checks That Cannot Be Overridden

```python
# The risk engine controls the exchange connection
# The trading engine only has a reference to a "broker proxy"
# that routes through the risk engine

class RiskGatedBroker:
    """
    The trading engine thinks this IS the broker.
    In reality, every order goes through risk checks first.
    The trading engine has no direct exchange access.
    """

    def __init__(self, real_broker: Exchange, risk_engine: RiskEngine):
        self._broker = real_broker  # private -- trading engine cannot access
        self._risk = risk_engine

    def place_order(self, order: Order) -> OrderResult:
        """The only way to reach the exchange."""
        decision = self._risk.evaluate(order)

        if not decision.approved:
            logger.warning("Order blocked by risk engine", order=order)
            return OrderResult(status="REJECTED", reason=decision.reason)

        return self._broker.place_order(order)

    def cancel_order(self, order_id: str) -> OrderResult:
        """Cancellations always allowed (reducing risk)."""
        return self._broker.cancel_order(order_id)


# The trading engine receives the gated broker, not the real one
class TradingEngine:
    def __init__(self, broker: RiskGatedBroker, model: Model):
        self.broker = broker  # This is actually the risk-gated proxy
        self.model = model
        # The trading engine has NO reference to the real exchange

    def on_signal(self, signal: Signal):
        order = self._generate_order(signal)
        result = self.broker.place_order(order)  # Goes through risk engine
        if result.status == "REJECTED":
            # Trading engine must accept rejections gracefully
            logger.info("Signal rejected by risk", signal=signal)
```

### 4.4 Hard Limits That Cannot Be Changed at Runtime

```python
# These limits are defined as frozen dataclass or constants
# They CANNOT be modified while the system is running
# Changing them requires: stop system -> edit config -> review -> restart

from dataclasses import dataclass

@dataclass(frozen=True)
class HardRiskLimits:
    """Immutable at runtime. Change requires system restart."""

    max_position_pct_of_nav: float = 0.20       # 20% max single position
    max_gross_exposure_pct: float = 0.40         # 40% max gross exposure
    max_daily_loss_pct: float = 0.03             # 3% daily loss halt
    max_drawdown_halt_pct: float = 0.10          # 10% drawdown halt
    max_drawdown_reduce_pct: float = 0.05        # 5% drawdown -> reduce
    max_daily_var_pct: float = 0.02              # 2% VaR limit
    max_orders_per_hour: int = 20                # Prevent runaway orders
    max_notional_per_order_usd: float = 10_000   # For $50K account
    min_time_between_orders_sec: int = 30        # Prevent rapid-fire
    forced_close_drawdown_pct: float = 0.15      # 15% -> close everything
```

---

## 5. Risk Committee Decisions for Solo Traders

At institutional funds, a risk committee meets regularly to review, audit, and adjust. For a solo trader, this must be systematized as scheduled reviews with checklists.

### 5.1 Weekly Risk Review (Every Sunday, ~30 minutes)

**Metrics to Review:**

```
WEEKLY RISK REVIEW CHECKLIST
============================

[ ] 1. P&L Summary
    - Total P&L this week ($ and %)
    - Sharpe ratio (rolling 30-day)
    - Win rate vs expected win rate
    - Average win vs average loss

[ ] 2. Risk Limit Compliance
    - Were any risk limits breached this week?
    - How many orders were rejected by risk engine?
    - What was the maximum drawdown this week?
    - What was the maximum daily VaR this week?

[ ] 3. Position Sizing Consistency
    - Were position sizes consistent with Kelly formula output?
    - Any deviation between planned risk and realized risk?
    - Slippage: planned vs actual

[ ] 4. Signal Quality
    - Model confidence distribution this week
    - Number of signals generated vs acted upon
    - False signal rate

[ ] 5. Operational Health
    - API uptime this week
    - Number of connection drops
    - Maximum latency observed
    - Any error alerts triggered?

[ ] 6. Market Regime Assessment
    - Current detected regime (trending/mean-reverting/volatile)
    - Is the model performing in-regime?
    - Any regime transitions this week?

DECISION: Continue / Reduce Size / Pause Trading
NOTES: ________________________________________
```

### 5.2 Monthly Risk Audit (First Saturday of each month, ~2 hours)

```
MONTHLY RISK AUDIT CHECKLIST
=============================

[ ] 1. Performance Deep Dive
    - Monthly P&L ($ and %)
    - Rolling Sharpe ratio (90-day)
    - Sortino ratio
    - Calmar ratio (return / max drawdown)
    - Maximum drawdown this month
    - Longest drawdown duration

[ ] 2. Model Health
    - Is the model's accuracy degrading? (CUSUM test)
    - Feature importance shifts
    - Prediction calibration: are 70% confidence signals winning 70%?
    - Compare live performance to backtest expectations

[ ] 3. Risk Engine Audit
    - Review ALL risk rejections: were they correct?
    - Review ALL risk limit breaches: root cause analysis
    - Were there trades that SHOULD have been rejected but weren't?
    - Backtest the month: if risk limits were tighter, what changes?

[ ] 4. Operational Review
    - Rotate API keys if >60 days old
    - Check exchange fee tier (has volume changed tier?)
    - Review withdrawal whitelist
    - Test backup systems (failover exchange, alert systems)
    - Check if any exchange policy changes affect operations

[ ] 5. Counterparty Risk
    - Check exchange proof of reserves (if available)
    - Review any news about exchange solvency
    - Ensure funds are distributed across exchanges if warranted
    - Check that withdrawal limits are adequate

[ ] 6. Stress Test Results
    - Run current portfolio through historical stress scenarios
    - Review: would the portfolio survive March 2020? May 2022?
    - Review: what is the current max loss at 99% confidence?

DECISION: Adjust risk parameters? Retrain model? Pause?
NEXT MONTH FOCUS: ________________________________
```

### 5.3 Quarterly Risk Parameter Review (Every 3 months, ~half day)

```
QUARTERLY RISK PARAMETER REVIEW
================================

[ ] 1. Should Risk Limits Change?
    - Review the past quarter's realized volatility
    - If vol is structurally higher/lower, adjust VaR limits
    - Review: are position sizes still appropriate for current NAV?
    - Review: is the daily loss limit still appropriate?

[ ] 2. Kelly Criterion Recalibration
    - Recalculate win rate from last 90 days of live data
    - Recalculate win/loss ratio from last 90 days
    - Update Kelly fraction if significantly different
    - Compare new Kelly to previous Kelly -- any red flags?

[ ] 3. Strategy Edge Assessment
    - Is the edge decaying? (Run formal CUSUM test)
    - Compare Sharpe ratio across quarters
    - If edge is declining: root cause analysis
    - Decision: continue / adjust / deprecate strategy

[ ] 4. Market Structure Review
    - Has BTC perp market structure changed? (spreads, depth, funding)
    - Has competitive landscape changed? (new bots, MEV, etc.)
    - Any regulatory changes affecting operations?
    - Are there new data sources that should be integrated?

[ ] 5. Capital Allocation Review
    - Is current capital allocation optimal?
    - Should capital be added or withdrawn?
    - Review: what is the target NAV for next quarter?
    - Review max drawdown tolerance vs. psychological tolerance

PARAMETER CHANGES (document every change with rationale):
| Parameter | Old Value | New Value | Rationale |
|-----------|-----------|-----------|-----------|
|           |           |           |           |
```

### 5.4 Annual Risk Framework Review (Once per year, full day)

```
ANNUAL RISK FRAMEWORK REVIEW
==============================

[ ] 1. Full Year Performance Analysis
    - Annual return ($ and %)
    - Annual Sharpe, Sortino, Calmar
    - Comparison to benchmarks (buy-and-hold BTC, S&P 500)
    - Max drawdown and recovery time
    - Number of trading days vs flat days

[ ] 2. Risk Framework Effectiveness
    - How many times did risk limits prevent catastrophic losses?
    - Were there false positives (unnecessary halts)?
    - Were there false negatives (losses that should have been prevented)?
    - Is the three-lines-of-defense model working?

[ ] 3. Full Stress Test Suite
    - Replay ALL historical crisis scenarios with current model
    - Add any new crisis scenarios from the past year
    - Reverse stress test: "what would cause a 50% drawdown?"
    - Monte Carlo: 10,000 simulated paths for next year

[ ] 4. Infrastructure Audit
    - Security audit of all API keys, access controls, permissions
    - Review and update disaster recovery plan
    - Test full system recovery from backup
    - Review cloud costs and optimize

[ ] 5. Personal Assessment
    - Did I override any risk rules this year? Why?
    - What was my emotional state during the worst drawdown?
    - Am I psychologically suited to continue?
    - Do I need to adjust my risk tolerance?

STRATEGIC DECISIONS:
- Continue / modify / stop trading
- Capital allocation for next year
- Technology investments needed
- Skills to develop
```

---

## 6. Value at Risk (VaR) in Practice

### 6.1 What VaR Is and Isn't

**VaR Definition:** "The maximum loss that will not be exceeded with a given confidence level over a given time horizon."

Example: "Daily 95% VaR of $1,500" means: "There is a 95% probability that we will not lose more than $1,500 in a single day."

**What VaR does NOT tell you:** What happens in the other 5% of cases. If your VaR is $1,500, the tail loss could be $2,000, $5,000, or $50,000. VaR is silent about this.

### 6.2 Three Methods of Computing VaR

#### Method 1: Historical VaR

```python
import numpy as np

def historical_var(returns: np.ndarray, confidence: float = 0.95) -> float:
    """
    Historical VaR: use actual return distribution.
    No distributional assumptions required.

    Args:
        returns: array of historical daily returns (e.g., last 252 days)
        confidence: confidence level (0.95 = 95%)

    Returns:
        VaR as a positive number representing potential loss
    """
    # Sort returns and find the percentile
    sorted_returns = np.sort(returns)
    index = int((1 - confidence) * len(sorted_returns))
    var = -sorted_returns[index]  # Convert to positive loss
    return var


# Example for BTC with $50K portfolio:
# daily_returns = array of past 252 daily BTC returns
# var_95 = historical_var(daily_returns, 0.95) * 50_000
# If var_95 = $1,500, then 95% of days you won't lose more than $1,500
```

**Pros:** No distributional assumptions. Captures fat tails if they're in the data.
**Cons:** Only as good as the historical window. If you use calm-period data, VaR will underestimate crisis risk.

#### Method 2: Parametric (Variance-Covariance) VaR

```python
from scipy.stats import norm

def parametric_var(
    portfolio_value: float,
    daily_return_mean: float,
    daily_return_std: float,
    confidence: float = 0.95,
) -> float:
    """
    Parametric VaR: assumes normal distribution of returns.
    Fast to compute but underestimates tail risk for crypto.

    Formula: VaR = -(mu + z * sigma) * portfolio_value
    """
    z_score = norm.ppf(1 - confidence)  # e.g., -1.645 for 95%
    var = -(daily_return_mean + z_score * daily_return_std) * portfolio_value
    return var


# Example:
# BTC daily mean return: ~0.03%
# BTC daily std dev: ~3.5%
# For $50K portfolio:
# parametric_var(50_000, 0.0003, 0.035, 0.95) = ~$2,850
```

**Pros:** Simple, fast, closed-form.
**Cons:** Assumes normal distribution. BTC returns have fat tails (kurtosis ~5-10x normal). This method will UNDERESTIMATE tail risk for crypto by 30-50%.

#### Method 3: Monte Carlo VaR

```python
def monte_carlo_var(
    portfolio_value: float,
    daily_return_mean: float,
    daily_return_std: float,
    confidence: float = 0.95,
    n_simulations: int = 100_000,
    use_t_distribution: bool = True,
    df: int = 5,  # degrees of freedom for t-distribution
) -> float:
    """
    Monte Carlo VaR: simulate many possible paths.
    Can use fat-tailed distributions (Student's t).

    For crypto: ALWAYS use t-distribution (df=3-5) instead of normal.
    """
    if use_t_distribution:
        # Student's t captures crypto fat tails much better
        from scipy.stats import t as t_dist
        simulated_returns = t_dist.rvs(
            df=df, loc=daily_return_mean,
            scale=daily_return_std, size=n_simulations
        )
    else:
        simulated_returns = np.random.normal(
            daily_return_mean, daily_return_std, n_simulations
        )

    simulated_pnl = simulated_returns * portfolio_value
    sorted_pnl = np.sort(simulated_pnl)
    index = int((1 - confidence) * n_simulations)
    var = -sorted_pnl[index]
    return var


# Example with t-distribution (fat tails):
# monte_carlo_var(50_000, 0.0003, 0.035, 0.95, use_t_distribution=True, df=4)
# Result: ~$3,500-$4,000 (higher than parametric due to fat tails)
```

**Pros:** Flexible. Can model any distribution. Best for crypto (use Student's t with df=3-5).
**Cons:** Computationally expensive (though 100K sims takes <1 second on modern hardware).

### 6.3 VaR Limits in Practice

**How funds set VaR limits:**

```
Daily VaR limit = X% of NAV

Typical institutional limits:
  Conservative fund: 1% of NAV
  Moderate fund: 2% of NAV
  Aggressive fund: 3-5% of NAV

For ep2-crypto ($50K account):
  Recommended: 2% of NAV = $1,000 daily VaR (95%)
  This means: 95% of days, you won't lose more than $1,000

If current VaR exceeds limit:
  1. Reduce position size until VaR < limit
  2. Do NOT wait for the loss to materialize
  3. VaR breach = automatic position reduction
```

### 6.4 Expected Shortfall (CVaR) -- Beyond VaR

VaR's fatal flaw: it tells you the boundary of the tail but nothing about what's inside it. Expected Shortfall (ES), also called Conditional VaR (CVaR), answers: "When losses exceed VaR, how bad are they on average?"

```python
def expected_shortfall(returns: np.ndarray, confidence: float = 0.95) -> float:
    """
    Expected Shortfall (CVaR): average loss in the tail beyond VaR.
    This is the metric that matters for crypto.

    Basel III/FRTB replaced 99% VaR with 97.5% ES as the primary
    risk measure for banks -- because ES captures tail risk.
    """
    sorted_returns = np.sort(returns)
    cutoff = int((1 - confidence) * len(sorted_returns))
    tail_returns = sorted_returns[:cutoff]
    es = -np.mean(tail_returns)
    return es


# Example:
# If 95% VaR = 3.5% (loss), the ES might be 5.5%
# This means: on the worst 5% of days, the AVERAGE loss is 5.5%
# For $50K portfolio: ES = $2,750
# Interpretation: "When things go wrong, expect to lose ~$2,750 on average"
```

**Why CVaR matters more than VaR for crypto:**
- BTC has kurtosis of ~5-10 (normal = 3). The tails are FAT.
- A March 2020 COVID-day would be in the tail of any VaR model
- CVaR tells you the average severity when the tail event occurs
- Regulators (Basel III FRTB) replaced VaR with CVaR for exactly this reason

### 6.5 Stressed VaR

**Concept:** Run VaR calculations using data from crisis periods, not calm periods.

```python
def stressed_var(
    current_position: float,
    crisis_period_returns: np.ndarray,
    confidence: float = 0.99,
) -> float:
    """
    Stressed VaR: compute VaR using crisis-period data.

    Use data from:
    - March 2020 COVID crash
    - May 2021 China ban crash
    - May 2022 Terra/Luna crash
    - November 2022 FTX collapse
    - Any period with BTC daily moves > 10%
    """
    sorted_returns = np.sort(crisis_period_returns)
    index = int((1 - confidence) * len(sorted_returns))
    stressed_var = -sorted_returns[index] * current_position
    return stressed_var

# The stressed VaR will be 2-5x the normal VaR
# This is the number you should actually worry about
```

**Risk limit framework combining VaR and CVaR:**

```
RISK LIMIT HIERARCHY:
=====================
Level 1 (Green):  Daily VaR (95%) < 2% of NAV
Level 2 (Yellow): Daily VaR (95%) between 2-3% of NAV
                  -> Reduce position size by 50%
Level 3 (Red):    Daily VaR (95%) > 3% of NAV
                  -> Close to flat, wait for vol to decline
Level 4 (Black):  CVaR (95%) > 5% of NAV
                  -> Close ALL positions immediately

Additionally:
  Stressed VaR (99%) should not exceed 10% of NAV
  If it does, the system is over-leveraged for crisis conditions
```

---

## 7. Stress Testing at Institutional Level

### 7.1 What Scenarios Do Funds Test?

Institutional hedge funds are required to report stress test results for:
- Equity price shocks (-10%, -20%, -40%)
- Interest rate shocks (+100bp, +200bp, +500bp)
- Credit spread widening
- Volatility spikes (VIX doubling, tripling)
- Liquidity dry-ups
- FX shocks
- Combined scenarios

For crypto, the equivalent scenarios are:

```
CRYPTO STRESS SCENARIO CATALOG
================================

Historical Replays (Must-Have):
1. March 2020 COVID:        BTC -52% in 36 hours
2. May 2021 China Ban:      BTC -53% over 2 weeks
3. May 2022 Terra/Luna:     BTC -40% in 1 week, cascading liquidations
4. November 2022 FTX:       BTC -25% in 3 days
5. April 2021 Leverage Flush: BTC -27% in 1 day ($10B liquidations)

Synthetic Scenarios (Hypothetical):
6. Flash crash:             BTC -20% in 5 minutes (order book gap)
7. Exchange outage:         Primary exchange down for 2 hours mid-trade
8. Funding rate spike:      Funding goes to -0.5% per 8 hours
9. Liquidity crisis:        Order book depth drops 80%
10. Correlation breakdown:  All assets become 0.95 correlated
11. Extended drawdown:      BTC -30% over 60 days (slow grind)
12. Black swan:             BTC -70% in 1 week (worse than any historical)
```

### 7.2 Historical Replay Requirements

For each historical scenario, the stress test must:

1. Use actual tick-by-tick or 1-minute data from the crisis period
2. Include realistic order book depth and spread data
3. Apply actual funding rates from the period
4. Model exchange outages that occurred
5. Apply slippage models appropriate to crisis liquidity

```python
class StressTestRunner:
    """Run strategy through historical crisis data."""

    SCENARIOS = {
        "covid_crash": {
            "start": "2020-03-11",
            "end": "2020-03-15",
            "description": "BTC -52% in 36 hours",
            "expected_max_loss_pct": 15.0,  # Max acceptable loss
        },
        "terra_luna": {
            "start": "2022-05-07",
            "end": "2022-05-13",
            "description": "BTC -40%, cascading liquidations",
            "expected_max_loss_pct": 12.0,
        },
        "ftx_collapse": {
            "start": "2022-11-06",
            "end": "2022-11-11",
            "description": "BTC -25%, exchange counterparty risk",
            "expected_max_loss_pct": 10.0,
        },
    }

    def run_scenario(self, scenario_name: str) -> StressTestResult:
        scenario = self.SCENARIOS[scenario_name]
        data = self.load_crisis_data(scenario["start"], scenario["end"])

        # Run the full pipeline through crisis data
        result = self.backtest_engine.run(
            data=data,
            slippage_model="crisis",  # Wider spreads, less depth
            funding_model="actual",   # Use actual funding rates
        )

        # Pass/fail criteria
        passed = result.max_drawdown_pct <= scenario["expected_max_loss_pct"]

        return StressTestResult(
            scenario=scenario_name,
            max_drawdown_pct=result.max_drawdown_pct,
            max_loss_usd=result.max_loss_usd,
            recovery_time=result.recovery_time,
            passed=passed,
        )
```

### 7.3 Reverse Stress Testing

**The most important question: "What would destroy us?"**

Reverse stress testing starts from the outcome (e.g., "lose 50% of NAV") and works backward to find what scenario would cause it.

```python
def reverse_stress_test(
    portfolio_nav: float,
    target_loss_pct: float,
    current_position: dict,
) -> list[dict]:
    """
    Find scenarios that would cause target_loss_pct loss.

    Ask: "What combination of market moves would cause a 50% loss?"
    """
    scenarios = []

    # Scenario 1: What BTC move would cause this loss?
    btc_move_needed = -target_loss_pct / current_position["leverage"]
    scenarios.append({
        "type": "price_move",
        "btc_move_pct": btc_move_needed * 100,
        "description": f"BTC needs to move {btc_move_needed*100:.1f}% to cause {target_loss_pct*100:.0f}% portfolio loss",
        "plausible": abs(btc_move_needed) < 0.30,  # Has BTC moved 30% in a day?
    })

    # Scenario 2: Liquidity failure
    scenarios.append({
        "type": "liquidity_failure",
        "description": "Exchange halt during max position, forced liquidation at -20% slippage",
        "loss_if_occurs": current_position["notional"] * 0.20,
        "plausible": True,  # This HAS happened (BitMEX 2020)
    })

    # Scenario 3: Cascading failures
    scenarios.append({
        "type": "cascade",
        "description": "Position auto-liquidated during cascade, then market recovers",
        "loss_if_occurs": current_position["notional"] * current_position["leverage"] * 0.15,
        "plausible": True,  # Regular occurrence in crypto
    })

    return scenarios
```

### 7.4 Stress Testing Frequency

| Test Type | Frequency | Trigger |
|-----------|-----------|---------|
| Historical replay | Monthly | Always run all scenarios |
| Reverse stress test | Monthly | Or when position size changes significantly |
| Liquidity stress | Weekly | Or when market depth changes >30% |
| Synthetic scenarios | Quarterly | Or when new risk factors emerge |
| Full stress suite | After any model change | Mandatory before deploying new model |
| Ad-hoc stress | Immediately | When market conditions change rapidly |

---

## 8. Operational Risk Management

### 8.1 Key Person Risk

For a solo trader, YOU are the single point of failure.

**Scenarios:**
- You are sick/hospitalized for a week with open positions
- Your internet goes down during a trade
- Your laptop dies
- You are traveling and have limited connectivity

**Mitigations:**

```
KEY PERSON RISK MITIGATIONS
============================

1. Dead Man's Switch
   - If the system doesn't receive a "keep alive" from you in 24 hours:
     -> Close all positions to flat
     -> Send alert to emergency contact
   - Implementation: daily check-in via Telegram bot

2. Emergency Contact Protocol
   - Trusted person with READ-ONLY access to trading dashboard
   - Written instructions: "If I'm unreachable for 48 hours, call
     [exchange support] and close all positions"
   - They should NOT have trading access (prevent social engineering)

3. Automatic Position Management
   - All positions must have stop-losses AT THE EXCHANGE LEVEL
     (not just in your local system)
   - Use exchange-native conditional orders, not software stops
   - Maximum position duration: X hours (force-close if exceeded)

4. Mobile Backup
   - Phone with exchange app as backup
   - VPN configured on phone
   - Emergency-only: can close positions from phone
   - Do NOT use phone for regular trading (screen too small for analysis)
```

### 8.2 Infrastructure Redundancy

```
INFRASTRUCTURE REDUNDANCY PLAN
================================

1. Internet Connection
   Primary:   Home fiber
   Backup:    Mobile hotspot (different carrier)
   Emergency: Can access from phone over cellular

2. Hardware
   Primary:   Desktop/laptop
   Backup:    Second machine with identical setup
   Emergency: Phone with exchange app
   Note:      System should be deployable from scratch in <1 hour

3. Exchange Access
   Primary:   Binance Futures (or your main exchange)
   Backup:    Bybit (with identical strategy deployed)
   Rule:      Never have >70% of capital on one exchange
   Note:      Post-FTX, this is non-negotiable

4. Alert System
   Primary:   Telegram bot notifications
   Backup:    SMS alerts (different system)
   Emergency: Email (slowest, but most reliable)

5. Code and Configuration
   Primary:   Git repository (GitHub/GitLab)
   Backup:    Local copy + encrypted USB
   Rule:      All config is version-controlled
   Note:      Secrets in Doppler or env vars, NEVER in repo
```

### 8.3 API Key Management

```
API KEY SECURITY PROTOCOL
==========================

1. Key Permissions (Principle of Least Privilege)
   - Trading keys: enable trading ONLY, disable withdrawals
   - Read-only keys: for monitoring dashboards
   - Withdrawal keys: DO NOT CREATE unless absolutely necessary
   - IP whitelist: restrict to your server IP(s)

2. Key Rotation Schedule
   - Trading keys: rotate every 60 days
   - Read-only keys: rotate every 90 days
   - After any suspected compromise: rotate IMMEDIATELY

3. Key Storage
   - NEVER store API keys in source code
   - NEVER store API keys in .env files in repos
   - Use Doppler or encrypted environment variables
   - Keys encrypted at rest using AES-256 or equivalent

4. Key Rotation Process (Zero-Downtime)
   a. Generate new key on exchange
   b. Update key in Doppler/secrets manager
   c. Deploy to system (new key active)
   d. Verify system works with new key
   e. Revoke old key on exchange
   f. Log rotation event with timestamp

5. Emergency Key Revocation
   - If ANY suspicion of compromise:
     1. Revoke ALL keys immediately on exchange
     2. Check recent trade/withdrawal history
     3. Enable 2FA re-verification
     4. Generate new keys with fresh IP whitelist
     5. Review system logs for unauthorized access
```

### 8.4 Two-Factor Authentication and Withdrawal Whitelisting

```
SECURITY HARDENING CHECKLIST
==============================

[ ] 2FA enabled on ALL exchange accounts (hardware key preferred, TOTP acceptable)
[ ] 2FA backup codes stored securely offline (not on the trading machine)
[ ] Withdrawal addresses whitelisted (only YOUR wallets)
[ ] New withdrawal addresses require 24-48 hour waiting period
[ ] Anti-phishing code set on exchange
[ ] Email notifications for ALL logins and trades
[ ] Separate email address for exchange accounts (not personal email)
[ ] Password manager for exchange credentials
[ ] No exchange passwords saved in browser
[ ] API keys restricted to IP whitelist
```

---

## 9. Liquidity Risk Management

### 9.1 Can You Exit Your Position?

The fundamental question: **if you need to close your entire position RIGHT NOW, how much slippage will you suffer?**

For BTC perpetual futures on Binance:
- Normal conditions: top-10 bids hold ~$5-20M depth. A $50K position exits with ~0.01% slippage.
- Stressed conditions (March 2020): depth dropped ~80%. A $50K position might see 0.1-0.5% slippage.
- Cascade conditions: depth can temporarily disappear. Slippage can be 1-5% or worse.

### 9.2 Position Size as a Function of Liquidity

```python
def max_position_from_liquidity(
    order_book_depth_usd: float,
    max_slippage_bps: float = 10,  # 0.1% max acceptable slippage
    depth_levels: int = 10,         # Use top-10 levels
) -> float:
    """
    Calculate maximum position size based on available liquidity.

    Rule of thumb: your order should be no more than 10% of
    visible depth at your acceptable slippage level.

    Why 10%? Because:
    1. Other traders are also competing for liquidity
    2. Order book can change between your decision and execution
    3. Iceberg orders mean visible depth != total depth (helps you)
    4. During stress, depth evaporates (hurts you badly)
    """
    # Conservative: position should be <10% of visible depth
    max_position = order_book_depth_usd * 0.10

    return max_position


# Example:
# Normal BTC perp depth (top-10 bids): $10M
# max_position = $10M * 0.10 = $1M
# For a $50K account with 5x leverage = $250K notional
# $250K << $1M -> OK in normal conditions

# Stressed BTC perp depth (top-10 bids): $2M (80% reduction)
# max_position = $2M * 0.10 = $200K
# $250K > $200K -> REDUCE POSITION or risk excessive slippage
```

### 9.3 Liquidity Stress Scenarios

```python
class LiquidityStressTest:
    """Test position exitability under various conditions."""

    def test_normal_exit(self, position_size: float) -> ExitAnalysis:
        """Can we exit cleanly in normal conditions?"""
        current_depth = self.get_order_book_depth()
        slippage = self.estimate_slippage(position_size, current_depth)
        return ExitAnalysis(
            slippage_bps=slippage,
            exit_time_seconds=position_size / current_depth["fill_rate"],
            feasible=slippage < 10,  # <0.1% slippage
        )

    def test_stressed_exit(self, position_size: float) -> ExitAnalysis:
        """Can we exit when depth drops 80%?"""
        stressed_depth = self.get_order_book_depth() * 0.20
        slippage = self.estimate_slippage(position_size, stressed_depth)
        return ExitAnalysis(
            slippage_bps=slippage,
            exit_time_seconds=position_size / stressed_depth["fill_rate"],
            feasible=slippage < 50,  # <0.5% slippage
        )

    def test_cascade_exit(self, position_size: float) -> ExitAnalysis:
        """Can we exit during a liquidation cascade?"""
        # During cascades, effective depth can drop 95%
        cascade_depth = self.get_order_book_depth() * 0.05
        slippage = self.estimate_slippage(position_size, cascade_depth)
        return ExitAnalysis(
            slippage_bps=slippage,
            exit_time_seconds=position_size / cascade_depth["fill_rate"],
            feasible=slippage < 200,  # <2% slippage
        )
```

### 9.4 Liquidity-Adjusted Position Sizing

```python
def liquidity_adjusted_position_size(
    kelly_position: float,        # Position size from Kelly criterion
    current_depth_usd: float,     # Current order book depth
    normal_depth_usd: float,      # 30-day average depth
    max_depth_usage_pct: float = 0.10,  # Max 10% of visible depth
) -> float:
    """
    Adjust Kelly-optimal position size for liquidity constraints.

    If liquidity is low, reduce position regardless of what
    Kelly says -- because Kelly doesn't account for slippage.
    """
    # Liquidity cap: never use more than 10% of visible depth
    liquidity_cap = current_depth_usd * max_depth_usage_pct

    # Liquidity ratio: is current depth normal?
    depth_ratio = current_depth_usd / normal_depth_usd

    # If depth is below 50% of normal, reduce position proportionally
    if depth_ratio < 0.50:
        liquidity_adjusted = kelly_position * depth_ratio
    else:
        liquidity_adjusted = kelly_position

    # Final position = minimum of Kelly, liquidity cap, and adjusted size
    final_position = min(kelly_position, liquidity_cap, liquidity_adjusted)

    return final_position
```

---

## 10. Risk Culture and Psychology

### 10.1 Why Traders Override Their Own Risk Rules

Every trader who blows up says the same thing: "I knew my rules, but I didn't follow them."

**The psychology of override:**

1. **"This time is different"** -- The market is doing something unusual, so the usual rules don't apply. (They always apply.)

2. **"I'll just widen the stop a little"** -- Small compromise that opens the door to larger compromises. $100 becomes $500 becomes $2,000.

3. **"I need to make back my losses"** -- Loss aversion and the disposition effect. Losers hold losers too long and cut winners too short.

4. **"My model says this is a great trade"** -- Overconfidence in the model. The model was trained on historical data; the market may have changed.

5. **"The risk engine is being too conservative"** -- The risk engine is doing its job. If you disagree with it repeatedly, the problem is your risk parameters, not the engine. Change them at the quarterly review, not in the heat of battle.

### 10.2 Key Behavioral Biases

| Bias | Definition | How It Kills Traders | Defense |
|------|-----------|---------------------|---------|
| **Loss Aversion** | Losses feel 2x worse than equivalent gains feel good | Hold losers too long, cut winners too short | Mechanical stop-losses, no manual override |
| **Disposition Effect** | Tendency to sell winners and hold losers | Exactly backward from optimal | Let the system manage exits |
| **Overconfidence** | Believing you're better than you are | Over-sizing positions after a win streak | Kelly criterion caps position size mathematically |
| **Recency Bias** | Overweighting recent events | Chasing momentum after it's extended | Use longer lookback periods for risk params |
| **Anchoring** | Fixating on a reference price | "BTC was $70K, it'll come back" | Trade the current price, not your entry price |
| **Sunk Cost Fallacy** | "I've already lost $5K, I can't close now" | Letting small losses become catastrophic | Hard stop-losses at exchange level |
| **Gambler's Fallacy** | "I've lost 5 in a row, I'm due for a win" | Increasing size after losses | Fixed position sizing regardless of recent results |
| **Confirmation Bias** | Seeking info that supports your position | Ignoring bearish signals while long | Let the model decide, not your narrative |

### 10.3 Designing Systems That Resist Human Override

**Principle 1: Physical Separation**
The risk engine runs as a separate process. The trading engine literally cannot reach the exchange without going through it. There is no "backdoor."

**Principle 2: Immutable Limits**
Hard risk limits are frozen at startup. Changing them requires stopping the entire system, editing a config file, and restarting. This adds friction to impulsive changes.

**Principle 3: Cooling-Off Periods**
After a risk limit is hit, enforce a mandatory waiting period before trading resumes:
- Daily loss limit hit: no trading until next day
- Drawdown limit hit: no trading for 48 hours minimum
- Hard drawdown hit: no trading for 1 week; requires manual review

**Principle 4: Audit Trail**
Every risk override, parameter change, and manual intervention is logged with timestamp and reason. Review these logs during the weekly review.

**Principle 5: Pre-Commitment Contracts**

```python
# Example pre-commitment: written BEFORE trading starts
class TradingPreCommitment:
    """
    Written during a calm, rational state.
    Enforced during emotional, irrational states.
    """

    RULES = {
        "max_loss_before_stopping": 0.03,  # 3% daily
        "max_consecutive_losses_before_review": 5,
        "max_position_after_3_losses": 0.5,  # Half normal size
        "mandatory_break_after_hard_stop": "48_hours",
        "allowed_to_override_risk_engine": False,  # NEVER
        "allowed_to_change_stops_intraday": False,  # NEVER
        "allowed_to_add_to_losing_position": False,  # NEVER
    }
```

### 10.4 The Pre-Mortem Technique

Before entering any new strategy or significant position change, conduct a pre-mortem:

> "It's 6 months from now. This strategy has failed catastrophically. What went wrong?"

Force yourself to write down 5 plausible failure modes. For each, write the mitigation. If you can't mitigate all 5, don't proceed.

### 10.5 Decision Journal

Maintain a decision journal separate from trading logs:

```
DECISION JOURNAL ENTRY
=======================
Date: 2026-03-23
Decision: Increase max position size from 15% to 20% of NAV
Reasoning: Win rate has improved from 52% to 55% over last 90 days
Expected outcome: ~15% higher returns with manageable additional risk
Pre-mortem risks:
  1. Win rate improvement may be noise, not signal
  2. Larger positions increase slippage costs
  3. Larger drawdowns may trigger emotional override
  4. If volatility spikes, 20% positions create larger VaR
Mitigations:
  1. Will review at next monthly audit; revert if win rate drops below 53%
  2. Monitor slippage metrics weekly
  3. Hard drawdown limit remains at 10% regardless
  4. VaR limit still enforced -- position will auto-reduce if vol spikes

Review date: 2026-04-23 (one month)
```

---

## Summary: The Complete Risk Framework for ep2-crypto

### Architecture

```
[Operational Watchdog - Third Line]
  |-- Dead man's switch / heartbeat monitor
  |-- Exchange connectivity check
  |-- System resource monitor
  |-- Can KILL everything
  |
  v
[Risk Engine - Second Line]
  |-- VaR computation (Monte Carlo, t-distribution)
  |-- CVaR / Expected Shortfall
  |-- Drawdown tracking
  |-- Position limit enforcement
  |-- Exposure limit enforcement
  |-- All orders pass through here
  |
  v
[Trading Engine - First Line]
  |-- Signal generation
  |-- Kelly-based position sizing
  |-- Stop-loss calculation
  |-- Order proposal (NOT execution)
  |
  Exchange (only reachable via Risk Engine)
```

### Hard Limits for $50K Account

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Max position | 20% of NAV ($10K) | Limit concentration |
| Max gross exposure | 40% of NAV ($20K) | Account for leverage |
| Daily VaR limit (95%) | 2% of NAV ($1,000) | Industry standard |
| CVaR limit (95%) | 4% of NAV ($2,000) | Tail risk protection |
| Soft drawdown | 5% ($2,500) | Reduce size by 50% |
| Hard drawdown | 10% ($5,000) | Halt trading 48 hours |
| Forced close drawdown | 15% ($7,500) | Close everything |
| Max daily loss | 3% ($1,500) | Halt for day |
| Max orders per hour | 20 | Prevent runaway |
| Position sizing | Quarter-Kelly | Conservative for crypto |
| Max leverage | 3x | Even if exchange allows more |

### Review Schedule

| Frequency | Focus | Time |
|-----------|-------|------|
| Daily | Check alerts, open positions, P&L | 5 min |
| Weekly | Full metric review, risk compliance | 30 min |
| Monthly | Deep audit, model health, stress tests | 2 hours |
| Quarterly | Parameter review, edge assessment | Half day |
| Annually | Framework review, strategic assessment | Full day |

### The Three Rules That Would Have Prevented Every Blowup in This Document

1. **Risk controls are immutable during trading.** Alameda exempted itself from liquidation. 3AC had no drawdown triggers. If you can override your risk rules while trading, you don't have risk rules.

2. **The risk engine wraps the trading engine, not the other way around.** Every order must pass through an independent risk check. The trading logic has no direct access to the exchange.

3. **Stress test for destruction, not comfort.** The question is not "will I be profitable?" but "what would destroy me?" If the answer is plausible (and in crypto, it always is), your position is too large.

---

## Sources

- [Renaissance Technologies - Wikipedia](https://en.wikipedia.org/wiki/Renaissance_Technologies)
- [Medallion Fund: The Curious Case - School of Hedge](https://www.schoolofhedge.com/pages/the-curious-case-of-medallion-fund)
- [Jim Simons Trading Strategy - QuantVPS](https://www.quantvps.com/blog/jim-simons-trading-strategy)
- [Risk and Reward: Leverage in Medallion Fund - Quantified Strategies](https://www.quantifiedstrategies.com/risk-and-reward-how-leverage-amplified-the-medallion-funds-gains/)
- [Renaissance Technologies Breakdown - Daniel Scrivner](https://www.danielscrivner.com/renaissance-technologies-business-breakdown/)
- [How Medallion Sustained 66% p.a. - Medium](https://t1mproject.medium.com/how-the-medallion-fund-sustained-66-p-a-for-30-years-and-generated-100-billion-f2a254c43eb7)
- [Inside Two Sigma - Institutional Investor](https://www.institutionalinvestor.com/article/2bsw4ehe37jv5y886qtxc/corner-office/inside-the-geeky-quirky-and-wildly-successful-world-of-quant-shop-two-sigma)
- [How Millennium, Citadel, Point72 Structure Pods - Navnoor Bawa](https://navnoorbawa.substack.com/p/how-millennium-citadel-and-point72)
- [Citadel Multi-Strategy Machine - HedgeCo](https://www.hedgeco.net/news/03/2026/citadels-multi-strategy-machine-is-winning-in-a-fractured-market.html)
- [Anatomy of a Run: Terra Luna Crash - Harvard Law](https://corpgov.law.harvard.edu/2023/05/22/anatomy-of-a-run-the-terra-luna-crash/)
- [Terra Luna Crash - NBER](https://www.nber.org/papers/w31160)
- [Jump Trading's Role in Terra - Coin360](https://coin360.com/news/jump-trading-terraform-labs-crypto-manipulation)
- [What Happened to 3AC - ZenLedger](https://zenledger.io/blog/three-arrows-capital-3ac-what-happened/)
- [3AC Collapse Lessons - QuantStrategy.io](https://quantstrategy.io/blog/the-billion-dollar-collapse-lessons-on-leverage-and-risk/)
- [How 3AC Dragged Down Crypto - CNBC](https://www.cnbc.com/2022/07/11/how-the-fall-of-three-arrows-or-3ac-dragged-down-crypto-investors.html)
- [Wintermute Hack - CoinDesk](https://www.coindesk.com/business/2022/09/20/crypto-market-maker-wintermute-hacked-for-160m-says-ceo)
- [Wintermute Hack Analysis - PostQuantum](https://postquantum.com/crypto-security/wintermute-hack/)
- [FTX Collapse Complete Guide - CoinLedger](https://coinledger.io/learn/the-ftx-collapse)
- [Alameda Gap and Crypto Liquidity - Cointelegraph](https://cointelegraph.com/explained/the-alameda-gap-and-crypto-liquidity-crisis-explained)
- [Three Lines of Defense - IIA](https://www.theiia.org/globalassets/documents/resources/the-iias-three-lines-model-an-update-of-the-three-lines-of-defense-july-2020/three-lines-model-updated-english.pdf)
- [Best Practices for Automated Trading Risk Controls - FIA](https://www.fia.org/sites/default/files/2024-07/FIA_WP_AUTOMATED%20TRADING%20RISK%20CONTROLS_FINAL_0.pdf)
- [Coinbase STARK Risk System](https://www.coinbase.com/blog/building-an-in-house-risk-management-system-for-futures-trading)
- [ESMA Algorithmic Trading Supervisory Briefing](https://www.esma.europa.eu/sites/default/files/2026-02/ESMA74-1505669079-10311_Supervisory_Briefing_on_Algorithmic_Trading_in_the_EU.pdf)
- [VaR Methods - Corporate Finance Institute](https://corporatefinanceinstitute.com/resources/career-map/sell-side/risk-management/value-at-risk-var/)
- [Expected Shortfall (CVaR) - Ryan O'Connell CFA](https://ryanoconnellfinance.com/expected-shortfall-cvar/)
- [Kelly Criterion for Crypto - CoinMarketCap](https://coinmarketcap.com/academy/article/what-is-the-kelly-bet-size-criterion-and-how-to-use-it-in-crypto-trading)
- [Kelly Criterion in Trading - Medium/Huma](https://medium.com/@humacapital/the-kelly-criterion-in-trading-05b9a095ca26)
- [Stress Testing for Hedge Funds - Arootah](https://arootah.com/blog/hedge-fund-and-family-office/how-funds-can-prepare-for-market-shocks)
- [Hedge Fund Stress Test Reporting - OFR](https://www.financialresearch.gov/hedge-fund-monitor/categories/risk-management/chart-56/)
- [Reverse Stress Testing - Rixtrema](https://www.rixtrema.com/pdf/RobustRiskEstimation.pdf)
- [API Key Security in Crypto Trading - Darkbot](https://darkbot.io/blog/what-is-api-key-security-in-automated-crypto-trading)
- [API Key Management Best Practices - MultitaskAI](https://multitaskai.com/blog/api-key-management-best-practices/)
- [Behavioral Biases in Trading - Quantified Strategies](https://www.quantifiedstrategies.com/trading-bias/)
- [Trading Psychology - Britannica Money](https://www.britannica.com/money/behavioral-biases-in-finance)
