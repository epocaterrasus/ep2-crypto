# Stress Testing Framework for ep2-crypto

> Comprehensive scenario catalog, setup code, pass/fail criteria, and CI/CD automation for validating system robustness under extreme conditions.

---

## Table of Contents

1. [Historical Stress Scenarios](#1-historical-stress-scenarios)
2. [Synthetic Stress Scenarios](#2-synthetic-stress-scenarios)
3. [Data Failure Scenarios](#3-data-failure-scenarios)
4. [Execution Failure Scenarios](#4-execution-failure-scenarios)
5. [Model Failure Scenarios](#5-model-failure-scenarios)
6. [Stress Test Metrics](#6-stress-test-metrics)
7. [Stress Test Automation & CI/CD](#7-stress-test-automation--cicd)

---

## 1. Historical Stress Scenarios

Each scenario replays real market data through the full pipeline (features -> regime -> model -> gating -> execution). The goal is to verify the system does not blow up during known extreme events. Historical data is fetched via `scripts/collect_history.py` and stored in the backtest database.

### 1.1 March 2020 COVID Crash (BTC -50% in 2 days)

**What happened (March 12-13, 2020 -- "Black Thursday"):**
- BTC fell from ~$7,900 to ~$3,800 in roughly 36 hours (-52%).
- BitMEX alone liquidated >$1.6B in 24 hours. Aggregate cross-exchange liquidations exceeded $4B.
- Binance order book depth collapsed: top-10 bid levels lost ~80% of volume within minutes.
- Bid-ask spreads on BTCUSDT perps widened from ~0.01% to >0.5% at the worst point.
- Funding rate went deeply negative (-0.375% on Bitmex in a single 8h period) as shorts dominated.
- Volume spiked 5-10x normal -- Binance spot BTC/USDT hit 400K BTC daily volume vs ~50K normal.
- Multiple exchanges experienced API outages (BitMEX went down for ~25 minutes, which actually halted the cascade).
- The crash was correlated across all asset classes: S&P 500 -9.5% on March 12, NQ followed.

**Setup code:**

```python
# tests/stress/test_historical_covid_crash.py
import pytest
from datetime import datetime, timezone
from ep2_crypto.backtest.walk_forward import WalkForwardEngine
from ep2_crypto.backtest.simulator import ExecutionSimulator
from ep2_crypto.backtest.metrics import compute_all_metrics
from ep2_crypto.stress.scenario_loader import load_historical_scenario

class TestCovidCrash:
    """Replay March 12-13, 2020 through the full system."""

    @pytest.fixture
    def scenario_data(self):
        return load_historical_scenario(
            start=datetime(2020, 3, 10, tzinfo=timezone.utc),
            end=datetime(2020, 3, 15, tzinfo=timezone.utc),
            symbol="BTC/USDT:USDT",
            include_orderbook=True,
            include_liquidations=True,
            include_funding=True,
            # Inject realistic conditions
            spread_multiplier=10.0,    # spreads widened ~10x
            depth_multiplier=0.2,      # book depth dropped to 20%
            slippage_multiplier=5.0,   # 5x normal slippage
        )

    def test_max_drawdown_bounded(self, scenario_data):
        """System must not exceed 15% drawdown (our max drawdown halt)."""
        result = self._run_backtest(scenario_data)
        assert result.max_drawdown_pct < 15.0, (
            f"Max drawdown {result.max_drawdown_pct:.1f}% exceeded 15% limit"
        )

    def test_kill_switches_activate(self, scenario_data):
        """Daily loss limit (3%) and drawdown gate must trigger."""
        result = self._run_backtest(scenario_data)
        assert result.kill_switch_activations > 0, (
            "Kill switches should have activated during COVID crash"
        )
        assert result.daily_loss_limit_triggered is True
        assert result.drawdown_gate_triggered is True

    def test_position_sizing_reduces(self, scenario_data):
        """Drawdown gate should progressively reduce position size."""
        result = self._run_backtest(scenario_data)
        # After 3% drawdown, position size should start reducing
        sizes = result.position_sizes_during_drawdown
        assert sizes[-1] < sizes[0] * 0.5, (
            "Position sizing did not reduce by at least 50% during drawdown"
        )

    def test_no_trades_during_spread_extreme(self, scenario_data):
        """System should abstain when spread > 0.1% (10x normal)."""
        result = self._run_backtest(scenario_data)
        extreme_spread_trades = [
            t for t in result.trades
            if t.entry_spread_pct > 0.1
        ]
        assert len(extreme_spread_trades) == 0, (
            f"{len(extreme_spread_trades)} trades executed during extreme spreads"
        )

    def test_pnl_vs_buy_and_hold(self, scenario_data):
        """System PnL must beat buy-and-hold (which lost -52%)."""
        result = self._run_backtest(scenario_data)
        bnh_return = scenario_data.buy_and_hold_return  # ~-52%
        assert result.total_return_pct > bnh_return, (
            f"System return {result.total_return_pct:.1f}% worse than "
            f"buy-and-hold {bnh_return:.1f}%"
        )

    def _run_backtest(self, scenario_data):
        engine = WalkForwardEngine(
            train_size=4032, test_size=288, step_size=288, gap=1
        )
        simulator = ExecutionSimulator(
            slippage_model="sqrt_impact",
            maker_fee_bps=2.0,
            taker_fee_bps=4.0,
            latency_ms=200,
        )
        return engine.run(scenario_data, simulator)
```

**Pass/fail criteria:**
| Criterion | Pass | Fail |
|-----------|------|------|
| Max drawdown | < 15% | >= 15% |
| Kill switch activation | Triggered within first 5% drawdown | Never triggered |
| Position sizing | Reduced by >= 50% during drawdown | No reduction |
| Trades during extreme spread | 0 | Any |
| System return vs buy-and-hold | Better than -52% | Worse |

**Hardening if test fails:**
- If drawdown exceeds 15%: Tighten daily loss limit from 3% to 2%. Add volatility-adaptive position sizing that cuts to zero when 5-min realized vol exceeds 200% annualized.
- If kill switch does not activate: Review kill switch thresholds in `config.py`. Ensure the drawdown gate monitors mark-to-market PnL, not just realized PnL.
- If trades happen during extreme spreads: Add a spread filter to the confidence gating pipeline: `if current_spread > 5 * rolling_median_spread: abstain`.

---

### 1.2 May 2021 China Mining Ban Crash (-30% in hours)

**What happened (May 19, 2021 -- "Black Wednesday"):**
- BTC dropped from ~$43K to ~$30K in roughly 8 hours (-30%).
- Triggered by China announcing a ban on crypto mining + financial institutions barred from crypto services.
- Aggregate liquidations exceeded $8B in 24 hours -- the largest single-day liquidation event in crypto history.
- Funding rate had been elevated at +0.05% to +0.10% for days prior (excessive long positioning).
- Open interest was at all-time highs (~$27B across exchanges).
- The cascade exhibited classic self-exciting dynamics: each wave of liquidations triggered the next.
- Order book depth on Binance dropped by ~70% within the first hour.
- NQ was relatively flat -- this was a crypto-specific event, meaning cross-market signals would have been misleading.

**Setup code:**

```python
class TestChinaBanCrash:
    """Replay May 19, 2021 through the full system."""

    @pytest.fixture
    def scenario_data(self):
        return load_historical_scenario(
            start=datetime(2021, 5, 17, tzinfo=timezone.utc),
            end=datetime(2021, 5, 22, tzinfo=timezone.utc),
            symbol="BTC/USDT:USDT",
            include_orderbook=True,
            include_liquidations=True,
            include_funding=True,
            spread_multiplier=8.0,
            depth_multiplier=0.3,
            slippage_multiplier=4.0,
            # Inject pre-crash conditions
            pre_crash_funding_rate=0.08,  # elevated
            pre_crash_oi_percentile=95,   # near ATH
        )

    def test_cascade_detector_fires(self, scenario_data):
        """Liquidation cascade detector must identify the cascade."""
        result = self._run_backtest(scenario_data)
        assert result.cascade_alerts > 0, (
            "Cascade detector did not fire during largest liquidation event ever"
        )
        # Must fire within first 30 minutes of the crash starting
        first_alert = result.cascade_alert_timestamps[0]
        crash_start = datetime(2021, 5, 19, 7, 0, tzinfo=timezone.utc)
        assert (first_alert - crash_start).total_seconds() < 1800

    def test_funding_rate_warning(self, scenario_data):
        """System should flag elevated funding as risk before crash."""
        result = self._run_backtest(scenario_data)
        # Funding z-score > 2 should trigger risk reduction
        assert result.pre_crash_risk_flags.get("funding_extreme") is True

    def test_cross_market_decorrelation_handled(self, scenario_data):
        """NQ was flat during crypto crash -- system must not assume
        correlation holds and trade based on NQ signal."""
        result = self._run_backtest(scenario_data)
        nq_based_trades = [
            t for t in result.trades
            if t.signal_source == "macro_event"
            and t.timestamp >= datetime(2021, 5, 19, tzinfo=timezone.utc)
        ]
        assert len(nq_based_trades) == 0, (
            "System traded on NQ signal during crypto-specific crash"
        )

    def test_max_drawdown_bounded(self, scenario_data):
        result = self._run_backtest(scenario_data)
        assert result.max_drawdown_pct < 15.0
```

**Pass/fail criteria:**
| Criterion | Pass | Fail |
|-----------|------|------|
| Cascade detector fires | Within 30 min of crash start | Missed entirely |
| Funding rate warning | Flagged before crash | Not flagged |
| Cross-market false signal | No NQ-based trades during crash | NQ trades taken |
| Max drawdown | < 15% | >= 15% |

**Hardening if test fails:**
- If cascade detector misses: Lower the Hawkes intensity threshold from `5 * mu` to `3 * mu`. Add a secondary detector based on pure price velocity (>0.5% per 5-min bar sustained for 3+ bars).
- If funding warning is missed: Add a pre-trade check: if funding_rate_zscore > 2.0 AND oi_percentile > 80, reduce max position size by 50%.

---

### 1.3 November 2022 FTX Collapse (-25% over 5 days)

**What happened (November 6-11, 2022):**
- BTC fell from ~$21K to ~$15.5K over 5 days (-25%).
- Unlike typical cascades, this was a slow-motion confidence crisis, not a leverage unwind.
- The crash was driven by exchange insolvency news (CoinDesk article -> Binance selling FTT -> FTX withdrawal halt -> bankruptcy).
- Funding rate went deeply negative as shorts piled in (unusual -- typically cascades have positive funding before long liquidations).
- Volume patterns were abnormal: massive spot selling (not just perp liquidations) as users withdrew from exchanges.
- Cross-exchange basis diverged: BTC traded at different prices on different exchanges as confidence in FTX collapsed.
- On-chain: massive exchange outflows from all exchanges (not just FTX) as users moved to self-custody.
- The regime was unique: it was not trending, not mean-reverting, not volatile in the typical sense. It was a structural break.

**Setup code:**

```python
class TestFTXCollapse:
    """Replay November 6-11, 2022 through the full system."""

    @pytest.fixture
    def scenario_data(self):
        return load_historical_scenario(
            start=datetime(2022, 11, 4, tzinfo=timezone.utc),
            end=datetime(2022, 11, 13, tzinfo=timezone.utc),
            symbol="BTC/USDT:USDT",
            include_orderbook=True,
            include_liquidations=True,
            include_funding=True,
            spread_multiplier=3.0,
            depth_multiplier=0.5,
            slippage_multiplier=2.5,
            # FTX-specific: negative funding, exchange-driven selling
            pre_crash_funding_rate=-0.03,
        )

    def test_regime_detector_flags_anomaly(self, scenario_data):
        """Regime detector should flag this as an unseen regime or
        high transition probability (instability)."""
        result = self._run_backtest(scenario_data)
        # BOCPD change point detection should fire
        assert result.regime_change_points > 0
        # Regime uncertainty should be elevated
        max_regime_entropy = max(result.regime_entropy_series)
        assert max_regime_entropy > 0.8, (
            f"Regime entropy peaked at {max_regime_entropy:.2f}, "
            "expected > 0.8 during structural break"
        )

    def test_slow_bleed_detection(self, scenario_data):
        """System must handle slow multi-day drawdown, not just flash crashes."""
        result = self._run_backtest(scenario_data)
        # Weekly loss limit (5%) should trigger
        assert result.weekly_loss_limit_triggered is True

    def test_no_buy_the_dip_during_crisis(self, scenario_data):
        """Model should not aggressively go long during a confidence crisis."""
        result = self._run_backtest(scenario_data)
        long_trades_during_crisis = [
            t for t in result.trades
            if t.direction == "long"
            and t.timestamp >= datetime(2022, 11, 8, tzinfo=timezone.utc)
            and t.timestamp <= datetime(2022, 11, 11, tzinfo=timezone.utc)
        ]
        # Some long trades are acceptable, but net direction should be flat/short
        assert len(long_trades_during_crisis) < 5, (
            f"{len(long_trades_during_crisis)} long trades during FTX crisis"
        )
```

**Pass/fail criteria:**
| Criterion | Pass | Fail |
|-----------|------|------|
| Regime anomaly detected | BOCPD fires, entropy > 0.8 | No change point detected |
| Weekly loss limit | Triggered | Not triggered after 5 days of losses |
| Long trades during crisis | < 5 | >= 5 (model buying the dip blindly) |
| Max drawdown | < 15% | >= 15% |

**Hardening if test fails:**
- If regime detector misses the structural break: Add a "correlation breakdown" feature -- when the rolling BTC-NQ correlation drops by >0.3 in 24 hours, flag as regime anomaly.
- If model keeps going long: Add an exchange health check to the confidence gating pipeline. If exchange withdrawal suspensions are detected (monitor via API status endpoints), reduce confidence by 50%.

---

### 1.4 March 2023 SVB Banking Crisis (BTC +20% on USDC depeg)

**What happened (March 10-13, 2023):**
- Silicon Valley Bank collapsed on March 10. USDC (backed partly by SVB deposits) depegged to ~$0.87.
- BTC/USD initially dropped ~5% on contagion fear, then rallied ~20% over the next 3 days as crypto was viewed as an alternative to the banking system.
- USDC-quoted pairs showed artificial BTC price inflation (BTC/USDC spiked because USDC was worth less, not because BTC was worth more in real terms).
- Funding rates spiked positive as longs piled in during the "digital gold" narrative.
- Cross-market: S&P 500/NQ dropped on banking fears, but BTC moved in the OPPOSITE direction -- a rare and significant decorrelation event.
- This was a regime where the BTC-NQ correlation flipped from positive to negative overnight.

**Setup code:**

```python
class TestSVBCrisis:
    """Replay March 10-13, 2023 through the full system."""

    @pytest.fixture
    def scenario_data(self):
        return load_historical_scenario(
            start=datetime(2023, 3, 9, tzinfo=timezone.utc),
            end=datetime(2023, 3, 15, tzinfo=timezone.utc),
            symbol="BTC/USDT:USDT",
            include_orderbook=True,
            include_funding=True,
            # USDC depeg creates data anomalies
            stablecoin_depeg=True,
            spread_multiplier=4.0,
            depth_multiplier=0.4,
        )

    def test_stablecoin_depeg_handling(self, scenario_data):
        """System must not confuse USDC depeg with real BTC price movement.
        If using USDT-quoted pairs, impact should be minimal, but cross-exchange
        basis divergence should trigger a warning."""
        result = self._run_backtest(scenario_data)
        assert result.data_quality_warnings.get("basis_divergence") is True

    def test_decorrelation_detected(self, scenario_data):
        """BTC-NQ correlation flipped from ~0.6 to negative.
        System must detect and adapt."""
        result = self._run_backtest(scenario_data)
        # NQ lead-lag feature should be down-weighted or disabled
        assert result.nq_signal_weight_during_crisis < 0.3, (
            "NQ signal weight should decrease during decorrelation"
        )

    def test_captures_upside(self, scenario_data):
        """This was a +20% rally. System should capture some of it,
        or at minimum not be short."""
        result = self._run_backtest(scenario_data)
        assert result.total_return_pct > -2.0, (
            f"System lost {result.total_return_pct:.1f}% during a +20% rally"
        )
```

**Pass/fail criteria:**
| Criterion | Pass | Fail |
|-----------|------|------|
| Stablecoin depeg warning | Flagged | Not detected |
| BTC-NQ decorrelation | Detected, NQ weight reduced | NQ signal still weighted > 0.3 |
| System return | > -2% (doesn't lose during a rally) | < -2% |

---

### 1.5 March 2024 New ATH Euphoria then -15% Correction

**What happened (March 5-20, 2024):**
- BTC hit a new all-time high near $73,800 on March 14 (driven by spot ETF inflows).
- Funding rates reached extreme levels: 0.10-0.20% on multiple exchanges (annualized 109-219%).
- Open interest hit all-time highs.
- Then BTC corrected -15% to ~$62K over the next 6 days.
- Liquidations exceeded $1.5B during the correction.
- The correction was a textbook "funding rate reversion" event -- excessively long positioning unwound.
- Volume during the ATH was 3-5x normal; spread was actually tight because of high participation.
- NQ was also near highs -- this was a correlated risk-on environment that reversed.

**Setup code:**

```python
class TestATHCorrection2024:
    """Replay March 5-20, 2024 through the full system."""

    @pytest.fixture
    def scenario_data(self):
        return load_historical_scenario(
            start=datetime(2024, 3, 3, tzinfo=timezone.utc),
            end=datetime(2024, 3, 22, tzinfo=timezone.utc),
            symbol="BTC/USDT:USDT",
            include_orderbook=True,
            include_liquidations=True,
            include_funding=True,
            # Spreads were actually tight during euphoria
            spread_multiplier=1.0,
            depth_multiplier=1.0,
            # Correction phase: moderate spread widening
            correction_spread_multiplier=3.0,
            correction_depth_multiplier=0.5,
        )

    def test_funding_rate_risk_flagged(self, scenario_data):
        """Funding at 0.10-0.20% for multiple days should trigger
        the cascade probability model."""
        result = self._run_backtest(scenario_data)
        assert result.cascade_probability_max > 0.5, (
            "Cascade probability should exceed 0.5 with funding at 0.10%+"
        )
        assert result.pre_crash_risk_flags.get("funding_extreme") is True

    def test_position_sizing_during_euphoria(self, scenario_data):
        """During extreme funding, position sizing should be reduced
        even if model is confident."""
        result = self._run_backtest(scenario_data)
        euphoria_trades = [
            t for t in result.trades
            if t.timestamp >= datetime(2024, 3, 12, tzinfo=timezone.utc)
            and t.timestamp <= datetime(2024, 3, 14, tzinfo=timezone.utc)
        ]
        for trade in euphoria_trades:
            assert trade.position_size_pct <= 3.0, (
                f"Position size {trade.position_size_pct:.1f}% too large "
                "during extreme funding"
            )

    def test_survives_correction(self, scenario_data):
        result = self._run_backtest(scenario_data)
        assert result.max_drawdown_pct < 15.0
        # Should not have given back all euphoria gains
        assert result.total_return_pct > -5.0
```

**Pass/fail criteria:**
| Criterion | Pass | Fail |
|-----------|------|------|
| Cascade probability | > 0.5 before correction | < 0.5 |
| Position sizing | <= 3% during extreme funding | > 3% |
| Max drawdown | < 15% | >= 15% |
| Total return | > -5% | < -5% |

---

## 2. Synthetic Stress Scenarios

These scenarios test conditions that have not occurred but are plausible. They are generated by manipulating real data or constructing artificial data streams.

### 2.1 Sustained 0% Volatility for 48 Hours

**What this tests:** Model makes no money because there are no price moves, but costs accumulate from any trades taken. Tests whether the confidence gating pipeline correctly abstains.

**Setup code:**

```python
# tests/stress/test_synthetic_zero_vol.py
import numpy as np
from ep2_crypto.stress.synthetic import SyntheticScenarioBuilder

class TestZeroVolatility:
    """48 hours of flat price action."""

    @pytest.fixture
    def scenario_data(self):
        builder = SyntheticScenarioBuilder(
            base_price=50000.0,
            duration_hours=48,
            bar_interval_minutes=5,
        )
        # Flat price with only microstructure noise (< 0.005% per bar)
        builder.set_price_path(
            volatility_annualized=0.0,
            microstructure_noise_bps=0.5,
        )
        builder.set_orderbook(
            spread_bps=1.0,
            depth_normal=True,
        )
        builder.set_funding_rate(0.01)  # normal baseline
        return builder.build()

    def test_model_abstains(self, scenario_data):
        """Model should produce mostly 'flat' predictions
        or low-confidence signals that get gated."""
        result = self._run_backtest(scenario_data)
        total_bars = 48 * 12  # 576 five-min bars
        trade_rate = len(result.trades) / total_bars
        assert trade_rate < 0.05, (
            f"Trade rate {trade_rate:.2f} too high during zero vol. "
            "Expected < 5% of bars to generate trades."
        )

    def test_no_death_by_thousand_cuts(self, scenario_data):
        """If trades are taken, net PnL must not be significantly negative
        (costs should not eat the account)."""
        result = self._run_backtest(scenario_data)
        assert result.total_return_pct > -0.5, (
            f"Lost {abs(result.total_return_pct):.2f}% during flat market. "
            "Confidence gating should prevent cost accumulation."
        )

    def test_volatility_filter_active(self, scenario_data):
        """The min volatility filter (15% annualized) should block trades."""
        result = self._run_backtest(scenario_data)
        assert result.vol_filter_blocks > 0, (
            "Volatility filter did not block any trades during zero-vol period"
        )
```

**Pass/fail criteria:**
| Criterion | Pass | Fail |
|-----------|------|------|
| Trade rate | < 5% of bars | >= 5% |
| Total PnL | > -0.5% | < -0.5% |
| Volatility filter | Blocked trades | Did not activate |

**Hardening if test fails:**
- If model trades too much: Tighten the min volatility threshold from 15% to 20% annualized. Add a "flat regime" detector in the Efficiency Ratio layer that outputs a regime probability for "no trend, no mean-reversion."
- If losses accumulate: Cap maximum trades per hour at 2 during low-volatility regimes.

---

### 2.2 Consecutive 10 Flash Crashes in a Week

**What this tests:** System resilience to repeated extreme events. Kill switches activate but the system must recover between events. Tests graceful re-entry logic.

**Setup code:**

```python
class TestRepeatedFlashCrashes:
    """10 flash crashes of -5% to -10% each, spaced 12-24 hours apart."""

    @pytest.fixture
    def scenario_data(self):
        builder = SyntheticScenarioBuilder(
            base_price=50000.0,
            duration_hours=168,  # 1 week
            bar_interval_minutes=5,
        )
        # Normal volatility with 10 injected crash events
        builder.set_price_path(volatility_annualized=60.0)
        crash_times_hours = [8, 22, 38, 55, 72, 88, 104, 118, 135, 155]
        for t in crash_times_hours:
            builder.inject_crash(
                time_offset_hours=t,
                magnitude_pct=np.random.uniform(-10, -5),
                duration_minutes=np.random.randint(5, 30),
                recovery_pct=np.random.uniform(0.3, 0.7),
                recovery_minutes=np.random.randint(30, 120),
            )
        builder.set_orderbook(
            spread_bps=1.0,
            depth_normal=True,
            crash_spread_multiplier=8.0,
            crash_depth_multiplier=0.2,
        )
        return builder.build()

    def test_weekly_loss_limit_triggers(self, scenario_data):
        """Weekly loss limit (5%) must trigger and halt trading."""
        result = self._run_backtest(scenario_data)
        assert result.weekly_loss_limit_triggered is True

    def test_kill_switch_recovery(self, scenario_data):
        """After kill switch triggers, system must recover trading
        after the defined cooldown period."""
        result = self._run_backtest(scenario_data)
        # Should have at least 2 kill switch activations (daily or weekly)
        assert result.kill_switch_activations >= 2
        # Should also have at least 1 recovery (resumed trading)
        assert result.kill_switch_recoveries >= 1

    def test_drawdown_gate_graduated_reentry(self, scenario_data):
        """After drawdown, position sizes should gradually increase,
        not snap back to full size."""
        result = self._run_backtest(scenario_data)
        # Find position sizes after first recovery
        recovery_sizes = result.post_recovery_position_sizes
        if len(recovery_sizes) >= 3:
            # First recovery trade should be small
            assert recovery_sizes[0] < 3.0  # < 3% vs normal 4-5%
            # Should gradually increase
            assert recovery_sizes[-1] > recovery_sizes[0]

    def test_cumulative_drawdown_bounded(self, scenario_data):
        """Even with 10 crashes, cumulative drawdown must be bounded."""
        result = self._run_backtest(scenario_data)
        assert result.max_drawdown_pct < 20.0, (
            f"Cumulative drawdown {result.max_drawdown_pct:.1f}% exceeded "
            "20% across 10 flash crashes"
        )
```

**Pass/fail criteria:**
| Criterion | Pass | Fail |
|-----------|------|------|
| Weekly loss limit | Triggered | Not triggered |
| Kill switch recovery | >= 1 recovery | No recovery (system stuck) |
| Graduated re-entry | First post-recovery trade < 3% | Full size immediately |
| Cumulative drawdown | < 20% | >= 20% |

---

### 2.3 Exchange Suspends Withdrawals (Confidence Crisis)

**What this tests:** A scenario where the exchange we trade on (Binance) announces withdrawal suspension. This is an FTX-like event where the correct action is to close all positions and wait.

```python
class TestExchangeWithdrawalSuspension:
    """Simulate exchange withdrawal suspension mid-trade."""

    @pytest.fixture
    def scenario_data(self):
        builder = SyntheticScenarioBuilder(
            base_price=50000.0,
            duration_hours=72,
            bar_interval_minutes=5,
        )
        builder.set_price_path(volatility_annualized=80.0)
        # At T+24h, inject withdrawal suspension event
        builder.inject_event(
            time_offset_hours=24,
            event_type="withdrawal_suspension",
            # Price drops -15% over 4 hours after announcement
            price_impact_pct=-15.0,
            impact_duration_hours=4,
            # Volume spikes 5x
            volume_multiplier=5.0,
            # Spread widens 10x
            spread_multiplier=10.0,
        )
        return builder.build()

    def test_emergency_close_on_suspension(self, scenario_data):
        """If system has an open position when withdrawal suspension
        is detected, it must close immediately regardless of PnL."""
        result = self._run_backtest(scenario_data)
        open_positions_after_event = [
            p for p in result.position_log
            if p.is_open
            and p.timestamp > scenario_data.event_timestamp
        ]
        # All positions should be closed within 5 minutes of event
        for pos in open_positions_after_event:
            time_to_close = (
                pos.close_timestamp - scenario_data.event_timestamp
            ).total_seconds()
            assert time_to_close < 300, (
                f"Position still open {time_to_close:.0f}s after "
                "withdrawal suspension"
            )

    def test_no_new_trades_after_suspension(self, scenario_data):
        """No new positions should be opened after withdrawal suspension."""
        result = self._run_backtest(scenario_data)
        post_event_entries = [
            t for t in result.trades
            if t.entry_timestamp > scenario_data.event_timestamp
        ]
        assert len(post_event_entries) == 0
```

---

### 2.4 Regulatory Ban Announcement (Step-Change Regime Shift)

**What this tests:** A permanent regime shift where the old model is suddenly invalid. Tests BOCPD change point detection and model adaptation speed.

```python
class TestRegulatoryBan:
    """Simulate a US regulatory ban announcement that permanently
    changes market dynamics."""

    @pytest.fixture
    def scenario_data(self):
        builder = SyntheticScenarioBuilder(
            base_price=50000.0,
            duration_hours=168,  # 1 week
            bar_interval_minutes=5,
        )
        # Normal regime for 2 days
        builder.set_price_path(
            volatility_annualized=50.0,
            trend_bps_per_bar=0.0,
        )
        # At T+48h: regulatory ban announcement
        # Immediate -20% drop, then new regime with:
        # - Higher volatility (100% annualized)
        # - Negative drift (ongoing selling pressure)
        # - Collapsed correlation with NQ (goes to 0)
        # - Negative funding (shorts dominate)
        builder.inject_regime_change(
            time_offset_hours=48,
            immediate_impact_pct=-20.0,
            new_volatility_annualized=100.0,
            new_trend_bps_per_bar=-0.5,  # persistent selling
            new_nq_correlation=0.0,
            new_funding_rate=-0.05,
        )
        return builder.build()

    def test_bocpd_detects_change_point(self, scenario_data):
        """BOCPD should detect the regime change within 12 bars (1 hour)."""
        result = self._run_backtest(scenario_data)
        change_point = result.first_change_point_after(
            scenario_data.regime_change_timestamp
        )
        assert change_point is not None, "BOCPD did not detect regime change"
        detection_delay = (
            change_point - scenario_data.regime_change_timestamp
        ).total_seconds()
        assert detection_delay < 3600, (
            f"BOCPD took {detection_delay:.0f}s to detect change. "
            "Expected < 3600s (1 hour)."
        )

    def test_model_stops_old_signals(self, scenario_data):
        """After regime change, model should not continue emitting
        signals based on old regime patterns."""
        result = self._run_backtest(scenario_data)
        # Confidence should drop after regime change
        post_change_confidences = result.get_confidences_after(
            scenario_data.regime_change_timestamp
        )
        avg_confidence = np.mean(post_change_confidences[:24])  # first 2 hours
        assert avg_confidence < 0.5, (
            f"Average confidence {avg_confidence:.2f} too high after regime change"
        )

    def test_adaptation_within_24_hours(self, scenario_data):
        """System should adapt to new regime within 24 hours
        (via warm-start retraining)."""
        result = self._run_backtest(scenario_data)
        # Predictions in the last 24 hours should have reasonable accuracy
        last_day_accuracy = result.get_accuracy_for_period(
            start=scenario_data.regime_change_timestamp + timedelta(hours=24),
            end=scenario_data.end_timestamp,
        )
        assert last_day_accuracy > 0.48, (
            f"Accuracy {last_day_accuracy:.2f} still below random "
            "24 hours after regime change"
        )
```

---

### 2.5 Correlated Asset Decorrelation (BTC-NQ drops from 0.6 to 0 overnight)

**What this tests:** The macro event module depends on BTC-NQ correlation. When that correlation breaks, the NQ lead-lag signal becomes noise. The system must detect this and disable the NQ-based signal.

```python
class TestCorrelationBreakdown:
    """BTC-NQ correlation drops from 0.6 to 0 overnight."""

    @pytest.fixture
    def scenario_data(self):
        builder = SyntheticScenarioBuilder(
            base_price=50000.0,
            duration_hours=120,  # 5 days
            bar_interval_minutes=5,
        )
        # Day 1-2: normal correlation (0.6)
        # Day 3-5: correlation drops to 0
        builder.set_cross_market(
            nq_correlation_phase1=0.6,
            nq_correlation_phase2=0.0,
            correlation_break_hour=48,
            # NQ moves up while BTC moves down
            nq_trend_bps_per_bar=0.3,
            btc_trend_bps_per_bar=-0.3,
        )
        return builder.build()

    def test_correlation_monitor_fires(self, scenario_data):
        """EWMA correlation monitor should detect the breakdown."""
        result = self._run_backtest(scenario_data)
        assert result.correlation_breakdown_detected is True
        detection_time = result.correlation_breakdown_timestamp
        expected_break = scenario_data.correlation_break_timestamp
        # Should detect within 6 hours (EWMA trailing 6h)
        assert (detection_time - expected_break).total_seconds() < 21600

    def test_nq_signal_disabled(self, scenario_data):
        """After correlation breakdown, NQ-based signals should be
        disabled or heavily down-weighted."""
        result = self._run_backtest(scenario_data)
        post_break_nq_trades = [
            t for t in result.trades
            if t.signal_source == "macro_event"
            and t.timestamp > result.correlation_breakdown_timestamp
        ]
        assert len(post_break_nq_trades) == 0, (
            f"{len(post_break_nq_trades)} NQ-based trades after correlation broke"
        )

    def test_no_false_signals_from_divergence(self, scenario_data):
        """When NQ goes up but BTC goes down, the divergence signal
        should NOT trigger mean-reversion trades."""
        result = self._run_backtest(scenario_data)
        divergence_longs = [
            t for t in result.trades
            if t.signal_reason == "btc_nq_divergence_long"
            and t.timestamp > scenario_data.correlation_break_timestamp
        ]
        assert len(divergence_longs) == 0
```

---

### 2.6 Funding Rate Stays at +0.3% for 2 Weeks Straight

**What this tests:** Extremely elevated funding (annualized 328%) sustained for 2 weeks. The cascade probability model should be at maximum alert. The system should trade with minimal position size or not at all.

```python
class TestPersistentExtremeFunding:
    """Funding rate stays at +0.3% per 8h period for 2 weeks."""

    @pytest.fixture
    def scenario_data(self):
        builder = SyntheticScenarioBuilder(
            base_price=80000.0,
            duration_hours=336,  # 14 days
            bar_interval_minutes=5,
        )
        # Slow grind up with extreme funding
        builder.set_price_path(
            volatility_annualized=40.0,
            trend_bps_per_bar=0.1,  # slow uptrend
        )
        builder.set_funding_rate(0.30)  # 0.3% per 8h
        builder.set_oi_percentile(98)   # near ATH OI
        return builder.build()

    def test_cascade_probability_elevated(self, scenario_data):
        """Cascade probability model should be near maximum."""
        result = self._run_backtest(scenario_data)
        avg_cascade_prob = np.mean(result.cascade_probability_series)
        assert avg_cascade_prob > 0.7, (
            f"Average cascade probability {avg_cascade_prob:.2f} too low "
            "for 0.3% sustained funding"
        )

    def test_position_size_capped(self, scenario_data):
        """Position size should be heavily reduced or zero."""
        result = self._run_backtest(scenario_data)
        for trade in result.trades:
            assert trade.position_size_pct <= 2.0, (
                f"Position size {trade.position_size_pct:.1f}% too large "
                "during extreme funding"
            )

    def test_short_bias_emerges(self, scenario_data):
        """System should develop a short bias or at minimum be cautious
        about longs during extreme long positioning."""
        result = self._run_backtest(scenario_data)
        long_trades = [t for t in result.trades if t.direction == "long"]
        short_trades = [t for t in result.trades if t.direction == "short"]
        # Net direction should not be aggressively long
        if len(long_trades) + len(short_trades) > 0:
            long_ratio = len(long_trades) / (len(long_trades) + len(short_trades))
            assert long_ratio < 0.7, (
                f"Long ratio {long_ratio:.2f} too high during extreme long funding"
            )
```

---

## 3. Data Failure Scenarios

These test the system's behavior when data sources fail or degrade. Each scenario injects specific data failures into the pipeline.

### 3.1 Binance WebSocket Drops for 5 Minutes

**What this tests:** The primary data source goes silent. The system must detect the gap, stop trading (no stale data predictions), and resume correctly after reconnection.

```python
# tests/stress/test_data_failures.py
import asyncio
from unittest.mock import AsyncMock, patch
from ep2_crypto.ingest.exchange import BinanceExchangeCollector
from ep2_crypto.ingest.orchestrator import DataOrchestrator

class TestWebSocketDrop:
    """Binance WebSocket disconnects for 5 minutes."""

    @pytest.fixture
    def orchestrator(self):
        return DataOrchestrator(config=test_config)

    async def test_staleness_detection(self, orchestrator):
        """System must detect that data is stale within 10 seconds
        of WebSocket dropping."""
        async with orchestrator:
            # Simulate 5 minutes of normal data, then drop
            await self._feed_normal_data(orchestrator, duration_sec=60)

            # Simulate WebSocket disconnect
            orchestrator.streams["binance_klines"].simulate_disconnect()

            # Wait 15 seconds
            await asyncio.sleep(15)

            health = await orchestrator.health_check()
            assert health["binance_klines"]["status"] == "stale", (
                "Binance klines not marked stale after 15s disconnect"
            )
            assert health["binance_klines"]["staleness_seconds"] > 10

    async def test_no_predictions_on_stale_data(self, orchestrator):
        """Predictions must not be emitted when primary data is stale."""
        async with orchestrator:
            await self._feed_normal_data(orchestrator, duration_sec=60)
            orchestrator.streams["binance_klines"].simulate_disconnect()

            await asyncio.sleep(30)

            predictions = orchestrator.get_predictions_since_disconnect()
            assert len(predictions) == 0, (
                f"{len(predictions)} predictions emitted on stale data"
            )

    async def test_gap_backfill_on_reconnect(self, orchestrator):
        """After reconnection, the 5-minute gap must be backfilled
        via REST API before resuming normal operation."""
        async with orchestrator:
            await self._feed_normal_data(orchestrator, duration_sec=60)

            disconnect_time = asyncio.get_event_loop().time()
            orchestrator.streams["binance_klines"].simulate_disconnect()

            # Wait 5 minutes (simulated)
            await self._advance_time(orchestrator, 300)

            # Reconnect
            orchestrator.streams["binance_klines"].simulate_reconnect()
            await asyncio.sleep(5)

            # Verify gap was detected and backfill requested
            assert orchestrator.backfill_requests > 0, (
                "No backfill request after 5-minute data gap"
            )
            # Verify no NaN in feature pipeline after reconnect
            latest_features = orchestrator.get_latest_features()
            nan_count = sum(1 for v in latest_features.values() if np.isnan(v))
            assert nan_count == 0, (
                f"{nan_count} NaN features after reconnection"
            )

    async def test_reconnection_with_backoff(self, orchestrator):
        """Reconnection attempts must use exponential backoff."""
        async with orchestrator:
            orchestrator.streams["binance_klines"].simulate_disconnect()

            # Record reconnection attempt timestamps
            attempts = orchestrator.streams["binance_klines"].reconnect_attempts
            await self._advance_time(orchestrator, 120)

            # Verify exponential backoff pattern
            if len(attempts) >= 3:
                gap1 = attempts[1] - attempts[0]
                gap2 = attempts[2] - attempts[1]
                assert gap2 > gap1, (
                    f"Reconnection not using backoff: "
                    f"gap1={gap1:.1f}s, gap2={gap2:.1f}s"
                )
```

**Pass/fail criteria:**
| Criterion | Pass | Fail |
|-----------|------|------|
| Staleness detection | Within 10 seconds | > 30 seconds or never |
| Predictions on stale data | Zero | Any |
| Gap backfill on reconnect | Backfill requested, no NaN | NaN in features or no backfill |
| Exponential backoff | Increasing delays | Fixed or no delay |

**Hardening if test fails:**
- If staleness is not detected: Add a heartbeat monitor that checks `last_message_timestamp` for each stream every 5 seconds. If `now - last_message_ts > 10s`, mark stream as stale.
- If predictions are emitted on stale data: Add a staleness gate in the confidence pipeline: `if any_primary_stream_stale: return confidence=0.0`.

---

### 3.2 Order Book Data Stops but Trade Data Continues

**What this tests:** Partial data failure. The system must detect the inconsistency (trades happening but order book frozen) and degrade gracefully -- microstructure features that depend on the order book should be disabled, but the system can still make predictions from other features.

```python
class TestPartialDataFailure:
    """Order book stream dies but trade stream continues."""

    async def test_inconsistency_detected(self, orchestrator):
        """System should detect that trades are updating but
        order book is frozen."""
        async with orchestrator:
            await self._feed_normal_data(orchestrator, duration_sec=60)

            # Kill order book stream only
            orchestrator.streams["binance_orderbook"].simulate_disconnect()

            # Trade stream still alive
            await self._feed_trades_only(orchestrator, duration_sec=30)

            health = await orchestrator.health_check()
            assert health["binance_orderbook"]["status"] == "stale"
            assert health["binance_trades"]["status"] == "ok"
            assert health["data_consistency"] == "partial_failure"

    async def test_graceful_degradation(self, orchestrator):
        """Features that depend on order book should be disabled.
        Features from trades/price/volume should still work."""
        async with orchestrator:
            await self._feed_normal_data(orchestrator, duration_sec=60)
            orchestrator.streams["binance_orderbook"].simulate_disconnect()
            await self._feed_trades_only(orchestrator, duration_sec=30)

            features = orchestrator.get_latest_features()

            # Order book features should be NaN or a sentinel value
            ob_features = ["obi_weighted", "ofi_multi", "microprice",
                           "book_pressure", "spread_bps"]
            for f in ob_features:
                assert np.isnan(features[f]) or features[f] == 0.0, (
                    f"Order book feature '{f}' not properly disabled"
                )

            # Trade/price features should still be valid
            trade_features = ["trade_flow_imbalance", "volume_delta",
                              "roc_1", "realized_vol_5m"]
            for f in trade_features:
                assert not np.isnan(features[f]), (
                    f"Trade feature '{f}' is NaN despite trade stream being active"
                )

    async def test_confidence_reduced(self, orchestrator):
        """Predictions should have lower confidence when order book
        data is missing (it represents 60-80% of the signal)."""
        async with orchestrator:
            # Get normal confidence
            await self._feed_normal_data(orchestrator, duration_sec=300)
            normal_confidence = orchestrator.get_latest_prediction().confidence

            # Disable order book
            orchestrator.streams["binance_orderbook"].simulate_disconnect()
            await self._feed_trades_only(orchestrator, duration_sec=300)
            degraded_confidence = orchestrator.get_latest_prediction().confidence

            assert degraded_confidence < normal_confidence * 0.5, (
                f"Degraded confidence {degraded_confidence:.2f} not sufficiently "
                f"lower than normal {normal_confidence:.2f}"
            )
```

---

### 3.3 Cross-Market NQ Data Delayed by 30 Minutes Instead of 15

**What this tests:** yfinance NQ data is normally ~15 minutes delayed. If it becomes 30 minutes delayed, the lead-lag signal becomes even less useful. The system must detect the increased delay and down-weight NQ features.

```python
class TestNQExtraDelay:
    """NQ data arrives with 30-minute delay instead of 15."""

    @pytest.fixture
    def scenario_data(self):
        return load_historical_scenario(
            start=datetime(2024, 1, 10, 14, 0, tzinfo=timezone.utc),
            end=datetime(2024, 1, 10, 21, 0, tzinfo=timezone.utc),
            symbol="BTC/USDT:USDT",
            nq_delay_minutes=30,  # double the expected delay
        )

    def test_delay_detected(self, scenario_data):
        """System should detect that NQ data is more delayed than expected."""
        result = self._run_backtest(scenario_data)
        assert result.nq_delay_warning is True

    def test_nq_features_downweighted(self, scenario_data):
        """NQ lead-lag features should be down-weighted or disabled."""
        result = self._run_backtest(scenario_data)
        assert result.nq_feature_weight < 0.5, (
            "NQ features not sufficiently down-weighted despite extra delay"
        )

    def test_macro_module_paused(self, scenario_data):
        """Macro event module should pause if NQ data is too stale."""
        result = self._run_backtest(scenario_data)
        assert result.macro_module_active is False
```

---

### 3.4 Feature Computation Produces NaN for 3 Consecutive Bars

**What this tests:** A bug or data corruption causes NaN to propagate through the feature pipeline. The system must detect and handle NaN before it reaches the model.

```python
class TestNaNPropagation:
    """Feature pipeline produces NaN values."""

    def test_nan_detected_and_handled(self):
        """NaN features must be caught before model inference."""
        features = self._create_normal_features()
        # Inject NaN into 3 consecutive bars
        for i in range(3):
            features.iloc[100 + i, :] = np.nan

        pipeline = FeaturePipeline(config=test_config)
        cleaned = pipeline.handle_nan(features)

        # No NaN should reach the model
        assert cleaned.iloc[100:103].isna().sum().sum() == 0, (
            "NaN values not handled in feature pipeline"
        )

    def test_nan_handling_method_correct(self):
        """NaN handling must use forward-fill for price features
        and zero for count features -- NOT interpolation (look-ahead)."""
        features = self._create_normal_features()
        features.iloc[100, features.columns.get_loc("roc_1")] = np.nan
        features.iloc[100, features.columns.get_loc("trade_count")] = np.nan

        pipeline = FeaturePipeline(config=test_config)
        cleaned = pipeline.handle_nan(features)

        # Price feature should be forward-filled
        assert cleaned.iloc[100]["roc_1"] == features.iloc[99]["roc_1"]
        # Count feature should be zero-filled
        assert cleaned.iloc[100]["trade_count"] == 0.0

    def test_consecutive_nan_triggers_alert(self):
        """3+ consecutive NaN bars should trigger a data quality alert."""
        features = self._create_normal_features()
        for i in range(3):
            features.iloc[100 + i, :] = np.nan

        pipeline = FeaturePipeline(config=test_config)
        cleaned, alerts = pipeline.handle_nan_with_alerts(features)

        assert len(alerts) > 0
        assert any(a.severity == "warning" for a in alerts)

    def test_model_prediction_quality_after_nan(self):
        """Predictions immediately after NaN recovery should have
        reduced confidence."""
        features = self._create_normal_features()
        for i in range(3):
            features.iloc[100 + i, :] = np.nan

        pipeline = FeaturePipeline(config=test_config)
        cleaned = pipeline.handle_nan(features)

        model = load_test_model()
        prediction = model.predict(cleaned.iloc[103:104])
        # Confidence should be penalized after NaN recovery
        assert prediction.confidence < 0.6
```

---

### 3.5 Model Inference Takes 10 Seconds Instead of 10ms

**What this tests:** The model becomes extremely slow (OOM, CPU spike, etc.). The system must time out and skip the prediction rather than blocking the pipeline.

```python
class TestSlowInference:
    """Model inference takes 10 seconds instead of normal 10ms."""

    async def test_inference_timeout(self):
        """Model inference must be bounded by a timeout."""
        slow_model = SlowModelWrapper(
            model=load_test_model(),
            artificial_delay_seconds=10.0,
        )
        predictor = Predictor(model=slow_model, inference_timeout_ms=1000)

        features = self._create_normal_features()
        result = await predictor.predict_with_timeout(features)

        assert result.timed_out is True
        assert result.prediction is None
        assert result.latency_ms < 1500  # some overhead over 1000ms timeout

    async def test_pipeline_not_blocked(self):
        """Slow inference must not block the data ingestion pipeline."""
        orchestrator = DataOrchestrator(config=test_config)
        orchestrator.model = SlowModelWrapper(
            model=load_test_model(),
            artificial_delay_seconds=10.0,
        )

        async with orchestrator:
            await self._feed_normal_data(orchestrator, duration_sec=30)

            # Data ingestion should still be processing
            assert orchestrator.messages_received_last_10s > 0, (
                "Data pipeline blocked by slow model inference"
            )

    async def test_skip_and_resume(self):
        """After a timeout, the next bar should attempt inference normally."""
        model = IntermittentSlowModel(
            model=load_test_model(),
            slow_every_n=3,
            delay_seconds=10.0,
        )
        predictor = Predictor(model=model, inference_timeout_ms=1000)

        results = []
        for i in range(6):
            features = self._create_normal_features_bar(i)
            result = await predictor.predict_with_timeout(features)
            results.append(result)

        # Some should time out, others should succeed
        timed_out = sum(1 for r in results if r.timed_out)
        succeeded = sum(1 for r in results if not r.timed_out)
        assert timed_out >= 1 and succeeded >= 1, (
            "System did not recover from slow inference"
        )
```

---

## 4. Execution Failure Scenarios

### 4.1 Order Rejected (Insufficient Margin)

```python
class TestInsufficientMargin:
    """Exchange rejects order due to insufficient margin."""

    async def test_rejection_handled_gracefully(self):
        """Order rejection must not crash the system."""
        executor = OrderExecutor(config=test_config)
        with patch_exchange_api(reject_reason="insufficient_margin"):
            result = await executor.place_order(
                symbol="BTC/USDT:USDT",
                side="buy",
                size=1.0,
                order_type="limit",
                price=50000.0,
            )
            assert result.status == "rejected"
            assert result.error_code == "insufficient_margin"

    async def test_no_phantom_position(self):
        """After rejection, internal position tracker must show no position."""
        executor = OrderExecutor(config=test_config)
        with patch_exchange_api(reject_reason="insufficient_margin"):
            await executor.place_order(
                symbol="BTC/USDT:USDT", side="buy", size=1.0,
                order_type="limit", price=50000.0,
            )
            position = executor.get_position("BTC/USDT:USDT")
            assert position.size == 0.0, (
                "Phantom position created after rejected order"
            )

    async def test_margin_check_before_order(self):
        """System should check available margin before sending order."""
        executor = OrderExecutor(config=test_config)
        executor.available_margin = 100.0  # very low
        order_size_usd = 50000.0  # way too large

        result = await executor.place_order(
            symbol="BTC/USDT:USDT", side="buy",
            size=order_size_usd / 50000.0,
            order_type="limit", price=50000.0,
        )
        assert result.status == "rejected_locally"
        assert result.reason == "insufficient_margin_pre_check"
```

**Pass/fail criteria:**
| Criterion | Pass | Fail |
|-----------|------|------|
| Graceful handling | No crash, proper error log | System crashes or hangs |
| No phantom position | Position size = 0 after rejection | Position shows open |
| Pre-check | Rejected locally before sending to exchange | Sent and rejected by exchange |

---

### 4.2 Partial Fill on Entry, Full Fill on Exit

```python
class TestPartialFill:
    """Entry order partially fills, but exit order fully fills,
    resulting in an accidental short position."""

    async def test_partial_fill_tracking(self):
        """System must track actual filled quantity, not requested quantity."""
        executor = OrderExecutor(config=test_config)
        with patch_exchange_api(fill_ratio=0.3):  # only 30% fills
            entry = await executor.place_order(
                symbol="BTC/USDT:USDT", side="buy", size=1.0,
                order_type="limit", price=50000.0,
            )
            assert entry.filled_size == pytest.approx(0.3, abs=0.01)

            # Exit should only close the filled amount
            exit_order = await executor.close_position("BTC/USDT:USDT")
            assert exit_order.requested_size == pytest.approx(0.3, abs=0.01), (
                f"Exit tried to close {exit_order.requested_size} "
                f"but only {entry.filled_size} was filled"
            )

    async def test_no_accidental_short(self):
        """Closing more than the open position must not create
        an accidental opposite position."""
        executor = OrderExecutor(config=test_config)
        executor.position = Position(size=0.3, side="long")

        with patch_exchange_api(fill_ratio=1.0):
            exit_order = await executor.close_position("BTC/USDT:USDT")
            assert exit_order.requested_size == pytest.approx(0.3, abs=0.01)

            position = executor.get_position("BTC/USDT:USDT")
            assert position.size == 0.0, (
                f"Position size is {position.size} after close, expected 0"
            )

    async def test_unfilled_portion_cancelled(self):
        """The unfilled portion of a partial fill should be cancelled
        after the timeout."""
        executor = OrderExecutor(config=test_config)
        with patch_exchange_api(fill_ratio=0.3, fill_delay_ms=100):
            entry = await executor.place_order(
                symbol="BTC/USDT:USDT", side="buy", size=1.0,
                order_type="limit", price=50000.0,
                timeout_ms=10000,  # 10s timeout
            )
            # After timeout, unfilled portion should be cancelled
            await asyncio.sleep(11)
            assert entry.unfilled_cancelled is True
```

---

### 4.3 Stop Loss Not Triggered (Price Gaps Through)

```python
class TestStopLossGap:
    """Price gaps through the stop loss level -- stop is never hit
    at the expected price."""

    def test_catastrophic_stop_with_gap(self):
        """When price gaps through the stop, the system must close at
        the first available price, not wait for the stop price."""
        simulator = ExecutionSimulator(config=test_config)

        # Entry at 50000, stop at 49000 (2% stop)
        trade = Trade(
            entry_price=50000.0,
            stop_loss=49000.0,
            direction="long",
        )

        # Price gaps from 49500 to 48000 (skipping the 49000 stop level)
        price_bars = [
            Bar(open=50000, high=50100, low=49800, close=49900),
            Bar(open=49900, high=49950, low=49500, close=49600),
            # GAP: no bar touches 49000, next bar opens at 48000
            Bar(open=48000, high=48500, low=47800, close=48200),
        ]

        result = simulator.process_trade(trade, price_bars)

        # Stop should trigger at the gap-open price, not the stop level
        assert result.exit_price == pytest.approx(48000.0, abs=100), (
            f"Exit at {result.exit_price}, expected ~48000 (gap-open price)"
        )
        assert result.slippage_bps > 200, (
            "Slippage not properly modeled for gap-through stop"
        )

    def test_gap_through_logged(self):
        """Gap-through events must be logged as risk events."""
        simulator = ExecutionSimulator(config=test_config)
        # ... same setup as above ...
        result = simulator.process_trade(trade, price_bars)
        assert result.gap_through_event is True
        assert result.gap_through_magnitude_pct > 2.0
```

---

### 4.4 Exchange Goes into Maintenance with Open Position

```python
class TestExchangeMaintenance:
    """Exchange announces maintenance mode while we have an open position."""

    async def test_maintenance_detection(self):
        """System must detect exchange maintenance status."""
        executor = OrderExecutor(config=test_config)
        with patch_exchange_api(maintenance_mode=True):
            status = await executor.check_exchange_status()
            assert status.maintenance is True

    async def test_no_new_orders_during_maintenance(self):
        """No new orders should be submitted during maintenance."""
        executor = OrderExecutor(config=test_config)
        executor.exchange_maintenance = True

        result = await executor.place_order(
            symbol="BTC/USDT:USDT", side="buy", size=0.1,
            order_type="limit", price=50000.0,
        )
        assert result.status == "rejected_locally"
        assert result.reason == "exchange_maintenance"

    async def test_position_risk_during_maintenance(self):
        """If we have an open position during maintenance, the system
        must log a risk warning and attempt to close ASAP when
        maintenance ends."""
        executor = OrderExecutor(config=test_config)
        executor.position = Position(size=0.5, side="long")

        with patch_exchange_api(maintenance_mode=True):
            risk_events = executor.check_risk_events()
            assert any(
                e.type == "open_position_during_maintenance"
                for e in risk_events
            )

        # When maintenance ends, system should attempt immediate close
        with patch_exchange_api(maintenance_mode=False):
            await executor.on_maintenance_end()
            # Position should be in "pending close" state
            assert executor.position.pending_emergency_close is True
```

---

### 4.5 API Rate Limit Hit During Critical Moment

```python
class TestRateLimitDuringCritical:
    """API rate limit is hit while trying to close a position
    during a fast-moving market."""

    async def test_rate_limit_retry_with_backoff(self):
        """Rate-limited requests must retry with exponential backoff."""
        executor = OrderExecutor(config=test_config)
        call_count = 0

        async def rate_limited_api(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 3:
                raise RateLimitError("429 Too Many Requests", retry_after=1)
            return {"status": "filled", "filled_size": 0.5}

        executor.exchange.create_order = rate_limited_api

        result = await executor.place_order(
            symbol="BTC/USDT:USDT", side="sell", size=0.5,
            order_type="market", price=None,
            is_emergency_close=True,  # critical close
        )
        assert result.status == "filled"
        assert call_count == 4  # 3 retries + 1 success

    async def test_emergency_close_uses_separate_rate_limit(self):
        """Emergency closes should have their own rate limit budget,
        separate from normal order placement."""
        executor = OrderExecutor(config=test_config)
        # Exhaust normal rate limit
        executor.rate_limiter.exhaust("orders")

        # Emergency close should still work
        result = await executor.place_order(
            symbol="BTC/USDT:USDT", side="sell", size=0.5,
            order_type="market", price=None,
            is_emergency_close=True,
        )
        assert result.status != "rate_limited", (
            "Emergency close blocked by rate limit"
        )
```

---

## 5. Model Failure Scenarios

### 5.1 Model Predicts Same Direction for 100 Consecutive Bars

**What this tests:** A stuck or broken model that outputs the same prediction regardless of input. The system must detect this anomaly.

```python
class TestStuckModel:
    """Model predicts 'long' for 100 consecutive bars."""

    def test_monotonic_prediction_detected(self):
        """System must detect when predictions are suspiciously uniform."""
        predictions = [
            Prediction(direction="long", confidence=0.65, bar_index=i)
            for i in range(100)
        ]
        detector = ModelHealthMonitor()
        for pred in predictions:
            detector.record_prediction(pred)

        assert detector.is_anomalous() is True
        assert detector.anomaly_type == "stuck_prediction"
        assert detector.consecutive_same_direction >= 100

    def test_stuck_model_triggers_halt(self):
        """After detecting stuck model, system should halt trading
        and trigger a model health alert."""
        monitor = ModelHealthMonitor(
            max_consecutive_same=20,  # halt after 20 same predictions
        )
        for i in range(25):
            monitor.record_prediction(
                Prediction(direction="long", confidence=0.65, bar_index=i)
            )

        assert monitor.should_halt_trading() is True
        assert monitor.alert_triggered is True

    def test_auto_retrain_on_stuck(self):
        """System should automatically trigger a model retrain
        when stuck prediction is detected."""
        monitor = ModelHealthMonitor(max_consecutive_same=20)
        for i in range(25):
            monitor.record_prediction(
                Prediction(direction="long", confidence=0.65, bar_index=i)
            )

        actions = monitor.recommended_actions()
        assert "force_retrain" in actions
        assert "reduce_confidence_to_zero" in actions
```

**Pass/fail criteria:**
| Criterion | Pass | Fail |
|-----------|------|------|
| Anomaly detection | Flagged after 20 same predictions | Not detected |
| Trading halt | Triggered | Continues trading on broken model |
| Retrain trigger | Auto-retrain initiated | No action taken |

---

### 5.2 Prediction Confidence is 0.99 for Every Prediction (Miscalibrated)

```python
class TestMiscalibratedConfidence:
    """Every prediction comes with 0.99 confidence -- miscalibrated model."""

    def test_calibration_drift_detected(self):
        """System must detect when confidence distribution is degenerate."""
        predictions = [
            Prediction(
                direction=random.choice(["long", "short"]),
                confidence=0.99,
                bar_index=i,
            )
            for i in range(100)
        ]
        monitor = ModelHealthMonitor()
        for pred in predictions:
            monitor.record_prediction(pred)

        assert monitor.is_anomalous() is True
        assert monitor.anomaly_type == "degenerate_confidence"

    def test_calibration_check_against_actuals(self):
        """If 99% confident predictions are only 55% accurate,
        calibration error should be flagged."""
        monitor = ModelHealthMonitor()
        for i in range(100):
            correct = random.random() < 0.55
            monitor.record_prediction_with_outcome(
                Prediction(direction="long", confidence=0.99, bar_index=i),
                actual_correct=correct,
            )

        calibration_error = monitor.expected_calibration_error()
        assert calibration_error > 0.3, (
            f"ECE {calibration_error:.2f} too low -- should flag miscalibration"
        )
        assert monitor.should_force_recalibration() is True
```

---

### 5.3 Feature Importance Completely Shifts Overnight

```python
class TestFeatureImportanceShift:
    """Top features change completely between consecutive retrains."""

    def test_importance_drift_detected(self):
        """Feature importance shift > PSI threshold should trigger alert."""
        yesterday_importance = {
            "obi_weighted": 0.25, "trade_flow_imbalance": 0.20,
            "volume_delta": 0.15, "realized_vol_5m": 0.10,
            "roc_1": 0.08, "microprice": 0.07,
        }
        today_importance = {
            "obi_weighted": 0.02, "trade_flow_imbalance": 0.03,
            "volume_delta": 0.01, "realized_vol_5m": 0.01,
            "roc_1": 0.40, "microprice": 0.01,
            # A previously unimportant feature now dominates
            "hour_sin": 0.30,
        }

        monitor = FeatureDriftMonitor()
        psi = monitor.compute_importance_shift(
            yesterday_importance, today_importance
        )
        assert psi > 0.2, (
            f"PSI {psi:.2f} too low -- should detect importance shift"
        )
        assert monitor.should_investigate() is True

    def test_importance_shift_reduces_confidence(self):
        """After importance shift, predictions should have lower confidence
        until validated on new data."""
        monitor = FeatureDriftMonitor()
        monitor.record_importance_shift(psi=0.5)

        confidence_penalty = monitor.get_confidence_penalty()
        assert confidence_penalty > 0.2, (
            "Confidence penalty too small after major importance shift"
        )
```

---

### 5.4 Training Data Corrupted by Bad Candles

```python
class TestCorruptedTrainingData:
    """Training data contains bad candles (e.g., exchange glitch that
    shows BTC at $1 for one candle)."""

    def test_outlier_detection_in_pipeline(self):
        """Feature pipeline must detect and filter extreme outliers
        before they reach the model."""
        candles = self._create_normal_candles(count=1000)
        # Inject a bad candle: price drops to $1 then recovers
        candles.iloc[500] = {
            "open": 1.0, "high": 50000.0, "low": 1.0,
            "close": 50000.0, "volume": 100.0,
        }

        pipeline = FeaturePipeline(config=test_config)
        cleaned = pipeline.clean_candles(candles)

        # The bad candle should be filtered or clamped
        assert cleaned.iloc[500]["low"] > 100.0, (
            "Bad candle with $1 low price not filtered"
        )

    def test_return_clipping(self):
        """Returns computed from bad candles must be clipped
        to prevent extreme feature values."""
        candles = self._create_normal_candles(count=100)
        candles.iloc[50, candles.columns.get_loc("close")] = 1.0

        pipeline = FeaturePipeline(config=test_config)
        features = pipeline.compute_features(candles)

        # roc_1 should be clipped, not -99.998%
        max_abs_return = abs(features["roc_1"]).max()
        assert max_abs_return < 0.50, (
            f"Maximum absolute return {max_abs_return:.2f} not clipped"
        )

    def test_model_robust_to_outlier_leakage(self):
        """Even if an outlier leaks through, the model should not produce
        extreme predictions."""
        model = load_test_model()
        features = self._create_normal_features()
        # Inject extreme values
        features.iloc[0, 0] = 1e6  # extreme feature value

        prediction = model.predict(features.iloc[0:1])
        assert 0.0 <= prediction.confidence <= 1.0
        assert prediction.direction in ("long", "short", "flat")
```

---

## 6. Stress Test Metrics

During every stress test, measure and record these metrics.

### 6.1 Core Metrics to Capture

```python
@dataclass
class StressTestResult:
    """Comprehensive metrics captured during stress test."""

    # Scenario identification
    scenario_name: str
    scenario_type: str  # "historical", "synthetic", "data_failure", "execution", "model"
    start_time: datetime
    end_time: datetime

    # PnL metrics
    total_return_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float
    profit_factor: float
    buy_and_hold_return_pct: float
    excess_return_pct: float  # system return - buy-and-hold

    # Recovery metrics
    time_to_max_drawdown_minutes: int
    time_to_recover_from_max_dd_minutes: int | None  # None if never recovered
    drawdown_recovery_ratio: float  # recovery time / drawdown time

    # Kill switch metrics
    kill_switch_activations: int
    kill_switch_type: list[str]  # "daily_loss", "weekly_loss", "max_drawdown"
    time_to_first_kill_switch_minutes: int | None
    kill_switch_recovery_time_minutes: int | None

    # Graceful degradation metrics
    data_gaps_detected: int
    data_gaps_handled_correctly: int
    predictions_on_stale_data: int  # should be 0
    features_with_nan: int
    features_nan_handled_correctly: int

    # Confidence gating metrics
    total_bars: int
    predictions_made: int
    predictions_gated: int  # blocked by confidence gate
    trade_rate: float  # predictions_made / total_bars
    gated_rate: float  # predictions_gated / total_bars

    # Execution metrics
    total_trades: int
    orders_rejected: int
    partial_fills: int
    gap_through_stops: int
    average_slippage_bps: float
    max_slippage_bps: float

    # Model health metrics
    consecutive_same_predictions_max: int
    calibration_error: float
    feature_importance_psi: float
    prediction_latency_p99_ms: float

    # Cascade detection metrics (if applicable)
    cascade_alerts_fired: int
    cascade_alerts_true_positive: int
    cascade_detection_latency_seconds: float | None

    def passed(self, criteria: dict) -> bool:
        """Check all pass/fail criteria."""
        for metric, threshold in criteria.items():
            value = getattr(self, metric)
            op, limit = threshold
            if op == "<" and not (value < limit):
                return False
            if op == ">" and not (value > limit):
                return False
            if op == "==" and not (value == limit):
                return False
        return True
```

### 6.2 Standard Pass/Fail Criteria

Every stress test must meet these minimum criteria:

```python
UNIVERSAL_PASS_CRITERIA = {
    "max_drawdown_pct": ("<", 20.0),
    "predictions_on_stale_data": ("==", 0),
    "orders_rejected": ("<", 5),          # some rejections OK, many = bug
    "consecutive_same_predictions_max": ("<", 30),
    "calibration_error": ("<", 0.3),
}

HISTORICAL_PASS_CRITERIA = {
    **UNIVERSAL_PASS_CRITERIA,
    "kill_switch_activations": (">", 0),  # should activate during crashes
    "excess_return_pct": (">", -10.0),    # don't lose 10% more than buy-and-hold
}

SYNTHETIC_PASS_CRITERIA = {
    **UNIVERSAL_PASS_CRITERIA,
    "trade_rate": ("<", 0.3),  # shouldn't trade more than 30% of bars
}

DATA_FAILURE_PASS_CRITERIA = {
    **UNIVERSAL_PASS_CRITERIA,
    "data_gaps_handled_correctly": ("==", lambda r: r.data_gaps_detected),
    "predictions_on_stale_data": ("==", 0),
}
```

### 6.3 Comparison Dashboard

For each scenario, produce a summary table:

```
Scenario: COVID Crash (March 2020)
Period:   March 10-15, 2020
Duration: 5 days

+-------------------------------+-----------+-------------+---------+
| Metric                        | System    | Buy & Hold  | Status  |
+-------------------------------+-----------+-------------+---------+
| Total Return                  | -4.2%     | -52.0%      | PASS    |
| Max Drawdown                  | -8.1%     | -52.0%      | PASS    |
| Sharpe Ratio                  | -1.2      | -5.8        | PASS    |
| Trades Taken                  | 12        | 1           | --      |
| Kill Switch Activations       | 2         | N/A         | PASS    |
| Time to Kill Switch           | 45 min    | N/A         | PASS    |
| Predictions on Stale Data     | 0         | N/A         | PASS    |
| Cascade Alerts                | 3         | N/A         | PASS    |
+-------------------------------+-----------+-------------+---------+
Overall: PASS (8/8 criteria met)
```

---

## 7. Stress Test Automation & CI/CD

### 7.1 Test Runner

```python
# src/ep2_crypto/stress/runner.py
import structlog
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime

logger = structlog.get_logger()

@dataclass
class StressTestConfig:
    """Configuration for stress test suite."""
    scenarios_dir: Path
    results_dir: Path
    historical_data_dir: Path
    max_parallel: int = 4
    timeout_per_scenario_minutes: int = 30
    fail_fast: bool = False  # stop on first failure

class StressTestRunner:
    """Runs all stress test scenarios and produces a report."""

    def __init__(self, config: StressTestConfig):
        self.config = config
        self.results: list[StressTestResult] = []

    async def run_all(self) -> StressTestReport:
        """Run all scenarios and return consolidated report."""
        scenarios = self._discover_scenarios()
        logger.info("stress_test_suite_start", scenario_count=len(scenarios))

        for scenario in scenarios:
            try:
                result = await asyncio.wait_for(
                    self._run_scenario(scenario),
                    timeout=self.config.timeout_per_scenario_minutes * 60,
                )
                self.results.append(result)
                logger.info(
                    "stress_test_scenario_complete",
                    scenario=scenario.name,
                    passed=result.passed(scenario.pass_criteria),
                    max_drawdown=result.max_drawdown_pct,
                    total_return=result.total_return_pct,
                )
                if not result.passed(scenario.pass_criteria) and self.config.fail_fast:
                    logger.error("stress_test_fail_fast", scenario=scenario.name)
                    break
            except asyncio.TimeoutError:
                logger.error(
                    "stress_test_timeout",
                    scenario=scenario.name,
                    timeout_minutes=self.config.timeout_per_scenario_minutes,
                )
                self.results.append(StressTestResult(
                    scenario_name=scenario.name,
                    scenario_type=scenario.type,
                    timed_out=True,
                ))
            except Exception as exc:
                logger.error(
                    "stress_test_exception",
                    scenario=scenario.name,
                    error=str(exc),
                    exc_info=True,
                )

        return self._build_report()

    def _discover_scenarios(self) -> list:
        """Auto-discover stress test scenarios from test files."""
        # Discovers all classes inheriting from StressScenario
        # in tests/stress/ directory
        pass

    def _build_report(self) -> StressTestReport:
        total = len(self.results)
        passed = sum(
            1 for r in self.results
            if not r.timed_out and r.passed(r.pass_criteria)
        )
        failed = total - passed

        return StressTestReport(
            timestamp=datetime.utcnow(),
            total_scenarios=total,
            passed=passed,
            failed=failed,
            results=self.results,
            summary=self._generate_summary(),
        )
```

### 7.2 Scenario Data Loader

```python
# src/ep2_crypto/stress/scenario_loader.py
import structlog
from datetime import datetime, timezone
from pathlib import Path

logger = structlog.get_logger()

def load_historical_scenario(
    start: datetime,
    end: datetime,
    symbol: str,
    data_dir: Path | None = None,
    include_orderbook: bool = True,
    include_liquidations: bool = True,
    include_funding: bool = True,
    spread_multiplier: float = 1.0,
    depth_multiplier: float = 1.0,
    slippage_multiplier: float = 1.0,
    **kwargs,
) -> ScenarioData:
    """Load historical data for a stress test scenario.

    If data is not available locally, fetches from Binance REST API
    (OHLCV and trades can be backfilled; order book cannot).

    For order book, uses synthetic reconstruction from OHLCV + spread/depth
    multipliers to approximate crisis conditions.
    """
    # 1. Load OHLCV data
    candles = _load_or_fetch_candles(symbol, start, end, data_dir)

    # 2. Load or synthesize order book data
    if include_orderbook:
        orderbook = _load_orderbook(symbol, start, end, data_dir)
        if orderbook is None:
            logger.warning(
                "orderbook_not_available_synthesizing",
                start=start.isoformat(),
                end=end.isoformat(),
            )
            orderbook = _synthesize_orderbook(
                candles,
                spread_multiplier=spread_multiplier,
                depth_multiplier=depth_multiplier,
            )

    # 3. Load liquidation data
    if include_liquidations:
        liquidations = _load_or_fetch_liquidations(symbol, start, end, data_dir)

    # 4. Load funding rate data
    if include_funding:
        funding = _load_or_fetch_funding(symbol, start, end, data_dir)

    # 5. Load cross-market data (NQ, ETH)
    cross_market = _load_cross_market(start, end, data_dir)

    return ScenarioData(
        candles=candles,
        orderbook=orderbook,
        liquidations=liquidations,
        funding=funding,
        cross_market=cross_market,
        spread_multiplier=spread_multiplier,
        depth_multiplier=depth_multiplier,
        slippage_multiplier=slippage_multiplier,
        buy_and_hold_return=_compute_buy_and_hold(candles),
        metadata=kwargs,
    )
```

### 7.3 Synthetic Scenario Builder

```python
# src/ep2_crypto/stress/synthetic.py
import numpy as np
import pandas as pd
from dataclasses import dataclass, field

class SyntheticScenarioBuilder:
    """Build synthetic stress scenarios with precise control."""

    def __init__(
        self,
        base_price: float,
        duration_hours: int,
        bar_interval_minutes: int = 5,
        seed: int = 42,
    ):
        self.base_price = base_price
        self.n_bars = int(duration_hours * 60 / bar_interval_minutes)
        self.bar_interval = bar_interval_minutes
        self.rng = np.random.default_rng(seed)
        self._prices = np.full(self.n_bars, base_price)
        self._volumes = np.full(self.n_bars, 1000.0)
        self._spreads = np.full(self.n_bars, 0.01)  # 1 bps
        self._funding = np.full(self.n_bars, 0.01)  # baseline 0.01%
        self._events: list = []

    def set_price_path(
        self,
        volatility_annualized: float,
        trend_bps_per_bar: float = 0.0,
        microstructure_noise_bps: float = 0.5,
    ) -> "SyntheticScenarioBuilder":
        """Generate a GBM price path with specified volatility."""
        bar_vol = volatility_annualized / np.sqrt(365.25 * 24 * 60 / self.bar_interval)
        noise = self.rng.normal(0, 1, self.n_bars)
        log_returns = (
            trend_bps_per_bar / 10000
            + bar_vol * noise
            + self.rng.normal(0, microstructure_noise_bps / 10000, self.n_bars)
        )
        log_returns[0] = 0
        self._prices = self.base_price * np.exp(np.cumsum(log_returns))
        return self

    def inject_crash(
        self,
        time_offset_hours: float,
        magnitude_pct: float,
        duration_minutes: int,
        recovery_pct: float = 0.0,
        recovery_minutes: int = 60,
    ) -> "SyntheticScenarioBuilder":
        """Inject a flash crash at a specific time."""
        start_bar = int(time_offset_hours * 60 / self.bar_interval)
        crash_bars = int(duration_minutes / self.bar_interval)
        recovery_bars = int(recovery_minutes / self.bar_interval)

        # Crash: price drops by magnitude_pct over crash_bars
        pre_crash_price = self._prices[start_bar]
        crash_target = pre_crash_price * (1 + magnitude_pct / 100)

        for i in range(crash_bars):
            bar_idx = start_bar + i
            if bar_idx < self.n_bars:
                progress = (i + 1) / crash_bars
                # Accelerating crash (convex path)
                self._prices[bar_idx] = pre_crash_price + (
                    crash_target - pre_crash_price
                ) * (progress ** 0.5)
                # Volume spikes during crash
                self._volumes[bar_idx] *= 3 + 5 * progress
                # Spread widens during crash
                self._spreads[bar_idx] *= 2 + 8 * progress

        # Recovery
        if recovery_pct > 0:
            bottom = crash_target
            recovery_target = bottom + (pre_crash_price - bottom) * recovery_pct
            for i in range(recovery_bars):
                bar_idx = start_bar + crash_bars + i
                if bar_idx < self.n_bars:
                    progress = (i + 1) / recovery_bars
                    self._prices[bar_idx] = bottom + (
                        recovery_target - bottom
                    ) * (1 - (1 - progress) ** 2)

        # Shift all subsequent prices
        end_bar = start_bar + crash_bars + recovery_bars
        if end_bar < self.n_bars:
            shift = self._prices[end_bar] / self._prices[end_bar - 1]
            # Don't shift -- let the normal GBM continue from new level
        return self

    def inject_regime_change(
        self,
        time_offset_hours: float,
        immediate_impact_pct: float,
        new_volatility_annualized: float,
        new_trend_bps_per_bar: float = 0.0,
        new_nq_correlation: float = 0.0,
        new_funding_rate: float = 0.01,
    ) -> "SyntheticScenarioBuilder":
        """Inject a permanent regime change at a specific time."""
        break_bar = int(time_offset_hours * 60 / self.bar_interval)
        # Immediate impact
        self._prices[break_bar] *= (1 + immediate_impact_pct / 100)
        # New regime from break_bar onward
        new_bar_vol = new_volatility_annualized / np.sqrt(
            365.25 * 24 * 60 / self.bar_interval
        )
        remaining = self.n_bars - break_bar - 1
        noise = self.rng.normal(0, 1, remaining)
        log_returns = new_trend_bps_per_bar / 10000 + new_bar_vol * noise
        for i in range(remaining):
            bar_idx = break_bar + 1 + i
            self._prices[bar_idx] = self._prices[bar_idx - 1] * np.exp(log_returns[i])
        # Update funding
        self._funding[break_bar:] = new_funding_rate
        return self

    def build(self) -> "ScenarioData":
        """Build the scenario data object."""
        timestamps = pd.date_range(
            start="2024-01-01",
            periods=self.n_bars,
            freq=f"{self.bar_interval}min",
            tz="UTC",
        )
        candles = pd.DataFrame({
            "timestamp": timestamps,
            "open": self._prices,
            "high": self._prices * (1 + abs(self.rng.normal(0, 0.001, self.n_bars))),
            "low": self._prices * (1 - abs(self.rng.normal(0, 0.001, self.n_bars))),
            "close": self._prices,
            "volume": self._volumes,
        })
        return ScenarioData(
            candles=candles,
            funding=self._funding,
            spreads=self._spreads,
        )
```

### 7.4 CI/CD Integration

```yaml
# .github/workflows/stress-tests.yml
name: Stress Tests

on:
  # Run weekly on Sunday at 02:00 UTC
  schedule:
    - cron: '0 2 * * 0'
  # Run on PR to main if stress test files changed
  pull_request:
    branches: [main]
    paths:
      - 'src/ep2_crypto/stress/**'
      - 'src/ep2_crypto/backtest/**'
      - 'src/ep2_crypto/models/**'
      - 'src/ep2_crypto/features/**'
      - 'tests/stress/**'
  # Manual trigger
  workflow_dispatch:
    inputs:
      scenario_filter:
        description: 'Run specific scenario (e.g., "covid", "ftx", "all")'
        required: false
        default: 'all'

jobs:
  stress-test:
    runs-on: ubuntu-latest
    timeout-minutes: 120

    steps:
      - uses: actions/checkout@v4

      - name: Install uv
        uses: astral-sh/setup-uv@v5

      - name: Install dependencies
        run: uv sync --all-extras

      - name: Download historical stress test data
        run: |
          uv run python scripts/download_stress_data.py \
            --scenarios all \
            --output-dir data/stress/

      - name: Run stress tests (fast subset)
        if: github.event_name == 'pull_request'
        run: |
          uv run pytest tests/stress/ \
            -m "fast" \
            --timeout=300 \
            --tb=short \
            -v \
            --stress-report=stress-report-fast.json

      - name: Run stress tests (full suite)
        if: github.event_name != 'pull_request'
        run: |
          uv run pytest tests/stress/ \
            --timeout=1800 \
            --tb=long \
            -v \
            --stress-report=stress-report-full.json \
            ${{ github.event.inputs.scenario_filter != 'all' && format('-k {0}', github.event.inputs.scenario_filter) || '' }}

      - name: Generate stress test report
        if: always()
        run: |
          uv run python scripts/stress_report.py \
            --input stress-report-*.json \
            --output stress-report.md \
            --format markdown

      - name: Upload stress test report
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: stress-test-report-${{ github.run_number }}
          path: |
            stress-report-*.json
            stress-report.md

      - name: Comment PR with results
        if: github.event_name == 'pull_request' && always()
        uses: actions/github-script@v7
        with:
          script: |
            const fs = require('fs');
            const report = fs.readFileSync('stress-report.md', 'utf8');
            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: `## Stress Test Results\n\n${report}`
            });

      - name: Fail if critical scenarios failed
        run: |
          uv run python -c "
          import json, sys
          with open('stress-report-full.json' if '${{ github.event_name }}' != 'pull_request' else 'stress-report-fast.json') as f:
              report = json.load(f)
          failed = [r for r in report['results'] if not r['passed']]
          critical_failed = [r for r in failed if r.get('severity') == 'critical']
          if critical_failed:
              for r in critical_failed:
                  print(f'CRITICAL FAILURE: {r[\"scenario_name\"]}')
              sys.exit(1)
          if len(failed) > len(report['results']) * 0.2:
              print(f'{len(failed)}/{len(report[\"results\"])} scenarios failed (>20%)')
              sys.exit(1)
          print(f'All critical scenarios passed. {len(failed)} non-critical failures.')
          "
```

### 7.5 Pytest Configuration

```python
# conftest.py (in tests/stress/)
import pytest

def pytest_addoption(parser):
    parser.addoption(
        "--stress-report",
        action="store",
        default=None,
        help="Output file for stress test report (JSON)",
    )

def pytest_configure(config):
    config.addinivalue_line("markers", "fast: fast stress tests (<30s)")
    config.addinivalue_line("markers", "slow: slow stress tests (>5min)")
    config.addinivalue_line("markers", "historical: historical replay scenarios")
    config.addinivalue_line("markers", "synthetic: synthetic stress scenarios")
    config.addinivalue_line("markers", "data_failure: data pipeline failure tests")
    config.addinivalue_line("markers", "execution: order execution failure tests")
    config.addinivalue_line("markers", "model: model failure tests")

@pytest.fixture(scope="session")
def stress_data_dir():
    """Path to pre-downloaded stress test data."""
    return Path("data/stress/")

@pytest.fixture(scope="session")
def test_config():
    """Standard test configuration with conservative risk parameters."""
    return TradingConfig(
        position_size_pct=4.0,
        daily_loss_limit_pct=3.0,
        weekly_loss_limit_pct=5.0,
        max_drawdown_halt_pct=15.0,
        catastrophic_stop_atr=3.0,
        max_trades_per_day=30,
        confidence_threshold=0.60,
        min_volatility_annualized=15.0,
        max_volatility_annualized=150.0,
        inference_timeout_ms=1000,
    )
```

### 7.6 Data Download Script

```python
# scripts/download_stress_data.py
"""Download historical data needed for stress test scenarios.

Run this once or as part of CI to populate the data/stress/ directory.
Data sources: Binance REST API (OHLCV, funding), yfinance (NQ, SPX).
"""
import asyncio
import structlog
from datetime import datetime, timezone
from pathlib import Path

logger = structlog.get_logger()

SCENARIOS = {
    "covid_crash": {
        "start": datetime(2020, 3, 10, tzinfo=timezone.utc),
        "end": datetime(2020, 3, 15, tzinfo=timezone.utc),
        "description": "March 2020 COVID crash",
    },
    "china_ban": {
        "start": datetime(2021, 5, 17, tzinfo=timezone.utc),
        "end": datetime(2021, 5, 22, tzinfo=timezone.utc),
        "description": "May 2021 China mining ban crash",
    },
    "ftx_collapse": {
        "start": datetime(2022, 11, 4, tzinfo=timezone.utc),
        "end": datetime(2022, 11, 13, tzinfo=timezone.utc),
        "description": "November 2022 FTX collapse",
    },
    "svb_crisis": {
        "start": datetime(2023, 3, 9, tzinfo=timezone.utc),
        "end": datetime(2023, 3, 15, tzinfo=timezone.utc),
        "description": "March 2023 SVB banking crisis",
    },
    "ath_correction_2024": {
        "start": datetime(2024, 3, 3, tzinfo=timezone.utc),
        "end": datetime(2024, 3, 22, tzinfo=timezone.utc),
        "description": "March 2024 ATH euphoria then correction",
    },
}

async def download_scenario(name: str, config: dict, output_dir: Path) -> None:
    """Download all data for a single scenario."""
    scenario_dir = output_dir / name
    scenario_dir.mkdir(parents=True, exist_ok=True)

    start = config["start"]
    end = config["end"]

    logger.info("downloading_scenario", name=name, start=start, end=end)

    # 1. OHLCV candles (1-minute from Binance)
    await download_binance_candles(
        symbol="BTCUSDT",
        interval="1m",
        start=start,
        end=end,
        output=scenario_dir / "candles_1m.parquet",
    )

    # 2. Funding rate history
    await download_binance_funding(
        symbol="BTCUSDT",
        start=start,
        end=end,
        output=scenario_dir / "funding.parquet",
    )

    # 3. Cross-market (NQ, SPX via yfinance)
    download_yfinance(
        symbols=["NQ=F", "ES=F", "GC=F", "^VIX"],
        start=start,
        end=end,
        interval="5m",
        output=scenario_dir / "cross_market.parquet",
    )

    logger.info("scenario_downloaded", name=name)
```

### 7.7 Report Generator

```python
# scripts/stress_report.py
"""Generate human-readable stress test report from JSON results."""

def generate_markdown_report(results: list[dict]) -> str:
    lines = [
        "# Stress Test Report",
        f"**Generated:** {datetime.utcnow().isoformat()}Z",
        f"**Total scenarios:** {len(results)}",
        f"**Passed:** {sum(1 for r in results if r['passed'])}",
        f"**Failed:** {sum(1 for r in results if not r['passed'])}",
        "",
        "## Summary",
        "",
        "| Scenario | Type | Max DD | Return | Kill Switches | Status |",
        "|----------|------|--------|--------|---------------|--------|",
    ]

    for r in results:
        status = "PASS" if r["passed"] else "**FAIL**"
        lines.append(
            f"| {r['scenario_name']} | {r['scenario_type']} | "
            f"{r['max_drawdown_pct']:.1f}% | {r['total_return_pct']:.1f}% | "
            f"{r['kill_switch_activations']} | {status} |"
        )

    lines.extend(["", "## Failed Scenarios Detail", ""])

    for r in results:
        if not r["passed"]:
            lines.extend([
                f"### {r['scenario_name']}",
                f"**Failed criteria:**",
            ])
            for criterion, detail in r.get("failed_criteria", {}).items():
                lines.append(
                    f"- `{criterion}`: got {detail['actual']}, "
                    f"expected {detail['operator']} {detail['threshold']}"
                )
            lines.append("")

    return "\n".join(lines)
```

---

## Summary: What Each Category Validates

| Category | What It Tests | Number of Scenarios | CI Frequency |
|----------|--------------|--------------------:|-------------|
| Historical replays | System behavior during known extreme events | 5 | Weekly |
| Synthetic scenarios | Resilience to events that haven't happened yet | 6 | Weekly |
| Data failures | Graceful degradation when data sources fail | 5 | Every PR |
| Execution failures | Order handling edge cases | 5 | Every PR |
| Model failures | Detection of broken/miscalibrated models | 4 | Every PR |
| **Total** | | **25** | |

### Priority Order for Implementation

1. **Data failure tests** -- build first because they protect against the most common production issue (WebSocket drops happen daily).
2. **Kill switch / drawdown gate tests** -- build second because they prevent catastrophic loss.
3. **Historical replays (COVID, China ban)** -- build third because they validate the complete pipeline.
4. **Model failure tests** -- build fourth because miscalibrated models are a slow drain, not an immediate crisis.
5. **Synthetic scenarios** -- build last because they require the most infrastructure (scenario builder).

### Key Insight from Research

The system's existing architecture already has many of the right components (circuit breakers, exponential backoff, cascade detection via Hawkes process, drawdown gates, confidence gating pipeline). The stress tests documented here are designed to *verify* that those components actually work correctly under extreme conditions -- which is fundamentally different from assuming they work because the code exists.
