"""Liquidation cascade detection using Hawkes process and multi-factor scoring.

Detects liquidation cascades in BTC perpetual futures by modeling liquidation
event intensity with a Hawkes self-exciting point process and combining it
with market structure indicators into a composite cascade probability.

Key parameters from research:
- Hawkes branching ratio: normal 0.3-0.5, pre-cascade >0.8, cascade >0.9
- BTC endogeneity ~80% (80% of price moves triggered by prior market activity)
- Cascade probability >0.7 → reduce position to 25% or halt
- Multi-factor score: OI percentile, funding z-score, liquidation burst rate,
  book depth, price velocity
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class CascadeState:
    """Current state of the cascade detector.

    Attributes:
        cascade_probability: Composite probability of cascade [0, 1].
        hawkes_intensity: Current Hawkes process intensity (events/sec).
        branching_ratio: Hawkes branching ratio (>0.8 = pre-cascade).
        liq_burst_rate: Liquidation events per second (recent window).
        oi_percentile: Open interest percentile vs 30-day history.
        funding_zscore: Funding rate z-score vs 90-day history.
        book_depth_ratio: Current depth vs recent average (< 1 = thinning).
        price_velocity: Absolute price change rate (bps/sec).
        position_size_multiplier: Recommended position size reduction [0, 1].
        alert_level: 0=normal, 1=elevated, 2=warning, 3=critical.
    """

    cascade_probability: float = 0.0
    hawkes_intensity: float = 0.0
    branching_ratio: float = 0.0
    liq_burst_rate: float = 0.0
    oi_percentile: float = 0.5
    funding_zscore: float = 0.0
    book_depth_ratio: float = 1.0
    price_velocity: float = 0.0
    position_size_multiplier: float = 1.0
    alert_level: int = 0


class HawkesProcess:
    """Univariate Hawkes process with exponential kernel for liquidation modeling.

    The Hawkes process models self-exciting point processes where each event
    increases the probability of future events (liquidation cascades).

    Intensity: lambda(t) = mu + sum alpha * exp(-beta * (t - t_i)) for all t_i < t

    Parameters:
        mu: Background intensity (baseline liquidation rate, events/sec).
        alpha: Excitation magnitude (how much each event boosts intensity).
        beta: Decay rate (how quickly excitation fades, 1/seconds).

    The branching ratio n = alpha/beta measures self-excitation:
    - n < 1: stable (sub-critical)
    - n -> 1: critical (cascade imminent)
    - n > 1: unstable (super-critical, cascade in progress)

    For BTC liquidations:
    - Normal: n ≈ 0.3-0.5
    - Pre-cascade: n > 0.8
    - Cascade: n > 0.9
    """

    def __init__(
        self,
        mu: float = 0.01,
        alpha: float = 0.05,
        beta: float = 0.1,
        *,
        max_history: int = 500,
    ) -> None:
        if beta <= 0:
            msg = "beta must be positive"
            raise ValueError(msg)
        if alpha < 0:
            msg = "alpha must be non-negative"
            raise ValueError(msg)

        self._mu = mu
        self._alpha = alpha
        self._beta = beta
        self._max_history = max_history

        # Store recent event timestamps (seconds)
        self._event_times: list[float] = []
        # Running sum for efficient intensity computation
        self._intensity_sum: float = 0.0
        self._last_update_time: float = 0.0

    @property
    def mu(self) -> float:
        return self._mu

    @property
    def alpha(self) -> float:
        return self._alpha

    @property
    def beta(self) -> float:
        return self._beta

    @property
    def branching_ratio(self) -> float:
        """n = alpha/beta. Measures self-excitation strength."""
        return self._alpha / self._beta

    def intensity(self, t: float) -> float:
        """Compute current intensity λ(t).

        Uses the recursive formula for efficiency:
        A(t) = sum exp(-beta*(t - t_i)) = exp(-beta*(t - t_last)) * (A(t_last) + 1)
        lambda(t) = mu + alpha * A(t)
        """
        if not self._event_times:
            return self._mu

        # Decay the running sum to current time
        dt = t - self._last_update_time
        if dt > 0:
            decayed_sum = self._intensity_sum * math.exp(-self._beta * dt)
        else:
            decayed_sum = self._intensity_sum

        return self._mu + self._alpha * decayed_sum

    def add_event(self, t: float) -> float:
        """Record a liquidation event at time t and return new intensity.

        Args:
            t: Event timestamp in seconds.

        Returns:
            Updated intensity after the event.
        """
        if self._event_times and t < self._event_times[-1]:
            logger.warning(
                "hawkes_event_out_of_order",
                t=t,
                last=self._event_times[-1],
            )

        # Decay running sum to current time, then add the new event
        if self._event_times:
            dt = t - self._last_update_time
            if dt > 0:
                self._intensity_sum = self._intensity_sum * math.exp(-self._beta * dt) + 1.0
            else:
                self._intensity_sum += 1.0
        else:
            self._intensity_sum = 1.0

        self._last_update_time = t
        self._event_times.append(t)

        # Prune old events to bound memory
        if len(self._event_times) > self._max_history:
            self._event_times = self._event_times[-self._max_history :]

        return self._mu + self._alpha * self._intensity_sum

    def estimated_branching_ratio(self, window_s: float = 300.0) -> float:
        """Estimate branching ratio from recent event data.

        Uses the method of moments: n̂ = 1 - μ̂/λ̄
        where λ̄ is the average intensity and μ̂ is the background rate.

        Args:
            window_s: Window in seconds for estimation.

        Returns:
            Estimated branching ratio [0, 1+].
        """
        if len(self._event_times) < 2:
            return 0.0

        t_now = self._event_times[-1]
        t_start = t_now - window_s

        # Count events in window
        events_in_window = sum(1 for t in self._event_times if t >= t_start)
        if events_in_window < 2:
            return 0.0

        # Average rate in window
        lambda_bar = events_in_window / window_s

        if lambda_bar <= self._mu:
            return 0.0

        # Method of moments estimator
        ratio = 1.0 - self._mu / lambda_bar
        return max(0.0, ratio)

    def reset(self) -> None:
        """Reset the process state."""
        self._event_times.clear()
        self._intensity_sum = 0.0
        self._last_update_time = 0.0

    @property
    def event_count(self) -> int:
        return len(self._event_times)


class CascadeDetector:
    """Multi-factor liquidation cascade detector.

    Combines Hawkes process intensity with market structure indicators
    to produce a composite cascade probability score.

    Factors:
    1. Hawkes branching ratio (self-excitation of liquidations)
    2. OI percentile vs 30-day history (high OI = more leverage = more risk)
    3. Funding rate z-score (extreme funding = crowded positioning)
    4. Liquidation burst rate (events/sec in short window)
    5. Book depth ratio (thinning = less liquidity to absorb cascades)
    6. Price velocity (fast moves trigger more liquidations)

    Output: cascade probability → position size multiplier
    - probability < 0.3: normal (multiplier = 1.0)
    - probability 0.3-0.5: elevated (multiplier = 0.75)
    - probability 0.5-0.7: warning (multiplier = 0.5)
    - probability > 0.7: critical (multiplier = 0.25)

    Parameters:
        hawkes_mu: Hawkes background rate.
        hawkes_alpha: Hawkes excitation.
        hawkes_beta: Hawkes decay.
        oi_history_size: Rolling window for OI percentile.
        funding_history_size: Rolling window for funding z-score.
        depth_history_size: Rolling window for depth ratio.
        burst_window_s: Window for burst rate calculation.
    """

    def __init__(
        self,
        *,
        hawkes_mu: float = 0.01,
        hawkes_alpha: float = 0.05,
        hawkes_beta: float = 0.1,
        oi_history_size: int = 8640,
        funding_history_size: int = 270,
        depth_history_size: int = 288,
        burst_window_s: float = 60.0,
    ) -> None:
        self._hawkes = HawkesProcess(
            mu=hawkes_mu,
            alpha=hawkes_alpha,
            beta=hawkes_beta,
        )
        self._burst_window_s = burst_window_s

        # Rolling histories for percentile/z-score calculations
        self._oi_history: list[float] = []
        self._oi_history_size = oi_history_size  # 30 days at 5-min bars
        self._funding_history: list[float] = []
        self._funding_history_size = funding_history_size  # 90 days at 8h intervals
        self._depth_history: list[float] = []
        self._depth_history_size = depth_history_size  # 1 day at 5-min bars

        # Recent liquidation timestamps for burst rate
        self._recent_liq_times: list[float] = []
        self._max_recent_liqs = 1000

        self._state = CascadeState()
        self._log = logger.bind(module="cascade_detector")

    def on_liquidation(self, timestamp_s: float, quantity: float, side: str) -> CascadeState:
        """Process a liquidation event.

        Args:
            timestamp_s: Event timestamp in seconds.
            quantity: Liquidation size (e.g., in BTC).
            side: "long" or "short" (which side was liquidated).

        Returns:
            Updated cascade state.
        """
        # Update Hawkes process
        self._hawkes.add_event(timestamp_s)

        # Track recent liquidations for burst rate
        self._recent_liq_times.append(timestamp_s)
        cutoff = timestamp_s - self._burst_window_s
        self._recent_liq_times = [t for t in self._recent_liq_times if t >= cutoff]
        if len(self._recent_liq_times) > self._max_recent_liqs:
            self._recent_liq_times = self._recent_liq_times[-self._max_recent_liqs :]

        # Update burst rate
        self._state.liq_burst_rate = len(self._recent_liq_times) / self._burst_window_s

        # Update Hawkes metrics
        self._state.hawkes_intensity = self._hawkes.intensity(timestamp_s)
        self._state.branching_ratio = self._hawkes.estimated_branching_ratio()

        # Recompute cascade probability
        self._update_cascade_probability()

        return self._state

    def on_bar(
        self,
        timestamp_s: float,
        *,
        open_interest: float | None = None,
        funding_rate: float | None = None,
        book_depth: float | None = None,
        price: float | None = None,
        prev_price: float | None = None,
        bar_duration_s: float = 300.0,
    ) -> CascadeState:
        """Process a new bar of market data.

        Updates OI percentile, funding z-score, book depth ratio,
        and price velocity.

        Args:
            timestamp_s: Bar timestamp in seconds.
            open_interest: Current open interest value.
            funding_rate: Current funding rate.
            book_depth: Total bid+ask depth near best price.
            price: Current price.
            prev_price: Previous bar price.
            bar_duration_s: Bar duration in seconds.

        Returns:
            Updated cascade state.
        """
        # Update OI history and percentile
        if open_interest is not None:
            self._oi_history.append(open_interest)
            if len(self._oi_history) > self._oi_history_size:
                self._oi_history = self._oi_history[-self._oi_history_size :]
            if len(self._oi_history) >= 10:
                sorted_oi = sorted(self._oi_history)
                rank = sum(1 for v in sorted_oi if v <= open_interest)
                self._state.oi_percentile = rank / len(sorted_oi)

        # Update funding z-score
        if funding_rate is not None:
            self._funding_history.append(funding_rate)
            if len(self._funding_history) > self._funding_history_size:
                self._funding_history = self._funding_history[-self._funding_history_size :]
            if len(self._funding_history) >= 10:
                arr = np.array(self._funding_history)
                mean = float(np.mean(arr))
                std = float(np.std(arr))
                if std > 1e-10:
                    self._state.funding_zscore = (funding_rate - mean) / std
                else:
                    self._state.funding_zscore = 0.0

        # Update book depth ratio
        if book_depth is not None:
            self._depth_history.append(book_depth)
            if len(self._depth_history) > self._depth_history_size:
                self._depth_history = self._depth_history[-self._depth_history_size :]
            if len(self._depth_history) >= 10:
                avg_depth = sum(self._depth_history) / len(self._depth_history)
                if avg_depth > 0:
                    self._state.book_depth_ratio = book_depth / avg_depth

        # Update price velocity (bps/sec)
        if price is not None and prev_price is not None and prev_price > 0:
            price_change_bps = abs(price - prev_price) / prev_price * 10_000
            self._state.price_velocity = price_change_bps / bar_duration_s

        # Update Hawkes intensity (decay without new events)
        self._state.hawkes_intensity = self._hawkes.intensity(timestamp_s)
        self._state.branching_ratio = self._hawkes.estimated_branching_ratio()

        # Recompute cascade probability
        self._update_cascade_probability()

        return self._state

    def _update_cascade_probability(self) -> None:
        """Compute composite cascade probability from all factors.

        Uses logistic combination of normalized factors:
        P(cascade) = sigmoid(w1*f1 + w2*f2 + ... + bias)
        """
        # Normalize each factor to [0, 1]

        # 1. Branching ratio: 0.3 = normal, 0.8+ = pre-cascade
        br = self._state.branching_ratio
        br_score = max(0.0, min(1.0, (br - 0.3) / 0.6))

        # 2. OI percentile: already [0, 1], but > 0.8 is high risk
        oi_score = max(0.0, min(1.0, (self._state.oi_percentile - 0.5) / 0.5))

        # 3. Funding z-score: |z| > 2 is extreme
        fz = abs(self._state.funding_zscore)
        funding_score = max(0.0, min(1.0, (fz - 1.0) / 2.0))

        # 4. Burst rate: > 1 event/sec is high
        burst_score = max(0.0, min(1.0, self._state.liq_burst_rate / 2.0))

        # 5. Book depth ratio: < 0.5 = severely thinned
        depth_score = max(0.0, min(1.0, 1.0 - self._state.book_depth_ratio))

        # 6. Price velocity: > 5 bps/sec is very fast
        vel_score = max(0.0, min(1.0, self._state.price_velocity / 5.0))

        # Weighted logistic combination
        weights = {
            "branching_ratio": 0.30,
            "burst_rate": 0.20,
            "oi_percentile": 0.15,
            "funding_zscore": 0.10,
            "book_depth": 0.15,
            "price_velocity": 0.10,
        }

        z = (
            weights["branching_ratio"] * br_score
            + weights["burst_rate"] * burst_score
            + weights["oi_percentile"] * oi_score
            + weights["funding_zscore"] * funding_score
            + weights["book_depth"] * depth_score
            + weights["price_velocity"] * vel_score
        )

        # Scale to [0, 1] — z is already in [0, 1] from the weighted sum
        # Apply sigmoid-like sharpening to create clearer thresholds
        # Using a shifted sigmoid: P = 1/(1 + exp(-k*(z - 0.5)))
        k = 8.0  # steepness
        prob = 1.0 / (1.0 + math.exp(-k * (z - 0.4)))

        self._state.cascade_probability = prob

        # Position size multiplier based on probability
        if prob < 0.3:
            self._state.position_size_multiplier = 1.0
            self._state.alert_level = 0
        elif prob < 0.5:
            self._state.position_size_multiplier = 0.75
            self._state.alert_level = 1
        elif prob < 0.7:
            self._state.position_size_multiplier = 0.5
            self._state.alert_level = 2
        else:
            self._state.position_size_multiplier = 0.25
            self._state.alert_level = 3

        if self._state.alert_level >= 2:
            self._log.warning(
                "cascade_alert",
                probability=prob,
                alert_level=self._state.alert_level,
                branching_ratio=self._state.branching_ratio,
                burst_rate=self._state.liq_burst_rate,
                multiplier=self._state.position_size_multiplier,
            )

    @property
    def state(self) -> CascadeState:
        """Current cascade detector state."""
        return self._state

    @property
    def hawkes(self) -> HawkesProcess:
        """Access the underlying Hawkes process."""
        return self._hawkes

    def reset(self) -> None:
        """Reset all state."""
        self._hawkes.reset()
        self._oi_history.clear()
        self._funding_history.clear()
        self._depth_history.clear()
        self._recent_liq_times.clear()
        self._state = CascadeState()


class OnlineHawkesEstimator:
    """Online MLE estimator for Hawkes process parameters via stochastic gradient.

    Fits mu, alpha, beta using stochastic gradient ascent on the Hawkes
    log-likelihood. After each event batch the parameters are updated toward
    higher log-likelihood without storing the full history.

    Log-likelihood of Hawkes process on [0, T] with events t_1,...,t_N:
      LL = sum_i log(lambda(t_i)) - integral_0^T lambda(s) ds
         = sum_i log(mu + alpha * A_i) - mu*T - alpha/beta * sum_i (1 - exp(-beta*(T-t_i)))

    where A_i = sum_{j<i} exp(-beta*(t_i - t_j)).

    Parameters:
        mu_init: Initial background rate.
        alpha_init: Initial excitation magnitude.
        beta_init: Initial decay rate (must be > alpha for stationarity).
        lr: Learning rate for stochastic gradient steps.
        lr_decay: Multiplicative decay applied to lr every `decay_every` events.
        decay_every: Number of events between lr decay steps.
        min_mu: Lower bound for mu.
        max_alpha_beta_ratio: Upper bound for alpha/beta (branching ratio cap).
    """

    def __init__(
        self,
        mu_init: float = 0.01,
        alpha_init: float = 0.05,
        beta_init: float = 0.1,
        lr: float = 1e-3,
        lr_decay: float = 0.995,
        decay_every: int = 50,
        min_mu: float = 1e-6,
        max_alpha_beta_ratio: float = 0.99,
    ) -> None:
        if beta_init <= 0:
            msg = "beta_init must be positive"
            raise ValueError(msg)
        if alpha_init < 0:
            msg = "alpha_init must be non-negative"
            raise ValueError(msg)

        self._mu = mu_init
        self._alpha = alpha_init
        self._beta = beta_init
        self._lr = lr
        self._lr_decay = lr_decay
        self._decay_every = decay_every
        self._min_mu = min_mu
        self._max_ratio = max_alpha_beta_ratio

        # Online state
        self._event_times: list[float] = []
        self._n_events: int = 0
        self._A: float = 0.0  # running recursive sum exp(-beta*(t - t_last))
        self._t_last: float = 0.0
        self._T: float = 0.0  # current time horizon

    @property
    def n_events(self) -> int:
        return self._n_events

    def get_params(self) -> dict[str, float]:
        """Return current estimated parameters."""
        return {
            "mu": self._mu,
            "alpha": self._alpha,
            "beta": self._beta,
            "branching_ratio": self._alpha / self._beta,
        }

    def add_event(self, t: float) -> None:
        """Record a new liquidation event and update parameters.

        Args:
            t: Event timestamp in seconds.
        """
        # Decay running sum to current time
        if self._n_events > 0:
            dt = t - self._t_last
            if dt > 0:
                self._A = self._A * math.exp(-self._beta * dt) + 1.0
            else:
                self._A += 1.0
        else:
            self._A = 1.0

        self._t_last = t
        self._T = t
        self._event_times.append(t)
        self._n_events += 1

        # Perform a gradient step every event
        self._update_params(t)

        # Decay learning rate periodically
        if self._n_events % self._decay_every == 0:
            self._lr *= self._lr_decay

        # Bound memory
        if len(self._event_times) > 1000:
            self._event_times = self._event_times[-500:]

    def _update_params(self, t_new: float) -> None:
        """One stochastic gradient ascent step on log-likelihood.

        Uses finite differences for beta gradient (analytically tricky).
        Mu and alpha have closed-form gradients.
        """
        if self._n_events < 2:
            return

        # Current intensity at t_new
        lam = self._mu + self._alpha * self._A
        if lam <= 0:
            return

        # Gradient w.r.t. mu: d/dmu [log lam - mu * T] = 1/lam - T (stochastic approx)
        grad_mu = 1.0 / lam - self._T

        # Gradient w.r.t. alpha: d/dalpha [log lam - alpha/beta * integral_term]
        # Stochastic: 1/lam * A - 1/beta * (1 - exp(-beta*(T - t_new)))
        integral_alpha = (1.0 - math.exp(-self._beta * max(0.0, self._T - t_new))) / self._beta
        grad_alpha = self._A / lam - integral_alpha

        # Gradient w.r.t. beta: finite difference
        grad_beta = self._fd_beta_gradient(t_new, lam)

        # Gradient ascent steps
        self._mu = max(self._min_mu, self._mu + self._lr * grad_mu)
        self._alpha = max(0.0, self._alpha + self._lr * grad_alpha)
        self._beta = max(self._alpha / self._max_ratio + 1e-9, self._beta + self._lr * grad_beta)

        # Enforce stationarity: alpha/beta < max_ratio
        if self._alpha / self._beta >= self._max_ratio:
            self._alpha = self._beta * self._max_ratio * 0.99

    def _fd_beta_gradient(self, t_new: float, lam: float, eps: float = 1e-5) -> float:
        """Finite difference approximation of d(log_lik)/d(beta)."""
        beta_plus = self._beta + eps
        beta_minus = max(1e-9, self._beta - eps)

        # A at t_new for perturbed beta values (recompute from recent events)
        def _compute_a(beta_val: float) -> float:
            a = 0.0
            for ti in self._event_times[-50:]:  # Use at most 50 recent events
                if ti < t_new:
                    a = a * math.exp(-beta_val * (t_new - ti)) + 1.0 if a > 0 else 1.0
            return a

        a_plus = _compute_a(beta_plus)
        a_minus = _compute_a(beta_minus)

        lam_plus = self._mu + self._alpha * a_plus
        lam_minus = self._mu + self._alpha * a_minus

        ll_plus = math.log(max(lam_plus, 1e-10))
        ll_minus = math.log(max(lam_minus, 1e-10))

        return (ll_plus - ll_minus) / (2 * eps)

    def reset(self) -> None:
        """Reset online estimator state (keep hyperparameters)."""
        self._event_times.clear()
        self._n_events = 0
        self._A = 0.0
        self._t_last = 0.0
        self._T = 0.0


class StateDependentAmplifier:
    """State-dependent cascade amplifier based on OI and funding conditions.

    Amplifies the base cascade probability when market conditions indicate
    elevated systemic risk (high OI + extreme funding = crowded positions
    that could unwind violently).

    Stress levels:
        0 = Normal: OI < 75th percentile AND |funding| < 1σ
        1 = Elevated: OI ≥ 75th percentile OR |funding| ≥ 1σ
        2 = High: OI ≥ 90th percentile AND |funding| ≥ 1.5σ
        3 = Critical: OI ≥ 95th percentile AND |funding| ≥ 2σ

    Amplification: P_amplified = 1 - (1 - P_base) * exp(-stress_factor)
    where stress_factor ∈ {0.0, 0.15, 0.35, 0.65} for levels 0-3.
    """

    _STRESS_FACTORS = {0: 0.0, 1: 0.15, 2: 0.35, 3: 0.65}

    def stress_level(self, oi_percentile: float, funding_zscore: float) -> int:
        """Compute stress level from OI percentile and funding z-score.

        Args:
            oi_percentile: OI rank in [0, 1] vs historical distribution.
            funding_zscore: Funding rate z-score (signed).

        Returns:
            Stress level 0-3.
        """
        abs_fz = abs(funding_zscore)

        if oi_percentile >= 0.95 and abs_fz >= 2.0:
            return 3
        if oi_percentile >= 0.90 and abs_fz >= 1.5:
            return 2
        if oi_percentile >= 0.75 or abs_fz >= 1.0:
            return 1
        return 0

    def amplify(
        self,
        base_prob: float,
        oi_percentile: float,
        funding_zscore: float,
    ) -> tuple[float, float]:
        """Amplify base cascade probability by market stress conditions.

        Args:
            base_prob: Base cascade probability from CascadeDetector in [0, 1].
            oi_percentile: OI rank vs history in [0, 1].
            funding_zscore: Funding rate z-score.

        Returns:
            Tuple of (amplified_probability, stress_factor) where
            amplified_probability ∈ [0, 1] and stress_factor ∈ [0, 0.65].
        """
        level = self.stress_level(oi_percentile, funding_zscore)
        sf = self._STRESS_FACTORS[level]

        if sf == 0.0:
            return base_prob, 0.0

        # P_amplified = 1 - (1 - P_base) * exp(-sf)
        # This maps [0,1] to [0,1], monotonically increases with sf
        amplified = 1.0 - (1.0 - base_prob) * math.exp(-sf)
        return min(1.0, amplified), sf
