# Model Retraining Strategies for Live 5-Min Crypto Prediction

> Deep research on when, how, and how much to retrain LightGBM + CatBoost + GRU stacking ensemble in production.

---

## 1. Fixed Schedule Retraining

### The Tradeoff: Latency vs Freshness

For 5-min crypto prediction, the key tension is:
- **Too frequent**: Wastes compute, risks overfitting to noise, destabilizes the ensemble
- **Too infrequent**: Model goes stale as crypto regimes shift (alpha half-life: 1-6 months per RESEARCH_SYNTHESIS.md)

### Evidence-Based Recommendations for 5-Min Crypto

| Schedule | Use Case | Evidence |
|----------|----------|----------|
| Every 2-4 hours | LightGBM/CatBoost warm-start | PLAN.md already specifies this; crypto microstructure patterns shift intraday |
| Every 4-8 hours | GRU fine-tune | More expensive to train; temporal patterns are slower-moving |
| Every 24 hours | Full cold retrain (all models) | Reset accumulated warm-start drift; recompute feature importance |
| Weekly | Hyperparameter re-optimization + meta-learner retrain | Expensive but catches structural shifts |

### Why 2-4 Hours for Tree Models

Crypto market microstructure has distinct intraday patterns:
- Asia session (00-08 UTC): Lower vol, different order flow dynamics
- Europe session (08-16 UTC): Increasing vol, cross-market correlations activate
- US session (16-00 UTC): Highest vol, NQ lead-lag most pronounced

A model trained on Asia session data will underperform during US session. The 2-4 hour window captures approximately one session transition, keeping the model adapted to current conditions.

### Implementation Pattern

```python
import asyncio
from datetime import datetime, timezone

RETRAIN_INTERVALS = {
    "lgbm_warmstart": 2 * 3600,      # 2 hours
    "catboost_warmstart": 2 * 3600,   # 2 hours
    "gru_finetune": 6 * 3600,         # 6 hours
    "full_cold_retrain": 24 * 3600,   # 24 hours
    "hyperparam_search": 7 * 86400,   # 7 days
}

class ScheduledRetrainer:
    def __init__(self):
        self.last_retrain: dict[str, float] = {}

    async def check_and_retrain(self, current_time: float) -> list[str]:
        actions = []
        for task, interval in RETRAIN_INTERVALS.items():
            last = self.last_retrain.get(task, 0.0)
            if current_time - last >= interval:
                actions.append(task)
                self.last_retrain[task] = current_time
        return actions
```

### Decision: Start with fixed schedule, layer on drift/performance triggers later

Fixed schedule is the simplest to implement and debug. It provides a predictable baseline. Performance and drift triggers (sections 2-3) act as *additional* triggers that can fire between scheduled retrains.

---

## 2. Performance-Triggered Retraining

### Metrics to Monitor

| Metric | Window | Threshold | Action |
|--------|--------|-----------|--------|
| Rolling Sharpe (after costs) | 48h (576 bars) | < 0.0 | Trigger retrain |
| Rolling directional accuracy | 24h (288 bars) | < 50% | Trigger retrain |
| Rolling hit rate on high-confidence trades | 12h (144 bars) | < 48% | Trigger retrain |
| Consecutive losing trades | N/A | > 8 | Trigger retrain |
| Prediction calibration error | 24h | > 0.15 | Recalibrate + retrain |

### Threshold Calibration: Avoiding Over-Reaction

The biggest risk is retraining on every bad day. A single bad day in crypto (a -5% flash crash) will tank all rolling metrics temporarily, but the model may be fine.

**Solution: Require sustained degradation, not point-in-time**

```python
import numpy as np

class PerformanceMonitor:
    """Track rolling performance and trigger retraining only on sustained degradation."""

    def __init__(
        self,
        sharpe_window: int = 576,       # 48h of 5-min bars
        sharpe_threshold: float = 0.0,
        accuracy_window: int = 288,      # 24h
        accuracy_threshold: float = 0.50,
        cooldown_bars: int = 144,        # 12h minimum between triggers
        confirmation_bars: int = 72,     # 6h of sustained degradation required
    ):
        self.sharpe_window = sharpe_window
        self.sharpe_threshold = sharpe_threshold
        self.accuracy_window = accuracy_window
        self.accuracy_threshold = accuracy_threshold
        self.cooldown_bars = cooldown_bars
        self.confirmation_bars = confirmation_bars
        self.bars_below_threshold = 0
        self.bars_since_last_trigger = 0

    def update(self, returns: np.ndarray, predictions: np.ndarray, actuals: np.ndarray) -> bool:
        """Returns True if retraining should be triggered."""
        self.bars_since_last_trigger += 1

        if self.bars_since_last_trigger < self.cooldown_bars:
            return False

        # Rolling Sharpe
        if len(returns) >= self.sharpe_window:
            recent = returns[-self.sharpe_window:]
            sharpe = recent.mean() / max(recent.std(), 1e-10) * np.sqrt(105_120)
            sharpe_degraded = sharpe < self.sharpe_threshold
        else:
            sharpe_degraded = False

        # Rolling accuracy
        if len(predictions) >= self.accuracy_window:
            recent_pred = predictions[-self.accuracy_window:]
            recent_actual = actuals[-self.accuracy_window:]
            accuracy = (np.sign(recent_pred) == np.sign(recent_actual)).mean()
            accuracy_degraded = accuracy < self.accuracy_threshold
        else:
            accuracy_degraded = False

        if sharpe_degraded or accuracy_degraded:
            self.bars_below_threshold += 1
        else:
            self.bars_below_threshold = 0

        # Require sustained degradation (confirmation_bars consecutive)
        if self.bars_below_threshold >= self.confirmation_bars:
            self.bars_below_threshold = 0
            self.bars_since_last_trigger = 0
            return True

        return False
```

### Key Insight: Cooldown Period is Critical

Without a cooldown, the system can enter a retrain loop:
1. Performance drops
2. Retrain triggered
3. New model has no track record yet, metrics still bad from old model's history
4. Retrain triggered again immediately

**Minimum cooldown: 12 hours (144 bars).** This gives the new model enough predictions to establish its own track record.

### Adaptive Thresholds by Regime

During volatile regimes, accuracy naturally drops. The thresholds should adjust:

```python
REGIME_THRESHOLDS = {
    "trending": {"sharpe": 0.5, "accuracy": 0.53},
    "mean_reverting": {"sharpe": 0.3, "accuracy": 0.52},
    "volatile": {"sharpe": -0.5, "accuracy": 0.48},  # More lenient
}
```

---

## 3. Drift-Triggered Retraining

### Why Drift Detection is More Principled

Performance-based triggers are *reactive* -- they detect problems after they hurt you. Drift detection is *proactive* -- it detects distributional shifts that *will* hurt you, potentially before performance degrades.

### Population Stability Index (PSI)

PSI is the standard metric for detecting distribution shifts. Standard thresholds:

| PSI Value | Interpretation | Action |
|-----------|----------------|--------|
| < 0.10 | No significant shift | Continue |
| 0.10 - 0.20 | Moderate shift | Monitor closely |
| 0.20 - 0.25 | Significant shift | Consider retraining |
| > 0.25 | Major shift | Retrain immediately |

**For crypto, use tighter thresholds** because the signal-to-noise ratio is already low. A PSI of 0.15 on a key microstructure feature (OBI, trade flow imbalance) warrants retraining.

### What to Monitor for Drift

**Tier 1 (retrain on drift):**
- Order book imbalance distribution
- Trade flow imbalance distribution
- Realized volatility distribution
- Spread distribution
- Model prediction score distribution (output drift)

**Tier 2 (investigate on drift):**
- Volume patterns
- ROC distributions
- Funding rate levels
- Cross-market correlation strength

**Tier 3 (log but don't trigger):**
- Hour-of-day patterns (expected to shift)
- On-chain features (slow-moving by nature)

### Implementation: Feature Drift Monitor

```python
import numpy as np

def compute_psi(reference: np.ndarray, current: np.ndarray, n_bins: int = 10) -> float:
    """Population Stability Index between reference and current distributions."""
    # Use reference quantiles as bin edges for consistency
    quantiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(reference, quantiles)
    bin_edges[-1] = np.inf
    bin_edges[0] = -np.inf

    ref_counts = np.histogram(reference, bins=bin_edges)[0]
    cur_counts = np.histogram(current, bins=bin_edges)[0]

    # Add small epsilon to avoid division by zero
    ref_pct = (ref_counts + 1e-6) / len(reference)
    cur_pct = (cur_counts + 1e-6) / len(current)

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)


class DriftMonitor:
    """Monitor feature distributions for drift using PSI."""

    def __init__(
        self,
        feature_names: list[str],
        reference_window: int = 4032,   # 14 days of 5-min bars
        current_window: int = 576,      # 48 hours
        psi_threshold: float = 0.15,    # Tighter than standard 0.20 for crypto
        min_features_drifted: int = 3,  # Don't trigger on single feature drift
    ):
        self.feature_names = feature_names
        self.reference_window = reference_window
        self.current_window = current_window
        self.psi_threshold = psi_threshold
        self.min_features_drifted = min_features_drifted

    def check_drift(self, feature_matrix: np.ndarray) -> dict:
        """Check if enough features have drifted to warrant retraining.

        Args:
            feature_matrix: shape (n_bars, n_features), ordered chronologically

        Returns:
            dict with 'should_retrain', 'drifted_features', 'psi_values'
        """
        n_bars = feature_matrix.shape[0]
        if n_bars < self.reference_window + self.current_window:
            return {"should_retrain": False, "drifted_features": [], "psi_values": {}}

        reference = feature_matrix[-(self.reference_window + self.current_window):-self.current_window]
        current = feature_matrix[-self.current_window:]

        psi_values = {}
        drifted = []

        for i, name in enumerate(self.feature_names):
            psi = compute_psi(reference[:, i], current[:, i])
            psi_values[name] = psi
            if psi > self.psi_threshold:
                drifted.append(name)

        return {
            "should_retrain": len(drifted) >= self.min_features_drifted,
            "drifted_features": drifted,
            "psi_values": psi_values,
        }
```

### Combining Drift + Performance: The Decision Matrix

| Performance OK | Drift Detected | Action |
|---------------|----------------|--------|
| Yes | No | Continue (no action) |
| Yes | Yes | Schedule retrain at next window (not urgent) |
| No | No | Investigate (may be bad luck, not drift) |
| No | Yes | Retrain immediately (confirmed degradation from drift) |

---

## 4. Warm-Start vs Cold-Start Retraining

### LightGBM: init_model Warm-Start

**How it works:** `init_model` passes a previously trained Booster. New trees are added on top. Previous trees are FROZEN -- they are never modified. Training focuses on residuals from the existing trees.

```python
import lightgbm as lgb

def warm_start_lgbm(
    existing_model: lgb.Booster,
    new_X: np.ndarray,
    new_y: np.ndarray,
    additional_rounds: int = 50,
    params: dict | None = None,
) -> lgb.Booster:
    """Add trees to existing model using new data."""
    train_data = lgb.Dataset(new_X, label=new_y)

    default_params = {
        "objective": "multiclass",
        "num_class": 3,
        "learning_rate": 0.05,  # Lower LR for warm-start to avoid overshooting
        "num_leaves": 31,
        "verbose": -1,
    }
    if params:
        default_params.update(params)

    updated_model = lgb.train(
        default_params,
        train_data,
        num_boost_round=additional_rounds,
        init_model=existing_model,  # Warm start from existing trees
    )
    return updated_model
```

**When warm-start works well:**
- Data distribution is similar to training data (PSI < 0.15)
- You're adding recent data from the same regime
- You want fast updates (warm-start is 5-10x faster than full retrain)
- You've accumulated 2-4 hours of new data

**When warm-start degrades (must cold-retrain):**
- Regime has changed significantly (PSI > 0.25 on multiple features)
- Model has been warm-started more than 10-15 times without a cold retrain
- Accumulated tree count exceeds 2-3x the original model size
- Feature importance has drifted significantly from baseline
- Validation performance of warm-started model is worse than the last cold-retrained version

**The accumulation problem:** Each warm-start adds trees. After many warm-starts, the model becomes bloated with stale trees that were trained on old regimes. These old trees still contribute to predictions but may be counterproductive. The only fix is a periodic cold retrain.

**Recommended cadence:**
- Warm-start every 2-4 hours
- Cold retrain every 24 hours (resets the tree accumulation)
- Emergency cold retrain when drift monitor fires

### CatBoost: Similar Pattern

CatBoost supports `init_model` in the same way. The same warm-start vs cold-retrain logic applies. CatBoost's ordered boosting makes it slightly more robust to warm-start degradation than LightGBM because the training order matters less when continuing from an existing model.

```python
from catboost import CatBoostClassifier, Pool

def warm_start_catboost(
    existing_model: CatBoostClassifier,
    new_X: np.ndarray,
    new_y: np.ndarray,
    additional_iterations: int = 50,
) -> CatBoostClassifier:
    """Continue training CatBoost from existing model."""
    train_pool = Pool(new_X, label=new_y)

    new_model = CatBoostClassifier(
        iterations=additional_iterations,
        learning_rate=0.05,
        loss_function="MultiClass",
        verbose=0,
    )
    new_model.fit(
        train_pool,
        init_model=existing_model,
    )
    return new_model
```

### GRU: Fine-Tune vs Retrain From Scratch

**Fine-tuning (warm-start equivalent):**
- Load existing weights
- Train with LOWER learning rate (1/5 to 1/10 of original)
- Fewer epochs (5-10 vs 50-100 for full train)
- Optionally freeze early layers

```python
import torch

def fine_tune_gru(
    model: torch.nn.Module,
    new_dataloader: torch.utils.data.DataLoader,
    original_lr: float = 1e-3,
    fine_tune_lr_factor: float = 0.1,
    epochs: int = 5,
    gradient_clip: float = 1.0,
) -> torch.nn.Module:
    """Fine-tune GRU with low learning rate to avoid catastrophic forgetting."""
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=original_lr * fine_tune_lr_factor,  # 1/10th of original LR
    )
    criterion = torch.nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        for X_batch, y_batch in new_dataloader:
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()

    return model
```

**Catastrophic forgetting risk for GRU:**
- HIGH if fine-tuning on small dataset (< 500 sequences)
- MEDIUM if fine-tuning on single-regime data (model forgets other regimes)
- LOW if fine-tuning on representative data with low LR

**Mitigation strategies:**
1. **Elastic Weight Consolidation (EWC):** Add penalty for changing weights that were important for old tasks. Effective but adds complexity.
2. **Replay buffer:** Mix 20-30% old data with new data during fine-tuning. Simple and effective.
3. **Low learning rate:** The simplest approach. 1/10th of original LR. Works well for gradual drift.

```python
def fine_tune_with_replay(
    model: torch.nn.Module,
    new_data: torch.utils.data.Dataset,
    replay_buffer: torch.utils.data.Dataset,
    replay_fraction: float = 0.3,
    epochs: int = 5,
) -> torch.nn.Module:
    """Fine-tune with replay buffer to prevent catastrophic forgetting."""
    # Mix old and new data
    replay_size = int(len(new_data) * replay_fraction / (1 - replay_fraction))
    replay_subset = torch.utils.data.Subset(
        replay_buffer,
        indices=torch.randperm(len(replay_buffer))[:replay_size].tolist(),
    )
    combined = torch.utils.data.ConcatDataset([new_data, replay_subset])
    # ... train on combined dataset
    return model
```

**Recommended cadence for GRU:**
- Fine-tune (low LR + replay) every 6-8 hours
- Full retrain from scratch every 48-72 hours
- Emergency retrain when drift monitor fires with PSI > 0.25

### Decision Matrix: Warm vs Cold

| Condition | LightGBM/CatBoost | GRU |
|-----------|-------------------|-----|
| Routine update (2-4h) | Warm-start (+50 trees) | Skip (too frequent) |
| Routine update (6-8h) | Warm-start (+50 trees) | Fine-tune (low LR, 5 epochs) |
| Daily reset | Cold retrain | Cold retrain |
| Drift detected (PSI > 0.20) | Cold retrain | Cold retrain |
| Performance degraded | Cold retrain | Cold retrain with replay |
| Regime change | Cold retrain | Cold retrain from scratch |

---

## 5. Training Data Window for Retraining

### Sliding Window vs Expanding Window

**The research consensus for crypto (from RESEARCH_SYNTHESIS.md): Sliding window.**

Crypto is non-stationary. Old data from a different regime actively hurts predictions. The PLAN.md already identifies this: "Sliding window, not expanding -- crypto non-stationarity means old data hurts more than helps."

### Optimal Window Sizes

| Model | Training Window | Why |
|-------|----------------|-----|
| LightGBM/CatBoost (warm-start) | Last 2-4h of data (new data only) | Warm-start just adds trees for recent residuals |
| LightGBM/CatBoost (cold retrain) | 7-14 days (2016-4032 bars) | Enough to capture multiple regime transitions |
| GRU (fine-tune) | Last 6-8h of sequences | Recent temporal patterns |
| GRU (cold retrain) | 14-30 days (4032-8640 bars) | Needs more data for stable gradient estimation |
| Stacking meta-learner | 7 days of OOF predictions | Needs diverse base model predictions |
| Regime detector (HMM) | 7 days sliding | Per PLAN.md: refit every 4h on 7-day window |

### Should You Include Data From ALL Regimes?

**For cold retrains: YES, but weighted.**

If you only train on recent regime data, the model will fail catastrophically when the regime changes. Include all regimes, but weight recent data more heavily.

```python
def compute_sample_weights(
    timestamps: np.ndarray,
    regime_labels: np.ndarray,
    current_regime: int,
    recency_halflife_bars: int = 1440,  # 5 days
    regime_boost: float = 1.5,
) -> np.ndarray:
    """Weight samples by recency and regime relevance."""
    n = len(timestamps)
    # Exponential recency weighting
    bars_ago = np.arange(n, 0, -1)
    recency_weights = np.exp(-np.log(2) * bars_ago / recency_halflife_bars)

    # Boost samples from current regime
    regime_weights = np.where(regime_labels == current_regime, regime_boost, 1.0)

    combined = recency_weights * regime_weights
    return combined / combined.sum() * n  # Normalize to sum = n
```

**For warm-starts: Recent data only (last window), no regime mixing needed.**

The warm-start is specifically adapting to current conditions. Including old regime data would dilute the signal.

### Window Size Sensitivity Analysis

Run this during initial backtesting to find the optimal window:

```python
WINDOW_SIZES_DAYS = [3, 5, 7, 10, 14, 21, 30]

def window_sensitivity_analysis(data, model_class, target):
    """Test different training window sizes via walk-forward."""
    results = {}
    for window_days in WINDOW_SIZES_DAYS:
        window_bars = window_days * 288
        # Run walk-forward with this window size
        sharpe = walk_forward_validate(data, model_class, target, train_size=window_bars)
        results[window_days] = sharpe
    return results
    # Typically: 7-14 day window wins for 5-min crypto
```

---

## 6. Feature Selection Stability During Retraining

### Should the Feature Set Change?

**Recommendation: Keep feature set FIXED during normal retrains. Re-evaluate monthly.**

Changing features during retraining creates several problems:
1. The stacking meta-learner expects consistent input dimensions
2. Historical comparison of model versions becomes impossible
3. Conformal prediction calibration breaks if features change
4. Regime detector may behave differently with different features

### Handling Feature Decay

Some features will become less useful over time (feature decay). The order book imbalance feature might lose predictive power as more traders use it (alpha decay).

**Monthly feature audit process:**

```python
def monthly_feature_audit(
    model: lgb.Booster,
    feature_names: list[str],
    recent_data: np.ndarray,
    recent_labels: np.ndarray,
    importance_threshold: float = 0.01,  # Features below 1% of top feature
) -> dict:
    """Identify features that have decayed in importance."""
    # Current feature importance
    importance = model.feature_importance(importance_type="gain")
    importance_pct = importance / importance.max()

    audit_results = {
        "decayed_features": [],
        "stable_features": [],
        "importance_ranking": {},
    }

    for i, name in enumerate(feature_names):
        audit_results["importance_ranking"][name] = float(importance_pct[i])
        if importance_pct[i] < importance_threshold:
            audit_results["decayed_features"].append(name)
        else:
            audit_results["stable_features"].append(name)

    return audit_results
```

### When Features Must Change

If a feature becomes *unavailable* (API shutdown, exchange change), or a new high-value feature is discovered:

1. **Add the new feature / remove the dead one**
2. **Cold retrain ALL models** (LightGBM, CatBoost, GRU) from scratch
3. **Retrain the stacking meta-learner** with new OOF predictions
4. **Recalibrate** isotonic regression and conformal prediction
5. **Run validation** against the previous model before deployment (Section 8)
6. **Reset drift monitoring baselines** for the changed features

This is essentially a minor version change of the model, not a routine retrain.

### Dual-Model Architecture for Feature Stability

Per the EvidentlyAI research, consider a dual approach:
- **Primary model**: Uses only stable, high-importance features (top 15)
- **Correction model**: Uses volatile/experimental features to adjust predictions

This way, feature instability in the correction model doesn't destabilize the primary signal.

---

## 7. Hyperparameter Re-Optimization

### The Cost Problem

Running Optuna with 50+ trials takes 1-4 hours depending on data size. Doing this at every retrain is wasteful and can actually hurt performance (overfitting hyperparameters to recent data).

### When to Re-Optimize vs Just Retrain

| Condition | Action |
|-----------|--------|
| Routine warm-start (every 2-4h) | Same hyperparameters. Never re-optimize. |
| Routine cold retrain (daily) | Same hyperparameters. Just retrain with fresh data. |
| Performance degraded for > 48h despite retraining | Re-optimize (the hyperparams may be stale) |
| Regime has fundamentally changed (PSI > 0.30 sustained) | Re-optimize with data from new regime |
| Monthly scheduled review | Light re-optimization (20 trials, warm-start from previous best) |
| Feature set changed | Full re-optimization (new features need new hyperparams) |

### Efficient Re-Optimization Pattern

Instead of cold-starting Optuna every time, use Optuna's study persistence to warm-start from previous best:

```python
import optuna

def efficient_reoptimize(
    study_name: str,
    storage: str,  # e.g., "sqlite:///optuna_studies.db"
    objective_fn,
    n_trials: int = 20,  # Fewer trials for re-optimization
) -> dict:
    """Re-optimize starting from previous best, with fewer trials."""
    # Load existing study (contains history of previous optimizations)
    study = optuna.load_study(
        study_name=study_name,
        storage=storage,
    )

    # Add new trials -- Optuna's TPE sampler will use history
    study.optimize(objective_fn, n_trials=n_trials)

    return study.best_params


def create_objective_with_deflated_sharpe(
    train_data,
    val_data,
    n_total_trials: int,
) -> callable:
    """Objective that accounts for multiple testing via Deflated Sharpe Ratio."""
    def objective(trial):
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 250),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.001, 25.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.001, 25.0, log=True),
        }
        # Train and evaluate...
        sharpe = walk_forward_sharpe(train_data, val_data, params)
        return sharpe  # Optuna maximizes by default with direction="maximize"

    return objective
```

### Hyperparameter Stability Check

Before accepting new hyperparameters, verify they're meaningfully different from current ones:

```python
def should_update_hyperparams(
    current_params: dict,
    new_params: dict,
    current_sharpe: float,
    new_sharpe: float,
    min_sharpe_improvement: float = 0.15,  # Must improve Sharpe by at least 0.15
) -> bool:
    """Only update hyperparams if improvement is significant."""
    if new_sharpe - current_sharpe < min_sharpe_improvement:
        return False  # Not enough improvement to justify the change risk

    return True
```

---

## 8. Model Validation Before Deployment

### The Cardinal Rule: Never Deploy Without Validation

After retraining, the new model must prove it's at least as good as the current model before going live.

### Shadow Mode (Champion-Challenger)

The most reliable approach: run both models simultaneously, but only act on the champion's predictions.

```python
import structlog

logger = structlog.get_logger()

class ChampionChallenger:
    """Run champion and challenger models side-by-side."""

    def __init__(
        self,
        champion_model,
        min_shadow_bars: int = 144,  # 12 hours minimum shadow period
    ):
        self.champion = champion_model
        self.challenger = None
        self.challenger_predictions: list[float] = []
        self.challenger_actuals: list[float] = []
        self.champion_predictions: list[float] = []
        self.champion_actuals: list[float] = []
        self.min_shadow_bars = min_shadow_bars
        self.shadow_bars_elapsed = 0

    def set_challenger(self, new_model) -> None:
        """Register a new challenger model for shadow evaluation."""
        self.challenger = new_model
        self.challenger_predictions = []
        self.challenger_actuals = []
        self.champion_predictions = []
        self.champion_actuals = []
        self.shadow_bars_elapsed = 0
        logger.info("challenger_registered", min_shadow_bars=self.min_shadow_bars)

    def predict(self, features):
        """Get prediction from champion. Log challenger prediction for comparison."""
        champion_pred = self.champion.predict(features)

        if self.challenger is not None:
            challenger_pred = self.challenger.predict(features)
            self.challenger_predictions.append(challenger_pred)
            self.champion_predictions.append(champion_pred)
            self.shadow_bars_elapsed += 1

        return champion_pred  # Always use champion for live decisions

    def record_actual(self, actual: float) -> None:
        """Record actual outcome for both models."""
        if self.challenger is not None and len(self.challenger_predictions) > len(self.challenger_actuals):
            self.challenger_actuals.append(actual)
            self.champion_actuals.append(actual)

    def should_promote_challenger(self) -> bool:
        """Evaluate if challenger should replace champion."""
        if self.challenger is None:
            return False
        if self.shadow_bars_elapsed < self.min_shadow_bars:
            return False

        n = len(self.challenger_actuals)
        if n < self.min_shadow_bars:
            return False

        # Compare directional accuracy
        ch_pred = np.array(self.champion_predictions[:n])
        cl_pred = np.array(self.challenger_predictions[:n])
        actuals = np.array(self.challenger_actuals)

        ch_accuracy = (np.sign(ch_pred) == np.sign(actuals)).mean()
        cl_accuracy = (np.sign(cl_pred) == np.sign(actuals)).mean()

        # Challenger must be at least as good (not strictly better for routine retrains)
        # For cold retrains, require no degradation
        # For major changes (feature set change), require improvement
        if cl_accuracy >= ch_accuracy - 0.01:  # Allow 1% margin
            logger.info(
                "challenger_promoted",
                champion_accuracy=ch_accuracy,
                challenger_accuracy=cl_accuracy,
                shadow_bars=n,
            )
            return True

        logger.warning(
            "challenger_rejected",
            champion_accuracy=ch_accuracy,
            challenger_accuracy=cl_accuracy,
            shadow_bars=n,
        )
        return False

    def promote_challenger(self) -> None:
        """Replace champion with challenger."""
        self.champion = self.challenger
        self.challenger = None
```

### Statistical Significance for Model Comparison

For warm-starts (routine updates), statistical significance is overkill -- just require non-degradation. For cold retrains or major changes, use a paired sign test:

```python
from scipy import stats

def paired_model_comparison(
    model_a_predictions: np.ndarray,
    model_b_predictions: np.ndarray,
    actuals: np.ndarray,
    alpha: float = 0.10,  # 10% significance (one-sided)
) -> dict:
    """Compare two models using paired sign test on directional accuracy."""
    a_correct = np.sign(model_a_predictions) == np.sign(actuals)
    b_correct = np.sign(model_b_predictions) == np.sign(actuals)

    # Cases where B is right and A is wrong
    b_better = np.sum(b_correct & ~a_correct)
    a_better = np.sum(a_correct & ~b_correct)
    n_discordant = b_better + a_better

    if n_discordant == 0:
        return {"significant": False, "p_value": 1.0, "winner": "tie"}

    # Binomial test: is B better more often than chance?
    p_value = stats.binom_test(b_better, n_discordant, 0.5, alternative="greater")

    return {
        "significant": p_value < alpha,
        "p_value": float(p_value),
        "winner": "b" if p_value < alpha else "none",
        "b_better_count": int(b_better),
        "a_better_count": int(a_better),
    }
```

### Validation Tiers by Retrain Type

| Retrain Type | Validation Required | Shadow Period |
|-------------|---------------------|---------------|
| Warm-start (2-4h) | None (auto-deploy) | 0 bars |
| Cold retrain (daily) | Non-degradation check on holdout | 72 bars (6h) shadow |
| Hyperparameter change | Statistical significance test | 144 bars (12h) shadow |
| Feature set change | Full shadow + significance test | 288 bars (24h) shadow |
| Architecture change | Full paper trading period | 2016 bars (7 days) shadow |

---

## 9. Rollback Procedures

### Model Version Management

Every model artifact must be versioned and stored:

```python
import json
import shutil
from pathlib import Path
from datetime import datetime, timezone

class ModelRegistry:
    """Simple file-based model registry with rollback support."""

    def __init__(self, registry_dir: str = "models/registry"):
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)

    def register_model(
        self,
        model_name: str,
        model_artifact_path: str,
        metadata: dict,
    ) -> str:
        """Register a new model version."""
        version_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        version_dir = self.registry_dir / model_name / version_id
        version_dir.mkdir(parents=True, exist_ok=True)

        # Copy artifact
        shutil.copy2(model_artifact_path, version_dir / "model.bin")

        # Save metadata
        metadata["version_id"] = version_id
        metadata["registered_at"] = datetime.now(timezone.utc).isoformat()
        metadata["status"] = "staging"

        with open(version_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        return version_id

    def promote_to_production(self, model_name: str, version_id: str) -> None:
        """Promote a staged model to production."""
        # Demote current production model
        current_prod = self._get_production_version(model_name)
        if current_prod:
            self._update_status(model_name, current_prod, "archived")

        self._update_status(model_name, version_id, "production")

    def rollback(self, model_name: str) -> str | None:
        """Roll back to the previous production model."""
        versions = self._list_versions(model_name)
        archived = [v for v in versions if v["status"] == "archived"]
        if not archived:
            return None

        # Most recent archived version
        prev_version = sorted(archived, key=lambda x: x["version_id"])[-1]
        self.promote_to_production(model_name, prev_version["version_id"])
        return prev_version["version_id"]

    def get_production_model_path(self, model_name: str) -> Path | None:
        """Get path to current production model artifact."""
        version_id = self._get_production_version(model_name)
        if version_id:
            return self.registry_dir / model_name / version_id / "model.bin"
        return None

    def _get_production_version(self, model_name: str) -> str | None:
        versions = self._list_versions(model_name)
        prod = [v for v in versions if v["status"] == "production"]
        return prod[0]["version_id"] if prod else None

    def _list_versions(self, model_name: str) -> list[dict]:
        model_dir = self.registry_dir / model_name
        if not model_dir.exists():
            return []
        versions = []
        for version_dir in sorted(model_dir.iterdir()):
            meta_path = version_dir / "metadata.json"
            if meta_path.exists():
                with open(meta_path) as f:
                    versions.append(json.load(f))
        return versions

    def _update_status(self, model_name: str, version_id: str, status: str) -> None:
        meta_path = self.registry_dir / model_name / version_id / "metadata.json"
        with open(meta_path) as f:
            meta = json.load(f)
        meta["status"] = status
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
```

### Automatic Rollback Triggers

```python
class AutoRollback:
    """Automatically rollback if new model performs poorly."""

    def __init__(
        self,
        registry: ModelRegistry,
        model_name: str,
        max_consecutive_losses: int = 12,    # 1 hour of consecutive losses
        min_bars_before_rollback: int = 36,  # 3 hours minimum trial period
        sharpe_floor: float = -1.0,          # Emergency rollback threshold
    ):
        self.registry = registry
        self.model_name = model_name
        self.max_consecutive_losses = max_consecutive_losses
        self.min_bars_before_rollback = min_bars_before_rollback
        self.sharpe_floor = sharpe_floor
        self.consecutive_losses = 0
        self.bars_since_deploy = 0
        self.returns_since_deploy: list[float] = []

    def update(self, trade_return: float) -> bool:
        """Returns True if rollback should be executed."""
        self.bars_since_deploy += 1
        self.returns_since_deploy.append(trade_return)

        if self.bars_since_deploy < self.min_bars_before_rollback:
            return False

        # Check consecutive losses
        if trade_return < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

        if self.consecutive_losses >= self.max_consecutive_losses:
            return True

        # Check rolling Sharpe floor
        if len(self.returns_since_deploy) >= 72:  # 6 hours
            returns = np.array(self.returns_since_deploy[-72:])
            sharpe = returns.mean() / max(returns.std(), 1e-10) * np.sqrt(105_120)
            if sharpe < self.sharpe_floor:
                return True

        return False

    def execute_rollback(self) -> str | None:
        """Execute the rollback and return the version rolled back to."""
        version = self.registry.rollback(self.model_name)
        if version:
            self.consecutive_losses = 0
            self.bars_since_deploy = 0
            self.returns_since_deploy = []
        return version
```

### Retention Policy

Keep the last N model versions per model type:
- **LightGBM**: Last 7 daily cold-retrain versions
- **CatBoost**: Last 7 daily cold-retrain versions
- **GRU**: Last 5 cold-retrain versions (larger artifacts)
- **Meta-learner**: Last 4 weekly versions

Warm-start checkpoints don't need permanent storage -- only keep the most recent warm-start and the last cold-retrain.

---

## 10. Retraining Pipeline Automation

### Architecture: Event-Driven with Scheduled Fallback

For a single-server deployment (per RESEARCH_SYNTHESIS.md: $40-100/mo Hetzner), heavy orchestration tools like Airflow are overkill. Prefect is a reasonable middle ground, but for a single pipeline on one machine, a simple Python-native solution is most appropriate.

### Recommended: asyncio-based Pipeline Manager

```python
import asyncio
import structlog
from dataclasses import dataclass
from enum import Enum

logger = structlog.get_logger()

class RetrainType(Enum):
    WARMSTART_TREE = "warmstart_tree"
    FINETUNE_GRU = "finetune_gru"
    COLD_RETRAIN_ALL = "cold_retrain_all"
    HYPERPARAM_SEARCH = "hyperparam_search"
    META_LEARNER_RETRAIN = "meta_learner_retrain"


@dataclass
class RetrainJob:
    retrain_type: RetrainType
    trigger: str  # "schedule", "drift", "performance"
    priority: int  # Lower = higher priority


class RetrainPipeline:
    """Orchestrates model retraining with both scheduled and event-driven triggers."""

    def __init__(self, model_registry: ModelRegistry):
        self.registry = model_registry
        self.job_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self._is_retraining = False
        self._lock = asyncio.Lock()

    async def schedule_loop(self) -> None:
        """Background loop that submits scheduled retrain jobs."""
        while True:
            # LightGBM/CatBoost warm-start every 2 hours
            await self.job_queue.put((1, RetrainJob(
                RetrainType.WARMSTART_TREE, "schedule", priority=1
            )))
            await asyncio.sleep(2 * 3600)

    async def schedule_gru_loop(self) -> None:
        """GRU fine-tune every 6 hours."""
        while True:
            await asyncio.sleep(6 * 3600)
            await self.job_queue.put((2, RetrainJob(
                RetrainType.FINETUNE_GRU, "schedule", priority=2
            )))

    async def schedule_cold_retrain_loop(self) -> None:
        """Full cold retrain every 24 hours."""
        while True:
            await asyncio.sleep(24 * 3600)
            await self.job_queue.put((0, RetrainJob(
                RetrainType.COLD_RETRAIN_ALL, "schedule", priority=0
            )))

    async def submit_event_triggered(self, retrain_type: RetrainType, trigger: str) -> None:
        """Submit an event-triggered retrain job (from drift or performance monitor)."""
        await self.job_queue.put((0, RetrainJob(
            retrain_type, trigger, priority=0  # High priority for event-triggered
        )))

    async def worker(self) -> None:
        """Process retrain jobs from the queue."""
        while True:
            priority, job = await self.job_queue.get()

            async with self._lock:  # Only one retrain at a time
                self._is_retraining = True
                try:
                    logger.info(
                        "retrain_started",
                        type=job.retrain_type.value,
                        trigger=job.trigger,
                    )
                    await self._execute_retrain(job)
                    logger.info("retrain_completed", type=job.retrain_type.value)
                except Exception:
                    logger.exception("retrain_failed", type=job.retrain_type.value)
                finally:
                    self._is_retraining = False
                    self.job_queue.task_done()

    async def _execute_retrain(self, job: RetrainJob) -> None:
        """Execute the actual retraining in a thread pool to avoid blocking."""
        loop = asyncio.get_event_loop()
        # Run CPU-intensive training in thread pool
        await loop.run_in_executor(None, self._retrain_sync, job)

    def _retrain_sync(self, job: RetrainJob) -> None:
        """Synchronous retraining logic (runs in thread pool)."""
        if job.retrain_type == RetrainType.WARMSTART_TREE:
            self._warmstart_trees()
        elif job.retrain_type == RetrainType.FINETUNE_GRU:
            self._finetune_gru()
        elif job.retrain_type == RetrainType.COLD_RETRAIN_ALL:
            self._cold_retrain_all()
        elif job.retrain_type == RetrainType.HYPERPARAM_SEARCH:
            self._hyperparameter_search()
        elif job.retrain_type == RetrainType.META_LEARNER_RETRAIN:
            self._retrain_meta_learner()

    def _warmstart_trees(self) -> None:
        """Steps: collect data -> warm-start LightGBM -> warm-start CatBoost -> validate -> deploy"""
        pass  # Implementation in model modules

    def _finetune_gru(self) -> None:
        """Steps: collect sequences -> fine-tune with replay -> validate -> deploy"""
        pass

    def _cold_retrain_all(self) -> None:
        """Steps: collect full window -> retrain LightGBM -> retrain CatBoost ->
        retrain GRU -> retrain meta-learner -> shadow validate -> deploy"""
        pass

    def _hyperparameter_search(self) -> None:
        """Steps: collect data -> Optuna search -> validate -> update params"""
        pass

    def _retrain_meta_learner(self) -> None:
        """Steps: generate OOF predictions -> retrain stacking -> validate -> deploy"""
        pass
```

### Pipeline Steps Per Retrain Type

**Warm-start trees (every 2-4h):**
1. Query last 2-4h of features + labels from DB
2. Warm-start LightGBM (`init_model`)
3. Warm-start CatBoost (`init_model`)
4. Quick validation on last 30 min holdout
5. Deploy (no shadow needed for warm-starts)

**Fine-tune GRU (every 6-8h):**
1. Query last 6-8h of sequences + replay buffer (20% from last 7 days)
2. Fine-tune with low LR (5 epochs)
3. Validate on last 1h holdout
4. Deploy if non-degrading

**Cold retrain (daily):**
1. Query full training window (14 days)
2. Compute sample weights (recency + regime)
3. Train LightGBM from scratch
4. Train CatBoost from scratch
5. Train GRU from scratch
6. Generate OOF predictions for meta-learner
7. Train stacking meta-learner
8. Calibrate (isotonic regression)
9. Shadow validate for 6h
10. Promote if non-degrading, else keep current

**Hyperparameter search (weekly):**
1. Query 14 days of data
2. Run Optuna (20 trials, warm-start from previous study)
3. Compare new params vs current params
4. If improvement > 0.15 Sharpe: accept new params
5. Cold retrain with new params
6. Shadow validate for 12h

### Why Not Airflow/Prefect?

For a single-server crypto prediction system:

| Tool | Pros | Cons for This System |
|------|------|---------------------|
| Airflow | Mature, battle-tested | Heavy (needs DB, webserver, scheduler). Overkill for 1 pipeline. |
| Prefect | Lighter, Python-native | Still adds dependency. Cloud features unnecessary. |
| systemd timer | Zero dependencies | No retry logic, no DAG awareness, poor observability |
| **asyncio + structlog** | **Already in stack, zero new dependencies, full control** | Must implement retry/monitoring yourself |

**Recommendation: asyncio pipeline manager (as shown above) with structlog for observability.** Add Prefect later if the system grows to multiple machines or pipelines.

---

## 11. Ensemble Model Retraining Coordination

### The Coordination Problem

LightGBM, CatBoost, and GRU have different optimal retraining schedules:

| Component | Warm-Start | Cold Retrain | Why Different |
|-----------|------------|-------------|---------------|
| LightGBM | Every 2h | Every 24h | Fast to train, adapts quickly |
| CatBoost | Every 2h | Every 24h | Similar to LightGBM |
| GRU | Every 6h (fine-tune) | Every 48-72h | Expensive, slower to adapt |
| Meta-learner | N/A | Every 24h (with cold retrain) | Needs fresh OOF from base models |
| Calibrator | N/A | Every 24h | Needs new predictions to calibrate |

### The Meta-Learner Dependency

The stacking meta-learner (logistic regression) takes base model OOF predictions as input. When base models are warm-started, the meta-learner's input distribution shifts slightly. This is usually tolerable for warm-starts but requires recalibration at cold-retrain time.

**Rule: The meta-learner is ONLY retrained during cold retrains, never during warm-starts.**

### Coordination Protocol

```
T+0h:   Warm-start LightGBM + CatBoost (meta-learner unchanged)
T+2h:   Warm-start LightGBM + CatBoost (meta-learner unchanged)
T+4h:   Warm-start LightGBM + CatBoost (meta-learner unchanged)
T+6h:   Warm-start LightGBM + CatBoost + Fine-tune GRU (meta-learner unchanged)
T+8h:   Warm-start LightGBM + CatBoost (meta-learner unchanged)
...
T+24h:  COLD RETRAIN ALL:
         1. Retrain LightGBM (scratch)
         2. Retrain CatBoost (scratch)
         3. Retrain GRU (scratch)
         4. Generate OOF predictions from all three
         5. Retrain meta-learner on new OOF
         6. Recalibrate isotonic regression
         7. Shadow validate ensemble
         8. Deploy if passing
```

### Handling Partial Retraining Failures

If one base model's retrain fails (e.g., GRU training diverges):

1. **Keep the previous version** of the failed model
2. **Still retrain other models** -- partial update is better than no update
3. **Skip meta-learner retrain** if any base model failed (OOF would be inconsistent)
4. **Log and alert** -- the failed model needs investigation
5. **Retry once** after 1 hour. If it fails again, keep previous and cold-retrain at next scheduled time.

```python
async def coordinated_cold_retrain(self) -> dict:
    """Cold retrain all models with failure handling."""
    results = {"lgbm": None, "catboost": None, "gru": None, "meta": None}

    # Phase 1: Retrain base models (can be parallel)
    try:
        results["lgbm"] = await self._cold_retrain_lgbm()
    except Exception:
        logger.exception("lgbm_cold_retrain_failed")

    try:
        results["catboost"] = await self._cold_retrain_catboost()
    except Exception:
        logger.exception("catboost_cold_retrain_failed")

    try:
        results["gru"] = await self._cold_retrain_gru()
    except Exception:
        logger.exception("gru_cold_retrain_failed")

    # Phase 2: Meta-learner (only if ALL base models succeeded)
    all_succeeded = all(r is not None for r in [results["lgbm"], results["catboost"], results["gru"]])

    if all_succeeded:
        try:
            results["meta"] = await self._retrain_meta_learner(
                results["lgbm"], results["catboost"], results["gru"]
            )
        except Exception:
            logger.exception("meta_learner_retrain_failed")
    else:
        logger.warning(
            "meta_learner_skipped",
            reason="base_model_failure",
            failed_models=[k for k, v in results.items() if v is None],
        )

    return results
```

### Meta-Learner Staleness

Between cold retrains, the meta-learner weights may become slightly stale as base models drift via warm-starts. To mitigate:

1. **Use simple meta-learner** (logistic regression) that is robust to small input shifts
2. **Monitor meta-learner prediction distribution** -- if it shifts significantly, trigger early cold retrain
3. **Never use a complex meta-learner** (neural net, GBM) for stacking -- it would be too sensitive to base model drift

---

## 12. Resource Management During Retraining

### The Contention Problem

On a single server ($40-100/mo Hetzner, typically 4-8 cores, 16-32 GB RAM):
- Live prediction loop needs to run uninterrupted
- Data ingestion (WebSocket streams) cannot lag
- Retraining is CPU-intensive (LightGBM) and memory-intensive (GRU)

### Architecture: Separate Process for Retraining

```
┌─────────────────────────────────────┐
│         MAIN PROCESS                │
│  - asyncio event loop               │
│  - Data ingestion (WebSockets)      │
│  - Feature computation              │
│  - Model inference (lightweight)    │
│  - API server                       │
│  - Drift/performance monitoring     │
│  Priority: HIGH (nice -10)          │
└─────────────┬───────────────────────┘
              │ IPC (file/socket)
              │ (submit job, receive model artifact)
┌─────────────┴───────────────────────┐
│       RETRAIN PROCESS               │
│  - Spawned via multiprocessing      │
│  - Reads training data from DB      │
│  - Trains model                     │
│  - Saves artifact to disk           │
│  - Signals main process on complete │
│  Priority: LOW (nice 10)            │
│  CPU affinity: cores 2-N            │
└─────────────────────────────────────┘
```

### Implementation: Process Isolation

```python
import multiprocessing as mp
import os
import signal

def retrain_worker(
    job_config: dict,
    result_path: str,
    done_event: mp.Event,
) -> None:
    """Run retraining in a separate process with resource limits."""
    # Lower priority
    os.nice(10)

    # Set CPU affinity (leave core 0 for main process)
    try:
        os.sched_setaffinity(0, set(range(1, mp.cpu_count())))
    except (AttributeError, OSError):
        pass  # Not available on macOS

    try:
        # Limit memory for GRU training
        if job_config.get("type") == "gru":
            import resource
            # Limit to 8GB for GRU training
            mem_limit = 8 * 1024 * 1024 * 1024
            try:
                resource.setrlimit(resource.RLIMIT_AS, (mem_limit, mem_limit))
            except (ValueError, OSError):
                pass  # Not all platforms support this

        # Execute retraining
        model = _do_retrain(job_config)
        _save_model(model, result_path)

    except Exception as e:
        _save_error(str(e), result_path + ".error")
    finally:
        done_event.set()


class RetrainProcessManager:
    """Manage retraining in separate processes."""

    def __init__(self, max_concurrent: int = 1):
        self.max_concurrent = max_concurrent
        self._active_processes: list[mp.Process] = []

    def submit_retrain(self, job_config: dict, result_path: str) -> mp.Event:
        """Submit a retrain job to run in a separate process."""
        # Clean up finished processes
        self._active_processes = [p for p in self._active_processes if p.is_alive()]

        if len(self._active_processes) >= self.max_concurrent:
            raise RuntimeError("Max concurrent retraining jobs reached")

        done_event = mp.Event()
        process = mp.Process(
            target=retrain_worker,
            args=(job_config, result_path, done_event),
            daemon=True,
        )
        process.start()
        self._active_processes.append(process)

        return done_event

    def cancel_all(self) -> None:
        """Cancel all running retrain jobs (e.g., on shutdown)."""
        for p in self._active_processes:
            if p.is_alive():
                p.terminate()
                p.join(timeout=5)
                if p.is_alive():
                    p.kill()
```

### Memory Budget

| Component | Memory Budget | Notes |
|-----------|--------------|-------|
| Main process (inference) | 2-4 GB | Feature buffers, loaded models, WebSocket state |
| LightGBM retraining | 2-4 GB | Dataset + tree building |
| CatBoost retraining | 2-4 GB | Similar to LightGBM |
| GRU retraining | 4-8 GB | PyTorch tensors, gradient state |
| **Total needed** | **8-16 GB** | Never retrain LightGBM + GRU simultaneously |

### Scheduling to Avoid Contention

```python
# Never run these simultaneously:
EXCLUSIVE_GROUPS = [
    {"cold_retrain_lgbm", "cold_retrain_catboost"},  # Both use heavy CPU
    {"cold_retrain_gru", "cold_retrain_lgbm"},        # GRU needs GPU/memory
    {"cold_retrain_gru", "cold_retrain_catboost"},
    {"hyperparam_search", "cold_retrain_gru"},         # Both very expensive
]

# Safe to run simultaneously:
# - warm_start_lgbm + warm_start_catboost (both lightweight)
# - Any warm-start + live inference (warm-start is fast)
```

### LightGBM Thread Management

LightGBM uses all available threads by default. During retraining, limit it:

```python
# In retrain process
params = {
    "num_threads": 2,  # Leave cores for main process
    # ... other params
}
```

### Hot-Swap Model Loading

After retraining completes in the separate process, the main process needs to load the new model without interrupting inference:

```python
import threading

class HotSwappableModel:
    """Thread-safe model that can be swapped without downtime."""

    def __init__(self, initial_model):
        self._model = initial_model
        self._lock = threading.RLock()

    def predict(self, features):
        with self._lock:
            return self._model.predict(features)

    def swap(self, new_model) -> None:
        """Atomically swap the model. Old model is garbage collected."""
        with self._lock:
            self._model = new_model
```

---

## Summary: Recommended Retraining Configuration for ep2-crypto

### Default Configuration

```python
RETRAIN_CONFIG = {
    # --- Scheduled Retraining ---
    "lgbm_warmstart_interval_hours": 2,
    "catboost_warmstart_interval_hours": 2,
    "gru_finetune_interval_hours": 6,
    "cold_retrain_interval_hours": 24,
    "hyperparam_search_interval_days": 7,

    # --- Performance Triggers ---
    "rolling_sharpe_window_bars": 576,       # 48h
    "rolling_sharpe_threshold": 0.0,
    "rolling_accuracy_window_bars": 288,     # 24h
    "rolling_accuracy_threshold": 0.50,
    "confirmation_bars": 72,                 # 6h sustained degradation
    "cooldown_bars": 144,                    # 12h between triggers

    # --- Drift Triggers ---
    "psi_reference_window_bars": 4032,       # 14 days
    "psi_current_window_bars": 576,          # 48h
    "psi_threshold": 0.15,
    "min_features_drifted": 3,

    # --- Warm-Start Parameters ---
    "lgbm_warmstart_rounds": 50,
    "lgbm_warmstart_lr": 0.05,
    "catboost_warmstart_iterations": 50,
    "gru_finetune_epochs": 5,
    "gru_finetune_lr_factor": 0.1,
    "gru_replay_fraction": 0.3,

    # --- Training Windows ---
    "tree_cold_retrain_window_days": 14,
    "gru_cold_retrain_window_days": 21,
    "meta_learner_window_days": 7,

    # --- Validation ---
    "warmstart_shadow_bars": 0,              # No shadow for warm-starts
    "cold_retrain_shadow_bars": 72,          # 6h shadow
    "hyperparam_change_shadow_bars": 144,    # 12h shadow

    # --- Rollback ---
    "max_consecutive_losses_before_rollback": 12,
    "min_bars_before_rollback": 36,
    "sharpe_floor": -1.0,
    "model_versions_to_keep": 7,

    # --- Resource Management ---
    "retrain_process_nice": 10,
    "lgbm_retrain_threads": 2,
    "max_concurrent_retrains": 1,
}
```

### Priority Order for Implementation

1. **Fixed schedule** (simplest, implement first)
2. **Model registry + rollback** (safety net before anything else)
3. **Champion-challenger shadow validation** (prevent bad models from going live)
4. **Process isolation** (prevent retraining from affecting live inference)
5. **Performance monitoring triggers** (reactive safety)
6. **Drift monitoring triggers** (proactive detection)
7. **Hyperparameter re-optimization pipeline** (refinement)
8. **Ensemble coordination** (advanced)

---

## Sources

- [LightGBM init_model Performance Issue #3781](https://github.com/microsoft/LightGBM/issues/3781)
- [LightGBM Retrain Mechanics Issue #1469](https://github.com/Microsoft/LightGBM/issues/1469)
- [LightGBM Incremental Training Issue #3747](https://github.com/microsoft/LightGBM/issues/3747)
- [LightGBM train API Documentation](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.train.html)
- [PSI for Drift Detection - Fiddler AI](https://www.fiddler.ai/blog/measuring-data-drift-population-stability-index)
- [PSI Guide - NannyML](https://www.nannyml.com/blog/population-stability-index-psi)
- [ML Model Maintenance: Drifting Features - EvidentlyAI](https://www.evidentlyai.com/blog/ml-model-maintenance-drifting-features)
- [Champion/Challenger Models - DataRobot](https://www.datarobot.com/blog/introducing-mlops-champion-challenger-models/)
- [Shadow Deployments - Wallaroo.AI](https://wallaroo.ai/validating-ml-models-in-production-in-the-cloud-or-at-the-edge-using-shadow-deployments/)
- [Canary Deployment Patterns - Medium](https://medium.com/@duckweave/canary-calm-rollback-fast-12-ml-deployment-patterns-d893d501041f)
- [Model Versioning Best Practices - lakeFS](https://lakefs.io/blog/model-versioning/)
- [Automated Rollback Mechanisms - APXML](https://apxml.com/courses/monitoring-managing-ml-models-production/chapter-4-automated-retraining-updates/automated-rollback)
- [Prefect vs Airflow for ML Pipelines](https://pr-peri.github.io/blogpost/2026/02/08/blogpost-airflow-vs-prefect.html)
- [Orchestration Comparison - ZenML](https://www.zenml.io/blog/orchestration-showdown-dagster-vs-prefect-vs-airflow)
- [Dynamic Neuroplastic Networks for Financial Decisions - Springer](https://link.springer.com/article/10.1007/s10614-025-11057-1)
- [Catastrophic Forgetting via Selective LoRA](https://arxiv.org/abs/2501.15377)
- [Warm-Start Training with Incremental Data](https://arxiv.org/html/2406.04484v1)
- [Deflated Sharpe Ratio - Bailey & Lopez de Prado](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf)
- [From Paper to Production: ML Crypto Papers - Medium](https://medium.com/@jamieyuu/from-paper-to-production-what-academic-ml-crypto-papers-dont-tell-you-d91384154a46)
- [High-Frequency Crypto Price Forecasting - MDPI](https://www.mdpi.com/2078-2489/16/4/300)
- [Model Drift Best Practices - Encord](https://encord.com/blog/model-drift-best-practices/)
