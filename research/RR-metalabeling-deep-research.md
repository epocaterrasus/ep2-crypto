# Deep Research: Meta-Labeling for 5-Min BTC Prediction System

> Lopez de Prado's meta-labeling technique adapted for the ep2-crypto architecture.
> Covers theory, implementation, pitfalls, and integration with the existing LightGBM + CatBoost + GRU stacking ensemble.

---

## Table of Contents

1. [Primary Model Setup](#1-primary-model-setup)
2. [Meta-Label Generation Algorithm](#2-meta-label-generation-algorithm)
3. [Meta-Feature Construction](#3-meta-feature-construction)
4. [Preventing Data Leakage](#4-preventing-data-leakage)
5. [Meta-Model Selection](#5-meta-model-selection)
6. [Meta-Label as Bet Sizing](#6-meta-label-as-bet-sizing)
7. [Meta-Labeling with Triple Barrier](#7-meta-labeling-with-triple-barrier)
8. [Evaluating Meta-Labeling Improvement](#8-evaluating-meta-labeling-improvement)
9. [Online Meta-Label Updating](#9-online-meta-label-updating)
10. [Meta-Labeling with Ensemble Primary](#10-meta-labeling-with-ensemble-primary)
11. [Stacking vs Meta-Labeling](#11-stacking-vs-meta-labeling)
12. [Complete Pipeline Implementation](#12-complete-pipeline-implementation)

---

## 1. Primary Model Setup

### The Role of the Primary Model

In Lopez de Prado's framework, the primary model has ONE job: predict direction (side). It does NOT predict size. The critical insight is that a high-recall, moderate-precision primary model is BETTER than a high-precision primary model, because the meta-model's job is to filter false positives (improve precision). If the primary model already has high precision but low recall, the meta-model has nothing useful to filter — it will just reduce trade count further without improving quality.

### What Model to Use as Primary

**Recommendation: Use the full stacking ensemble as primary, not a simpler model.**

Rationale:
- The primary model needs to be your BEST directional predictor. Lopez de Prado's original formulation assumes the primary model is a "white box" (e.g., a fundamental model or rules-based system), but the technique works equally well when the primary is a black-box ML ensemble.
- The stacking ensemble (LightGBM + CatBoost + GRU with logistic regression meta-learner) from RESEARCH_SYNTHESIS.md is already designed to maximize directional accuracy.
- A simpler model (e.g., just LightGBM) would miss signals that the ensemble captures, and those missed signals can never be recovered by the meta-model.

**Alternative considered and rejected**: Using a simple model (e.g., single LightGBM or even a rule-based system like MA crossover) as primary, with the meta-model being more complex. This is Lopez de Prado's original "quantamental" use case but is suboptimal here because:
1. We do not have a fundamental model with economic intuition to preserve.
2. A simple primary model has lower recall, meaning fewer correct trades to filter from.
3. The meta-model cannot discover new directions — it can only filter existing ones.

### What Accuracy Should the Primary Target?

Per RESEARCH_SYNTHESIS.md, realistic directional accuracy at 5-min BTC is 52-56%. The primary model should target:

| Metric | Target | Rationale |
|--------|--------|-----------|
| Recall (per class) | >60% | High recall = more correct signals for meta-model to keep |
| Precision (per class) | >45% | Can be low — meta-model will improve this |
| Overall accuracy | 52-56% | Realistic for 5-min crypto |
| Ternary F1 (macro) | >0.40 | Balanced across up/flat/down |

**Key insight**: Tune the primary model to maximize RECALL at the expense of precision. Use a lower confidence threshold for the primary (e.g., 0.40 instead of 0.60). The meta-model will then filter the false positives.

### Primary Model Configuration

```python
import lightgbm as lgb
import numpy as np
from sklearn.linear_model import LogisticRegression

# Primary LightGBM — tuned for high recall
PRIMARY_LGBM_PARAMS = {
    "objective": "multiclass",
    "num_class": 3,  # up, flat, down
    "metric": "multi_logloss",
    "boosting_type": "gbdt",
    "num_leaves": 63,
    "learning_rate": 0.05,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 50,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "verbose": -1,
    "n_jobs": -1,
    # Lower class weight on "flat" to encourage directional predictions
    # This increases recall for up/down at the expense of flat precision
    "class_weight": {0: 1.2, 1: 0.6, 2: 1.2},  # up, flat, down
}

# Primary CatBoost — ordered boosting for diversity
PRIMARY_CATBOOST_PARAMS = {
    "iterations": 500,
    "learning_rate": 0.05,
    "depth": 6,
    "loss_function": "MultiClass",
    "eval_metric": "MultiClass",
    "random_seed": 42,
    "verbose": 0,
    "class_weights": [1.2, 0.6, 1.2],
}
```

### Extracting the "Side" from Primary

The primary model outputs a ternary prediction. For meta-labeling, we convert this to a side signal:

```python
def extract_side_from_primary(
    primary_proba: np.ndarray,  # shape (n_samples, 3): [p_up, p_flat, p_down]
    min_directional_prob: float = 0.35,
) -> np.ndarray:
    """Convert primary model probabilities to side signal.

    Returns:
        side: array of {-1, 0, +1}
        - +1 when primary predicts up with sufficient confidence
        - -1 when primary predicts down with sufficient confidence
        - 0 when primary predicts flat OR no direction is confident enough

    We use a LOW threshold (0.35) to maximize recall. The meta-model
    will filter the bad trades.
    """
    sides = np.zeros(len(primary_proba), dtype=np.int8)

    p_up = primary_proba[:, 0]
    p_down = primary_proba[:, 2]

    # Go long when up probability exceeds threshold AND exceeds down
    long_mask = (p_up > min_directional_prob) & (p_up > p_down)
    sides[long_mask] = 1

    # Go short when down probability exceeds threshold AND exceeds up
    short_mask = (p_down > min_directional_prob) & (p_down > p_up)
    sides[short_mask] = -1

    return sides
```

---

## 2. Meta-Label Generation Algorithm

### Core Algorithm

Given a primary model prediction at time t and the actual outcome, generate a binary label: "was the primary model correct AND profitable after costs?"

```python
import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass(frozen=True)
class MetaLabelConfig:
    """Configuration for meta-label generation."""
    transaction_cost_bps: float = 8.0      # Round-trip: 4 bps maker + 4 bps slippage
    min_return_bps: float = 5.0            # Minimum return above costs to count as "correct"
    horizon_bars: int = 1                   # How many bars forward to evaluate (1 = next 5-min bar)
    use_triple_barrier: bool = True         # Whether to use triple barrier method
    pt_multiplier: float = 2.0             # Profit-take = pt_multiplier * volatility
    sl_multiplier: float = 1.0             # Stop-loss = sl_multiplier * volatility
    max_holding_bars: int = 6              # Vertical barrier = 30 minutes at 5-min bars


def generate_meta_labels(
    primary_sides: np.ndarray,       # +1, -1, 0 from primary model
    close_prices: pd.Series,         # Close prices indexed by timestamp
    config: MetaLabelConfig,
) -> pd.Series:
    """Generate binary meta-labels: 1 = primary was correct & profitable, 0 = was not.

    The meta-label is 1 if and only if:
    1. The primary model predicted a direction (side != 0)
    2. The actual forward return IN THAT DIRECTION exceeded transaction costs + min_return

    This incorporates Lopez de Prado's key insight: the meta-label encodes
    "would following this signal have been profitable?"
    """
    n = len(primary_sides)
    meta_labels = np.zeros(n, dtype=np.int8)
    total_cost = config.transaction_cost_bps / 10_000

    for i in range(n):
        side = primary_sides[i]
        if side == 0:
            # Primary said flat — meta-label is 0 (no trade to evaluate)
            meta_labels[i] = 0
            continue

        if i + config.horizon_bars >= n:
            # Not enough forward data to evaluate
            meta_labels[i] = 0
            continue

        # Forward return
        entry_price = close_prices.iloc[i]
        exit_price = close_prices.iloc[i + config.horizon_bars]
        raw_return = (exit_price - entry_price) / entry_price

        # Directional return (positive means primary was correct)
        directional_return = side * raw_return

        # Net of costs
        net_return = directional_return - total_cost
        min_threshold = config.min_return_bps / 10_000

        # Meta-label = 1 only if net return exceeds minimum threshold
        meta_labels[i] = 1 if net_return > min_threshold else 0

    return pd.Series(meta_labels, index=close_prices.index, name="meta_label")


def generate_meta_labels_triple_barrier(
    primary_sides: np.ndarray,
    close_prices: pd.Series,
    high_prices: pd.Series,
    low_prices: pd.Series,
    volatility: pd.Series,           # Rolling volatility (e.g., 30-bar realized vol)
    config: MetaLabelConfig,
) -> tuple[pd.Series, pd.Series]:
    """Generate meta-labels using the triple barrier method.

    For each trade signal from the primary:
    1. Set upper barrier = entry + pt_multiplier * volatility (profit take)
    2. Set lower barrier = entry - sl_multiplier * volatility (stop loss)
    3. Set vertical barrier = max_holding_bars forward
    4. Whichever barrier is touched FIRST determines the label

    Returns:
        meta_labels: Series of 0/1 (was this trade profitable after costs?)
        touch_times: Series of timestamps when each barrier was touched
    """
    n = len(primary_sides)
    meta_labels = np.zeros(n, dtype=np.int8)
    touch_times = pd.Series(pd.NaT, index=close_prices.index)
    total_cost = config.transaction_cost_bps / 10_000

    for i in range(n):
        side = primary_sides[i]
        if side == 0:
            continue

        entry_price = close_prices.iloc[i]
        vol = volatility.iloc[i]
        if vol <= 0 or np.isnan(vol):
            continue

        # Set barriers relative to entry price
        upper_barrier = entry_price * (1 + config.pt_multiplier * vol)
        lower_barrier = entry_price * (1 - config.sl_multiplier * vol)

        # If short, flip the barriers
        if side == -1:
            upper_barrier, lower_barrier = (
                entry_price * (1 + config.sl_multiplier * vol),  # SL is up for shorts
                entry_price * (1 - config.pt_multiplier * vol),  # PT is down for shorts
            )

        # Walk forward looking for first barrier touch
        max_j = min(i + config.max_holding_bars, n)
        touched = False

        for j in range(i + 1, max_j):
            bar_high = high_prices.iloc[j]
            bar_low = low_prices.iloc[j]

            hit_upper = bar_high >= upper_barrier
            hit_lower = bar_low <= lower_barrier

            if side == 1:  # Long position
                if hit_upper:  # Profit-take touched
                    net_return = config.pt_multiplier * vol - total_cost
                    meta_labels[i] = 1 if net_return > 0 else 0
                    touch_times.iloc[i] = close_prices.index[j]
                    touched = True
                    break
                elif hit_lower:  # Stop-loss touched
                    meta_labels[i] = 0  # Loss
                    touch_times.iloc[i] = close_prices.index[j]
                    touched = True
                    break
            else:  # Short position
                if hit_lower:  # Profit-take for short
                    net_return = config.pt_multiplier * vol - total_cost
                    meta_labels[i] = 1 if net_return > 0 else 0
                    touch_times.iloc[i] = close_prices.index[j]
                    touched = True
                    break
                elif hit_upper:  # Stop-loss for short
                    meta_labels[i] = 0
                    touch_times.iloc[i] = close_prices.index[j]
                    touched = True
                    break

        if not touched:
            # Vertical barrier: evaluate return at expiry
            if max_j < n:
                exit_price = close_prices.iloc[max_j]
                raw_return = side * (exit_price - entry_price) / entry_price
                net_return = raw_return - total_cost
                meta_labels[i] = 1 if net_return > 0 else 0
                touch_times.iloc[i] = close_prices.index[max_j]

    return (
        pd.Series(meta_labels, index=close_prices.index, name="meta_label"),
        touch_times,
    )
```

### Expected Class Distribution

At 5-min BTC with realistic costs (8 bps round-trip):
- Primary predicts direction ~60-70% of bars (rest are flat)
- Of those directional predictions, roughly 45-55% are correct and profitable
- Meta-label distribution: approximately 45% positive (1) / 55% negative (0) among directional bars
- This mild imbalance is fine — do NOT use SMOTE or oversampling (per RESEARCH_SYNTHESIS.md)

### Pitfalls

1. **Do not include flat predictions in meta-label training.** Only bars where the primary predicted a direction (side != 0) get meta-labels. If you include flat bars, the meta-model learns to predict "no trade" which is trivial and dilutes the signal.

2. **Transaction costs MUST be in the meta-label.** Without costs, you are teaching the meta-model that a +0.01% return is "correct." In reality, that trade loses money after costs.

3. **Avoid forward-looking bias in volatility estimation.** The volatility used for triple barriers must be computed from data BEFORE time t, not including time t or later.

---

## 3. Meta-Feature Construction

### Feature Categories and Ranking

Research from Singh & Joubert (Hudson & Thames JFDS paper) and practical implementations consistently show this ranking of meta-feature importance:

| Rank | Feature Category | Importance | Why |
|------|-----------------|------------|-----|
| 1 | Primary model confidence | ~30% | Directly measures prediction certainty |
| 2 | Current volatility state | ~20% | Models fail differently in different vol regimes |
| 3 | Rolling accuracy of primary | ~15% | Captures model regime fitness |
| 4 | Regime / market state | ~12% | Structural regime changes degrade primary |
| 5 | Time features | ~8% | Session effects on primary accuracy |
| 6 | Original features (subset) | ~10% | Captures market conditions primary misreads |
| 7 | Order book state | ~5% | Liquidity affects execution quality |

### Recommended Meta-Feature Set (22 features)

```python
import numpy as np
import pandas as pd


def construct_meta_features(
    primary_proba: np.ndarray,       # (n, 3) probabilities from primary
    primary_sides: np.ndarray,       # Side predictions from primary
    original_features: pd.DataFrame, # Full feature matrix from feature pipeline
    close_prices: pd.Series,
    meta_label_history: pd.Series,   # Historical meta-labels (for rolling accuracy)
    regime_probs: np.ndarray,        # (n, n_regimes) regime probabilities
) -> pd.DataFrame:
    """Construct meta-features for the meta-model.

    Returns DataFrame with 22 meta-features. These are the features the
    meta-model uses to predict "will this trade be profitable?"
    """
    n = len(primary_proba)
    meta_features = pd.DataFrame(index=close_prices.index)

    # ---- Category 1: Primary Model Confidence (5 features) ----

    # Max probability across classes (how confident is the primary?)
    meta_features["primary_max_prob"] = primary_proba.max(axis=1)

    # Probability of the predicted class specifically
    predicted_class = primary_proba.argmax(axis=1)
    meta_features["primary_pred_prob"] = primary_proba[
        np.arange(n), predicted_class
    ]

    # Entropy of probability distribution (high entropy = uncertain)
    epsilon = 1e-10
    entropy = -np.sum(
        primary_proba * np.log(primary_proba + epsilon), axis=1
    )
    meta_features["primary_entropy"] = entropy

    # Margin between top-2 classes (large margin = more certain)
    sorted_probs = np.sort(primary_proba, axis=1)[:, ::-1]
    meta_features["primary_margin"] = sorted_probs[:, 0] - sorted_probs[:, 1]

    # Probability of the directional prediction (up or down, not flat)
    p_directional = primary_proba[:, 0] + primary_proba[:, 2]  # p_up + p_down
    meta_features["primary_directional_prob"] = p_directional

    # ---- Category 2: Volatility State (4 features) ----

    returns = close_prices.pct_change()

    # Realized volatility (rolling 12-bar = 1 hour)
    meta_features["vol_1h"] = returns.rolling(12).std()

    # Realized volatility (rolling 60-bar = 5 hours)
    meta_features["vol_5h"] = returns.rolling(60).std()

    # Vol-of-vol (how unstable is volatility itself?)
    rolling_vol = returns.rolling(12).std()
    meta_features["vol_of_vol"] = rolling_vol.rolling(24).std()

    # Volatility ratio (short/long — rising = vol expanding)
    meta_features["vol_ratio"] = (
        meta_features["vol_1h"] / meta_features["vol_5h"].clip(lower=1e-10)
    )

    # ---- Category 3: Rolling Accuracy of Primary (3 features) ----

    # Rolling accuracy over last 50 trades
    meta_features["rolling_accuracy_50"] = (
        meta_label_history.rolling(50, min_periods=10).mean()
    )

    # Rolling accuracy over last 200 trades
    meta_features["rolling_accuracy_200"] = (
        meta_label_history.rolling(200, min_periods=50).mean()
    )

    # Accuracy trend (is primary getting better or worse?)
    acc_50 = meta_features["rolling_accuracy_50"]
    acc_200 = meta_features["rolling_accuracy_200"]
    meta_features["accuracy_trend"] = acc_50 - acc_200

    # ---- Category 4: Regime / Market State (3 features) ----

    # Regime probability of the dominant regime
    meta_features["regime_max_prob"] = regime_probs.max(axis=1)

    # Regime uncertainty (entropy of regime probs)
    regime_entropy = -np.sum(
        regime_probs * np.log(regime_probs + epsilon), axis=1
    )
    meta_features["regime_entropy"] = regime_entropy

    # Regime stability (has regime changed recently?)
    dominant_regime = regime_probs.argmax(axis=1)
    regime_change = np.diff(dominant_regime, prepend=dominant_regime[0]) != 0
    meta_features["recent_regime_changes"] = (
        pd.Series(regime_change.astype(float), index=close_prices.index)
        .rolling(12)
        .sum()
    )

    # ---- Category 5: Time Features (3 features) ----

    if hasattr(close_prices.index, 'hour'):
        hour = close_prices.index.hour
    else:
        hour = pd.to_datetime(close_prices.index).hour

    # Cyclical hour encoding
    meta_features["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    meta_features["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    # Session indicator (0=Asia, 1=Europe, 2=US)
    session = np.where(
        hour < 8, 0,
        np.where(hour < 16, 1, 2)
    )
    meta_features["session"] = session.astype(float)

    # ---- Category 6: Select Original Features (3 features) ----
    # Choose features that capture CONDITIONS, not direction

    if "obi_weighted" in original_features.columns:
        meta_features["obi_abs"] = original_features["obi_weighted"].abs()

    if "spread_relative" in original_features.columns:
        meta_features["spread"] = original_features["spread_relative"]

    if "trade_intensity_5m" in original_features.columns:
        meta_features["trade_intensity"] = original_features["trade_intensity_5m"]

    # ---- Category 7: Order Book State (1 feature) ----

    if "book_depth_ratio" in original_features.columns:
        meta_features["book_depth_ratio"] = original_features["book_depth_ratio"]

    return meta_features
```

### Which Combination Works Best?

Based on ablation studies in the Hudson & Thames paper and empirical results from crypto implementations:

1. **Primary confidence alone**: ~60% of total meta-labeling improvement. This is the single most important meta-feature.
2. **Primary confidence + volatility state**: ~80% of total improvement. Volatility determines whether the market is "tradeable."
3. **All 22 features**: Full improvement, but diminishing returns past the first 10 features.

**Critical rule**: Do NOT pass all original features (30+) to the meta-model. This makes the meta-model too similar to the primary and defeats the purpose. The meta-model should focus on WHEN the primary is right, not re-learn WHAT the primary predicts.

### Pitfalls

1. **Rolling accuracy is a lagging indicator.** It tells you if the primary WAS good, not if it WILL BE good. Use it as context, not primary signal.
2. **Do not include raw price or return features.** These make the meta-model learn direction, which is the primary's job.
3. **Absolute values of directional features.** When passing features like OBI to the meta-model, use the absolute value. The meta-model should not learn direction — only magnitude of signals.

---

## 4. Preventing Data Leakage in Meta-Labeling

### The Core Problem

The meta-model learns to predict "will the primary be correct?" If the meta-model sees the primary's IN-SAMPLE predictions during training, it is effectively seeing predictions that were fitted to the same data. The primary model will appear more accurate than it really is on in-sample data, and the meta-model will learn an overly optimistic view of "when the primary is right."

### Solution: Out-of-Fold (OOF) Predictions with Purging

The meta-model must ONLY see out-of-fold predictions from the primary model. This means the primary's prediction for time t was made by a model that never trained on data from time t (or nearby times).

```python
import numpy as np
import pandas as pd
import lightgbm as lgb
import structlog

logger = structlog.get_logger()


def generate_oof_predictions_walk_forward(
    features: pd.DataFrame,
    labels: pd.Series,
    train_window: int = 4032,     # 14 days of 5-min bars
    test_window: int = 288,       # 1 day
    purge_gap: int = 6,           # 30 minutes (max feature lookback)
    embargo_bars: int = 12,       # 1 hour post-test embargo
    model_params: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate out-of-fold predictions using purged walk-forward validation.

    This is the CORRECT way to generate primary model predictions for
    meta-label training. Each prediction is made by a model that:
    1. Was trained on data BEFORE the prediction point
    2. Has a purge gap between training end and test start
    3. Has an embargo after test end before the next training window

    Returns:
        oof_proba: (n_samples, n_classes) out-of-fold predicted probabilities
        oof_valid_mask: boolean array, True where OOF prediction exists
    """
    if model_params is None:
        model_params = PRIMARY_LGBM_PARAMS.copy()

    n = len(features)
    n_classes = labels.nunique()
    oof_proba = np.full((n, n_classes), np.nan)
    oof_valid_mask = np.zeros(n, dtype=bool)

    fold = 0
    test_start = train_window + purge_gap

    while test_start < n:
        test_end = min(test_start + test_window, n)

        # Training window: [test_start - purge_gap - train_window, test_start - purge_gap)
        train_end = test_start - purge_gap
        train_start = max(0, train_end - train_window)

        if train_end - train_start < train_window // 2:
            break  # Not enough training data

        # Embargo: exclude embargo_bars after test_end from FUTURE training
        # (This is handled implicitly by sliding forward)

        X_train = features.iloc[train_start:train_end]
        y_train = labels.iloc[train_start:train_end]
        X_test = features.iloc[test_start:test_end]

        # Train primary model on this fold
        train_data = lgb.Dataset(X_train, label=y_train)
        model = lgb.train(
            model_params,
            train_data,
            num_boost_round=300,
            valid_sets=[train_data],
            callbacks=[lgb.log_evaluation(period=0)],  # Suppress output
        )

        # Generate OOF predictions for the test window
        fold_proba = model.predict(X_test)
        oof_proba[test_start:test_end] = fold_proba
        oof_valid_mask[test_start:test_end] = True

        logger.info(
            "walk_forward_fold",
            fold=fold,
            train_range=f"{train_start}-{train_end}",
            test_range=f"{test_start}-{test_end}",
            n_train=train_end - train_start,
            n_test=test_end - test_start,
        )

        fold += 1
        test_start += test_window

    valid_count = oof_valid_mask.sum()
    logger.info(
        "oof_generation_complete",
        total_folds=fold,
        valid_predictions=int(valid_count),
        coverage=f"{valid_count / n:.1%}",
    )

    return oof_proba, oof_valid_mask
```

### Walk-Forward Implementation Detail

```
Time ────────────────────────────────────────────────────►

Fold 0:
  [===== TRAIN (14d) =====][purge][== TEST (1d) ==]

Fold 1:
       [===== TRAIN (14d) =====][purge][== TEST (1d) ==]

Fold 2:
            [===== TRAIN (14d) =====][purge][== TEST (1d) ==]

Fold 3:
                 [===== TRAIN (14d) =====][purge][== TEST (1d) ==]

Key rules:
- Train window SLIDES (not expands) — crypto non-stationarity
- Purge gap = 6 bars (30 min) between train end and test start
- Test windows are contiguous and non-overlapping
- Embargo of 12 bars after each test window (not used in next train)
```

### Why NOT Standard K-Fold

Standard k-fold with time series creates MASSIVE leakage:
- Fold 3 trains on data from days 1-15 and 21-30, but tests on days 16-20
- The model has seen the FUTURE (days 21-30) when predicting days 16-20
- This inflates accuracy by 5-15%, making the meta-labels unrealistically optimistic

### Combinatorial Purged Cross-Validation (CPCV) — Optional Enhancement

For maximum robustness, use Lopez de Prado's CPCV which generates more test paths than standard walk-forward:

```python
def generate_oof_predictions_cpcv(
    features: pd.DataFrame,
    labels: pd.Series,
    n_splits: int = 6,
    purge_gap: int = 6,
    embargo_pct: float = 0.01,
    model_params: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """CPCV: Combinatorial Purged Cross-Validation.

    Generates (n_splits choose 2) backtest paths instead of just n_splits.
    More paths = better estimate of OOF performance, less variance in meta-labels.

    Note: CPCV is computationally expensive. For 6 splits, it generates
    C(6,2) = 15 backtest paths instead of 6.
    """
    if model_params is None:
        model_params = PRIMARY_LGBM_PARAMS.copy()

    n = len(features)
    n_classes = labels.nunique()

    # Create temporal splits
    split_size = n // n_splits
    splits = [(i * split_size, min((i + 1) * split_size, n)) for i in range(n_splits)]

    # Accumulate predictions across all CPCV paths
    oof_proba_sum = np.zeros((n, n_classes))
    oof_count = np.zeros(n, dtype=int)

    from itertools import combinations

    for test_splits in combinations(range(n_splits), 2):
        # Test set = union of the two held-out splits
        test_indices = []
        for s in test_splits:
            start, end = splits[s]
            test_indices.extend(range(start, end))

        # Train set = everything else, with purging
        train_indices = []
        for s in range(n_splits):
            if s in test_splits:
                continue
            start, end = splits[s]
            # Purge: remove purge_gap bars adjacent to test boundaries
            for test_s in test_splits:
                t_start, t_end = splits[test_s]
                embargo_len = int(embargo_pct * n)
                start = max(start, 0)
                # Remove overlap with purge zone
                purge_start = t_start - purge_gap
                purge_end = t_end + embargo_len
                indices = [
                    idx for idx in range(start, end)
                    if idx < purge_start or idx >= purge_end
                ]
                train_indices.extend(indices)

        train_indices = sorted(set(train_indices))
        test_indices = sorted(set(test_indices))

        if len(train_indices) < 1000 or len(test_indices) < 100:
            continue

        X_train = features.iloc[train_indices]
        y_train = labels.iloc[train_indices]
        X_test = features.iloc[test_indices]

        train_data = lgb.Dataset(X_train, label=y_train)
        model = lgb.train(
            model_params,
            train_data,
            num_boost_round=300,
            callbacks=[lgb.log_evaluation(period=0)],
        )

        fold_proba = model.predict(X_test)
        for k, idx in enumerate(test_indices):
            oof_proba_sum[idx] += fold_proba[k]
            oof_count[idx] += 1

    # Average predictions across paths
    valid_mask = oof_count > 0
    oof_proba = np.zeros((n, n_classes))
    oof_proba[valid_mask] = oof_proba_sum[valid_mask] / oof_count[valid_mask, np.newaxis]

    return oof_proba, valid_mask
```

### Pitfalls

1. **Most common mistake**: Training the meta-model on in-sample primary predictions. This makes meta-labeling appear to give 20-30% improvement in backtest, but ZERO improvement live. Always use OOF predictions.

2. **Purge gap must cover max feature lookback.** If features use a 30-bar rolling window, the purge gap must be at least 30 bars. For the ep2-crypto feature set, the longest lookback is ~60 bars (5h volatility), so purge_gap should be 60.

3. **Embargo handles autocorrelation.** Even with purging, adjacent bars are correlated. The embargo ensures the model trained on bars up to t cannot exploit autocorrelation when predicting bars close to t.

---

## 5. Meta-Model Selection

### Recommendation: Logistic Regression (with LightGBM as Alternative)

The meta-task is fundamentally simpler than the primary task. The primary model must learn complex nonlinear patterns in 30+ features. The meta-model only needs to learn "is this prediction likely correct?" from ~22 features that are mostly confidence/volatility metrics.

| Model | Pros | Cons | Recommendation |
|-------|------|------|---------------|
| **Logistic Regression** | Simple, fast, well-calibrated probabilities, minimal overfitting | Cannot capture feature interactions | **Primary choice** |
| **LightGBM (shallow)** | Captures interactions (e.g., high confidence + low vol = good) | Can overfit the meta-task, needs more tuning | **Alternative** |
| Random Forest | Robust, good probabilities | Slow inference, overkill | Not recommended |
| Neural Network | Flexible | Massively overkill, overfitting risk | Not recommended |
| SVM | Good margins | Poor probability calibration | Not recommended |

### Why Logistic Regression is Preferred

1. **Calibrated probabilities out of the box.** The meta-model's predicted probability IS the bet size (Section 6). Logistic regression gives well-calibrated probabilities without additional calibration.

2. **Regularization prevents overfitting.** With L2 regularization (C=0.1 to 1.0), logistic regression is highly resistant to overfitting on the ~22 meta-features.

3. **Interpretable.** You can read the coefficients to understand WHAT makes the meta-model accept/reject a trade. This is invaluable for debugging.

4. **Lopez de Prado's own recommendation.** In AFML Ch. 3, he suggests the meta-model should be simpler than the primary to avoid "meta-overfitting."

### Implementation

```python
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd


def train_meta_model_logistic(
    meta_features: pd.DataFrame,
    meta_labels: pd.Series,
    valid_mask: np.ndarray,
) -> Pipeline:
    """Train a logistic regression meta-model.

    Uses LogisticRegressionCV for automatic regularization tuning.
    Only trains on samples where:
    1. The OOF prediction is valid (valid_mask)
    2. The primary predicted a direction (meta_label is defined)
    """
    # Filter to valid samples with directional predictions
    mask = valid_mask & (~meta_labels.isna()) & (meta_labels.index.isin(meta_features.index))
    X = meta_features.loc[mask].copy()
    y = meta_labels.loc[mask].copy()

    # Drop rows with NaN features
    nan_mask = X.isna().any(axis=1)
    X = X[~nan_mask]
    y = y[~nan_mask]

    # Pipeline: scale + logistic regression
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegressionCV(
            Cs=np.logspace(-3, 1, 20),  # Search regularization strength
            cv=5,                        # Internal CV for C selection
            scoring="f1",                # Optimize F1 (balance precision/recall)
            max_iter=1000,
            class_weight="balanced",     # Handle mild class imbalance
            random_state=42,
        )),
    ])

    pipeline.fit(X, y)

    # Log feature importance (coefficients)
    coefs = pipeline.named_steps["classifier"].coef_[0]
    feature_importance = pd.Series(
        np.abs(coefs),
        index=X.columns,
    ).sort_values(ascending=False)

    logger.info(
        "meta_model_trained",
        n_samples=len(X),
        positive_rate=f"{y.mean():.3f}",
        best_C=float(pipeline.named_steps["classifier"].C_[0]),
        top_features=feature_importance.head(5).to_dict(),
    )

    return pipeline


def train_meta_model_lgbm(
    meta_features: pd.DataFrame,
    meta_labels: pd.Series,
    valid_mask: np.ndarray,
) -> lgb.Booster:
    """Alternative: shallow LightGBM meta-model.

    Use max_depth=3 and few leaves to prevent overfitting.
    This captures interactions like (high_confidence AND low_vol → good trade)
    that logistic regression misses.
    """
    mask = valid_mask & (~meta_labels.isna())
    X = meta_features.loc[mask].dropna()
    y = meta_labels.loc[X.index]

    META_LGBM_PARAMS = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "num_leaves": 8,           # Very shallow — prevents overfitting
        "max_depth": 3,            # Hard depth limit
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 100,  # High minimum to prevent overfitting
        "reg_alpha": 1.0,
        "reg_lambda": 5.0,         # Strong regularization
        "verbose": -1,
        "is_unbalance": True,
    }

    train_data = lgb.Dataset(X, label=y)
    model = lgb.train(
        META_LGBM_PARAMS,
        train_data,
        num_boost_round=100,       # Few rounds — simple model
        callbacks=[lgb.log_evaluation(period=0)],
    )

    return model
```

### When to Use LightGBM Instead of Logistic Regression

Use the LightGBM meta-model when:
- You observe interaction effects in meta-feature analysis (e.g., high confidence is only predictive when volatility is in a specific range)
- The logistic regression meta-model's cross-validated F1 is below 0.55
- You have a large enough meta-training set (>5000 directional samples)

### Pitfalls

1. **Do not make the meta-model complex.** If the meta-model is as complex as the primary, you are just building a second primary model that will overfit differently.
2. **Always use class_weight="balanced" or is_unbalance=True.** The meta-labels are mildly imbalanced, and you want the meta-model to learn to detect BOTH good and bad trades.
3. **Calibrate the LightGBM meta-model.** Unlike logistic regression, LightGBM probabilities are NOT well-calibrated. Apply isotonic regression calibration if using LightGBM as meta-model (see Section 6).

---

## 6. Meta-Label as Bet Sizing

### Lopez de Prado's Bet Sizing Formula

The meta-model's predicted probability p = P(primary is correct) is converted to a bet size using:

```
z = (p - 0.5) / sqrt(p * (1 - p))
bet_size = 2 * Phi(z) - 1
```

Where Phi is the CDF of the standard normal distribution.

This mapping has desirable properties:
- p = 0.5 -> bet_size = 0 (no edge, no bet)
- p = 0.6 -> bet_size ~ 0.16 (small bet)
- p = 0.7 -> bet_size ~ 0.34 (moderate bet)
- p = 0.8 -> bet_size ~ 0.53 (larger bet)
- p = 0.9 -> bet_size ~ 0.74 (aggressive)
- p = 1.0 -> bet_size = 1.0 (full size — never happens in practice)

The relationship is nonlinear and conservative: near p=0.5, bet sizes are tiny, preventing over-trading on marginal signals.

### Implementation

```python
import numpy as np
from scipy.stats import norm


def meta_prob_to_bet_size(
    meta_proba: np.ndarray,
    max_bet_fraction: float = 0.05,   # Max 5% of capital per trade
    min_prob_threshold: float = 0.55,  # Below this = no trade
) -> np.ndarray:
    """Convert meta-model predicted probability to position size.

    Uses Lopez de Prado's formula from AFML Chapter 10.

    Args:
        meta_proba: P(primary correct) from meta-model, shape (n,)
        max_bet_fraction: Maximum position size as fraction of capital
        min_prob_threshold: Minimum meta-probability to take any trade

    Returns:
        bet_sizes: Position size as fraction of capital, shape (n,)
                   0 means no trade, max_bet_fraction means full size
    """
    bet_sizes = np.zeros_like(meta_proba)

    # Only bet when meta-probability exceeds threshold
    active = meta_proba > min_prob_threshold

    p = meta_proba[active]

    # Lopez de Prado's formula
    # z = (p - 0.5) / sqrt(p * (1-p))
    z = (p - 0.5) / np.sqrt(p * (1 - p) + 1e-10)

    # bet_size = 2 * Phi(z) - 1, where Phi is standard normal CDF
    raw_bet = 2 * norm.cdf(z) - 1

    # Scale to max position size
    bet_sizes[active] = raw_bet * max_bet_fraction

    return bet_sizes


def apply_bet_sizing_with_concurrent_bets(
    meta_proba: np.ndarray,
    primary_sides: np.ndarray,
    current_positions: float,         # Current open position (fraction of capital)
    max_portfolio_exposure: float = 0.15,  # Max total exposure
    max_bet_fraction: float = 0.05,
    min_prob_threshold: float = 0.55,
) -> np.ndarray:
    """Bet sizing accounting for existing positions.

    Lopez de Prado (AFML Ch. 10) notes that bet sizes should account
    for concurrent active bets. If you already have 10% exposure,
    a new trade's max size is reduced.
    """
    raw_bet = meta_prob_to_bet_size(
        meta_proba, max_bet_fraction, min_prob_threshold
    )

    # Reduce available capacity by current exposure
    available = max(0, max_portfolio_exposure - abs(current_positions))
    scaled_bet = np.minimum(raw_bet, available)

    return scaled_bet


def quarter_kelly_bet_sizing(
    meta_proba: np.ndarray,
    avg_win_return: float,   # Average winning trade return (e.g., 0.003 = 30 bps)
    avg_loss_return: float,  # Average losing trade return (e.g., -0.002 = -20 bps)
    max_bet_fraction: float = 0.05,
    min_prob_threshold: float = 0.55,
) -> np.ndarray:
    """Quarter-Kelly position sizing using meta-model probabilities.

    Kelly formula: f* = (p * b - q) / b
    where p = P(win), q = 1-p, b = win/loss ratio

    Quarter-Kelly = f*/4 for robustness against estimation error.
    This is the approach recommended in RESEARCH_SYNTHESIS.md.
    """
    bet_sizes = np.zeros_like(meta_proba)
    active = meta_proba > min_prob_threshold

    p = meta_proba[active]
    q = 1 - p
    b = abs(avg_win_return / avg_loss_return) if avg_loss_return != 0 else 1.0

    # Full Kelly
    kelly = (p * b - q) / b

    # Clamp negative Kelly (no edge) to zero
    kelly = np.maximum(kelly, 0)

    # Quarter Kelly for safety
    quarter_kelly = kelly / 4

    # Scale to max bet fraction
    bet_sizes[active] = np.minimum(quarter_kelly, max_bet_fraction)

    return bet_sizes
```

### Combining Side and Size

The final trading signal combines the primary model's direction with the meta-model's sizing:

```python
def combine_side_and_size(
    primary_sides: np.ndarray,     # +1, -1, 0
    bet_sizes: np.ndarray,         # 0.0 to max_bet_fraction
) -> np.ndarray:
    """Combine direction from primary with size from meta-model.

    Final position = side * size
    Examples:
        side=+1, size=0.03 -> position = +0.03 (long 3% of capital)
        side=-1, size=0.05 -> position = -0.05 (short 5% of capital)
        side=+1, size=0.00 -> position = 0 (meta-model filtered this trade)
        side=0,  size=any  -> position = 0 (primary said flat)
    """
    return primary_sides * bet_sizes
```

### Expected Bet Size Distribution

For a realistic 5-min BTC system with meta-labeling:
- ~60-70% of bars: position = 0 (primary says flat OR meta-model filters)
- ~15-20% of bars: position in [0.01, 0.03] (low-confidence trades)
- ~8-12% of bars: position in [0.03, 0.05] (high-confidence trades)
- ~2-5% of bars: position = max (very high confidence)

This selectivity is the primary source of Sharpe improvement: you are trading less but better.

### Pitfalls

1. **Never use raw probability as bet size.** p=0.55 should NOT mean 55% position. The Lopez de Prado formula correctly maps marginal edges to small bets.

2. **Quarter-Kelly, not full Kelly.** Full Kelly is optimal only with perfect probability estimates. In practice, our probability estimates have estimation error. Quarter-Kelly sacrifices ~30% of expected return but reduces variance by ~75%.

3. **Probability calibration is critical.** If the meta-model says p=0.7 but the true accuracy at that confidence level is p=0.6, your bets are too large. Always verify calibration with a reliability diagram.

---

## 7. Meta-Labeling with Triple Barrier

### Combining the Two Concepts

The triple barrier method and meta-labeling serve complementary roles:

1. **Triple barrier method**: Defines WHAT constitutes a successful trade (hit profit-take before stop-loss)
2. **Meta-labeling**: Predicts WHETHER the current primary signal will lead to a successful triple-barrier outcome

### The Full Integration

```python
def full_triple_barrier_meta_labeling(
    features: pd.DataFrame,
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    primary_model,                   # Trained primary ensemble
    meta_model,                      # Trained meta-model
    volatility: pd.Series,          # Rolling volatility
    config: MetaLabelConfig,
) -> pd.DataFrame:
    """End-to-end triple barrier meta-labeling for live inference.

    Pipeline:
    1. Primary model predicts direction (side)
    2. Triple barriers are set based on side + current volatility
    3. Meta-model predicts P(trade hits profit-take before stop-loss)
    4. Bet size is derived from meta-model probability
    5. Final position = side * bet_size

    Returns DataFrame with columns:
        side, meta_prob, bet_size, position, upper_barrier, lower_barrier, vertical_barrier
    """
    # Step 1: Primary model predicts probabilities
    primary_proba = primary_model.predict_proba(features)
    sides = extract_side_from_primary(primary_proba)

    # Step 2: Construct meta-features
    # Note: in live inference, we need historical meta-labels for rolling accuracy
    # This is maintained in a rolling buffer
    meta_feats = construct_meta_features(
        primary_proba=primary_proba,
        primary_sides=sides,
        original_features=features,
        close_prices=close,
        meta_label_history=_get_rolling_meta_label_history(),  # From live state
        regime_probs=_get_current_regime_probs(),              # From regime module
    )

    # Step 3: Meta-model predicts P(trade is profitable)
    meta_proba = meta_model.predict_proba(meta_feats)[:, 1]  # P(class=1)

    # Step 4: Convert to bet sizes
    bet_sizes = meta_prob_to_bet_size(meta_proba)

    # Step 5: Combine side and size
    positions = combine_side_and_size(sides, bet_sizes)

    # Step 6: Set barriers for active trades
    result = pd.DataFrame(index=features.index)
    result["side"] = sides
    result["meta_prob"] = meta_proba
    result["bet_size"] = bet_sizes
    result["position"] = positions

    # Barriers for active trades
    active = positions != 0
    result["upper_barrier"] = np.where(
        active & (sides == 1),
        close * (1 + config.pt_multiplier * volatility),
        np.where(
            active & (sides == -1),
            close * (1 + config.sl_multiplier * volatility),
            np.nan,
        ),
    )
    result["lower_barrier"] = np.where(
        active & (sides == 1),
        close * (1 - config.sl_multiplier * volatility),
        np.where(
            active & (sides == -1),
            close * (1 - config.pt_multiplier * volatility),
            np.nan,
        ),
    )
    result["vertical_barrier_bars"] = np.where(
        active, config.max_holding_bars, 0
    )

    return result
```

### Asymmetric Barriers

A key insight for 5-min BTC: the profit-take and stop-loss multipliers should NOT be symmetric.

```python
# Asymmetric barriers by regime
BARRIER_CONFIGS = {
    "trending": MetaLabelConfig(
        pt_multiplier=3.0,   # Wide profit-take in trends
        sl_multiplier=1.0,   # Tight stop-loss
        max_holding_bars=6,  # Hold up to 30 min
    ),
    "mean_reverting": MetaLabelConfig(
        pt_multiplier=1.5,   # Tighter profit-take in ranges
        sl_multiplier=1.5,   # Wider stop-loss (allow bounce)
        max_holding_bars=4,  # Shorter hold
    ),
    "volatile": MetaLabelConfig(
        pt_multiplier=2.0,   # Moderate
        sl_multiplier=2.0,   # Wide stop-loss (volatile moves)
        max_holding_bars=3,  # Short hold — get out fast
    ),
}
```

### Pitfalls

1. **Volatility scaling is essential.** Fixed-pip barriers fail because BTC volatility changes 5x intraday. Always use volatility-scaled barriers.

2. **Vertical barrier must be tight at 5-min.** Beyond 6 bars (30 min), the signal decays (per RESEARCH_SYNTHESIS.md, signal half-life is ~3 min for microstructure features). Do not hold beyond max_holding_bars.

3. **Barrier calibration via backtest.** The pt/sl multipliers should be tuned to maximize the meta-labeled Sharpe, not just win rate. A 3:1 pt:sl ratio has higher win rate but fewer triggering trades.

---

## 8. Evaluating Meta-Labeling Improvement

### Metrics Framework

Meta-labeling should be evaluated as A/B: system WITHOUT meta-labeling vs WITH meta-labeling.

```python
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class MetaLabelingEvaluation:
    """Complete evaluation of meta-labeling impact."""

    # Before meta-labeling (all primary directional trades, equal size)
    sharpe_before: float
    precision_before: float
    recall_before: float
    f1_before: float
    trades_per_day_before: float
    avg_return_before: float

    # After meta-labeling (filtered + sized by meta-model)
    sharpe_after: float
    precision_after: float
    recall_after: float
    f1_after: float
    trades_per_day_after: float
    avg_return_after: float

    # Improvement metrics
    sharpe_improvement_pct: float
    precision_improvement_pct: float
    trade_reduction_pct: float
    is_statistically_significant: bool
    p_value: float


def evaluate_meta_labeling(
    primary_sides: np.ndarray,
    meta_positions: np.ndarray,      # Sized positions from meta-model
    forward_returns: pd.Series,       # Actual forward returns
    trading_cost_bps: float = 8.0,
    n_bootstrap: int = 10000,
) -> MetaLabelingEvaluation:
    """Compare system performance with and without meta-labeling.

    Before: Take every trade the primary suggests, equal size.
    After: Take only meta-approved trades, with meta-sized positions.
    """
    cost = trading_cost_bps / 10_000

    # --- Before meta-labeling ---
    # Every directional prediction becomes a trade at fixed size
    active_before = primary_sides != 0
    returns_before = (
        primary_sides[active_before] * forward_returns.values[active_before] - cost
    )
    sharpe_before = _compute_sharpe(returns_before)

    # Was primary correct? (positive return after costs)
    correct_before = returns_before > 0
    precision_before = correct_before.mean() if len(correct_before) > 0 else 0
    n_trades_before = active_before.sum()

    # --- After meta-labeling ---
    active_after = meta_positions != 0
    returns_after = (
        meta_positions[active_after] * forward_returns.values[active_after] - cost
    )
    sharpe_after = _compute_sharpe(returns_after)

    correct_after = returns_after > 0
    precision_after = correct_after.mean() if len(correct_after) > 0 else 0
    n_trades_after = active_after.sum()

    # --- Recall ---
    # Recall = of ALL profitable trades, how many did we take?
    all_profitable = (primary_sides * forward_returns.values - cost) > 0
    meta_took = (meta_positions != 0)

    recall_before = 1.0  # Before meta-labeling, we take all trades
    recall_after = (
        (all_profitable & meta_took).sum() / max(all_profitable.sum(), 1)
    )

    # --- F1 ---
    f1_before = 2 * precision_before * recall_before / max(precision_before + recall_before, 1e-10)
    f1_after = 2 * precision_after * recall_after / max(precision_after + recall_after, 1e-10)

    # --- Statistical significance (bootstrap Sharpe difference) ---
    sharpe_diffs = np.zeros(n_bootstrap)
    for b in range(n_bootstrap):
        # Resample with replacement
        idx_b = np.random.choice(len(returns_before), size=len(returns_before), replace=True)
        idx_a = np.random.choice(len(returns_after), size=len(returns_after), replace=True)
        sharpe_diffs[b] = _compute_sharpe(returns_after[idx_a]) - _compute_sharpe(returns_before[idx_b])

    p_value = float((sharpe_diffs <= 0).mean())
    significant = p_value < 0.05

    bars_per_day = 288
    n_days = len(primary_sides) / bars_per_day

    return MetaLabelingEvaluation(
        sharpe_before=float(sharpe_before),
        precision_before=float(precision_before),
        recall_before=float(recall_before),
        f1_before=float(f1_before),
        trades_per_day_before=float(n_trades_before / max(n_days, 1)),
        avg_return_before=float(returns_before.mean()) if len(returns_before) > 0 else 0.0,
        sharpe_after=float(sharpe_after),
        precision_after=float(precision_after),
        recall_after=float(recall_after),
        f1_after=float(f1_after),
        trades_per_day_after=float(n_trades_after / max(n_days, 1)),
        avg_return_after=float(returns_after.mean()) if len(returns_after) > 0 else 0.0,
        sharpe_improvement_pct=float(
            (sharpe_after - sharpe_before) / max(abs(sharpe_before), 1e-10) * 100
        ),
        precision_improvement_pct=float(
            (precision_after - precision_before) / max(precision_before, 1e-10) * 100
        ),
        trade_reduction_pct=float(
            (n_trades_before - n_trades_after) / max(n_trades_before, 1) * 100
        ),
        is_statistically_significant=bool(significant),
        p_value=float(p_value),
    )


def _compute_sharpe(returns: np.ndarray, bars_per_year: float = 105120) -> float:
    """Annualized Sharpe from per-bar returns."""
    if len(returns) < 2:
        return 0.0
    mean = returns.mean()
    std = returns.std(ddof=1)
    if std < 1e-15:
        return 0.0
    return float(mean / std * np.sqrt(bars_per_year))
```

### Expected Improvements

Based on the Hudson & Thames research (Singh & Joubert JFDS paper) and empirical crypto applications:

| Metric | Before Meta-Labeling | After Meta-Labeling | Change |
|--------|---------------------|---------------------|--------|
| Sharpe Ratio | 0.8-1.2 | 1.5-2.5 | +50-100% |
| Precision | 48-52% | 55-62% | +10-15pp |
| Recall | 100% (take all) | 60-75% | -25-40pp |
| F1 Score | 0.48-0.52 | 0.57-0.68 | +15-25% |
| Trades/Day | 30-80 | 10-30 | -50-70% |
| Avg Return/Trade | 2-5 bps | 8-15 bps | +100-200% |
| Max Drawdown | 15-25% | 8-15% | -30-50% |

**Key insight**: Meta-labeling sacrifices recall for precision. You take fewer trades, but each trade is better. The Sharpe improves because the average return per trade increases faster than the reduction in trade count.

### Pitfalls

1. **Sharpe improvement from trade reduction alone is not real alpha.** If you just randomly remove 50% of trades, Sharpe often increases due to reduced variance. The meta-model must selectively remove BAD trades, not random ones. Test by comparing against random trade filtering.

2. **Use Deflated Sharpe Ratio.** If you tried multiple meta-model configurations, the best one's Sharpe is upward-biased. Apply the DSR correction from Lopez de Prado.

3. **Regime-decompose the evaluation.** Meta-labeling might improve Sharpe overall but destroy it in one specific regime. Check per-regime.

---

## 9. Online Meta-Label Updating

### The Synchronization Problem

When the primary model is retrained, its prediction characteristics change:
- A retrained primary model makes DIFFERENT predictions on the same data
- The meta-labels (which encode "was primary correct?") become stale
- The meta-model was trained on the OLD primary's error patterns

### Retraining Schedule

```python
from datetime import timedelta
import structlog

logger = structlog.get_logger()


class MetaLabelRetrainingSchedule:
    """Manages the meta-model retraining lifecycle.

    The meta-model must be retrained AFTER the primary model is retrained,
    because the meta-labels depend on the primary's predictions.

    Schedule:
    - Primary model: retrain every 2-4 hours (warm-start)
    - Meta-model: retrain every 4-8 hours (after primary retrain + new meta-labels)

    The meta-model retraining is ALWAYS triggered by a primary retrain,
    never independently.
    """

    def __init__(
        self,
        primary_retrain_interval_bars: int = 48,   # Every 4 hours (48 * 5min)
        meta_retrain_interval_bars: int = 96,       # Every 8 hours
        meta_label_lookback_bars: int = 4032,       # 14 days of meta-label history
        min_meta_labels_for_retrain: int = 500,     # Minimum directional samples
    ) -> None:
        self.primary_retrain_interval = primary_retrain_interval_bars
        self.meta_retrain_interval = meta_retrain_interval_bars
        self.meta_label_lookback = meta_label_lookback_bars
        self.min_meta_labels = min_meta_labels_for_retrain
        self._bars_since_primary_retrain = 0
        self._bars_since_meta_retrain = 0
        self._primary_version = 0
        self._meta_version = 0

    def on_new_bar(self) -> dict[str, bool]:
        """Called every 5-min bar. Returns dict of what needs retraining."""
        self._bars_since_primary_retrain += 1
        self._bars_since_meta_retrain += 1

        actions = {
            "retrain_primary": False,
            "retrain_meta": False,
        }

        # Check if primary needs retraining
        if self._bars_since_primary_retrain >= self.primary_retrain_interval:
            actions["retrain_primary"] = True
            self._bars_since_primary_retrain = 0
            self._primary_version += 1

        # Check if meta needs retraining
        # Meta only retrains if: sufficient time has passed AND primary was recently retrained
        if (
            self._bars_since_meta_retrain >= self.meta_retrain_interval
            and self._primary_version > self._meta_version
        ):
            actions["retrain_meta"] = True
            self._bars_since_meta_retrain = 0
            self._meta_version = self._primary_version

        return actions


def retrain_meta_model_online(
    current_primary_model,
    features_buffer: pd.DataFrame,       # Rolling buffer of recent features
    close_buffer: pd.Series,             # Rolling buffer of recent prices
    high_buffer: pd.Series,
    low_buffer: pd.Series,
    volatility_buffer: pd.Series,
    meta_label_config: MetaLabelConfig,
    previous_meta_model=None,            # For warm-start comparison
) -> tuple:
    """Retrain meta-model using the CURRENT primary model's OOF predictions.

    This is the critical function: after the primary is retrained, we must
    regenerate OOF predictions using the NEW primary, generate fresh
    meta-labels, and retrain the meta-model.

    Steps:
    1. Generate OOF predictions from current primary (walk-forward on buffer)
    2. Generate meta-labels from OOF predictions + actual outcomes
    3. Construct meta-features
    4. Train new meta-model
    5. Validate: new meta-model should beat previous (or at least not be worse)
    """
    # Step 1: OOF predictions from current primary
    oof_proba, valid_mask = generate_oof_predictions_walk_forward(
        features=features_buffer,
        labels=_compute_ternary_labels(close_buffer),  # Actual ternary labels
        train_window=min(2016, len(features_buffer) // 3),  # 7 days or 1/3 of buffer
        test_window=288,
    )

    # Step 2: Extract sides and generate meta-labels
    sides = extract_side_from_primary(oof_proba[valid_mask])
    all_sides = np.zeros(len(features_buffer), dtype=np.int8)
    all_sides[valid_mask] = sides

    meta_labels, _ = generate_meta_labels_triple_barrier(
        primary_sides=all_sides,
        close_prices=close_buffer,
        high_prices=high_buffer,
        low_prices=low_buffer,
        volatility=volatility_buffer,
        config=meta_label_config,
    )

    # Step 3: Construct meta-features
    meta_features = construct_meta_features(
        primary_proba=oof_proba,
        primary_sides=all_sides,
        original_features=features_buffer,
        close_prices=close_buffer,
        meta_label_history=meta_labels,
        regime_probs=_get_regime_probs_for_buffer(features_buffer),
    )

    # Step 4: Train new meta-model
    new_meta_model = train_meta_model_logistic(
        meta_features=meta_features,
        meta_labels=meta_labels,
        valid_mask=valid_mask,
    )

    # Step 5: Validate
    # Quick out-of-sample check on last day of buffer
    val_start = len(features_buffer) - 288
    val_features = meta_features.iloc[val_start:]
    val_labels = meta_labels.iloc[val_start:]

    if previous_meta_model is not None:
        old_score = previous_meta_model.score(val_features.dropna(), val_labels[val_features.dropna().index])
        new_score = new_meta_model.score(val_features.dropna(), val_labels[val_features.dropna().index])

        if new_score < old_score - 0.05:  # Allow 5% degradation margin
            logger.warning(
                "meta_model_degradation",
                old_score=old_score,
                new_score=new_score,
                action="keeping_old_model",
            )
            return previous_meta_model, False

    logger.info(
        "meta_model_retrained",
        n_samples=int(valid_mask.sum()),
        meta_label_positive_rate=f"{meta_labels[valid_mask].mean():.3f}",
    )

    return new_meta_model, True
```

### Drift Detection for Meta-Model

```python
def detect_meta_model_drift(
    recent_meta_probs: np.ndarray,      # Last 288 bars (1 day) of meta predictions
    recent_outcomes: np.ndarray,         # Actual outcomes (1 = profitable, 0 = not)
    historical_calibration: float = 0.6, # Expected accuracy at p>0.55
) -> dict[str, float]:
    """Detect if the meta-model's predictions have drifted from reality.

    Triggers:
    1. Calibration drift: meta-model says p=0.7 but actual accuracy is 0.5
    2. Distribution drift: meta-model suddenly predicting much higher/lower
    3. Performance drift: rolling Sharpe of meta-filtered trades declining
    """
    # 1. Calibration check (Brier score)
    active = recent_meta_probs > 0.55
    if active.sum() < 20:
        return {"drift_detected": False, "reason": "insufficient_trades"}

    actual = recent_outcomes[active]
    predicted = recent_meta_probs[active]

    brier = float(((predicted - actual) ** 2).mean())
    # Brier > 0.25 means worse than random
    calibration_ok = brier < 0.25

    # 2. Mean prediction drift (should be ~0.55-0.65 for active trades)
    mean_pred = float(predicted.mean())
    pred_drift = abs(mean_pred - 0.60) > 0.10

    # 3. Actual accuracy vs expected
    actual_accuracy = float(actual.mean())
    accuracy_drift = actual_accuracy < historical_calibration - 0.10

    drift_detected = not calibration_ok or pred_drift or accuracy_drift

    return {
        "drift_detected": drift_detected,
        "brier_score": brier,
        "mean_prediction": mean_pred,
        "actual_accuracy": actual_accuracy,
        "calibration_ok": calibration_ok,
        "prediction_distribution_ok": not pred_drift,
        "accuracy_ok": not accuracy_drift,
    }
```

### Pitfalls

1. **Never retrain meta-model without retraining primary first.** If primary changes but meta-model does not, the meta-model is calibrated to the OLD primary's error patterns.

2. **Allow overlap period.** After retraining, run BOTH old and new meta-models in parallel for 1 day. If new model performs significantly worse, rollback.

3. **Version everything.** Track which primary model version each meta-model was trained on. Log this for debugging.

---

## 10. Meta-Labeling with Ensemble Primary

### The Problem

When the primary is an ensemble (LightGBM + CatBoost + GRU with logistic regression stacker), what serves as the "primary prediction"?

### Recommendation: Use the Stacking Meta-Learner Output

```python
def ensemble_to_primary_signal(
    lgbm_proba: np.ndarray,         # (n, 3) from LightGBM
    catboost_proba: np.ndarray,     # (n, 3) from CatBoost
    gru_features: np.ndarray,       # (n, hidden_dim) GRU hidden states
    stacking_model,                  # Logistic regression stacker
) -> tuple[np.ndarray, np.ndarray]:
    """Extract primary signal from ensemble for meta-labeling.

    The stacking meta-learner's output IS the primary prediction.
    Its predicted probability IS the primary confidence.

    Why not use individual model outputs?
    - Each model's probability is uncalibrated relative to the others
    - The stacker already optimally combines them
    - Meta-labeling on top of stacking is the "second layer" — it filters,
      while stacking combines
    """
    # Build stacking features
    stacking_features = np.hstack([
        lgbm_proba,           # 3 cols
        catboost_proba,       # 3 cols
        gru_features,         # hidden_dim cols
    ])

    # Stacker prediction = primary prediction
    primary_proba = stacking_model.predict_proba(stacking_features)
    primary_sides = extract_side_from_primary(primary_proba)

    return primary_proba, primary_sides
```

### Additional Ensemble Diversity Features for Meta-Model

The ensemble provides unique meta-features that a single primary model cannot:

```python
def ensemble_disagreement_features(
    lgbm_proba: np.ndarray,
    catboost_proba: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute ensemble disagreement features for the meta-model.

    Ensemble disagreement is a POWERFUL meta-feature because:
    - When all models agree, the prediction is more reliable
    - When models disagree, the market is ambiguous
    """
    # Predicted class for each model
    lgbm_class = lgbm_proba.argmax(axis=1)
    catboost_class = catboost_proba.argmax(axis=1)

    # Binary agreement
    agreement = (lgbm_class == catboost_class).astype(float)

    # Probability distance (how different are the probability vectors?)
    prob_distance = np.sqrt(((lgbm_proba - catboost_proba) ** 2).sum(axis=1))

    # Max probability variance across models (per class)
    stacked = np.stack([lgbm_proba, catboost_proba], axis=0)  # (2, n, 3)
    prob_variance = stacked.var(axis=0).max(axis=1)  # Max variance across classes

    return {
        "ensemble_agreement": agreement,
        "ensemble_prob_distance": prob_distance,
        "ensemble_prob_variance": prob_variance,
    }
```

### Alternative: Meta-Label Each Model Separately

An alternative architecture meta-labels each base model independently, then combines:

```
LightGBM → Meta-Model A → sized_position_lgbm
CatBoost → Meta-Model B → sized_position_catboost
GRU      → Meta-Model C → sized_position_gru

Final position = weighted_average(sized_position_lgbm, sized_position_catboost, sized_position_gru)
```

**This is NOT recommended** because:
1. It requires 3 separate meta-models, tripling the leakage risk
2. The weight between models should be learned by the stacker, not duplicated
3. It loses the ensemble diversity signal (agreement/disagreement)

---

## 11. Stacking vs Meta-Labeling

### Fundamental Difference

| Aspect | Stacking | Meta-Labeling |
|--------|----------|---------------|
| **Purpose** | Combine diverse predictions into better prediction | Filter predictions, deciding whether to act |
| **Output** | A new prediction (direction + confidence) | A binary decision (trade / don't trade) + bet size |
| **Target** | Same as base models (up/flat/down) | Different: "was base model correct?" (binary) |
| **When it helps** | When base models have uncorrelated errors | When base model has decent recall but poor precision |
| **Position in pipeline** | Replaces individual model outputs | Sits on top of the ensemble output |

### Architecture: Using BOTH Stacking and Meta-Labeling

They serve different purposes and should be used together:

```
Layer 0 (Base Models):
    LightGBM → proba_lgbm
    CatBoost → proba_catboost
    GRU      → hidden_features
         │
         ▼
Layer 1 (Stacking):
    Logistic Regression Meta-Learner
    Input: proba_lgbm + proba_catboost + hidden_features
    Output: ensemble_proba (direction + confidence)
    Purpose: Optimally COMBINE diverse predictions
         │
         ▼
Layer 2 (Meta-Labeling):
    Logistic Regression (or shallow LightGBM)
    Input: ensemble_confidence + volatility + rolling_accuracy + regime + time + disagreement
    Output: P(trade will be profitable)
    Purpose: FILTER bad predictions + SIZE good ones
         │
         ▼
Layer 3 (Bet Sizing):
    Lopez de Prado formula or quarter-Kelly
    Input: side (from Layer 1) + P(profitable) (from Layer 2)
    Output: final position = side * bet_size
```

### Why This Architecture Works

1. **Stacking improves directional accuracy** by combining models with different inductive biases (gradient boosting vs neural nets)
2. **Meta-labeling improves risk-adjusted return** by filtering trades where the ensemble is likely wrong
3. **They operate on different information**: stacking uses raw feature predictions, meta-labeling uses confidence/regime/volatility

### Pitfalls

1. **Do not stack the meta-model.** The meta-model is already the last layer. Adding another meta-meta-model is overfitting.
2. **Do not use stacker features as meta-model input.** The meta-model should not see the raw stacking features (lgbm_proba, catboost_proba). It should see the ENSEMBLE output + context features.
3. **The stacker and meta-model have DIFFERENT targets.** Stacker target = ternary direction. Meta-model target = binary (profitable/not).

---

## 12. Complete Meta-Labeling Pipeline

### End-to-End Implementation

```python
"""
Complete meta-labeling pipeline for ep2-crypto.

Modules:
    meta_labeling/
        config.py       - Configuration dataclasses
        labels.py       - Meta-label generation (simple + triple barrier)
        features.py     - Meta-feature construction
        model.py        - Meta-model training and inference
        bet_sizing.py   - Probability-to-position mapping
        evaluation.py   - A/B evaluation framework
        pipeline.py     - End-to-end orchestration
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import structlog
from dataclasses import dataclass, field
from pathlib import Path
from scipy.stats import norm
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

logger = structlog.get_logger()


# ============================================================
# CONFIG
# ============================================================

@dataclass(frozen=True)
class MetaLabelPipelineConfig:
    """Full configuration for the meta-labeling pipeline."""

    # Meta-label generation
    transaction_cost_bps: float = 8.0
    min_return_bps: float = 5.0
    use_triple_barrier: bool = True
    pt_multiplier: float = 2.0
    sl_multiplier: float = 1.0
    max_holding_bars: int = 6
    volatility_lookback: int = 30     # Bars for rolling volatility

    # Primary model OOF
    oof_train_window: int = 4032      # 14 days
    oof_test_window: int = 288        # 1 day
    oof_purge_gap: int = 60           # 5 hours (max feature lookback)
    oof_embargo: int = 12             # 1 hour

    # Meta-model
    meta_model_type: str = "logistic"  # "logistic" or "lgbm"
    min_directional_prob: float = 0.35  # Primary confidence threshold for side

    # Bet sizing
    max_bet_fraction: float = 0.05
    min_meta_prob: float = 0.55        # Below this = no trade
    use_kelly: bool = True

    # Retraining
    primary_retrain_bars: int = 48     # 4 hours
    meta_retrain_bars: int = 96        # 8 hours
    meta_label_history_bars: int = 4032  # 14 days lookback


# ============================================================
# PIPELINE ORCHESTRATOR
# ============================================================

class MetaLabelingPipeline:
    """End-to-end meta-labeling pipeline.

    Usage for TRAINING:
        pipeline = MetaLabelingPipeline(config)
        pipeline.train(features, close, high, low, ternary_labels, regime_probs)

    Usage for INFERENCE:
        result = pipeline.predict(features, close, regime_probs)
        # result.position = side * bet_size
    """

    def __init__(self, config: MetaLabelPipelineConfig) -> None:
        self.config = config
        self.primary_model: lgb.Booster | None = None
        self.meta_model: Pipeline | lgb.Booster | None = None
        self._meta_label_history: pd.Series | None = None
        self._avg_win: float = 0.001
        self._avg_loss: float = -0.001

    def train(
        self,
        features: pd.DataFrame,
        close: pd.Series,
        high: pd.Series,
        low: pd.Series,
        ternary_labels: pd.Series,
        regime_probs: np.ndarray,
    ) -> MetaLabelingEvaluation:
        """Full training pipeline.

        Steps:
        1. Train primary model via walk-forward, collect OOF predictions
        2. Extract sides from OOF predictions
        3. Generate meta-labels (was primary correct + profitable?)
        4. Construct meta-features
        5. Train meta-model
        6. Evaluate improvement
        """
        logger.info("meta_labeling_training_start", n_samples=len(features))

        # --- Step 1: Generate OOF predictions from primary ---
        oof_proba, valid_mask = generate_oof_predictions_walk_forward(
            features=features,
            labels=ternary_labels,
            train_window=self.config.oof_train_window,
            test_window=self.config.oof_test_window,
            purge_gap=self.config.oof_purge_gap,
            embargo_bars=self.config.oof_embargo,
        )

        # Also train a final primary model on all data (for live inference)
        train_data = lgb.Dataset(features, label=ternary_labels)
        self.primary_model = lgb.train(
            PRIMARY_LGBM_PARAMS,
            train_data,
            num_boost_round=300,
        )

        # --- Step 2: Extract sides ---
        sides = np.zeros(len(features), dtype=np.int8)
        valid_proba = oof_proba[valid_mask]
        sides[valid_mask] = extract_side_from_primary(
            valid_proba, self.config.min_directional_prob
        )

        # --- Step 3: Generate meta-labels ---
        volatility = close.pct_change().rolling(self.config.volatility_lookback).std()

        if self.config.use_triple_barrier:
            meta_labels, touch_times = generate_meta_labels_triple_barrier(
                primary_sides=sides,
                close_prices=close,
                high_prices=high,
                low_prices=low,
                volatility=volatility,
                config=MetaLabelConfig(
                    transaction_cost_bps=self.config.transaction_cost_bps,
                    min_return_bps=self.config.min_return_bps,
                    pt_multiplier=self.config.pt_multiplier,
                    sl_multiplier=self.config.sl_multiplier,
                    max_holding_bars=self.config.max_holding_bars,
                ),
            )
        else:
            meta_labels = generate_meta_labels(
                primary_sides=sides,
                close_prices=close,
                config=MetaLabelConfig(
                    transaction_cost_bps=self.config.transaction_cost_bps,
                    min_return_bps=self.config.min_return_bps,
                ),
            )

        self._meta_label_history = meta_labels

        # --- Step 4: Construct meta-features ---
        meta_features = construct_meta_features(
            primary_proba=oof_proba,
            primary_sides=sides,
            original_features=features,
            close_prices=close,
            meta_label_history=meta_labels,
            regime_probs=regime_probs,
        )

        # --- Step 5: Train meta-model ---
        if self.config.meta_model_type == "logistic":
            self.meta_model = train_meta_model_logistic(
                meta_features=meta_features,
                meta_labels=meta_labels,
                valid_mask=valid_mask & (sides != 0),
            )
        else:
            self.meta_model = train_meta_model_lgbm(
                meta_features=meta_features,
                meta_labels=meta_labels,
                valid_mask=valid_mask & (sides != 0),
            )

        # Compute avg win/loss for Kelly sizing
        directional = valid_mask & (sides != 0)
        if directional.any():
            forward_ret = close.pct_change().shift(-1)
            trade_returns = sides[directional] * forward_ret.values[directional]
            cost = self.config.transaction_cost_bps / 10_000
            net_returns = trade_returns - cost
            wins = net_returns[net_returns > 0]
            losses = net_returns[net_returns <= 0]
            self._avg_win = float(wins.mean()) if len(wins) > 0 else 0.001
            self._avg_loss = float(losses.mean()) if len(losses) > 0 else -0.001

        # --- Step 6: Evaluate ---
        # Get meta-model predictions on OOF data for evaluation
        eval_mask = valid_mask & (sides != 0) & (~meta_features.isna().any(axis=1))
        eval_features = meta_features.loc[eval_mask].dropna()

        if isinstance(self.meta_model, Pipeline):
            meta_proba = self.meta_model.predict_proba(eval_features)[:, 1]
        else:
            meta_proba = self.meta_model.predict(eval_features)

        # Compute bet sizes
        if self.config.use_kelly:
            bet_sizes_eval = quarter_kelly_bet_sizing(
                meta_proba=meta_proba,
                avg_win_return=self._avg_win,
                avg_loss_return=self._avg_loss,
                max_bet_fraction=self.config.max_bet_fraction,
                min_prob_threshold=self.config.min_meta_prob,
            )
        else:
            bet_sizes_eval = meta_prob_to_bet_size(
                meta_proba=meta_proba,
                max_bet_fraction=self.config.max_bet_fraction,
                min_prob_threshold=self.config.min_meta_prob,
            )

        # Reconstruct full-length arrays for evaluation
        full_meta_positions = np.zeros(len(features))
        eval_indices = np.where(eval_mask)[0]
        valid_eval = eval_features.index
        for k, idx in enumerate(eval_indices[:len(bet_sizes_eval)]):
            full_meta_positions[idx] = sides[idx] * bet_sizes_eval[k]

        forward_returns = close.pct_change().shift(-1).fillna(0)

        evaluation = evaluate_meta_labeling(
            primary_sides=sides,
            meta_positions=full_meta_positions,
            forward_returns=forward_returns,
            trading_cost_bps=self.config.transaction_cost_bps,
        )

        logger.info(
            "meta_labeling_training_complete",
            sharpe_before=evaluation.sharpe_before,
            sharpe_after=evaluation.sharpe_after,
            sharpe_improvement=f"{evaluation.sharpe_improvement_pct:.1f}%",
            precision_improvement=f"{evaluation.precision_improvement_pct:.1f}%",
            trade_reduction=f"{evaluation.trade_reduction_pct:.1f}%",
            significant=evaluation.is_statistically_significant,
        )

        return evaluation

    def predict(
        self,
        features: pd.DataFrame,
        close: pd.Series,
        regime_probs: np.ndarray,
    ) -> pd.DataFrame:
        """Live inference: primary prediction → meta-filtering → bet sizing.

        Returns DataFrame with:
            side: +1/-1/0 from primary
            meta_prob: P(trade profitable) from meta-model
            bet_size: position fraction from bet sizing
            position: side * bet_size (the actual trading signal)
        """
        if self.primary_model is None or self.meta_model is None:
            raise RuntimeError("Pipeline not trained. Call train() first.")

        # Primary prediction
        primary_proba = self.primary_model.predict(features)
        sides = extract_side_from_primary(
            primary_proba, self.config.min_directional_prob
        )

        # Meta-features
        meta_feats = construct_meta_features(
            primary_proba=primary_proba,
            primary_sides=sides,
            original_features=features,
            close_prices=close,
            meta_label_history=(
                self._meta_label_history
                if self._meta_label_history is not None
                else pd.Series(0.5, index=close.index)
            ),
            regime_probs=regime_probs,
        )

        # Meta-model prediction
        clean_feats = meta_feats.fillna(0)
        if isinstance(self.meta_model, Pipeline):
            meta_proba = self.meta_model.predict_proba(clean_feats)[:, 1]
        else:
            meta_proba = self.meta_model.predict(clean_feats)

        # Bet sizing
        if self.config.use_kelly:
            bet_sizes = quarter_kelly_bet_sizing(
                meta_proba=meta_proba,
                avg_win_return=self._avg_win,
                avg_loss_return=self._avg_loss,
                max_bet_fraction=self.config.max_bet_fraction,
                min_prob_threshold=self.config.min_meta_prob,
            )
        else:
            bet_sizes = meta_prob_to_bet_size(
                meta_proba=meta_proba,
                max_bet_fraction=self.config.max_bet_fraction,
                min_prob_threshold=self.config.min_meta_prob,
            )

        # Combine
        positions = combine_side_and_size(sides, bet_sizes)

        result = pd.DataFrame(index=features.index)
        result["side"] = sides
        result["meta_prob"] = meta_proba
        result["bet_size"] = bet_sizes
        result["position"] = positions

        return result
```

### Integration with ep2-crypto Architecture

This pipeline fits into the existing architecture at Phase 8 (Confidence Gating):

```
Phase 7 (Models) → Stacking Ensemble output
                          │
                          ▼
Phase 8 (Confidence Gating):
    Step 1: Isotonic calibration (calibrate ensemble probabilities)
    Step 2: META-LABELING GATE  ← THIS PIPELINE
    Step 3: Ensemble agreement check
    Step 4: Conformal prediction gate
    Step 5: Signal filters (vol, regime, liquidity)
    Step 6: Adaptive confidence threshold
    Step 7: Drawdown gate
    Step 8: Quarter-Kelly position sizing
                          │
                          ▼
Phase 9+ (Execution)
```

The meta-labeling pipeline replaces steps 2 AND 8 — it IS the filter AND the sizer. The other gates (conformal, drawdown, signal filters) remain as additional safety layers.

---

## Summary of Key Decisions for ep2-crypto

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Primary model | Full stacking ensemble | Best directional accuracy = best recall |
| Primary tuning | High recall, lower precision | Meta-model improves precision |
| Meta-label type | Triple barrier with costs | Realistic profit evaluation |
| Meta-features | 22 features (confidence + vol + accuracy + regime + time) | Ablation-tested optimal set |
| Leakage prevention | Purged walk-forward OOF | Gold standard for time series |
| Meta-model | Logistic regression (primary) or shallow LightGBM (alternative) | Simple model for simple task |
| Bet sizing | Quarter-Kelly from meta-probability | Per RESEARCH_SYNTHESIS.md recommendation |
| Retraining | Meta-model retrains 4-8h AFTER primary retrains | Synchronization requirement |
| Barrier calibration | Regime-dependent pt/sl multipliers | Different regimes need different barriers |
| Stacking + meta-labeling | YES, both — stacking combines, meta-labeling filters | Complementary, not redundant |

## Expected End-to-End Performance Impact

Based on the Hudson & Thames JFDS research and crypto-specific empirical results:

| Scenario | Sharpe Without | Sharpe With | Improvement |
|----------|---------------|-------------|-------------|
| Conservative (LR meta, simple barriers) | 1.0 | 1.4-1.7 | +40-70% |
| Moderate (LR meta, triple barrier, regime barriers) | 1.0 | 1.6-2.2 | +60-120% |
| Aggressive (LightGBM meta, full feature set) | 1.0 | 1.8-2.5 | +80-150% |

These improvements compound with other gating layers (conformal, drawdown gate, signal filters) for the full 2-4x Sharpe improvement cited in RESEARCH_SYNTHESIS.md.

---

## Sources

- [Does Meta Labeling Add to Signal Efficacy? - Hudson & Thames](https://hudsonthames.org/does-meta-labeling-add-to-signal-efficacy-triple-barrier-method/)
- [Singh & Joubert JFDS Paper (PDF)](https://hudsonthames.org/wp-content/uploads/2022/04/Does-Meta-Labeling-Add-to-Signal-Efficacy.pdf)
- [Meta-Labeling: Theory and Framework - JFDS](https://jfds.pm-research.com/content/early/2022/06/23/jfds.2022.1.098)
- [Meta-Labeling - Wikipedia](https://en.wikipedia.org/wiki/Meta-Labeling)
- [Triple-Barrier and Meta-Labelling - mlfinlab docs](https://www.mlfinlab.com/en/latest/labeling/tb_meta_labeling.html)
- [Bet Sizing in ML - mlfinlab docs](https://www.mlfinlab.com/en/latest/bet_sizing/bet_sizing.html)
- [Hudson & Thames meta-labeling GitHub](https://github.com/hudson-and-thames/meta-labeling)
- [Purged Cross-Validation - Wikipedia](https://en.wikipedia.org/wiki/Purged_cross-validation)
- [CPCV Method - Towards AI](https://towardsai.net/p/l/the-combinatorial-purged-cross-validation-method)
- [Meta-labeling in Cryptocurrencies Market - Medium](https://medium.com/@liangnguyen612/meta-labeling-in-cryptocurrencies-market-95f761410fac)
- [Corrective AI to Improve Trading Systems - Don Brady](https://www.comintel.com/meetup/DonBrady/Meta-Labeling.pdf)
- [Background: Meta-Labeling Method - AI Quantitative Trading](https://www.waylandz.com/quant-book-en/Meta-Labeling-Method/)
- [Why Meta-Labeling Is Not a Silver Bullet - QuantConnect](https://www.quantconnect.com/forum/discussion/14706/why-meta-labeling-is-not-a-silver-bullet/)
- [Advances in Financial Machine Learning - Reasonable Deviations](https://reasonabledeviations.com/notes/adv_fin_ml/)
- [Chapter 10 Bet Sizing - AFML O'Reilly](https://www.oreilly.com/library/view/advances-in-financial/9781119482086/c10.xhtml)
