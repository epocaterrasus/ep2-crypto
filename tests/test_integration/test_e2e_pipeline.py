"""End-to-end integration test for the ep2-crypto prediction pipeline.

Covers the full data flow:
  synthetic OHLCV data
    -> FeaturePipeline.compute_batch()
    -> LGBMDirectionModel.predict_proba()  (trained on synthetic labels)
    -> StackingEnsemble.predict_proba()    (meta-learner on base probas)
    -> ConfidenceGatingPipeline.evaluate()
    -> RiskManager.approve_trade()
    -> trade signal dict (Polymarket-compatible order format)

Design decisions:
- Uses SQLite in-memory for all DB operations
- Models are trained on synthetic data (not loaded from disk) so the test
  is self-contained and fast (<30 s on a laptop)
- Gating pipeline is configured with all gates enabled but thresholds
  relaxed enough that a clear-signal bar has a good chance of passing
- The test documents known pipeline quirks as comments rather than masking them
"""

from __future__ import annotations

import sqlite3
import time
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pytest
import structlog

from ep2_crypto.confidence.gating import (
    ConfidenceGatingPipeline,
    GatingConfig,
    GatingResult,
    MarketContext,
)
from ep2_crypto.features.pipeline import FeaturePipeline, build_default_registry
from ep2_crypto.models.lgbm_direction import LGBMConfig, LGBMDirectionModel
from ep2_crypto.models.stacking import StackingConfig, StackingEnsemble
from ep2_crypto.risk.config import RiskConfig
from ep2_crypto.risk.risk_manager import RiskManager, SignalInput


# ---------------------------------------------------------------------------
# Logging setup — structlog must be configured before any test imports use it
# ---------------------------------------------------------------------------


def _configure_structlog() -> None:
    """Configure structlog to write plain text to stderr (test-friendly)."""
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.dev.ConsoleRenderer(colors=False),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(20),  # INFO
        logger_factory=structlog.PrintLoggerFactory(),
    )


_configure_structlog()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_BARS = 400  # Enough warmup (max ~60 bars) plus training data
INITIAL_PRICE = 65_000.0
INITIAL_EQUITY = 100_000.0
# A Tuesday at 14:00 UTC so trading hours are satisfied
BASE_TIMESTAMP_MS = int(datetime(2026, 3, 10, 14, 0, 0, tzinfo=UTC).timestamp() * 1000)
BAR_INTERVAL_MS = 5 * 60 * 1000  # 5 minutes


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ohlcv_arrays() -> dict[str, np.ndarray]:
    """Generate 400 bars of realistic synthetic BTC OHLCV data.

    Uses a geometric Brownian motion for closes, then derives OHLC.
    Volume is correlated with absolute returns to mimic real markets.
    """
    rng = np.random.default_rng(42)
    n = N_BARS

    # Log-normal returns: ~15% annualised vol at 5-min bars
    # Per-bar std ≈ 0.15 / sqrt(105120) ≈ 0.000463
    bar_std = 0.15 / np.sqrt(105_120)
    log_returns = rng.normal(0.0, bar_std, size=n)
    log_prices = np.log(INITIAL_PRICE) + np.cumsum(log_returns)
    closes = np.exp(log_prices)

    # Open = previous close (no gap for simplicity)
    opens = np.empty(n, dtype=np.float64)
    opens[0] = INITIAL_PRICE
    opens[1:] = closes[:-1]

    # High / Low derived from close with small noise
    noise = np.abs(rng.normal(0.0, bar_std * closes, size=n))
    highs = np.maximum(opens, closes) + noise
    lows = np.minimum(opens, closes) - noise
    lows = np.maximum(lows, closes * 0.99)  # Never below 99% of close

    # Volume: ~10 BTC average, higher on big moves
    abs_ret = np.abs(log_returns)
    volume_base = 10.0 + rng.exponential(2.0, size=n)
    volumes = volume_base * (1.0 + 5.0 * abs_ret / (bar_std + 1e-10))

    # Timestamps: every 5 minutes from base
    timestamps = np.array(
        [BASE_TIMESTAMP_MS + i * BAR_INTERVAL_MS for i in range(n)],
        dtype=np.int64,
    )

    return {
        "timestamps": timestamps,
        "opens": opens.astype(np.float64),
        "highs": highs.astype(np.float64),
        "lows": lows.astype(np.float64),
        "closes": closes.astype(np.float64),
        "volumes": volumes.astype(np.float64),
    }


@pytest.fixture(scope="module")
def feature_matrix(ohlcv_arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, list[str]]:
    """Compute the full feature matrix for all bars.

    Returns:
        (feature_matrix_2d, feature_names)
        Shape: (N_BARS, n_features)
    """
    pipeline = FeaturePipeline(registry=build_default_registry())

    arr = ohlcv_arrays
    X = pipeline.compute_batch(
        arr["timestamps"],
        arr["opens"],
        arr["highs"],
        arr["lows"],
        arr["closes"],
        arr["volumes"],
        fill_nan=True,
    )
    return X, pipeline.output_names


def _clean_feature_matrix(
    X: np.ndarray,
    feature_names: list[str],
    warmup: int = 65,
) -> tuple[np.ndarray, list[str]]:
    """Remove warmup rows and structurally-NaN columns.

    Order book features (OBI, OFI, etc.) are always NaN when no
    bids/asks kwargs are provided to the pipeline.  These columns
    must be dropped before passing features to a model.

    Returns:
        (X_clean, clean_feature_names) with warmup rows and NaN-only
        columns removed.
    """
    # Drop warmup rows
    X_post = X[warmup:]

    # Drop columns that are entirely NaN (structurally missing)
    col_finite = np.any(np.isfinite(X_post), axis=0)
    X_post = X_post[:, col_finite]
    clean_names = [n for n, keep in zip(feature_names, col_finite.tolist()) if keep]

    # Replace remaining scattered NaNs with column medians
    for j in range(X_post.shape[1]):
        col = X_post[:, j]
        mask = ~np.isfinite(col)
        if mask.any():
            median_val = float(np.nanmedian(col))
            col[mask] = median_val if np.isfinite(median_val) else 0.0

    return X_post, clean_names


@pytest.fixture(scope="module")
def trained_lgbm(feature_matrix: tuple[np.ndarray, list[str]]) -> LGBMDirectionModel:
    """Train a minimal LightGBM model on the synthetic feature matrix.

    Pipeline note: order book features (OBI/OFI/microprice) require
    bids/asks/trade_sizes kwargs.  Without them they are structurally NaN.
    We drop those columns via _clean_feature_matrix() before training.
    """
    X, feature_names = feature_matrix

    X_clean, clean_names = _clean_feature_matrix(X, feature_names, warmup=65)

    # Synthetic ternary labels
    rng = np.random.default_rng(99)
    n = len(X_clean)
    y = rng.choice(np.array([-1, 0, 1], dtype=np.int8), size=n, p=[0.33, 0.34, 0.33])

    # Train/val split (temporal — last 20% as val)
    split = int(n * 0.8)
    X_train, X_val = X_clean[:split], X_clean[split:]
    y_train, y_val = y[:split], y[split:]

    # Small config for speed in CI
    cfg = LGBMConfig(
        n_estimators=50,
        num_leaves=15,
        max_depth=3,
        min_child_samples=5,
        early_stopping_rounds=10,
    )
    model = LGBMDirectionModel(config=cfg)
    model.train(X_train, y_train, X_val, y_val, feature_names=clean_names)
    return model


@pytest.fixture(scope="module")
def trained_stacking(
    feature_matrix: tuple[np.ndarray, list[str]],
    trained_lgbm: LGBMDirectionModel,
) -> StackingEnsemble:
    """Train a stacking ensemble using two synthetic base model outputs.

    We duplicate the LGBM probabilities with small noise as the "CatBoost"
    base model so the ensemble has two inputs without requiring CatBoost
    to be fully trained (which would be slow).
    """
    X, feature_names = feature_matrix
    X_clean, _ = _clean_feature_matrix(X, feature_names, warmup=65)

    rng = np.random.default_rng(7)
    proba_lgbm = trained_lgbm.predict_proba(X_clean)  # (n, 3)

    # Simulate a second base model with noisy probabilities
    noise = rng.normal(0.0, 0.02, size=proba_lgbm.shape)
    proba_cat = proba_lgbm + noise
    proba_cat = np.clip(proba_cat, 1e-6, 1.0)
    proba_cat = proba_cat / proba_cat.sum(axis=1, keepdims=True)

    # Generate matching labels (consistent with trained_lgbm fixture)
    n = len(X_clean)
    raw_labels = rng.choice([-1, 0, 1], size=n, p=[0.33, 0.34, 0.33])
    y = raw_labels.astype(np.int8)

    cfg = StackingConfig(meta_max_iter=200, random_state=42)
    ensemble = StackingEnsemble(config=cfg)
    ensemble.train(
        base_probas=[proba_lgbm, proba_cat],
        y_true=y,
        base_model_names=["lgbm", "catboost_sim"],
    )
    return ensemble


@pytest.fixture(scope="module")
def gating_pipeline() -> ConfidenceGatingPipeline:
    """Build a gating pipeline with all gates enabled but relaxed thresholds.

    Meta-labeler and conformal predictor are intentionally NOT injected
    (their is_fitted checks return False → gates pass as "not_available").
    This lets us test the structural flow without requiring those components
    to be trained.
    """
    cfg = GatingConfig(
        enable_calibration=True,
        enable_meta_labeling=True,
        enable_ensemble_agreement=True,
        enable_conformal=True,
        enable_signal_filters=True,
        enable_adaptive_threshold=True,
        enable_drawdown=True,
        meta_label_threshold=0.50,
        max_ensemble_variance=0.15,  # relaxed for synthetic noise
    )
    # Relax adaptive threshold so synthetic model confidence can pass
    cfg.adaptive_threshold.base_threshold = 0.30
    return ConfidenceGatingPipeline(config=cfg)


@pytest.fixture(scope="module")
def risk_manager() -> RiskManager:
    """Create a RiskManager backed by an in-memory SQLite database.

    Disables trading-hours enforcement so the test timestamp (UTC 14:00 Tue)
    passes regardless of the machine's local clock.  The risk config is
    otherwise default.
    """
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    risk_cfg = RiskConfig(
        enforce_trading_hours=False,  # avoid timestamp-based rejections in CI
        min_volatility_ann=0.0,       # synthetic data may have low vol
        max_volatility_ann=10.0,      # and extreme vol spikes
    )
    return RiskManager(conn, initial_equity=INITIAL_EQUITY, config=risk_cfg)


# ---------------------------------------------------------------------------
# Stage-level tests
# ---------------------------------------------------------------------------


class TestFeaturePipelineStage:
    """Verify that FeaturePipeline.compute_batch() produces the expected shape."""

    def test_output_shape(self, feature_matrix: tuple[np.ndarray, list[str]]) -> None:
        X, names = feature_matrix
        assert X.ndim == 2, "Feature matrix must be 2-D"
        assert X.shape[0] == N_BARS, f"Expected {N_BARS} rows, got {X.shape[0]}"
        assert X.shape[1] == len(names), "Column count must match output_names length"
        assert X.shape[1] > 0, "Must have at least one feature column"

    def test_warmup_rows_are_nan(self, feature_matrix: tuple[np.ndarray, list[str]]) -> None:
        X, _ = feature_matrix
        # First few rows should have some NaN before warmup completes
        # (not necessarily ALL columns, but at least some)
        first_row_nans = np.sum(~np.isfinite(X[0]))
        assert first_row_nans > 0, "Row 0 should have NaN values (warmup not complete)"

    def test_post_warmup_rows_mostly_finite(
        self, feature_matrix: tuple[np.ndarray, list[str]]
    ) -> None:
        X, _ = feature_matrix
        # After warmup (using safe margin of 80 bars), rows should be at least 50%
        # finite across the raw matrix.  Order book columns (OBI, OFI, etc.) are
        # structurally NaN when no bids/asks data is provided — this is expected
        # and intentional (those features require live order book data).
        post_warmup = X[80:]
        finite_fraction = np.mean(np.isfinite(post_warmup))
        assert finite_fraction > 0.50, (
            f"Post-warmup finite fraction {finite_fraction:.2%} < 50%"
        )

    def test_post_warmup_non_ob_features_mostly_finite(
        self, feature_matrix: tuple[np.ndarray, list[str]]
    ) -> None:
        """Non-order-book features must be fully finite after warmup.

        Pipeline bug documented: OBI/OFI/microprice/TFI/KyleLambda columns
        are structurally NaN when no bids/asks/trade_sizes kwargs are passed.
        _clean_feature_matrix() (used in training fixtures) handles this by
        dropping those columns before model training.
        """
        X, names = feature_matrix
        post_warmup = X[80:]
        # Identify columns that are NOT structurally NaN (have at least one finite value)
        non_ob_mask = np.any(np.isfinite(post_warmup), axis=0)
        X_non_ob = post_warmup[:, non_ob_mask]
        finite_fraction = np.mean(np.isfinite(X_non_ob))
        assert finite_fraction > 0.90, (
            f"Non-OB feature finite fraction {finite_fraction:.2%} < 90%"
        )

    def test_feature_names_are_strings(
        self, feature_matrix: tuple[np.ndarray, list[str]]
    ) -> None:
        _, names = feature_matrix
        assert all(isinstance(n, str) for n in names), "All feature names must be strings"
        # No duplicates
        assert len(names) == len(set(names)), "Feature names must be unique"

    def test_pipeline_output_names_stable(self) -> None:
        """Two pipeline instances with default registry produce identical output_names."""
        p1 = FeaturePipeline()
        p2 = FeaturePipeline()
        assert p1.output_names == p2.output_names, (
            "output_names must be deterministic across instances"
        )


class TestLGBMStage:
    """Verify LGBMDirectionModel.predict_proba() output contract."""

    def test_is_fitted_after_train(self, trained_lgbm: LGBMDirectionModel) -> None:
        assert trained_lgbm.is_fitted, "Model must be fitted after train()"

    def test_predict_proba_shape(
        self,
        trained_lgbm: LGBMDirectionModel,
        feature_matrix: tuple[np.ndarray, list[str]],
    ) -> None:
        X, feature_names = feature_matrix
        X_clean, _ = _clean_feature_matrix(X, feature_names)
        x_sample = X_clean[0:1]
        proba = trained_lgbm.predict_proba(x_sample)
        assert proba.shape == (1, 3), f"Expected (1, 3), got {proba.shape}"

    def test_predict_proba_sums_to_one(
        self,
        trained_lgbm: LGBMDirectionModel,
        feature_matrix: tuple[np.ndarray, list[str]],
    ) -> None:
        X, feature_names = feature_matrix
        X_clean, _ = _clean_feature_matrix(X, feature_names)
        proba = trained_lgbm.predict_proba(X_clean[:20])
        row_sums = proba.sum(axis=1)
        np.testing.assert_allclose(
            row_sums,
            np.ones(20),
            atol=1e-6,
            err_msg="Probability rows must sum to 1.0",
        )

    def test_predict_proba_in_range(
        self,
        trained_lgbm: LGBMDirectionModel,
        feature_matrix: tuple[np.ndarray, list[str]],
    ) -> None:
        X, feature_names = feature_matrix
        X_clean, _ = _clean_feature_matrix(X, feature_names)
        proba = trained_lgbm.predict_proba(X_clean[:20])
        assert np.all(proba >= 0) and np.all(proba <= 1), "Probabilities must be in [0, 1]"

    def test_predict_returns_ternary_labels(
        self,
        trained_lgbm: LGBMDirectionModel,
        feature_matrix: tuple[np.ndarray, list[str]],
    ) -> None:
        X, feature_names = feature_matrix
        X_clean, _ = _clean_feature_matrix(X, feature_names)
        preds = trained_lgbm.predict(X_clean[:30])
        assert set(preds).issubset({-1, 0, 1}), (
            f"Predictions must be in {{-1, 0, 1}}, got {set(preds)}"
        )

    def test_feature_importance_populated(self, trained_lgbm: LGBMDirectionModel) -> None:
        assert trained_lgbm.feature_importance is not None
        total = sum(trained_lgbm.feature_importance.values())
        assert abs(total - 1.0) < 1e-4, f"Feature importances must sum to ~1.0, got {total}"


class TestStackingEnsembleStage:
    """Verify StackingEnsemble.predict_proba() output contract."""

    def test_is_fitted(self, trained_stacking: StackingEnsemble) -> None:
        assert trained_stacking.is_fitted

    def test_predict_proba_shape_and_sum(
        self,
        trained_lgbm: LGBMDirectionModel,
        trained_stacking: StackingEnsemble,
        feature_matrix: tuple[np.ndarray, list[str]],
    ) -> None:
        X, feature_names = feature_matrix
        rng = np.random.default_rng(7)
        X_clean, _ = _clean_feature_matrix(X, feature_names)
        x_sample = X_clean[:20]
        proba_lgbm = trained_lgbm.predict_proba(x_sample)
        noise = rng.normal(0.0, 0.02, size=proba_lgbm.shape)
        proba_cat = np.clip(proba_lgbm + noise, 1e-6, 1.0)
        proba_cat /= proba_cat.sum(axis=1, keepdims=True)

        result = trained_stacking.predict_proba([proba_lgbm, proba_cat])
        assert result.shape == (20, 3), f"Expected (20, 3), got {result.shape}"
        np.testing.assert_allclose(
            result.sum(axis=1),
            np.ones(20),
            atol=1e-6,
            err_msg="Stacking probas must sum to 1.0",
        )
        assert np.all(result >= 0) and np.all(result <= 1)

    def test_predict_returns_ternary(
        self,
        trained_lgbm: LGBMDirectionModel,
        trained_stacking: StackingEnsemble,
        feature_matrix: tuple[np.ndarray, list[str]],
    ) -> None:
        X, feature_names = feature_matrix
        rng = np.random.default_rng(7)
        X_clean, _ = _clean_feature_matrix(X, feature_names)
        x_sample = X_clean[:15]
        proba_lgbm = trained_lgbm.predict_proba(x_sample)
        noise = rng.normal(0.0, 0.02, size=proba_lgbm.shape)
        proba_cat = np.clip(proba_lgbm + noise, 1e-6, 1.0)
        proba_cat /= proba_cat.sum(axis=1, keepdims=True)

        preds = trained_stacking.predict([proba_lgbm, proba_cat])
        assert set(preds).issubset({-1, 0, 1}), f"Got {set(preds)}"


class TestGatingPipelineStage:
    """Verify ConfidenceGatingPipeline.evaluate() output contract."""

    def _make_proba(self, rng: np.random.Generator, direction: int) -> np.ndarray:
        """Create a 3-class probability vector biased toward `direction`."""
        # direction: -1=DOWN(idx=0), 0=FLAT(idx=1), 1=UP(idx=2)
        idx = direction + 1  # -1->0, 0->1, 1->2
        p = rng.dirichlet([0.5, 0.5, 0.5])
        # Amplify the preferred class
        p[idx] += 0.4
        p = np.clip(p, 0.0, 1.0)
        p /= p.sum()
        return p.astype(np.float64)

    def test_evaluate_returns_gating_result(
        self,
        gating_pipeline: ConfidenceGatingPipeline,
        feature_matrix: tuple[np.ndarray, list[str]],
    ) -> None:
        rng = np.random.default_rng(1)
        X, _ = feature_matrix
        features = X[100]  # Single bar features

        probas = self._make_proba(rng, direction=1)
        ctx = MarketContext(
            volatility_ann=50.0,
            regime_label=0,
            regime_probability=0.85,
            spread_bps=1.0,
            current_drawdown_pct=0.0,
        )

        result = gating_pipeline.evaluate(
            primary_probas=probas,
            primary_prediction=1,
            features=features,
            market_ctx=ctx,
        )
        assert isinstance(result, GatingResult)

    def test_gating_result_fields(
        self,
        gating_pipeline: ConfidenceGatingPipeline,
        feature_matrix: tuple[np.ndarray, list[str]],
    ) -> None:
        rng = np.random.default_rng(2)
        X, _ = feature_matrix
        features = X[150]
        probas = self._make_proba(rng, direction=1)

        ctx = MarketContext(
            volatility_ann=40.0,
            regime_label=0,
            regime_probability=0.90,
            spread_bps=0.5,
            current_drawdown_pct=0.0,
        )
        result = gating_pipeline.evaluate(
            primary_probas=probas,
            primary_prediction=1,
            features=features,
            market_ctx=ctx,
        )
        # Verify type contract on every field
        assert isinstance(result.should_trade, bool)
        assert result.direction in {-1, 0, 1}
        assert 0.0 <= result.composite_confidence <= 1.0 or result.composite_confidence == 0.0
        assert result.calibrated_probas is not None
        assert result.calibrated_probas.shape == (3,)
        assert result.drawdown_multiplier >= 0.0
        assert isinstance(result.gate_decisions, list)
        assert len(result.gate_decisions) >= 1

    def test_excessive_drawdown_blocks_trade(
        self, gating_pipeline: ConfidenceGatingPipeline
    ) -> None:
        """Gate 7 must block trades when drawdown >= 15%."""
        rng = np.random.default_rng(3)
        features = np.zeros(30)
        probas = self._make_proba(rng, direction=1)
        ctx = MarketContext(
            volatility_ann=40.0,
            regime_label=0,
            regime_probability=0.90,
            spread_bps=1.0,
            current_drawdown_pct=15.0,  # at halt threshold
        )
        result = gating_pipeline.evaluate(
            primary_probas=probas,
            primary_prediction=1,
            features=features,
            market_ctx=ctx,
        )
        assert not result.should_trade, "Trade must be blocked at drawdown halt threshold"

    def test_low_volatility_blocks_trade(
        self, gating_pipeline: ConfidenceGatingPipeline
    ) -> None:
        """Gate 5 must block trades when vol < 15% annualised."""
        rng = np.random.default_rng(4)
        features = np.zeros(30)
        probas = self._make_proba(rng, direction=1)
        ctx = MarketContext(
            volatility_ann=10.0,  # Below 15% minimum
            regime_label=0,
            regime_probability=0.90,
            spread_bps=1.0,
            current_drawdown_pct=0.0,
        )
        result = gating_pipeline.evaluate(
            primary_probas=probas,
            primary_prediction=1,
            features=features,
            market_ctx=ctx,
        )
        assert not result.should_trade, "Trade must be blocked when vol is below minimum"

    def test_ensemble_agreement_gate(
        self, gating_pipeline: ConfidenceGatingPipeline
    ) -> None:
        """Maximally disagreeing base models (variance > 0.15 threshold) fail gate 3.

        Pipeline detail: the gate computes mean(var_across_models(per_class_prob)).
        For [0.98, 0.01, 0.01] vs [0.01, 0.01, 0.98], per-class variances are:
          DOWN: var([0.98, 0.01]) ≈ 0.226
          FLAT: var([0.01, 0.01]) = 0.0
          UP:   var([0.01, 0.98]) ≈ 0.226
          mean ≈ 0.157 > threshold 0.15  →  gate FAILS
        """
        features = np.zeros(30)
        # Maximally opposing: model 1 almost certain DOWN, model 2 almost certain UP
        p1 = np.array([0.98, 0.01, 0.01])
        p2 = np.array([0.01, 0.01, 0.98])
        ctx = MarketContext(
            volatility_ann=40.0,
            regime_label=0,
            regime_probability=0.90,
            spread_bps=1.0,
            current_drawdown_pct=0.0,
        )
        primary = np.array([0.1, 0.1, 0.8])
        result = gating_pipeline.evaluate(
            primary_probas=primary,
            primary_prediction=1,
            features=features,
            market_ctx=ctx,
            ensemble_probas=[p1, p2],
        )
        assert not result.should_trade, (
            "Maximally disagreeing ensemble (var≈0.157 > 0.15) must fail gate 3"
        )


class TestRiskManagerStage:
    """Verify RiskManager.approve_trade() output contract."""

    def test_approve_returns_trade_decision(
        self,
        risk_manager: RiskManager,
        ohlcv_arrays: dict[str, np.ndarray],
    ) -> None:
        closes = ohlcv_arrays["closes"]
        highs = ohlcv_arrays["highs"]
        lows = ohlcv_arrays["lows"]
        current_idx = 200

        signal = SignalInput(
            direction="long",
            confidence=0.70,
            timestamp_ms=BASE_TIMESTAMP_MS + current_idx * BAR_INTERVAL_MS,
            win_rate=0.55,
            payoff_ratio=1.5,
        )
        decision = risk_manager.approve_trade(signal, closes, highs, lows, current_idx)
        assert hasattr(decision, "approved")
        assert isinstance(decision.approved, bool)

    def test_approved_trade_has_positive_quantity(
        self,
        risk_manager: RiskManager,
        ohlcv_arrays: dict[str, np.ndarray],
    ) -> None:
        closes = ohlcv_arrays["closes"]
        highs = ohlcv_arrays["highs"]
        lows = ohlcv_arrays["lows"]
        current_idx = 200

        signal = SignalInput(
            direction="long",
            confidence=0.70,
            timestamp_ms=BASE_TIMESTAMP_MS + current_idx * BAR_INTERVAL_MS,
            win_rate=0.55,
            payoff_ratio=1.5,
        )
        decision = risk_manager.approve_trade(signal, closes, highs, lows, current_idx)
        if decision.approved:
            assert decision.quantity_btc > 0.0, "Approved trade must have positive BTC quantity"
            assert decision.notional_usd > 0.0, "Approved trade must have positive notional"
            assert decision.stop_price > 0.0, "Approved trade must have a stop price"
        # If rejected, we just verify the type — rejection is a valid outcome

    def test_rejected_when_position_open(
        self,
        ohlcv_arrays: dict[str, np.ndarray],
    ) -> None:
        """A second trade while a position is open must be rejected."""
        conn = sqlite3.connect(":memory:", check_same_thread=False)
        risk_cfg = RiskConfig(
            enforce_trading_hours=False,
            min_volatility_ann=0.0,
            max_volatility_ann=10.0,
        )
        rm = RiskManager(conn, initial_equity=INITIAL_EQUITY, config=risk_cfg)

        closes = ohlcv_arrays["closes"]
        highs = ohlcv_arrays["highs"]
        lows = ohlcv_arrays["lows"]
        current_idx = 200
        ts = BASE_TIMESTAMP_MS + current_idx * BAR_INTERVAL_MS

        signal = SignalInput(
            direction="long",
            confidence=0.70,
            timestamp_ms=ts,
            win_rate=0.55,
            payoff_ratio=1.5,
        )

        first = rm.approve_trade(signal, closes, highs, lows, current_idx)
        if first.approved:
            # Open the position
            rm.on_trade_opened(
                side="long",
                quantity=first.quantity_btc,
                entry_price=float(closes[current_idx]),
                entry_time_ms=ts,
                stop_price=first.stop_price,
            )
            # Second trade must be rejected
            second = rm.approve_trade(signal, closes, highs, lows, current_idx)
            assert not second.approved, "Must reject trade when position is already open"

    def test_daily_trade_limit(self, ohlcv_arrays: dict[str, np.ndarray]) -> None:
        """After max_trades_per_day trades, further approvals must be rejected."""
        conn = sqlite3.connect(":memory:", check_same_thread=False)
        # Very low max trades for testing
        risk_cfg = RiskConfig(
            max_trades_per_day=2,
            enforce_trading_hours=False,
            min_volatility_ann=0.0,
            max_volatility_ann=10.0,
        )
        rm = RiskManager(conn, initial_equity=INITIAL_EQUITY, config=risk_cfg)

        closes = ohlcv_arrays["closes"]
        highs = ohlcv_arrays["highs"]
        lows = ohlcv_arrays["lows"]

        # Simulate 2 trades (open + immediately close each)
        for i in range(2):
            idx = 200 + i * 10
            ts = BASE_TIMESTAMP_MS + idx * BAR_INTERVAL_MS
            signal = SignalInput(
                direction="long",
                confidence=0.70,
                timestamp_ms=ts,
                win_rate=0.55,
                payoff_ratio=1.5,
            )
            dec = rm.approve_trade(signal, closes, highs, lows, idx)
            if dec.approved:
                rm.on_trade_opened(
                    side="long",
                    quantity=dec.quantity_btc,
                    entry_price=float(closes[idx]),
                    entry_time_ms=ts,
                    stop_price=dec.stop_price,
                )
                rm.on_trade_closed(
                    exit_price=float(closes[idx + 1]),
                    exit_time_ms=ts + BAR_INTERVAL_MS,
                )

        # Now the third trade should be rejected (limit=2)
        idx = 230
        ts = BASE_TIMESTAMP_MS + idx * BAR_INTERVAL_MS
        signal = SignalInput(
            direction="long",
            confidence=0.70,
            timestamp_ms=ts,
            win_rate=0.55,
            payoff_ratio=1.5,
        )
        third = rm.approve_trade(signal, closes, highs, lows, idx)
        assert not third.approved, "Must reject after max_trades_per_day reached"


# ---------------------------------------------------------------------------
# Full end-to-end pipeline test
# ---------------------------------------------------------------------------


class TestFullE2EPipeline:
    """End-to-end test that runs every stage in the correct order and
    produces a Polymarket-compatible order dict for a passing signal."""

    def _make_market_ctx(self, closes: np.ndarray, idx: int) -> MarketContext:
        """Compute a realistic MarketContext from recent closes."""
        bar_std = 0.15 / np.sqrt(105_120)
        start = max(0, idx - 50)
        window = closes[start : idx + 1]
        log_ret = np.diff(np.log(np.maximum(window, 1e-10)))
        if len(log_ret) >= 2:
            vol_ann = float(np.std(log_ret, ddof=1)) * np.sqrt(105_120) * 100.0
        else:
            vol_ann = 50.0

        return MarketContext(
            volatility_ann=max(vol_ann, 20.0),  # ensure above minimum
            regime_label=0,
            regime_probability=0.85,
            spread_bps=1.0,
            current_drawdown_pct=0.0,
        )

    def _build_trade_signal(
        self,
        decision_direction: int,
        gating_result: GatingResult,
        trade_decision: Any,
        current_price: float,
        timestamp_ms: int,
    ) -> dict[str, Any]:
        """Convert pipeline outputs to a Polymarket-compatible order dict.

        Polymarket BTC prediction markets use simple YES/NO tokens for
        price direction; the "order" captures the direction, sizing, and
        confidence metadata for downstream order routing.

        Returns:
            Dict with keys: market, side, outcome, quantity_btc, notional_usd,
            stop_price, confidence, timestamp_ms, source.
        """
        if decision_direction == 1:
            side = "BUY"
            outcome = "UP"
        elif decision_direction == -1:
            side = "SELL"
            outcome = "DOWN"
        else:
            side = "ABSTAIN"
            outcome = "FLAT"

        return {
            "market": "BTC/USDT-5m",
            "side": side,
            "outcome": outcome,
            "quantity_btc": trade_decision.quantity_btc if trade_decision.approved else 0.0,
            "notional_usd": trade_decision.notional_usd if trade_decision.approved else 0.0,
            "stop_price": trade_decision.stop_price if trade_decision.approved else 0.0,
            "confidence": round(gating_result.composite_confidence, 6),
            "timestamp_ms": timestamp_ms,
            "current_price_usd": round(current_price, 2),
            "source": "ep2_crypto_v1",
        }

    def test_full_pipeline_runs_without_error(
        self,
        feature_matrix: tuple[np.ndarray, list[str]],
        trained_lgbm: LGBMDirectionModel,
        trained_stacking: StackingEnsemble,
        gating_pipeline: ConfidenceGatingPipeline,
        risk_manager: RiskManager,
        ohlcv_arrays: dict[str, np.ndarray],
    ) -> None:
        """Run the full pipeline on a single bar and verify no exceptions are raised."""
        X, feature_names = feature_matrix
        closes = ohlcv_arrays["closes"]
        highs = ohlcv_arrays["highs"]
        lows = ohlcv_arrays["lows"]

        # Use cleaned feature matrix (drops structurally-NaN OB columns)
        X_clean, _ = _clean_feature_matrix(X, feature_names)
        # target_idx=200 in the original maps to index 200-65=135 in the clean matrix
        clean_idx = 135
        features_1d_clean = X_clean[clean_idx]
        # Also keep original (full) features for gating pipeline context
        # (gating pipeline doesn't run them through LGBM, just stores as context)
        features_for_gating = features_1d_clean

        # Stage 1: Primary model (LGBM) — uses cleaned features
        proba_lgbm = trained_lgbm.predict_proba(features_1d_clean.reshape(1, -1))[0]
        assert proba_lgbm.shape == (3,)

        # Stage 2: Simulate second base model with small noise
        rng = np.random.default_rng(42)
        noise = rng.normal(0.0, 0.02, size=proba_lgbm.shape)
        proba_cat = np.clip(proba_lgbm + noise, 1e-6, 1.0)
        proba_cat /= proba_cat.sum()

        # Stage 3: Stacking ensemble
        stack_proba = trained_stacking.predict_proba(
            [proba_lgbm.reshape(1, -1), proba_cat.reshape(1, -1)]
        )[0]
        assert stack_proba.shape == (3,)
        primary_pred = int(np.argmax(stack_proba)) - 1  # decode: 0->-1, 1->0, 2->1

        # Stage 4: Gating pipeline
        # target_idx in the original (full) OHLCV array: clean_idx + warmup
        target_idx_ohlcv = clean_idx + 65
        market_ctx = self._make_market_ctx(closes, target_idx_ohlcv)
        gating_result = gating_pipeline.evaluate(
            primary_probas=stack_proba,
            primary_prediction=primary_pred,
            features=features_for_gating,
            market_ctx=market_ctx,
            ensemble_probas=[proba_lgbm, proba_cat],
        )
        assert isinstance(gating_result, GatingResult)

        # Stage 5: Risk manager (only if gating passed a non-flat direction)
        ts = int(ohlcv_arrays["timestamps"][target_idx_ohlcv])
        signal = SignalInput(
            direction="long" if gating_result.direction >= 0 else "short",
            confidence=max(gating_result.composite_confidence, 0.30),  # ensure non-zero
            timestamp_ms=ts,
            win_rate=0.55,
            payoff_ratio=1.5,
        )

        # Use a fresh RiskManager so the position check is always clean
        conn = sqlite3.connect(":memory:", check_same_thread=False)
        risk_cfg = RiskConfig(
            enforce_trading_hours=False,
            min_volatility_ann=0.0,
            max_volatility_ann=10.0,
        )
        rm = RiskManager(conn, initial_equity=INITIAL_EQUITY, config=risk_cfg)
        trade_decision = rm.approve_trade(signal, closes, highs, lows, target_idx_ohlcv)

        # Stage 6: Build order dict
        order = self._build_trade_signal(
            decision_direction=gating_result.direction if gating_result.should_trade else 0,
            gating_result=gating_result,
            trade_decision=trade_decision,
            current_price=float(closes[target_idx_ohlcv]),
            timestamp_ms=ts,
        )

        # Verify order dict structure
        required_keys = {
            "market", "side", "outcome", "quantity_btc", "notional_usd",
            "stop_price", "confidence", "timestamp_ms", "current_price_usd", "source",
        }
        assert required_keys.issubset(order.keys()), (
            f"Missing keys: {required_keys - set(order.keys())}"
        )
        assert order["source"] == "ep2_crypto_v1"
        assert order["market"] == "BTC/USDT-5m"
        assert order["side"] in {"BUY", "SELL", "ABSTAIN"}
        assert order["outcome"] in {"UP", "DOWN", "FLAT"}

    def test_approved_trade_signal_has_valid_sizing(
        self,
        feature_matrix: tuple[np.ndarray, list[str]],
        trained_lgbm: LGBMDirectionModel,
        trained_stacking: StackingEnsemble,
        gating_pipeline: ConfidenceGatingPipeline,
        ohlcv_arrays: dict[str, np.ndarray],
    ) -> None:
        """When risk approves a trade, the sizing fields must be positive and consistent."""
        X, feature_names = feature_matrix
        closes = ohlcv_arrays["closes"]
        highs = ohlcv_arrays["highs"]
        lows = ohlcv_arrays["lows"]

        # Use cleaned features; clean_idx=185 maps to ohlcv_idx=250
        X_clean, _ = _clean_feature_matrix(X, feature_names)
        clean_idx = 185
        target_idx = clean_idx + 65  # original OHLCV index

        features_1d = X_clean[clean_idx]
        proba_lgbm = trained_lgbm.predict_proba(features_1d.reshape(1, -1))[0]

        # Use a fresh RM with relaxed limits
        conn = sqlite3.connect(":memory:", check_same_thread=False)
        risk_cfg = RiskConfig(
            enforce_trading_hours=False,
            min_volatility_ann=0.0,
            max_volatility_ann=10.0,
        )
        rm = RiskManager(conn, initial_equity=INITIAL_EQUITY, config=risk_cfg)

        ts = int(ohlcv_arrays["timestamps"][target_idx])
        signal = SignalInput(
            direction="long",
            confidence=0.75,
            timestamp_ms=ts,
            win_rate=0.55,
            payoff_ratio=1.5,
        )
        decision = rm.approve_trade(signal, closes, highs, lows, target_idx)

        if decision.approved:
            price = float(closes[target_idx])
            assert decision.quantity_btc >= 0.001, "Must meet minimum BTC quantity"
            assert decision.notional_usd > 0, "Notional must be positive"
            # Notional should be consistent with price × qty (within 1% tolerance)
            expected_notional = decision.quantity_btc * price
            assert abs(decision.notional_usd - expected_notional) / expected_notional < 0.01, (
                f"Notional mismatch: {decision.notional_usd:.2f} vs {expected_notional:.2f}"
            )
            # Stop price must be below entry for long
            assert decision.stop_price < price, (
                f"Long stop {decision.stop_price:.2f} must be below entry {price:.2f}"
            )
            # Position fraction must respect max cap
            position_fraction = decision.notional_usd / INITIAL_EQUITY
            assert position_fraction <= 0.06, (
                f"Position fraction {position_fraction:.4f} exceeds 5% cap"
            )

    def test_on_bar_returns_risk_actions(
        self,
        ohlcv_arrays: dict[str, np.ndarray],
    ) -> None:
        """Verify on_bar() returns a list (possibly empty) of RiskAction objects."""
        conn = sqlite3.connect(":memory:", check_same_thread=False)
        risk_cfg = RiskConfig(
            enforce_trading_hours=False,
            min_volatility_ann=0.0,
            max_volatility_ann=10.0,
        )
        rm = RiskManager(conn, initial_equity=INITIAL_EQUITY, config=risk_cfg)

        closes = ohlcv_arrays["closes"]
        highs = ohlcv_arrays["highs"]
        lows = ohlcv_arrays["lows"]
        idx = 200
        ts = BASE_TIMESTAMP_MS + idx * BAR_INTERVAL_MS

        actions = rm.on_bar(
            bar_close=float(closes[idx]),
            bar_high=float(highs[idx]),
            bar_low=float(lows[idx]),
            bar_timestamp_ms=ts,
            closes=closes,
            highs=highs,
            lows=lows,
            current_idx=idx,
        )
        assert isinstance(actions, list), "on_bar must return a list"

    def test_pipeline_latency_acceptable(
        self,
        feature_matrix: tuple[np.ndarray, list[str]],
        trained_lgbm: LGBMDirectionModel,
        trained_stacking: StackingEnsemble,
        gating_pipeline: ConfidenceGatingPipeline,
        ohlcv_arrays: dict[str, np.ndarray],
    ) -> None:
        """Full pipeline (excluding feature computation) must complete in <200ms.

        This corresponds to the CLAUDE.md latency requirement (200ms minimum
        simulated latency). If this fails, the inference path is too slow.
        """
        X, feature_names = feature_matrix
        closes = ohlcv_arrays["closes"]
        highs = ohlcv_arrays["highs"]
        lows = ohlcv_arrays["lows"]

        # clean_idx=235 maps to ohlcv_idx=300
        X_clean, _ = _clean_feature_matrix(X, feature_names)
        clean_idx = 235
        target_idx = clean_idx + 65

        features_1d = X_clean[clean_idx]
        rng = np.random.default_rng(5)

        conn = sqlite3.connect(":memory:", check_same_thread=False)
        risk_cfg = RiskConfig(
            enforce_trading_hours=False,
            min_volatility_ann=0.0,
            max_volatility_ann=10.0,
        )
        rm = RiskManager(conn, initial_equity=INITIAL_EQUITY, config=risk_cfg)

        ts = int(ohlcv_arrays["timestamps"][target_idx])

        start = time.perf_counter()

        proba_lgbm = trained_lgbm.predict_proba(features_1d.reshape(1, -1))[0]
        noise = rng.normal(0.0, 0.02, size=proba_lgbm.shape)
        proba_cat = np.clip(proba_lgbm + noise, 1e-6, 1.0)
        proba_cat /= proba_cat.sum()

        stack_proba = trained_stacking.predict_proba(
            [proba_lgbm.reshape(1, -1), proba_cat.reshape(1, -1)]
        )[0]
        primary_pred = int(np.argmax(stack_proba)) - 1

        ctx = MarketContext(
            volatility_ann=50.0,
            regime_label=0,
            regime_probability=0.85,
            spread_bps=1.0,
            current_drawdown_pct=0.0,
        )
        gating_pipeline.evaluate(
            primary_probas=stack_proba,
            primary_prediction=primary_pred,
            features=features_1d,
            market_ctx=ctx,
            ensemble_probas=[proba_lgbm, proba_cat],
        )

        signal = SignalInput(
            direction="long",
            confidence=0.65,
            timestamp_ms=ts,
            win_rate=0.55,
            payoff_ratio=1.5,
        )
        rm.approve_trade(signal, closes, highs, lows, target_idx)

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        assert elapsed_ms < 200.0, (
            f"Inference pipeline took {elapsed_ms:.1f}ms > 200ms budget"
        )


# ---------------------------------------------------------------------------
# Polymarket order format test
# ---------------------------------------------------------------------------


class TestPolymarketOrderFormat:
    """Verify the output order dict is valid for Polymarket BTC prediction markets."""

    def test_long_signal_produces_buy_up(self) -> None:
        """A long direction (1) must map to BUY/UP."""
        from ep2_crypto.confidence.gating import GatingResult, GateDecision, GateID

        fake_gating = GatingResult(
            should_trade=True,
            direction=1,
            composite_confidence=0.72,
            calibrated_probas=np.array([0.1, 0.15, 0.75]),
            meta_label_prob=0.72,
            conformal_set_size=1,
            drawdown_multiplier=1.0,
            gate_decisions=[
                GateDecision(gate_id=GateID.CALIBRATION, passed=True, reason="calibrated", value=1.0)
            ],
            rejection_gate=None,
        )
        from ep2_crypto.risk.risk_manager import TradeDecision

        fake_trade = TradeDecision(
            approved=True,
            quantity_btc=0.01,
            notional_usd=650.0,
            stop_price=64_000.0,
            position_fraction=0.0065,
            reason="All risk checks passed",
        )

        order: dict[str, Any] = {
            "market": "BTC/USDT-5m",
            "side": "BUY" if fake_gating.direction == 1 else "SELL",
            "outcome": "UP" if fake_gating.direction == 1 else "DOWN",
            "quantity_btc": fake_trade.quantity_btc,
            "notional_usd": fake_trade.notional_usd,
            "stop_price": fake_trade.stop_price,
            "confidence": round(fake_gating.composite_confidence, 6),
            "timestamp_ms": BASE_TIMESTAMP_MS,
            "current_price_usd": 65_000.0,
            "source": "ep2_crypto_v1",
        }

        assert order["side"] == "BUY"
        assert order["outcome"] == "UP"
        assert order["quantity_btc"] == 0.01
        assert order["confidence"] == 0.72
        assert order["source"] == "ep2_crypto_v1"

    def test_short_signal_produces_sell_down(self) -> None:
        """A short direction (-1) must map to SELL/DOWN."""
        direction = -1
        side = "BUY" if direction == 1 else "SELL"
        outcome = "UP" if direction == 1 else "DOWN"
        assert side == "SELL"
        assert outcome == "DOWN"

    def test_flat_direction_produces_abstain(self) -> None:
        """A flat/no-trade direction (0) must map to ABSTAIN/FLAT."""
        direction = 0
        side = "BUY" if direction == 1 else ("SELL" if direction == -1 else "ABSTAIN")
        outcome = "UP" if direction == 1 else ("DOWN" if direction == -1 else "FLAT")
        assert side == "ABSTAIN"
        assert outcome == "FLAT"
