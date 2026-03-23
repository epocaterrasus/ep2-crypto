"""Tests for ablation study: full system vs component-removed variants."""

from __future__ import annotations

import numpy as np
import pytest

from ep2_crypto.backtest.ablation import (
    AblationResult,
    AblationStudy,
    AblationVariant,
    AblationVariantResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ohlcv(
    n: int = 500,
    seed: int = 0,
    drift: float = 0.0001,
) -> tuple:
    """Generate synthetic OHLCV with mild upward drift."""
    rng = np.random.default_rng(seed)
    log_returns = rng.normal(drift, 0.002, n)
    closes = 30_000.0 * np.cumprod(1 + log_returns)
    highs = closes * (1 + abs(rng.normal(0, 0.001, n)))
    lows = closes * (1 - abs(rng.normal(0, 0.001, n)))
    opens = np.roll(closes, 1)
    opens[0] = closes[0]
    volumes = rng.uniform(100, 1000, n)
    timestamps_ms = np.arange(n, dtype=np.int64) * 300_000  # 5-min bars
    return opens, highs, lows, closes, volumes, timestamps_ms


def _make_signals(n: int, seed: int = 0, win_rate: float = 0.55) -> tuple:
    """Generate signals with a slight edge (win_rate > 0.5)."""
    rng = np.random.default_rng(seed)
    signals = np.where(rng.random(n) < win_rate, 1, -1).astype(np.int8)
    signals[: n // 5] = 0  # warm-up: no signal for first 20%
    confidences = rng.uniform(0.5, 0.9, n)
    return signals, confidences


def _run_full_study(n: int = 600, seed: int = 42) -> AblationResult:
    opens, highs, lows, closes, volumes, timestamps_ms = _make_ohlcv(n, seed)
    signals, confidences = _make_signals(n, seed)
    regime_labels = (np.arange(n) // (n // 3)).astype(np.int32)
    study = AblationStudy(initial_equity=50_000.0, seed=seed)
    return study.run(
        opens=opens,
        highs=highs,
        lows=lows,
        closes=closes,
        volumes=volumes,
        timestamps_ms=timestamps_ms,
        signals=signals,
        confidences=confidences,
        regime_labels=regime_labels,
    )


# ---------------------------------------------------------------------------
# AblationVariantResult
# ---------------------------------------------------------------------------

class TestAblationVariantResult:
    def test_fields_accessible(self):
        r = AblationVariantResult(
            variant=AblationVariant.FULL,
            sharpe=1.5,
            sortino=2.0,
            max_drawdown=0.05,
            total_trades=100,
            win_rate=0.55,
            total_return=0.15,
        )
        assert r.sharpe == 1.5
        assert r.delta_sharpe == 0.0  # default

    def test_delta_sharpe_default_zero(self):
        r = AblationVariantResult(
            variant=AblationVariant.NO_CONFIDENCE_GATE,
            sharpe=0.8,
            sortino=1.0,
            max_drawdown=0.08,
            total_trades=200,
            win_rate=0.50,
            total_return=0.05,
        )
        assert r.delta_sharpe == 0.0


# ---------------------------------------------------------------------------
# AblationResult
# ---------------------------------------------------------------------------

class TestAblationResult:
    def _make_result(self) -> AblationResult:
        variants = [
            AblationVariantResult(
                variant=AblationVariant.FULL,
                sharpe=2.0,
                sortino=3.0,
                max_drawdown=0.05,
                total_trades=100,
                win_rate=0.55,
                total_return=0.20,
            ),
            AblationVariantResult(
                variant=AblationVariant.NO_CONFIDENCE_GATE,
                sharpe=1.2,
                sortino=1.8,
                max_drawdown=0.08,
                total_trades=180,
                win_rate=0.51,
                total_return=0.10,
            ),
            AblationVariantResult(
                variant=AblationVariant.NO_RISK_ENGINE,
                sharpe=0.5,
                sortino=0.6,
                max_drawdown=0.20,
                total_trades=300,
                win_rate=0.50,
                total_return=0.03,
            ),
        ]
        return AblationResult(variants=variants)

    def test_delta_sharpe_computed_from_full(self):
        r = self._make_result()
        full = next(v for v in r.variants if v.variant == AblationVariant.FULL)
        no_conf = next(v for v in r.variants if v.variant == AblationVariant.NO_CONFIDENCE_GATE)
        no_risk = next(v for v in r.variants if v.variant == AblationVariant.NO_RISK_ENGINE)

        assert full.delta_sharpe == pytest.approx(0.0)
        assert no_conf.delta_sharpe == pytest.approx(1.2 - 2.0)
        assert no_risk.delta_sharpe == pytest.approx(0.5 - 2.0)

    def test_summary_contains_all_variants(self):
        r = self._make_result()
        s = r.summary()
        assert "full" in s
        assert "no_confidence_gate" in s
        assert "no_risk_engine" in s

    def test_to_dict_returns_list(self):
        r = self._make_result()
        d = r.to_dict()
        assert isinstance(d, list)
        assert len(d) == 3
        assert d[0]["variant"] == "full"

    def test_summary_shows_delta_sign(self):
        r = self._make_result()
        s = r.summary()
        # FULL should show +0.000, others should show negative delta
        assert "+" in s or "-" in s


# ---------------------------------------------------------------------------
# AblationStudy — configuration building
# ---------------------------------------------------------------------------

class TestAblationStudyConfigBuilding:
    def setup_method(self):
        self.study = AblationStudy(initial_equity=10_000.0, seed=0)
        n = 50
        self.signals = np.ones(n, dtype=np.int8)
        self.confidences = np.full(n, 0.7)

    def test_full_variant_keeps_signals_unchanged(self):
        cfg, sigs, confs = self.study._build_config(
            AblationVariant.FULL, self.signals, self.confidences
        )
        assert np.array_equal(sigs, self.signals)
        assert np.array_equal(confs, self.confidences)

    def test_no_confidence_gate_sets_threshold_zero(self):
        cfg, sigs, confs = self.study._build_config(
            AblationVariant.NO_CONFIDENCE_GATE, self.signals, self.confidences
        )
        assert cfg.confidence_threshold == 0.0
        assert np.all(confs == 1.0)

    def test_no_risk_engine_wide_limits(self):
        cfg, sigs, confs = self.study._build_config(
            AblationVariant.NO_RISK_ENGINE, self.signals, self.confidences
        )
        assert cfg.risk_config is not None
        assert cfg.risk_config.daily_loss_limit >= 0.9
        assert cfg.risk_config.max_drawdown_halt >= 0.9

    def test_no_drawdown_gate_high_dd_threshold(self):
        cfg, sigs, confs = self.study._build_config(
            AblationVariant.NO_DRAWDOWN_GATE, self.signals, self.confidences
        )
        assert cfg.risk_config is not None
        assert cfg.risk_config.max_drawdown_halt >= 0.9

    def test_no_kill_switches_high_limits(self):
        cfg, sigs, confs = self.study._build_config(
            AblationVariant.NO_KILL_SWITCHES, self.signals, self.confidences
        )
        assert cfg.risk_config is not None
        assert cfg.risk_config.daily_loss_limit >= 0.9
        assert cfg.risk_config.consecutive_loss_limit >= 1000

    def test_unknown_variant_raises(self):
        with pytest.raises((ValueError, AttributeError)):
            self.study._build_config("unknown_variant", self.signals, self.confidences)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# AblationStudy — end-to-end runs
# ---------------------------------------------------------------------------

class TestAblationStudyEndToEnd:
    def test_all_variants_produce_results(self):
        result = _run_full_study(n=400)
        assert len(result.variants) == len(AblationVariant)
        for v in result.variants:
            assert isinstance(v.sharpe, float)
            assert not np.isnan(v.sharpe)

    def test_full_variant_present(self):
        result = _run_full_study()
        variants = {v.variant for v in result.variants}
        assert AblationVariant.FULL in variants

    def test_delta_sharpe_is_zero_for_full(self):
        result = _run_full_study()
        full = next(v for v in result.variants if v.variant == AblationVariant.FULL)
        assert full.delta_sharpe == pytest.approx(0.0)

    def test_no_confidence_gate_has_more_trades(self):
        result = _run_full_study(n=600, seed=7)
        full = next(v for v in result.variants if v.variant == AblationVariant.FULL)
        no_conf = next(v for v in result.variants if v.variant == AblationVariant.NO_CONFIDENCE_GATE)
        # No confidence gate passes ALL signals (threshold=0), expect >= trades
        assert no_conf.total_trades >= full.total_trades

    def test_selective_variants_run(self):
        opens, highs, lows, closes, volumes, timestamps_ms = _make_ohlcv(300)
        signals, confidences = _make_signals(300)
        study = AblationStudy(initial_equity=10_000.0)
        result = study.run(
            opens=opens, highs=highs, lows=lows, closes=closes,
            volumes=volumes, timestamps_ms=timestamps_ms,
            signals=signals, confidences=confidences,
            variants=[AblationVariant.FULL, AblationVariant.NO_RISK_ENGINE],
        )
        assert len(result.variants) == 2
        variant_types = {v.variant for v in result.variants}
        assert AblationVariant.FULL in variant_types
        assert AblationVariant.NO_RISK_ENGINE in variant_types

    def test_results_have_valid_metrics(self):
        result = _run_full_study(n=400)
        for v in result.variants:
            assert 0.0 <= v.max_drawdown <= 1.0
            assert 0.0 <= v.win_rate <= 1.0
            assert v.total_trades >= 0

    def test_summary_string_not_empty(self):
        result = _run_full_study(n=400)
        s = result.summary()
        assert len(s) > 100
        assert "full" in s

    def test_to_dict_all_fields_present(self):
        result = _run_full_study(n=400)
        dicts = result.to_dict()
        for d in dicts:
            assert "variant" in d
            assert "sharpe" in d
            assert "delta_sharpe" in d
            assert "total_trades" in d

    def test_no_regime_filter_runs_without_regime(self):
        opens, highs, lows, closes, volumes, timestamps_ms = _make_ohlcv(300)
        signals, confidences = _make_signals(300)
        study = AblationStudy(initial_equity=10_000.0)
        result = study.run(
            opens=opens, highs=highs, lows=lows, closes=closes,
            volumes=volumes, timestamps_ms=timestamps_ms,
            signals=signals, confidences=confidences,
            regime_labels=np.zeros(300, dtype=np.int32),
            variants=[AblationVariant.FULL, AblationVariant.NO_REGIME_FILTER],
        )
        assert len(result.variants) == 2
