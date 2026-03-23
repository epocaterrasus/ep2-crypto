"""Tests for feature computation interface and registry."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from ep2_crypto.features.base import FeatureComputer, FeatureRegistry

if TYPE_CHECKING:
    from numpy.typing import NDArray


class DummyFeature(FeatureComputer):
    """Concrete feature for testing."""

    @property
    def name(self) -> str:
        return "dummy"

    @property
    def warmup_bars(self) -> int:
        return 5

    def compute(
        self,
        idx: int,
        timestamps: NDArray[np.int64],
        opens: NDArray[np.float64],
        highs: NDArray[np.float64],
        lows: NDArray[np.float64],
        closes: NDArray[np.float64],
        volumes: NDArray[np.float64],
        **kwargs: NDArray[np.float64] | None,
    ) -> dict[str, float]:
        if idx < self.warmup_bars:
            return {"dummy_val": float("nan")}
        return {"dummy_val": float(closes[idx])}

    def output_names(self) -> list[str]:
        return ["dummy_val"]


class MultiOutputFeature(FeatureComputer):
    """Feature that produces multiple outputs."""

    @property
    def name(self) -> str:
        return "multi"

    @property
    def warmup_bars(self) -> int:
        return 3

    def compute(
        self,
        idx: int,
        timestamps: NDArray[np.int64],
        opens: NDArray[np.float64],
        highs: NDArray[np.float64],
        lows: NDArray[np.float64],
        closes: NDArray[np.float64],
        volumes: NDArray[np.float64],
        **kwargs: NDArray[np.float64] | None,
    ) -> dict[str, float]:
        if idx < self.warmup_bars:
            return {"multi_a": float("nan"), "multi_b": float("nan")}
        return {
            "multi_a": float(closes[idx] - opens[idx]),
            "multi_b": float(highs[idx] - lows[idx]),
        }

    def output_names(self) -> list[str]:
        return ["multi_a", "multi_b"]


def _make_arrays(
    n: int = 20,
) -> tuple[
    NDArray[np.int64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    rng = np.random.default_rng(42)
    timestamps = np.arange(n, dtype=np.int64) * 60_000
    closes = 100.0 + np.cumsum(rng.standard_normal(n))
    opens = closes - rng.uniform(0, 1, n)
    highs = np.maximum(opens, closes) + rng.uniform(0, 0.5, n)
    lows = np.minimum(opens, closes) - rng.uniform(0, 0.5, n)
    volumes = rng.uniform(100, 1000, n)
    return timestamps, opens, highs, lows, closes, volumes


class TestFeatureComputer:
    def test_abc_cannot_instantiate(self) -> None:
        with pytest.raises(TypeError):
            FeatureComputer()  # type: ignore[abstract]

    def test_concrete_implementation(self) -> None:
        feat = DummyFeature()
        assert feat.name == "dummy"
        assert feat.warmup_bars == 5

    def test_compute_returns_nan_before_warmup(self) -> None:
        feat = DummyFeature()
        ts, o, h, lo, c, v = _make_arrays()
        result = feat.compute(2, ts, o, h, lo, c, v)
        assert np.isnan(result["dummy_val"])

    def test_compute_returns_value_after_warmup(self) -> None:
        feat = DummyFeature()
        ts, o, h, lo, c, v = _make_arrays()
        result = feat.compute(10, ts, o, h, lo, c, v)
        assert result["dummy_val"] == pytest.approx(c[10])

    def test_output_names(self) -> None:
        feat = DummyFeature()
        assert feat.output_names() == ["dummy_val"]

    def test_multi_output(self) -> None:
        feat = MultiOutputFeature()
        ts, o, h, lo, c, v = _make_arrays()
        result = feat.compute(5, ts, o, h, lo, c, v)
        assert set(result.keys()) == {"multi_a", "multi_b"}
        assert result["multi_a"] == pytest.approx(c[5] - o[5])


class TestFeatureRegistry:
    def test_register_and_get(self) -> None:
        reg = FeatureRegistry()
        feat = DummyFeature()
        reg.register(feat)
        assert reg.get("dummy") is feat

    def test_register_duplicate_raises(self) -> None:
        reg = FeatureRegistry()
        reg.register(DummyFeature())
        with pytest.raises(ValueError, match="already registered"):
            reg.register(DummyFeature())

    def test_get_missing_raises(self) -> None:
        reg = FeatureRegistry()
        with pytest.raises(KeyError, match="not found"):
            reg.get("nonexistent")

    def test_names(self) -> None:
        reg = FeatureRegistry()
        reg.register(DummyFeature())
        reg.register(MultiOutputFeature())
        assert set(reg.names) == {"dummy", "multi"}

    def test_get_all(self) -> None:
        reg = FeatureRegistry()
        reg.register(DummyFeature())
        reg.register(MultiOutputFeature())
        assert len(reg.get_all()) == 2

    def test_select(self) -> None:
        reg = FeatureRegistry()
        reg.register(DummyFeature())
        reg.register(MultiOutputFeature())
        selected = reg.select(["multi"])
        assert len(selected) == 1
        assert selected[0].name == "multi"

    def test_max_warmup(self) -> None:
        reg = FeatureRegistry()
        assert reg.max_warmup() == 0
        reg.register(DummyFeature())  # warmup=5
        reg.register(MultiOutputFeature())  # warmup=3
        assert reg.max_warmup() == 5

    def test_compute_all(self) -> None:
        reg = FeatureRegistry()
        reg.register(DummyFeature())
        reg.register(MultiOutputFeature())
        ts, o, h, lo, c, v = _make_arrays()
        result = reg.compute_all(10, ts, o, h, lo, c, v)
        assert "dummy_val" in result
        assert "multi_a" in result
        assert "multi_b" in result
        assert result["dummy_val"] == pytest.approx(c[10])

    def test_compute_all_before_warmup(self) -> None:
        reg = FeatureRegistry()
        reg.register(DummyFeature())
        ts, o, h, lo, c, v = _make_arrays()
        result = reg.compute_all(2, ts, o, h, lo, c, v)
        assert np.isnan(result["dummy_val"])
