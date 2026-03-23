"""Unified feature pipeline combining all feature modules.

Combines microstructure, volume, volatility, momentum, cross-market,
temporal, and regime features into a single pipeline with consistent
output shape and configurable feature selection.
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

import numpy as np

from ep2_crypto.features.base import FeatureComputer, FeatureRegistry
from ep2_crypto.features.cross_market import (
    DivergenceComputer,
    ETHRatioComputer,
    LeadLagComputer,
    NQReturnComputer,
)
from ep2_crypto.features.microstructure import (
    KyleLambdaComputer,
    MicropriceComputer,
    OBIComputer,
    OFIComputer,
    TFIComputer,
)
from ep2_crypto.features.momentum import (
    LinRegSlopeComputer,
    QuantileRankComputer,
    ROCComputer,
    RSIComputer,
)
from ep2_crypto.features.regime_features import (
    ERFeatureComputer,
    GARCHFeatureComputer,
    HMMFeatureComputer,
)
from ep2_crypto.features.temporal import (
    CyclicalTimeComputer,
    FundingTimeComputer,
    SessionComputer,
)
from ep2_crypto.features.volatility import (
    EWMAVolComputer,
    ParkinsonVolComputer,
    RealizedVolComputer,
    VolOfVolComputer,
)
from ep2_crypto.features.polymarket import PolymarketProbComputer
from ep2_crypto.features.volume import (
    VolumeDeltaComputer,
    VolumeROCComputer,
    VWAPComputer,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray


def build_default_registry() -> FeatureRegistry:
    """Build a registry with all default feature computers.

    Returns a registry containing all Sprint 3-5 features with
    standard parameters.
    """
    reg = FeatureRegistry()

    # Sprint 3: Microstructure
    reg.register(OBIComputer())
    reg.register(OFIComputer())
    reg.register(MicropriceComputer())
    reg.register(TFIComputer())
    reg.register(KyleLambdaComputer(window=20))

    # Sprint 4: Volume
    reg.register(VolumeDeltaComputer())
    reg.register(VWAPComputer(window=12))
    reg.register(VolumeROCComputer())

    # Sprint 4: Volatility
    reg.register(RealizedVolComputer(short_window=6, long_window=12))
    reg.register(ParkinsonVolComputer(short_window=6, long_window=12))
    reg.register(EWMAVolComputer(decay=0.94))
    reg.register(VolOfVolComputer(inner_window=6, outer_window=12))

    # Sprint 4: Momentum
    reg.register(ROCComputer())
    reg.register(RSIComputer(window=14))
    reg.register(LinRegSlopeComputer(window=20))
    reg.register(QuantileRankComputer(window=60))

    # Sprint 5: Cross-market
    reg.register(NQReturnComputer())
    reg.register(ETHRatioComputer())
    reg.register(LeadLagComputer(window=20))
    reg.register(DivergenceComputer(lookback=6))

    # Sprint 5: Temporal
    reg.register(CyclicalTimeComputer())
    reg.register(SessionComputer())
    reg.register(FundingTimeComputer())

    # Sprint 5: Regime
    reg.register(ERFeatureComputer(window=10))
    reg.register(GARCHFeatureComputer())
    reg.register(HMMFeatureComputer(vol_window=20, smooth_window=10))

    # Polymarket crowd consensus (optional — gracefully degrades without data)
    reg.register(PolymarketProbComputer(accuracy_window=20))

    return reg


def _get_accepted_kwargs(computer: FeatureComputer) -> tuple[set[str], bool]:
    """Get accepted keyword argument names for a compute method.

    Returns (set of accepted kwarg names, whether **kwargs is accepted).
    """
    sig = inspect.signature(computer.compute)
    accepted = set()
    has_var_keyword = False
    for name, param in sig.parameters.items():
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            has_var_keyword = True
        elif param.kind == inspect.Parameter.KEYWORD_ONLY:
            accepted.add(name)
    return accepted, has_var_keyword


# Cache accepted kwargs per class to avoid repeated introspection
_KWARGS_CACHE: dict[type, tuple[set[str], bool]] = {}


def _filter_kwargs(
    computer: FeatureComputer,
    kwargs: dict[str, np.ndarray | None],
) -> dict[str, np.ndarray | None]:
    """Filter kwargs to only those accepted by the computer's compute method."""
    cls = type(computer)
    if cls not in _KWARGS_CACHE:
        _KWARGS_CACHE[cls] = _get_accepted_kwargs(computer)
    accepted, has_var_keyword = _KWARGS_CACHE[cls]
    if has_var_keyword:
        return kwargs
    return {k: v for k, v in kwargs.items() if k in accepted}


class FeaturePipeline:
    """Unified feature pipeline combining all modules.

    Handles:
    - Consistent output shape regardless of missing data sources
    - NaN filling (forward-fill for prices, zero for indicators)
    - Configurable feature selection
    - Warmup management
    """

    def __init__(
        self,
        registry: FeatureRegistry | None = None,
        selected_features: list[str] | None = None,
    ) -> None:
        self._registry = registry or build_default_registry()
        self._selected = selected_features
        self._output_names: list[str] | None = None

    @property
    def registry(self) -> FeatureRegistry:
        return self._registry

    @property
    def warmup_bars(self) -> int:
        """Maximum warmup across all (selected) computers."""
        computers = self._get_computers()
        if not computers:
            return 0
        return max(c.warmup_bars for c in computers)

    @property
    def output_names(self) -> list[str]:
        """Ordered list of all output feature names."""
        if self._output_names is None:
            names: list[str] = []
            for computer in self._get_computers():
                names.extend(computer.output_names())
            self._output_names = names
        return self._output_names

    @property
    def n_features(self) -> int:
        return len(self.output_names)

    def _get_computers(self) -> list[FeatureComputer]:
        if self._selected is not None:
            return self._registry.select(self._selected)
        return self._registry.get_all()

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
        """Compute all features at the given index.

        Missing cross-market data (nq_closes, eth_closes) is handled
        gracefully — features that need them return NaN or zero.

        Args:
            idx: Current bar index.
            timestamps, opens, highs, lows, closes, volumes: OHLCV arrays.
            **kwargs: Optional arrays (bids, asks, bid_sizes, ask_sizes,
                      trade_sizes, trade_sides, nq_closes, eth_closes).

        Returns:
            Dict of feature_name -> float for all selected features.
        """
        result: dict[str, float] = {}

        for computer in self._get_computers():
            filtered = _filter_kwargs(computer, kwargs)
            features = computer.compute(
                idx,
                timestamps,
                opens,
                highs,
                lows,
                closes,
                volumes,
                **filtered,
            )
            result.update(features)

        return result

    def compute_batch(
        self,
        timestamps: NDArray[np.int64],
        opens: NDArray[np.float64],
        highs: NDArray[np.float64],
        lows: NDArray[np.float64],
        closes: NDArray[np.float64],
        volumes: NDArray[np.float64],
        *,
        fill_nan: bool = True,
        **kwargs: NDArray[np.float64] | None,
    ) -> NDArray[np.float64]:
        """Compute features for all bars, returning a 2D array.

        Works with OHLCV-only data (no order book or trade data required).
        Features that need missing optional arrays (bids, asks, trade_sizes,
        nq_closes, eth_closes, etc.) return NaN for those specific columns;
        all OHLCV-derived features (volatility, momentum, temporal, regime)
        compute normally.

        Column order matches ``self.output_names`` exactly.  Use
        ``pipeline.output_names`` to get the feature name for each column.

        Args:
            timestamps: Bar open timestamps in milliseconds (int64).
            opens, highs, lows, closes, volumes: OHLCV float64 arrays,
                all the same length.
            fill_nan: If True (default), forward-fill NaN values column-wise
                after warmup.  NaN columns that never have a valid value
                (e.g. OBI when no order-book data supplied) remain all-NaN.
            **kwargs: Optional supplementary arrays keyed by name:
                ``bids``, ``asks``, ``bid_sizes``, ``ask_sizes`` — 2-D
                (n_bars, n_levels) order-book snapshots;
                ``trade_sizes``, ``trade_sides`` — per-bar aggregated trade
                data (1-D);
                ``nq_closes``, ``eth_closes`` — cross-market close prices
                (1-D, same length as OHLCV).

        Returns:
            float64 array of shape ``(n_bars, n_features)``.
            Rows before ``warmup_bars`` contain NaN (or forward-filled values
            when ``fill_nan=True``).
        """
        n_bars = len(timestamps)
        names = self.output_names
        n_features = len(names)

        output = np.full((n_bars, n_features), float("nan"), dtype=np.float64)

        for i in range(n_bars):
            result = self.compute(
                i,
                timestamps,
                opens,
                highs,
                lows,
                closes,
                volumes,
                **kwargs,
            )
            for j, name in enumerate(names):
                output[i, j] = result.get(name, float("nan"))

        if fill_nan:
            output = _forward_fill_nan(output)

        return output


def _forward_fill_nan(data: NDArray[np.float64]) -> NDArray[np.float64]:
    """Forward-fill NaN values column-wise.

    Only fills after the first valid value in each column.
    """
    result = data.copy()
    for j in range(result.shape[1]):
        col = result[:, j]
        last_valid = float("nan")
        for i in range(len(col)):
            if np.isfinite(col[i]):
                last_valid = col[i]
            elif np.isfinite(last_valid):
                col[i] = last_valid
    return result
