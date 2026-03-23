"""Feature computation interface and registry.

Provides the base class for all feature computers and a registry
for selecting features by name.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


class FeatureComputer(ABC):
    """Abstract base class for feature computation.

    Each feature computer produces one or more named float values
    from market data arrays. Implementations must declare their
    warmup requirement and compute features using only data at
    times <= the current index (no look-ahead).
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this feature computer."""

    @property
    @abstractmethod
    def warmup_bars(self) -> int:
        """Number of bars required before valid output."""

    @abstractmethod
    def compute(
        self,
        idx: int,
        timestamps: NDArray[np.int64],
        opens: NDArray[np.float64],
        highs: NDArray[np.float64],
        lows: NDArray[np.float64],
        closes: NDArray[np.float64],
        volumes: NDArray[np.float64],
        *,
        bids: NDArray[np.float64] | None = None,
        asks: NDArray[np.float64] | None = None,
        bid_sizes: NDArray[np.float64] | None = None,
        ask_sizes: NDArray[np.float64] | None = None,
        trade_prices: NDArray[np.float64] | None = None,
        trade_sizes: NDArray[np.float64] | None = None,
        trade_sides: NDArray[np.float64] | None = None,
    ) -> dict[str, float]:
        """Compute features at the given index.

        Args:
            idx: Current bar index. Only data at indices <= idx may be used.
            timestamps: Bar timestamps in milliseconds.
            opens, highs, lows, closes, volumes: OHLCV arrays.
            bids, asks: Best bid/ask prices per bar (optional).
            bid_sizes, ask_sizes: Bid/ask sizes per bar (optional).
            trade_prices, trade_sizes, trade_sides: Trade-level data (optional).

        Returns:
            Dict mapping feature names to float values.
            Returns NaN values if idx < warmup_bars.
        """

    def output_names(self) -> list[str]:
        """Return the list of feature names this computer produces.

        Default implementation creates a dummy call. Subclasses should
        override for efficiency.
        """
        dummy = np.zeros(1, dtype=np.float64)
        dummy_ts = np.zeros(1, dtype=np.int64)
        result = self.compute(
            0, dummy_ts, dummy, dummy, dummy, dummy, dummy,
        )
        return list(result.keys())


class FeatureRegistry:
    """Registry for feature computers. Allows selection by name."""

    def __init__(self) -> None:
        self._computers: dict[str, FeatureComputer] = {}

    def register(self, computer: FeatureComputer) -> None:
        """Register a feature computer."""
        if computer.name in self._computers:
            msg = f"Feature computer '{computer.name}' already registered"
            raise ValueError(msg)
        self._computers[computer.name] = computer

    def get(self, name: str) -> FeatureComputer:
        """Get a feature computer by name."""
        if name not in self._computers:
            msg = f"Feature computer '{name}' not found. Available: {list(self._computers.keys())}"
            raise KeyError(msg)
        return self._computers[name]

    def get_all(self) -> list[FeatureComputer]:
        """Get all registered feature computers."""
        return list(self._computers.values())

    @property
    def names(self) -> list[str]:
        """List all registered feature computer names."""
        return list(self._computers.keys())

    def select(self, names: list[str]) -> list[FeatureComputer]:
        """Select multiple feature computers by name."""
        return [self.get(name) for name in names]

    def max_warmup(self) -> int:
        """Return the maximum warmup bars across all registered computers."""
        if not self._computers:
            return 0
        return max(c.warmup_bars for c in self._computers.values())

    def compute_all(
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
        """Compute all registered features at the given index."""
        result: dict[str, float] = {}
        for computer in self._computers.values():
            features = computer.compute(
                idx, timestamps, opens, highs, lows, closes, volumes, **kwargs,
            )
            result.update(features)
        return result
