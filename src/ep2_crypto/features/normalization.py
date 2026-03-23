"""Feature normalization pipeline: dual path for tree and neural models.

Tree models (LightGBM, CatBoost) work best with raw features.
Neural models (GRU) need robust scaling + rank-to-gaussian transform.

All normalizers are fitted on training data only (per-fold) to prevent
information leakage. Never use global normalization.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


class RawPassthrough:
    """Pass-through normalizer for tree models. No transformation applied.

    Tree models are invariant to monotonic transformations and handle
    raw feature values natively. This exists to maintain a consistent
    interface with other normalizers.
    """

    def fit(self, data: NDArray[np.float64]) -> RawPassthrough:
        """No-op fit for interface compatibility."""
        return self

    def transform(self, data: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return data unchanged."""
        return data.copy()

    def fit_transform(self, data: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return data unchanged."""
        return data.copy()


class RobustScaler:
    """Robust scaling using median and IQR.

    Robust to outliers, which are common in crypto data.
    x_scaled = (x - median) / IQR

    Must be fit on training data only (per-fold).
    """

    def __init__(self) -> None:
        self._median: NDArray[np.float64] | None = None
        self._iqr: NDArray[np.float64] | None = None

    @property
    def is_fitted(self) -> bool:
        return self._median is not None

    def fit(self, data: NDArray[np.float64]) -> RobustScaler:
        """Fit on training data. Computes median and IQR per feature.

        Args:
            data: 2D array (n_samples, n_features).
        """
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        self._median = np.median(data, axis=0)
        q75 = np.percentile(data, 75, axis=0)
        q25 = np.percentile(data, 25, axis=0)
        self._iqr = q75 - q25
        # Avoid division by zero: set zero IQR to 1.0
        self._iqr = np.where(self._iqr > 0, self._iqr, 1.0)
        return self

    def transform(self, data: NDArray[np.float64]) -> NDArray[np.float64]:
        """Transform data using fitted median and IQR.

        Args:
            data: 2D array (n_samples, n_features).

        Returns:
            Scaled array.
        """
        if self._median is None or self._iqr is None:
            msg = "RobustScaler not fitted. Call fit() first."
            raise RuntimeError(msg)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        return (data - self._median) / self._iqr

    def fit_transform(self, data: NDArray[np.float64]) -> NDArray[np.float64]:
        """Fit and transform in one step."""
        return self.fit(data).transform(data)


class RankGaussianTransformer:
    """Rank-to-Gaussian (quantile) transformation.

    Maps features to a standard normal distribution via rank transform.
    This is the recommended normalization for neural models on financial data.

    Applied after RobustScaler for the neural path.
    Must be fit on training data only (per-fold).
    """

    def __init__(self) -> None:
        self._quantiles: list[NDArray[np.float64]] | None = None
        self._n_samples: int = 0

    @property
    def is_fitted(self) -> bool:
        return self._quantiles is not None

    def fit(self, data: NDArray[np.float64]) -> RankGaussianTransformer:
        """Fit by storing sorted training values per feature.

        Args:
            data: 2D array (n_samples, n_features).
        """
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        self._n_samples = data.shape[0]
        self._quantiles = [np.sort(data[:, j]) for j in range(data.shape[1])]
        return self

    def transform(self, data: NDArray[np.float64]) -> NDArray[np.float64]:
        """Transform by rank-mapping then applying inverse normal CDF.

        Args:
            data: 2D array (n_samples, n_features).

        Returns:
            Gaussian-transformed array.
        """
        if self._quantiles is None:
            msg = "RankGaussianTransformer not fitted. Call fit() first."
            raise RuntimeError(msg)
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        result = np.zeros_like(data)
        for j in range(data.shape[1]):
            # Rank: count of training values <= each test value
            ranks = np.searchsorted(self._quantiles[j], data[:, j], side="right")
            # Normalize to (0, 1) — clamp to avoid infinity at edges
            uniform = np.clip(ranks / (self._n_samples + 1), 0.001, 0.999)
            # Inverse normal CDF (probit)
            result[:, j] = _inv_normal_cdf(uniform)

        return result

    def fit_transform(self, data: NDArray[np.float64]) -> NDArray[np.float64]:
        """Fit and transform in one step."""
        return self.fit(data).transform(data)


def _inv_normal_cdf(p: NDArray[np.float64]) -> NDArray[np.float64]:
    """Approximate inverse normal CDF (probit function).

    Uses the rational approximation by Peter Acklam.
    Accurate to ~1e-9 for 0.001 < p < 0.999.
    """
    # Coefficients
    a1 = -3.969683028665376e1
    a2 = 2.209460984245205e2
    a3 = -2.759285104469687e2
    a4 = 1.383577518672690e2
    a5 = -3.066479806614716e1
    a6 = 2.506628277459239e0

    b1 = -5.447609879822406e1
    b2 = 1.615858368580409e2
    b3 = -1.556989798598866e2
    b4 = 6.680131188771972e1
    b5 = -1.328068155288572e1

    c1 = -7.784894002430293e-3
    c2 = -3.223964580411365e-1
    c3 = -2.400758277161838e0
    c4 = -2.549732539343734e0
    c5 = 4.374664141464968e0
    c6 = 2.938163982698783e0

    d1 = 7.784695709041462e-3
    d2 = 3.224671290700398e-1
    d3 = 2.445134137142996e0
    d4 = 3.754408661907416e0

    p_low = 0.02425
    p_high = 1.0 - p_low

    result = np.zeros_like(p)

    # Lower region
    mask_low = p < p_low
    if np.any(mask_low):
        q = np.sqrt(-2.0 * np.log(p[mask_low]))
        result[mask_low] = (
            ((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6
        ) / ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)

    # Central region
    mask_mid = (~mask_low) & (p <= p_high)
    if np.any(mask_mid):
        q = p[mask_mid] - 0.5
        r = q * q
        result[mask_mid] = (
            ((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6
        ) * q / (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)

    # Upper region
    mask_high = p > p_high
    if np.any(mask_high):
        q = np.sqrt(-2.0 * np.log(1.0 - p[mask_high]))
        result[mask_high] = -(
            ((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6
        ) / ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)

    return result


class DualNormalizationPipeline:
    """Dual normalization: raw for trees, robust+rank-gaussian for neural.

    Usage:
        pipeline = DualNormalizationPipeline()
        pipeline.fit(train_data)
        tree_features = pipeline.transform_tree(test_data)
        neural_features = pipeline.transform_neural(test_data)
    """

    def __init__(self) -> None:
        self._raw = RawPassthrough()
        self._robust = RobustScaler()
        self._rank_gauss = RankGaussianTransformer()
        self._fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def fit(self, data: NDArray[np.float64]) -> DualNormalizationPipeline:
        """Fit normalizers on training data.

        Args:
            data: 2D array (n_samples, n_features).
        """
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        self._robust.fit(data)
        self._rank_gauss.fit(data)
        self._fitted = True
        return self

    def transform_tree(self, data: NDArray[np.float64]) -> NDArray[np.float64]:
        """Transform for tree models (raw pass-through)."""
        return self._raw.transform(data)

    def transform_neural(self, data: NDArray[np.float64]) -> NDArray[np.float64]:
        """Transform for neural models (robust scaling + rank-gaussian)."""
        if not self._fitted:
            msg = "Pipeline not fitted. Call fit() first."
            raise RuntimeError(msg)
        return self._rank_gauss.transform(data)
