"""Feature drift detection using Population Stability Index (PSI).

PSI measures how much a feature's distribution has shifted from a reference
(training) distribution. Higher PSI = more drift.

Thresholds (industry standard):
  PSI < 0.1:  No significant drift
  PSI 0.1-0.2: Moderate drift — monitor
  PSI > 0.2:  Significant drift — alert / retrain
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import structlog

logger = structlog.get_logger(__name__)

PSI_WARN_THRESHOLD = 0.1
PSI_ALERT_THRESHOLD = 0.2
PSI_CRITICAL_THRESHOLD = 0.3


@dataclass(frozen=True)
class DriftReport:
    """Report for a single feature's drift status."""

    feature_name: str
    psi: float
    is_drifted: bool
    severity: str  # "none", "moderate", "significant", "critical"
    reference_bins: int
    current_bins: int


@dataclass
class DailyDriftSummary:
    """Aggregate drift report across all features."""

    timestamp_ms: int
    total_features: int
    drifted_features: int
    max_psi: float
    max_psi_feature: str
    feature_reports: list[DriftReport] = field(default_factory=list)
    any_alert: bool = False

    @property
    def drift_ratio(self) -> float:
        if self.total_features == 0:
            return 0.0
        return self.drifted_features / self.total_features


def compute_psi(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10,
    epsilon: float = 1e-6,
) -> float:
    """Compute Population Stability Index between two distributions.

    Uses equal-frequency binning from the reference distribution
    to avoid sensitivity to bin edge choices.

    Args:
        reference: Reference (training) feature values.
        current: Current (live) feature values.
        n_bins: Number of bins for histogram.
        epsilon: Small constant to avoid log(0).

    Returns:
        PSI value (>= 0). 0 means identical distributions.
    """
    if len(reference) < n_bins or len(current) < n_bins:
        return 0.0

    # Equal-frequency bins from reference distribution
    quantiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(reference, quantiles)
    # Ensure unique edges by adding small perturbation
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) < 3:
        return 0.0

    ref_counts, _ = np.histogram(reference, bins=bin_edges)
    cur_counts, _ = np.histogram(current, bins=bin_edges)

    ref_pct = ref_counts / len(reference) + epsilon
    cur_pct = cur_counts / len(current) + epsilon

    psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
    return max(0.0, psi)


def classify_psi(psi: float) -> str:
    """Classify PSI into severity levels."""
    if psi >= PSI_CRITICAL_THRESHOLD:
        return "critical"
    if psi >= PSI_ALERT_THRESHOLD:
        return "significant"
    if psi >= PSI_WARN_THRESHOLD:
        return "moderate"
    return "none"


class FeatureDriftDetector:
    """Monitors feature distributions for drift using PSI.

    Stores reference distributions (from training) and compares
    incoming data to detect when features have shifted.
    """

    def __init__(
        self,
        n_bins: int = 10,
        alert_threshold: float = PSI_ALERT_THRESHOLD,
        window_size: int = 2016,  # ~7 days of 5-min bars
    ) -> None:
        self._n_bins = n_bins
        self._alert_threshold = alert_threshold
        self._window_size = window_size
        self._reference: dict[str, np.ndarray] = {}
        self._current_buffer: dict[str, list[float]] = {}
        self._latest_psi: dict[str, float] = {}

    def set_reference(self, feature_name: str, values: np.ndarray) -> None:
        """Set the reference distribution for a feature (from training data)."""
        self._reference[feature_name] = np.asarray(values, dtype=np.float64)
        self._current_buffer.setdefault(feature_name, [])
        logger.info(
            "drift_reference_set",
            feature=feature_name,
            n_samples=len(values),
        )

    def set_references_batch(self, feature_values: dict[str, np.ndarray]) -> None:
        """Set reference distributions for multiple features at once."""
        for name, values in feature_values.items():
            self.set_reference(name, values)

    def update(self, feature_values: dict[str, float]) -> None:
        """Update current buffer with new feature values from one bar."""
        for name, value in feature_values.items():
            if name in self._reference:
                buf = self._current_buffer.setdefault(name, [])
                buf.append(value)
                # Keep buffer bounded
                if len(buf) > self._window_size:
                    buf.pop(0)

    def compute_drift(self, feature_name: str) -> DriftReport:
        """Compute PSI for a single feature against its reference."""
        if feature_name not in self._reference:
            return DriftReport(
                feature_name=feature_name,
                psi=0.0,
                is_drifted=False,
                severity="none",
                reference_bins=0,
                current_bins=0,
            )

        ref = self._reference[feature_name]
        buf = self._current_buffer.get(feature_name, [])
        current = np.array(buf, dtype=np.float64)

        psi = compute_psi(ref, current, n_bins=self._n_bins)
        self._latest_psi[feature_name] = psi
        severity = classify_psi(psi)
        is_drifted = psi >= self._alert_threshold

        if is_drifted:
            logger.warning(
                "feature_drift_detected",
                feature=feature_name,
                psi=round(psi, 4),
                severity=severity,
            )

        return DriftReport(
            feature_name=feature_name,
            psi=psi,
            is_drifted=is_drifted,
            severity=severity,
            reference_bins=len(ref),
            current_bins=len(current),
        )

    def compute_all_drift(self) -> list[DriftReport]:
        """Compute PSI for all tracked features."""
        reports = []
        for name in sorted(self._reference.keys()):
            reports.append(self.compute_drift(name))
        return reports

    def generate_daily_report(self, timestamp_ms: int) -> DailyDriftSummary:
        """Generate a summary drift report across all features."""
        reports = self.compute_all_drift()
        drifted = [r for r in reports if r.is_drifted]

        max_psi = 0.0
        max_psi_feature = ""
        for r in reports:
            if r.psi > max_psi:
                max_psi = r.psi
                max_psi_feature = r.feature_name

        summary = DailyDriftSummary(
            timestamp_ms=timestamp_ms,
            total_features=len(reports),
            drifted_features=len(drifted),
            max_psi=max_psi,
            max_psi_feature=max_psi_feature,
            feature_reports=reports,
            any_alert=len(drifted) > 0,
        )

        logger.info(
            "daily_drift_report",
            total=summary.total_features,
            drifted=summary.drifted_features,
            max_psi=round(summary.max_psi, 4),
            max_feature=summary.max_psi_feature,
        )

        return summary

    def get_drifted_features(self) -> list[str]:
        """Return names of features currently exceeding the alert threshold."""
        return [name for name, psi in self._latest_psi.items() if psi >= self._alert_threshold]

    def get_psi(self, feature_name: str) -> float | None:
        """Get the latest PSI value for a feature."""
        return self._latest_psi.get(feature_name)

    @property
    def feature_names(self) -> list[str]:
        """Names of all tracked features."""
        return sorted(self._reference.keys())

    def reset_buffer(self, feature_name: str | None = None) -> None:
        """Clear current buffer(s), keeping references."""
        if feature_name:
            self._current_buffer[feature_name] = []
            self._latest_psi.pop(feature_name, None)
        else:
            self._current_buffer = {k: [] for k in self._reference}
            self._latest_psi.clear()
