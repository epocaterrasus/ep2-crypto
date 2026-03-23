"""Feature rotation tracking: per-feature PSI daily, auto-downweight drifted features.

Tracks daily PSI for each feature and flags features that have been
drifted (PSI > 0.3) for 7+ consecutive days for downweighting.
Generates monthly feature importance reports.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import structlog

logger = structlog.get_logger(__name__)

DRIFT_THRESHOLD = 0.3
CONSECUTIVE_DAYS_FLAG = 7
DEFAULT_DOWNWEIGHT = 0.5


@dataclass
class FeatureRotationState:
    """Tracks a single feature's drift history."""

    name: str
    daily_psi: list[float] = field(default_factory=list)
    consecutive_drift_days: int = 0
    flagged: bool = False
    weight: float = 1.0
    importance_history: list[float] = field(default_factory=list)


@dataclass(frozen=True)
class RotationReport:
    """Monthly feature rotation summary."""

    total_features: int
    flagged_features: list[str]
    downweighted_features: dict[str, float]
    importance_changes: dict[str, float]  # feature → delta importance


class FeatureRotationTracker:
    """Monitors per-feature PSI daily and manages feature weights.

    Features with persistent drift get automatically downweighted
    rather than removed, preserving the feature pipeline.
    """

    def __init__(
        self,
        drift_threshold: float = DRIFT_THRESHOLD,
        consecutive_days_to_flag: int = CONSECUTIVE_DAYS_FLAG,
        downweight_factor: float = DEFAULT_DOWNWEIGHT,
    ) -> None:
        self._drift_threshold = drift_threshold
        self._consecutive_days_to_flag = consecutive_days_to_flag
        self._downweight_factor = downweight_factor
        self._features: dict[str, FeatureRotationState] = {}

    def register_feature(
        self, name: str, initial_importance: float = 0.0
    ) -> None:
        """Register a feature for tracking."""
        self._features[name] = FeatureRotationState(
            name=name,
            importance_history=[initial_importance] if initial_importance > 0 else [],
        )

    def register_features(
        self, importances: dict[str, float] | None = None, names: list[str] | None = None
    ) -> None:
        """Register multiple features at once."""
        if importances:
            for name, imp in importances.items():
                self.register_feature(name, imp)
        elif names:
            for name in names:
                self.register_feature(name)

    def record_daily_psi(self, psi_values: dict[str, float]) -> list[str]:
        """Record daily PSI for each feature and return newly flagged features."""
        newly_flagged: list[str] = []
        for name, psi in psi_values.items():
            state = self._features.get(name)
            if state is None:
                self.register_feature(name)
                state = self._features[name]

            state.daily_psi.append(psi)

            if psi >= self._drift_threshold:
                state.consecutive_drift_days += 1
            else:
                state.consecutive_drift_days = 0

            was_flagged = state.flagged
            if state.consecutive_drift_days >= self._consecutive_days_to_flag:
                state.flagged = True
                state.weight = self._downweight_factor
                if not was_flagged:
                    newly_flagged.append(name)
                    logger.warning(
                        "feature_flagged_for_rotation",
                        feature=name,
                        consecutive_days=state.consecutive_drift_days,
                        weight=state.weight,
                    )
            elif state.consecutive_drift_days == 0 and state.flagged:
                # Recovered — restore weight
                state.flagged = False
                state.weight = 1.0
                logger.info(
                    "feature_recovered",
                    feature=name,
                )

        return newly_flagged

    def record_importance(self, importances: dict[str, float]) -> None:
        """Record feature importance snapshot (e.g., from SHAP)."""
        for name, imp in importances.items():
            state = self._features.get(name)
            if state:
                state.importance_history.append(imp)

    def get_weight(self, feature_name: str) -> float:
        """Get current weight for a feature (1.0 = normal, <1.0 = downweighted)."""
        state = self._features.get(feature_name)
        return state.weight if state else 1.0

    def get_weights(self) -> dict[str, float]:
        """Get all feature weights."""
        return {name: s.weight for name, s in self._features.items()}

    def get_flagged_features(self) -> list[str]:
        """Get features currently flagged for rotation."""
        return [name for name, s in self._features.items() if s.flagged]

    def get_downweighted_features(self) -> dict[str, float]:
        """Get features with weight < 1.0."""
        return {
            name: s.weight
            for name, s in self._features.items()
            if s.weight < 1.0
        }

    def generate_monthly_report(self) -> RotationReport:
        """Generate monthly feature rotation report."""
        importance_changes: dict[str, float] = {}
        for name, state in self._features.items():
            if len(state.importance_history) >= 2:
                recent = state.importance_history[-1]
                older = state.importance_history[0]
                importance_changes[name] = recent - older

        report = RotationReport(
            total_features=len(self._features),
            flagged_features=self.get_flagged_features(),
            downweighted_features=self.get_downweighted_features(),
            importance_changes=importance_changes,
        )

        logger.info(
            "monthly_rotation_report",
            total=report.total_features,
            flagged=len(report.flagged_features),
            downweighted=len(report.downweighted_features),
        )
        return report

    def get_feature_state(self, name: str) -> FeatureRotationState | None:
        """Get full state for a feature."""
        return self._features.get(name)

    @property
    def feature_count(self) -> int:
        return len(self._features)

    def reset(self) -> None:
        """Reset all tracking state."""
        self._features.clear()
