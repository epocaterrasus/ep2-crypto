"""Temporal feature computers: cyclical time encoding, session indicator, funding time.

Time-based features capture intraday seasonality (volume patterns differ by session),
day-of-week effects, and time-to-funding (positions adjust before 8h settlements).
All computations use only data at times <= current index (no look-ahead).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from ep2_crypto.features.base import FeatureComputer

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


class CyclicalTimeComputer(FeatureComputer):
    """Cyclical encoding of time components using sin/cos.

    Encodes minute-of-hour, hour-of-day, and day-of-week as cyclical
    features to avoid discontinuities (e.g., 23:55 is close to 00:05).

    Uses timestamps in milliseconds since epoch.
    """

    @property
    def name(self) -> str:
        return "cyclical_time"

    @property
    def warmup_bars(self) -> int:
        return 1

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
        if idx < self.warmup_bars - 1:
            return {
                "minute_sin": float("nan"), "minute_cos": float("nan"),
                "hour_sin": float("nan"), "hour_cos": float("nan"),
                "dow_sin": float("nan"), "dow_cos": float("nan"),
            }

        ts_ms = int(timestamps[idx])
        ts_s = ts_ms // 1000

        minute = (ts_s // 60) % 60
        hour = (ts_s // 3600) % 24
        # Day of week: 0=Thursday for epoch (1970-01-01 was Thursday)
        # days_since_epoch -> dow: (days + 3) % 7 gives 0=Monday
        days = ts_s // 86400
        dow = (days + 3) % 7  # 0=Monday, 6=Sunday

        two_pi = 2.0 * math.pi

        return {
            "minute_sin": math.sin(two_pi * minute / 60.0),
            "minute_cos": math.cos(two_pi * minute / 60.0),
            "hour_sin": math.sin(two_pi * hour / 24.0),
            "hour_cos": math.cos(two_pi * hour / 24.0),
            "dow_sin": math.sin(two_pi * dow / 7.0),
            "dow_cos": math.cos(two_pi * dow / 7.0),
        }

    def output_names(self) -> list[str]:
        return ["minute_sin", "minute_cos", "hour_sin", "hour_cos", "dow_sin", "dow_cos"]


class SessionComputer(FeatureComputer):
    """Trading session indicator.

    Classifies current bar into trading sessions:
    - Asia: 00:00-08:00 UTC -> 1.0
    - Europe: 08:00-16:00 UTC -> 2.0
    - US: 16:00-00:00 UTC -> 3.0

    Also provides overlap indicators for session transitions.
    """

    @property
    def name(self) -> str:
        return "session"

    @property
    def warmup_bars(self) -> int:
        return 1

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
        if idx < self.warmup_bars - 1:
            return {
                "session_asia": float("nan"),
                "session_europe": float("nan"),
                "session_us": float("nan"),
            }

        ts_ms = int(timestamps[idx])
        hour = int((ts_ms // 3_600_000) % 24)

        asia = 1.0 if 0 <= hour < 8 else 0.0
        europe = 1.0 if 8 <= hour < 16 else 0.0
        us = 1.0 if 16 <= hour < 24 else 0.0

        return {
            "session_asia": asia,
            "session_europe": europe,
            "session_us": us,
        }

    def output_names(self) -> list[str]:
        return ["session_asia", "session_europe", "session_us"]


class FundingTimeComputer(FeatureComputer):
    """Time-to-funding: minutes until next 8-hour funding settlement.

    Perpetual futures funding occurs every 8 hours at 00:00, 08:00, 16:00 UTC.
    As funding approaches, positions adjust — especially when funding rate is extreme.

    Outputs minutes until next funding (0-480) normalized to [0, 1].
    """

    @property
    def name(self) -> str:
        return "funding_time"

    @property
    def warmup_bars(self) -> int:
        return 1

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
        if idx < self.warmup_bars - 1:
            return {
                "time_to_funding_norm": float("nan"),
                "time_to_funding_min": float("nan"),
            }

        ts_ms = int(timestamps[idx])
        ts_s = ts_ms // 1000

        # Seconds since midnight UTC
        seconds_today = ts_s % 86400
        minutes_today = seconds_today / 60.0

        # Funding times at 0, 480, 960 minutes (0:00, 8:00, 16:00 UTC)
        funding_times = [0.0, 480.0, 960.0, 1440.0]  # 1440 = next day 0:00

        # Find minutes until next funding
        time_to_funding = 1440.0  # max
        for ft in funding_times:
            if ft > minutes_today:
                time_to_funding = ft - minutes_today
                break

        return {
            "time_to_funding_norm": time_to_funding / 480.0,  # Normalize to [0, 1]
            "time_to_funding_min": time_to_funding,
        }

    def output_names(self) -> list[str]:
        return ["time_to_funding_norm", "time_to_funding_min"]
