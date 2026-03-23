"""2-state GaussianHMM regime detector — core layer.

Uses hmmlearn's GaussianHMM for offline fitting and a custom forward
algorithm for online filtered probability computation. States are
sorted by volatility for semantic stability across refits.

Key design choices:
- Forward algorithm (not Viterbi) for online inference: gives P(state|data_{1:t})
- Weekly refit on 7-day sliding window
- States sorted by emission variance → state 0 = low vol, state 1 = high vol
- BIC model selection (n=2,3,4,5) to pick optimal state count
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import structlog
from hmmlearn.hmm import GaussianHMM

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class HMMResult:
    """Output from the HMM detector."""

    state_probabilities: tuple[float, ...]  # P(state_i | data_{1:t}) for each state
    most_likely_state: int  # argmax of state_probabilities
    n_states: int  # Number of states in the model
    is_fitted: bool  # Whether the model has been fitted


class HMMDetector:
    """2-state GaussianHMM with forward algorithm for online regime detection.

    Parameters
    ----------
    n_states : int
        Number of hidden states (default 2: low-vol, high-vol).
    fit_window : int
        Number of bars for fitting window (default 2016 = 7 days at 5-min).
    min_fit_samples : int
        Minimum samples required before fitting (default 500).
    n_iter : int
        Max EM iterations for fitting (default 100).
    auto_select_states : bool
        If True, use BIC to select n_states from {2,3,4,5} during fit.
    """

    def __init__(
        self,
        n_states: int = 2,
        fit_window: int = 2016,
        min_fit_samples: int = 500,
        n_iter: int = 100,
        auto_select_states: bool = False,
    ) -> None:
        if n_states < 2:
            msg = "n_states must be >= 2"
            raise ValueError(msg)

        self._n_states = n_states
        self._fit_window = fit_window
        self._min_fit_samples = min_fit_samples
        self._n_iter = n_iter
        self._auto_select = auto_select_states

        # Model state
        self._model: GaussianHMM | None = None
        self._is_fitted: bool = False
        # Sorted state mapping: sorted_idx -> original_idx
        self._state_order: NDArray[np.intp] | None = None

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @property
    def n_states(self) -> int:
        return self._n_states

    @property
    def warmup_bars(self) -> int:
        return self._min_fit_samples

    def fit(
        self,
        returns: NDArray[np.float64],
        volatilities: NDArray[np.float64] | None = None,
    ) -> bool:
        """Fit HMM on a window of features.

        Parameters
        ----------
        returns : NDArray[np.float64]
            Log returns array (1D).
        volatilities : NDArray[np.float64] | None
            Optional realized volatility array (1D). If provided, uses
            (return, vol) as 2D features. Otherwise uses returns only.

        Returns
        -------
        bool : True if fitting succeeded.
        """
        if len(returns) < self._min_fit_samples:
            logger.warning(
                "hmm_fit_insufficient_data",
                n_samples=len(returns),
                min_required=self._min_fit_samples,
            )
            return False

        # Use most recent fit_window samples
        window = returns[-self._fit_window :]
        if volatilities is not None:
            vol_window = volatilities[-self._fit_window :]
            features = np.column_stack([window, vol_window])
        else:
            features = window.reshape(-1, 1)

        if self._auto_select:
            return self._fit_with_bic(features)

        return self._fit_single(features, self._n_states)

    def _fit_single(self, features: NDArray[np.float64], n_states: int) -> bool:
        """Fit a single HMM with n_states."""
        model = GaussianHMM(
            n_components=n_states,
            covariance_type="full",
            n_iter=self._n_iter,
            random_state=42,
            init_params="stmc",
        )

        try:
            model.fit(features)
        except Exception:
            logger.exception("hmm_fit_failed", n_states=n_states)
            return False

        if not model.monitor_.converged:
            logger.warning("hmm_fit_not_converged", n_states=n_states)

        self._model = model
        self._n_states = n_states
        self._is_fitted = True

        # Sort states by emission variance (ascending) for semantic stability
        # State 0 = lowest vol, State N-1 = highest vol
        self._sort_states()

        return True

    def _fit_with_bic(self, features: NDArray[np.float64]) -> bool:
        """Select optimal n_states via BIC from {2, 3, 4, 5}."""
        best_bic = float("inf")
        best_n = 2

        for n in range(2, 6):
            model = GaussianHMM(
                n_components=n,
                covariance_type="full",
                n_iter=self._n_iter,
                random_state=42,
                init_params="stmc",
            )
            try:
                model.fit(features)
                bic = -2 * model.score(features) + n * np.log(len(features))
                if bic < best_bic:
                    best_bic = bic
                    best_n = n
            except Exception:
                logger.warning("hmm_bic_fit_failed", n_states=n)
                continue

        logger.info("hmm_bic_selected", n_states=best_n, bic=best_bic)
        return self._fit_single(features, best_n)

    def _sort_states(self) -> None:
        """Sort states by emission variance for semantic stability.

        After sorting, state 0 is always the lowest-volatility state.
        """
        if self._model is None:
            return

        # For each state, compute the trace of its covariance (total variance)
        covars = self._model.covars_
        if covars.ndim == 3:
            # Full covariance: (n_states, n_features, n_features)
            total_var = np.array([np.trace(covars[i]) for i in range(self._n_states)])
        else:
            # Diagonal or other: sum variances
            total_var = np.sum(covars, axis=-1) if covars.ndim == 2 else covars

        self._state_order = np.argsort(total_var)

    def predict_proba(
        self,
        returns: NDArray[np.float64],
        volatilities: NDArray[np.float64] | None = None,
    ) -> HMMResult:
        """Compute filtered state probabilities using the forward algorithm.

        Uses hmmlearn's score_samples which internally runs the forward
        algorithm and returns per-sample log-likelihoods and posteriors.

        Parameters
        ----------
        returns : NDArray[np.float64]
            Log returns up to current time (1D).
        volatilities : NDArray[np.float64] | None
            Optional volatilities matching returns.

        Returns
        -------
        HMMResult with state probabilities for the last time step.
        """
        if not self._is_fitted or self._model is None:
            return HMMResult(
                state_probabilities=tuple(1.0 / self._n_states for _ in range(self._n_states)),
                most_likely_state=0,
                n_states=self._n_states,
                is_fitted=False,
            )

        if volatilities is not None:
            features = np.column_stack([returns, volatilities])
        else:
            features = returns.reshape(-1, 1)

        try:
            # score_samples returns (log_likelihood, posteriors)
            # posteriors shape: (T, n_states) — these are filtered probabilities
            _, posteriors = self._model.score_samples(features)
        except Exception:
            logger.exception("hmm_predict_failed")
            return HMMResult(
                state_probabilities=tuple(1.0 / self._n_states for _ in range(self._n_states)),
                most_likely_state=0,
                n_states=self._n_states,
                is_fitted=True,
            )

        # Get probabilities for the last time step
        last_probs = posteriors[-1]

        # Reorder by our sorted state mapping
        if self._state_order is not None:
            sorted_probs = np.zeros(self._n_states)
            for sorted_idx, original_idx in enumerate(self._state_order):
                sorted_probs[sorted_idx] = last_probs[original_idx]
            last_probs = sorted_probs

        # Normalize (should already sum to 1, but ensure numerical stability)
        prob_sum = np.sum(last_probs)
        if prob_sum > 0:
            last_probs = last_probs / prob_sum

        most_likely = int(np.argmax(last_probs))

        return HMMResult(
            state_probabilities=tuple(float(p) for p in last_probs),
            most_likely_state=most_likely,
            n_states=self._n_states,
            is_fitted=True,
        )

    def update(
        self,
        idx: int,
        closes: NDArray[np.float64],
    ) -> HMMResult:
        """Convenience method matching the detector interface.

        Computes log returns from closes, auto-fits if not fitted,
        and returns filtered probabilities.
        """
        if idx < 1:
            return HMMResult(
                state_probabilities=tuple(1.0 / self._n_states for _ in range(self._n_states)),
                most_likely_state=0,
                n_states=self._n_states,
                is_fitted=False,
            )

        log_prices = np.log(closes[: idx + 1])
        returns = np.diff(log_prices)

        if len(returns) < self._min_fit_samples:
            return HMMResult(
                state_probabilities=tuple(1.0 / self._n_states for _ in range(self._n_states)),
                most_likely_state=0,
                n_states=self._n_states,
                is_fitted=False,
            )

        # Auto-fit on first call or when not fitted
        if not self._is_fitted:
            self.fit(returns)

        return self.predict_proba(returns)
