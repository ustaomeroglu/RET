"""
SVM probe using sklearn's LinearSVC with StandardScaler.

Matches the DOMProbe API: fit(X, y) / score(X) / predict(X).
Decision boundary is at 0 (the learned hyperplane).
Features are standardised (zero mean, unit variance) internally —
the scaler is fit on training data only.
"""
from __future__ import annotations

import logging

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

logger = logging.getLogger(__name__)


class SVMProbe:
    """Linear SVM probe for binary classification.

    Parameters
    ----------
    C             : regularisation strength (inverse)
    class_weight  : 'balanced' handles imbalanced sycophancy data
    max_iter      : convergence iterations
    """

    def __init__(
        self,
        C: float = 1.0,
        class_weight: str = "balanced",
        max_iter: int = 10000,
    ):
        self.scaler = StandardScaler()
        self.svc = LinearSVC(
            C=C,
            class_weight=class_weight,
            dual="auto",
            max_iter=max_iter,
        )

    def fit(self, X: np.ndarray, y: np.ndarray,
            X_val=None, y_val=None) -> "SVMProbe":
        """Fit scaler + SVM.  X_val / y_val are ignored (API compat)."""
        X_scaled = self.scaler.fit_transform(X)
        self.svc.fit(X_scaled, y)
        logger.debug("SVM fitted: %d samples, %d features", X.shape[0], X.shape[1])
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Return continuous decision-function values [N]."""
        return self.svc.decision_function(self.scaler.transform(X)).astype(np.float64)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return binary predictions (threshold = 0)."""
        return self.svc.predict(self.scaler.transform(X))
