"""
Difference-of-Means (DOM) probe implemented in PyTorch.

Binary case
-----------
Given training hidden states split by label (positive / negative):
    μ_true  = mean of positive hidden states
    μ_false = mean of negative hidden states
    w       = μ_true − μ_false          (direction vector)
    μ       = 0.5 * (μ_true + μ_false)  (midpoint)

Score for a test sample h:
    score(h) = ((h − μ) · w) / ||w||

Prediction: sign(score) → positive → class 1, negative → class 0.

Multi-class case
----------------
    μ_c    = mean hidden state of class c
    μ_avg  = mean over all class centroids
    v_c    = μ_c − μ_avg               (class direction vector)

Score a test sample h:
    Subtract μ_avg: h' = h − μ_avg
    Score for class c: h' · v_c
    Predict argmax_c score_c
"""
from __future__ import annotations

import logging
from typing import List, Literal, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


class DOMProbe:
    """Difference-of-Means probe.

    Parameters
    ----------
    task_type : "binary" | "multiclass"
    device    : "cuda" | "cpu"
    """

    def __init__(
        self,
        task_type: Literal["binary", "multiclass"] = "binary",
        device: str = "cuda",
    ):
        self.task_type = task_type
        self.device = device if torch.cuda.is_available() else "cpu"
        self._classes: Optional[List[int]] = None
        # Binary
        self._w: Optional[torch.Tensor] = None      # direction vector
        self._mu: Optional[torch.Tensor] = None     # midpoint
        # Multi-class
        self._class_vecs: Optional[torch.Tensor] = None  # [n_classes, hidden_dim]
        self._mu_avg: Optional[torch.Tensor] = None      # [hidden_dim]

    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray,
            X_val=None, y_val=None) -> "DOMProbe":
        """Compute class centroids.  No gradient descent; pure statistics.

        Parameters
        ----------
        X : [N, hidden_dim]
        y : [N]  integer labels
        X_val, y_val : ignored (accepted for API compatibility)
        """
        self._classes = sorted(set(y.tolist()))
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_t = torch.tensor(y, dtype=torch.long).to(self.device)

        if self.task_type == "binary":
            if len(self._classes) != 2:
                raise ValueError(
                    f"Binary DOM requires exactly 2 classes, got: {self._classes}"
                )
            mask_true  = (y_t == 1)
            mask_false = (y_t == 0)
            mu_true  = X_t[mask_true].mean(dim=0)
            mu_false = X_t[mask_false].mean(dim=0)
            self._w  = mu_true - mu_false                   # [hidden_dim]
            self._mu = 0.5 * (mu_true + mu_false)           # [hidden_dim]
            logger.debug("DOM binary probe fitted. ||w|| = %.4f", self._w.norm().item())

        else:  # multiclass
            centroids = []
            for c in self._classes:
                mask = (y_t == c)
                centroids.append(X_t[mask].mean(dim=0))
            centroids_t = torch.stack(centroids, dim=0)     # [n_classes, hidden_dim]
            self._mu_avg = centroids_t.mean(dim=0)          # [hidden_dim]
            self._class_vecs = centroids_t - self._mu_avg.unsqueeze(0)
            logger.debug(
                "DOM multiclass probe fitted. %d classes, hidden_dim=%d",
                len(self._classes), X_t.shape[1],
            )

        return self

    # ------------------------------------------------------------------
    @torch.no_grad()
    def score(self, X: np.ndarray) -> np.ndarray:
        """Return raw projection scores.

        Binary   → 1-D [N] (positive = class 1)
        Multiclass → 2-D [N, n_classes]
        """
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)

        if self.task_type == "binary":
            if self._w is None:
                raise RuntimeError("Call fit() before score().")
            h_centered = X_t - self._mu.unsqueeze(0)       # [N, D]
            w_norm = self._w / (self._w.norm() + 1e-8)     # [D]
            scores = (h_centered @ w_norm).cpu().numpy()   # [N]
            return scores

        else:
            if self._class_vecs is None:
                raise RuntimeError("Call fit() before score().")
            h_centered = X_t - self._mu_avg.unsqueeze(0)  # [N, D]
            scores = (h_centered @ self._class_vecs.T)     # [N, n_classes]
            return scores.cpu().numpy()

    @torch.no_grad()
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted integer class labels."""
        scores = self.score(X)
        if self.task_type == "binary":
            return (scores >= 0).astype(int)
        else:
            return np.argmax(scores, axis=1)

    def fit_from_dataloader(self, dataloader) -> "DOMProbe":
        """Compute class centroids by streaming through a DataLoader.

        Each batch yields (X [B, D], y [B]). Memory usage is O(D), not O(N*D).
        """
        class_sums = {}
        class_counts = {}
        for X_batch, y_batch in dataloader:
            if isinstance(y_batch, torch.Tensor):
                y_batch = y_batch.numpy()
            if isinstance(X_batch, torch.Tensor):
                X_batch = X_batch.numpy()
            for c in np.unique(y_batch):
                c = int(c)
                mask = y_batch == c
                batch_sum = X_batch[mask].sum(axis=0)
                if c not in class_sums:
                    class_sums[c] = batch_sum.astype(np.float64)
                    class_counts[c] = int(mask.sum())
                else:
                    class_sums[c] += batch_sum.astype(np.float64)
                    class_counts[c] += int(mask.sum())

        self._classes = sorted(class_sums.keys())
        centroids = []
        for c in self._classes:
            centroids.append(class_sums[c] / class_counts[c])
        centroids_np = np.stack(centroids, axis=0)  # [n_classes, D]
        centroids_t = torch.tensor(centroids_np, dtype=torch.float32).to(self.device)

        if self.task_type == "binary" and len(self._classes) == 2:
            self._w = centroids_t[1] - centroids_t[0]
            self._mu = 0.5 * (centroids_t[1] + centroids_t[0])
        else:
            self._mu_avg = centroids_t.mean(dim=0)
            self._class_vecs = centroids_t - self._mu_avg.unsqueeze(0)

        return self

    def save(self, path: str) -> None:
        """Save trained probe weights."""
        payload = {"task_type": self.task_type, "classes": self._classes}
        if self.task_type == "binary":
            payload.update({"w": self._w, "mu": self._mu})
        else:
            payload.update({"class_vecs": self._class_vecs, "mu_avg": self._mu_avg})
        torch.save(payload, path)
