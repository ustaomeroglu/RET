"""
Torch-based Linear Probe.

Implements logistic regression as a single linear layer trained with
cross-entropy loss on CUDA (or CPU fallback).  Supports both binary and
multi-class tasks.
"""
from __future__ import annotations

import logging
from typing import Literal, List

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


class LinearProbe:
    """Logistic regression probe implemented in PyTorch.

    Parameters
    ----------
    task_type : "binary" | "multiclass"
    lr        : learning rate for AdamW
    max_epochs: maximum training epochs
    l2        : L2 regularisation (weight decay)
    device    : "cuda" | "cpu"
    """

    def __init__(
        self,
        task_type: Literal["binary", "multiclass"] = "binary",
        lr: float = 1e-3,
        max_epochs: int = 1000,
        l2: float = 1e-4,
        device: str = "cuda",
        train_batch_size: int = 256,
    ):
        self.task_type = task_type
        self.lr = lr
        self.max_epochs = max_epochs
        self.l2 = l2
        self.device = device if torch.cuda.is_available() else "cpu"
        self.train_batch_size = train_batch_size
        self._model: nn.Linear | None = None
        self._classes: List[int] | None = None

    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray,
            X_val: np.ndarray | None = None,
            y_val: np.ndarray | None = None) -> "LinearProbe":
        """Train the probe.

        Parameters
        ----------
        X     : [N, hidden_dim]  training features
        y     : [N]              training integer class labels
        X_val : [M, hidden_dim]  optional validation features
        y_val : [M]              optional validation labels
        """
        self._classes = sorted(set(y.tolist()))
        n_classes = len(self._classes)
        if self.task_type == "binary" and n_classes != 2:
            raise ValueError(
                f"Binary task requires exactly 2 classes, got {n_classes}: {self._classes}"
            )

        out_dim = 1 if self.task_type == "binary" else n_classes
        hidden_dim = X.shape[1]

        self._model = nn.Linear(hidden_dim, out_dim, bias=True).to(self.device)
        nn.init.zeros_(self._model.weight)
        nn.init.zeros_(self._model.bias)

        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.long)

        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=self.train_batch_size, shuffle=True)

        optimizer = AdamW(self._model.parameters(), lr=self.lr, weight_decay=self.l2)

        if self.task_type == "binary":
            loss_fn = nn.BCEWithLogitsLoss()
        else:
            loss_fn = nn.CrossEntropyLoss()

        # Training history
        has_val = X_val is not None and y_val is not None
        history = {"train_loss": [], "train_acc": []}
        if has_val:
            history["val_loss"] = []
            history["val_acc"] = []
            history["val_epoch"] = []
        eval_every = max(1, self.max_epochs // 20)  # ~20 val checkpoints

        self._model.train()
        for epoch in range(self.max_epochs):
            total_loss = 0.0
            total_correct = 0
            total_samples = 0
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                logits = self._model(xb)
                if self.task_type == "binary":
                    loss = loss_fn(logits.squeeze(-1), yb.float())
                    preds = (logits.squeeze(-1) >= 0).long()
                else:
                    loss = loss_fn(logits, yb)
                    preds = logits.argmax(dim=-1)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                total_correct += (preds == yb).sum().item()
                total_samples += yb.shape[0]

            history["train_loss"].append(total_loss / len(loader))
            history["train_acc"].append(total_correct / total_samples)

            # Validation metrics at regular intervals
            if has_val and ((epoch + 1) % eval_every == 0 or epoch == self.max_epochs - 1):
                self._model.eval()
                with torch.no_grad():
                    xv = torch.tensor(X_val, dtype=torch.float32).to(self.device)
                    yv = torch.tensor(y_val, dtype=torch.long).to(self.device)
                    v_logits = self._model(xv)
                    if self.task_type == "binary":
                        v_loss = loss_fn(v_logits.squeeze(-1), yv.float()).item()
                        v_preds = (v_logits.squeeze(-1) >= 0).long()
                    else:
                        v_loss = loss_fn(v_logits, yv).item()
                        v_preds = v_logits.argmax(dim=-1)
                    v_acc = (v_preds == yv).float().mean().item()
                history["val_loss"].append(v_loss)
                history["val_acc"].append(v_acc)
                history["val_epoch"].append(epoch)
                self._model.train()

            if (epoch + 1) % 100 == 0:
                msg = "Epoch %d/%d  loss=%.4f  acc=%.3f"
                args = [epoch + 1, self.max_epochs,
                        history["train_loss"][-1], history["train_acc"][-1]]
                if has_val and history["val_acc"]:
                    msg += "  val_loss=%.4f  val_acc=%.3f"
                    args += [history["val_loss"][-1], history["val_acc"][-1]]
                logger.debug(msg, *args)

        self._model.eval()
        self.history_ = history
        return self

    def fit_from_dataloader(
        self, train_loader, val_loader=None, hidden_dim: int = None,
    ) -> "LinearProbe":
        """Train from external DataLoaders (for lazy per-token datasets).

        Each batch yields (X [B, D], y [B]).
        """
        # Infer hidden_dim from first batch
        if hidden_dim is None:
            first_batch = next(iter(train_loader))
            hidden_dim = first_batch[0].shape[1]

        self._classes = [0, 1] if self.task_type == "binary" else None
        out_dim = 1 if self.task_type == "binary" else (
            max(y.max().item() for _, y in train_loader) + 1
        )

        self._model = nn.Linear(hidden_dim, out_dim, bias=True).to(self.device)
        nn.init.zeros_(self._model.weight)
        nn.init.zeros_(self._model.bias)

        optimizer = AdamW(self._model.parameters(), lr=self.lr, weight_decay=self.l2)
        loss_fn = nn.BCEWithLogitsLoss() if self.task_type == "binary" else nn.CrossEntropyLoss()

        history = {"train_loss": [], "train_acc": []}
        has_val = val_loader is not None
        if has_val:
            history["val_loss"] = []
            history["val_acc"] = []
            history["val_epoch"] = []
        eval_every = max(1, self.max_epochs // 20)

        self._model.train()
        for epoch in range(self.max_epochs):
            total_loss = 0.0
            total_correct = 0
            total_samples = 0
            n_batches = 0

            for xb, yb in train_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device) if isinstance(yb, torch.Tensor) else torch.tensor(yb, device=self.device)
                optimizer.zero_grad()
                logits = self._model(xb)
                if self.task_type == "binary":
                    loss = loss_fn(logits.squeeze(-1), yb.float())
                    preds = (logits.squeeze(-1) >= 0).long()
                else:
                    loss = loss_fn(logits, yb.long())
                    preds = logits.argmax(dim=-1)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                total_correct += (preds == yb).sum().item()
                total_samples += yb.shape[0]
                n_batches += 1

            history["train_loss"].append(total_loss / max(n_batches, 1))
            history["train_acc"].append(total_correct / max(total_samples, 1))

            if has_val and ((epoch + 1) % eval_every == 0 or epoch == self.max_epochs - 1):
                self._model.eval()
                v_loss_sum, v_correct, v_total, v_n = 0.0, 0, 0, 0
                with torch.no_grad():
                    for xv, yv in val_loader:
                        xv = xv.to(self.device)
                        yv = yv.to(self.device) if isinstance(yv, torch.Tensor) else torch.tensor(yv, device=self.device)
                        v_logits = self._model(xv)
                        if self.task_type == "binary":
                            v_loss_sum += loss_fn(v_logits.squeeze(-1), yv.float()).item()
                            v_preds = (v_logits.squeeze(-1) >= 0).long()
                        else:
                            v_loss_sum += loss_fn(v_logits, yv.long()).item()
                            v_preds = v_logits.argmax(dim=-1)
                        v_correct += (v_preds == yv).sum().item()
                        v_total += yv.shape[0]
                        v_n += 1
                history["val_loss"].append(v_loss_sum / max(v_n, 1))
                history["val_acc"].append(v_correct / max(v_total, 1))
                history["val_epoch"].append(epoch)
                self._model.train()

            if (epoch + 1) % 100 == 0:
                msg = "Epoch %d/%d  loss=%.4f  acc=%.3f"
                args = [epoch + 1, self.max_epochs,
                        history["train_loss"][-1], history["train_acc"][-1]]
                if has_val and history["val_acc"]:
                    msg += "  val_loss=%.4f  val_acc=%.3f"
                    args += [history["val_loss"][-1], history["val_acc"][-1]]
                logger.debug(msg, *args)

        self._model.eval()
        self.history_ = history
        return self

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted integer class labels."""
        return (self.predict_proba(X) >= 0.5).astype(int) if self.task_type == "binary" \
            else np.argmax(self._get_logits(X), axis=1)

    @torch.no_grad()
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probabilities.

        Binary  → 1-D array of P(class=1).
        Multi   → 2-D array [N, n_classes].
        """
        logits = self._get_logits(X)
        if self.task_type == "binary":
            return torch.sigmoid(torch.tensor(logits)).numpy().squeeze(-1)
        else:
            return torch.softmax(torch.tensor(logits), dim=-1).numpy()

    def _get_logits(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Call fit() before predict().")
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        return self._model(X_t).cpu().numpy()

    def save(self, path: str) -> None:
        """Save trained probe weights."""
        torch.save({
            "model_state_dict": self._model.state_dict(),
            "classes": self._classes,
            "task_type": self.task_type,
        }, path)
