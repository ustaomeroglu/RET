"""
Attention Probe — learns a soft attention over input tokens.

Binary setting
--------------
Learnable parameters:
    θ_q ∈ ℝ^D   (query-like weights → selects which tokens to focus on)
    θ_v ∈ ℝ^D   (value / concept weights → what to read from those tokens)

Given hidden states A ∈ ℝ^{S×D} for a single sample (S = #tokens):
    α = softmax(A θ_q)              attention weights  [S]
    v = A θ_v                       value scores       [S]
    score = αᵀ v                    scalar
    p(correct) = sigmoid(score)

Multi-class setting (K classes)
--------------------------------
    θ_q ∈ ℝ^D   (single attention distribution, shared across classes)
    Θ_v ∈ ℝ^{D×K} (class-specific value projections)

    α = softmax(A θ_q)              [S]
    logits = αᵀ (A Θ_v)             [K]
    p(class) = softmax(logits)

Padding is handled by masking out pad positions in the softmax
(setting their pre-softmax score to -inf).

Input format
------------
fit(X, y)  where X is a dict:
    X["hidden_states"]  : np.ndarray [N, S, D]
    X["attention_mask"] : np.ndarray [N, S]  (1 = valid, 0 = padding)
y : np.ndarray [N] integer class labels
"""
from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)

_NINF = float("-inf")


class AttentionProbe(nn.Module):
    """Attention probe over all input-token hidden states.

    Parameters
    ----------
    hidden_dim : D — hidden state dimension (inferred from first fit call)
    task_type  : "binary" | "multiclass"
    n_classes  : number of classes (only used for multiclass)
    lr, max_epochs, l2 : training hyper-parameters
    device : "cuda" | "cpu"
    """

    def __init__(
        self,
        task_type: Literal["binary", "multiclass"] = "binary",
        n_classes: int = 2,
        lr: float = 1e-3,
        max_epochs: int = 1000,
        l2: float = 1e-4,
        device: str = "cuda",
        train_batch_size: int = 64,
    ):
        super().__init__()
        self.task_type       = task_type
        self.n_classes       = n_classes
        self.lr              = lr
        self.max_epochs      = max_epochs
        self.l2              = l2
        self.device          = device if torch.cuda.is_available() else "cpu"
        self.train_batch_size = train_batch_size

        # Parameters initialized lazily in fit() once hidden_dim is known
        self._theta_q: nn.Parameter | None = None   # [D]
        self._theta_v: nn.Parameter | None = None   # [D] (binary) or [D,K] (multi)
        self._fitted = False

    # -----------------------------------------------------------------------
    def _build_params(self, D: int) -> None:
        K = self.n_classes
        self._theta_q = nn.Parameter(torch.zeros(D))
        if self.task_type == "binary":
            self._theta_v = nn.Parameter(torch.zeros(D))
            self._bias    = nn.Parameter(torch.zeros(1))
        else:
            self._theta_v = nn.Parameter(torch.zeros(D, K))
            self._bias    = nn.Parameter(torch.zeros(K))
        self.to(self.device)

    # -----------------------------------------------------------------------
    def _forward(
        self,
        A: torch.Tensor,          # [B, S, D]
        mask: torch.Tensor,       # [B, S]  1=valid
        per_position: bool = False,
    ) -> torch.Tensor:
        """Run the attention probe forward pass.

        Returns
        -------
        per_position=False (default):
            Binary    : [B] raw logits (before sigmoid)
            Multiclass: [B, K] raw logits (before softmax)
        per_position=True:
            Binary    : [B, S] raw logits at each position
            (uses cumulative attention up to each position)
        """
        if per_position and self.task_type == "binary":
            # Per-position: at each position t, compute the cumulative
            # attention-weighted score using only tokens 0..t (causal).
            v = A @ self._theta_v          # [B, S]
            q = A @ self._theta_q          # [B, S]
            # Causal mask: lower triangular
            S = A.shape[1]
            causal = torch.tril(torch.ones(S, S, device=A.device))  # [S, S]
            pad_mask = mask.unsqueeze(1)  # [B, 1, S] broadcast over query positions
            combined_mask = causal.unsqueeze(0) * pad_mask  # [B, S, S]
            # Attention logits for each query position attending to keys 0..t
            attn = q.unsqueeze(2) * torch.ones_like(q.unsqueeze(1))  # broadcast trick — just use q as gate
            # Actually simpler: per-position score = running weighted average
            # For causal attention: at position t, softmax over logits[0..t]
            q_expanded = q.unsqueeze(1).expand(-1, S, -1)  # [B, S, S] — same q at every position
            attn_logits = q.unsqueeze(1).expand_as(combined_mask)  # Not right — q is per-key, not per-query
            # The attention probe uses q as key-scoring, so at position t:
            # alpha[t, j] = softmax_j(A[j] @ theta_q, j=0..t)
            # score[t] = sum_j alpha[t,j] * v[j] + bias
            q_all = A @ self._theta_q  # [B, S] — scoring of each key
            q_all = q_all.unsqueeze(1).expand(-1, S, -1)  # [B, S_query, S_key]
            q_all = q_all.masked_fill(combined_mask == 0, _NINF)
            alpha = torch.softmax(q_all, dim=-1)  # [B, S_query, S_key]
            scores = (alpha * v.unsqueeze(1)).sum(dim=-1) + self._bias.squeeze()  # [B, S]
            return scores

        # Original pooled path
        attn_logits = A @ self._theta_q          # [B, S]
        attn_logits = attn_logits.masked_fill(mask == 0, _NINF)
        alpha = torch.softmax(attn_logits, dim=-1)  # [B, S]

        if self.task_type == "binary":
            v = A @ self._theta_v                 # [B, S]
            score = (alpha * v).sum(dim=-1) + self._bias.squeeze()  # [B]
            return score

        else:  # multiclass
            values  = A @ self._theta_v           # [B, S, K]
            logits  = (alpha.unsqueeze(-1) * values).sum(dim=1) + self._bias  # [B, K]
            return logits

    # -----------------------------------------------------------------------
    def fit(self, X, y: np.ndarray,
            X_val: dict | None = None,
            y_val: np.ndarray | None = None,
            batch_size: int | None = None) -> "AttentionProbe":
        """Train the probe using mini-batches (data stays on CPU, batches pushed to GPU).

        Parameters
        ----------
        X     : dict with keys "hidden_states" [N,S,D] and "attention_mask" [N,S],
                OR a ``MemmapTokenDataset`` for lazy per-batch loading.
        y     : [N] integer class labels (ignored when X is a MemmapTokenDataset,
                since labels are embedded in the dataset).
        X_val : optional validation dict (same format as X)
        y_val : optional validation labels
        batch_size : mini-batch size. Defaults to self.train_batch_size.
        """
        from evaluation.probing.hidden_state_io import MemmapTokenDataset, collate_pad

        batch_size = batch_size if batch_size is not None else self.train_batch_size

        # Detect lazy memmap dataset vs pre-padded dict
        if isinstance(X, MemmapTokenDataset):
            # Lazy path: dynamic per-batch padding via collate_pad
            # Infer hidden_dim from first sample
            sample_seq, _ = X[0]
            D = sample_seq.shape[1]
            N = len(X)
            loader = DataLoader(X, batch_size=batch_size, shuffle=True,
                                collate_fn=collate_pad, num_workers=0)
        else:
            # Original path: pre-padded dict
            hs   = torch.tensor(X["hidden_states"],  dtype=torch.float32)
            mask = torch.tensor(X["attention_mask"], dtype=torch.float32)
            y_t  = torch.tensor(y, dtype=torch.long)
            N, S, D = hs.shape
            dataset  = TensorDataset(hs, mask, y_t)
            loader   = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Infer n_classes from the actual training labels
        if self.task_type == "multiclass":
            self.n_classes = int(y.max()) + 1
        elif self.task_type == "binary":
            self.n_classes = 2

        if not self._fitted:
            self._build_params(D)
            self._fitted = True
        optimizer = AdamW(self.parameters(), lr=self.lr, weight_decay=self.l2)
        loss_fn   = (nn.BCEWithLogitsLoss() if self.task_type == "binary"
                     else nn.CrossEntropyLoss())

        # Training history
        has_val = X_val is not None and y_val is not None
        history = {"train_loss": [], "train_acc": []}
        if has_val:
            history["val_loss"] = []
            history["val_acc"] = []
            history["val_epoch"] = []
        eval_every = max(1, self.max_epochs // 20)

        self.train()
        for epoch in range(self.max_epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_samples = 0
            for A_b, m_b, y_b in loader:
                A_b = A_b.to(self.device)
                m_b = m_b.to(self.device)
                y_b = y_b.to(self.device)

                optimizer.zero_grad()
                logits = self._forward(A_b, m_b)
                if self.task_type == "binary":
                    loss = loss_fn(logits, y_b.float())
                    preds = (logits >= 0).long()
                else:
                    loss = loss_fn(logits, y_b)
                    preds = logits.argmax(dim=-1)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                epoch_correct += (preds == y_b).sum().item()
                epoch_samples += y_b.shape[0]

            history["train_loss"].append(epoch_loss / len(loader))
            history["train_acc"].append(epoch_correct / epoch_samples)

            # Validation metrics at regular intervals
            if has_val and ((epoch + 1) % eval_every == 0 or epoch == self.max_epochs - 1):
                self.eval()
                v_preds = self.predict(X_val, batch_size=batch_size)
                v_acc = float((v_preds == y_val).mean())
                # Compute val loss
                with torch.no_grad():
                    v_hs = torch.tensor(X_val["hidden_states"], dtype=torch.float32).to(self.device)
                    v_mk = torch.tensor(X_val["attention_mask"], dtype=torch.float32).to(self.device)
                    v_yt = torch.tensor(y_val, dtype=torch.long).to(self.device)
                    v_logits = self._forward(v_hs, v_mk)
                    if self.task_type == "binary":
                        v_loss = loss_fn(v_logits, v_yt.float()).item()
                    else:
                        v_loss = loss_fn(v_logits, v_yt).item()
                history["val_loss"].append(v_loss)
                history["val_acc"].append(v_acc)
                history["val_epoch"].append(epoch)
                self.train()

            if (epoch + 1) % 200 == 0:
                msg = "AttentionProbe epoch %d/%d  loss=%.4f  acc=%.3f"
                args = [epoch + 1, self.max_epochs,
                        history["train_loss"][-1], history["train_acc"][-1]]
                if has_val and history["val_acc"]:
                    msg += "  val_loss=%.4f  val_acc=%.3f"
                    args += [history["val_loss"][-1], history["val_acc"][-1]]
                logger.debug(msg, *args)

        self.eval()
        self.history_ = history
        return self

    # -----------------------------------------------------------------------
    @torch.no_grad()
    def predict(self, X: dict, batch_size: int = 64) -> np.ndarray:
        """Predict class labels. Batched to avoid OOM with large sequences."""
        hs   = torch.tensor(X["hidden_states"],  dtype=torch.float32)  # CPU
        mask = torch.tensor(X["attention_mask"], dtype=torch.float32)  # CPU
        all_preds = []
        for start in range(0, hs.shape[0], batch_size):
            A_b = hs[start:start+batch_size].to(self.device)
            m_b = mask[start:start+batch_size].to(self.device)
            logits = self._forward(A_b, m_b)
            if self.task_type == "binary":
                all_preds.append((logits >= 0).cpu().numpy().astype(int))
            else:
                all_preds.append(logits.argmax(dim=-1).cpu().numpy())
        return np.concatenate(all_preds, axis=0)

    @torch.no_grad()
    def predict_proba(self, X: dict, batch_size: int = 64) -> np.ndarray:
        """Return predicted probabilities. Batched to avoid OOM."""
        hs   = torch.tensor(X["hidden_states"],  dtype=torch.float32)  # CPU
        mask = torch.tensor(X["attention_mask"], dtype=torch.float32)  # CPU
        all_proba = []
        for start in range(0, hs.shape[0], batch_size):
            A_b = hs[start:start+batch_size].to(self.device)
            m_b = mask[start:start+batch_size].to(self.device)
            logits = self._forward(A_b, m_b)
            if self.task_type == "binary":
                all_proba.append(torch.sigmoid(logits).cpu().numpy())
            else:
                all_proba.append(torch.softmax(logits, dim=-1).cpu().numpy())
        return np.concatenate(all_proba, axis=0)

    def fit_from_dataloader(
        self, train_loader, val_loader=None,
    ) -> "AttentionProbe":
        """Train from external DataLoaders (for lazy CausalPrefixDataset).

        Each batch yields (dict{"hidden_states": [B,S,D], "attention_mask": [B,S]}, labels [B]).
        """
        # Infer D from first batch
        first_batch = next(iter(train_loader))
        D = first_batch[0]["hidden_states"].shape[2]

        if self.task_type == "binary":
            self.n_classes = 2

        if not self._fitted:
            self._build_params(D)
            self._fitted = True

        optimizer = AdamW(self.parameters(), lr=self.lr, weight_decay=self.l2)
        loss_fn = (nn.BCEWithLogitsLoss() if self.task_type == "binary"
                   else nn.CrossEntropyLoss())

        has_val = val_loader is not None
        history = {"train_loss": [], "train_acc": []}
        if has_val:
            history["val_loss"] = []
            history["val_acc"] = []
            history["val_epoch"] = []
        eval_every = max(1, self.max_epochs // 20)

        self.train()
        for epoch in range(self.max_epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_samples = 0
            for batch_dict, y_b in train_loader:
                A_b = batch_dict["hidden_states"].to(self.device)
                m_b = batch_dict["attention_mask"].to(self.device)
                y_b = y_b.to(self.device)

                optimizer.zero_grad()
                logits = self._forward(A_b, m_b)
                if self.task_type == "binary":
                    loss = loss_fn(logits, y_b.float())
                    preds = (logits >= 0).long()
                else:
                    loss = loss_fn(logits, y_b)
                    preds = logits.argmax(dim=-1)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                epoch_correct += (preds == y_b).sum().item()
                epoch_samples += y_b.shape[0]

            history["train_loss"].append(epoch_loss / max(len(train_loader), 1))
            history["train_acc"].append(epoch_correct / max(epoch_samples, 1))

            if has_val and ((epoch + 1) % eval_every == 0 or epoch == self.max_epochs - 1):
                self.eval()
                v_loss_sum, v_correct, v_total, v_n = 0.0, 0, 0, 0
                with torch.no_grad():
                    for batch_dict, y_b in val_loader:
                        A_b = batch_dict["hidden_states"].to(self.device)
                        m_b = batch_dict["attention_mask"].to(self.device)
                        y_b = y_b.to(self.device)
                        logits = self._forward(A_b, m_b)
                        if self.task_type == "binary":
                            v_loss_sum += loss_fn(logits, y_b.float()).item()
                            preds = (logits >= 0).long()
                        else:
                            v_loss_sum += loss_fn(logits, y_b).item()
                            preds = logits.argmax(dim=-1)
                        v_correct += (preds == y_b).sum().item()
                        v_total += y_b.shape[0]
                        v_n += 1
                history["val_loss"].append(v_loss_sum / max(v_n, 1))
                history["val_acc"].append(v_correct / max(v_total, 1))
                history["val_epoch"].append(epoch)
                self.train()

            if (epoch + 1) % 200 == 0:
                msg = "AttentionProbe epoch %d/%d  loss=%.4f  acc=%.3f"
                args = [epoch + 1, self.max_epochs,
                        history["train_loss"][-1], history["train_acc"][-1]]
                if has_val and history["val_acc"]:
                    msg += "  val_loss=%.4f  val_acc=%.3f"
                    args += [history["val_loss"][-1], history["val_acc"][-1]]
                logger.debug(msg, *args)

        self.eval()
        self.history_ = history
        return self

    def save(self, path: str) -> None:
        """Save trained probe weights."""
        torch.save({
            "state_dict": self.state_dict(),
            "task_type": self.task_type,
            "n_classes": self.n_classes,
        }, path)
