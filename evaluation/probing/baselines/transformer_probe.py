"""
Transformer Block Probe — single transformer encoder layer over input tokens.

Architecture
------------
Input [B, S, D] → down_proj Linear(D, proj_dim) →
  nn.TransformerEncoderLayer(d_model=proj_dim, nhead, dim_feedforward) →
  masked mean pool → Linear(proj_dim, 1) → sigmoid  (binary)

The down-projection keeps the probe lightweight (~1.5M params with proj_dim=256)
rather than running self-attention at D=5120 (~105M params for Q/K/V/O).

Input format: same as AttentionProbe:
    X["hidden_states"]  : np.ndarray [N, S, D]
    X["attention_mask"] : np.ndarray [N, S]  (1 = valid, 0 = padding)
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


class TransformerBlockProbe(nn.Module):
    """Single-layer transformer encoder probe over token sequences.

    Parameters
    ----------
    proj_dim       : dimension after down-projection (before transformer)
    n_heads        : number of attention heads in the transformer layer
    dim_feedforward: feedforward dimension in the transformer layer
    dropout        : dropout rate in the transformer layer
    task_type      : "binary" | "multiclass"
    n_classes      : number of classes (only used for multiclass)
    lr, max_epochs, l2 : training hyper-parameters
    device         : "cuda" | "cpu"
    train_batch_size : mini-batch size for training
    """

    def __init__(
        self,
        task_type: Literal["binary", "multiclass"] = "binary",
        n_classes: int = 2,
        proj_dim: int | None = 256,
        n_heads: int = 8,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        lr: float = 1e-3,
        max_epochs: int = 1000,
        l2: float = 1e-4,
        device: str = "cuda",
        train_batch_size: int = 64,
    ):
        """proj_dim=None disables the down-projection — the transformer
        block runs directly on the input hidden states (d_model = D)."""
        super().__init__()
        self.task_type = task_type
        self.n_classes = n_classes
        self.proj_dim = proj_dim   # None = no down-projection
        self.n_heads = n_heads
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.lr = lr
        self.max_epochs = max_epochs
        self.l2 = l2
        self.device = device if torch.cuda.is_available() else "cpu"
        self.train_batch_size = train_batch_size

        self._fitted = False

    def _build(self, D: int) -> None:
        """Lazily build layers once input dimension is known."""
        if self.proj_dim is None:
            # No down-projection: transformer runs at full D, with the
            # standard 4x feed-forward expansion (d_model=D, dim_ff=4D).
            self.down_proj = nn.Identity()
            d_model = D
            dim_ff = 4 * d_model
        else:
            self.down_proj = nn.Linear(D, self.proj_dim)
            d_model = self.proj_dim
            dim_ff = self.dim_feedforward
        if d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({self.n_heads})."
            )
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=self.n_heads,
            dim_feedforward=dim_ff,
            dropout=self.dropout,
            batch_first=True,
        )
        out_dim = 1 if self.task_type == "binary" else self.n_classes
        self.head = nn.Linear(d_model, out_dim)
        self.to(self.device)

    def _forward(
        self,
        A: torch.Tensor,      # [B, S, D]
        mask: torch.Tensor,    # [B, S]  1=valid, 0=pad
        per_position: bool = False,
    ) -> torch.Tensor:
        """Forward pass. Returns raw logits (before sigmoid/softmax).

        If per_position=True, returns logits at every position [B, S]
        instead of pooling to a single [B] prediction. Used for full-
        conversation probing where loss is masked to specific positions.
        """
        # Down-project
        x = self.down_proj(A)  # [B, S, proj_dim]

        # Transformer encoder layer
        key_padding_mask = (mask == 0)  # [B, S] True=ignore
        if per_position:
            # Causal mask for per-position prediction (no future leakage)
            S = x.shape[1]
            causal_mask = torch.triu(
                torch.full((S, S), float("-inf"), device=x.device), diagonal=1
            )
            x = self.encoder_layer(
                x, src_mask=causal_mask, src_key_padding_mask=key_padding_mask,
            )
            logits = self.head(x)  # [B, S, 1] or [B, S, K]
            if self.task_type == "binary":
                logits = logits.squeeze(-1)  # [B, S]
            return logits

        # Bidirectional for pooled prediction
        x = self.encoder_layer(x, src_key_padding_mask=key_padding_mask)  # [B, S, proj_dim]

        # Masked mean pooling (original single-prediction path)
        mask_expanded = mask.unsqueeze(-1)  # [B, S, 1]
        x = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)  # [B, proj_dim]

        # Classification head
        logits = self.head(x)  # [B, 1] or [B, K]
        if self.task_type == "binary":
            logits = logits.squeeze(-1)  # [B]
        return logits

    def fit(
        self,
        X: dict,
        y: np.ndarray,
        X_val: dict | None = None,
        y_val: np.ndarray | None = None,
        batch_size: int | None = None,
    ) -> "TransformerBlockProbe":
        """Train the probe using mini-batches.

        Parameters
        ----------
        X     : dict with "hidden_states" [N,S,D] and "attention_mask" [N,S]
        y     : [N] integer class labels
        X_val : optional validation dict
        y_val : optional validation labels
        """
        batch_size = batch_size if batch_size is not None else self.train_batch_size

        hs = torch.tensor(X["hidden_states"], dtype=torch.float32)
        mask_t = torch.tensor(X["attention_mask"], dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.long)
        N, S, D = hs.shape

        if self.task_type == "multiclass":
            self.n_classes = int(y.max()) + 1

        if not self._fitted:
            self._build(D)
            self._fitted = True

        optimizer = AdamW(self.parameters(), lr=self.lr, weight_decay=self.l2)
        loss_fn = (nn.BCEWithLogitsLoss() if self.task_type == "binary"
                   else nn.CrossEntropyLoss())

        dataset = TensorDataset(hs, mask_t, y_t)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

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

            if has_val and ((epoch + 1) % eval_every == 0 or epoch == self.max_epochs - 1):
                self.eval()
                v_preds = self.predict(X_val, batch_size=batch_size)
                v_acc = float((v_preds == y_val).mean())
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
                msg = "TransformerBlockProbe epoch %d/%d  loss=%.4f  acc=%.3f"
                args = [epoch + 1, self.max_epochs,
                        history["train_loss"][-1], history["train_acc"][-1]]
                if has_val and history["val_acc"]:
                    msg += "  val_loss=%.4f  val_acc=%.3f"
                    args += [history["val_loss"][-1], history["val_acc"][-1]]
                logger.debug(msg, *args)

        self.eval()
        self.history_ = history
        return self

    @torch.no_grad()
    def predict(self, X: dict, batch_size: int = 64) -> np.ndarray:
        """Predict class labels. Batched to avoid OOM."""
        hs = torch.tensor(X["hidden_states"], dtype=torch.float32)
        mask = torch.tensor(X["attention_mask"], dtype=torch.float32)
        all_preds = []
        for start in range(0, hs.shape[0], batch_size):
            A_b = hs[start:start + batch_size].to(self.device)
            m_b = mask[start:start + batch_size].to(self.device)
            logits = self._forward(A_b, m_b)
            if self.task_type == "binary":
                all_preds.append((logits >= 0).cpu().numpy().astype(int))
            else:
                all_preds.append(logits.argmax(dim=-1).cpu().numpy())
        return np.concatenate(all_preds, axis=0)

    @torch.no_grad()
    def predict_proba(self, X: dict, batch_size: int = 64) -> np.ndarray:
        """Return predicted probabilities. Batched to avoid OOM."""
        hs = torch.tensor(X["hidden_states"], dtype=torch.float32)
        mask = torch.tensor(X["attention_mask"], dtype=torch.float32)
        all_proba = []
        for start in range(0, hs.shape[0], batch_size):
            A_b = hs[start:start + batch_size].to(self.device)
            m_b = mask[start:start + batch_size].to(self.device)
            logits = self._forward(A_b, m_b)
            if self.task_type == "binary":
                all_proba.append(torch.sigmoid(logits).cpu().numpy())
            else:
                all_proba.append(torch.softmax(logits, dim=-1).cpu().numpy())
        return np.concatenate(all_proba, axis=0)

    def fit_from_dataloader(
        self, train_loader, val_loader=None,
    ) -> "TransformerBlockProbe":
        """Train from external DataLoaders (for lazy CausalPrefixDataset).

        Each batch yields (dict{"hidden_states": [B,S,D], "attention_mask": [B,S]}, labels [B]).
        """
        first_batch = next(iter(train_loader))
        D = first_batch[0]["hidden_states"].shape[2]

        if self.task_type == "multiclass":
            self.n_classes = max(int(first_batch[1].max()) + 1, self.n_classes)

        if not self._fitted:
            self._build(D)
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
                msg = "TransformerProbe epoch %d/%d  loss=%.4f  acc=%.3f"
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
            "proj_dim": self.proj_dim,
            "n_heads": self.n_heads,
        }, path)
