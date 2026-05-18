"""Closure predictor builders + loss / R² utilities.

The predictor and token-projector architectures live canonically in
``ret/models/predictor.py``; closure callers reuse them via thin builders
that take raw int dims (no ``PredictorConfig`` ceremony for the closure
use case where input_dim ≠ output_dim is common).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ret.config import PredictorConfig as _PredictorConfig
from ret.models.predictor import TokenProjector as _TokenProjector
from ret.models.predictor import build_mlp_net as _build_mlp_net


def build_closure_predictor(
    input_dim: int,
    output_dim: int,
    hidden_dim: int = 256,
    num_layers: int = 2,
) -> nn.Sequential:
    """Build a residual MLP predictor for closure-probe training.

    Delegates to ``ret.models.predictor.build_mlp_net`` so the closure and
    JEPA predictors share the exact same architecture (residual MLP).
    """
    return _build_mlp_net(input_dim, output_dim, hidden_dim, num_layers)


def build_token_projector(d_h: int, control_dim: int) -> _TokenProjector:
    """Build a standalone token projector (Linear + LayerNorm).

    Returns the canonical ``ret.models.predictor.TokenProjector`` instance so
    the closure pipeline shares the same module as JEPA training.
    """
    return _TokenProjector(d_h, _PredictorConfig(control_dim=control_dim))


def compute_loss(pred: torch.Tensor, target: torch.Tensor, loss_type: str) -> torch.Tensor:
    """Compute prediction loss (cosine or MSE)."""
    if loss_type == "cosine":
        pred_n = F.normalize(pred, dim=-1)
        target_n = F.normalize(target, dim=-1)
        return 1.0 - (pred_n * target_n).sum(dim=-1).mean()
    elif loss_type == "mse":
        return F.mse_loss(pred, target)
    else:
        raise ValueError(f"Unsupported closure loss_type: {loss_type!r}")


def compute_r2(pred: torch.Tensor, target: torch.Tensor) -> float:
    """R² = 1 - SS_res / SS_tot. Works for any dimensionality."""
    ss_res = (pred - target).pow(2).sum().item()
    ss_tot = (target - target.mean(dim=0)).pow(2).sum().item()
    if ss_tot < 1e-12:
        return 0.0
    return 1.0 - ss_res / ss_tot


def compute_r2_streaming(ss_res: float, target_sum: torch.Tensor,
                          target_sq_sum: torch.Tensor, n: int) -> float:
    """R² from streaming statistics, without materializing full tensors.

    Uses the identity: SS_tot = sum(y^2) - sum(y)^2 / N  (per-dim, then sum).
    """
    ss_tot = (target_sq_sum - target_sum.pow(2) / n).sum().item()
    if ss_tot < 1e-12:
        return 0.0
    return 1.0 - ss_res / ss_tot


def compute_per_element_loss(pred: torch.Tensor, target: torch.Tensor, loss_type: str) -> torch.Tensor:
    """Compute per-element loss (no reduction)."""
    if loss_type == "cosine":
        pred_n = F.normalize(pred, dim=-1)
        target_n = F.normalize(target, dim=-1)
        return 1.0 - (pred_n * target_n).sum(dim=-1)
    elif loss_type == "mse":
        return (pred - target).pow(2).mean(dim=-1)
    else:
        raise ValueError(f"Unsupported closure loss_type: {loss_type!r}")
