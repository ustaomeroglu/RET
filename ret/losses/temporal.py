"""
Temporal smoothness regularization for RET z trajectories.

Total Variation (TV) loss that penalizes large token-to-token changes in z,
encouraging smooth trajectories while still allowing sparse abrupt jumps
(via L1 norm).

Usage:
    loss, metrics = temporal_smoothness_loss(z, attention_mask, loss_type="l1")
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def temporal_smoothness_loss(
    z: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    loss_type: str = "l1",
    smooth_dims: int | None = None,
) -> tuple[torch.Tensor, dict]:
    """Total Variation temporal smoothness loss on z trajectories.

    Parameters
    ----------
    z : (B, T, d_z)
        Encoder output across the sequence.
    attention_mask : (B, T), optional
        1 for valid tokens, 0 for padding.
    loss_type : "l1" | "l2" | "huber" | "relative" | "cosine"
        How to penalize temporal differences.
        l1 promotes sparse changes (smooth + occasional jumps).
        l2 penalizes all changes equally.
        huber is smooth-L1 (ignores tiny jitter, L1-like for larger changes).
        relative: ||z_{t+1} - z_t||_2 / (||z_t||_2 + eps), scale-invariant —
          penalizes change relative to current z magnitude (prevents z-shrink
          cheating when no VICReg).
        cosine: 1 - cos_sim(z_t, z_{t+1}), purely directional, scale-invariant.
    smooth_dims : int, optional
        Regularize only the first N dimensions of z. None = all dimensions.

    Returns
    -------
    (loss, metrics) where loss is a scalar and metrics is a dict for logging.
    """
    if smooth_dims is not None:
        z = z[:, :, :smooth_dims]

    # Temporal differences: (B, T-1, D)
    dz = z[:, 1:] - z[:, :-1]

    # Mask: both z_t and z_{t+1} must be non-padding
    if attention_mask is not None:
        pair_mask = (attention_mask[:, 1:] * attention_mask[:, :-1]).float()
        pair_mask = pair_mask.unsqueeze(-1)  # (B, T-1, 1)
        n_valid = pair_mask.sum().clamp(min=1)
    else:
        pair_mask = 1.0
        n_valid = float(dz.numel())

    if loss_type == "l1":
        loss = (dz.abs() * pair_mask).sum() / n_valid
    elif loss_type == "l2":
        loss = (dz.pow(2) * pair_mask).sum() / n_valid
    elif loss_type == "huber":
        # smooth_l1_loss is Huber loss with delta=1.0
        raw = F.smooth_l1_loss(dz, torch.zeros_like(dz), reduction="none")
        loss = (raw * pair_mask).sum() / n_valid
    elif loss_type == "relative":
        # ||dz||_2 / (||z_t||_2 + eps), scale-invariant
        eps = 1e-6
        dz_norm = dz.norm(dim=-1)                  # (B, T-1)
        z_norm = z[:, :-1].norm(dim=-1)            # (B, T-1)
        ratio = dz_norm / (z_norm + eps)           # (B, T-1)
        pair_mask_flat = pair_mask.squeeze(-1) if torch.is_tensor(pair_mask) else 1.0
        loss = (ratio * pair_mask_flat).sum() / n_valid
    elif loss_type == "cosine":
        # 1 - cos_sim(z_t, z_{t+1}), fully scale-invariant (directional only)
        z_a = z[:, :-1]
        z_b = z[:, 1:]
        cos = F.cosine_similarity(z_a, z_b, dim=-1)  # (B, T-1)
        pair_mask_flat = pair_mask.squeeze(-1) if torch.is_tensor(pair_mask) else 1.0
        loss = ((1.0 - cos) * pair_mask_flat).sum() / n_valid
    else:
        raise ValueError(f"Unknown smooth_loss_type: {loss_type!r}")

    with torch.no_grad():
        mean_abs = (dz.abs() * pair_mask).sum().item() / (n_valid if isinstance(n_valid, float) else n_valid.item())

    metrics = {
        "smooth/tv_loss": loss.item(),
        "smooth/mean_abs_dz": mean_abs,
    }
    return loss, metrics
