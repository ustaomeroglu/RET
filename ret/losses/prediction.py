from __future__ import annotations

import torch
import torch.nn.functional as F


def cosine_prediction_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Cosine prediction loss: 1 - mean(cos_sim(pred, target)).

    Args:
        pred: (B, T-1, d_z)
        target: (B, T-1, d_z) — should be stop-gradiented before calling

    Returns:
        Scalar loss
    """
    pred = F.normalize(pred, dim=-1)
    target = F.normalize(target, dim=-1)
    cos_sim = (pred * target).sum(dim=-1)
    return 1.0 - cos_sim.mean()


def mse_prediction_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MSE prediction loss.

    Args:
        pred: (B, T-1, d_z)
        target: (B, T-1, d_z) — should be stop-gradiented before calling

    Returns:
        Scalar loss
    """
    return F.mse_loss(pred, target)


def l1_prediction_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L1 (MAE) prediction loss.

    Args:
        pred: (B, T-1, d_z)
        target: (B, T-1, d_z) — should be stop-gradiented before calling

    Returns:
        Scalar loss
    """
    return F.l1_loss(pred, target)


def masked_prediction_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    pair_mask: torch.Tensor,
    loss_type: str,
) -> torch.Tensor:
    """Compute prediction loss only on valid (non-padding) position pairs.

    Parameters
    ----------
    pred       : (B, T-1, d_z)
    target     : (B, T-1, d_z) — should be stop-gradiented before calling
    pair_mask  : (B, T-1) — 1 where both z_t and z_{t+1} are valid
    loss_type  : "mse", "l1", or "cosine"
    """
    pair_mask = pair_mask.to(pred.device)
    mask_exp = pair_mask.unsqueeze(-1)  # (B, T-1, 1)
    n_valid = pair_mask.sum().clamp(min=1)

    if loss_type == "cosine":
        pred_n = F.normalize(pred, dim=-1)
        tgt_n = F.normalize(target, dim=-1)
        cos_sim = (pred_n * tgt_n).sum(dim=-1)  # (B, T-1)
        loss = ((1.0 - cos_sim) * pair_mask).sum() / n_valid
    elif loss_type == "mse":
        loss = (((pred - target) ** 2) * mask_exp).sum() / (n_valid * pred.shape[-1])
    elif loss_type == "l1":
        loss = ((pred - target).abs() * mask_exp).sum() / (n_valid * pred.shape[-1])
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
    return loss


def build_prediction_loss(loss_type: str):
    """Factory for prediction loss functions."""
    if loss_type == "cosine":
        return cosine_prediction_loss
    elif loss_type == "mse":
        return mse_prediction_loss
    elif loss_type == "l1":
        return l1_prediction_loss
    else:
        raise ValueError(f"Unknown prediction loss type: {loss_type}")
