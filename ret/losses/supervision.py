from __future__ import annotations

import torch
import torch.nn.functional as F


def supervision_reward_loss(
    r_pred: torch.Tensor,
    r_target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """MSE reward prediction loss.

    Args:
        r_pred:   (B, T) — predicted rewards
        r_target: (B, T) — ground-truth rewards
        mask:     (B, T) — 1 for valid positions, 0 for padding. Optional.

    Returns:
        (scalar_loss, metrics_dict)
    """
    if mask is not None:
        valid = mask.bool()
        r_pred_valid = r_pred[valid]
        r_target_valid = r_target[valid]
        if r_pred_valid.shape[0] == 0:
            zero = torch.tensor(0.0, device=r_pred.device, requires_grad=True)
            return zero, {"supervision/reward_loss": 0.0, "supervision/reward_mae": 0.0}
        loss = F.mse_loss(r_pred_valid, r_target_valid)
        with torch.no_grad():
            mae = (r_pred_valid - r_target_valid).abs().mean().item()
    else:
        loss = F.mse_loss(r_pred, r_target)
        with torch.no_grad():
            mae = (r_pred - r_target).abs().mean().item()

    return loss, {"supervision/reward_loss": loss.item(), "supervision/reward_mae": mae}


def supervision_value_loss(
    v_pred: torch.Tensor,
    v_target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """MSE value prediction loss.

    Args:
        v_pred:   (B, T) — predicted values
        v_target: (B, T) — TD target values
        mask:     (B, T) — 1 for valid positions, 0 for padding. Optional.

    Returns:
        (scalar_loss, metrics_dict)
    """
    if mask is not None:
        valid = mask.bool()
        v_pred_valid = v_pred[valid]
        v_target_valid = v_target[valid]
        if v_pred_valid.shape[0] == 0:
            zero = torch.tensor(0.0, device=v_pred.device, requires_grad=True)
            return zero, {"supervision/value_loss": 0.0, "supervision/value_mae": 0.0}
        loss = F.mse_loss(v_pred_valid, v_target_valid)
        with torch.no_grad():
            mae = (v_pred_valid - v_target_valid).abs().mean().item()
    else:
        loss = F.mse_loss(v_pred, v_target)
        with torch.no_grad():
            mae = (v_pred - v_target).abs().mean().item()

    return loss, {"supervision/value_loss": loss.item(), "supervision/value_mae": mae}
