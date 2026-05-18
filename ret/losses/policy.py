from __future__ import annotations

import torch
import torch.nn.functional as F


def policy_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Cross-entropy next-token prediction loss.

    Args:
        logits:  (B, T-1, V) — policy network output
        targets: (B, T-1)    — ground-truth next tokens (input_ids[:, 1:])
        mask:    (B, T-1)    — 1 for valid positions, 0 for padding. Optional.

    Returns:
        (scalar_loss, metrics_dict) where metrics_dict contains
        policy/loss, policy/accuracy, policy/top5_accuracy.
    """
    B, T_minus1, V = logits.shape

    if mask is not None:
        valid = mask.bool()
        logits_valid = logits[valid]      # (N_valid, V)
        targets_valid = targets[valid]    # (N_valid,)
        if logits_valid.shape[0] == 0:
            zero = torch.tensor(0.0, device=logits.device, requires_grad=True)
            return zero, {
                "policy/loss": 0.0,
                "policy/accuracy": 0.0,
                "policy/top5_accuracy": 0.0,
            }
        loss = F.cross_entropy(logits_valid, targets_valid)
        with torch.no_grad():
            preds = logits_valid.argmax(dim=-1)
            acc = (preds == targets_valid).float().mean().item()
            top5 = logits_valid.topk(min(5, V), dim=-1).indices
            top5_acc = (top5 == targets_valid.unsqueeze(-1)).any(dim=-1).float().mean().item()
    else:
        logits_flat = logits.reshape(-1, V)
        targets_flat = targets.reshape(-1)
        loss = F.cross_entropy(logits_flat, targets_flat)
        with torch.no_grad():
            preds = logits_flat.argmax(dim=-1)
            acc = (preds == targets_flat).float().mean().item()
            top5 = logits_flat.topk(min(5, V), dim=-1).indices
            top5_acc = (top5 == targets_flat.unsqueeze(-1)).any(dim=-1).float().mean().item()

    return loss, {
        "policy/loss": loss.item(),
        "policy/accuracy": acc,
        "policy/top5_accuracy": top5_acc,
    }
