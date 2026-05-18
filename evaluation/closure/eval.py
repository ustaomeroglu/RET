"""Per-step evaluation for closure predictors."""
from __future__ import annotations

import torch
import torch.nn as nn

from evaluation.closure.predictors import compute_per_element_loss


def evaluate_per_step(
    model: nn.Module,
    x_seq: torch.Tensor,
    target_seq: torch.Tensor,
    loss_type: str,
) -> torch.Tensor:
    """Evaluate model per time step on a single sequence.

    Args:
        model: trained closure predictor
        x_seq: (T-1, input_dim) input features for one sequence
        target_seq: (T-1, output_dim) target z_{t+1} values
        loss_type: "cosine" or "mse"

    Returns:
        (T-1,) tensor of per-step losses
    """
    model.eval()
    with torch.no_grad():
        pred = model(x_seq)  # (T-1, output_dim)
        return compute_per_element_loss(pred, target_seq, loss_type)
