from __future__ import annotations

import torch


def variance_loss(z: torch.Tensor, target_std: float = 1.0, epsilon: float = 1e-4) -> torch.Tensor:
    """VICReg variance loss: penalize per-dimension std < target.

    Args:
        z: (B*T, d_z) flattened representations
        target_std: target standard deviation
        epsilon: small constant for numerical stability

    Returns:
        Scalar variance loss
    """
    std = (z.var(dim=0) + epsilon).sqrt()
    return torch.relu(target_std - std).mean()


def covariance_loss(z: torch.Tensor) -> torch.Tensor:
    """VICReg covariance loss: penalize off-diagonal correlations.

    Args:
        z: (B*T, d_z) flattened representations

    Returns:
        Scalar covariance loss
    """
    N, d = z.shape
    if N <= 1:
        return torch.tensor(0.0, device=z.device)
    z_centered = z - z.mean(dim=0)
    cov = (z_centered.T @ z_centered) / (N - 1)

    # Off-diagonal elements
    off_diag = cov.pow(2)
    # Zero out diagonal
    mask = ~torch.eye(d, device=z.device, dtype=torch.bool)
    off_diag = off_diag[mask]
    return off_diag.mean()


def vicreg_loss(
    z: torch.Tensor,
    lambda_var: float = 25.0,
    lambda_cov: float = 1.0,
    var_target: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Combined VICReg regularization loss.

    Args:
        z: (B, T, d_z) encoder output
        lambda_var: weight for variance loss
        lambda_cov: weight for covariance loss
        var_target: target standard deviation

    Returns:
        (total_loss, metrics_dict)
    """
    # Flatten to (B*T, d_z)
    z_flat = z.reshape(-1, z.shape[-1])

    v_loss = variance_loss(z_flat, target_std=var_target)
    c_loss = covariance_loss(z_flat)
    total = lambda_var * v_loss + lambda_cov * c_loss

    mean_std = z_flat.std(dim=0).mean().item()
    metrics = {
        "vicreg/variance_loss": v_loss.item(),
        "vicreg/covariance_loss": c_loss.item(),
        "vicreg/total": total.item(),
        "vicreg/mean_std": mean_std,
        "reg/mean_std": mean_std,
    }
    return total, metrics
