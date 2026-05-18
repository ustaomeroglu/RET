"""SIGReg — Sketch Isotropic Gaussian Regularizer.

Enforces Z ~ N(0, I) via random projections and the Epps-Pulley
characteristic function test statistic. From arXiv:2511.08544.

Reference implementation: github.com/lucas-maes/le-wm
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SIGReg(nn.Module):
    """Sketch Isotropic Gaussian Regularizer (Epps-Pulley test).

    Args:
        knots: number of quadrature knots for the trapezoidal integration
        num_proj: number of random projection directions per forward pass
    """

    def __init__(self, knots: int = 17, num_proj: int = 1024):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt  # trapezoidal rule: half-weight at endpoints
        window = torch.exp(-t.square() / 2.0)  # Gaussian window φ(t) = e^{-t²/2}
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Compute the SIGReg statistic.

        Args:
            z: (N, D) flat sample matrix — N samples of D-dimensional embeddings

        Returns:
            Scalar SIGReg statistic (lower = closer to N(0,I))
        """
        # Random projection directions, L2-normalized
        A = torch.randn(z.size(-1), self.num_proj, device=z.device, dtype=z.dtype)
        A = A.div_(A.norm(p=2, dim=0))

        # Project and scale by quadrature knots: (N, num_proj, knots)
        x_t = (z @ A).unsqueeze(-1) * self.t

        # Epps-Pulley: compare empirical CF to N(0,1) CF
        # ECF = mean(cos(t·h)) + i·mean(sin(t·h)); target CF = φ(t)
        err = (x_t.cos().mean(0) - self.phi).square() + x_t.sin().mean(0).square()

        # Weighted integration, scaled by sample count
        statistic = (err @ self.weights) * z.size(0)
        return statistic.mean()


def sigreg_loss(
    z: torch.Tensor,
    sigreg_module: SIGReg,
    lambda_sigreg: float = 0.09,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute SIGReg regularization loss.

    Matches the return signature of ``vicreg_loss`` for drop-in use.

    Args:
        z: (B, T, D) encoder output (or any shape with last dim = embedding dim)
        sigreg_module: instantiated SIGReg module (holds buffers on correct device)
        lambda_sigreg: weight for the SIGReg loss

    Returns:
        (weighted_loss, metrics_dict)
    """
    z_flat = z.reshape(-1, z.shape[-1])  # (B*T, D)
    raw = sigreg_module(z_flat)
    total = lambda_sigreg * raw
    mean_std = z_flat.std(dim=0).mean().item()
    return total, {
        "sigreg/statistic": raw.item(),
        "sigreg/total": total.item(),
        "sigreg/mean_std": mean_std,
        "reg/mean_std": mean_std,
    }
