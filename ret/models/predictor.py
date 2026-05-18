from __future__ import annotations

import torch
import torch.nn as nn

from ret.config import PredictorConfig


class ResidualBlock(nn.Module):
    """Linear → LayerNorm → GELU with a residual skip connection."""
    def __init__(self, dim: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


def build_mlp_net(
    input_dim: int,
    output_dim: int,
    hidden_dim: int = 256,
    num_layers: int = 2,
) -> nn.Sequential:
    """Build a residual MLP: Linear→LayerNorm→GELU + (num_layers-1) ResidualBlocks + Linear.

    Used by both ``Predictor`` (z_t → z_{t+1}, with input_dim == output_dim == z_dim
    or input_dim == z_dim + control_dim) and by closure-probe builders (with
    arbitrary input_dim, output_dim).
    """
    layers: list[nn.Module] = [
        nn.Linear(input_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.GELU(),
    ]
    for _ in range(num_layers - 1):
        layers.append(ResidualBlock(hidden_dim))
    layers.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*layers)


class Predictor(nn.Module):
    """Predicts z_{t+1} from z_t (autonomous) or (z_t, u_{t+1}) (controlled).

    Thin wrapper around ``build_mlp_net``: same residual-MLP architecture, just
    with the input dim derived from the autonomous/controlled mode.
    """

    def __init__(self, z_dim: int, config: PredictorConfig):
        super().__init__()
        self.mode = config.mode
        input_dim = (
            z_dim + config.control_dim if config.mode == "controlled" else z_dim
        )
        self.net = build_mlp_net(
            input_dim, z_dim, config.hidden_dim, config.num_layers,
        )

    def forward(self, z_t: torch.Tensor, u_next: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            z_t: (B, T-1, d_z)
            u_next: (B, T-1, d_u) or None for autonomous mode

        Returns:
            z_hat_next: (B, T-1, d_z)
        """
        if self.mode == "controlled":
            assert u_next is not None, "Controlled predictor requires u_next"
            x = torch.cat([z_t, u_next], dim=-1)
        else:
            x = z_t
        return self.net(x)


class TokenProjector(nn.Module):
    """Projects raw token embeddings to control signal u_{t+1}.

    Linear(d_h, d_u) + LayerNorm
    """

    def __init__(self, d_h: int, config: PredictorConfig):
        super().__init__()
        self.proj = nn.Linear(d_h, config.control_dim)
        self.norm = nn.LayerNorm(config.control_dim)

    def forward(self, token_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_embeddings: (B, T, d_h)

        Returns:
            u: (B, T, d_u)
        """
        return self.norm(self.proj(token_embeddings))
