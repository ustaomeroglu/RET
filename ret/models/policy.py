from __future__ import annotations

import torch
import torch.nn as nn

from ret.models.predictor import ResidualBlock


class PolicyNetwork(nn.Module):
    """Maps z_t -> next-token logits.

    Architecture: Linear(z_dim, hidden) -> LayerNorm -> GELU
                  -> [ResidualBlock(hidden)] * (num_layers - 1)
                  -> Linear(hidden, vocab_size)
    """

    def __init__(self, z_dim: int, vocab_size: int, hidden_dim: int = 256,
                 num_layers: int = 3):
        super().__init__()
        layers: list[nn.Module] = [
            nn.Linear(z_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        ]
        for _ in range(num_layers - 1):
            layers.append(ResidualBlock(hidden_dim))
        layers.append(nn.Linear(hidden_dim, vocab_size))
        self.net = nn.Sequential(*layers)

    def forward(self, z_t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_t: (B, T-1, z_dim) — representations at positions 0..T-2
        Returns:
            logits: (B, T-1, vocab_size)
        """
        return self.net(z_t)
