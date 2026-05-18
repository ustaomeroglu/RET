from __future__ import annotations

import torch
import torch.nn as nn

from ret.models.predictor import ResidualBlock


class SupervisionRewardHead(nn.Module):
    """Maps z_t -> scalar reward prediction.

    Architecture: Linear(z_dim, hidden) -> LayerNorm -> GELU
                  -> [ResidualBlock(hidden)] * (num_layers - 1)
                  -> Linear(hidden, 1)
    """

    def __init__(self, z_dim: int, hidden_dim: int = 256, num_layers: int = 3):
        super().__init__()
        layers: list[nn.Module] = [
            nn.Linear(z_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        ]
        for _ in range(num_layers - 1):
            layers.append(ResidualBlock(hidden_dim))
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, T, z_dim)
        Returns:
            r_pred: (B, T, 1)
        """
        return self.net(z)


class SupervisionClassificationHead(nn.Module):
    """Maps z_t -> class logits for a categorical per-token target.

    Same backbone as SupervisionRewardHead but outputs `num_classes` logits
    instead of a single scalar. Used for supervised RET variants where the
    reward-head slot carries class indices (e.g. MMLU subject group).
    """

    def __init__(self, z_dim: int, num_classes: int,
                 hidden_dim: int = 256, num_layers: int = 3):
        super().__init__()
        layers: list[nn.Module] = [
            nn.Linear(z_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        ]
        for _ in range(num_layers - 1):
            layers.append(ResidualBlock(hidden_dim))
        layers.append(nn.Linear(hidden_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, T, z_dim) -> logits: (B, T, num_classes)."""
        return self.net(z)


class SupervisionValueHead(nn.Module):
    """Maps z_t -> scalar value prediction.

    Same architecture as SupervisionRewardHead.
    """

    def __init__(self, z_dim: int, hidden_dim: int = 256, num_layers: int = 3):
        super().__init__()
        layers: list[nn.Module] = [
            nn.Linear(z_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        ]
        for _ in range(num_layers - 1):
            layers.append(ResidualBlock(hidden_dim))
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, T, z_dim)
        Returns:
            v_pred: (B, T, 1)
        """
        return self.net(z)
