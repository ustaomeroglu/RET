from __future__ import annotations

import copy

import torch
import torch.nn as nn


class EMATeacher(nn.Module):
    """Exponential Moving Average teacher encoder.

    Deep-copies the student encoder with all parameters frozen.
    Supports momentum warmup from warmup_momentum to target momentum.
    """

    def __init__(
        self,
        student: nn.Module,
        momentum: float = 0.996,
        warmup_momentum: float = 0.99,
        warmup_steps: int = 1000,
    ):
        super().__init__()
        self.teacher = copy.deepcopy(student)
        for p in self.teacher.parameters():
            p.requires_grad_(False)

        self.target_momentum = momentum
        self.warmup_momentum = warmup_momentum
        self.warmup_steps = warmup_steps
        self._step = 0

    @property
    def momentum(self) -> float:
        if self._step >= self.warmup_steps:
            return self.target_momentum
        alpha = self._step / self.warmup_steps
        return self.warmup_momentum + alpha * (self.target_momentum - self.warmup_momentum)

    @torch.no_grad()
    def update(self, student: nn.Module) -> None:
        """Update teacher parameters and buffers: theta_bar = tau * theta_bar + (1 - tau) * theta."""
        tau = self.momentum
        for p_teacher, p_student in zip(self.teacher.parameters(), student.parameters()):
            p_teacher.data.mul_(tau).add_(p_student.data, alpha=1.0 - tau)
        for b_teacher, b_student in zip(self.teacher.buffers(), student.buffers()):
            b_teacher.data.copy_(b_student.data)
        self._step += 1

    def forward(self, h: torch.Tensor, **kwargs) -> torch.Tensor:
        """Run teacher encoder (no gradient)."""
        z, _aux = self.teacher(h, **kwargs)
        return z
