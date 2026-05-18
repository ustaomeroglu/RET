"""Gradient-based steering optimizer for unsupervised steering.

Two modes controlled by ``gradient_steps``:
  - ``gradient_steps=1``: Single gradient step per token (fast).
  - ``gradient_steps=K``: K small gradient steps per token, re-evaluating z
    after each step to follow manifold curvature (more precise).

The objective is displacement-based:
  maximize  (z_proc_new - z_proc_old) · d
where d = normalize(target_center - current_center).

Since z_proc_old is constant w.r.t. δh, the gradient simplifies to
  ∇_h [z_proc · d]
"""
from __future__ import annotations

import torch
import torch.nn as nn

from evaluation.unsupervised_steering.targets import preprocess_z_torch


class GradientSteerer:
    """Displacement-based gradient steering optimizer.

    Optionally includes a closure-predictor regularizer that penalizes
    excess prediction loss above a per-step baseline:
        L = L_steer  -  lambda_pred * clamp(MSE_steered − MSE_baseline, min=−eps_pred)
    where L_steer = (z_proc · displacement) is maximized via gradient ascent.
    """

    def __init__(
        self,
        *,
        strength: float,
        gradient_steps: int,
        d_h: int,
        device: str,
        lambda_pred: float = 0.0,
        epsilon_pred: float = 0.0,
        normalize_z: bool = True,
    ):
        self.strength = strength  # step size as fraction of ||h||: δh = strength * ||h|| * grad_dir
        self.gradient_steps = max(gradient_steps, 1)
        self.d_h = d_h
        self.device = device
        self.lambda_pred = lambda_pred
        self.epsilon_pred = epsilon_pred
        # When False, the gradient objective uses centered-only z (no L2 norm),
        # matching the Euclidean clustering pipeline.
        self.normalize_z = normalize_z

    def reset(self) -> None:
        """Reset state between generations (no persistent state for now)."""
        pass

    def _compute_gradient_direction(
        self,
        h_t: torch.Tensor,
        h_history_tensor: torch.Tensor,
        encoder: nn.Module,
        mu: torch.Tensor,
        displacement: torch.Tensor,
        predictor: nn.Module | None = None,
        z_prev_raw: torch.Tensor | None = None,
        baseline_loss: float | None = None,
    ) -> torch.Tensor:
        """Compute gradient direction for the steering objective (with optional
        prediction-loss regularizer). Returns unit-norm direction in h-space.
        """
        use_reg = (
            predictor is not None
            and z_prev_raw is not None
            and baseline_loss is not None
            and self.lambda_pred > 0.0
        )
        with torch.enable_grad():
            h_t_grad = h_t.detach().clone().requires_grad_(True)
            h_full = torch.cat([h_history_tensor.detach(), h_t_grad], dim=1)

            z_full, _ = encoder(h_full)
            z_raw_t = z_full[:, -1, :]  # (1, d_z) raw
            z_proc_t = preprocess_z_torch(z_raw_t, mu, normalize=self.normalize_z)
            objective = (z_proc_t * displacement.unsqueeze(0)).sum()

            if use_reg:
                z_prev = z_prev_raw.detach().to(z_raw_t.device)
                if z_prev.dim() == 1:
                    z_prev = z_prev.unsqueeze(0)  # (1, d_z)
                pred_target = predictor(z_prev)  # (1, d_z)
                mse_steered = ((pred_target - z_raw_t) ** 2).mean()
                excess = mse_steered - baseline_loss
                reg = self.lambda_pred * torch.clamp(
                    excess, min=-self.epsilon_pred,
                )
                objective = objective - reg

            objective.backward()

        grad = h_t_grad.grad  # (1, 1, d_h)
        grad_norm = grad.reshape(-1).norm()
        if grad_norm < 1e-10:
            return torch.zeros_like(h_t)
        return (grad / grad_norm).detach()

    def step(
        self,
        h_t: torch.Tensor,
        h_history: list[torch.Tensor],
        encoder: nn.Module,
        mu: torch.Tensor,
        target_center: torch.Tensor,
        current_center: torch.Tensor,
        invert: bool = False,
        predictor: nn.Module | None = None,
        z_prev_raw: torch.Tensor | None = None,
        baseline_loss: float | None = None,
    ) -> torch.Tensor:
        """Compute the optimal perturbation for the current token.

        Args:
            h_t: (1, 1, d_h) current hidden state (unmodified)
            h_history: list of (1, 1, d_h) prior (modified) hidden states
            encoder: RET encoder mapping h -> z
            mu: (d_z,) centering vector
            target_center: (d_z,) L2-normalized target center
            current_center: (d_z,) L2-normalized current cluster center
            invert: if True, use repulsion displacement (current - target)
                instead of attraction (target - current). Used by
                proximity-gated avoid_cluster rules.

        Returns:
            delta_h: (1, 1, d_h) perturbation to add to h_t
        """
        h_norm = h_t.reshape(-1).norm().item()
        if h_norm < 1e-8:
            return torch.zeros_like(h_t)

        # Displacement direction in z-space, normalized
        if invert:
            displacement = current_center - target_center  # repel
        else:
            displacement = target_center - current_center  # attract
        d_norm = displacement.norm()
        if d_norm < 1e-8:
            return torch.zeros_like(h_t)
        displacement = displacement / d_norm

        # Build history tensor once
        if h_history:
            h_hist = torch.cat(h_history, dim=1)  # (1, T-1, d_h)
        else:
            h_hist = h_t.new_zeros(1, 0, self.d_h)

        if self.gradient_steps == 1:
            # --- Single gradient step ---
            grad_dir = self._compute_gradient_direction(
                h_t, h_hist, encoder, mu, displacement,
                predictor=predictor, z_prev_raw=z_prev_raw,
                baseline_loss=baseline_loss,
            )
            return (self.strength * h_norm) * grad_dir
        else:
            # --- Iterative gradient: K small steps ---
            strength_per_step = self.strength / self.gradient_steps
            h_current = h_t.clone()
            total_delta = torch.zeros_like(h_t)

            for _k in range(self.gradient_steps):
                grad_dir = self._compute_gradient_direction(
                    h_current, h_hist, encoder, mu, displacement,
                    predictor=predictor, z_prev_raw=z_prev_raw,
                    baseline_loss=baseline_loss,
                )
                step_delta = (strength_per_step * h_norm) * grad_dir
                h_current = h_current + step_delta
                total_delta = total_delta + step_delta

            return total_delta

    # ------------------------------------------------------------------
    # Multi-layer steering (control slice of concatenated h)
    # ------------------------------------------------------------------

    def _compute_gradient_direction_multi(
        self,
        h_t_concat: torch.Tensor,
        h_history_concat: torch.Tensor,
        encoder: nn.Module,
        mu: torch.Tensor,
        displacement: torch.Tensor,
        control_slice: slice,
    ) -> torch.Tensor:
        """Gradient of (z_proc · displacement) w.r.t. only the control slice
        of the concatenated h_t tensor.

        Args:
            h_t_concat: (1, 1, L*d_h) current step's concatenated h
            h_history_concat: (1, T-1, L*d_h) prior steps' concatenated h
            encoder: RET encoder (with MultiLayerEncoder wrapper)
            mu: (d_z,) centering vector
            displacement: (d_z,) unit-norm displacement direction in z-space
            control_slice: which columns of the last dim belong to the
                control layer (e.g. slice(-d_h, None) when control is the
                last layer)

        Returns:
            (1, 1, d_h) unit-norm gradient direction for the control layer
        """
        with torch.enable_grad():
            # Split h_t_concat into non-control and control parts
            control_start = control_slice.start
            control_stop = control_slice.stop
            if control_start is None or control_stop is None:
                raise ValueError(
                    "control_slice must have explicit start and stop"
                )

            h_before = h_t_concat[..., :control_start].detach()  # may be empty
            h_after = h_t_concat[..., control_stop:].detach()    # may be empty
            h_control = (
                h_t_concat[..., control_slice]
                .detach()
                .clone()
                .requires_grad_(True)
            )

            # Stitch along feature dim
            parts = []
            if h_before.shape[-1] > 0:
                parts.append(h_before)
            parts.append(h_control)
            if h_after.shape[-1] > 0:
                parts.append(h_after)
            h_t_stitched = torch.cat(parts, dim=-1)

            # Combine with detached history
            h_full = torch.cat(
                [h_history_concat.detach(), h_t_stitched], dim=1,
            )

            z_full, _ = encoder(h_full)
            z_raw_t = z_full[:, -1, :]
            z_proc_t = preprocess_z_torch(z_raw_t, mu, normalize=self.normalize_z)
            objective = (z_proc_t * displacement.unsqueeze(0)).sum()
            objective.backward()

        grad = h_control.grad  # (1, 1, d_h)
        if grad is None:
            return torch.zeros(
                1, 1, self.d_h, device=h_t_concat.device, dtype=h_t_concat.dtype,
            )
        grad_norm = grad.reshape(-1).norm()
        if grad_norm < 1e-10:
            return torch.zeros_like(grad)
        return (grad / grad_norm).detach()

    def step_multi_layer(
        self,
        *,
        h_t_concat: torch.Tensor,       # (1, 1, L*d_h)
        h_history_concat: torch.Tensor, # (1, T-1, L*d_h)
        encoder: nn.Module,
        mu: torch.Tensor,
        target_center: torch.Tensor,
        current_center: torch.Tensor,
        control_slice: slice,
        invert: bool = False,
    ) -> torch.Tensor:
        """Compute δh_control for the control layer in a multi-layer setup.

        Args:
            invert: if True, use repulsion displacement (current - target)
                instead of attraction (target - current). Used by
                proximity-gated avoid_cluster rules.

        Returns:
            (1, 1, d_h) perturbation to add to the CONTROL layer's h_t only.
        """
        # Use control layer's h_t norm as the step-size scale
        h_control = h_t_concat[..., control_slice]
        h_norm = h_control.reshape(-1).norm().item()
        if h_norm < 1e-8:
            return torch.zeros_like(h_control)

        if invert:
            displacement = current_center - target_center  # repel
        else:
            displacement = target_center - current_center  # attract
        d_norm = displacement.norm()
        if d_norm < 1e-8:
            return torch.zeros_like(h_control)
        displacement = displacement / d_norm

        if self.gradient_steps == 1:
            grad_dir = self._compute_gradient_direction_multi(
                h_t_concat=h_t_concat,
                h_history_concat=h_history_concat,
                encoder=encoder,
                mu=mu,
                displacement=displacement,
                control_slice=control_slice,
            )
            return (self.strength * h_norm) * grad_dir
        else:
            strength_per_step = self.strength / self.gradient_steps
            # Iterative: update only the control slice between iterations
            h_t_concat_cur = h_t_concat.clone()
            total_delta = torch.zeros_like(h_control)

            for _k in range(self.gradient_steps):
                grad_dir = self._compute_gradient_direction_multi(
                    h_t_concat=h_t_concat_cur,
                    h_history_concat=h_history_concat,
                    encoder=encoder,
                    mu=mu,
                    displacement=displacement,
                    control_slice=control_slice,
                )
                step_delta = (strength_per_step * h_norm) * grad_dir
                total_delta = total_delta + step_delta
                # Update the control slice in-place on the concat tensor
                h_t_concat_cur = h_t_concat_cur.clone()
                h_t_concat_cur[..., control_slice] = (
                    h_t_concat_cur[..., control_slice] + step_delta
                )

            return total_delta
