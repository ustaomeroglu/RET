"""
Reward-gradient steering hook for supervised RET activation steering.

At each decode step, backpropagates through the RET encoder and reward head
to compute ∂reward/∂h_t, then perturbs h_t along that gradient to shift the
reward prediction up (more sycophantic) or down (less sycophantic).

The direction is **context-dependent**: it changes at every token based on the
full conversation history, because the RET encoder is a causal transformer
that processes the entire prefix h[0:t+1].

Multi-step gradient (K > 1) re-evaluates the gradient at the updated h_t
after each small step, following the manifold curvature — same idea as the
unsupervised steering's GradientSteerer in ``evaluation/unsupervised_steering/mppi.py``.

Perturbation scaling follows the user convention:
    delta = (alpha / K) * mean_norm * (grad / ||grad||)
where mean_norm = mean(||h_layer||) is the average activation magnitude at the
target layer. alpha > 0 pushes reward UP; alpha < 0 pushes reward DOWN.
"""
from __future__ import annotations

import logging
from typing import List, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class RewardGradientHook:
    """Forward hook that steers via ∂reward/∂h at each decode step.

    Attaches to a single transformer block and detects prefill vs decode
    by checking the sequence length of the layer output:
      - prefill (seq_len > 1): accumulate h_history, no perturbation
      - decode  (seq_len == 1): compute multi-step gradient perturbation

    Parameters
    ----------
    layer_module : transformer block to hook (e.g. ``model.model.layers[23]``)
    encoder      : frozen RET encoder mapping h → z (e.g. CausalTransformerEncoder)
    reward_head  : frozen reward head mapping z → scalar logit (SupervisionRewardHead)
    alpha        : steering coefficient. ``alpha > 0`` pushes reward UP (label=1),
                   ``alpha < 0`` pushes reward DOWN (label=0). When exactly 0, the
                   hook is a no-op (baseline reproduction).
    mean_norm    : mean(||h||) at the target layer, used for scaling delta.
    gradient_steps : number of gradient steps per token (default 3).
    device       : device for encoder/reward_head forward passes (typically "cuda").
    llm_dtype    : dtype of the LLM layer outputs (typically bfloat16). The hook
                   casts to float32 for gradient computation and back for injection.
    """

    def __init__(
        self,
        layer_module: nn.Module,
        encoder: nn.Module,
        reward_head: nn.Module,
        alpha: float,
        mean_norm: float,
        gradient_steps: int = 3,
        device: str = "cuda",
        llm_dtype: torch.dtype = torch.bfloat16,
        prediction_scaled: bool = False,
        self_predictor: Optional[nn.Module] = None,
        reg_lambda: float = 0.0,
        reg_epsilon: float = 0.0,
    ):
        self.layer_module = layer_module
        self.encoder = encoder
        self.reward_head = reward_head
        self.alpha = float(alpha)
        self.mean_norm = float(mean_norm)
        self.gradient_steps = max(gradient_steps, 1)
        self.device = device
        self.llm_dtype = llm_dtype
        self.prediction_scaled = prediction_scaled
        self.self_predictor = self_predictor
        self.reg_lambda = float(reg_lambda)
        self.reg_epsilon = float(reg_epsilon)
        self.reg_enabled = (self_predictor is not None) and (self.reg_lambda > 0.0)
        self.handle: Optional[torch.utils.hooks.RemovableHandle] = None

        # h_history: list of (1, 1, D) tensors in float32 on self.device
        self.h_history: List[torch.Tensor] = []
        # z_history: list of (1, 1, z_dim) detached tensors — populated only when reg_enabled
        self.z_history: List[torch.Tensor] = []

        # Diagnostics: collect per-token stats (optional, for debugging)
        self._grad_norms: List[float] = []
        self._delta_norms: List[float] = []
        self._pred_scales: List[float] = []
        self._l_base: List[float] = []
        self._l_steer: List[float] = []
        self._reg_active: List[float] = []  # fraction of K-iters with R > 0

    def reset(self):
        """Clear h_history between conversations."""
        self.h_history = []
        self.z_history = []
        self._grad_norms = []
        self._delta_norms = []
        self._pred_scales = []
        self._l_base = []
        self._l_steer = []
        self._reg_active = []
        # Free fragmented GPU memory between conversations to avoid creep.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _build_history_tensor(self) -> torch.Tensor:
        """Concatenate h_history into (1, T, D) tensor."""
        if self.h_history:
            return torch.cat(self.h_history, dim=1)  # (1, T, D)
        D = self.encoder.output_proj.in_features if hasattr(self.encoder, 'output_proj') else 5120
        return torch.zeros(1, 0, D, device=self.device, dtype=torch.float32)

    @torch.no_grad()
    def _encode_last_z(self, h_full: torch.Tensor) -> torch.Tensor:
        """Run encoder on h_full and return last-position z, detached."""
        mask = torch.ones(1, h_full.shape[1], device=self.device, dtype=torch.float32)
        z_full, _ = self.encoder(h_full, attention_mask=mask)
        return z_full[:, -1:, :].detach()

    def _compute_gradient_direction(
        self,
        h_t: torch.Tensor,
        h_hist: torch.Tensor,
        z_hat_t: Optional[torch.Tensor] = None,
        L_base: Optional[float] = None,
        return_z: bool = False,
    ):
        """Compute ∂J_reg/∂h_t via backprop through encoder + reward head.

        When ``z_hat_t`` and ``L_base`` are provided AND self.reg_enabled, the
        objective being differentiated is the regularized one:
            J_reg(h) = J(h) - λ · max(0, L_steer(h) - L_base + ε)
        where J(h) is the reward head's logit on z(h) and
        L_steer(h) = mean((z(h) - z_hat_t)^2).
        Otherwise (default) only J(h) is used (original behavior).

        Args:
            h_t      : (1, 1, D) current hidden state (float32, requires_grad)
            h_hist   : (1, T, D) prior hidden states (detached)
            z_hat_t  : (1, 1, z_dim) predictor's expectation, detached, or None
            L_base   : scalar baseline MSE (no gradient), or None
            return_z : if True, also return the detached z_t computed in the
                       forward pass (used to cache L_base on the first iter)

        Returns:
            direction: (1, 1, D) unit-norm gradient direction (detached).
                       Returns zeros if gradient is near-zero.
            logit:     float, raw reward head output (for prediction scaling).
            extras:    dict with optional fields {z_detached, l_steer_value,
                       reg_active}. Only populated when relevant.
        """
        extras: dict = {}
        with torch.enable_grad():
            h_t_grad = h_t.detach().clone().requires_grad_(True)
            h_full = torch.cat([h_hist.detach(), h_t_grad], dim=1)  # (1, T+1, D)

            # Encoder: h -> z
            mask = torch.ones(
                1, h_full.shape[1], device=self.device, dtype=torch.float32,
            )
            z_full, _ = self.encoder(h_full, attention_mask=mask)

            # Last-position z (with grad)
            z_t = z_full[:, -1:, :]  # (1, 1, z_dim)
            r_pred = self.reward_head(z_t)  # (1, 1, 1)

            # Build the scalar to backprop. Default: J = r_pred.sum().
            objective = r_pred.sum()

            # Regularizer term (subtractive): -λ · max(0, L_steer - L_base + ε)
            if self.reg_enabled and z_hat_t is not None and L_base is not None:
                # MSE per-element mean, single scalar (z_t and z_hat_t both (1, 1, z_dim))
                diff = z_t - z_hat_t.to(z_t.dtype)
                L_steer = (diff * diff).mean()
                hinge = L_steer - float(L_base) + self.reg_epsilon
                # max(0, hinge); when hinge <= 0, no gradient flows from R
                reg_term = self.reg_lambda * torch.clamp(hinge, min=0.0)
                objective = objective - reg_term
                extras["l_steer_value"] = float(L_steer.detach().item())
                extras["reg_active"] = 1.0 if hinge.detach().item() > 0.0 else 0.0

            objective.backward()

            if return_z:
                extras["z_detached"] = z_t.detach()

        logit = r_pred.detach().item()
        grad = h_t_grad.grad  # (1, 1, D)
        grad_norm = grad.reshape(-1).norm().item()
        self._grad_norms.append(grad_norm)

        if grad_norm < 1e-10:
            direction = torch.zeros_like(h_t)
        else:
            direction = (grad / grad_norm).detach()

        # Free intermediates before returning so the next K-iter has clean memory.
        del h_t_grad, h_full, z_full, z_t, r_pred, objective, mask, grad
        return direction, logit, extras

    def _compute_perturbation(self, h_t_raw: torch.Tensor) -> torch.Tensor:
        """Multi-step gradient perturbation.

        Args:
            h_t_raw: (1, 1, D) layer output in float32

        Returns:
            h_modified: (1, 1, D) perturbed hidden state in float32
        """
        if self.alpha == 0.0:
            return h_t_raw

        h_hist = self._build_history_tensor()  # (1, T, D)

        # ----- Regularizer setup (once per decode step) -----
        z_hat_t: Optional[torch.Tensor] = None
        L_base: Optional[float] = None
        if self.reg_enabled:
            # Lazy populate z_history with the prefill's last-position z if empty.
            if len(self.z_history) == 0 and h_hist.shape[1] > 0:
                z_prefill_last = self._encode_last_z(h_hist).cpu()  # store on CPU
                self.z_history.append(z_prefill_last)
            # Predictor's expectation given the previous step's post-steered z.
            if len(self.z_history) > 0:
                with torch.no_grad():
                    z_prev = self.z_history[-1].to(self.device, dtype=torch.float32)
                    z_hat_t = self.self_predictor(z_prev).detach()

        alpha_per_step = self.alpha / self.gradient_steps
        h_current = h_t_raw.clone()
        total_delta = torch.zeros_like(h_t_raw)
        reg_active_count = 0

        for _k in range(self.gradient_steps):
            # On the first iter we also extract z(h_t) (== z(h^0)) to set L_base.
            need_z = self.reg_enabled and (L_base is None) and (z_hat_t is not None)
            grad_dir, logit, extras = self._compute_gradient_direction(
                h_current, h_hist,
                z_hat_t=z_hat_t, L_base=L_base, return_z=need_z,
            )
            # Cache L_base after the first iter (h^0 == h_t_raw).
            if need_z and "z_detached" in extras and z_hat_t is not None:
                z0 = extras["z_detached"]  # (1, 1, z_dim)
                diff0 = z0 - z_hat_t.to(z0.dtype)
                L_base = float((diff0 * diff0).mean().item())
                self._l_base.append(L_base)
            # Track diagnostics
            if "l_steer_value" in extras:
                self._l_steer.append(extras["l_steer_value"])
            if "reg_active" in extras:
                reg_active_count += int(extras["reg_active"] > 0)

            if self.prediction_scaled:
                # Modulate push strength by how sycophantic the reward head
                # thinks the current state is. Under the unified label
                # convention the reward target is 1 = SYCOPHANTIC for both
                # debate and false_presupposition, and the head is an MSE
                # regressor onto that target, so a higher r_pred ⇒ more
                # sycophantic. sigmoid(r_pred) is therefore larger when
                # sycophantic (strong push) and smaller when not (weak push).
                scale = torch.sigmoid(torch.tensor(logit)).item()
            else:
                scale = 1.0
            step_delta = scale * alpha_per_step * self.mean_norm * grad_dir
            h_current = h_current + step_delta
            total_delta = total_delta + step_delta

        delta_norm = total_delta.reshape(-1).norm().item()
        self._delta_norms.append(delta_norm)
        if self.prediction_scaled:
            self._pred_scales.append(scale)  # last step's scale
        if self.reg_enabled and z_hat_t is not None:
            self._reg_active.append(reg_active_count / self.gradient_steps)

        h_modified = h_t_raw + total_delta

        # ----- Update z_history with post-steered z (one extra encoder call) -----
        if self.reg_enabled:
            h_full_post = torch.cat([h_hist.detach(), h_modified.detach()], dim=1)
            z_post = self._encode_last_z(h_full_post).cpu()  # store on CPU
            self.z_history.append(z_post)
            del h_full_post

        return h_modified

    def _hook_fn(self, module, module_input, module_output):
        if isinstance(module_output, tuple):
            h = module_output[0]
        else:
            h = module_output

        h_f32 = h.float().to(self.device)  # (B, S, D)

        if h.shape[1] > 1:
            # ---- Prefill: accumulate all positions, no perturbation ----
            for t in range(h_f32.shape[1]):
                self.h_history.append(h_f32[:, t:t+1, :].detach().clone())
            return module_output  # unchanged

        # ---- Decode: single token ----
        h_t = h_f32[:, 0:1, :]  # (1, 1, D)
        h_modified = self._compute_perturbation(h_t)
        self.h_history.append(h_modified.detach().clone())

        # Cast back to LLM dtype and inject
        h_out = h_modified.to(dtype=self.llm_dtype)
        if isinstance(module_output, tuple):
            return (h_out,) + module_output[1:]
        return h_out

    def __enter__(self) -> "RewardGradientHook":
        self.handle = self.layer_module.register_forward_hook(self._hook_fn)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None

    @property
    def diagnostics(self) -> dict:
        """Return per-token grad/delta norm statistics (for debugging)."""
        import numpy as np
        g = np.array(self._grad_norms) if self._grad_norms else np.array([0.0])
        d = np.array(self._delta_norms) if self._delta_norms else np.array([0.0])
        result = {
            "n_decode_steps": len(self._grad_norms),
            "grad_norm_mean": float(g.mean()),
            "grad_norm_std": float(g.std()),
            "delta_norm_mean": float(d.mean()),
            "delta_norm_std": float(d.std()),
            "delta_over_mean_norm": float(d.mean() / self.mean_norm) if self.mean_norm > 0 else 0,
        }
        if self._pred_scales:
            s = np.array(self._pred_scales)
            result["pred_scale_mean"] = float(s.mean())
            result["pred_scale_std"] = float(s.std())
        if self._l_base:
            result["l_base_mean"] = float(np.array(self._l_base).mean())
        if self._l_steer:
            result["l_steer_mean"] = float(np.array(self._l_steer).mean())
        if self._reg_active:
            result["reg_active_frac"] = float(np.array(self._reg_active).mean())
        return result
