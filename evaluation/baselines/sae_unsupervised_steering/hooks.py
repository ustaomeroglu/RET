"""Forward hook that intercepts the LLM residual stream and adds a scaled
SAE feature direction when a rule is active.

The hook satisfies the same interface contract as
``evaluation/unsupervised_steering/hooks.py:SteeringHook`` so the existing
``SteeredGenerator`` can drive generation without modification.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from evaluation.baselines.sae import TopKSAE
from evaluation.baselines.sae_unsupervised_steering.config import SAESteeringRule
from evaluation.baselines.sae_unsupervised_steering.directions import build_composite_direction
from evaluation.baselines.sae_unsupervised_steering.steerer import SAESteerer

logger = logging.getLogger(__name__)


@dataclass
class _SAERuleState:
    """Mutable per-run rule tracking."""

    active_rule_idx: int | None = None
    remaining: int = 0
    triggered: set = field(default_factory=set)         # rule indices already fired
    avoid_fired: dict[int, int] = field(default_factory=dict)  # rule idx -> gate-open count


def _topk_latent_pairs(
    pre_acts_or_post: torch.Tensor,  # (d_sae,)
    top_k: int,
) -> torch.Tensor:
    """Return the top-``top_k`` latent_ids and activations as a (top_k, 2)
    tensor. Stored values are floats: column 0 is the latent index (cast to
    float for packed storage), column 1 is the (post-ReLU) activation.
    """
    k = min(top_k, pre_acts_or_post.shape[-1])
    vals, idx = pre_acts_or_post.topk(k)
    pairs = torch.stack([idx.float(), vals.float()], dim=-1)  # (k, 2)
    if k < top_k:
        pad = torch.zeros(top_k - k, 2, device=pairs.device)
        pairs = torch.cat([pairs, pad], dim=0)
    return pairs.detach().cpu()


class SAESteeringHook:
    """SAE-direction steering hook for one LLM layer.

    Pre-builds composite h-space directions for every rule at __init__ so
    the per-token cost is O(d_sae * d_in) for the SAE encode (the same
    work an SAE eval would do) plus O(d_in) for the addition.

    Phases (controlled by ``enabled``):
      * **prompt** (False): record h per token; do not modify.
      * **generation** (True): encode h, evaluate rules, optionally add δh.
    """

    def __init__(
        self,
        *,
        sae: TopKSAE,
        decoder_weight: torch.Tensor | None,
        encoder_weight: torch.Tensor,
        b_dec: torch.Tensor,
        encoder_bias: torch.Tensor,
        rules: list[SAESteeringRule],
        steerer: SAESteerer,
        direction_source: str,
        device: str,
        top_k_diag: int = 16,
    ):
        self.sae = sae
        self.decoder_weight = decoder_weight
        self.encoder_weight = encoder_weight
        self.b_dec = b_dec
        self.encoder_bias = encoder_bias
        self.rules = list(rules)
        self.steerer = steerer
        self.direction_source = direction_source
        self.device = device
        self.top_k_diag = int(top_k_diag)

        # Pre-build composite direction + sign + kind for every rule.
        self._rule_dirs: list[torch.Tensor | None] = []
        self._rule_signs: list[int] = []
        self._rule_kinds: list[str] = []
        for r in self.rules:
            if r.target_latents is not None:
                d = build_composite_direction(
                    latent_ids=r.target_latents,
                    decoder_weight=decoder_weight,
                    encoder_weight=encoder_weight,
                    direction_source=direction_source,
                )
                self._rule_dirs.append(d)
                self._rule_signs.append(+1)
                self._rule_kinds.append("attractor")
            else:
                d = build_composite_direction(
                    latent_ids=r.avoid_latents,
                    decoder_weight=decoder_weight,
                    encoder_weight=encoder_weight,
                    direction_source=direction_source,
                )
                self._rule_dirs.append(d)
                self._rule_signs.append(-1)
                self._rule_kinds.append("avoid")

        # Mutable state (matches SteeringHook contract)
        self.h_history: list[torch.Tensor] = []
        self.enabled: bool = False
        self._step: int = 0
        self._rule_state = _SAERuleState()
        self.diagnostics: list[dict] = []

        # z_features holds packed top-K latent activations per generated token,
        # parallel to the RET hook's z_features. Each entry is a (top_k_diag, 2)
        # CPU tensor [[latent_id, activation], ...]. Filled with the post-
        # perturbation top-K when a token is steered, otherwise pre-perturbation.
        self.z_features: list[torch.Tensor] = []
        # _raw alias kept for parity with the RET pipeline's saved npy.
        self.z_features_raw: list[torch.Tensor] = []

        # Populated by compute_prompt_features().
        self.prompt_diagnostics: list[dict] = []
        self.prompt_z_features: list[torch.Tensor] = []
        self.prompt_z_features_raw: list[torch.Tensor] = []

        # Lazily resolved on first call so we can match the layer's runtime
        # device (handles `device_map="auto"`).
        self._tensors_on_layer_device: bool = False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _ensure_layer_device(self, like: torch.Tensor) -> None:
        """Move SAE weights and rule directions onto ``like.device`` exactly
        once. Safe to call on every step; cheap after the first."""
        if self._tensors_on_layer_device:
            return
        dev = like.device
        if self.b_dec.device != dev:
            self.b_dec = self.b_dec.to(dev).contiguous()
        if self.encoder_weight.device != dev:
            self.encoder_weight = self.encoder_weight.to(dev).contiguous()
        if self.encoder_bias.device != dev:
            self.encoder_bias = self.encoder_bias.to(dev).contiguous()
        if self.decoder_weight is not None and self.decoder_weight.device != dev:
            self.decoder_weight = self.decoder_weight.to(dev).contiguous()
        self._rule_dirs = [
            (d.to(dev) if d is not None else None) for d in self._rule_dirs
        ]
        self._tensors_on_layer_device = True

    def _pre_acts(self, h: torch.Tensor) -> torch.Tensor:
        """Compute raw SAE pre-activations (post-bias, pre-ReLU, pre-TopK).

        ``h``: any shape ending in ``d_in``. Returns same leading shape with
        last dim == d_sae. Computed in float32 regardless of input dtype.
        """
        h_f = h.float()
        h_centered = h_f - self.b_dec
        return h_centered @ self.encoder_weight.T + self.encoder_bias

    @staticmethod
    def _post_topk(pre_acts: torch.Tensor, k: int) -> torch.Tensor:
        """Apply TopK + ReLU sparsification used by ``TopKSAE.encode``.

        Returns same shape as ``pre_acts`` with at most ``k`` nonzero entries
        along the last dim.
        """
        topk_vals, topk_idx = pre_acts.topk(k, dim=-1)
        topk_vals = torch.relu(topk_vals)
        out = torch.zeros_like(pre_acts)
        out.scatter_(-1, topk_idx, topk_vals)
        return out

    # ------------------------------------------------------------------
    # SteeredGenerator contract
    # ------------------------------------------------------------------

    def compute_prompt_features(self) -> None:
        """After the prompt forward pass, encode each prompt-token h through
        the SAE and record per-token top-K latent activations.

        Mirrors ``SteeringHook.compute_prompt_features`` shape/keys so the
        downstream code path (HTML renderer, npy save) is symmetric.
        """
        if not self.h_history:
            return
        self._ensure_layer_device(self.h_history[0])

        h_full = torch.cat(self.h_history, dim=1)  # (1, P, d_h)
        with torch.no_grad():
            pre_acts = self._pre_acts(h_full).squeeze(0)  # (P, d_sae)
            post_acts = self._post_topk(pre_acts, self.sae.k)
        P = pre_acts.shape[0]

        for i in range(P):
            top_pairs = _topk_latent_pairs(post_acts[i], self.top_k_diag)
            self.prompt_z_features.append(top_pairs)
            self.prompt_z_features_raw.append(top_pairs.clone())
            top1_lid = int(top_pairs[0, 0].item())
            top1_act = float(top_pairs[0, 1].item())
            self.prompt_diagnostics.append({
                "step": -(P - i),
                "top1_lid": top1_lid,
                "top1_act": round(top1_act, 4),
                "h_norm": round(h_full[0, i].float().norm().item(), 4),
                "was_steered": False,
                "active_rule": None,
                "rule_kind": None,
            })

    def reset(self) -> None:
        """Clear all state for a fresh generation."""
        self.h_history.clear()
        self.enabled = False
        self._step = 0
        self._rule_state = _SAERuleState()
        self.diagnostics.clear()
        self.z_features.clear()
        self.z_features_raw.clear()
        self.prompt_diagnostics.clear()
        self.prompt_z_features.clear()
        self.prompt_z_features_raw.clear()

    # ------------------------------------------------------------------
    # Rule engine
    # ------------------------------------------------------------------

    def _check_avoid_gates(self, pre_acts_t: torch.Tensor) -> dict[int, float]:
        """Evaluate every avoid rule's gate signal at this step.

        Returns ``{rule_idx: max_pre_act_over_avoid_latents}`` for each
        avoid rule, used both for diagnostics and as the gate predicate.
        ``pre_acts_t`` is a 1-D tensor of shape ``(d_sae,)``.
        """
        out: dict[int, float] = {}
        for i, r in enumerate(self.rules):
            if self._rule_kinds[i] != "avoid":
                continue
            lids = torch.tensor(
                r.avoid_latents, device=pre_acts_t.device, dtype=torch.long,
            )
            max_pre = float(pre_acts_t.index_select(0, lids).max())
            out[i] = max_pre
        return out

    def _maybe_fire_avoid(
        self,
        gate_signals: dict[int, float],
        step: int,
    ) -> None:
        """Mirror ``_check_avoid_rule`` from the RET hook: first avoid rule
        whose gate is open this step fires (sticky top-up of remaining)."""
        for i, r in enumerate(self.rules):
            if self._rule_kinds[i] != "avoid":
                continue
            if r.avoid_from_step is not None and step < r.avoid_from_step:
                continue
            if r.avoid_max_tokens is not None:
                if self._rule_state.avoid_fired.get(i, 0) >= r.avoid_max_tokens:
                    continue
            if i not in gate_signals:
                continue
            if gate_signals[i] >= r.avoid_pre_threshold:
                self._rule_state.active_rule_idx = i
                self._rule_state.remaining = max(
                    self._rule_state.remaining, r.avoid_sticky_duration,
                )
                self._rule_state.avoid_fired[i] = (
                    self._rule_state.avoid_fired.get(i, 0) + 1
                )
                logger.info(
                    "Avoid rule %d fired at step %d: max pre-act over %s = "
                    "%.3f >= %.3f (sticky=%d, remaining=%d)",
                    i, step, r.avoid_latents, gate_signals[i],
                    r.avoid_pre_threshold, r.avoid_sticky_duration,
                    self._rule_state.remaining,
                )
                return

    def _maybe_fire_attractor(self, step: int) -> None:
        """Position-triggered attractor rules. Mirrors the RET path: skipped
        while another rule is still in its sticky window."""
        if self._rule_state.remaining > 0:
            return
        for i, r in enumerate(self.rules):
            if self._rule_kinds[i] != "attractor":
                continue
            if i in self._rule_state.triggered:
                continue
            if r.trigger_position is None:
                continue
            if step == r.trigger_position:
                self._rule_state.active_rule_idx = i
                self._rule_state.remaining = r.duration
                self._rule_state.triggered.add(i)
                logger.info(
                    "Attractor rule %d fired at step %d: target %s for %d tokens",
                    i, step, r.target_latents, r.duration,
                )
                return

    # ------------------------------------------------------------------
    # Hook entry point
    # ------------------------------------------------------------------

    def __call__(
        self,
        module: nn.Module,
        input: tuple,
        output: tuple | torch.Tensor,
    ):
        h_raw = output[0] if isinstance(output, tuple) else output
        orig_dtype = h_raw.dtype
        self._ensure_layer_device(h_raw)

        h = h_raw.float()

        if not self.enabled:
            for i in range(h.shape[1]):
                self.h_history.append(h[:, i : i + 1, :].detach().clone())
            return None

        # --- Generation phase: single token -----------------------------------
        h_t = h  # (1, 1, d_h)
        with torch.no_grad():
            pre_acts = self._pre_acts(h_t).squeeze(0).squeeze(0)  # (d_sae,)
            post_acts = self._post_topk(pre_acts, self.sae.k)
        h_norm_pre = h_t.float().norm().item()

        # Avoid gates & rule firing
        gate_signals = self._check_avoid_gates(pre_acts)
        self._maybe_fire_avoid(gate_signals, self._step)
        self._maybe_fire_attractor(self._step)

        # Apply perturbation if a rule is active
        was_steered = False
        h_modified = h_t
        delta_h_norm = 0.0
        active_idx = self._rule_state.active_rule_idx
        if active_idx is not None and self._rule_state.remaining > 0:
            composite = self._rule_dirs[active_idx]
            sign = self._rule_signs[active_idx]
            delta_h = self.steerer.step(h_t.float(), composite, sign=sign)
            h_modified = h_t + delta_h
            delta_h_norm = float(delta_h.norm())
            was_steered = True
            self._rule_state.remaining -= 1
            if self._rule_state.remaining == 0:
                self._rule_state.active_rule_idx = None

        # Post-perturbation top-K (for HTML / diagnostics)
        if was_steered:
            with torch.no_grad():
                post_pre_acts = self._pre_acts(h_modified).squeeze(0).squeeze(0)
                post_post_acts = self._post_topk(post_pre_acts, self.sae.k)
            post_top_pairs = _topk_latent_pairs(post_post_acts, self.top_k_diag)
        else:
            post_top_pairs = _topk_latent_pairs(post_acts, self.top_k_diag)

        pre_top_pairs = _topk_latent_pairs(post_acts, self.top_k_diag)
        self.z_features.append(post_top_pairs)
        self.z_features_raw.append(post_top_pairs.clone())

        # Append modified h to history (so context for downstream layers in
        # later steps reflects the perturbation).
        self.h_history.append(h_modified.detach().clone())

        # Diagnostics
        diag = {
            "step": self._step,
            "was_steered": was_steered,
            "active_rule": active_idx,
            "rule_kind": (
                self._rule_kinds[active_idx] if active_idx is not None else None
            ),
            "top1_lid": int(pre_top_pairs[0, 0].item()),
            "top1_act": round(float(pre_top_pairs[0, 1].item()), 4),
            "post_top1_lid": int(post_top_pairs[0, 0].item()),
            "post_top1_act": round(float(post_top_pairs[0, 1].item()), 4),
            "h_norm_pre": round(h_norm_pre, 4),
            "h_norm_post": round(float(h_modified.float().norm()), 4),
            "delta_h_norm": round(delta_h_norm, 4),
        }
        if gate_signals:
            diag["avoid_max_pre_act"] = {
                str(k): round(v, 4) for k, v in gate_signals.items()
            }
        self.diagnostics.append(diag)
        self._step += 1

        h_out = h_modified.to(orig_dtype)
        if isinstance(output, tuple):
            return (h_out,) + output[1:]
        return h_out
