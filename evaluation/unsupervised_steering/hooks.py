"""Steering hooks — forward hooks that intercept and perturb hidden states.

Library entry: :func:`attach_steering_hook` picks the right hook class
(``SteeringHook`` for single-layer, ``MultiLayerSteeringHook`` for multi-layer
encoders) and registers it on ``ctx.layers``. Returns ``(hook, handles)``.
The caller must invoke ``h.remove()`` on each handle when done (use a
``try/finally``).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from evaluation.unsupervised_steering.config import SteeringRule
from evaluation.unsupervised_steering.mppi import GradientSteerer
from evaluation.unsupervised_steering.targets import (
    assign_cluster_torch,
    compute_target_center,
    preprocess_z_torch,
)

logger = logging.getLogger(__name__)


def attach_steering_hook(
    ctx,
    rules: list[SteeringRule],
    optimizer: GradientSteerer,
    device: str,
    *,
    metric: str = "cosine",
) -> tuple[object, list]:
    """Build the appropriate steering hook and register it on the LLM.

    Returns ``(hook, handles)``. Caller is responsible for ``h.remove()`` on
    each ``h in handles`` after the steered generation finishes (best done in
    a ``try/finally`` around the generator call).
    """
    handles: list = []
    if ctx.is_multi_layer:
        hook = MultiLayerSteeringHook(
            layer_indices=ctx.layer_indices,
            control_layer=ctx.layer_idx,
            hidden_dim=ctx.hidden_dim,
            encoder=ctx.encoder,
            optimizer=optimizer,
            mu=ctx.mu,
            centers=ctx.centers,
            centers_normed=ctx.centers_normed,
            group_labels=ctx.group_labels,
            group_to_clusters=ctx.group_to_clusters,
            rules=rules,
            device=device,
            metric=metric,
        )
        for li in ctx.layer_indices:
            if li == ctx.layer_idx:
                handles.append(
                    ctx.layers[li].register_forward_hook(hook.make_active_hook())
                )
            else:
                handles.append(
                    ctx.layers[li].register_forward_hook(hook.make_passive_hook(li))
                )
    else:
        hook = SteeringHook(
            encoder=ctx.encoder,
            optimizer=optimizer,
            mu=ctx.mu,
            centers=ctx.centers,
            centers_normed=ctx.centers_normed,
            group_labels=ctx.group_labels,
            group_to_clusters=ctx.group_to_clusters,
            rules=rules,
            device=device,
            metric=metric,
        )
        handles.append(ctx.layers[ctx.layer_idx].register_forward_hook(hook))
    return hook, handles


# ---------------------------------------------------------------------------
# Group centroid helper (used by both single- and multi-layer hooks)
# ---------------------------------------------------------------------------


def _build_group_centroids(
    centers: torch.Tensor,
    group_to_clusters: dict[int, list[int]],
    normalize: bool = True,
) -> dict[int, torch.Tensor]:
    """For each group, compute the mean of its member cluster centers.

    Used for both gate signals and repulsion targets in ``avoid_group`` rules.

    Args:
        normalize: if True (default), L2-normalize each centroid (cosine mode).
            If False, return the raw mean (Euclidean mode).
    """
    out: dict[int, torch.Tensor] = {}
    for g, members in group_to_clusters.items():
        c = centers[members].mean(dim=0)
        if normalize:
            c = c / c.norm().clamp(min=1e-8)
        out[int(g)] = c
    return out


# ---------------------------------------------------------------------------
# Rule engine state
# ---------------------------------------------------------------------------


@dataclass
class _RuleState:
    """Mutable per-rule tracking state."""

    # Current consecutive dwell in each group
    dwell_counts: dict[int, int] = field(default_factory=dict)
    last_group: int | None = None

    # Active steering
    active_rule: SteeringRule | None = None
    remaining: int = 0
    triggered: set = field(default_factory=set)  # rule indices already fired

    # Proximity-gated repulsion: per-rule-index count of how many tokens
    # have been repelled so far (for avoid_max_tokens budget enforcement).
    avoid_fired: dict[int, int] = field(default_factory=dict)


def _check_avoid_rule(
    state: _RuleState,
    rules: list[SteeringRule],
    gate_margin: dict[int, float],
    step: int,
) -> None:
    """Check every avoid rule (cluster- or group-level) every step.

    For each avoid rule, if the margin gate is currently open
    (``cos(z, c_current) − cos(z, c_avoid) ≤ avoid_margin_threshold``),
    fire it: top up the sticky countdown to ``avoid_sticky_duration``
    (so repulsion keeps acting for that many more tokens even after the
    gate closes). The first rule whose gate is open wins; later avoid
    rules in the same step are ignored. While remaining > 0 from a
    previous step but this step's gate is closed, we let remaining
    decrement naturally.

    ``gate_margin`` maps rule index -> ``cos(z, c_current) − cos(z, c_avoid)``
    where ``c_current`` is the argmax cluster's center for avoid_cluster
    rules and the centroid of the group containing the argmax cluster
    for avoid_group rules. Margin is ≥ 0 since c_current is the argmax;
    a small margin means z is leaning toward the avoid target.

    Must be called BEFORE ``_update_dwell_and_check`` so attractor rules
    don't fire while an avoid rule is already in its sticky window.
    """
    for i, rule in enumerate(rules):
        is_avoid = rule.avoid_cluster is not None or rule.avoid_group is not None
        if not is_avoid:
            continue
        if rule.avoid_from_step is not None and step < rule.avoid_from_step:
            continue
        if rule.avoid_max_tokens is not None:
            if state.avoid_fired.get(i, 0) >= rule.avoid_max_tokens:
                continue
        margin = gate_margin.get(i)
        if margin is None:
            continue
        if margin <= rule.avoid_margin_threshold:
            target_label = (
                f"C{rule.avoid_cluster}" if rule.avoid_cluster is not None
                else f"G{rule.avoid_group}"
            )
            state.active_rule = rule
            state.remaining = max(state.remaining, rule.avoid_sticky_duration)
            state.avoid_fired[i] = state.avoid_fired.get(i, 0) + 1
            logger.info(
                "Avoid rule %d gate open at step %d: margin(current,%s)=%.3f <= %.3f "
                "(sticky=%d, remaining=%d)",
                i, step, target_label, margin, rule.avoid_margin_threshold,
                rule.avoid_sticky_duration, state.remaining,
            )
            return


def _update_dwell_and_check(
    state: _RuleState,
    group_id: int,
    rules: list[SteeringRule],
    step: int,
) -> None:
    """Update dwell counters and check if any rule should fire.

    Two trigger modes are supported:
      * ``rule.trigger_position`` (position trigger): fires when ``step``
        matches the rule's position. Checked first so position rules take
        precedence over any still-accumulating dwell counters on the same
        step.
      * ``rule.trigger_group`` + ``rule.trigger_dwell`` (group-dwell trigger):
        fires when the model has been in ``trigger_group`` for at least
        ``trigger_dwell`` consecutive tokens.
    """
    # Already steering — don't re-trigger
    if state.remaining > 0:
        return

    # --- Position-based triggers (checked before group-dwell) ---
    for i, rule in enumerate(rules):
        if i in state.triggered:
            continue
        if rule.trigger_position is None:
            continue
        if step == rule.trigger_position:
            state.active_rule = rule
            state.remaining = rule.duration
            state.triggered.add(i)
            logger.info(
                "Rule %d triggered: at step %d → steer %s for %d tokens",
                i, step,
                f"G{rule.target_group}" if rule.target_group is not None
                else f"C{rule.target_cluster}",
                rule.duration,
            )
            return

    # --- Group-dwell triggers (legacy) ---
    if group_id == state.last_group:
        state.dwell_counts[group_id] = state.dwell_counts.get(group_id, 0) + 1
    else:
        state.dwell_counts = {group_id: 1}
        state.last_group = group_id

    current_dwell = state.dwell_counts.get(group_id, 0)

    for i, rule in enumerate(rules):
        if i in state.triggered:
            continue
        if rule.trigger_position is not None:
            continue  # handled above
        if rule.trigger_group == group_id and current_dwell >= rule.trigger_dwell:
            state.active_rule = rule
            state.remaining = rule.duration
            state.triggered.add(i)
            logger.info(
                "Rule %d triggered: after G%d dwell %d → steer %s for %d tokens",
                i, group_id, current_dwell,
                f"G{rule.target_group}" if rule.target_group is not None
                else f"C{rule.target_cluster}",
                rule.duration,
            )
            return


# ---------------------------------------------------------------------------
# Steering hook
# ---------------------------------------------------------------------------


class SteeringHook:
    """PyTorch forward hook that optionally modifies layer output.

    Phases:
        - **prompt** (enabled=False): passively record h for each position.
        - **generation** (enabled=True): compute z, check rules, and apply
          CEM/MPPI perturbation if a rule is active.
    """

    def __init__(
        self,
        *,
        encoder: nn.Module,
        optimizer: GradientSteerer,
        mu: torch.Tensor,
        centers: torch.Tensor,
        centers_normed: torch.Tensor,
        group_labels: torch.Tensor,
        group_to_clusters: dict[int, list[int]],
        rules: list[SteeringRule],
        device: str,
        metric: str = "cosine",
    ):
        self.encoder = encoder
        self.optimizer = optimizer
        self.mu = mu
        self.centers = centers
        self.centers_normed = centers_normed
        self.group_labels = group_labels
        self.group_to_clusters = group_to_clusters
        self.rules = rules
        self.device = device
        self.metric = metric
        self.normalize_z = (metric == "cosine")

        # Pre-compute group centroids for avoid_group rule gating and repulsion
        # direction. L2-normalized in cosine mode; raw means in euclidean mode.
        self.group_centroids_normed = _build_group_centroids(
            centers, group_to_clusters, normalize=self.normalize_z,
        )

        # Mutable state
        self.h_history: list[torch.Tensor] = []  # each (1, 1, d_h)
        self.enabled: bool = False
        self._step: int = 0
        self._rule_state = _RuleState()
        self.diagnostics: list[dict] = []
        self.z_features: list[torch.Tensor] = []  # post-perturbation z_proc per gen step
        self.z_features_raw: list[torch.Tensor] = []  # post-perturbation raw z (encoder output)

        # Optional regularizer wiring — set by the runner before generation.
        # When ``closure_predictor`` is set, the hook computes a per-step
        # live baseline L_t = ||z_t_unperturbed - P(z_{t-1})||² using the
        # same z_{t-1} as the steered loss, then GradientSteerer applies
        # the hinge penalty on (L_steered - L_t).
        self.closure_predictor: nn.Module | None = None

    def compute_prompt_features(self) -> None:
        """Compute z features and cluster assignments for all prompt tokens.

        Call this after prompt processing (enabled=False phase) and before
        generation starts.  Populates ``prompt_diagnostics`` and appends
        prompt z_features to ``z_features``.
        """
        if not self.h_history:
            return
        h_full = torch.cat(self.h_history, dim=1)  # (1, P, d_h)
        with torch.no_grad():
            z_full, _ = self.encoder(h_full)
        # z_full: (1, P, d_z)  — raw encoder output
        z_raw_all = z_full.squeeze(0)  # (P, d_z)
        z_proc_all = preprocess_z_torch(z_raw_all, self.mu, normalize=self.normalize_z)
        if self.metric == "euclidean":
            # Per-row argmin of Euclidean distance to centers
            # dists: (P, K)
            dists = torch.cdist(z_proc_all, self.centers)
            cluster_ids = dists.argmin(dim=-1)
            top1_cos = -dists[torch.arange(len(cluster_ids)), cluster_ids]  # negative distance as score
        else:
            sims = z_proc_all @ self.centers_normed.T  # (P, K)
            cluster_ids = sims.argmax(dim=-1)
            top1_cos = sims[torch.arange(len(cluster_ids)), cluster_ids]

        self.prompt_diagnostics: list[dict] = []
        self.prompt_z_features: list[torch.Tensor] = []
        self.prompt_z_features_raw: list[torch.Tensor] = []
        for i in range(z_proc_all.shape[0]):
            cid = cluster_ids[i].item()
            gid = self.group_labels[cid].item()
            self.prompt_diagnostics.append({
                "step": -(z_proc_all.shape[0] - i),  # negative = prompt
                "cluster_id": cid,
                "group_id": gid,
                "cos_sim": round(top1_cos[i].item(), 4),
                "z_centered_norm": round(
                    (z_full[0, i] - self.mu).norm().item(), 4
                ),
                "was_steered": False,
                "post_cluster_id": cid,
                "post_group_id": gid,
                "post_cos_sim": round(top1_cos[i].item(), 4),
            })
            self.prompt_z_features.append(z_proc_all[i].detach().cpu())
            self.prompt_z_features_raw.append(z_raw_all[i].detach().cpu())

    def reset(self) -> None:
        """Clear all state for a fresh generation."""
        self.h_history.clear()
        self.enabled = False
        self._step = 0
        self._rule_state = _RuleState()
        self.diagnostics.clear()
        self.z_features.clear()
        self.z_features_raw.clear()
        self.optimizer.reset()

    # -- hook entry point ------------------------------------------------------

    def __call__(
        self,
        module: nn.Module,
        input: tuple,
        output: tuple | torch.Tensor,
    ):
        h_raw = output[0] if isinstance(output, tuple) else output
        orig_dtype = h_raw.dtype
        # Convert to float32 for encoder (LLM may output bfloat16)
        h = h_raw.float()

        if not self.enabled:
            # Prompt phase: store one entry per token position
            for i in range(h.shape[1]):
                self.h_history.append(h[:, i : i + 1, :].detach().clone())
            return None  # no modification

        # --- Generation phase (single token at a time) -----------------------
        h_t = h  # (1, 1, d_h)

        # Compute z_t via encoder over full history
        h_full = torch.cat(self.h_history + [h_t], dim=1)  # (1, T, d_h)
        with torch.no_grad():
            z_full, _ = self.encoder(h_full)
        z_raw_t = z_full[:, -1:, :]  # (1, 1, d_z)
        z_proc_t = preprocess_z_torch(z_raw_t.squeeze(1), self.mu, normalize=self.normalize_z)
        z_centered_norm = (z_raw_t.squeeze() - self.mu).norm().item()

        cluster_id, group_id, cos_sim = assign_cluster_torch(
            z_proc_t, self.centers_normed, self.group_labels,
            metric=self.metric, centers_raw=self.centers,
        )

        # Per-rule gate margins.
        # Cosine mode:    margin = cos(z, c_current) − cos(z, c_avoid)
        # Euclidean mode: margin = dist(z, c_avoid) − dist(z, c_current)
        # Both are ≥ 0 by construction (current_cluster is the argmax/argmin)
        # and "small margin = leaning toward avoid target".
        gate_margin: dict[int, float] = {}
        z_vec = z_proc_t.squeeze(0) if z_proc_t.dim() == 2 else z_proc_t
        for i, r in enumerate(self.rules):
            if r.avoid_cluster is not None:
                if self.metric == "euclidean":
                    avoid_c = self.centers[r.avoid_cluster]
                    cur_c = self.centers[cluster_id]
                    d_avoid = float((z_vec - avoid_c).norm())
                    d_cur = float((z_vec - cur_c).norm())
                    gate_margin[i] = d_avoid - d_cur
                else:
                    target = self.centers_normed[r.avoid_cluster]
                    cos_target = float((z_vec * target).sum())
                    gate_margin[i] = cos_sim - cos_target
            elif r.avoid_group is not None:
                if self.metric == "euclidean":
                    avoid_centroid = self.group_centroids_normed[r.avoid_group]
                    cur_centroid = self.group_centroids_normed[group_id]
                    d_avoid = float((z_vec - avoid_centroid).norm())
                    d_cur = float((z_vec - cur_centroid).norm())
                    gate_margin[i] = d_avoid - d_cur
                else:
                    target = self.group_centroids_normed[r.avoid_group]
                    cos_target = float((z_vec * target).sum())
                    current_group_centroid = self.group_centroids_normed[group_id]
                    cos_current = float((z_vec * current_group_centroid).sum())
                    gate_margin[i] = cos_current - cos_target
            else:
                continue

        # Rule engine: check avoid gates (sticky) then dwell/position.
        _check_avoid_rule(
            self._rule_state, self.rules, gate_margin, self._step,
        )
        _update_dwell_and_check(
            self._rule_state, group_id, self.rules, self._step,
        )

        # Apply perturbation if a rule is active
        was_steered = False
        if (
            self._rule_state.active_rule is not None
            and self._rule_state.remaining > 0
        ):
            active = self._rule_state.active_rule
            # In euclidean mode: use raw (un-normalized) centers as
            # target/current so that displacement = (target_raw − current_raw)
            # is in the same space as a centered-only z.
            current_center = (
                self.centers[cluster_id] if self.metric == "euclidean"
                else self.centers_normed[cluster_id]
            )
            # Regularizer inputs: z_{t-1} from the SAME run (whatever past we
            # actually have — possibly steered, but that's fine), plus a
            # live baseline L_t = || z_t_unperturbed - P(z_{t-1}) ||² that
            # uses the same z_{t-1} as the steered loss. Holds the past
            # fixed between baseline and steered terms.
            z_prev_raw = (
                self.z_features_raw[-1].to(self.device)
                if self.z_features_raw
                else None
            )
            base_loss = None
            if (
                self.closure_predictor is not None
                and z_prev_raw is not None
            ):
                z_prev_input = (
                    z_prev_raw.unsqueeze(0)
                    if z_prev_raw.dim() == 1 else z_prev_raw
                )
                with torch.no_grad():
                    pred_target_natural = self.closure_predictor(z_prev_input)
                    z_t_natural = z_raw_t.squeeze(1)  # (1, d_z) un-perturbed
                    base_loss = (
                        ((pred_target_natural - z_t_natural) ** 2).mean().item()
                    )
            if active.avoid_cluster is not None:
                avoid_center = (
                    self.centers[active.avoid_cluster]
                    if self.metric == "euclidean"
                    else self.centers_normed[active.avoid_cluster]
                )
                zero_current = torch.zeros_like(avoid_center)
                delta_h = self.optimizer.step(
                    h_t, self.h_history, self.encoder, self.mu,
                    avoid_center, zero_current, invert=True,
                    predictor=self.closure_predictor,
                    z_prev_raw=z_prev_raw,
                    baseline_loss=base_loss,
                )
            elif active.avoid_group is not None:
                avoid_center = self.group_centroids_normed[active.avoid_group]
                zero_current = torch.zeros_like(avoid_center)
                delta_h = self.optimizer.step(
                    h_t, self.h_history, self.encoder, self.mu,
                    avoid_center, zero_current, invert=True,
                    predictor=self.closure_predictor,
                    z_prev_raw=z_prev_raw,
                    baseline_loss=base_loss,
                )
            else:
                target = compute_target_center(
                    self.centers,
                    self.group_labels,
                    self.group_to_clusters,
                    active,
                    normalize=self.normalize_z,
                )
                delta_h = self.optimizer.step(
                    h_t, self.h_history, self.encoder, self.mu,
                    target, current_center,
                    predictor=self.closure_predictor,
                    z_prev_raw=z_prev_raw,
                    baseline_loss=base_loss,
                )
            h_modified = h_t + delta_h
            self._rule_state.remaining -= 1
            was_steered = True
            if self._rule_state.remaining == 0:
                if (
                    active.avoid_cluster is None
                    and active.avoid_group is None
                ):
                    logger.info(
                        "Steering rule exhausted after %d tokens",
                        active.duration,
                    )
                self._rule_state.active_rule = None
        else:
            h_modified = h_t

        # Compute POST-perturbation z for diagnostics and features
        post_cluster_id, post_group_id, post_cos_sim = cluster_id, group_id, cos_sim
        z_feat = z_proc_t.squeeze(0)  # default: pre-perturbation
        z_feat_raw = z_raw_t.squeeze(0).squeeze(0)  # default: pre-perturbation raw
        if was_steered:
            h_full_post = torch.cat(
                self.h_history + [h_modified], dim=1,
            )
            with torch.no_grad():
                z_post, _ = self.encoder(h_full_post)
            z_raw_post = z_post[:, -1:, :].squeeze(1)  # (1, d_z) raw
            z_proc_post = preprocess_z_torch(z_raw_post, self.mu, normalize=self.normalize_z)
            post_cluster_id, post_group_id, post_cos_sim = assign_cluster_torch(
                z_proc_post, self.centers_normed, self.group_labels,
                metric=self.metric, centers_raw=self.centers,
            )
            z_feat = z_proc_post.squeeze(0)
            z_feat_raw = z_raw_post.squeeze(0)
        self.z_features.append(z_feat.detach().cpu())
        self.z_features_raw.append(z_feat_raw.detach().cpu())

        # Store modified h and log
        self.h_history.append(h_modified.detach().clone())
        diag = {
            "step": self._step,
            "cluster_id": cluster_id,
            "group_id": group_id,
            "cos_sim": round(cos_sim, 4),
            "z_centered_norm": round(z_centered_norm, 4),
            "was_steered": was_steered,
            "post_cluster_id": post_cluster_id,
            "post_group_id": post_group_id,
            "post_cos_sim": round(post_cos_sim, 4),
        }
        if gate_margin:
            diag["gate_margin"] = {
                str(rule_idx): round(m, 4) for rule_idx, m in gate_margin.items()
            }
        self.diagnostics.append(diag)
        self._step += 1

        # Return modified output (cast back to original dtype for LLM layers above)
        h_out = h_modified.to(orig_dtype)
        if isinstance(output, tuple):
            return (h_out,) + output[1:]
        return h_out


# ---------------------------------------------------------------------------
# Multi-layer steering hook
# ---------------------------------------------------------------------------


class MultiLayerSteeringHook:
    """Multi-layer steering: passive hooks collect h from all layers, active
    hook on the control (last) layer computes z from concatenated h across
    all layers and steers by perturbing only the control layer's h_t.

    Exposes the same interface as ``SteeringHook`` so the generation loop and
    HTML rendering code don't need to branch:
      - ``h_history``: list of (1, 1, d_h) tensors — the CONTROL layer's
        history (used for len() logging and compute_prompt_features).
      - ``enabled``, ``diagnostics``, ``z_features``, ``prompt_diagnostics``,
        ``prompt_z_features``, ``compute_prompt_features()``, ``reset()``.
    """

    def __init__(
        self,
        *,
        layer_indices: list[int],
        control_layer: int,
        hidden_dim: int,
        encoder: nn.Module,
        optimizer: GradientSteerer,
        mu: torch.Tensor,
        centers: torch.Tensor,
        centers_normed: torch.Tensor,
        group_labels: torch.Tensor,
        group_to_clusters: dict[int, list[int]],
        rules: list[SteeringRule],
        device: str,
        metric: str = "cosine",
    ):
        self.layer_indices = list(layer_indices)
        self.control_layer = control_layer
        self.hidden_dim = hidden_dim
        self.encoder = encoder
        self.optimizer = optimizer
        self.mu = mu
        self.centers = centers
        self.centers_normed = centers_normed
        self.group_labels = group_labels
        self.group_to_clusters = group_to_clusters
        self.rules = rules
        self.device = device
        self.metric = metric
        self.normalize_z = (metric == "cosine")

        # Pre-compute group centroids for avoid_group rules.
        self.group_centroids_normed = _build_group_centroids(
            centers, group_to_clusters, normalize=self.normalize_z,
        )

        # Per-layer history (all positions accumulated across prompt+generation).
        # Keys are actual layer indices from the LLM.  Each entry is a list of
        # (1, 1, d_h) tensors in temporal order.
        self._h_history_per_layer: dict[int, list[torch.Tensor]] = {
            li: [] for li in self.layer_indices
        }
        # Scratch: h of the current generation step for passive layers.
        # Cleared after the active hook processes the step.
        self._cur_step_h: dict[int, torch.Tensor] = {}

        # Mutable state (matches SteeringHook interface)
        self.enabled: bool = False
        self._step: int = 0
        self._rule_state = _RuleState()
        self.diagnostics: list[dict] = []
        self.z_features: list[torch.Tensor] = []

    @property
    def h_history(self) -> list[torch.Tensor]:
        """Control layer's history — used for len() logging + prompt feature
        computation. Matches the single-layer SteeringHook interface."""
        return self._h_history_per_layer[self.control_layer]

    # ------------------------------------------------------------------
    # Passive hook (for layers other than the control layer)
    # ------------------------------------------------------------------

    def make_passive_hook(self, layer_idx: int):
        """Return a forward-hook closure for a non-control layer.

        During prompt phase: stores each position's h into per-layer history.
        During generation phase: stores h for the current step into scratch.
        Always returns None (no modification).
        """
        def hook_fn(module, input, output):
            h_raw = output[0] if isinstance(output, tuple) else output
            h = h_raw.float()
            if not self.enabled:
                # Prompt phase: split into per-token entries
                for i in range(h.shape[1]):
                    self._h_history_per_layer[layer_idx].append(
                        h[:, i : i + 1, :].detach().clone()
                    )
            else:
                # Generation phase: store current step for later aggregation
                self._cur_step_h[layer_idx] = h.detach().clone()
            return None  # do not modify
        return hook_fn

    # ------------------------------------------------------------------
    # Concatenation helpers
    # ------------------------------------------------------------------

    def _build_history_concat(self) -> torch.Tensor:
        """Concatenate per-layer histories along the feature dim.

        Returns:
            (1, T, L*d_h) tensor — T is the number of already-collected positions.
            If no history (T=0), returns a (1, 0, L*d_h) tensor.
        """
        per_layer = []
        for li in self.layer_indices:
            hist = self._h_history_per_layer[li]
            if hist:
                per_layer.append(torch.cat(hist, dim=1))  # (1, T, d_h)
            else:
                per_layer.append(
                    torch.zeros(1, 0, self.hidden_dim, device=self.device)
                )
        return torch.cat(per_layer, dim=-1)  # (1, T, L*d_h)

    def _build_current_concat(self, h_t_control: torch.Tensor) -> torch.Tensor:
        """Concatenate the current step's h across all layers.

        The control layer's h_t comes from the active hook's argument; all
        other layers' h_t come from ``_cur_step_h`` (populated by passive hooks
        that fired earlier in this forward pass).

        Returns:
            (1, 1, L*d_h)
        """
        per_layer = []
        for li in self.layer_indices:
            if li == self.control_layer:
                per_layer.append(h_t_control)
            else:
                if li not in self._cur_step_h:
                    raise RuntimeError(
                        f"Passive hook for layer {li} did not fire before "
                        f"control layer {self.control_layer}. Check hook "
                        f"registration order or that layer_indices are in "
                        f"ascending order."
                    )
                per_layer.append(self._cur_step_h[li])
        return torch.cat(per_layer, dim=-1)  # (1, 1, L*d_h)

    # ------------------------------------------------------------------
    # Prompt feature computation (analog of SteeringHook method)
    # ------------------------------------------------------------------

    def compute_prompt_features(self) -> None:
        """Run the encoder on the concatenated prompt h and store per-position
        z features + cluster assignments in ``prompt_diagnostics`` / ``prompt_z_features``."""
        if not self._h_history_per_layer[self.control_layer]:
            return
        h_full_concat = self._build_history_concat()  # (1, P, L*d_h)
        with torch.no_grad():
            z_full, _ = self.encoder(h_full_concat)
        z_proc_all = preprocess_z_torch(z_full.squeeze(0), self.mu, normalize=self.normalize_z)
        if self.metric == "euclidean":
            dists = torch.cdist(z_proc_all, self.centers)
            cluster_ids = dists.argmin(dim=-1)
            top1_cos = -dists[torch.arange(len(cluster_ids)), cluster_ids]
        else:
            sims = z_proc_all @ self.centers_normed.T  # (P, K)
            cluster_ids = sims.argmax(dim=-1)
            top1_cos = sims[torch.arange(len(cluster_ids)), cluster_ids]

        self.prompt_diagnostics: list[dict] = []
        self.prompt_z_features: list[torch.Tensor] = []
        P = z_proc_all.shape[0]
        for i in range(P):
            cid = cluster_ids[i].item()
            gid = self.group_labels[cid].item()
            self.prompt_diagnostics.append({
                "step": -(P - i),
                "cluster_id": cid,
                "group_id": gid,
                "cos_sim": round(top1_cos[i].item(), 4),
                "z_centered_norm": round(
                    (z_full[0, i] - self.mu).norm().item(), 4
                ),
                "was_steered": False,
                "post_cluster_id": cid,
                "post_group_id": gid,
                "post_cos_sim": round(top1_cos[i].item(), 4),
            })
            self.prompt_z_features.append(z_proc_all[i].detach().cpu())

    def reset(self) -> None:
        for li in self.layer_indices:
            self._h_history_per_layer[li].clear()
        self._cur_step_h = {}
        self.enabled = False
        self._step = 0
        self._rule_state = _RuleState()
        self.diagnostics.clear()
        self.z_features.clear()
        self.optimizer.reset()

    # ------------------------------------------------------------------
    # Active hook (for the control layer)
    # ------------------------------------------------------------------

    def make_active_hook(self):
        def hook_fn(module, input, output):
            h_raw = output[0] if isinstance(output, tuple) else output
            orig_dtype = h_raw.dtype
            h = h_raw.float()

            if not self.enabled:
                # Prompt phase: append each position to CONTROL layer history.
                # Passive hooks on other layers already populated their own lists.
                for i in range(h.shape[1]):
                    self._h_history_per_layer[self.control_layer].append(
                        h[:, i : i + 1, :].detach().clone()
                    )
                return None

            # --- Generation phase (single token at a time) ---
            h_t_control = h  # (1, 1, d_h)

            # Build current-step concat (uses h_t_control for control layer
            # and _cur_step_h for others)
            h_t_concat = self._build_current_concat(h_t_control)  # (1, 1, L*d_h)
            h_hist_concat = self._build_history_concat()  # (1, T-1, L*d_h)

            # Encode the full sequence
            h_full_concat = torch.cat([h_hist_concat, h_t_concat], dim=1)
            with torch.no_grad():
                z_full, _ = self.encoder(h_full_concat)
            z_raw_t = z_full[:, -1:, :]  # (1, 1, d_z)
            z_proc_t = preprocess_z_torch(z_raw_t.squeeze(1), self.mu, normalize=self.normalize_z)
            z_centered_norm = (z_raw_t.squeeze() - self.mu).norm().item()

            cluster_id, group_id, cos_sim = assign_cluster_torch(
                z_proc_t, self.centers_normed, self.group_labels,
                metric=self.metric, centers_raw=self.centers,
            )

            # Per-rule gate margins (see SteeringHook for both formulas).
            gate_margin: dict[int, float] = {}
            z_vec = z_proc_t.squeeze(0) if z_proc_t.dim() == 2 else z_proc_t
            for i, r in enumerate(self.rules):
                if r.avoid_cluster is not None:
                    if self.metric == "euclidean":
                        avoid_c = self.centers[r.avoid_cluster]
                        cur_c = self.centers[cluster_id]
                        d_avoid = float((z_vec - avoid_c).norm())
                        d_cur = float((z_vec - cur_c).norm())
                        gate_margin[i] = d_avoid - d_cur
                    else:
                        target = self.centers_normed[r.avoid_cluster]
                        cos_target = float((z_vec * target).sum())
                        gate_margin[i] = cos_sim - cos_target
                elif r.avoid_group is not None:
                    if self.metric == "euclidean":
                        avoid_centroid = self.group_centroids_normed[r.avoid_group]
                        cur_centroid = self.group_centroids_normed[group_id]
                        d_avoid = float((z_vec - avoid_centroid).norm())
                        d_cur = float((z_vec - cur_centroid).norm())
                        gate_margin[i] = d_avoid - d_cur
                    else:
                        target = self.group_centroids_normed[r.avoid_group]
                        cos_target = float((z_vec * target).sum())
                        current_group_centroid = self.group_centroids_normed[group_id]
                        cos_current = float((z_vec * current_group_centroid).sum())
                        gate_margin[i] = cos_current - cos_target
                else:
                    continue

            _check_avoid_rule(
                self._rule_state, self.rules, gate_margin, self._step,
            )
            _update_dwell_and_check(
                self._rule_state, group_id, self.rules, self._step,
            )

            was_steered = False
            h_modified = h_t_control
            post_cluster_id, post_group_id, post_cos_sim = (
                cluster_id, group_id, cos_sim,
            )
            z_feat = z_proc_t.squeeze(0)

            if (
                self._rule_state.active_rule is not None
                and self._rule_state.remaining > 0
            ):
                active = self._rule_state.active_rule
                current_center = (
                    self.centers[cluster_id] if self.metric == "euclidean"
                    else self.centers_normed[cluster_id]
                )
                control_slice = slice(
                    h_t_concat.shape[-1] - self.hidden_dim,
                    h_t_concat.shape[-1],
                )
                if active.avoid_cluster is not None:
                    avoid_center = (
                        self.centers[active.avoid_cluster]
                        if self.metric == "euclidean"
                        else self.centers_normed[active.avoid_cluster]
                    )
                    zero_current = torch.zeros_like(avoid_center)
                    delta_h_control = self.optimizer.step_multi_layer(
                        h_t_concat=h_t_concat,
                        h_history_concat=h_hist_concat,
                        encoder=self.encoder,
                        mu=self.mu,
                        target_center=avoid_center,
                        current_center=zero_current,
                        control_slice=control_slice,
                        invert=True,
                    )
                elif active.avoid_group is not None:
                    avoid_center = self.group_centroids_normed[active.avoid_group]
                    zero_current = torch.zeros_like(avoid_center)
                    delta_h_control = self.optimizer.step_multi_layer(
                        h_t_concat=h_t_concat,
                        h_history_concat=h_hist_concat,
                        encoder=self.encoder,
                        mu=self.mu,
                        target_center=avoid_center,
                        current_center=zero_current,
                        control_slice=control_slice,
                        invert=True,
                    )
                else:
                    target = compute_target_center(
                        self.centers,
                        self.group_labels,
                        self.group_to_clusters,
                        active,
                        normalize=self.normalize_z,
                    )
                    delta_h_control = self.optimizer.step_multi_layer(
                        h_t_concat=h_t_concat,
                        h_history_concat=h_hist_concat,
                        encoder=self.encoder,
                        mu=self.mu,
                        target_center=target,
                        current_center=current_center,
                        control_slice=control_slice,
                    )
                h_modified = h_t_control + delta_h_control

                # Recompute z from modified h for diagnostics
                h_t_concat_post = torch.cat(
                    [
                        h_t_concat[..., :-self.hidden_dim],
                        h_modified,
                    ],
                    dim=-1,
                )
                h_full_post = torch.cat(
                    [h_hist_concat, h_t_concat_post], dim=1,
                )
                with torch.no_grad():
                    z_post, _ = self.encoder(h_full_post)
                z_proc_post = preprocess_z_torch(
                    z_post[:, -1:, :].squeeze(1), self.mu,
                    normalize=self.normalize_z,
                )
                post_cluster_id, post_group_id, post_cos_sim = assign_cluster_torch(
                    z_proc_post, self.centers_normed, self.group_labels,
                    metric=self.metric, centers_raw=self.centers,
                )
                z_feat = z_proc_post.squeeze(0)

                self._rule_state.remaining -= 1
                was_steered = True
                if self._rule_state.remaining == 0:
                    if (
                        active.avoid_cluster is None
                        and active.avoid_group is None
                    ):
                        logger.info(
                            "Steering rule exhausted after %d tokens",
                            active.duration,
                        )
                    self._rule_state.active_rule = None

            self.z_features.append(z_feat.detach().cpu())

            # Append (modified) control h_t and passive layers' h_t to history
            self._h_history_per_layer[self.control_layer].append(
                h_modified.detach().clone()
            )
            for li in self.layer_indices:
                if li != self.control_layer:
                    self._h_history_per_layer[li].append(self._cur_step_h[li])
            self._cur_step_h = {}

            diag = {
                "step": self._step,
                "cluster_id": cluster_id,
                "group_id": group_id,
                "cos_sim": round(cos_sim, 4),
                "z_centered_norm": round(z_centered_norm, 4),
                "was_steered": was_steered,
                "post_cluster_id": post_cluster_id,
                "post_group_id": post_group_id,
                "post_cos_sim": round(post_cos_sim, 4),
            }
            if gate_margin:
                diag["gate_margin"] = {
                    str(rule_idx): round(m, 4) for rule_idx, m in gate_margin.items()
                }
            self.diagnostics.append(diag)
            self._step += 1

            h_out = h_modified.to(orig_dtype)
            if isinstance(output, tuple):
                return (h_out,) + output[1:]
            return h_out

        return hook_fn
