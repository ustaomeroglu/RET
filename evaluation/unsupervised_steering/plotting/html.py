"""Interactive-HTML rendering for unsupervised-steering runs.

Builds on `evaluation.clustering.plotting.html.render_interactive_trajectory_html`
and adds a steering-aware info box plus red underline on steered tokens.

Public entry: ``save_steered_html(config, ctx, hook, diagnostics, out_dir)``.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch

from evaluation.unsupervised_steering.config import SteeringConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rule rendering
# ---------------------------------------------------------------------------


def describe_rule(rule, *, html: bool) -> str:
    """Format a SteeringRule as a human-readable string.

    Two rendering modes:
      * ``html=False``: plain text (for title bar / logs), uses arrow "→".
      * ``html=True``: HTML (for info box), wraps target in <b>, uses &rarr;.

    Three rule flavors:
      * attractor position trigger → "From step N → G/C ..."
      * attractor group-dwell (legacy) → "After G<x> dwell <y> → ..."
      * proximity-gated repulsion → "Repel from C<c> when cos ≥ T"
    """
    import html as html_mod

    # Repulsion branch (cluster- or group-level)
    if rule.avoid_cluster is not None or rule.avoid_group is not None:
        if rule.avoid_cluster is not None:
            t_raw = f"C{rule.avoid_cluster}"
        else:
            t_raw = f"G{rule.avoid_group}"
        t = f"<b>{html_mod.escape(t_raw)}</b>" if html else t_raw
        gate = (
            f" after step {rule.avoid_from_step}"
            if rule.avoid_from_step is not None else ""
        )
        budget = (
            f" (max {rule.avoid_max_tokens} fires)"
            if rule.avoid_max_tokens is not None else ""
        )
        sticky = (
            f" sticky={rule.avoid_sticky_duration}"
            if rule.avoid_sticky_duration > 1 else ""
        )
        if html:
            return (
                f"Repel from {t} when margin &le; {rule.avoid_margin_threshold}"
                f"{sticky}{gate}{budget}"
            )
        return (
            f"avoid {t_raw} @ margin<={rule.avoid_margin_threshold}{sticky}{gate}{budget}"
        )

    # Target label
    if rule.target_group is not None:
        tgt_raw = f"G{rule.target_group}"
    else:
        tgt_raw = f"C{rule.target_cluster}"
    tgt = f"<b>{html_mod.escape(tgt_raw)}</b>" if html else tgt_raw

    # Trigger prefix
    if rule.trigger_position is not None:
        if rule.trigger_position == 0:
            trig = (
                "From the very first generated token"
                if html else "from gen step 0"
            )
        else:
            trig = (
                f"From generation step {rule.trigger_position}"
                if html else f"from gen step {rule.trigger_position}"
            )
    else:
        # Legacy group-dwell trigger
        trig = (
            f"After G{rule.trigger_group} dwell {rule.trigger_dwell}"
            if html
            else f"after G{rule.trigger_group} dwell {rule.trigger_dwell}"
        )

    if html:
        return (
            f"{trig} &rarr; steer toward {tgt} for {rule.duration} tokens"
        )
    return f"{trig} → {tgt} for {rule.duration}t"


def load_cluster_group_names(
    config: SteeringConfig,
) -> tuple[dict | None, dict | None]:
    """Try to load cluster/group naming files from the analysis subtree."""
    base = (
        Path(config.clustering_dir)
        / f"w{config.window_size}"
        / f"K{config.K}"
        / f"seed{config.seed}"
        / "analysis"
        / f"weighted_tau{config.tau}"
        / f"grouped_alpha{config.alpha}"
    )
    naming_path = base / "cluster_naming" / "naming.txt"
    if not naming_path.exists():
        return None, None

    cluster_names: dict[int, tuple[str, str]] = {}
    group_names: dict[int, tuple[str, str]] = {}
    for line in naming_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("G"):
            parts = line.split(":", 1)
            gid = int(parts[0][1:])
            rest = parts[1].strip()
            if "|" in rest:
                short, long = rest.split("|", 1)
                group_names[gid] = (short.strip(), long.strip())
            else:
                group_names[gid] = (rest, rest)
        elif line.startswith("C"):
            parts = line.split(":", 1)
            cid_str = parts[0].split("[")[0].strip()
            cid = int(cid_str[1:])
            rest = parts[1].strip()
            if "|" in rest:
                short, long = rest.split("|", 1)
                cluster_names[cid] = (short.strip(), long.strip())
            else:
                cluster_names[cid] = (rest, rest)
    return cluster_names or None, group_names or None


# ---------------------------------------------------------------------------
# Top-level rendering entry
# ---------------------------------------------------------------------------


def save_steered_html(
    config: SteeringConfig,
    ctx,
    hook,  # SteeringHook | MultiLayerSteeringHook
    diagnostics: list[dict],
    out: Path,
) -> None:
    """Render interactive HTML for the steered trajectory."""
    from evaluation.clustering.plotting.html import render_interactive_trajectory_html

    if not diagnostics or not hook.z_features:
        logger.warning("No diagnostics/z_features to render HTML")
        return

    # Combine prompt + generation diagnostics and features
    all_diag = getattr(hook, "prompt_diagnostics", []) + diagnostics
    prompt_z = getattr(hook, "prompt_z_features", [])
    all_z = prompt_z + hook.z_features

    # Tokens — from combined diagnostics. Token IDs are passed through to
    # the renderer so it can repair multi-byte UTF-8 characters split
    # across BPE token boundaries (e.g. ≈ → two cells of �).
    tokens = [d.get("token", "?") for d in all_diag]
    token_ids_per_cell = [d.get("token_id") for d in all_diag]
    if not all(tid is not None for tid in token_ids_per_cell):
        token_ids_per_cell = None

    # Post-perturbation labels and features
    post_labels = np.array([d["post_cluster_id"] for d in all_diag])
    post_cos = np.array([d["post_cos_sim"] for d in all_diag], dtype=np.float32)

    # Compute margin from z_features vs centers
    z_feats = torch.stack(all_z).numpy()
    centers_np = ctx.centers.cpu().numpy()
    cn_np = ctx.centers_normed.cpu().numpy()
    sims = z_feats @ cn_np.T
    top1 = sims[np.arange(len(sims)), post_labels]
    sims_copy = sims.copy()
    sims_copy[np.arange(len(sims)), post_labels] = -np.inf
    top2 = sims_copy.max(axis=1)
    margin = top1 - top2

    # Group labels (post-perturbation)
    gl_np = ctx.group_labels.cpu().numpy()
    post_group_labels = gl_np[post_labels]

    # Build title — only mention steering if tokens were actually steered
    any_steered = any(d.get("was_steered") for d in all_diag)
    if any_steered:
        rules_desc = [describe_rule(r, html=False) for r in config.rules]
        steer_note = (
            f"Steered (strength={config.strength}, steps={config.gradient_steps}) "
            f"| {'; '.join(rules_desc)}"
        )
    else:
        steer_note = "Baseline (no steering)"

    # Load names
    cluster_names, group_names = load_cluster_group_names(config)

    K_coarse = len(ctx.group_to_clusters)

    # Load state machine artifacts if available
    sm_kwargs: dict = {}
    analysis_dir = (
        Path(config.clustering_dir)
        / f"w{config.window_size}"
        / f"K{config.K}"
        / f"seed{config.seed}"
        / "analysis"
    )
    tau_dir = analysis_dir / f"weighted_tau{config.tau}"
    grp_dir = tau_dir / f"grouped_alpha{config.alpha}"

    # Micro-level matrices (in tau_dir)
    pruned = tau_dir / "transition_matrix_pruned.npy"
    if pruned.exists():
        sm_kwargs["P_exit_micro"] = np.load(pruned)
    micro_exit = tau_dir / "transition_matrix_micro_exit.npy"
    if micro_exit.exists():
        P_weighted = np.load(tau_dir / "transition_matrix_weighted.npy")
        sm_kwargs["P_self_loop"] = np.diag(P_weighted)

    # Macro-level matrix (in grp_dir)
    macro_npy = grp_dir / "transition_matrix_macro.npy"
    if macro_npy.exists():
        P_macro = np.load(macro_npy)
        # Compute P_exit from P_macro (remove self-loops, renormalize)
        P_exit = np.zeros_like(P_macro)
        for g in range(P_macro.shape[0]):
            off_diag = 1.0 - P_macro[g, g]
            if off_diag > 1e-8:
                for j in range(P_macro.shape[1]):
                    if j != g:
                        P_exit[g, j] = P_macro[g, j] / off_diag
        sm_kwargs["P_exit"] = P_exit

    # Cluster usage
    usage_path = analysis_dir / "cluster_usage.npy"
    if usage_path.exists():
        sm_kwargs["usage"] = np.load(usage_path)

    render_interactive_trajectory_html(
        tokens=tokens,
        labels=post_labels,
        features=z_feats,
        K=config.K,
        centers=centers_np,
        out_path=out / "interactive_trajectory.html",
        title=steer_note,
        stride=1,
        window_size=1,
        top1_cos=post_cos,
        margin=margin,
        group_labels=gl_np,
        K_coarse=K_coarse,
        cluster_names=cluster_names,
        group_names=group_names,
        token_ids=token_ids_per_cell,
        tokenizer=ctx.tokenizer,
        **sm_kwargs,
    )

    # Inject a steering info box and highlight steered tokens
    inject_steering_info(out / "interactive_trajectory.html", config, all_diag)
    logger.info("Saved %s", out / "interactive_trajectory.html")


def inject_steering_info(
    html_path: Path,
    config: SteeringConfig,
    all_diag: list[dict],
) -> None:
    """Insert steering info box and highlight steered tokens in the HTML."""
    steered = [d for d in all_diag if d.get("was_steered")]
    if not steered:
        return

    # Find token indices that were steered (in the combined prompt+gen list)
    steered_tok_ids = [
        i for i, d in enumerate(all_diag) if d.get("was_steered")
    ]
    first_idx = steered_tok_ids[0]
    last_idx = steered_tok_ids[-1]

    # Build rule descriptions (handles both position and group-dwell triggers)
    rule_lines = [describe_rule(r, html=True) for r in config.rules]

    # Deduplicate
    if rule_lines:
        seen: list[str] = []
        for line in rule_lines:
            if line not in seen:
                seen.append(line)
        rule_lines = seen

    # Group distribution during steering
    from collections import Counter
    post_groups = Counter(d["post_group_id"] for d in steered)
    group_dist = ", ".join(
        f"G{g}: {n}t" for g, n in sorted(post_groups.items())
    )

    # Determine trigger variant label from the (deduplicated) rule set.
    def _is_avoid(r):
        return r.avoid_cluster is not None or r.avoid_group is not None

    has_position = any(
        not _is_avoid(r) and r.trigger_position is not None
        for r in config.rules
    )
    has_dwell = any(
        not _is_avoid(r)
        and r.trigger_position is None
        and r.trigger_group is not None
        for r in config.rules
    )
    has_avoid = any(_is_avoid(r) for r in config.rules)
    if has_avoid and not has_position and not has_dwell:
        variant_label = (
            "margin-gated repulsion (fires when cos(z,current) − cos(z,avoid) ≤ threshold)"
        )
        header_label = "Steering Info &mdash; avoid/repel variant"
    elif has_position and not has_dwell and not has_avoid:
        variant_label = "position-triggered (fires at a fixed generation step)"
        header_label = "Steering Info &mdash; position-triggered variant"
    elif has_dwell and not has_position and not has_avoid:
        variant_label = "group-dwell triggered (fires after N tokens in a group)"
        header_label = "Steering Info &mdash; group-dwell variant"
    else:
        variant_label = "mixed triggers"
        header_label = "Steering Info"

    info_html = (
        '<div style="margin:8px 0 12px; padding:10px 14px; '
        'background:#fff3cd; border:1px solid #ffc107; border-radius:6px; '
        'font-size:12px; line-height:1.6; max-width:800px;">'
        f'<b style="font-size:13px;">{header_label}</b><br>'
        f'<b>Trigger type:</b> {variant_label}<br>'
        f'<b>Steered tokens:</b> tok-{first_idx}&ndash;tok-{last_idx} '
        f'({len(steered)} tokens)<br>'
        f'<b>Method:</b> displacement gradient, '
        f'strength={config.strength}, steps={config.gradient_steps}<br>'
        f'<b>Rule{"s" if len(rule_lines) > 1 else ""}:</b> '
        f'{"<br>".join(rule_lines)}<br>'
        f'<b>Post-steer groups:</b> {group_dist}<br>'
        '<span style="display:inline-block;width:20px;height:4px;'
        'background:#e00;margin-right:4px;margin-bottom:2px;'
        'vertical-align:middle;"></span>'
        '<span style="font-size:11px;">= steered token (red underline)</span>'
        '</div>'
    )

    steered_css = (
        '.token.steered-token {\n'
        '  box-shadow: 0 4px 0 0 #e00 !important;\n'
        '  padding-bottom: 4px !important;\n'
        '  position: relative;\n'
        '}\n'
    )

    steered_ids_js = ",".join(str(i) for i in steered_tok_ids)
    steered_js = (
        '<script>\n'
        'document.addEventListener("DOMContentLoaded", function() {\n'
        f'  var ids = [{steered_ids_js}];\n'
        '  ids.forEach(function(i) {\n'
        '    var el = document.getElementById("tok-" + i);\n'
        '    if (el) el.classList.add("steered-token");\n'
        '  });\n'
        '});\n'
        '</script>\n'
    )

    content = html_path.read_text(encoding="utf-8")
    if "</style>" in content:
        content = content.replace("</style>", steered_css + "</style>", 1)
    else:
        content = content.replace("</head>", steered_css + "</head>", 1)
    content = content.replace("</h2>", "</h2>" + info_html, 1)
    content = content.replace("</body>", steered_js + "</body>", 1)
    html_path.write_text(content, encoding="utf-8")
