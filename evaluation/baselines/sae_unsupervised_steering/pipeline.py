"""End-to-end SAE-based steering pipeline.

Parallel to ``evaluation/unsupervised_steering/pipeline.py`` but uses
direct SAE feature directions (decoder columns by default) instead of
gradient-on-cluster-displacement.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import yaml

from evaluation.baselines.sae_unsupervised_steering.config import SAESteeringConfig, SAESteeringRule
from evaluation.baselines.sae_unsupervised_steering.hooks import SAESteeringHook
from evaluation.baselines.sae_unsupervised_steering.loader import load_sae_steering_context
from evaluation.baselines.sae_unsupervised_steering.steerer import SAESteerer
from evaluation.unsupervised_steering.generation import SteeredGenerator

logger = logging.getLogger(__name__)


def run_sae_steering(config: SAESteeringConfig) -> dict:
    """Load context, register hook, generate text, save artifacts.

    Returns ``{"text": ..., "diagnostics": [...]}``.
    """
    ctx = load_sae_steering_context(config)
    steerer = SAESteerer(
        strength=config.strength, d_h=ctx.hidden_dim, device=ctx.device,
    )
    hook = SAESteeringHook(
        sae=ctx.sae,
        decoder_weight=ctx.decoder_weight,
        encoder_weight=ctx.encoder_weight,
        b_dec=ctx.b_dec,
        encoder_bias=ctx.encoder_bias,
        rules=config.rules,
        steerer=steerer,
        direction_source=config.direction_source,
        device=ctx.device,
        top_k_diag=config.top_k_diag,
    )
    handle = ctx.layers[ctx.layer_idx].register_forward_hook(hook)
    try:
        gen = SteeredGenerator(ctx.model, ctx.tokenizer, hook)
        text, diagnostics = gen.generate(
            config.prompt, config.max_new_tokens,
            config.temperature, config.top_p,
        )
    finally:
        handle.remove()

    if config.output_dir is not None:
        out = Path(config.output_dir)
        out.mkdir(parents=True, exist_ok=True)

        (out / "steered_output.txt").write_text(text, encoding="utf-8")

        with open(out / "diagnostics.jsonl", "w") as f:
            for d in diagnostics:
                f.write(json.dumps(d) + "\n")

        prompt_z = list(hook.prompt_z_features)
        gen_z = list(hook.z_features)
        if prompt_z or gen_z:
            stacked = torch.stack(prompt_z + gen_z)  # (P+G, top_k_diag, 2)
            np.save(out / "z_sae_trajectory.npy", stacked.numpy())
            (out / "prompt_len.txt").write_text(str(len(prompt_z)))

        cfg_dict = asdict(config)
        with open(out / "config.yaml", "w") as f:
            yaml.dump(cfg_dict, f, default_flow_style=False)

        _save_sae_steered_html(config, ctx, hook, diagnostics, out)
        logger.info("Results saved to %s", out)

    return {"text": text, "diagnostics": diagnostics}


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------


def _describe_rule(rule: SAESteeringRule, names: dict[int, list[str]] | None) -> str:
    """One-line description of a rule, with named latents inlined when names
    are available."""
    def label(lids: list[int]) -> str:
        if names is None:
            return ",".join(f"L{i}" for i in lids)
        parts = []
        for i in lids:
            toks = names.get(i)
            if toks:
                parts.append(f"L{i}({toks[0]!r})")
            else:
                parts.append(f"L{i}")
        return ",".join(parts)

    if rule.target_latents is not None:
        return (
            f"from step {rule.trigger_position} steer toward {label(rule.target_latents)} "
            f"for {rule.duration} tokens"
        )
    sticky = (
        f" sticky={rule.avoid_sticky_duration}"
        if rule.avoid_sticky_duration > 1 else ""
    )
    gate = f" from step {rule.avoid_from_step}" if rule.avoid_from_step else ""
    budget = f" max {rule.avoid_max_tokens} fires" if rule.avoid_max_tokens else ""
    return (
        f"avoid {label(rule.avoid_latents)} when pre-act >= "
        f"{rule.avoid_pre_threshold}{sticky}{gate}{budget}"
    )


def _palette(n: int) -> list[str]:
    """``n`` visually distinct colors via tab20-like cycling. Returns hex strings."""
    base = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
        "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5",
        "#393b79", "#637939", "#8c6d31", "#843c39", "#7b4173",
    ]
    return [base[i % len(base)] for i in range(n)]


def _text_color_for_bg_hex(hex_color: str) -> str:
    """Pick black/white text per background luminance (matches RET helper)."""
    s = hex_color.lstrip("#")
    r, g, b = int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    return "#000" if lum > 140 else "#fff"


def _build_top_n_palette(
    all_z: list, top_n: int = 20,
) -> tuple[dict[int, str], list[tuple[int, int]]]:
    """Pick the top-N most-frequent top-1 latent ids across the whole run and
    assign each a stable color. Returns ``(lid -> hex, [(lid, count), ...])``.
    Latents not in the top-N render with a default gray.
    """
    from collections import Counter
    top1_counts: Counter[int] = Counter()
    for z in all_z:
        if len(z) == 0:
            continue
        top1_counts[int(z[0, 0].item())] += 1
    most = top1_counts.most_common(top_n)
    palette = _palette(len(most))
    return ({lid: palette[i] for i, (lid, _) in enumerate(most)}, most)


_HTML_HEADER = """<!DOCTYPE html>
<html><head><meta charset='utf-8'><title>{title}</title>
<style>
body {{ margin:0; padding:16px; font-family:sans-serif; background:#fafafa; }}
h2 {{ margin:0 0 8px; font-size:16px; }}
.info-box {{ margin: 0 0 10px; padding: 10px 14px; background: #fff3cd;
             border: 1px solid #ffc107; border-radius: 6px; font-size: 12px;
             line-height: 1.6; max-width: 980px; }}
.container {{ display:flex; gap:12px; height:calc(100vh - 180px); }}
.text-panel {{ flex:0 0 60%; overflow-y:auto; padding:10px; background:#fff;
               border:1px solid #ddd; border-radius:4px; }}
.text-content {{ font-family:'Courier New',monospace; font-size:13px;
                 line-height:1.8; font-size:0; }}
.right-panel {{ flex:1; display:flex; flex-direction:column;
                overflow-y:auto; gap:10px; }}
.legend-panel {{ background:#fff; border:1px solid #ddd; border-radius:4px;
                 padding:8px 10px; }}
.legend-title {{ font-size:12px; font-weight:bold; margin-bottom:6px; color:#333; }}
.legend-row {{ display:flex; align-items:center; gap:6px; padding:2px 0;
              font-size:11px; font-family:'Courier New',monospace; }}
.legend-swatch {{ display:inline-block; width:14px; height:14px;
                 border-radius:2px; border:1px solid #888; flex:0 0 14px; }}
.legend-lid {{ color:#06c; flex:0 0 60px; }}
.legend-count {{ color:#888; flex:0 0 38px; text-align:right; }}
.legend-toks {{ color:#333; overflow:hidden; text-overflow:ellipsis;
               white-space:nowrap; }}
.token {{ font-size:13px; padding:1px 0; border-radius:2px;
        cursor:pointer; white-space:pre-wrap; }}
.token.hl {{ outline:3px solid #333; outline-offset:1px;
            z-index:1; position:relative; }}
.token.steered {{ box-shadow:0 4px 0 0 #e00; padding-bottom:4px; }}
.token.unlabeled {{ background:#e8e8e8; color:#666; }}
.tooltip {{ position:fixed; background:rgba(0,0,0,0.92); color:#fff;
           padding:6px 10px; border-radius:4px; font-size:11px;
           pointer-events:none; display:none; z-index:1000; max-width:380px;
           font-family:'Courier New',monospace; white-space:pre-wrap;
           line-height:1.5; }}
</style></head><body>
<h2>{title}</h2>
"""


def _save_sae_steered_html(
    config: SAESteeringConfig,
    ctx,
    hook: SAESteeringHook,
    diagnostics: list[dict],
    out: Path,
) -> None:
    """Two-panel interactive HTML: token-strip on the left (colored by top-1
    SAE latent), legend + run info on the right. Hover a token to see its
    top-5 active latents with names. Steered tokens carry a red underline.

    Layout intentionally mirrors the RET clustering HTML's text-panel /
    right-panel split, but without UMAP/state-machine since SAE features
    aren't a small discrete clustering.
    """
    import html as html_mod
    import json as json_mod

    all_diag = list(hook.prompt_diagnostics) + list(diagnostics)
    all_z = list(hook.prompt_z_features) + list(hook.z_features)
    if not all_diag:
        logger.warning("No diagnostics to render for HTML")
        return

    names = ctx.sae_latent_names or {}

    # --- Top-N latent palette (most-frequent top-1 across the run) ----------
    color_map, top_counts = _build_top_n_palette(all_z, top_n=20)

    # --- Header & info box --------------------------------------------------
    any_steered = any(d.get("was_steered") for d in all_diag)
    if any_steered:
        rules_html = "<br>".join(
            html_mod.escape(_describe_rule(r, names)) for r in config.rules
        )
        title_note = (
            f"SAE steered (strength={config.strength}, "
            f"direction_source={config.direction_source})"
        )
    else:
        rules_html = "(no active rules — baseline trajectory)"
        title_note = "SAE baseline (no steering)"

    n_steered = sum(1 for d in all_diag if d.get("was_steered"))
    info_box = (
        '<div class="info-box">'
        f'<b>Run:</b> layer {ctx.layer_idx} ({ctx.sae_release}/{ctx.sae_id})'
        f' &middot; d_sae={ctx.sae.d_sae}, k={ctx.sae.k}<br>'
        f'<b>Method:</b> δh = sign · {config.strength} · ‖h‖ · '
        f'{config.direction_source} direction<br>'
        f'<b>Rule(s):</b> {rules_html}<br>'
        f'<b>Steered tokens:</b> {n_steered}/{len(all_diag)} '
        f'(<span style="display:inline-block;width:18px;height:4px;'
        f'background:#e00;margin:0 3px;vertical-align:middle;"></span> '
        f'red underline)'
        '</div>'
    )

    h: list[str] = [_HTML_HEADER.format(title=html_mod.escape(title_note)),
                    info_box,
                    '<div class="container">',
                    '<div class="text-panel"><div class="text-content">']

    # --- Tokens (left panel) ------------------------------------------------
    for i, (d, z) in enumerate(zip(all_diag, all_z)):
        tok = d.get("token", "")
        if not tok:
            tok = ""
        tok_esc = html_mod.escape(tok)
        has_nl = "\n" in tok
        tok_disp = tok_esc.replace("\n", "↵")

        if len(z) > 0:
            top1_lid = int(z[0, 0].item())
        else:
            top1_lid = -1

        klass_parts = ["token"]
        if d.get("was_steered"):
            klass_parts.append("steered")

        # Build top-5 tooltip payload
        tip_lines: list[str] = []
        tip_lines.append(f"step {d.get('step')}  tok={tok!r}")
        if d.get("was_steered"):
            tip_lines.append(
                f"STEERED: rule {d.get('active_rule')} "
                f"({d.get('rule_kind')}) δh={d.get('delta_h_norm')}"
            )
        tip_lines.append("Top-5 active latents:")
        for j in range(min(5, z.shape[0])):
            lid = int(z[j, 0].item())
            act = float(z[j, 1].item())
            if act <= 0.0 and j > 0:
                break
            toks = names.get(lid, [])
            label = " ".join(map(repr, toks[:5])) if toks else "(unnamed)"
            tip_lines.append(f"  L{lid:>6}  act={act:7.2f}  {label}")
        tip = "\n".join(tip_lines)

        if top1_lid in color_map:
            bg = color_map[top1_lid]
            fg = _text_color_for_bg_hex(bg)
            klass_parts.append("colored")
            style = f'background:{bg};color:{fg};'
        elif top1_lid < 0:
            klass_parts.append("unlabeled")
            style = ""
        else:
            klass_parts.append("unlabeled")
            style = ""

        klass = " ".join(klass_parts)
        h.append(
            f'<span class="{klass}" id="tok-{i}" '
            f'data-lid="{top1_lid}" data-tip="{html_mod.escape(tip)}" '
            f'style="{style}">{tok_disp}</span>'
        )
        if has_nl:
            h.append("<br>")

    h.append("</div></div>")  # text-content, text-panel

    # --- Right panel: legend with top-N latents -----------------------------
    h.append('<div class="right-panel">')
    h.append('<div class="legend-panel">')
    h.append(f'<div class="legend-title">Top-{len(top_counts)} most-active '
             'latents in this run</div>')
    for lid, cnt in top_counts:
        bg = color_map[lid]
        toks = names.get(lid, [])
        toks_str = ", ".join(repr(t) for t in toks[:3]) if toks else "(unnamed)"
        h.append('<div class="legend-row">')
        h.append(f'  <span class="legend-swatch" style="background:{bg};"></span>')
        h.append(f'  <span class="legend-lid">L{lid}</span>')
        h.append(f'  <span class="legend-count">×{cnt}</span>')
        h.append(f'  <span class="legend-toks" title="{html_mod.escape(toks_str)}">'
                 f'{html_mod.escape(toks_str)}</span>')
        h.append('</div>')
    h.append('</div>')  # legend-panel

    h.append('<div class="legend-panel">')
    h.append('<div class="legend-title">How to read</div>')
    h.append('<div style="font-size:11px;line-height:1.6;color:#444;">')
    h.append('Each token is colored by its <b>most-active SAE latent</b> '
             '(top-1 of the SAE\'s top-K). Latents not in the top-20 of this '
             'run are gray. Hover any token for the top-5 active latents '
             'with their max-activating tokens. Tokens with a <span '
             'style="box-shadow:0 3px 0 0 #e00;padding-bottom:3px;">red '
             'underline</span> were perturbed by an active rule.')
    h.append('</div></div>')

    h.append('</div>')  # right-panel
    h.append('</div>')  # container

    # --- Tooltip + JS -------------------------------------------------------
    h.append('<div class="tooltip" id="tooltip"></div>')
    h.append('<script>')
    h.append("""
const tooltip = document.getElementById('tooltip');
function showTip(ev, text) {
  tooltip.style.display = 'block';
  tooltip.textContent = text;
  tooltip.style.left = (ev.clientX + 14) + 'px';
  tooltip.style.top = (ev.clientY + 14) + 'px';
}
function hideTip() { tooltip.style.display = 'none'; }
document.querySelectorAll('.token').forEach(sp => {
  sp.addEventListener('mouseenter', ev => {
    sp.classList.add('hl');
    showTip(ev, sp.dataset.tip);
  });
  sp.addEventListener('mousemove', ev => {
    tooltip.style.left = (ev.clientX + 14) + 'px';
    tooltip.style.top = (ev.clientY + 14) + 'px';
  });
  sp.addEventListener('mouseleave', () => {
    sp.classList.remove('hl');
    hideTip();
  });
});
""")
    h.append('</script>')
    h.append('</body></html>')

    (out / "interactive_trajectory.html").write_text(
        "".join(h), encoding="utf-8",
    )
    logger.info("Saved %s", out / "interactive_trajectory.html")
