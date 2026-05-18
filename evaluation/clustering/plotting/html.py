"""Interactive trajectory HTML rendering."""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


from evaluation.clustering.plotting.colors import _cluster_colors, _group_colors, _margin_to_alpha, _text_color_for_bg
from evaluation.clustering.plotting.state_machine import (
    _compute_state_machine_layout,
    _render_state_machine_svg,
    _render_trajectory_svg,
    _render_live_trajectory_base_svg,
)

def repair_utf8_split_tokens(
    tokens: list[str],
    token_ids: list[int],
    tokenizer,
    max_merge: int = 4,
) -> list[str]:
    """Repair multi-byte UTF-8 characters split across BPE token boundaries.

    Byte-level BPE tokenizers (tiktoken, GPT-2 BPE, etc.) sometimes split
    a single multi-byte UTF-8 character (e.g. ``≈``, ``√``, ``…``, ``½``)
    across consecutive tokens. Decoding each token in isolation then
    yields the Unicode replacement character (``\\ufffd``) for those
    partial-byte tokens.

    This helper walks the per-token strings; whenever a token contains
    ``\\ufffd``, it tries merging with up to ``max_merge - 1`` following
    tokens and re-decoding their IDs as a group. If the merged result is
    clean, the leading partial-byte cells are blanked and the merged
    string is placed at the LAST cell of the merge window; if not even
    ``max_merge`` consecutive tokens combine cleanly, every cell is left
    untouched (we do **not** wait indefinitely — that would risk hanging
    on genuine model-emitted byte corruption).

    Args:
        tokens: per-token decoded strings (may contain ``\\ufffd``).
        token_ids: per-token integer IDs, same length as ``tokens``.
        tokenizer: any tokenizer with ``decode(list_of_ids) -> str``.
        max_merge: maximum window size for merge attempts (default 4).

    Returns:
        New list with partial-byte tokens collapsed into their owning
        character cell. Same length as the input.
    """
    if len(tokens) != len(token_ids):
        raise ValueError(
            f"tokens and token_ids must be same length: "
            f"{len(tokens)} vs {len(token_ids)}"
        )
    REPL = "�"
    out = list(tokens)
    n = len(out)
    i = 0
    while i < n:
        if REPL not in out[i]:
            i += 1
            continue
        repaired_str = None
        end = None
        for k in range(2, max_merge + 1):
            if i + k > n:
                break
            try:
                merged = tokenizer.decode(list(token_ids[i:i + k]))
            except Exception:
                break
            if REPL not in merged:
                repaired_str = merged
                end = i + k
                break
        if repaired_str is not None:
            for j in range(i, end - 1):
                out[j] = ""
            out[end - 1] = repaired_str
            i = end
        else:
            i += 1
    return out


def render_colored_text_html(
    tokens: list[str],
    labels: np.ndarray,
    K: int,
    out_path: Path,
    title: str = "",
    stride: int = 1,
    window_size: int = 1,
    top1_cos: np.ndarray | None = None,
    margin: np.ndarray | None = None,
    group_labels: np.ndarray | None = None,
) -> None:
    """Generate an HTML file with tokens colored by cluster assignment.

    Each token is rendered as a ``<span>`` whose background color corresponds
    to the assigned cluster.  Hover tooltips show cluster ID, cosine
    similarity, and margin.

    For stride > 1 the labels are expanded back to token length via
    forward-fill so that every token between two strided sample points
    inherits the color of the preceding sample.

    Args:
        tokens: per-token decoded strings (length N_tokens, padding removed).
        labels: (N_labels,) int array of cluster IDs.
        K: total number of clusters.
        out_path: path to write the HTML file.
        title: optional title shown at the top of the page.
        stride: stride used when producing labels.
        window_size: window size used when producing labels.
        top1_cos: optional (N_labels,) cosine similarity to assigned center.
        margin: optional (N_labels,) top1−top2 margin.
    """
    import html as html_mod

    colors = _cluster_colors(K, group_labels)
    n_tokens = len(tokens)

    # Expand labels back to per-token length when stride > 1
    if stride > 1:
        expanded = np.full(n_tokens, -1, dtype=np.int64)
        exp_cos = np.full(n_tokens, np.nan, dtype=np.float64) if top1_cos is not None else None
        exp_margin = np.full(n_tokens, np.nan, dtype=np.float64) if margin is not None else None
        for i, lbl in enumerate(labels):
            start = i * stride
            end = min(start + stride, n_tokens)
            expanded[start:end] = lbl
            if exp_cos is not None:
                exp_cos[start:end] = top1_cos[i]
            if exp_margin is not None:
                exp_margin[start:end] = margin[i]
        labels = expanded
        top1_cos = exp_cos
        margin = exp_margin
    else:
        # Trim labels to token length (labels may be shorter due to windowing)
        min_len = min(n_tokens, len(labels))
        if len(labels) < n_tokens:
            padded = np.full(n_tokens, -1, dtype=np.int64)
            padded[:len(labels)] = labels
            labels = padded
            if top1_cos is not None:
                pc = np.full(n_tokens, np.nan)
                pc[:len(top1_cos)] = top1_cos
                top1_cos = pc
            if margin is not None:
                pm = np.full(n_tokens, np.nan)
                pm[:len(margin)] = margin
                margin = pm

    # Build cluster-to-group lookup for data attributes
    cluster_to_group: dict[int, int] = {}
    if group_labels is not None:
        for k in range(K):
            cluster_to_group[k] = int(group_labels[k])

    # Build HTML
    parts: list[str] = []
    parts.append("<!DOCTYPE html>\n<html><head><meta charset='utf-8'>")
    parts.append("<style>")
    parts.append("body { font-family: 'Courier New', monospace; font-size: 14px; "
                 "line-height: 1.8; padding: 20px; background: #fafafa; }")
    parts.append(".text { font-size: 0; }")
    parts.append(".token { font-size: 14px; padding: 1px 0; border-radius: 2px; "
                 "cursor: default; white-space: pre-wrap; }")
    parts.append(".legend { display: flex; flex-wrap: wrap; gap: 4px 8px; "
                 "margin-bottom: 16px; font-size: 14px; align-items: flex-start; }")
    parts.append(".legend-item { display: inline-block; margin: 3px 6px; "
                 "padding: 3px 8px; border-radius: 3px; font-size: 12px; "
                 "cursor: pointer; transition: outline 0.1s; }")
    parts.append(".legend-group { display: inline-flex; flex-wrap: wrap; "
                 "align-items: center; cursor: pointer; }")
    parts.append(".legend-item.active { outline: 3px solid #333; }")
    parts.append(".legend-group.active > b { text-decoration: underline; }")
    parts.append(".legend-item.highlight { outline: 2px solid #333; }")
    parts.append(".unlabeled { background: #ddd; color: #666; }")
    parts.append("</style></head><body>")

    parts.append(f"<h2>{html_mod.escape(title or 'Cluster-Colored Text')}</h2>")

    # Legend
    parts.append('<div class="legend">')
    if group_labels is not None:
        K_coarse = int(group_labels.max()) + 1
        g_colors = _group_colors(K_coarse)
        for g in range(K_coarse):
            g_bg = g_colors[g]
            parts.append(
                f'<div class="legend-group" data-group="{g}" '
                f'style="border:2px solid {g_bg};'
                f'border-radius:6px;padding:2px 4px;margin:4px 4px;">'
                f'<b style="font-size:11px;">G{g}</b> '
            )
            members = [i for i in range(K) if group_labels[i] == g]
            for k in members:
                bg = colors[k]
                fg = _text_color_for_bg(bg)
                parts.append(
                    f'<span class="legend-item" data-cluster="{k}" '
                    f'style="background:{bg};color:{fg};">'
                    f'C{k}</span>'
                )
            parts.append("</div>")
    else:
        for k in range(K):
            bg = colors[k]
            fg = _text_color_for_bg(bg)
            parts.append(
                f'<span class="legend-item" data-cluster="{k}" '
                f'style="background:{bg};color:{fg};">'
                f'Cluster {k}</span>'
            )
    parts.append("</div>")

    # Token spans — join without newlines to avoid whitespace between inline spans
    token_spans: list[str] = []
    for i, tok in enumerate(tokens):
        lbl = int(labels[i]) if i < len(labels) else -1
        tok_escaped = html_mod.escape(tok)

        # Replace newline chars with a visible ↵ symbol inside the span,
        # then append a <br> *after* the span so text still wraps.
        has_newline = "\n" in tok_escaped
        tok_display = tok_escaped.replace("\n", "↵")

        if lbl < 0 or lbl >= K:
            token_spans.append(f'<span class="token unlabeled">{tok_display}</span>')
        else:
            bg = colors[lbl]
            alpha = _margin_to_alpha(margin[i]) if margin is not None and i < len(margin) else 1.0
            fg = "#000" if alpha < 0.5 else _text_color_for_bg(bg)
            r, g, b = mcolors.to_rgb(bg)
            bg_rgba = f"rgba({int(r*255)},{int(g*255)},{int(b*255)},{alpha:.2f})"
            tooltip_parts = [f"cluster={lbl}"]
            if top1_cos is not None and i < len(top1_cos) and not np.isnan(top1_cos[i]):
                tooltip_parts.append(f"cos={top1_cos[i]:.3f}")
            if margin is not None and i < len(margin) and not np.isnan(margin[i]):
                tooltip_parts.append(f"margin={margin[i]:.3f}")
            tooltip = ", ".join(tooltip_parts)
            grp_attr = (f' data-group="{cluster_to_group[lbl]}"'
                        if lbl in cluster_to_group else "")
            token_spans.append(
                f'<span class="token" data-cluster="{lbl}"{grp_attr} '
                f'style="background:{bg_rgba};color:{fg};" '
                f'title="{tooltip}">{tok_display}</span>'
            )
        if has_newline:
            token_spans.append("<br>")
    parts.append('<div class="text">' + "".join(token_spans) + "</div>")

    # Interactive JavaScript
    parts.append("""<script>
(function() {
  var activeCluster = null, activeGroup = null;
  var tokens = document.querySelectorAll('.token[data-cluster]');
  tokens.forEach(function(el) {
    el.dataset.origBg = el.style.background;
    el.dataset.origColor = el.style.color;
  });
  function resetAll() {
    tokens.forEach(function(el) {
      el.style.background = el.dataset.origBg;
      el.style.color = el.dataset.origColor;
    });
    document.querySelectorAll('.legend-item.active').forEach(function(el) {
      el.classList.remove('active');
    });
    document.querySelectorAll('.legend-group.active').forEach(function(el) {
      el.classList.remove('active');
    });
    activeCluster = null;
    activeGroup = null;
  }
  function filterByCluster(cid) {
    if (activeCluster === cid) { resetAll(); return; }
    resetAll();
    activeCluster = cid;
    tokens.forEach(function(el) {
      if (el.dataset.cluster !== String(cid)) {
        el.style.background = 'transparent';
        el.style.color = '#ccc';
      }
    });
    var li = document.querySelector('.legend-item[data-cluster="'+cid+'"]');
    if (li) li.classList.add('active');
  }
  function filterByGroup(gid) {
    if (activeGroup === gid) { resetAll(); return; }
    resetAll();
    activeGroup = gid;
    tokens.forEach(function(el) {
      if (el.dataset.group !== String(gid)) {
        el.style.background = 'transparent';
        el.style.color = '#ccc';
      }
    });
    var lg = document.querySelector('.legend-group[data-group="'+gid+'"]');
    if (lg) lg.classList.add('active');
  }
  document.querySelectorAll('.legend-item[data-cluster]').forEach(function(el) {
    el.addEventListener('click', function(e) {
      e.stopPropagation();
      filterByCluster(parseInt(el.dataset.cluster));
    });
  });
  document.querySelectorAll('.legend-group[data-group]').forEach(function(el) {
    el.querySelector('b').addEventListener('click', function(e) {
      e.stopPropagation();
      filterByGroup(parseInt(el.dataset.group));
    });
  });
  tokens.forEach(function(el) {
    el.addEventListener('mouseenter', function() {
      var cid = el.dataset.cluster;
      var li = document.querySelector('.legend-item[data-cluster="'+cid+'"]');
      if (li) li.classList.add('highlight');
    });
    el.addEventListener('mouseleave', function() {
      document.querySelectorAll('.legend-item.highlight').forEach(function(h) {
        h.classList.remove('highlight');
      });
    });
  });
})();
</script>""")

    parts.append("</body></html>")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(parts), encoding="utf-8")
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# LLM-friendly cluster naming input
# ---------------------------------------------------------------------------


def _build_js_name_dict(names: dict[int, tuple[str, str]] | None) -> str:
    """Build a JS object literal from a names dict."""
    if not names:
        return "{}"
    entries = []
    for k in sorted(names):
        name_esc = json.dumps(names[k][0])
        desc_esc = json.dumps(names[k][1])
        entries.append(f"{k}: {{name: {name_esc}, desc: {desc_esc}}}")
    return "{" + ", ".join(entries) + "}"


# ---------------------------------------------------------------------------
# Interactive HTML: side-by-side colored text + UMAP trajectory with Voronoi
# ---------------------------------------------------------------------------


def render_interactive_trajectory_html(
    tokens: list[str],
    labels: np.ndarray,
    features: np.ndarray,
    K: int,
    centers: np.ndarray,
    out_path: Path,
    title: str = "",
    stride: int = 1,
    window_size: int = 1,
    top1_cos: np.ndarray | None = None,
    margin: np.ndarray | None = None,
    group_labels: np.ndarray | None = None,
    P_exit_micro: np.ndarray | None = None,
    P_exit: np.ndarray | None = None,
    usage: np.ndarray | None = None,
    K_coarse: int | None = None,
    P_self_loop: np.ndarray | None = None,
    micro_top_r: int = 3,
    micro_min_prob: float = 0.03,
    macro_top_r: int = 3,
    macro_min_prob: float = 0.03,
    cluster_names: dict[int, tuple[str, str]] | None = None,
    group_names: dict[int, tuple[str, str]] | None = None,
    token_ids: list[int] | None = None,
    tokenizer=None,
) -> None:
    """Generate an interactive HTML with colored text + UMAP trajectory.

    Left panel: cluster-colored text. Right panel: SVG UMAP scatter with
    Voronoi cells (group-level when *group_labels* is given, else per-cluster)
    and points colored by temporal position (viridis).
    Bidirectional hover: text ↔ UMAP point.

    Args:
        tokens: per-token decoded strings (length N_tokens).
        labels: (N_labels,) cluster assignments (pre-expansion).
        features: (N_labels, D) preprocessed feature vectors for UMAP.
        K: number of clusters.
        centers: (K, D) cluster centers in original feature space.
        out_path: path to write the HTML file.
        title: figure title.
        stride: stride used when producing labels.
        window_size: window size used when producing labels.
        top1_cos: optional (N_labels,) cosine similarity to assigned center.
        margin: optional (N_labels,) top1-top2 margin.
        token_ids, tokenizer: optional. When both are provided, multi-byte
            UTF-8 characters split across token boundaries (which decode to
            ``\\ufffd`` per-token) are repaired before rendering. Bounded
            look-ahead so genuinely-corrupt sequences don't hang the loop.
    """
    import html as html_mod
    import json as json_mod
    import umap as umap_lib
    from scipy.spatial import Voronoi

    if token_ids is not None and tokenizer is not None:
        tokens = repair_utf8_split_tokens(tokens, token_ids, tokenizer)

    colors = _cluster_colors(K, group_labels)
    n_tokens = len(tokens)
    N_labels = len(labels)

    # --- 1. Expand labels to per-token for text panel ---
    if stride > 1:
        exp_labels = np.full(n_tokens, -1, dtype=np.int64)
        exp_cos = np.full(n_tokens, np.nan) if top1_cos is not None else None
        exp_margin = np.full(n_tokens, np.nan) if margin is not None else None
        for i, lbl in enumerate(labels):
            s = i * stride
            e = min(s + stride, n_tokens)
            exp_labels[s:e] = lbl
            if exp_cos is not None:
                exp_cos[s:e] = top1_cos[i]
            if exp_margin is not None:
                exp_margin[s:e] = margin[i]
    else:
        exp_labels = np.full(n_tokens, -1, dtype=np.int64)
        n = min(n_tokens, N_labels)
        exp_labels[:n] = labels[:n]
        exp_cos = None
        if top1_cos is not None:
            exp_cos = np.full(n_tokens, np.nan)
            exp_cos[:min(n_tokens, len(top1_cos))] = top1_cos[:min(n_tokens, len(top1_cos))]
        exp_margin = None
        if margin is not None:
            exp_margin = np.full(n_tokens, np.nan)
            exp_margin[:min(n_tokens, len(margin))] = margin[:min(n_tokens, len(margin))]

    # --- 2. UMAP fitting ---
    T = features.shape[0]
    if T < 3:
        print(f"  Skipping interactive HTML (only {T} features): {out_path}")
        return

    n_neighbors = min(15, T - 1)
    reducer = umap_lib.UMAP(
        n_components=2, n_neighbors=n_neighbors, min_dist=0.1, random_state=42,
    )
    coords_2d = reducer.fit_transform(features)
    centers_2d = reducer.transform(centers)

    # --- 3. Voronoi (group-level when group_labels provided, else per-cluster) ---
    all_pts = np.vstack([coords_2d, centers_2d])
    x_min, x_max = float(all_pts[:, 0].min()), float(all_pts[:, 0].max())
    y_min, y_max = float(all_pts[:, 1].min()), float(all_pts[:, 1].max())
    data_range = max(x_max - x_min, y_max - y_min, 1e-6)
    pad = 0.15 * data_range
    x_min -= pad
    x_max += pad
    y_min -= pad
    y_max += pad

    far = 100 * data_range
    dummy = np.array([
        [x_min - far, y_min - far], [x_max + far, y_min - far],
        [x_min - far, y_max + far], [x_max + far, y_max + far],
    ])

    if group_labels is not None:
        # Group-level Voronoi
        K_coarse = int(group_labels.max()) + 1
        vor_centers = np.zeros((K_coarse, 2))
        for g in range(K_coarse):
            members = [i for i in range(K) if group_labels[i] == g]
            vor_centers[g] = centers_2d[members].mean(axis=0)
        vor_colors = _group_colors(K_coarse)
        vor_n = K_coarse
    else:
        vor_centers = centers_2d
        vor_colors = colors
        vor_n = K

    try:
        vor = Voronoi(np.vstack([vor_centers, dummy]))
        voronoi_ok = True
    except Exception:
        voronoi_ok = False

    # --- 4. SVG coordinate mapping ---
    svg_w, svg_h = 700, 700
    pl, pt_y, pr, pb = 10, 10, 690, 660  # plot area within SVG
    pw, ph = pr - pl, pb - pt_y

    def sx(x):
        return pl + (x - x_min) / max(x_max - x_min, 1e-9) * pw

    def sy(y):
        return pb - (y - y_min) / max(y_max - y_min, 1e-9) * ph

    # --- 5. Viridis time colors ---
    viridis_cm = plt.cm.viridis
    time_colors = [
        mcolors.to_hex(viridis_cm(i / max(N_labels - 1, 1)))
        for i in range(N_labels)
    ]

    # Viridis gradient stops for colorbar
    viridis_stops = [
        mcolors.to_hex(viridis_cm(i / 10)) for i in range(11)
    ]
    gradient_css = "linear-gradient(to right, " + ", ".join(viridis_stops) + ")"

    # --- 6. Build HTML ---
    h = []  # accumulate HTML parts

    h.append("<!DOCTYPE html>\n<html><head><meta charset='utf-8'><style>")
    h.append("""
body { margin:0; padding:16px; font-family:sans-serif; background:#fafafa; }
h2 { margin:0 0 8px; font-size:16px; }
.legend { margin-bottom:10px; display:flex; flex-wrap:wrap; gap:4px 8px; }
.legend-item { display:inline-block; margin:2px 3px; padding:2px 6px;
  border-radius:3px; font-size:11px; }
.container { display:flex; gap:12px; height:calc(100vh - 100px); }
.text-panel { flex:0 0 45%; overflow-y:auto; padding:10px; background:#fff;
  border:1px solid #ddd; border-radius:4px; }
.text-content { font-family:'Courier New',monospace; font-size:13px;
  line-height:1.8; font-size:0; }
.right-panel { flex:1; display:flex; flex-direction:column; }
.tab-bar { display:flex; gap:4px; margin-bottom:6px; }
.tab-btn { padding:6px 16px; border:1px solid #ccc; border-radius:4px 4px 0 0;
  background:#eee; cursor:pointer; font-size:12px; border-bottom:none; }
.tab-btn.active { background:#fff; font-weight:bold; border-bottom:1px solid #fff; }
.tab-content { display:none; flex:1; flex-direction:column; overflow:auto; }
.tab-content.active { display:flex; }
.umap-svg { flex:1; width:100%; }
.token { font-size:13px; padding:1px 0; border-radius:2px;
  cursor:pointer; white-space:pre-wrap; }
.token.hl { outline:3px solid #333; outline-offset:1px;
  z-index:1; position:relative; }
.unlabeled { background:#ddd; color:#666; }
.umap-point { cursor:pointer; }
.umap-point.hl { stroke:#000; stroke-width:2.5; }
.tooltip { position:fixed; background:rgba(0,0,0,0.88); color:#fff;
  padding:5px 9px; border-radius:4px; font-size:11px;
  pointer-events:none; display:none; z-index:1000; max-width:320px;
  font-family:'Courier New',monospace; white-space:pre-wrap; }
.legend-item { cursor:pointer; transition:outline 0.1s; }
.legend-group { cursor:pointer; display:inline-flex; flex-wrap:wrap; gap:2px; align-items:center; }
.legend-item.active { outline:3px solid #333; }
.legend-group.active > b { text-decoration:underline; }
.legend-item.highlight { outline:2px solid #333; }
.sm-container { border:1px solid #ddd; border-radius:4px; background:#fff; }
.sm-svg { width:100%; height:auto; }
.sm-node { cursor:pointer; }
.sm-bubble { cursor:pointer; }
.sm-node.sm-hl { filter:url(#sm-glow); stroke:red; stroke-width:3; }
.sm-bubble.sm-hl { fill-opacity:0.35; stroke:#333; stroke-width:2.5; }
.sm-micro-edge.sm-hl { stroke-opacity:1; stroke:#333; }
.sm-macro-edge.sm-hl { stroke-opacity:1; stroke:red; stroke-width:4; }
.traj-edge.sm-hl { stroke-opacity:1; stroke:#333; stroke-width:3; }
.token.sm-faded { opacity:0.2; }
.sm-controls { padding:6px 12px; background:#f5f5f5; border:1px solid #ddd;
  display:flex; gap:16px; flex-wrap:wrap; align-items:center; }
.sm-controls label { font-size:11px; color:#555; display:flex; gap:4px;
  align-items:center; white-space:nowrap; }
.sm-controls input[type=range] { width:100px; }
.sm-controls span { font-weight:bold; min-width:28px; }
.legend-tabs { display:flex; gap:0; }
.legend-tab-btn { padding:3px 10px; font-size:11px; border:1px solid #ccc;
  background:#eee; cursor:pointer; border-bottom:none; border-radius:4px 4px 0 0; }
.legend-tab-btn.active { background:white; font-weight:bold; }
.legend-tab-content { display:none; }
.legend-tab-content.active { display:block; }
.names-panel { max-height:200px; overflow-y:auto; padding:6px;
  font-size:11px; line-height:1.5; border:1px solid #ddd; border-top:none;
  background:white; }
.names-group-header { font-weight:bold; padding:3px 6px; margin:4px 0 2px;
  background:#f0f0f0; border-radius:3px; }
.names-cluster { padding:1px 6px 1px 16px; color:#333; }
.names-swatch { display:inline-block; width:10px; height:10px;
  border-radius:2px; margin-right:4px; vertical-align:middle; }
.colorbar-row { display:flex; align-items:center; gap:6px; margin:4px 10px;
  font-size:10px; color:#555; }
.colorbar-gradient { flex:1; height:12px; border-radius:2px; }
.token.live-current { outline:2px solid gold; z-index:2; position:relative; }
.token.live-window { outline:1px solid #999; position:relative; }
#live-info { padding:4px 8px; font-size:12px; background:#f8f8f8;
  border-bottom:1px solid #ddd; }
#live-controls { padding:6px 8px; display:flex; gap:12px; align-items:center;
  flex-wrap:wrap; font-size:12px; background:#f5f5f5; border:1px solid #ddd; }
#live-controls label { display:flex; align-items:center; gap:4px;
  white-space:nowrap; }
#live-pos-slider { width:300px; }
#live-play-btn { padding:2px 10px; cursor:pointer; }
""")
    h.append("</style></head><body>")

    # Title
    h.append(f"<h2>{html_mod.escape(title or 'Interactive Trajectory')}</h2>")

    # Legend area with optional Names sub-tab
    _has_names = bool(cluster_names or group_names)
    _K_coarse_leg = K_coarse if K_coarse is not None else (
        int(group_labels.max()) + 1 if group_labels is not None else 0
    )

    if _has_names:
        h.append('<div class="legend-tabs">')
        h.append('  <button class="legend-tab-btn active" data-ltab="legend">Legend</button>')
        h.append('  <button class="legend-tab-btn" data-ltab="names">Names</button>')
        h.append('</div>')

    # --- Legend tab ---
    if _has_names:
        h.append('<div class="legend-tab-content active" id="ltab-legend">')
    h.append('<div class="legend">')
    if group_labels is not None:
        g_colors = _group_colors(_K_coarse_leg)
        for g in range(_K_coarse_leg):
            g_bg = g_colors[g]
            h.append(
                f'<span class="legend-group" data-group="{g}" '
                f'style="display:inline-flex;flex-wrap:wrap;align-items:center;'
                f'border:2px solid {g_bg};border-radius:6px;padding:2px 4px;'
                f'margin:2px 3px;cursor:pointer;">'
                f'<b style="font-size:10px;">G{g}</b> '
            )
            members = [i for i in range(K) if group_labels[i] == g]
            for k in members:
                bg = colors[k]
                fg = _text_color_for_bg(bg)
                h.append(
                    f'<span class="legend-item" data-cluster="{k}" '
                    f'style="background:{bg};color:{fg};">'
                    f"C{k}</span>"
                )
            h.append("</span>")
    else:
        for k in range(K):
            bg = colors[k]
            fg = _text_color_for_bg(bg)
            h.append(
                f'<span class="legend-item" data-cluster="{k}" '
                f'style="background:{bg};color:{fg};">'
                f"C{k}</span>"
            )
    h.append("</div>")
    if _has_names:
        h.append('</div>')  # ltab-legend

    # --- Names tab ---
    if _has_names:
        h.append('<div class="legend-tab-content" id="ltab-names">')
        h.append('<div class="names-panel">')
        if group_labels is not None:
            g_colors_n = _group_colors(_K_coarse_leg)
            for g in range(_K_coarse_leg):
                gname, gdesc = (group_names or {}).get(g, (f"Group {g}", ""))
                h.append(
                    f'<div class="names-group-header" '
                    f'style="border-left:4px solid {g_colors_n[g]}">'
                    f'G{g}: {html_mod.escape(gname)}'
                )
                if gdesc:
                    h.append(f' &mdash; <span style="font-weight:normal">{html_mod.escape(gdesc)}</span>')
                h.append('</div>')
                members = [i for i in range(K) if group_labels[i] == g]
                for ci in members:
                    cname, cdesc = (cluster_names or {}).get(ci, (f"Cluster {ci}", ""))
                    h.append(
                        f'<div class="names-cluster">'
                        f'<span class="names-swatch" style="background:{colors[ci]}"></span>'
                        f'C{ci}: {html_mod.escape(cname)}'
                    )
                    if cdesc:
                        h.append(f' &mdash; <span style="color:#666">{html_mod.escape(cdesc)}</span>')
                    h.append('</div>')
        else:
            for ci in range(K):
                cname, cdesc = (cluster_names or {}).get(ci, (f"Cluster {ci}", ""))
                h.append(
                    f'<div class="names-cluster">'
                    f'<span class="names-swatch" style="background:{colors[ci]}"></span>'
                    f'C{ci}: {html_mod.escape(cname)}'
                )
                if cdesc:
                    h.append(f' &mdash; <span style="color:#666">{html_mod.escape(cdesc)}</span>')
                h.append('</div>')
        h.append('</div>')  # names-panel
        h.append('</div>')  # ltab-names

    h.append('<div class="container">')

    # ---- Left panel: colored text ----
    h.append('<div class="text-panel"><div class="text-content">')
    for t_idx, tok in enumerate(tokens):
        lbl = int(exp_labels[t_idx]) if t_idx < len(exp_labels) else -1
        tok_esc = html_mod.escape(tok)
        has_nl = "\n" in tok_esc
        tok_disp = tok_esc.replace("\n", "\u21b5")

        # Which UMAP index does this token map to?
        umap_idx = t_idx // stride if stride > 1 else t_idx
        if umap_idx >= N_labels:
            umap_idx = -1

        # Tooltip text
        tip_parts = [f"pos={t_idx}"]
        if lbl >= 0:
            tip_parts.append(f"cluster={lbl}")
        if exp_cos is not None and t_idx < len(exp_cos) and not np.isnan(exp_cos[t_idx]):
            tip_parts.append(f"cos={exp_cos[t_idx]:.3f}")
        if exp_margin is not None and t_idx < len(exp_margin) and not np.isnan(exp_margin[t_idx]):
            tip_parts.append(f"margin={exp_margin[t_idx]:.3f}")
        tip = ", ".join(tip_parts)

        # Data attributes for cluster/group
        grp_attr = ""
        if lbl >= 0 and lbl < K and group_labels is not None:
            grp_attr = f' data-cluster="{lbl}" data-group="{int(group_labels[lbl])}"'
        elif lbl >= 0 and lbl < K:
            grp_attr = f' data-cluster="{lbl}"'

        if lbl < 0 or lbl >= K:
            h.append(
                f'<span class="token unlabeled" id="tok-{t_idx}" '
                f'data-umap="{umap_idx}" data-tip="{tip}">{tok_disp}</span>'
            )
        else:
            bg = colors[lbl]
            alpha = _margin_to_alpha(exp_margin[t_idx]) if exp_margin is not None and t_idx < len(exp_margin) else 1.0
            fg = "#000" if alpha < 0.5 else _text_color_for_bg(bg)
            r, g, b = mcolors.to_rgb(bg)
            bg_rgba = f"rgba({int(r*255)},{int(g*255)},{int(b*255)},{alpha:.2f})"
            h.append(
                f'<span class="token" id="tok-{t_idx}" data-umap="{umap_idx}"'
                f'{grp_attr} '
                f'style="background:{bg_rgba};color:{fg};" data-tip="{tip}">'
                f"{tok_disp}</span>"
            )
        if has_nl:
            h.append("<br>")

    h.append("</div></div>")

    # ---- Right panel: tabbed (UMAP / State Machine) ----
    h.append('<div class="right-panel">')
    h.append('<div class="tab-bar">')
    h.append('  <button class="tab-btn active" data-tab="umap">UMAP Trajectory</button>')
    h.append('  <button class="tab-btn" data-tab="sm">State Machine</button>')
    h.append('  <button class="tab-btn" data-tab="traj">Trajectory</button>')
    h.append('  <button class="tab-btn" data-tab="live">Live</button>')
    h.append('</div>')
    h.append('<div class="tab-content active" id="tab-umap">')
    h.append(f'<svg class="umap-svg" viewBox="0 0 {svg_w} {svg_h}">')

    # Voronoi polygons
    if voronoi_ok:
        for k in range(vor_n):
            region_idx = vor.point_region[k]
            region = vor.regions[region_idx]
            if -1 not in region and len(region) > 0:
                poly_pts = " ".join(
                    f"{sx(vor.vertices[vi][0]):.1f},{sy(vor.vertices[vi][1]):.1f}"
                    for vi in region
                )
                h.append(
                    f'<polygon points="{poly_pts}" '
                    f'fill="{vor_colors[k]}" fill-opacity="0.15" '
                    f'stroke="none"/>'
                )
        # Voronoi edges
        for simplex in vor.ridge_vertices:
            if -1 not in simplex:
                v0, v1 = vor.vertices[simplex[0]], vor.vertices[simplex[1]]
                h.append(
                    f'<line x1="{sx(v0[0]):.1f}" y1="{sy(v0[1]):.1f}" '
                    f'x2="{sx(v1[0]):.1f}" y2="{sy(v1[1]):.1f}" '
                    f'stroke="#999" stroke-width="0.5" stroke-opacity="0.5"/>'
                )

    # Trajectory line
    line_pts = " ".join(
        f"{sx(coords_2d[i, 0]):.1f},{sy(coords_2d[i, 1]):.1f}"
        for i in range(N_labels)
    )
    h.append(
        f'<polyline points="{line_pts}" fill="none" '
        f'stroke="#888" stroke-width="0.8" stroke-opacity="0.3"/>'
    )

    # Scatter points (viridis time colors)
    for j in range(N_labels):
        cx, cy = sx(coords_2d[j, 0]), sy(coords_2d[j, 1])
        lbl_j = int(labels[j]) if j < len(labels) else -1
        # Info for tooltip when hovering UMAP point
        info_parts = [f"pos={j}"]
        if lbl_j >= 0:
            info_parts.append(f"C{lbl_j}")
        if top1_cos is not None and j < len(top1_cos):
            info_parts.append(f"cos={top1_cos[j]:.3f}")
        if margin is not None and j < len(margin):
            info_parts.append(f"margin={margin[j]:.3f}")
        # Token text snippet
        tok_start = j * stride if stride > 1 else j
        tok_end = min(tok_start + (stride if stride > 1 else 1), n_tokens)
        tok_snippet = "".join(tokens[tok_start:tok_end])
        if len(tok_snippet) > 40:
            tok_snippet = tok_snippet[:37] + "..."
        info_parts.append(html_mod.escape(repr(tok_snippet)))
        info = " | ".join(info_parts)

        h.append(
            f'<circle class="umap-point" id="pt-{j}" '
            f'cx="{cx:.1f}" cy="{cy:.1f}" r="3" '
            f'fill="{time_colors[j]}" fill-opacity="0.8" '
            f'stroke="#333" stroke-width="0.3" '
            f'data-idx="{j}" data-info="{info}"/>'
        )

    # Cluster centers
    for k in range(K):
        cx, cy = sx(centers_2d[k, 0]), sy(centers_2d[k, 1])
        h.append(
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="7" '
            f'fill="{colors[k]}" stroke="#000" stroke-width="1.2"/>'
        )
        h.append(
            f'<text x="{cx:.1f}" y="{cy - 10:.1f}" '
            f'text-anchor="middle" font-size="9" font-weight="bold" '
            f'fill="#333">C{k}</text>'
        )

    # Start / end markers
    s0x, s0y = sx(coords_2d[0, 0]), sy(coords_2d[0, 1])
    e0x, e0y = sx(coords_2d[-1, 0]), sy(coords_2d[-1, 1])
    h.append(
        f'<circle cx="{s0x:.1f}" cy="{s0y:.1f}" r="6" '
        f'fill="lime" stroke="#000" stroke-width="1.2"/>'
    )
    h.append(
        f'<text x="{s0x:.1f}" y="{s0y - 9:.1f}" text-anchor="middle" '
        f'font-size="8" fill="#000">START</text>'
    )
    h.append(
        f'<circle cx="{e0x:.1f}" cy="{e0y:.1f}" r="6" '
        f'fill="red" stroke="#000" stroke-width="1.2"/>'
    )
    h.append(
        f'<text x="{e0x:.1f}" y="{e0y - 9:.1f}" text-anchor="middle" '
        f'font-size="8" fill="#000">END</text>'
    )

    h.append("</svg>")

    # Colorbar
    h.append('<div class="colorbar-row">')
    h.append("<span>Start</span>")
    h.append(f'<div class="colorbar-gradient" style="background:{gradient_css};"></div>')
    h.append("<span>End</span>")
    h.append("</div>")

    h.append("</div>")  # tab-umap

    # ---- State machine SVG (second tab) ----
    has_sm = (P_exit_micro is not None and group_labels is not None
              and P_exit is not None and usage is not None)
    h.append('<div class="tab-content" id="tab-sm">')
    if has_sm:
        _kc = K_coarse if K_coarse is not None else int(group_labels.max()) + 1
        sm_layout = _compute_state_machine_layout(
            P_exit_micro, P_exit, group_labels, usage, _kc,
            micro_top_r, micro_min_prob, macro_top_r, macro_min_prob,
        )
        sm_colors = _cluster_colors(K, group_labels)
        sm_macro_colors = _group_colors(_kc)
        sm_svg = _render_state_machine_svg(
            sm_layout, K, _kc, group_labels, usage, P_self_loop,
            sm_colors, sm_macro_colors,
            default_micro_top_r=micro_top_r,
            default_micro_min_prob=micro_min_prob,
            default_macro_top_r=macro_top_r,
            default_macro_min_prob=macro_min_prob,
        )
        # Controls panel
        h.append('<div class="sm-controls">')
        h.append(f'  <label>Micro top-r: <input type="range" id="sm-micro-top-r" '
                 f'min="1" max="10" value="{micro_top_r}" step="1">'
                 f'<span id="sm-micro-top-r-val">{micro_top_r}</span></label>')
        h.append(f'  <label>Micro min-p: <input type="range" id="sm-micro-min-prob" '
                 f'min="0" max="0.50" value="{micro_min_prob}" step="0.01">'
                 f'<span id="sm-micro-min-prob-val">{micro_min_prob:.2f}</span></label>')
        h.append(f'  <label>Macro top-r: <input type="range" id="sm-macro-top-r" '
                 f'min="1" max="10" value="{macro_top_r}" step="1">'
                 f'<span id="sm-macro-top-r-val">{macro_top_r}</span></label>')
        h.append(f'  <label>Macro min-p: <input type="range" id="sm-macro-min-prob" '
                 f'min="0" max="0.50" value="{macro_min_prob}" step="0.01">'
                 f'<span id="sm-macro-min-prob-val">{macro_min_prob:.2f}</span></label>')
        h.append('</div>')
        h.append('<div class="sm-container">')
        h.append(sm_svg)
        h.append("</div>")
    else:
        h.append('<p style="padding:20px;color:#888;">No state machine data available.</p>')
    h.append("</div>")  # tab-sm

    # ---- Trajectory SVG (third tab) ----
    h.append('<div class="tab-content" id="tab-traj">')
    if has_sm:
        traj_svg = _render_trajectory_svg(
            sm_layout, labels, K, _kc, group_labels, usage,
            sm_colors, sm_macro_colors,
        )
        # Controls panel
        h.append('<div class="sm-controls">')
        h.append('  <label>Min transition count: <input type="range" id="traj-min-count" '
                 'min="1" max="10" value="1" step="1">'
                 '<span id="traj-min-count-val">1</span></label>')
        h.append('</div>')
        h.append('<div class="sm-container">')
        h.append(traj_svg)
        h.append("</div>")
    else:
        h.append('<p style="padding:20px;color:#888;">No state machine data available.</p>')
    h.append("</div>")  # tab-traj

    # ---- Live Trajectory (fourth tab) ----
    h.append('<div class="tab-content" id="tab-live">')
    if has_sm:
        # Compute RLE state sequence
        live_state_seq = []
        _i = 0
        while _i < len(labels):
            _c = int(labels[_i])
            _j = _i + 1
            while _j < len(labels) and int(labels[_j]) == _c:
                _j += 1
            live_state_seq.append((_c, _i, _j - 1))
            _i = _j

        # Render base SVG (nodes only, no edges)
        live_svg, live_meta = _render_live_trajectory_base_svg(
            sm_layout, labels, K, _kc, group_labels, usage,
            sm_colors, sm_macro_colors,
        )

        # Controls
        max_pos = len(live_state_seq) - 1
        h.append('<div id="live-controls">')
        h.append(f'  <label>Position: <input type="range" id="live-pos-slider" '
                 f'min="0" max="{max_pos}" value="0" step="1">'
                 f'  <span id="live-pos-val">0</span>/{max_pos}</label>')
        h.append('  <label>Window: <input type="range" id="live-w-slider" '
                 'min="1" max="10" value="2" step="1">'
                 '  <span id="live-w-val">2</span></label>')
        h.append('  <button id="live-play-btn">&#9654; Play</button>')
        h.append('  <label>Speed: <input type="range" id="live-speed" '
                 'min="100" max="2000" value="500" step="100">'
                 '  <span id="live-speed-val">500</span>ms</label>')
        h.append('</div>')

        # Info bar
        h.append('<div id="live-info">Cluster: &mdash; | Group: &mdash; | Tokens: &mdash;</div>')

        # SVG container
        h.append('<div class="sm-container">')
        h.append(live_svg)
        h.append('</div>')

        # Emit JS data
        import json as json_mod
        seq_js = json_mod.dumps([[c, s, e] for c, s, e in live_state_seq])
        pos_js = json_mod.dumps({str(k): [round(v[0], 1), round(v[1], 1)]
                                 for k, v in live_meta["node_pos"].items()})
        sizes_js = json_mod.dumps({str(k): round(float(v), 1)
                                   for k, v in live_meta["node_sizes"].items()})
        gl_js = json_mod.dumps({str(i): int(group_labels[i]) for i in range(K)})

        h.append(f'<script>var LIVE_SEQ={seq_js}; var LIVE_NODE_POS={pos_js}; '
                 f'var LIVE_NODE_SIZES={sizes_js}; var LIVE_NODE_R={live_meta["node_r"]}; '
                 f'var LIVE_GL={gl_js};</script>')
    else:
        h.append('<p style="padding:20px;color:#888;">No state machine data available.</p>')
    h.append("</div>")  # tab-live

    h.append("</div>")  # right-panel
    h.append("</div>")  # container

    # Tooltip div
    h.append('<div class="tooltip" id="tooltip"></div>')

    # ---- JavaScript: bidirectional hover ----
    h.append("<script>")
    h.append(f"const STRIDE={stride}, N_LABELS={N_labels}, N_TOKENS={n_tokens};")
    h.append(f"var CLUSTER_NAMES = {_build_js_name_dict(cluster_names)};")
    h.append(f"var GROUP_NAMES = {_build_js_name_dict(group_names)};")
    h.append("""
const tooltip = document.getElementById('tooltip');

function umapIdxForToken(t) {
  const j = STRIDE > 1 ? Math.floor(t / STRIDE) : t;
  return j < N_LABELS ? j : -1;
}
function tokenRangeForUmap(j) {
  const s = STRIDE > 1 ? j * STRIDE : j;
  const e = STRIDE > 1 ? Math.min(s + STRIDE, N_TOKENS) : Math.min(s + 1, N_TOKENS);
  return [s, e];
}

function highlight(umapIdx) {
  if (umapIdx < 0 || umapIdx >= N_LABELS) return;
  const c = document.getElementById('pt-' + umapIdx);
  if (c) { c.setAttribute('r', '7'); c.classList.add('hl'); }
  const [s, e] = tokenRangeForUmap(umapIdx);
  for (let t = s; t < e; t++) {
    const sp = document.getElementById('tok-' + t);
    if (sp) sp.classList.add('hl');
  }
}
function unhighlight(umapIdx) {
  if (umapIdx < 0 || umapIdx >= N_LABELS) return;
  const c = document.getElementById('pt-' + umapIdx);
  if (c) { c.setAttribute('r', '3'); c.classList.remove('hl'); }
  const [s, e] = tokenRangeForUmap(umapIdx);
  for (let t = s; t < e; t++) {
    const sp = document.getElementById('tok-' + t);
    if (sp) sp.classList.remove('hl');
  }
}
function showTip(ev, text) {
  tooltip.style.display = 'block';
  tooltip.textContent = text;
  tooltip.style.left = (ev.clientX + 14) + 'px';
  tooltip.style.top = (ev.clientY + 14) + 'px';
}
function hideTip() { tooltip.style.display = 'none'; }

// --- Tab switching ---
document.querySelectorAll('.tab-btn').forEach(function(btn) {
  btn.addEventListener('click', function() {
    document.querySelectorAll('.tab-btn').forEach(function(b) { b.classList.remove('active'); });
    document.querySelectorAll('.tab-content').forEach(function(tc) {
      tc.style.display = 'none';
      tc.classList.remove('active');
    });
    btn.classList.add('active');
    var tab = document.getElementById('tab-' + btn.dataset.tab);
    if (tab) { tab.style.display = 'flex'; tab.classList.add('active'); }
    // Clean up live-tab highlights when leaving
    if (btn.dataset.tab !== 'live') {
      document.querySelectorAll('.token.live-current').forEach(function(el) { el.classList.remove('live-current'); });
      document.querySelectorAll('.token.live-window').forEach(function(el) { el.classList.remove('live-window'); });
    }
  });
});

// Token spans → UMAP + SM + legend
document.querySelectorAll('.token[data-umap]').forEach(sp => {
  const u = parseInt(sp.dataset.umap);
  // Store original styles for filter restore
  sp.dataset.origBg = sp.style.background || '';
  sp.dataset.origColor = sp.style.color || '';
  sp.addEventListener('mouseenter', ev => {
    highlight(u);
    // Build tooltip with optional cluster/group names
    var tipText = sp.dataset.tip;
    var cid = sp.dataset.cluster;
    var gid = sp.dataset.group;
    var nameParts = [];
    if (gid !== undefined && GROUP_NAMES[gid]) nameParts.push('G' + gid + ': ' + GROUP_NAMES[gid].name);
    if (cid !== undefined && CLUSTER_NAMES[cid]) nameParts.push('C' + cid + ': ' + CLUSTER_NAMES[cid].name);
    if (nameParts.length > 0) tipText += '\\n' + nameParts.join(' | ');
    showTip(ev, tipText);
    // SM + Trajectory highlight: node + bubble
    if (cid !== undefined) {
      var node = document.getElementById('sm-node-' + cid);
      if (node) node.classList.add('sm-hl');
      var trajNode = document.getElementById('traj-node-' + cid);
      if (trajNode) trajNode.classList.add('sm-hl');
      var liveNode = document.getElementById('live-node-' + cid);
      if (liveNode) liveNode.classList.add('sm-hl');
    }
    if (gid !== undefined) {
      var bubble = document.getElementById('sm-bubble-' + gid);
      if (bubble) bubble.classList.add('sm-hl');
      var trajBubble = document.getElementById('traj-bubble-' + gid);
      if (trajBubble) trajBubble.classList.add('sm-hl');
      var liveBubble = document.getElementById('live-bubble-' + gid);
      if (liveBubble) liveBubble.classList.add('sm-hl');
    }
    // SM highlight: transition arrow to next token
    var tIdx = parseInt(sp.id.replace('tok-', ''));
    var nextTok = document.getElementById('tok-' + (tIdx + 1));
    if (nextTok && cid !== undefined) {
      var nextCid = nextTok.dataset.cluster;
      var nextGid = nextTok.dataset.group;
      if (nextCid !== undefined && nextCid !== cid) {
        document.querySelectorAll('.sm-micro-edge[data-src="'+cid+'"][data-dst="'+nextCid+'"]')
          .forEach(function(e) { e.classList.add('sm-hl'); });
        // Also highlight trajectory edge
        document.querySelectorAll('.traj-edge[data-src="'+cid+'"][data-dst="'+nextCid+'"]')
          .forEach(function(e) { e.classList.add('sm-hl'); });
      }
      if (nextGid !== undefined && nextGid !== gid) {
        document.querySelectorAll('.sm-macro-edge[data-src-group="'+gid+'"][data-dst-group="'+nextGid+'"]')
          .forEach(function(e) { e.classList.add('sm-hl'); });
      }
    }
    // Legend highlight
    if (cid !== undefined) {
      var li = document.querySelector('.legend-item[data-cluster="'+cid+'"]');
      if (li) li.classList.add('highlight');
    }
  });
  sp.addEventListener('mouseleave', () => {
    unhighlight(u);
    hideTip();
    document.querySelectorAll('.sm-hl').forEach(function(el) { el.classList.remove('sm-hl'); });
    document.querySelectorAll('.legend-item.highlight').forEach(function(el) { el.classList.remove('highlight'); });
  });
});

// UMAP circles → text
document.querySelectorAll('.umap-point').forEach(circle => {
  const u = parseInt(circle.dataset.idx);
  circle.addEventListener('mouseenter', ev => {
    highlight(u);
    showTip(ev, circle.dataset.info);
    const [s] = tokenRangeForUmap(u);
    const sp = document.getElementById('tok-' + s);
    if (sp) sp.scrollIntoView({behavior:'smooth', block:'center'});
  });
  circle.addEventListener('mouseleave', () => { unhighlight(u); hideTip(); });
});

// --- Legend click-to-filter ---
var activeCluster = null, activeGroup = null;
function resetFilter() {
  document.querySelectorAll('.token[data-umap]').forEach(function(el) {
    el.style.background = el.dataset.origBg;
    el.style.color = el.dataset.origColor;
  });
  document.querySelectorAll('.legend-item.active').forEach(function(el) { el.classList.remove('active'); });
  document.querySelectorAll('.legend-group.active').forEach(function(el) { el.classList.remove('active'); });
  activeCluster = null;
  activeGroup = null;
}
function filterByCluster(cid) {
  if (activeCluster === cid) { resetFilter(); return; }
  resetFilter();
  activeCluster = cid;
  document.querySelectorAll('.token[data-umap]').forEach(function(el) {
    if (el.dataset.cluster !== String(cid)) {
      el.style.background = 'transparent';
      el.style.color = '#ccc';
    }
  });
  var li = document.querySelector('.legend-item[data-cluster="'+cid+'"]');
  if (li) li.classList.add('active');
}
function filterByGroup(gid) {
  if (activeGroup === gid) { resetFilter(); return; }
  resetFilter();
  activeGroup = gid;
  document.querySelectorAll('.token[data-umap]').forEach(function(el) {
    if (el.dataset.group !== String(gid)) {
      el.style.background = 'transparent';
      el.style.color = '#ccc';
    }
  });
  var lg = document.querySelector('.legend-group[data-group="'+gid+'"]');
  if (lg) lg.classList.add('active');
}
document.querySelectorAll('.legend-item[data-cluster]').forEach(function(el) {
  el.addEventListener('click', function(e) {
    e.stopPropagation();
    filterByCluster(parseInt(el.dataset.cluster));
  });
});
document.querySelectorAll('.legend-group[data-group]').forEach(function(el) {
  el.querySelector('b').addEventListener('click', function(e) {
    e.stopPropagation();
    filterByGroup(parseInt(el.dataset.group));
  });
});

// --- Legend sub-tabs (Legend / Names) ---
document.querySelectorAll('.legend-tab-btn').forEach(function(btn) {
  btn.addEventListener('click', function() {
    document.querySelectorAll('.legend-tab-btn').forEach(function(b) { b.classList.remove('active'); });
    document.querySelectorAll('.legend-tab-content').forEach(function(c) { c.classList.remove('active'); });
    btn.classList.add('active');
    var target = document.getElementById('ltab-' + btn.dataset.ltab);
    if (target) target.classList.add('active');
  });
});

// --- Legend hover: show cluster/group names ---
document.querySelectorAll('.legend-item[data-cluster]').forEach(function(el) {
  el.addEventListener('mouseenter', function(ev) {
    var cid = el.dataset.cluster;
    if (CLUSTER_NAMES[cid]) {
      showTip(ev, 'C' + cid + ': ' + CLUSTER_NAMES[cid].name + '\\n' + CLUSTER_NAMES[cid].desc);
    }
  });
  el.addEventListener('mouseleave', hideTip);
});
document.querySelectorAll('.legend-group[data-group]').forEach(function(el) {
  el.addEventListener('mouseenter', function(ev) {
    var gid = el.dataset.group;
    if (GROUP_NAMES[gid]) {
      showTip(ev, 'G' + gid + ': ' + GROUP_NAMES[gid].name + '\\n' + GROUP_NAMES[gid].desc);
    }
  });
  el.addEventListener('mouseleave', hideTip);
});

// --- SM node/bubble hover → text ---
document.querySelectorAll('.sm-node').forEach(function(node) {
  node.addEventListener('mouseenter', function(ev) {
    var cid = node.dataset.cluster;
    node.classList.add('sm-hl');
    document.querySelectorAll('.sm-micro-edge[data-src="'+cid+'"]').forEach(function(e) { e.classList.add('sm-hl'); });
    document.querySelectorAll('.token[data-umap]').forEach(function(el) {
      if (el.dataset.cluster !== cid) el.classList.add('sm-faded');
    });
    if (CLUSTER_NAMES[cid]) {
      showTip(ev, 'C' + cid + ': ' + CLUSTER_NAMES[cid].name + '\\n' + CLUSTER_NAMES[cid].desc);
    }
  });
  node.addEventListener('mouseleave', function() {
    document.querySelectorAll('.sm-hl').forEach(function(el) { el.classList.remove('sm-hl'); });
    document.querySelectorAll('.token.sm-faded').forEach(function(el) { el.classList.remove('sm-faded'); });
    hideTip();
  });
});
document.querySelectorAll('.sm-bubble').forEach(function(bubble) {
  bubble.addEventListener('mouseenter', function(ev) {
    var gid = bubble.dataset.group;
    bubble.classList.add('sm-hl');
    document.querySelectorAll('.token[data-umap]').forEach(function(el) {
      if (el.dataset.group !== gid) el.classList.add('sm-faded');
    });
    if (GROUP_NAMES[gid]) {
      showTip(ev, 'G' + gid + ': ' + GROUP_NAMES[gid].name + '\\n' + GROUP_NAMES[gid].desc);
    }
  });
  bubble.addEventListener('mouseleave', function() {
    document.querySelectorAll('.sm-hl').forEach(function(el) { el.classList.remove('sm-hl'); });
    document.querySelectorAll('.token.sm-faded').forEach(function(el) { el.classList.remove('sm-faded'); });
    hideTip();
  });
});

// --- SM pruning controls ---
function applySmPruning() {
  var microTopR = parseInt(document.getElementById('sm-micro-top-r').value);
  var microMinProb = parseFloat(document.getElementById('sm-micro-min-prob').value);
  var macroTopR = parseInt(document.getElementById('sm-macro-top-r').value);
  var macroMinProb = parseFloat(document.getElementById('sm-macro-min-prob').value);
  document.getElementById('sm-micro-top-r-val').textContent = microTopR;
  document.getElementById('sm-micro-min-prob-val').textContent = microMinProb.toFixed(2);
  document.getElementById('sm-macro-top-r-val').textContent = macroTopR;
  document.getElementById('sm-macro-min-prob-val').textContent = macroMinProb.toFixed(2);
  document.querySelectorAll('.sm-micro-edge').forEach(function(el) {
    var rank = parseInt(el.dataset.rank);
    var prob = parseFloat(el.dataset.prob);
    el.setAttribute('display', (rank <= microTopR && prob >= microMinProb) ? 'inline' : 'none');
  });
  document.querySelectorAll('.sm-macro-edge, .sm-macro-label').forEach(function(el) {
    var rank = parseInt(el.dataset.rank);
    var prob = parseFloat(el.dataset.prob);
    el.setAttribute('display', (rank <= macroTopR && prob >= macroMinProb) ? 'inline' : 'none');
  });
}
['sm-micro-top-r','sm-micro-min-prob','sm-macro-top-r','sm-macro-min-prob'].forEach(function(id) {
  var el = document.getElementById(id);
  if (el) el.addEventListener('input', applySmPruning);
});

// --- Trajectory pruning controls ---
function applyTrajPruning() {
  var minCount = parseInt(document.getElementById('traj-min-count').value);
  document.getElementById('traj-min-count-val').textContent = minCount;
  document.querySelectorAll('.traj-edge').forEach(function(el) {
    el.setAttribute('display', parseInt(el.dataset.count) >= minCount ? 'inline' : 'none');
  });
  document.querySelectorAll('.traj-label').forEach(function(el) {
    el.setAttribute('display', parseInt(el.dataset.count) >= minCount ? 'inline' : 'none');
  });
}
var trajSlider = document.getElementById('traj-min-count');
if (trajSlider) trajSlider.addEventListener('input', applyTrajPruning);
// Set max from SVG data attribute
var trajSvg = document.querySelector('#tab-traj .sm-svg');
if (trajSvg && trajSlider) {
  var mc = parseInt(trajSvg.dataset.maxCount) || 10;
  trajSlider.max = Math.max(mc, 2);
}

// --- Live Trajectory ---
(function() {
  if (typeof LIVE_SEQ === 'undefined') return;
  var edgesG = document.getElementById('live-edges');
  if (!edgesG) return;
  var posSlider = document.getElementById('live-pos-slider');
  var wSlider = document.getElementById('live-w-slider');
  var playBtn = document.getElementById('live-play-btn');
  var speedSlider = document.getElementById('live-speed');
  var posVal = document.getElementById('live-pos-val');
  var wVal = document.getElementById('live-w-val');
  var speedVal = document.getElementById('live-speed-val');
  var infoBar = document.getElementById('live-info');
  var playTimer = null;

  function liveBezier(srcId, dstId) {
    var p1 = LIVE_NODE_POS[srcId], p2 = LIVE_NODE_POS[dstId];
    if (!p1 || !p2) return null;
    var dx = p2[0]-p1[0], dy = p2[1]-p1[1];
    var len = Math.sqrt(dx*dx + dy*dy);
    if (len < 1) return null;
    var ux = dx/len, uy = dy/len;
    var r1 = LIVE_NODE_SIZES[srcId] || LIVE_NODE_R;
    var r2 = LIVE_NODE_SIZES[dstId] || LIVE_NODE_R;
    var x1 = p1[0]+ux*r1, y1 = p1[1]+uy*r1;
    var x2 = p2[0]-ux*r2, y2 = p2[1]-uy*r2;
    var nx = -dy/len, ny = dx/len;
    var off = 0.12 * len;
    var qx = (x1+x2)/2 + nx*off, qy = (y1+y2)/2 + ny*off;
    return 'M '+x1.toFixed(1)+' '+y1.toFixed(1)+' Q '+qx.toFixed(1)+' '+qy.toFixed(1)+' '+x2.toFixed(1)+' '+y2.toFixed(1);
  }

  function updateLive() {
    var pos = parseInt(posSlider.value);
    var w = parseInt(wSlider.value);
    posVal.textContent = pos;
    wVal.textContent = w;

    // 1. Clear previous edges
    while (edgesG.firstChild) edgesG.removeChild(edgesG.firstChild);

    // 2. Remove previous token highlights
    document.querySelectorAll('.token.live-current').forEach(function(el) { el.classList.remove('live-current'); });
    document.querySelectorAll('.token.live-window').forEach(function(el) { el.classList.remove('live-window'); });

    // 3. Reset all node strokes to defaults
    document.querySelectorAll('#tab-live .sm-node').forEach(function(nd) {
      nd.setAttribute('stroke', nd.dataset.defaultStroke);
      nd.setAttribute('stroke-width', nd.dataset.defaultSw);
      nd.removeAttribute('filter');
    });

    if (LIVE_SEQ.length === 0) return;

    // 4. Current segment
    var cur = LIVE_SEQ[pos];
    var curCluster = cur[0], curStart = cur[1], curEnd = cur[2];

    // 5. Highlight current node with gold glow
    var curNode = document.getElementById('live-node-' + curCluster);
    if (curNode) {
      curNode.setAttribute('stroke', 'gold');
      curNode.setAttribute('stroke-width', '3');
      curNode.setAttribute('filter', 'url(#live-glow)');
    }

    // 6. Highlight text tokens
    for (var t = curStart; t <= curEnd; t++) {
      var tok = document.getElementById('tok-' + t);
      if (tok) tok.classList.add('live-current');
    }
    // Window segments
    var wStart = Math.max(0, pos - w);
    var wEnd = Math.min(LIVE_SEQ.length - 1, pos + w);
    for (var si = wStart; si <= wEnd; si++) {
      if (si === pos) continue;
      var seg = LIVE_SEQ[si];
      for (var t2 = seg[1]; t2 <= seg[2]; t2++) {
        var tok2 = document.getElementById('tok-' + t2);
        if (tok2) tok2.classList.add('live-window');
      }
    }
    // Scroll current into view
    var firstTok = document.getElementById('tok-' + curStart);
    if (firstTok) firstTok.scrollIntoView({behavior:'smooth', block:'center'});

    // 7. Draw transition arrows
    var iStart = Math.max(0, pos - w);
    var iEnd = Math.min(LIVE_SEQ.length - 2, pos + w - 1);
    for (var i = iStart; i <= iEnd; i++) {
      var src = String(LIVE_SEQ[i][0]);
      var dst = String(LIVE_SEQ[i+1][0]);
      if (src === dst) continue;
      var d = liveBezier(src, dst);
      if (!d) continue;

      var isPast = (i < pos);
      var dist = Math.abs(i - pos);
      var maxDist = w;
      var opacity = 0.3 + 0.7 * (1 - dist / Math.max(maxDist, 1));

      var color, marker;
      if (isPast) {
        // Blue gradient: darker = further in past
        var blue_l = 40 + 30 * (1 - dist / Math.max(maxDist, 1));
        color = 'hsl(210, 60%, ' + blue_l.toFixed(0) + '%)';
        marker = 'url(#live-arrow-past)';
      } else {
        // Orange gradient: darker = further in future
        var orange_l = 40 + 30 * (1 - dist / Math.max(maxDist, 1));
        color = 'hsl(30, 80%, ' + orange_l.toFixed(0) + '%)';
        marker = 'url(#live-arrow-future)';
      }

      var path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
      path.setAttribute('d', d);
      path.setAttribute('stroke', color);
      path.setAttribute('stroke-width', '2');
      path.setAttribute('stroke-opacity', opacity.toFixed(2));
      path.setAttribute('fill', 'none');
      path.setAttribute('marker-end', marker);
      edgesG.appendChild(path);
    }

    // 8. Update info bar
    var gid = LIVE_GL[String(curCluster)];
    var cName = CLUSTER_NAMES[String(curCluster)] ? CLUSTER_NAMES[String(curCluster)].name : '';
    var gName = GROUP_NAMES[String(gid)] ? GROUP_NAMES[String(gid)].name : '';
    infoBar.textContent = 'Cluster: C' + curCluster + (cName ? ' (' + cName + ')' : '') +
      ' | Group: G' + gid + (gName ? ' (' + gName + ')' : '') +
      ' | Tokens: ' + curStart + '-' + curEnd +
      ' | Segment: ' + (pos+1) + '/' + LIVE_SEQ.length;
  }

  posSlider.addEventListener('input', updateLive);
  wSlider.addEventListener('input', updateLive);
  speedSlider.addEventListener('input', function() {
    speedVal.textContent = speedSlider.value;
    if (playTimer !== null) {
      clearInterval(playTimer);
      playTimer = setInterval(advance, parseInt(speedSlider.value));
    }
  });

  function advance() {
    var v = parseInt(posSlider.value);
    if (v >= parseInt(posSlider.max)) {
      clearInterval(playTimer);
      playTimer = null;
      playBtn.textContent = '\\u25B6 Play';
      return;
    }
    posSlider.value = v + 1;
    updateLive();
  }

  playBtn.addEventListener('click', function() {
    if (playTimer !== null) {
      clearInterval(playTimer);
      playTimer = null;
      playBtn.textContent = '\\u25B6 Play';
    } else {
      if (parseInt(posSlider.value) >= parseInt(posSlider.max)) posSlider.value = 0;
      playTimer = setInterval(advance, parseInt(speedSlider.value));
      playBtn.textContent = '\\u23F8 Pause';
    }
  });

  // Initialize on first tab switch
  document.querySelectorAll('.tab-btn[data-tab="live"]').forEach(function(btn) {
    btn.addEventListener('click', function() { setTimeout(updateLive, 50); });
  });
})();
""")
    h.append("</script></body></html>")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(h), encoding="utf-8")
    print(f"Saved {out_path}")
