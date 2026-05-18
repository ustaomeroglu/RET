#!/usr/bin/env python3
"""Per-story cosine-similarity heatmap with event-boundary overlays.

Reproduces the (a)/(b)-style panel of Figure 7 in Lubana et al.
(arxiv 2511.01836): for a story with abrupt event transitions, a good
latent code's pairwise cosine-similarity heatmap shows hierarchical block
structure aligned with event boundaries. Raw hidden states do not.

For each story (loaded from a JSON with per-event text strings so that
event boundaries are exact at the token level), the script produces a
side-by-side heatmap of:

  * h  — raw hidden states
  * z  — encoder output
  * ẑ  — predictor output

Baseline modes (--baseline-hh/sae/pca) emit a 2-column h vs baseline figure.

Usage:
    # Single checkpoint, all 5 default stories
    python scripts/event_boundary_structure.py outputs/.../checkpoint_final.pt

    # Subset of stories
    python scripts/event_boundary_structure.py outputs/.../checkpoint_final.pt \\
        --story-ids abrupt_01 abrupt_03

    # All checkpoints
    python scripts/event_boundary_structure.py --all-checkpoints

    # Baseline mode
    python scripts/event_boundary_structure.py --baseline-hh
    python scripts/event_boundary_structure.py --baseline-pca --pca-dim 128
"""

import argparse
import json
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea, VPacker
import numpy as np
import torch
import umap

# One distinct color per event in the header text (cycles if #events > 5).
EVENT_COLORS = ["tab:blue", "tab:red", "tab:green", "tab:purple", "tab:orange"]

# Reuse helpers from event_structure.py (co-located)
sys.path.insert(0, str(Path(__file__).parent))
from event_structure import (  # noqa: E402
    compute_z_zhat,
    load_text_sequences,
    pairwise_cosine_similarity,
    tortuosity,
    tortuosity_cosine,
)

from ret.utils.prompt_logger import PromptLogger  # noqa: E402
from evaluation.baseline_utils import (  # noqa: E402
    add_baseline_args,
    get_baseline_mode,
    setup_baseline,
    fit_pca_on_h,
    transform_h,
)
from ret.extraction.hooks import HiddenStateExtractor  # noqa: E402
from ret.utils.checkpointing import load_encoder  # noqa: E402
from ret.models.predictor import Predictor  # noqa: E402
from ret.models.predictor import TokenProjector  # noqa: E402
from evaluation.baselines.sae import load_sae_baseline, encode_sae  # noqa: E402


# ---------------------------------------------------------------------------
# Window-mean baseline + numeric diagnostics
# ---------------------------------------------------------------------------
def compute_windowed_mean(h: np.ndarray, W: int) -> np.ndarray:
    """Causal sliding-window mean of ``h`` along time.

    For position ``t``, returns ``mean(h[max(0, t-W+1) : t+1])``.
    Edge positions ``t < W-1`` average over ``t+1`` tokens (not ``W``).

    Args:
        h: ``(T, D)`` per-token features.
        W: window size (number of tokens to average over).
    Returns:
        ``(T, D)`` array of windowed means, dtype matching ``h``.
    """
    T, D = h.shape
    csum = np.cumsum(h, axis=0, dtype=np.float64)
    csum0 = np.concatenate([np.zeros((1, D), dtype=csum.dtype), csum], axis=0)
    t_idx = np.arange(T)
    denom = np.minimum(t_idx + 1, W).astype(np.float64)[:, None]
    out = (csum0[t_idx + 1] - csum0[np.maximum(t_idx + 1 - W, 0)]) / denom
    return out.astype(h.dtype)


def boundary_contrast(sim: np.ndarray, boundaries: list[int]) -> dict:
    """Within-event vs cross-event off-diagonal cosine similarity.

    ``boundaries`` is the list of cumulative event-end indices (last entry
    equals ``T``). Returns within-mean, cross-mean, and contrast = within − cross.
    """
    T = sim.shape[0]
    event_indices: list[list[int]] = []
    prev = 0
    for b in boundaries:
        event_indices.append(list(range(prev, min(int(b), T))))
        prev = int(b)

    within_vals = []
    for idx in event_indices:
        if len(idx) >= 2:
            within_vals.append(_mean_offdiag(sim[np.ix_(idx, idx)]))

    cross_vals = []
    for i in range(len(event_indices)):
        for j in range(i + 1, len(event_indices)):
            ai, aj = event_indices[i], event_indices[j]
            if not ai or not aj:
                continue
            cross_vals.append(float(np.nanmean(sim[np.ix_(ai, aj)])))

    within_mean = float(np.nanmean(within_vals)) if within_vals else float("nan")
    cross_mean = float(np.nanmean(cross_vals)) if cross_vals else float("nan")
    return {
        "within_event_mean": within_mean,
        "cross_event_mean": cross_mean,
        "contrast": within_mean - cross_mean,
    }


def boundary_sharpness(sim: np.ndarray, boundaries: list[int]) -> dict:
    """Local cosine drop across planted scene boundaries.

    For each interior boundary index ``b`` (``b`` is the first token of the
    next event), we compare two adjacent-token cosines:

      * ``within_step`` = sim(b-2, b-1)  — adjacent tokens *inside* prior event.
      * ``cross_step``  = sim(b-1, b)    — adjacent tokens straddling boundary.

    ``drop = within_step − cross_step`` quantifies how sharply similarity
    falls when crossing the planted boundary. A representation that smears
    across boundaries (e.g. windowed mean) shows ``drop ≈ 0`` because
    overlapping windows make both cosines nearly identical. RET should
    produce a large drop.
    """
    T = sim.shape[0]
    interior = [int(b) for b in boundaries[:-1]]
    drops, withins, crosses = [], [], []
    for b in interior:
        if 2 <= b < T:
            w = float(sim[b - 2, b - 1])
            c = float(sim[b - 1, b])
            withins.append(w)
            crosses.append(c)
            drops.append(w - c)
    if not drops:
        return {"drop": float("nan"), "within_step": float("nan"),
                "cross_step": float("nan"), "n_boundaries": 0}
    return {
        "within_step": float(np.mean(withins)),
        "cross_step": float(np.mean(crosses)),
        "drop": float(np.mean(drops)),
        "n_boundaries": len(drops),
    }


def and_context_score(
    sim: np.ndarray,
    highlight_positions: list[int],
    boundaries: list[int],
) -> dict:
    """Scene-vs-lexical alignment for repeated-word probe tokens.

    For each highlighted (e.g. "and") token:
      * scene_align = mean cosine sim to other tokens in the SAME event,
        excluding other highlighted tokens.
      * lex_align   = mean cosine sim to OTHER highlighted tokens (any event).

    Returns the mean of (scene_align − lex_align) over all highlighted tokens.
    Positive values indicate the representation places repeated words near
    their surrounding scene rather than near each other.
    """
    T = sim.shape[0]
    valid = [int(p) for p in highlight_positions if 0 <= int(p) < T]
    if len(valid) < 2:
        return {"score": float("nan"), "scene_align": float("nan"),
                "lex_align": float("nan"), "n_highlights": len(valid)}

    token_event = np.array([_event_of_token(t, boundaries) for t in range(T)])
    valid_set = set(valid)

    scene_aligns: list[float] = []
    lex_aligns: list[float] = []
    for p in valid:
        my_event = token_event[p]
        scene_idx = [i for i in range(T)
                     if i != p and token_event[i] == my_event and i not in valid_set]
        lex_idx = [i for i in valid if i != p]
        if scene_idx and lex_idx:
            scene_aligns.append(float(np.mean(sim[p, scene_idx])))
            lex_aligns.append(float(np.mean(sim[p, lex_idx])))

    if not scene_aligns:
        return {"score": float("nan"), "scene_align": float("nan"),
                "lex_align": float("nan"), "n_highlights": len(valid)}

    sa = float(np.mean(scene_aligns))
    la = float(np.mean(lex_aligns))
    return {"scene_align": sa, "lex_align": la, "score": sa - la,
            "n_highlights": len(valid)}


# ---------------------------------------------------------------------------
# Stimulus loading
# ---------------------------------------------------------------------------
def load_abrupt_change_stories(
    json_path: Path,
    tokenizer,
    story_ids: list[str] | None = None,
    prepend_bos: bool = True,
) -> list[dict]:
    """Load hand-crafted stories with ground-truth event boundaries.

    Two preprocessing fixes to avoid outlier tokens contaminating the
    cosine-similarity plots:

    * A leading space is prepended to every event after the first so BPE
      emits the natural mid-document form of each event's first token
      (e.g. ``" Without"``, not ``"Without"``). Without it, the no-space
      variant — which the model only sees at start-of-document during
      training — behaves as an outlier at every boundary.
    * When ``prepend_bos`` is True and the tokenizer exposes a BOS id, a
      single BOS token (for Pythia: ``<|endoftext|>``) is prepended so the
      first content token sits at position 1 with a natural preceding
      context (rather than being the naked start-of-byte-stream token,
      which Pythia also treats as an outlier). The BOS position is dropped
      before plotting — ``boundaries`` stays in content-token coordinates.

    Returns a list of dicts per story: id, title, text (joined), events,
    input_ids (includes BOS if enabled), tokens, boundaries (content-token
    coords), bos_offset (0 or 1).
    """
    with open(json_path) as f:
        raw = json.load(f)
    if story_ids is not None:
        want = set(story_ids)
        raw = [s for s in raw if s["id"] in want]
        missing = want - {s["id"] for s in raw}
        if missing:
            raise ValueError(f"Story ids not found in {json_path}: {sorted(missing)}")

    bos_id = tokenizer.bos_token_id if prepend_bos else None

    out: list[dict] = []
    for story in raw:
        events = story["events"]
        input_ids: list[int] = []
        if bos_id is not None:
            input_ids.append(bos_id)
        bos_offset = len(input_ids)  # 0 or 1

        boundaries: list[int] = []  # in content-token coords (exclude BOS)
        for i, ev in enumerate(events):
            prefix = " " if i > 0 else ""
            ids = tokenizer.encode(prefix + ev, add_special_tokens=False)
            input_ids.extend(ids)
            boundaries.append(len(input_ids) - bos_offset)
        tokens = [tokenizer.decode([tid]) for tid in input_ids]
        out.append({
            "id": story["id"],
            "title": story.get("title", story["id"]),
            "text": " ".join(events),
            "events": events,
            "input_ids": input_ids,
            "tokens": tokens,
            "boundaries": boundaries,
            "bos_offset": bos_offset,
        })
    return out


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def load_random_tinystories(
    tokenizer,
    num_stories: int,
    seed: int,
    min_tokens: int = 60,
    max_tokens: int = 150,
    split: str = "validation",
    prepend_bos: bool = True,
) -> list[dict]:
    """Sample ``num_stories`` random TinyStories validation entries.

    Returns dicts with the same schema as :func:`load_abrupt_change_stories`
    but with a single "event" (the whole story), so interior boundaries are
    empty. That signals downstream plotting to use viridis-by-position
    coloring instead of the blue/red/green event palette.
    """
    texts = load_text_sequences(
        hf_dataset="roneneldan/TinyStories",
        split=split,
        text_column="text",
        num_sequences=num_stories,
        tokenizer=tokenizer,
        min_tokens=min_tokens,
        max_tokens=max_tokens,
        seed=seed,
    )

    bos_id = tokenizer.bos_token_id if prepend_bos else None
    out: list[dict] = []
    for i, text in enumerate(texts):
        content_ids = tokenizer.encode(text.strip(), add_special_tokens=False)
        input_ids: list[int] = []
        if bos_id is not None:
            input_ids.append(bos_id)
        bos_offset = len(input_ids)
        input_ids.extend(content_ids)
        tokens = [tokenizer.decode([tid]) for tid in input_ids]
        out.append({
            "id": f"tinystory_{i:02d}",
            "title": f"TinyStory #{i:02d}",
            "text": text.strip(),
            "events": [text.strip()],
            "input_ids": input_ids,
            "tokens": tokens,
            "boundaries": [len(content_ids)],
            "bos_offset": bos_offset,
        })
    return out


def _add_event_header(
    fig, events: list[str], boundaries: list[int],
    highlight_token: str | None = None,
    width: int = 110, top_y: float = 0.965,
) -> None:
    """Render the full story as one wrapped paragraph with each event colored.

    Words from each event keep that event's color (cycled from
    ``EVENT_COLORS``). If ``highlight_token`` is set, every occurrence of
    that word is rendered inside a rounded rectangle so the token is visually
    marked in-place. Events are concatenated with a single space (as the
    model sees them), and the paragraph is wrapped at ``width`` characters
    at word boundaries — there are no artificial line breaks between events.
    """
    ht = highlight_token.lower() if highlight_token else None
    # Single-event stories (e.g. random TinyStories) don't get colored events —
    # render the whole paragraph in a neutral dark gray instead.
    single_event = len(events) <= 1

    # Build a flat list of (word, event_index) in reading order.
    word_evi: list[tuple[str, int]] = []
    for i, ev in enumerate(events):
        for word in ev.strip().split():
            word_evi.append((word, i))

    # Greedy word-level wrap to ``width`` characters.
    lines: list[list[tuple[str, int]]] = []
    cur: list[tuple[str, int]] = []
    cur_len = 0
    for word, evi in word_evi:
        add = len(word) + (1 if cur else 0)
        if cur and cur_len + add > width:
            lines.append(cur)
            cur = [(word, evi)]
            cur_len = len(word)
        else:
            cur.append((word, evi))
            cur_len += add
    if cur:
        lines.append(cur)

    # Each line becomes an HPacker of one TextArea per word. Highlighted
    # words get a bbox so the rectangle doesn't swallow surrounding spaces.
    # We measure word widths with TextPath so that we can stretch the inter-word
    # gap on each non-final line to produce justified (block-aligned) text.
    from matplotlib.font_manager import FontProperties
    from matplotlib.textpath import TextPath

    fontsize_pt = 24
    default_sep = 6

    def _word_width(word: str) -> float:
        fp = FontProperties(size=fontsize_pt, style="italic")
        return TextPath((0, 0), word, prop=fp).get_extents().width

    line_widths: list[list[float]] = [
        [_word_width(w) for w, _ in line] for line in lines
    ]
    # Target line ink + gap width: use the longest natural line as the target.
    natural_per_line = [sum(ws) for ws in line_widths]
    max_n_gaps = max((len(line) - 1 for line in lines), default=0)
    target_width = (max(natural_per_line, default=0.0)
                    + default_sep * max_n_gaps)

    vbox_children = []
    n_lines = len(lines)
    for line_idx, (line, widths) in enumerate(zip(lines, line_widths)):
        n_gaps = len(line) - 1
        is_last = (line_idx == n_lines - 1)
        if n_gaps == 0 or is_last:
            sep = default_sep
        else:
            sep = max(default_sep,
                      (target_width - sum(widths)) / n_gaps)

        hchildren = []
        for word, evi in line:
            color = "0.15" if single_event else EVENT_COLORS[evi % len(EVENT_COLORS)]
            textprops = dict(color=color, size=fontsize_pt, style="italic")
            if ht is not None and word.strip(".,!?;:'\"").lower() == ht:
                textprops["bbox"] = dict(
                    boxstyle="round,pad=0.2",
                    facecolor="khaki",
                    edgecolor="black",
                    linewidth=1.0,
                    alpha=0.7,
                )
            hchildren.append(TextArea(word, textprops=textprops))
        vbox_children.append(HPacker(
            children=hchildren, align="baseline", pad=0, sep=sep,
        ))

    vbox = VPacker(children=vbox_children, align="left", pad=0, sep=4)
    anchored = AnchoredOffsetbox(
        loc="upper center", child=vbox, pad=0, borderpad=0, frameon=False,
        bbox_to_anchor=(0.5, top_y), bbox_transform=fig.transFigure,
    )
    fig.add_artist(anchored)


def _count_header_lines(events: list[str], width: int) -> int:
    """Count wrapped lines the event header will produce (mirrors _add_event_header logic)."""
    words = [w for ev in events for w in ev.strip().split()]
    lines, cur_len = 0, 0
    for word in words:
        add = len(word) + (1 if cur_len > 0 else 0)
        if cur_len > 0 and cur_len + add > width:
            lines += 1
            cur_len = len(word)
        else:
            cur_len += add
    if cur_len > 0:
        lines += 1
    return lines


def _event_of_token(t: int, boundaries: list[int]) -> int:
    """Return the event index that token ``t`` belongs to (given cumulative
    event-end boundaries)."""
    for i, b in enumerate(boundaries):
        if t < b:
            return i
    return len(boundaries) - 1


def _mean_offdiag(sim: np.ndarray) -> float:
    """Mean of the off-diagonal entries of a (T, T) similarity matrix."""
    n = sim.shape[0]
    if n < 2:
        return float("nan")
    total = sim.sum() - n  # diagonal entries are 1 each
    return float(total / (n * (n - 1)))


def _subset_offdiag_mean(sim_full: np.ndarray, indices: list[int]) -> float:
    """Mean pairwise sim among ``indices`` drawn from a full (T, T) sim matrix."""
    if len(indices) < 2:
        return float("nan")
    sub = sim_full[np.ix_(indices, indices)]
    return _mean_offdiag(sub)


def plot_boundary_similarity(
    representations: list[tuple[str, np.ndarray, list[int]]],
    out_path: Path,
    story_id: str,
    title: str,
    n_events: int,
    events: list[str] | None = None,
    boundaries: list[int] | None = None,
    highlight_positions: list[int] | None = None,
    highlight_label: str | None = None,
    show_suptitle: bool = True,
    rect_top_cap: float = 0.92,
) -> None:
    """Per-story figure: event-text header + sim heatmap row + 2D UMAP row.

    Each element of ``representations`` is ``(label, X, interior_boundaries)``
    where ``X`` has shape ``(T, d)``. For each column, the top subplot is a
    mean-centered cosine-similarity heatmap with boundary lines overlaid, and
    the bottom subplot is a 2D UMAP of ``X`` with boundary-token positions
    marked and tortuosity annotated.
    """
    n_cols = len(representations)
    fig, axes = plt.subplots(2, n_cols, figsize=(6.4 * n_cols, 14.5))
    if n_cols == 1:
        axes = axes.reshape(2, 1)

    if show_suptitle:
        fig.suptitle(title, fontsize=26, y=0.995, fontweight="bold")

    if events is not None and boundaries is not None:
        # Wrap width tracks figure width: original 110 chars was tuned for 3 cols;
        # scale linearly with n_cols so the header spans the full figure for any
        # column count.
        header_width = max(110, round(110 * n_cols / 3))
        n_header_lines = _count_header_lines(events, width=header_width)
        # When the suptitle is hidden, the header can sit higher (no title bar
        # above it), so anchor the header's top closer to the figure's edge.
        header_top = 0.995 if not show_suptitle else 0.965
        _add_event_header(fig, events, boundaries,
                          highlight_token=highlight_label,
                          width=header_width, top_y=header_top)

    for col, (label, X, interior) in enumerate(representations):
        T = X.shape[0]

        # Compute similarity matrix once — reused for both UMAP-title stats
        # and the heatmap below. Uses GLOBAL mean-centering (paper convention).
        sim = pairwise_cosine_similarity(X)
        avg_sim_all = _mean_offdiag(sim)

        # Top: 2D UMAP with tortuosity and boundary markers
        ax_um = axes[0, col]
        n_neighbors = min(15, max(2, T - 1))
        reducer = umap.UMAP(
            n_components=2, n_neighbors=n_neighbors,
            min_dist=0.1, random_state=42,
        )
        coords = reducer.fit_transform(X)
        # Tortuosity computed in the original feature space (cosine distance),
        # not on the UMAP projection. UMAP non-linearly distorts global
        # distances, so geometric path-length ratios in 2D coords are not a
        # reliable smoothness metric across representations of different
        # dimensionality. Cosine distance is scale- and dim-invariant.
        tau = tortuosity_cosine(X)

        ax_um.plot(coords[:, 0], coords[:, 1],
                   color="gray", alpha=0.4, linewidth=0.8)
        # Use event-colored dots only when there are >1 events (interior
        # boundaries exist). Otherwise fall back to viridis-by-position so the
        # temporal progression is still visible.
        if boundaries is not None and interior:
            token_colors = [
                EVENT_COLORS[_event_of_token(t, boundaries) % len(EVENT_COLORS)]
                for t in range(T)
            ]
            ax_um.scatter(coords[:, 0], coords[:, 1],
                          c=token_colors, s=110,
                          edgecolors="k", linewidths=0.5)
        else:
            pos_color = np.arange(T) / max(T - 1, 1)
            ax_um.scatter(coords[:, 0], coords[:, 1],
                          c=pos_color, cmap="viridis",
                          s=110, edgecolors="k", linewidths=0.5)
        for b in interior:
            if 0 <= b < T:
                ax_um.scatter(coords[b, 0], coords[b, 1],
                              marker="s", s=360, facecolors="none",
                              edgecolors="red", linewidths=3.0,
                              label="event boundary" if b == interior[0] else None)

        if highlight_positions:
            hp = [p for p in highlight_positions if 0 <= p < T]
            if hp:
                ax_um.scatter(
                    coords[hp, 0], coords[hp, 1],
                    marker="o", s=520, facecolors="none",
                    edgecolors="black", linewidths=2.6, zorder=5,
                    label=f"‘{highlight_label}’" if highlight_label else "highlight",
                )

        ax_um.set_title(f"{label}   τ = {tau:.2f}", fontsize=26)
        ax_um.set_xticks([])
        ax_um.set_yticks([])
        ax_um.set_box_aspect(1)
        if interior:
            ax_um.legend(loc="best", fontsize=20, framealpha=0.8, markerscale=1.0)

        # Bottom: similarity heatmap with event-boundary lines (reuse sim)
        ax_hm = axes[1, col]
        im = ax_hm.imshow(sim, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
        for b in interior:
            ax_hm.axvline(b - 0.5, color="black", linewidth=1.2, alpha=0.85)
            ax_hm.axhline(b - 0.5, color="black", linewidth=1.2, alpha=0.85)
        ax_hm.set_xlabel("Token position", fontsize=22)
        if col == 0:
            ax_hm.set_ylabel("Token position", fontsize=22)
        else:
            ax_hm.set_ylabel("")
            ax_hm.tick_params(axis="y", labelleft=False)
        ax_hm.tick_params(axis="both", labelsize=18)
        ax_hm.set_box_aspect(1)

    # Reserve top slice of the figure for the event-text header and a thin
    # right-side strip for the heatmap colorbar. Lower rect_top when the
    # header is long (many wrapped lines) to prevent overlap with the plots.
    # LINE_HEIGHT_FRAC: ~(fontsize 24pt + sep 4pt) per line / (14.5in * 72pt/in)
    if events is not None and boundaries is not None:
        LINE_HEIGHT_FRAC = 0.028
        # When suptitle is hidden, the header sits higher (no title bar above
        # it), so the plot area can extend further up — shift the header anchor
        # up by ~0.03.
        header_anchor_top = 0.995 if not show_suptitle else 0.965
        rect_top = min(rect_top_cap,
                       header_anchor_top - n_header_lines * LINE_HEIGHT_FRAC - 0.005)
    else:
        rect_top = rect_top_cap
    plt.tight_layout(rect=[0, 0, 0.965, rect_top])
    # Equal, tight column and row gaps. The figure aspect was chosen so that
    # the gridspec slots are themselves nearly square — combined with
    # ``set_box_aspect(1)`` on every axis this keeps inter-panel whitespace
    # uniform horizontally and vertically.
    plt.subplots_adjust(wspace=0.04, hspace=0.04)
    # Place the shared colorbar manually next to the bottom (heatmap) row only,
    # without shrinking any of the plot axes.
    pos_bot_right = axes[1, -1].get_position()
    cax = fig.add_axes([
        pos_bot_right.x1 + 0.006,
        pos_bot_right.y0,
        0.013,
        pos_bot_right.height,
    ])
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=16)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    pdf_path = out_path.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}  and  {pdf_path.name}")


# ---------------------------------------------------------------------------
# Diagnostics dump
# ---------------------------------------------------------------------------
def _dump_diagnostics(
    out_path: Path,
    representations: list[tuple[str, np.ndarray, list[int]]],
    boundaries: list[int],
    highlight_positions: list[int] | None,
    story_meta: dict,
) -> None:
    """Compute and write per-representation cosine-space diagnostics to JSON.

    For each representation column we dump:
      * ``boundary_contrast`` — within-event vs cross-event off-diagonal
        cosine sim. Captures whether scene boundaries produce sharp blocks.
      * ``and_context_score`` — for repeated highlighted tokens (e.g. "and"),
        scene-vs-lexical alignment. Positive: token placed near its scene.
    """
    record_out: dict = {"meta": story_meta, "representations": []}
    for label, X, _interior in representations:
        sim = pairwise_cosine_similarity(X)
        bc = boundary_contrast(sim, boundaries)
        bs = boundary_sharpness(sim, boundaries)
        record: dict = {
            "label": label,
            "boundary_contrast": bc,
            "boundary_sharpness": bs,
        }
        if highlight_positions:
            record["and_context_score"] = and_context_score(
                sim, highlight_positions, boundaries,
            )
        record_out["representations"].append(record)
    with open(out_path, "w") as f:
        json.dump(record_out, f, indent=2)
    print(f"  Diagnostics → {out_path.name}")
    for rep in record_out["representations"]:
        bc = rep["boundary_contrast"]
        bs = rep["boundary_sharpness"]
        line = (f"    {rep['label']:35s}  within={bc['within_event_mean']:+.3f} "
                f"cross={bc['cross_event_mean']:+.3f} contrast={bc['contrast']:+.3f}"
                f"  drop={bs['drop']:+.3f}")
        if "and_context_score" in rep:
            ac = rep["and_context_score"]
            line += f"  and={ac['score']:+.3f}"
        print(line)


# ---------------------------------------------------------------------------
# Story text dump
# ---------------------------------------------------------------------------
def _write_story_texts(stories: list[dict], out_dir: Path) -> None:
    f_out = out_dir / "story_texts.txt"
    with open(f_out, "w") as f:
        for story in stories:
            f.write(f"{'=' * 72}\n")
            f.write(f"ID:       {story['id']}\n")
            f.write(f"Title:    {story['title']}\n")
            f.write(f"Tokens:   {len(story['input_ids'])}\n")
            f.write(f"Events:   {len(story['events'])}\n")
            f.write(f"Boundary: {story['boundaries']}  (cumulative token ends)\n")
            f.write(f"{'=' * 72}\n")
            for i, ev in enumerate(story["events"]):
                b_start = 0 if i == 0 else story["boundaries"][i - 1]
                b_end = story["boundaries"][i]
                f.write(f"[Event {i}  tokens {b_start}..{b_end}  ({b_end - b_start} tokens)]\n")
                f.write(ev + "\n\n")
    print(f"Saved story texts to {f_out}")


# ---------------------------------------------------------------------------
# Checkpoint processing
# ---------------------------------------------------------------------------
def _load_predictor_and_projector(config, ckpt, device: str):
    predictor = Predictor(config.encoder.z_dim, config.predictor).to(device)
    try:
        predictor.load_state_dict(ckpt["predictor"])
    except RuntimeError as e:
        import warnings
        warnings.warn(f"Predictor state_dict mismatch, loading with strict=False: {e}")
        info = predictor.load_state_dict(ckpt["predictor"], strict=False)
        if info.missing_keys:
            warnings.warn(f"  Missing keys (randomly initialized): {info.missing_keys}")
        if info.unexpected_keys:
            warnings.warn(f"  Unexpected keys (ignored): {info.unexpected_keys}")
    predictor.eval()

    token_projector = None
    if config.predictor.mode == "controlled" and "token_projector" in ckpt:
        token_projector = TokenProjector(config.llm.hidden_dim, config.predictor).to(device)
        token_projector.load_state_dict(ckpt["token_projector"])
        token_projector.eval()
    return predictor, token_projector


def _extract_h(extractor: HiddenStateExtractor, input_ids: list[int], device: str) -> np.ndarray:
    input_ids_t = torch.tensor([input_ids], device=device)
    with torch.no_grad():
        h = extractor.extract(input_ids_t)  # (1, T, d_h)
    return h[0].cpu().numpy()


def process_checkpoint(
    ckpt_path: Path,
    stories_json: Path,
    story_ids: list[str] | None,
    device: str,
    skip_existing: bool,
    highlight_token: str | None = None,
    random_tinystories: int | None = None,
    random_seed: int = 42,
    min_tokens: int = 60,
    max_tokens: int = 150,
    add_window_mean: bool = False,
    window_size: int = 8,
    output_suffix: str | None = None,
) -> None:
    """Run full event structure analysis for one checkpoint.

    Two modes:
      * ``random_tinystories=None`` (default): load the abrupt-change JSON and
        write to ``analysis_boundary/`` with ``boundary_sim_*`` files.
      * ``random_tinystories=N``: sample N random TinyStories validation
        stories, single-event, no blue/red/green coloring, write to
        ``analysis_natural/`` with ``natural_sim_*`` files.

    When ``add_window_mean=True``, an extra column ``h̄`` (sliding-window mean
    of ``h`` with window ``window_size``) is inserted between ``h`` and
    ``SAE``. ``output_suffix`` (if set) is appended to the output directory
    name (``analysis_boundary_<suffix>/``) and to each PNG/PDF filename so the
    new figures never overwrite the suffix-less originals.
    """
    ckpt_dir = ckpt_path.parent
    natural_mode = random_tinystories is not None
    base_dir = "analysis_natural" if natural_mode else "analysis_boundary"
    if output_suffix:
        base_dir = f"{base_dir}_{output_suffix}"
    out_dir = ckpt_dir / base_dir
    fname_prefix = "natural_sim" if natural_mode else "boundary_sim"
    fname_suffix = f"_{output_suffix}" if output_suffix else ""
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading config and checkpoint from {ckpt_dir} ...")
    encoder, config, ckpt = load_encoder(ckpt_path, device=device)

    predictor, token_projector = _load_predictor_and_projector(config, ckpt, device)

    extractor = HiddenStateExtractor(config.llm, device=device)

    sae, sae_label = None, None
    sae_result = load_sae_baseline(config.llm, device=device)
    if sae_result is not None:
        sae, sae_release, _ = sae_result
        sae_label = sae_release.split("/")[-1] if "/" in sae_release else sae_release

    if natural_mode:
        stories = load_random_tinystories(
            extractor.tokenizer,
            num_stories=random_tinystories,
            seed=random_seed,
            min_tokens=min_tokens, max_tokens=max_tokens,
        )
        print(f"Loaded {len(stories)} random TinyStories validation stories")
    else:
        stories = load_abrupt_change_stories(stories_json, extractor.tokenizer, story_ids)
        if stories:
            print(f"Loaded {len(stories)} stories from {stories_json}")
    if not stories:
        print("  No stories loaded; nothing to do.")
        extractor.cleanup()
        return

    if skip_existing:
        marker = out_dir / f"{fname_prefix}_{stories[0]['id']}{fname_suffix}.png"
        if marker.exists():
            print(f"  SKIPPED — analysis already exists ({marker})")
            extractor.cleanup()
            return

    _write_story_texts(stories, out_dir)

    prompt_logger = PromptLogger(out_dir)

    for story in stories:
        T = len(story["input_ids"])
        print(f"\n--- Story {story['id']} ({T} tokens, {len(story['events'])} events) ---")
        print(f"  Title: {story['title']}")

        if not prompt_logger.is_full:
            prompt_logger.log_prompts([story["text"]], extractor.tokenizer, source="document")

        h_np = _extract_h(extractor, story["input_ids"], device)

        seq = {
            "h": h_np,
            "input_ids": story["input_ids"],
            "tokens": story["tokens"],
            "text": story["text"],
        }
        z, _ = compute_z_zhat(
            seq, encoder, predictor, token_projector, extractor, device,
        )

        # Drop the BOS-prepend position (if any) so all reps align with
        # content tokens.
        b_off = story.get("bos_offset", 0)
        h_c = h_np[b_off:]
        z_c = z[b_off:]

        interior = story["boundaries"][:-1]
        representations: list[tuple[str, np.ndarray, list[int]]] = [
            ("h (hidden states)", h_c, interior),
        ]
        if add_window_mean:
            bar_h = compute_windowed_mean(h_c, window_size)
            representations.append(
                (r"$h^{\mathrm{pooled}}$", bar_h, interior)
            )
        if sae is not None:
            h_tensor = torch.from_numpy(h_c).to(device)
            sae_feats = encode_sae(sae, h_tensor, device=device).cpu().numpy()
            representations.append(
                ("SAE", sae_feats, interior)
            )
        representations.append(
            ("z (macro state)", z_c, interior)
        )
        highlight_positions = None
        if highlight_token is not None:
            target = highlight_token.strip().lower()
            highlight_positions = [
                i for i, t in enumerate(story["tokens"][b_off:])
                if t.strip().lower() == target
            ]

        plot_boundary_similarity(
            representations,
            out_dir / f"{fname_prefix}_{story['id']}{fname_suffix}.png",
            story_id=story["id"],
            title=story["title"],
            n_events=len(story["events"]),
            events=story["events"],
            boundaries=story["boundaries"],
            highlight_positions=highlight_positions,
            highlight_label=highlight_token,
        )

        if add_window_mean:
            _dump_diagnostics(
                out_dir / f"{fname_prefix}_{story['id']}{fname_suffix}_diagnostics.json",
                representations,
                story["boundaries"],
                highlight_positions,
                story_meta={
                    "id": story["id"],
                    "title": story["title"],
                    "n_events": len(story["events"]),
                    "n_tokens": int(h_c.shape[0]),
                    "window_size": int(window_size),
                    "highlight_token": highlight_token,
                },
            )

    prompt_logger.save()
    extractor.cleanup()
    print(f"\nDone. All plots saved to {out_dir}")


def process_baseline(
    ctx,
    stories_json: Path,
    story_ids: list[str] | None,
    device: str,
    skip_existing: bool,
    highlight_token: str | None = None,
    random_tinystories: int | None = None,
    random_seed: int = 42,
    min_tokens: int = 60,
    max_tokens: int = 150,
) -> None:
    """Run event-boundary analysis in baseline mode (no encoder/predictor)."""
    # Use a sibling analysis_boundary dir next to ctx.output_dir
    out_dir = ctx.output_dir.parent / "analysis_boundary"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Baseline mode requires an extractor even for h (transform_h needs it for SAE).
    # setup_baseline already gives us ctx.extractor (unless skip_extractor was passed).
    extractor = ctx.extractor
    if extractor is None:
        # We need an extractor here because hand-written stories aren't cached.
        extractor = HiddenStateExtractor(ctx.config.llm, device=device)

    stories = load_abrupt_change_stories(stories_json, extractor.tokenizer, story_ids)
    if not stories:
        print(f"  No stories loaded from {stories_json}; nothing to do.")
        return
    print(f"Loaded {len(stories)} stories from {stories_json}")

    if skip_existing:
        marker = out_dir / f"boundary_sim_{stories[0]['id']}.png"
        if marker.exists():
            print(f"  SKIPPED — analysis already exists ({marker})")
            return

    # For PCA baseline, fit on pooled h from all stories (content tokens only)
    if ctx.mode == "pca" and ctx.pca_components is None:
        print("Fitting PCA on pooled h from all stories ...")
        h_list = []
        for s in stories:
            h_full = _extract_h(extractor, s["input_ids"], device)
            h_list.append(h_full[s.get("bos_offset", 0):])
        h_pooled = np.concatenate(h_list, axis=0)
        comps, mean = fit_pca_on_h(h_pooled, ctx.pca_dim)
        ctx.pca_components = comps
        ctx.pca_mean = mean

    _write_story_texts(stories, out_dir)

    prompt_logger = PromptLogger(out_dir)

    for story in stories:
        T = len(story["input_ids"])
        print(f"\n--- Story {story['id']} ({T} tokens, {len(story['events'])} events) ---")
        print(f"  Title: {story['title']}")

        if not prompt_logger.is_full:
            prompt_logger.log_prompts([story["text"]], extractor.tokenizer, source="document")

        h_np = _extract_h(extractor, story["input_ids"], device)
        b_off = story.get("bos_offset", 0)
        h_c = h_np[b_off:]

        repr_tensor = transform_h(torch.from_numpy(h_c), ctx, device=device)
        repr_np = repr_tensor.cpu().numpy()

        interior = story["boundaries"][:-1]
        representations = [
            ("h (hidden states)", h_c, interior),
            (ctx.representation_label, repr_np, interior),
        ]
        highlight_positions = None
        if highlight_token is not None:
            target = highlight_token.strip().lower()
            highlight_positions = [
                i for i, t in enumerate(story["tokens"][b_off:])
                if t.strip().lower() == target
            ]

        plot_boundary_similarity(
            representations,
            out_dir / f"boundary_sim_{story['id']}.png",
            story_id=story["id"],
            title=story["title"],
            n_events=len(story["events"]),
            events=story["events"],
            boundaries=story["boundaries"],
            highlight_positions=highlight_positions,
            highlight_label=highlight_token,
        )

    prompt_logger.save()
    extractor.cleanup()
    print(f"\nDone. All plots saved to {out_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Per-story cosine-similarity heatmap with event-boundary overlays "
            "(h, z, ẑ side-by-side)."
        )
    )
    parser.add_argument(
        "checkpoint", type=str, nargs="?", default=None,
        help="Path to checkpoint_final.pt",
    )
    parser.add_argument(
        "--all-checkpoints", action="store_true",
        help="Process all checkpoint_final.pt under outputs/",
    )
    parser.add_argument(
        "--stories-json", type=str,
        default="data/abrupt_change_stories.json",
        help="Path to the hand-curated abrupt-change stories JSON.",
    )
    parser.add_argument(
        "--story-ids", type=str, nargs="*", default=None,
        help="Subset of story ids to analyze (default: all in the JSON).",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip checkpoints that already have boundary analysis.",
    )
    parser.add_argument(
        "--highlight-token", type=str, default=None,
        help=("Circle every occurrence of this word in each UMAP and box it "
              "in the header text (case-insensitive, leading spaces trimmed). "
              "Use to probe lexical vs contextual clustering "
              "(e.g. --highlight-token and)."),
    )
    parser.add_argument(
        "--random-tinystories", type=int, default=None,
        help=("Sample N random TinyStories validation stories instead of the "
              "abrupt-change JSON. Single-event, no blue/red/green coloring. "
              "Writes to analysis_natural/ with natural_sim_*.png files."),
    )
    parser.add_argument(
        "--random-seed", type=int, default=42,
        help="Seed for sampling random TinyStories.",
    )
    parser.add_argument(
        "--min-tokens", type=int, default=60,
        help="Minimum tokens per sampled story (only for --random-tinystories).",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=150,
        help="Maximum tokens per sampled story (only for --random-tinystories).",
    )
    parser.add_argument(
        "--add-window-mean", action=argparse.BooleanOptionalAction, default=True,
        help=("Insert an extra column for the causal sliding-window mean of h "
              "(h^pooled, window size via --window-size). On by default = the "
              "paper Fig 3 / 17-21 configuration; pass --no-add-window-mean to disable."),
    )
    parser.add_argument(
        "--window-size", type=int, default=4,
        help="Window size W for h^pooled (default: 4 = paper Fig 3 / 17-21).",
    )
    parser.add_argument(
        "--output-suffix", type=str, default=None,
        help=("If set, output dir becomes ``analysis_boundary_<suffix>/`` and "
              "every PNG/PDF gets ``_<suffix>`` before the extension. Use to "
              "avoid overwriting prior figures (e.g. --output-suffix winmean8)."),
    )
    add_baseline_args(parser)
    args = parser.parse_args()

    stories_json = Path(args.stories_json)
    if not stories_json.exists():
        raise SystemExit(f"ERROR: stories file not found: {stories_json}")

    is_baseline = any([
        getattr(args, "baseline_hh", False),
        getattr(args, "baseline_sae", False),
        getattr(args, "baseline_pca", False),
    ])

    common_kwargs = dict(
        stories_json=stories_json,
        story_ids=args.story_ids,
        device=args.device,
        skip_existing=args.skip_existing,
        highlight_token=args.highlight_token,
        random_tinystories=args.random_tinystories,
        random_seed=args.random_seed,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
    )
    ckpt_kwargs = dict(
        common_kwargs,
        add_window_mean=args.add_window_mean,
        window_size=args.window_size,
        output_suffix=args.output_suffix,
    )

    if args.all_checkpoints:
        if is_baseline:
            raise SystemExit("ERROR: --all-checkpoints cannot be combined with --baseline-* flags")
        import glob
        ckpt_paths = sorted(glob.glob("outputs/**/checkpoint_final.pt", recursive=True))
        if not ckpt_paths:
            print("ERROR: no checkpoint_final.pt files found under outputs/")
            return
        print(f"Found {len(ckpt_paths)} checkpoints")
        for p in ckpt_paths:
            print(f"\n{'=' * 60}")
            print(f"  {p}")
            print(f"{'=' * 60}")
            try:
                process_checkpoint(Path(p), **ckpt_kwargs)
            except Exception as e:
                import traceback
                print(f"  FAILED: {e}")
                traceback.print_exc()
    elif is_baseline:
        mode = get_baseline_mode(args)
        ctx = setup_baseline(mode, args, device=args.device)
        process_baseline(ctx, **common_kwargs)
    elif args.checkpoint is not None:
        process_checkpoint(Path(args.checkpoint), **ckpt_kwargs)
    else:
        parser.error(
            "Either provide a checkpoint path, use --all-checkpoints, "
            "or use a --baseline-* flag."
        )


if __name__ == "__main__":
    main()
