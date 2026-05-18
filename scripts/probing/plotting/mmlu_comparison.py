#!/usr/bin/env python
"""Paper-ready MMLU t-SNE comparison: RET vs h / h_pca128 / sae baselines.

Loads the cached per-token embeddings produced by scripts/tsne_mmlu.py for
each method, subsamples the same tokens across methods (seed=42), runs t-SNE
per method, and produces comparison figures with a single shared legend.

Outputs to Paper_figures_2/mmlu_analysis/:
  mmlu_semantic_7way_comparison.png
  mmlu_semantic_4way_comparison.png
  mmlu_contextual_comparison.png
  mmlu_syntactic_comparison.png
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import spacy
from sklearn.manifold import TSNE
from transformers import AutoTokenizer

from evaluation.datasets.mmlu.data import (
    POS_MERGE,
    SUBJECT_TO_GROUP_4WAY,
    SUBJECT_TO_GROUP_7WAY,
    format_question,
)
from scripts.probing.plotting.tsne_mmlu import _load_mmlu_subsampled

ROOT = Path(__file__).resolve().parents[3]  # repo root (scripts/probing/plotting/ -> 3 up)
OUT_DIR = ROOT / "Paper_figures_2" / "mmlu_analysis"

# Baseline methods — shared across RET variants
BASELINE_METHODS = [
    ("h (768)",        ROOT / "outputs/Pythia160M/pile/baselines/h/analysis/mmlu_embeddings_hh.npz"),
    ("h-PCA128",       ROOT / "outputs/Pythia160M/pile/baselines/h_pca128/analysis/mmlu_embeddings_pca_128.npz"),
    ("SAE (32k, k=32)", ROOT / "outputs/Pythia160M/pile/baselines/sae/analysis/mmlu_embeddings_sae.npz"),
]
RET_VARIANTS = {
    "z128": ("RET z=128", ROOT / "outputs/Pythia160M/pile/ret_autonomous/analysis/mmlu_embeddings.npz"),
    "z256": ("RET z=256", ROOT / "outputs/Pythia160M/pile/ret_dense_autonomous_z256/analysis/mmlu_embeddings.npz"),
}

N_TSNE = 5_000
SEED = 42
DPI = 300

# Extraction settings — must match scripts/tsne_mmlu.py defaults
MAX_Q_PER_SUBJECT = 100
MAX_LENGTH = 256
BATCH_SIZE = 32
LLM_TOKENIZER = "EleutherAI/pythia-160m"


def _load(path: Path) -> dict:
    npz = np.load(path, allow_pickle=True)
    return {
        "z": npz["z_all"],
        "subjects": list(npz["subjects"]),
        "q_indices": npz["q_indices"],
        "pos_tags": list(npz["pos_tags"]),
    }


def _subsample_indices(n_total: int) -> np.ndarray:
    rng = np.random.default_rng(SEED)
    idx = rng.choice(n_total, size=min(N_TSNE, n_total), replace=False)
    idx.sort()
    return idx


def _tsne(z: np.ndarray) -> np.ndarray:
    print(f"  t-SNE on {z.shape[0]} tokens (d={z.shape[1]}) ...")
    return TSNE(n_components=2, perplexity=30, random_state=SEED, init="pca").fit_transform(z)


def _build_raw_pos_tags() -> list[str]:
    """Regenerate raw (unmerged) POS tags in the exact iteration order used
    by scripts/tsne_mmlu.py ``extract_embeddings`` / ``extract_mmlu_h``.

    Returns one tag per cached token (same length as cached z_all).
    """
    print("Re-tagging MMLU tokens (raw POS, no merge) ...")
    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
    tokenizer = AutoTokenizer.from_pretrained(LLM_TOKENIZER)
    tokenizer.pad_token = tokenizer.eos_token

    dataset = _load_mmlu_subsampled(MAX_Q_PER_SUBJECT)

    raw_tags: list[str] = []
    n = len(dataset)
    for batch_start in range(0, n, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, n)
        batch_rows = dataset.select(range(batch_start, batch_end))
        texts = [format_question(row) for row in batch_rows]

        enc = tokenizer(
            texts, return_tensors="pt", truncation=True,
            max_length=MAX_LENGTH, padding="max_length",
            return_offsets_mapping=True,
        )
        offsets = enc["offset_mapping"]           # (B, T, 2)
        mask = enc["attention_mask"].numpy()       # (B, T)

        for i in range(len(batch_rows)):
            text = texts[i]
            doc = nlp(text)
            char_to_pos: dict[int, str] = {}
            for tok in doc:
                for k in range(tok.idx, tok.idx + len(tok.text)):
                    char_to_pos[k] = tok.pos_

            offs_i = offsets[i].tolist()
            for t, (s, e) in enumerate(offs_i):
                if mask[i, t] == 0:
                    continue
                if s == e:            # SPEC token (same filter as extraction)
                    continue
                raw = char_to_pos.get(int(s), "X")
                merged = POS_MERGE.get(raw, raw)
                if merged == "SPEC":  # extraction also skipped these
                    continue
                raw_tags.append(raw)
        print(f"  {batch_end}/{n} questions, {len(raw_tags)} tokens", end="\r")
    print()
    return raw_tags


def _colors_categorical(labels: list[str], categories: list[str], palette_name: str):
    cmap = plt.cm.get_cmap(palette_name, len(categories))
    cat_to_color = {c: cmap(i) for i, c in enumerate(categories)}
    return [cat_to_color[l] for l in labels], cat_to_color


def _plot_combined(
    coords: list[np.ndarray],
    colors_top: list,
    colors_bot: list,
    handles_top: list,
    handles_bot: list,
    col_titles: list[str],
    row_labels: tuple[str, str],
    out_path: Path,
):
    """2-row figure: semantic (top) and syntactic (bottom) across 4 methods.

    Each row gets a y-label on the leftmost panel and a shared legend on
    the right. Column titles appear above the top row only.
    """
    n = len(coords)
    fig, axes = plt.subplots(2, n, figsize=(4.4 * n, 6.8))

    for j in range(n):
        # Top row — semantic
        ax = axes[0, j]
        xy = coords[j]
        ax.scatter(xy[:, 0], xy[:, 1], c=colors_top, s=6.4, alpha=0.7, linewidths=0)
        ax.set_title(col_titles[j], fontsize=21)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.6)

        # Bottom row — syntactic
        ax = axes[1, j]
        ax.scatter(xy[:, 0], xy[:, 1], c=colors_bot, s=6.4, alpha=0.7, linewidths=0)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.6)

    # Row labels (rotated) on the leftmost panels
    axes[0, 0].set_ylabel(row_labels[0], fontsize=18, labelpad=10)
    axes[1, 0].set_ylabel(row_labels[1], fontsize=18, labelpad=10)

    fig.tight_layout(rect=[0, 0, 0.86, 1.0])

    # Shared legends, one per row, anchored to the right of each row's
    # rightmost axes (so y=0.5 is exactly the vertical center of that row).
    fig.legend(
        handles=handles_top,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        bbox_transform=axes[0, -1].transAxes,
        frameon=True,
        fontsize=9,
        title="Subject group",
        title_fontsize=10,
    )
    fig.legend(
        handles=handles_bot,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        bbox_transform=axes[1, -1].transAxes,
        frameon=True,
        fontsize=9,
        title="POS tag",
        title_fontsize=10,
    )

    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    pdf_path = out_path.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")
    print(f"Saved {pdf_path}")


def _plot_comparison(
    coords_per_method: list[np.ndarray],
    method_names: list[str],
    colors_per_method: list[list],
    title: str,
    out_path: Path,
    legend_handles: list | None,
    legend_ncol: int = 1,
    legend_fontsize: int = 9,
):
    n = len(method_names)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.6))
    for ax, xy, colors, name in zip(axes, coords_per_method, colors_per_method, method_names):
        ax.scatter(xy[:, 0], xy[:, 1], c=colors, s=3.5, alpha=0.7, linewidths=0)
        ax.set_title(name, fontsize=12)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.6)

    if legend_handles is not None:
        fig.legend(
            handles=legend_handles,
            loc="center right",
            bbox_to_anchor=(1.0, 0.5),
            frameon=True,
            fontsize=legend_fontsize,
            ncol=legend_ncol,
        )
        fig.suptitle(title, fontsize=13, y=1.02)
        fig.tight_layout(rect=[0, 0, 0.88, 1.0])
    else:
        fig.suptitle(title, fontsize=13, y=1.02)
        fig.tight_layout()

    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ret-variant", choices=list(RET_VARIANTS.keys()), default="z128",
        help="Which RET checkpoint to use as the rightmost column",
    )
    parser.add_argument(
        "--suffix", default=None,
        help="Filename suffix for outputs (default: empty for z128, '_z256' for z256)",
    )
    args = parser.parse_args()

    ret_name, ret_cache = RET_VARIANTS[args.ret_variant]
    suffix = args.suffix if args.suffix is not None else (
        "" if args.ret_variant == "z128" else f"_{args.ret_variant}"
    )
    methods = BASELINE_METHODS + [(ret_name, ret_cache)]

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"RET variant: {args.ret_variant} -> suffix '{suffix}'")
    print("Loading cached embeddings ...")
    data = []
    for name, path in methods:
        print(f"  {name}: {path}")
        data.append((name, _load(path)))

    # --- Verify token identity across methods ---
    ref = data[0][1]
    n_tokens = len(ref["subjects"])
    for name, d in data[1:]:
        assert len(d["subjects"]) == n_tokens, f"{name}: token count mismatch"
        assert ref["subjects"] == d["subjects"], f"{name}: subjects differ"
        assert np.array_equal(ref["q_indices"], d["q_indices"]), f"{name}: q_indices differ"
        assert ref["pos_tags"] == d["pos_tags"], f"{name}: pos_tags differ"
    print(f"Token identity verified across {len(data)} methods ({n_tokens} tokens).")

    # --- Regenerate raw POS tags, filter out X (BPE alignment artifacts) ---
    raw_tags = _build_raw_pos_tags()
    assert len(raw_tags) == n_tokens, (
        f"Raw POS tag count ({len(raw_tags)}) != cached token count ({n_tokens})."
    )
    # Sanity: merged raw tags should equal cached pos_tags
    merged = [POS_MERGE.get(t, t) for t in raw_tags]
    mismatches = sum(1 for a, b in zip(merged, ref["pos_tags"]) if a != b)
    print(f"Merged-tag sanity check: {mismatches}/{n_tokens} mismatches.")

    keep = np.array([t != "X" for t in raw_tags], dtype=bool)
    n_kept = int(keep.sum())
    print(f"Filtering out X tokens: keeping {n_kept}/{n_tokens} ({100*n_kept/n_tokens:.1f}%).")

    # Apply filter to cached data in-place
    filtered_pos = [merged[i] for i in range(n_tokens) if keep[i]]
    filtered_subjects = [ref["subjects"][i] for i in range(n_tokens) if keep[i]]
    filtered_q_idx = ref["q_indices"][keep]
    for _name, d in data:
        d["z"] = d["z"][keep]

    # --- Subsample same indices for all methods from filtered set ---
    idx = _subsample_indices(n_kept)
    subjects_sub = [filtered_subjects[i] for i in idx]
    q_idx_sub = filtered_q_idx[idx]
    pos_sub = [filtered_pos[i] for i in idx]

    # --- Run t-SNE per method ---
    coords = []
    names = []
    for name, d in data:
        print(f"\n{name}:")
        z_sub = d["z"][idx]
        # SAE is 32k-dim sparse; TSNE can handle but is slow. Keep as-is for fairness.
        coords.append(_tsne(z_sub.astype(np.float32)))
        names.append(name)

    # ------------------------------------------------------------------
    # Figure 1: semantic (7-way)
    # ------------------------------------------------------------------
    groups_7 = sorted(set(SUBJECT_TO_GROUP_7WAY.values()))
    labels_7 = [SUBJECT_TO_GROUP_7WAY.get(s, "Other") for s in subjects_sub]
    colors_7, map_7 = _colors_categorical(labels_7, groups_7, "tab10")
    handles_7 = [mpatches.Patch(color=map_7[c], label=c) for c in groups_7]
    _plot_comparison(
        coords, names, [colors_7] * len(names),
        title="Semantic structure on MMLU (7-way subject groups)",
        out_path=OUT_DIR / f"mmlu_semantic_7way_comparison{suffix}.png",
        legend_handles=handles_7, legend_ncol=1, legend_fontsize=9,
    )

    # ------------------------------------------------------------------
    # Figure 2: semantic (4-way)
    # ------------------------------------------------------------------
    groups_4 = sorted(set(SUBJECT_TO_GROUP_4WAY.values()))
    labels_4 = [SUBJECT_TO_GROUP_4WAY.get(s, "Other") for s in subjects_sub]
    colors_4, map_4 = _colors_categorical(labels_4, groups_4, "tab10")
    handles_4 = [mpatches.Patch(color=map_4[c], label=c) for c in groups_4]
    _plot_comparison(
        coords, names, [colors_4] * len(names),
        title="Semantic structure on MMLU (4-way subject groups)",
        out_path=OUT_DIR / f"mmlu_semantic_4way_comparison{suffix}.png",
        legend_handles=handles_4, legend_ncol=1, legend_fontsize=9,
    )

    # ------------------------------------------------------------------
    # Figure 3: contextual (question index, cyclic HSV)
    # ------------------------------------------------------------------
    n_q = int(q_idx_sub.max()) + 1
    ctx_colors = plt.cm.hsv(q_idx_sub / max(n_q, 1))
    _plot_comparison(
        coords, names, [ctx_colors] * len(names),
        title="Contextual structure on MMLU (color = question index)",
        out_path=OUT_DIR / f"mmlu_contextual_comparison{suffix}.png",
        legend_handles=None,
    )

    # ------------------------------------------------------------------
    # Figure 4: syntactic (POS tag)
    # ------------------------------------------------------------------
    pos_categories = sorted(set(pos_sub))
    palette = "tab20" if len(pos_categories) > 10 else "tab10"
    pos_colors, map_pos = _colors_categorical(pos_sub, pos_categories, palette)
    handles_pos = [mpatches.Patch(color=map_pos[c], label=c) for c in pos_categories]

    # ------------------------------------------------------------------
    # Figure 5: combined 2-row (semantic + syntactic) for the paper
    # ------------------------------------------------------------------
    col_titles = [
        "h (hidden states)",
        r"$h^{PCA}$ (top-128 PCs)",
        "SAE",
        "RET",
    ]
    _plot_combined(
        coords=coords,
        colors_top=colors_7,
        colors_bot=pos_colors,
        handles_top=handles_7,
        handles_bot=handles_pos,
        col_titles=col_titles,
        row_labels=("Semantic Structure", "Syntactic Structure"),
        out_path=OUT_DIR / f"mmlu_semantic_syntactic_combined{suffix}.png",
    )

    _plot_comparison(
        coords, names, [pos_colors] * len(names),
        title="Syntactic structure on MMLU (color = POS tag)",
        out_path=OUT_DIR / f"mmlu_syntactic_comparison{suffix}.png",
        legend_handles=handles_pos, legend_ncol=1, legend_fontsize=8,
    )

    print(f"\nAll figures saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
