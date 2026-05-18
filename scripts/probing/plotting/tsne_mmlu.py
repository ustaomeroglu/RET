#!/usr/bin/env python3
"""MMLU t-SNE visualization: semantic, contextual, and syntactic coloring.

Three token-level t-SNE plots where each dot is one token from an MMLU
question, colored by:
  1. Semantic   — MMLU subject / broad group
  2. Contextual — question index (tokens from same question share a hue)
  3. Syntactic  — Universal POS tag (NOUN, VERB, ADJ, ...)

Usage:
    # All three plots, fine-grained subjects
    python scripts/tsne_mmlu.py outputs/transformer_autonomous/checkpoint_final.pt

    # All three plots, grouped subjects in semantic plot
    python scripts/tsne_mmlu.py outputs/transformer_autonomous/checkpoint_final.pt --group

    # Custom limits
    python scripts/tsne_mmlu.py checkpoint.pt --max-questions-per-subject 30 \\
        --max-tsne-tokens 8000 --device cuda
"""

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from sklearn.manifold import TSNE

import spacy

from ret.utils.checkpointing import load_encoder
from ret.utils.prompt_logger import PromptLogger
from evaluation.baseline_utils import (
    add_baseline_args,
    get_baseline_mode,
    setup_baseline,
    fit_pca_on_h,
    transform_h,
)
from ret.extraction.hooks import HiddenStateExtractor


from evaluation.datasets.mmlu.data import (
    SUBJECT_GROUPS_4WAY,
    SUBJECT_GROUPS_7WAY,
    SUBJECT_GROUPS,
    SUBJECT_TO_GROUP_4WAY,
    SUBJECT_TO_GROUP_7WAY,
    SUBJECT_TO_GROUP,
    POS_MERGE,
    format_question,
)


# ---------------------------------------------------------------------------
# POS tagging
# ---------------------------------------------------------------------------
def get_pos_tags(
    nlp,
    text: str,
    offset_mapping: list[tuple[int, int]],
) -> list[str]:
    """Map each token's character span to a Universal POS tag.

    Special tokens (start == end, e.g. padding/BOS/EOS) get tag "SPEC".
    """
    doc = nlp(text)
    char_to_pos: dict[int, str] = {}
    for tok in doc:
        for i in range(tok.idx, tok.idx + len(tok.text)):
            char_to_pos[i] = tok.pos_

    tags = []
    for start, end in offset_mapping:
        if start == end:
            tags.append("SPEC")
        else:
            raw = char_to_pos.get(start, "X")
            tags.append(POS_MERGE.get(raw, raw))
    return tags


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------
def extract_embeddings(
    dataset,
    extractor: HiddenStateExtractor,
    encoder: torch.nn.Module,
    nlp,
    max_length: int = 256,
    batch_size: int = 32,
    device: str = "cuda",
    prompt_logger: PromptLogger | None = None,
) -> tuple[np.ndarray, list[str], np.ndarray, list[str]]:
    """Tokenize, extract h, encode to z, collect per-token labels.

    Returns
    -------
    z_all     : np.ndarray (N_tokens, d_z)
    subjects  : list[str]  length N_tokens
    q_indices : np.ndarray (N_tokens,) int — original question index
    pos_tags  : list[str]  length N_tokens
    """
    tokenizer = extractor.tokenizer

    all_z: list[np.ndarray] = []
    all_subjects: list[str] = []
    all_q_idx: list[int] = []
    all_pos: list[str] = []

    n = len(dataset)

    for batch_start in range(0, n, batch_size):
        batch_end = min(batch_start + batch_size, n)
        batch_rows = dataset.select(range(batch_start, batch_end))

        texts = [format_question(row) for row in batch_rows]
        subjects_batch = [row["subject"] for row in batch_rows]
        q_indices_batch = list(range(batch_start, batch_end))

        if prompt_logger is not None and not prompt_logger.is_full:
            prompt_logger.log_prompts(texts, tokenizer, source="individual")

        # Tokenize with offset mapping for POS alignment
        enc = tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_offsets_mapping=True,
        )

        input_ids = enc["input_ids"].to(device)           # (B, T)
        attention_mask = enc["attention_mask"].to(device)  # (B, T)
        offset_mapping = enc["offset_mapping"]             # (B, T, 2) on CPU

        with torch.no_grad():
            h = extractor.extract(input_ids, attention_mask=attention_mask)
            z, aux = encoder(h, attention_mask=attention_mask)
            # For VQ encoders, use continuous z_pre instead of quantized z_q
            z = aux.get("z_pre", z)
            z = z.cpu().numpy()                 # (B, T, d_z)

        mask_np = attention_mask.cpu().numpy()  # (B, T)

        for i in range(len(batch_rows)):
            offsets_i = [
                (int(s), int(e))
                for s, e in offset_mapping[i].tolist()
            ]
            pos_tags_i = get_pos_tags(nlp, texts[i], offsets_i)

            for t in range(enc["input_ids"].shape[1]):
                if mask_np[i, t] == 0:
                    continue
                if pos_tags_i[t] == "SPEC":
                    continue
                all_z.append(z[i, t])
                all_subjects.append(subjects_batch[i])
                all_q_idx.append(q_indices_batch[i])
                all_pos.append(pos_tags_i[t])

        print(
            f"  Processed {batch_end}/{n} questions,"
            f" {len(all_z)} tokens so far",
            end="\r",
        )

    print()
    return (
        np.array(all_z, dtype=np.float32),
        all_subjects,
        np.array(all_q_idx, dtype=np.int32),
        all_pos,
    )


def extract_mmlu_h(
    dataset,
    extractor: HiddenStateExtractor,
    nlp,
    max_length: int = 256,
    batch_size: int = 32,
    device: str = "cuda",
    prompt_logger: PromptLogger | None = None,
) -> tuple[np.ndarray, list[str], np.ndarray, list[str]]:
    """Extract raw hidden states h with per-token labels (no encoder).

    Same iteration logic as :func:`extract_embeddings` but returns h
    instead of encoder z.  Used by baseline modes in both tsne_mmlu.py
    and probe_mmlu.py.

    Returns
    -------
    h_all     : np.ndarray (N_tokens, d_h)
    subjects  : list[str]  length N_tokens
    q_indices : np.ndarray (N_tokens,) int
    pos_tags  : list[str]  length N_tokens
    """
    tokenizer = extractor.tokenizer

    all_h: list[np.ndarray] = []
    all_subjects: list[str] = []
    all_q_idx: list[int] = []
    all_pos: list[str] = []

    n = len(dataset)

    for batch_start in range(0, n, batch_size):
        batch_end = min(batch_start + batch_size, n)
        batch_rows = dataset.select(range(batch_start, batch_end))

        texts = [format_question(row) for row in batch_rows]
        subjects_batch = [row["subject"] for row in batch_rows]
        q_indices_batch = list(range(batch_start, batch_end))

        if prompt_logger is not None and not prompt_logger.is_full:
            prompt_logger.log_prompts(texts, tokenizer, source="individual")

        enc = tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_offsets_mapping=True,
        )

        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        offset_mapping = enc["offset_mapping"]

        with torch.no_grad():
            h = extractor.extract(input_ids, attention_mask=attention_mask)
            h_np = h.cpu().numpy()

        mask_np = attention_mask.cpu().numpy()

        for i in range(len(batch_rows)):
            offsets_i = [
                (int(s), int(e))
                for s, e in offset_mapping[i].tolist()
            ]
            pos_tags_i = get_pos_tags(nlp, texts[i], offsets_i)

            for t in range(enc["input_ids"].shape[1]):
                if mask_np[i, t] == 0:
                    continue
                if pos_tags_i[t] == "SPEC":
                    continue
                all_h.append(h_np[i, t])
                all_subjects.append(subjects_batch[i])
                all_q_idx.append(q_indices_batch[i])
                all_pos.append(pos_tags_i[t])

        print(
            f"  Processed {batch_end}/{n} questions,"
            f" {len(all_h)} tokens so far",
            end="\r",
        )

    print()
    return (
        np.array(all_h, dtype=np.float32),
        all_subjects,
        np.array(all_q_idx, dtype=np.int32),
        all_pos,
    )


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
def _save_fig(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def plot_semantic(
    z_2d: np.ndarray,
    subjects: list[str],
    out_path: Path,
    use_groups: bool,
    subject_to_group: dict[str, str] | None = None,
    title_suffix: str = "",
) -> None:
    """Plot 1: color by MMLU subject or broad group.

    Parameters
    ----------
    subject_to_group : optional override for the subject→group mapping.
        Defaults to SUBJECT_TO_GROUP (4-way) if not provided.
    title_suffix : appended to the title (e.g. " (4-way)" or " (7-way)").
    """
    if use_groups:
        mapping = subject_to_group or SUBJECT_TO_GROUP
        labels = [mapping.get(s, "Other") for s in subjects]
        categories = sorted(set(labels))
        cmap = plt.cm.get_cmap("tab10", len(categories))
        cat_to_color = {c: cmap(i) for i, c in enumerate(categories)}
        legend_fontsize = 9
        legend_ncol = 1
    else:
        labels = subjects
        categories = sorted(set(labels))
        # Build a 60-colour palette from tab20 + tab20b + tab20c
        palette: list = []
        for cmap_name in ("tab20", "tab20b", "tab20c"):
            cm = plt.cm.get_cmap(cmap_name)
            palette.extend([cm(i / 20) for i in range(20)])
        cat_to_color = {c: palette[i % len(palette)] for i, c in enumerate(categories)}
        legend_fontsize = 5
        legend_ncol = 3

    colors = [cat_to_color[lbl] for lbl in labels]

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(z_2d[:, 0], z_2d[:, 1], c=colors, s=6, alpha=0.7, linewidths=0)
    title = "Semantic: color = MMLU " + ("group" if use_groups else "subject")
    if title_suffix:
        title += f" {title_suffix}"
    ax.set_title(title, fontsize=13)
    ax.set_xticks([])
    ax.set_yticks([])

    patches = [mpatches.Patch(color=cat_to_color[c], label=c) for c in categories]
    ax.legend(
        handles=patches,
        bbox_to_anchor=(1.01, 1),
        loc="upper left",
        borderaxespad=0,
        frameon=True,
        fontsize=legend_fontsize,
        ncol=legend_ncol,
    )

    _save_fig(fig, out_path)


def plot_contextual(
    z_2d: np.ndarray,
    q_indices: np.ndarray,
    out_path: Path,
) -> None:
    """Plot 2: color by question index using cyclic HSV colormap."""
    n_questions = int(q_indices.max()) + 1
    colors = plt.cm.hsv(q_indices / max(n_questions, 1))

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(z_2d[:, 0], z_2d[:, 1], c=colors, s=6, alpha=0.7, linewidths=0)
    ax.set_title("Contextual: color = question index", fontsize=13)
    ax.set_xticks([])
    ax.set_yticks([])

    _save_fig(fig, out_path)


def plot_syntactic(
    z_2d: np.ndarray,
    pos_tags: list[str],
    out_path: Path,
) -> None:
    """Plot 3: color by Universal POS tag."""
    categories = sorted(set(pos_tags))
    n_cats = len(categories)
    cmap = plt.cm.get_cmap("tab20" if n_cats > 10 else "tab10", n_cats)
    cat_to_color = {c: cmap(i) for i, c in enumerate(categories)}
    colors = [cat_to_color[t] for t in pos_tags]

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(z_2d[:, 0], z_2d[:, 1], c=colors, s=6, alpha=0.7, linewidths=0)
    ax.set_title("Syntactic: color = POS tag", fontsize=13)
    ax.set_xticks([])
    ax.set_yticks([])

    patches = [mpatches.Patch(color=cat_to_color[c], label=c) for c in categories]
    ax.legend(
        handles=patches,
        bbox_to_anchor=(1.01, 1),
        loc="upper left",
        borderaxespad=0,
        frameon=True,
    )

    _save_fig(fig, out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _load_mmlu_subsampled(max_questions_per_subject: int):
    """Load MMLU test split and subsample up to N questions per subject."""
    print("Loading MMLU dataset …")
    dataset = load_dataset("cais/mmlu", "all", split="test")
    print(f"  Total questions: {len(dataset)}")

    subject_rows: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(dataset):
        subject_rows[row["subject"]].append(i)

    rng = np.random.default_rng(42)
    selected_indices: list[int] = []
    for _subj, idxs in sorted(subject_rows.items()):
        if len(idxs) > max_questions_per_subject:
            chosen = rng.choice(idxs, size=max_questions_per_subject, replace=False)
        else:
            chosen = np.array(idxs)
        selected_indices.extend(chosen.tolist())

    selected_indices.sort()
    dataset_sub = dataset.select(selected_indices)
    print(f"  After subsampling: {len(dataset_sub)} questions")
    return dataset_sub


def _run_tsne_and_plot(
    z_all: np.ndarray,
    subjects: list[str],
    q_indices: np.ndarray,
    pos_tags: list[str],
    out_dir: Path,
    max_tsne_tokens: int,
) -> None:
    """Subsample, run t-SNE, generate all 5 plots."""
    n_total = len(z_all)
    n_tsne = min(max_tsne_tokens, n_total)
    rng = np.random.default_rng(42)
    idx = rng.choice(n_total, size=n_tsne, replace=False)
    idx.sort()

    z_sub = z_all[idx]
    subjects_sub = [subjects[i] for i in idx]
    q_idx_sub = q_indices[idx]
    pos_sub = [pos_tags[i] for i in idx]

    # Check for degenerate embeddings (e.g. collapsed VQ)
    z_std = np.std(z_sub, axis=0)
    n_active_dims = int(np.sum(z_std > 1e-6))
    if n_active_dims < 2:
        print(
            f"ERROR: embeddings are degenerate ({n_active_dims} active dims). "
            f"t-SNE would fail. Skipping all plots."
        )
        print(
            "This typically means the encoder has collapsed "
            "(e.g. VQ with too few active codes)."
        )
        return

    print(f"Running t-SNE on {n_tsne} tokens (perplexity=30) …")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    z_2d = tsne.fit_transform(z_sub)
    print("  t-SNE done.")

    print("Generating plots …")
    plot_semantic(
        z_2d, subjects_sub,
        out_dir / "tsne_mmlu_semantic.png", use_groups=False,
    )
    plot_semantic(
        z_2d, subjects_sub,
        out_dir / "tsne_mmlu_semantic_grouped_4way.png",
        use_groups=True, subject_to_group=SUBJECT_TO_GROUP_4WAY,
        title_suffix="(4-way)",
    )
    plot_semantic(
        z_2d, subjects_sub,
        out_dir / "tsne_mmlu_semantic_grouped_7way.png",
        use_groups=True, subject_to_group=SUBJECT_TO_GROUP_7WAY,
        title_suffix="(7-way)",
    )
    plot_contextual(z_2d, q_idx_sub, out_dir / "tsne_mmlu_contextual.png")
    plot_syntactic(z_2d, pos_sub, out_dir / "tsne_mmlu_syntactic.png")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Token-level t-SNE on MMLU: semantic / contextual / syntactic."
    )
    parser.add_argument(
        "checkpoint", type=str, nargs="?", default=None,
        help="Path to checkpoint_final.pt (required unless a --baseline-* flag is set)",
    )
    parser.add_argument(
        "--group",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Color semantic plot by 7 broad groups (default = paper Fig 22); "
             "--no-group uses 57 fine-grained subjects",
    )
    parser.add_argument(
        "--max-questions-per-subject",
        type=int,
        default=50,
        help="Subsample up to N questions per MMLU subject (default: 50 → ≤2850 total)",
    )
    parser.add_argument(
        "--max-tsne-tokens",
        type=int,
        default=10000,
        help="Max tokens fed into t-SNE (random subsample, default: 10000 = "
             "paper Fig 22, same 10,000 tokens shown in each panel)",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    add_baseline_args(parser)
    args = parser.parse_args()

    mode = get_baseline_mode(args)

    # ------------------------------------------------------------------
    # Load spacy (needed for all modes)
    # ------------------------------------------------------------------
    print("Loading spacy model …")
    try:
        nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
    except OSError:
        raise RuntimeError(
            "spacy model 'en_core_web_sm' not found. "
            "Run: python -m spacy download en_core_web_sm"
        )

    if mode == "checkpoint":
        # ==============================================================
        # Checkpoint mode (original behaviour)
        # ==============================================================
        ctx = setup_baseline(mode, args, device=args.device)
        out_dir = ctx.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        prompt_logger = PromptLogger(out_dir)
        cache_path = out_dir / "mmlu_embeddings.npz"

        print("Loading config and checkpoint …")
        encoder, _cfg, _ckpt = load_encoder(args.checkpoint, device=args.device, config=ctx.config)

        if cache_path.exists():
            print(f"Loading cached embeddings from {cache_path} …")
            npz = np.load(cache_path, allow_pickle=True)
            z_all = npz["z_all"]
            subjects = list(npz["subjects"])
            q_indices = npz["q_indices"]
            pos_tags = list(npz["pos_tags"])
            print(f"  {len(z_all)} cached tokens, d_z={z_all.shape[1]}")
        else:
            dataset_sub = _load_mmlu_subsampled(args.max_questions_per_subject)

            print("Extracting token-level embeddings …")
            z_all, subjects, q_indices, pos_tags = extract_embeddings(
                dataset_sub, ctx.extractor, encoder, nlp,
                max_length=256, batch_size=args.batch_size, device=args.device,
                prompt_logger=prompt_logger,
            )
            print(f"  Total real tokens: {len(z_all)}, d_z={z_all.shape[1]}")

            np.savez(
                cache_path, z_all=z_all,
                subjects=np.array(subjects), q_indices=q_indices,
                pos_tags=np.array(pos_tags),
            )
            print(f"Cached embeddings to {cache_path}")

        prompt_logger.save()
        extractor = ctx.extractor

    else:
        # ==============================================================
        # Baseline modes: hh / sae / pca
        # ==============================================================
        ctx = setup_baseline(mode, args, device=args.device)
        out_dir = ctx.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        prompt_logger = PromptLogger(out_dir)
        cache_name = f"mmlu_embeddings_{mode}"
        if mode == "pca":
            cache_name += f"_{args.pca_dim}"
        cache_path = out_dir / f"{cache_name}.npz"

        if cache_path.exists():
            print(f"Loading cached embeddings from {cache_path} …")
            npz = np.load(cache_path, allow_pickle=True)
            z_all = npz["z_all"]
            subjects = list(npz["subjects"])
            q_indices = npz["q_indices"]
            pos_tags = list(npz["pos_tags"])
            print(f"  {len(z_all)} cached tokens, d={z_all.shape[1]}")
        else:
            dataset_sub = _load_mmlu_subsampled(args.max_questions_per_subject)

            print(f"Extracting hidden states h for {mode} baseline …")
            h_all, subjects, q_indices, pos_tags = extract_mmlu_h(
                dataset_sub, ctx.extractor, nlp,
                max_length=256, batch_size=args.batch_size, device=args.device,
                prompt_logger=prompt_logger,
            )
            print(f"  Total real tokens: {len(h_all)}, d_h={h_all.shape[1]}")

            # Fit PCA if needed
            if mode == "pca":
                comps, mean = fit_pca_on_h(h_all, args.pca_dim)
                ctx.pca_components = comps
                ctx.pca_mean = mean

            # Transform h -> representation
            h_tensor = torch.from_numpy(h_all)
            z_all = transform_h(h_tensor, ctx, device=args.device).cpu().numpy()
            print(f"  Representation dim: {z_all.shape[1]}")

            np.savez(
                cache_path, z_all=z_all,
                subjects=np.array(subjects), q_indices=q_indices,
                pos_tags=np.array(pos_tags),
            )
            print(f"Cached embeddings to {cache_path}")

        prompt_logger.save()
        extractor = ctx.extractor

    # ------------------------------------------------------------------
    # t-SNE + plots (shared for all modes)
    # ------------------------------------------------------------------
    _run_tsne_and_plot(
        z_all, subjects, q_indices, pos_tags, out_dir, args.max_tsne_tokens,
    )

    extractor.cleanup()
    print("Done. All plots saved to", out_dir)


if __name__ == "__main__":
    main()
