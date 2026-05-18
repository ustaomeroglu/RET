#!/usr/bin/env python3
"""Per-sequence event structure evaluation.

For each selected sequence, produces two plots comparing representation types
(h, z, z_hat) side by side:

  * 3D UMAP trajectory + pairwise cosine similarity heatmap (with tortuosity)
  * Hierarchical clustering dendrograms (cosine distance, average linkage)

Data sources (two modes):

  * HuggingFace stream (default): downloads text, runs the LLM forward to
    obtain hidden states h.
  * Cached memmap (``--cached-hidden-states-dir``): reads h and input_ids
    from a precomputed cache directory produced by
    ``scripts/data_preparation_scripts/cache_hidden_states.py``. No LLM is
    loaded unless the checkpoint uses a controlled predictor (which still
    needs token embeddings).

Usage:
    # Single checkpoint, default 3 sequences from TinyStories (HF stream)
    python scripts/event_structure.py outputs/sparse_mlp_autonomous/checkpoint_final.pt

    # Custom number of sequences
    python scripts/event_structure.py outputs/.../checkpoint_final.pt --num-sequences 5

    # Different dataset via HF stream (e.g. math)
    python scripts/event_structure.py outputs/.../checkpoint_final.pt \\
        --hf-dataset AI-MO/NuminaMath-CoT --split train \\
        --text-column problem --min-tokens 40 --max-tokens 200

    # Use a cached hidden-states directory (no LLM load)
    python scripts/event_structure.py outputs/.../checkpoint_final.pt \\
        --cached-hidden-states-dir data/tiny_stories/GptOss20B/layer11 \\
        --split val

    # All checkpoints under outputs/
    python scripts/event_structure.py --all-checkpoints --num-sequences 3
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from transformers import AutoTokenizer
import umap

from ret.config import load_config
from ret.data.dataset import CachedHiddenStateDataset
from ret.utils.prompt_logger import PromptLogger
from evaluation.baseline_utils import (
    add_baseline_args,
    get_baseline_mode,
    setup_baseline,
    fit_pca_on_h,
    transform_h,
)
from ret.extraction.hooks import HiddenStateExtractor
from ret.utils.checkpointing import load_encoder
from ret.models.predictor import Predictor
from ret.models.predictor import TokenProjector


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------
def load_text_sequences(
    hf_dataset: str,
    split: str,
    text_column: str,
    num_sequences: int,
    tokenizer,
    min_tokens: int,
    max_tokens: int,
    seed: int,
) -> list[str]:
    """Stream texts from an HF dataset and return sequences whose tokenized
    length is in ``[min_tokens, max_tokens]``.

    Two-stage filter: a coarse char-length pre-filter (~3-6 chars/token) builds
    a candidate pool so we don't tokenize the whole dataset, then an exact
    token-count filter selects ``num_sequences`` entries after shuffling.
    """
    ds = load_dataset(hf_dataset, split=split, streaming=True)
    rng = np.random.default_rng(seed)

    candidates: list[str] = []
    for row in ds:
        text = row.get(text_column)
        if not isinstance(text, str):
            continue
        text = text.strip()
        char_len = len(text)
        if char_len < min_tokens * 3 or char_len > max_tokens * 6:
            continue
        candidates.append(text)
        if len(candidates) >= num_sequences * 20:
            break

    if not candidates:
        raise ValueError(
            f"No candidates found in {hf_dataset}[{split}] column '{text_column}' "
            f"matching char length ~{min_tokens * 3}..{max_tokens * 6}. "
            "Try relaxing --min-tokens/--max-tokens or check --text-column."
        )

    indices = rng.permutation(len(candidates))
    shuffled = [candidates[i] for i in indices]

    accepted: list[str] = []
    for text in shuffled:
        ids = tokenizer.encode(text)
        if min_tokens <= len(ids) <= max_tokens:
            accepted.append(text)
            if len(accepted) >= num_sequences:
                break

    if len(accepted) < num_sequences:
        print(
            f"  WARNING: only found {len(accepted)} sequences in "
            f"{hf_dataset}[{split}] with token count in "
            f"[{min_tokens}, {max_tokens}] (requested {num_sequences})."
        )
    return accepted


def load_cached_sequences(
    cached_dir: Path,
    split: str,
    num_sequences: int,
    min_tokens: int,
    max_tokens: int,
    seed: int,
) -> tuple[list[dict], "AutoTokenizer"]:
    """Load precomputed h + input_ids from a cached memmap directory.

    Returns ``(sequences, tokenizer)`` where each sequence dict contains:
        * ``h``: ``np.ndarray`` of shape ``(T, d_h)``
        * ``input_ids``: ``list[int]`` of length ``T``
        * ``tokens``: ``list[str]`` of length ``T`` (per-token decoded strings)
        * ``text``: ``str`` (full decoded text, may include chat scaffolding)
    """
    split_dir = Path(cached_dir) / split
    if not split_dir.exists():
        raise FileNotFoundError(
            f"Cache split dir not found: {split_dir}. "
            f"Expected layout: {cached_dir}/{{train,val,test}}/"
        )

    with open(split_dir / "metadata.json") as f:
        meta = json.load(f)
    tokenizer = AutoTokenizer.from_pretrained(
        meta["model_name"], trust_remote_code=True,
    )

    ds = CachedHiddenStateDataset(split_dir)
    n_total = len(ds)
    # Filter by pre-stored lengths (avoid loading h/ids for docs we'll drop)
    keep = [i for i in range(n_total) if min_tokens <= int(ds.lengths[i]) <= max_tokens]
    if not keep:
        raise ValueError(
            f"No cached sequences at {split_dir} with length in "
            f"[{min_tokens}, {max_tokens}] (cache has {n_total} docs)."
        )
    rng = np.random.default_rng(seed)
    order = np.array(keep)[rng.permutation(len(keep))].tolist()
    order = order[:num_sequences]

    sequences: list[dict] = []
    for idx in order:
        item = ds[idx]
        h_np = item["hidden_states"].numpy()  # (T, d_h) float32
        input_ids = item["input_ids"].tolist()
        tokens = [tokenizer.decode([tid]) for tid in input_ids]
        text = tokenizer.decode(input_ids, skip_special_tokens=False)
        sequences.append({
            "h": h_np,
            "input_ids": input_ids,
            "tokens": tokens,
            "text": text,
        })

    if len(sequences) < num_sequences:
        print(
            f"  WARNING: only found {len(sequences)} cached sequences in "
            f"{split_dir} with token count in [{min_tokens}, {max_tokens}] "
            f"(requested {num_sequences})."
        )
    return sequences, tokenizer


def _sequences_from_extractor(
    texts: list[str],
    extractor: HiddenStateExtractor,
    device: str,
) -> list[dict]:
    """Run the LLM forward on each text and build the sequence-dict list."""
    tokenizer = extractor.tokenizer
    out: list[dict] = []
    for text in texts:
        enc = tokenizer(text, return_tensors="pt")
        input_ids_t = enc["input_ids"].to(device)  # (1, T)
        with torch.no_grad():
            h = extractor.extract(input_ids_t)  # (1, T, d_h)
        input_ids = input_ids_t[0].cpu().tolist()
        tokens = [tokenizer.decode([tid]) for tid in input_ids]
        out.append({
            "h": h[0].cpu().numpy(),
            "input_ids": input_ids,
            "tokens": tokens,
            "text": text,
        })
    return out


def prepare_sequences(
    *,
    num_sequences: int,
    min_tokens: int,
    max_tokens: int,
    seed: int,
    cached_dir: Path | None,
    split: str,
    hf_dataset: str,
    text_column: str,
    extractor: HiddenStateExtractor | None,
    device: str,
) -> tuple[list[dict], object]:
    """Dispatch to the cached or HF-stream loader.

    Returns ``(sequences, tokenizer)``. The tokenizer is either the one bundled
    with the extractor (HF mode) or the one built from the cache metadata.
    """
    if cached_dir is not None:
        return load_cached_sequences(
            cached_dir=cached_dir, split=split,
            num_sequences=num_sequences,
            min_tokens=min_tokens, max_tokens=max_tokens, seed=seed,
        )
    if extractor is None:
        raise RuntimeError(
            "HF-stream mode requires an extractor; pass --cached-hidden-states-dir "
            "to use a cache instead."
        )
    texts = load_text_sequences(
        hf_dataset=hf_dataset, split=split, text_column=text_column,
        num_sequences=num_sequences, tokenizer=extractor.tokenizer,
        min_tokens=min_tokens, max_tokens=max_tokens, seed=seed,
    )
    sequences = _sequences_from_extractor(texts, extractor, device)
    return sequences, extractor.tokenizer


# ---------------------------------------------------------------------------
# Encoder + predictor rollout on precomputed h
# ---------------------------------------------------------------------------
def compute_z_zhat(
    seq: dict,
    encoder: torch.nn.Module,
    predictor: Predictor,
    token_projector: TokenProjector | None,
    extractor: HiddenStateExtractor | None,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(z, z_hat)`` given a sequence dict with precomputed ``h``.

    Shapes: ``z`` is ``(T, d_z)``, ``z_hat`` is ``(T-1, d_z)``.

    For controlled predictors, ``extractor`` is required so that token
    embeddings can be looked up for ``u_next``.
    """
    h = torch.from_numpy(seq["h"]).to(device).unsqueeze(0)  # (1, T, d_h)
    with torch.no_grad():
        z_full, aux = encoder(h)
        # For VQ encoders, use continuous z_pre
        z_full = aux.get("z_pre", z_full)
        z_t = z_full[:, :-1, :]

        if predictor.mode == "controlled" and token_projector is not None:
            if extractor is None:
                raise RuntimeError(
                    "Controlled predictor requires an LLM extractor for token "
                    "embeddings. Rerun without --cached-hidden-states-dir, or "
                    "load the LLM alongside the cache."
                )
            ids_t = torch.tensor([seq["input_ids"]], device=device)
            token_emb = extractor.get_token_embeddings(ids_t)  # (1, T, d_h)
            u = token_projector(token_emb)
            u_next = u[:, 1:, :]
            z_hat = predictor(z_t, u_next)
        else:
            z_hat = predictor(z_t)

    return z_full[0].cpu().numpy(), z_hat[0].cpu().numpy()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def pairwise_cosine_similarity(X: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine similarity matrix after robust (median) centering.

    Centering removes the dominant "common direction" present in LLM hidden
    states so off-diagonal structure is visible. We use the per-feature
    median rather than the mean: a single token with a massive activation
    (commonly seen in LLMs at sentence-boundary punctuation tokens — see
    Sun et al. 2024 on "massive activations") drags the mean toward itself,
    which after subtraction makes every other token look antipodal to that
    one outlier and produces a misleading −1 stripe in row/column ``i``.
    The median is unaffected by such single outliers, so the resulting
    cosine matrix reflects genuine pairwise structure rather than the
    influence of one extreme token.

    Args:
        X: (T, d) array
    Returns:
        (T, T) cosine similarity matrix
    """
    X = X - np.median(X, axis=0, keepdims=True)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    X_normed = X / norms
    return X_normed @ X_normed.T


def tortuosity(coords_3d: np.ndarray) -> float:
    """Compute tortuosity = path_length / chord_length in UMAP space."""
    diffs = np.diff(coords_3d, axis=0)
    segment_lengths = np.linalg.norm(diffs, axis=1)
    path_length = segment_lengths.sum()
    chord_length = np.linalg.norm(coords_3d[-1] - coords_3d[0])
    if chord_length < 1e-10:
        return float("inf")
    return path_length / chord_length


def tortuosity_cosine(X: np.ndarray) -> float:
    """Tortuosity computed directly in the original feature space using cosine
    distance d(x, y) = 1 - cos_sim(x, y).

    Scale- and dimension-invariant, so it is comparable across representations
    of different widths (h, h^pooled, SAE, z). Median-centering matches
    ``pairwise_cosine_similarity``, so the τ value is consistent with the
    similarity heatmap displayed below it.
    """
    eps = 1e-10
    Xc = X - np.median(X, axis=0, keepdims=True)
    norms = np.linalg.norm(Xc, axis=1, keepdims=True)
    Xn = Xc / np.maximum(norms, eps)
    cos_consec = np.sum(Xn[:-1] * Xn[1:], axis=1)
    path = float((1.0 - cos_consec).sum())
    chord = 1.0 - float(Xn[0] @ Xn[-1])
    if chord < eps:
        return float("inf")
    return path / chord


# ---------------------------------------------------------------------------
# UMAP + Cosine Similarity + Tortuosity
# ---------------------------------------------------------------------------
def plot_umap_similarity(
    representations: list[tuple[str, np.ndarray, list[str]]],
    out_path: Path,
    seq_idx: int,
) -> None:
    """N-column figure. Each column: 3D UMAP (top) + cosine sim heatmap (bottom).

    Parameters
    ----------
    representations : list of (label, X, tokens) tuples.
        ``X`` is ``(T, d)``; ``tokens`` is length ``T``.
    """
    n_cols = len(representations)
    fig = plt.figure(figsize=(6 * n_cols, 10))
    fig.suptitle(
        f"Sequence {seq_idx}: Event Structure (UMAP + Cosine Similarity)",
        fontsize=14,
    )

    for col, (label, X, toks) in enumerate(representations):
        T = X.shape[0]

        # 3D UMAP
        n_neighbors = min(15, T - 1)
        reducer = umap.UMAP(n_components=3, n_neighbors=n_neighbors, min_dist=0.1, random_state=42)
        coords_3d = reducer.fit_transform(X)

        tau = tortuosity(coords_3d)

        ax_3d = fig.add_subplot(2, n_cols, col + 1, projection="3d")
        colors = np.arange(T) / max(T - 1, 1)
        ax_3d.scatter(
            coords_3d[:, 0], coords_3d[:, 1], coords_3d[:, 2],
            c=colors, cmap="viridis", s=30, edgecolors="k", linewidths=0.3,
        )
        # Draw path
        ax_3d.plot(
            coords_3d[:, 0], coords_3d[:, 1], coords_3d[:, 2],
            color="gray", alpha=0.4, linewidth=0.8,
        )
        ax_3d.set_title(f"{label}\n\u03c4 = {tau:.2f}", fontsize=11)
        ax_3d.set_xticks([])
        ax_3d.set_yticks([])
        ax_3d.set_zticks([])

        # Cosine similarity heatmap
        sim = pairwise_cosine_similarity(X)
        ax_hm = fig.add_subplot(2, n_cols, n_cols + col + 1)
        im = ax_hm.imshow(sim, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax_hm.set_xlabel("Token position")
        ax_hm.set_ylabel("Token position")
        ax_hm.set_title(f"Cosine similarity ({label})", fontsize=10)
        plt.colorbar(im, ax=ax_hm, fraction=0.046, pad=0.04)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# Hierarchical Clustering Dendrograms
# ---------------------------------------------------------------------------
def _reorder_linkage_by_position(Z, n, start_idx):
    """Reorder linkage so subtrees with lower average token position go left.

    At every merge node the left/right child ordering is a free choice that
    does not affect the clustering — we choose the order that puts earlier
    tokens on the left side of the plot.
    """
    Z_ordered = Z.copy()
    avg_pos = {i: float(start_idx + i) for i in range(n)}

    for k in range(len(Z_ordered)):
        left_id = int(Z_ordered[k, 0])
        right_id = int(Z_ordered[k, 1])

        left_avg = avg_pos[left_id]
        right_avg = avg_pos[right_id]

        if left_avg > right_avg:
            Z_ordered[k, 0], Z_ordered[k, 1] = Z_ordered[k, 1], Z_ordered[k, 0]

        left_count = 1 if left_id < n else int(Z_ordered[left_id - n, 3])
        right_count = 1 if right_id < n else int(Z_ordered[right_id - n, 3])
        total = left_count + right_count
        avg_pos[n + k] = (left_avg * left_count + right_avg * right_count) / total

    return Z_ordered


def plot_dendrograms(
    representations: list[tuple[str, np.ndarray, list[str], int]],
    out_path: Path,
    seq_idx: int,
    sequence_text: str = "",
) -> None:
    """N-row dendrogram with token labels including position indices.

    Parameters
    ----------
    representations : list of (label, X, tokens, start_idx) tuples.
        ``X`` is ``(T, d)``; ``tokens`` is length ``T``;
        ``start_idx`` is the token offset (0 for h/z, 1 for z_hat).
    """
    n_rows = len(representations)
    fig, axes = plt.subplots(n_rows, 1, figsize=(28, 7 * n_rows + 1))
    if n_rows == 1:
        axes = [axes]
    fig.suptitle(
        f"Sequence {seq_idx}: Hierarchical Clustering (Cosine Distance)",
        fontsize=14, y=0.99,
    )

    if sequence_text:
        import textwrap
        wrapped = textwrap.fill(sequence_text.strip(), width=140)
        fig.text(
            0.5, 0.96, wrapped,
            ha="center", va="top", fontsize=8,
            fontstyle="italic", color="0.3",
            wrap=True,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.7),
        )

    for ax, (label, X, toks, start_idx) in zip(axes, representations):
        n = X.shape[0]
        X_centered = X - X.mean(axis=0, keepdims=True)
        norms = np.linalg.norm(X_centered, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        X_normed = X_centered / norms

        sim = X_normed @ X_normed.T
        dist = 1.0 - sim
        np.fill_diagonal(dist, 0.0)
        dist = np.maximum(dist, 0.0)

        condensed = squareform(dist, checks=False)
        Z = linkage(condensed, method="average")
        Z = _reorder_linkage_by_position(Z, n, start_idx)

        short_labels = []
        for i, t in enumerate(toks):
            clean = t.replace("\n", "\\n")[:12]
            short_labels.append(f"{clean} ({start_idx + i})")

        dendrogram(
            Z,
            ax=ax,
            labels=short_labels,
            leaf_rotation=90,
            leaf_font_size=7,
            color_threshold=0.7 * max(Z[:, 2]) if len(Z) > 0 else 0,
        )
        ax.set_title(label, fontsize=12)
        ax.set_ylabel("Cosine distance")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# Shared helper: write sequence_texts.txt
# ---------------------------------------------------------------------------
def _write_sequence_texts(sequences: list[dict], out_dir: Path) -> None:
    texts_file = out_dir / "sequence_texts.txt"
    with open(texts_file, "w") as f:
        for idx, seq in enumerate(sequences):
            n_tok = len(seq["input_ids"])
            f.write(f"{'=' * 72}\n")
            f.write(f"Sequence {idx}  ({n_tok} tokens)\n")
            f.write(f"{'=' * 72}\n")
            f.write(seq["text"].strip() + "\n\n")
    print(f"Saved sequence texts to {texts_file}")


# ---------------------------------------------------------------------------
# Checkpoint processing
# ---------------------------------------------------------------------------
def process_checkpoint(
    ckpt_path: Path,
    num_sequences: int,
    hf_dataset: str,
    split: str,
    text_column: str,
    min_tokens: int,
    max_tokens: int,
    seed: int,
    device: str,
    skip_existing: bool,
    cached_dir: Path | None = None,
) -> None:
    """Run full event structure analysis for one checkpoint."""
    ckpt_dir = ckpt_path.parent
    out_dir = ckpt_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Skip check
    if skip_existing:
        marker = out_dir / "event_umap_sim_0.png"
        if marker.exists():
            print(f"  SKIPPED — analysis already exists ({marker})")
            return

    # Load config + checkpoint
    print(f"Loading config and checkpoint from {ckpt_dir} ...")
    encoder, config, ckpt = load_encoder(ckpt_path, device=device)

    # Build models
    predictor = Predictor(config.encoder.z_dim, config.predictor).to(device)
    try:
        predictor.load_state_dict(ckpt["predictor"])
    except RuntimeError as e:
        # Legacy checkpoint with flat nn.Sequential keys (pre-residual architecture)
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

    # LLM extractor is only needed when (a) no cache is provided, or (b) the
    # predictor is controlled (token embeddings still come from the LLM).
    needs_extractor = (cached_dir is None) or (
        predictor.mode == "controlled" and token_projector is not None
    )
    extractor = HiddenStateExtractor(config.llm, device=device) if needs_extractor else None
    if cached_dir is not None and extractor is not None:
        print("  Controlled predictor: LLM loaded for token embeddings (h from cache).")

    # Load sequences
    if cached_dir is not None:
        print(
            f"Loading cached sequences from {cached_dir}/{split} "
            f"(need {num_sequences} with {min_tokens}-{max_tokens} tokens) ..."
        )
    else:
        print(
            f"Loading {hf_dataset} [{split}] "
            f"(need {num_sequences} sequences with {min_tokens}-{max_tokens} tokens) ..."
        )
    sequences, tokenizer = prepare_sequences(
        num_sequences=num_sequences, min_tokens=min_tokens, max_tokens=max_tokens,
        seed=seed, cached_dir=cached_dir, split=split,
        hf_dataset=hf_dataset, text_column=text_column,
        extractor=extractor, device=device,
    )
    print(f"  Selected {len(sequences)} sequences")

    _write_sequence_texts(sequences, out_dir)

    # Process each sequence
    prompt_logger = PromptLogger(out_dir)
    for idx, seq in enumerate(sequences):
        T = len(seq["input_ids"])
        print(f"\n--- Sequence {idx} ({T} tokens) ---")
        print(f"  \"{seq['text'][:80]}...\"")

        if not prompt_logger.is_full:
            prompt_logger.log_prompts([seq["text"]], tokenizer, source="document")

        z, z_hat = compute_z_zhat(
            seq, encoder, predictor, token_projector, extractor, device,
        )
        h, tokens = seq["h"], seq["tokens"]

        umap_reprs = [
            ("h (hidden states)", h, tokens),
            ("z (encoder)", z, tokens),
            ("\u017a\u0302 (predictor)", z_hat, tokens[1:]),
        ]
        dendro_reprs = [
            ("h (hidden states)", h, tokens, 0),
            ("z (encoder)", z, tokens, 0),
            ("\u017a\u0302 (predictor)", z_hat, tokens[1:], 1),
        ]

        plot_umap_similarity(
            umap_reprs, out_dir / f"event_umap_sim_{idx}.png", seq_idx=idx,
        )
        plot_dendrograms(
            dendro_reprs, out_dir / f"event_dendrogram_{idx}.png",
            seq_idx=idx, sequence_text=seq["text"],
        )

    prompt_logger.save()
    if extractor is not None:
        extractor.cleanup()
    print(f"\nDone. All plots saved to {out_dir}")


def process_baseline(
    ctx,
    num_sequences: int,
    hf_dataset: str,
    split: str,
    text_column: str,
    min_tokens: int,
    max_tokens: int,
    seed: int,
    device: str,
    cached_dir: Path | None = None,
) -> None:
    """Run event structure analysis using a baseline representation."""
    out_dir = ctx.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load sequences (cached or HF stream)
    if cached_dir is not None:
        print(
            f"Loading cached sequences from {cached_dir}/{split} "
            f"(need {num_sequences} with {min_tokens}-{max_tokens} tokens) ..."
        )
    else:
        print(
            f"Loading {hf_dataset} [{split}] "
            f"(need {num_sequences} sequences with {min_tokens}-{max_tokens} tokens) ..."
        )
    sequences, tokenizer = prepare_sequences(
        num_sequences=num_sequences, min_tokens=min_tokens, max_tokens=max_tokens,
        seed=seed, cached_dir=cached_dir, split=split,
        hf_dataset=hf_dataset, text_column=text_column,
        extractor=ctx.extractor, device=device,
    )
    print(f"  Selected {len(sequences)} sequences")

    # For PCA: fit on pooled h from all sequences
    if ctx.mode == "pca" and ctx.pca_components is None:
        print("Fitting PCA on pooled h from all sequences ...")
        h_pooled = np.concatenate([s["h"] for s in sequences], axis=0)
        comps, mean = fit_pca_on_h(h_pooled, ctx.pca_dim)
        ctx.pca_components = comps
        ctx.pca_mean = mean

    _write_sequence_texts(sequences, out_dir)

    # Process each sequence
    prompt_logger = PromptLogger(out_dir)
    for idx, seq in enumerate(sequences):
        T = len(seq["input_ids"])
        print(f"\n--- Sequence {idx} ({T} tokens) ---")
        print(f"  \"{seq['text'][:80]}...\"")

        if not prompt_logger.is_full:
            prompt_logger.log_prompts([seq["text"]], tokenizer, source="document")

        h_np = seq["h"]
        repr_tensor = transform_h(torch.from_numpy(h_np), ctx, device=device)
        repr_np = repr_tensor.cpu().numpy()

        tokens = seq["tokens"]
        umap_reprs = [
            ("h (hidden states)", h_np, tokens),
            (ctx.representation_label, repr_np, tokens),
        ]
        dendro_reprs = [
            ("h (hidden states)", h_np, tokens, 0),
            (ctx.representation_label, repr_np, tokens, 0),
        ]

        plot_umap_similarity(
            umap_reprs, out_dir / f"event_umap_sim_{idx}.png", seq_idx=idx,
        )
        plot_dendrograms(
            dendro_reprs, out_dir / f"event_dendrogram_{idx}.png",
            seq_idx=idx, sequence_text=seq["text"],
        )

    prompt_logger.save()
    if ctx.extractor is not None:
        ctx.extractor.cleanup()
    print(f"\nDone. All plots saved to {out_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Per-sequence event structure evaluation: UMAP + cosine similarity "
            "+ hierarchical clustering of h, z, z_hat."
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
        "--num-sequences", type=int, default=3,
        help="Number of sequences to analyze per checkpoint.",
    )
    parser.add_argument(
        "--cached-hidden-states-dir", type=str, default=None,
        help=(
            "Path to a cached hidden-states directory produced by "
            "scripts/data_preparation_scripts/cache_hidden_states.py. "
            "When set, skips HF streaming and the LLM forward pass (except "
            "for controlled predictors, which still need token embeddings). "
            "Layout: <dir>/{train,val,test}/{hidden_states.bin,input_ids.bin,"
            "offsets.npy,lengths.npy,metadata.json}."
        ),
    )
    parser.add_argument(
        "--hf-dataset", type=str, default="roneneldan/TinyStories",
        help="HuggingFace dataset ID to stream texts from (HF-stream mode only).",
    )
    parser.add_argument(
        "--text-column", type=str, default="text",
        help="Name of the column that contains raw text (HF-stream mode only).",
    )
    parser.add_argument(
        "--split", type=str, default="validation",
        help=(
            "Dataset split. For HF mode use HF conventions (e.g. 'validation'); "
            "for cached mode use the cache subdir name (e.g. 'val')."
        ),
    )
    parser.add_argument(
        "--min-tokens", type=int, default=60,
        help="Minimum token count per sequence.",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=150,
        help="Maximum token count per sequence.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip checkpoints that already have event structure analysis",
    )
    add_baseline_args(parser)
    args = parser.parse_args()

    cached_dir = Path(args.cached_hidden_states_dir) if args.cached_hidden_states_dir else None

    # Determine mode: --all-checkpoints takes priority, then baselines
    is_baseline = any([
        getattr(args, "baseline_hh", False),
        getattr(args, "baseline_sae", False),
        getattr(args, "baseline_pca", False),
    ])

    loader_kwargs = dict(
        num_sequences=args.num_sequences,
        hf_dataset=args.hf_dataset,
        split=args.split,
        text_column=args.text_column,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
        seed=args.seed,
        device=args.device,
        cached_dir=cached_dir,
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
                process_checkpoint(
                    Path(p), skip_existing=args.skip_existing, **loader_kwargs,
                )
            except Exception as e:
                import traceback
                print(f"  FAILED: {e}")
                traceback.print_exc()
    elif is_baseline:
        mode = get_baseline_mode(args)
        # In cache mode, don't load the LLM inside setup_baseline.
        ctx = setup_baseline(
            mode, args, device=args.device,
            skip_extractor=cached_dir is not None,
        )
        process_baseline(ctx, **loader_kwargs)
    elif args.checkpoint is not None:
        process_checkpoint(
            Path(args.checkpoint), skip_existing=args.skip_existing, **loader_kwargs,
        )
    else:
        parser.error(
            "Either provide a checkpoint path, use --all-checkpoints, "
            "or use a --baseline-* flag"
        )


if __name__ == "__main__":
    main()
