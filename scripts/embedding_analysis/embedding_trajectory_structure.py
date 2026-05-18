#!/usr/bin/env python3
"""Generalized document event structure evaluation (Figure 4a & 4b from "Priors in Time").

Per-document analysis of temporal/event structure in latent codes:
  - Figure 4a: 3D UMAP + pairwise cosine similarity heatmap + tortuosity
  - Figure 4b: Hierarchical clustering dendrograms

Supports any dataset (Arrow document-mode or HuggingFace streaming), any LLM,
and any combination of representations: h, z, z_hat, pca-N, sae.

Usage:
    # RET checkpoint, default representations (h, z, z_hat), Arrow dataset
    python scripts/embedding_trajectory_structure.py \
        --checkpoint outputs/GptOss20B/RET_numina_math_gen/checkpoint_final.pt

    # Custom repr set, 5 documents
    python scripts/embedding_trajectory_structure.py \
        --checkpoint outputs/GptOss20B/RET_numina_math_gen/checkpoint_final.pt \
        --repr h z pca-64 --num-docs 5

    # h only (no encoder needed), using config directly
    python scripts/embedding_trajectory_structure.py \
        --config configs/GptOss20B/numina_math_gen.yaml \
        --repr h sae --num-docs 3

    # Fallback to HuggingFace streaming (TinyStories) when config.data.hf_dataset is set
    python scripts/embedding_trajectory_structure.py \
        --checkpoint outputs/mlp_autonomous/checkpoint_final.pt \
        --repr h z z_hat
"""

from __future__ import annotations

import argparse
import textwrap
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
import umap

from transformers import AutoTokenizer

from ret.config import load_config, llm_short_name
from ret.data.dataset import maybe_apply_chat_template, CachedHiddenStateDataset
from evaluation.datasets.tiny_stories import load_stories, filter_by_token_count
from ret.extraction.hooks import HiddenStateExtractor
from ret.utils.checkpointing import load_encoder
from ret.models.predictor import Predictor
from ret.models.predictor import TokenProjector
from ret.utils.prompt_logger import PromptLogger
from evaluation.baseline_utils import fit_pca_on_h
from evaluation.baselines.sae import load_sae_baseline, encode_sae


# ---------------------------------------------------------------------------
# Label constants
# ---------------------------------------------------------------------------
_LABEL_H = "h (hidden states)"
_LABEL_Z = "z (encoder)"
_LABEL_Z_HAT = "\u017a\u0302 (predictor)"  # ź̂
_LABEL_SAE = "SAE"


def _pca_label(n: int) -> str:
    return f"PCA-{n}"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_documents(
    config,
    tokenizer,
    num_docs: int,
    seed: int,
    data_dir: str | None = None,
    data_split: str = "val",
    max_tokens: int | None = None,
) -> list[dict]:
    """Load documents for analysis.

    For Arrow datasets (document mode: local_dir + max_seq_length set), loads
    from ``<data_dir>/<split>`` via ``datasets.load_from_disk()``, applies
    scaffolding, tokenizes with truncation, and samples ``num_docs`` randomly.

    For HuggingFace streaming datasets (hf_dataset set), falls back to the
    TinyStories-style streaming loader.

    Parameters
    ----------
    config : ExperimentConfig
        Full experiment config. ``config.data`` provides data source settings.
    tokenizer :
        Tokenizer for the LLM.
    num_docs : int
        Number of documents to return.
    seed : int
        Random seed for sampling.
    data_dir : str | None
        Override ``config.data.local_dir``. Ignored when hf_dataset is used.
    data_split : str
        Which split subfolder to load (``"train"``, ``"val"``, or ``"test"``).
    max_tokens : int | None
        Truncate each document to at most this many tokens (applied after
        tokenization, in addition to ``config.data.max_seq_length``).

    Returns
    -------
    list of dict, each with keys:
        text (str), input_ids (1, T), attention_mask (1, T) or None,
        tokens (list[str]), doc_idx (int)
    """
    data_cfg = config.data
    mode = data_cfg.mode
    rng = np.random.default_rng(seed)

    effective_local_dir = data_dir or data_cfg.local_dir

    if mode == "cached":
        cache_dir = Path(data_cfg.cached_hidden_states_dir) / data_split
        print(f"Loading cached hidden states from {cache_dir} ...")
        cached_ds = CachedHiddenStateDataset(
            cache_dir, llm_config=config.llm,
            max_seq_length=data_cfg.max_seq_length,
        )
        print(f"  Dataset size: {len(cached_ds)} documents")

        n_avail = len(cached_ds)
        if n_avail < num_docs:
            print(f"  WARNING: only {n_avail} documents available (requested {num_docs})")
            indices = list(range(n_avail))
        else:
            indices = rng.choice(n_avail, size=num_docs, replace=False).tolist()

        docs = []
        for doc_idx, row_idx in enumerate(indices):
            item = cached_ds[int(row_idx)]
            h = item["hidden_states"]       # (L, d_h) float32
            ids = item["input_ids"]         # (L,) int64
            L = item["length"]

            input_ids = ids.unsqueeze(0)            # (1, T)
            hidden_states = h.unsqueeze(0)          # (1, T, d_h)
            attention_mask = torch.ones(1, L, dtype=torch.long)

            token_ids = ids.tolist()
            tokens = [tokenizer.decode([tid]) for tid in token_ids]
            text = tokenizer.decode(token_ids, skip_special_tokens=False)

            docs.append({
                "text": text,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "tokens": tokens,
                "doc_idx": doc_idx,
                "hidden_states": hidden_states,
            })

        print(f"  Loaded {len(docs)} documents (cached)")
        return docs

    if mode == "document":
        from datasets import load_from_disk

        split_dir = Path(effective_local_dir) / data_split
        print(f"Loading Arrow dataset from {split_dir} ...")
        dataset = load_from_disk(str(split_dir))
        print(f"  Dataset size: {len(dataset)} documents")

        max_length = data_cfg.max_seq_length

        # Sample random indices
        n_avail = len(dataset)
        if n_avail < num_docs:
            print(f"  WARNING: only {n_avail} documents available (requested {num_docs})")
            indices = list(range(n_avail))
        else:
            indices = rng.choice(n_avail, size=num_docs, replace=False).tolist()

        docs = []
        for doc_idx, row_idx in enumerate(indices):
            row = dataset[int(row_idx)]

            # Apply scaffolding to get text
            if data_cfg.scaffolding == "proper_chat":
                messages = row[data_cfg.messages_field]
                text = maybe_apply_chat_template(messages, tokenizer, data_cfg)
            else:
                raw_text = row[data_cfg.text_field]
                text = maybe_apply_chat_template(raw_text, tokenizer, data_cfg)

            # Tokenize with truncation
            enc = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                add_special_tokens=False,
            )
            input_ids = enc["input_ids"]  # (1, T)
            T = input_ids.shape[1]

            # For single documents without padding, attention_mask is all-ones
            attention_mask = torch.ones(1, T, dtype=torch.long)

            # Decode tokens
            token_ids = input_ids[0].tolist()
            tokens = [tokenizer.decode([tid]) for tid in token_ids]

            docs.append({
                "text": text,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "tokens": tokens,
                "doc_idx": doc_idx,
            })

        print(f"  Loaded {len(docs)} documents")
        return docs

    # HuggingFace streaming fallback (TinyStories-style)
    print("Loading TinyStories (HF streaming fallback) ...")
    candidate_stories = load_stories(num_docs, seed)
    stories = filter_by_token_count(
        candidate_stories, tokenizer,
        min_tokens=60, max_tokens=150, num_stories=num_docs,
    )
    print(f"  Selected {len(stories)} stories")

    if len(stories) < num_docs:
        print(f"  WARNING: only found {len(stories)} stories (requested {num_docs})")

    docs = []
    for doc_idx, text in enumerate(stories):
        enc = tokenizer(text, return_tensors="pt")
        input_ids = enc["input_ids"]  # (1, T)
        tokens = [tokenizer.decode([tid]) for tid in input_ids[0].tolist()]
        docs.append({
            "text": text,
            "input_ids": input_ids,
            "attention_mask": None,  # No masking needed for un-padded stories
            "tokens": tokens,
            "doc_idx": doc_idx,
        })

    return docs


# ---------------------------------------------------------------------------
# Representation computation
# ---------------------------------------------------------------------------
def compute_representation(
    repr_name: str,
    h: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    extractor: HiddenStateExtractor,
    encoder: torch.nn.Module | None,
    predictor: Predictor | None,
    token_projector: TokenProjector | None,
    pca_context: dict,
    sae,
    device: str,
) -> tuple[np.ndarray, int]:
    """Compute a single representation from hidden states.

    Parameters
    ----------
    repr_name : str
        One of ``"h"``, ``"z"``, ``"z_hat"``, ``"pca-N"``, ``"sae"``.
    h : torch.Tensor
        ``(1, T, d_h)`` hidden states (already extracted).
    input_ids : torch.Tensor
        ``(1, T)`` token IDs.
    attention_mask : torch.Tensor | None
        ``(1, T)`` attention mask (1=valid, 0=padding), or None.
    extractor : HiddenStateExtractor
        Used only for ``z_hat`` (controlled mode) to get token embeddings.
    encoder : nn.Module | None
        RET encoder; required for ``z`` and ``z_hat``.
    predictor : Predictor | None
        RET predictor; required for ``z_hat``.
    token_projector : TokenProjector | None
        Token projector for controlled-mode ``z_hat``; may be None.
    pca_context : dict
        Mapping ``{pca_dim: (components, mean)}`` populated during PCA pre-pass.
    sae :
        Loaded SAE model; required for ``"sae"``.
    device : str
        Compute device.

    Returns
    -------
    repr_np : np.ndarray
        ``(T_valid, d)`` representation array (before attention mask filtering).
    start_idx : int
        Token offset (0 for h/z/pca/sae, 1 for z_hat).
    """
    with torch.no_grad():
        if repr_name == "h":
            return h[0].cpu().numpy(), 0

        elif repr_name == "z":
            assert encoder is not None, "Encoder required for repr='z'"
            z, aux = encoder(h, attention_mask=attention_mask)
            # For VQ encoders, use continuous z_pre
            z = aux.get("z_pre", z)
            return z[0].cpu().numpy(), 0

        elif repr_name == "z_hat":
            assert encoder is not None, "Encoder required for repr='z_hat'"
            assert predictor is not None, "Predictor required for repr='z_hat'"
            z_full, aux = encoder(h, attention_mask=attention_mask)
            z_full = aux.get("z_pre", z_full)

            z_t = z_full[:, :-1, :]  # (1, T-1, d_z)

            if predictor.mode == "controlled" and token_projector is not None:
                token_emb = extractor.get_token_embeddings(input_ids)  # (1, T, d_h)
                u = token_projector(token_emb)                          # (1, T, d_u)
                u_next = u[:, 1:, :]                                    # (1, T-1, d_u)
                z_hat = predictor(z_t, u_next)                          # (1, T-1, d_z)
            else:
                z_hat = predictor(z_t)                                  # (1, T-1, d_z)

            return z_hat[0].cpu().numpy(), 1

        elif repr_name.startswith("pca-"):
            pca_dim = int(repr_name[4:])
            assert pca_dim in pca_context, (
                f"PCA-{pca_dim} not found in pca_context. "
                "Was the PCA pre-pass run?"
            )
            components, mean = pca_context[pca_dim]  # (pca_dim, d_h), (d_h,)
            h_np = h[0].cpu().numpy()                # (T, d_h)
            z_pca = (h_np - mean) @ components.T     # (T, pca_dim)
            return z_pca, 0

        elif repr_name == "sae":
            assert sae is not None, "SAE required for repr='sae'"
            sae_acts = encode_sae(sae, h[0].cpu())  # (T, d_sae) on CPU
            return sae_acts.numpy(), 0

        else:
            raise ValueError(f"Unknown repr name: {repr_name!r}")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def pairwise_cosine_similarity(X: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine similarity matrix after mean-centering.

    Args:
        X: (T, d) array
    Returns:
        (T, T) cosine similarity matrix
    """
    X = X - X.mean(axis=0, keepdims=True)  # mean-center
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


# ---------------------------------------------------------------------------
# Figure 4a: UMAP + Similarity + Tortuosity
# ---------------------------------------------------------------------------
def plot_umap_similarity(
    representations: list[tuple[str, np.ndarray, list[str]]],
    out_path: Path,
    story_idx: int,
) -> None:
    """N-column figure. Each column: 3D UMAP (top) + cosine sim heatmap (bottom).

    Parameters
    ----------
    representations : list of (label, X, tokens) tuples.
        ``X`` is ``(T, d)``; ``tokens`` is length ``T``.
    """
    n_cols = len(representations)
    fig = plt.figure(figsize=(6 * n_cols, 10))
    fig.suptitle(f"Document {story_idx}: Event Structure (UMAP + Cosine Similarity)", fontsize=14)

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
# Figure 4a (2D): 2D UMAP + Similarity + Tortuosity
# ---------------------------------------------------------------------------
def plot_umap_similarity_2d(
    representations: list[tuple[str, np.ndarray, list[str]]],
    out_path: Path,
    story_idx: int,
) -> None:
    """N-column figure. Each column: 2D UMAP (top) + cosine sim heatmap (bottom).

    Parameters
    ----------
    representations : list of (label, X, tokens) tuples.
        ``X`` is ``(T, d)``; ``tokens`` is length ``T``.
    """
    n_cols = len(representations)
    fig, axes = plt.subplots(2, n_cols, figsize=(6 * n_cols, 10))
    fig.suptitle(f"Document {story_idx}: Event Structure (2D UMAP + Cosine Similarity)", fontsize=14)

    if n_cols == 1:
        axes = axes.reshape(2, 1)

    for col, (label, X, toks) in enumerate(representations):
        T = X.shape[0]

        # 2D UMAP
        n_neighbors = min(15, T - 1)
        reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=0.1, random_state=42)
        coords_2d = reducer.fit_transform(X)

        tau = tortuosity(coords_2d)

        ax = axes[0, col]
        colors = np.arange(T) / max(T - 1, 1)
        sc = ax.scatter(
            coords_2d[:, 0], coords_2d[:, 1],
            c=colors, cmap="viridis", s=30, edgecolors="k", linewidths=0.3,
            zorder=2,
        )
        # Draw path
        ax.plot(
            coords_2d[:, 0], coords_2d[:, 1],
            color="gray", alpha=0.4, linewidth=0.8, zorder=1,
        )
        ax.set_title(f"{label}\n\u03c4 = {tau:.2f}", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="Position")

        # Cosine similarity heatmap
        sim = pairwise_cosine_similarity(X)
        ax_hm = axes[1, col]
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
# Figure 4b: Hierarchical Clustering Dendrograms
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
    story_idx: int,
    story_text: str = "",
) -> None:
    """N-row dendrogram with token labels including position indices.

    Parameters
    ----------
    representations : list of (label, X, tokens, start_idx) tuples.
        ``X`` is ``(T, d)``; ``tokens`` is length ``T``;
        ``start_idx`` is the token offset (0 for h/z, 1 for z_hat).
    """
    n_rows = len(representations)
    max_tokens = max(X.shape[0] for _, X, _, _ in representations)
    fig_width = max(28, max_tokens * 0.25)
    fig, axes = plt.subplots(n_rows, 1, figsize=(fig_width, 7 * n_rows + 1))
    if n_rows == 1:
        axes = [axes]
    fig.suptitle(
        f"Document {story_idx}: Hierarchical Clustering (Cosine Distance)",
        fontsize=14, y=0.99,
    )

    if story_text:
        wrapped = textwrap.fill(story_text.strip(), width=140)
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
# Repr label helper
# ---------------------------------------------------------------------------
def _repr_label(repr_name: str) -> str:
    """Map internal repr name to display label."""
    if repr_name == "h":
        return _LABEL_H
    elif repr_name == "z":
        return _LABEL_Z
    elif repr_name == "z_hat":
        return _LABEL_Z_HAT
    elif repr_name == "sae":
        return _LABEL_SAE
    elif repr_name.startswith("pca-"):
        return _pca_label(int(repr_name[4:]))
    else:
        return repr_name


# ---------------------------------------------------------------------------
# Attention mask helper
# ---------------------------------------------------------------------------
def _apply_mask(
    repr_np: np.ndarray,
    tokens: list[str],
    attention_mask: torch.Tensor | None,
    start_idx: int,
) -> tuple[np.ndarray, list[str]]:
    """Apply attention mask to filter out padding tokens.

    For z_hat (start_idx=1), uses attention_mask[:, 1:] since z_hat has T-1 rows.

    Args:
        repr_np: (T, d) representation array
        tokens: length-T token list (already sliced to match repr_np length)
        attention_mask: (1, T_full) or None
        start_idx: 0 for h/z/pca/sae, 1 for z_hat

    Returns:
        Filtered (T_valid, d) array and token list.
    """
    if attention_mask is None:
        return repr_np, tokens

    mask_np = attention_mask[0].cpu().numpy().astype(bool)  # (T_full,)

    if start_idx == 0:
        valid_mask = mask_np  # (T,)
    else:
        # z_hat has T-1 tokens; its position i corresponds to full-sequence position i+1
        valid_mask = mask_np[1:]  # (T-1,)

    repr_np = repr_np[valid_mask]
    tokens = [t for t, m in zip(tokens, valid_mask) if m]
    return repr_np, tokens


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generalized document event structure evaluation. "
            "Supports any dataset, LLM, and representation set."
        )
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to checkpoint_final.pt. Required if z or z_hat are in --repr.",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help=(
            "Path to config YAML. Inferred from checkpoint directory if omitted. "
            "Required if --checkpoint is not given."
        ),
    )
    parser.add_argument(
        "--repr", nargs="+", default=["h", "z", "z_hat"],
        metavar="REPR",
        help=(
            "Representations to visualize. Options: h, z, z_hat, pca-N (e.g. pca-64), sae. "
            "Default: h z z_hat"
        ),
    )
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Override data directory (Arrow dataset root with train/val/test subdirs).",
    )
    parser.add_argument(
        "--data-split", type=str, default="val",
        choices=["train", "val", "test"],
        help="Which split to load from the Arrow dataset (default: val).",
    )
    parser.add_argument(
        "--num-docs", type=int, default=3,
        help="Number of documents to analyze (default: 3).",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=None,
        help="Truncate documents to at most this many tokens after loading.",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Override output directory. Defaults to <checkpoint_dir>/analysis/ or outputs/doc_event/.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42).",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Compute device (default: cuda).",
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip if analysis already exists in output directory.",
    )
    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # Validate arguments
    # -----------------------------------------------------------------------
    if args.checkpoint is None and args.config is None:
        parser.error("Must provide --checkpoint or --config.")

    needs_encoder = any(r in args.repr for r in ("z", "z_hat"))
    if needs_encoder and args.checkpoint is None:
        parser.error("--checkpoint is required when repr includes 'z' or 'z_hat'.")

    # -----------------------------------------------------------------------
    # Load config
    # -----------------------------------------------------------------------
    if args.checkpoint is not None:
        ckpt_path = Path(args.checkpoint)
        ckpt_dir = ckpt_path.parent
        config_path = ckpt_dir / "config.yaml"
        if not config_path.exists() and args.config is not None:
            config_path = Path(args.config)
        print(f"Loading config from {config_path} ...")
        config = load_config(config_path)
    else:
        config_path = Path(args.config)
        print(f"Loading config from {config_path} ...")
        config = load_config(config_path)
        ckpt_path = None
        ckpt_dir = None

    # -----------------------------------------------------------------------
    # Resolve output directory
    # -----------------------------------------------------------------------
    if args.output_dir is not None:
        out_dir = Path(args.output_dir)
    elif ckpt_dir is not None:
        out_dir = ckpt_dir / "analysis" / "embedding_trajectory_structure"
    else:
        lsn = llm_short_name(config.llm)
        out_dir = Path("outputs") / lsn / "doc_event"

    out_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Skip check
    # -----------------------------------------------------------------------
    if args.skip_existing:
        marker = out_dir / "doc_event_umap_sim_0.png"
        if marker.exists():
            print(f"SKIPPED — analysis already exists ({marker})")
            return

    # -----------------------------------------------------------------------
    # Build extractor (or just tokenizer for cached mode)
    # -----------------------------------------------------------------------
    is_cached = config.data.mode == "cached"

    if is_cached:
        print(f"Cached mode: loading tokenizer only for {config.llm.name} ...")
        tok_kwargs = {}
        if config.llm.trust_remote_code:
            tok_kwargs["trust_remote_code"] = True
        tokenizer = AutoTokenizer.from_pretrained(config.llm.name, **tok_kwargs)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        extractor = None
    else:
        print(f"Building HiddenStateExtractor for {config.llm.name} (layer {config.llm.layer_index}) ...")
        extractor = HiddenStateExtractor(config.llm, device=args.device)
        tokenizer = extractor.tokenizer

    # -----------------------------------------------------------------------
    # Build encoder, predictor, token_projector (if needed)
    # -----------------------------------------------------------------------
    encoder = None
    predictor = None
    token_projector = None

    if needs_encoder:
        print(f"Loading checkpoint from {ckpt_path} ...")
        encoder, config, ckpt = load_encoder(ckpt_path, device=args.device, config=config)

        predictor = Predictor(config.encoder.z_dim, config.predictor).to(args.device)
        try:
            predictor.load_state_dict(ckpt["predictor"])
        except RuntimeError as e:
            warnings.warn(f"Predictor state_dict mismatch, loading with strict=False: {e}")
            info = predictor.load_state_dict(ckpt["predictor"], strict=False)
            if info.missing_keys:
                warnings.warn(f"  Missing keys (randomly initialized): {info.missing_keys}")
            if info.unexpected_keys:
                warnings.warn(f"  Unexpected keys (ignored): {info.unexpected_keys}")
        predictor.eval()

        if config.predictor.mode == "controlled" and "token_projector" in ckpt:
            if is_cached:
                print(
                    "WARNING: Cached mode does not support controlled z_hat "
                    "(no token embeddings available). Using autonomous fallback."
                )
            else:
                token_projector = TokenProjector(config.llm.hidden_dim, config.predictor).to(args.device)
                token_projector.load_state_dict(ckpt["token_projector"])
                token_projector.eval()

    # -----------------------------------------------------------------------
    # Load documents
    # -----------------------------------------------------------------------
    docs = load_documents(
        config=config,
        tokenizer=tokenizer,
        num_docs=args.num_docs,
        seed=args.seed,
        data_dir=args.data_dir,
        data_split=args.data_split,
    )

    if len(docs) == 0:
        print("ERROR: No documents loaded. Aborting.")
        if extractor is not None:
            extractor.cleanup()
        return

    # -----------------------------------------------------------------------
    # PCA pre-pass: fit on all docs before per-doc loop
    # -----------------------------------------------------------------------
    pca_dims = set()
    for r in args.repr:
        if r.startswith("pca-"):
            pca_dims.add(int(r[4:]))

    pca_context: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    if pca_dims:
        print(f"PCA pre-pass: collecting h from {len(docs)} documents ...")
        all_h_flat: list[np.ndarray] = []
        for doc in docs:
            if "hidden_states" in doc:
                h = doc["hidden_states"].to(args.device)
            else:
                input_ids = doc["input_ids"].to(args.device)
                attention_mask = doc["attention_mask"]
                attn_dev = attention_mask.to(args.device) if attention_mask is not None else None
                with torch.no_grad():
                    h = extractor.extract(input_ids, attention_mask=attn_dev)

            h_np = h[0].cpu().numpy()  # (T, d_h)

            # Filter padding tokens
            if attention_mask is not None:
                valid_mask = attention_mask[0].cpu().numpy().astype(bool)
                h_np = h_np[valid_mask]

            all_h_flat.append(h_np)

        h_pooled = np.concatenate(all_h_flat, axis=0)  # (N_total, d_h)
        print(f"  Pooled {h_pooled.shape[0]} tokens for PCA fitting")

        for pca_dim in sorted(pca_dims):
            print(f"  Fitting PCA-{pca_dim} ...")
            comps, mean = fit_pca_on_h(h_pooled, pca_dim)
            pca_context[pca_dim] = (comps, mean)

    # -----------------------------------------------------------------------
    # SAE loading
    # -----------------------------------------------------------------------
    sae = None
    if "sae" in args.repr:
        print("Loading SAE baseline ...")
        sae_result = load_sae_baseline(config.llm, device=args.device)
        if sae_result is None:
            print("WARNING: Could not load SAE. 'sae' repr will be skipped.")
            args.repr = [r for r in args.repr if r != "sae"]
        else:
            sae, _, _ = sae_result

    # -----------------------------------------------------------------------
    # Save document texts
    # -----------------------------------------------------------------------
    texts_file = out_dir / "doc_texts.txt"
    with open(texts_file, "w") as f:
        for doc in docs:
            n_tok = len(doc["tokens"])
            f.write(f"{'=' * 72}\n")
            f.write(f"Document {doc['doc_idx']}  ({n_tok} tokens)\n")
            f.write(f"{'=' * 72}\n")
            f.write(doc["text"].strip() + "\n\n")
    print(f"Saved document texts to {texts_file}")

    # -----------------------------------------------------------------------
    # Per-document analysis loop
    # -----------------------------------------------------------------------
    prompt_logger = PromptLogger(out_dir)

    for doc in docs:
        doc_idx = doc["doc_idx"]
        text = doc["text"]
        input_ids = doc["input_ids"].to(args.device)
        attention_mask = doc["attention_mask"]
        attn_dev = attention_mask.to(args.device) if attention_mask is not None else None
        tokens_full = doc["tokens"]

        n_tok = len(tokens_full)
        print(f"\n--- Document {doc_idx} ({n_tok} tokens) ---")
        preview = text[:80].replace("\n", " ")
        print(f'  "{preview}..."')

        if not prompt_logger.is_full:
            prompt_logger.log_prompts([text], tokenizer, source="document")

        # Extract hidden states once per doc
        if "hidden_states" in doc:
            h = doc["hidden_states"].to(args.device)
        else:
            with torch.no_grad():
                h = extractor.extract(input_ids, attention_mask=attn_dev)

        # Compute all requested representations
        umap_reprs: list[tuple[str, np.ndarray, list[str]]] = []
        dendro_reprs: list[tuple[str, np.ndarray, list[str], int]] = []

        for repr_name in args.repr:
            label = _repr_label(repr_name)

            try:
                repr_np, start_idx = compute_representation(
                    repr_name=repr_name,
                    h=h,
                    input_ids=input_ids,
                    attention_mask=attn_dev,
                    extractor=extractor,
                    encoder=encoder,
                    predictor=predictor,
                    token_projector=token_projector,
                    pca_context=pca_context,
                    sae=sae,
                    device=args.device,
                )
            except Exception as e:
                print(f"  WARNING: failed to compute repr={repr_name!r}: {e}")
                continue

            # Slice tokens to match repr length (z_hat has T-1 rows)
            if start_idx == 1:
                tokens_repr = tokens_full[1:]
            else:
                tokens_repr = tokens_full

            # Apply attention mask to filter padding tokens
            repr_np, tokens_repr = _apply_mask(repr_np, tokens_repr, attention_mask, start_idx)

            # Apply --max-tokens truncation
            if args.max_tokens is not None and repr_np.shape[0] > args.max_tokens:
                repr_np = repr_np[: args.max_tokens]
                tokens_repr = tokens_repr[: args.max_tokens]

            # Need at least 2 tokens for UMAP / dendrogram
            if repr_np.shape[0] < 2:
                print(f"  WARNING: repr={repr_name!r} has <2 valid tokens, skipping.")
                continue

            umap_reprs.append((label, repr_np, tokens_repr))
            dendro_reprs.append((label, repr_np, tokens_repr, start_idx))

        if not umap_reprs:
            print(f"  WARNING: No representations computed for document {doc_idx}. Skipping plots.")
            continue

        # Generate plots
        plot_umap_similarity(
            umap_reprs,
            out_dir / f"doc_event_umap_sim_{doc_idx}.png",
            story_idx=doc_idx,
        )
        plot_umap_similarity_2d(
            umap_reprs,
            out_dir / f"doc_event_umap_sim_2d_{doc_idx}.png",
            story_idx=doc_idx,
        )
        plot_dendrograms(
            dendro_reprs,
            out_dir / f"doc_event_dendrogram_{doc_idx}.png",
            story_idx=doc_idx,
            story_text=text,
        )

    prompt_logger.save()
    if extractor is not None:
        extractor.cleanup()
    print(f"\nDone. All plots saved to {out_dir}")


if __name__ == "__main__":
    main()
