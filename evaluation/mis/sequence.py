"""Per-token sequence builders for interp score.

RET: hidden → z (via trained encoder) → window+stride → preprocessor.transform
     → kmeans.predict → (cluster_id, group_id) per token.
SAE: hidden → topK latent indices per token.

Both paths return token strings aligned with the ID sequence, plus a formatted
text representation suitable for an LLM prompt.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# RET inference
# ---------------------------------------------------------------------------

@dataclass
class RETInference:
    """Holds loaded RET + clustering artifacts for per-sample inference.

    Keeps everything on CPU/GPU in eval mode; callers pass hidden_states +
    input_ids and get back (tokens, cluster_ids, group_ids).
    """
    encoder: torch.nn.Module
    preprocessor: object                # evaluation.clustering.core.StreamingPreprocessor
    kmeans: object                      # MiniBatchKMeans
    group_labels: np.ndarray            # (K,) cluster_id -> group_id
    tokenizer: object
    window_size: int
    stride: int
    device: str

    @torch.no_grad()
    def sequence_for_sample(
        self,
        hidden_states: torch.Tensor,    # (T, D) float
        input_ids: torch.Tensor,        # (T,) int
    ) -> Tuple[List[str], np.ndarray, np.ndarray]:
        from evaluation.clustering.core import (
            apply_window_and_stride,
            predict_with_scores,
        )

        T = int(hidden_states.shape[0])

        # Encoder expects (B, T, D) → (B, T, z_dim)
        h = hidden_states.unsqueeze(0).to(self.device, dtype=torch.float32)
        mask = torch.ones((1, T), dtype=torch.long, device=h.device)
        z, _ = self.encoder(h, attention_mask=mask)

        # Windowing + stride (reuses the same function the clustering pipeline used)
        windowed, doc_lens = apply_window_and_stride(
            z, mask, self.window_size, self.stride,
        )
        assert len(doc_lens) == 1, f"expected 1 doc, got {len(doc_lens)}"
        T_out = int(doc_lens[0])

        preprocessed = self.preprocessor.transform(windowed)
        labels, _, _ = predict_with_scores(
            self.kmeans, preprocessed, raw=self.preprocessor.raw,
        )
        labels = np.asarray(labels[:T_out], dtype=np.int64)
        groups = self.group_labels[labels].astype(np.int64)

        # Token strings: align with the windowed/stride output — for w=1 stride=1
        # this matches T one-to-one. For w>1 the window reduces token count.
        ids_list = input_ids.tolist()
        if len(ids_list) == T_out:
            tokens = [self.tokenizer.decode([tid]) for tid in ids_list]
        else:
            # When windowing reduces length, align to the tail (matches apply_window_and_stride)
            keep = ids_list[-T_out:]
            tokens = [self.tokenizer.decode([tid]) for tid in keep]

        return tokens, labels, groups


def format_ret_sequence(
    tokens: List[str],
    labels: np.ndarray,
    groups: np.ndarray,
) -> str:
    """Produce 'G2.C5, G2.C5, G3.C15, ...' (one entry per token)."""
    parts = [f"G{int(g)}.C{int(c)}" for g, c in zip(groups, labels)]
    return ", ".join(parts)


def find_assistant_response_start(tokens: List[str]) -> Optional[int]:
    """Return the token index where the assistant's generated response starts.

    Looks for the first '<|start|>' immediately followed by 'assistant'.
    Returns None if the pattern is not present.
    """
    for i in range(len(tokens) - 1):
        if tokens[i] == "<|start|>" and tokens[i + 1] == "assistant":
            return i
    return None


def format_ret_sequence_rle(
    tokens: List[str],
    labels: np.ndarray,
    groups: np.ndarray,
) -> str:
    """Run-length encode the RET sequence as phase events with durations.

    Output header describes the row format; each row is
    ``[tokens a-b | N toks | G<g>.C<c>]``.
    """
    lines: List[str] = []
    lines.append("# RET phase sequence (run-length encoded).")
    lines.append(
        "# Each row: [tokens a-b | N toks | G<group>.C<cluster>] "
        "— the mental-state phase held during that token range."
    )
    lines.append("")
    T = len(labels)
    if T == 0:
        return "\n".join(lines)

    i = 0
    while i < T:
        j = i + 1
        while j < T and int(labels[j]) == int(labels[i]) and int(groups[j]) == int(groups[i]):
            j += 1
        run_len = j - i
        lines.append(
            f"[tokens {i}-{j - 1} | {run_len} toks | G{int(groups[i])}.C{int(labels[i])}]"
        )
        i = j
    return "\n".join(lines)


def format_ret_sequence_windowed_group1(
    tokens: List[str],
    labels: np.ndarray,
    groups: np.ndarray,
    window_size: int,
) -> str:
    """Chunk into fixed windows and emit only the single most-represented
    Group ID per window (by duration).

    Output: one row per window of the form '[tokens a-b | G<g>]'. No cluster
    info is shown — deliberately handicaps RET relative to the SAE windowed
    format.
    """
    T = len(labels)
    lines: List[str] = []
    lines.append(
        f"# RET windowed phase sequence (window={window_size} tokens, "
        "single most-represented Group per window)."
    )
    lines.append(
        "# Each row: [tokens a-b | G<group>] — the group-level mental-state "
        "phase that the model spent the most tokens in during that window."
    )
    lines.append("")
    if T == 0:
        return "\n".join(lines)

    num_windows = (T + window_size - 1) // window_size
    for w in range(num_windows):
        start = w * window_size
        end = min(start + window_size, T)
        win_groups = groups[start:end]
        counts: Dict[int, int] = {}
        for g in win_groups:
            g = int(g)
            counts[g] = counts.get(g, 0) + 1
        top_g = max(counts.items(), key=lambda kv: kv[1])[0]
        lines.append(f"[tokens {start}-{end - 1} | G{top_g}]")
    return "\n".join(lines)


def format_ret_sequence_windowed(
    tokens: List[str],
    labels: np.ndarray,
    groups: np.ndarray,
    window_size: int,
    top_k_out: int,
) -> str:
    """Chunk the per-token RET labels into fixed windows and within each
    window report the top ``top_k_out`` clusters ranked by duration
    (number of tokens in the window spent in that cluster).

    Format mirrors ``format_sae_sequence_windowed`` for parallel structure.
    """
    T = len(labels)
    lines: List[str] = []
    lines.append(
        f"# RET windowed phase sequence (window={window_size} tokens, "
        f"top-{top_k_out} clusters per window by duration)."
    )
    lines.append(
        "# Each row: [tokens a-b | G<g>.C<c>(N toks), G<g>.C<c>(N toks), ...] "
        "— mental-state clusters that the model spent the most tokens in "
        "during that window."
    )
    lines.append("")
    if T == 0:
        return "\n".join(lines)

    num_windows = (T + window_size - 1) // window_size
    for w in range(num_windows):
        start = w * window_size
        end = min(start + window_size, T)
        win_labels = labels[start:end]
        win_groups = groups[start:end]
        # Count durations per (group, cluster) pair.
        keys = [(int(g), int(c)) for g, c in zip(win_groups, win_labels)]
        counts: Dict[tuple, int] = {}
        for k in keys:
            counts[k] = counts.get(k, 0) + 1
        picks = sorted(counts.items(), key=lambda kv: -kv[1])[:top_k_out]
        body = ", ".join(
            f"G{gc[0]}.C{gc[1]}({n} toks)" for gc, n in picks
        )
        lines.append(f"[tokens {start}-{end - 1} | {body}]")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# SAE inference
# ---------------------------------------------------------------------------

@dataclass
class SAEInference:
    sae: torch.nn.Module
    tokenizer: object
    top_k: int
    device: str

    @torch.no_grad()
    def sequence_for_sample(
        self,
        hidden_states: torch.Tensor,    # (T, D)
        input_ids: torch.Tensor,        # (T,)
    ) -> Tuple[List[str], np.ndarray, np.ndarray]:
        """Return (tokens, top_k_latent_indices [T, top_k], top_k_values [T, top_k]).

        Values have ReLU applied (negative pre-activations zeroed), matching
        how the SAE would use them.
        """
        sae = self.sae
        h = hidden_states.to(self.device, dtype=torch.float32)
        centered = h - sae.b_dec
        pre = centered @ sae.encoder_weight.T + sae.encoder_bias
        topk_vals, topk_idx = pre.topk(self.top_k, dim=-1)
        topk_vals = torch.relu(topk_vals)
        topk_idx = topk_idx.cpu().numpy().astype(np.int64)   # (T, top_k)
        topk_vals_np = topk_vals.cpu().numpy().astype(np.float32)  # (T, top_k)

        ids_list = input_ids.tolist()
        tokens = [self.tokenizer.decode([tid]) for tid in ids_list]
        return tokens, topk_idx, topk_vals_np


def format_sae_sequence(
    tokens: List[str],
    top_k_latents: np.ndarray,          # (T, K)
) -> str:
    """Produce '[L42, L101, L15], [L42, L17, L15], ...' one entry per token."""
    parts = []
    for row in top_k_latents:
        ids = ",".join(f"L{int(i)}" for i in row)
        parts.append(f"[{ids}]")
    return ", ".join(parts)


def format_sae_sequence_windowed(
    tokens: List[str],
    top_k_idx: np.ndarray,      # (T, K)
    top_k_vals: np.ndarray,     # (T, K)
    window_size: int,
    top_k_out: int,
) -> Tuple[str, np.ndarray]:
    """Chunk the per-token SAE activations into fixed windows, and within each
    window report the ``top_k_out`` latents ranked by cumulative activation.

    Returns (formatted_string, active_latent_ids) where active_latent_ids is
    the set of latent IDs that appear in ANY window's output (used to filter
    the naming table).
    """
    T = top_k_idx.shape[0]
    lines: List[str] = []
    lines.append(
        f"# SAE windowed activation sequence (window={window_size} tokens, "
        f"top-{top_k_out} latents per window ranked by cumulative activation)."
    )
    lines.append(
        "# Each row: [tokens a-b | L<id1>(sum=..), L<id2>(sum=..), ...] "
        "— latents that were most strongly active across that token window."
    )
    lines.append("")

    active_ids: set = set()
    num_windows = (T + window_size - 1) // window_size

    for w in range(num_windows):
        start = w * window_size
        end = min(start + window_size, T)
        # For this window, accumulate activation sums per latent ID.
        win_idx = top_k_idx[start:end]           # (W, K)
        win_vals = top_k_vals[start:end]         # (W, K)
        flat_idx = win_idx.reshape(-1)
        flat_vals = win_vals.reshape(-1)
        # Group-by latent id using numpy.
        unique_ids, inverse = np.unique(flat_idx, return_inverse=True)
        sums = np.zeros(unique_ids.shape[0], dtype=np.float32)
        np.add.at(sums, inverse, flat_vals)
        # Take top `top_k_out` by cumulative activation.
        order = np.argsort(-sums)[:top_k_out]
        picks = [(int(unique_ids[i]), float(sums[i])) for i in order]
        for lid, _ in picks:
            active_ids.add(lid)
        body = ", ".join(f"L{lid}(sum={s:.1f})" for lid, s in picks)
        lines.append(f"[tokens {start}-{end - 1} | {body}]")

    return "\n".join(lines), np.array(sorted(active_ids), dtype=np.int64)


# ---------------------------------------------------------------------------
# Loader helpers (glue: clustering_meta → RETInference ; sae ckpt → SAEInference)
# ---------------------------------------------------------------------------

def load_ret_inference(
    clustering_dir: Path | str,
    device: str = "cuda",
    stride: int = 1,
) -> Tuple[RETInference, object]:
    """Materialise an RETInference from a `.../grouped_alpha<alpha>` directory.

    Walks up from the grouping dir to find:
        .../clustering/<centers_dir>/clustering_meta.json
        .../w{W}/K{K}/seed{S}/kmeans_model.pkl
        .../w{W}/mu.npy (+ preprocessor state)
        .../w{W}/K{K}/seed{S}/analysis/weighted_tau<T>/grouped_alpha<A>/group_labels.npy
    Returns (RETInference, config).
    """
    from evaluation.clustering.config import load_config_with_overlay
    from evaluation.clustering.core import StreamingPreprocessor, load_kmeans

    grouping_dir = Path(clustering_dir).resolve()
    if not grouping_dir.exists():
        raise FileNotFoundError(f"clustering_dir not found: {grouping_dir}")

    # Expected shape: .../clustering/w{W}/K{K}/seed{S}/analysis/weighted_tau{T}/grouped_alpha{A}
    parts = grouping_dir.parts
    # Walk upwards to extract w/K/seed
    try:
        grp_idx = [i for i, p in enumerate(parts) if p.startswith("grouped_alpha")][-1]
        tau_idx = [i for i, p in enumerate(parts) if p.startswith("weighted_tau")][-1]
        analysis_idx = parts.index("analysis", 0, tau_idx)
        seed_dir = Path(*parts[: analysis_idx])                   # .../seed{S}
        K_dir = seed_dir.parent                                   # .../K{K}
        w_dir = K_dir.parent                                      # .../w{W}
        centers_dir = w_dir.parent                                # .../clustering
    except (ValueError, IndexError):
        raise ValueError(
            f"Unexpected clustering path layout: {grouping_dir}. "
            "Expected .../clustering/w{W}/K{K}/seed{S}/analysis/weighted_tau{T}/grouped_alpha{A}."
        )

    seed_name = seed_dir.name  # "seed1"
    K_name = K_dir.name        # "K64"
    w_name = w_dir.name        # "w1"
    window_size = int(w_name[1:])

    meta_path = centers_dir / "clustering_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"clustering_meta.json not found at {meta_path}")
    meta = json.loads(meta_path.read_text())
    variant = meta["variant"]
    if not variant.startswith("z"):
        raise ValueError(
            f"load_ret_inference only handles z variants; got variant={variant!r}"
        )
    config_yaml = meta["config_yaml_path"]
    checkpoint = meta.get("checkpoint_path")
    if not checkpoint:
        raise ValueError("clustering_meta.json missing checkpoint_path for z variant")

    config = load_config_with_overlay(config_yaml)

    # Build + load encoder
    from ret.models.encoders import build_encoder

    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    li = config.llm.layer_index
    if li == "all":
        dp_key = "down_proj.0.weight"
        concat_dim = ckpt["encoder"][dp_key].shape[1] if dp_key in ckpt["encoder"] else None
    elif isinstance(li, list) and len(li) > 1:
        concat_dim = len(li) * config.llm.hidden_dim
    else:
        concat_dim = None
    encoder = build_encoder(config.llm.hidden_dim, config.encoder, concat_dim=concat_dim).to(device)
    encoder.load_state_dict(ckpt["encoder"])
    encoder.eval()

    # Preprocessor (variant, w)
    preprocessor = StreamingPreprocessor.load(w_dir, variant, window_size, stride)

    # K-means
    kmeans = load_kmeans(seed_dir)

    # Group labels
    gl_path = grouping_dir / "group_labels.npy"
    if not gl_path.exists():
        raise FileNotFoundError(f"group_labels.npy not found: {gl_path}")
    group_labels = np.load(gl_path)

    # Tokenizer (extractor not needed — we use cached hiddens)
    from transformers import AutoTokenizer

    tok_kwargs: dict = {}
    if config.llm.trust_remote_code:
        tok_kwargs["trust_remote_code"] = True
    tokenizer = AutoTokenizer.from_pretrained(config.llm.name, **tok_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    infer = RETInference(
        encoder=encoder,
        preprocessor=preprocessor,
        kmeans=kmeans,
        group_labels=group_labels,
        tokenizer=tokenizer,
        window_size=window_size,
        stride=stride,
        device=device,
    )
    return infer, config


def load_sae_inference(
    sae_checkpoint: Optional[str],
    llm_name_for_tokenizer: str,
    trust_remote_code: bool,
    top_k: int,
    device: str = "cuda",
) -> SAEInference:
    from evaluation.probing.sae_utils import load_sae_encoder
    from transformers import AutoTokenizer

    sae, sae_cfg = load_sae_encoder(checkpoint_path=sae_checkpoint, device=device)
    logger.info(
        "SAE loaded: d_in=%d, d_sae=%d, k=%d, source=%s",
        sae_cfg["d_in"], sae_cfg["d_sae"], sae_cfg["k"], sae_cfg["source"],
    )

    tok_kwargs = {"trust_remote_code": True} if trust_remote_code else {}
    tokenizer = AutoTokenizer.from_pretrained(llm_name_for_tokenizer, **tok_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return SAEInference(sae=sae, tokenizer=tokenizer, top_k=top_k, device=device)
