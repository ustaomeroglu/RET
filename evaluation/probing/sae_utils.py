"""
Utilities for loading SAE encoders and transforming hidden states h -> sparse features.

Analogous to ret_utils.py — loads a pre-trained Sparse Autoencoder, encodes
cached hidden states token-by-token, then pools into average / assistant_token
modes for downstream probing.

Public API
----------
load_sae_encoder(checkpoint_path_or_hf_repo, device)
    Load a trained SAE encoder (from local .pt or HuggingFace repo).

apply_sae_encoder(sae, hs_all_tokens, modes, device, batch_size, ...)
    Encode full-sequence hidden states through SAE, then pool.

free_sae_encoder(sae)
    Delete SAE from GPU and empty CUDA cache.
"""
from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Dict, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn

from evaluation.baselines.sae import TopKSAE, _load_sae_from_state_dict, DEFAULT_SAE_HF_REPO, DEFAULT_SAE_FILENAME, DEFAULT_SAE_K

logger = logging.getLogger(__name__)


def load_sae_encoder(
    checkpoint_path: Optional[str] = None,
    hf_repo: Optional[str] = None,
    device: str = "cuda",
) -> Tuple[TopKSAE, Dict]:
    """Load a pre-trained SAE encoder.

    Supports two loading modes:
    1. Local .pt checkpoint (from our own SAE training or downloaded HF model)
    2. HuggingFace repo (auto-downloads the checkpoint)

    The checkpoint is expected to contain an OrderedDict with keys like:
        encoder.weight, encoder.bias, b_dec (for TopK SAEs from Goodfire/EleutherAI)
    OR a full Autoencoder state dict from our sparse_autoencoder codebase.

    Parameters
    ----------
    checkpoint_path : path to local .pt checkpoint (preferred)
    hf_repo         : HuggingFace repo ID (used if checkpoint_path is None)
    device          : device to place the encoder on

    Returns
    -------
    sae    : loaded TopKSAE module in eval mode
    config : dict with metadata (d_in, d_sae, k, source)
    """
    if checkpoint_path is not None:
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"SAE checkpoint not found: {ckpt_path}")
        state_dict = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        source = str(ckpt_path)
    else:
        repo = hf_repo or DEFAULT_SAE_HF_REPO
        from huggingface_hub import hf_hub_download
        local_path = hf_hub_download(repo, DEFAULT_SAE_FILENAME)
        state_dict = torch.load(local_path, map_location="cpu", weights_only=False)
        source = f"hf://{repo}/{DEFAULT_SAE_FILENAME}"

    # --- Detect checkpoint format and extract weights ---
    sae = _load_sae_from_state_dict(state_dict, device)
    config = {
        "d_in": sae.d_in,
        "d_sae": sae.d_sae,
        "k": sae.k,
        "source": source,
    }

    logger.info(
        "Loaded SAE encoder: d_in=%d, d_sae=%d, k=%d, source=%s",
        sae.d_in, sae.d_sae, sae.k, source,
    )
    return sae, config


def free_sae_encoder(sae: nn.Module) -> None:
    """Delete SAE encoder from GPU and empty CUDA cache."""
    del sae
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("SAE encoder freed from GPU.")


@torch.no_grad()
def apply_sae_encoder(
    sae: TopKSAE,
    hs_all_tokens: dict,
    modes: Set[str],
    device: str = "cuda",
    batch_size: int = 512,
    system_prefix_len: int = 0,
) -> dict:
    """Encode full-sequence hidden states through SAE, then pool.

    Pools in a streaming fashion — processes one sample at a time so we never
    materialize the full [N, S, d_sae] array (which would be 100+ GB for
    large datasets with d_sae=46080).

    Parameters
    ----------
    sae           : loaded TopKSAE encoder in eval mode
    hs_all_tokens : dict with "hidden_states" [N, S, D] and
                    "attention_mask" [N, S]
    modes         : set of requested modes, subset of
                    {"average", "assistant_token"}
    device        : device for SAE inference
    batch_size    : number of tokens to encode per GPU batch
    system_prefix_len : number of system-prompt prefix tokens to exclude
                        from probe pooling (0 = use all tokens)

    Returns
    -------
    sae_cache : dict with mode keys mapping to pooled SAE representations.
                "average"         -> np.ndarray [N, d_sae]
                "assistant_token" -> np.ndarray [N, d_sae]
    """
    h = hs_all_tokens["hidden_states"]   # [N, S, D]
    full_mask = hs_all_tokens.get("original_attention_mask",
                                   hs_all_tokens["attention_mask"])  # [N, S]
    probe_mask = hs_all_tokens["attention_mask"]  # [N, S] — may be user-only

    # --- Input validation ---
    assert h.ndim == 3, f"hidden_states must be 3D [N,S,D], got {h.shape}"
    assert full_mask.ndim == 2, f"attention_mask must be 2D [N,S], got {full_mask.shape}"
    assert h.shape[:2] == full_mask.shape, \
        f"shape mismatch: h {h.shape[:2]} vs mask {full_mask.shape}"
    N, S, D = h.shape
    d_sae = sae.d_sae

    assert D == sae.d_in, \
        f"Hidden dim mismatch: h has D={D} but SAE expects d_in={sae.d_in}"

    # --- Probe pooling mask ---
    has_original = "original_attention_mask" in hs_all_tokens
    if has_original:
        pool_mask = probe_mask
    elif system_prefix_len > 0:
        from evaluation.probing.model_utils import make_user_token_mask
        pool_mask = make_user_token_mask(full_mask, system_prefix_len)
    else:
        pool_mask = full_mask

    # --- Pre-allocate output arrays (only pooled, not full [N,S,d_sae]) ---
    want_avg = "average" in modes
    want_last = "assistant_token" in modes
    if want_avg:
        sae_avg = np.zeros((N, d_sae), dtype=np.float32)
    if want_last:
        sae_last = np.zeros((N, d_sae), dtype=np.float32)

    # --- Stream: encode + pool one sample at a time ---
    for i in range(N):
        h_i = h[i]            # [S, D]
        pm_i = pool_mask[i]   # [S]
        fm_i = full_mask[i]   # [S]

        # Encode tokens in sub-batches to limit GPU memory
        acts_chunks = []
        for start in range(0, S, batch_size):
            end = min(start + batch_size, S)
            chunk = torch.from_numpy(h_i[start:end]).to(device, dtype=torch.float32)
            acts = sae.encode(chunk)  # [chunk_len, d_sae]
            acts_chunks.append(acts.cpu().numpy())
        sae_i = np.concatenate(acts_chunks, axis=0)  # [S, d_sae]

        if want_avg:
            mask_f = pm_i.astype(np.float32)  # [S]
            n_tokens = mask_f.sum()
            if n_tokens > 0:
                sae_avg[i] = (sae_i * mask_f[:, np.newaxis]).sum(axis=0) / n_tokens

        if want_last:
            seq_len = max(int(fm_i.sum()) - 1, 0)
            sae_last[i] = sae_i[seq_len]

        if (i + 1) % 200 == 0 or i == N - 1:
            logger.info("  SAE encode+pool: %d/%d samples", i + 1, N)

    logger.info(
        "SAE encoded: h %s -> pooled (d_sae=%d, k=%d)",
        h.shape, d_sae, sae.k,
    )

    sae_cache: dict = {}
    if want_avg:
        sae_cache["average"] = sae_avg
    if want_last:
        sae_cache["assistant_token"] = sae_last

    # --- Output validation ---
    for mode_key, val in sae_cache.items():
        assert val.shape == (N, d_sae), \
            f"sae_cache['{mode_key}'] shape {val.shape} != expected ({N}, {d_sae})"
        assert not np.any(np.isnan(val)), \
            f"sae_cache['{mode_key}'] contains NaN"

    return sae_cache
