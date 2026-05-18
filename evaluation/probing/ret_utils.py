"""
Utilities for loading RET encoders and transforming hidden states h -> z.

Used by probing experiment runners when --ret_checkpoint is provided.
The encoder operates on full token sequences, then pooling is applied to z.

Public API
----------
load_ret_encoder(checkpoint_path, device)
    Load a trained RET encoder from a checkpoint file.

apply_ret_encoder(encoder, hs_all_tokens, modes, device, batch_size)
    Encode full-sequence hidden states through RET, then pool.
"""
from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Set, Tuple

import numpy as np
import torch
import torch.nn as nn

from ret.config import ExperimentConfig
from ret.utils.checkpointing import load_encoder as _load_encoder

logger = logging.getLogger(__name__)


def load_ret_encoder(
    checkpoint_path: str,
    device: str = "cuda",
) -> Tuple[nn.Module, ExperimentConfig]:
    """Load a trained RET encoder from a checkpoint file.

    Looks for config.yaml in the same directory as the checkpoint.

    Parameters
    ----------
    checkpoint_path : path to checkpoint_final.pt (or any .pt checkpoint)
    device : device to place the encoder on

    Returns
    -------
    encoder : the loaded encoder module in eval mode
    config  : the ExperimentConfig saved alongside the checkpoint
    """
    encoder, config, _ckpt = _load_encoder(checkpoint_path, device=device)

    logger.info(
        "Loaded RET encoder from %s (type=%s, z_dim=%d, hidden_dim=%d)",
        checkpoint_path,
        config.encoder.type,
        config.encoder.z_dim,
        config.llm.hidden_dim,
    )
    return encoder, config


def free_ret_encoder(encoder: nn.Module) -> None:
    """Delete RET encoder from GPU and empty CUDA cache."""
    del encoder
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("RET encoder freed from GPU.")


@torch.no_grad()
def apply_ret_encoder(
    encoder: nn.Module,
    hs_all_tokens: dict,
    modes: Set[str],
    device: str = "cuda",
    batch_size: int = 16,
    system_prefix_len: int = 0,
    include_system_in_ret: bool = True,
) -> dict:
    """Encode full-sequence hidden states through RET, then pool.

    The RET encoder operates on token sequences.  Two separate masks
    control (a) which tokens the encoder *sees* and (b) which z positions
    are used for downstream probe pooling:

    * **RET input mask**: When ``include_system_in_ret=True`` (default),
      the encoder sees ALL real tokens (system + user).  When False,
      system-prefix tokens are masked out before encoding.
    * **Probe pooling mask**: Always excludes system-prefix tokens — probe
      average / assistant_token / all_tokens modes only use user-input z.

    Parameters
    ----------
    encoder       : loaded RET encoder in eval mode
    hs_all_tokens : dict with "hidden_states" [N, S, D] and
                    "attention_mask" [N, S]
    modes         : set of requested modes, subset of
                    {"average", "assistant_token", "all_tokens"}
    device        : device for encoder inference
    batch_size    : batch size for encoding
    system_prefix_len : number of system-prompt prefix tokens to exclude
                        from probe pooling (0 = use all tokens, backward compat)
    include_system_in_ret : if True, RET encoder sees all tokens including
                            system prompt; if False, system tokens are masked
                            out before encoding

    Returns
    -------
    z_cache : dict with mode keys mapping to pooled z representations.
              "average"         -> np.ndarray [N, z_dim]
              "assistant_token" -> np.ndarray [N, z_dim]
              "all_tokens"      -> dict{"hidden_states": [N,S,z_dim],
                                        "attention_mask": [N,S]}
    """
    h = hs_all_tokens["hidden_states"]   # [N, S, D]
    # Use original (full) mask when available — repool_from_all_tokens stores
    # user-only mask in "attention_mask" but preserves the full one in
    # "original_attention_mask" so RET can still include system tokens.
    full_mask = hs_all_tokens.get("original_attention_mask",
                                  hs_all_tokens["attention_mask"])  # [N, S]
    probe_mask = hs_all_tokens["attention_mask"]  # [N, S] — may be user-only

    # --- Input validation ---
    assert h.ndim == 3, f"hidden_states must be 3D [N,S,D], got {h.shape}"
    assert full_mask.ndim == 2, f"attention_mask must be 2D [N,S], got {full_mask.shape}"
    assert h.shape[:2] == full_mask.shape, \
        f"shape mismatch: h {h.shape[:2]} vs mask {full_mask.shape}"
    assert system_prefix_len >= 0, \
        f"system_prefix_len must be >= 0, got {system_prefix_len}"
    if system_prefix_len > 0:
        assert system_prefix_len < h.shape[1], \
            f"system_prefix_len ({system_prefix_len}) >= seq_len ({h.shape[1]})"
    N, S, D = h.shape

    # --- RET input mask ---
    if include_system_in_ret or system_prefix_len <= 0:
        ret_mask = full_mask
    else:
        from evaluation.probing.model_utils import make_user_token_mask
        ret_mask = make_user_token_mask(full_mask, system_prefix_len)
        logger.info("RET forward: excluding %d system-prefix tokens", system_prefix_len)

    # Encode all sequences through the RET encoder
    all_z = []
    for start in range(0, len(h), batch_size):
        chunk_h = torch.from_numpy(h[start : start + batch_size]).to(device)
        chunk_mask = torch.from_numpy(
            ret_mask[start : start + batch_size].astype(np.float32)
        ).to(device)
        z, _aux = encoder(chunk_h, attention_mask=chunk_mask)
        all_z.append(z.cpu().numpy())
    z_full = np.concatenate(all_z, axis=0)  # [N, S, z_dim]

    # --- Post-encoding validation ---
    assert z_full.shape[0] == N, \
        f"z_full N mismatch: expected {N}, got {z_full.shape[0]}"
    assert z_full.shape[1] == S, \
        f"z_full S mismatch: expected {S}, got {z_full.shape[1]}"
    z_dim = z_full.shape[2]
    assert not np.any(np.isnan(z_full)), \
        f"RET output contains {np.sum(np.isnan(z_full))} NaN values"

    logger.info(
        "RET encoded: h %s -> z %s (system_prefix_len=%d, include_system=%s)",
        h.shape, z_full.shape, system_prefix_len, include_system_in_ret,
    )

    # --- Probe pooling mask: user tokens only ---
    # If repooling has already been done, probe_mask is already user-only.
    # Otherwise, derive it from full_mask + system_prefix_len.
    has_original = "original_attention_mask" in hs_all_tokens
    if has_original:
        pool_mask = probe_mask
    elif system_prefix_len > 0:
        from evaluation.probing.model_utils import make_user_token_mask
        pool_mask = make_user_token_mask(full_mask, system_prefix_len)
        real_per_sample = full_mask.sum(axis=1)
        pool_per_sample = pool_mask.sum(axis=1)
        expected_user = np.clip(real_per_sample - system_prefix_len, 0, None)
        assert np.allclose(pool_per_sample, expected_user), \
            "pool_mask token counts don't match expected user token counts"
    else:
        pool_mask = full_mask

    z_cache: dict = {}

    if "average" in modes:
        pm = pool_mask[..., np.newaxis].astype(np.float32)  # [N, S, 1]
        z_avg = (z_full * pm).sum(axis=1) / pm.sum(axis=1).clip(min=1)
        z_cache["average"] = z_avg  # [N, z_dim]

    if "assistant_token" in modes:
        seq_lens = np.maximum(full_mask.sum(axis=1).astype(int) - 1, 0)  # [N]
        z_last = np.array([z_full[i, seq_lens[i]] for i in range(len(z_full))])
        z_cache["assistant_token"] = z_last  # [N, z_dim]

    if "all_tokens" in modes:
        z_cache["all_tokens"] = {
            "hidden_states": z_full,
            "attention_mask": pool_mask,
        }

    # --- Output validation ---
    for mode_key, val in z_cache.items():
        if isinstance(val, dict):
            assert val["hidden_states"].shape == (N, S, z_dim), \
                f"z_cache['{mode_key}'].hidden_states shape {val['hidden_states'].shape} " \
                f"!= expected ({N}, {S}, {z_dim})"
            assert val["attention_mask"].shape == (N, S), \
                f"z_cache['{mode_key}'].attention_mask shape {val['attention_mask'].shape} " \
                f"!= expected ({N}, {S})"
        else:
            assert val.shape == (N, z_dim), \
                f"z_cache['{mode_key}'] shape {val.shape} != expected ({N}, {z_dim})"
            assert not np.any(np.isnan(val)), \
                f"z_cache['{mode_key}'] contains NaN"

    return z_cache
