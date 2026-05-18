"""
Hidden-state disk caching: save and load extracted hidden states + metadata.

Saves per-experiment .pt files so that LLM forward passes and generation
are only done once.  Subsequent runs load from disk.

File layout per experiment (e.g. sycophancy)::

    cache_dir/
      sycophancy/
        debate.pt
        false_presupposition.pt
        ...

Each .pt file contains::

    {
        "hidden_states": {
            "average":         Tensor [N, D],
            "assistant_token": Tensor [N, D],
            "all_tokens": {
                "hidden_states": Tensor [N, S, D],
                "attention_mask": Tensor [N, S],
            },
        },
        "labels":   Tensor [N],          # integer labels
        "metadata": [...],               # list of per-sample dicts
        "dataset_name": str,
        "experiment":   str,
        "system_prefix_len": int,        # tokens in system prompt prefix (0 if unknown)
    }

For models_under_pressure (single dataset, multiple subtask labels)::

    {
        "hidden_states": { ... },        # same structure
        "labels": {
            "tone":         Tensor [N],
            "prompt_style": Tensor [N],
            "high_stakes":  Tensor [N],
        },
        "metadata": [...],
        "dataset_name": "pressure",
        "experiment":   "models_under_pressure",
        "system_prefix_len": int,
    }

Memmap format (written by cache_hidden_states.py --dataset mode)::

    cache_dir/
        metadata.json
        offsets.npy           # (num_docs,) int64
        lengths.npy           # (num_docs,) int32
        hidden_states.bin     # memmap (total_tokens, hidden_dim)
        input_ids.bin         # memmap (total_tokens,) int32
        labels.npy / labels.npz   # optional

Use ``load_hidden_states_from_memmap(cache_dir)`` to read this format and
pool per-document representations on the fly.
``load_hidden_states()`` auto-detects the format (memmap vs .pt).
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np
import torch
import torch.utils.data

logger = logging.getLogger(__name__)


class MemmapTokenDataset(torch.utils.data.Dataset):
    """Per-doc sequence dataset from memmap hidden states.

    Preloads all sequences into RAM at creation time so that subsequent
    epochs are pure memory access (no memmap I/O).  Padding happens
    per-batch via :func:`collate_pad`.

    Parameters
    ----------
    h_mmap         : memmap array (total_tokens, hidden_dim)
    offsets        : (N,) int64 — start position of each doc in the memmap
    prompt_lengths : (N,) int32 — number of prompt tokens per doc
    sys_prefix     : int — system prefix tokens to skip
    labels         : (N,) int array — integer class labels
    """

    def __init__(self, h_mmap, offsets, prompt_lengths, sys_prefix, labels):
        self.labels = labels
        # Preload all sequences into RAM (unpadded, variable length).
        # For pressure train split: ~2.8 GB.
        self._seqs = []
        for i in range(len(offsets)):
            start = int(offsets[i]) + sys_prefix
            end = int(offsets[i]) + int(prompt_lengths[i])
            self._seqs.append(h_mmap[start:end].astype(np.float32))
        total_mb = sum(s.nbytes for s in self._seqs) / 1e6
        logger.info("MemmapTokenDataset: preloaded %d sequences (%.0f MB)",
                    len(self._seqs), total_mb)

    def __len__(self):
        return len(self._seqs)

    def __getitem__(self, idx):
        return self._seqs[idx], int(self.labels[idx])


def collate_pad(batch):
    """Pad variable-length sequences to the batch-local max length.

    Returns ``(padded_hs, attention_mask, labels)`` — the same tuple
    format that the attention probe's training loop expects.
    """
    seqs, labels = zip(*batch)
    max_len = max(s.shape[0] for s in seqs)
    D = seqs[0].shape[1]
    padded = torch.zeros(len(seqs), max_len, D)
    mask = torch.zeros(len(seqs), max_len)
    for i, s in enumerate(seqs):
        L = s.shape[0]
        padded[i, :L] = torch.from_numpy(s)
        mask[i, :L] = 1.0
    labels_t = torch.tensor(labels, dtype=torch.long)
    return padded, mask, labels_t


def pad_all_tokens_from_memmap(h_mmap, offsets, prompt_lengths, sys_prefix):
    """Load all docs from memmap and pad to max prompt length.

    Use for small splits (e.g. val/test) where full materialization is fine.

    Returns
    -------
    dict with ``"hidden_states"`` (N, max_len, D) and ``"attention_mask"`` (N, max_len).
    """
    num_docs = len(offsets)
    hidden_dim = h_mmap.shape[1]
    max_len = int(prompt_lengths.max()) - sys_prefix
    # Ensure max_len is at least 1
    max_len = max(max_len, 1)

    all_hs = np.zeros((num_docs, max_len, hidden_dim), dtype=np.float32)
    all_mask = np.zeros((num_docs, max_len), dtype=np.float32)

    for i in range(num_docs):
        off = int(offsets[i])
        start = off + sys_prefix
        end = off + int(prompt_lengths[i])
        seq_len = end - start
        if seq_len > max_len:
            seq_len = max_len
        if seq_len > 0:
            all_hs[i, :seq_len] = h_mmap[start:start + seq_len].astype(np.float32)
            all_mask[i, :seq_len] = 1.0

    return {"hidden_states": all_hs, "attention_mask": all_mask}


def _np_to_tensor(obj):
    """Recursively convert numpy arrays to torch tensors."""
    if isinstance(obj, np.ndarray):
        return torch.from_numpy(obj)
    if isinstance(obj, dict):
        return {k: _np_to_tensor(v) for k, v in obj.items()}
    return obj


def _tensor_to_np(obj):
    """Recursively convert torch tensors to numpy arrays."""
    if isinstance(obj, torch.Tensor):
        return obj.numpy()
    if isinstance(obj, dict):
        return {k: _tensor_to_np(v) for k, v in obj.items()}
    return obj


def save_hidden_states(
    cache_dir: str,
    experiment: str,
    dataset_name: str,
    hidden_states: dict,
    labels,
    metadata: Optional[List[Dict[str, Any]]] = None,
    system_prefix_len: int = 0,
) -> str:
    """Save hidden states + labels + metadata for one dataset to disk.

    Parameters
    ----------
    cache_dir     : root cache directory
    experiment    : experiment name (e.g. "sycophancy")
    dataset_name  : dataset name (e.g. "debate")
    hidden_states : dict with mode keys -> numpy arrays or dicts
    labels        : np.ndarray [N] or dict of np.ndarray (pressure subtasks)
    metadata      : optional list of per-sample dicts
    system_prefix_len : number of system-prompt prefix tokens (for RET / pooling)

    Returns
    -------
    Path to saved .pt file.
    """
    # --- Validate inputs ---
    assert isinstance(hidden_states, dict), \
        f"hidden_states must be a dict, got {type(hidden_states)}"
    for mode_key in hidden_states:
        v = hidden_states[mode_key]
        if isinstance(v, dict):
            assert "hidden_states" in v and "attention_mask" in v, \
                f"all_tokens dict must have 'hidden_states' and 'attention_mask', got {list(v.keys())}"
            hs_arr = v["hidden_states"]
            assert hasattr(hs_arr, 'shape') and hs_arr.ndim == 3, \
                f"all_tokens/hidden_states must be 3D [N,S,D], got shape={getattr(hs_arr, 'shape', '?')}"
        elif hasattr(v, 'shape'):
            assert v.ndim == 2, \
                f"hidden_states['{mode_key}'] must be 2D [N,D], got shape={v.shape}"

    if isinstance(labels, dict):
        for k, v in labels.items():
            assert hasattr(v, '__len__'), f"labels['{k}'] must be array-like"
    else:
        assert hasattr(labels, '__len__'), "labels must be array-like"

    assert isinstance(system_prefix_len, int) and system_prefix_len >= 0, \
        f"system_prefix_len must be non-negative int, got {system_prefix_len}"

    out_dir = os.path.join(cache_dir, experiment)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{dataset_name}.pt")

    payload = {
        "hidden_states": _np_to_tensor(hidden_states),
        "labels": _np_to_tensor(labels),
        "metadata": metadata or [],
        "dataset_name": dataset_name,
        "experiment": experiment,
        "system_prefix_len": system_prefix_len,
    }
    torch.save(payload, path)
    logger.info("Saved hidden-state cache: %s (system_prefix_len=%d)", path, system_prefix_len)
    return path


def load_hidden_states_from_memmap(
    cache_dir,
    modes: Optional[Set[str]] = None,
    split: Optional[str] = None,
) -> Dict[str, Any]:
    """Load cached hidden states from memmap format and pool per-document.

    Reads the memmap layout written by ``cache_hidden_states.py --dataset``:
    ``metadata.json``, ``hidden_states.bin``, ``offsets.npy``,
    ``lengths.npy``, and optionally ``labels.npy`` / ``labels.npz``.

    Parameters
    ----------
    cache_dir : str or Path
        Directory containing the memmap files.
    modes : set of pooling modes.  Supported: ``"average"``,
        ``"assistant_token"``.  Defaults to both.
    split : optional split name (``"train"``, ``"val"``, ``"test"``).
        When provided, reads ``splits.json`` from the cache directory
        and returns only the samples in that split.

    Returns
    -------
    dict with keys:

    * ``"average"``         — np.ndarray (N, D), mean over user tokens
    * ``"assistant_token"`` — np.ndarray (N, D), last token per doc
    * ``"labels"``          — dict or np.ndarray or None
    * ``"system_prefix_len"`` — int
    * ``"metadata"``        — the raw metadata.json dict
    """
    cache_dir = Path(cache_dir)
    with open(cache_dir / "metadata.json") as f:
        meta = json.load(f)

    hidden_dim = meta["hidden_dim"]
    total_tokens = meta["total_tokens"]
    num_docs = meta["num_docs"]
    dtype_str = meta.get("dtype", "float16")
    sys_prefix = meta.get("system_prefix_len", 0)

    offsets = np.load(cache_dir / "offsets.npy")   # (N,) int64
    lengths = np.load(cache_dir / "lengths.npy")   # (N,) int32
    h = np.memmap(
        cache_dir / "hidden_states.bin",
        dtype=dtype_str,
        mode="r",
        shape=(total_tokens, hidden_dim),
    )

    # Load prompt lengths if available (for generation caches).
    # When present, pooling uses the prompt boundary instead of full sequence
    # so that "average" covers input tokens only and "assistant_token" returns
    # the last prompt token (not the last generated token).
    prompt_lengths = None
    if (cache_dir / "prompt_lengths.npy").exists():
        prompt_lengths = np.load(cache_dir / "prompt_lengths.npy")  # (N,) int32

    # Apply split filtering if requested
    split_indices = None
    if split is not None:
        # Format A: per-split subdirectory (e.g. cache_dir/train/)
        split_subdir = cache_dir / split
        if (split_subdir / "metadata.json").exists():
            logger.info("Detected per-split subdirectory: %s", split_subdir)
            return load_hidden_states_from_memmap(split_subdir, modes=modes, split=None)

        # Format B fallback: single dir with splits.json
        splits_path = cache_dir / "splits.json"
        if not splits_path.exists():
            raise FileNotFoundError(
                f"Neither {split_subdir}/metadata.json nor {splits_path} found. "
                f"Re-run cache_hidden_states.py to generate splits."
            )
        with open(splits_path) as f:
            splits_data = json.load(f)
        key = f"{split}_indices"
        if key not in splits_data:
            raise ValueError(f"Unknown split '{split}'. Available: train, val, test")
        split_indices = np.array(splits_data[key])
        offsets = offsets[split_indices]
        lengths = lengths[split_indices]
        if prompt_lengths is not None:
            prompt_lengths = prompt_lengths[split_indices]
        num_docs = len(split_indices)
        logger.info("Using '%s' split (splits.json): %d / %d samples", split, num_docs, splits_data["n_total"])

    # Determine the effective end boundary per doc for pooling.
    # With prompt_lengths: pool over prompt tokens only (exclude generation).
    # Without: pool over the full sequence (backward compatible).
    pool_ends = prompt_lengths if prompt_lengths is not None else lengths

    if prompt_lengths is not None:
        logger.info("Using prompt_lengths for pooling (prompt-only, excluding generation)")

    if modes is None:
        modes = {"average", "assistant_token"}

    result: Dict[str, Any] = {}

    if "average" in modes:
        avg = np.zeros((num_docs, hidden_dim), dtype=np.float32)
        for i in range(num_docs):
            start = int(offsets[i]) + sys_prefix
            end = int(offsets[i]) + int(pool_ends[i])
            if end > start:
                avg[i] = h[start:end].astype(np.float32).mean(axis=0)
            else:
                # fall back to full doc if prefix covers entire sequence
                avg[i] = h[int(offsets[i]): int(offsets[i]) + int(lengths[i])].astype(
                    np.float32
                ).mean(axis=0)
        result["average"] = avg

    if "assistant_token" in modes:
        last = np.zeros((num_docs, hidden_dim), dtype=np.float32)
        for i in range(num_docs):
            idx = int(offsets[i]) + int(pool_ends[i]) - 1
            last[i] = h[idx].astype(np.float32)
        result["assistant_token"] = last

    # Load labels (filtered by split if applicable)
    labels = None
    if (cache_dir / "labels.npz").exists():
        labels = dict(np.load(cache_dir / "labels.npz"))
        if split_indices is not None:
            labels = {k: v[split_indices] for k, v in labels.items()}
    elif (cache_dir / "labels.npy").exists():
        labels = np.load(cache_dir / "labels.npy")
        if split_indices is not None:
            labels = labels[split_indices]

    result["labels"] = labels
    result["system_prefix_len"] = sys_prefix
    result["metadata"] = meta

    # Expose raw handles so callers can build MemmapTokenDataset for
    # lazy per-batch loading (used by attention probe).
    result["_memmap_handle"] = h
    result["_offsets"] = offsets
    result["_prompt_lengths"] = pool_ends
    result["_sys_prefix"] = sys_prefix

    logger.info(
        "Loaded memmap cache: %s  (%d docs, hidden_dim=%d, sys_prefix=%d)",
        cache_dir, num_docs, hidden_dim, sys_prefix,
    )
    return result


def load_hidden_states(
    cache_dir: str,
    experiment: str,
    dataset_name: str,
    as_numpy: bool = True,
) -> Dict[str, Any]:
    """Load cached hidden states for one dataset.

    Auto-detects format:

    * **Memmap format** — if ``<cache_dir>/metadata.json`` exists (written by
      ``cache_hidden_states.py --dataset``).  Returns the same dict shape as
      the .pt format (with ``hidden_states`` key containing pooled arrays).
    * **Memmap in sub-directory** — if
      ``<cache_dir>/<experiment>/<dataset_name>/metadata.json`` exists.
    * **.pt format** (legacy) — falls back to the original torch.load path.

    Parameters
    ----------
    as_numpy : if True, convert tensors back to numpy arrays (applies to .pt
        format only; memmap format always returns numpy arrays).

    Returns
    -------
    Dict with keys: hidden_states, labels, metadata, dataset_name, experiment.
    """
    cache_dir = Path(cache_dir)

    # ---- Auto-detect memmap format at root level ----
    if (cache_dir / "metadata.json").exists():
        raw = load_hidden_states_from_memmap(cache_dir)
        # Wrap to match the .pt format expected by callers
        return {
            "hidden_states": {
                k: v for k, v in raw.items()
                if k not in ("labels", "system_prefix_len", "metadata")
            },
            "labels": raw["labels"],
            "metadata": [],
            "dataset_name": dataset_name,
            "experiment": experiment,
            "system_prefix_len": raw["system_prefix_len"],
        }

    # ---- Auto-detect memmap in experiment/dataset_name subdirectory ----
    if experiment and dataset_name:
        subdir = cache_dir / experiment / dataset_name
        if (subdir / "metadata.json").exists():
            raw = load_hidden_states_from_memmap(subdir)
            return {
                "hidden_states": {
                    k: v for k, v in raw.items()
                    if k not in ("labels", "system_prefix_len", "metadata")
                },
                "labels": raw["labels"],
                "metadata": [],
                "dataset_name": dataset_name,
                "experiment": experiment,
                "system_prefix_len": raw["system_prefix_len"],
            }

    # ---- Legacy .pt format ----
    path = os.path.join(str(cache_dir), experiment, f"{dataset_name}.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cache not found: {path}")

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if as_numpy:
        payload["hidden_states"] = _tensor_to_np(payload["hidden_states"])
        payload["labels"] = _tensor_to_np(payload["labels"])

    # --- Validate loaded data ---
    assert "hidden_states" in payload, f"Cache missing 'hidden_states' key: {path}"
    assert "labels" in payload, f"Cache missing 'labels' key: {path}"
    hs = payload["hidden_states"]
    assert isinstance(hs, dict), f"hidden_states must be dict, got {type(hs)} in {path}"

    if "system_prefix_len" not in payload:
        logger.warning("Cache %s missing 'system_prefix_len' — setting to 0", path)
        payload["system_prefix_len"] = 0

    logger.info("Loaded hidden-state cache: %s", path)
    return payload


def repool_from_all_tokens(hs: dict, system_prefix_len: int) -> dict:
    """Recompute ``average``, ``assistant_token``, and ``all_tokens`` mask
    from the ``all_tokens`` data using only user+assistant tokens
    (excluding system prefix).

    The original extraction stores ``average`` as the mean over *all*
    non-padding tokens (including system prompt).  This function corrects
    that so probes only see user-input + assistant-start tokens.

    Operates in-place on the hidden-states dict and also returns it.

    Token layout (left-padded)::

        [padding] [system_prompt] [user_input] [assistant_start_token]

    After repooling:
        - ``average``         = mean of h at user+assistant positions
        - ``assistant_token`` = h at the last non-padding position
                                (same as before — the assistant start token)
        - ``all_tokens.attention_mask`` = user-only mask (system zeroed out)

    Parameters
    ----------
    hs : hidden states dict (with keys ``average``, ``assistant_token``,
         ``all_tokens``)
    system_prefix_len : number of system-prompt prefix tokens to exclude
    """
    all_tok = hs.get("all_tokens")
    if all_tok is None:
        logger.warning("repool_from_all_tokens: no all_tokens data, skipping")
        return hs

    h_full = all_tok["hidden_states"]    # [N, S, D]
    # Use original mask if we've already been repooled (idempotency)
    mask_full = all_tok.get("original_attention_mask", all_tok["attention_mask"])  # [N, S]

    if system_prefix_len > 0:
        from evaluation.probing.model_utils import make_user_token_mask
        user_mask = make_user_token_mask(mask_full, system_prefix_len)
    else:
        user_mask = mask_full.copy()

    # Recompute average over user+assistant tokens only
    um = user_mask[..., np.newaxis].astype(np.float32)  # [N, S, 1]
    hs["average"] = (h_full * um).sum(axis=1) / um.sum(axis=1).clip(min=1)

    # assistant_token = last real token (unchanged, but recompute for consistency)
    seq_lens = np.maximum(mask_full.sum(axis=1).astype(int) - 1, 0)
    hs["assistant_token"] = np.array(
        [h_full[i, seq_lens[i]] for i in range(len(h_full))]
    )

    # Replace the attention mask so downstream attention probes
    # only attend over user+assistant tokens.
    # Preserve the original full mask for RET encoder input.
    hs["all_tokens"] = {
        "hidden_states": h_full,
        "attention_mask": user_mask,
        "original_attention_mask": mask_full,
    }

    logger.info("Repooled hidden states: system_prefix_len=%d, "
                "avg user tokens/sample=%.1f",
                system_prefix_len, user_mask.sum(axis=1).mean())

    return hs


def repool_hidden_states(cache: Dict[str, Any]) -> Dict[str, Any]:
    """Repool a full cache dict (convenience wrapper around ``repool_from_all_tokens``).

    Reads ``system_prefix_len`` from the cache and calls
    ``repool_from_all_tokens`` on the hidden states.
    Operates in-place and returns the cache.
    """
    spl = cache.get("system_prefix_len", 0)
    cache["hidden_states"] = repool_from_all_tokens(cache["hidden_states"], spl)
    return cache


def cache_exists(cache_dir: str, experiment: str, dataset_name: str) -> bool:
    """Check whether a cache file exists for this dataset."""
    path = os.path.join(cache_dir, experiment, f"{dataset_name}.pt")
    return os.path.exists(path)


def list_cached_datasets(cache_dir: str, experiment: str) -> List[str]:
    """Return dataset names for which cache files exist."""
    exp_dir = os.path.join(cache_dir, experiment)
    if not os.path.isdir(exp_dir):
        return []
    return [f.replace(".pt", "") for f in os.listdir(exp_dir) if f.endswith(".pt")]
