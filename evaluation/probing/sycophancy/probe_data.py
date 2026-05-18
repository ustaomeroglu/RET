"""
Per-token data loading for sycophancy probing experiments.

Each assistant-token position within turns 2-5 is one sample. The probe
predicts whether the model is being sycophantic at that turn, from the
hidden state at that token position (or a running mean, or a causal prefix).

Input modes:
  h_t                  — hidden state at position t  [D]
  running_mean_global  — mean(h_0, ..., h_t)         [D]
  running_mean_turn    — mean(h_{turn_start}, ..., h_t)  [D]
  causal_prefix        — h[0:t+1]                    [t+1, D]  (variable length)
  z_t                  — RET-encoded z at position t  [z_dim]
"""
from __future__ import annotations

import json
import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from evaluation.datasets.sycophancy.storage import MemmapReader
# Label production lives on the data side (datasets/sycophancy/labels.py);
# re-exported here so probe_data's own callers + existing importers are
# unaffected, while ret/ depends only on datasets/ (not the probing code).
from evaluation.datasets.sycophancy.labels import (  # noqa: F401
    compute_sycophancy_labels,
    load_judge_results,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Canonical splits
# ---------------------------------------------------------------------------

def load_canonical_splits(
    data_dir: str,
    reader: MemmapReader,
    judge_results: Dict[str, List[int]],
    split_file: str = "split_indices.json",
) -> Dict[str, List[str]]:
    """Load canonical conversation-level splits, filtering to judged conversations.

    Reads *split_file* from the parent of *data_dir* (the scenario dir, e.g.
    ``data/sycophancy/debate/``), maps indices to conversation IDs, and keeps
    only those with judge results. Splits are shared across models within a
    scenario, so the file lives one level above the per-model data_dir.

    Returns {"train": [conv_id, ...], "val": [...], "test": [...]}
    """
    split_path = os.path.join(os.path.dirname(os.path.normpath(data_dir)), split_file)
    if not os.path.exists(split_path):
        raise FileNotFoundError(
            f"{split_file} not found at {split_path}. "
            "Splits are scenario-level (one dir above the per-model data_dir). "
            "Create it before running probing."
        )

    with open(split_path, "r") as f:
        saved = json.load(f)

    all_conv_ids = reader.conversation_ids

    splits = {}
    for name in ("train", "val", "test"):
        indices = saved[name]
        conv_ids = [all_conv_ids[i] for i in indices if i < len(all_conv_ids)]
        # Filter to judged only
        judged = [cid for cid in conv_ids if cid in judge_results]
        splits[name] = judged
        logger.info("  Split %s: %d/%d conversations have judge results",
                    name, len(judged), len(conv_ids))

    return splits


# ---------------------------------------------------------------------------
# Token index builder
# ---------------------------------------------------------------------------

def build_token_index(
    reader: MemmapReader,
    conv_ids: List[str],
    judge_results: Dict[str, List[int]],
    num_turns: int = 5,
    max_tokens_per_turn: Optional[int] = None,
    seed: int = 42,
    mode: str = "debate",
    truncate_on: Optional[int] = None,
    user_context_only: bool = False,
) -> List[Tuple[str, int, int, int, int, int]]:
    """Build a flat index of per-token sample tuples.

    Each tuple: (conv_id, turn_k, position_t, turn_start, turn_n_tokens, label)

    For mode='debate' (default), conversations where the model's turn-1
    reply did not defend the assigned argument (judgment[0] != 1) are dropped
    entirely (stance-overlap filter). Turn 1 is the anchor and excluded
    from training; only turns 2..num_turns are indexed.

    For mode='false_presupposition', all conversations are kept, turn 1
    is included, and the index truncates at the first sycophantic turn
    (post-cave turns are noise).

    Parameters
    ----------
    reader              : MemmapReader for the hidden state cache
    conv_ids            : conversation IDs to index
    judge_results       : {conv_id: [j1, ..., j5]}
    num_turns           : total turns per conversation (5)
    max_tokens_per_turn : if set, randomly subsample this many tokens per turn
    seed                : random seed for subsampling
    mode                : 'debate', 'false_presupposition', or 'absolute'
    truncate_on         : if set, stop including turns after the first turn with
                          this label value. For 'false_presupposition' mode this
                          defaults to 1 (stop after first sycophantic turn) since
                          post-cave turns have no meaningful information.

    Returns
    -------
    List of (conv_id, turn_k, position_t, turn_start_pos, turn_n_tokens, label) tuples.
    """
    # FP default: truncate at first sycophantic turn (label==1). Post-cave
    # turns are noise (model won't un-cave) and for 35B are marked with judge -1.
    if mode == "false_presupposition" and truncate_on is None:
        truncate_on = 1

    rng = np.random.default_rng(seed)
    index = []
    n_skipped_unaligned = 0
    for conv_id in conv_ids:
        if conv_id not in judge_results:
            continue

        judgments = judge_results[conv_id]
        labels = compute_sycophancy_labels(judgments, mode=mode)
        if labels is None:
            n_skipped_unaligned += 1
            continue  # debate: model's turn-1 reply did not defend the argument

        rec = reader.get_index_record(reader._id_to_idx[conv_id])
        offset = rec["offset"]
        length = rec["length"]

        role_ids = np.array(reader._role[offset:offset + length])
        turn_ids = np.array(reader._turn[offset:offset + length])

        # For false_presupposition, turn 1 is a real response (not a trivial
        # anchor like in debate), so include it.
        start_turn = 1 if mode == "false_presupposition" else 2
        for k in range(start_turn, num_turns + 1):
            mask = (role_ids == 2) & (turn_ids == k)
            positions = np.where(mask)[0]
            if len(positions) == 0:
                continue

            label = labels[k - 1]  # k is 1-indexed, labels are 0-indexed
            if label < 0:
                # FP: judgment -1 = judge early-stopped after first cave.
                # Subsequent turns are also -1; safe to break.
                break
            turn_start = int(positions[0])
            turn_n_tokens = len(positions)

            if user_context_only:
                # Take only positions[1] — the `assistant` role-name token
                # right after `<|im_start|>`. Hidden state there encodes full
                # prior context (system + prior turns + this turn's user
                # message) via causal attention, but no generated content.
                if len(positions) < 2:
                    continue  # not enough tokens to reach positions[1]
                positions = positions[1:2]
            elif max_tokens_per_turn is not None and len(positions) > max_tokens_per_turn:
                positions = rng.choice(positions, size=max_tokens_per_turn, replace=False)
                positions.sort()

            for pos in positions:
                index.append((conv_id, k, int(pos), turn_start, turn_n_tokens, label))

            if truncate_on is not None and label == truncate_on:
                break

    if mode == "debate" and n_skipped_unaligned > 0:
        logger.info(
            "  Stance-overlap filter dropped %d/%d conversations "
            "(turn-1 reply did not defend the assigned argument)",
            n_skipped_unaligned, len(conv_ids),
        )
    logger.info("Built token index: %d samples from %d conversations (mode=%s)",
                len(index), len(conv_ids), mode)
    return index


# ---------------------------------------------------------------------------
# Lazy per-token Dataset for pooled probes
# ---------------------------------------------------------------------------

class TokenProbingDataset(Dataset):
    """Lazy per-token dataset for sycophancy probing (pooled modes).

    Each sample is one assistant-token position in turns 2-5.
    Features are computed on-the-fly from the memmap cache.

    Modes:
      h_t                 — h[t]                           [D]
      running_mean_global — cumsum(h[0:t+1])[-1] / (t+1)  [D]
      running_mean_turn   — mean(h[turn_start:t+1])        [D]
      z_t                 — z_cache[conv_id][t]            [z_dim]

    Optional PCA projection: if pca_components and pca_mean are provided,
    the feature is projected as (feature - mean) @ components.T after
    computation. Works with any pooled mode.
    """

    def __init__(
        self,
        reader: MemmapReader,
        token_index: List[Tuple[str, int, int, int, int]],
        mode: str = "h_t",
        z_cache: Optional[Dict[str, np.ndarray]] = None,
        pca_components: Optional[np.ndarray] = None,
        pca_mean: Optional[np.ndarray] = None,
    ):
        self.reader = reader
        self.token_index = token_index
        self.mode = mode
        self.z_cache = z_cache
        self.pca_components = pca_components  # [pca_dim, D]
        self.pca_mean = pca_mean              # [D]

        if mode == "z_t" and z_cache is None:
            raise ValueError("z_cache required for mode='z_t'")

        # Simple single-entry cache (most samples are from same conversation)
        self._cache_id = None
        self._cache_h = None
        self._cache_cumsum = None

    def __len__(self) -> int:
        return len(self.token_index)

    def _load_conv_data(self, conv_id: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load and cache conversation hidden states + cumulative sum."""
        if conv_id == self._cache_id:
            return self._cache_h, self._cache_cumsum
        rec = self.reader.get_index_record(self.reader._id_to_idx[conv_id])
        offset, length = rec["offset"], rec["length"]
        h = np.array(self.reader._hs[offset:offset + length], dtype=np.float32)
        cumsum = np.cumsum(h, axis=0, dtype=np.float64)
        self._cache_id = conv_id
        self._cache_h = h
        self._cache_cumsum = cumsum
        return h, cumsum

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        conv_id, turn_k, pos, turn_start, turn_n_tokens, label = self.token_index[idx]

        if self.mode == "z_t":
            feature = self.z_cache[conv_id][pos]
        else:
            h, cumsum = self._load_conv_data(conv_id)
            if self.mode == "h_t":
                feature = h[pos]
            elif self.mode == "running_mean_global":
                feature = (cumsum[pos] / (pos + 1)).astype(np.float32)
            elif self.mode == "running_mean_turn":
                if turn_start == 0:
                    feature = (cumsum[pos] / (pos + 1)).astype(np.float32)
                else:
                    feature = ((cumsum[pos] - cumsum[turn_start - 1]) / (pos - turn_start + 1)).astype(np.float32)
            else:
                raise ValueError(f"Unknown mode: {self.mode}")

        if self.pca_components is not None:
            feature = (feature - self.pca_mean) @ self.pca_components.T

        return torch.tensor(feature, dtype=torch.float32), label


# ---------------------------------------------------------------------------
# Lazy per-token Dataset for sequence probes (causal prefix)
# ---------------------------------------------------------------------------

def materialize_pooled_features(
    reader: MemmapReader,
    token_index: List[Tuple[str, int, int, int, int]],
    mode: str = "h_t",
    z_cache: Optional[Dict[str, np.ndarray]] = None,
    z_turn_cache: Optional[Dict[str, Dict[int, np.ndarray]]] = None,
    pca_components: Optional[np.ndarray] = None,
    pca_mean: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Precompute all pooled features into numpy arrays.

    Streams through conversations once (sorted by conv_id for cache locality),
    computes features, and stacks into contiguous arrays.

    Modes:
      h_t, running_mean_global, running_mean_turn — from raw h
      z_t — from z_cache (full-conversation RET encoding)
      z_t_turn — from z_turn_cache (per-turn RET encoding, turn-only context)

    Returns (X [N, D], y [N]).
    """
    # Sort index by conv_id for cache locality
    sorted_idx = sorted(range(len(token_index)), key=lambda i: token_index[i][0])

    # Determine output dim
    if mode in ("z_t", "z_running_mean_global", "z_running_mean_turn") and z_cache is not None:
        first_conv = next(iter(z_cache.values()))
        D = first_conv.shape[1]
    elif mode == "z_t_turn" and z_turn_cache is not None:
        first_conv = next(iter(z_turn_cache.values()))
        first_turn = next(iter(first_conv.values()))
        D = first_turn.shape[1]
    else:
        D = reader.hidden_dim
    if pca_components is not None:
        D = pca_components.shape[0]

    N = len(token_index)
    X = np.empty((N, D), dtype=np.float32)
    y = np.empty(N, dtype=np.int64)

    cache_id = None
    cache_h = None
    cache_cumsum = None
    z_cache_id = None       # separate cache for z-based running means
    z_cache_cumsum = None

    for out_idx, src_idx in enumerate(sorted_idx):
        conv_id, turn_k, pos, turn_start, turn_n_tokens, label = token_index[src_idx]

        if mode == "z_t":
            feature = z_cache[conv_id][pos].astype(np.float32)
        elif mode == "z_t_turn":
            turn_z = z_turn_cache[conv_id].get(turn_k)
            if turn_z is not None:
                pos_in_turn = pos - turn_start
                feature = turn_z[min(pos_in_turn, len(turn_z) - 1)].astype(np.float32)
            else:
                feature = np.zeros(D, dtype=np.float32)
        elif mode in ("z_running_mean_global", "z_running_mean_turn"):
            if conv_id != z_cache_id:
                z_conv = z_cache[conv_id].astype(np.float64)
                z_cache_cumsum = np.cumsum(z_conv, axis=0)
                z_cache_id = conv_id
            if mode == "z_running_mean_global":
                feature = (z_cache_cumsum[pos] / (pos + 1)).astype(np.float32)
            else:  # z_running_mean_turn
                if turn_start == 0:
                    feature = (z_cache_cumsum[pos] / (pos + 1)).astype(np.float32)
                else:
                    feature = ((z_cache_cumsum[pos] - z_cache_cumsum[turn_start - 1]) / (pos - turn_start + 1)).astype(np.float32)
        else:
            if conv_id != cache_id:
                rec = reader.get_index_record(reader._id_to_idx[conv_id])
                offset, length = rec["offset"], rec["length"]
                cache_h = np.array(reader._hs[offset:offset + length], dtype=np.float32)
                cache_cumsum = np.cumsum(cache_h, axis=0, dtype=np.float64)
                cache_id = conv_id

            if mode == "h_t":
                feature = cache_h[pos]
            elif mode == "running_mean_global":
                feature = (cache_cumsum[pos] / (pos + 1)).astype(np.float32)
            elif mode == "running_mean_turn":
                if turn_start == 0:
                    feature = (cache_cumsum[pos] / (pos + 1)).astype(np.float32)
                else:
                    feature = ((cache_cumsum[pos] - cache_cumsum[turn_start - 1]) / (pos - turn_start + 1)).astype(np.float32)
            else:
                raise ValueError(f"Unknown mode: {mode}")

        if pca_components is not None:
            feature = (feature - pca_mean) @ pca_components.T

        X[out_idx] = feature
        y[out_idx] = label

        if (out_idx + 1) % 100000 == 0:
            logger.info("  Materialized %d/%d features", out_idx + 1, N)

    # Unsort back to original order
    unsort = np.empty(N, dtype=np.int64)
    for out_idx, src_idx in enumerate(sorted_idx):
        unsort[src_idx] = out_idx
    X = X[unsort]
    y = y[unsort]

    logger.info("Materialized %s: %d samples, D=%d (%.1f GB)",
                mode, N, D, N * D * 4 / 1e9)
    return X, y


class CausalPrefixDataset(Dataset):
    """Lazy dataset that returns h[0:t+1] as a variable-length prefix.

    For use with attention/transformer probes that process full sequences.
    """

    def __init__(
        self,
        reader: MemmapReader,
        token_index: List[Tuple[str, int, int, int, int]],
    ):
        self.reader = reader
        self.token_index = token_index
        self._cache_id = None
        self._cache_h = None

    def __len__(self) -> int:
        return len(self.token_index)

    def _load_hidden_states(self, conv_id: str) -> np.ndarray:
        if conv_id == self._cache_id:
            return self._cache_h
        rec = self.reader.get_index_record(self.reader._id_to_idx[conv_id])
        offset, length = rec["offset"], rec["length"]
        h = np.array(self.reader._hs[offset:offset + length], dtype=np.float32)
        self._cache_id = conv_id
        self._cache_h = h
        return h

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        conv_id, turn_k, pos, turn_start, turn_n_tokens, label = self.token_index[idx]
        h = self._load_hidden_states(conv_id)
        prefix = h[:pos + 1]  # [t+1, D]
        return torch.tensor(prefix, dtype=torch.float32), label


def collate_causal_prefix(batch):
    """Collate variable-length causal prefixes with right-padding.

    Returns dict compatible with attention/transformer probe input:
      {"hidden_states": [B, S_max, D], "attention_mask": [B, S_max]}, labels [B]
    """
    prefixes, labels = zip(*batch)
    max_len = max(p.shape[0] for p in prefixes)
    D = prefixes[0].shape[1]
    B = len(prefixes)

    padded = torch.zeros(B, max_len, D)
    mask = torch.zeros(B, max_len)
    for i, p in enumerate(prefixes):
        L = p.shape[0]
        padded[i, :L] = p
        mask[i, :L] = 1.0

    return (
        {"hidden_states": padded, "attention_mask": mask},
        torch.tensor(labels, dtype=torch.long),
    )


# ---------------------------------------------------------------------------
# Turn-segment dataset for sequence probes (turn-only context)
# ---------------------------------------------------------------------------

class TurnSegmentDataset(Dataset):
    """Each sample is one turn's assistant tokens as an independent sequence.

    Unlike ConversationProbingDataset (full conversation), this gives the
    probe only within-turn context — a fair comparison against running_mean_turn.
    All tokens in a segment share the same label.
    """

    def __init__(
        self,
        reader: MemmapReader,
        conv_ids: List[str],
        judge_results: Dict[str, List[int]],
        mode: str = "debate",
        num_turns: int = 5,
        truncate_on: Optional[int] = None,
    ):
        # FP default: truncate at first sycophantic turn (label==1 = cave).
        if mode == "false_presupposition" and truncate_on is None:
            truncate_on = 1

        self.reader = reader
        self.segments = []  # (conv_id, turn_k, label)
        n_skipped_unaligned = 0

        start_turn = 1 if mode == "false_presupposition" else 2
        for cid in conv_ids:
            if cid not in judge_results:
                continue
            labels = compute_sycophancy_labels(judge_results[cid], mode=mode)
            if labels is None:
                n_skipped_unaligned += 1
                continue
            for k in range(start_turn, num_turns + 1):
                label = labels[k - 1]
                if label < 0:
                    break  # unevaluated turn — stop (e.g. after first cave for FP)
                self.segments.append((cid, k, label))
                if truncate_on is not None and label == truncate_on:
                    break

        if mode == "debate" and n_skipped_unaligned > 0:
            logger.info(
                "TurnSegmentDataset: stance-overlap filter dropped %d/%d "
                "conversations (turn-1 reply did not defend the assigned argument)",
                n_skipped_unaligned, len(conv_ids),
            )

    def __len__(self) -> int:
        return len(self.segments)

    def __getitem__(self, idx: int) -> Dict:
        conv_id, turn_k, label = self.segments[idx]
        rec = self.reader.get_index_record(self.reader._id_to_idx[conv_id])
        off, length = rec["offset"], rec["length"]

        role_ids = np.array(self.reader._role[off:off + length])
        turn_ids = np.array(self.reader._turn[off:off + length])
        mask = (role_ids == 2) & (turn_ids == turn_k)
        positions = np.where(mask)[0]

        if len(positions) == 0:
            D = self.reader.hidden_dim
            return {
                "hidden_states": torch.zeros(1, D),
                "labels": torch.tensor([float(label)]),
                "supervision_mask": torch.ones(1),
                "rel_positions": torch.zeros(1),
                "length": 1,
            }

        h_segment = np.array(self.reader._hs[off + positions[0]:off + positions[-1] + 1],
                             dtype=np.float32)
        L = h_segment.shape[0]

        # Relative position within turn: 0 = first token, 1 = last token
        rel_pos = np.arange(L, dtype=np.float32) / max(L - 1, 1)

        return {
            "hidden_states": torch.from_numpy(h_segment),
            "labels": torch.full((L,), float(label)),
            "supervision_mask": torch.ones(L),
            "rel_positions": torch.from_numpy(rel_pos),
            "length": L,
        }


# ---------------------------------------------------------------------------
# Conversation-level dataset for sequence probes (full-sequence, masked loss)
# ---------------------------------------------------------------------------

class ConversationProbingDataset(Dataset):
    """Full-conversation dataset with per-position labels and supervision mask.

    Each item is one full conversation. The probe processes the entire
    sequence in a single forward pass (with causal masking) and the loss
    is applied only at supervised positions (assistant tokens in turns 2-5).

    This is dramatically faster than CausalPrefixDataset: one forward pass
    per conversation instead of one per token position.
    """

    def __init__(
        self,
        reader: MemmapReader,
        conv_ids: List[str],
        judge_results: Dict[str, List[int]],
        mode: str = "debate",
        truncate_on: Optional[int] = None,
        user_context_only: bool = False,
    ):
        # FP default: truncate at first sycophantic turn (label==1 = cave).
        if mode == "false_presupposition" and truncate_on is None:
            truncate_on = 1

        self.reader = reader
        self.mode = mode
        self.truncate_on = truncate_on
        self.user_context_only = user_context_only

        # Filter to judged conversations (and stance-overlap filter for debate)
        self.conv_ids = []
        self.labels_map: Dict[str, List[int]] = {}
        n_skipped_unaligned = 0
        for cid in conv_ids:
            if cid not in judge_results:
                continue
            labels = compute_sycophancy_labels(judge_results[cid], mode=mode)
            if labels is None:
                n_skipped_unaligned += 1
                continue
            self.conv_ids.append(cid)
            self.labels_map[cid] = labels

        if mode == "debate" and n_skipped_unaligned > 0:
            logger.info(
                "ConversationProbingDataset: stance-overlap filter dropped %d/%d "
                "conversations (turn-1 reply did not defend the assigned argument)",
                n_skipped_unaligned, len(conv_ids),
            )

    def __len__(self) -> int:
        return len(self.conv_ids)

    def __getitem__(self, idx: int) -> Dict:
        conv_id = self.conv_ids[idx]
        rec = self.reader.get_index_record(self.reader._id_to_idx[conv_id])
        off, length = rec["offset"], rec["length"]

        h = np.array(self.reader._hs[off:off + length], dtype=np.float32)
        role_ids = np.array(self.reader._role[off:off + length])
        turn_ids = np.array(self.reader._turn[off:off + length])
        labels = self.labels_map[conv_id]

        # Per-position labels, supervision mask, and relative positions
        y = np.zeros(length, dtype=np.float32)
        sup_mask = np.zeros(length, dtype=np.float32)
        rel_pos = np.zeros(length, dtype=np.float32)

        start_turn = 1 if self.mode == "false_presupposition" else 2
        for k in range(start_turn, len(labels) + 1):
            label = labels[k - 1]
            if label < 0:
                # FP: judgment -1 = judge early-stopped after first cave.
                # Subsequent turns are also -1; safe to break.
                break
            mask = (role_ids == 2) & (turn_ids == k)
            positions = np.where(mask)[0]
            if len(positions) == 0:
                continue
            if self.user_context_only:
                # Supervise only the `assistant` role-name token (positions[1]).
                if len(positions) < 2:
                    continue
                p = int(positions[1])
                sup_mask[p] = 1.0
                y[p] = float(label)
                rel_pos[p] = 0.0  # single-position bin
            else:
                sup_mask[mask] = 1.0
                y[mask] = float(label)
                # Relative position within this turn
                turn_len = len(positions)
                for j, p in enumerate(positions):
                    rel_pos[p] = j / max(turn_len - 1, 1)
            if self.truncate_on is not None and label == self.truncate_on:
                break

        return {
            "hidden_states": torch.from_numpy(h),
            "labels": torch.from_numpy(y),
            "supervision_mask": torch.from_numpy(sup_mask),
            "rel_positions": torch.from_numpy(rel_pos),
            "length": length,
        }


def collate_conversations(batch: List[Dict]) -> Dict:
    """Collate variable-length conversations with right-padding."""
    max_len = max(item["length"] for item in batch)
    B = len(batch)
    D = batch[0]["hidden_states"].shape[1]

    padded_h = torch.zeros(B, max_len, D)
    padded_y = torch.zeros(B, max_len)
    padded_sup = torch.zeros(B, max_len)
    padded_mask = torch.zeros(B, max_len)
    padded_rel = torch.zeros(B, max_len)

    for i, item in enumerate(batch):
        L = item["length"]
        padded_h[i, :L] = item["hidden_states"]
        padded_y[i, :L] = item["labels"]
        padded_sup[i, :L] = item["supervision_mask"]
        padded_mask[i, :L] = 1.0
        if "rel_positions" in item:
            padded_rel[i, :L] = item["rel_positions"]

    return {
        "hidden_states": padded_h,
        "labels": padded_y,
        "supervision_mask": padded_sup,
        "attention_mask": padded_mask,
        "rel_positions": padded_rel,
    }


# ---------------------------------------------------------------------------
# Position-stratified evaluation
# ---------------------------------------------------------------------------

def compute_relative_positions(
    token_index: List[Tuple],
) -> np.ndarray:
    """Compute relative position within turn for each sample in the index.

    Returns array of floats in [0, 1] where 0 = first token, 1 = last token.
    """
    rel_pos = np.empty(len(token_index), dtype=np.float32)
    for i, entry in enumerate(token_index):
        pos, turn_start, turn_n_tokens = entry[2], entry[3], entry[4]
        offset_in_turn = pos - turn_start
        rel_pos[i] = offset_in_turn / max(turn_n_tokens - 1, 1)
    return rel_pos


def position_stratified_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    rel_positions: np.ndarray,
    n_bins: int = 20,
) -> Dict:
    """Compute accuracy as a function of relative position within the turn.

    Parameters
    ----------
    y_true        : [N] ground truth labels
    y_pred        : [N] predicted labels
    rel_positions : [N] relative position in [0, 1]
    n_bins        : number of position bins

    Returns
    -------
    dict with keys:
      "bin_centers":   [n_bins] float — center of each bin
      "bin_accuracy":  [n_bins] float — accuracy per bin
      "bin_bal_accuracy": [n_bins] float — balanced accuracy per bin
      "bin_n_samples": [n_bins] int — samples per bin
      "auc":           float — area under the accuracy-vs-position curve
      "bal_auc":       float — area under balanced-accuracy-vs-position curve
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    bin_acc = np.zeros(n_bins)
    bin_bal_acc = np.zeros(n_bins)
    bin_n = np.zeros(n_bins, dtype=int)

    correct = (y_true == y_pred)

    for b in range(n_bins):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        if b == n_bins - 1:
            mask = (rel_positions >= lo) & (rel_positions <= hi)
        else:
            mask = (rel_positions >= lo) & (rel_positions < hi)

        n = mask.sum()
        bin_n[b] = n
        if n == 0:
            continue

        bin_acc[b] = correct[mask].mean()

        # Balanced accuracy per bin
        y_t = y_true[mask]
        y_p = y_pred[mask]
        n_pos = (y_t == 1).sum()
        n_neg = (y_t == 0).sum()
        tpr = ((y_p == 1) & (y_t == 1)).sum() / max(n_pos, 1)
        tnr = ((y_p == 0) & (y_t == 0)).sum() / max(n_neg, 1)
        bin_bal_acc[b] = (tpr + tnr) / 2

    # AUC: trapezoidal integration over bins (normalized to [0, 1])
    valid = bin_n > 0
    if valid.sum() >= 2:
        auc = float(np.trapz(bin_acc[valid], bin_centers[valid]))
        bal_auc = float(np.trapz(bin_bal_acc[valid], bin_centers[valid]))
    else:
        auc = 0.0
        bal_auc = 0.0

    return {
        "bin_centers": bin_centers.tolist(),
        "bin_accuracy": bin_acc.tolist(),
        "bin_bal_accuracy": bin_bal_acc.tolist(),
        "bin_n_samples": bin_n.tolist(),
        "auc": auc,
        "bal_auc": bal_auc,
    }


def plot_position_accuracy_curve(
    curves: Dict[str, Dict],
    output_path: str,
    title: str = "Accuracy vs. Position in Turn",
) -> None:
    """Plot accuracy-vs-position curves for multiple methods.

    Parameters
    ----------
    curves : {method_name: position_stratified_metrics output}
    output_path : path to save PNG
    title : plot title
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for name, data in curves.items():
        centers = data["bin_centers"]
        ax1.plot(centers, data["bin_accuracy"],
                 marker=".", markersize=4, label=f"{name} (AUC={data['auc']:.3f})")
        ax2.plot(centers, data["bin_bal_accuracy"],
                 marker=".", markersize=4, label=f"{name} (AUC={data['bal_auc']:.3f})")

    for ax, metric in [(ax1, "Accuracy"), (ax2, "Balanced Accuracy")]:
        ax.set_xlabel("Relative Position in Turn (0=start, 1=end)")
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} vs. Position")
        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="chance")
        ax.set_ylim(0.3, 1.0)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Position-accuracy curve saved to %s", output_path)
