"""
Sycophancy-specific glue for supervised activation steering.

The *generic* steering machinery (probe loading, forward hooks, quantile stats)
lives in ``evaluation.steering``. This file holds only the sycophancy-specific
wiring: streaming hidden states out of the sycophancy memmap cache and
filtering to the same token population the per-token sycophancy probes were
trained on (assistant tokens in turns 2..N — turn 1 has a trivially fixed
label and is excluded from probe training; see
``evaluation/probing/sycophancy/probe_data.py`` ``build_token_index``).

Without this filtering the projection statistics would be dominated by
system / user / turn-1 tokens whose mid-layer residual-stream norms in
Qwen2.5-14B can be orders of magnitude larger than what the probe actually
saw, yielding an uninterpretable (and generation-destroying) steering scale.
"""
from __future__ import annotations

import logging
from typing import Iterator, List, Optional

import numpy as np
import torch

from evaluation.datasets.sycophancy.storage import MemmapReader
from evaluation.steering.stats import projection_stats

logger = logging.getLogger(__name__)


def _iter_assistant_token_blocks(
    reader: MemmapReader,
    conv_ids: List[str],
    assistant_only: bool = True,
    min_turn: int = 2,
) -> Iterator[np.ndarray]:
    """Yield hidden-state blocks from the sycophancy memmap cache, filtered
    to match the per-token probe's training population.

    Yields one ``np.ndarray [k, D]`` per conversation (or skips empty ones).
    """
    for cid in conv_ids:
        if not reader.has_conversation(cid):
            continue
        idx = reader._id_to_idx[cid]
        rec = reader.get_index_record(idx)
        offset = rec["offset"]
        length = rec["length"]

        block = np.array(reader.get_hidden_states_slice(offset, length))  # [length, D]

        keep = np.ones(length, dtype=bool)
        if assistant_only:
            roles = np.array(reader._role[offset:offset + length])
            keep &= (roles == 2)
        if min_turn > 0:
            turns = np.array(reader._turn[offset:offset + length])
            keep &= (turns >= min_turn)

        if keep.any():
            yield block[keep]


def compute_m_max_from_memmap(
    direction: torch.Tensor,
    memmap_dir: str,
    conv_ids: List[str],
    assistant_only: bool = True,
    min_turn: int = 2,
    quantile: Optional[float] = 0.99,
) -> dict:
    """Compute reference projection-magnitude statistics (``m_max``) for a
    steering direction from the sycophancy layer-activations memmap cache.

    Thin wrapper around ``evaluation.steering.projection_stats`` that supplies
    a sycophancy-specific token iterator: hidden states from ``conv_ids``,
    filtered to assistant tokens in turns ``>= min_turn``.

    Parameters
    ----------
    direction      : raw direction tensor ``[D]``; unit-normalized internally.
    memmap_dir     : path to ``layer_<L>_activations_cache/`` directory.
    conv_ids       : conversation IDs to include (typically the training split).
    assistant_only : if True, only keep role_id == 2 tokens.
    min_turn       : exclude turns < ``min_turn`` (default 2 to match probe training).
    quantile       : quantile to report as m_max (default 0.99). See
                     ``projection_stats`` docstring for rationale.

    Returns
    -------
    stats : dict (see ``projection_stats``) augmented with filter metadata.
    """
    reader = MemmapReader(memmap_dir)
    blocks = _iter_assistant_token_blocks(
        reader, conv_ids, assistant_only=assistant_only, min_turn=min_turn,
    )
    stats = projection_stats(direction, blocks, quantile=quantile)
    stats["assistant_only"] = bool(assistant_only)
    stats["min_turn"] = int(min_turn)
    stats["n_conversations"] = len(conv_ids)
    return stats
