"""
Reference projection-magnitude statistics for supervised activation steering.

Given a steering direction ``w`` and a stream of hidden-state token batches,
compute summary statistics of ``|h · w_hat|`` (absolute projection). These
statistics define the "unit of steering strength" — following AxBench
(Wu et al. 2025, arXiv:2501.17148, Sec H), we apply steering as

    h ← h + alpha * w_hat,    alpha = factor * m_max

where ``m_max`` is some reference projection magnitude of real training
tokens. ``factor=1.0`` thus means "as much as a real high-projection token
already sees". AxBench uses the bare maximum; we additionally expose
quantile statistics because in large models the bare max can be dominated
by ~0.01% of outlier / attention-sink tokens whose projections are 100×
larger than typical, producing an uninterpretable scale.

This module is dataset-agnostic: the caller supplies an iterable of
``(block: np.ndarray of shape [k, D])`` and we only care about the matmul.
Dataset-specific token filtering (e.g. "only assistant tokens in turns 2-5")
lives in the dataset-specific wrapper that builds the iterable.
"""
from __future__ import annotations

import logging
from typing import Iterable, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


def projection_stats(
    direction: torch.Tensor,
    token_blocks: Iterable[np.ndarray],
    quantile: Optional[float] = 0.99,
) -> dict:
    """Compute projection-magnitude statistics of ``|h · w_hat|``.

    Parameters
    ----------
    direction    : raw direction tensor ``[D]``; unit-normalized internally.
    token_blocks : iterable of numpy arrays ``[k, D]`` yielding hidden-state
                   blocks. The caller decides which tokens to include
                   (e.g. assistant-only, turns >= 2, etc.).
    quantile     : the quantile to report as ``m_max`` (default 0.99).
                   Pass ``None`` to use the bare maximum (AxBench's literal
                   recipe; not recommended for mid-layer large-model
                   residuals where a handful of sink tokens dominate).

    Returns
    -------
    stats : dict with keys
        "m_max"        : the reported scale (q-quantile, or max if quantile=None)
        "max"          : bare maximum of |h·w_hat|
        "q99", "q95", "q90", "median" : percentile statistics for diagnostics
        "n_tokens"     : number of tokens contributed to the statistics
    """
    w_np = direction.detach().cpu().numpy().astype(np.float32)
    w_np = w_np / (np.linalg.norm(w_np) + 1e-12)

    all_proj_abs = []
    n_tokens = 0
    for block in token_blocks:
        if block.size == 0:
            continue
        proj = block @ w_np  # [k]
        all_proj_abs.append(np.abs(proj))
        n_tokens += int(block.shape[0])

    if not all_proj_abs:
        raise RuntimeError("projection_stats: no tokens supplied")

    flat = np.concatenate(all_proj_abs)
    stats = {
        "max": float(flat.max()),
        "q99": float(np.quantile(flat, 0.99)),
        "q95": float(np.quantile(flat, 0.95)),
        "q90": float(np.quantile(flat, 0.90)),
        "median": float(np.median(flat)),
        "n_tokens": n_tokens,
    }
    if quantile is None:
        stats["m_max"] = stats["max"]
        stat_name = "max"
    else:
        stats["m_max"] = float(np.quantile(flat, quantile))
        stat_name = f"q{quantile}"

    logger.info(
        "projection_stats: %d tokens; "
        "median=%.3f q90=%.3f q95=%.3f q99=%.3f max=%.3f  --> using %s = %.3f",
        n_tokens,
        stats["median"], stats["q90"], stats["q95"],
        stats["q99"], stats["max"],
        stat_name, stats["m_max"],
    )
    return stats
