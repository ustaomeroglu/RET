"""
Utilities for PCA baseline encoding in the probing pipeline.

Loads pre-fitted PCA components or auto-fits on the train split,
then projects hidden states h -> h_pca for downstream probing.

Public API
----------
load_or_fit_pca(pca_dir, cache_dir, pca_dim, split, pooling_mode)
    Load PCA components; fit and save if not found.

apply_pca(h, components, mean)
    Project h through PCA components.
"""
from __future__ import annotations

import datetime
import json
import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def load_or_fit_pca(
    pca_dir: str,
    cache_dir: Optional[str] = None,
    pca_dim: Optional[int] = None,
    split: str = "train",
    pooling_mode: str = "average",
) -> Tuple[np.ndarray, np.ndarray]:
    """Load PCA components from *pca_dir*; fit on cache if not found.

    Parameters
    ----------
    pca_dir      : directory containing (or to save) ``pca_components.npz``
    cache_dir    : memmap cache directory (for auto-fitting)
    pca_dim      : number of PCA components (required for auto-fitting)
    split        : which split to fit on (default: train)
    pooling_mode : pooling mode for hidden states (default: average)

    Returns
    -------
    components : np.ndarray (pca_dim, hidden_dim)
    mean       : np.ndarray (hidden_dim,)
    """
    pca_dir = Path(pca_dir)
    npz_path = pca_dir / "pca_components.npz"

    if npz_path.exists():
        data = np.load(npz_path)
        logger.info(
            "Loaded PCA components from %s (dim=%d, hidden_dim=%d)",
            npz_path, data["components"].shape[0], data["components"].shape[1],
        )
        return data["components"], data["mean"]

    # Auto-fit PCA
    if not cache_dir or not pca_dim:
        raise ValueError(
            f"PCA components not found at {npz_path}. "
            f"Provide --cache_dir and --pca_dim to auto-fit, "
            f"or run scripts/baselines/fit_pca.py first."
        )

    from evaluation.probing.hidden_state_io import load_hidden_states_from_memmap
    from sklearn.decomposition import PCA

    logger.info(
        "PCA components not found at %s — auto-fitting (dim=%d) on %s split...",
        npz_path, pca_dim, split,
    )
    data = load_hidden_states_from_memmap(
        cache_dir, modes={pooling_mode}, split=split,
    )
    h = data[pooling_mode]
    logger.info("Fitting PCA on %d samples (hidden_dim=%d)...", h.shape[0], h.shape[1])

    pca = PCA(n_components=pca_dim)
    pca.fit(h)

    explained = float(pca.explained_variance_ratio_.sum())
    logger.info("PCA explained variance: %.4f (%.1f%%)", explained, explained * 100)

    # Save components
    pca_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        npz_path,
        components=pca.components_,
        mean=pca.mean_,
        explained_variance_ratio=pca.explained_variance_ratio_,
    )
    logger.info("Saved PCA components to %s", npz_path)

    # Save info.json (same format as scripts/baselines/fit_pca.py)
    info = {
        "pca_dim": pca_dim,
        "hidden_dim": int(h.shape[1]),
        "n_train": int(h.shape[0]),
        "explained_variance": explained,
        "source": {
            "type": "auto_fit",
            "cache_dir": str(cache_dir),
            "split": split,
            "pooling_mode": pooling_mode,
        },
        "date": datetime.date.today().isoformat(),
    }
    with open(pca_dir / "info.json", "w") as f:
        json.dump(info, f, indent=2)

    return pca.components_, pca.mean_


def apply_pca(
    h: np.ndarray,
    components: np.ndarray,
    mean: np.ndarray,
) -> np.ndarray:
    """Project hidden states through PCA.

    Parameters
    ----------
    h          : (N, hidden_dim)
    components : (pca_dim, hidden_dim)
    mean       : (hidden_dim,)

    Returns
    -------
    h_pca : (N, pca_dim)  float32
    """
    return ((h - mean) @ components.T).astype(np.float32)
