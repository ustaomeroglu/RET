"""Load + apply the single global 3D projection (PCA & UMAP).

Shared by build_3d.py (mental-state) and build_grid_assets.py (steering grid)
so every demo lives in the SAME constant-origin, fixed-axis space.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

# persisted in the repo but kept OUT of the published webpage/ (umap.pkl is ~66 MB)
_PROJ = Path(__file__).resolve().parents[2] / "webpage_files"


def load_global():
    g = np.load(_PROJ / "global.npz")
    with open(_PROJ / "umap.pkl", "rb") as f:
        reducer = pickle.load(f)
    return {
        "pca_mean": g["pca_mean"], "pca_comp": g["pca_comp"],
        "pca_ctr": g["pca_ctr"], "pca_scale": float(g["pca_scale"]),
        "umap": reducer, "umap_ctr": g["umap_ctr"],
        "umap_scale": float(g["umap_scale"]),
    }


def _round(a):
    return [[round(float(v), 3) for v in row] for row in a]


def pick(n: int, maxn: int):
    """Evenly-spaced indices (incl. first & last) for downsampling a path."""
    if n <= maxn:
        return list(range(n))
    return sorted(set(
        [round(i * (n - 1) / (maxn - 1)) for i in range(maxn)]
    ))


def project_both(G, z: np.ndarray):
    """z: (T,d) preprocessed macrostates -> (pca_pts, umap_pts), global space."""
    if z is None or len(z) == 0:
        return [], []
    z = z.astype(np.float32)
    pc = (z - G["pca_mean"]) @ G["pca_comp"].T          # sklearn PCA.transform
    pc = (pc - G["pca_ctr"]) / G["pca_scale"]
    um = G["umap"].transform(z)
    um = (um - G["umap_ctr"]) / G["umap_scale"]
    return _round(pc), _round(um)
