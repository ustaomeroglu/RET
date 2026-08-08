#!/usr/bin/env python3
"""Fit ONE global 3D projection (PCA + UMAP) reused by every demo.

A single mapping with a constant origin and fixed axes makes trajectories
directly comparable across examples and far less cluttered. We sample z_t
from many cached test docs (encoder over CACHED layer-11 h — no 20B forward),
preprocess exactly like the clustering pipeline, then fit both a PCA(3) and a
UMAP(3) on the same pooled sample and save them under <repo>/webpage_files/
(persisted in the repo so the global-projection parameters are never lost).

  conda run -n eet python webpage/tools/fit_global_proj.py
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

CKPT = REPO / "outputs/GptOss20B/numina_math/ret_jepa_auth_l11/checkpoint_final.pt"
CACHED_HS = REPO / "data/numina_math_64k_samples/GptOss20B/layer11"
CLUSTERING = REPO / "outputs/GptOss20B/numina_math/ret_jepa_auth_l11/clustering"
OUT = REPO / "webpage_files"  # persisted in repo (kept out of published webpage/)
N_DOCS = 1000        # test docs pooled for the fit (richer global space)
FIT_SAMPLE = 60000   # max vectors used to fit UMAP


def main() -> None:
    import torch
    from sklearn.decomposition import PCA
    import umap as umap_lib

    from evaluation.clustering.config import build_dataloader
    from evaluation.clustering.core import StreamingPreprocessor, extract_features
    from ret.config import load_config
    from ret.utils.checkpointing import load_encoder

    OUT.mkdir(parents=True, exist_ok=True)
    pp = StreamingPreprocessor.load(CLUSTERING / "w1", "z", 1, 1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = load_config(str(CKPT.parent / "config.yaml"))
    cfg.data.cached_hidden_states_dir = str(CACHED_HS)
    encoder, cfg, _ = load_encoder(str(CKPT), device=device, config=cfg)

    loader = build_dataloader(cfg, split="test", batch_size=8, seed=42,
                              shuffle=False, drop_last=False)
    pooled = []
    n = 0
    print(f"sampling z from up to {N_DOCS} test docs (device={device}) …")
    for batch in loader:
        feats = extract_features(batch, None, encoder, need_h=False,
                                 need_z=True, device=device)
        z = feats["z"]
        mask = batch.get("attention_mask")
        for b in range(z.shape[0]):
            zb = z[b][mask[b].bool()].numpy() if mask is not None else z[b].numpy()
            pooled.append(pp.transform(zb.astype(np.float32)))
            n += 1
            if n >= N_DOCS:
                break
        if n >= N_DOCS:
            break
    pooled = np.concatenate(pooled, axis=0).astype(np.float32)
    print(f"pooled {pooled.shape[0]} preprocessed z vectors from {n} docs")

    # ---- PCA(3): constant origin (mean) + fixed orthogonal axes ----
    pca = PCA(n_components=3, random_state=42).fit(pooled)
    pca_pts = pca.transform(pooled)
    pca_scale = float(np.abs(pca_pts - pca_pts.mean(0)).max()) or 1.0
    pca_ctr = pca_pts.mean(0).astype(np.float32)

    # ---- UMAP(3): one fitted reducer, reused via .transform ----
    rng = np.random.default_rng(42)
    idx = rng.choice(len(pooled), size=min(FIT_SAMPLE, len(pooled)),
                     replace=False)
    sub = pooled[np.sort(idx)]
    print(f"fitting global UMAP(3) on {sub.shape[0]} vectors …")
    reducer = umap_lib.UMAP(n_components=3, n_neighbors=15, min_dist=0.1,
                            random_state=42).fit(sub)
    u = reducer.transform(sub)
    umap_ctr = u.mean(0).astype(np.float32)
    umap_scale = float(np.abs(u - u.mean(0)).max()) or 1.0

    np.savez(
        OUT / "global.npz",
        pca_mean=pca.mean_.astype(np.float32),
        pca_comp=pca.components_.astype(np.float32),   # (3, 128)
        pca_ctr=pca_ctr, pca_scale=np.float32(pca_scale),
        umap_ctr=umap_ctr, umap_scale=np.float32(umap_scale),
    )
    with open(OUT / "umap.pkl", "wb") as f:
        pickle.dump(reducer, f)
    print(f"saved global projections to {OUT} "
          f"(PCA explained var ratio {pca.explained_variance_ratio_.round(3).tolist()})")


if __name__ == "__main__":
    main()
