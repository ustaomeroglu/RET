#!/usr/bin/env python3
"""Compute 3D-UMAP macrostate (z_t) trajectories for the webpage demos.

Read-only with respect to outputs/: reads cached clustering artifacts and
steering z-trajectories, recomputes z_t for the 3 mental-state test docs from
*cached* layer-11 hidden states (no 20B forward), and writes per-demo JSON
ONLY into webpage/demos/_coords/. Nothing in outputs/ is modified, and the
clustering pipeline is NOT re-run.

Usage (from repo root):
    conda run -n eet python webpage/tools/build_3d.py
"""
from __future__ import annotations

import colorsys
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

CLUSTERING = REPO / "outputs/GptOss20B/numina_math/ret_jepa_auth_l11/clustering"
GROUPED = (
    CLUSTERING
    / "w1/K64/seed1/analysis/weighted_tau0.1/grouped_alpha1.0"
)
CKPT = REPO / "outputs/GptOss20B/numina_math/ret_jepa_auth_l11/checkpoint_final.pt"
CACHED_HS = REPO / "data/numina_math_64k_samples/GptOss20B/layer11"  # configured path is stale
OUT = REPO / "webpage/demos/_coords"
OUT.mkdir(parents=True, exist_ok=True)

# steering demo file stem -> source condition dir (relative to GROUPED/unsupervised_steering)
STEER = {
    "sum_baseline": "paper_fig6_sum_1_to_11/n11_str0.5_gs3/baseline",
    "sum_steered_C35": "paper_fig6_sum_1_to_11/n11_str0.5_gs3/steer_C35_dur50",
    "train_baseline": "robustness_exploration_disambiguation/word_problem/baseline",
    "train_steered_C30": "robustness_exploration_disambiguation/word_problem/steer_C30_dur600",
    "mean_baseline": "wording_disambiguation_C57/numina_test_4/baseline",
    "mean_steered_C57": "wording_disambiguation_C57/numina_test_4/steer_C57_dur300",
}
# mental-state demo file stem -> test-split doc index (deterministic, shuffle=False)
MENTAL = {
    "trace-1-function-range": 0,
    "trace-2-hyperbola": 1,
    "trace-3-statistics": 2,
    "trace-4-word-problem": 3,
}

GROUP_LABELS = np.load(GROUPED / "group_labels.npy")  # (K,) cluster -> group
K_COARSE = int(GROUP_LABELS.max()) + 1


def group_hex(g: int) -> str:
    """Same per-group color the text legend uses (colors.py _cluster_colors)."""
    r, gr, b = colorsys.hls_to_rgb(g / K_COARSE, 0.45, 0.65)
    return "#%02x%02x%02x" % (int(r * 255), int(gr * 255), int(b * 255))


_G = None  # global projection, loaded once in main()


def emit(stem: str, feats: np.ndarray, groups, clusters, steered,
         cos=None, mg=None) -> None:
    import proj as _proj

    pca, um = _proj.project_both(_G, feats)
    n = len(um)
    idx = _proj.pick(n, 500)  # mental traces are long; a 500-pt path is plenty
    payload = {
        "n": len(idx),
        "idx": [int(i) for i in idx],  # downsampled point -> original token position
        "points_umap": [um[i] for i in idx],
        "points_pca": [pca[i] for i in idx],
        "color": [group_hex(int(groups[i])) for i in idx],
        "steered": [bool(steered[i]) for i in idx],
    }
    # full-length per-token cos to assigned center + margin to runner-up
    # (used by the demo HTML's hover tooltip; not downsampled)
    if cos is not None:
        payload["tok_cos"] = [round(float(v), 4) for v in cos]
    if mg is not None:
        payload["tok_mg"] = [round(float(v), 4) for v in mg]
    (OUT / f"{stem}.json").write_text(json.dumps(payload, separators=(",", ":")))
    ns = int(np.sum(steered))
    print(f"  {stem:24s} n={n:5d}  steered={ns:4d}  -> {stem}.json")


def load_preprocessor():
    from evaluation.clustering.core import StreamingPreprocessor

    return StreamingPreprocessor.load(CLUSTERING / "w1", "z", 1, 1)


def load_kmeans():
    import joblib

    return joblib.load(CLUSTERING / "w1/K64/seed1/kmeans_model.pkl")


def build_mental(pp, km) -> None:
    import torch

    from evaluation.clustering.config import build_dataloader
    from evaluation.clustering.core import extract_features, predict_with_scores
    from ret.config import load_config
    from ret.utils.checkpointing import load_encoder

    raw_flag = bool(pp.raw or pp.centered_only)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = load_config(str(CKPT.parent / "config.yaml"))
    cfg.data.cached_hidden_states_dir = str(CACHED_HS)  # in-memory only; stale path in yaml
    encoder, cfg, _ = load_encoder(str(CKPT), device=device, config=cfg)

    want = max(MENTAL.values()) + 1
    loader = build_dataloader(
        cfg, split="test", batch_size=8, seed=42, shuffle=False, drop_last=False
    )
    docs_z: list[np.ndarray] = []
    print(f"Mental-state demos (replay test split, encoder over cached h, device={device}):")
    for batch in loader:
        feats = extract_features(batch, None, encoder, need_h=False, need_z=True, device=device)
        z = feats["z"]  # (B,T,128) cpu float32
        mask = batch.get("attention_mask")
        B = z.shape[0]
        for b in range(B):
            if mask is not None:
                valid = mask[b].bool()
                zb = z[b][valid].numpy()
            else:
                zb = z[b].numpy()
            docs_z.append(zb.astype(np.float32))
            if len(docs_z) >= want:
                break
        if len(docs_z) >= want:
            break

    inv = {v: k for k, v in MENTAL.items()}
    for idx, stem in sorted(inv.items()):
        zb = docs_z[idx]
        feats = pp.transform(zb)
        labels, cos, mg = predict_with_scores(km, feats, raw=raw_flag)
        groups = GROUP_LABELS[labels]
        emit(stem, feats, groups, labels, [False] * len(feats),
             cos=cos, mg=mg)


def main() -> None:
    global _G
    import proj as _proj
    _G = _proj.load_global()
    pp = load_preprocessor()
    km = load_kmeans()
    build_mental(pp, km)  # fixed-steering demos superseded by the playground
    print(f"\nDone. Wrote {len(list(OUT.glob('*.json')))} coord files to {OUT}")


if __name__ == "__main__":
    main()
