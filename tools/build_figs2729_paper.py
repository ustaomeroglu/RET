#!/usr/bin/env python3
"""Build the figs2729 playground cells DIRECTLY from the persisted paper runs.

GPT-OSS-20B generation (bf16 / MXFP4) is NOT reproducible run-to-run, so
re-running the grid does not match the paper figures token-for-token. The
paper's exact Figs 27-29 (and a G7 sweep) on the four-digit problem are
already saved under `…/unsupervised_steering/cluster_sweep/…/numina_test_3/`
at the paper recipe (s=0.5, K=3, cosine, trigger 0, dur 600, 2048 tok). We
read those persisted outputs verbatim — token-exact reproduction, no GPU.

Emits webpage/demos/steering_grid/figs2729_highlevel/ with a single-recipe
manifest (strengths [0,0.5] → baseline / paper; starts [0]; durations [600])
so the existing playground UI works unchanged, and rewrites prompts.js.
"""
from __future__ import annotations

import argparse
import colorsys
import json
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

CS = (REPO / "outputs/GptOss20B/numina_math/ret_jepa_auth_l11/clustering/w1/"
      "K64/seed1/analysis/weighted_tau0.1/grouped_alpha1.0/"
      "unsupervised_steering/cluster_sweep")
CLUSTERING = REPO / "outputs/GptOss20B/numina_math/ret_jepa_auth_l11/clustering"
OUTROOT = REPO / "webpage/demos/steering_grid"
OUT = OUTROOT / "figs2729_highlevel"
MAXP = 300
K_COARSE = 12

# (cluster id, short label, paper note) — paper-overlay targets only.
# All four runs exist in the persisted cluster_sweep on disk.
TARGETS = [
    (18, "→ C18 verification closure (G1) — Fig 27",
     "Paper Figure 27: a compact verification-closure response."),
    (20, "→ C20 analysis entry stub (G4) — Fig 28",
     "Paper Figure 28: repeated “We need to…” analysis openers."),
    (27, "→ C27 boxed choice conclusions (G2) — Fig 29",
     "Paper Figure 29: a brief, boxed-conclusion response."),
    (38, "→ C38 error-checking & reinterpretation (G7)",
     "Holds the model in an error-checking & reinterpretation phase."),
]


def ghex(g: int) -> str:
    r, gr, b = colorsys.hls_to_rgb((g % K_COARSE) / K_COARSE, 0.45, 0.65)
    return "#%02x%02x%02x" % (int(r * 255), int(gr * 255), int(b * 255))


def final_answer(text: str) -> str:
    t = str(text)
    for m in ("assistantfinal", "<|channel|>final<|message|>", "final<|message|>"):
        i = t.rfind(m)
        if i != -1:
            t = t[i + len(m):]
            break
    return re.split(r"<\|(?:end|return|start)\|>", t)[0].strip()[:400]


def cluster_dir(cid: int) -> Path:
    hits = sorted(CS.glob(f"C{cid}_*/numina_test_3/steer_C{cid}_dur600"))
    if not hits:
        raise FileNotFoundError(f"no persisted cluster_sweep run for C{cid}")
    return hits[0]


def cell_from_run(run: Path, G, _proj, km=None, target_cluster=None):
    diag = [json.loads(l) for l in (run / "diagnostics.jsonl").read_text()
            .splitlines() if l.strip()]
    z = np.load(run / "z_trajectory.npy").astype(np.float32)
    # z_trajectory.npy holds prompt + generation z; diagnostics.jsonl holds
    # generation tokens only. Drop the prompt rows so z aligns with diag
    # (previously z[:n] zipped prompt-z against generation tokens).
    pl = int((run / "prompt_len.txt").read_text().strip())
    if len(z) >= pl + len(diag):
        z = z[pl:]
    n = min(len(z), len(diag))
    diag, z = diag[:n], z[:n]
    toks, grp, clu, steer = [], [], [], []
    pre_clu = []
    for d in diag:
        st = bool(d.get("was_steered", False))
        gid = (d["post_group_id"] if (st and d.get("post_group_id") is not None)
               else d["group_id"])
        cid = (d["post_cluster_id"] if (st and d.get("post_cluster_id") is not None)
               else d["cluster_id"])
        toks.append(d["token"]); grp.append(int(gid))
        clu.append(int(cid)); steer.append(st)
        pre_clu.append(int(d["cluster_id"]))
    # z_trajectory.npy is already the processed (L2-normalized) clustering
    # space — project directly, exactly as build_grid_assets does for the
    # generated cells (no StreamingPreprocessor.transform → consistent path).
    pca, um = _proj.project_both(G, z)
    idx = _proj.pick(n, MAXP)
    # filter "steered" to tokens where the model wasn't already at target.
    # baseline (no target) keeps the empty set; attractor uses pre_cluster.
    if target_cluster is not None:
        sset = {i for i in range(n)
                if steer[i] and pre_clu[i] != target_cluster}
    else:
        sset = {i for i, s in enumerate(steer) if s}
    # per-token cos to assigned center + margin to runner-up
    cos, mg = [], []
    if km is not None:
        from evaluation.clustering.core import predict_with_scores
        _, _cos, _mg = predict_with_scores(km, z, raw=False)
        cos = [round(float(v), 4) for v in _cos]
        mg = [round(float(v), 4) for v in _mg]
    return {
        "tokens": toks, "g": grp, "cl": clu,
        "cos": cos, "mg": mg, "steered": sorted(sset),
        "points_umap": [um[i] for i in idx], "points_pca": [pca[i] for i in idx],
        "pidx": [int(i) for i in idx],
        "pgi": [grp[i] for i in idx],
        "pst": [k for k, i in enumerate(idx) if i in sset],
        "answer": final_answer((run / "steered_output.txt").read_text()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--overlay", action="store_true",
        help="only (re)write the token-exact paper cells (baseline + each "
             "C{cid}_s0p5_p0_d600); leave manifest.js/json + prompts.js as "
             "written by build_grid_assets so the full generated grid (extra "
             "durations/strengths, G7) survives.",
    )
    args = ap.parse_args()

    import proj as _proj
    from transformers import AutoTokenizer
    from ret.config import load_config
    from scripts.unsupervised_steering.steer import build_chat_prompt

    from evaluation.clustering.core import load_kmeans

    OUT.mkdir(parents=True, exist_ok=True)
    G = _proj.load_global()
    KM = load_kmeans(CLUSTERING / "w1/K64/seed1")

    def emit(key, payload):
        payload["key"] = key
        (OUT / f"{key}.js").write_text(
            f'window.__RETGRID__&&window.__RETGRID__("{key}",'
            + json.dumps(payload, separators=(",", ":")) + ");")

    base = CS / "__baselines__/numina_test_3/baseline"
    emit("baseline", cell_from_run(base, G, _proj, KM, target_cluster=None))
    print("baseline  <-", base.relative_to(REPO))
    for cid, label, desc in TARGETS:
        run = cluster_dir(cid)
        emit(f"C{cid}_s0p5_p0_d600",
             cell_from_run(run, G, _proj, KM, target_cluster=cid))
        print(f"C{cid:<3} <-", run.relative_to(REPO))

    if args.overlay:
        tot = sum(f.stat().st_size for f in OUT.glob("*.js")) / 1e6
        print(f"\n[overlay] rewrote {len(TARGETS)+1} token-exact paper cells "
              f"in {OUT} ({tot:.1f} MB); manifest/prompts left untouched.")
        return

    model_name = load_config(json.loads(
        (CLUSTERING / "clustering_meta.json").read_text())["config_yaml_path"]
    ).llm.name
    # the baseline dir has no user_prompt.txt; it lives in the numina_test_3
    # dir of any steered run (verbatim paper four-digit prompt)
    raw = (cluster_dir(18).parent / "user_prompt.txt").read_text().strip()
    tok = AutoTokenizer.from_pretrained(model_name)
    prompt_tokens = [tok.decode([i]) for i in tok.encode(build_chat_prompt(model_name, raw))]

    manifest = {
        "pid": "figs2729_highlevel",
        "label": "Four-digit numbers (Figs 27-29)  ·  exact paper runs",
        "blurb": "The paper's verbatim Figures 27-29 (and a G7 cluster sweep) "
                 "on the held-out four-digit problem, loaded straight from the "
                 "saved runs — token-identical to the paper. Recipe is fixed "
                 "at the paper setting (strength 0.5, K=3, cosine, from token 0, "
                 "duration 600); set strength 0 for the unsteered baseline.",
        "strengths": [0, 0.5], "starts": [0], "durations": [600],
        "baseline_key": "baseline",
        "palette": [ghex(i) for i in range(K_COARSE)],
        "prompt_tokens": prompt_tokens,
        "targets": [{"key": f"C{c}", "kind": "steer_C", "value": c,
                     "label": lab, "desc": d} for c, lab, d in TARGETS],
        "fixed_recipe": True,
        "n_cells": len(TARGETS) + 1,
    }
    (OUT / "manifest.js").write_text(
        "window.__RETGRID_MANIFEST__&&window.__RETGRID_MANIFEST__("
        + json.dumps(manifest, separators=(",", ":")) + ");")
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # rewrite the top-level prompt index (sum · figs2729 · fig30)
    import grid_spec as GS
    order = ["paper_fig6_sum_1_to_11", "figs2729_highlevel", "fig30_rs_C57"]
    idx = []
    for p in order:
        if (OUTROOT / p / "manifest.js").exists():
            lbl = (manifest["label"] if p == "figs2729_highlevel"
                   else GS.PROMPTS[p]["label"])
            idx.append({"pid": p, "label": lbl})
    (OUTROOT / "prompts.js").write_text(
        "window.__RETGRID_PROMPTS__&&window.__RETGRID_PROMPTS__("
        + json.dumps(idx, separators=(",", ":")) + ");")
    tot = sum(f.stat().st_size for f in OUT.glob("*.js")) / 1e6
    print(f"\nwrote {len(TARGETS)+1} paper cells + manifest to {OUT} "
          f"({tot:.1f} MB); prompts.js = {[d['pid'] for d in idx]}")


if __name__ == "__main__":
    main()
