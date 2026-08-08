#!/usr/bin/env python3
"""Turn the staged steering-grid .npz cells into compact web assets.

Fits ONE shared 3D-UMAP per prompt (on a subsample pooled across all cells)
so trajectories are visually comparable as knobs change, transforms every
cell, and writes:

  webpage/demos/steering_grid/<pid>/<cellkey>.json   (compact, per cell)
  webpage/demos/steering_grid/<pid>/manifest.json    (knob options + meta)

Run after the grid shards finish:
  conda run -n eet python webpage/tools/build_grid_assets.py --pid paper_fig6_sum_1_to_11
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
import grid_spec as G  # noqa: E402

STAGE = Path("/scratch/mustaome/ret_grid")
OUTROOT = REPO / "webpage/demos/steering_grid"
K_COARSE = 12

# Each playground prompt has a persisted steering run that used the SAME
# prompt; its z_trajectory.npy stores prompt + generation z (prompt first,
# `prompt_len` rows). We read the prompt rows once per pid to colour the
# prompt tokens by their real macrostate group.
_US = ("clustering/w1/K64/seed1/analysis/weighted_tau0.1/grouped_alpha1.0/"
       "unsupervised_steering")
PERSISTED_PROMPT_RUN = {
    "figs2729_highlevel":
        f"{_US}/cluster_sweep/C18_verification_closure/numina_test_3/"
        "steer_C18_dur600",
    "paper_fig6_sum_1_to_11":
        f"{_US}/paper_fig6_sum_1_to_11/n11_str0.5_gs3/baseline",
    "fig30_rs_C57":
        f"{_US}/wording_disambiguation_C57/numina_test_5/baseline",
}
RJA = REPO / "outputs/GptOss20B/numina_math/ret_jepa_auth_l11"
CLUSTERING = RJA / "clustering"
# Canonical macrostate cluster & group names (the same naming the
# mental-state demos use). Baked into each manifest so tooltips can show
# the cluster / group title + description.
NAMING_TXT = (CLUSTERING / "w1/K64/seed1/analysis/weighted_tau0.1/"
              "grouped_alpha1.0/cluster_naming/naming.txt")


def load_naming():
    """Parse `naming.txt` into {clusters:{id:{name,desc}}, groups:{...}}."""
    out = {"clusters": {}, "groups": {}}
    for line in NAMING_TXT.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        # "G7: exploratory_... | desc..." or "C18 [G1]: verification_... | desc..."
        if line.startswith("G"):
            head, _, body = line.partition(":")
            gid = int(head[1:])
            name, _, desc = body.strip().partition("|")
            out["groups"][gid] = {"name": name.strip(), "desc": desc.strip()}
        elif line.startswith("C"):
            head, _, body = line.partition(":")
            # "C18 [G1]"
            cid = int(head.split()[0][1:])
            name, _, desc = body.strip().partition("|")
            out["clusters"][cid] = {"name": name.strip(), "desc": desc.strip()}
    return out


def compute_prompt_macrostate(pid: str, n_prompt_tokens: int):
    """Per-prompt-token (group_id, cos_sim, margin) for `pid`.

    Reads the prompt z from the persisted run, projects through the SAME
    k-means the clustering pipeline used, validates the recipe against the
    run's own generation diagnostics, then maps cluster -> group. Raises
    loudly on any mismatch (no silent fallback).

    Returns: dict {g, cos, mg} — each is a Python list of length
    n_prompt_tokens.
    """
    import numpy as np

    from evaluation.clustering.core import load_kmeans, predict_with_scores

    rel = PERSISTED_PROMPT_RUN.get(pid)
    if rel is None:
        raise KeyError(f"no persisted prompt run mapped for pid {pid!r}")
    run = RJA / rel
    if not run.is_dir():
        raise FileNotFoundError(f"persisted prompt run missing: {run}")
    z = np.load(run / "z_trajectory.npy").astype(np.float32)
    pl = int((run / "prompt_len.txt").read_text().strip())
    diag = [json.loads(l) for l in
            (run / "diagnostics.jsonl").read_text().splitlines() if l.strip()]
    if pl != n_prompt_tokens:
        raise ValueError(
            f"{pid}: persisted prompt_len {pl} != tokenizer prompt_tokens "
            f"{n_prompt_tokens} — prompt text drift; refusing to mis-colour")
    if len(z) < pl + len(diag):
        raise ValueError(
            f"{pid}: z_trajectory {len(z)} < prompt_len {pl} + gen "
            f"{len(diag)} — unexpected layout")

    km = load_kmeans(CLUSTERING / "w1/K64/seed1")
    glab = np.load(CLUSTERING / "w1/K64/seed1/analysis/weighted_tau0.1/"
                   "grouped_alpha1.0/group_labels.npy")

    # z_trajectory.npy is ALREADY the processed clustering space (L2-norm),
    # so assign directly (no StreamingPreprocessor.transform). The recorded
    # z is post-perturbation when steered, so the matching diagnostics
    # cluster is post_cluster_id (else cluster_id) — validate against that.
    truth = np.array([
        (d["post_cluster_id"] if d.get("was_steered") and
         d.get("post_cluster_id") is not None else d["cluster_id"])
        for d in diag])
    best_raw, best_agree = None, -1.0
    for raw in (False, True):
        lbl = predict_with_scores(km, z[pl:pl + len(diag)], raw=raw)[0]
        agree = float((lbl == truth).mean())
        if agree > best_agree:
            best_raw, best_agree = raw, agree
    if best_agree < 0.95:
        raise RuntimeError(
            f"{pid}: cluster-assignment recipe disagrees with persisted "
            f"diagnostics (best agree {best_agree:.2f}); space mismatch")
    pc, pcos, pmg = predict_with_scores(km, z[:pl], raw=best_raw)
    print(f"  prompt[{pid}]: {pl} tokens, recipe "
          f"{'euclid' if best_raw else 'cosine'} (gen-agree {best_agree:.2f})")
    return {
        "cl": [int(c) for c in pc],
        "g": [int(g) for g in glab[pc]],
        "cos": [round(float(v), 4) for v in pcos],
        "mg": [round(float(v), 4) for v in pmg],
    }


def cos_margin(km, z):
    """Per-token (cos to assigned center, margin to runner-up) via cosine."""
    from evaluation.clustering.core import predict_with_scores
    _, cos, mg = predict_with_scores(km, z, raw=False)
    return ([round(float(v), 4) for v in cos],
            [round(float(v), 4) for v in mg])


def ghex(g: int) -> str:
    r, gr, b = colorsys.hls_to_rgb((g % K_COARSE) / K_COARSE, 0.45, 0.65)
    return "#%02x%02x%02x" % (int(r * 255), int(gr * 255), int(b * 255))


def final_answer(text: str) -> str:
    """Best-effort extract the model's final-channel answer for a quick read-out."""
    t = str(text)
    for marker in ("assistantfinal", "<|channel|>final<|message|>", "final<|message|>"):
        i = t.rfind(marker)
        if i != -1:
            t = t[i + len(marker):]
            break
    t = re.split(r"<\|(?:end|return|start)\|>", t)[0]
    return t.strip()[:400]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", required=True)
    ap.add_argument("--fit-sample", type=int, default=9000,
                    help="max pooled z vectors used to fit the shared UMAP")
    ap.add_argument("--manifest-only", action="store_true",
                    help="only rewrite manifest (prompt context + strengths); "
                         "skip UMAP + cell files")
    args = ap.parse_args()

    sdir = STAGE / args.pid
    spec = G.PROMPTS[args.pid]
    expected = G.n_cells(args.pid)
    outdir = OUTROOT / args.pid
    outdir.mkdir(parents=True, exist_ok=True)

    # ---- prompt context tokens (shown muted before the generation) ----------
    # Reconstruct exactly how the generator sees the prompt: encode the chat
    # prompt, decode per token. Tokenizer only — no model weights.
    from transformers import AutoTokenizer

    from ret.config import LLMConfig, llm_short_name, load_config
    from scripts.unsupervised_steering.steer import (
        build_chat_prompt, load_curated_problem, load_numina_test_prompt,
    )

    meta = json.loads(
        (REPO / "outputs/GptOss20B/numina_math/ret_jepa_auth_l11/clustering"
                "/clustering_meta.json").read_text()
    )
    model_name = load_config(meta["config_yaml_path"]).llm.name
    src = spec["source"]
    if "prompt_file" in src:                        # exact paper prompt file
        pf = Path(src["prompt_file"])
        if not pf.is_absolute():
            pf = REPO / pf
        chat = build_chat_prompt(model_name, pf.read_text().strip())
    elif "user_prompt" in src:                      # exact raw problem text
        chat = build_chat_prompt(model_name, src["user_prompt"])
    elif "numina_test_index" in src:
        short = llm_short_name(LLMConfig(name=model_name, layer_index=0))
        chat = load_numina_test_prompt(short, src["numina_test_index"])
    else:
        raw = load_curated_problem(
            REPO / "data/math_problems/problems.json", src["problem_id"],
        )["prompt"]
        chat = build_chat_prompt(model_name, raw)
    tok = AutoTokenizer.from_pretrained(model_name)
    prompt_ids = tok.encode(chat)
    prompt_tokens = [tok.decode([i]) for i in prompt_ids]
    prompt_macro = compute_prompt_macrostate(args.pid, len(prompt_tokens))
    prompt_cl = prompt_macro["cl"]
    prompt_g = prompt_macro["g"]
    prompt_cos = prompt_macro["cos"]
    prompt_mg = prompt_macro["mg"]

    n_have = len(list(sdir.glob("*.npz")))
    _ax = G.axes(args.pid)
    manifest = {
        "pid": args.pid,
        "label": spec["label"],
        "blurb": spec["blurb"],
        # 0 == baseline (no steering); the rest are the generated strengths
        "strengths": [0] + list(_ax[0]),
        "starts": list(_ax[1]),
        "durations": list(_ax[2]),
        "targets": [
            {"key": t["key"], "kind": t["kind"], "label": t["label"],
             "desc": t["desc"],
             # optional per-target strength axis (e.g. C18/C27 drop s=1)
             "strengths": t.get("strengths")} for t in spec["targets"]
        ],
        "baseline_key": "baseline",
        "defaults": spec.get("defaults", {}),
        "cells_omit": spec.get("cells_omit", []),    # UI-suppressed combos
        "palette": [ghex(i) for i in range(K_COARSE)],
        "prompt_tokens": prompt_tokens,
        "prompt_g": prompt_g,                  # per-prompt-token group colour
        "prompt_cl": prompt_cl,                # tooltip: cluster id
        "prompt_cos": prompt_cos,              # tooltip: cos to assigned ctr
        "prompt_mg": prompt_mg,                # tooltip: margin to runner-up
        "cluster_names": load_naming()["clusters"],   # tooltip data
        "group_names": load_naming()["groups"],
        "cell_key_fmt": "{target}_s{strength}_p{start}_d{dur}  (strength dot->p)",
        "n_cells": n_have,
        "n_expected": expected,
    }
    (outdir / "manifest.js").write_text(
        "window.__RETGRID_MANIFEST__&&window.__RETGRID_MANIFEST__("
        + json.dumps(manifest, separators=(",", ":")) + ");"
    )
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"manifest written ({len(prompt_tokens)} prompt-context tokens, "
          f"strengths={manifest['strengths']})")
    if args.manifest_only:
        return

    import proj as _proj
    from evaluation.clustering.core import load_kmeans
    PROJ = _proj.load_global()  # ONE global PCA+UMAP shared by every demo
    KM = load_kmeans(CLUSTERING / "w1/K64/seed1")   # for per-token cos+margin

    # emit ONLY cells in the current spec's exact key set — robust to a
    # staging dir that mixes old- and new-spec runs, or a trimmed target list
    valid = {c["key"] for c in G.enumerate_cells(args.pid)}
    files = [f for f in sorted(sdir.glob("*.npz")) if f.stem in valid]
    if not files:
        raise SystemExit(f"no staged cells in {sdir}")
    MAXP = 300  # 3D path is downsampled (text panel keeps every token)
    print(f"{args.pid}: emitting {len(files)}/{len(valid)} valid cells "
          f"— global UMAP, ≤{MAXP}-pt path")
    for f in files:
        d = np.load(f, allow_pickle=True)
        z = d["z"].astype(np.float32)
        pca, um = _proj.project_both(PROJ, z)
        n = len(um)
        grp = d["group"].astype(int).tolist()[:n]
        clu = d["cluster"].astype(int).tolist()[:n]
        steer = d["steered"].astype(bool).tolist()[:n]
        # filter the steered set: red-underbar only on tokens where the
        # PRE-perturbation macrostate wasn't already at the target (else
        # the steering had nothing to push). Falls back to plain
        # was_steered for legacy npz files that lack pre_cluster/pre_group.
        meta = json.loads(str(d["meta"]))
        kind, tval = meta.get("kind"), meta.get("value")
        if "pre_cluster" in d.files and kind == "steer_C":
            pre_c = d["pre_cluster"].astype(int).tolist()[:n]
            sset = {i for i in range(n) if steer[i] and pre_c[i] != tval}
        elif "pre_group" in d.files and kind == "steer_G":
            pre_g = d["pre_group"].astype(int).tolist()[:n]
            sset = {i for i in range(n) if steer[i] and pre_g[i] != tval}
        else:                                                 # avoid_*, baseline, or legacy
            sset = {i for i, s in enumerate(steer) if s}
        cos, mg = cos_margin(KM, z[:n])                        # per-token
        idx = _proj.pick(n, MAXP)
        payload = {
            "key": f.stem,
            "tokens": [str(x) for x in d["tokens"]][:n],       # text panel
            "g": grp,                                          # text colour
            "cl": clu,                                         # tooltip cluster id
            "cos": cos,                                        # tooltip cos
            "mg": mg,                                          # tooltip margin
            "steered": sorted(sset),                           # text underline
            "points_umap": [um[i] for i in idx],               # 3D path (UMAP)
            "points_pca": [pca[i] for i in idx],               # 3D path (PCA)
            "pidx": [int(i) for i in idx],                     # downsample→tok idx
            "pgi": [grp[i] for i in idx],                      # 3D pt colour
            "pst": [k for k, i in enumerate(idx) if i in sset],  # 3D steered
            "answer": final_answer(str(d["text"])),
        }
        body = json.dumps(payload, separators=(",", ":"))
        (outdir / f"{f.stem}.js").write_text(
            f'window.__RETGRID__&&window.__RETGRID__("{f.stem}",{body});'
        )
    total = sum(f.stat().st_size for f in outdir.glob("*.js"))
    print(f"wrote {len(files)} cell JS + manifest to {outdir} "
          f"({total/1e6:.1f} MB total)")

    # top-level prompt index for the playground's Prompt selector
    idx = [{"pid": p, "label": G.PROMPTS[p]["label"]}
           for p in G.PROMPTS
           if (OUTROOT / p / "manifest.js").exists()]
    (OUTROOT / "prompts.js").write_text(
        "window.__RETGRID_PROMPTS__&&window.__RETGRID_PROMPTS__("
        + json.dumps(idx, separators=(",", ":")) + ");"
    )
    print(f"prompt index: {[d['pid'] for d in idx]}")


if __name__ == "__main__":
    main()
