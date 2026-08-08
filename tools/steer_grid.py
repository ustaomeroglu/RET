#!/usr/bin/env python3
"""Generate one shard of the steering knob-grid (one GPU, ctx reused).

Replicates the run_unsupervised_steering body so the steering hook is kept
and the per-token z_t trajectory can be harvested without writing the heavy
interactive HTML. Per cell, writes a COMPACT .npz to the staging dir:
  tokens(str[]), cluster(int16[]), group(int16[]), steered(bool[]),
  z(float32[T,dz]), plus text + meta.

Read-only w.r.t. outputs/. Resumable: existing stage files are skipped.

Usage (one process per GPU):
  CUDA_VISIBLE_DEVICES is set via --gpu; run from repo root in env `eet`:
    python webpage/tools/steer_grid.py --pid paper_fig6_sum_1_to_11 \
        --shard 0 --nshards 4 --gpu 0 --stage /scratch/mustaome/ret_grid
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", required=True)
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--nshards", type=int, required=True)
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--stage", required=True)
    ap.add_argument("--limit", type=int, default=0, help="debug: cap #cells")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    import numpy as np
    import torch

    from evaluation.unsupervised_steering.conditions import build_rules_for_condition
    from evaluation.unsupervised_steering.config import SteeringConfig, SteeringRule
    from evaluation.unsupervised_steering.generation import SteeredGenerator
    from evaluation.unsupervised_steering.hooks import attach_steering_hook
    from evaluation.unsupervised_steering.mppi import GradientSteerer
    from evaluation.unsupervised_steering.targets import load_steering_context
    from ret.config import load_config
    from scripts.unsupervised_steering.steer import build_chat_prompt, resolve_prompt

    import grid_spec as G

    CLUSTERING = str(
        REPO / "outputs/GptOss20B/numina_math/ret_jepa_auth_l11/clustering"
    )
    stage = Path(args.stage) / args.pid
    stage.mkdir(parents=True, exist_ok=True)

    spec = G.PROMPTS[args.pid]
    import zlib
    # hash-shard by cell key so ANY subset (e.g. a newly-added duration
    # column) spreads evenly across GPUs — index%nshards would dump a whole
    # column on one shard when #durations == nshards.
    cells = [
        c for c in G.enumerate_cells(args.pid)
        if zlib.crc32(c["key"].encode()) % args.nshards == args.shard
    ]
    if args.limit:
        cells = cells[: args.limit]
    todo = [c for c in cells if not (stage / f"{c['key']}.npz").exists()]
    print(f"[shard {args.shard}/{args.nshards} gpu{args.gpu}] "
          f"{len(cells)} assigned, {len(todo)} to do", flush=True)
    if not todo:
        return

    # ---- load model + clustering context ONCE (the expensive part) ----------
    base_cfg = SteeringConfig(
        clustering_dir=CLUSTERING, window_size=1, K=64, seed=1,
        tau=0.1, alpha=1.0, device="cuda",
    )
    t0 = time.time()
    ctx = load_steering_context(base_cfg)
    print(f"[shard {args.shard}] ctx loaded in {time.time()-t0:.0f}s", flush=True)

    meta_path = Path(CLUSTERING) / "clustering_meta.json"
    ret_cfg = load_config(json.loads(meta_path.read_text())["config_yaml_path"])
    d_h = ret_cfg.llm.hidden_dim
    model_name = ret_cfg.llm.name

    src = spec["source"]
    _pf = src.get("prompt_file")
    if _pf and not os.path.isabs(_pf):
        _pf = str(REPO / _pf)
    pa = SimpleNamespace(
        problem_id=src.get("problem_id"),
        problems_file=str(REPO / "data/math_problems/problems.json"),
        numina_test_index=src.get("numina_test_index"),
        prompt_file=_pf, user_prompt=src.get("user_prompt"),
        problem_id_override=args.pid,
    )
    _raw, chat_prompt, _pid = resolve_prompt(pa, model_name)

    def run_cell(c: dict) -> None:
        if c["kind"] in ("avoid_C", "avoid_G"):
            # The `duration` knob window-limits the repulsor: it fires on
            # proximity to the avoided target (margin gate, as usual, with a
            # small fixed stickiness) but is only ACTIVE for the tokens in
            # [start, start+dur). `duration` is NOT the sticky parameter.
            is_c = c["kind"] == "avoid_C"
            rules = [SteeringRule(
                avoid_cluster=c["value"] if is_c else None,
                avoid_group=None if is_c else c["value"],
                avoid_margin_threshold=G.AVOID_MARGIN,
                avoid_sticky_duration=G.AVOID_STICKY,
                avoid_from_step=c["start"],
                avoid_until_step=c["start"] + (c["dur"] or 1),
            )]
        elif c["kind"] == "baseline":
            _label, rules = build_rules_for_condition(("baseline",))
        else:
            _label, rules = build_rules_for_condition(
                (c["kind"], c["value"]),
                attractor_duration=c["dur"] or 1,
                trigger_position=c["start"],
            )
        is_base = c["kind"] == "baseline"
        cfg = SteeringConfig(
            clustering_dir=CLUSTERING, window_size=1, K=64, seed=1,
            tau=0.1, alpha=1.0, rules=rules, metric="cosine",
            strength=0.0 if is_base else c["strength"],
            gradient_steps=1 if is_base else G.GRADIENT_STEPS,
            prompt=chat_prompt,
            max_new_tokens=spec["max_new_tokens"],
            temperature=G.TEMPERATURE, top_p=G.TOP_P,
            device="cuda", output_dir=None,
        )
        optimizer = GradientSteerer(
            strength=cfg.strength, gradient_steps=cfg.gradient_steps,
            d_h=d_h, device=cfg.device, lambda_pred=0.0, epsilon_pred=0.0,
            normalize_z=(cfg.metric == "cosine"),
        )
        hook, handles = attach_steering_hook(
            ctx, cfg.rules, optimizer, cfg.device, metric=cfg.metric
        )
        try:
            torch.manual_seed(G.TORCH_SEED)
            gen = SteeredGenerator(ctx.model, ctx.tokenizer, hook)
            text, diag = gen.generate(
                cfg.prompt, cfg.max_new_tokens, cfg.temperature, cfg.top_p
            )
        finally:
            for h in handles:
                h.remove()

        # per-generation-step z (post-perturbation, processed = clustering space)
        zf = list(hook.z_features)
        T = min(len(diag), len(zf))
        diag = diag[:T]
        z = torch.stack(zf[:T]).float().cpu().numpy() if T else np.zeros((0, 128), np.float32)

        toks = [d["token"] for d in diag]
        clu, grp, steer = [], [], []
        pre_c, pre_g = [], []
        for d in diag:
            st = bool(d.get("was_steered", False))
            # "displayed" cluster: post if steered, else pre (used for colour)
            clu.append(d["post_cluster_id"] if (st and d.get("post_cluster_id") is not None) else d["cluster_id"])
            grp.append(d["post_group_id"] if (st and d.get("post_group_id") is not None) else d["group_id"])
            # PRE-perturbation cluster/group every step — lets the build
            # filter out tokens where z was already at the target (no push
            # needed), so the red-underbar tracks "meaningfully steered".
            pre_c.append(d["cluster_id"])
            pre_g.append(d["group_id"])
            steer.append(st)
        np.savez_compressed(
            stage / f"{c['key']}.npz",
            tokens=np.array(toks, dtype=object),
            cluster=np.array(clu, np.int16),
            group=np.array(grp, np.int16),
            pre_cluster=np.array(pre_c, np.int16),
            pre_group=np.array(pre_g, np.int16),
            steered=np.array(steer, np.bool_),
            z=z.astype(np.float32),
            text=text,
            meta=json.dumps({k: c[k] for k in ("key", "target_key", "kind",
                                               "value", "strength", "start", "dur")}),
        )

    done = 0
    t_run = time.time()
    for c in todo:
        ct = time.time()
        try:
            run_cell(c)
            done += 1
            if done <= 3 or done % 25 == 0:
                el = time.time() - ct
                rate = (time.time() - t_run) / done
                eta = rate * (len(todo) - done) / 60
                print(f"[shard {args.shard}] {done}/{len(todo)} {c['key']} "
                      f"{el:.1f}s  eta~{eta:.0f}min", flush=True)
        except Exception as e:  # loud, but keep going so one bad cell != lost shard
            print(f"[shard {args.shard}] FAIL {c['key']}: {type(e).__name__}: {e}",
                  flush=True)
    print(f"[shard {args.shard}] DONE {done}/{len(todo)} "
          f"in {(time.time()-t_run)/60:.1f}min", flush=True)


if __name__ == "__main__":
    main()
