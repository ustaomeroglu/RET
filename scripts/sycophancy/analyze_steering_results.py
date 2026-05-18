#!/usr/bin/env python
"""
Analyze a supervised-steering sweep: aggregate judge results across all
(probe, mode, factor) runs and produce dose-response plots + a summary CSV.

For debate, sycophancy fraction at turn t is defined as
``mean(1 - judgment[t])`` over kept conversations: judgment[t]=1 means the
model is still defending the assigned stance at turn t (not sycophantic),
and 0 means it caved (sycophantic).

Usage
-----
    python scripts/sycophancy/analyze_steering_results.py \\
        --output-dir outputs/.../supervised_steering
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import logging
import os
import re
from collections import defaultdict
from typing import Dict, List

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


_FACTOR_RE = re.compile(r"factor_([+-]?\d+(?:\.\d+)?)")


def parse_args():
    p = argparse.ArgumentParser(description="Analyze supervised steering sweep")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--num-turns", type=int, default=5)
    return p.parse_args()


def _load_judge(path: str) -> Dict[str, List[int]]:
    """{conv_id: [j1, ..., jT]}"""
    out = {}
    if not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out[rec["id"]] = rec["turn_judgments"]
    return out


def _load_splits(path: str) -> Dict[str, str]:
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return json.load(f)


def _aggregate_run(judge: Dict[str, List[int]],
                   splits: Dict[str, str],
                   num_turns: int) -> Dict[str, dict]:
    """For each split (train/test), compute fraction-sycophantic per turn
    and "ever caves" fraction. Returns {split_name: {...}}."""
    by_split = defaultdict(list)
    for conv_id, judgments in judge.items():
        if len(judgments) < num_turns:
            continue
        split = splits.get(conv_id, "unknown")
        by_split[split].append(judgments)

    out = {}
    for split, jl in by_split.items():
        arr = np.asarray(jl, dtype=np.int8)  # [N, T] where 1 = aligned, 0 = caved
        # sycophantic = caved = 1 - judgment
        syc = 1 - arr  # [N, T]
        per_turn = syc.mean(axis=0)  # [T]
        ever_caves = (syc.sum(axis=1) > 0).mean()
        out[split] = {
            "n_conversations": int(arr.shape[0]),
            "fraction_sycophantic_per_turn": per_turn.tolist(),
            "fraction_ever_caves": float(ever_caves),
        }
    return out


def _walk_runs(root: str):
    """Yield (probe, mode, factor, run_dir) for every steered run."""
    for run_cfg_path in sorted(glob.glob(os.path.join(root, "*", "factor_*", "run_config.json"))):
        run_dir = os.path.dirname(run_cfg_path)
        with open(run_cfg_path) as f:
            cfg = json.load(f)
        yield cfg["probe"], cfg["input_mode"], float(cfg["factor"]), run_dir, cfg


def main():
    args = parse_args()
    root = args.output_dir
    rows = []  # CSV rows
    by_pm: Dict[tuple, List[dict]] = defaultdict(list)  # (probe, mode) -> [{factor, train, test}]

    n_runs = 0
    n_unjudged = 0
    for probe, mode, factor, run_dir, run_cfg in _walk_runs(root):
        n_runs += 1
        judge_path = os.path.join(run_dir, "judge_results.jsonl")
        judge = _load_judge(judge_path)
        if not judge:
            n_unjudged += 1
            logger.warning("No judge results in %s — skipping", run_dir)
            continue
        splits = _load_splits(os.path.join(run_dir, "splits.json"))
        agg = _aggregate_run(judge, splits, num_turns=args.num_turns)

        for split, m in agg.items():
            for t, frac in enumerate(m["fraction_sycophantic_per_turn"], start=1):
                rows.append({
                    "probe": probe,
                    "mode": mode,
                    "factor": factor,
                    "alpha": run_cfg.get("alpha", float("nan")),
                    "m_max": run_cfg.get("m_max", float("nan")),
                    "split": split,
                    "turn": t,
                    "fraction_sycophantic": frac,
                    "n_conversations": m["n_conversations"],
                    "fraction_ever_caves": m["fraction_ever_caves"],
                })

        by_pm[(probe, mode)].append({
            "factor": factor,
            "alpha": run_cfg.get("alpha", float("nan")),
            "splits": agg,
        })

    logger.info("Walked %d runs (%d unjudged)", n_runs, n_unjudged)
    if not rows:
        logger.error("No judged runs found — did you run `run_judge.py --input-dir <output-dir>` first?")
        return

    # ---- Save CSV ----
    csv_path = os.path.join(root, "summary.csv")
    fieldnames = [
        "probe", "mode", "factor", "alpha", "m_max", "split", "turn",
        "fraction_sycophantic", "n_conversations", "fraction_ever_caves",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    logger.info("Wrote %d rows to %s", len(rows), csv_path)

    # ---- Plot dose-response curves ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available; skipping plots")
        return

    pm_keys = sorted(by_pm.keys())
    if not pm_keys:
        return

    n_pm = len(pm_keys)
    fig, axes = plt.subplots(n_pm, 2, figsize=(12, 3.0 * n_pm), squeeze=False)

    for row_idx, (probe, mode) in enumerate(pm_keys):
        runs = sorted(by_pm[(probe, mode)], key=lambda r: r["factor"])
        factors = [r["factor"] for r in runs]
        for col_idx, split in enumerate(["train", "test"]):
            ax = axes[row_idx][col_idx]
            n_turns = args.num_turns
            for t in range(1, n_turns + 1):
                ys = []
                xs = []
                for r in runs:
                    sm = r["splits"].get(split)
                    if sm is None:
                        continue
                    fpt = sm["fraction_sycophantic_per_turn"]
                    if t - 1 >= len(fpt):
                        continue
                    xs.append(r["factor"])
                    ys.append(fpt[t - 1])
                if xs:
                    ax.plot(xs, ys, marker="o", label=f"turn {t}")
            ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
            ax.set_xlabel("steering factor")
            ax.set_ylabel("fraction sycophantic")
            ax.set_title(f"{probe}/{mode}  ({split})")
            ax.set_ylim(-0.02, 1.02)
            ax.grid(True, alpha=0.3)
            if row_idx == 0 and col_idx == 1:
                ax.legend(fontsize=8, loc="best")

    fig.tight_layout()
    plot_path = os.path.join(root, "dose_response.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    logger.info("Wrote dose-response plot to %s", plot_path)

    # ---- Summary JSON: best factor per direction ----
    summary = {}
    for (probe, mode), runs in by_pm.items():
        # Aggregate "fraction_ever_caves" over test split for each factor.
        best_pos = None
        best_neg = None
        baseline = None
        for r in runs:
            test = r["splits"].get("test")
            if test is None:
                continue
            ever = test["fraction_ever_caves"]
            if r["factor"] == 0.0:
                baseline = ever
            elif r["factor"] > 0:
                if best_pos is None or ever > best_pos[1]:
                    best_pos = (r["factor"], ever)
            else:
                if best_neg is None or ever < best_neg[1]:
                    best_neg = (r["factor"], ever)
        summary[f"{probe}/{mode}"] = {
            "baseline_fraction_ever_caves": baseline,
            "max_pos_factor": best_pos[0] if best_pos else None,
            "max_pos_fraction_ever_caves": best_pos[1] if best_pos else None,
            "min_neg_factor": best_neg[0] if best_neg else None,
            "min_neg_fraction_ever_caves": best_neg[1] if best_neg else None,
        }

    with open(os.path.join(root, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Wrote summary.json")


if __name__ == "__main__":
    main()
