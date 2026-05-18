#!/usr/bin/env python
"""
Generate split_indices_balanced.json from existing split_indices.json.

Removes conversations that never exhibit sycophancy (all judge labels = 1)
and conversations with -1 sentinel labels (unevaluated turns). Only
conversations that flip at some turn are kept in train/val/test.

split_indices.json and split_indices_balanced.json live at the SCENARIO
level (one dir above the per-model data_dir), so splits are uniform across
models. The judge_results.jsonl driving the balanced filter is still
model-specific — passing multiple --data_dir values is supported but each
call OVERWRITES the scenario-level balanced split.

Usage:
    python scripts/sycophancy/generate_balanced_splits.py \
        --data_dir data/sycophancy/augmented_false_presupposition/Qwen2.5-14B-Instruct
"""
import argparse
import json
import logging
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def generate_balanced_split(data_dir: str) -> None:
    scenario_dir = os.path.dirname(os.path.normpath(data_dir))
    split_path = os.path.join(scenario_dir, "split_indices.json")
    judge_path = os.path.join(data_dir, "judge_results.jsonl")
    out_path = os.path.join(scenario_dir, "split_indices_balanced.json")

    if not os.path.exists(split_path):
        raise FileNotFoundError(
            f"split_indices.json not found at scenario level: {split_path}"
        )
    if not os.path.exists(judge_path):
        raise FileNotFoundError(f"judge_results.jsonl not found: {judge_path}")

    # Load judge results: id -> turn_judgments
    judge = {}
    with open(judge_path) as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                judge[d["id"]] = d["turn_judgments"]

    # Identify conversations to EXCLUDE: those that never flip (all labels = 1).
    # Conversations with -1 sentinels (from early-stopping judge) are valid
    # flippers — the -1 appears AFTER a 0, so truncation handles it correctly.
    exclude_ids = set()
    for conv_id, labels in judge.items():
        # Exclude only if every evaluated label is 1 (never sycophantic)
        evaluated = [l for l in labels if l != -1]
        if evaluated and all(l == 1 for l in evaluated):
            exclude_ids.add(conv_id)

    # Load existing splits
    with open(split_path) as f:
        splits = json.load(f)

    # We need to map indices -> conv_ids to check exclusion.
    # The memmap reader knows the mapping, but we can also load conv_index.jsonl
    # to get the ordered list of conv IDs.
    # Find the layer cache dir for the conv_index
    layer_dirs = [d for d in os.listdir(data_dir)
                  if d.startswith("layer_") and d.endswith("_activations_cache")]
    if not layer_dirs:
        raise FileNotFoundError(f"No layer_*_activations_cache dir in {data_dir}")
    conv_index_path = os.path.join(data_dir, layer_dirs[0], "conv_index.jsonl")
    all_conv_ids = []
    with open(conv_index_path) as f:
        for line in f:
            if line.strip():
                all_conv_ids.append(json.loads(line)["id"])

    # Filter each split
    balanced = {}
    for name in ("train", "val", "test"):
        original_indices = splits[name]
        kept = []
        for idx in original_indices:
            if idx >= len(all_conv_ids):
                continue
            conv_id = all_conv_ids[idx]
            if conv_id not in exclude_ids:
                kept.append(idx)
        balanced[name] = kept
        logger.info("  %s: %d -> %d (removed %d non-flipping/sentinel convs)",
                    name, len(original_indices), len(kept),
                    len(original_indices) - len(kept))

    total_orig = sum(len(splits[s]) for s in ("train", "val", "test"))
    total_bal = sum(len(balanced[s]) for s in ("train", "val", "test"))

    with open(out_path, "w") as f:
        json.dump(balanced, f, indent=2)

    logger.info("Wrote %s: %d -> %d conversations (%.1f%% kept)",
                out_path, total_orig, total_bal, 100 * total_bal / total_orig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", nargs="+", required=True,
                   help="Path(s) to model data dirs containing split_indices.json + judge_results.jsonl")
    args = p.parse_args()

    for data_dir in args.data_dir:
        logger.info("Processing %s", data_dir)
        generate_balanced_split(data_dir)


if __name__ == "__main__":
    main()
