#!/usr/bin/env python3
"""Subsample existing NuminaMath splits to 64K/3K/1K for fair comparison.

Loads pre-split data from data/numina_math/{train,val,test} and subsamples
with stratification by source category.

Creates:
  data/numina_math_64k_samples/train/  — 64,000 samples
  data/numina_math_64k_samples/val/    —  3,000 samples
  data/numina_math_64k_samples/test/   —  1,000 samples
  data/numina_math_64k_samples/split_info.json

Usage:
    python scripts/data_preparation_scripts/prepare_numina_math_64k_samples.py
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from datasets import load_from_disk
from sklearn.model_selection import train_test_split


def stratified_subsample(dataset, max_samples: int, seed: int = 42):
    """Subsample with stratification by source."""
    if len(dataset) <= max_samples:
        return dataset

    sources = [row["source"] for row in dataset]
    counts = Counter(sources)
    strat_labels = [s if counts[s] >= 50 else "other" for s in sources]

    all_idx = np.arange(len(dataset))
    _, sample_idx = train_test_split(
        all_idx,
        test_size=max_samples,
        random_state=seed,
        stratify=strat_labels,
    )
    return dataset.select(sorted(sample_idx.tolist()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="data/numina_math")
    parser.add_argument("--output-dir", default="data/numina_math_64k_samples")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-train", type=int, default=64_000)
    parser.add_argument("--num-val", type=int, default=3_000)
    parser.add_argument("--num-test", type=int, default=1_000)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    targets = {"train": args.num_train, "val": args.num_val, "test": args.num_test}

    split_info = {"seed": args.seed, "source": str(input_dir)}

    for split_name, target_n in targets.items():
        print(f"Loading {input_dir / split_name} ...")
        ds = load_from_disk(str(input_dir / split_name))
        print(f"  {len(ds):,} samples loaded")

        sub = stratified_subsample(ds, target_n, seed=args.seed)
        print(f"  Subsampled to {len(sub):,}")

        out_path = output_dir / split_name
        out_path.mkdir(parents=True, exist_ok=True)
        sub.save_to_disk(str(out_path))
        print(f"  Saved -> {out_path}")

        split_info[split_name] = len(sub)

    info_path = output_dir / "split_info.json"
    with open(info_path, "w") as f:
        json.dump(split_info, f, indent=2)
    print(f"\nSaved split info -> {info_path}")
    print("Done.")


if __name__ == "__main__":
    main()
