#!/usr/bin/env python3
"""Prepare TinyStories data for RET training.

Loads roneneldan/TinyStories from HuggingFace, randomly samples
train/val/test splits, and saves as Arrow datasets.

Creates:
  data/tiny_stories/train/          — 64,000 stories
  data/tiny_stories/val/            —  3,000 stories
  data/tiny_stories/test/           —  1,000 stories
  data/tiny_stories/split_info.json — seed, counts

Usage:
    python scripts/data_preparation_scripts/prepare_tiny_stories_splits.py
    python scripts/data_preparation_scripts/prepare_tiny_stories_splits.py --output-dir data --seed 42
"""

import argparse
import json
from pathlib import Path

import numpy as np
from datasets import load_dataset, Dataset


def prepare_tiny_stories_splits(
    output_dir: Path,
    seed: int = 42,
    num_train: int = 64_000,
    num_val: int = 3_000,
    num_test: int = 1_000,
) -> None:
    """Load, sample, split, and save TinyStories datasets."""

    total_needed = num_train + num_val + num_test
    rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # 1. Load dataset (train split has ~2.1M stories)
    # ------------------------------------------------------------------
    print("Loading roneneldan/TinyStories (train split) from HuggingFace ...")
    ds = load_dataset("roneneldan/TinyStories", split="train")
    print(f"  {len(ds):,} stories loaded")

    if len(ds) < total_needed:
        raise ValueError(
            f"Need {total_needed:,} stories but only {len(ds):,} available."
        )

    # ------------------------------------------------------------------
    # 2. Random sample indices
    # ------------------------------------------------------------------
    print(f"Sampling {total_needed:,} stories (seed={seed}) ...")
    indices = rng.choice(len(ds), size=total_needed, replace=False)
    rng.shuffle(indices)

    train_idx = sorted(indices[:num_train].tolist())
    val_idx = sorted(indices[num_train : num_train + num_val].tolist())
    test_idx = sorted(indices[num_train + num_val :].tolist())

    print(f"  train: {len(train_idx):,}")
    print(f"  val:   {len(val_idx):,}")
    print(f"  test:  {len(test_idx):,}")

    # ------------------------------------------------------------------
    # 3. Select subsets
    # ------------------------------------------------------------------
    train_ds = ds.select(train_idx)
    val_ds = ds.select(val_idx)
    test_ds = ds.select(test_idx)

    # ------------------------------------------------------------------
    # 4. Save to disk
    # ------------------------------------------------------------------
    base = output_dir / "tiny_stories"

    for split_name, split_ds in [("train", train_ds), ("val", val_ds), ("test", test_ds)]:
        split_path = base / split_name
        split_path.mkdir(parents=True, exist_ok=True)
        split_ds.save_to_disk(str(split_path))
        print(f"  Saved {split_name} ({len(split_ds):,} stories) -> {split_path}")

    # ------------------------------------------------------------------
    # 5. Save split_info.json
    # ------------------------------------------------------------------
    split_info = {
        "seed": seed,
        "source": "roneneldan/TinyStories",
        "total_sampled": total_needed,
        "train": len(train_ds),
        "val": len(val_ds),
        "test": len(test_ds),
    }

    info_path = base / "split_info.json"
    with open(info_path, "w") as f:
        json.dump(split_info, f, indent=2)
    print(f"  Saved split info -> {info_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare TinyStories data for RET training."
    )
    parser.add_argument(
        "--output-dir", type=str, default="data",
        help="Base output directory (default: data)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible sampling (default: 42)",
    )
    parser.add_argument("--num-train", type=int, default=64_000)
    parser.add_argument("--num-val", type=int, default=3_000)
    parser.add_argument("--num-test", type=int, default=1_000)
    args = parser.parse_args()

    prepare_tiny_stories_splits(
        Path(args.output_dir),
        seed=args.seed,
        num_train=args.num_train,
        num_val=args.num_val,
        num_test=args.num_test,
    )
    print("\nDone. Splits saved under:", Path(args.output_dir) / "tiny_stories")


if __name__ == "__main__":
    main()
