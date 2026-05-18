#!/usr/bin/env python3
"""Prepare NuminaMath-CoT data for RET training.

Loads AI-MO/NuminaMath-CoT (train split, ~859K samples) from HuggingFace,
formats each sample as a combined problem+solution text, then creates a
3-way stratified split by source category.

Creates:
  data/numina_math/train/          — 85% of data
  data/numina_math/val/            — 10% of data
  data/numina_math/test/           —  5% of data
  data/numina_math/split_info.json — seed, counts, per-source breakdowns

Split procedure:
  1. Group source categories with fewer than 50 samples into "other"
     so that sklearn stratification does not error on rare classes.
  2. First split: peel off 5% as the test set (stratified by source).
  3. Second split: divide the remaining 95% into ~89.5% / ~10.5%
     (stratified), yielding ~85% / ~10% of the full dataset.

Usage:
    python scripts/prepare_numina_splits.py
    python scripts/prepare_numina_splits.py --output-dir data --seed 42
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def format_sample(row: dict) -> dict:
    """Combine problem and solution into a single 'text' field."""
    return {"text": f"Problem: {row['problem']}\n\nSolution: {row['solution']}"}


def remap_rare_sources(sources: list, min_count: int = 50) -> list:
    """Replace source labels that appear fewer than min_count times with 'other'.

    This prevents sklearn from raising an error when a stratify class has
    fewer samples than the number of splits required.
    """
    counts = Counter(sources)
    remapped = [s if counts[s] >= min_count else "other" for s in sources]
    return remapped


# ---------------------------------------------------------------------------
# Core preparation function
# ---------------------------------------------------------------------------

def prepare_numina_splits(output_dir: Path, seed: int = 42) -> None:
    """Load, format, split, and save NuminaMath-CoT datasets."""

    # ------------------------------------------------------------------
    # 1. Load dataset
    # ------------------------------------------------------------------
    print("Loading AI-MO/NuminaMath-CoT (train split) from HuggingFace ...")
    ds = load_dataset("AI-MO/NuminaMath-CoT", split="train")
    print(f"  {len(ds):,} samples loaded")

    # ------------------------------------------------------------------
    # 2. Format: add 'text' column
    # ------------------------------------------------------------------
    print("Formatting samples (problem + solution -> text) ...")
    ds = ds.map(format_sample)
    print(f"  Formatting complete")

    # ------------------------------------------------------------------
    # 3. Remap rare sources before stratification
    # ------------------------------------------------------------------
    raw_sources = [row["source"] for row in ds]
    strat_labels = remap_rare_sources(raw_sources, min_count=50)

    original_source_counts = Counter(raw_sources)
    remapped_source_counts = Counter(strat_labels)
    print(f"  Source distribution (after rare-group remapping):")
    for src, cnt in sorted(remapped_source_counts.items(), key=lambda x: -x[1]):
        print(f"    {src}: {cnt:,}")

    indices = np.arange(len(ds))

    # ------------------------------------------------------------------
    # 4. First split: peel off 5% test set
    # ------------------------------------------------------------------
    print("\nCreating 3-way stratified split (85% train / 10% val / 5% test) ...")
    trainval_idx, test_idx = train_test_split(
        indices,
        test_size=0.05,
        random_state=seed,
        stratify=strat_labels,
    )

    # Gather stratify labels for the trainval portion
    strat_labels_arr = np.array(strat_labels)
    strat_trainval = strat_labels_arr[trainval_idx]

    # ------------------------------------------------------------------
    # 5. Second split: divide trainval into ~89.5% train / ~10.5% val
    #    (89.5% of 95% ≈ 85% of full; 10.5% of 95% ≈ 10% of full)
    # ------------------------------------------------------------------
    val_fraction_of_trainval = 10.0 / 95.0   # ≈ 0.10526...

    train_idx, val_idx = train_test_split(
        trainval_idx,
        test_size=val_fraction_of_trainval,
        random_state=seed,
        stratify=strat_trainval,
    )

    train_idx = sorted(train_idx.tolist())
    val_idx = sorted(val_idx.tolist())
    test_idx = sorted(test_idx.tolist())

    print(f"  train: {len(train_idx):,} samples  ({100*len(train_idx)/len(ds):.1f}%)")
    print(f"  val:   {len(val_idx):,} samples  ({100*len(val_idx)/len(ds):.1f}%)")
    print(f"  test:  {len(test_idx):,} samples  ({100*len(test_idx)/len(ds):.1f}%)")

    # ------------------------------------------------------------------
    # 6. Select subsets
    # ------------------------------------------------------------------
    train_ds = ds.select(train_idx)
    val_ds = ds.select(val_idx)
    test_ds = ds.select(test_idx)

    # ------------------------------------------------------------------
    # 7. Save to disk
    # ------------------------------------------------------------------
    base = output_dir / "numina_math"

    for split_name, split_ds in [("train", train_ds), ("val", val_ds), ("test", test_ds)]:
        split_path = base / split_name
        split_path.mkdir(parents=True, exist_ok=True)
        split_ds.save_to_disk(str(split_path))
        print(f"  Saved {split_name} ({len(split_ds):,} samples) -> {split_path}")

    # ------------------------------------------------------------------
    # 8. Save split_info.json
    # ------------------------------------------------------------------
    def source_counts(split_ds: Dataset) -> dict:
        counts = Counter(row["source"] for row in split_ds)
        return dict(sorted(counts.items()))

    split_info = {
        "seed": seed,
        "total": len(ds),
        "train": len(train_ds),
        "val": len(val_ds),
        "test": len(test_ds),
        "train_source_counts": source_counts(train_ds),
        "val_source_counts": source_counts(val_ds),
        "test_source_counts": source_counts(test_ds),
    }

    info_path = base / "split_info.json"
    with open(info_path, "w") as f:
        json.dump(split_info, f, indent=2)
    print(f"  Saved split info -> {info_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare NuminaMath-CoT data for RET training."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Base output directory (default: data)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible splits (default: 42)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    prepare_numina_splits(output_dir, seed=args.seed)

    print("\nDone. Splits saved under:", output_dir / "numina_math")


if __name__ == "__main__":
    main()
