#!/usr/bin/env python3
"""Prepare MMLU data for RET training and probing.

Creates:
  data/mmlu/train/            — auxiliary_train split formatted with "text" column
                                 for JEPA self-supervised training
  data/mmlu/eval/train/       — 80% of test split (stratified by subject)
  data/mmlu/eval/val/         — 20% of test split (stratified by subject)
  data/mmlu/eval/split_info.json — seed, indices, per-subject counts

Usage:
    python scripts/probing/data/prepare_mmlu_splits.py
    python scripts/probing/data/prepare_mmlu_splits.py --output-dir data/mmlu_custom
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split

from evaluation.datasets.mmlu.data import format_question


def prepare_jepa_train(output_dir: Path) -> None:
    """Load auxiliary_train, format as text, save to disk."""
    print("Loading MMLU auxiliary_train split ...")
    ds = load_dataset("cais/mmlu", "all", split="auxiliary_train")
    print(f"  {len(ds)} questions loaded")

    # Add formatted "text" column
    ds = ds.map(lambda row: {"text": format_question(row)})

    save_path = output_dir / "mmlu" / "train"
    save_path.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(save_path))
    print(f"  Saved JEPA training data to {save_path}")


def prepare_probe_splits(
    output_dir: Path,
    test_size: float = 0.2,
    seed: int = 42,
) -> None:
    """Load test split, do 80/20 stratified split by subject, save both."""
    print("\nLoading MMLU test split ...")
    ds = load_dataset("cais/mmlu", "all", split="test")
    print(f"  {len(ds)} questions loaded")

    # Add formatted "text" column and original question index
    ds = ds.map(lambda row, idx: {"text": format_question(row), "question_idx": idx},
                with_indices=True)

    subjects = [row["subject"] for row in ds]
    indices = np.arange(len(ds))

    # Stratified split by subject
    train_idx, val_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=seed,
        stratify=subjects,
    )
    train_idx = sorted(train_idx.tolist())
    val_idx = sorted(val_idx.tolist())

    train_ds = ds.select(train_idx)
    val_ds = ds.select(val_idx)

    split_dir = output_dir / "mmlu" / "eval"
    train_path = split_dir / "train"
    val_path = split_dir / "val"
    train_path.mkdir(parents=True, exist_ok=True)
    val_path.mkdir(parents=True, exist_ok=True)

    train_ds.save_to_disk(str(train_path))
    val_ds.save_to_disk(str(val_path))
    print(f"  Saved probe train ({len(train_ds)} questions) to {train_path}")
    print(f"  Saved probe val   ({len(val_ds)} questions) to {val_path}")

    # Save split info
    train_subjects = Counter(row["subject"] for row in train_ds)
    val_subjects = Counter(row["subject"] for row in val_ds)
    split_info = {
        "seed": seed,
        "test_size": test_size,
        "total_questions": len(ds),
        "train_questions": len(train_ds),
        "val_questions": len(val_ds),
        "train_indices": train_idx,
        "val_indices": val_idx,
        "train_subjects": dict(sorted(train_subjects.items())),
        "val_subjects": dict(sorted(val_subjects.items())),
    }
    info_path = split_dir / "split_info.json"
    with open(info_path, "w") as f:
        json.dump(split_info, f, indent=2)
    print(f"  Saved split info to {info_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare MMLU data for RET training and probing."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Base output directory (default: data)",
    )
    args = parser.parse_args()
    output_dir = Path(args.output_dir)

    prepare_jepa_train(output_dir)
    prepare_probe_splits(output_dir)

    print("\nDone. Next steps:")
    print(f"  1. Train RET:  python scripts/train.py configs/Pythia160M/mmlu.yaml")
    print(f"  2. Probe:      python scripts/probe_mmlu.py outputs/mmlu/checkpoint_final.pt")


if __name__ == "__main__":
    main()
