#!/usr/bin/env python3
"""Prepare probing experiment data as Arrow datasets for RET training.

Saves pre-template text (chat template applied at training time) into Arrow
datasets with a "text" column, compatible with ``build_padded_dataloader``.

Usage:
    conda run -n eet python scripts/probing/data/prepare_probing_splits.py --experiment pressure
    conda run -n eet python scripts/probing/data/prepare_probing_splits.py --experiment mmlu
    conda run -n eet python scripts/probing/data/prepare_probing_splits.py --experiment all
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from datasets import Dataset, load_from_disk

# Ensure repo root is on sys.path (scripts/probing/data/ -> repo root)
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")))

SEED = 42
VAL_RATIO = 0.2

PROBE_EXPERIMENTS = ["pressure"]


# ---------------------------------------------------------------------------
# Per-experiment text extraction (pre-template, no chat wrapping)
# ---------------------------------------------------------------------------

def _load_pressure_texts(max_samples: int | None) -> list[str]:
    """Load models-under-pressure raw input texts (no chat template)."""
    from evaluation.datasets.pressure.data import load_pressure_dataset

    prompts, _ = load_pressure_dataset(max_samples=max_samples, tokenizer=None)
    return prompts


def _load_mmlu_texts(data_dir: str, max_samples: int | None) -> list[str]:
    """Load MMLU texts from existing Arrow dataset."""
    ds = load_from_disk(data_dir)
    texts = []
    for row in ds:
        text = row.get("text", "")
        if isinstance(text, str) and text.strip():
            texts.append(text)
            if max_samples and len(texts) >= max_samples:
                break
    return texts


# ---------------------------------------------------------------------------
# Per-experiment record extraction (for probe JSONL splits)
# ---------------------------------------------------------------------------

def _load_pressure_records(max_samples: int | None) -> list[dict]:
    """Return per-sample records for pressure probe JSONL saving.

    Uses ``load_pressure_dataset(tokenizer=None)`` to get raw text + encoded
    integer labels.  Iteration order matches ``_load_pressure_texts``.
    """
    from evaluation.datasets.pressure.data import load_pressure_dataset

    prompts, labels_dict = load_pressure_dataset(max_samples=max_samples, tokenizer=None)
    records: list[dict] = []
    for i, text in enumerate(prompts):
        records.append({
            "text": text,
            "tone": int(labels_dict["tone"][i]),
            "prompt_style": int(labels_dict["prompt_style"][i]),
            "high_stakes": int(labels_dict["high_stakes"][i]),
        })
    return records


# ---------------------------------------------------------------------------
# Probe JSONL split saving
# ---------------------------------------------------------------------------

def _write_jsonl(path: str, records: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _save_probe_splits(
    records: list[dict],
    output_dir: Path,
    experiment: str,
    dataset_key: str = "dataset",
    seed: int = SEED,
    val_ratio: float = VAL_RATIO,
) -> None:
    """Save per-subdataset train/test JSONL files using the same split as RET.

    Uses the identical RNG seed and split logic as ``_split_and_save`` so
    that the train/test assignment of each sample matches the RET Arrow split.

    Parameters
    ----------
    records : list of dicts, one per sample (same length and order as the
              texts list passed to ``_split_and_save``).
    output_dir : base output directory for probe splits.
    experiment : experiment name (subdirectory).
    dataset_key : key in each record that identifies the subdataset.
                  Records without this key are assigned to ``experiment``.
    """
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(records))
    n_val = max(1, int(len(records) * val_ratio))

    test_indices = set(indices[:n_val].tolist())

    from collections import defaultdict
    train_by_ds: dict[str, list[dict]] = defaultdict(list)
    test_by_ds: dict[str, list[dict]] = defaultdict(list)

    for i, rec in enumerate(records):
        ds = rec.get(dataset_key, experiment)
        if i in test_indices:
            test_by_ds[ds].append(rec)
        else:
            train_by_ds[ds].append(rec)

    exp_dir = output_dir / experiment
    all_ds = sorted(set(train_by_ds.keys()) | set(test_by_ds.keys()))
    for ds in all_ds:
        train_path = exp_dir / "train" / f"{ds}.jsonl"
        test_path = exp_dir / "test" / f"{ds}.jsonl"
        _write_jsonl(str(train_path), train_by_ds.get(ds, []))
        _write_jsonl(str(test_path), test_by_ds.get(ds, []))
        print(f"  Probe split {ds}: train={len(train_by_ds.get(ds, []))}, "
              f"test={len(test_by_ds.get(ds, []))}")


# ---------------------------------------------------------------------------
# Split and save
# ---------------------------------------------------------------------------

def _split_and_save(
    texts: list[str],
    output_dir: Path,
    experiment: str,
    seed: int = SEED,
    val_ratio: float = VAL_RATIO,
) -> None:
    """Shuffle, split, and save as Arrow datasets with split_info.json."""
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(texts))
    n_val = max(1, int(len(texts) * val_ratio))

    val_texts = [texts[i] for i in indices[:n_val]]
    train_texts = [texts[i] for i in indices[n_val:]]

    train_ds = Dataset.from_dict({"text": train_texts})
    val_ds = Dataset.from_dict({"text": val_texts})

    train_path = output_dir / "train"
    val_path = output_dir / "val"
    train_path.mkdir(parents=True, exist_ok=True)
    val_path.mkdir(parents=True, exist_ok=True)

    train_ds.save_to_disk(str(train_path))
    val_ds.save_to_disk(str(val_path))

    split_info = {
        "experiment": experiment,
        "seed": seed,
        "val_ratio": val_ratio,
        "total": len(texts),
        "train": len(train_texts),
        "val": len(val_texts),
    }
    with open(output_dir / "split_info.json", "w") as f:
        json.dump(split_info, f, indent=2)

    print(f"  Saved train ({len(train_texts)}) to {train_path}")
    print(f"  Saved val   ({len(val_texts)}) to {val_path}")


# ---------------------------------------------------------------------------
# Dispatchers
# ---------------------------------------------------------------------------

EXPERIMENTS = ["pressure", "mmlu"]


def prepare_experiment(
    experiment: str,
    output_base: Path,
    mmlu_dir: str = "./data/mmlu/train",
    max_samples: int | None = None,
    probe_splits_dir: Path | None = None,
) -> None:
    output_dir = output_base / experiment
    print(f"\n=== Preparing {experiment} ===")

    if experiment == "pressure":
        texts = _load_pressure_texts(max_samples)
    elif experiment == "mmlu":
        texts = _load_mmlu_texts(mmlu_dir, max_samples)
    else:
        raise ValueError(f"Unknown experiment: {experiment}")

    print(f"  Loaded {len(texts)} texts")
    _split_and_save(texts, output_dir, experiment)

    if probe_splits_dir is not None and experiment in PROBE_EXPERIMENTS:
        print(f"\n  --- Saving probe JSONL splits for {experiment} ---")
        if experiment == "pressure":
            records = _load_pressure_records(max_samples)
        else:
            records = []

        if records:
            assert len(records) == len(texts), (
                f"Record count ({len(records)}) != text count ({len(texts)}) "
                f"for {experiment}. Probe split indices would not match RET split."
            )
            if experiment == "pressure":
                for r in records:
                    r["dataset"] = "pressure"
            _save_probe_splits(records, probe_splits_dir, experiment,
                               dataset_key="dataset")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare probing experiment data as Arrow datasets for RET training.",
    )
    parser.add_argument(
        "--experiment", required=True,
        choices=EXPERIMENTS + ["all"],
        help="Which experiment to prepare (or 'all').",
    )
    parser.add_argument(
        "--output-dir", default="data/probing/ret_training",
        help="Base output directory (default: data/probing/ret_training).",
    )
    parser.add_argument(
        "--mmlu-dir", default="./data/mmlu/train",
        help="Directory with MMLU Arrow dataset.",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--probe-splits-dir", default=None,
        help="If set, also save per-subdataset probe train/test JSONL files "
             "here (pressure only). "
             "Uses the exact same sample split as the RET Arrow datasets.",
    )
    args = parser.parse_args()

    output_base = Path(args.output_dir)
    probe_splits_dir = Path(args.probe_splits_dir) if args.probe_splits_dir else None
    experiments = EXPERIMENTS if args.experiment == "all" else [args.experiment]

    for exp in experiments:
        prepare_experiment(
            exp, output_base,
            mmlu_dir=args.mmlu_dir,
            max_samples=args.max_samples,
            probe_splits_dir=probe_splits_dir,
        )

    print("\nDone. Next steps:")
    print("  conda run -n eet python scripts/train.py configs/probing/ret_training/<experiment>.yaml")


if __name__ == "__main__":
    main()
