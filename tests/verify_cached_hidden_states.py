#!/usr/bin/env python3
"""Verify cached hidden states match live LLM extraction.

Loads batches from the dataloader, runs the LLM live, and compares against
the cached memmap files. Reports per-doc error statistics.

Usage:
    conda run -n eet python scripts/verify_cached_hidden_states.py \
        --config configs/GptOss20B/numina_math_gen.yaml \
        --cache-dir outputs/test_h_cache/val
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description="Verify cached hidden states.")
    parser.add_argument("--config", type=str, required=True, help="YAML config path")
    parser.add_argument("--cache-dir", type=str, required=True, help="Path to cached split dir")
    parser.add_argument("--split", type=str, default=None, help="Split name (auto-detected from metadata)")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-batches", type=int, default=None, help="Max batches to verify (default: all cached docs)")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    from torch.utils.data import DataLoader
    from ret.config import load_config
    from ret.data.dataset import build_padded_dataloader, collate_padded
    from ret.extraction.hooks import HiddenStateExtractor

    # Load and validate cache
    cache_dir = Path(args.cache_dir)
    required_files = ["metadata.json", "offsets.npy", "lengths.npy", "hidden_states.bin", "input_ids.bin"]
    for fname in required_files:
        if not (cache_dir / fname).exists():
            raise FileNotFoundError(f"Cache incomplete: missing {cache_dir / fname}")

    with open(cache_dir / "metadata.json") as f:
        meta = json.load(f)
    offsets = np.load(cache_dir / "offsets.npy")
    lengths = np.load(cache_dir / "lengths.npy")

    # Validate metadata consistency
    if meta["num_docs"] != len(offsets):
        raise ValueError(f"metadata says num_docs={meta['num_docs']} but offsets has {len(offsets)} entries")
    if meta["num_docs"] != len(lengths):
        raise ValueError(f"metadata says num_docs={meta['num_docs']} but lengths has {len(lengths)} entries")

    h_cached = np.memmap(
        cache_dir / "hidden_states.bin", dtype=meta["dtype"], mode="r",
        shape=(meta["total_tokens"], meta["hidden_dim"]),
    )
    ids_cached = np.memmap(
        cache_dir / "input_ids.bin", dtype="int32", mode="r",
        shape=(meta["total_tokens"],),
    )

    if meta["total_tokens"] != h_cached.shape[0]:
        raise ValueError(f"metadata says total_tokens={meta['total_tokens']} but memmap has {h_cached.shape[0]} rows")

    split = args.split or meta["split"]
    if args.split and args.split != meta["split"]:
        print(f"WARNING: --split={args.split} differs from cache metadata split={meta['split']}")

    print(f"Cache: {cache_dir}")
    print(f"  model={meta['model_name']}, layer={meta['layer_index']}, "
          f"dim={meta['hidden_dim']}, dtype={meta['dtype']}")
    print(f"  num_docs={meta['num_docs']}, total_tokens={meta['total_tokens']}, split={split}")

    # Load config and cross-validate against cached metadata
    config = load_config(args.config)
    if config.llm.name != meta["model_name"]:
        print(f"WARNING: config model={config.llm.name} != cache model={meta['model_name']}")
    if config.llm.layer_index != meta["layer_index"]:
        print(f"WARNING: config layer={config.llm.layer_index} != cache layer={meta['layer_index']}")
    if config.llm.hidden_dim != meta["hidden_dim"]:
        print(f"WARNING: config hidden_dim={config.llm.hidden_dim} != cache hidden_dim={meta['hidden_dim']}")

    extractor = HiddenStateExtractor(config.llm, device=args.device)

    # Build DataLoader directly with drop_last=False to verify ALL docs
    # (build_padded_dataloader uses drop_last=True for train split)
    loader_tmp = build_padded_dataloader(
        config.data, config.llm, split=split,
        batch_size=args.batch_size, shuffle=False,
    )
    dataset = loader_tmp.dataset
    pad_id = extractor.tokenizer.pad_token_id
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_padded(b, pad_id),
        num_workers=config.data.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # Verify batches
    num_cached_docs = len(offsets)
    doc_idx = 0
    batch_count = 0
    all_ok = True
    has_warnings = False
    total_verified = 0

    print(f"\n=== Verifying cached vs live (up to {num_cached_docs} docs) ===")

    for batch in loader:
        if doc_idx >= num_cached_docs:
            break
        if args.num_batches is not None and batch_count >= args.num_batches:
            break

        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask")
        B = input_ids.shape[0]

        with torch.no_grad():
            h_live = extractor.extract(
                input_ids.to(extractor.input_device),
                attention_mask=attention_mask.to(extractor.input_device)
                if attention_mask is not None else None,
            ).cpu().numpy()

        mask_np = (
            attention_mask.cpu().numpy()
            if attention_mask is not None
            else np.ones_like(input_ids.numpy())
        )
        ids_np = input_ids.numpy()

        for b in range(B):
            if doc_idx >= num_cached_docs:
                break

            valid_len = int(mask_np[b].sum())
            live = h_live[b, :valid_len]
            off, length = int(offsets[doc_idx]), int(lengths[doc_idx])

            # Bounds check
            if off + length > h_cached.shape[0]:
                print(f"doc {doc_idx}: FAIL — offset {off} + length {length} "
                      f"exceeds cache size {h_cached.shape[0]}")
                all_ok = False
                doc_idx += 1
                continue

            cached = h_cached[off: off + length].astype(np.float32)

            # Shape check
            if live.shape != cached.shape:
                print(f"doc {doc_idx}: FAIL — shape mismatch live={live.shape} cached={cached.shape}")
                all_ok = False
                doc_idx += 1
                continue

            # Input IDs check
            live_ids = ids_np[b, :valid_len]
            cached_ids = ids_cached[off: off + length]
            ids_match = np.array_equal(live_ids, cached_ids)

            # Hidden state error (skip near-zero values for relative error)
            abs_err = np.abs(live - cached)
            magnitude = np.abs(live)
            large_mask = magnitude > 1e-3
            if large_mask.any():
                rel_vals = abs_err[large_mask] / magnitude[large_mask]
                mean_rel_err = float(rel_vals.mean())
                max_rel_err = float(rel_vals.max())
            else:
                mean_rel_err = 0.0
                max_rel_err = 0.0

            status = "OK"
            if not ids_match:
                status = "FAIL (ids mismatch)"
                all_ok = False
            elif max_rel_err > 0.05:
                status = "FAIL (extreme rel_err)"
                all_ok = False
            elif mean_rel_err > 0.002:
                status = "WARN (rel_err above float16 precision)"
                has_warnings = True

            print(f"doc {doc_idx}: len={valid_len}, ids_match={ids_match}, "
                  f"max_abs_err={abs_err.max():.4f}, "
                  f"mean_rel_err={mean_rel_err:.6f}, "
                  f"max_rel_err={max_rel_err:.6f}  [{status}]")

            total_verified += 1
            doc_idx += 1

        batch_count += 1

    extractor.cleanup()

    print(f"\nVerified {total_verified} / {num_cached_docs} cached documents.")
    if all_ok and not has_warnings:
        print("All checks passed. Cache is correct (within float16 rounding).")
    elif all_ok and has_warnings:
        print("All checks passed with WARNINGS. Relative error slightly above float16 "
              "precision for some docs — may be expected for bfloat16 models.")
    else:
        print("Some checks FAILED. Investigate above.")


if __name__ == "__main__":
    main()
