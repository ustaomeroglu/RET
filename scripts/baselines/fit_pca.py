#!/usr/bin/env python3
"""Fit PCA on hidden states and save components for reuse.

Fits PCA on the train split of cached hidden states (or live LLM extraction)
and saves components, mean, and explained variance to disk. The saved
components can be loaded by probing, closure, clustering, or any other
downstream experiment.

Usage:
    # From cached hidden states (fast, no LLM needed)
    python scripts/fit_pca.py \
        --cache-dir data/models_under_pressure/GptOss20B/layer11 \
        --pca-dim 128 \
        --output-dir outputs/GptOss20B/models_under_pressure/baselines/h_pca128

    # From live LLM extraction
    python scripts/fit_pca.py \
        --dataset models_under_pressure \
        --model openai/gpt-oss-20b --layer 11 \
        --pca-dim 128 \
        --output-dir outputs/GptOss20B/models_under_pressure/baselines/h_pca128

    # Different PCA dimensions
    python scripts/fit_pca.py \
        --cache-dir data/models_under_pressure/GptOss20B/layer11 \
        --pca-dim 64 \
        --output-dir outputs/GptOss20B/models_under_pressure/baselines/h_pca64
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Fit PCA on hidden states and save components.")

    src = p.add_argument_group("Data source (provide one)")
    src.add_argument("--cache-dir", type=str, default=None,
                     help="Path to cached hidden states (memmap format)")
    src.add_argument("--dataset", type=str, default=None,
                     choices=["models_under_pressure",
                              "numina_math", "numina_math_64k_samples", "mmlu"],
                     help="Dataset name for live LLM extraction")
    src.add_argument("--model", type=str, default="openai/gpt-oss-20b",
                     help="HuggingFace model ID (used with --dataset)")
    src.add_argument("--layer", type=int, default=12,
                     help="Layer index (used with --dataset)")

    p.add_argument("--pca-dim", type=int, required=True,
                   help="Number of PCA components to keep")
    p.add_argument("--output-dir", type=str, required=True,
                   help="Directory to save pca_components.npz and info.json")
    p.add_argument("--split", type=str, default="train",
                   help="Which split to fit PCA on (default: train)")
    p.add_argument("--pooling-mode", type=str, default="average",
                   choices=["average", "assistant_token"],
                   help="Pooling mode for hidden states (default: average)")

    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)

    if not args.cache_dir and not args.dataset:
        print("Error: provide either --cache-dir or --dataset", file=sys.stderr)
        sys.exit(1)

    t0 = time.time()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if already fitted
    npz_path = output_dir / "pca_components.npz"
    if npz_path.exists():
        logger.info("PCA components already exist at %s. Delete to refit.", npz_path)
        return

    # ------------------------------------------------------------------
    # Load hidden states
    # ------------------------------------------------------------------
    if args.cache_dir:
        from evaluation.probing.hidden_state_io import load_hidden_states_from_memmap
        logger.info("Loading cached hidden states from %s (split=%s, mode=%s)...",
                     args.cache_dir, args.split, args.pooling_mode)
        data = load_hidden_states_from_memmap(
            args.cache_dir, modes={args.pooling_mode}, split=args.split)
        h = data[args.pooling_mode]
        source_info = {"type": "cache", "cache_dir": str(args.cache_dir),
                       "split": args.split, "pooling_mode": args.pooling_mode}
    else:
        # Live LLM extraction
        import torch
        from ret.extraction.hooks import HiddenStateExtractor
        from ret.config import LLMConfig

        model_dtype = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
        llm_config = LLMConfig(name=args.model, layer_index=args.layer, dtype=model_dtype)
        logger.info("Loading LLM %s layer %d for extraction...", args.model, args.layer)
        extractor = HiddenStateExtractor(llm_config, device="cuda")
        tokenizer = extractor.tokenizer
        hidden_dim = extractor.model.config.hidden_size

        # Load dataset
        from scripts.data_preparation_scripts.cache_hidden_states import _load_dataset
        prompts, _ = _load_dataset(args.dataset, tokenizer, args)
        logger.info("Loaded %d prompts, extracting hidden states...", len(prompts))

        # Extract and pool
        all_h = []
        for i, p in enumerate(prompts):
            enc = tokenizer(p, return_tensors="pt")
            input_ids = enc["input_ids"].to("cuda")
            with torch.no_grad():
                h_seq = extractor.extract(input_ids)  # (1, T, D)
            if args.pooling_mode == "average":
                all_h.append(h_seq[0].mean(dim=0).cpu().numpy())
            else:
                all_h.append(h_seq[0, -1].cpu().numpy())
            if (i + 1) % 500 == 0:
                logger.info("  %d / %d extracted", i + 1, len(prompts))

        h = np.stack(all_h).astype(np.float32)
        extractor.cleanup()
        source_info = {"type": "live", "model": args.model, "layer": args.layer,
                       "dataset": args.dataset, "pooling_mode": args.pooling_mode}

    logger.info("Hidden states shape: %s", h.shape)

    # ------------------------------------------------------------------
    # Fit PCA
    # ------------------------------------------------------------------
    from sklearn.decomposition import PCA

    logger.info("Fitting PCA (n_components=%d) on %d samples...", args.pca_dim, h.shape[0])
    pca = PCA(n_components=args.pca_dim)
    pca.fit(h)

    explained = float(pca.explained_variance_ratio_.sum())
    logger.info("Explained variance: %.4f (%.1f%%)", explained, explained * 100)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    np.savez(
        npz_path,
        components=pca.components_,                    # (pca_dim, hidden_dim)
        mean=pca.mean_,                                # (hidden_dim,)
        explained_variance_ratio=pca.explained_variance_ratio_,  # (pca_dim,)
    )
    logger.info("Saved PCA components to %s", npz_path)

    info = {
        "pca_dim": args.pca_dim,
        "hidden_dim": int(h.shape[1]),
        "n_train": int(h.shape[0]),
        "explained_variance": explained,
        "source": source_info,
        "date": datetime.date.today().isoformat(),
    }
    with open(output_dir / "info.json", "w") as f:
        json.dump(info, f, indent=2)
    logger.info("Saved info.json")

    elapsed = time.time() - t0
    logger.info("Done in %.1f s. PCA %d-dim explains %.1f%% of variance.",
                elapsed, args.pca_dim, explained * 100)


if __name__ == "__main__":
    main()
