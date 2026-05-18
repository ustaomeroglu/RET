#!/usr/bin/env python3
"""Pre-compute PCA components for SAE clustering variants.

IncrementalPCA on 131072-d SAE activations is infeasible (each partial_fit
does an SVD that is effectively O(d^2 * k)). Instead we draw a fixed sample
of SAE activations, fit a randomized PCA once, and save the components to
``<clustering_dir>/pca_components.npz`` so ``train_centers.py`` skips Pass 0.

Usage
-----
    python scripts/clustering/precompute_sae_pca.py <sae_config.yaml> [--samples N]

The config should have ``closure.variant: sae`` and ``closure.pca_dim``
set. The output file is written under
``<config.logging.output_dir>/clustering/``.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ret.config import load_config  # noqa: E402
from ret.data.dataset import build_cached_dataloader, build_dataloader, build_padded_dataloader  # noqa: E402
from evaluation.probing.sae_utils import load_sae_encoder  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def build_train_loader(config, batch_size: int):
    mode = config.data.mode
    if mode == "cached":
        return build_cached_dataloader(
            config.data, config.llm, split="train",
            batch_size=batch_size, seed=42,
        )
    if mode == "document":
        return build_padded_dataloader(
            config.data, config.llm, split="train",
            batch_size=batch_size, seed=42,
        )
    return build_dataloader(
        config.data, config.llm, split="train",
        batch_size=batch_size, seed=42,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="SAE baseline config YAML")
    parser.add_argument("--samples", type=int, default=100_000,
                        help="Number of SAE activation vectors to sample (default 100k)")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Dataloader batch size")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing pca_components.npz")
    args = parser.parse_args()

    config = load_config(args.config)
    assert config.closure.variant == "sae", "Expected closure.variant=sae"
    pca_dim = config.closure.pca_dim
    assert pca_dim is not None, "Expected closure.pca_dim to be set"

    out_dir = Path(config.logging.output_dir) / "clustering"
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / "pca_components.npz"
    if npz_path.exists() and not args.force:
        logger.info("Already exists: %s (use --force to overwrite)", npz_path)
        return

    logger.info("Loading SAE encoder ...")
    # Auto-fetch andyrdt SAE for gpt-oss-20b if applicable.
    ckpt_arg = None
    if config.llm.name == "openai/gpt-oss-20b" and int(config.llm.layer_index) in (3, 11, 15, 19, 23):
        from huggingface_hub import hf_hub_download
        ckpt_arg = hf_hub_download(
            "andyrdt/saes-gpt-oss-20b",
            f"resid_post_layer_{int(config.llm.layer_index)}/trainer_0/ae.pt",
        )
        logger.info("Using andyrdt SAE: %s", ckpt_arg)
    sae, sae_cfg = load_sae_encoder(checkpoint_path=ckpt_arg, device=args.device)
    logger.info("SAE: d_in=%d, d_sae=%d, k=%d", sae_cfg["d_in"], sae_cfg["d_sae"], sae_cfg["k"])
    d_sae = sae.d_sae

    logger.info("Building TRAIN dataloader ...")
    loader = build_train_loader(config, args.batch_size)

    # Draw up to `args.samples` SAE activation rows, accumulating on CPU
    needed = args.samples
    acc = np.empty((needed, d_sae), dtype=np.float32)
    filled = 0
    with torch.no_grad():
        for batch in loader:
            if "hidden_states" in batch:
                h = batch["hidden_states"].to(args.device)
            else:
                raise RuntimeError("SAE PCA precompute expects cached mode")
            B, T, dH = h.shape
            mask = batch.get("attention_mask")
            if mask is not None:
                mask = mask.to(h.device).bool()
                h_valid = h[mask]
            else:
                h_valid = h.reshape(-1, dH)
            # Encode in chunks to bound GPU memory
            for i in range(0, h_valid.shape[0], 4096):
                chunk = sae.encode(h_valid[i : i + 4096]).to("cpu").numpy().astype(np.float32)
                take = min(chunk.shape[0], needed - filled)
                if take <= 0:
                    break
                acc[filled : filled + take] = chunk[:take]
                filled += take
                if filled >= needed:
                    break
            if filled >= needed:
                break
            if filled % 20000 < 4096:
                logger.info("  accumulated %d / %d samples", filled, needed)
    acc = acc[:filled]
    logger.info("Collected %d SAE activation rows (d=%d). Fitting randomized PCA ...", filled, d_sae)

    pca = PCA(n_components=pca_dim, svd_solver="randomized", random_state=0)
    pca.fit(acc)
    logger.info("Total explained variance: %.4f", pca.explained_variance_ratio_.sum())

    np.savez(
        npz_path,
        components=pca.components_.astype(np.float32),
        mean=pca.mean_.astype(np.float32),
        explained_variance_ratio=pca.explained_variance_ratio_.astype(np.float32),
        pca_dim=pca_dim,
    )
    logger.info("Saved PCA to %s", npz_path)


if __name__ == "__main__":
    main()
