#!/usr/bin/env python3
"""Fit streaming k-means cluster centers on RET/LLM representations (TRAIN split).

Three-pass streaming architecture:
  Pass 0 (h_pca/sae only): fit IncrementalPCA on raw h to learn projection directions.
  Pass 1 (subsampled): accumulate mean statistics for preprocessing.
  Pass 2 (full TRAIN): stream through all batches and call partial_fit on MiniBatchKMeans.

Supported variants (set via config.closure.variant):
  z_ret  - encoder output z, center + L2-normalized (requires RET checkpoint)
  h      - raw LLM hidden states, center + L2-normalized
  h_pca  - PCA-projected h, center + L2-normalized (pca_dim from config.closure.pca_dim)
  sae    - SAE-encoded h through a pre-trained SAE then applies PCA

Usage:
    # RET z variant (pass checkpoint or its config.yaml):
    python scripts/clustering/train_centers.py outputs/.../ret_jepa_auth/config.yaml

    # h baseline:
    python scripts/clustering/train_centers.py configs/GptOss20B/baselines/h.yaml

    # h_pca128 baseline:
    python scripts/clustering/train_centers.py configs/GptOss20B/baselines/h_pca128.yaml

    # Custom grid:
    python scripts/clustering/train_centers.py configs/GptOss20B/baselines/h.yaml \\
        --K-list 32 64 128 --seeds 42 43 44 --window-sizes 1 4 8
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from ret.config import ExperimentConfig
from ret.extraction.hooks import HiddenStateExtractor
from ret.models.encoders import build_encoder
from ret.utils.seeding import seed_everything
from evaluation.baselines.sae import encode_sae
from evaluation.clustering.config import (
    build_dataloader,
    load_config_with_overlay,
)
from evaluation.clustering.core import (
    StreamingPreprocessor,
    apply_pca_projection,
    apply_window_and_stride,
    create_kmeans,
    extract_features,
    save_kmeans,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Map from config closure.variant names to internal variant tokens
VARIANT_MAP = {"z_ret": "z", "h": "h", "h_pca": "h_pca", "sae": "sae"}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit streaming k-means cluster centers on RET/LLM representations."
    )

    # Single positional arg: path to .yaml config OR .pt checkpoint
    parser.add_argument(
        "config",
        type=str,
        help=(
            "Path to a YAML config file or an RET .pt checkpoint. "
            "If .pt: uses parent_dir/config.yaml for the config and the .pt for the encoder. "
            "If .yaml: derives checkpoint from logging.output_dir/checkpoint_final.pt when "
            "the z_ret variant is selected."
        ),
    )

    # SAE
    parser.add_argument(
        "--sae-checkpoint", type=str, default=None,
        help="Path to SAE checkpoint .pt. If unset and LLM is openai/gpt-oss-20b with "
             "layer_index in {3,11,15,19,23}, auto-fetches andyrdt/saes-gpt-oss-20b for "
             "that layer; otherwise auto-downloads Goodfire L15 from HF.",
    )

    # Grid parameters
    parser.add_argument(
        "--K-list", type=int, nargs="+", default=[16, 32, 64],
        help="List of K values for k-means (default: 16 32 64)",
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[42, 43, 44],
        help="Explicit list of random seeds per (variant, w, K) combo (default: 42 43 44)",
    )
    parser.add_argument(
        "--window-sizes", type=int, nargs="+", default=[1, 8],
        help="Causal moving-average window sizes (default: 1 8)",
    )
    parser.add_argument(
        "--stride", type=int, default=1,
        help="Token stride when subsampling (default: 1)",
    )

    # Pass 1 settings
    parser.add_argument(
        "--pass1-docs", type=int, default=10000,
        help="Number of documents to use in pass 1 for stat accumulation (default: 10000)",
    )

    # Training
    parser.add_argument(
        "--kmeans-batch-size", type=int, default=65536,
        help="Mini-batch size for MiniBatchKMeans.partial_fit (default: 65536)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=4,
        help="Document batch size for the dataloader (default: 4)",
    )
    parser.add_argument(
        "--max-train-docs", type=int, default=None,
        help="Max documents to process in pass 2 (default: all). Useful for quick tests.",
    )

    # Output
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Override root directory for output artifacts "
             "(default: logging.output_dir/clustering/ from config)",
    )
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device (default: cuda if available, else cpu)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Global random seed (default: 42)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing output artifacts",
    )
    parser.add_argument(
        "--center-only", action="store_true",
        help="Mean-subtract features but skip L2 normalization (variant suffix '_centered'). "
             "Use a separate --output-dir to avoid clobbering existing centered+normalized runs.",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Saving helpers
# ---------------------------------------------------------------------------


def _save_metrics(path: Path, model, variant: str, w: int, K: int, actual_seed: int) -> None:
    """Save metrics.json with summary statistics for a fitted k-means model."""
    metrics = {
        "inertia": float(model.inertia_),
        "n_clusters": int(model.n_clusters),
        "seed": int(actual_seed),
        "dim": int(model.cluster_centers_.shape[1]),
        "n_samples_seen": int(getattr(model, "n_steps_", getattr(model, "n_iter_", -1))),
    }
    with open(path / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)


def _save_config_snapshot(
    path: Path,
    config: ExperimentConfig,
    config_yaml_path: str,
    checkpoint: str | None,
) -> None:
    """Save a full config snapshot (config dict + resolved paths) to config.json."""
    snapshot = {
        "ret_config": asdict(config),
        "config_yaml_path": config_yaml_path,
        "checkpoint_path": checkpoint,
    }
    with open(path / "config.json", "w") as f:
        json.dump(snapshot, f, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv=None) -> None:
    args = _parse_args(argv)

    # ------------------------------------------------------------------
    # 1. Detect mode and load config
    # ------------------------------------------------------------------
    arg = args.config
    if arg.endswith(".pt"):
        ckpt_path = Path(arg)
        config_yaml = str(ckpt_path.parent / "config.yaml")
        config = load_config_with_overlay(config_yaml)
        checkpoint = str(ckpt_path)
        logger.info("Loaded config from %s (inferred from checkpoint)", config_yaml)
    else:
        config_yaml = arg
        config = load_config_with_overlay(config_yaml)
        checkpoint = None
        logger.info("Loaded config from %s", config_yaml)

    # ------------------------------------------------------------------
    # 2. Derive variant
    # ------------------------------------------------------------------
    closure_variant = config.closure.variant
    if closure_variant is None:
        logger.error(
            "Config must have closure.variant set (z_ret, h, h_pca, or sae). "
            "Add 'closure:\\n  variant: h' (or similar) to your config YAML."
        )
        sys.exit(1)
    variant = VARIANT_MAP.get(closure_variant)
    if variant is None:
        logger.error(
            "Unknown closure.variant: %s. Expected one of %s",
            closure_variant,
            list(VARIANT_MAP.keys()),
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # 3. Derive checkpoint for z variant
    # ------------------------------------------------------------------
    if variant == "z" and checkpoint is None:
        checkpoint = str(Path(config.logging.output_dir) / "checkpoint_final.pt")
        if not Path(checkpoint).exists():
            logger.error(
                "z variant requires a checkpoint at %s (not found). "
                "Either pass the .pt file directly or ensure checkpoint_final.pt exists.",
                checkpoint,
            )
            sys.exit(1)

    # ------------------------------------------------------------------
    # 4. Derive output directory
    # ------------------------------------------------------------------
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(config.logging.output_dir) / "clustering"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: %s", output_dir)

    # ------------------------------------------------------------------
    # 5. Derive PCA settings
    # ------------------------------------------------------------------
    pca_dim = config.closure.pca_dim
    pca_dir = Path(config.logging.output_dir)

    # ------------------------------------------------------------------
    # 6. Setup (seed, device)
    # ------------------------------------------------------------------
    seed_everything(args.seed)
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU.")
        device = "cpu"

    if args.center_only:
        if variant in ("h_pca", "sae"):
            logger.error(
                "--center-only is currently supported only for z/h variants, not '%s' "
                "(SAE/PCA paths gate behavior on exact variant name).",
                variant,
            )
            sys.exit(1)
        variant = f"{variant}_centered"
        logger.info("--center-only enabled: variant -> %s (mean-subtract, no L2 normalize)", variant)

    logger.info("Variant: %s, Output: %s", variant, output_dir)

    # Single-element list — forward-compatible with the pass loops below
    expanded_variants = [variant]

    # ------------------------------------------------------------------
    # 7. Build HiddenStateExtractor (skip in cached mode)
    # ------------------------------------------------------------------
    is_cached = config.data.mode == "cached"
    extractor = None
    if not is_cached:
        logger.info(
            "Building HiddenStateExtractor (LLM: %s, layer: %s) ...",
            config.llm.name,
            config.llm.layer_index,
        )
        extractor = HiddenStateExtractor(config.llm, device=device)
    else:
        logger.info("Cached mode: skipping HiddenStateExtractor (hidden states from dataloader)")

    # ------------------------------------------------------------------
    # 8. Build encoder (if z variant)
    # ------------------------------------------------------------------
    encoder = None
    if variant.startswith("z"):
        logger.info("Loading RET encoder from checkpoint: %s", checkpoint)
        ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
        li = config.llm.layer_index
        if li == "all":
            # Infer concat_dim from checkpoint weights
            dp_key = "down_proj.0.weight"
            concat_dim = ckpt["encoder"][dp_key].shape[1] if dp_key in ckpt["encoder"] else None
        elif isinstance(li, list) and len(li) > 1:
            concat_dim = len(li) * config.llm.hidden_dim
        else:
            concat_dim = None
        encoder = build_encoder(config.llm.hidden_dim, config.encoder, concat_dim=concat_dim).to(device)
        encoder.load_state_dict(ckpt["encoder"])
        encoder.eval()
        logger.info("Encoder loaded and set to eval mode.")

    # ------------------------------------------------------------------
    # 9. Build TRAIN dataloader
    # ------------------------------------------------------------------
    logger.info("Building TRAIN dataloader (batch_size=%d) ...", args.batch_size)
    train_loader = build_dataloader(config, split="train", batch_size=args.batch_size, seed=args.seed)

    # ------------------------------------------------------------------
    # 10. Determine which raw features are needed
    # ------------------------------------------------------------------
    need_h = variant.startswith("h") or variant == "sae"
    need_z = variant.startswith("z")
    logger.info("need_h=%s  need_z=%s", need_h, need_z)

    # ------------------------------------------------------------------
    # 11. Load SAE model (if sae variant)
    # ------------------------------------------------------------------
    sae_model = None
    sae_source_path: str | None = None
    if variant == "sae":
        from evaluation.probing.sae_utils import load_sae_encoder

        ckpt_arg = args.sae_checkpoint
        # Auto-fetch andyrdt SAE for gpt-oss-20b layers 3,11,15,19,23.
        if ckpt_arg is None and config.llm.name == "openai/gpt-oss-20b" and int(config.llm.layer_index) in (3, 11, 15, 19, 23):
            from huggingface_hub import hf_hub_download
            ckpt_arg = hf_hub_download(
                "andyrdt/saes-gpt-oss-20b",
                f"resid_post_layer_{int(config.llm.layer_index)}/trainer_0/ae.pt",
            )
            logger.info("Auto-fetched andyrdt SAE: %s", ckpt_arg)

        sae_model, sae_cfg = load_sae_encoder(
            checkpoint_path=ckpt_arg, device=device,
        )
        sae_source_path = sae_cfg["source"]
        logger.info(
            "SAE loaded: d_in=%d, d_sae=%d, k=%d, source=%s",
            sae_cfg["d_in"], sae_cfg["d_sae"], sae_cfg["k"], sae_cfg["source"],
        )

    # ------------------------------------------------------------------
    # 12. PASS 0: Fit IncrementalPCA (for h_pca and/or sae variants)
    # ------------------------------------------------------------------
    # PCA components keyed by variant name
    pca_components: dict[str, np.ndarray] = {}
    pca_mean: dict[str, np.ndarray] = {}

    pca_variants = [v for v in expanded_variants if v in ("h_pca", "sae")]
    if pca_variants:
        from sklearn.decomposition import IncrementalPCA

        if pca_dim is None:
            logger.error(
                "h_pca/sae variant requires config.closure.pca_dim to be set. "
                "Add 'closure:\\n  pca_dim: 128' to your config YAML."
            )
            sys.exit(1)

        for pca_var in pca_variants:
            # PCA artifacts live in pca_dir (== logging.output_dir)
            pca_save_dir = pca_dir
            pca_npz_path = pca_save_dir / "pca_components.npz"

            # Auto-fit: load if exists, fit if not
            if pca_npz_path.exists() and not args.force:
                saved = np.load(pca_npz_path)
                pca_components[pca_var] = saved["components"].astype(np.float32)
                pca_mean[pca_var] = saved["mean"].astype(np.float32)
                logger.info(
                    "=== Pass 0: loaded cached PCA for '%s' from %s ===",
                    pca_var, pca_npz_path,
                )
                logger.info(
                    "  PCA dim: %d, explained variance (top-5): %s",
                    int(saved.get("pca_dim", saved["components"].shape[0])),
                    saved["explained_variance_ratio"][:5],
                )
                continue

            logger.info(
                "=== Pass 0: fitting IncrementalPCA for '%s' (dim=%d, up to %d docs) ===",
                pca_var, pca_dim, args.pass1_docs,
            )
            ipca = IncrementalPCA(n_components=pca_dim)

            pca_loader = build_dataloader(
                config, split="train", batch_size=args.batch_size, seed=args.seed,
            )
            pca_doc_count = 0
            for batch in pca_loader:
                features = extract_features(
                    batch, extractor, encoder, need_h=True, need_z=False, device=device,
                )
                mask = batch.get("attention_mask")
                h = features["h"]  # (B, T, d_h) on CPU
                B, T, d_h = h.shape

                # SAE encoding before PCA
                if pca_var == "sae":
                    h = encode_sae(sae_model, h)
                    d_h = h.shape[-1]

                # Flatten valid tokens
                if mask is not None:
                    valid_mask = mask.bool()
                    tokens_flat = (
                        h[valid_mask].numpy()
                        if isinstance(h, torch.Tensor)
                        else h[valid_mask.numpy()]
                    )
                else:
                    tokens_flat = (
                        h.reshape(-1, d_h).numpy()
                        if isinstance(h, torch.Tensor)
                        else h.reshape(-1, d_h)
                    )

                if tokens_flat.shape[0] >= pca_dim:
                    ipca.partial_fit(tokens_flat.astype(np.float32))

                pca_doc_count += B
                if pca_doc_count % max(args.batch_size, 100) < args.batch_size:
                    logger.info(
                        "  Pass 0 [%s]: processed %d / %d docs",
                        pca_var, pca_doc_count, args.pass1_docs,
                    )
                if pca_doc_count >= args.pass1_docs:
                    break

            pca_components[pca_var] = ipca.components_.astype(np.float32)
            pca_mean[pca_var] = ipca.mean_.astype(np.float32)

            # Save PCA artifacts
            pca_save_dir.mkdir(parents=True, exist_ok=True)
            np.savez(
                pca_npz_path,
                components=pca_components[pca_var],
                mean=pca_mean[pca_var],
                explained_variance_ratio=ipca.explained_variance_ratio_.astype(np.float32),
                pca_dim=pca_dim,
            )
            logger.info(
                "Pass 0 [%s] complete. PCA fitted on %d docs, saved to %s",
                pca_var, pca_doc_count, pca_npz_path,
            )
            logger.info("  Explained variance (top-5): %s", ipca.explained_variance_ratio_[:5])
            logger.info("  Total explained variance: %.4f", ipca.explained_variance_ratio_.sum())

    # ------------------------------------------------------------------
    # 13. Create preprocessors: one per (variant_name, window_size)
    # ------------------------------------------------------------------
    preprocessors: dict[tuple[str, int], StreamingPreprocessor] = {}
    for variant_name in expanded_variants:
        for w in args.window_sizes:
            pp = StreamingPreprocessor(variant_name, w, args.stride)
            preprocessors[(variant_name, w)] = pp

    # ------------------------------------------------------------------
    # PASS 1: Subsampled TRAIN — accumulate stats for preprocessors
    # ------------------------------------------------------------------
    logger.info("=== Pass 1: accumulating statistics (up to %d docs) ===", args.pass1_docs)
    doc_count = 0

    for batch in train_loader:
        features = extract_features(batch, extractor, encoder, need_h, need_z, device)
        mask = batch.get("attention_mask")
        if mask is not None:
            mask = mask.to("cpu")

        for variant_name in expanded_variants:
            raw_key = "z" if variant_name.startswith("z") else "h"
            raw_feat = features[raw_key]  # (B, T, D) on CPU

            if variant_name == "sae":
                raw_feat = encode_sae(sae_model, raw_feat)
            if variant_name in ("h_pca", "sae"):
                raw_feat = apply_pca_projection(
                    raw_feat, pca_components[variant_name], pca_mean[variant_name],
                )

            for w in args.window_sizes:
                windowed, _ = apply_window_and_stride(raw_feat, mask, w, args.stride)
                if windowed.shape[0] > 0:
                    preprocessors[(variant_name, w)].accumulate(windowed)

        batch_docs = batch["input_ids"].shape[0]
        doc_count += batch_docs

        if doc_count % max(args.batch_size, 100) < args.batch_size:
            logger.info("  Pass 1: processed %d / %d docs", doc_count, args.pass1_docs)

        if doc_count >= args.pass1_docs:
            break

    logger.info("Pass 1 complete. Total docs processed: %d", doc_count)

    # Finalize and save preprocessors (flat layout: output_dir/w{W}/)
    for (variant_name, w), pp in preprocessors.items():
        pp.finalize_fit()
        pp_dir = output_dir / f"w{w}"
        pp.save(pp_dir)
        logger.info("  Saved preprocessor: %s", pp_dir)

    # ------------------------------------------------------------------
    # PASS 2: Full TRAIN — fit k-means
    # ------------------------------------------------------------------
    logger.info("=== Pass 2: fitting k-means (full TRAIN) ===")

    K_list = args.K_list
    seed_list = args.seeds

    kmeans_dict: dict[tuple[str, int, int, int], object] = {}
    for variant_name in expanded_variants:
        for w in args.window_sizes:
            for K in K_list:
                for seed in seed_list:
                    km = create_kmeans(K, seed=seed, batch_size=args.kmeans_batch_size)
                    kmeans_dict[(variant_name, w, K, seed)] = km

    total_kmeans = len(kmeans_dict)
    logger.info("Created %d k-means instances.", total_kmeans)

    # Build a fresh dataloader for pass 2
    train_loader_p2 = build_dataloader(
        config, split="train", batch_size=args.batch_size, seed=args.seed,
    )

    batch_idx = 0
    p2_doc_count = 0
    for batch in train_loader_p2:
        features = extract_features(batch, extractor, encoder, need_h, need_z, device)
        mask = batch.get("attention_mask")
        if mask is not None:
            mask = mask.to("cpu")

        for variant_name in expanded_variants:
            raw_key = "z" if variant_name.startswith("z") else "h"
            raw_feat = features[raw_key]

            if variant_name == "sae":
                raw_feat = encode_sae(sae_model, raw_feat)
            if variant_name in ("h_pca", "sae"):
                raw_feat = apply_pca_projection(
                    raw_feat, pca_components[variant_name], pca_mean[variant_name],
                )

            for w in args.window_sizes:
                windowed, _ = apply_window_and_stride(raw_feat, mask, w, args.stride)
                if windowed.shape[0] == 0:
                    continue
                preprocessed = preprocessors[(variant_name, w)].transform(windowed)

                for K in K_list:
                    for seed in seed_list:
                        kmeans_dict[(variant_name, w, K, seed)].partial_fit(preprocessed)

        batch_idx += 1
        p2_doc_count += batch["input_ids"].shape[0]
        if batch_idx % 100 == 0:
            logger.info(
                "  Pass 2: processed %d docs (%d batches)", p2_doc_count, batch_idx,
            )

        if args.max_train_docs is not None and p2_doc_count >= args.max_train_docs:
            logger.info(
                "  Pass 2: reached --max-train-docs limit (%d docs)", p2_doc_count,
            )
            break

    logger.info("Pass 2 complete. Processed %d docs (%d batches).", p2_doc_count, batch_idx)

    # ------------------------------------------------------------------
    # Save k-means models and metadata (flat layout)
    # ------------------------------------------------------------------
    logger.info("Saving k-means models ...")
    abs_config_yaml = str(Path(config_yaml).resolve())
    for (variant_name, w, K, actual_seed), km in kmeans_dict.items():
        # Flat layout: output_dir/w{W}/K{K}/seed{S}/
        save_dir = output_dir / f"w{w}" / f"K{K}" / f"seed{actual_seed}"
        save_dir.mkdir(parents=True, exist_ok=True)

        # Skip if already exists and --force not set
        if (save_dir / "kmeans_model.pkl").exists() and not args.force:
            logger.info("  Skipping (exists): %s", save_dir)
            continue

        save_kmeans(km, save_dir)
        _save_metrics(save_dir, km, variant_name, w, K, actual_seed)
        _save_config_snapshot(save_dir, config, abs_config_yaml, checkpoint)

        from evaluation.clustering.plotting.plots import plot_center_scatter_2d
        plot_center_scatter_2d(
            km.cluster_centers_,
            save_dir / "center_scatter_2d.png",
            title=f"Cluster Centers — K={K} seed={actual_seed}",
        )

        logger.info(
            "  Saved: %s  (K=%d, seed=%d, inertia=%.2f)",
            save_dir, K, actual_seed, km.inertia_,
        )

    # ------------------------------------------------------------------
    # Save clustering_meta.json at output_dir
    # ------------------------------------------------------------------
    meta = {
        "variant": variant,
        "pca_dim": pca_dim,
        "config_yaml_path": abs_config_yaml,
        "checkpoint_path": checkpoint,
        "pca_dir": str(pca_dir) if pca_dim is not None else None,
        "sae_checkpoint": sae_source_path,
    }
    with open(output_dir / "clustering_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    logger.info("Saved clustering_meta.json to %s", output_dir)

    logger.info("Done. All artifacts saved to: %s", output_dir)


if __name__ == "__main__":
    main()
