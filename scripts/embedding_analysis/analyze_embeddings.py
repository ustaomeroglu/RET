#!/usr/bin/env python3
"""Analyze learned embeddings: PCA, t-SNE, effective dimensionality."""

import argparse
import json
from pathlib import Path

from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from ret.data.dataset import build_dataloader, build_cached_dataloader, build_padded_dataloader
from evaluation.baseline_utils import (
    add_baseline_args,
    get_baseline_mode,
    setup_baseline,
    fit_pca_on_h,
    transform_h,
)
from ret.extraction.hooks import HiddenStateExtractor
from ret.utils.checkpointing import load_encoder
from ret.utils.seeding import seed_everything
from ret.utils.prompt_logger import PromptLogger


def effective_dimensionality(z: np.ndarray) -> float:
    """Compute participation ratio as effective dimensionality."""
    z_centered = z - z.mean(axis=0)
    cov = np.cov(z_centered.T)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = eigenvalues[eigenvalues > 0]
    p = eigenvalues / eigenvalues.sum()
    return float(1.0 / (p ** 2).sum())


def temporal_autocorrelation(z: np.ndarray, max_lag: int = 50) -> np.ndarray:
    """Compute mean autocorrelation of z across time lags."""
    T, d = z.shape
    z_centered = z - z.mean(axis=0)
    var = (z_centered ** 2).sum(axis=-1).mean()
    autocorr = []
    for lag in range(max_lag):
        if lag == 0:
            autocorr.append(1.0)
        else:
            corr = (z_centered[:-lag] * z_centered[lag:]).sum(axis=-1).mean() / var
            autocorr.append(float(corr))
    return np.array(autocorr)


def temporal_auto_cosine(z: np.ndarray, max_lag: int = 50) -> np.ndarray:
    """Compute mean cosine similarity between z_t and z_{t+k} for each lag."""
    norms = np.linalg.norm(z, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    z_normed = z / norms
    auto_cos = []
    for lag in range(max_lag):
        if lag == 0:
            auto_cos.append(1.0)
        else:
            cos_sim = (z_normed[:-lag] * z_normed[lag:]).sum(axis=1).mean()
            auto_cos.append(float(cos_sim))
    return np.array(auto_cos)


def svd_effective_rank(z: np.ndarray) -> float:
    """Compute entropy-based effective rank from SVD singular values.

    Different from the participation ratio in effective_dimensionality().
    Effective rank = exp(H(sigma / sum(sigma))) where H is Shannon entropy.

    Reference: Roy & Vetterli (2007).
    """
    z_centered = z - z.mean(axis=0)
    sigma = np.linalg.svd(z_centered, compute_uv=False)
    sigma = sigma[sigma > 1e-12]
    if sigma.size == 0:
        return 0.0
    p = sigma / sigma.sum()
    H = -np.sum(p * np.log(p))
    return float(np.exp(H))


def per_dimension_kurtosis(z: np.ndarray) -> np.ndarray:
    """Compute excess kurtosis for each dimension of z.

    Gaussian excess kurtosis = 0. High values indicate heavy tails or collapse.
    """
    return stats.kurtosis(z, axis=0, fisher=True)


def shapiro_wilk_normality(z: np.ndarray, max_samples: int = 5000) -> np.ndarray:
    """Compute Shapiro-Wilk normality test p-values per dimension.

    Low p-values (< 0.05) indicate the dimension is significantly non-Gaussian.
    Subsamples to max_samples if needed (Shapiro-Wilk hard limit).
    """
    N, d_z = z.shape
    if N > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(N, max_samples, replace=False)
        z = z[idx]

    p_values = np.zeros(d_z)
    for dim in range(d_z):
        _, p = stats.shapiro(z[:, dim])
        p_values[dim] = p
    return p_values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "checkpoint", type=str, nargs="?", default=None,
        help="Path to checkpoint .pt file (required unless a --baseline-* flag is set)",
    )
    parser.add_argument("--max-batches", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-dir", type=str, default=None)
    add_baseline_args(parser, include_output_dir=False)
    args = parser.parse_args()

    mode = get_baseline_mode(args)
    ctx = setup_baseline(mode, args, device=args.device)
    config = ctx.config
    seed_everything(config.training.seed)

    # Resolve output dir: explicit --output-dir > ctx.output_dir/embedding_statistics
    out_dir = Path(args.output_dir) / "embedding_statistics" if args.output_dir else ctx.output_dir / "embedding_statistics"
    out_dir.mkdir(parents=True, exist_ok=True)

    prompt_logger = PromptLogger(out_dir)

    if mode == "checkpoint":
        # Load encoder from checkpoint
        encoder, config, _ckpt = load_encoder(args.checkpoint, device=args.device, config=config)

    # Collect embeddings
    data_mode = config.data.mode
    if data_mode == "cached":
        val_loader = build_cached_dataloader(
            config.data, config.llm, split="val",
            batch_size=config.training.batch_size, shuffle=False,
        )
    elif data_mode == "document":
        val_loader = build_padded_dataloader(
            config.data, config.llm, split="val",
            batch_size=config.training.batch_size, shuffle=False,
        )
    else:
        val_loader = build_dataloader(
            config.data, config.llm, split="val",
            max_chunks=config.training.num_chunks,
            batch_size=config.training.batch_size, shuffle=False,
        )

    all_z = []
    all_h = []  # only used for PCA fitting
    seq_lengths = []  # track per-sequence lengths for autocorrelation
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= args.max_batches:
                break

            if data_mode == "cached":
                h = batch["hidden_states"].to(args.device)
                mask = batch["attention_mask"]  # (B, T)
            else:
                if not prompt_logger.is_full:
                    source = "document" if data_mode == "document" else "concat_chunk"
                    prompt_logger.log_forward_pass_batch(
                        batch["input_ids"], ctx.extractor.tokenizer, source=source,
                    )
                h = ctx.extractor.extract(batch["input_ids"])  # (B, T, d_h)
                mask = None

            if mode == "checkpoint":
                z, aux = encoder(h)
                z = aux.get("z_pre", z)
                if mask is not None:
                    # Variable-length: extract real tokens per sequence
                    for b in range(z.shape[0]):
                        L = int(mask[b].sum())
                        all_z.append(z[b, :L].cpu())
                        seq_lengths.append(L)
                else:
                    for b in range(z.shape[0]):
                        all_z.append(z[b].cpu())
                        seq_lengths.append(z.shape[1])
            else:
                # Baseline: collect h first (PCA needs fitting pass)
                if mask is not None:
                    for b in range(h.shape[0]):
                        L = int(mask[b].sum())
                        all_h.append(h[b, :L].cpu())
                        seq_lengths.append(L)
                else:
                    for b in range(h.shape[0]):
                        all_h.append(h[b].cpu())
                        seq_lengths.append(h.shape[1])

    prompt_logger.save()

    if mode == "checkpoint":
        z_flat = torch.cat(all_z, dim=0).numpy()  # (total_tokens, d_z)
    else:
        h_flat_all = torch.cat(all_h, dim=0)  # (total_tokens, d_h)

        # For PCA: fit on flat h
        if mode == "pca":
            h_flat_np = h_flat_all.numpy()
            comps, mean = fit_pca_on_h(h_flat_np, args.pca_dim)
            ctx.pca_components = comps
            ctx.pca_mean = mean

        # Transform h -> representation (needs 3D input)
        # Process per-sequence to handle variable lengths
        z_parts = []
        offset = 0
        for L in seq_lengths:
            h_seq = h_flat_all[offset:offset + L].unsqueeze(0)  # (1, L, d_h)
            z_seq = transform_h(h_seq, ctx, device=args.device).cpu()
            z_parts.append(z_seq.squeeze(0))  # (L, d_repr)
            offset += L
        z_flat = torch.cat(z_parts, dim=0).numpy()

    d_z = z_flat.shape[-1]
    n_tokens = z_flat.shape[0]
    n_seqs = len(seq_lengths)

    if ctx.extractor is not None:
        ctx.extractor.cleanup()

    print(f"Collected {n_seqs} sequences, {n_tokens} token embeddings, d_z={d_z}")

    # Mean norm and norm of mean
    mean_norm = float(np.linalg.norm(z_flat, axis=1).mean())
    norm_of_mean = float(np.linalg.norm(z_flat.mean(axis=0)))
    print(f"Mean ||z_t||: {mean_norm:.4f}, ||mean(z_t)||: {norm_of_mean:.4f}")

    # Effective dimensionality
    eff_dim = effective_dimensionality(z_flat)
    print(f"Effective dimensionality: {eff_dim:.1f} / {d_z}")

    # PCA
    pca = PCA(n_components=min(d_z, 20))
    pca.fit(z_flat)
    explained = np.cumsum(pca.explained_variance_ratio_)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(1, len(explained) + 1), pca.explained_variance_ratio_, alpha=0.7)
    ax.plot(range(1, len(explained) + 1), explained, "r-o", markersize=3)
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Variance Explained")
    ax.set_title(f"PCA Spectrum (eff. dim = {eff_dim:.1f})")
    fig.savefig(out_dir / "pca_spectrum.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved PCA spectrum to {out_dir / 'pca_spectrum.png'}")

    # t-SNE on subset (skip when effective dim is very low — degenerate/collapsed z)
    try:
        if eff_dim < 2.5:
            print(f"Skipping t-SNE (effective dim {eff_dim:.1f} too low, would be numerically unstable)")
        else:
            n_tsne = min(5000, len(z_flat))
            idx = np.random.choice(len(z_flat), n_tsne, replace=False)
            z_sub = z_flat[idx]
            tsne = TSNE(n_components=2, perplexity=min(30, n_tsne - 1), random_state=42)
            z_2d = tsne.fit_transform(z_sub)
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.scatter(z_2d[:, 0], z_2d[:, 1], s=1, alpha=0.3)
            ax.set_title("t-SNE of Learned Embeddings")
            fig.savefig(out_dir / "tsne.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved t-SNE to {out_dir / 'tsne.png'}")
    except Exception as e:
        import traceback
        print(f"Skipping t-SNE (failed: {e})")
        traceback.print_exc()

    # Temporal autocorrelation (first sequence)
    z_first_seq = z_flat[:seq_lengths[0]]
    autocorr = temporal_autocorrelation(z_first_seq)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(autocorr)
    ax.set_xlabel("Lag")
    ax.set_ylabel("Autocorrelation")
    ax.set_title("Temporal Autocorrelation of z")
    ax.axhline(y=0, color="k", linestyle="--", alpha=0.3)
    fig.savefig(out_dir / "autocorrelation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved autocorrelation to {out_dir / 'autocorrelation.png'}")

    # Auto cosine similarity
    auto_cos = temporal_auto_cosine(z_first_seq)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(auto_cos)
    ax.set_xlabel("Lag")
    ax.set_ylabel("Cosine Similarity")
    ax.set_title("Temporal Auto-Cosine of z")
    ax.axhline(y=0, color="k", linestyle="--", alpha=0.3)
    fig.savefig(out_dir / "auto_cosine.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved auto-cosine to {out_dir / 'auto_cosine.png'}")

    # SVD Spectrum Analysis
    print("\n--- Spectrum Analysis ---")

    z_centered = z_flat - z_flat.mean(axis=0)
    singular_values = np.linalg.svd(z_centered, compute_uv=False)
    sv_positive = singular_values[singular_values > 1e-12]
    if sv_positive.size == 0:
        eff_rank_svd = 0.0
    else:
        p_sv = sv_positive / sv_positive.sum()
        eff_rank_svd = float(np.exp(-np.sum(p_sv * np.log(p_sv))))
    sv_total = singular_values.sum()
    sv_normalized = singular_values / sv_total if sv_total > 0 else singular_values

    print(f"SVD effective rank: {eff_rank_svd:.1f} / {d_z}")
    print(f"Participation ratio (existing): {eff_dim:.1f} / {d_z}")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(1, len(sv_normalized) + 1), sv_normalized, alpha=0.7, color="steelblue")
    ax.set_xlabel("Singular Value Index")
    ax.set_ylabel("Normalized Singular Value")
    ax.set_title(f"SVD Spectrum (eff. rank = {eff_rank_svd:.1f}, participation ratio = {eff_dim:.1f})")
    ax.grid(True, alpha=0.3)
    fig.savefig(out_dir / "svd_spectrum.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved SVD spectrum to {out_dir / 'svd_spectrum.png'}")

    # Per-Dimension Kurtosis
    kurt = per_dimension_kurtosis(z_flat)
    print(f"Kurtosis: mean={np.nanmean(kurt):.2f}, std={np.nanstd(kurt):.2f}, "
          f"max={np.nanmax(kurt):.2f}, min={np.nanmin(kurt):.2f}")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(1, d_z + 1), kurt, alpha=0.7, color="darkorange")
    ax.axhline(y=0, color="black", linestyle="--", linewidth=0.8, alpha=0.5, label="Gaussian (k=0)")
    ax.set_xlabel("Dimension Index")
    ax.set_ylabel("Excess Kurtosis")
    ax.set_title(f"Per-Dimension Kurtosis (mean={np.nanmean(kurt):.2f})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(out_dir / "kurtosis.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved kurtosis plot to {out_dir / 'kurtosis.png'}")

    # Shapiro-Wilk Normality Test
    sw_pvals = shapiro_wilk_normality(z_flat)
    n_nongaussian = (sw_pvals < 0.05).sum()
    print(f"Shapiro-Wilk: {n_nongaussian}/{d_z} dimensions significantly non-Gaussian (p<0.05)")

    # Save all spectrum metrics to JSON (sanitize NaN → null for JSON compliance)
    def _sanitize(val):
        if isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
            return None
        if isinstance(val, list):
            return [_sanitize(v) for v in val]
        return val

    spectrum_results = {
        "mean_norm": mean_norm,
        "norm_of_mean": norm_of_mean,
        "effective_rank_svd": float(eff_rank_svd),
        "effective_dim_participation": float(eff_dim),
        "d_z": int(d_z),
        "num_samples": int(z_flat.shape[0]),
        "svd_singular_values_normalized": sv_normalized.tolist(),
        "kurtosis_per_dimension": _sanitize(kurt.tolist()),
        "kurtosis_summary": {
            "mean": _sanitize(float(np.nanmean(kurt))),
            "std": _sanitize(float(np.nanstd(kurt))),
            "min": _sanitize(float(np.nanmin(kurt))),
            "max": _sanitize(float(np.nanmax(kurt))),
            "median": _sanitize(float(np.nanmedian(kurt))),
        },
        "shapiro_wilk_p_values": sw_pvals.tolist(),
        "shapiro_wilk_summary": {
            "num_nongaussian_p005": int(n_nongaussian),
            "fraction_nongaussian_p005": float(n_nongaussian / d_z),
            "min_p": float(sw_pvals.min()),
            "max_p": float(sw_pvals.max()),
            "median_p": float(np.median(sw_pvals)),
        },
        "auto_cosine_first_seq": auto_cos.tolist(),
        "pca_explained_variance_top20": pca.explained_variance_ratio_.tolist(),
    }

    with open(out_dir / "spectrum_analysis.json", "w") as f:
        json.dump(spectrum_results, f, indent=2)
    print(f"Saved spectrum analysis to {out_dir / 'spectrum_analysis.json'}")


if __name__ == "__main__":
    main()
