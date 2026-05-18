#!/usr/bin/env python3
"""Compute effective dimension and rank for raw h and PCA projections.

Reads cached hidden states directly via memmap. No LLM loading needed.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CACHE_DIR = Path("data/numina_math_64k_samples/GptOss20B/layer11/val")
MAX_TOKENS = 100_000  # plenty for spectral statistics on 2880-dim


def load_h_sample(cache_dir: Path, max_tokens: int) -> np.ndarray:
    """Load a subsample of cached hidden states."""
    with open(cache_dir / "metadata.json") as f:
        meta = json.load(f)
    h_mmap = np.memmap(
        cache_dir / "hidden_states.bin",
        dtype=meta["dtype"], mode="r",
        shape=(meta["total_tokens"], meta["hidden_dim"]),
    )
    n = min(max_tokens, meta["total_tokens"])
    rng = np.random.default_rng(42)
    idx = rng.choice(meta["total_tokens"], n, replace=False)
    idx.sort()
    h = np.array(h_mmap[idx], dtype=np.float32)
    print(f"Loaded {n} tokens, hidden_dim={meta['hidden_dim']}, dtype={meta['dtype']}")
    return h


def eigenspectrum(X: np.ndarray) -> np.ndarray:
    """Return eigenvalues of covariance matrix, sorted descending."""
    X_c = X - X.mean(axis=0)
    cov = (X_c.T @ X_c) / (X_c.shape[0] - 1)
    eigvals = np.linalg.eigvalsh(cov)
    return np.sort(eigvals)[::-1]


def participation_ratio_from_eigvals(eigvals: np.ndarray) -> float:
    pos = eigvals[eigvals > 0]
    p = pos / pos.sum()
    return float(1.0 / (p ** 2).sum())


def svd_effective_rank_from_eigvals(eigvals: np.ndarray) -> float:
    sigma = np.sqrt(np.maximum(eigvals, 0))
    sigma = sigma[sigma > 1e-12]
    p = sigma / sigma.sum()
    H = -np.sum(p * np.log(p))
    return float(np.exp(H))


def plot_pca_spectrum(var_ratio: np.ndarray, pr: float, er: float, d: int,
                      label: str, out_path: Path, max_show: int = 50):
    """Bar chart of per-component variance ratio + cumulative line."""
    n_show = min(len(var_ratio), max_show)
    var_show = var_ratio[:n_show]
    cum = np.cumsum(var_ratio)[:n_show]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(1, n_show + 1), var_show, alpha=0.7, color="steelblue",
           label="Per-component")
    ax.plot(range(1, n_show + 1), cum, "r-o", markersize=2, label="Cumulative")
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Variance Explained")
    ax.set_title(f"PCA Spectrum — {label}\n"
                 f"PR = {pr:.1f}/{d}, EffRank = {er:.1f}/{d}"
                 + (f"  (showing top {n_show}/{len(var_ratio)})" if n_show < len(var_ratio) else ""))
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_svd_spectrum(eigvals: np.ndarray, er: float, d: int,
                      label: str, out_path: Path, max_show: int = 50):
    """Bar chart of normalized singular values."""
    sigma = np.sqrt(np.maximum(eigvals, 0))
    sigma_norm = sigma / sigma.sum() if sigma.sum() > 0 else sigma
    n_show = min(len(sigma_norm), max_show)
    sv_show = sigma_norm[:n_show]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(1, n_show + 1), sv_show, alpha=0.7, color="darkorange")
    ax.set_xlabel("Singular Value Index")
    ax.set_ylabel("Normalized Singular Value")
    ax.set_title(f"SVD Spectrum — {label}\n"
                 f"EffRank = {er:.1f}/{d}"
                 + (f"  (showing top {n_show}/{len(sigma_norm)})" if n_show < len(sigma_norm) else ""))
    ax.grid(True, alpha=0.3)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def compute_and_plot(X: np.ndarray, label: str, tag: str, out_dir: Path,
                     pca_var_ratio: np.ndarray | None = None,
                     max_show: int = 50) -> dict:
    """Compute stats and generate spectrum plots."""
    d = X.shape[1]
    eigvals = eigenspectrum(X)
    pr = participation_ratio_from_eigvals(eigvals)
    er = svd_effective_rank_from_eigvals(eigvals)
    mean_norm = float(np.linalg.norm(X, axis=1).mean())
    norm_of_mean = float(np.linalg.norm(X.mean(axis=0)))

    # Variance ratios for PCA plot
    if pca_var_ratio is None:
        pos = eigvals[eigvals > 0]
        pca_var_ratio = pos / pos.sum()

    print(f"\n{'='*60}")
    print(f"  {label}  (dim={d}, N={X.shape[0]})")
    print(f"{'='*60}")
    print(f"  Participation ratio:   {pr:8.1f} / {d}")
    print(f"  SVD effective rank:    {er:8.1f} / {d}")
    print(f"  Mean ||x||:            {mean_norm:.4f}")
    print(f"  ||mean(x)||:           {norm_of_mean:.4f}")
    print(f"  Top-5 variance ratios: {pca_var_ratio[:5]}")

    # Plots
    plot_pca_spectrum(pca_var_ratio, pr, er, d, label,
                      out_dir / f"pca_spectrum_{tag}.png", max_show)
    plot_svd_spectrum(eigvals, er, d, label,
                      out_dir / f"svd_spectrum_{tag}.png", max_show)

    return {
        "label": label,
        "dim": d,
        "n_tokens": int(X.shape[0]),
        "participation_ratio": pr,
        "svd_effective_rank": er,
        "mean_norm": mean_norm,
        "norm_of_mean": norm_of_mean,
        "top20_variance_ratios": pca_var_ratio[:20].tolist(),
    }


def main():
    out_dir = Path("outputs/GptOss20B/baselines")
    out_dir.mkdir(parents=True, exist_ok=True)

    h = load_h_sample(CACHE_DIR, MAX_TOKENS)
    results = {}

    # 1. Raw hidden states
    results["h_raw"] = compute_and_plot(
        h, "Raw hidden states h (2880-dim)", "h_raw", out_dir, max_show=50)

    # 2. PCA-128
    from sklearn.decomposition import PCA
    pca128 = PCA(n_components=128)
    h_pca128 = pca128.fit_transform(h)
    results["h_pca_128"] = compute_and_plot(
        h_pca128, f"PCA-128 (expl. var = {pca128.explained_variance_ratio_.sum():.4f})",
        "pca128", out_dir, pca_var_ratio=pca128.explained_variance_ratio_)

    # 3. PCA-64
    pca64 = PCA(n_components=64)
    h_pca64 = pca64.fit_transform(h)
    results["h_pca_64"] = compute_and_plot(
        h_pca64, f"PCA-64 (expl. var = {pca64.explained_variance_ratio_.sum():.4f})",
        "pca64", out_dir, pca_var_ratio=pca64.explained_variance_ratio_)

    # Summary table
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Representation':<35} {'Dim':>5} {'PR':>8} {'EffRank':>8}")
    print(f"  {'-'*56}")
    for v in results.values():
        print(f"  {v['label'][:35]:<35} {v['dim']:>5} {v['participation_ratio']:>8.1f} {v['svd_effective_rank']:>8.1f}")

    with open(out_dir / "spectral_baseline_stats.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_dir / 'spectral_baseline_stats.json'}")


if __name__ == "__main__":
    main()
