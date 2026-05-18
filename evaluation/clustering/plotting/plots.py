"""Basic 2D / matplotlib plotting helpers for clustering analyses.

State-machine SVG rendering lives in ``state_machine.py``; cluster-
naming I/O helpers live in ``naming.py``; trajectory HTML rendering
in ``html.py``; small color/format utilities in ``colors.py``.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


from evaluation.clustering.plotting.colors import _cluster_colors, _group_colors, _margin_to_alpha, _text_color_for_bg

def plot_transition_heatmap(
    P: np.ndarray,
    out_path: Path,
    title: str = "",
) -> None:
    """Plot a (K, K) row-normalised transition matrix as a heatmap.

    Args:
        P: (K, K) row-normalised transition matrix where P[i, j] = P(j | i).
        out_path: destination path for the saved figure.
        title: optional figure title; defaults to "Transition Matrix".
    """
    K = P.shape[0]
    fig, ax = plt.subplots(figsize=(max(6, K * 0.5), max(5, K * 0.45)))

    im = ax.imshow(P, cmap="YlOrRd", vmin=0.0, vmax=1.0, aspect="auto")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("P(j|i)", fontsize=10)

    tick_labels = [str(k) for k in range(K)]
    ax.set_xticks(range(K))
    ax.set_yticks(range(K))
    ax.set_xticklabels(tick_labels, fontsize=max(4, min(9, 90 // K)))
    ax.set_yticklabels(tick_labels, fontsize=max(4, min(9, 90 // K)))
    ax.set_xlabel("Next cluster j", fontsize=10)
    ax.set_ylabel("Current cluster i", fontsize=10)

    # Annotate cells whose value exceeds the threshold
    threshold = 0.05
    for i in range(K):
        for j in range(K):
            v = P[i, j]
            if v > threshold:
                text_color = "black" if v < 0.6 else "white"
                ax.text(
                    j, i, f"{v:.2f}",
                    ha="center", va="center",
                    fontsize=max(4, min(8, 72 // K)),
                    color=text_color,
                )

    ax.set_title(title if title else "Transition Matrix", fontsize=12)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_dwell_histogram(
    dwell_stats: dict,
    out_path: Path,
    title: str = "",
) -> None:
    """Two-panel figure: dwell-length histogram (left) and per-cluster mean dwell (right).

    Args:
        dwell_stats: dict from compute_dwell_times with keys "all_dwells" (list/array
            of dwell lengths) and "mean_dwell_per_cluster" (dict or array of per-cluster
            means indexed by cluster id).
        out_path: destination path for the saved figure.
        title: optional suptitle.
    """
    all_dwells = np.asarray(dwell_stats["all_dwells"], dtype=float)
    mean_per_cluster_raw = dwell_stats["mean_dwell_per_cluster"]

    # Normalise mean_per_cluster to a sorted array of (cluster_id, mean) pairs
    if isinstance(mean_per_cluster_raw, dict):
        cluster_ids = sorted(mean_per_cluster_raw.keys())
        mean_dwells = np.array([mean_per_cluster_raw[k] for k in cluster_ids], dtype=float)
    else:
        mean_dwells = np.asarray(mean_per_cluster_raw, dtype=float)
        cluster_ids = list(range(len(mean_dwells)))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Left: histogram of all dwell lengths ---
    ax_hist = axes[0]
    if len(all_dwells) > 0:
        max_dwell = int(all_dwells.max())
        bins = min(max_dwell, 80)
        ax_hist.hist(all_dwells, bins=bins, color="steelblue", edgecolor="black", linewidth=0.4)
    ax_hist.set_yscale("log")
    ax_hist.set_xlabel("Dwell length (tokens)", fontsize=10)
    ax_hist.set_ylabel("Count (log scale)", fontsize=10)
    ax_hist.set_title("Dwell Length Distribution", fontsize=11)
    ax_hist.grid(True, axis="y", alpha=0.3)

    if len(all_dwells) > 0:
        overall_mean = float(np.mean(all_dwells))
        overall_median = float(np.median(all_dwells))
        ax_hist.axvline(overall_mean, color="crimson", linestyle="--", linewidth=1.5,
                        label=f"mean={overall_mean:.2f}")
        ax_hist.axvline(overall_median, color="darkorange", linestyle=":", linewidth=1.5,
                        label=f"median={overall_median:.2f}")
        ax_hist.legend(fontsize=9)

        annotation = f"mean={overall_mean:.2f}\nmedian={overall_median:.2f}"
        ax_hist.text(
            0.97, 0.97, annotation,
            transform=ax_hist.transAxes,
            ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8),
        )

    # --- Right: horizontal bar chart of mean dwell per cluster ---
    ax_bar = axes[1]
    y_pos = np.arange(len(cluster_ids))
    ax_bar.barh(y_pos, mean_dwells, color="mediumseagreen", edgecolor="black", linewidth=0.4)
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels([str(c) for c in cluster_ids],
                            fontsize=max(4, min(9, 180 // max(len(cluster_ids), 1))))
    ax_bar.set_xlabel("Mean dwell time (tokens)", fontsize=10)
    ax_bar.set_ylabel("Cluster index", fontsize=10)
    ax_bar.set_title("Mean Dwell Time per Cluster", fontsize=11)
    ax_bar.grid(True, axis="x", alpha=0.3)

    if len(all_dwells) > 0:
        annotation = f"overall mean={overall_mean:.2f}\noverall median={overall_median:.2f}"
        ax_bar.text(
            0.97, 0.03, annotation,
            transform=ax_bar.transAxes,
            ha="right", va="bottom", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8),
        )

    if title:
        fig.suptitle(title, fontsize=13)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_timeline_strips(
    trajectories: list[np.ndarray],
    K: int,
    out_path: Path,
    num_docs: int = 5,
    title: str = "",
) -> None:
    """Visualise cluster assignment trajectories as horizontal colour strips.

    Each row corresponds to one document; x-axis is token position; colour
    encodes the cluster ID.

    Args:
        trajectories: list of (T_i,) integer arrays of cluster assignments.
        K: total number of clusters.
        out_path: destination path for the saved figure.
        num_docs: number of documents to show (takes the first num_docs).
        title: optional figure title.
    """
    docs = trajectories[:num_docs]
    n_docs = len(docs)
    if n_docs == 0:
        fig, ax = plt.subplots(figsize=(10, 2))
        ax.text(0.5, 0.5, "No trajectories provided.", ha="center", va="center",
                transform=ax.transAxes)
        plt.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out_path}")
        return

    # Choose colormap
    if K <= 20:
        cmap = plt.cm.tab20
        norm = mcolors.BoundaryNorm(boundaries=np.arange(-0.5, K + 0.5), ncolors=cmap.N)
    else:
        colors_list = plt.cm.hsv(np.linspace(0, 1, K, endpoint=False))
        cmap = mcolors.ListedColormap(colors_list)
        norm = mcolors.BoundaryNorm(boundaries=np.arange(-0.5, K + 0.5), ncolors=K)

    strip_height = 1.0
    fig_height = max(3, n_docs * 0.8 + 1.5)
    fig, ax = plt.subplots(figsize=(16, fig_height))

    for row_idx, traj in enumerate(docs):
        T = len(traj)
        # Build a 2-D array (1, T) for imshow
        data = traj.reshape(1, T).astype(float)
        y_bottom = row_idx * strip_height
        ax.imshow(
            data,
            aspect="auto",
            cmap=cmap,
            norm=norm,
            extent=[0, T, y_bottom, y_bottom + strip_height],
            interpolation="nearest",
        )

    ax.set_xlim(0, max(len(t) for t in docs))
    ax.set_ylim(0, n_docs * strip_height)
    ax.set_yticks([i * strip_height + strip_height / 2 for i in range(n_docs)])
    ax.set_yticklabels([f"doc {i}" for i in range(n_docs)], fontsize=9)
    ax.set_xlabel("Token position", fontsize=10)
    ax.set_title(title if title else "Cluster Assignment Timelines", fontsize=12)

    # Colorbar showing cluster IDs
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation="vertical", pad=0.02)
    cbar.set_label("Cluster ID", fontsize=9)
    tick_step = max(1, K // 20)
    cbar_ticks = list(range(0, K, tick_step))
    cbar.set_ticks(cbar_ticks)
    cbar.set_ticklabels([str(t) for t in cbar_ticks], fontsize=max(5, min(9, 120 // K)))

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_cluster_usage(
    usage: np.ndarray,
    out_path: Path,
    title: str = "",
) -> None:
    """Bar chart of cluster usage fractions, sorted by descending frequency.

    Includes a horizontal dashed reference line at 1/K (uniform) and annotates
    the Gini coefficient.

    Args:
        usage: (K,) array of fractions summing to 1.
        out_path: destination path for the saved figure.
        title: optional figure title.
    """
    usage = np.asarray(usage, dtype=float)
    K = len(usage)

    # Sort descending
    sort_idx = np.argsort(usage)[::-1]
    sorted_usage = usage[sort_idx]
    sorted_labels = [str(i) for i in sort_idx]

    # Gini coefficient
    u_sorted_asc = np.sort(usage)
    n = K
    cumsum = np.cumsum(u_sorted_asc)
    if usage.sum() > 0:
        gini = (2 * np.sum((np.arange(1, n + 1)) * u_sorted_asc) / (n * usage.sum())) - (n + 1) / n
        gini = float(np.clip(gini, 0.0, 1.0))
    else:
        gini = float("nan")

    fig, ax = plt.subplots(figsize=(max(8, K * 0.35), 5))

    x = np.arange(K)
    ax.bar(x, sorted_usage, color="steelblue", edgecolor="black", linewidth=0.4)
    ax.axhline(y=1.0 / K, color="crimson", linestyle="--", linewidth=1.5,
               label=f"Uniform (1/K = {1.0/K:.4f})")

    ax.set_xticks(x)
    ax.set_xticklabels(sorted_labels, rotation=90, fontsize=max(4, min(9, 90 // max(K, 1))))
    ax.set_xlabel("Cluster index (sorted by frequency)", fontsize=10)
    ax.set_ylabel("Fraction of assignments", fontsize=10)
    ax.set_title(title if title else "Cluster Usage", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    gini_text = f"Gini = {gini:.3f}" if not np.isnan(gini) else "Gini = N/A"
    ax.text(
        0.98, 0.97, gini_text,
        transform=ax.transAxes,
        ha="right", va="top", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8),
    )

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_elbow(
    inertias_by_K: dict[int, list[float]],
    out_path: Path,
    title: str = "",
) -> None:
    """Elbow plot of k-means inertia vs K with error bars across seeds.

    Args:
        inertias_by_K: mapping {K: [inertia_seed0, inertia_seed1, ...]} where
            each list contains the final inertia for different random seeds.
        out_path: destination path for the saved figure.
        title: optional figure title.
    """
    Ks = sorted(inertias_by_K.keys())
    means = np.array([np.mean(inertias_by_K[k]) for k in Ks], dtype=float)
    stds = np.array([np.std(inertias_by_K[k]) for k in Ks], dtype=float)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(
        Ks, means, yerr=stds,
        fmt="o-", color="steelblue", ecolor="steelblue",
        elinewidth=1.5, capsize=4, linewidth=2, markersize=6,
        label="mean ± std across seeds",
    )
    ax.scatter(Ks, means, color="steelblue", zorder=5)

    ax.set_xlabel("K (number of clusters)", fontsize=10)
    ax.set_ylabel("Final inertia", fontsize=10)
    ax.set_title(title if title else "Elbow Plot: Inertia vs K", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_cosine_margin_histogram(
    margins: np.ndarray,
    out_path: Path,
    title: str = "",
) -> None:
    """Histogram of cosine margin values (top-1 minus top-2 similarity).

    Args:
        margins: (N,) array of margin values, one per token/sample.
        out_path: destination path for the saved figure.
        title: optional figure title.
    """
    margins = np.asarray(margins, dtype=float)

    fig, ax = plt.subplots(figsize=(8, 5))

    if len(margins) > 0:
        bins = min(80, max(10, len(margins) // 20))
        ax.hist(margins, bins=bins, color="mediumpurple", edgecolor="black", linewidth=0.4)

        overall_mean = float(np.mean(margins))
        overall_median = float(np.median(margins))

        ax.axvline(overall_median, color="crimson", linestyle="--", linewidth=2.0,
                   label=f"median={overall_median:.4f}")

        annotation = f"mean   = {overall_mean:.4f}\nmedian = {overall_median:.4f}"
        ax.text(
            0.97, 0.97, annotation,
            transform=ax.transAxes,
            ha="right", va="top", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8),
        )
        ax.legend(fontsize=9)

    ax.set_xlabel("Cosine margin (top-1 − top-2 similarity)", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title(title if title else "Cosine Margin Distribution", fontsize=12)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_center_similarity(
    centers: np.ndarray,
    out_path: Path,
    title: str = "",
) -> None:
    """Plot a K x K cosine similarity heatmap between cluster centers.

    Args:
        centers: (K, D) array of cluster center vectors.
        out_path: destination path for the saved figure.
        title: optional figure title.
    """
    # L2-normalize centers then compute cosine similarity matrix
    norms = np.linalg.norm(centers, axis=1, keepdims=True).clip(min=1e-8)
    normed = centers / norms
    sim = normed @ normed.T  # (K, K)

    K = sim.shape[0]
    fig, ax = plt.subplots(figsize=(max(6, K * 0.5), max(5, K * 0.45)))

    im = ax.imshow(sim, cmap="RdBu_r", vmin=-1.0, vmax=1.0, aspect="auto")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Cosine similarity", fontsize=10)

    tick_labels = [str(k) for k in range(K)]
    ax.set_xticks(range(K))
    ax.set_yticks(range(K))
    ax.set_xticklabels(tick_labels, fontsize=max(4, min(9, 90 // K)))
    ax.set_yticklabels(tick_labels, fontsize=max(4, min(9, 90 // K)))
    ax.set_xlabel("Cluster j", fontsize=10)
    ax.set_ylabel("Cluster i", fontsize=10)

    # Annotate cells
    for i in range(K):
        for j in range(K):
            v = sim[i, j]
            if i == j:
                continue  # skip diagonal (always 1.0)
            text_color = "black" if abs(v) < 0.6 else "white"
            ax.text(
                j, i, f"{v:.2f}",
                ha="center", va="center",
                fontsize=max(4, min(8, 72 // K)),
                color=text_color,
            )

    ax.set_title(title if title else "Cluster Center Similarity", fontsize=12)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_center_scatter_2d(
    centers: np.ndarray,
    out_path: Path,
    group_labels: np.ndarray | None = None,
    cluster_names: dict[int, str] | None = None,
    group_names: dict[int, str] | None = None,
    title: str = "",
) -> None:
    """2D scatter of cluster centers via MDS on cosine distance.

    Args:
        centers: (K, D) cluster center vectors.
        out_path: destination for the saved figure.
        group_labels: (K,) mapping cluster → group (colors by group when given).
        cluster_names: optional mapping cluster_id → name string.
        group_names: optional mapping group_id → name string.
        title: figure title.
    """
    from sklearn.manifold import MDS

    K, D = centers.shape
    out_path = Path(out_path)

    # Cosine distance matrix
    norms = np.linalg.norm(centers, axis=1, keepdims=True).clip(min=1e-8)
    normed = centers / norms
    cos_sim = normed @ normed.T
    cos_dist = np.clip(1.0 - cos_sim, 0.0, 2.0)

    # Embed to 2D
    stress_val = 0.0
    if K <= 1:
        pts = np.zeros((K, 2))
    elif K == 2:
        pts = np.array([[0.0, 0.0], [cos_dist[0, 1], 0.0]])
    else:
        mds = MDS(
            n_components=2,
            dissimilarity="precomputed",
            normalized_stress="auto",
            random_state=42,
            n_init=4,
        )
        pts = mds.fit_transform(cos_dist)
        stress_val = mds.stress_

    fig, ax = plt.subplots(figsize=(10, 8))

    # Colors
    colors = _cluster_colors(K, group_labels)

    # Scatter cluster centers
    for k in range(K):
        ax.scatter(
            pts[k, 0], pts[k, 1],
            c=colors[k], s=120, edgecolors="black", linewidths=0.6, zorder=3,
        )

    # Annotate clusters
    for k in range(K):
        label = str(k)
        if cluster_names and k in cluster_names:
            label = f"{k}: {cluster_names[k]}"
        elif cluster_names:
            sk = str(k)
            if sk in cluster_names:
                label = f"{k}: {cluster_names[sk]}"
        ax.annotate(
            label,
            (pts[k, 0], pts[k, 1]),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=max(5, min(8, 64 // K)),
            zorder=4,
        )

    # Group centroids + legend
    if group_labels is not None:
        K_coarse = int(group_labels.max()) + 1
        grp_colors = _group_colors(K_coarse)
        handles = []
        for g in range(K_coarse):
            members = [i for i in range(K) if group_labels[i] == g]
            if not members:
                continue
            centroid = pts[members].mean(axis=0)
            ax.scatter(
                centroid[0], centroid[1],
                c=grp_colors[g], s=350, marker="*",
                edgecolors="black", linewidths=0.8, zorder=5,
            )
            ax.annotate(
                f"G{g}", (centroid[0], centroid[1]),
                textcoords="offset points", xytext=(-8, -12),
                fontsize=8, fontweight="bold", zorder=6,
            )
            g_label = f"G{g}"
            if group_names and g in group_names:
                g_label = f"G{g}: {group_names[g]}"
            handles.append(
                plt.Line2D(
                    [0], [0], marker="*", color="w",
                    markerfacecolor=grp_colors[g], markeredgecolor="black",
                    markersize=14, label=g_label,
                )
            )
        if handles:
            ax.legend(handles=handles, loc="best", fontsize=7, framealpha=0.8)

    ax.set_xlabel("MDS dim 1", fontsize=10)
    ax.set_ylabel("MDS dim 2", fontsize=10)
    subtitle = f"MDS on cosine distance (K={K})"
    if K > 2:
        subtitle += f"  —  stress={stress_val:.4f}"
    if title:
        ax.set_title(f"{title}\n{subtitle}", fontsize=11)
    else:
        ax.set_title(subtitle, fontsize=11)

    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# State machine diagram plots
# ---------------------------------------------------------------------------


def plot_dendrogram(
    linkage_matrix: np.ndarray,
    K_coarse: int,
    out_path: Path,
    title: str = "",
) -> None:
    """Plot the hierarchical clustering dendrogram with a cut line.

    Args:
        linkage_matrix: (K-1, 4) scipy linkage matrix.
        K_coarse: number of groups (used to draw a horizontal cut line).
        out_path: output file path.
        title: figure title.
    """
    from scipy.cluster.hierarchy import dendrogram as scipy_dendrogram

    fig, ax = plt.subplots(figsize=(max(10, K_coarse * 1.5), 6))

    # Determine the cut height from the linkage matrix:
    # the (K_coarse - 1)-th merge from the end gives the threshold
    n_merges = linkage_matrix.shape[0]
    if K_coarse <= n_merges:
        cut_height = linkage_matrix[-(K_coarse - 1), 2] if K_coarse > 1 else linkage_matrix[-1, 2] * 1.1
    else:
        cut_height = 0.0

    scipy_dendrogram(
        linkage_matrix,
        ax=ax,
        color_threshold=cut_height,
        above_threshold_color="gray",
        leaf_font_size=9,
    )

    ax.axhline(y=cut_height, color="crimson", linestyle="--", linewidth=1.5,
               label=f"Cut → {K_coarse} groups")
    ax.legend(fontsize=9)

    ax.set_xlabel("Micro cluster index", fontsize=10)
    ax.set_ylabel("Distance", fontsize=10)
    ax.set_title(title if title else "Hierarchical Clustering Dendrogram", fontsize=12)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_umap_trajectories(
    coords_2d: np.ndarray,
    doc_offsets: np.ndarray,
    labels_per_doc: list[np.ndarray],
    K: int,
    centers_2d: np.ndarray,
    out_dir: Path,
    title_prefix: str = "",
    num_docs: int = 5,
) -> None:
    """Plot per-document UMAP trajectories colored by cluster assignment.

    Args:
        coords_2d: (N_total, 2) global UMAP coordinates for ALL tokens.
        doc_offsets: (num_docs+1,) cumulative token counts, so doc i spans
            ``coords_2d[doc_offsets[i]:doc_offsets[i+1]]``.
        labels_per_doc: cluster labels for the first *num_docs* documents.
        K: number of clusters.
        centers_2d: (K, 2) cluster centers projected to the same 2D space.
        out_dir: directory to save plots into.
        title_prefix: prefix for figure titles.
        num_docs: how many example documents to plot.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    colors = _cluster_colors(K)
    n_plot = min(num_docs, len(labels_per_doc))

    # --- Per-document plots ---
    for doc_idx in range(n_plot):
        start = int(doc_offsets[doc_idx])
        end = int(doc_offsets[doc_idx + 1])
        xy = coords_2d[start:end]
        labels = labels_per_doc[doc_idx]
        n = min(len(xy), len(labels))
        xy, labels = xy[:n], labels[:n]

        fig, ax = plt.subplots(figsize=(10, 8))

        # Trajectory line (thin, gray)
        ax.plot(xy[:, 0], xy[:, 1], color="gray", alpha=0.15, linewidth=0.5, zorder=1)

        # Scatter colored by cluster
        for k in range(K):
            mask = labels == k
            if mask.any():
                ax.scatter(
                    xy[mask, 0], xy[mask, 1],
                    c=colors[k], s=8, alpha=0.6, zorder=2, label=f"C{k}",
                )

        # Cluster centers
        ax.scatter(
            centers_2d[:, 0], centers_2d[:, 1],
            c=colors[:K], s=200, marker="*", edgecolors="black",
            linewidths=0.8, zorder=3,
        )
        for k in range(K):
            ax.annotate(
                f"C{k}", (centers_2d[k, 0], centers_2d[k, 1]),
                fontsize=7, fontweight="bold", ha="center", va="bottom",
                xytext=(0, 6), textcoords="offset points",
            )

        # Start / end markers
        ax.scatter(xy[0, 0], xy[0, 1], c="lime", s=80, marker="o",
                   edgecolors="black", linewidths=1.2, zorder=4, label="start")
        ax.scatter(xy[-1, 0], xy[-1, 1], c="red", s=80, marker="X",
                   edgecolors="black", linewidths=1.2, zorder=4, label="end")

        ax.set_title(f"{title_prefix} — doc {doc_idx} ({n} tokens)", fontsize=11)
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
        ax.grid(True, alpha=0.2)

        out_path = out_dir / f"umap_trajectory_doc_{doc_idx}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out_path}")

    # --- Combined overview plot ---
    fig, ax = plt.subplots(figsize=(12, 10))

    for doc_idx in range(n_plot):
        start = int(doc_offsets[doc_idx])
        end = int(doc_offsets[doc_idx + 1])
        xy = coords_2d[start:end]
        labels = labels_per_doc[doc_idx]
        n = min(len(xy), len(labels))
        xy, labels = xy[:n], labels[:n]

        ax.plot(xy[:, 0], xy[:, 1], alpha=0.15, linewidth=0.4, zorder=1)
        for k in range(K):
            mask = labels == k
            if mask.any():
                ax.scatter(
                    xy[mask, 0], xy[mask, 1],
                    c=colors[k], s=4, alpha=0.3, zorder=2,
                )

    ax.scatter(
        centers_2d[:, 0], centers_2d[:, 1],
        c=colors[:K], s=250, marker="*", edgecolors="black",
        linewidths=0.8, zorder=3,
    )
    for k in range(K):
        ax.annotate(
            f"C{k}", (centers_2d[k, 0], centers_2d[k, 1]),
            fontsize=7, fontweight="bold", ha="center", va="bottom",
            xytext=(0, 6), textcoords="offset points",
        )

    ax.set_title(f"{title_prefix} — {n_plot} docs combined", fontsize=11)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.grid(True, alpha=0.2)

    out_path = out_dir / "umap_trajectory_combined.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# Naming file helpers (used by interactive HTML and inject_cluster_names.py)
# ---------------------------------------------------------------------------


