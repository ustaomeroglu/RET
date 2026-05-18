"""Plotting functions for closure analysis — used by both training and comparative scripts."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _robust_ylim(all_vals: list[np.ndarray], pad: float = 0.15) -> tuple[float, float] | None:
    """Compute a robust y-axis range using the 2nd-98th percentile."""
    if not all_vals:
        return None
    combined = np.concatenate(all_vals)
    if len(combined) == 0:
        return None
    lo = np.percentile(combined, 2)
    hi = np.percentile(combined, 98)
    margin = (hi - lo) * pad
    if margin < 1e-8:
        margin = abs(hi) * 0.1 + 1e-6
    return (lo - margin, hi + margin)


def plot_multi_prediction(
    method_losses: dict[str, np.ndarray],
    tokens: list[str],
    out_path: Path,
    doc_idx: int,
    color_map: dict[str, str] | None = None,
) -> None:
    """Plot per-token prediction loss with multiple methods overlaid.

    Args:
        method_losses: {label: (T-1,) array of per-step losses}
        tokens: list of token strings (length T)
        out_path: save path for the figure
        doc_idx: story index for the title
        color_map: optional {label: color} mapping; uses tab10 by default
    """
    fig, ax = plt.subplots(figsize=(16, 5))
    colors = color_map or {}
    tab10 = plt.cm.tab10.colors

    for i, (label, vals) in enumerate(method_losses.items()):
        color = colors.get(label, tab10[i % len(tab10)])

        # Style logic: A=solid, B=dashed; auto=thick, ctrl=thin; hh=gray dotted
        if "(h)" in label or "hh" in label:
            ls, lw, color = ":", 1.5, "gray"
        elif "B " in label or "B(" in label:
            ls = "--"
            lw = 2.0 if "auto" in label else 1.2
        else:
            ls = "-"
            lw = 2.0 if "auto" in label else 1.2

        ax.plot(vals, label=label, color=color, linestyle=ls, linewidth=lw)

    ax.set_yscale("log")
    all_vals = np.concatenate(list(method_losses.values()))
    positive = all_vals[all_vals > 0]
    if len(positive) > 0:
        ax.set_ylim(bottom=positive.min() * 0.5)

    T = len(list(method_losses.values())[0])
    positions = list(range(T))
    tok_labels = [t.replace("\n", "\\n").strip()[:8] for t in tokens[1 : T + 1]]
    ax.set_xticks(positions)
    ax.set_xticklabels(tok_labels, rotation=90, fontsize=6)

    ax.set_xlabel("Token position (target)")
    ax.set_ylabel("Prediction loss")
    ax.set_title(f"Document {doc_idx}: Per-step Prediction Loss")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_multi_prediction_nse(
    method_nse: dict[str, np.ndarray],
    tokens: list[str],
    out_path: Path,
    doc_idx: int,
    color_map: dict[str, str] | None = None,
) -> None:
    """Plot per-token normalized prediction quality 1 - SE/Var with methods overlaid.

    Higher = better. 1 = perfect, 0 = error equals population variance.
    Comparable across methods since each is normalized by its own population variance.
    Note: per-step values are single-sample (SE not MSE), so not strictly R².

    Args:
        method_nse: {label: (T-1,) array of per-step 1 - SE/Var values}
        tokens: list of token strings (length T)
        out_path: save path for the figure
        doc_idx: story index for the title
        color_map: optional {label: color} mapping; uses tab10 by default
    """
    fig, ax = plt.subplots(figsize=(16, 5))
    colors = color_map or {}
    tab10 = plt.cm.tab10.colors

    for i, (label, vals) in enumerate(method_nse.items()):
        color = colors.get(label, tab10[i % len(tab10)])

        if "(h)" in label or "hh" in label:
            ls, lw, color = ":", 1.5, "gray"
        elif "B " in label or "B(" in label:
            ls = "--"
            lw = 2.0 if "auto" in label else 1.2
        else:
            ls = "-"
            lw = 2.0 if "auto" in label else 1.2

        ax.plot(vals, label=label, color=color, linestyle=ls, linewidth=lw)

    ax.axhline(y=0, color="black", linewidth=0.8, linestyle="-", alpha=0.5)

    ylim = _robust_ylim(list(method_nse.values()))
    if ylim is not None:
        # Ensure y=0 is always visible so outlier region is apparent
        ylim = (min(ylim[0], -0.1), ylim[1])
        ax.set_ylim(*ylim)

    # Annotate per-method mean so true average is visible despite y-axis clipping
    mean_parts = []
    for label, vals in method_nse.items():
        mean_parts.append(f"{label}: mean={vals.mean():.3f}")
    mean_text = "\n".join(mean_parts)
    ax.text(
        0.01, 0.02, mean_text, transform=ax.transAxes, fontsize=6,
        verticalalignment="bottom", fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8),
    )

    T = len(list(method_nse.values())[0])
    positions = list(range(T))
    tok_labels = [t.replace("\n", "\\n").strip()[:8] for t in tokens[1 : T + 1]]
    ax.set_xticks(positions)
    ax.set_xticklabels(tok_labels, rotation=90, fontsize=6)

    ax.set_xlabel("Token position (target)")
    ax.set_ylabel("1 - SE / Var(target)")
    ax.set_title(f"Document {doc_idx}: Normalized Prediction Quality (higher = better)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_multi_closure_gap(
    method_gaps: dict[str, np.ndarray],
    tokens: list[str],
    out_path: Path,
    doc_idx: int,
    color_map: dict[str, str] | None = None,
) -> None:
    """Plot closure gap (A - B) per method overlaid.

    Args:
        method_gaps: {label: (T-1,) array of A-B gaps}
        tokens: list of token strings (length T)
        out_path: save path
        doc_idx: story index for the title
        color_map: optional {label: color} mapping
    """
    fig, ax = plt.subplots(figsize=(16, 5))
    colors = color_map or {}
    tab10 = plt.cm.tab10.colors

    for i, (label, vals) in enumerate(method_gaps.items()):
        color = colors.get(label, tab10[i % len(tab10)])
        ls = "--" if "ctrl" in label else "-"
        ax.plot(vals, label=label, color=color, linestyle=ls, linewidth=1.5)

    ax.axhline(y=0, color="black", linewidth=0.8, linestyle="-", alpha=0.5)

    ylim = _robust_ylim(list(method_gaps.values()))
    if ylim is not None:
        ax.set_ylim(*ylim)

    T = len(list(method_gaps.values())[0])
    positions = list(range(T))
    tok_labels = [t.replace("\n", "\\n").strip()[:8] for t in tokens[1 : T + 1]]
    ax.set_xticks(positions)
    ax.set_xticklabels(tok_labels, rotation=90, fontsize=6)

    ax.set_xlabel("Token position (target)")
    ax.set_ylabel("Closure gap (A - B)")
    ax.set_title(f"Document {doc_idx}: Closure Gap (positive = info lost in compression)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_multi_closure_gap_normalized(
    method_gaps: dict[str, np.ndarray],
    tokens: list[str],
    out_path: Path,
    doc_idx: int,
    color_map: dict[str, str] | None = None,
) -> None:
    """Plot normalized closure gap (SE_A - SE_B) / Var(target) per method overlaid.

    Positive = model B (with h) has lower error than A (without h),
    i.e. information was lost in compression. Cross-method comparable.

    Args:
        method_gaps: {label: (T-1,) array of normalized gap values}
        tokens: list of token strings (length T)
        out_path: save path
        doc_idx: story index for the title
        color_map: optional {label: color} mapping
    """
    fig, ax = plt.subplots(figsize=(16, 5))
    colors = color_map or {}
    tab10 = plt.cm.tab10.colors

    for i, (label, vals) in enumerate(method_gaps.items()):
        color = colors.get(label, tab10[i % len(tab10)])
        ls = "--" if "ctrl" in label else "-"
        ax.plot(vals, label=label, color=color, linestyle=ls, linewidth=1.5)

    ax.axhline(y=0, color="black", linewidth=0.8, linestyle="-", alpha=0.5)

    ylim = _robust_ylim(list(method_gaps.values()))
    if ylim is not None:
        ax.set_ylim(*ylim)

    T = len(list(method_gaps.values())[0])
    positions = list(range(T))
    tok_labels = [t.replace("\n", "\\n").strip()[:8] for t in tokens[1 : T + 1]]
    ax.set_xticks(positions)
    ax.set_xticklabels(tok_labels, rotation=90, fontsize=6)

    ax.set_xlabel("Token position (target)")
    ax.set_ylabel("(SE_A - SE_B) / Var(target)")
    ax.set_title(f"Document {doc_idx}: Normalized Closure Gap (positive = info lost in compression)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_summary_bar(
    method_results: dict[str, dict[str, float]],
    out_path: Path,
    metric: str = "val_loss_final",
    title: str | None = None,
) -> None:
    """Bar chart of a metric across methods and variants.

    Args:
        method_results: {label: {"val_loss_final": float, "val_r2_final": float, ...}}
        out_path: save path
        metric: key to plot — "val_loss_final" or "val_r2_final"
        title: optional title override; if None, an auto-generated title is used
    """
    labels = []
    vals = []
    for label, info in method_results.items():
        vl = info.get(metric)
        if vl is None:
            continue
        labels.append(label)
        vals.append(vl)

    if not labels:
        print(f"  No results to plot for summary bar chart (metric={metric}).")
        return

    if metric == "val_r2_final":
        ylabel = "R²"
        auto_title = "Closure: Final Validation R² per Variant"
        color = "mediumseagreen"
    else:
        ylabel = "Val Loss"
        auto_title = "Closure: Final Validation Loss per Variant"
        color = "steelblue"

    if title is None:
        title = auto_title

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 1.2), 5))
    x = np.arange(len(labels))
    bars = ax.bar(x, vals, color=color, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)

    for bar, v in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{v:.4f}",
            ha="center", va="bottom", fontsize=8,
        )

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_relative_closure_gap(
    method_gaps: dict[str, np.ndarray],
    tokens: list[str],
    out_path: Path,
    doc_idx: int,
    color_map: dict[str, str] | None = None,
) -> None:
    """Plot relative closure gap (A - B) / A per method overlaid.

    Args:
        method_gaps: {label: (T-1,) array of relative gap values}
        tokens: list of token strings (length T)
        out_path: save path
        doc_idx: story index for the title
        color_map: optional {label: color} mapping
    """
    fig, ax = plt.subplots(figsize=(16, 5))
    colors = color_map or {}
    tab10 = plt.cm.tab10.colors

    for i, (label, vals) in enumerate(method_gaps.items()):
        color = colors.get(label, tab10[i % len(tab10)])
        ls = "--" if "ctrl" in label else "-"
        ax.plot(vals, label=label, color=color, linestyle=ls, linewidth=1.5)

    ax.axhline(y=0, color="black", linewidth=0.8, linestyle="-", alpha=0.5)

    ylim = _robust_ylim(list(method_gaps.values()))
    if ylim is not None:
        ax.set_ylim(*ylim)

    T = len(list(method_gaps.values())[0])
    positions = list(range(T))
    tok_labels = [t.replace("\n", "\\n").strip()[:8] for t in tokens[1 : T + 1]]
    ax.set_xticks(positions)
    ax.set_xticklabels(tok_labels, rotation=90, fontsize=6)

    ax.set_xlabel("Token position (target)")
    ax.set_ylabel("Relative closure gap (A - B) / A")
    ax.set_title(f"Document {doc_idx}: Relative Closure Gap (fraction of error from compression)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_training_curves(
    variant_curves: dict[str, dict[str, list[float]]],
    out_path: Path,
) -> None:
    """Subplot grid of train/val loss per variant.

    Args:
        variant_curves: {variant_name: {"train_losses": [...], "val_losses": [...]}}
        out_path: save path
    """
    curves = {
        name: info for name, info in variant_curves.items()
        if info.get("train_losses") and info.get("val_losses")
    }
    if not curves:
        print("  No loss curves to plot (no training history available).")
        return

    n = len(curves)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows), squeeze=False)

    for idx, (name, info) in enumerate(curves.items()):
        ax = axes[idx // cols][idx % cols]
        train_l = info["train_losses"]
        val_l = info["val_losses"]
        epochs = list(range(1, len(train_l) + 1))
        ax.plot(epochs, train_l, label="train", color="tab:blue")
        ax.plot(epochs, val_l, label="val", color="tab:orange")
        ax.set_xlabel("Eval cycle")
        ax.set_ylabel("Loss")
        ax.set_title(name, fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    for idx in range(n, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    fig.suptitle("Closure Predictor Training Curves", fontsize=12)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")
