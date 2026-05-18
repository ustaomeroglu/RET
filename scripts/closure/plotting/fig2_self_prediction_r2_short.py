#!/usr/bin/env python3
"""Render a shortened version of self_prediction_r2_summary_1row.

Reads Paper_figures_2/closure/self_prediction_r2_summary.json (the
already-saved Pile-as-main summary) and renders a 1x4 bar figure with
ylim=(0.35, 1.05) and proportionally reduced height.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


COLOR = {
    "h": "#9aa0a6",
    "h_pca128": "#4285f4",
    "SAE": "#fbbc04",
    "RET": "#ea4335",
}
LABEL = {
    "h": r"$h$",
    "h_pca128": r"$h^{\mathrm{PCA}}$",
    "SAE": "SAE",
    "RET": "RET",
}
DS_SHORT = {"numina_math": "NuminaMath", "tiny_stories": "TinyStories"}
MODEL_ORDER = ["GPT-OSS-20B (L11)", "Pythia-160M"]
MODEL_SHORT = {"GPT-OSS-20B (L11)": "GPT-OSS-20B", "Pythia-160M": "Pythia-160M"}
DATASETS = ["numina_math", "tiny_stories"]
METHODS = ["h", "h_pca128", "SAE", "RET"]


def render(json_path: Path, out_stem: Path, ymin: float = 0.35, height: float = 2.7):
    summary = json.loads(json_path.read_text())

    panels = []
    for ds in DATASETS:
        for label in MODEL_ORDER:
            row = summary[ds][label]
            vals = [row.get("h"), row.get("h_pca128"),
                    row.get("SAE_auto"), row.get("RET_auto")]
            panels.append((vals, f"{MODEL_SHORT[label]} · {DS_SHORT[ds]}"))

    fig, axes = plt.subplots(1, 4, figsize=(12, height), sharey=True)
    x = np.arange(len(METHODS))
    for idx, (ax, (vals, title)) in enumerate(zip(axes, panels)):
        bars = ax.bar(
            x, [v if v is not None else 0 for v in vals], width=0.55,
            color=[COLOR[m] for m in METHODS],
            edgecolor="black", linewidth=0.5,
        )
        for bar, v in zip(bars, vals):
            if v is None:
                continue
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.008,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels([LABEL[m] for m in METHODS], fontsize=13)
        ax.set_ylim(ymin, 1.05)
        ax.set_title(title, fontsize=14)
        ax.grid(axis="y", linestyle=":", alpha=0.4)
        ax.tick_params(axis="y", labelsize=13)
        if idx == 0:
            ax.set_ylabel("self-prediction R²", fontsize=14)
        else:
            ax.set_ylabel("")
            ax.tick_params(axis="y", left=False)

    fig.tight_layout(pad=0.4, w_pad=0.25)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    titles = [ax.title for ax in axes]
    fontsize = 22
    while fontsize > 7:
        too_wide = False
        for ax, t in zip(axes, titles):
            t.set_fontsize(fontsize)
            tb = t.get_window_extent(renderer=renderer)
            ax_bb = ax.get_window_extent(renderer=renderer)
            if tb.width > ax_bb.width - 4:
                too_wide = True
                break
        if not too_wide:
            break
        fontsize -= 0.5
    for t in titles:
        t.set_fontsize(fontsize)
    print(f"chose title fontsize = {fontsize}")
    png = out_stem.with_suffix(".png")
    pdf = out_stem.with_suffix(".pdf")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {png}")
    print(f"Saved: {pdf}")


def main():
    repo = Path(__file__).resolve().parents[3]
    json_path = repo / "Paper_figures_2/closure/self_prediction_r2_summary.json"
    out_dir = repo / "latex/Figures/closure"
    out_dir.mkdir(parents=True, exist_ok=True)
    render(json_path, out_dir / "self_prediction_r2_summary_1row_short")


if __name__ == "__main__":
    main()
