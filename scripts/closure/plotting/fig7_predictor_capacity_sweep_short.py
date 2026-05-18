#!/usr/bin/env python3
"""Render a shortened version of the predictor-capacity sweep (paper Figure 7).

Mirrors the styling of fig2_self_prediction_r2_short.py: larger fonts,
"GPT-OSS-20B · NuminaMath" panel titles, and "RET" labels.
Reuses collect_sweeps() from fig2_self_prediction_r2.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fig2_self_prediction_r2 import DATASETS, MODELS, collect_sweeps  # noqa: E402


COLOR = {
    "h": "#9aa0a6",
    "h_pca128": "#4285f4",
    "SAE": "#fbbc04",
    "RET": "#ea4335",
    "RET (Pile)": "#8b1a1a",
}
LEGEND = {
    "h": r"$h$",
    "h_pca128": r"$h^{\mathrm{PCA}}$",
    "SAE": "SAE",
    "RET": "RET",
    "RET (Pile)": "RET (Pile)",
}
DS_SHORT = {"numina_math": "NuminaMath", "tiny_stories": "TinyStories"}
MODEL_SHORT = {"GptOss20B": "GPT-OSS-20B", "Pythia160M": "Pythia-160M"}


def render(out_stem: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.4), sharey=True)
    for i, ds in enumerate(DATASETS):
        for j, (label, model_dir, include_sae, l11, include_pile_ret) in enumerate(MODELS):
            ax = axes[i, j]
            sweeps = collect_sweeps(
                model_dir, ds, include_sae, l11, include_pile_ret,
                pile_ret_as_main=True,
            )
            for method, series in sweeps.items():
                if not series:
                    continue
                hs = sorted(series)
                rs = [series[h] for h in hs]
                ax.plot(
                    hs, rs, marker="o", linewidth=2.5, markersize=8,
                    color=COLOR[method], label=LEGEND.get(method, method),
                )
            ax.set_xscale("log", base=2)
            ax.set_ylim(0, 1.05)
            ax.set_title(f"{MODEL_SHORT[model_dir]} · {DS_SHORT[ds]}", fontsize=18)
            ax.grid(True, which="both", alpha=0.3)
            ax.tick_params(axis="both", labelsize=14)
            if i == 1:
                ax.set_xlabel("predictor hidden dim", fontsize=16)
            if j == 0:
                ax.set_ylabel("self-prediction R²", fontsize=16)
            ax.legend(loc="lower right", fontsize=13)

    fig.tight_layout()
    png = out_stem.with_suffix(".png")
    pdf = out_stem.with_suffix(".pdf")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {png}")
    print(f"Saved: {pdf}")


def main() -> None:
    repo = Path(__file__).resolve().parents[3]
    out_dir = repo / "latex/Figures/closure"
    out_dir.mkdir(parents=True, exist_ok=True)
    render(out_dir / "self_prediction_r2_sweep_short")


if __name__ == "__main__":
    main()
