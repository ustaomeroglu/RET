#!/usr/bin/env python3
"""Plot self-prediction R² summary for Numina + TinyStories across methods
on two LLM backbones: GPT-OSS-20B (layer 11) and Pythia-160M.

Produces a 2×2 grouped-bar panel figure.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_r2(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def best_r2(sweep: dict, key: str) -> float | None:
    sub = sweep.get(key)
    if not sub:
        return None
    return sub.get("_best_r2")


DATASETS = ["numina_math", "tiny_stories"]
# (label, model_dir, include_sae, l11, include_pile_ret)
# include_pile_ret=True adds an extra "RET (Pile)" bar reading closure from
# outputs/Pythia160M/pile/ret_autonomous/closure_on_{numina,tinystories}.
MODELS = [
    ("GPT-OSS-20B (L11)", "GptOss20B", True, True, False),
    ("Pythia-160M", "Pythia160M", True, False, True),
]
METHODS_FULL = ["h", "h_pca128", "SAE", "RET", "RET_pile"]
COLOR = {
    "h": "#9aa0a6",
    "h_pca128": "#4285f4",
    "SAE": "#fbbc04",
    "RET": "#ea4335",
    "RET_pile": "#8b1a1a",
}
KEY_AUTO = {
    "h": "h/hh",
    "h_pca128": "h_pca/pca",
    "SAE": "sae/autonomous",
    "RET": "z_ret/autonomous",
}
KEY_CTRL = {
    "RET": "z_ret/controlled",
    "SAE": "sae/controlled",
}


def _sweep_r2_from_predictors(variant_dir: Path, method_key: str) -> dict[int, float]:
    """Return {hdim: val_r2_final} across every hdim* subdir."""
    import torch

    out: dict[int, float] = {}
    if not variant_dir.exists():
        return out
    for hdim_dir in sorted(variant_dir.glob("hdim*")):
        try:
            h = int(hdim_dir.name.replace("hdim", ""))
        except ValueError:
            continue
        pt = hdim_dir / "predictors.pt"
        if not pt.exists():
            continue
        try:
            d = torch.load(str(pt), map_location="cpu", weights_only=False)
        except Exception:
            continue
        entry = d.get(method_key)
        if entry is None:
            continue
        r2 = entry.get("val_r2_final")
        if r2 is None:
            continue
        out[h] = r2
    return out


def _best_r2_from_predictors(variant_dir: Path, method_key: str) -> float | None:
    sweep = _sweep_r2_from_predictors(variant_dir, method_key)
    return max(sweep.values()) if sweep else None


def _median_r2_from_predictors(variant_dir: Path, method_key: str) -> float | None:
    sweep = _sweep_r2_from_predictors(variant_dir, method_key)
    return float(np.median(list(sweep.values()))) if sweep else None


def _reduce_r2(variant_dir: Path, method_key: str, reducer: str) -> float | None:
    if reducer == "median":
        return _median_r2_from_predictors(variant_dir, method_key)
    return _best_r2_from_predictors(variant_dir, method_key)


_PILE_DS_TAG = {"numina_math": "numina", "tiny_stories": "tinystories"}


def collect(
    model_dir: str,
    dataset: str,
    include_sae: bool,
    l11: bool,
    include_pile_ret: bool = False,
    reducer: str = "best",
) -> dict:
    base = Path("outputs") / model_dir / dataset
    if l11:
        h_dir = base / "baselines" / "h_l11" / "closure"
        hpca_dir = base / "baselines" / "h_pca128_l11" / "closure"
        sae_dir = base / "baselines" / "sae_l11" / "closure"
        ret_dir = base / "ret_jepa_auth_l11" / "closure"
    else:
        h_dir = base / "baselines" / "h" / "closure"
        hpca_dir = base / "baselines" / "h_pca128" / "closure"
        sae_dir = base / "baselines" / "sae" / "closure"
        # Pythia RET path varies; fall back to the closure_analysis sweep if present.
        ret_dir = base / "ret_jepa_auth" / "closure"

    row = {
        "h": _reduce_r2(h_dir, "hh", reducer),
        "h_pca128": _reduce_r2(hpca_dir, "pca", reducer),
        "RET_auto": _reduce_r2(ret_dir, "autonomous", reducer),
        "RET_ctrl": _reduce_r2(ret_dir, "controlled", reducer),
    }
    if include_sae:
        row["SAE_auto"] = _reduce_r2(sae_dir, "autonomous", reducer)
        row["SAE_ctrl"] = _reduce_r2(sae_dir, "controlled", reducer)
    if include_pile_ret:
        pile_tag = _PILE_DS_TAG.get(dataset)
        pile_ret_dir = (
            Path("outputs") / model_dir / "pile" / "ret_dense_autonomous"
            / f"closure_on_{pile_tag}"
        )
        row["RET_pile_auto"] = _reduce_r2(pile_ret_dir, "autonomous", reducer)
        row["RET_pile_ctrl"] = _reduce_r2(pile_ret_dir, "controlled", reducer)
        if row["RET_pile_auto"] is None or row["RET_pile_ctrl"] is None:
            pile_sweep = load_r2(
                Path("outputs") / model_dir / "pile"
                / f"closure_analysis_{pile_tag}" / "capacity_sweep_summary.json"
            )
            if row["RET_pile_auto"] is None:
                row["RET_pile_auto"] = best_r2(pile_sweep, "z_ret/autonomous")
            if row["RET_pile_ctrl"] is None:
                row["RET_pile_ctrl"] = best_r2(pile_sweep, "z_ret/controlled")
    # Fallback to sweep_summary.json for cases where predictors.pt isn't available.
    ret_sweep = load_r2(base / "closure_analysis" / "capacity_sweep_summary.json")
    if row["RET_auto"] is None:
        row["RET_auto"] = best_r2(ret_sweep, "z_ret/autonomous")
    if row["RET_ctrl"] is None:
        row["RET_ctrl"] = best_r2(ret_sweep, "z_ret/controlled")
    baseline_sweep = load_r2(base / "baselines" / "closure_analysis" / "capacity_sweep_summary.json")
    if row["h"] is None:
        row["h"] = best_r2(baseline_sweep, "h/hh")
    if row["h_pca128"] is None:
        row["h_pca128"] = best_r2(baseline_sweep, "h_pca/pca")
    return row


_METHOD_XLABEL = {
    "h": "h",
    "h_pca128": "h_pca128",
    "SAE": "SAE",
    "RET": "RET",
    "RET_pile": "RET\n(Pile)",
}

_METHOD_XLABEL_PRETTY = {
    "h": r"$h$",
    "h_pca128": r"$h^{\mathrm{PCA}}$",
    "SAE": "SAE",
    "RET": "RET",
    "RET_pile": "RET\n(Pile)",
}


def plot_panel(
    ax,
    row: dict,
    title: str,
    include_sae: bool,
    include_pile_ret: bool = False,
    pile_ret_as_main: bool = False,
    pretty: bool = False,
    show_ylabel: bool = True,
) -> None:
    """Autonomous-only bar chart per panel.

    If ``pile_ret_as_main`` is True (only meaningful when ``include_pile_ret``),
    the single RET bar uses the Pile-trained values and is labeled just "RET"
    (no in-domain RET bar is shown).
    """
    methods = ["h", "h_pca128"]
    if include_sae:
        methods.append("SAE")
    if include_pile_ret and pile_ret_as_main:
        methods.append("RET_from_pile")
    else:
        methods.append("RET")
        if include_pile_ret:
            methods.append("RET_pile")

    vals = []
    for m in methods:
        if m == "RET":
            vals.append(row.get("RET_auto"))
        elif m == "RET_pile":
            vals.append(row.get("RET_pile_auto"))
        elif m == "RET_from_pile":
            vals.append(row.get("RET_pile_auto"))
        elif m == "SAE":
            vals.append(row.get("SAE_auto"))
        else:
            vals.append(row.get(m))

    x = np.arange(len(methods))
    color_key = {"RET_from_pile": "RET"}  # use RET color when relabeled
    bars = ax.bar(
        x, [v if v is not None else 0 for v in vals], width=0.6,
        color=[COLOR[color_key.get(m, m)] for m in methods],
        edgecolor="black", linewidth=0.5,
    )
    for bar, v in zip(bars, vals):
        if v is None:
            continue
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01,
                f"{v:.2f}", ha="center", va="bottom", fontsize=13)

    base_labels = _METHOD_XLABEL_PRETTY if pretty else _METHOD_XLABEL
    label_map = dict(base_labels, RET_from_pile="RET")
    ax.set_xticks(x)
    ax.set_xticklabels([label_map[m] for m in methods], fontsize=15)
    ax.set_ylim(0, 1.05)
    ax.set_title(title, fontsize=16)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.tick_params(axis="y", labelsize=14)
    if show_ylabel:
        ax.set_ylabel("self-prediction R²", fontsize=16)
    else:
        ax.set_ylabel("")
        ax.tick_params(axis="y", left=False)


def _render_figure(
    out_stem: Path, pile_ret_as_main: bool, pretty: bool = False,
    reducer: str = "best",
) -> None:
    fig, axes = plt.subplots(
        len(DATASETS), len(MODELS), figsize=(11, 7), sharey=True,
    )
    for i, ds in enumerate(DATASETS):
        for j, (label, model_dir, include_sae, l11, include_pile_ret) in enumerate(MODELS):
            row = collect(model_dir, ds, include_sae, l11, include_pile_ret,
                          reducer=reducer)
            title = f"{ds}   |   {label}"
            plot_panel(
                axes[i, j], row, title, include_sae, include_pile_ret,
                pile_ret_as_main=pile_ret_as_main, pretty=pretty,
                show_ylabel=(j == 0),
            )
    if not pretty:
        reducer_tag = "median" if reducer == "median" else "best"
        fig.suptitle(
            f"Self-prediction R²  (autonomous, {reducer_tag} hidden_dim)",
            y=1.01, fontsize=12,
        )
    fig.tight_layout()
    png = out_stem.with_suffix(".png")
    pdf = out_stem.with_suffix(".pdf")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {png}")
    print(f"Saved: {pdf}")


def collect_sweeps(
    model_dir: str,
    dataset: str,
    include_sae: bool,
    l11: bool,
    include_pile_ret: bool = False,
    pile_ret_as_main: bool = False,
) -> dict[str, dict[int, float]]:
    """Return {method_label: {hdim: R²}} for the autonomous predictors."""
    base = Path("outputs") / model_dir / dataset
    if l11:
        h_dir = base / "baselines" / "h_l11" / "closure"
        hpca_dir = base / "baselines" / "h_pca128_l11" / "closure"
        sae_dir = base / "baselines" / "sae_l11" / "closure"
        ret_dir = base / "ret_jepa_auth_l11" / "closure"
    else:
        h_dir = base / "baselines" / "h" / "closure"
        hpca_dir = base / "baselines" / "h_pca128" / "closure"
        sae_dir = base / "baselines" / "sae" / "closure"
        ret_dir = base / "ret_jepa_auth" / "closure"

    out: dict[str, dict[int, float]] = {
        "h": _sweep_r2_from_predictors(h_dir, "hh"),
        "h_pca128": _sweep_r2_from_predictors(hpca_dir, "pca"),
    }
    if include_sae:
        out["SAE"] = _sweep_r2_from_predictors(sae_dir, "autonomous")

    if include_pile_ret and pile_ret_as_main:
        pile_tag = _PILE_DS_TAG.get(dataset)
        pile_ret_dir = (
            Path("outputs") / model_dir / "pile" / "ret_dense_autonomous"
            / f"closure_on_{pile_tag}"
        )
        out["RET"] = _sweep_r2_from_predictors(pile_ret_dir, "autonomous")
    else:
        out["RET"] = _sweep_r2_from_predictors(ret_dir, "autonomous")
        if include_pile_ret:
            pile_tag = _PILE_DS_TAG.get(dataset)
            pile_ret_dir = (
                Path("outputs") / model_dir / "pile" / "ret_dense_autonomous"
                / f"closure_on_{pile_tag}"
            )
            out["RET (Pile)"] = _sweep_r2_from_predictors(pile_ret_dir, "autonomous")
    return out


def _sweep_plot_color(method: str) -> str:
    return {
        "h": COLOR["h"],
        "h_pca128": COLOR["h_pca128"],
        "SAE": COLOR["SAE"],
        "RET": COLOR["RET"],
        "RET (Pile)": COLOR["RET_pile"],
    }[method]


_SWEEP_LEGEND_PRETTY = {
    "h": r"$h$",
    "h_pca128": r"$h^{\mathrm{PCA}}$",
    "SAE": "SAE",
    "RET": "RET",
    "RET (Pile)": "RET (Pile)",
}


def plot_sweep_panel(
    ax, sweeps: dict[str, dict[int, float]], title: str, pretty: bool = False,
) -> None:
    for method, series in sweeps.items():
        if not series:
            continue
        hs = sorted(series)
        rs = [series[h] for h in hs]
        legend_label = _SWEEP_LEGEND_PRETTY.get(method, method) if pretty else method
        ax.plot(hs, rs, marker="o", linewidth=2, markersize=6,
                color=_sweep_plot_color(method), label=legend_label)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("predictor hidden_dim")
    ax.set_ylabel("self-prediction R²")
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)


def _render_sweep_figure(
    out_stem: Path, pile_ret_as_main: bool, pretty: bool = False,
) -> None:
    fig, axes = plt.subplots(
        len(DATASETS), len(MODELS), figsize=(12, 8), sharey=True,
    )
    for i, ds in enumerate(DATASETS):
        for j, (label, model_dir, include_sae, l11, include_pile_ret) in enumerate(MODELS):
            sweeps = collect_sweeps(
                model_dir, ds, include_sae, l11, include_pile_ret,
                pile_ret_as_main=pile_ret_as_main,
            )
            plot_sweep_panel(axes[i, j], sweeps, f"{ds}   |   {label}", pretty=pretty)
    if not pretty:
        fig.suptitle("Self-prediction R² vs predictor capacity  (autonomous)",
                     y=1.01, fontsize=12)
    fig.tight_layout()
    png = out_stem.with_suffix(".png")
    pdf = out_stem.with_suffix(".pdf")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {png}")
    print(f"Saved: {pdf}")


def _write_summary_table(summary_simple: dict, out_stem: Path) -> None:
    """Write the 16-number table as CSV + markdown.

    Rows: (dataset, model)  -  Cols: h, h_pca128, SAE, RET
    """
    methods = ["h", "h_pca128", "SAE", "RET"]
    csv_lines = ["dataset,model," + ",".join(methods)]
    md_lines = [
        "| dataset | model | " + " | ".join(methods) + " |",
        "|---|---|" + "|".join(["---:"] * len(methods)) + "|",
    ]
    for ds in DATASETS:
        for label, _, _, _, _ in MODELS:
            row = summary_simple[ds][label]
            vals = [row.get("h"), row.get("h_pca128"),
                    row.get("SAE_auto"), row.get("RET_auto")]
            csv_vals = ",".join("" if v is None else f"{v:.4f}" for v in vals)
            csv_lines.append(f"{ds},{label},{csv_vals}")
            md_vals = " | ".join("—" if v is None else f"{v:.3f}" for v in vals)
            md_lines.append(f"| {ds} | {label} | {md_vals} |")
    out_csv = out_stem.with_suffix(".csv")
    out_md = out_stem.with_suffix(".md")
    out_csv.write_text("\n".join(csv_lines) + "\n")
    out_md.write_text("\n".join(md_lines) + "\n")
    print(f"Saved: {out_csv}")
    print(f"Saved: {out_md}")


def _render_figure_compact(
    out_stem: Path, pile_ret_as_main: bool, pretty: bool = False,
    reducer: str = "best",
) -> None:
    """1×4 compact layout: all four panels in a single row.

    Order: NuminaMath/GPT-OSS, NuminaMath/Pythia, TinyStories/GPT-OSS, TinyStories/Pythia.
    Designed to occupy roughly half the vertical space of the 2×2 figure.
    """
    DS_SHORT = {"numina_math": "NuminaMath", "tiny_stories": "TinyStories"}
    MODEL_SHORT = {"GptOss20B": "GPT-OSS-20B", "Pythia160M": "Pythia-160M"}

    panels: list[tuple] = []
    for ds in DATASETS:
        for (label, model_dir, include_sae, l11, include_pile_ret) in MODELS:
            row = collect(model_dir, ds, include_sae, l11, include_pile_ret, reducer=reducer)
            title = f"{MODEL_SHORT[model_dir]}\n{DS_SHORT[ds]}"
            panels.append((row, title, include_sae, include_pile_ret))

    fig, axes = plt.subplots(1, 4, figsize=(12, 3.6), sharey=True)

    for idx, (ax, (row, title, include_sae, include_pile_ret)) in enumerate(zip(axes, panels)):
        # Reuse the same method/value logic as plot_panel
        methods = ["h", "h_pca128"]
        if include_sae:
            methods.append("SAE")
        if include_pile_ret and pile_ret_as_main:
            methods.append("RET_from_pile")
        else:
            methods.append("RET")
            if include_pile_ret:
                methods.append("RET_pile")

        vals = []
        for m in methods:
            if m in ("RET", "RET_from_pile"):
                vals.append(row.get("RET_pile_auto" if m == "RET_from_pile" else "RET_auto"))
            elif m == "RET_pile":
                vals.append(row.get("RET_pile_auto"))
            elif m == "SAE":
                vals.append(row.get("SAE_auto"))
            else:
                vals.append(row.get(m))

        x = np.arange(len(methods))
        color_key = {"RET_from_pile": "RET"}
        bars = ax.bar(
            x, [v if v is not None else 0 for v in vals], width=0.55,
            color=[COLOR[color_key.get(m, m)] for m in methods],
            edgecolor="black", linewidth=0.5,
        )
        for bar, v in zip(bars, vals):
            if v is None:
                continue
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.012,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=10)

        base_labels = _METHOD_XLABEL_PRETTY if pretty else _METHOD_XLABEL
        label_map = dict(base_labels, RET_from_pile="RET")
        ax.set_xticks(x)
        ax.set_xticklabels([label_map[m] for m in methods], fontsize=13)
        ax.set_ylim(0, 1.05)
        ax.set_title(title, fontsize=14)
        ax.grid(axis="y", linestyle=":", alpha=0.4)
        ax.tick_params(axis="y", labelsize=13)
        if idx == 0:
            ax.set_ylabel("self-prediction R²", fontsize=14)
        else:
            ax.set_ylabel("")
            ax.tick_params(axis="y", left=False)

    fig.tight_layout(pad=0.4, w_pad=0.25)
    png = out_stem.with_suffix(".png")
    pdf = out_stem.with_suffix(".pdf")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {png}")
    print(f"Saved: {pdf}")


def main() -> None:
    out_dir = Path("Paper_figures_2/closure")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Full version: both in-domain RET and Pile-trained RET bars.
    _render_figure(out_dir / "self_prediction_r2_full_summary", pile_ret_as_main=False)
    # Summary version: single RET bar (Pile-trained values, labeled "RET").
    _render_figure(out_dir / "self_prediction_r2_summary", pile_ret_as_main=True, pretty=True)
    # Compact 1×4 version.
    _render_figure_compact(out_dir / "self_prediction_r2_summary_1row", pile_ret_as_main=True, pretty=True)

    # Median-aggregated variants (median R² across the predictor hidden_dim sweep).
    _render_figure(
        out_dir / "self_prediction_r2_full_summary_median",
        pile_ret_as_main=False, reducer="median",
    )
    _render_figure(
        out_dir / "self_prediction_r2_summary_median",
        pile_ret_as_main=True, pretty=True, reducer="median",
    )

    # Full JSON dump retains all fields (used by downstream code + audit trail).
    summary = {}
    for ds in DATASETS:
        summary[ds] = {}
        for label, model_dir, include_sae, l11, include_pile_ret in MODELS:
            summary[ds][label] = collect(
                model_dir, ds, include_sae, l11, include_pile_ret
            )
    full_json = out_dir / "self_prediction_r2_full_summary.json"
    with open(full_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {full_json}")

    # Summary JSON: single "RET" entry per Pythia row (Pile-trained values).
    summary_simple: dict = {}
    for ds, per_model in summary.items():
        summary_simple[ds] = {}
        for label, row in per_model.items():
            new_row = {k: v for k, v in row.items()
                       if k not in {"RET_auto", "RET_ctrl",
                                    "RET_pile_auto", "RET_pile_ctrl"}}
            if row.get("RET_pile_auto") is not None:
                new_row["RET_auto"] = row["RET_pile_auto"]
                new_row["RET_ctrl"] = row.get("RET_pile_ctrl")
            else:
                new_row["RET_auto"] = row.get("RET_auto")
                new_row["RET_ctrl"] = row.get("RET_ctrl")
            summary_simple[ds][label] = new_row
    simple_json = out_dir / "self_prediction_r2_summary.json"
    with open(simple_json, "w") as f:
        json.dump(summary_simple, f, indent=2)
    print(f"Saved: {simple_json}")

    # Median-aggregated JSON dump (mirrors summary_simple but with median R²).
    summary_median: dict = {}
    for ds in DATASETS:
        summary_median[ds] = {}
        for label, model_dir, include_sae, l11, include_pile_ret in MODELS:
            row = collect(
                model_dir, ds, include_sae, l11, include_pile_ret,
                reducer="median",
            )
            new_row = {k: v for k, v in row.items()
                       if k not in {"RET_auto", "RET_ctrl",
                                    "RET_pile_auto", "RET_pile_ctrl"}}
            if row.get("RET_pile_auto") is not None:
                new_row["RET_auto"] = row["RET_pile_auto"]
                new_row["RET_ctrl"] = row.get("RET_pile_ctrl")
            else:
                new_row["RET_auto"] = row.get("RET_auto")
                new_row["RET_ctrl"] = row.get("RET_ctrl")
            summary_median[ds][label] = new_row
    median_json = out_dir / "self_prediction_r2_summary_median.json"
    with open(median_json, "w") as f:
        json.dump(summary_median, f, indent=2)
    print(f"Saved: {median_json}")

    # Table of the 16 autonomous-R² numbers backing the summary figure.
    _write_summary_table(summary_simple, out_dir / "self_prediction_r2_summary_table")

    # Capacity sweep figures (same 4-panel layout, lines per method).
    _render_sweep_figure(out_dir / "self_prediction_r2_full_sweep",
                         pile_ret_as_main=False)
    _render_sweep_figure(out_dir / "self_prediction_r2_sweep",
                         pile_ret_as_main=True, pretty=True)


if __name__ == "__main__":
    main()
