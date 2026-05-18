"""Sweep utilities: discover, summarize, and print capacity sweep results."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from evaluation.closure.config import (
    load_config_with_overlay,
    closure_data_tag,
)
from ret.config import llm_short_name


def _read_r2_from_predictors(pt_path: Path) -> dict[str, float | None]:
    """Read val_r2_final per variant from a predictors.pt file."""
    data = torch.load(pt_path, map_location="cpu", weights_only=False)
    results = {}
    for key, val in data.items():
        if key == "__meta__":
            continue
        if isinstance(val, dict) and "val_r2_final" in val:
            results[key] = val["val_r2_final"]
    return results


def _discover_sweep_results(
    base_dir: Path,
    closure_name: str | None,
) -> dict[str, dict[str, float | None]]:
    """Scan for hdim subdirs and extract val_r2_final from predictors.pt.

    Handles two naming conventions:
      - New: base_dir/hdim{N}/predictors.pt
      - Old: base_dir/{closure_name}_hdim{N}/predictors.pt

    Returns {variant_key: {str(hdim): r2_value}}.
    """
    results: dict[str, dict[str, float | None]] = {}

    # New-style: hdim{N}/ directly under base_dir
    for d in sorted(base_dir.glob("hdim*")):
        if not d.is_dir():
            continue
        pt = d / "predictors.pt"
        if not pt.exists():
            continue
        hdim_str = d.name.replace("hdim", "")
        for key, r2 in _read_r2_from_predictors(pt).items():
            results.setdefault(key, {})[hdim_str] = r2

    # Old-style: {closure_name}_hdim{N}/ under base_dir
    if closure_name:
        for d in sorted(base_dir.glob(f"{closure_name}_hdim*")):
            if not d.is_dir():
                continue
            pt = d / "predictors.pt"
            if not pt.exists():
                continue
            hdim_str = d.name.rsplit("_hdim", 1)[-1]
            for key, r2 in _read_r2_from_predictors(pt).items():
                results.setdefault(key, {})[hdim_str] = r2

    return results


def _print_sweep_summary(
    summary: dict[str, dict[str, float | None]],
    saturation_threshold: float = 0.005,
) -> None:
    """Print formatted summary table with saturation detection and best hdim."""
    # Collect all hdim values across methods (excluding metadata keys)
    all_hdims: set[int] = set()
    for r2_dict in summary.values():
        for h in r2_dict:
            if not h.startswith("_"):
                try:
                    all_hdims.add(int(h))
                except ValueError:
                    pass
    all_hdims_sorted = sorted(all_hdims)

    dim_headers = "".join(f"{'hdim=' + str(h):>12s}" for h in all_hdims_sorted)
    header = f"{'Method':<16s}{dim_headers}  {'best':>8s}  {'sat@':>8s}"
    print(header)
    print("-" * len(header))

    for method, r2_dict in summary.items():
        row = f"{method:<16s}"
        valid: dict[str, float] = {}
        for h in all_hdims_sorted:
            hs = str(h)
            if hs in r2_dict and r2_dict[hs] is not None:
                v = r2_dict[hs]
                row += f"{v:12.4f}"
                valid[hs] = v
            else:
                row += f"{'---':>12s}"

        # Best hdim
        if valid:
            best_h = max(valid, key=lambda h: valid[h])
            row += f"  {best_h:>8s}"
        else:
            row += f"  {'?':>8s}"

        # Saturation detection
        sorted_items = sorted(
            ((int(h), v) for h, v in valid.items()),
            key=lambda x: x[0],
        )
        prev_val = None
        sat = None
        for hdim_val, r2_val in sorted_items:
            if prev_val is not None and r2_val - prev_val < saturation_threshold:
                sat = hdim_val
                break
            prev_val = r2_val
        row += f"  {str(sat) if sat else '?':>8s}"

        print(row)


def run_summarize_only(args):
    """Read existing sweep results and print summary + saturation detection."""
    ckpt_path = Path(args.checkpoint)
    ckpt_dir = ckpt_path.parent
    config = load_config_with_overlay(ckpt_dir / "config.yaml", args.config)

    data_tag = closure_data_tag(config)
    lsn = llm_short_name(config.llm)
    closure_name = args.closure_name

    print(f"Scanning for sweep results ...")
    print(f"  Checkpoint dir: {ckpt_dir}")
    print(f"  Data tag: {data_tag}")
    print(f"  Closure name: {closure_name}")

    # Discover RET results from checkpoint dir (new layout: closure/, fallback: flat)
    closure_root = ckpt_dir / "closure"
    ret_results = _discover_sweep_results(closure_root, closure_name)
    if not ret_results:
        ret_results = _discover_sweep_results(ckpt_dir, closure_name)
    if ret_results:
        print(f"  Found RET variants: {list(ret_results.keys())}")

    # Discover baseline results
    baselines_root = Path("outputs") / lsn / "baselines" / data_tag
    hh_results = _discover_sweep_results(baselines_root / "baseline_hh", None)
    if hh_results:
        print(f"  Found h->h variants: {list(hh_results.keys())}")

    pca_all_results: dict[str, dict[str, float | None]] = {}
    for pca_dir in sorted(baselines_root.glob("baseline_pca_*")):
        pd = pca_dir.name.split("_")[-1]
        pca_results = _discover_sweep_results(pca_dir, None)
        if pca_results:
            # Rename keys: "pca" -> "pca_{pd}"
            for key, r2_dict in pca_results.items():
                label = f"{key}_{pd}" if key == "pca" else key
                pca_all_results[label] = r2_dict
            print(f"  Found PCA-{pd} variants: {list(pca_results.keys())}")

    # Merge all results
    summary: dict[str, dict[str, float | None]] = {}
    summary.update(ret_results)
    summary.update(hh_results)
    summary.update(pca_all_results)

    if not summary:
        print("\nNo sweep results found.")
        return

    # Add _best_hdim and _saturated_at metadata
    for method, r2_dict in summary.items():
        valid = {h: v for h, v in r2_dict.items()
                 if v is not None and not h.startswith("_")}
        if valid:
            best_h = max(valid, key=lambda h: valid[h])
            summary[method]["_best_hdim"] = int(best_h)
            summary[method]["_best_r2"] = valid[best_h]

    print(f"\n=== Capacity Sweep Summary ===")
    _print_sweep_summary(summary)

    # Save
    analysis_dir = ckpt_dir / "closure"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    summary_path = analysis_dir / "self_prediction_capacity_sweep_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {summary_path}")
