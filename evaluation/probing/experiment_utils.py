"""
Shared utilities for experiment runners.

build_probe_list(cfg)              -> binary probe factories (original experiments)
build_probe_list_for_task(cfg, ..) -> probe factories for specified task_type/n_classes
needed_modes(cfg)                  -> set of mode strings to extract
probe_output_dir(root, probe, exp) -> per-probe output directory
get_pooled_modes(modes)            -> list of pooled mode strings
save_probe_results(...)            -> save results JSON with numpy serialization
run_eval_loop(...)                 -> dispatch cross/within dataset evaluation
apply_ret_encoding(...)            -> apply RET encoder if active
run_probes_and_save(...)           -> full probe loop for multi-dataset experiments
"""
from __future__ import annotations
import json
import logging
import os
from typing import Dict, List, Set, Tuple, Callable

import numpy as np

logger = logging.getLogger(__name__)


def probe_output_dir(output_root: str, probe_name: str, experiment_name: str = "") -> str:
    """Return per-probe output dir: {root}/{probe}_probe/"""
    return os.path.join(output_root, f"{probe_name}_probe")


def build_probe_list(cfg) -> List[Tuple[str, Callable]]:
    """Return (probe_name, factory) pairs for binary probes (original experiments)."""
    return build_probe_list_for_task(cfg, task_type="binary", n_classes=2)


def build_probe_list_for_task(cfg, task_type: str = "binary",
                               n_classes: int = 2) -> List[Tuple[str, Callable]]:
    """Return (probe_name, factory) pairs for a given task type.

    Parameters
    ----------
    task_type : "binary" | "multiclass"
    n_classes : number of output classes (2 for binary)
    """
    from evaluation.probing.baselines.linear_probe    import LinearProbe
    from evaluation.probing.baselines.dom_probe       import DOMProbe
    from evaluation.probing.baselines.attention_probe import AttentionProbe
    from evaluation.probing.baselines.transformer_probe import TransformerBlockProbe

    probes = []
    pt = cfg.probe.probe_type
    # Support comma-separated probe types (e.g. "linear,dom")
    requested = {p.strip() for p in pt.split(",")} if "," in pt else {pt}

    if "linear" in requested or "all" in requested:
        probes.append(("linear", lambda: LinearProbe(
            task_type=task_type,
            lr=cfg.probe.lr, max_epochs=cfg.probe.max_epochs,
            l2=cfg.probe.l2, device=cfg.model.device,
            train_batch_size=cfg.probe.probe_batch_size)))

    if "dom" in requested or "all" in requested:
        probes.append(("dom", lambda: DOMProbe(
            task_type=task_type,
            device=cfg.model.device)))

    if "attention" in requested or "all" in requested:
        probes.append(("attention", lambda: AttentionProbe(
            task_type=task_type,
            lr=cfg.probe.lr, max_epochs=cfg.probe.max_epochs,
            l2=cfg.probe.l2, device=cfg.model.device,
            train_batch_size=cfg.probe.probe_batch_size)))

    if "transformer" in requested or "all" in requested:
        probes.append(("transformer", lambda: TransformerBlockProbe(
            task_type=task_type,
            lr=cfg.probe.lr, max_epochs=cfg.probe.max_epochs,
            l2=cfg.probe.l2, device=cfg.model.device,
            train_batch_size=cfg.probe.probe_batch_size)))

    if "transformer_no_proj" in requested:
        # Transformer block without the input down-projection — d_model = D.
        probes.append(("transformer_no_proj", lambda: TransformerBlockProbe(
            task_type=task_type,
            proj_dim=None,
            lr=cfg.probe.lr, max_epochs=cfg.probe.max_epochs,
            l2=cfg.probe.l2, device=cfg.model.device,
            train_batch_size=cfg.probe.probe_batch_size)))

    if not probes:
        raise ValueError(f"Unknown probe_type: {cfg.probe.probe_type!r}")
    return probes


def needed_modes(cfg) -> Set[str]:
    """Return the set of hidden-state modes needed by all probes in cfg."""
    pt = cfg.probe.probe_type
    hs = cfg.model.hidden_state_type
    modes: Set[str] = set()

    if pt in ("linear", "dom", "all"):
        if hs == "all":
            modes.update(["average", "assistant_token"])
        else:
            modes.add(hs)

    if pt in ("attention", "transformer", "all"):
        modes.add("all_tokens")

    return modes


def needed_modes_with_ret(cfg, ret_active: bool) -> Set[str]:
    """Return modes to extract from the LLM, accounting for RET.

    When RET is active, always include ``all_tokens`` because the encoder
    must operate on full sequences before pooling.
    """
    modes = needed_modes(cfg)
    if ret_active:
        modes.add("all_tokens")
    return modes


# ---------------------------------------------------------------------------
# Shared helpers extracted from experiment runners
# ---------------------------------------------------------------------------

def get_pooled_modes(modes) -> List[str]:
    """Return the subset of pooled modes present in *modes*."""
    return [m for m in ["average", "assistant_token"] if m in modes]


def _strip_training_history(obj):
    """Deep-copy a results dict with training_history keys removed."""
    if isinstance(obj, dict):
        return {k: _strip_training_history(v)
                for k, v in obj.items() if k != "training_history"}
    if isinstance(obj, list):
        return [_strip_training_history(v) for v in obj]
    return obj


def save_probe_results(probe_dir: str, probe_name: str, probe_results: dict) -> str:
    """Save probe results as JSON (training_history stripped). Returns path."""
    out_path = os.path.join(probe_dir, "results.json")
    clean = _strip_training_history(probe_results)
    with open(out_path, "w") as f:
        json.dump(clean, f, indent=2,
                  default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x)
    logger.info("Probe %s results saved to %s", probe_name, out_path)
    return out_path


def save_experiment_summary(output_dir: str, all_results: Dict, experiment_name: str = "") -> str:
    """Write summary.json with key metrics from all probes.

    Extracts only accuracy, balanced_accuracy, f1, n_samples from each
    probe/mode/split combination into a compact summary for quick comparison.
    """
    KEEP_KEYS = {"accuracy", "balanced_accuracy", "f1", "n_samples", "n_positive"}

    def _extract_metrics(obj):
        """Recursively extract only key metrics from nested result dicts."""
        if isinstance(obj, dict):
            if "accuracy" in obj:
                return {k: v for k, v in obj.items() if k in KEEP_KEYS}
            return {k: _extract_metrics(v) for k, v in obj.items()
                    if k != "training_history"}
        return obj

    import datetime
    summary = {
        "experiment": experiment_name,
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "probes": {},
    }
    for probe_name, probe_results in all_results.items():
        summary["probes"][probe_name] = _extract_metrics(probe_results)

    out_path = os.path.join(output_dir, "summary.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2,
                  default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x)
    logger.info("Experiment summary saved to %s", out_path)
    return out_path


def run_eval_loop(cfg, dataset_names, hidden_states, labels,
                  probe_name, probe_factory, mode_key, probe_results,
                  output_dir, experiment_name):
    """Dispatch cross-dataset and/or within-dataset evaluation for one probe + mode.

    This is the shared eval loop used by multi-dataset probing experiments.
    """
    from evaluation.probing.evaluation import cross_dataset_eval, within_dataset_eval, print_results

    eval_types = (["cross_dataset", "within_dataset"]
                  if cfg.eval.eval_type == "all" else [cfg.eval.eval_type])
    cm_dir = os.path.join(output_dir, "confusion_matrices", mode_key)
    os.makedirs(cm_dir, exist_ok=True)

    for eval_type in eval_types:
        logger.info("Probe=%s  eval=%s  hs_mode=%s", probe_name, eval_type, mode_key)
        fn = cross_dataset_eval if eval_type == "cross_dataset" else within_dataset_eval
        kwargs = dict(output_dir=cm_dir, probe_name=probe_name)
        if eval_type == "within_dataset":
            kwargs["val_ratio"] = cfg.eval.val_ratio
        r = fn(dataset_names, hidden_states, labels, probe_factory, **kwargs)
        probe_results.setdefault(f"{mode_key}/{eval_type}", {})[probe_name] = r
        print_results({probe_name: r},
                      title=f"{experiment_name} | hs={mode_key} | eval={eval_type}")


def apply_pca_encoding_to_cache(cfg, hs_cache):
    """Apply PCA projection to pooled hidden states if configured.

    When ``cfg.pca.pca_dir`` is set, loads (or auto-fits) PCA components
    and projects all pooled modes through PCA.  Returns the transformed
    cache dict (modified in-place).

    Skips ``all_tokens`` — PCA on per-token data requires streaming from
    memmap and is not supported here.
    """
    pca_cfg = getattr(cfg, "pca", None)
    if pca_cfg is None or pca_cfg.pca_dir is None:
        return hs_cache

    from evaluation.probing.pca_utils import load_or_fit_pca, apply_pca

    cache_dir = getattr(cfg.cache, "cache_dir", None)
    components, mean = load_or_fit_pca(
        pca_cfg.pca_dir,
        cache_dir=cache_dir,
        pca_dim=pca_cfg.pca_dim,
    )

    for mode in ["average", "assistant_token"]:
        if mode in hs_cache:
            hs_cache[mode] = apply_pca(hs_cache[mode], components, mean)
            logger.info("  PCA projected %s: %d -> %d",
                        mode, components.shape[1], components.shape[0])

    return hs_cache


def apply_sae_encoding_to_cache(cfg, hs_cache, device: str = "cuda"):
    """Apply SAE encoding to pooled hidden states if configured.

    When ``cfg.sae.checkpoint_path`` or ``cfg.sae.hf_repo`` is set,
    loads the SAE and encodes all pooled modes.  Returns the transformed
    cache dict (modified in-place).

    Encodes the already-pooled representations (not per-token).
    """
    import torch

    sae_cfg = getattr(cfg, "sae", None)
    if sae_cfg is None:
        return hs_cache
    if sae_cfg.checkpoint_path is None and sae_cfg.hf_repo is None:
        return hs_cache

    from evaluation.probing.sae_utils import load_sae_encoder, free_sae_encoder

    sae, sae_info = load_sae_encoder(
        checkpoint_path=sae_cfg.checkpoint_path,
        hf_repo=sae_cfg.hf_repo,
        device=device,
    )

    for mode in ["average", "assistant_token"]:
        if mode in hs_cache:
            h = torch.from_numpy(hs_cache[mode]).to(device, dtype=torch.float32)
            with torch.no_grad():
                encoded = sae.encode(h)
            hs_cache[mode] = encoded.cpu().numpy()
            logger.info("  SAE encoded %s: %d -> %d",
                        mode, sae_info["d_in"], sae_info["d_sae"])

    free_sae_encoder(sae)
    return hs_cache


def apply_ret_encoding(cfg, hs_cache, modes, system_prefix_len: int = 0):
    """Apply RET encoder to cached hidden states if active. Returns (hs_cache, modes).

    Handles two cache layouts:
    - Per-dataset: ``{ds_name: {mode: array, ...}, ...}`` — encodes each dataset
    - Flat: ``{mode: array, ...}`` — encodes the single cache directly

    When RET is not active (no checkpoint), returns inputs unchanged.

    Parameters
    ----------
    system_prefix_len : number of system-prompt prefix tokens.  Passed to
        ``apply_ret_encoder`` so that probe pooling excludes system tokens
        and the RET forward-pass inclusion can be controlled via
        ``cfg.ret.include_system_in_ret``.
    """
    if cfg.ret.checkpoint_path is None:
        return hs_cache, modes

    from evaluation.probing.ret_utils import load_ret_encoder, apply_ret_encoder, free_ret_encoder

    ret_encoder, _ret_cfg = load_ret_encoder(cfg.ret.checkpoint_path, device=cfg.ret.device)
    probe_modes = needed_modes(cfg)

    include_sys = getattr(cfg.ret, "include_system_in_ret", True)

    first_val = next(iter(hs_cache.values()))
    is_per_dataset = isinstance(first_val, dict) and not (
        set(first_val.keys()) <= {"hidden_states", "attention_mask"}
    )

    if is_per_dataset:
        for name in list(hs_cache.keys()):
            hs_cache[name] = apply_ret_encoder(
                ret_encoder, hs_cache[name]["all_tokens"],
                modes=probe_modes, device=cfg.ret.device,
                system_prefix_len=system_prefix_len,
                include_system_in_ret=include_sys,
            )
            logger.info("  RET encoded %s: modes=%s", name, list(hs_cache[name].keys()))
    else:
        hs_cache = apply_ret_encoder(
            ret_encoder, hs_cache["all_tokens"],
            modes=probe_modes, device=cfg.ret.device,
            system_prefix_len=system_prefix_len,
            include_system_in_ret=include_sys,
        )
        logger.info("  RET encoded: modes=%s", list(hs_cache.keys()))

    free_ret_encoder(ret_encoder)
    return hs_cache, probe_modes


def _probe_loop(cfg, probe_list, experiment_name, eval_one) -> Dict:
    """Shared probe-loop skeleton for the two run_probes_and_save drivers.

    ``eval_one(probe_name, probe_factory, probe_dir, probe_results)`` fills
    ``probe_results`` for one probe; this owns the loop, per-probe dir,
    results JSON, training curves, and aggregation.
    """
    all_results: Dict = {}
    for probe_name, probe_factory in probe_list:
        probe_dir = probe_output_dir(cfg.output_dir, probe_name, experiment_name)
        os.makedirs(probe_dir, exist_ok=True)
        probe_results: Dict = {}

        eval_one(probe_name, probe_factory, probe_dir, probe_results)

        save_probe_results(probe_dir, probe_name, probe_results)
        save_training_curves(probe_dir, probe_results, probe_name)
        all_results[probe_name] = probe_results

    return all_results


def run_probes_and_save_presplit(
    cfg, probe_list,
    train_hs, train_labels,
    test_hs, test_labels,
    dataset_names, pooled_modes, experiment_name,
    eval_type: str = "cross_dataset",
) -> Dict:
    """Run all probes using pre-split train/test hidden states and save results.

    Parameters
    ----------
    eval_type : "cross_dataset", "within_dataset", or "all".
    """
    from evaluation.probing.evaluation import (
        presplit_cross_dataset_eval, presplit_within_dataset_eval, print_results,
    )

    eval_types = (["cross_dataset", "within_dataset"]
                  if eval_type == "all" else [eval_type])

    def _eval_one(probe_name, probe_factory, probe_dir, probe_results):
        for et in eval_types:
            eval_fn = (presplit_cross_dataset_eval if et == "cross_dataset"
                       else presplit_within_dataset_eval)

            if probe_name == "attention":
                tr = {n: train_hs[n]["all_tokens"] for n in dataset_names if n in train_hs}
                te = {n: test_hs[n]["all_tokens"] for n in dataset_names if n in test_hs}
                r = eval_fn(dataset_names, tr, train_labels, te, test_labels,
                            probe_factory, output_dir=probe_dir, probe_name=probe_name)
                probe_results[f"all_tokens/{et}"] = {probe_name: r}
                print_results({probe_name: r},
                              title=f"{experiment_name} | hs=all_tokens | eval={et}")
            else:
                for hs_mode in pooled_modes:
                    tr = {n: train_hs[n][hs_mode] for n in dataset_names if n in train_hs}
                    te = {n: test_hs[n][hs_mode] for n in dataset_names if n in test_hs}
                    r = eval_fn(dataset_names, tr, train_labels, te, test_labels,
                                probe_factory, output_dir=probe_dir, probe_name=probe_name)
                    probe_results[f"{hs_mode}/{et}"] = {probe_name: r}
                    print_results({probe_name: r},
                                  title=f"{experiment_name} | hs={hs_mode} | eval={et}")

    return _probe_loop(cfg, probe_list, experiment_name, _eval_one)


def run_probes_and_save(cfg, probe_list, hs_cache, labels,
                        dataset_names, pooled_modes, experiment_name) -> Dict:
    """Run all probes on multi-dataset cached hidden states and save results.

    For each probe: dispatches attention (all_tokens) and pooled modes (average/assistant_token)
    through run_eval_loop, then saves per-probe results JSON.

    Used by multi-dataset probing experiments.
    """
    def _eval_one(probe_name, probe_factory, probe_dir, probe_results):
        if probe_name == "attention":
            hs_by_ds = {n: hs_cache[n]["all_tokens"] for n in dataset_names}
            run_eval_loop(cfg, dataset_names, hs_by_ds, labels,
                          probe_name, probe_factory, "all_tokens", probe_results,
                          output_dir=probe_dir, experiment_name=experiment_name)
        else:
            for hs_mode in pooled_modes:
                hs_by_ds = {n: hs_cache[n][hs_mode] for n in dataset_names}
                run_eval_loop(cfg, dataset_names, hs_by_ds, labels,
                              probe_name, probe_factory, hs_mode, probe_results,
                              output_dir=probe_dir, experiment_name=experiment_name)

    return _probe_loop(cfg, probe_list, experiment_name, _eval_one)


def save_training_curves(probe_dir: str, probe_results: dict, probe_name: str) -> None:
    """Save training curve plots from training_history in probe results.

    Creates ``training_curves.png`` in *probe_dir* with one row per
    subtask/mode combination that has a training history, showing loss
    and accuracy over epochs.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available; skipping training curve plots")
        return

    # Collect all (key, history) pairs from arbitrarily nested results
    curves = []

    def _walk(obj, prefix=""):
        if isinstance(obj, dict):
            if "training_history" in obj:
                curves.append((prefix, obj["training_history"]))
            else:
                for k, v in obj.items():
                    _walk(v, f"{prefix}/{k}" if prefix else k)

    _walk(probe_results)

    if not curves:
        return

    n = len(curves)
    fig, axes = plt.subplots(n, 2, figsize=(12, 3.5 * n), squeeze=False)

    for i, (label, hist) in enumerate(curves):
        epochs_train = list(range(len(hist["train_loss"])))

        # Loss subplot
        ax_loss = axes[i, 0]
        ax_loss.plot(epochs_train, hist["train_loss"], label="train", alpha=0.7)
        if "val_loss" in hist and hist["val_loss"]:
            ax_loss.plot(hist["val_epoch"], hist["val_loss"],
                         "o-", label="val", markersize=3)
        ax_loss.set_xlabel("epoch")
        ax_loss.set_ylabel("loss")
        ax_loss.set_title(f"{label} — Loss")
        ax_loss.legend()
        ax_loss.grid(True, alpha=0.3)

        # Accuracy subplot
        ax_acc = axes[i, 1]
        ax_acc.plot(epochs_train, hist["train_acc"], label="train", alpha=0.7)
        if "val_acc" in hist and hist["val_acc"]:
            ax_acc.plot(hist["val_epoch"], hist["val_acc"],
                        "o-", label="val", markersize=3)
        ax_acc.set_xlabel("epoch")
        ax_acc.set_ylabel("accuracy")
        ax_acc.set_title(f"{label} — Accuracy")
        ax_acc.legend()
        ax_acc.grid(True, alpha=0.3)

    fig.suptitle(f"{probe_name} probe — Training Curves", fontsize=14, y=1.01)
    fig.tight_layout()
    path = os.path.join(probe_dir, "training_curves.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved training curves: %s", path)
