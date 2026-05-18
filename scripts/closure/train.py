#!/usr/bin/env python3
"""Train closure predictor probes from one or more config files.

Each config YAML declares a ``closure.variant`` field that identifies the
representation type.  Multiple configs can be passed at once; they all share
one LLM forward-pass per training step (same extractor, one dataloader).

Supported variants
------------------
``z_ret``   — RET encoder checkpoint.  Loads checkpoint from
              ``logging.output_dir/checkpoint_final.pt``.
              Trains both autonomous and controlled sub-variants.
``h``       — Raw hidden states h→h.
``h_pca``   — PCA-projected h.  PCA loaded from
              ``logging.output_dir/pca_components.npz`` (auto-fitted if missing).
``sae``     — SAE-encoded representations.

Output dir comes from ``config.closure.output_dir`` if set, else
``config.logging.output_dir``.  Predictors are saved under::

    <output_dir>/closure/hdim{N}/predictors.pt

Usage
-----
::

    # Multiple configs = multiple variants sharing one LLM extraction pass
    python scripts/closure/train.py \\
        configs/GptOss20B/cached_autonomous.yaml \\
        configs/GptOss20B/baselines/h.yaml \\
        configs/GptOss20B/baselines/h_pca128.yaml

    # Single variant
    python scripts/closure/train.py configs/GptOss20B/baselines/h.yaml

    # Print existing sweep results (no training)
    python scripts/closure/train.py configs/GptOss20B/cached_autonomous.yaml --summarize-only
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from ret.config import save_config, ExperimentConfig
from ret.extraction.hooks import HiddenStateExtractor
from ret.utils.checkpointing import load_encoder
from ret.utils.seeding import seed_everything
from ret.utils.prompt_logger import PromptLogger
from evaluation.baselines.sae import encode_sae, load_sae_baseline
from evaluation.closure import (
    build_closure_predictor,
    build_token_projector,
    compute_loss,
    compute_r2_streaming,
    collect_val_data,
    save_closure_predictors,
)
from evaluation.closure.config import (
    ensure_list,
    load_config_with_overlay,
    build_closure_dataloader,
    closure_data_tag,
    closure_prompt_source,
    make_train_iter,
)
from evaluation.closure.plotting.plots import plot_training_curves
from evaluation.closure.sweep import (
    _discover_sweep_results,
    _print_sweep_summary,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _resolve_output_dir(config: ExperimentConfig) -> Path:
    """Return the output dir for a variant config."""
    if config.closure.output_dir:
        return Path(config.closure.output_dir)
    return Path(config.logging.output_dir)


def _llm_data_fingerprint(config: ExperimentConfig) -> tuple:
    """Return a tuple that uniquely identifies (LLM, data) settings.

    Used for cross-config validation: all configs in one invocation must
    agree on the LLM and data source.
    """
    d = config.data
    return (
        config.llm.name,
        config.llm.layer_index,
        config.llm.hidden_dim,
        d.mode,
        d.hf_dataset,
        d.local_dir,
        d.cached_hidden_states_dir,
        d.scaffolding,
        d.text_field,
    )


# ---------------------------------------------------------------------------
# Cache-skip helpers
# ---------------------------------------------------------------------------

def _hparams_from_config(config: ExperimentConfig, hdim: int) -> dict:
    """Build the hparams dict to compare against saved closure_config.json."""
    c = config.closure
    hp: dict = {
        "hidden_dim": hdim,
        "num_layers": c.num_layers,
        "lr": c.lr,
        "max_steps": c.max_steps,
        "batch_size": c.batch_size,
        "patience": c.patience,
        "val_batches": c.val_batches,
        "eval_every": c.eval_every,
    }
    if c.variant == "z_ret":
        hp["control_dim"] = c.control_dim
    elif c.variant == "h_pca":
        hp["pca_dim"] = c.pca_dim
    return hp


def _should_skip(save_dir: Path, hparams: dict, force: bool, label: str) -> bool:
    """Return True if this predictor dir is already up-to-date."""
    if force:
        return False
    pred_pt = save_dir / "predictors.pt"
    cfg_json = save_dir / "closure_config.json"
    if not pred_pt.exists():
        return False
    if cfg_json.exists():
        with open(cfg_json) as f:
            saved = json.load(f)
        if saved == hparams:
            print(f"  {label}: up-to-date at {save_dir}. Skipping.")
            return True
        print(f"  {label}: config changed, retraining...")
        return False
    print(f"  {label}: already exists at {save_dir}. Use --force to retrain.")
    return True


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------

def _data_meta(config: ExperimentConfig) -> dict:
    d = config.data
    return {
        "data_hf_dataset": d.hf_dataset,
        "data_local_dir": d.local_dir,
        "data_text_field": d.text_field,
        "data_scaffolding": d.scaffolding,
        "data_system_message": d.system_message,
    }


# ---------------------------------------------------------------------------
# Core training function
# ---------------------------------------------------------------------------

def run_config_driven(args: argparse.Namespace) -> None:
    """Train closure predictors for all variants defined by the given config files.

    All variants share a single LLM forward pass per training step.
    """
    # ------------------------------------------------------------------ #
    # 1. Load configs and resolve variants
    # ------------------------------------------------------------------ #
    overlay_yaml = args.config_overlay

    configs: list[ExperimentConfig] = []
    for cfg_path in args.configs:
        configs.append(load_config_with_overlay(cfg_path, overlay_yaml))

    # Validate: all configs must agree on LLM + data
    if len(configs) > 1:
        ref_fp = _llm_data_fingerprint(configs[0])
        for i, cfg in enumerate(configs[1:], 1):
            fp = _llm_data_fingerprint(cfg)
            if fp != ref_fp:
                raise ValueError(
                    f"Config mismatch: configs[0] ({args.configs[0]}) and "
                    f"configs[{i}] ({args.configs[i]}) have different llm/data sections.\n"
                    f"  ref:  {ref_fp}\n  got:  {fp}"
                )

    # Use the first config as the "primary" for shared settings
    primary_cfg = configs[0]

    # Variant registry: list of (variant_type, output_dir, config)
    variant_specs: list[tuple[str, Path, ExperimentConfig]] = []
    for cfg in configs:
        variant = cfg.closure.variant
        if variant is None:
            raise ValueError(
                f"Config loaded from {args.configs[configs.index(cfg)]} has no "
                f"closure.variant field. Set it to one of: z_ret, h, h_pca, sae."
            )
        if variant not in ("z_ret", "h", "h_pca", "sae"):
            raise ValueError(
                f"Unknown closure.variant={variant!r}. "
                f"Must be one of: z_ret, h, h_pca, sae."
            )
        output_dir = _resolve_output_dir(cfg)
        variant_specs.append((variant, output_dir, cfg))

    has_ret = any(v == "z_ret" for v, _, _ in variant_specs)
    has_hpca = any(v == "h_pca" for v, _, _ in variant_specs)
    has_sae = any(v == "sae" for v, _, _ in variant_specs)

    # Closure subdirectory name (changed when --exclude-scaffold)
    closure_subdir = "closure_wo_system_prompt" if args.exclude_scaffold else "closure"

    # ------------------------------------------------------------------ #
    # 2. Enumerate all (method_key, hdim, output_dir, config) tuples
    #    and check cache
    # ------------------------------------------------------------------ #
    # A "method_key" is a unique string per training predictor:
    #   z_ret -> "autonomous", "controlled"
    #   h     -> "hh"
    #   h_pca -> "pca"   (one per config, since pca_dim lives in the config)
    #   sae   -> "sae_autonomous", "sae_controlled"

    # For each (variant, output_dir, config), expand into (method_key, hdim, save_dir, config) tuples
    all_keys: list[tuple[str, int, Path, ExperimentConfig]] = []
    for variant, output_dir, cfg in variant_specs:
        dims = ensure_list(cfg.closure.hidden_dim)
        if variant == "z_ret":
            for hdim in dims:
                save_dir = output_dir / closure_subdir / f"hdim{hdim}"
                all_keys.append(("autonomous", hdim, save_dir, cfg))
                all_keys.append(("controlled", hdim, save_dir, cfg))
        elif variant == "h":
            for hdim in dims:
                save_dir = output_dir / closure_subdir / f"hdim{hdim}"
                all_keys.append(("hh", hdim, save_dir, cfg))
        elif variant == "h_pca":
            for hdim in dims:
                save_dir = output_dir / closure_subdir / f"hdim{hdim}"
                all_keys.append(("pca", hdim, save_dir, cfg))
        elif variant == "sae":
            for hdim in dims:
                save_dir = output_dir / closure_subdir / f"hdim{hdim}"
                all_keys.append(("sae_autonomous", hdim, save_dir, cfg))
                all_keys.append(("sae_controlled", hdim, save_dir, cfg))

    # De-duplicate save_dirs for same-variant same-hdim (z_ret: autonomous+controlled share a dir)
    # The hparams check is per-save_dir, not per-key.
    # We check caching at the save_dir level (each save_dir stores multiple keys).
    checked_dirs: set[Path] = set()
    skip_save_dirs: set[Path] = set()
    cached_r2: dict[tuple[str, int], float | None] = {}

    for method_key, hdim, save_dir, cfg in all_keys:
        if save_dir in checked_dirs:
            continue
        checked_dirs.add(save_dir)
        hp = _hparams_from_config(cfg, hdim)
        label = f"{cfg.closure.variant} hdim={hdim}"
        if _should_skip(save_dir, hp, args.force, label):
            skip_save_dirs.add(save_dir)
            # Read cached R² for later summary
            pred_pt = save_dir / "predictors.pt"
            if pred_pt.exists():
                try:
                    cached_data = torch.load(pred_pt, map_location="cpu", weights_only=False)
                    for k, v in cached_data.items():
                        if k != "__meta__" and isinstance(v, dict) and "val_r2_final" in v:
                            cached_r2[(k, hdim)] = v.get("val_r2_final")
                except Exception:
                    pass

    train_tuples = [
        t for t in all_keys if t[2] not in skip_save_dirs
    ]

    if not train_tuples:
        print("All predictors up-to-date. Nothing to train.")
        return

    # Gather distinct method_keys that need training (for building predictors)
    # Use a unique key: (method_key, hdim, save_dir) — save_dir disambiguates same method for diff variants
    train_key_set: set[tuple[str, int, Path]] = {
        (mk, hdim, sd) for mk, hdim, sd, _ in train_tuples
    }
    # Map (method_key, hdim, save_dir) -> cfg
    train_key_cfg: dict[tuple[str, int, Path], ExperimentConfig] = {
        (mk, hdim, sd): cfg for mk, hdim, sd, cfg in train_tuples
    }

    print(f"Training {len(train_key_set)} predictor stream(s), "
          f"skipping {len(skip_save_dirs)} cached save dir(s).")

    # ------------------------------------------------------------------ #
    # 3. Seed + build shared infrastructure
    # ------------------------------------------------------------------ #
    seed_everything(args.seed)

    d_h = primary_cfg.llm.hidden_dim
    loss_type = "mse"

    print("Loading LLM extractor ...")
    extractor = HiddenStateExtractor(primary_cfg.llm, device=args.device)
    print("LLM extractor ready.")

    # Compute scaffold prefix length for --exclude-scaffold
    scaffold_prefix_len = 0
    if args.exclude_scaffold:
        from ret.data.dataset import _compute_scaffold_tokens
        _prefix_ids, _suffix_ids = _compute_scaffold_tokens(
            primary_cfg.data, extractor.tokenizer,
        )
        scaffold_prefix_len = len(_prefix_ids) if _prefix_ids is not None else 0
        if scaffold_prefix_len > 0:
            print(f"  Excluding scaffold prefix: {scaffold_prefix_len} tokens per document")
        else:
            print("  No scaffold detected (scaffolding=raw); --exclude-scaffold is a no-op")

    # RET encoder (needed if any z_ret variant)
    encoder = None
    d_z: int | None = None
    ret_configs: list[tuple[Path, ExperimentConfig]] = [
        (od, cfg) for v, od, cfg in variant_specs if v == "z_ret"
    ]
    if ret_configs:
        # All z_ret variants must use the same encoder; use the first one
        ret_output_dir, ret_cfg = ret_configs[0]
        ckpt_path = ret_output_dir / "checkpoint_final.pt"
        print(f"Loading RET encoder from {ckpt_path} ...")
        encoder, _, _ = load_encoder(ckpt_path, device=args.device, config=ret_cfg)
        d_z = ret_cfg.encoder.z_dim
        print(f"  RET encoder ready. z_dim={d_z}")

    # SAE model (needed if any sae variant)
    sae_model = None
    d_sae: int | None = None
    sae_configs: list[tuple[Path, ExperimentConfig]] = [
        (od, cfg) for v, od, cfg in variant_specs if v == "sae"
    ]
    if sae_configs:
        sae_output_dir, sae_cfg_ref = sae_configs[0]
        print("Loading SAE baseline ...")
        sae_result = load_sae_baseline(primary_cfg.llm, device=args.device)
        if sae_result is None:
            extractor.cleanup()
            raise RuntimeError("Could not load SAE baseline. Aborting.")
        sae_model, _sae_release, _sae_id = sae_result
        sample_sae = encode_sae(sae_model, torch.zeros(1, d_h, device=args.device))
        d_sae = sample_sae.shape[-1]
        print(f"  SAE ready. d_sae={d_sae}")

    # PCA components (needed if any h_pca variant); one per (output_dir, pca_dim) pair
    pca_specs: list[tuple[Path, ExperimentConfig]] = [
        (od, cfg) for v, od, cfg in variant_specs if v == "h_pca"
    ]
    # pca_comp_tensors[(output_dir, pca_dim)] = (components_tensor, mean_tensor, explained)
    pca_comp_tensors: dict[tuple[str, int], torch.Tensor] = {}
    pca_mean_tensors: dict[tuple[str, int], torch.Tensor] = {}
    pca_explained: dict[tuple[str, int], float] = {}

    # ------------------------------------------------------------------ #
    # 4. Build shared dataloader + collect val data
    # ------------------------------------------------------------------ #
    print("Building dataloader ...")
    train_loader = build_closure_dataloader(primary_cfg)
    print("Dataloader ready.")

    # Prompt logging (one log per distinct output_dir, max_entries=50 for RET, 5 for baselines)
    _source = closure_prompt_source(primary_cfg)
    for v, output_dir, cfg in variant_specs:
        max_ent = 50 if v == "z_ret" else 5
        closure_log_dir = output_dir / closure_subdir
        closure_log_dir.mkdir(parents=True, exist_ok=True)
        pl = PromptLogger(closure_log_dir, max_entries=max_ent)
        _log_dl = build_closure_dataloader(primary_cfg, split="train")
        for _batch in _log_dl:
            if pl.is_full:
                break
            pl.log_forward_pass_batch(
                _batch["input_ids"], extractor.tokenizer,
                attention_mask=_batch.get("attention_mask"),
                source=_source,
            )
        pl.save()
        del _log_dl

    # Collect val data ONCE (encoder included for z_ret if needed)
    val_batches = max(cfg.closure.val_batches for _, _, cfg in variant_specs)
    val_loader = build_closure_dataloader(primary_cfg, split="val")
    print(f"Collecting val data ({val_batches} batches from val split) ...")
    val_data = collect_val_data(
        extractor, val_loader, val_batches, args.device,
        encoder=encoder,
        scaffold_prefix_len=scaffold_prefix_len,
    )
    del val_loader
    N_val = val_data["h_t"].shape[0]
    print(f"  Collected {N_val} val temporal pairs")

    # ------------------------------------------------------------------ #
    # 5. Fit / load PCA components for each h_pca variant
    # ------------------------------------------------------------------ #
    pca_needs_fit: list[tuple[Path, ExperimentConfig]] = []
    for output_dir, cfg in pca_specs:
        pca_dim = cfg.closure.pca_dim
        if pca_dim is None:
            raise ValueError(
                f"h_pca variant at {output_dir} has no closure.pca_dim set."
            )
        pca_key = (str(output_dir), pca_dim)
        npz_path = output_dir / "pca_components.npz"
        if not args.force and npz_path.exists():
            print(f"Loading cached PCA-{pca_dim} from {npz_path}")
            saved = np.load(npz_path)
            pca_comp_tensors[pca_key] = torch.from_numpy(saved["components"].copy()).float().to(args.device)
            pca_mean_tensors[pca_key] = torch.from_numpy(saved["mean"].copy()).float().to(args.device)
            pca_explained[pca_key] = float(saved["explained_variance_ratio"].sum())
            print(f"  PCA-{pca_dim} explained variance: {pca_explained[pca_key]:.4f} (cached)")
        else:
            pca_needs_fit.append((output_dir, cfg))

    if pca_needs_fit:
        from sklearn.decomposition import PCA as _PCA
        pca_fit_loader = build_closure_dataloader(primary_cfg, split="train")
        pca_fit_batches = max(cfg.closure.val_batches for _, cfg in pca_needs_fit)
        pca_fit_data = collect_val_data(extractor, pca_fit_loader, pca_fit_batches, args.device)
        del pca_fit_loader
        h_t_np_fit = pca_fit_data["h_t"].cpu().numpy()

        for output_dir, cfg in pca_needs_fit:
            pca_dim = cfg.closure.pca_dim
            pca_key = (str(output_dir), pca_dim)
            print(f"Fitting PCA (d_h={d_h} -> pca_dim={pca_dim}) on {h_t_np_fit.shape[0]} samples ...")
            pca_obj = _PCA(n_components=pca_dim)
            pca_obj.fit(h_t_np_fit)
            explained = pca_obj.explained_variance_ratio_.sum()
            print(f"  PCA-{pca_dim} explained variance: {explained:.4f}")
            pca_comp_tensors[pca_key] = torch.from_numpy(pca_obj.components_).float().to(args.device)
            pca_mean_tensors[pca_key] = torch.from_numpy(pca_obj.mean_).float().to(args.device)
            pca_explained[pca_key] = float(explained)
            # Save for future reuse
            output_dir.mkdir(parents=True, exist_ok=True)
            np.savez(
                output_dir / "pca_components.npz",
                components=pca_obj.components_,
                mean=pca_obj.mean_,
                explained_variance_ratio=pca_obj.explained_variance_ratio_,
                pca_dim=pca_dim,
            )
            print(f"  Saved PCA components to {output_dir / 'pca_components.npz'}")

        del pca_fit_data, h_t_np_fit

    # ------------------------------------------------------------------ #
    # 6. Pre-compute val representations per variant type
    # ------------------------------------------------------------------ #
    val_h_t = val_data["h_t"]
    val_h_next = val_data["h_next"]
    val_te_next = val_data["token_emb_next"]

    # val_repr: dict[(method_key, save_dir)] -> {"repr_t": tensor, "repr_next": tensor}
    # For "controlled" types we store (repr_t, h_t, te_next, repr_next)
    val_simple: dict[tuple[str, Path], tuple[torch.Tensor, torch.Tensor]] = {}
    val_ctrl: dict[tuple[str, Path], tuple] = {}
    target_variances: dict[tuple[str, Path], float] = {}

    for variant, output_dir, cfg in variant_specs:
        dims = ensure_list(cfg.closure.hidden_dim)
        for hdim in dims:
            save_dir = output_dir / closure_subdir / f"hdim{hdim}"

            if variant == "z_ret":
                # z_t, z_next already in val_data (collected with encoder)
                z_t = val_data["z_t"]
                z_next = val_data["z_next"]
                tv = z_next.to(args.device).var(dim=0).mean().item()
                for method_key in ("autonomous", "controlled"):
                    key = (method_key, save_dir)
                    if method_key == "autonomous":
                        val_simple[key] = (z_t, z_next)
                    else:
                        val_ctrl[key] = (z_t, val_h_t, val_te_next, z_next)
                    target_variances[key] = tv

            elif variant == "h":
                tv = val_h_next.to(args.device).var(dim=0).mean().item()
                key = ("hh", save_dir)
                val_simple[key] = (val_h_t, val_h_next)
                target_variances[key] = tv

            elif variant == "h_pca":
                pca_dim = cfg.closure.pca_dim
                pca_key = (str(output_dir), pca_dim)
                comp_cpu = pca_comp_tensors[pca_key].cpu()
                mean_cpu = pca_mean_tensors[pca_key].cpu()
                val_z_t_pca = (val_h_t.float() - mean_cpu) @ comp_cpu.T
                val_z_next_pca = (val_h_next.float() - mean_cpu) @ comp_cpu.T
                tv = val_z_next_pca.to(args.device).var(dim=0).mean().item()
                key = ("pca", save_dir)
                val_simple[key] = (val_z_t_pca, val_z_next_pca)
                target_variances[key] = tv

            elif variant == "sae":
                val_sae_t = encode_sae(sae_model, val_h_t.to(args.device))
                val_sae_next = encode_sae(sae_model, val_h_next.to(args.device))
                # val_sae_next is CPU float32 ~N×131072 — computing var on GPU needs
                # the full tensor there (~50GB for 100k pairs). Compute on CPU instead.
                tv = val_sae_next.var(dim=0).mean().item()
                for method_key in ("sae_autonomous", "sae_controlled"):
                    key = (method_key, save_dir)
                    if method_key == "sae_autonomous":
                        val_simple[key] = (val_sae_t, val_sae_next)
                    else:
                        val_ctrl[key] = (val_sae_t, val_h_t, val_te_next, val_sae_next)
                    target_variances[key] = tv

    print(f"\nTarget variances:")
    seen_tvs: set[float] = set()
    for (mk, sd), tv in target_variances.items():
        if tv not in seen_tvs:
            print(f"  {mk} @ {sd.parent.name}: {tv:.6f}")
            seen_tvs.add(tv)

    # ------------------------------------------------------------------ #
    # 7. Build predictors and optimizers for all training keys
    # ------------------------------------------------------------------ #
    # Map: (method_key, hdim, save_dir) -> config
    # Derive input/output dims
    def _dims_for_key(method_key: str, cfg: ExperimentConfig) -> tuple[int, int]:
        if method_key == "autonomous":
            return (d_z, d_z)
        elif method_key == "controlled":
            return (d_z + cfg.closure.control_dim, d_z)
        elif method_key == "hh":
            return (d_h, d_h)
        elif method_key == "pca":
            pca_dim = cfg.closure.pca_dim
            return (pca_dim, pca_dim)
        elif method_key == "sae_autonomous":
            return (d_sae, d_sae)
        elif method_key == "sae_controlled":
            return (d_sae + cfg.closure.control_dim, d_sae)
        else:
            raise ValueError(f"Unknown method_key: {method_key!r}")

    CTRL_KEYS = {"controlled", "sae_controlled"}

    predictors_map: dict[tuple[str, int, Path], nn.Module] = {}
    optimizers_map: dict[tuple[str, int, Path], torch.optim.Optimizer] = {}
    projectors_map: dict[tuple[str, int, Path], nn.Module] = {}

    for tk in train_key_set:
        method_key, hdim, save_dir = tk
        cfg = train_key_cfg[tk]
        ccfg = cfg.closure
        in_dim, out_dim = _dims_for_key(method_key, cfg)
        pred = build_closure_predictor(in_dim, out_dim, hdim, ccfg.num_layers).to(args.device)
        predictors_map[tk] = pred
        if method_key in CTRL_KEYS:
            proj = build_token_projector(d_h, ccfg.control_dim).to(args.device)
            projectors_map[tk] = proj
            optimizers_map[tk] = torch.optim.Adam(
                list(pred.parameters()) + list(proj.parameters()), lr=ccfg.lr
            )
        else:
            optimizers_map[tk] = torch.optim.Adam(pred.parameters(), lr=ccfg.lr)

    all_train_keys = sorted(train_key_set, key=lambda t: (t[0], t[1], str(t[2])))

    # ------------------------------------------------------------------ #
    # 8. Training state
    # ------------------------------------------------------------------ #
    best_states: dict[tuple, dict | None] = {k: None for k in all_train_keys}
    best_proj_states: dict[tuple, dict | None] = {k: None for k in projectors_map}
    best_val_losses: dict[tuple, float] = {k: float("inf") for k in all_train_keys}
    evals_without_improvement: dict[tuple, int] = {k: 0 for k in all_train_keys}
    active_keys: set[tuple] = set(all_train_keys)
    train_loss_accum: dict[tuple, list] = {k: [0.0, 0] for k in all_train_keys}
    all_train_losses_rec: dict[tuple, list] = {k: [] for k in all_train_keys}
    all_val_losses_rec: dict[tuple, list] = {k: [] for k in all_train_keys}
    all_val_r2_rec: dict[tuple, list] = {k: [] for k in all_train_keys}

    def _ccfg_for(tk):
        return train_key_cfg[tk].closure

    max_steps = max(_ccfg_for(tk).max_steps for tk in all_train_keys)
    eval_every = min(_ccfg_for(tk).eval_every for tk in all_train_keys)
    patience_map = {tk: _ccfg_for(tk).patience for tk in all_train_keys}
    closure_batch_size = min(_ccfg_for(tk).batch_size for tk in all_train_keys)
    min_delta = 1e-6

    # ------------------------------------------------------------------ #
    # 9. Unified training loop
    # ------------------------------------------------------------------ #
    train_it = make_train_iter(train_loader)

    print(f"\nTraining {len(all_train_keys)} predictor(s) "
          f"(max_steps={max_steps}, eval_every={eval_every}) ...")
    for mk, hdim, sd in all_train_keys:
        print(f"  {mk} hdim={hdim} -> {sd}")

    pbar = tqdm(range(1, max_steps + 1), desc="Closure training")
    for step in pbar:
        if not active_keys:
            break

        # Shared LLM forward pass
        batch = next(train_it)
        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask")
        with torch.no_grad():
            if "hidden_states" in batch:
                h = batch["hidden_states"].to(args.device)
            else:
                h = extractor.extract(input_ids, attention_mask=attention_mask)
            if attention_mask is not None:
                attention_mask = attention_mask.to(h.device)
            token_emb = extractor.get_token_embeddings(input_ids)  # (B, T, d_h)

            # RET z (only if needed)
            z = None
            if encoder is not None and any(
                mk in ("autonomous", "controlled") for mk, _, _ in active_keys
            ):
                z, _aux = (
                    encoder(h, attention_mask=attention_mask)
                    if attention_mask is not None else encoder(h)
                )

        # Temporal pairs
        if attention_mask is not None:
            pair_mask = (attention_mask[:, :-1] * attention_mask[:, 1:]).bool()
            if scaffold_prefix_len > 0:
                pair_mask[:, :scaffold_prefix_len] = False
            h_t = h[:, :-1][pair_mask]
            h_next = h[:, 1:][pair_mask]
            te_next = token_emb[:, 1:][pair_mask]
            if z is not None:
                z_t = z[:, :-1][pair_mask]
                z_next = z[:, 1:][pair_mask]
        else:
            B, T, _ = h.shape
            if scaffold_prefix_len > 0:
                pair_mask = torch.ones(B, T - 1, dtype=torch.bool, device=h.device)
                pair_mask[:, :scaffold_prefix_len] = False
                h_t = h[:, :-1][pair_mask]
                h_next = h[:, 1:][pair_mask]
                te_next = token_emb[:, 1:][pair_mask]
                if z is not None:
                    z_t = z[:, :-1][pair_mask]
                    z_next = z[:, 1:][pair_mask]
            else:
                h_t = h[:, :-1].reshape(-1, d_h)
                h_next = h[:, 1:].reshape(-1, d_h)
                te_next = token_emb[:, 1:].reshape(-1, d_h)
                if z is not None:
                    z_t = z[:, :-1].reshape(-1, d_z)
                    z_next = z[:, 1:].reshape(-1, d_z)

        # SAE projections (compute once if needed this step)
        sae_t: torch.Tensor | None = None
        sae_next: torch.Tensor | None = None
        if sae_model is not None and any(
            mk in ("sae_autonomous", "sae_controlled") for mk, _, _ in active_keys
        ):
            with torch.no_grad():
                sae_acts = encode_sae(sae_model, h)  # (B, T, d_sae), CPU
                if attention_mask is not None:
                    pair_mask_cpu = pair_mask.cpu()
                    sae_t = sae_acts[:, :-1][pair_mask_cpu].to(args.device)
                    sae_next = sae_acts[:, 1:][pair_mask_cpu].to(args.device)
                else:
                    sae_t = sae_acts[:, :-1].reshape(-1, d_sae).to(args.device)
                    sae_next = sae_acts[:, 1:].reshape(-1, d_sae).to(args.device)

        # PCA projections (one per (output_dir, pca_dim) key)
        pca_proj_cache: dict[tuple[str, int], tuple[torch.Tensor, torch.Tensor]] = {}
        for mk, hdim_, sd_ in active_keys:
            if mk != "pca":
                continue
            cfg_ = train_key_cfg[(mk, hdim_, sd_)]
            pca_dim_ = cfg_.closure.pca_dim
            pca_key_ = (str(sd_.parent.parent), pca_dim_)
            if pca_key_ not in pca_proj_cache:
                comp = pca_comp_tensors[pca_key_]
                mu = pca_mean_tensors[pca_key_]
                with torch.no_grad():
                    pca_zt = (h_t.float() - mu) @ comp.T
                    pca_znext = (h_next.float() - mu) @ comp.T
                pca_proj_cache[pca_key_] = (pca_zt, pca_znext)

        # Sub-batch loop over pairs
        n_pairs = h_t.shape[0]
        for start in range(0, n_pairs, closure_batch_size):
            end = min(start + closure_batch_size, n_pairs)

            hb = h_t[start:end]
            hnb = h_next[start:end]
            teb = te_next[start:end]
            zb = z_t[start:end] if z is not None else None
            znb = z_next[start:end] if z is not None else None
            saetb = sae_t[start:end] if sae_t is not None else None
            saenb = sae_next[start:end] if sae_next is not None else None

            for tk in list(active_keys):
                method_key, hdim, save_dir = tk
                model = predictors_map[tk]
                model.train()
                if tk in projectors_map:
                    projectors_map[tk].train()

                if method_key == "autonomous":
                    x, y = zb, znb
                elif method_key == "controlled":
                    u = projectors_map[tk](teb)
                    x, y = torch.cat([zb, u], dim=-1), znb
                elif method_key == "hh":
                    x, y = hb, hnb
                elif method_key == "pca":
                    cfg_ = train_key_cfg[tk]
                    pca_dim_ = cfg_.closure.pca_dim
                    pca_key_ = (str(save_dir.parent.parent), pca_dim_)
                    pz_t, pz_n = pca_proj_cache[pca_key_]
                    x, y = pz_t[start:end], pz_n[start:end]
                elif method_key == "sae_autonomous":
                    x, y = saetb, saenb
                elif method_key == "sae_controlled":
                    u = projectors_map[tk](teb)
                    x, y = torch.cat([saetb, u], dim=-1), saenb
                else:
                    continue

                pred_out = model(x)
                loss = compute_loss(pred_out, y, loss_type)
                optimizers_map[tk].zero_grad()
                loss.backward()
                optimizers_map[tk].step()

                train_loss_accum[tk][0] += loss.item()
                train_loss_accum[tk][1] += 1

        # Periodic validation
        if step % eval_every == 0:
            for tk in list(active_keys):
                method_key, hdim, save_dir = tk
                s, c = train_loss_accum[tk]
                avg_train = s / max(c, 1)
                all_train_losses_rec[tk].append(avg_train)
                train_loss_accum[tk] = [0.0, 0]

                predictors_map[tk].eval()
                if tk in projectors_map:
                    projectors_map[tk].eval()

                with torch.no_grad():
                    val_key = (method_key, save_dir)
                    if val_key in val_ctrl:
                        vz, vh, vte, vy = val_ctrl[val_key]
                        vds = TensorDataset(vz, vh, vte, vy)
                        vldr = DataLoader(vds, batch_size=closure_batch_size, shuffle=False)
                        val_loss_sum, val_n = 0.0, 0
                        ss_res_sum = 0.0
                        tgt_sum = tgt_sq_sum = None
                        for vzb, vhb, vteb, vyb in vldr:
                            vzb = vzb.to(args.device)
                            vhb = vhb.to(args.device)
                            vteb = vteb.to(args.device)
                            vyb = vyb.to(args.device)
                            vu = projectors_map[tk](vteb)
                            vx = torch.cat([vzb, vu], dim=-1)
                            vpred = predictors_map[tk](vx)
                            val_loss_sum += compute_loss(vpred, vyb, loss_type).item() * vzb.shape[0]
                            val_n += vzb.shape[0]
                            ss_res_sum += (vpred - vyb).cpu().pow(2).sum().item()
                            vyb_c = vyb.cpu()
                            if tgt_sum is None:
                                tgt_sum = vyb_c.sum(dim=0)
                                tgt_sq_sum = vyb_c.pow(2).sum(dim=0)
                            else:
                                tgt_sum += vyb_c.sum(dim=0)
                                tgt_sq_sum += vyb_c.pow(2).sum(dim=0)
                    else:
                        vx_t, vy_t = val_simple[val_key]
                        vds = TensorDataset(vx_t, vy_t)
                        vldr = DataLoader(vds, batch_size=closure_batch_size, shuffle=False)
                        val_loss_sum, val_n = 0.0, 0
                        ss_res_sum = 0.0
                        tgt_sum = tgt_sq_sum = None
                        for vxb, vyb in vldr:
                            vxb = vxb.to(args.device)
                            vyb = vyb.to(args.device)
                            vpred = predictors_map[tk](vxb)
                            val_loss_sum += compute_loss(vpred, vyb, loss_type).item() * vxb.shape[0]
                            val_n += vxb.shape[0]
                            ss_res_sum += (vpred - vyb).cpu().pow(2).sum().item()
                            vyb_c = vyb.cpu()
                            if tgt_sum is None:
                                tgt_sum = vyb_c.sum(dim=0)
                                tgt_sq_sum = vyb_c.pow(2).sum(dim=0)
                            else:
                                tgt_sum += vyb_c.sum(dim=0)
                                tgt_sq_sum += vyb_c.pow(2).sum(dim=0)

                    val_loss = val_loss_sum / max(val_n, 1)
                    val_r2 = compute_r2_streaming(ss_res_sum, tgt_sum, tgt_sq_sum, val_n)

                all_val_losses_rec[tk].append(val_loss)
                all_val_r2_rec[tk].append(val_r2)

                if val_loss < best_val_losses[tk] - min_delta:
                    best_val_losses[tk] = val_loss
                    best_states[tk] = {k: v.clone() for k, v in predictors_map[tk].state_dict().items()}
                    if tk in projectors_map:
                        best_proj_states[tk] = {k: v.clone() for k, v in projectors_map[tk].state_dict().items()}
                    evals_without_improvement[tk] = 0
                else:
                    evals_without_improvement[tk] += 1

                if evals_without_improvement[tk] >= patience_map[tk]:
                    active_keys.discard(tk)
                    print(f"  {method_key} hdim={hdim}: early stop @ step {step}")

            if active_keys:
                best_tk = min(active_keys, key=lambda k: best_val_losses[k])
                r2_list = all_val_r2_rec[best_tk]
                pbar.set_postfix(
                    active=len(active_keys),
                    best_R2=f"{r2_list[-1]:.4f}" if r2_list else "?",
                )

    pbar.close()

    # Restore best states
    for tk in all_train_keys:
        if best_states[tk] is not None:
            predictors_map[tk].load_state_dict(best_states[tk])
    for tk in projectors_map:
        if best_proj_states.get(tk) is not None:
            projectors_map[tk].load_state_dict(best_proj_states[tk])

    # ------------------------------------------------------------------ #
    # 10. Print final R²
    # ------------------------------------------------------------------ #
    print("\nFinal R² results:")
    for tk in all_train_keys:
        method_key, hdim, save_dir = tk
        r2_list = all_val_r2_rec[tk]
        r2_final = r2_list[-1] if r2_list else None
        if r2_final is not None:
            print(f"  {method_key} hdim={hdim}: val_R²={r2_final:.4f}")
        else:
            print(f"  {method_key} hdim={hdim}: no eval")
    for (mk, hdim_), r2 in cached_r2.items():
        if r2 is not None:
            print(f"  {mk} hdim={hdim_}: val_R²={r2:.4f} (cached)")

    # ------------------------------------------------------------------ #
    # 11. Save per-save_dir
    # ------------------------------------------------------------------ #
    # Group all training keys by save_dir
    save_dir_to_keys: dict[Path, list[tuple[str, int, Path]]] = {}
    for tk in all_train_keys:
        method_key, hdim, save_dir = tk
        save_dir_to_keys.setdefault(save_dir, []).append(tk)

    for save_dir, tks in save_dir_to_keys.items():
        # All keys in the same save_dir share the same config and hdim
        _, hdim, _ = tks[0]
        cfg = train_key_cfg[tks[0]]
        ccfg = cfg.closure
        variant = ccfg.variant
        output_dir = _resolve_output_dir(cfg)

        save_dir.mkdir(parents=True, exist_ok=True)

        # Determine the target variance (same for all keys in this dir)
        tv_key = (tks[0][0], save_dir)
        tv = target_variances.get(tv_key)

        # Build trained dict
        trained: dict[str, dict] = {}
        for tk in tks:
            method_key = tk[0]
            in_dim, out_dim = _dims_for_key(method_key, cfg)
            val_r2_final = all_val_r2_rec[tk][-1] if all_val_r2_rec[tk] else None
            # Use canonical name for the predictor file (drop "sae_" prefix for file)
            save_name = method_key.replace("sae_", "")  # "sae_autonomous" -> "autonomous"
            trained[save_name] = {
                "model": predictors_map[tk],
                "input_dim": in_dim,
                "output_dim": out_dim,
                "loss_type": loss_type,
                "hidden_dim": hdim,
                "num_layers": ccfg.num_layers,
                "train_losses": all_train_losses_rec[tk],
                "val_losses": all_val_losses_rec[tk],
                "val_r2_final": val_r2_final,
            }

        save_closure_predictors(trained, save_dir / "predictors.pt", target_variance=tv)

        # Save token projector if controlled variant present
        ctrl_key_map = {"controlled": ("controlled",), "sae_controlled": ("sae_controlled",)}
        for tk in tks:
            if tk in projectors_map:
                method_key = tk[0]
                save_name = method_key.replace("sae_", "")
                torch.save({
                    save_name: projectors_map[tk].state_dict(),
                    "d_h": d_h,
                    "control_dim": ccfg.control_dim,
                }, save_dir / "token_projector.pt")
                print(f"Saved token projector to {save_dir / 'token_projector.pt'}")
                break

        # Save closure_config.json (for cache invalidation)
        with open(save_dir / "closure_config.json", "w") as f:
            json.dump(_hparams_from_config(cfg, hdim), f, indent=2)

        # Save metadata
        meta: dict = {
            "variant": variant,
            "llm_name": cfg.llm.name,
            "llm_layer": cfg.llm.layer_index,
            "d_h": d_h,
            "is_controlled": any(mk in CTRL_KEYS for mk, _, _ in tks),
            "variants_trained": [k for k in trained],
            "closure_hidden_dim": hdim,
            "closure_num_layers": ccfg.num_layers,
            "num_val_pairs": N_val,
            "max_steps": ccfg.max_steps,
            "patience": ccfg.patience,
            "seed": args.seed,
            "loss_type": loss_type,
            **_data_meta(cfg),
        }
        if variant == "z_ret":
            meta["z_dim"] = d_z
            meta["control_dim"] = ccfg.control_dim
        elif variant == "h_pca":
            pca_dim = ccfg.pca_dim
            pca_key = (str(output_dir), pca_dim)
            meta["pca_dim"] = pca_dim
            meta["pca_explained_variance"] = pca_explained.get(pca_key)
        elif variant == "sae":
            meta["z_dim"] = d_sae
            meta["control_dim"] = ccfg.control_dim

        with open(save_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        # Training curves
        curves = {
            save_name_: {
                "train_losses": all_train_losses_rec[tk_],
                "val_losses": all_val_losses_rec[tk_],
            }
            for tk_ in tks
            for save_name_ in [tk_[0].replace("sae_", "")]
        }
        plot_training_curves(curves, save_dir / "training_curves.png")

        # Save config under closure/ subdir (avoid polluting baseline/RET root)
        closure_cfg_dir = output_dir / closure_subdir
        closure_cfg_dir.mkdir(parents=True, exist_ok=True)
        save_config(cfg, closure_cfg_dir / "config.yaml")

        print(f"Saved {variant} hdim={hdim} -> {save_dir}")

    extractor.cleanup()
    print("\nClosure predictor training complete.")


# ---------------------------------------------------------------------------
# Summarize-only mode
# ---------------------------------------------------------------------------

def run_summarize_only(args: argparse.Namespace) -> None:
    """Read existing sweep results from all config output_dirs and print summary."""
    overlay_yaml = args.config_overlay
    closure_subdir = "closure_wo_system_prompt" if args.exclude_scaffold else "closure"

    print("Scanning for sweep results ...")
    all_results: dict[str, dict[str, float | None]] = {}

    for cfg_path in args.configs:
        cfg = load_config_with_overlay(cfg_path, overlay_yaml)
        output_dir = _resolve_output_dir(cfg)
        variant = cfg.closure.variant or "unknown"

        closure_root = output_dir / closure_subdir
        results = _discover_sweep_results(closure_root, closure_name=None)
        if results:
            # Prefix keys with variant to avoid collisions
            for key, r2_dict in results.items():
                label = f"{variant}/{key}"
                all_results[label] = r2_dict
            print(f"  {variant} @ {output_dir}: found variants {list(results.keys())}")
        else:
            print(f"  {variant} @ {output_dir}: no results found")

    if not all_results:
        print("\nNo sweep results found.")
        return

    print(f"\n=== Capacity Sweep Summary ===")
    _print_sweep_summary(all_results)

    # Add _best_hdim and _best_r2 per variant
    for key, r2_dict in all_results.items():
        hdims = {k: v for k, v in r2_dict.items() if not k.startswith("_") and v is not None}
        if hdims:
            best_k = max(hdims, key=hdims.get)
            r2_dict["_best_hdim"] = int(best_k)
            r2_dict["_best_r2"] = hdims[best_k]

    # Determine output directory
    if args.output_dir:
        summary_dir = Path(args.output_dir)
    else:
        # Default: parent of first config's output_dir / closure_analysis
        first_cfg = load_config_with_overlay(args.configs[0], args.config_overlay)
        first_output_dir = _resolve_output_dir(first_cfg)
        summary_dir = first_output_dir.parent / "closure_analysis"
    summary_dir.mkdir(parents=True, exist_ok=True)

    # Save sweep summary JSON
    summary_path = summary_dir / "capacity_sweep_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved sweep summary to {summary_path}")

    # Auto-generate sweep plots
    try:
        from scripts.closure.plot_sweep import plot_best_r2_bar, plot_capacity_sweep
        plot_best_r2_bar(all_results, summary_dir / "best_r2_per_variant.png")
        plot_capacity_sweep(all_results, summary_dir / "capacity_sweep.png")
        print(f"Saved sweep plots to {summary_dir}")
    except Exception as e:
        print(f"Warning: Could not generate sweep plots: {e}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train closure predictors from one or more config files."
    )
    parser.add_argument(
        "configs", nargs="+",
        help="Config YAML files (one per variant). All configs must share the same llm/data settings.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Retrain even if predictors already exist.",
    )
    parser.add_argument(
        "--config-overlay", type=str, default=None,
        help="Optional YAML overlay merged on top of each config.",
    )
    parser.add_argument(
        "--summarize-only", action="store_true",
        help="Skip training; read existing predictors.pt files and print summary table.",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for sweep summary and plots (used with --summarize-only).",
    )
    parser.add_argument(
        "--exclude-scaffold", action="store_true",
        help="Exclude scaffold/system-prompt prefix tokens from closure R² computation. "
             "Results saved to closure_wo_system_prompt/ subdirs.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.summarize_only:
        run_summarize_only(args)
        return

    run_config_driven(args)
    # Auto-summarize after training
    run_summarize_only(args)


if __name__ == "__main__":
    main()
