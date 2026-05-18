"""Shared baseline utilities for analysis scripts.

Provides argparse helpers, mode resolution, and representation transformers
for hh / sae / pca baselines, so that probe_mmlu, tsne_mmlu,
event_structure, and analyze_embeddings can share baseline logic.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from ret.config import load_config, llm_short_name, ExperimentConfig
from ret.extraction.hooks import HiddenStateExtractor
from evaluation.baselines.sae import load_sae_baseline, encode_sae


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------
BaselineMode = Literal["checkpoint", "hh", "sae", "pca"]


@dataclass
class BaselineContext:
    """Holds all state needed for baseline representation computation."""

    mode: BaselineMode
    config: ExperimentConfig
    extractor: HiddenStateExtractor | None
    output_dir: Path
    representation_label: str = ""

    # SAE fields (populated when mode == "sae")
    sae: object | None = None
    sae_dim: int | None = None

    # PCA fields (populated after fit_pca_on_h is called)
    pca_components: np.ndarray | None = None  # (pca_dim, d_h)
    pca_mean: np.ndarray | None = None  # (d_h,)
    pca_dim: int | None = None


# ---------------------------------------------------------------------------
# Argparse helpers
# ---------------------------------------------------------------------------
def add_baseline_args(
    parser: argparse.ArgumentParser,
    include_output_dir: bool = True,
) -> None:
    """Add --baseline-hh, --baseline-sae, --baseline-pca, --pca-dim to a parser.

    Optionally adds --output-dir (set include_output_dir=False if the caller
    already defines that argument).
    """
    group = parser.add_argument_group("Baseline modes")
    group.add_argument(
        "--baseline-hh",
        action="store_true",
        help="Use raw hidden states h as the representation (no encoder)",
    )
    group.add_argument(
        "--baseline-sae",
        action="store_true",
        help="Use SAE-encoded features as the representation",
    )
    group.add_argument(
        "--baseline-pca",
        action="store_true",
        help="Use PCA-projected h as the representation",
    )
    group.add_argument(
        "--pca-dim",
        type=int,
        default=64,
        help="PCA projection dimensionality (default: 64, only with --baseline-pca)",
    )
    group.add_argument(
        "--config",
        type=str,
        default=None,
        help="Config YAML for baseline modes (default: Pythia-160m layer 6)",
    )
    if include_output_dir:
        parser.add_argument(
            "--output-dir",
            type=str,
            default=None,
            help="Override output directory (results saved to <output-dir>/analysis/)",
        )


def get_baseline_mode(args: argparse.Namespace) -> BaselineMode:
    """Determine which mode is active and validate mutual exclusivity.

    Returns ``"checkpoint"`` if no baseline flag is set.
    """
    baseline_flags = sum(
        [
            getattr(args, "baseline_hh", False),
            getattr(args, "baseline_sae", False),
            getattr(args, "baseline_pca", False),
        ]
    )
    has_checkpoint = getattr(args, "checkpoint", None) is not None

    if baseline_flags > 1:
        raise SystemExit(
            "ERROR: Specify at most one baseline mode: "
            "--baseline-hh, --baseline-sae, or --baseline-pca"
        )
    if baseline_flags == 0 and not has_checkpoint:
        raise SystemExit(
            "ERROR: Must provide a checkpoint path or use one of: "
            "--baseline-hh, --baseline-sae, --baseline-pca"
        )
    if baseline_flags > 0 and has_checkpoint:
        raise SystemExit(
            "ERROR: Cannot specify both a checkpoint and a --baseline-* flag"
        )

    if getattr(args, "baseline_hh", False):
        return "hh"
    if getattr(args, "baseline_sae", False):
        return "sae"
    if getattr(args, "baseline_pca", False):
        return "pca"
    return "checkpoint"


# ---------------------------------------------------------------------------
# Baseline setup
# ---------------------------------------------------------------------------
def setup_baseline(
    mode: BaselineMode,
    args: argparse.Namespace,
    device: str = "cuda",
    skip_extractor: bool = False,
) -> BaselineContext:
    """Create extractor + load SAE if needed + resolve output directory.

    For ``"checkpoint"`` mode returns a minimal context — the caller is
    responsible for loading the encoder from the checkpoint.

    For baseline modes loads the default config (Pythia-160m, layer 6).

    When ``skip_extractor=True``, the LLM is not loaded (caller supplies h
    from elsewhere, e.g. a cache).
    """
    if mode == "checkpoint":
        ckpt_path = Path(args.checkpoint)
        ckpt_dir = ckpt_path.parent
        config = load_config(ckpt_dir / "config.yaml")
        # Skip loading the full LLM when hidden states are pre-cached
        extractor = None
        if not skip_extractor and config.data.mode != "cached":
            extractor = HiddenStateExtractor(config.llm, device=device)
        out_dir = Path(args.output_dir) if args.output_dir else ckpt_dir
        return BaselineContext(
            mode=mode,
            config=config,
            extractor=extractor,
            output_dir=out_dir / "analysis",
            representation_label="z (encoder)",
        )

    # --- Baseline modes: load config (default or user-specified) ---
    config_path = getattr(args, "config", None)
    config = load_config(yaml_path=config_path)
    extractor = None
    if not skip_extractor and config.data.mode != "cached":
        extractor = HiddenStateExtractor(config.llm, device=device)
    lsn = llm_short_name(config.llm)

    sae = None
    sae_dim = None
    pca_dim = None

    if mode == "sae":
        sae_result = load_sae_baseline(config.llm, device=device)
        if sae_result is None:
            if extractor is not None:
                extractor.cleanup()
            raise RuntimeError("Could not load SAE. Aborting.")
        sae, _, _ = sae_result
        # Probe for SAE output dimensionality
        probe_in = torch.zeros(1, config.llm.hidden_dim, device=device)
        probe_out = encode_sae(sae, probe_in)
        sae_dim = probe_out.shape[-1]
        label = "SAE"
        default_out = Path("outputs") / lsn / "baselines" / f"SAE_{lsn}_L{config.llm.layer_index}"

    elif mode == "pca":
        pca_dim = getattr(args, "pca_dim", 64)
        label = f"PCA-{pca_dim}"
        default_out = Path("outputs") / lsn / "baselines" / f"baseline_pca_{pca_dim}"

    else:  # hh
        label = "h (hidden states)"
        default_out = Path("outputs") / lsn / "baselines" / "baseline_hh"

    out_dir = Path(args.output_dir) if args.output_dir else default_out

    return BaselineContext(
        mode=mode,
        config=config,
        extractor=extractor,
        output_dir=out_dir / "analysis",
        representation_label=label,
        sae=sae,
        sae_dim=sae_dim,
        pca_dim=pca_dim,
    )


# ---------------------------------------------------------------------------
# PCA fitting
# ---------------------------------------------------------------------------
def fit_pca_on_h(
    h_flat: np.ndarray,
    pca_dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit PCA on flat hidden states, return ``(components, mean)``.

    Args:
        h_flat: ``(N, d_h)`` numpy array of hidden states.
        pca_dim: Number of PCA components.

    Returns:
        components: ``(pca_dim, d_h)`` float32 numpy array.
        mean: ``(d_h,)`` float32 numpy array.
    """
    from sklearn.decomposition import PCA

    pca = PCA(n_components=pca_dim)
    pca.fit(h_flat)
    explained = pca.explained_variance_ratio_.sum()
    print(f"  PCA fit: {pca_dim} components, explained variance = {explained:.4f}")
    return (
        pca.components_.astype(np.float32),
        pca.mean_.astype(np.float32),
    )


# ---------------------------------------------------------------------------
# h -> representation transform
# ---------------------------------------------------------------------------
def transform_h(
    h: torch.Tensor,
    ctx: BaselineContext,
    device: str = "cuda",
) -> torch.Tensor:
    """Transform hidden states *h* to the baseline representation.

    Args:
        h: ``(..., d_h)`` tensor of hidden states.
        ctx: :class:`BaselineContext` with mode and loaded SAE / PCA.
        device: Device for computation.

    Returns:
        ``(..., d_repr)`` tensor on CPU.
    """
    if ctx.mode == "hh":
        return h

    if ctx.mode == "sae":
        return encode_sae(ctx.sae, h.to(device))

    if ctx.mode == "pca":
        if ctx.pca_components is None or ctx.pca_mean is None:
            raise RuntimeError(
                "PCA components not fitted yet — call fit_pca_on_h first"
            )
        comp = torch.from_numpy(ctx.pca_components).to(h.device)  # (pca_dim, d_h)
        mean = torch.from_numpy(ctx.pca_mean).to(h.device)  # (d_h,)
        return (h - mean) @ comp.T

    raise ValueError(f"transform_h called with mode={ctx.mode!r}")
