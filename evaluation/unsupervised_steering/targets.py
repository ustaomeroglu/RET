"""Load clustering artifacts and compute steering targets."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluation.unsupervised_steering.config import SteeringConfig, SteeringRule

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Steering context (holds all loaded artifacts)
# ---------------------------------------------------------------------------


@dataclass
class SteeringContext:
    model: AutoModelForCausalLM
    tokenizer: AutoTokenizer
    encoder: nn.Module
    predictor: nn.Module | None
    centers: torch.Tensor          # (K, d_z) in preprocessed space
    centers_normed: torch.Tensor   # (K, d_z) L2-normalized for cosine sim
    mu: torch.Tensor               # (d_z,) centering vector
    group_labels: torch.Tensor     # (K,) cluster -> group mapping
    group_to_clusters: dict[int, list[int]]
    layer_idx: int                 # control layer (= max(layer_indices))
    layers: nn.ModuleList
    device: str
    # Multi-layer support: list of all extracted layer indices.
    # For single-layer configs, layer_indices = [layer_idx].
    # For multi-layer, layer_indices contains all layers and layer_idx is the
    # control layer (the last one to fire in the forward pass, chosen for
    # steering).
    layer_indices: list[int] = None  # type: ignore[assignment]
    hidden_dim: int = 0              # per-layer hidden dim (d_h)

    @property
    def is_multi_layer(self) -> bool:
        return len(self.layer_indices) > 1

    @property
    def concat_dim(self) -> int:
        return len(self.layer_indices) * self.hidden_dim


def load_steering_context(config: SteeringConfig) -> SteeringContext:
    """Load all artifacts needed for steering from clustering outputs."""
    clustering_dir = Path(config.clustering_dir)
    meta_path = clustering_dir / "clustering_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"clustering_meta.json not found in {clustering_dir}"
        )

    with open(meta_path) as f:
        meta = json.load(f)

    # -- Load RET config + encoder + predictor ---------------------------------
    from ret.config import load_config
    from ret.utils.checkpointing import load_encoder, load_checkpoint

    ret_config_path = meta["config_yaml_path"]
    ret_cfg = load_config(ret_config_path)

    ckpt_path = Path(meta["checkpoint_path"])
    if not ckpt_path.is_absolute():
        # Resolve relative to project root (common convention)
        project_root = Path(__file__).resolve().parents[2]
        ckpt_path = project_root / ckpt_path

    encoder, ret_cfg, ckpt = load_encoder(
        str(ckpt_path), device=config.device, config=ret_cfg,
    )

    predictor = None
    if config.horizon > 0 and "predictor" in ckpt:
        from ret.models.predictor import Predictor

        predictor = Predictor(ret_cfg.encoder.z_dim, ret_cfg.predictor)
        predictor.load_state_dict(ckpt["predictor"])
        predictor.to(config.device).eval()
        logger.info("Loaded predictor for MPPI rollouts (horizon=%d)", config.horizon)

    # -- Load clustering artifacts ---------------------------------------------
    w_dir = clustering_dir / f"w{config.window_size}"
    if not w_dir.exists():
        raise FileNotFoundError(
            f"Window dir {w_dir} not found. Steering requires w{config.window_size}/ artifacts."
        )

    mu_path = w_dir / "mu.npy"
    if not mu_path.exists():
        raise FileNotFoundError(f"mu.npy not found in {w_dir}")
    mu = torch.from_numpy(np.load(mu_path)).float().to(config.device)

    seed_dir = w_dir / f"K{config.K}" / f"seed{config.seed}"
    centers_np = np.load(seed_dir / "centers.npy")
    centers = torch.from_numpy(centers_np).float().to(config.device)
    centers_normed = centers / centers.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    # Group labels from hierarchy
    analysis_dir = seed_dir / "analysis"
    tau_dir = analysis_dir / f"weighted_tau{config.tau}"
    grp_dir = tau_dir / f"grouped_alpha{config.alpha}"
    group_labels_np = np.load(grp_dir / "group_labels.npy")
    group_labels = torch.from_numpy(group_labels_np).long().to(config.device)

    with open(grp_dir / "hierarchy.json") as f:
        hierarchy = json.load(f)
    group_to_clusters = {
        int(k): v for k, v in hierarchy["groups"].items()
    }

    # -- Load LLM + tokenizer --------------------------------------------------
    from ret.extraction.hooks import _find_layers_and_embed

    llm_cfg = ret_cfg.llm
    dtype = getattr(torch, llm_cfg.dtype)
    load_kwargs: dict = dict(torch_dtype=dtype)
    if llm_cfg.trust_remote_code:
        load_kwargs["trust_remote_code"] = True

    if llm_cfg.device_map is not None:
        load_kwargs["device_map"] = llm_cfg.device_map
        model = AutoModelForCausalLM.from_pretrained(llm_cfg.name, **load_kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(llm_cfg.name, **load_kwargs)
        model.to(config.device)

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    tokenizer_kwargs: dict = {}
    if llm_cfg.trust_remote_code:
        tokenizer_kwargs["trust_remote_code"] = True
    tokenizer = AutoTokenizer.from_pretrained(llm_cfg.name, **tokenizer_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    layers, _embed = _find_layers_and_embed(model)

    # Resolve layer_index (int, list[int], or "all") into layer_indices list
    raw_layer_index = llm_cfg.layer_index
    if isinstance(raw_layer_index, str):
        if raw_layer_index == "all":
            layer_indices = list(range(len(layers)))
        else:
            raise ValueError(
                f"Unsupported layer_index string: {raw_layer_index!r} "
                f"(only 'all' is supported)"
            )
    elif isinstance(raw_layer_index, (list, tuple)):
        layer_indices = [int(x) for x in raw_layer_index]
    else:
        layer_indices = [int(raw_layer_index)]

    # Control layer: the one that fires LAST in the forward pass (max index)
    # so that by the time its hook runs, all earlier passive hooks have
    # already collected h for the current step.
    layer_idx = max(layer_indices)

    logger.info(
        "Loaded: LLM=%s layer_indices=%s control_layer=%d, "
        "encoder z_dim=%d, K=%d, %d groups",
        llm_cfg.name, layer_indices, layer_idx, ret_cfg.encoder.z_dim, config.K,
        len(group_to_clusters),
    )

    return SteeringContext(
        model=model,
        tokenizer=tokenizer,
        encoder=encoder,
        predictor=predictor,
        centers=centers,
        centers_normed=centers_normed,
        mu=mu,
        group_labels=group_labels,
        group_to_clusters=group_to_clusters,
        layer_idx=layer_idx,
        layers=layers,
        device=config.device,
        layer_indices=layer_indices,
        hidden_dim=llm_cfg.hidden_dim,
    )


# ---------------------------------------------------------------------------
# Preprocessing & target computation (PyTorch, on-device)
# ---------------------------------------------------------------------------


def preprocess_z_torch(
    z_raw: torch.Tensor,
    mu: torch.Tensor,
    normalize: bool = True,
) -> torch.Tensor:
    """Center (and optionally L2-normalize) z, matching StreamingPreprocessor.

    Args:
        z_raw: (..., d_z) raw encoder output
        mu: (d_z,) centering vector
        normalize: if True (default), L2-normalize after centering (matches
            the cosine-clustering pipeline). If False, return only the
            centered vector (matches the centered-only / Euclidean pipeline).

    Returns:
        z_proc: (..., d_z) on the unit hypersphere (normalize=True) or
                centered-only (normalize=False).
    """
    z_centered = z_raw - mu
    if not normalize:
        return z_centered
    return z_centered / z_centered.norm(dim=-1, keepdim=True).clamp(min=1e-8)


def assign_cluster_torch(
    z_proc: torch.Tensor,
    centers_normed: torch.Tensor,
    group_labels: torch.Tensor,
    metric: str = "cosine",
    centers_raw: torch.Tensor | None = None,
) -> tuple[int, int, float]:
    """Assign a single z_proc vector to its nearest cluster and group.

    Args:
        z_proc: (1, d_z) or (d_z,) preprocessed z (unit-norm for cosine,
            centered-only for euclidean).
        centers_normed: (K, d_z) L2-normalized cluster centers (used for
            cosine).
        group_labels: (K,) cluster -> group mapping
        metric: "cosine" (default) or "euclidean".
        centers_raw: (K, d_z) raw cluster centers (required for euclidean).

    Returns:
        (cluster_id, group_id, score) where score is cosine similarity for
        cosine metric, or negative Euclidean distance for euclidean metric
        (so that "higher score = closer" in both cases).
    """
    z = z_proc.squeeze(0) if z_proc.dim() == 2 else z_proc
    if metric == "euclidean":
        if centers_raw is None:
            raise ValueError("euclidean metric requires centers_raw")
        # Euclidean distance to each center; argmin = closest cluster.
        diffs = centers_raw - z.unsqueeze(0)  # (K, d_z)
        dists = diffs.norm(dim=-1)            # (K,)
        cluster_id = dists.argmin().item()
        score = -dists[cluster_id].item()
        return cluster_id, group_labels[cluster_id].item(), score
    # Cosine
    sims = z @ centers_normed.T  # (K,)
    cluster_id = sims.argmax().item()
    return cluster_id, group_labels[cluster_id].item(), sims[cluster_id].item()


def compute_target_center(
    centers: torch.Tensor,
    group_labels: torch.Tensor,
    group_to_clusters: dict[int, list[int]],
    rule: SteeringRule,
    normalize: bool = True,
) -> torch.Tensor:
    """Compute the target vector in preprocessed space for a steering rule.

    Args:
        normalize: if True (default), return an L2-normalized target (cosine
            mode). If False, return the raw target (Euclidean mode), where the
            unnormalized cluster center / group centroid lives in the same
            space as a centered-only z.
    """
    if rule.target_cluster is not None:
        c = centers[rule.target_cluster]
        if not normalize:
            return c
        return c / c.norm().clamp(min=1e-8)
    # Group centroid: average of member cluster centers
    member_ids = group_to_clusters[rule.target_group]
    member_centers = centers[member_ids]  # (M, d_z)
    centroid = member_centers.mean(dim=0)
    if not normalize:
        return centroid
    return centroid / centroid.norm().clamp(min=1e-8)
