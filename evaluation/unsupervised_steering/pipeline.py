"""End-to-end unsupervised steering pipeline.

The single library entry is :func:`run_unsupervised_steering`. It takes a
``SteeringConfig`` plus optional pre-loaded ``ctx`` (from
``targets.load_steering_context``) and optional ``regularizer`` (a closure
predictor for OOD-aware steering). Returns ``{"text", "diagnostics"}`` and
writes outputs under ``config.output_dir/``.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import yaml

from evaluation.unsupervised_steering.config import SteeringConfig
from evaluation.unsupervised_steering.generation import SteeredGenerator
from evaluation.unsupervised_steering.hooks import attach_steering_hook
from evaluation.unsupervised_steering.mppi import GradientSteerer
from evaluation.unsupervised_steering.plotting.html import save_steered_html
from evaluation.unsupervised_steering.targets import load_steering_context

logger = logging.getLogger(__name__)


def run_unsupervised_steering(
    config: SteeringConfig,
    *,
    ctx=None,
    regularizer=None,
) -> dict:
    """Run one steered (or baseline) generation.

    Parameters
    ----------
    config : SteeringConfig
        Run specification (model/clustering, prompt, rules, strength, …).
    ctx : SteeringContext, optional
        Pre-loaded context. If ``None``, loaded from ``config.clustering_dir``.
        Pass a shared ``ctx`` when sweeping multiple problems so the LLM and
        clustering artifacts are loaded once.
    regularizer : dict, optional
        Optional ``{"predictor": nn.Module}`` for closure-predictor OOD
        regularization (used when ``config.lambda_pred > 0``).

    Returns
    -------
    dict
        ``{"text": str, "diagnostics": list[dict]}``.
    """
    if ctx is None:
        ctx = load_steering_context(config)

    # hidden_dim comes from the RET config tied to the clustering checkpoint
    from ret.config import load_config
    meta_path = Path(config.clustering_dir) / "clustering_meta.json"
    ret_cfg = load_config(json.loads(meta_path.read_text())["config_yaml_path"])
    d_h = ret_cfg.llm.hidden_dim

    optimizer = GradientSteerer(
        strength=config.strength,
        gradient_steps=config.gradient_steps,
        d_h=d_h,
        device=config.device,
        lambda_pred=config.lambda_pred,
        epsilon_pred=config.epsilon_pred,
        normalize_z=(config.metric == "cosine"),
    )

    hook, handles = attach_steering_hook(ctx, config.rules, optimizer, config.device,
                                          metric=config.metric)
    if regularizer is not None and config.lambda_pred > 0.0:
        hook.closure_predictor = regularizer["predictor"]

    try:
        generator = SteeredGenerator(ctx.model, ctx.tokenizer, hook)
        text, diagnostics = generator.generate(
            config.prompt,
            config.max_new_tokens,
            config.temperature,
            config.top_p,
        )
    finally:
        for h in handles:
            h.remove()

    if config.output_dir is not None:
        _save_outputs(config, ctx, hook, text, diagnostics)

    return {"text": text, "diagnostics": diagnostics}


def _save_outputs(config, ctx, hook, text, diagnostics) -> None:
    """Persist a run's outputs: text + diagnostics + z-trajectory + HTML."""
    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    (out / "steered_output.txt").write_text(text, encoding="utf-8")
    with open(out / "diagnostics.jsonl", "w") as f:
        for d in diagnostics:
            f.write(json.dumps(d) + "\n")

    prompt_z = getattr(hook, "prompt_z_features", []) or []
    if prompt_z or hook.z_features:
        z_full = torch.stack(list(prompt_z) + list(hook.z_features))
        np.save(out / "z_trajectory.npy", z_full.numpy())
        (out / "prompt_len.txt").write_text(str(len(prompt_z)))

    prompt_z_raw = getattr(hook, "prompt_z_features_raw", []) or []
    z_features_raw = getattr(hook, "z_features_raw", []) or []
    if prompt_z_raw or z_features_raw:
        z_full_raw = torch.stack(list(prompt_z_raw) + list(z_features_raw))
        np.save(out / "z_trajectory_raw.npy", z_full_raw.numpy())

    with open(out / "config.yaml", "w") as f:
        yaml.dump(asdict(config), f, default_flow_style=False)

    try:
        save_steered_html(config, ctx, hook, diagnostics, out)
    except Exception as e:
        logger.warning("HTML render failed for %s: %s", out, e)

    logger.info("Results saved to %s", out)
