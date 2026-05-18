"""
Probe-direction loading for supervised activation steering.

Loads the weight vector from a trained binary probe checkpoint so it can be
used as a steering direction (adds ``alpha * w_hat`` to a residual stream).

This module is dataset-agnostic — it works with any probe trained via
``evaluation.probing.baselines.{linear_probe, dom_probe}``, regardless of
which dataset / experiment produced it.

Sign convention (shared by both probe types in this repo):
    A positive projection of ``h`` onto the returned direction corresponds
    to class label = 1. Adding ``+alpha * w_hat`` pushes the model toward
    class 1; adding ``-alpha * w_hat`` pushes it toward class 0.
"""
from __future__ import annotations

from typing import Literal

import torch


def load_probe_direction(
    probe_path: str,
    probe_type: Literal["linear", "dom"],
) -> torch.Tensor:
    """Load a binary probe checkpoint and return the raw direction vector.

    Parameters
    ----------
    probe_path : path to ``model_{mode}.pt`` saved by LinearProbe or DOMProbe.
    probe_type : "linear" or "dom".

    Returns
    -------
    w : torch.Tensor of shape ``[D]``  (CPU, float32). Not unit-normalized —
        the caller is responsible for normalization (``SteeringHook`` does it).
    """
    ckpt = torch.load(probe_path, map_location="cpu", weights_only=False)

    if probe_type == "linear":
        task_type = ckpt.get("task_type", "binary")
        if task_type != "binary":
            raise NotImplementedError(
                f"load_probe_direction only supports binary linear probes, got {task_type}"
            )
        weight = ckpt["model_state_dict"]["weight"]  # [1, D] for binary
        if weight.ndim != 2 or weight.shape[0] != 1:
            raise ValueError(
                f"Expected binary linear probe weight shape [1, D], got {tuple(weight.shape)}"
            )
        w = weight.squeeze(0).float().contiguous()  # [D]

    elif probe_type == "dom":
        task_type = ckpt.get("task_type", "binary")
        if task_type != "binary":
            raise NotImplementedError(
                f"load_probe_direction only supports binary DoM probes, got {task_type}"
            )
        if "w" not in ckpt:
            raise KeyError(
                f"DoM checkpoint at {probe_path} missing 'w' key. Found: {list(ckpt.keys())}"
            )
        w = ckpt["w"].float().contiguous()  # [D]
        if w.ndim != 1:
            raise ValueError(
                f"Expected binary DoM 'w' shape [D], got {tuple(w.shape)}"
            )

    else:
        raise ValueError(f"Unknown probe_type {probe_type!r}; expected 'linear' or 'dom'")

    return w
