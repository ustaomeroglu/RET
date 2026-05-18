"""
Forward-hook utilities for supervised activation steering.

``SteeringHook`` adds a fixed vector ``alpha * w_hat`` to the output of a
single transformer block at every forward pass. It is dataset-agnostic and
works with any HF model architecture whose layer modules return either a
Tensor ``[B, S, D]`` or a tuple ``(hidden_states, ...)`` (LLaMA/Qwen/Mistral/
GPT-OSS/GPTNeoX/GPT-2 all do).
"""
from __future__ import annotations

from typing import Optional

import torch

from evaluation.probing.model_utils import _get_layer_from, _unwrap


class SteeringHook:
    """Context manager that adds ``alpha * w_hat`` to the output of a single
    transformer layer at every forward pass.

    Usage
    -----
    >>> with SteeringHook(layer_module, w_raw, alpha, device, dtype):
    ...     out = model.generate(...)

    Parameters
    ----------
    layer_module : the ``nn.Module`` to attach the hook to (e.g. a transformer
                   block whose output is the residual stream entering the next
                   block).
    w_hat        : direction tensor ``[D]``; automatically unit-normalized at
                   construction so ``alpha`` is in absolute units of ``||h||``.
    alpha        : steering coefficient (signed scalar). When exactly 0.0, the
                   hook is a runtime no-op so ``alpha=0`` reproduces the
                   unsteered baseline byte-for-byte.
    device       : device to cast ``w_hat`` to (should match the layer's
                   parameter device — with ``device_map='auto'`` this may
                   NOT be the same as the model's embedding device).
    dtype        : dtype to cast ``w_hat`` to (typically ``torch.bfloat16``).
    """

    def __init__(
        self,
        layer_module: torch.nn.Module,
        w_hat: torch.Tensor,
        alpha: float,
        device: torch.device | str,
        dtype: torch.dtype,
    ):
        self.layer_module = layer_module
        self.alpha = float(alpha)
        norm = w_hat.float().norm().item()
        if norm < 1e-12:
            raise ValueError("Steering direction has near-zero norm; cannot normalize")
        self.w_unit = (
            (w_hat.float() / norm).to(device=device, dtype=dtype).contiguous()
        )  # [D]
        self.handle: Optional[torch.utils.hooks.RemovableHandle] = None

    def _hook_fn(self, module, module_input, module_output):
        # alpha=0 → true no-op path (bit-exact baseline reproduction).
        if self.alpha == 0.0:
            return module_output

        if isinstance(module_output, tuple):
            h = module_output[0]
            h_new = h + self.alpha * self.w_unit  # broadcast over [B, S, D]
            return (h_new,) + module_output[1:]
        else:
            return module_output + self.alpha * self.w_unit

    def __enter__(self) -> "SteeringHook":
        self.handle = self.layer_module.register_forward_hook(self._hook_fn)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


def get_layer_module(model, layer_idx: int) -> torch.nn.Module:
    """Return the transformer block at ``layer_idx`` for steering hook attachment.

    Thin wrapper that unwraps multi-GPU shells (``MultiGPURunner``,
    ``DataParallel``) and dispatches to the existing architecture-aware
    helper in ``evaluation.probing.model_utils``. Supports LLaMA/Mistral/
    Qwen/GPT-OSS (``model.model.layers``), GPTNeoX/Pythia
    (``model.gpt_neox.layers``), and GPT-2/GPT-J (``model.transformer.h``).
    """
    return _get_layer_from(_unwrap(model), layer_idx)
