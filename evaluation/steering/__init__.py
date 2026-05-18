"""
Supervised activation-steering library — reusable across datasets.

Public API:

- ``load_probe_direction(path, probe_type)``: load a binary probe checkpoint
  (linear or DoM) and return its direction vector.
- ``SteeringHook(layer_module, w_hat, alpha, device, dtype)``: context
  manager that adds ``alpha * w_hat`` to a layer's output at every forward
  pass.
- ``get_layer_module(model, layer_idx)``: architecture-aware layer fetch.
- ``projection_stats(direction, token_blocks, quantile=0.99)``: compute
  reference projection magnitudes (``m_max``) from an iterable of token
  batches.

Dataset-specific wrappers (e.g. sycophancy-specific m_max computation that
filters on role_id / turn_id from the sycophancy memmap schema) live in
``evaluation/datasets/<dataset>/steering.py`` and import from here.
"""
from evaluation.steering.directions import load_probe_direction
from evaluation.steering.hooks import SteeringHook, get_layer_module
from evaluation.steering.stats import projection_stats
from evaluation.steering.reward_gradient import RewardGradientHook

__all__ = [
    "load_probe_direction",
    "SteeringHook",
    "get_layer_module",
    "projection_stats",
    "RewardGradientHook",
]
