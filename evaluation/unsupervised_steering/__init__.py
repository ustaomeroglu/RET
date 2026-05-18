"""Unsupervised steering of LLM generation via RET group trajectories.

Public entries:
- :func:`run_unsupervised_steering` — run one steered (or baseline) generation.
- :func:`attach_steering_hook` — hook factory (single- or multi-layer).
- :func:`build_rules_for_condition` — short condition specs to ``SteeringRule`` lists.
"""

from evaluation.unsupervised_steering.conditions import build_rules_for_condition
from evaluation.unsupervised_steering.hooks import attach_steering_hook
from evaluation.unsupervised_steering.pipeline import run_unsupervised_steering

__all__ = [
    "run_unsupervised_steering",
    "attach_steering_hook",
    "build_rules_for_condition",
]
