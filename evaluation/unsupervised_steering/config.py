"""Configuration dataclasses for unsupervised steering."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SteeringRule:
    """One event-driven steering instruction.

    Three modes (exactly one must be used):

    * **Attractor — group-dwell trigger** (legacy): set ``trigger_group``
      and ``trigger_dwell``. Rule fires once the model has been in
      ``trigger_group`` for at least ``trigger_dwell`` consecutive tokens,
      then steers toward ``target_group``/``target_cluster`` for
      ``duration`` tokens.

    * **Attractor — position trigger**: set ``trigger_position`` to a
      non-negative integer N. Rule fires on generation step N and steers
      toward ``target_group``/``target_cluster`` for ``duration`` tokens.

    * **Margin-gated repulsion (cluster or group)**: set exactly one of
      ``avoid_cluster`` / ``avoid_group``, plus ``avoid_margin_threshold``.
      The rule watches every step. The gate is "open" whenever
      ``cos(z, c_current) − cos(z, c_avoid) ≤ avoid_margin_threshold``,
      where ``c_current`` is the argmax cluster's center for
      ``avoid_cluster`` rules and the centroid of the group containing the
      argmax cluster for ``avoid_group`` rules. The margin is ≥ 0 by
      construction (current_cluster is the argmax); a small margin means
      z is leaning toward the avoid target. While the gate is open,
      repulsion fires each step. ``avoid_sticky_duration`` controls how
      many additional tokens repulsion keeps firing after the gate
      closes (default 1 = stop immediately). Optional
      ``avoid_max_tokens`` caps the total number of gate-open events
      across the run; optional ``avoid_from_step`` delays gate activation
      until after that generation step.

      Repulsion direction is pure: displacement = ``−c_avoid`` (no
      anchor to ``c_current``).
    """

    # Attractor — triggers
    trigger_group: int | None = None
    trigger_dwell: int | None = None
    trigger_position: int | None = None
    # Attractor — target
    target_group: int | None = None
    target_cluster: int | None = None
    duration: int = 20
    # Repulsion — margin-gated, always-on watcher
    avoid_cluster: int | None = None
    avoid_group: int | None = None
    avoid_margin_threshold: float | None = None
    avoid_sticky_duration: int = 1
    avoid_max_tokens: int | None = None
    avoid_from_step: int | None = None

    def __post_init__(self):
        is_avoid = (
            self.avoid_cluster is not None or self.avoid_group is not None
        )

        if is_avoid:
            if self.avoid_cluster is not None and self.avoid_group is not None:
                raise ValueError(
                    "Specify exactly one of avoid_cluster / avoid_group, "
                    "not both."
                )
            if self.avoid_margin_threshold is None:
                raise ValueError(
                    "avoid_cluster / avoid_group rules must also set "
                    "avoid_margin_threshold"
                )
            # Avoid rules have no attractor target or dwell/position trigger.
            if (
                self.target_group is not None
                or self.target_cluster is not None
                or self.trigger_group is not None
                or self.trigger_dwell is not None
                or self.trigger_position is not None
            ):
                raise ValueError(
                    "avoid rules must not specify target_* or trigger_* "
                    "fields; use avoid_from_step for a delayed gate"
                )
            if self.avoid_max_tokens is not None and self.avoid_max_tokens < 1:
                raise ValueError(
                    f"avoid_max_tokens must be >= 1, got {self.avoid_max_tokens}"
                )
            if self.avoid_from_step is not None and self.avoid_from_step < 0:
                raise ValueError(
                    f"avoid_from_step must be >= 0, got {self.avoid_from_step}"
                )
            if self.avoid_sticky_duration < 1:
                raise ValueError(
                    f"avoid_sticky_duration must be >= 1, got "
                    f"{self.avoid_sticky_duration}"
                )
            return

        # Attractor rule validation (existing behavior)
        if (self.target_group is None) == (self.target_cluster is None):
            raise ValueError(
                "Exactly one of target_group / target_cluster must be set"
            )
        # Validate trigger mode: exactly one of (position) or (group+dwell).
        has_position = self.trigger_position is not None
        has_group_dwell = (
            self.trigger_group is not None and self.trigger_dwell is not None
        )
        if has_position and has_group_dwell:
            raise ValueError(
                "Specify either trigger_position OR (trigger_group + trigger_dwell), "
                "not both."
            )
        if not has_position and not has_group_dwell:
            raise ValueError(
                "Must specify either trigger_position, or both trigger_group "
                "and trigger_dwell."
            )
        if has_position and self.trigger_position < 0:
            raise ValueError(
                f"trigger_position must be >= 0, got {self.trigger_position}"
            )


@dataclass
class SteeringConfig:
    """Full configuration for an unsupervised steering run."""

    # -- Clustering artifacts --------------------------------------------------
    clustering_dir: str = ""
    window_size: int = 1
    K: int = 64
    seed: int = 42
    tau: float = 0.1
    alpha: float = 1.0

    # -- Steering rules --------------------------------------------------------
    rules: list[SteeringRule] = field(default_factory=list)

    # -- Steering method -------------------------------------------------------
    strength: float = 0.5            # step size as fraction of ||h||: δh = strength * ||h|| * grad_dir
    gradient_steps: int = 1          # 1 = single step, K>1 = iterative refinement

    # -- Cluster-assignment metric --------------------------------------------
    # "cosine"   (default, original behavior): center + L2-normalize z, assign
    #            by cosine similarity, target/current centers L2-normalized.
    #            Matches the original L2-normalized clustering pipeline.
    # "euclidean": center only (no L2 norm), assign by Euclidean distance,
    #            target/current centers used in raw form. For clusterings
    #            produced with the centered-only preprocessor (variant
    #            ``z_centered``).
    metric: str = "cosine"

    # -- Prediction-loss regularizer (single-layer steering only) -------------
    # When lambda_pred > 0 AND a closure predictor + per-step baseline_loss
    # are wired into the hook, the gradient objective is augmented with:
    #     -lambda_pred * clamp(MSE_steered - baseline_loss[t], min=-epsilon_pred)
    # which penalizes excess prediction loss above the baseline trajectory.
    lambda_pred: float = 0.0
    epsilon_pred: float = 0.0

    # -- CEM / MPPI (reserved for future horizon > 0) -------------------------
    horizon: int = 0
    num_samples: int = 256
    num_elites: int = 32
    iterations: int = 6
    temperature_mppi: float = 0.5
    lambda_reg: float = 1.0
    gamma: float = 0.99

    # -- Generation ------------------------------------------------------------
    prompt: str = ""
    max_new_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.9
    device: str = "cuda"
    output_dir: str | None = None
