"""Configuration dataclasses for SAE-based unsupervised steering."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SAESteeringRule:
    """One event-driven SAE-steering instruction.

    Two mutually exclusive modes:

    * **Position-triggered attractor**: set ``trigger_position`` and
      ``target_latents`` (list of SAE latent indices). On generation step
      ``trigger_position`` the rule fires and the model is steered toward
      the composite target direction for ``duration`` tokens.

    * **Activation-gated repulsion**: set ``avoid_latents`` (list of SAE
      latent indices) and ``avoid_pre_threshold``. Every step, the gate is
      "open" whenever ``max_i pre_acts[i] >= avoid_pre_threshold`` for
      ``i in avoid_latents``, where ``pre_acts = (h - b_dec) @ W_enc^T + b_enc``
      (raw pre-activation, NOT post-TopK). While open, repulsion fires.
      ``avoid_sticky_duration`` keeps repulsion firing for that many extra
      tokens after the gate closes (default 1 = stop immediately).
      ``avoid_max_tokens`` caps the total number of gate-open events;
      ``avoid_from_step`` delays gate activation until after that step.
    """

    # Attractor
    trigger_position: int | None = None
    target_latents: list[int] | None = None
    duration: int = 20

    # Repulsion
    avoid_latents: list[int] | None = None
    avoid_pre_threshold: float | None = None
    avoid_sticky_duration: int = 1
    avoid_max_tokens: int | None = None
    avoid_from_step: int | None = None

    def __post_init__(self):
        is_avoid = self.avoid_latents is not None
        is_attract = self.target_latents is not None

        if is_avoid and is_attract:
            raise ValueError(
                "A rule is either attractor (target_latents) or avoid "
                "(avoid_latents), not both."
            )
        if not is_avoid and not is_attract:
            raise ValueError(
                "Must specify exactly one of target_latents / avoid_latents."
            )

        if is_avoid:
            if not self.avoid_latents:
                raise ValueError("avoid_latents must be a non-empty list")
            if self.avoid_pre_threshold is None:
                raise ValueError(
                    "avoid_latents rules must also set avoid_pre_threshold"
                )
            if self.trigger_position is not None:
                raise ValueError(
                    "avoid rules must not specify trigger_position; use "
                    "avoid_from_step for delayed gate activation"
                )
            if any(lid < 0 for lid in self.avoid_latents):
                raise ValueError(
                    f"avoid_latents must all be >= 0, got {self.avoid_latents}"
                )
            if self.avoid_sticky_duration < 1:
                raise ValueError(
                    f"avoid_sticky_duration must be >= 1, got "
                    f"{self.avoid_sticky_duration}"
                )
            if self.avoid_max_tokens is not None and self.avoid_max_tokens < 1:
                raise ValueError(
                    f"avoid_max_tokens must be >= 1, got {self.avoid_max_tokens}"
                )
            if self.avoid_from_step is not None and self.avoid_from_step < 0:
                raise ValueError(
                    f"avoid_from_step must be >= 0, got {self.avoid_from_step}"
                )
            return

        # Attractor branch
        if not self.target_latents:
            raise ValueError("target_latents must be a non-empty list")
        if any(lid < 0 for lid in self.target_latents):
            raise ValueError(
                f"target_latents must all be >= 0, got {self.target_latents}"
            )
        if self.trigger_position is None:
            raise ValueError(
                "Attractor rules must specify trigger_position (>= 0)"
            )
        if self.trigger_position < 0:
            raise ValueError(
                f"trigger_position must be >= 0, got {self.trigger_position}"
            )
        if self.duration < 1:
            raise ValueError(f"duration must be >= 1, got {self.duration}")


@dataclass
class SAESteeringConfig:
    """Full configuration for one SAE-steering run."""

    # -- SAE selection ---------------------------------------------------------
    # Defaults work for the GPT-OSS-20B layer-11 SAE used in this project.
    sae_release: str | None = "andyrdt/saes-gpt-oss-20b"
    sae_id: str | None = "resid_post_layer_11/trainer_0"

    # YAML config path that defines the LLM (we reuse configs/.../sae_l11.yaml).
    llm_config_path: str = ""

    # Direction source: "decoder" uses W_dec[:, latent_id] (recommended);
    # "encoder" uses W_enc[latent_id]. Decoder is unit-norm by training
    # construction and is the standard choice in published SAE-steering work.
    direction_source: str = "decoder"

    # Optional path to a JSON file of named latents (latent_id -> top-5 tokens),
    # used for diagnostics & HTML viz only.
    sae_latent_names_path: str | None = None

    # -- Steering rules --------------------------------------------------------
    rules: list[SAESteeringRule] = field(default_factory=list)

    # -- Steering method -------------------------------------------------------
    # δh = sign * strength * ||h_t|| * unit_dir   (single closed-form step).
    # Parity with RET pipeline: same numeric range so a value works similarly.
    strength: float = 0.5

    # -- Generation ------------------------------------------------------------
    prompt: str = ""
    max_new_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    device: str = "cuda"
    output_dir: str | None = None

    # How many top-active latents to log per token (for diagnostics & HTML).
    top_k_diag: int = 16

    def __post_init__(self):
        if self.direction_source not in ("decoder", "encoder"):
            raise ValueError(
                f"direction_source must be 'decoder' or 'encoder', "
                f"got {self.direction_source!r}"
            )
        if self.strength < 0:
            raise ValueError(f"strength must be >= 0, got {self.strength}")
        if self.top_k_diag < 1:
            raise ValueError(f"top_k_diag must be >= 1, got {self.top_k_diag}")
        if self.max_new_tokens < 1:
            raise ValueError(
                f"max_new_tokens must be >= 1, got {self.max_new_tokens}"
            )
