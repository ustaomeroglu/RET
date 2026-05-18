"""
Configuration dataclasses for sycophancy evaluation (SYCON-Bench).
"""
from dataclasses import dataclass, field
from typing import List, Optional, Union

from evaluation.probing.config import (
    CacheConfig, RETConfig, EvalConfig, ModelConfig, PCAConfig,
    ProbeConfig, SAEConfig, SplitPathsConfig,
)


@dataclass
class SycophancyModelConfig:
    """Model and hidden-state extraction settings."""
    model_id: str = "Qwen/Qwen2.5-14B-Instruct"
    # Layers to extract hidden states from (-1 = last layer)
    layers: List[int] = field(default_factory=lambda: [-1])
    max_new_tokens: int = 2048
    device: str = "cuda"
    n_gpus: int = 0


@dataclass
class SycophancyConfig:
    """Top-level config for SYCON-Bench evaluation."""
    model: SycophancyModelConfig = field(default_factory=SycophancyModelConfig)
    # Which scenario(s) to run: debate | false_presupposition | all
    scenario: str = "all"
    num_turns: int = 5
    data_dir: str = "data/sycophancy/debate"
    # Limit samples per scenario (None = use all)
    max_samples: Optional[int] = None
    seed: int = 42


@dataclass
class SycophancyProbingConfig:
    """Config for sycophancy probing experiments.

    Uses pre-extracted hidden states from the generation pipeline
    (no LLM forward pass needed). Consumed by scripts/probing/probe.py
    via evaluation.probing.sycophancy.runner.
    """
    model: ModelConfig = field(default_factory=ModelConfig)
    probe: ProbeConfig = field(default_factory=ProbeConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    ret: RETConfig = field(default_factory=RETConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    splits: SplitPathsConfig = field(default_factory=SplitPathsConfig)
    pca: PCAConfig = field(default_factory=PCAConfig)
    sae: SAEConfig = field(default_factory=SAEConfig)
    # Path to conversation data dir (e.g. data/sycophancy/augmented_debate/Qwen2.5-14B-Instruct)
    data_dir: str = "data/sycophancy/augmented_debate/Qwen2.5-14B-Instruct"
    # Layer index for pre-extracted hidden states
    layer: int = 23
    output_dir: str = "outputs/Qwen2.5-14B-Instruct/sycophancy/debate/baselines/h/probing"
    # Limit conversations for smoke tests (None = all)
    max_conversations: Optional[int] = None
    # Subsample tokens per turn (None = all tokens, e.g. 20 = random 20 per turn)
    max_tokens_per_turn: Optional[int] = None
    # Sycophancy label semantics: "debate" or "false_presupposition".
    # **Convention: label==1 = sycophantic in both modes.**
    # debate: drops convs where turn-1 reply did not defend the assigned argument;
    #         label[0]=0, label[t]=1 if judgment[t] != judgment[t-1] (stance flipped).
    # false_presupposition: label[t] = 1 - judgment[t]; never drops convs.
    # No default — must be set explicitly.
    mode: Optional[str] = None
    # When set, exclude turns after the first turn with this label value.
    # For false_presupposition: defaults to truncate_on=1 (stop after the
    # first sycophantic turn; post-cave turns carry the -1 sentinel).
    truncate_on: Optional[int] = None
    # Split file name within data_dir. Default is the full split; use
    # "split_indices_balanced.json" for balanced splits (flipping convs only).
    split_file: str = "split_indices.json"
    # Random seed: controls per-turn token subsampling (build_token_index)
    # AND probe init / dataloader shuffle (via torch.manual_seed at runner start).
    # Existing runs were unseeded; new multi-seed runs use 42/43/44 with separate
    # output_dirs so prior results are not overwritten.
    seed: int = 42
    # If True, each turn contributes exactly ONE sample: the hidden state at
    # the assistant role-name token (positions[1] within the assistant portion
    # of the turn, i.e. the `assistant` token right after `<|im_start|>`). This
    # hidden state encodes the full prior context (system prompt, every prior
    # turn, and this turn's user message) via causal attention, but no
    # generated content from the current assistant reply. Used for the
    # "user-context-only" probing experiment.
    # When True, `max_tokens_per_turn` is ignored; sequence probes skip the
    # `turn_segment` mode (degenerate at one position per turn).
    user_context_only: bool = False
