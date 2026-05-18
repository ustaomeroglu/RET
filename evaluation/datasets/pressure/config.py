from dataclasses import dataclass, field

from evaluation.probing.config import (
    CacheConfig, RETConfig, EvalConfig, ModelConfig, PCAConfig,
    ProbeConfig, SAEConfig, SplitPathsConfig,
)


@dataclass
class PressureConfig:
    """Config for the models_under_pressure experiment.

    No generation needed — labels come directly from the dataset.
    Hidden states are extracted once and shared across all subtasks.
    """
    model: ModelConfig = field(default_factory=ModelConfig)
    probe: ProbeConfig = field(default_factory=ProbeConfig)
    eval: EvalConfig  = field(default_factory=EvalConfig)
    ret: RETConfig = field(default_factory=RETConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    splits: SplitPathsConfig = field(default_factory=SplitPathsConfig)
    pca: PCAConfig = field(default_factory=PCAConfig)
    sae: SAEConfig = field(default_factory=SAEConfig)
    # Output directory
    output_dir: str = "outputs/probing/models_under_pressure"
    include_gpt_info: bool = True
