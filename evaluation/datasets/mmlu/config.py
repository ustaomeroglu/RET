from dataclasses import dataclass, field

from evaluation.probing.config import (
    CacheConfig, RETConfig, EvalConfig, ModelConfig, PCAConfig,
    ProbeConfig, SAEConfig, SplitPathsConfig,
)


@dataclass
class MMLUConfig:
    """Config for the mmlu probing experiment.

    Uses pre-split Arrow datasets (produced by prepare_mmlu_splits.py).
    Runs multiple classification tasks: subject (57-class), group_4way
    (4-class), group_7way (7-class), and question_idx (contextual).
    """
    model: ModelConfig = field(default_factory=ModelConfig)
    probe: ProbeConfig = field(default_factory=ProbeConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    ret: RETConfig = field(default_factory=RETConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    splits: SplitPathsConfig = field(default_factory=SplitPathsConfig)
    pca: PCAConfig = field(default_factory=PCAConfig)
    sae: SAEConfig = field(default_factory=SAEConfig)
    # Directory containing pre-split train/ and val/ Arrow datasets
    mmlu_split_dir: str = "data/mmlu/eval"
    # Output directory
    output_dir: str = "outputs/probing/mmlu"
    include_gpt_info: bool = True
