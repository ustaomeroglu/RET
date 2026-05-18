"""
Experiment configuration dataclasses for probing tasks.

Each experiment can import the base configs and extend/override as needed.
"""
from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class ModelConfig:
    """Configuration for the model and hidden-state extraction."""
    model_id: str = "openai/gpt-oss-20b"
    layer: int = 15
    hidden_state_type: str = "all"  # "average", "assistant_token", or "all"
    batch_size: int = 8
    device: str = "cuda"
    # Number of GPUs to use. 0 = use all available GPUs.
    n_gpus: int = 0


@dataclass
class ProbeConfig:
    """Configuration for probe training."""
    # "linear"      : torch logistic regression
    # "dom"         : difference-of-means
    # "attention"   : attention probe over all token hidden states
    # "transformer" : single transformer encoder layer probe
    # "all"         : run all four
    probe_type: str = "all"
    # Linear probe training hyper-params
    lr: float = 1e-3
    max_epochs: int = 1000
    l2: float = 1e-4
    # Mini-batch size for probe training on GPU (attention probe + linear probe).
    # Reduce if GPU OOM during backward pass after model is freed.
    probe_batch_size: int = 64


@dataclass
class EvalConfig:
    """Configuration for the evaluation protocol."""
    # "cross_dataset"  : leave-one-out across datasets
    # "within_dataset" : 80/20 train/val split per dataset
    # "all"            : both
    eval_type: Literal["cross_dataset", "within_dataset", "all"] = "all"
    # Fraction of data for validation in within_dataset mode
    val_ratio: float = 0.05
    # Optional cap on samples per dataset (useful for fast smoke tests)
    max_samples_per_dataset: Optional[int] = None


@dataclass
class RETConfig:
    """Configuration for RET encoder integration in probing.

    When checkpoint_path is set, hidden states h are encoded to z via
    a pre-trained RET encoder before probe training.
    """
    # Path to RET checkpoint (checkpoint_final.pt). None = baseline (probe raw h).
    checkpoint_path: Optional[str] = None
    device: str = "cuda"
    # Whether the RET encoder should see system-prompt hidden states.
    # True  = encoder sees full sequence (recommended: preserves context).
    # False = encoder only sees user-input tokens.
    # Probe pooling always excludes system tokens regardless of this flag.
    include_system_in_ret: bool = True


@dataclass
class CacheConfig:
    """Configuration for hidden-state disk caching.

    When cache_dir is set and use_cache is True, experiment runners will
    load hidden states from disk instead of running the LLM.
    """
    cache_dir: Optional[str] = None
    use_cache: bool = False


@dataclass
class PCAConfig:
    """PCA baseline configuration.

    When pca_dir is set, hidden states are projected through PCA components
    before probing.  If pca_components.npz doesn't exist in pca_dir,
    PCA is auto-fitted on the train split (requires pca_dim and cache_dir).
    """
    pca_dir: Optional[str] = None
    pca_dim: Optional[int] = None


@dataclass
class SAEConfig:
    """SAE baseline configuration.

    When checkpoint_path or hf_repo is set, hidden states are encoded
    through a pre-trained Sparse Autoencoder before probing.
    """
    checkpoint_path: Optional[str] = None
    hf_repo: Optional[str] = None


@dataclass
class SplitPathsConfig:
    """Pre-split probe train/test data paths.

    When both directories are set, experiment runners load data from
    per-subdataset JSONL files instead of loading from raw sources and
    splitting internally.  This ensures reproducible, fixed splits that
    match RET training splits produced by ``prepare_probing_splits.py``.

    Directory layout expected::

        probe_train_dir/
          <dataset_name>.jsonl
        probe_test_dir/
          <dataset_name>.jsonl
    """
    probe_train_dir: Optional[str] = None
    probe_test_dir: Optional[str] = None
