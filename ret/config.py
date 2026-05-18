from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field, fields, asdict
from pathlib import Path
from typing import Literal

import yaml

logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    name: str = "EleutherAI/pythia-160m"
    layer_index: int | list[int] | str = 6
    #   int        = single-layer extraction (existing behaviour, with early exit).
    #   list[int]  = extract & concatenate the listed layers.
    #   "all"      = extract all layers (resolved at runtime after model is loaded).
    hidden_dim: int = 768
    dtype: str = "float16"
    trust_remote_code: bool = False
    device_map: str | None = None  # None = single GPU .to(device); "auto" = multi-GPU
    vocab_size: int = 50304        # tokenizer vocab size (GPT-NeoX/Pythia default)


@dataclass
class DataConfig:
    # --- Data mode ---
    mode: Literal["chunked", "document", "cached"] = "chunked"
    #   chunked  = concat-and-chunk tokenized text into fixed-length segments.
    #   document = variable-length documents with per-batch padding + attention masks.
    #   cached   = pre-extracted hidden states from memmap files (no LLM loaded).

    # --- Data source (mutually exclusive) ---
    hf_dataset: str | None = "monology/pile-uncopyrighted"
    #   HuggingFace dataset ID. Streamed, concat-and-chunked. Used in chunked mode.
    local_dir: str | None = None
    #   Path to Arrow dataset on disk. Used in chunked or document mode.
    text_field: str = "text"
    # --- Scaffolding ---
    scaffolding: Literal["raw", "chatify", "proper_chat"] = "raw"
    #   raw         = text fed as-is to tokenizer (base models or pre-formatted data)
    #   chatify     = wrap entire text as a single user message: [system, user: <text>]
    #   proper_chat = read structured messages from messages_field and pass to
    #                 apply_chat_template(): [system, user: problem, assistant: solution, ...]
    #                 Requires document mode.
    messages_field: str = "messages"
    #   Column name for structured messages (list of {role, content} dicts).
    #   Only used when scaffolding="proper_chat".
    system_message: str = "You are a helpful assistant."
    #   Content of the system role. Used by scaffolding="chatify" and "proper_chat".
    chat_template_kwargs: dict = field(default_factory=dict)
    #   Extra kwargs forwarded to tokenizer.apply_chat_template().
    #   Model-specific, e.g. {"reasoning_effort": "high", "model_identity": "You are a math tutor."}
    # --- Chunked mode settings ---
    chunk_length: int = 512
    val_fraction: float = 0.02
    num_workers: int = 4
    shuffle_buffer_size: int = 10_000
    val_chunks: int = 500
    # --- Sequence length cap (works in any mode) ---
    max_seq_length: int | None = None  # max tokens per document; None = no truncation
    # --- Cached hidden states (only used when mode="cached") ---
    cached_hidden_states_dir: str | None = None
    #   Path to root directory of cached hidden states. Two layouts supported:
    #   1. Split layout: contains train/, val/ subdirs with memmap files
    #      (from scripts/cache_hidden_states.py).
    #   2. Unified layout: contains conv_index.jsonl + *.bin files
    #      (sycophancy unified memmap format). Requires split_indices_path.
    split_indices_path: str | None = None
    #   Path to split_indices.json for unified memmap format (conversation-level splits).
    #   Only used when cached_hidden_states_dir points to a unified memmap cache.
    judge_results_path: str | None = None
    #   Path to judge_results.jsonl for reward supervision. When set, each batch
    #   includes per-token reward targets and reward_mask (dataset-specific).
    sycophancy_mode: str | None = None
    #   REQUIRED when judge_results_path is set.
    #   "debate" or "false_presupposition" — controls sycophancy label semantics.
    #   No default to prevent silent misuse.
    truncate_at_first_cave: bool = False
    #   When True (requires judge_results_path), truncate each conversation
    #   at the end of the first turn where judgment[t]==0. Conversations
    #   that never cave keep all turns. Useful for false-presupposition data
    #   where post-cave turns are not meaningful.


@dataclass
class EncoderConfig:
    type: Literal["mlp", "transformer"] = "mlp"
    z_dim: int = 64
    z_regularization: Literal["none", "sparse", "vq"] = "none"
    # MLP-specific
    mlp_hidden_dims: list[int] = field(default_factory=lambda: [512, 256])
    # Transformer-specific
    num_heads: int = 4
    num_layers: int = 1
    ff_dim: int | None = None  # defaults to 4 * d_model at construction time
    dropout: float = 0.0
    proj_position: Literal["begin", "end"] = "begin"
    # Multi-layer bottleneck projection (only used when layer_index is a list/"all")
    down_proj_bottleneck: int | None = None  # e.g. 128: concat_dim → 128 → out_dim
    down_proj_out_dim: int | None = None     # output dim after up-projection (default: hidden_dim)
    # Sparse-specific
    lambda_sparse: float = 1e-3
    # VQ-specific
    codebook_size: int = 512
    commitment_weight: float = 0.25


@dataclass
class PredictorConfig:
    mode: Literal["autonomous", "controlled"] = "autonomous"
    control_dim: int = 32
    hidden_dim: int = 256
    num_layers: int = 2


@dataclass
class EMAConfig:
    enabled: bool = True       # set False to disable EMA teacher entirely
    momentum: float = 0.996
    warmup_momentum: float = 0.99
    warmup_steps: int = -1  # -1 = auto (10% of max_steps)


@dataclass
class LossConfig:
    prediction_type: Literal["cosine", "mse", "l1"] = "cosine"
    regularizer: Literal["vicreg", "sigreg", "none"] = "none"  # paper RET uses no collapse regularizer (Table 1); EMA teacher + predictor bottleneck suffice
    # VICReg params (used when regularizer="vicreg")
    lambda_var: float = 25.0
    lambda_cov: float = 1.0
    var_target: float = 1.0
    # SIGReg params (used when regularizer="sigreg")
    lambda_sigreg: float = 0.09
    sigreg_knots: int = 17
    sigreg_num_proj: int = 1024
    # Autoregressive
    ar_horizon: int = 1       # autoregressive rollout steps (1 = current 1-step pred)
    ar_rho: float = 0.9       # temporal discount per step
    # Temporal smoothness (TV regularization on z trajectories)
    lambda_smooth: float = 0.0       # 0 = disabled (backward compat)
    smooth_loss_type: str = "l1"     # "l1", "l2", "huber"
    smooth_dims: int = -1            # -1 = all z dims, else first N dims


@dataclass
class PolicyConfig:
    enabled: bool = False
    lambda_policy: float = 1.0   # weight on policy loss (no scaling needed when detached)
    hidden_dim: int = 256        # MLP hidden layer width
    num_layers: int = 3          # total layers (1 projection + (num_layers-1) residual blocks + output)
    warmup_steps: int = 0        # steps before policy loss is applied
    detach_z: bool = True        # True = policy trains independently; False = auxiliary loss to encoder


@dataclass
class SupervisionConfig:
    reward_enabled: bool = False   # reward head (can be used standalone)
    value_enabled: bool = False    # value head (requires reward_enabled — TD target uses rewards)
    reward_hidden_dim: int = 256
    reward_num_layers: int = 3
    value_hidden_dim: int = 256
    value_num_layers: int = 3
    lambda_reward: float = 1.0
    lambda_value: float = 1.0
    value_ema_momentum: float = 0.995
    warmup_steps: int = 0


@dataclass
class TrainingConfig:
    batch_size: int = 64
    gradient_accumulation_steps: int = 1
    max_steps: int = 6_000
    lr: float = 3e-4
    lr_min_fraction: float = 0.1
    weight_decay: float = 0.01
    warmup_steps: int = -1  # -1 = auto (10% of max_steps)
    grad_clip: float = 1.0
    eval_every: int = 200
    checkpoint_every: int = 1_000_000_000_000_000_000
    num_chunks: int = 400_000
    seed: int = 42
    resume_from: str | None = None  # path to checkpoint to resume training from


@dataclass
class LoggingConfig:
    wandb_project: str = "Initial_Variants"
    wandb_entity: str | None = "ret"
    wandb_run_name: str | None = None
    log_every: int = 20
    output_dir: str = "outputs"


@dataclass
class ClosureSAEConfig:
    """Closure predictor config for SAE baseline (autonomous/controlled variants).

    ``hidden_dim`` may be a single int (one run) or a list of ints (sweep
    over predictor capacities).
    """
    hidden_dim: int | list[int] = 256
    num_layers: int = 2
    lr: float = 1e-3
    max_steps: int = 5000
    batch_size: int = 1024
    patience: int = 15
    val_batches: int = 20
    eval_every: int = 200


@dataclass
class ClosureSingleConfig:
    """Closure predictor config for single-variant baselines (h→h, PCA).

    ``hidden_dim`` may be a single int (one run) or a list of ints (sweep).
    ``pca_dim`` controls PCA projection dimensionality (ignored for h→h).
    """
    hidden_dim: int | list[int] = 3328
    pca_dim: int | list[int] = 64   # PCA components; list for sweep (ignored for hh)
    num_layers: int = 2
    lr: float = 1e-3
    max_steps: int = 5000
    batch_size: int = 1024
    patience: int = 15
    val_batches: int = 20
    eval_every: int = 200


@dataclass
class ClosureConfig:
    """Closure predictor training configuration.

    ``hidden_dim`` may be a single int (one run) or a list of ints (sweep
    over predictor capacities).  Used by both autonomous and controlled
    variants.

    ``variant`` identifies the representation type for config-driven closure
    predictor training: "z_ret", "h", "h_pca", "sae", or None (legacy mode).
    """
    variant: str | None = None     # "z_ret", "h", "h_pca", "sae" — config-driven variant
    output_dir: str | None = None  # output dir for this variant (baselines); z_ret uses logging.output_dir
    pca_dim: int | None = None     # PCA dimensionality (h_pca variant only)
    hidden_dim: int | list[int] = 256      # autonomous + controlled predictor hidden dim
    num_layers: int = 2
    control_dim: int = 32          # projection dim for ctrl variants
    lr: float = 1e-3
    max_steps: int = 5000
    batch_size: int = 1024
    llm_batch_size: int = 0        # sequences per LLM forward pass; 0 = use training.batch_size
    patience: int = 15
    val_batches: int = 20          # batches to pre-collect for val set
    eval_every: int = 200          # steps between val evaluations
    # Baseline-specific configs (completely separate, legacy — use variant-specific config files instead)
    sae: ClosureSAEConfig = field(default_factory=ClosureSAEConfig)
    hh: ClosureSingleConfig = field(default_factory=ClosureSingleConfig)
    pca: ClosureSingleConfig = field(default_factory=lambda: ClosureSingleConfig(hidden_dim=256))


@dataclass
class ExperimentConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    data: DataConfig = field(default_factory=DataConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    predictor: PredictorConfig = field(default_factory=PredictorConfig)
    ema: EMAConfig = field(default_factory=EMAConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    supervision: SupervisionConfig = field(default_factory=SupervisionConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    closure: ClosureConfig = field(default_factory=ClosureConfig)


def _nested_update(d: dict, keys: list[str], value: str) -> None:
    """Set d[keys[0]][keys[1]]...[keys[-1]] = value with type coercion."""
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    leaf = keys[-1]
    # Coerce value to the right type
    if value.lower() in ("true", "false"):
        d[leaf] = value.lower() == "true"
    elif value.lower() == "none":
        d[leaf] = None
    else:
        for caster in (int, float):
            try:
                d[leaf] = caster(value)
                return
            except ValueError:
                continue
        d[leaf] = value


def _dict_to_config(d: dict, cls: type, _path: str = "") -> object:
    """Recursively build a dataclass from a nested dict.

    Warns about unknown keys that don't match any dataclass field (likely typos).
    """
    import typing
    hints = typing.get_type_hints(cls)
    valid_names = {f.name for f in fields(cls)}

    # Warn about unknown keys
    for key in d:
        if key not in valid_names:
            full_path = f"{_path}.{key}" if _path else key
            logger.warning(
                "Unknown config key '%s' — will be ignored. "
                "Check for typos. Valid keys for %s: %s",
                full_path, cls.__name__, sorted(valid_names),
            )

    kwargs = {}
    for f in fields(cls):
        if f.name not in d:
            continue
        val = d[f.name]
        resolved_type = hints.get(f.name)
        child_path = f"{_path}.{f.name}" if _path else f.name
        if isinstance(val, dict) and isinstance(resolved_type, type) and hasattr(resolved_type, "__dataclass_fields__"):
            kwargs[f.name] = _dict_to_config(val, resolved_type, _path=child_path)
        else:
            # Coerce to declared type when value is a string but field expects numeric/bool
            if isinstance(val, str) and resolved_type in (int, float, bool):
                if resolved_type is bool:
                    val = val.lower() == "true"
                else:
                    val = resolved_type(val)
            kwargs[f.name] = val
    return cls(**kwargs)


def validate_config(config: ExperimentConfig) -> None:
    """Cross-field validation. Raises ValueError on inconsistencies."""
    d = config.data
    llm = config.llm
    enc = config.encoder

    # Multi-layer extraction validation
    is_multi_layer = isinstance(llm.layer_index, list) or llm.layer_index == "all"
    if is_multi_layer:
        if d.mode == "cached":
            raise ValueError(
                "Multi-layer extraction (layer_index as list/'all') is incompatible "
                "with data.mode='cached'. Use mode='document' or 'chunked' instead."
            )
        if enc.down_proj_bottleneck is None:
            logger.warning(
                "Multi-layer extraction without encoder.down_proj_bottleneck set. "
                "The concatenated input dimension will be very large (%s * %d). "
                "Consider setting down_proj_bottleneck (e.g. 128).",
                llm.layer_index, llm.hidden_dim,
            )

    # Mode-specific data source validation
    if d.mode == "cached":
        if d.cached_hidden_states_dir is None:
            raise ValueError(
                "data.mode='cached' requires data.cached_hidden_states_dir to be set."
            )
        if d.hf_dataset is not None or d.local_dir is not None:
            raise ValueError(
                "data.mode='cached' is incompatible with data.hf_dataset and "
                "data.local_dir. Set them to null."
            )
        return  # skip remaining data-source and scaffolding checks

    if d.mode == "document":
        if d.local_dir is None:
            raise ValueError(
                "data.mode='document' requires data.local_dir to be set."
            )
        if d.hf_dataset is not None:
            raise ValueError(
                "data.mode='document' is incompatible with data.hf_dataset. "
                "Set it to null."
            )

    if d.mode == "chunked":
        if d.hf_dataset is None and d.local_dir is None:
            raise ValueError(
                "data.mode='chunked' requires either data.hf_dataset or "
                "data.local_dir to be set."
            )
        if d.hf_dataset is not None and d.local_dir is not None:
            raise ValueError(
                "data.hf_dataset and data.local_dir are mutually exclusive. "
                "Set one to null."
            )

    # proper_chat requires document mode (can't chunk multi-turn conversations)
    if d.scaffolding == "proper_chat" and d.mode != "document":
        raise ValueError(
            "data.scaffolding='proper_chat' requires data.mode='document'. "
            "Multi-turn messages cannot be meaningfully chunked across "
            "arbitrary boundaries."
        )

    # chatify + chunked: loud warning
    if d.scaffolding == "chatify" and d.mode == "chunked":
        logger.warning(
            "=" * 70 + "\n"
            "WARNING: scaffolding='chatify' with chunked mode.\n"
            "Each fixed-length chunk will be independently wrapped in a chat\n"
            "template. Chunks contain arbitrary text boundaries — this is\n"
            "intentional for large-corpus training but may not be appropriate\n"
            "for structured data. If you need per-document scaffolding, use\n"
            "mode='document' instead.\n" +
            "=" * 70
        )

    # Chat scaffolding requires a non-empty system_message
    if d.scaffolding in ("chatify", "proper_chat") and not d.system_message:
        raise ValueError(
            f"data.scaffolding='{d.scaffolding}' requires a non-empty "
            f"data.system_message."
        )

    # Controlled mode: warn if control_dim is set but mode is autonomous
    p = config.predictor
    if p.mode == "autonomous" and p.control_dim is not None and p.control_dim > 0:
        logger.warning(
            "predictor.mode='autonomous' but control_dim=%d is set. "
            "control_dim will be ignored.", p.control_dim,
        )


def _migrate_deprecated_keys(d: dict) -> dict:
    """Migrate deprecated config keys with warnings."""
    data = d.get("data", {})

    # dataset_name → hf_dataset (always overrides — variant takes precedence)
    if "dataset_name" in data:
        data["hf_dataset"] = data.pop("dataset_name")
        logger.warning("Deprecated: 'data.dataset_name' — use 'data.hf_dataset'.")

    # data_dir → local_dir
    if "data_dir" in data:
        old_val = data.pop("data_dir")
        if old_val is not None:
            data["local_dir"] = old_val
            data["hf_dataset"] = None  # Force mutual exclusivity
            logger.warning("Deprecated: 'data.data_dir' — use 'data.local_dir'.")
        else:
            logger.warning("Deprecated: 'data.data_dir=null' — remove it.")

    # system_prompt → scaffolding + system_message
    if "system_prompt" in data:
        sp = data.pop("system_prompt")
        if sp is None:
            data.setdefault("scaffolding", "raw")
        else:
            data["scaffolding"] = "chatify"
            data["system_message"] = sp
        logger.warning(
            "Deprecated: 'data.system_prompt' — use 'data.scaffolding' + "
            "'data.system_message'.",
        )

    # scaffolding: chat → chatify (renamed)
    if data.get("scaffolding") == "chat":
        data["scaffolding"] = "chatify"
        logger.warning(
            "Deprecated: scaffolding='chat' has been renamed to 'chatify'.",
        )

    # mode was previously auto-derived; now it's an explicit field — keep it

    # Remove dead fields
    for dead in ("dataset_subset", "hf_split", "streaming"):
        if dead in data:
            data.pop(dead)
            logger.warning("Deprecated: 'data.%s' removed. Ignoring.", dead)

    # --- Closure config migration ---
    closure = d.get("closure", {})

    # Remove obsolete data fields
    for dead in ("predictor_dataset", "predictor_data_dir",
                 "predictor_hf_dataset", "predictor_local_dir", "predictor_text_field"):
        if dead in closure:
            closure.pop(dead)
            logger.warning("Deprecated: 'closure.%s' removed. Closure always uses training data.", dead)

    # Migrate old hidden_dim_a / hidden_dim_b → hidden_dim
    for _old in ("hidden_dim_a", "hidden_dim_b",
                 "auto_hidden_dim_a", "ctrl_hidden_dim_a",
                 "auto_hidden_dim_b", "ctrl_hidden_dim_b"):
        if _old in closure:
            val = closure.pop(_old)
            closure.setdefault("hidden_dim", val)
            logger.warning("Deprecated: 'closure.%s' → 'closure.hidden_dim'.", _old)

    # Migrate SAE sub-config similarly
    sae_cfg = closure.get("sae", {})
    if isinstance(sae_cfg, dict):
        for _old in ("hidden_dim_a", "hidden_dim_b",
                     "auto_hidden_dim_a", "ctrl_hidden_dim_a",
                     "auto_hidden_dim_b", "ctrl_hidden_dim_b"):
            if _old in sae_cfg:
                val = sae_cfg.pop(_old)
                sae_cfg.setdefault("hidden_dim", val)
                logger.warning("Deprecated: 'closure.sae.%s' → 'closure.sae.hidden_dim'.", _old)

    return d


def load_config(
    yaml_path: str | Path | None = None,
    overrides: list[str] | None = None,
    default_yaml: str | Path | None = None,
) -> ExperimentConfig:
    """Load config: [default YAML ->] main YAML -> CLI overrides.

    If ``default_yaml`` is provided, it is loaded first as the base and the
    main ``yaml_path`` is deep-merged on top (variant-style).  If not, the
    main ``yaml_path`` is the full config — dataclass defaults fill any
    unspecified fields.

    Parameters
    ----------
    yaml_path : path, optional
        Main config YAML.  When ``default_yaml`` is also given this acts as
        a variant overlay; otherwise it is the complete config.
    overrides : list of str, optional
        CLI ``key=value`` overrides applied last.
    default_yaml : path, optional
        Base config YAML loaded first.  Use ``--default`` on the CLI to set
        this.  Pass ``None`` (the default) for self-contained configs.
    """
    if default_yaml is not None:
        with open(default_yaml) as f:
            base = yaml.safe_load(f) or {}
        if yaml_path is not None:
            with open(yaml_path) as f:
                variant = yaml.safe_load(f) or {}
            _deep_merge(base, variant)
    elif yaml_path is not None:
        with open(yaml_path) as f:
            base = yaml.safe_load(f) or {}
    else:
        base = {}

    # Apply CLI overrides
    for override in overrides or []:
        key, _, value = override.partition("=")
        if not value:
            raise ValueError(f"Invalid override: {override!r}. Expected key=value.")
        _nested_update(base, key.split("."), value)

    base = _migrate_deprecated_keys(base)
    config = _dict_to_config(base, ExperimentConfig)
    validate_config(config)
    return config


def _deep_merge(base: dict, override: dict) -> dict:
    """In-place deep merge override into base."""
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def save_config(config: ExperimentConfig, path: str | Path) -> None:
    """Save config to YAML."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(asdict(config), f, default_flow_style=False, sort_keys=False)


def llm_short_name(llm_config: LLMConfig) -> str:
    """Derive a short, filesystem-safe name from the LLM config.

    Examples:
        "EleutherAI/pythia-160m" -> "Pythia160M"
        "openai/gpt-oss-20b" -> "GptOss20B"
    """
    model_part = llm_config.name.split("/")[-1]
    parts = model_part.split("-")
    result = ""
    for p in parts:
        if p.lower().endswith(("m", "b", "t")) and len(p) > 1 and p[:-1].replace('.', '', 1).isdigit():
            result += p[:-1] + p[-1].upper()
        else:
            result += p.capitalize()
    return result


def config_from_cli() -> ExperimentConfig:
    """Parse config from CLI: train.py [--default base.yaml] [config.yaml] [key=value ...]."""
    args = sys.argv[1:]

    # Extract --default <path> if present
    default_yaml = None
    if "--default" in args:
        idx = args.index("--default")
        if idx + 1 >= len(args):
            raise ValueError("--default requires a path argument.")
        default_yaml = args[idx + 1]
        args = args[:idx] + args[idx + 2:]

    yaml_path = None
    overrides = []
    for arg in args:
        if "=" in arg:
            overrides.append(arg)
        elif yaml_path is None and Path(arg).suffix in (".yaml", ".yml"):
            yaml_path = arg
        else:
            overrides.append(arg)
    return load_config(yaml_path, overrides, default_yaml=default_yaml)
