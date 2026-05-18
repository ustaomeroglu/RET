"""Config and dataloader helpers for closure predictor training."""
from __future__ import annotations

import logging
from pathlib import Path

from dataclasses import asdict

from ret.config import (
    load_config, _deep_merge, _dict_to_config, _migrate_deprecated_keys,
    validate_config, ExperimentConfig,
)
from ret.data.dataset import build_dataloader, build_padded_dataloader, build_cached_dataloader

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ensure_list(val: int | list[int]) -> list[int]:
    """Normalise a scalar-or-list config value to a list."""
    return val if isinstance(val, list) else [val]


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config_with_overlay(base_yaml=None, overlay_yaml=None):
    """Load config from base_yaml, then deep-merge an optional overlay YAML."""
    import yaml as _yaml
    config = load_config(base_yaml)
    if overlay_yaml is not None:
        with open(overlay_yaml) as f:
            overlay = _yaml.safe_load(f) or {}
        base_dict = asdict(config)
        _deep_merge(base_dict, overlay)
        base_dict = _migrate_deprecated_keys(base_dict)
        config = _dict_to_config(base_dict, ExperimentConfig)
        validate_config(config)
    return config


def closure_data_config(config):
    """Return the DataConfig for closure predictor training.

    Closure predictors always use the exact same data as the main RET
    training — same dataset, scaffolding, and tokenization.
    """
    data_cfg = config.data
    logger.info(
        "Closure data config — hf_dataset=%s, local_dir=%s, scaffolding=%s, mode=%s "
        "(same as training data)",
        data_cfg.hf_dataset, data_cfg.local_dir, data_cfg.scaffolding, data_cfg.mode,
    )
    return data_cfg


def closure_data_tag(config) -> str:
    """Derive a short dataset tag for the closure subfolder name.

    Under the nested layout ``data/{dataset}/{model}/...``, returns
    ``"{dataset}_{model}"`` (e.g. ``"numina_math_GptOss20B"``). For base
    text datasets at ``data/{dataset}`` (no model nesting), returns
    ``"{dataset}"``. Falls back to ``hf_dataset`` basename or "default".
    """
    def _tag_from(path: str) -> str:
        p = Path(path)
        # If the immediate parent is "data", we're at dataset level only.
        if p.parent.name == "data":
            return p.name
        # Nested: <data>/<dataset>/<model>[/layerN] -> <dataset>_<model>
        # cached path is <data>/<dataset>/<model>/layerN; step up to <model>.
        # local_dir path is typically <data>/<dataset>/<model>.
        return f"{p.parent.name}_{p.name}"

    d = config.data
    if d.cached_hidden_states_dir:
        # e.g. "data/numina_math_64k_samples/GptOss20B/layer11" -> parent is GptOss20B -> tag from parent
        return _tag_from(str(Path(d.cached_hidden_states_dir).parent))
    if d.local_dir:
        return _tag_from(d.local_dir)
    if d.hf_dataset:
        return d.hf_dataset.split("/")[-1]
    return "default"


def closure_prompt_source(config) -> str:
    """Derive the prompt source tag from data mode for prompt logging."""
    if config.data.mode in ("document", "cached"):
        return "document"
    return "concat_chunk"


def build_closure_dataloader(config, split="train"):
    """Build the appropriate dataloader for closure probe training."""
    data_cfg = closure_data_config(config)
    mode = data_cfg.mode
    bs = config.closure.llm_batch_size or config.training.batch_size
    if mode == "cached":
        return build_cached_dataloader(
            data_cfg, config.llm, split=split,
            batch_size=bs,
            seed=config.training.seed,
        )
    elif mode == "document":
        return build_padded_dataloader(
            data_cfg, config.llm, split=split,
            batch_size=bs,
            seed=config.training.seed,
        )
    else:
        return build_dataloader(
            data_cfg, config.llm, split=split,
            batch_size=bs,
            seed=config.training.seed,
        )


# ---------------------------------------------------------------------------
# Streaming extraction helpers
# ---------------------------------------------------------------------------

def make_train_iter(train_loader):
    """Create an infinite iterator over a DataLoader, restarting on exhaustion."""
    it = iter(train_loader)
    while True:
        try:
            yield next(it)
        except StopIteration:
            it = iter(train_loader)
            yield next(it)
