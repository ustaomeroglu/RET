"""Config and dataloader helpers shared by clustering scripts.

(SAE encoding lives in evaluation.baselines.sae.encode_sae.)
"""
from __future__ import annotations

from dataclasses import asdict

from ret.config import (
    ExperimentConfig,
    _deep_merge,
    _dict_to_config,
    _migrate_deprecated_keys,
    load_config,
    validate_config,
)
from ret.data.dataset import build_cached_dataloader, build_padded_dataloader


def load_config_with_overlay(base_yaml=None, overlay_yaml=None) -> ExperimentConfig:
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


def build_dataloader(config, split, batch_size, seed, shuffle=None, drop_last=None):
    """Build the appropriate dataloader based on data mode."""
    if config.data.mode == "cached":
        return build_cached_dataloader(
            config.data, config.llm, split=split, batch_size=batch_size, seed=seed,
            shuffle=shuffle, drop_last=drop_last,
        )
    return build_padded_dataloader(
        config.data, config.llm, split=split, batch_size=batch_size, seed=seed,
        shuffle=shuffle, drop_last=drop_last,
    )


