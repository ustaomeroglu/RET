from __future__ import annotations

from pathlib import Path

import torch

from ret.config import ExperimentConfig, save_config


def save_checkpoint(
    path: str | Path,
    step: int,
    encoder: torch.nn.Module,
    predictor: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: ExperimentConfig,
    token_projector: torch.nn.Module | None = None,
    policy_network: torch.nn.Module | None = None,
    supervision_reward: torch.nn.Module | None = None,
    supervision_value: torch.nn.Module | None = None,
    extra: dict | None = None,
) -> None:
    """Save training checkpoint.

    The ``extra`` dict is merged into the saved state and can carry
    resume-critical state like EMA weights, LR scheduler, and global_step.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    state = {
        "step": step,
        "encoder": encoder.state_dict(),
        "predictor": predictor.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    if token_projector is not None:
        state["token_projector"] = token_projector.state_dict()
    if policy_network is not None:
        state["policy_network"] = policy_network.state_dict()
    if supervision_reward is not None:
        state["supervision_reward"] = supervision_reward.state_dict()
    if supervision_value is not None:
        state["supervision_value"] = supervision_value.state_dict()
    if extra:
        state.update(extra)

    torch.save(state, path)
    save_config(config, path.parent / "config.yaml")


def load_checkpoint(path: str | Path) -> dict:
    """Load training checkpoint."""
    return torch.load(path, map_location="cpu", weights_only=False)


def load_encoder(
    checkpoint_path: "str | Path",
    device: str = "cuda",
    config: "ExperimentConfig | None" = None,
) -> tuple:
    """Load encoder from checkpoint with config.

    Convenience wrapper that handles the common pattern of loading a checkpoint,
    building the encoder from config, and loading weights.

    Parameters
    ----------
    checkpoint_path : path to the checkpoint .pt file
    device : device to place the encoder on
    config : optional pre-loaded config; if None, loads from config.yaml
             in the same directory as the checkpoint

    Returns
    -------
    (encoder, config, checkpoint_dict) tuple
    """
    from ret.config import load_config
    from ret.models.encoders import build_encoder

    ckpt_path = Path(checkpoint_path)
    if config is None:
        config = load_config(str(ckpt_path.parent / "config.yaml"))
    ckpt = load_checkpoint(ckpt_path)
    # Multi-layer: compute concat_dim from config
    li = config.llm.layer_index
    if li == "all":
        # Cannot resolve "all" without the model; use checkpoint state_dict to infer
        # the down_proj input dimension if the encoder is a MultiLayerEncoder.
        concat_dim = None
        if config.encoder.down_proj_bottleneck is not None:
            # Infer from saved weights: down_proj.0.weight has shape (bottleneck, concat_dim)
            dp_key = "down_proj.0.weight"
            if dp_key in ckpt["encoder"]:
                concat_dim = ckpt["encoder"][dp_key].shape[1]
    elif isinstance(li, list):
        n_layers = len(li)
        concat_dim = n_layers * config.llm.hidden_dim if n_layers > 1 else None
    else:
        concat_dim = None
    encoder = build_encoder(config.llm.hidden_dim, config.encoder, concat_dim=concat_dim)
    encoder.load_state_dict(ckpt["encoder"])
    encoder.to(device).eval()
    return encoder, config, ckpt
