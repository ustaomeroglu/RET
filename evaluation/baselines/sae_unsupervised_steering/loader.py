"""Load all artifacts needed for an SAE-steering run.

We reuse ``evaluation.baselines.sae.load_sae_baseline`` for the encoder-side
TopKSAE; the decoder weight matrix (which the existing loader discards)
is fetched separately from the same cached checkpoint.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluation.baselines.sae import TopKSAE, _load_sae_from_state_dict, load_sae_baseline
from evaluation.baselines.sae_unsupervised_steering.config import SAESteeringConfig
from evaluation.baselines.sae_unsupervised_steering.latent_search import load_sae_latent_names

logger = logging.getLogger(__name__)


@dataclass
class SAESteeringContext:
    """All artifacts for an SAE-steering run."""

    model: AutoModelForCausalLM
    tokenizer: AutoTokenizer
    sae: TopKSAE
    decoder_weight: torch.Tensor | None  # (d_in, d_sae); None if checkpoint had none
    encoder_weight: torch.Tensor         # (d_sae, d_in)
    encoder_bias: torch.Tensor           # (d_sae,)
    b_dec: torch.Tensor                  # (d_in,)
    layer_idx: int
    layers: nn.ModuleList
    hidden_dim: int
    device: str
    sae_release: str
    sae_id: str
    sae_latent_names: dict[int, list[str]] | None


def _load_andyrdt_sae_with_decoder(
    sae_release: str,
    sae_id: str,
    device: str,
) -> tuple[TopKSAE, torch.Tensor | None]:
    """Direct andyrdt loader: downloads ``ae.pt`` once and extracts both the
    encoder-side ``TopKSAE`` and the decoder weight matrix.

    The closure pipeline's ``load_sae_baseline`` only takes the andyrdt path
    when ``sae_release is None``, so explicit sae_release strings would
    otherwise bypass the andyrdt-specific format handling. This helper
    handles that case directly.

    Returns ``(sae, decoder_weight_or_None)``.
    """
    from huggingface_hub import hf_hub_download

    ae_path = hf_hub_download(sae_release, f"{sae_id}/ae.pt")
    state = torch.load(ae_path, map_location="cpu", weights_only=False)
    sae = _load_sae_from_state_dict(state, device)

    decoder_weight = None
    if "decoder.weight" in state:
        decoder_weight = state["decoder.weight"].float().to(device).contiguous()
    else:
        logger.warning(
            "Checkpoint at %s/%s has no 'decoder.weight'; decoder-mode "
            "steering will not be available.",
            sae_release, sae_id,
        )

    del state
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info(
        "Loaded SAE (andyrdt direct): %s / %s — d_in=%d d_sae=%d k=%d "
        "decoder=%s",
        sae_release, sae_id, sae.d_in, sae.d_sae, sae.k,
        "yes" if decoder_weight is not None else "no",
    )
    return sae, decoder_weight


def load_sae_steering_context(
    config: SAESteeringConfig,
) -> SAESteeringContext:
    """Build an ``SAESteeringContext`` from the configuration.

    Steps:
      1. Load RET-style LLM config from ``config.llm_config_path`` for
         model name / dtype / layer_index.
      2. Load the SAE (encoder side) via the existing closure helper.
      3. Best-effort load of ``decoder.weight`` from the cached checkpoint.
      4. Load LLM + tokenizer.
      5. Resolve the target layer (single int only in v1).
      6. Optionally load the named-latents JSON.
    """
    if not config.llm_config_path:
        raise ValueError("SAESteeringConfig.llm_config_path is required")

    # -- LLM config (mirrors evaluation/unsupervised_steering/targets.py) ------
    from ret.config import load_config

    ret_cfg = load_config(config.llm_config_path)
    llm_cfg = ret_cfg.llm

    # -- SAE (encoder + decoder in one shot for andyrdt; otherwise fallback) ---
    if config.sae_release and "andyrdt" in config.sae_release:
        resolved_release = config.sae_release
        resolved_id = config.sae_id or f"resid_post_layer_{llm_cfg.layer_index}/trainer_0"
        sae, decoder_weight = _load_andyrdt_sae_with_decoder(
            resolved_release, resolved_id, config.device,
        )
    else:
        sae_result = load_sae_baseline(
            llm_cfg, device=config.device,
            sae_release=config.sae_release, sae_id=config.sae_id,
        )
        if sae_result is None:
            raise RuntimeError(
                f"Failed to load SAE for release={config.sae_release} "
                f"sae_id={config.sae_id}; see logs for details."
            )
        sae, resolved_release, resolved_id = sae_result
        decoder_weight = None  # non-andyrdt loaders don't expose decoder

    if config.direction_source == "decoder" and decoder_weight is None:
        raise RuntimeError(
            f"direction_source='decoder' requested but no decoder.weight "
            f"available for SAE {resolved_release}/{resolved_id}. "
            f"Switch to --direction-source encoder or pick a checkpoint "
            f"that includes a separate decoder."
        )

    # -- LLM + tokenizer (mirrors targets.py recipe) --------------------------
    dtype = getattr(torch, llm_cfg.dtype)
    load_kwargs: dict = dict(torch_dtype=dtype)
    if llm_cfg.trust_remote_code:
        load_kwargs["trust_remote_code"] = True

    if llm_cfg.device_map is not None:
        load_kwargs["device_map"] = llm_cfg.device_map
        model = AutoModelForCausalLM.from_pretrained(llm_cfg.name, **load_kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(llm_cfg.name, **load_kwargs)
        model.to(config.device)

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    tokenizer_kwargs: dict = {}
    if llm_cfg.trust_remote_code:
        tokenizer_kwargs["trust_remote_code"] = True
    tokenizer = AutoTokenizer.from_pretrained(llm_cfg.name, **tokenizer_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    from ret.extraction.hooks import _find_layers_and_embed
    layers, _embed = _find_layers_and_embed(model)

    # -- Layer resolution (single int only in v1) -----------------------------
    raw_layer_index = llm_cfg.layer_index
    if isinstance(raw_layer_index, (list, tuple)):
        raise ValueError(
            f"SAE steering v1 supports a single layer; got "
            f"layer_index={raw_layer_index!r}. Use a config with a scalar "
            f"layer_index that matches the SAE's training layer."
        )
    if isinstance(raw_layer_index, str):
        raise ValueError(
            f"layer_index string {raw_layer_index!r} not supported; "
            f"use an integer matching the SAE training layer."
        )
    layer_idx = int(raw_layer_index)

    # Sanity: SAE d_in matches LLM hidden dim
    if sae.d_in != llm_cfg.hidden_dim:
        raise ValueError(
            f"SAE d_in {sae.d_in} != LLM hidden_dim {llm_cfg.hidden_dim}; "
            f"the SAE was likely trained on a different model / layer."
        )

    # -- Optional latent name table -------------------------------------------
    sae_latent_names = None
    if config.sae_latent_names_path is not None:
        sae_latent_names = load_sae_latent_names(config.sae_latent_names_path)

    logger.info(
        "Loaded SAE-steering ctx: LLM=%s layer=%d, SAE=%s/%s d_sae=%d k=%d "
        "decoder=%s names=%s",
        llm_cfg.name, layer_idx, resolved_release, resolved_id,
        sae.d_sae, sae.k,
        "yes" if decoder_weight is not None else "no",
        f"{len(sae_latent_names)} entries" if sae_latent_names else "none",
    )

    return SAESteeringContext(
        model=model,
        tokenizer=tokenizer,
        sae=sae,
        decoder_weight=decoder_weight,
        encoder_weight=sae.encoder_weight,
        encoder_bias=sae.encoder_bias,
        b_dec=sae.b_dec,
        layer_idx=layer_idx,
        layers=layers,
        hidden_dim=int(llm_cfg.hidden_dim),
        device=config.device,
        sae_release=resolved_release,
        sae_id=resolved_id,
        sae_latent_names=sae_latent_names,
    )
