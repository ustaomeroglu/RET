"""Build a non-LLM SAE latent-naming table.

For each SAE latent, records the top-K distinct token strings that most
strongly activate it, computed on the train split (never the test split).
Saves to <output_json>. Use the produced JSON with
``scripts/mis/run_mis.py sae --sae-latent-names <json>``.

Usage
-----
    python scripts/mis/build_sae_latent_names.py \\
        --llm-config configs/GptOss20B/baselines/sae_l11.yaml \\
        --top-k 5 \\
        --max-tokens 2000000 \\
        --output-dir outputs/GptOss20B/numina_math/baselines/sae_l11/interp_score

Flags
-----
--sae-checkpoint     local .pt (optional; default downloads Goodfire HF)
--max-tokens         cap tokens streamed for naming (speed vs quality)
--batch-size         dataloader batch size
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch

from evaluation.clustering.config import build_dataloader, load_config_with_overlay
from evaluation.mis.naming import (
    build_sae_latent_names,
    save_sae_latent_names,
)
from evaluation.probing.sae_utils import load_sae_encoder

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    p = argparse.ArgumentParser(description="Build top-K activating-token naming for SAE latents.")
    p.add_argument("--llm-config", type=str, required=True,
                   help="YAML config that supplies LLM + cached_hidden_states_dir (train split).")
    p.add_argument("--sae-checkpoint", type=str, default=None,
                   help="Local SAE .pt (optional; default Goodfire HF).")
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--max-tokens", type=int, default=2_000_000)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    config = load_config_with_overlay(args.llm_config)
    logger.info("LLM: %s  layer=%s  cache=%s", config.llm.name,
                config.llm.layer_index, config.data.cached_hidden_states_dir)

    sae, sae_cfg = load_sae_encoder(checkpoint_path=args.sae_checkpoint, device=args.device)
    logger.info("SAE loaded: d_in=%d d_sae=%d k=%d source=%s",
                sae_cfg["d_in"], sae_cfg["d_sae"], sae_cfg["k"], sae_cfg["source"])

    # train split
    loader = build_dataloader(
        config, split="train", batch_size=args.batch_size,
        seed=42, shuffle=False, drop_last=False,
    )

    from transformers import AutoTokenizer
    tok_kwargs = {"trust_remote_code": True} if config.llm.trust_remote_code else {}
    tokenizer = AutoTokenizer.from_pretrained(config.llm.name, **tok_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    latent_names = build_sae_latent_names(
        sae=sae,
        hidden_loader=loader,
        tokenizer=tokenizer,
        device=args.device,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "sae_latent_names.json"
    save_sae_latent_names(
        latent_names, out_path,
        meta={
            "sae_source": sae_cfg["source"],
            "d_sae": int(sae_cfg["d_sae"]),
            "sae_k": int(sae_cfg["k"]),
            "llm": config.llm.name,
            "layer": config.llm.layer_index,
            "split": "train",
            "top_k_per_latent": args.top_k,
            "max_tokens_streamed": args.max_tokens,
        },
    )
    logger.info("Saved %d latent names -> %s", len(latent_names), out_path)


if __name__ == "__main__":
    main()
