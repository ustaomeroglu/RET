"""Interpretability score: compare RET or SAE codes to text via LLM judges.

Subcommands:
    ret   — score RET clustering (requires cluster_naming/naming.txt present).
    sae   — score SAE top-K latents (requires sae_latent_names.json).

Both write:
    <output-dir>/samples/doc_XXXX/{original_text.txt, codes_sequence.txt,
                                   desc_from_codes.txt, desc_from_text.txt,
                                   judge.json}
    <output-dir>/per_sample_scores.jsonl
    <output-dir>/summary.json
    <output-dir>/config.json
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys

import torch

from evaluation.mis.runner import (
    MISConfig,
    run_combined,
    run_ret,
    run_sae,
)

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--num-samples", type=int, default=100)
    p.add_argument("--word-limit", type=int, default=80)
    p.add_argument("--model-id", type=str, default="claude-sonnet-4-20250514")
    p.add_argument("--max-concurrent", type=int, default=5)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max-sample-tokens", type=int, default=20000)
    p.add_argument("--skip-judge", action="store_true",
                   help="Run describers only; skip judge + scoring (useful for prompt verification).")


def main():
    parser = argparse.ArgumentParser(description="Interpretability score evaluation")
    sub = parser.add_subparsers(dest="method", required=True)

    p_ret = sub.add_parser("ret", help="Score RET clustering")
    p_ret.add_argument("--clustering-dir", type=str, required=True,
                       help="Path to .../clustering/.../grouped_alpha<A>/ containing "
                            "cluster_naming/naming.txt")
    _add_common(p_ret)

    p_sae = sub.add_parser("sae", help="Score SAE top-K latents")
    p_sae.add_argument("--sae-checkpoint", type=str, default=None,
                       help="Local SAE .pt (optional; default Goodfire HF).")
    p_sae.add_argument("--sae-latent-names", type=str, required=True,
                       help="Path to sae_latent_names.json from build_sae_latent_names.py")
    p_sae.add_argument("--llm-config", type=str, required=True,
                       help="YAML config providing LLM + cached_hidden_states_dir.")
    p_sae.add_argument("--top-k", type=int, default=5)
    _add_common(p_sae)

    p_both = sub.add_parser(
        "both",
        help="Score RET clustering AND SAE top-K latents on the same samples "
             "with both judges, saving all artifacts in one output dir.",
    )
    p_both.add_argument("--clustering-dir", type=str, required=True,
                        help="Path to .../grouped_alpha<A>/ containing cluster_naming/naming.txt")
    p_both.add_argument("--sae-checkpoint", type=str, default=None,
                        help="Local SAE .pt (optional; default Goodfire HF).")
    p_both.add_argument("--sae-latent-names", type=str, required=True,
                        help="Path to sae_latent_names.json from build_sae_latent_names.py")
    p_both.add_argument("--top-k", type=int, default=3,
                        help="Top-K per window (RET: by duration; SAE: by cumulative activation). Default 3.")
    p_both.add_argument("--window-size", type=int, default=50,
                        help="Tokens per window for both RET and SAE aggregations. "
                             "Default 50 = the paper MIS configuration.")
    p_both.add_argument("--variant", type=str, default="afterassistant",
                        choices=["full", "afterassistant"],
                        help="'full' = full chat trajectory; 'afterassistant' = only the model's "
                             "generated response. Default 'afterassistant' = the paper MIS configuration.")
    p_both.add_argument("--ret-format", type=str, default="window-group1",
                        choices=["window-topk", "window-group1"],
                        help="'window-topk': top-K (G.C) pairs per window by duration. "
                             "'window-group1' (default): single top Group per window, cluster names "
                             "omitted — the paper MIS configuration (deliberate handicap for RET).")
    p_both.add_argument("--copy-sae-from", type=str, default=None,
                        help="If set, copy SAE describer output, text describer output, and SAE judge result from the "
                             "corresponding sample dirs under this output-dir. Skips those API calls — only the RET "
                             "describer + RET judge are computed. Use when only the RET representation has changed.")
    p_both.add_argument("--copy-ret-from", type=str, default=None,
                        help="Mirror of --copy-sae-from for the RET side. Copies RET describer output, text describer "
                             "output, and RET judge result from the source. Only SAE describer + SAE judge are computed. "
                             "Use when only the SAE representation has changed.")
    p_both.add_argument("--sae-top-k", type=int, default=10,
                        help="SAE top-K latents per window. Default 10 = the paper MIS "
                             "configuration; pass a value to override (None falls back to --top-k).")
    _add_common(p_both)

    args = parser.parse_args()

    if args.method == "ret":
        cfg = MISConfig(
            method="ret",
            output_dir=args.output_dir,
            num_samples=args.num_samples,
            word_limit=args.word_limit,
            model_id=args.model_id,
            max_concurrent=args.max_concurrent,
            device=args.device,
            clustering_dir=args.clustering_dir,
            max_sample_tokens=args.max_sample_tokens,
            skip_judge=args.skip_judge,
        )
        summary = asyncio.run(run_ret(cfg))
    elif args.method == "sae":
        cfg = MISConfig(
            method="sae",
            output_dir=args.output_dir,
            num_samples=args.num_samples,
            word_limit=args.word_limit,
            model_id=args.model_id,
            max_concurrent=args.max_concurrent,
            device=args.device,
            sae_checkpoint=args.sae_checkpoint,
            sae_latent_names_path=args.sae_latent_names,
            top_k=args.top_k,
            llm_config_yaml=args.llm_config,
            max_sample_tokens=args.max_sample_tokens,
            skip_judge=args.skip_judge,
        )
        summary = asyncio.run(run_sae(cfg))
    elif args.method == "both":
        cfg = MISConfig(
            method="both",
            output_dir=args.output_dir,
            num_samples=args.num_samples,
            word_limit=args.word_limit,
            model_id=args.model_id,
            max_concurrent=args.max_concurrent,
            device=args.device,
            clustering_dir=args.clustering_dir,
            sae_checkpoint=args.sae_checkpoint,
            sae_latent_names_path=args.sae_latent_names,
            top_k=args.top_k,
            sae_top_k=args.sae_top_k,
            window_size=args.window_size,
            variant=args.variant,
            ret_format=args.ret_format,
            copy_sae_from=args.copy_sae_from,
            copy_ret_from=args.copy_ret_from,
            max_sample_tokens=args.max_sample_tokens,
            skip_judge=args.skip_judge,
        )
        summary = asyncio.run(run_combined(cfg))
    else:
        parser.error(f"unknown method {args.method}")
        sys.exit(1)

    print("\n=== SUMMARY ===")
    for k, v in summary.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
