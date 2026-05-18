"""Strength sweep for the most promising hypotheses on n=11.

Loads LLM+SAE once, runs a 2-D grid (hypothesis × strength), saves each
to ``_exploration_strength_n11/<label>/str{S}/`` for inspection.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import yaml

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)s %(levelname)s  %(message)s',
                    datefmt='%H:%M:%S')
logger = logging.getLogger("sae_explore_str")

REPO_ROOT = Path(__file__).resolve().parents[4]
LLM_CFG = REPO_ROOT / "configs/GptOss20B/baselines/sae_l11.yaml"
SAE_NAMES = (
    REPO_ROOT
    / "outputs/GptOss20B/numina_math/baselines/sae_l11/interp_score/sae_latent_names.json"
)
OUT_ROOT = (
    REPO_ROOT
    / "outputs/GptOss20B/numina_math/baselines/sae_l11"
    / "sae_steering/sum_1_to_n/_exploration_strength_n11"
)

UPPER = 11
MAX_NEW = 512
SEED = 12345

# ( label, kind, latent_ids )
HYPOS = [
    ("H1_python_code",   "attract", [501, 566, 779]),
    ("H2_step_by_step",  "attract", [7775, 21632]),
    ("H4_proof_mode",    "attract", [4075, 3575, 5039]),
    ("H7_single_python", "attract", [501]),
    ("H5_avoid_digits",  "avoid",   [77349]),
]
STRENGTHS = [0.2, 0.3, 0.4]


def main():
    from evaluation.baselines.sae_unsupervised_steering.config import (
        SAESteeringConfig, SAESteeringRule,
    )
    from evaluation.baselines.sae_unsupervised_steering.hooks import SAESteeringHook
    from evaluation.baselines.sae_unsupervised_steering.loader import load_sae_steering_context
    from evaluation.baselines.sae_unsupervised_steering.pipeline import _save_sae_steered_html
    from evaluation.baselines.sae_unsupervised_steering.steerer import SAESteerer
    from evaluation.unsupervised_steering.generation import SteeredGenerator
    from ret.config import load_config

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    ret_cfg = load_config(str(LLM_CFG))
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(ret_cfg.llm.name)
    terms = " + ".join(str(k) for k in range(1, UPPER + 1))
    user_prompt = f"{terms} = ?"
    chat_prompt = tok.apply_chat_template(
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_prompt},
        ],
        tokenize=False, add_generation_prompt=True,
    )

    loader_cfg = SAESteeringConfig(
        sae_release="andyrdt/saes-gpt-oss-20b",
        sae_id="resid_post_layer_11/trainer_0",
        llm_config_path=str(LLM_CFG),
        direction_source="decoder",
        sae_latent_names_path=str(SAE_NAMES),
        rules=[], prompt=chat_prompt, device="cuda",
    )
    logger.info("Loading SAE-steering context...")
    ctx = load_sae_steering_context(loader_cfg)
    logger.info("Context ready.")

    summary = []
    overall_t0 = time.time()
    for label, kind, lids in HYPOS:
        for s in STRENGTHS:
            if kind == "attract":
                rules = [SAESteeringRule(
                    trigger_position=0, target_latents=lids, duration=MAX_NEW,
                )]
            else:
                rules = [SAESteeringRule(
                    avoid_latents=lids,
                    avoid_pre_threshold=0.0,
                    avoid_sticky_duration=10, avoid_from_step=0,
                )]
            run_dir = OUT_ROOT / label / f"str{s}"
            run_dir.mkdir(parents=True, exist_ok=True)

            cfg = SAESteeringConfig(
                sae_release="andyrdt/saes-gpt-oss-20b",
                sae_id="resid_post_layer_11/trainer_0",
                llm_config_path=str(LLM_CFG),
                direction_source="decoder",
                sae_latent_names_path=str(SAE_NAMES),
                rules=rules, strength=s, prompt=chat_prompt,
                max_new_tokens=MAX_NEW, device="cuda",
                output_dir=str(run_dir),
            )
            steerer = SAESteerer(strength=s, d_h=ctx.hidden_dim, device=ctx.device)
            hook = SAESteeringHook(
                sae=ctx.sae, decoder_weight=ctx.decoder_weight,
                encoder_weight=ctx.encoder_weight, b_dec=ctx.b_dec,
                encoder_bias=ctx.encoder_bias, rules=cfg.rules,
                steerer=steerer, direction_source=cfg.direction_source,
                device=ctx.device, top_k_diag=cfg.top_k_diag,
            )
            handle = ctx.layers[ctx.layer_idx].register_forward_hook(hook)
            torch.manual_seed(SEED)
            t0 = time.time()
            try:
                gen = SteeredGenerator(ctx.model, ctx.tokenizer, hook)
                text, diagnostics = gen.generate(
                    cfg.prompt, cfg.max_new_tokens,
                    cfg.temperature, cfg.top_p,
                )
            finally:
                handle.remove()
            dt = time.time() - t0

            (run_dir / "steered_output.txt").write_text(text, encoding="utf-8")
            with open(run_dir / "diagnostics.jsonl", "w") as f:
                for d in diagnostics:
                    f.write(json.dumps(d) + "\n")
            with open(run_dir / "config.yaml", "w") as f:
                yaml.dump(asdict(cfg), f, default_flow_style=False)
            try:
                _save_sae_steered_html(cfg, ctx, hook, diagnostics, run_dir)
            except Exception as e:
                logger.warning("HTML render failed for %s/str%s: %s", label, s, e)

            n_steered = sum(1 for d in diagnostics if d.get("was_steered"))
            head = text[:250].replace("\n", " ")
            logger.info("[%s str=%s] %.1fs gen=%d st=%d", label, s, dt, len(diagnostics), n_steered)
            logger.info("    %s...", head)
            summary.append({
                "label": label, "kind": kind, "lids": lids, "strength": s,
                "seconds": round(dt, 2), "n_gen": len(diagnostics),
                "n_steered": n_steered, "text_head": text[:500],
            })

    total = time.time() - overall_t0
    with open(OUT_ROOT / "summary.json", "w") as f:
        json.dump({
            "total_seconds": round(total, 1),
            "params": {"upper": UPPER, "max_new_tokens": MAX_NEW,
                       "seed": SEED, "user_prompt": user_prompt},
            "runs": summary,
        }, f, indent=2)
    logger.info("All done in %.1fs.", total)


if __name__ == "__main__":
    main()
