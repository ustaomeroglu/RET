#!/usr/bin/env python3
"""SAE-baseline strength sweep for G7 ("exploratory_search_and_disambiguation").

Two target sets sweep:
  SINGLE   = [L91407]                                  (existing top-1 from
                                                        comparison_behavior_change)
  COMPOSITE= [253, 21674, 227, 12146, 115608]           (G7-keyword latents)

Phase 1 (this script): sweep strength in {0.1, 0.2, 0.3, 0.5, 0.7} on a
10-prompt subset of the C30 robustness set; produces baseline + 10 steered
runs per prompt = 110 runs.

Outputs to:
  outputs/GptOss20B/numina_math/baselines/sae_l11/sae_steering/
    g7_strength_sweep_phase1/<prompt_tag>/(baseline|<target>_str<S>)/
                                 (steered_output.txt | config.yaml)
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import torch

from evaluation.baselines.sae_unsupervised_steering.robustness_prompts import load_robustness_33

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("sae_g7_sweep")

REPO_ROOT = Path(__file__).resolve().parents[4]
LLM_CFG = REPO_ROOT / "configs/GptOss20B/baselines/sae_l11.yaml"
SAE_NAMES = (
    REPO_ROOT
    / "outputs/GptOss20B/numina_math/baselines/sae_l11/interp_score/sae_latent_names.json"
)
OUT_ROOT = (
    REPO_ROOT
    / "outputs/GptOss20B/numina_math/baselines/sae_l11"
    / "sae_steering" / "g7_strength_sweep_phase1"
)

SYSTEM_PROMPT = "You are a helpful assistant."
MAX_NEW = 2048
DURATION = 600
SEED = 12345

STRENGTHS = [0.1, 0.2, 0.3, 0.5, 0.7]

TARGETS = {
    "single_L91407":  [91407],
    "composite_g7":   [253, 21674, 227, 12146, 115608],
}

# 10-prompt sweep subset spanning the robustness set. Prompt text is
# single-sourced from data/math_problems/paper_robustness_set_33.json; only
# the subset selection (these 10 legacy labels) lives here.
SWEEP_LABELS = [
    "algebra_quadratic", "prob_dice", "combinatorics", "number_theory",
    "calc_deriv", "geom_circle", "arith_decimal", "word_problem",
    "alg_radical", "creative",
]
_ROBUSTNESS_33 = load_robustness_33()
SWEEP_PROMPTS = {label: _ROBUSTNESS_33[label] for label in SWEEP_LABELS}


def build_chat_prompt(model_name, user):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    return tok.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT},
         {"role": "user", "content": user}],
        tokenize=False, add_generation_prompt=True,
    )


def _gen(ctx, chat, hook):
    from evaluation.unsupervised_steering.generation import SteeredGenerator
    handle = ctx.layers[ctx.layer_idx].register_forward_hook(hook)
    try:
        gen = SteeredGenerator(ctx.model, ctx.tokenizer, hook)
        torch.manual_seed(SEED)
        text, diags = gen.generate(chat, MAX_NEW, 0.7, 0.9)
    finally:
        handle.remove()
    return text, diags


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    logger.info("Out root: %s", OUT_ROOT)

    from evaluation.baselines.sae_unsupervised_steering.config import SAESteeringConfig, SAESteeringRule
    from evaluation.baselines.sae_unsupervised_steering.hooks import SAESteeringHook
    from evaluation.baselines.sae_unsupervised_steering.loader import load_sae_steering_context
    from evaluation.baselines.sae_unsupervised_steering.steerer import SAESteerer

    cfg0 = SAESteeringConfig(
        sae_release="andyrdt/saes-gpt-oss-20b",
        sae_id="resid_post_layer_11/trainer_0",
        llm_config_path=str(LLM_CFG), direction_source="decoder",
        sae_latent_names_path=str(SAE_NAMES),
        rules=[], prompt="dummy", device="cuda",
    )
    logger.info("Loading SAE context...")
    ctx = load_sae_steering_context(cfg0)

    runs = []
    overall_t0 = time.time()
    for tag, user_prompt in SWEEP_PROMPTS.items():
        prompt_dir = OUT_ROOT / tag
        prompt_dir.mkdir(parents=True, exist_ok=True)
        chat = build_chat_prompt(ctx.model.name_or_path, user_prompt)
        (prompt_dir / "user_prompt.txt").write_text(user_prompt)

        # Baseline
        baseline_dir = prompt_dir / "baseline"
        if not (baseline_dir / "steered_output.txt").exists():
            baseline_dir.mkdir(exist_ok=True)
            steerer = SAESteerer(strength=0.0, d_h=ctx.hidden_dim, device=ctx.device)
            hook = SAESteeringHook(
                sae=ctx.sae, decoder_weight=ctx.decoder_weight,
                encoder_weight=ctx.encoder_weight, b_dec=ctx.b_dec,
                encoder_bias=ctx.encoder_bias, rules=[], steerer=steerer,
                direction_source="decoder", device=ctx.device, top_k_diag=8,
            )
            t0 = time.time()
            text, _ = _gen(ctx, chat, hook)
            (baseline_dir / "steered_output.txt").write_text(text)
            elapsed = round(time.time() - t0, 2)
            runs.append({"prompt": tag, "kind": "baseline", "secs": elapsed,
                         "chars": len(text)})
            logger.info("[%s] baseline %.1fs (%d chars)", tag, elapsed, len(text))
        else:
            logger.info("[%s] baseline exists, skipping", tag)

        # Steered: each (target, strength)
        for tname, tlatents in TARGETS.items():
            for s in STRENGTHS:
                cond = f"{tname}_str{s}"
                cond_dir = prompt_dir / cond
                if (cond_dir / "steered_output.txt").exists():
                    continue
                cond_dir.mkdir(exist_ok=True)
                steerer = SAESteerer(strength=s, d_h=ctx.hidden_dim, device=ctx.device)
                rule = SAESteeringRule(
                    trigger_position=0, target_latents=tlatents,
                    duration=DURATION,
                )
                hook = SAESteeringHook(
                    sae=ctx.sae, decoder_weight=ctx.decoder_weight,
                    encoder_weight=ctx.encoder_weight, b_dec=ctx.b_dec,
                    encoder_bias=ctx.encoder_bias, rules=[rule], steerer=steerer,
                    direction_source="decoder", device=ctx.device, top_k_diag=8,
                )
                t0 = time.time()
                text, _ = _gen(ctx, chat, hook)
                (cond_dir / "steered_output.txt").write_text(text)
                elapsed = round(time.time() - t0, 2)
                runs.append({"prompt": tag, "kind": cond, "secs": elapsed,
                             "chars": len(text)})
                logger.info("[%s/%s] %.1fs (%d chars)",
                            tag, cond, elapsed, len(text))

    total = round(time.time() - overall_t0, 1)
    summary = OUT_ROOT / "summary.json"
    with open(summary, "w") as f:
        json.dump({
            "targets": TARGETS,
            "strengths": STRENGTHS,
            "duration": DURATION,
            "max_new": MAX_NEW,
            "seed": SEED,
            "n_prompts": len(SWEEP_PROMPTS),
            "total_seconds": total,
            "runs": runs,
        }, f, indent=2)
    logger.info("Done. %d runs in %.1fs.", len(runs), total)


if __name__ == "__main__":
    main()
