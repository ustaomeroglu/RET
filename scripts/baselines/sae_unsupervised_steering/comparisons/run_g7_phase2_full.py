#!/usr/bin/env python3
"""SAE Phase 2: full 33-prompt comparison at str=0.5 for both targets.

Targets:
  single_L91407   : [91407]
  composite_g7    : [253, 21674, 227, 12146, 115608]

Runs only:
  - 33 baselines  (SAE-side, no steering)
  - 33 single_L91407_str0.5
  - 33 composite_g7_str0.5

Skips any prompt+condition where output already exists (Phase 1's 10
prompts will not be re-run for str=0.5 since they're already on disk).

Output: outputs/.../sae_l11/sae_steering/g7_phase2_full/<prompt>/...
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
logger = logging.getLogger("sae_g7_p2")

REPO_ROOT = Path(__file__).resolve().parents[4]
LLM_CFG = REPO_ROOT / "configs/GptOss20B/baselines/sae_l11.yaml"
SAE_NAMES = (
    REPO_ROOT
    / "outputs/GptOss20B/numina_math/baselines/sae_l11/interp_score/sae_latent_names.json"
)
SAE_OUT_ROOT = (
    REPO_ROOT
    / "outputs/GptOss20B/numina_math/baselines/sae_l11"
    / "sae_steering" / "g7_phase2_full"
)
PHASE1_DIR = (
    REPO_ROOT
    / "outputs/GptOss20B/numina_math/baselines/sae_l11"
    / "sae_steering" / "g7_strength_sweep_phase1"
)

SYSTEM_PROMPT = "You are a helpful assistant."
MAX_NEW = 2048
DURATION = 600
SEED = 12345
STRENGTH = 0.5

TARGETS = {
    "single_L91407":  [91407],
    "composite_g7":   [253, 21674, 227, 12146, 115608],
}

# 33-prompt robustness set — single-sourced from
# data/math_problems/paper_robustness_set_33.json (paper Appendix L.1).
# Legacy inline labels preserved (JSON id minus the 'paper_robustness_NN_'
# prefix); canonical id order. Verified set-identical to the prior inline copy.
PROMPTS = load_robustness_33()


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


def _import_phase1_if_present(prompt_dir, prompt_tag):
    """If Phase 1 already produced this prompt's outputs, copy them in."""
    src = PHASE1_DIR / prompt_tag
    if not src.exists():
        return
    for sub in ("baseline", f"single_L91407_str{STRENGTH}", f"composite_g7_str{STRENGTH}"):
        sf = src / sub / "steered_output.txt"
        df_dir = prompt_dir / sub
        df = df_dir / "steered_output.txt"
        if sf.exists() and not df.exists():
            df_dir.mkdir(parents=True, exist_ok=True)
            df.write_text(sf.read_text())


def main():
    SAE_OUT_ROOT.mkdir(parents=True, exist_ok=True)
    logger.info("Out root: %s", SAE_OUT_ROOT)

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
    for tag, user_prompt in PROMPTS.items():
        prompt_dir = SAE_OUT_ROOT / tag
        prompt_dir.mkdir(parents=True, exist_ok=True)
        chat = build_chat_prompt(ctx.model.name_or_path, user_prompt)
        (prompt_dir / "user_prompt.txt").write_text(user_prompt)
        # Reuse Phase 1 outputs where possible
        _import_phase1_if_present(prompt_dir, tag)

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
            logger.info("[%s] baseline ok", tag)

        for tname, tlatents in TARGETS.items():
            cond = f"{tname}_str{STRENGTH}"
            cond_dir = prompt_dir / cond
            if (cond_dir / "steered_output.txt").exists():
                logger.info("[%s/%s] ok", tag, cond)
                continue
            cond_dir.mkdir(exist_ok=True)
            steerer = SAESteerer(strength=STRENGTH, d_h=ctx.hidden_dim, device=ctx.device)
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
            logger.info("[%s/%s] %.1fs (%d chars)", tag, cond, elapsed, len(text))

    total = round(time.time() - overall_t0, 1)
    summary = SAE_OUT_ROOT / "summary.json"
    with open(summary, "w") as f:
        json.dump({
            "strength": STRENGTH,
            "duration": DURATION,
            "max_new": MAX_NEW,
            "seed": SEED,
            "targets": TARGETS,
            "n_prompts": len(PROMPTS),
            "total_seconds": total,
            "runs": runs,
        }, f, indent=2)
    logger.info("Done. %d new runs in %.1fs.", len(runs), total)


if __name__ == "__main__":
    main()
