"""Exploration: test several composite-latent steering hypotheses on sum 1..11.

Loads the LLM + SAE once, runs every hypothesis at one strength, saves to
an ``_exploration/`` subfolder for inspection. Once a winner is identified,
``run_sum_1_to_n.py`` (with custom latents) saves in the canonical layout.

Hypotheses (composite directions):
  H1 python_code      = [L501, L566, L779]    # Python language + return + scipy
  H2 step_by_step     = [L7775, L21632]       # Firstly + Second/Finally
  H3 word_numbers     = [L40847, L91210, L90309]  # eleven/twelve word forms
  H4 proof_mode       = [L4075, L3575, L5039] # theorem + derive + proof
  H5 avoid_digits     = avoid L77349 (top digit detector)
  H6 avoid_structure  = avoid L10990 (analysis-channel structure)
  H7 single_python    = [L501] only — control to compare composite vs single
  H8 verbose_explain  = [L7775, L21632, L4075] — Firstly + ordinals + theorem
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("sae_explore")

REPO_ROOT = Path(__file__).resolve().parents[4]
LLM_CFG = REPO_ROOT / "configs/GptOss20B/baselines/sae_l11.yaml"
SAE_NAMES = (
    REPO_ROOT
    / "outputs/GptOss20B/numina_math/baselines/sae_l11/interp_score/sae_latent_names.json"
)
OUT_ROOT = (
    REPO_ROOT
    / "outputs/GptOss20B/numina_math/baselines/sae_l11"
    / "sae_steering/sum_1_to_n/_exploration_n11"
)

UPPER = 11
STRENGTH = 0.15
MAX_NEW = 1024
SEED = 12345

HYPOTHESES = [
    ("baseline",         "none",   None),
    ("H1_python_code",   "attract", [501, 566, 779]),
    ("H2_step_by_step",  "attract", [7775, 21632]),
    ("H3_word_numbers",  "attract", [40847, 91210, 90309]),
    ("H4_proof_mode",    "attract", [4075, 3575, 5039]),
    ("H7_single_python", "attract", [501]),
    ("H8_verbose",       "attract", [7775, 21632, 4075]),
    ("H5_avoid_digits",  "avoid",   [77349]),
    ("H6_avoid_structure","avoid",  [10990]),
]


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

    # Build the chat-templated prompt
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

    # Load context once
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
    for label, kind, lids in HYPOTHESES:
        if kind == "none":
            rules: list = []
        elif kind == "attract":
            rules = [SAESteeringRule(
                trigger_position=0, target_latents=lids, duration=MAX_NEW,
            )]
        else:  # avoid
            rules = [SAESteeringRule(
                avoid_latents=lids,
                avoid_pre_threshold=0.0,  # fire when feature pre-act is positive
                avoid_sticky_duration=10,
                avoid_from_step=0,
            )]

        out_dir = OUT_ROOT / label
        out_dir.mkdir(parents=True, exist_ok=True)

        cfg = SAESteeringConfig(
            sae_release="andyrdt/saes-gpt-oss-20b",
            sae_id="resid_post_layer_11/trainer_0",
            llm_config_path=str(LLM_CFG),
            direction_source="decoder",
            sae_latent_names_path=str(SAE_NAMES),
            rules=rules, strength=STRENGTH, prompt=chat_prompt,
            max_new_tokens=MAX_NEW, device="cuda",
            output_dir=str(out_dir),
        )

        # Build hook
        steerer = SAESteerer(strength=cfg.strength, d_h=ctx.hidden_dim, device=ctx.device)
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

        # Save
        (out_dir / "steered_output.txt").write_text(text, encoding="utf-8")
        with open(out_dir / "diagnostics.jsonl", "w") as f:
            for d in diagnostics:
                f.write(json.dumps(d) + "\n")
        prompt_z = list(hook.prompt_z_features)
        gen_z = list(hook.z_features)
        if prompt_z or gen_z:
            stacked = torch.stack(prompt_z + gen_z)
            np.save(out_dir / "z_sae_trajectory.npy", stacked.numpy())
            (out_dir / "prompt_len.txt").write_text(str(len(prompt_z)))
        with open(out_dir / "config.yaml", "w") as f:
            yaml.dump(asdict(cfg), f, default_flow_style=False)
        try:
            _save_sae_steered_html(cfg, ctx, hook, diagnostics, out_dir)
        except Exception as e:
            logger.warning("HTML render failed for %s: %s", label, e)

        n_steered = sum(1 for d in diagnostics if d.get("was_steered"))
        logger.info("[%s] %.1fs gen=%d steered=%d", label, dt, len(diagnostics), n_steered)
        # First 200 chars of generation, after the assistantfinal marker if any
        head = text[:300].replace("\n", " ")
        logger.info("    text: %s...", head)
        summary.append({
            "label": label, "kind": kind, "lids": lids,
            "seconds": round(dt, 2),
            "n_gen": len(diagnostics), "n_steered": n_steered,
            "text_head": text[:500],
        })

    total = time.time() - overall_t0
    with open(OUT_ROOT / "summary.json", "w") as f:
        json.dump({
            "total_seconds": round(total, 1),
            "params": {"upper": UPPER, "strength": STRENGTH, "max_new_tokens": MAX_NEW,
                       "seed": SEED, "user_prompt": user_prompt},
            "hypotheses": HYPOTHESES,
            "runs": summary,
        }, f, indent=2)
    logger.info("All done in %.1fs. Summary at %s", total, OUT_ROOT / "summary.json")


if __name__ == "__main__":
    main()
