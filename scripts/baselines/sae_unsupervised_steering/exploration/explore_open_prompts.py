"""Test SAE steering on more open-ended prompts where the model has multiple
plausible continuations and steering has room to land.

We want demos where the SAE feature steers the *style* / *mode* of the
answer (Python, story, formal proof, etc.), not just the value.
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
logger = logging.getLogger("sae_open")

REPO_ROOT = Path(__file__).resolve().parents[4]
LLM_CFG = REPO_ROOT / "configs/GptOss20B/baselines/sae_l11.yaml"
SAE_NAMES = (
    REPO_ROOT
    / "outputs/GptOss20B/numina_math/baselines/sae_l11/interp_score/sae_latent_names.json"
)
OUT_ROOT = (
    REPO_ROOT
    / "outputs/GptOss20B/numina_math/baselines/sae_l11"
    / "sae_steering/sum_1_to_n/_exploration_open_prompts"
)

UPPER = 11
MAX_NEW = 512
SEED = 12345

# (prompt-tag, user-prompt-template)
PROMPTS = [
    ("explain_method", "Explain how to compute 1 + 2 + 3 + ... + {n}."),
    ("describe_sum",   "Write a short note about the sum 1 + 2 + 3 + ... + {n}."),
]

# (label, kind, latent_ids, strength)
CONDITIONS = [
    ("baseline",          "none",    None,                        0.10),
    ("steer_python",      "attract", [501, 566, 779, 217],         0.10),
    ("steer_python_str02","attract", [501, 566, 779, 217],         0.20),
    ("steer_python_str03","attract", [501, 566, 779, 217],         0.30),
    ("steer_python_str05","attract", [501, 566, 779, 217],         0.50),
    ("steer_proof",       "attract", [4075, 3575, 5039, 2578],     0.20),
    ("steer_proof_str03", "attract", [4075, 3575, 5039, 2578],     0.30),
    ("steer_step_strong", "attract", [7775, 21632, 5108, 17818],   0.30),
    ("avoid_digits_str03","avoid",   [77349],                      0.30),
    ("avoid_digits_str05","avoid",   [77349],                      0.50),
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

    ret_cfg = load_config(str(LLM_CFG))
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(ret_cfg.llm.name)

    loader_cfg = SAESteeringConfig(
        sae_release="andyrdt/saes-gpt-oss-20b",
        sae_id="resid_post_layer_11/trainer_0",
        llm_config_path=str(LLM_CFG),
        direction_source="decoder",
        sae_latent_names_path=str(SAE_NAMES),
        rules=[], prompt="dummy", device="cuda",
    )
    logger.info("Loading SAE-steering context...")
    ctx = load_sae_steering_context(loader_cfg)

    summary = []
    overall_t0 = time.time()
    for prompt_tag, prompt_tpl in PROMPTS:
        user = prompt_tpl.format(n=UPPER)
        chat_prompt = tok.apply_chat_template(
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user},
            ],
            tokenize=False, add_generation_prompt=True,
        )
        for label, kind, lids, s in CONDITIONS:
            if kind == "attract":
                rules = [SAESteeringRule(
                    trigger_position=0, target_latents=lids, duration=MAX_NEW,
                )]
            elif kind == "avoid":
                rules = [SAESteeringRule(
                    avoid_latents=lids,
                    avoid_pre_threshold=0.0,
                    avoid_sticky_duration=10, avoid_from_step=0,
                )]
            else:
                rules = []

            run_dir = OUT_ROOT / prompt_tag / label
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
                logger.warning("HTML render failed: %s", e)

            n_steered = sum(1 for d in diagnostics if d.get("was_steered"))
            head = text[:300].replace("\n", " | ")
            logger.info("[%s/%s] %.1fs gen=%d st=%d %s", prompt_tag, label, dt,
                        len(diagnostics), n_steered, head[:200])
            summary.append({
                "prompt": prompt_tag, "label": label, "lids": lids,
                "strength": s, "kind": kind, "seconds": round(dt, 2),
                "n_gen": len(diagnostics), "n_steered": n_steered,
                "text_head": text[:600],
            })

    total = time.time() - overall_t0
    with open(OUT_ROOT / "summary.json", "w") as f:
        json.dump({
            "total_seconds": round(total, 1),
            "params": {"upper": UPPER, "max_new_tokens": MAX_NEW, "seed": SEED},
            "runs": summary,
        }, f, indent=2)
    logger.info("All done in %.1fs.", total)


if __name__ == "__main__":
    main()
