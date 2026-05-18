"""Canonical SAE-steering demo on the open-ended ``Explain how to compute
1 + 2 + ... + 11`` prompt.

This is the SAE-baseline parallel of the RET ``run_sum_1_to_10.py`` launcher.
Unlike the closed-form arithmetic prompt (``1+2+...+11=?`` which has only
one plausible continuation: ``66``), this prompt invites multiple coherent
*styles* — pairing trick, arithmetic-series formula, Python code, formal
proof, etc. SAE feature directions can land cleanly on those styles.

Conditions
----------
  baseline                  — no rule, model picks pairing trick + formula
  steer_python_code         — composite of Python-related latents at str=0.7
  steer_formal_proof        — composite of proof/theorem/derive at str=0.3
  steer_step_by_step        — Firstly + Second/Finally at str=0.3
  avoid_digits              — informative null: try to suppress L77349

Output layout (mirrors RET sum_1_to_10)::

    outputs/GptOss20B/numina_math/baselines/sae_l11/sae_steering/sum_1_to_n/
        explain_n11_demo/
            summary.json
            baseline/
            steer_python_code/
            steer_formal_proof/
            steer_step_by_step/
            avoid_digits/
"""
from __future__ import annotations

import argparse
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
logger = logging.getLogger("sae_explain_n11")

REPO_ROOT = Path(__file__).resolve().parents[4]
LLM_CFG = REPO_ROOT / "configs/GptOss20B/baselines/sae_l11.yaml"
SAE_NAMES = (
    REPO_ROOT
    / "outputs/GptOss20B/numina_math/baselines/sae_l11/interp_score/sae_latent_names.json"
)
OUT_ROOT = (
    REPO_ROOT
    / "outputs/GptOss20B/numina_math/baselines/sae_l11"
    / "sae_steering/sum_1_to_n"
)


# (run-label, kind, latent_ids, strength, extra params for avoid)
def _conditions():
    return [
        ("baseline",            "none",    None,                                 0.0,  {}),
        ("steer_python_code",   "attract", [501, 566, 779, 217, 34, 616],        0.7,  {}),
        ("steer_formal_proof",  "attract", [8498, 9406, 5039, 4075, 3575],       0.3,  {}),
        ("steer_step_by_step",  "attract", [7775, 21632],                        0.4,  {}),
        ("avoid_digits",        "avoid",   [77349],                              0.5,  {"threshold": 0.0, "sticky": 10}),
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upper", type=int, default=11)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--experiment-tag", type=str, default=None)
    args = parser.parse_args()

    if args.experiment_tag is None:
        args.experiment_tag = f"explain_n{args.upper}_demo"
    out_dir = OUT_ROOT / args.experiment_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Experiment dir: %s", out_dir)

    from evaluation.baselines.sae_unsupervised_steering.config import (
        SAESteeringConfig, SAESteeringRule,
    )
    from evaluation.baselines.sae_unsupervised_steering.hooks import SAESteeringHook
    from evaluation.baselines.sae_unsupervised_steering.loader import load_sae_steering_context
    from evaluation.baselines.sae_unsupervised_steering.pipeline import _save_sae_steered_html
    from evaluation.baselines.sae_unsupervised_steering.steerer import SAESteerer
    from evaluation.unsupervised_steering.generation import SteeredGenerator
    from ret.config import load_config
    from transformers import AutoTokenizer

    ret_cfg = load_config(str(LLM_CFG))
    tok = AutoTokenizer.from_pretrained(ret_cfg.llm.name)
    user_prompt = (
        f"Explain how to compute 1 + 2 + 3 + ... + {args.upper}."
    )
    chat_prompt = tok.apply_chat_template(
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_prompt},
        ],
        tokenize=False, add_generation_prompt=True,
    )
    logger.info("User prompt: %s", user_prompt)

    loader_cfg = SAESteeringConfig(
        sae_release="andyrdt/saes-gpt-oss-20b",
        sae_id="resid_post_layer_11/trainer_0",
        llm_config_path=str(LLM_CFG),
        direction_source="decoder",
        sae_latent_names_path=str(SAE_NAMES),
        rules=[], prompt=chat_prompt, device=args.device,
    )
    logger.info("Loading SAE-steering context...")
    ctx = load_sae_steering_context(loader_cfg)
    logger.info("Context ready.")

    overall_t0 = time.time()
    summary: list[dict] = []
    for label, kind, lids, strength, extras in _conditions():
        if kind == "attract":
            rules = [SAESteeringRule(
                trigger_position=0, target_latents=lids,
                duration=args.max_new_tokens,
            )]
        elif kind == "avoid":
            rules = [SAESteeringRule(
                avoid_latents=lids,
                avoid_pre_threshold=extras["threshold"],
                avoid_sticky_duration=extras["sticky"],
                avoid_from_step=0,
            )]
        else:
            rules = []

        run_dir = out_dir / label
        run_dir.mkdir(parents=True, exist_ok=True)
        cfg = SAESteeringConfig(
            sae_release="andyrdt/saes-gpt-oss-20b",
            sae_id="resid_post_layer_11/trainer_0",
            llm_config_path=str(LLM_CFG),
            direction_source="decoder",
            sae_latent_names_path=str(SAE_NAMES),
            rules=rules, strength=strength, prompt=chat_prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature, top_p=args.top_p,
            device=args.device, output_dir=str(run_dir),
        )
        steerer = SAESteerer(strength=strength, d_h=ctx.hidden_dim, device=ctx.device)
        hook = SAESteeringHook(
            sae=ctx.sae, decoder_weight=ctx.decoder_weight,
            encoder_weight=ctx.encoder_weight, b_dec=ctx.b_dec,
            encoder_bias=ctx.encoder_bias, rules=cfg.rules,
            steerer=steerer, direction_source="decoder",
            device=ctx.device, top_k_diag=cfg.top_k_diag,
        )
        handle = ctx.layers[ctx.layer_idx].register_forward_hook(hook)
        torch.manual_seed(args.seed)
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
        prompt_z = list(hook.prompt_z_features)
        gen_z = list(hook.z_features)
        if prompt_z or gen_z:
            stacked = torch.stack(prompt_z + gen_z)
            np.save(run_dir / "z_sae_trajectory.npy", stacked.numpy())
            (run_dir / "prompt_len.txt").write_text(str(len(prompt_z)))
        with open(run_dir / "config.yaml", "w") as f:
            yaml.dump(asdict(cfg), f, default_flow_style=False)
        try:
            _save_sae_steered_html(cfg, ctx, hook, diagnostics, run_dir)
        except Exception as e:
            logger.warning("HTML render failed for %s: %s", label, e)

        n_steered = sum(1 for d in diagnostics if d.get("was_steered"))
        text_final = (
            text.split("assistantfinal", 1)[1][:300].replace("\n", " | ")
            if "assistantfinal" in text else text[:300].replace("\n", " | ")
        )
        logger.info("[%s] str=%.2f %.1fs gen=%d st=%d FINAL: %s",
                    label, strength, dt, len(diagnostics), n_steered, text_final[:300])
        summary.append({
            "label": label, "kind": kind, "lids": lids, "strength": strength,
            "extras": extras, "seconds": round(dt, 2),
            "n_gen": len(diagnostics), "n_steered": n_steered,
            "text_head": text[:1200],
        })

    total_dt = time.time() - overall_t0
    with open(out_dir / "summary.json", "w") as f:
        json.dump({
            "total_seconds": round(total_dt, 1),
            "user_prompt": user_prompt,
            "params": {
                "upper": args.upper, "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature, "top_p": args.top_p,
                "seed": args.seed,
            },
            "runs": summary,
        }, f, indent=2)
    logger.info("Done. %d runs in %.1fs. Summary: %s",
                len(summary), total_dt, out_dir / "summary.json")


if __name__ == "__main__":
    main()
