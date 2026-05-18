#!/usr/bin/env python3
"""SAE-steering parallel of ``scripts/unsupervised_steering/run_sum_1_to_10.py``.

Direct-enumeration vs Gauss-formula steering on the prompt
"1 + 2 + ... + N = ?", driven by single-feature SAE latents.

RET cluster targets and their SAE-latent counterparts:

    Direct enumeration  : C63 exhaustive_favorable_tally
                          → L938  ['(', ' (', ' sum', ' +', ')+']
    Formula retrieval   : C35 known_formula_retrieval
                          → L509  [' formula', 'Formula', ' Formula',
                                    ' formulas', 'formula']

Conditions matched to the RET launcher (5 runs per parameter set):

    baseline                        — no rule
    steer_L938_addition             — attract toward addition operator
    steer_L509_formula              — attract toward formula word
    avoid_L938_addition             — repel from addition operator
    avoid_L509_formula              — repel from formula word

Output layout mirrors the RET sum_1_to_10 tree::

    outputs/GptOss20B/numina_math/baselines/sae_l11/sae_steering/sum_1_to_n/
        explicit_n11_str0.05_dur1024_st20/
            summary.json
            baseline/
            steer_L938_addition/
            steer_L509_formula/
            avoid_L938_addition/
            avoid_L509_formula/
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("sae_sum_1_to_n")

REPO_ROOT = Path(__file__).resolve().parents[4]

LLM_CONFIG_PATH = REPO_ROOT / "configs/GptOss20B/baselines/sae_l11.yaml"
SAE_LATENT_NAMES = (
    REPO_ROOT
    / "outputs/GptOss20B/numina_math/baselines/sae_l11"
    / "interp_score/sae_latent_names.json"
)

OUT_ROOT = (
    REPO_ROOT
    / "outputs/GptOss20B/numina_math/baselines/sae_l11"
    / "sae_steering/sum_1_to_n"
)

# SAE latent counterparts of the RET targets.
TARGET_ENUMERATION = 938   # ['(', ' (', ' sum', ' +', ')+']
TARGET_FORMULA = 509       # [' formula', 'Formula', ' Formula', ...]

SYSTEM_PROMPT = "You are a helpful assistant."


def make_user_prompt(style: str, n: int) -> str:
    if style == "question":
        return f"What is the sum of the integers from 1 to {n}?"
    if style == "explicit":
        terms = " + ".join(str(k) for k in range(1, n + 1))
        return f"{terms} = ?"
    raise ValueError(
        f"Unknown prompt-style: {style!r}. Expected 'question' or 'explicit'."
    )


def build_chat_prompt(model_name: str, user_prompt: str) -> str:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    return tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )


def make_config(
    *, prompt: str, rules: list, strength: float, max_new_tokens: int,
    output_dir: Path, device: str, temperature: float, top_p: float,
):
    from evaluation.baselines.sae_unsupervised_steering.config import SAESteeringConfig
    return SAESteeringConfig(
        sae_release="andyrdt/saes-gpt-oss-20b",
        sae_id="resid_post_layer_11/trainer_0",
        llm_config_path=str(LLM_CONFIG_PATH),
        direction_source="decoder",
        sae_latent_names_path=str(SAE_LATENT_NAMES),
        rules=rules,
        strength=strength,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        device=device,
        output_dir=str(output_dir),
    )


def run_one(ctx, config, label: str) -> dict:
    """Generate one run with the supplied config, save artifacts, return summary."""
    from collections import Counter

    from evaluation.baselines.sae_unsupervised_steering.hooks import SAESteeringHook
    from evaluation.baselines.sae_unsupervised_steering.pipeline import _save_sae_steered_html
    from evaluation.baselines.sae_unsupervised_steering.steerer import SAESteerer
    from evaluation.unsupervised_steering.generation import SteeredGenerator

    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    steerer = SAESteerer(
        strength=config.strength, d_h=ctx.hidden_dim, device=config.device,
    )
    hook = SAESteeringHook(
        sae=ctx.sae, decoder_weight=ctx.decoder_weight,
        encoder_weight=ctx.encoder_weight, b_dec=ctx.b_dec,
        encoder_bias=ctx.encoder_bias, rules=config.rules,
        steerer=steerer, direction_source=config.direction_source,
        device=ctx.device, top_k_diag=config.top_k_diag,
    )
    handle = ctx.layers[ctx.layer_idx].register_forward_hook(hook)
    t0 = time.time()
    try:
        gen = SteeredGenerator(ctx.model, ctx.tokenizer, hook)
        text, diagnostics = gen.generate(
            config.prompt, config.max_new_tokens,
            config.temperature, config.top_p,
        )
    finally:
        handle.remove()
    dt = time.time() - t0

    (out_dir / "steered_output.txt").write_text(text, encoding="utf-8")
    with open(out_dir / "diagnostics.jsonl", "w") as f:
        for d in diagnostics:
            f.write(json.dumps(d) + "\n")
    cfg_dict = asdict(config)
    with open(out_dir / "config.yaml", "w") as f:
        yaml.dump(cfg_dict, f, default_flow_style=False)

    prompt_z = list(hook.prompt_z_features)
    gen_z = list(hook.z_features)
    if prompt_z or gen_z:
        stacked = torch.stack(prompt_z + gen_z)
        np.save(out_dir / "z_sae_trajectory.npy", stacked.numpy())
        (out_dir / "prompt_len.txt").write_text(str(len(prompt_z)))

    try:
        _save_sae_steered_html(config, ctx, hook, diagnostics, out_dir)
    except Exception as e:
        logger.warning("HTML render failed for %s: %s", label, e)

    n_steered = sum(1 for d in diagnostics if d.get("was_steered"))
    post_top1 = Counter(
        d["post_top1_lid"] for d in diagnostics if d.get("was_steered")
    )
    avoid_fires = Counter()
    for d in diagnostics:
        if d.get("rule_kind") == "avoid" and d.get("was_steered"):
            avoid_fires[d.get("active_rule", -1)] += 1

    logger.info(
        "[%s] %.1fs gen=%d steered=%d post_top1_top3=%s -> %s",
        label, dt, len(diagnostics), n_steered,
        dict(post_top1.most_common(3)),
        out_dir.relative_to(REPO_ROOT),
    )
    return {
        "label": label,
        "seconds": round(dt, 2),
        "n_gen": len(diagnostics),
        "n_steered": n_steered,
        "post_top1_hist_top5": dict(post_top1.most_common(5)),
        "avoid_fires_per_rule": dict(avoid_fires),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument(
        "--strength", type=float, default=0.05,
        help="δh = strength·‖h‖·u (decoder unit dir). Effective α ≈ "
             "strength·1200 in pre-act units; 0.05 → α≈60.",
    )
    parser.add_argument(
        "--avoid-pre-threshold", type=float, default=5.0,
        help="Avoid-gate fires when max(pre_act over avoid_latents) >= T.",
    )
    parser.add_argument(
        "--avoid-sticky-duration", type=int, default=20,
        help="Tokens to keep repelling after gate closes.",
    )
    parser.add_argument(
        "--attractor-duration", type=int, default=2048,
        help="Tokens to keep attracting once trigger fires.",
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument(
        "--target-enumeration", type=int, default=TARGET_ENUMERATION,
        help=f"SAE latent for the addition/sum path (default L{TARGET_ENUMERATION}).",
    )
    parser.add_argument(
        "--target-formula", type=int, default=TARGET_FORMULA,
        help=f"SAE latent for the formula path (default L{TARGET_FORMULA}).",
    )
    parser.add_argument(
        "--prompt-style", choices=("question", "explicit"),
        default="explicit",
    )
    parser.add_argument(
        "--upper", type=int, default=11,
        help="Upper bound N of the sum 1..N.",
    )
    parser.add_argument(
        "--experiment-tag", type=str, default=None,
    )
    args = parser.parse_args()

    if args.experiment_tag is None:
        dur = min(args.attractor_duration, args.max_new_tokens)
        args.experiment_tag = (
            f"{args.prompt_style}_n{args.upper}_str{args.strength}"
            f"_dur{dur}_st{args.avoid_sticky_duration}"
        )
    out_dir = OUT_ROOT / args.experiment_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Experiment dir: %s", out_dir)

    user_prompt = make_user_prompt(args.prompt_style, args.upper)
    logger.info("User prompt: %s", user_prompt)

    from ret.config import load_config
    ret_cfg = load_config(str(LLM_CONFIG_PATH))
    chat_prompt = build_chat_prompt(ret_cfg.llm.name, user_prompt)
    logger.info("Templated prompt:\n%s", chat_prompt)

    from evaluation.baselines.sae_unsupervised_steering.config import SAESteeringRule
    from evaluation.baselines.sae_unsupervised_steering.loader import load_sae_steering_context

    from evaluation.baselines.sae_unsupervised_steering.config import SAESteeringConfig
    loader_cfg = SAESteeringConfig(
        sae_release="andyrdt/saes-gpt-oss-20b",
        sae_id="resid_post_layer_11/trainer_0",
        llm_config_path=str(LLM_CONFIG_PATH),
        direction_source="decoder",
        sae_latent_names_path=str(SAE_LATENT_NAMES),
        rules=[], prompt=chat_prompt, device=args.device,
    )
    logger.info("Loading SAE steering context...")
    ctx = load_sae_steering_context(loader_cfg)
    logger.info(
        "Context ready: layer=%d d_sae=%d k=%d",
        ctx.layer_idx, ctx.sae.d_sae, ctx.sae.k,
    )

    summary: list[dict] = []
    overall_t0 = time.time()
    dur = min(args.attractor_duration, args.max_new_tokens)

    # ---------- baseline ----------
    torch.manual_seed(args.seed)
    cfg = make_config(
        prompt=chat_prompt, rules=[], strength=args.strength,
        max_new_tokens=args.max_new_tokens,
        output_dir=out_dir / "baseline", device=args.device,
        temperature=args.temperature, top_p=args.top_p,
    )
    summary.append(run_one(ctx, cfg, "baseline"))

    # ---------- steer_L{enum}_addition ----------
    torch.manual_seed(args.seed)
    cfg = make_config(
        prompt=chat_prompt,
        rules=[SAESteeringRule(
            trigger_position=0,
            target_latents=[args.target_enumeration],
            duration=dur,
        )],
        strength=args.strength,
        max_new_tokens=args.max_new_tokens,
        output_dir=out_dir / f"steer_L{args.target_enumeration}_addition",
        device=args.device, temperature=args.temperature, top_p=args.top_p,
    )
    summary.append(run_one(ctx, cfg, f"steer_L{args.target_enumeration}_addition"))

    # ---------- steer_L{formula}_formula ----------
    torch.manual_seed(args.seed)
    cfg = make_config(
        prompt=chat_prompt,
        rules=[SAESteeringRule(
            trigger_position=0,
            target_latents=[args.target_formula],
            duration=dur,
        )],
        strength=args.strength,
        max_new_tokens=args.max_new_tokens,
        output_dir=out_dir / f"steer_L{args.target_formula}_formula",
        device=args.device, temperature=args.temperature, top_p=args.top_p,
    )
    summary.append(run_one(ctx, cfg, f"steer_L{args.target_formula}_formula"))

    # ---------- avoid_L{enum}_addition ----------
    torch.manual_seed(args.seed)
    cfg = make_config(
        prompt=chat_prompt,
        rules=[SAESteeringRule(
            avoid_latents=[args.target_enumeration],
            avoid_pre_threshold=args.avoid_pre_threshold,
            avoid_sticky_duration=args.avoid_sticky_duration,
            avoid_from_step=0,
        )],
        strength=args.strength,
        max_new_tokens=args.max_new_tokens,
        output_dir=out_dir / f"avoid_L{args.target_enumeration}_addition",
        device=args.device, temperature=args.temperature, top_p=args.top_p,
    )
    summary.append(run_one(ctx, cfg, f"avoid_L{args.target_enumeration}_addition"))

    # ---------- avoid_L{formula}_formula ----------
    torch.manual_seed(args.seed)
    cfg = make_config(
        prompt=chat_prompt,
        rules=[SAESteeringRule(
            avoid_latents=[args.target_formula],
            avoid_pre_threshold=args.avoid_pre_threshold,
            avoid_sticky_duration=args.avoid_sticky_duration,
            avoid_from_step=0,
        )],
        strength=args.strength,
        max_new_tokens=args.max_new_tokens,
        output_dir=out_dir / f"avoid_L{args.target_formula}_formula",
        device=args.device, temperature=args.temperature, top_p=args.top_p,
    )
    summary.append(run_one(ctx, cfg, f"avoid_L{args.target_formula}_formula"))

    # ---------- summary ----------
    total_dt = time.time() - overall_t0
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(
            {
                "total_seconds": round(total_dt, 1),
                "params": {
                    "strength": args.strength,
                    "avoid_pre_threshold": args.avoid_pre_threshold,
                    "avoid_sticky_duration": args.avoid_sticky_duration,
                    "attractor_duration": dur,
                    "max_new_tokens": args.max_new_tokens,
                    "target_enumeration": args.target_enumeration,
                    "target_formula": args.target_formula,
                    "prompt_style": args.prompt_style,
                    "upper": args.upper,
                    "direction_source": "decoder",
                },
                "user_prompt": user_prompt,
                "runs": summary,
            },
            f, indent=2,
        )
    logger.info(
        "Done. %d runs in %.1fs. Summary: %s",
        len(summary), total_dt, summary_path,
    )


if __name__ == "__main__":
    main()
