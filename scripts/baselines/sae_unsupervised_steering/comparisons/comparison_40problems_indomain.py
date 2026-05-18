"""Fair (in-domain) head-to-head: SAE math-method composites vs RET clusters.

Replaces the earlier off-distribution attempt (steer toward "Python") with
**math-domain** SAE latent composites, since both RET and the SAE were
trained on numina_math.

Pairings:
  RET C35 (known_formula_retrieval)      ↔  SAE [L509, L7353, L5028, L1454]
                                              ' formula' / ' rule'-' function' /
                                              ' computed,formula' / ' Plug in equation'
  RET C63 (exhaustive_favorable_tally)   ↔  SAE [L938, L12, L1333]
                                              ' (',' sum',' +' / '+, +x' /
                                              ' found',' total',' computed'

Both SAE composites at strength=0.3 (validated probe at sum_1_to_n/_math_indomain_sweep_n11/).

Reuses the RET outputs from the previous ``comparison_40problems`` run
(those were unaffected by the SAE-target choice). Only Phase 1 (SAE)
re-runs with the new composites.
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import re
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import yaml

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)s %(levelname)s  %(message)s',
                    datefmt='%H:%M:%S')
logger = logging.getLogger("compare40_id")

REPO_ROOT = Path(__file__).resolve().parents[4]
LLM_CFG = REPO_ROOT / "configs/GptOss20B/baselines/sae_l11.yaml"
SAE_NAMES = (
    REPO_ROOT
    / "outputs/GptOss20B/numina_math/baselines/sae_l11/interp_score/sae_latent_names.json"
)

PREV_ROOT = (
    REPO_ROOT
    / "outputs/GptOss20B/numina_math/baselines/sae_l11"
    / "sae_steering/comparison_40problems"
)
OUT_ROOT = (
    REPO_ROOT
    / "outputs/GptOss20B/numina_math/baselines/sae_l11"
    / "sae_steering/comparison_40problems_indomain"
)

# In-domain math composites (validated by strength probe on n=11 sum prompt)
SAE_FORMULA_LIDS = [509, 7353, 5028, 1454]
SAE_FORMULA_STRENGTH = 0.3
SAE_ENUMERATION_LIDS = [938, 12, 1333]
SAE_ENUMERATION_STRENGTH = 0.3

MAX_NEW = 384
SEED = 12345
SYSTEM_PROMPT = "You are a helpful assistant."


# ---------------------------------------------------------------------------
# Detectors (same as previous run for direct comparability)
# ---------------------------------------------------------------------------

def _detect_styles(text: str) -> dict[str, bool]:
    t = text.lower()

    formula_patterns = [
        r"n\s*\(\s*n\s*\+\s*1\s*\)\s*/\s*2",
        r"\(\s*n\s*\+\s*1\s*\)\s*\*\s*n\s*/\s*2",
        r"n\s*\*\s*\(\s*n\s*\+\s*1\s*\)\s*/\s*2",
        r"frac\{\s*n\s*\(\s*n\s*\+\s*1\s*\)\s*\}\s*\{\s*2\s*\}",
        r"n\s*\(\s*n\s*\+\s*1\s*\)\s*\(\s*2\s*n\s*\+\s*1\s*\)\s*/\s*6",
        r"n\^2\s*\(\s*n\s*\+\s*1\s*\)\^2\s*/\s*4",
        r"\\frac\{[^}]*\}\{[^}]*\}",
        r"\bs[\\_]*n\s*=\s*n\s*\(.*?n\s*\+\s*1.*?\)\s*/\s*2",
    ]
    has_formula = any(re.search(p, t) for p in formula_patterns)

    enum_patterns = [
        r"1\s*\+\s*2\s*=\s*3.*?\+\s*3\s*=\s*6",
        r"\b\d+\s*\+\s*\d+\s*=\s*\d+\b.*?\b\d+\s*\+\s*\d+\s*=\s*\d+\b.*?\b\d+\s*\+\s*\d+\s*=\s*\d+\b",
        r"\+\s*\d+\s*=\s*\d+,\s*\+\s*\d+\s*=\s*\d+,\s*\+\s*\d+\s*=\s*\d+",
    ]
    has_enumeration = any(re.search(p, t) for p in enum_patterns)

    python_patterns = [
        r"```python", r"sum\(\s*range\s*\(",
        r"^\s*for\s+\w+\s+in\s+range\s*\(", r"\bdef\s+\w+\s*\(",
        r"\.append\b", r"^\s*import\s+\w+", r"print\s*\(",
    ]
    has_python = any(re.search(p, t, re.MULTILINE) for p in python_patterns)

    proof_patterns = [
        r"\barithmetic\s+progression\b", r"\bcommon\s+difference\b",
        r"first\s+term[s]?\s*[a-d_]{0,4}\s*[=:]",
        r"\bs[\\_]*n\s*=", r"a[_\\]+1\s*=", r"\binduction\b", r"\bproof\b.*\bby\b",
    ]
    has_proof = any(re.search(p, t) for p in proof_patterns)

    return {
        "formula": has_formula, "enumeration": has_enumeration,
        "python": has_python, "proof": has_proof,
    }


def build_chat_prompt(model_name: str, user_prompt: str) -> str:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    return tok.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT},
         {"role": "user", "content": user_prompt}],
        tokenize=False, add_generation_prompt=True,
    )


def _gen_with_hook(ctx, prompt, hook, max_new=MAX_NEW, temperature=0.7,
                    top_p=0.9, seed=SEED):
    from evaluation.unsupervised_steering.generation import SteeredGenerator
    handle = ctx.layers[ctx.layer_idx].register_forward_hook(hook)
    try:
        gen = SteeredGenerator(ctx.model, ctx.tokenizer, hook)
        torch.manual_seed(seed)
        text, diags = gen.generate(prompt, max_new, temperature, top_p)
    finally:
        handle.remove()
    return text, diags


# ---------------------------------------------------------------------------
# Phase 1: SAE in-domain
# ---------------------------------------------------------------------------

def phase_sae_indomain(problems, save_dir):
    from evaluation.baselines.sae_unsupervised_steering.config import (
        SAESteeringConfig, SAESteeringRule,
    )
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
    sae_ctx = load_sae_steering_context(cfg0)

    results = {}
    for prob in problems:
        pid = prob["id"]
        chat = build_chat_prompt(sae_ctx.model.name_or_path, prob["user"])
        results[pid] = {"user_prompt": prob["user"], "category": prob["category"]}

        # Baseline (no rule)
        steerer = SAESteerer(strength=0.0, d_h=sae_ctx.hidden_dim, device=sae_ctx.device)
        hook = SAESteeringHook(
            sae=sae_ctx.sae, decoder_weight=sae_ctx.decoder_weight,
            encoder_weight=sae_ctx.encoder_weight, b_dec=sae_ctx.b_dec,
            encoder_bias=sae_ctx.encoder_bias, rules=[], steerer=steerer,
            direction_source="decoder", device=sae_ctx.device, top_k_diag=8,
        )
        t0 = time.time()
        text, _ = _gen_with_hook(sae_ctx, chat, hook)
        results[pid]["baseline"] = text
        results[pid]["baseline_secs"] = round(time.time() - t0, 2)

        # SAE formula (in-domain composite)
        steerer_f = SAESteerer(strength=SAE_FORMULA_STRENGTH,
                                d_h=sae_ctx.hidden_dim, device=sae_ctx.device)
        rule_f = SAESteeringRule(
            trigger_position=0, target_latents=SAE_FORMULA_LIDS, duration=MAX_NEW,
        )
        hook_f = SAESteeringHook(
            sae=sae_ctx.sae, decoder_weight=sae_ctx.decoder_weight,
            encoder_weight=sae_ctx.encoder_weight, b_dec=sae_ctx.b_dec,
            encoder_bias=sae_ctx.encoder_bias, rules=[rule_f], steerer=steerer_f,
            direction_source="decoder", device=sae_ctx.device, top_k_diag=8,
        )
        t0 = time.time()
        text_f, _ = _gen_with_hook(sae_ctx, chat, hook_f)
        results[pid]["sae_formula"] = text_f
        results[pid]["sae_formula_secs"] = round(time.time() - t0, 2)

        # SAE enumeration (in-domain composite)
        steerer_e = SAESteerer(strength=SAE_ENUMERATION_STRENGTH,
                                d_h=sae_ctx.hidden_dim, device=sae_ctx.device)
        rule_e = SAESteeringRule(
            trigger_position=0, target_latents=SAE_ENUMERATION_LIDS, duration=MAX_NEW,
        )
        hook_e = SAESteeringHook(
            sae=sae_ctx.sae, decoder_weight=sae_ctx.decoder_weight,
            encoder_weight=sae_ctx.encoder_weight, b_dec=sae_ctx.b_dec,
            encoder_bias=sae_ctx.encoder_bias, rules=[rule_e], steerer=steerer_e,
            direction_source="decoder", device=sae_ctx.device, top_k_diag=8,
        )
        t0 = time.time()
        text_e, _ = _gen_with_hook(sae_ctx, chat, hook_e)
        results[pid]["sae_enumeration"] = text_e
        results[pid]["sae_enumeration_secs"] = round(time.time() - t0, 2)

        logger.info("[SAE %s] base/formula/enum: %.1f/%.1f/%.1fs",
                    pid, results[pid]["baseline_secs"],
                    results[pid]["sae_formula_secs"],
                    results[pid]["sae_enumeration_secs"])

    with open(save_dir / "phase1_sae_indomain.json", "w") as f:
        json.dump(results, f, indent=2)
    del sae_ctx, hook, hook_f, hook_e
    gc.collect()
    torch.cuda.empty_cache()
    return results


# ---------------------------------------------------------------------------
# Score
# ---------------------------------------------------------------------------

def phase_score(results, save_dir):
    detected = {}
    for pid, r in results.items():
        detected[pid] = {}
        for cond in ["baseline", "ret_formula", "ret_enumeration",
                     "sae_formula", "sae_enumeration"]:
            text = r.get(cond, "")
            detected[pid][cond] = _detect_styles(text)

    methods = {
        "ret_formula":      ("formula",     "ret_formula"),
        "ret_enumeration":  ("enumeration", "ret_enumeration"),
        "sae_formula":      ("formula",     "sae_formula"),
        "sae_enumeration":  ("enumeration", "sae_enumeration"),
    }
    per_method = {m: {"success": 0, "no_op": 0, "harmful": 0, "failed": 0,
                      "induced": [], "no_op_pids": [], "harmful_pids": [], "failed_pids": []}
                  for m in methods}

    for pid in detected:
        for m, (style, cond) in methods.items():
            in_b = detected[pid]["baseline"].get(style, False)
            in_s = detected[pid][cond].get(style, False)
            if in_s and not in_b:
                per_method[m]["success"] += 1
                per_method[m]["induced"].append(pid)
            elif in_s and in_b:
                per_method[m]["no_op"] += 1
                per_method[m]["no_op_pids"].append(pid)
            elif (not in_s) and in_b:
                per_method[m]["harmful"] += 1
                per_method[m]["harmful_pids"].append(pid)
            else:
                per_method[m]["failed"] += 1
                per_method[m]["failed_pids"].append(pid)

    # Aggregate by side
    ret_methods = ["ret_formula", "ret_enumeration"]
    sae_methods = ["sae_formula", "sae_enumeration"]

    def _agg(ms):
        s = sum(per_method[m]["success"] for m in ms)
        n = sum(sum(per_method[m][k] for k in ("success","no_op","harmful","failed")) for m in ms)
        return s, n, (s/n if n else 0.0)

    ret_s, ret_n, ret_rate = _agg(ret_methods)
    sae_s, sae_n, sae_rate = _agg(sae_methods)

    # Pair-wise (same target, different method): formula RET vs formula SAE
    pairs = {
        "formula": ("ret_formula", "sae_formula"),
        "enumeration": ("ret_enumeration", "sae_enumeration"),
    }
    pair_summary = {}
    for tgt, (em, sm) in pairs.items():
        em_succ = set(per_method[em]["induced"])
        sm_succ = set(per_method[sm]["induced"])
        em_pres = em_succ | set(per_method[em]["no_op_pids"])
        sm_pres = sm_succ | set(per_method[sm]["no_op_pids"])
        pair_summary[tgt] = {
            "ret_success": len(em_succ), "sae_success": len(sm_succ),
            "ret_target_present": len(em_pres), "sae_target_present": len(sm_pres),
            "both_success": len(em_succ & sm_succ),
            "ret_only_success": len(em_succ - sm_succ),
            "sae_only_success": len(sm_succ - em_succ),
            "neither_success": 40 - len(em_succ | sm_succ),
        }

    ret_any = set(); sae_any = set()
    for pid in detected:
        if any(pid in per_method[m]["induced"] for m in ret_methods):
            ret_any.add(pid)
        if any(pid in per_method[m]["induced"] for m in sae_methods):
            sae_any.add(pid)

    summary = {
        "params": {
            "max_new_tokens": MAX_NEW, "seed": SEED,
            "sae_formula_lids": SAE_FORMULA_LIDS, "sae_formula_strength": SAE_FORMULA_STRENGTH,
            "sae_enumeration_lids": SAE_ENUMERATION_LIDS, "sae_enumeration_strength": SAE_ENUMERATION_STRENGTH,
        },
        "n_problems": len(results),
        "per_method": per_method,
        "pair_summary": pair_summary,
        "aggregate": {
            "ret_total_success": ret_s, "ret_total_attempts": ret_n, "ret_success_rate": ret_rate,
            "sae_total_success": sae_s, "sae_total_attempts": sae_n, "sae_success_rate": sae_rate,
            "ret_any_method_induced": len(ret_any), "sae_any_method_induced": len(sae_any),
        },
        "detected": detected,
    }
    with open(save_dir / "comparison_summary_indomain.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", default="all", choices=["all", "sae", "score"])
    args = parser.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    # Reuse problems and RET runs from previous comparison
    problems = json.load(open(PREV_ROOT / "problems.json"))
    prev_results = json.load(open(PREV_ROOT / "phase2_ret_added.json"))
    logger.info("Loaded %d problems and prev RET results", len(problems))

    # Save problems.json mirror
    with open(OUT_ROOT / "problems.json", "w") as f:
        json.dump(problems, f, indent=2)

    sae_results = {}
    if args.phase in ("all", "sae"):
        logger.info("=== Phase 1 (in-domain SAE) ===")
        t0 = time.time()
        sae_results = phase_sae_indomain(problems, OUT_ROOT)
        logger.info("Phase 1 done in %.1fs", time.time() - t0)
    elif (OUT_ROOT / "phase1_sae_indomain.json").exists():
        sae_results = json.load(open(OUT_ROOT / "phase1_sae_indomain.json"))

    # Merge: keep ret_formula and ret_enumeration from prev_results, replace SAE columns
    merged = {}
    for pid in sae_results:
        merged[pid] = dict(sae_results[pid])
        merged[pid]["ret_formula"] = prev_results[pid].get("ret_formula", "")
        merged[pid]["ret_enumeration"] = prev_results[pid].get("ret_enumeration", "")
    with open(OUT_ROOT / "all_outputs_indomain.json", "w") as f:
        json.dump(merged, f, indent=2)

    if args.phase in ("all", "score"):
        logger.info("=== Phase 3: Score ===")
        summary = phase_score(merged, OUT_ROOT)
        agg = summary["aggregate"]
        logger.info("=== RESULTS (in-domain SAE vs RET) ===")
        for m, v in summary["per_method"].items():
            logger.info("  %-22s success=%2d  no_op=%2d  harmful=%2d  failed=%2d",
                        m, v["success"], v["no_op"], v["harmful"], v["failed"])
        logger.info("RET: %d/%d (%.1f%%); any-method induced on %d/40",
                    agg["ret_total_success"], agg["ret_total_attempts"],
                    100*agg["ret_success_rate"], agg["ret_any_method_induced"])
        logger.info("SAE: %d/%d (%.1f%%); any-method induced on %d/40",
                    agg["sae_total_success"], agg["sae_total_attempts"],
                    100*agg["sae_success_rate"], agg["sae_any_method_induced"])
        logger.info("Pair-wise:")
        for tgt, p in summary["pair_summary"].items():
            logger.info("  %s: RET succ=%d  SAE succ=%d  both=%d  RET-only=%d  SAE-only=%d  neither=%d",
                        tgt, p["ret_success"], p["sae_success"], p["both_success"],
                        p["ret_only_success"], p["sae_only_success"], p["neither_success"])


if __name__ == "__main__":
    main()
