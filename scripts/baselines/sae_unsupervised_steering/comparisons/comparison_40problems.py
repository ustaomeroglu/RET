"""40-problem head-to-head comparison: SAE-baseline steering vs RET steering.

For each of 40 math problems we run five conditions:

  baseline                  — no steering, natural model output
  RET steer_C35_formula     — attract toward RET cluster 35 (known formula)
  RET steer_C63_enumeration — attract toward RET cluster 63 (exhaustive sum)
  SAE steer_python          — composite Python latents, strength=0.7
  SAE steer_proof           — composite proof latents, strength=0.3

Each output is then scored by keyword-based detectors for four styles
(formula, enumeration, python, proof), and we compute a success rate per
steering method:

    success(method) = fraction of problems where the targeted style
                       appeared in steered AND not in baseline.

Phases (sequential to limit memory pressure on a single H100 80GB):

  Phase 1: load SAE context, run {baseline, sae_python, sae_proof} for all 40 problems
  Phase 2: free SAE, load RET context, run {ret_formula, ret_enum} for all 40 problems
  Phase 3: score everything, save summary

The comparison output lives at:
  outputs/GptOss20B/numina_math/baselines/sae_l11/sae_steering/comparison_40problems/
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

import numpy as np
import torch
import yaml

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)s %(levelname)s  %(message)s',
                    datefmt='%H:%M:%S')
logger = logging.getLogger("compare40")

REPO_ROOT = Path(__file__).resolve().parents[4]
LLM_CFG = REPO_ROOT / "configs/GptOss20B/baselines/sae_l11.yaml"
SAE_NAMES = (
    REPO_ROOT
    / "outputs/GptOss20B/numina_math/baselines/sae_l11/interp_score/sae_latent_names.json"
)
RET_CHECKPOINT_DIR = REPO_ROOT / "outputs/GptOss20B/numina_math/ret_jepa_auth_l11"
RET_CLUSTERING_DIR = RET_CHECKPOINT_DIR / "clustering"

OUT_ROOT = (
    REPO_ROOT
    / "outputs/GptOss20B/numina_math/baselines/sae_l11"
    / "sae_steering/comparison_40problems"
)

# RET hyperparameters from the validated RET sweet-spot
RET_W = 1
RET_K = 64
RET_SEED = 1
RET_TAU = 0.1
RET_ALPHA_GROUPING = 1.0
RET_STRENGTH = 0.5
RET_GRADIENT_STEPS = 3
RET_TARGET_FORMULA = 35
RET_TARGET_ENUMERATION = 63
RET_DURATION = 600

# SAE composites (validated working configurations)
SAE_PYTHON_LIDS = [501, 566, 779, 217, 34, 616]
SAE_PYTHON_STRENGTH = 0.7
SAE_PROOF_LIDS = [8498, 9406, 5039, 4075, 3575]
SAE_PROOF_STRENGTH = 0.3

MAX_NEW = 384
SEED = 12345
SYSTEM_PROMPT = "You are a helpful assistant."


# ---------------------------------------------------------------------------
# Problem set: 40 prompts
# ---------------------------------------------------------------------------

def _build_problems() -> list[dict[str, str]]:
    """Compose 40 math prompts spanning multiple categories.

    Each entry has: id, category, user_prompt.
    Mix of open-ended ("Explain how to compute X") and closed-form
    ("Compute X.") since SAE prefers the former and RET works on both.
    """
    out: list[dict[str, str]] = []

    def add(cat: str, user: str):
        out.append({"id": f"P{len(out)+1:02d}", "category": cat, "user": user})

    # Group A — sum 1..N (mix of phrasings, vary N)  [12 prompts]
    add("sum_1_to_N", "Explain how to compute 1 + 2 + 3 + ... + 7.")
    add("sum_1_to_N", "Explain how to compute 1 + 2 + 3 + ... + 9.")
    add("sum_1_to_N", "Explain how to compute 1 + 2 + 3 + ... + 11.")
    add("sum_1_to_N", "Explain how to compute 1 + 2 + 3 + ... + 13.")
    add("sum_1_to_N", "Explain how to compute 1 + 2 + 3 + ... + 17.")
    add("sum_1_to_N", "Explain how to compute 1 + 2 + 3 + ... + 20.")
    add("sum_1_to_N", "How would you compute the sum 1 + 2 + 3 + ... + 25?")
    add("sum_1_to_N", "Describe a way to find 1 + 2 + 3 + ... + 50.")
    add("sum_1_to_N", "What is the sum 1 + 2 + 3 + ... + 14?")
    add("sum_1_to_N", "Give a method to compute 1 + 2 + 3 + ... + 30.")
    add("sum_1_to_N", "Compute 1 + 2 + 3 + ... + 12.")
    add("sum_1_to_N", "Find the value of 1 + 2 + 3 + ... + 18.")

    # Group B — sum of squares 1²+2²+...+N²  [5 prompts]
    add("sum_squares", "Explain how to compute 1² + 2² + 3² + ... + 5².")
    add("sum_squares", "Explain how to compute 1² + 2² + 3² + ... + 8².")
    add("sum_squares", "How would you find the sum of the squares from 1 to 10?")
    add("sum_squares", "Compute 1² + 2² + 3² + ... + 6².")
    add("sum_squares", "Describe a way to compute the sum of the first 12 perfect squares.")

    # Group C — sum of cubes 1³+2³+...+N³  [4 prompts]
    add("sum_cubes", "Explain how to compute 1³ + 2³ + 3³ + ... + 4³.")
    add("sum_cubes", "How would you compute 1³ + 2³ + 3³ + ... + 6³?")
    add("sum_cubes", "Find a method to compute the sum of the first 7 cubes.")
    add("sum_cubes", "Compute 1³ + 2³ + ... + 5³.")

    # Group D — sum of evens  [4 prompts]
    add("sum_evens", "Explain how to compute 2 + 4 + 6 + ... + 20.")
    add("sum_evens", "How would you find the sum of the even numbers from 2 to 30?")
    add("sum_evens", "Compute 2 + 4 + 6 + ... + 16.")
    add("sum_evens", "What is 2 + 4 + 6 + ... + 24?")

    # Group E — sum of odds  [4 prompts]
    add("sum_odds", "Explain how to compute 1 + 3 + 5 + ... + 19.")
    add("sum_odds", "Compute the sum of the first 10 odd positive integers.")
    add("sum_odds", "What is 1 + 3 + 5 + ... + 25?")
    add("sum_odds", "How would you find 1 + 3 + 5 + ... + 31?")

    # Group F — geometric series  [4 prompts]
    add("geometric", "Explain how to compute 1 + 2 + 4 + 8 + ... + 64.")
    add("geometric", "Compute 1 + 2 + 4 + ... + 256.")
    add("geometric", "Compute 1 + 3 + 9 + 27 + 81.")
    add("geometric", "How would you compute 2 + 6 + 18 + 54 + 162?")

    # Group G — counting / divisible  [4 prompts]
    add("counting", "How many positive integers from 1 to 100 are divisible by 3?")
    add("counting", "Count the multiples of 5 in the range 1 to 60.")
    add("counting", "How many even numbers are between 1 and 99?")
    add("counting", "How many integers from 1 to 50 are divisible by 7?")

    # Group H — mean / average  [3 prompts]
    add("mean", "Compute the average of the integers from 1 to 11.")
    add("mean", "What is the mean of the numbers 1, 2, 3, ..., 20?")
    add("mean", "Find the average of the first 15 positive integers.")

    assert len(out) == 40, f"expected 40 problems, got {len(out)}"
    return out


# ---------------------------------------------------------------------------
# Style detectors (keyword / regex)
# ---------------------------------------------------------------------------

def _detect_styles(text: str) -> dict[str, bool]:
    """Apply keyword-based detectors to the model output. Each is heuristic
    but consistent across conditions so the SUCCESS = "in steered AND NOT
    in baseline" comparison is well-defined.

    Returns a dict with bools for: formula, enumeration, python, proof.
    """
    t = text.lower()

    # FORMULA: closed-form expression for arithmetic-series-style sums
    formula_patterns = [
        r"n\s*\(\s*n\s*\+\s*1\s*\)\s*/\s*2",
        r"\(\s*n\s*\+\s*1\s*\)\s*\*\s*n\s*/\s*2",
        r"n\s*\*\s*\(\s*n\s*\+\s*1\s*\)\s*/\s*2",
        r"frac\{\s*n\s*\(\s*n\s*\+\s*1\s*\)\s*\}\s*\{\s*2\s*\}",
        r"n\s*\(\s*n\s*\+\s*1\s*\)\s*\(\s*2\s*n\s*\+\s*1\s*\)\s*/\s*6",  # squares
        r"n\^2\s*\(\s*n\s*\+\s*1\s*\)\^2\s*/\s*4",                       # cubes
        r"\\frac\{[^}]*\}\{[^}]*\}",                                       # any LaTeX frac
    ]
    has_formula = any(re.search(p, t) for p in formula_patterns)

    # ENUMERATION: explicit chain like "1+2=3, then 3+3=6"
    # we look for at least 3 consecutive cumulative-add steps
    enum_patterns = [
        r"1\s*\+\s*2\s*=\s*3.*?\+\s*3\s*=\s*6",
        r"\b\d+\s*\+\s*\d+\s*=\s*\d+\b.*?\b\d+\s*\+\s*\d+\s*=\s*\d+\b.*?\b\d+\s*\+\s*\d+\s*=\s*\d+\b",
        r"\+\s*\d+\s*=\s*\d+,\s*\+\s*\d+\s*=\s*\d+,\s*\+\s*\d+\s*=\s*\d+",
    ]
    has_enumeration = any(re.search(p, t) for p in enum_patterns)

    # PYTHON: explicit Python code
    python_patterns = [
        r"```python",
        r"sum\(\s*range\s*\(",
        r"^\s*for\s+\w+\s+in\s+range\s*\(",   # multi-line context
        r"\bdef\s+\w+\s*\(",
        r"\.append\b",
        r"^\s*import\s+\w+",
        r"print\s*\(",
    ]
    has_python = any(re.search(p, t, re.MULTILINE) for p in python_patterns)

    # PROOF: formal arithmetic-series notation
    proof_patterns = [
        r"\barithmetic\s+progression\b",
        r"\bcommon\s+difference\b",
        r"first\s+term[s]?\s*[a-d_]{0,4}\s*[=:]",
        r"\bs[\\_]*n\s*=",                       # S_n =
        r"a[_\\]+1\s*=",                          # a_1 =
        r"\binduction\b",
        r"\bproof\b.*\bby\b",
    ]
    has_proof = any(re.search(p, t) for p in proof_patterns)

    return {
        "formula": has_formula,
        "enumeration": has_enumeration,
        "python": has_python,
        "proof": has_proof,
    }


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------

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


def _gen_with_hook(ctx, prompt: str, hook, max_new: int = MAX_NEW,
                    temperature: float = 0.7, top_p: float = 0.9,
                    seed: int = SEED) -> tuple[str, list[dict]]:
    """Run a single generation with the given hook attached to ctx.layers[ctx.layer_idx]."""
    from evaluation.unsupervised_steering.generation import SteeredGenerator
    handle = ctx.layers[ctx.layer_idx].register_forward_hook(hook)
    try:
        gen = SteeredGenerator(ctx.model, ctx.tokenizer, hook)
        torch.manual_seed(seed)
        text, diagnostics = gen.generate(prompt, max_new, temperature, top_p)
    finally:
        handle.remove()
    return text, diagnostics


# ---------------------------------------------------------------------------
# Phase 1: SAE
# ---------------------------------------------------------------------------

def phase_sae(problems: list[dict], save_dir: Path) -> dict[str, dict]:
    """Run baseline + SAE python + SAE proof on all problems. Returns a dict
    mapping problem_id -> {baseline: text, sae_python: text, sae_proof: text}."""
    from evaluation.baselines.sae_unsupervised_steering.config import (
        SAESteeringConfig, SAESteeringRule,
    )
    from evaluation.baselines.sae_unsupervised_steering.hooks import SAESteeringHook
    from evaluation.baselines.sae_unsupervised_steering.loader import load_sae_steering_context
    from evaluation.baselines.sae_unsupervised_steering.steerer import SAESteerer

    save_dir.mkdir(parents=True, exist_ok=True)

    cfg0 = SAESteeringConfig(
        sae_release="andyrdt/saes-gpt-oss-20b",
        sae_id="resid_post_layer_11/trainer_0",
        llm_config_path=str(LLM_CFG),
        direction_source="decoder",
        sae_latent_names_path=str(SAE_NAMES),
        rules=[], prompt="dummy", device="cuda",
    )
    logger.info("Loading SAE context...")
    sae_ctx = load_sae_steering_context(cfg0)

    results: dict[str, dict] = {}
    for prob in problems:
        pid = prob["id"]
        chat = build_chat_prompt(sae_ctx.model.name_or_path, prob["user"])
        results[pid] = {
            "user_prompt": prob["user"], "category": prob["category"],
        }

        # Baseline
        steerer0 = SAESteerer(strength=0.0, d_h=sae_ctx.hidden_dim, device=sae_ctx.device)
        hook0 = SAESteeringHook(
            sae=sae_ctx.sae, decoder_weight=sae_ctx.decoder_weight,
            encoder_weight=sae_ctx.encoder_weight, b_dec=sae_ctx.b_dec,
            encoder_bias=sae_ctx.encoder_bias, rules=[], steerer=steerer0,
            direction_source="decoder", device=sae_ctx.device, top_k_diag=8,
        )
        t0 = time.time()
        text, _ = _gen_with_hook(sae_ctx, chat, hook0)
        results[pid]["baseline"] = text
        results[pid]["baseline_secs"] = round(time.time() - t0, 2)

        # SAE python
        steerer_py = SAESteerer(strength=SAE_PYTHON_STRENGTH, d_h=sae_ctx.hidden_dim, device=sae_ctx.device)
        rule_py = SAESteeringRule(
            trigger_position=0, target_latents=SAE_PYTHON_LIDS, duration=MAX_NEW,
        )
        hook_py = SAESteeringHook(
            sae=sae_ctx.sae, decoder_weight=sae_ctx.decoder_weight,
            encoder_weight=sae_ctx.encoder_weight, b_dec=sae_ctx.b_dec,
            encoder_bias=sae_ctx.encoder_bias, rules=[rule_py], steerer=steerer_py,
            direction_source="decoder", device=sae_ctx.device, top_k_diag=8,
        )
        t0 = time.time()
        text_py, _ = _gen_with_hook(sae_ctx, chat, hook_py)
        results[pid]["sae_python"] = text_py
        results[pid]["sae_python_secs"] = round(time.time() - t0, 2)

        # SAE proof
        steerer_pr = SAESteerer(strength=SAE_PROOF_STRENGTH, d_h=sae_ctx.hidden_dim, device=sae_ctx.device)
        rule_pr = SAESteeringRule(
            trigger_position=0, target_latents=SAE_PROOF_LIDS, duration=MAX_NEW,
        )
        hook_pr = SAESteeringHook(
            sae=sae_ctx.sae, decoder_weight=sae_ctx.decoder_weight,
            encoder_weight=sae_ctx.encoder_weight, b_dec=sae_ctx.b_dec,
            encoder_bias=sae_ctx.encoder_bias, rules=[rule_pr], steerer=steerer_pr,
            direction_source="decoder", device=sae_ctx.device, top_k_diag=8,
        )
        t0 = time.time()
        text_pr, _ = _gen_with_hook(sae_ctx, chat, hook_pr)
        results[pid]["sae_proof"] = text_pr
        results[pid]["sae_proof_secs"] = round(time.time() - t0, 2)

        logger.info("[SAE %s] base/py/proof: %.1f/%.1f/%.1fs",
                    pid, results[pid]["baseline_secs"],
                    results[pid]["sae_python_secs"],
                    results[pid]["sae_proof_secs"])

    # Save phase 1 results
    with open(save_dir / "phase1_sae_outputs.json", "w") as f:
        json.dump(results, f, indent=2)

    # Free
    del sae_ctx, hook0, hook_py, hook_pr, steerer0, steerer_py, steerer_pr
    gc.collect()
    torch.cuda.empty_cache()

    return results


# ---------------------------------------------------------------------------
# Phase 2: RET
# ---------------------------------------------------------------------------

def phase_ret(problems: list[dict], existing: dict[str, dict],
              save_dir: Path) -> dict[str, dict]:
    """Run RET steering for both targets on all problems. Mutates and returns
    ``existing`` with the ret_formula / ret_enumeration text added."""
    from evaluation.unsupervised_steering.config import (
        SteeringConfig, SteeringRule,
    )
    from evaluation.unsupervised_steering.hooks import SteeringHook
    from evaluation.unsupervised_steering.mppi import GradientSteerer
    from evaluation.unsupervised_steering.targets import load_steering_context

    cfg0 = SteeringConfig(
        clustering_dir=str(RET_CLUSTERING_DIR),
        window_size=RET_W, K=RET_K, seed=RET_SEED,
        tau=RET_TAU, alpha=RET_ALPHA_GROUPING,
        rules=[], strength=RET_STRENGTH,
        gradient_steps=RET_GRADIENT_STEPS,
        prompt="dummy", max_new_tokens=MAX_NEW,
        temperature=0.7, top_p=0.9, device="cuda",
    )
    logger.info("Loading RET context...")
    ret_ctx = load_steering_context(cfg0)
    logger.info("RET ready: layer=%d K=%d", ret_ctx.layer_idx, RET_K)

    optimizer = GradientSteerer(
        strength=RET_STRENGTH, gradient_steps=RET_GRADIENT_STEPS,
        d_h=ret_ctx.hidden_dim, device="cuda",
    )

    for prob in problems:
        pid = prob["id"]
        chat = build_chat_prompt(ret_ctx.model.name_or_path, prob["user"])

        # RET formula attractor
        rule_f = SteeringRule(
            trigger_position=0, target_cluster=RET_TARGET_FORMULA,
            duration=RET_DURATION,
        )
        hook_f = SteeringHook(
            encoder=ret_ctx.encoder, optimizer=optimizer,
            mu=ret_ctx.mu, centers=ret_ctx.centers,
            centers_normed=ret_ctx.centers_normed,
            group_labels=ret_ctx.group_labels,
            group_to_clusters=ret_ctx.group_to_clusters,
            rules=[rule_f], device="cuda",
        )
        t0 = time.time()
        text_f, _ = _gen_with_hook(ret_ctx, chat, hook_f)
        existing[pid]["ret_formula"] = text_f
        existing[pid]["ret_formula_secs"] = round(time.time() - t0, 2)

        # RET enumeration attractor
        rule_e = SteeringRule(
            trigger_position=0, target_cluster=RET_TARGET_ENUMERATION,
            duration=RET_DURATION,
        )
        hook_e = SteeringHook(
            encoder=ret_ctx.encoder, optimizer=optimizer,
            mu=ret_ctx.mu, centers=ret_ctx.centers,
            centers_normed=ret_ctx.centers_normed,
            group_labels=ret_ctx.group_labels,
            group_to_clusters=ret_ctx.group_to_clusters,
            rules=[rule_e], device="cuda",
        )
        t0 = time.time()
        text_e, _ = _gen_with_hook(ret_ctx, chat, hook_e)
        existing[pid]["ret_enumeration"] = text_e
        existing[pid]["ret_enumeration_secs"] = round(time.time() - t0, 2)

        logger.info("[RET %s] formula/enum: %.1f/%.1fs", pid,
                    existing[pid]["ret_formula_secs"],
                    existing[pid]["ret_enumeration_secs"])

    with open(save_dir / "phase2_ret_added.json", "w") as f:
        json.dump(existing, f, indent=2)
    return existing


# ---------------------------------------------------------------------------
# Phase 3: Score
# ---------------------------------------------------------------------------

def phase_score(results: dict[str, dict], save_dir: Path) -> dict[str, Any]:
    """Apply detectors and compute success rates."""
    detected: dict[str, dict[str, dict[str, bool]]] = {}
    for pid, r in results.items():
        detected[pid] = {}
        for cond in ["baseline", "ret_formula", "ret_enumeration",
                     "sae_python", "sae_proof"]:
            text = r.get(cond, "")
            detected[pid][cond] = _detect_styles(text)

    # Per-method success: target appears in steered AND not in baseline.
    methods = {
        "ret_formula":      ("formula",     "ret_formula"),
        "ret_enumeration":  ("enumeration", "ret_enumeration"),
        "sae_python":       ("python",      "sae_python"),
        "sae_proof":        ("proof",       "sae_proof"),
    }
    per_method = {m: {"success": 0, "no_op": 0, "harmful": 0, "failed": 0,
                      "induced": [], "no_op_pids": [], "harmful_pids": [],
                      "failed_pids": []}
                  for m in methods}

    for pid in detected:
        for m, (style, cond) in methods.items():
            in_base = detected[pid]["baseline"].get(style, False)
            in_steer = detected[pid][cond].get(style, False)
            if in_steer and not in_base:
                per_method[m]["success"] += 1
                per_method[m]["induced"].append(pid)
            elif in_steer and in_base:
                per_method[m]["no_op"] += 1
                per_method[m]["no_op_pids"].append(pid)
            elif (not in_steer) and in_base:
                per_method[m]["harmful"] += 1
                per_method[m]["harmful_pids"].append(pid)
            else:
                per_method[m]["failed"] += 1
                per_method[m]["failed_pids"].append(pid)

    # Aggregate per-side (RET vs SAE)
    ret_methods = ["ret_formula", "ret_enumeration"]
    sae_methods = ["sae_python", "sae_proof"]

    def _agg(ms):
        s = sum(per_method[m]["success"] for m in ms)
        n = sum(per_method[m]["success"] + per_method[m]["no_op"]
                + per_method[m]["harmful"] + per_method[m]["failed"]
                for m in ms)
        return s, n, (s / n if n else 0.0)

    ret_s, ret_n, ret_rate = _agg(ret_methods)
    sae_s, sae_n, sae_rate = _agg(sae_methods)

    # "any-success" per problem: did EITHER RET method induce its style? same for SAE
    ret_induced_any = set()
    sae_induced_any = set()
    for pid in detected:
        if any(pid in per_method[m]["induced"] for m in ret_methods):
            ret_induced_any.add(pid)
        if any(pid in per_method[m]["induced"] for m in sae_methods):
            sae_induced_any.add(pid)

    summary = {
        "params": {
            "max_new_tokens": MAX_NEW, "seed": SEED,
            "ret_strength": RET_STRENGTH, "ret_gradient_steps": RET_GRADIENT_STEPS,
            "ret_target_formula": RET_TARGET_FORMULA,
            "ret_target_enumeration": RET_TARGET_ENUMERATION,
            "sae_python_lids": SAE_PYTHON_LIDS, "sae_python_strength": SAE_PYTHON_STRENGTH,
            "sae_proof_lids": SAE_PROOF_LIDS, "sae_proof_strength": SAE_PROOF_STRENGTH,
        },
        "n_problems": len(results),
        "per_method": per_method,
        "aggregate": {
            "ret_total_success": ret_s, "ret_total_attempts": ret_n,
            "ret_success_rate": ret_rate,
            "sae_total_success": sae_s, "sae_total_attempts": sae_n,
            "sae_success_rate": sae_rate,
            "ret_any_method_induced": len(ret_induced_any),
            "sae_any_method_induced": len(sae_induced_any),
        },
        "detected": detected,
    }
    with open(save_dir / "comparison_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", default="all", choices=["all", "sae", "ret", "score"],
                        help="Which phase to run.")
    parser.add_argument("--out-root", default=str(OUT_ROOT))
    args = parser.parse_args()

    save_dir = Path(args.out_root)
    save_dir.mkdir(parents=True, exist_ok=True)

    problems = _build_problems()
    with open(save_dir / "problems.json", "w") as f:
        json.dump(problems, f, indent=2)
    logger.info("Loaded %d problems", len(problems))

    results: dict[str, dict] = {}

    if args.phase in ("all", "sae"):
        logger.info("=== Phase 1: SAE ===")
        t0 = time.time()
        results = phase_sae(problems, save_dir)
        logger.info("Phase 1 SAE done in %.1fs", time.time() - t0)
    elif (save_dir / "phase1_sae_outputs.json").exists():
        results = json.load(open(save_dir / "phase1_sae_outputs.json"))

    if args.phase in ("all", "ret"):
        if not results:
            results = json.load(open(save_dir / "phase1_sae_outputs.json"))
        logger.info("=== Phase 2: RET ===")
        t0 = time.time()
        results = phase_ret(problems, results, save_dir)
        logger.info("Phase 2 RET done in %.1fs", time.time() - t0)
    elif (save_dir / "phase2_ret_added.json").exists():
        results = json.load(open(save_dir / "phase2_ret_added.json"))

    if args.phase in ("all", "score"):
        logger.info("=== Phase 3: Score ===")
        summary = phase_score(results, save_dir)
        # Pretty print
        agg = summary["aggregate"]
        logger.info("=== RESULTS ===")
        logger.info("Per-method success:")
        for m, s in summary["per_method"].items():
            n = s["success"] + s["no_op"] + s["harmful"] + s["failed"]
            logger.info("  %-20s success=%2d/%-2d  no_op=%2d  harmful=%2d  failed=%2d",
                        m, s["success"], n, s["no_op"], s["harmful"], s["failed"])
        logger.info("RET aggregate: %d/%d successes (rate=%.1f%%); any-method induced on %d/%d problems",
                    agg["ret_total_success"], agg["ret_total_attempts"],
                    100 * agg["ret_success_rate"], agg["ret_any_method_induced"],
                    summary["n_problems"])
        logger.info("SAE aggregate: %d/%d successes (rate=%.1f%%); any-method induced on %d/%d problems",
                    agg["sae_total_success"], agg["sae_total_attempts"],
                    100 * agg["sae_success_rate"], agg["sae_any_method_induced"],
                    summary["n_problems"])


if __name__ == "__main__":
    main()
