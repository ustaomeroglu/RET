"""Larger behavior-change steering comparison.

Same methodology as ``comparison_behavior_change.py`` but with 70+
ambiguity-prone problems chosen specifically because they admit a clear
alternative interpretation that a 'wording_disambiguation' style steering
should surface.

Reuses the algorithmically-selected targets from the smaller comparison:
  RET top-1 = C57 wording_disambiguation
  SAE top-1 = L91407 [' potential', ' possible', ' interpretation', ...]

(Selection was done over identical search spaces with sentence-transformers
all-MiniLM-L6-v2 cosine similarity to the same fixed concept query.)
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

import torch

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)s %(levelname)s  %(message)s',
                    datefmt='%H:%M:%S')
logger = logging.getLogger("compare_behavior_L")

REPO_ROOT = Path(__file__).resolve().parents[4]
LLM_CFG = REPO_ROOT / "configs/GptOss20B/baselines/sae_l11.yaml"
SAE_NAMES = (
    REPO_ROOT
    / "outputs/GptOss20B/numina_math/baselines/sae_l11/interp_score/sae_latent_names.json"
)
RET_CLUSTERING_DIR = (
    REPO_ROOT / "outputs/GptOss20B/numina_math/ret_jepa_auth_l11/clustering"
)

PREV_ROOT = (
    REPO_ROOT
    / "outputs/GptOss20B/numina_math/baselines/sae_l11"
    / "sae_steering/comparison_behavior_change"
)
OUT_ROOT = (
    REPO_ROOT
    / "outputs/GptOss20B/numina_math/baselines/sae_l11"
    / "sae_steering/comparison_behavior_change_large"
)

# Algorithmically-selected targets (top-1 each, reused from prev run)
RET_TOP1_CLUSTER = 57       # wording_disambiguation
SAE_TOP1_LATENT  = 91407    # ' potential',' possible',' interpretation',' if',' answers'

RET_W = 1; RET_K = 64; RET_SEED = 1; RET_TAU = 0.1; RET_ALPHA_GROUPING = 1.0
RET_STRENGTH = 0.5; RET_GRADIENT_STEPS = 3; RET_DURATION = 600
SAE_STRENGTH = 0.3
MAX_NEW = 384; SEED = 12345
SYSTEM_PROMPT = "You are a helpful assistant."


# ---------------------------------------------------------------------------
# 70+ ambiguity-prone problems
# ---------------------------------------------------------------------------

def _build_problems():
    out = []
    def add(cat, user):
        out.append({"id": f"P{len(out)+1:02d}", "category": cat, "user": user})

    # A) Operator precedence ambiguity (15)
    add("precedence", "Evaluate -3^2.")
    add("precedence", "Compute 8/4/2.")
    add("precedence", "What is 2^3^2?")
    add("precedence", "Evaluate -(-3)^2.")
    add("precedence", "What is 16/2(2+2)?")
    add("precedence", "What is 6/2*(1+2)?")
    add("precedence", "Compute 4 * 3 + 2 * 5.")
    add("precedence", "Evaluate 2 - 3 + 4.")
    add("precedence", "What is 100 - 50 + 10?")
    add("precedence", "Compute 10 / 2 * 5.")
    add("precedence", "Evaluate 3 + 4 * 2.")
    add("precedence", "What is -2^4?")
    add("precedence", "Compute 2 * 3^2.")
    add("precedence", "Evaluate 5 - -3.")
    add("precedence", "What is 6 / 2 * 3?")

    # B) Domain ambiguity (15)
    add("domain", "Solve x^2 = 9.")
    add("domain", "Solve x^2 = -1.")
    add("domain", "How many roots does x^4 = 1 have?")
    add("domain", "Solve x^3 = -8.")
    add("domain", "How many solutions does cos(x) = 0 have?")
    add("domain", "Solve sin(x) = 1.")
    add("domain", "How many real roots does x^4 - 1 = 0 have?")
    add("domain", "What is sqrt(9)?")
    add("domain", "What is sqrt(-4)?")
    add("domain", "Solve |x| = 5.")
    add("domain", "Solve |x - 2| = 3.")
    add("domain", "How many solutions does x^2 + 4 = 0 have?")
    add("domain", "Solve cos(x) = 1 in [0, 4π].")
    add("domain", "What's the largest real number x such that x^3 = 27?")
    add("domain", "How many integer solutions does x^2 = 4 have?")

    # C) Convention-dependent (15)
    add("convention", "Is 0 a natural number?")
    add("convention", "Is 1 a prime number?")
    add("convention", "Compute 0!.")
    add("convention", "What is 0^0?")
    add("convention", "Compute log(1).")
    add("convention", "Compute log(0).")
    add("convention", "Compute 1/0.")
    add("convention", "Compute 0/0.")
    add("convention", "What is the sign of 0?")
    add("convention", "Is 0 even or odd?")
    add("convention", "Compute (-2)^(1/2).")
    add("convention", "What is the sum of the empty set?")
    add("convention", "What is the smallest natural number?")
    add("convention", "Is 0.999... equal to 1?")
    add("convention", "What is 0 divided by 0?")

    # D) Function ambiguity (10)
    add("function", "What is arcsin(0.5)?")
    add("function", "What is the inverse of f(x) = x^2?")
    add("function", "Compute log(100).")
    add("function", "What is the derivative of |x|?")
    add("function", "What is |3 - 5|?")
    add("function", "What is the floor of 3.7?")
    add("function", "What is the ceiling of -1.3?")
    add("function", "Round 2.5 to nearest integer.")
    add("function", "What is the GCD of 0 and 5?")
    add("function", "What is the LCM of 0 and 5?")

    # E) Counting ambiguity (10)
    add("counting", "How many factors does 12 have?")
    add("counting", "How many vowels are in the word 'rendezvous'?")
    add("counting", "How many integers are between 1 and 10?")
    add("counting", "How many corners does a polygon have?")
    add("counting", "How many sides does a circle have?")
    add("counting", "How many ways can you arrange the letters of ABBA?")
    add("counting", "How many ways can you arrange 4 books on a shelf?")
    add("counting", "How many digits does the number 0 have?")
    add("counting", "How many significant figures does 1.0 have?")
    add("counting", "How many divisors does 6 have?")

    # F) Probability / statistics (8)
    add("statistics", "What is the probability of getting 2 heads in 3 coin flips?")
    add("statistics", "What is the average of 1 and 2?")
    add("statistics", "What is the median of {1, 2, 3, 4}?")
    add("statistics", "What is the variance of {1, 2, 3}?")
    add("statistics", "What is the mode of {1, 2, 2, 3, 3}?")
    add("statistics", "What is the standard deviation of {1, 2, 3}?")
    add("statistics", "What is the expected value of a fair die roll?")
    add("statistics", "What's the average grade if scores are 80, 90, 100?")

    # G) Word problem ambiguity (5)
    add("word", "If a shirt costs $50 and is on 20% sale, what's the final price?")
    add("word", "If you double 6 and add 3, what do you get?")
    add("word", "What's half of three quarters?")
    add("word", "If x is twice y and y is 5, what is x?")
    add("word", "A train leaves at 3pm; how long until 5pm?")

    return out


# ---------------------------------------------------------------------------
# Detectors
# ---------------------------------------------------------------------------

_HEDGE_PATTERNS = [
    r"\bdepends?\s+on\b", r"\bdepending\s+on\b",
    r"\bin\s+(?:some|certain|other)\s+conventions?\b",
    r"\bconvention(?:al(?:ly)?|s|\b)",
    r"\bambig\w*", r"\binterpret\w*",
    r"\b(?:could|might|may)\s+(?:be|mean|refer)\b",
    r"\b(?:if|when)\s+we\s+(?:assume|interpret|consider|allow|require|exclude|include|use|take)\b",
    r"\beither\s+\w+\s+or\b",
    r"\b(?:two|three|several|multiple)\s+(?:possible|interpretations?|conventions?|answers?|values?|solutions?|cases?)\b",
    r"\bmultiple\s+\w+s?\b",
    r"\b(?:assuming|suppose|presume|expect)\b",
    r"\bclarif\w*",
    r"\bdistinct\s+(?:answers?|values?|interpretations?)\b",
    r"\b(?:vs|versus)\b",
    r"\b(?:over|in)\s+(?:the\s+)?(?:reals?|real\s+numbers?|complex|integers?)\b",
    r"\bdomain\s+(?:of|matters|depends|is)\b",
    r"\bif\s+(?:by|the\s+question)\b",
    r"\bthe\s+question\s+(?:is\s+)?(?:not\s+(?:fully\s+)?clear|under(?:specified|defined)|ambig\w*)\b",
    r"\b(?:answer|value|result)s?\s+(?:can|could|may|might)\s+(?:be|vary|differ)\b",
    r"\b(?:there\s+are|we\s+have)\s+(?:two|three|several|multiple)\b",
    r"\b(?:I'?m|am)\s+not\s+sure\b",
]

_ALT_INTERP_PATTERNS = [
    r'\bif\s+you\s+(?:intended|mean|meant|wanted|intend)\b',
    r'\bif\s+(?:instead|on\s+the\s+other|the\s+question|by)\b',
    r'\b(?:on\s+the\s+other\s+hand|alternatively|otherwise)\b',
    r'\b(?:depending\s+on|depends\s+on)\s+(?:whether|the|how|what)\b',
    r'\bin\s+(?:that|the|some)\s+case\b',
    r'\binterpretation\b', r'\binterpret\s+\w+\s+as\b',
    r'\b(?:if|when)\s+we\s+(?:assume|consider|allow|require|treat|take|interpret)\b',
    r'\b(?:could|may|might)\s+(?:also\s+)?(?:mean|refer\s+to|be\s+interpreted)\b',
    r'\bunder\s+(?:the|a|some|certain)\s+convention\b',
    r'\bambig\w*', r'\bin\s+(?:some|other|certain)\s+conventions?\b',
    r'\b(?:two|three|several|multiple|different)\s+(?:interpretations?|conventions?|meanings?|possibilities?|possible\s+answers?)\b',
    r'\b(?:so|hence)[\s,]+(?:depending|the\s+answer\s+depends)\b',
    r'\b(?:you\s+can|one\s+can|we\s+can)\s+also\b',
    r"\b(?:vs|versus)\b\s+\w+",
]

def _hedge_count(text):
    t = text.lower()
    return sum(len(re.findall(p, t)) for p in _HEDGE_PATTERNS)

def _alt_interp_count(text):
    t = text.lower()
    return sum(len(re.findall(p, t)) for p in _ALT_INTERP_PATTERNS)


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------

def build_chat_prompt(model_name, user):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    return tok.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT},
         {"role": "user", "content": user}],
        tokenize=False, add_generation_prompt=True,
    )

def _gen(ctx, prompt, hook, max_new=MAX_NEW, temperature=0.7, top_p=0.9, seed=SEED):
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
# Phase: SAE
# ---------------------------------------------------------------------------

def phase_sae(problems, save_dir):
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

    results = {}
    for prob in problems:
        pid = prob["id"]
        chat = build_chat_prompt(ctx.model.name_or_path, prob["user"])
        results[pid] = {"user_prompt": prob["user"], "category": prob["category"]}

        # Baseline
        steerer = SAESteerer(strength=0.0, d_h=ctx.hidden_dim, device=ctx.device)
        hook = SAESteeringHook(
            sae=ctx.sae, decoder_weight=ctx.decoder_weight,
            encoder_weight=ctx.encoder_weight, b_dec=ctx.b_dec,
            encoder_bias=ctx.encoder_bias, rules=[], steerer=steerer,
            direction_source="decoder", device=ctx.device, top_k_diag=8,
        )
        t0 = time.time()
        text, _ = _gen(ctx, chat, hook)
        results[pid]["baseline"] = text
        results[pid]["baseline_secs"] = round(time.time() - t0, 2)

        # SAE algo target
        steerer_s = SAESteerer(strength=SAE_STRENGTH, d_h=ctx.hidden_dim, device=ctx.device)
        rule = SAESteeringRule(
            trigger_position=0, target_latents=[SAE_TOP1_LATENT], duration=MAX_NEW,
        )
        hook_s = SAESteeringHook(
            sae=ctx.sae, decoder_weight=ctx.decoder_weight,
            encoder_weight=ctx.encoder_weight, b_dec=ctx.b_dec,
            encoder_bias=ctx.encoder_bias, rules=[rule], steerer=steerer_s,
            direction_source="decoder", device=ctx.device, top_k_diag=8,
        )
        t0 = time.time()
        text_s, _ = _gen(ctx, chat, hook_s)
        results[pid]["sae"] = text_s
        results[pid]["sae_secs"] = round(time.time() - t0, 2)

        logger.info("[SAE %s] base/sae: %.1f/%.1fs",
                    pid, results[pid]["baseline_secs"], results[pid]["sae_secs"])

    with open(save_dir / "phase_sae.json", "w") as f:
        json.dump(results, f, indent=2)

    del ctx, hook, hook_s
    gc.collect()
    torch.cuda.empty_cache()
    return results


# ---------------------------------------------------------------------------
# Phase: RET
# ---------------------------------------------------------------------------

def phase_ret(problems, results, save_dir):
    from evaluation.unsupervised_steering.config import SteeringConfig, SteeringRule
    from evaluation.unsupervised_steering.hooks import SteeringHook
    from evaluation.unsupervised_steering.mppi import GradientSteerer
    from evaluation.unsupervised_steering.targets import load_steering_context

    cfg0 = SteeringConfig(
        clustering_dir=str(RET_CLUSTERING_DIR),
        window_size=RET_W, K=RET_K, seed=RET_SEED, tau=RET_TAU, alpha=RET_ALPHA_GROUPING,
        rules=[], strength=RET_STRENGTH, gradient_steps=RET_GRADIENT_STEPS,
        prompt="dummy", max_new_tokens=MAX_NEW,
        temperature=0.7, top_p=0.9, device="cuda",
    )
    logger.info("Loading RET context...")
    ctx = load_steering_context(cfg0)

    optimizer = GradientSteerer(
        strength=RET_STRENGTH, gradient_steps=RET_GRADIENT_STEPS,
        d_h=ctx.hidden_dim, device="cuda",
    )

    for prob in problems:
        pid = prob["id"]
        chat = build_chat_prompt(ctx.model.name_or_path, prob["user"])

        rule = SteeringRule(
            trigger_position=0, target_cluster=RET_TOP1_CLUSTER,
            duration=RET_DURATION,
        )
        hook = SteeringHook(
            encoder=ctx.encoder, optimizer=optimizer,
            mu=ctx.mu, centers=ctx.centers,
            centers_normed=ctx.centers_normed,
            group_labels=ctx.group_labels,
            group_to_clusters=ctx.group_to_clusters,
            rules=[rule], device="cuda",
        )
        t0 = time.time()
        text, _ = _gen(ctx, chat, hook)
        results[pid]["ret"] = text
        results[pid]["ret_secs"] = round(time.time() - t0, 2)
        logger.info("[RET %s] %.1fs", pid, results[pid]["ret_secs"])

    with open(save_dir / "phase_ret.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


# ---------------------------------------------------------------------------
# Score
# ---------------------------------------------------------------------------

def phase_score(results, save_dir):
    per = {}
    for pid in results:
        b = results[pid]["baseline"]; s = results[pid]["sae"]; e = results[pid]["ret"]
        per[pid] = {
            "category": results[pid]["category"],
            "user": results[pid]["user_prompt"],
            "hedge": {"baseline": _hedge_count(b), "sae": _hedge_count(s), "ret": _hedge_count(e)},
            "alt":   {"baseline": _alt_interp_count(b), "sae": _alt_interp_count(s), "ret": _alt_interp_count(e)},
        }
        per[pid]["hedge"]["sae_delta"] = per[pid]["hedge"]["sae"] - per[pid]["hedge"]["baseline"]
        per[pid]["hedge"]["ret_delta"] = per[pid]["hedge"]["ret"] - per[pid]["hedge"]["baseline"]
        per[pid]["alt"]["sae_delta"] = per[pid]["alt"]["sae"] - per[pid]["alt"]["baseline"]
        per[pid]["alt"]["ret_delta"] = per[pid]["alt"]["ret"] - per[pid]["alt"]["baseline"]

    n = len(results)
    def _agg(metric):
        sae_strict = sum(1 for d in per.values() if d[metric]["sae_delta"] > 0)
        ret_strict = sum(1 for d in per.values() if d[metric]["ret_delta"] > 0)
        sae_d = sum(d[metric]["sae_delta"] for d in per.values())
        ret_d = sum(d[metric]["ret_delta"] for d in per.values())
        sae_iz = sum(1 for d in per.values()
                     if d[metric]["baseline"] == 0 and d[metric]["sae"] >= 1)
        ret_iz = sum(1 for d in per.values()
                     if d[metric]["baseline"] == 0 and d[metric]["ret"] >= 1)
        sae_w = sum(1 for d in per.values()
                    if d[metric]["sae_delta"] > d[metric]["ret_delta"])
        ret_w = sum(1 for d in per.values()
                    if d[metric]["ret_delta"] > d[metric]["sae_delta"])
        ties = n - sae_w - ret_w
        return {"sae_strict_more": sae_strict, "ret_strict_more": ret_strict,
                "sae_total_delta": sae_d, "ret_total_delta": ret_d,
                "sae_induced_from_zero": sae_iz, "ret_induced_from_zero": ret_iz,
                "sae_per_problem_wins": sae_w, "ret_per_problem_wins": ret_w,
                "ties": ties}

    summary = {
        "n_problems": n,
        "ret_top1_cluster": RET_TOP1_CLUSTER,
        "sae_top1_latent": SAE_TOP1_LATENT,
        "sae_strength": SAE_STRENGTH, "ret_strength": RET_STRENGTH,
        "metrics": {"hedge": _agg("hedge"), "alt": _agg("alt")},
        "per_problem": per,
    }
    with open(save_dir / "comparison_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", default="all", choices=["all", "sae", "ret", "score"])
    args = parser.parse_args()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    problems = _build_problems()
    with open(OUT_ROOT / "problems.json", "w") as f:
        json.dump(problems, f, indent=2)
    logger.info("Loaded %d problems", len(problems))

    results = {}
    if args.phase in ("all", "sae"):
        logger.info("=== Phase SAE ===")
        results = phase_sae(problems, OUT_ROOT)
    elif (OUT_ROOT / "phase_sae.json").exists():
        results = json.load(open(OUT_ROOT / "phase_sae.json"))

    if args.phase in ("all", "ret"):
        logger.info("=== Phase RET ===")
        results = phase_ret(problems, results, OUT_ROOT)
    elif (OUT_ROOT / "phase_ret.json").exists():
        results = json.load(open(OUT_ROOT / "phase_ret.json"))

    if args.phase in ("all", "score"):
        logger.info("=== Score ===")
        s = phase_score(results, OUT_ROOT)
        for metric_name, m in s["metrics"].items():
            logger.info("[%s] strict-more SAE=%d RET=%d  induced-from-zero SAE=%d RET=%d  per-problem-wins SAE=%d RET=%d ties=%d  total-Δ SAE=%+d RET=%+d",
                        metric_name, m["sae_strict_more"], m["ret_strict_more"],
                        m["sae_induced_from_zero"], m["ret_induced_from_zero"],
                        m["sae_per_problem_wins"], m["ret_per_problem_wins"], m["ties"],
                        m["sae_total_delta"], m["ret_total_delta"])


if __name__ == "__main__":
    main()
