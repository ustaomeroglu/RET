"""Behavior-change steering comparison: RET clusters vs SAE latents,
with **symmetric algorithmic target selection**.

Methodology
-----------
* Concept query (single, fixed) — describes the high-level behavior shift
  we want to induce. We use:

      "consider alternative interpretations of the question, acknowledge
       ambiguity, hedge and express uncertainty about which answer is
       intended"

* Both methods select their target by the SAME procedure:
  - Embed the query phrase with all-MiniLM-L6-v2 (sentence-transformers).
  - Embed every candidate target with the same encoder:
      RET: long_description from cluster_naming/naming.txt for each of 64 clusters.
      SAE: top-5 max-activating tokens joined with spaces, for each of 131k latents.
  - Pick TOP-1 by cosine similarity. (Same K for both methods.)

* Strength is committed:
  - SAE: strength=0.3 (math-domain sweet spot identified earlier).
  - RET: strength=0.5, gradient_steps=3 (validated RET sweet spot).

* Problems are 30 deliberately *ambiguity-prone* math questions where
  the natural baseline gives a single confident answer but a more
  "exploratory / disambiguating" mode would mention multiple
  interpretations, conventions, or hedge.

* Score: detect hedging / multi-answer / interpretation markers in
  the steered output and compare against baseline.
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

import numpy as np
import torch
import yaml

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)s %(levelname)s  %(message)s',
                    datefmt='%H:%M:%S')
logger = logging.getLogger("compare_behavior")

REPO_ROOT = Path(__file__).resolve().parents[4]
LLM_CFG = REPO_ROOT / "configs/GptOss20B/baselines/sae_l11.yaml"
SAE_NAMES = (
    REPO_ROOT
    / "outputs/GptOss20B/numina_math/baselines/sae_l11/interp_score/sae_latent_names.json"
)
RET_CLUSTERING_DIR = (
    REPO_ROOT / "outputs/GptOss20B/numina_math/ret_jepa_auth_l11/clustering"
)
RET_NAMING = (
    RET_CLUSTERING_DIR
    / "w1/K64/seed1/analysis/weighted_tau0.1/grouped_alpha1.0/cluster_naming/naming.txt"
)

OUT_ROOT = (
    REPO_ROOT
    / "outputs/GptOss20B/numina_math/baselines/sae_l11"
    / "sae_steering/comparison_behavior_change"
)

RET_W = 1
RET_K = 64
RET_SEED = 1
RET_TAU = 0.1
RET_ALPHA_GROUPING = 1.0
RET_STRENGTH = 0.5
RET_GRADIENT_STEPS = 3
RET_DURATION = 600

SAE_STRENGTH = 0.3
SAE_TOP_K_LATENTS = 1     # symmetric: top-1 for both methods

CONCEPT_QUERY = (
    "consider alternative interpretations of the question, acknowledge "
    "ambiguity, hedge and express uncertainty about which answer is intended"
)

MAX_NEW = 384
SEED = 12345
SYSTEM_PROMPT = "You are a helpful assistant."


# ---------------------------------------------------------------------------
# 30 ambiguity-prone problems
# ---------------------------------------------------------------------------

def _build_problems():
    """30 math/reasoning prompts that admit multiple interpretations
    or convention-dependent answers. Baseline tends to give one confident
    answer; with 'consider alternatives' steering, model should hedge,
    enumerate cases, mention conventions, or ask for clarification."""
    out = []
    def add(cat, user):
        out.append({"id": f"P{len(out)+1:02d}", "category": cat, "user": user})

    # Convention-dependent
    add("convention", "Is 0 a natural number?")
    add("convention", "Is 1 a prime number?")
    add("convention", "What is 0^0?")
    add("convention", "What is the smallest natural number?")
    add("convention", "Compute log(1).")
    add("convention", "Compute 0! (zero factorial).")

    # Operator precedence / ambiguous expressions
    add("precedence", "Compute 2^2^3.")
    add("precedence", "What is 6 / 2 * 3?")
    add("precedence", "Evaluate -3^2.")
    add("precedence", "What is 2 + 3 * 4?")

    # Equation domain (real vs complex, single vs multiple solutions)
    add("domain", "How many roots does x^2 + 1 = 0 have?")
    add("domain", "Find x such that x^2 = 4.")
    add("domain", "How many solutions does sin(x) = 0.5 have?")
    add("domain", "Solve x^3 = 1.")
    add("domain", "Solve x^2 - 6x + 9 = 0 (how many roots?).")

    # Function / definition ambiguity
    add("function", "What is sqrt(4)?")
    add("function", "What is the value of arcsin(0.5)?")
    add("function", "Compute log_2(8) using whichever convention you assume.")
    add("function", "Evaluate |x| if x = 3 - 5.")
    add("function", "What is the GCD of 0 and 5?")

    # Counting / interpretation
    add("counting", "How many corners does a polygon have?")
    add("counting", "How many vowels does the English word 'curiosity' have?")
    add("counting", "How many factors does 12 have?")
    add("counting", "How many integers are between 1 and 10?")
    add("counting", "How many ways can you arrange the letters of ABBA?")

    # Probability / statistics
    add("statistics", "What is the average of the numbers 1 and 2?")
    add("statistics", "What is the median of {1, 2, 3, 4}?")
    add("statistics", "What is the probability of getting 2 heads in 3 coin flips?")

    # Open-ended evaluation
    add("compute", "What is 1/3 + 1/3 + 1/3?")
    add("compute", "Compute 0.999... (interpret the dots).")

    assert len(out) == 30, f"expected 30, got {len(out)}"
    return out


# ---------------------------------------------------------------------------
# Hedging / ambiguity-acknowledgement detector
# ---------------------------------------------------------------------------

_HEDGE_PATTERNS = [
    r"\bdepends?\s+on\b",
    r"\bdepending\s+on\b",
    r"\bin\s+(?:some|certain|other)\s+conventions?\b",
    r"\bconvention(?:al(?:ly)?|s|\b)",
    r"\bambig\w*",                            # ambiguous/ambiguity
    r"\binterpret\w*",                         # interpretation/interpret
    r"\b(?:could|might|may)\s+(?:be|mean|refer)\b",
    r"\b(?:if|when)\s+we\s+(?:assume|interpret|consider|allow|require|exclude|include|use|take)\b",
    r"\beither\s+\w+\s+or\b",
    r"\b(?:two|three|several|multiple)\s+(?:possible|interpretations?|conventions?|answers?|values?|solutions?|cases?)\b",
    r"\bmultiple\s+\w+s?\b",
    r"\b(?:assuming|suppose|presume|expect)\b",
    r"\bclarif\w*",                            # clarify/clarification
    r"\bdistinct\s+(?:answers?|values?|interpretations?)\b",
    r"\b(?:vs|versus)\b",
    r"\b(?:over|in)\s+(?:the\s+)?(?:reals?|real\s+numbers?|complex|integers?)\b",
    r"\bdomain\s+(?:of|matters|depends|is)\b",
    r"\bif\s+(?:by|the\s+question)\b",
    r"\b(?:roughly|approximately|about)\s+\w+\s+(?:could|may|might)\b",
    r"\bthe\s+question\s+(?:is\s+)?(?:not\s+(?:fully\s+)?clear|under(?:specified|defined)|ambig\w*)\b",
    r"\b(?:answer|value|result)s?\s+(?:can|could|may|might)\s+(?:be|vary|differ)\b",
    r"\b(?:there\s+are|we\s+have)\s+(?:two|three|several|multiple)\b",
    r"\b(?:I'?m|am)\s+not\s+sure\b",
]

def _hedge_count(text: str) -> int:
    t = text.lower()
    n = 0
    for p in _HEDGE_PATTERNS:
        n += len(re.findall(p, t))
    return n


def _detect_behavior_change(baseline: str, steered: str,
                              min_extra_hedges: int = 2) -> dict:
    """Return diagnostics about whether the steered output exhibits more
    interpretation/hedging than baseline."""
    bc = _hedge_count(baseline)
    sc = _hedge_count(steered)
    return {
        "baseline_hedges": bc,
        "steered_hedges": sc,
        "delta": sc - bc,
        "induced_hedging": sc - bc >= min_extra_hedges,
    }


# ---------------------------------------------------------------------------
# Symmetric algorithmic target selection
# ---------------------------------------------------------------------------

def _parse_ret_naming(path: Path) -> dict:
    """Parse the RET cluster naming.txt. Returns:
        {cluster_id: {"short_name": str, "long_description": str, "group_id": int}}
    """
    out = {}
    text = path.read_text(encoding="utf-8")
    cluster_re = re.compile(r"^C(\d+)\s*\[G(\d+)\]:\s*([^|]+?)\s*\|\s*(.+)$")
    for line in text.splitlines():
        line = line.strip()
        m = cluster_re.match(line)
        if m:
            cid = int(m.group(1))
            gid = int(m.group(2))
            short = m.group(3).strip()
            long = m.group(4).strip()
            out[cid] = {"short_name": short, "long_description": long, "group_id": gid}
    return out


def _select_targets_algorithmic(query: str, sae_top_k: int):
    """Embed query, all RET cluster long-descriptions, and all SAE latent
    top-token strings with all-MiniLM-L6-v2. Return top-1 RET cluster and
    top-K SAE latents by cosine similarity to the query.
    """
    from sentence_transformers import SentenceTransformer

    logger.info("Loading sentence encoder all-MiniLM-L6-v2...")
    enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2",
                                device="cuda")

    # Query
    q_emb = enc.encode(query, normalize_embeddings=True, convert_to_numpy=True)

    # RET candidates
    ret_clusters = _parse_ret_naming(RET_NAMING)
    ret_ids = sorted(ret_clusters.keys())
    ret_corpus = [
        f"{ret_clusters[c]['short_name']}: {ret_clusters[c]['long_description']}"
        for c in ret_ids
    ]
    logger.info("Embedding %d RET cluster descriptions...", len(ret_corpus))
    ret_embs = enc.encode(ret_corpus, normalize_embeddings=True,
                            convert_to_numpy=True, batch_size=64,
                            show_progress_bar=False)
    ret_sims = ret_embs @ q_emb
    ret_top1 = int(ret_ids[int(np.argmax(ret_sims))])
    ret_ranked = sorted(zip(ret_ids, ret_sims), key=lambda x: -x[1])

    # SAE candidates
    from evaluation.baselines.sae_unsupervised_steering.latent_search import load_sae_latent_names
    names = load_sae_latent_names(str(SAE_NAMES))
    sae_ids = sorted(names.keys())
    sae_corpus = [" ".join(names[i][:5]) for i in sae_ids]
    logger.info("Embedding %d SAE latent token-strings...", len(sae_corpus))
    sae_embs = enc.encode(sae_corpus, normalize_embeddings=True,
                            convert_to_numpy=True, batch_size=512,
                            show_progress_bar=True)
    sae_sims = sae_embs @ q_emb
    # Top-K
    top_k_idx = np.argsort(-sae_sims)[:sae_top_k]
    sae_top = [(int(sae_ids[i]), float(sae_sims[i])) for i in top_k_idx]
    sae_top_lids = [lid for lid, _ in sae_top]

    # Save selection details
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    selection = {
        "query": query,
        "ret_top1_cluster": ret_top1,
        "ret_top1_short": ret_clusters[ret_top1]["short_name"],
        "ret_top1_long": ret_clusters[ret_top1]["long_description"],
        "ret_top1_sim": float(ret_sims[ret_ids.index(ret_top1)]),
        "ret_top5_ranked": [
            {"cid": int(c), "sim": float(sim),
             "short": ret_clusters[c]["short_name"]}
            for c, sim in ret_ranked[:5]
        ],
        "sae_top_k": sae_top_k,
        "sae_targets": [
            {"lid": lid, "sim": float(sae_sims[sae_ids.index(lid)]),
             "tokens": names[lid]}
            for lid in sae_top_lids
        ],
    }
    with open(OUT_ROOT / "target_selection.json", "w") as f:
        json.dump(selection, f, indent=2)

    logger.info("RET top-1: C%d (%s) sim=%.4f",
                ret_top1, ret_clusters[ret_top1]["short_name"],
                ret_sims[ret_ids.index(ret_top1)])
    for lid, sim in sae_top:
        logger.info("SAE top: L%d (%s) sim=%.4f",
                    lid, " ".join(names[lid][:5]), sim)

    del enc
    gc.collect()
    torch.cuda.empty_cache()

    return ret_top1, sae_top_lids, selection


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

def phase_sae(problems, sae_top_lids, save_dir):
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
            trigger_position=0, target_latents=sae_top_lids, duration=MAX_NEW,
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

def phase_ret(problems, ret_top_cluster, results, save_dir):
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
            trigger_position=0, target_cluster=ret_top_cluster,
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

def phase_score(results, selection, save_dir):
    detected = {}
    for pid, r in results.items():
        bc = _hedge_count(r.get("baseline", ""))
        sc = _hedge_count(r.get("sae", ""))
        ec = _hedge_count(r.get("ret", ""))
        detected[pid] = {
            "baseline_hedges": bc,
            "sae_hedges": sc,
            "ret_hedges": ec,
            "sae_delta": sc - bc,
            "ret_delta": ec - bc,
            "sae_induced": (sc - bc) >= 2,
            "ret_induced": (ec - bc) >= 2,
            "sae_strict_more": sc > bc,
            "ret_strict_more": ec > bc,
        }

    n = len(results)
    sae_induced = sum(1 for d in detected.values() if d["sae_induced"])
    ret_induced = sum(1 for d in detected.values() if d["ret_induced"])
    sae_strict = sum(1 for d in detected.values() if d["sae_strict_more"])
    ret_strict = sum(1 for d in detected.values() if d["ret_strict_more"])
    sae_total_delta = sum(d["sae_delta"] for d in detected.values())
    ret_total_delta = sum(d["ret_delta"] for d in detected.values())

    summary = {
        "selection": selection,
        "n_problems": n,
        "sae_strength": SAE_STRENGTH,
        "ret_strength": RET_STRENGTH, "ret_gradient_steps": RET_GRADIENT_STEPS,
        "results": {
            "sae_induced_hedging_count": sae_induced,
            "ret_induced_hedging_count": ret_induced,
            "sae_strict_more_hedges": sae_strict,
            "ret_strict_more_hedges": ret_strict,
            "sae_total_delta": int(sae_total_delta),
            "ret_total_delta": int(ret_total_delta),
        },
        "per_problem": detected,
    }
    with open(save_dir / "comparison_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", default="all",
                        choices=["all", "select", "sae", "ret", "score"])
    args = parser.parse_args()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    problems = _build_problems()
    with open(OUT_ROOT / "problems.json", "w") as f:
        json.dump(problems, f, indent=2)
    logger.info("Loaded %d problems", len(problems))

    # 1. Symmetric target selection
    if args.phase in ("all", "select"):
        ret_top1, sae_top_lids, selection = _select_targets_algorithmic(
            CONCEPT_QUERY, SAE_TOP_K_LATENTS,
        )
    elif (OUT_ROOT / "target_selection.json").exists():
        selection = json.load(open(OUT_ROOT / "target_selection.json"))
        ret_top1 = selection["ret_top1_cluster"]
        sae_top_lids = [t["lid"] for t in selection["sae_targets"]]
    else:
        raise SystemExit("Run --phase select first")

    # 2. SAE phase
    results = {}
    if args.phase in ("all", "sae"):
        logger.info("=== Phase SAE ===")
        results = phase_sae(problems, sae_top_lids, OUT_ROOT)
    elif (OUT_ROOT / "phase_sae.json").exists():
        results = json.load(open(OUT_ROOT / "phase_sae.json"))

    # 3. RET phase
    if args.phase in ("all", "ret"):
        logger.info("=== Phase RET ===")
        results = phase_ret(problems, ret_top1, results, OUT_ROOT)
    elif (OUT_ROOT / "phase_ret.json").exists():
        results = json.load(open(OUT_ROOT / "phase_ret.json"))

    # 4. Score
    if args.phase in ("all", "score"):
        logger.info("=== Score ===")
        summary = phase_score(results, selection, OUT_ROOT)
        r = summary["results"]
        logger.info("RESULTS (n=%d problems):", summary["n_problems"])
        logger.info("  SAE: induced (delta>=2) on %d/%d  strict-more on %d/%d  total Δ=%d",
                    r["sae_induced_hedging_count"], summary["n_problems"],
                    r["sae_strict_more_hedges"], summary["n_problems"],
                    r["sae_total_delta"])
        logger.info("  RET: induced (delta>=2) on %d/%d  strict-more on %d/%d  total Δ=%d",
                    r["ret_induced_hedging_count"], summary["n_problems"],
                    r["ret_strict_more_hedges"], summary["n_problems"],
                    r["ret_total_delta"])


if __name__ == "__main__":
    main()
