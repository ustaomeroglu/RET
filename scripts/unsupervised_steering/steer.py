#!/usr/bin/env python3
"""Workhorse single-experiment steering runner.

One invocation = one steered (or baseline) generation. Specify the
clustering checkpoint, the problem (id + prompt), how to steer (target /
direction / duration / when), and where to write outputs. No experiment
defaults are hardcoded — every knob is a CLI flag.

Baseline mode: pass no ``--target-*`` / ``--avoid-*`` flag. The steering
hook still attaches (so per-token cluster diagnostics are recorded) but
no rule ever fires, so the model generates unsteered.

Examples
--------
::

    # Baseline (no steering, but full diagnostics)
    python scripts/unsupervised_steering/steer.py \\
        --clustering-dir outputs/GptOss20B/numina_math/ret_jepa_auth_l11/clustering \\
        --problem-id amgm_vs_calc \\
        --user-prompt "For x > 0, find the minimum value of x + 4/x."

    # Attract toward cluster 29 from the very first generated token
    python scripts/unsupervised_steering/steer.py \\
        --clustering-dir outputs/.../clustering \\
        --problem-id amgm_vs_calc \\
        --user-prompt "..." \\
        --target-cluster 29 --duration 50 --strength 0.5 --gradient-steps 3

    # Avoid group 5 (margin-gated repulsion) once we're past step 20
    python scripts/unsupervised_steering/steer.py \\
        --clustering-dir outputs/.../clustering \\
        --problem-id avoid_geometry \\
        --user-prompt "..." \\
        --avoid-group 5 --avoid-margin-threshold 0.15 \\
        --avoid-sticky-duration 5 --avoid-from-step 20 \\
        --strength 0.5 --gradient-steps 3

    # Multi-attractor (parallel rules)
    python scripts/unsupervised_steering/steer.py \\
        --clustering-dir outputs/.../clustering \\
        --problem-id telescoping \\
        --user-prompt "..." --target-clusters 6,10,25 --duration 50
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict
from pathlib import Path

import torch
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("steer")

REPO_ROOT = Path(__file__).resolve().parents[2]


SYSTEM_PROMPT = "You are a helpful assistant."
DEFAULT_PROBLEMS_FILE = REPO_ROOT / "data/math_problems/problems.json"

# Marker used in pre-templated numina test prompts where generation starts.
NUMINA_CUT_MARKER = "<|start|>assistant"


# ---------------------------------------------------------------------------
# Prompt sources
# ---------------------------------------------------------------------------


def build_chat_prompt(model_name: str, user_prompt: str) -> str:
    """Apply the LLM's chat template to a single user message."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    return tok.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT},
         {"role": "user", "content": user_prompt}],
        tokenize=False, add_generation_prompt=True,
    )


def load_curated_problem(problems_file: str | Path, problem_id: str) -> dict:
    """Look up a problem by id in a curated problems JSON file.

    Schema (top-level JSON array):
        [{"id": str, "prompt": str, "tags": [str, ...] (optional), ...}, ...]
    """
    problems_file = Path(problems_file)
    if not problems_file.exists():
        raise FileNotFoundError(f"Problems file not found: {problems_file}")
    data = json.loads(problems_file.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{problems_file}: top-level JSON must be a list")
    for entry in data:
        if entry.get("id") == problem_id:
            if "prompt" not in entry:
                raise ValueError(
                    f"Problem {problem_id!r} in {problems_file} has no 'prompt' field"
                )
            return entry
    available = ", ".join(repr(e["id"]) for e in data if "id" in e)
    raise KeyError(
        f"Problem id {problem_id!r} not found in {problems_file}. "
        f"Available: {available or '(none)'}"
    )


def load_numina_test_prompt(model_short: str, index: int) -> str:
    """Load a pre-templated numina-math test prompt, cut at the assistant
    marker so generation replays the model's answer from scratch.

    Path convention: ``data/numina_math_64k_samples/<model_short>/test``.
    """
    import datasets as hfds

    path = REPO_ROOT / "data/numina_math_64k_samples" / model_short / "test"
    if not path.exists():
        raise FileNotFoundError(
            f"Numina-math test split not found at {path}. "
            f"(Expected layout: data/numina_math_64k_samples/<model_short>/test)"
        )
    ds = hfds.load_from_disk(str(path))
    if index < 0 or index >= len(ds):
        raise IndexError(
            f"Numina test index {index} out of range [0, {len(ds)})"
        )
    txt = ds[index]["text"]
    cut = txt.find(NUMINA_CUT_MARKER)
    if cut < 0:
        raise RuntimeError(
            f"Numina test doc {index} has no {NUMINA_CUT_MARKER!r} marker"
        )
    return txt[: cut + len(NUMINA_CUT_MARKER)]


def resolve_prompt(args, model_name: str) -> tuple[str, str, str]:
    """Pick one of the four prompt sources and return (raw_user_prompt,
    chat_formatted_prompt, problem_id).

    ``raw_user_prompt`` is the plain user message (or empty for numina
    samples — those are already chat-templated). ``chat_formatted_prompt``
    is what gets fed to the LLM.
    """
    if args.problem_id is not None:
        entry = load_curated_problem(args.problems_file, args.problem_id)
        raw = entry["prompt"]
        return raw, build_chat_prompt(model_name, raw), args.problem_id

    if args.numina_test_index is not None:
        from ret.config import llm_short_name, LLMConfig
        # We need the LLM short name to find the dataset path. Construct a
        # minimal LLMConfig with just the name set.
        short = llm_short_name(LLMConfig(name=model_name, layer_index=0))
        chat = load_numina_test_prompt(short, args.numina_test_index)
        pid = args.problem_id_override or f"numina_test_{args.numina_test_index}"
        return "", chat, pid

    if args.prompt_file is not None:
        raw = Path(args.prompt_file).read_text(encoding="utf-8").strip()
    else:
        raw = args.user_prompt
    pid = args.problem_id_override or "adhoc"
    return raw, build_chat_prompt(model_name, raw), pid


def parse_int_list(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def derive_condition_spec(args) -> tuple:
    """Inspect target-/avoid- flags and build a condition tuple.

    Mutually exclusive: at most one of the flags below may be set. If none,
    returns ``("baseline",)``.
    """
    flags = [
        ("steer_C",   args.target_cluster),
        ("steer_G",   args.target_group),
        ("steer_Cs",  parse_int_list(args.target_clusters) if args.target_clusters else None),
        ("steer_Gs",  parse_int_list(args.target_groups) if args.target_groups else None),
        ("avoid_C",   args.avoid_cluster),
        ("avoid_G",   args.avoid_group),
        ("avoid_Cs",  parse_int_list(args.avoid_clusters) if args.avoid_clusters else None),
        ("avoid_Gs",  parse_int_list(args.avoid_groups) if args.avoid_groups else None),
    ]
    chosen = [(kind, val) for kind, val in flags if val is not None]
    if len(chosen) > 1:
        raise SystemExit(
            "ERROR: pass at most one of --target-cluster / --target-group / "
            "--target-clusters / --target-groups / --avoid-cluster / "
            "--avoid-group / --avoid-clusters / --avoid-groups."
        )
    if not chosen:
        return ("baseline",)
    kind, val = chosen[0]
    return (kind, val)


def derive_output_dir(
    *, output_dir, output_base, clustering_dir,
    K, seed, window_size, tau, alpha,
    problem_id, experiment_tag, condition_label,
) -> Path:
    """Choose the run's output directory.

    Order of precedence:
      1. ``--output-dir PATH`` — used verbatim.
      2. ``--output-base PATH`` — uses ``<base>/<problem_id>/<exp_tag>/<cond>``.
      3. Default — derives from the clustering analysis subtree.
    """
    if output_dir is not None:
        return Path(output_dir)
    if output_base is not None:
        return Path(output_base) / problem_id / experiment_tag / condition_label
    # Default: anchor under the clustering analysis subtree
    return (
        Path(clustering_dir)
        / f"w{window_size}" / f"K{K}" / f"seed{seed}" / "analysis"
        / f"weighted_tau{tau}" / f"grouped_alpha{alpha}"
        / "unsupervised_steering" / "problems"
        / problem_id / experiment_tag / condition_label
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Clustering
    p.add_argument("--clustering-dir", required=True,
                   help="Path to clustering/ (must contain clustering_meta.json)")

    # Prompt source — one of these four is required
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--problem-id",
                   help="Look up the problem by id in --problems-file.")
    g.add_argument("--numina-test-index", type=int,
                   help="Index into data/numina_math_64k_samples/<model>/test.")
    g.add_argument("--user-prompt",
                   help="Inline user message (will be chat-template-wrapped).")
    g.add_argument("--prompt-file",
                   help="File containing the user message (chat-template-wrapped).")

    p.add_argument("--problems-file", default=str(DEFAULT_PROBLEMS_FILE),
                   help="Curated problems JSON (default: %(default)s).")
    p.add_argument("--problem-id-override", default=None,
                   help="Override the problem-id used in output dirs (default: "
                        "the chosen --problem-id, 'numina_test_<N>' for numina "
                        "samples, or 'adhoc' for raw prompts).")

    # Steering target (mutually exclusive; none = baseline)
    p.add_argument("--target-cluster", type=int, default=None)
    p.add_argument("--target-group",   type=int, default=None)
    p.add_argument("--target-clusters", type=str, default=None,
                   help='Comma-separated cluster ids, e.g. "10,25,38".')
    p.add_argument("--target-groups",   type=str, default=None,
                   help='Comma-separated group ids.')
    p.add_argument("--avoid-cluster",  type=int, default=None)
    p.add_argument("--avoid-group",    type=int, default=None)
    p.add_argument("--avoid-clusters", type=str, default=None)
    p.add_argument("--avoid-groups",   type=str, default=None)

    # Trigger ("when")
    p.add_argument("--trigger-position", type=int, default=0,
                   help="Generation step at which the attractor rule fires.")

    # Duration and avoid-rule parameters
    p.add_argument("--duration", type=int, default=50,
                   help="How many tokens the attractor holds (attractor rules).")
    p.add_argument("--avoid-margin-threshold", type=float, default=0.15)
    p.add_argument("--avoid-sticky-duration", type=int, default=5)
    p.add_argument("--avoid-from-step", type=int, default=0)

    # Steering strength
    p.add_argument("--strength", type=float, default=0.5,
                   help="δh = strength * ||h|| * grad_dir. 0.0 = pure baseline.")
    p.add_argument("--gradient-steps", type=int, default=3,
                   help="K=1 single-step; K>1 iterative gradient refinement.")

    # Closure-predictor OOD regularization (optional)
    p.add_argument("--lambda-pred", type=float, default=0.0)
    p.add_argument("--epsilon-pred", type=float, default=0.0)
    p.add_argument("--closure-dir", type=str, default=None,
                   help="Closure-predictor dir; required iff --lambda-pred > 0.")

    # Clustering selection (defaults match the canonical training)
    p.add_argument("--K", type=int, default=64)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--window-size", type=int, default=1)
    p.add_argument("--tau", type=float, default=0.1)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--metric", choices=["cosine", "l2"], default="cosine")

    # Generation
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)

    # System
    p.add_argument("--device", default="cuda")
    p.add_argument("--torch-seed", type=int, default=12345,
                   help="torch.manual_seed for generation reproducibility.")

    # Output
    p.add_argument("--output-dir", default=None,
                   help="Explicit per-run output directory (overrides --output-base).")
    p.add_argument("--output-base", default=None,
                   help="Base dir; output becomes <base>/<problem_id>/<exp_tag>/<cond>/.")
    p.add_argument("--experiment-tag", default=None,
                   help="Default: 'str{strength}_gs{gradient_steps}[_lp{λ}]'.")

    return p


def main(argv: list[str] | None = None):
    args = build_parser().parse_args(argv)

    # Validate closure-reg flags
    if args.lambda_pred > 0.0 and not args.closure_dir:
        raise SystemExit("--lambda-pred > 0 requires --closure-dir")

    # 1. Condition + label
    from evaluation.unsupervised_steering.conditions import build_rules_for_condition
    condition = derive_condition_spec(args)
    cond_label, rules = build_rules_for_condition(
        condition,
        attractor_duration=args.duration,
        avoid_margin_threshold=args.avoid_margin_threshold,
        avoid_sticky_duration=args.avoid_sticky_duration,
        trigger_position=args.trigger_position,
        avoid_from_step=args.avoid_from_step,
    )

    # 2. Experiment tag
    if args.experiment_tag is None:
        lp = f"_lp{args.lambda_pred}" if args.lambda_pred > 0.0 else ""
        args.experiment_tag = f"str{args.strength}_gs{args.gradient_steps}{lp}"

    # 3. Resolve prompt source — needs model name first
    meta = json.loads((Path(args.clustering_dir) / "clustering_meta.json").read_text())
    from ret.config import load_config
    model_name = load_config(meta["config_yaml_path"]).llm.name
    raw_user_prompt, chat_prompt, problem_id = resolve_prompt(args, model_name)

    # 4. Output dir
    out_dir = derive_output_dir(
        output_dir=args.output_dir, output_base=args.output_base,
        clustering_dir=args.clustering_dir,
        K=args.K, seed=args.seed, window_size=args.window_size,
        tau=args.tau, alpha=args.alpha,
        problem_id=problem_id, experiment_tag=args.experiment_tag,
        condition_label=cond_label,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output dir: %s", out_dir)
    logger.info("Problem:    %s", problem_id)
    logger.info("Condition:  %s (%d rule%s)", cond_label, len(rules),
                "" if len(rules) == 1 else "s")

    # 5. Persist the raw user prompt + invocation context next to the outputs
    if raw_user_prompt:
        (out_dir / "user_prompt.txt").write_text(raw_user_prompt)

    # 6. SteeringConfig — strength=0 / no rules => model generates as baseline
    from evaluation.unsupervised_steering.config import SteeringConfig
    is_baseline = condition[0] == "baseline"
    cfg = SteeringConfig(
        clustering_dir=str(args.clustering_dir),
        window_size=args.window_size, K=args.K, seed=args.seed,
        tau=args.tau, alpha=args.alpha,
        rules=rules,
        metric=args.metric,
        strength=0.0 if is_baseline else args.strength,
        gradient_steps=1 if is_baseline else args.gradient_steps,
        lambda_pred=0.0 if is_baseline else args.lambda_pred,
        epsilon_pred=0.0 if is_baseline else args.epsilon_pred,
        prompt=chat_prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature, top_p=args.top_p,
        device=args.device,
        output_dir=str(out_dir),
    )

    # 7. Optional regularizer (closure-predictor)
    regularizer = None
    if cfg.lambda_pred > 0.0:
        from evaluation.closure.io import pick_best_closure_predictor
        predictor, _info = pick_best_closure_predictor(
            args.closure_dir, device=args.device, loss_type="mse",
        )
        regularizer = {"predictor": predictor}

    # 8. Generate
    torch.manual_seed(args.torch_seed)
    from evaluation.unsupervised_steering import run_unsupervised_steering
    t0 = time.time()
    result = run_unsupervised_steering(cfg, regularizer=regularizer)
    dt = time.time() - t0

    # 9. Compact summary
    n_steered = sum(1 for d in result["diagnostics"] if d.get("was_steered"))
    summary = {
        "problem_id": problem_id,
        "condition": cond_label,
        "experiment_tag": args.experiment_tag,
        "user_prompt": raw_user_prompt or None,
        "seconds": round(dt, 2),
        "n_generated": len(result["diagnostics"]),
        "n_steered": n_steered,
        "config": asdict(cfg),
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    logger.info(
        "Done %s/%s in %.1fs (%d generated, %d steered) -> %s",
        problem_id, cond_label, dt, len(result["diagnostics"]),
        n_steered, out_dir,
    )


if __name__ == "__main__":
    main()
