#!/usr/bin/env python3
"""CLI entry point for SAE-based unsupervised steering.

Examples
--------
    python scripts/sae_steering/run_sae_steering.py \\
        --llm-config configs/GptOss20B/baselines/sae_l11.yaml \\
        --sae-latent-names outputs/GptOss20B/numina_math/baselines/sae_l11/interp_score/sae_latent_names.json \\
        --rule "from_step 0 steer_sae 1770 for 600" \\
        --strength 0.5 --max-new-tokens 1024 \\
        --prompt "Compute 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10 + 11." \\
        --output-dir outputs/GptOss20B/numina_math/baselines/sae_l11/sae_steering/smoke_steer_L1770

Rule grammar
------------
    from_step <N> steer_sae <id1>,<id2>,... for <M>
    avoid_sae <id1>,<id2>,... activation_threshold <T> [max_tokens <N>]
        [from_step <S>] [sticky_duration <D>]
"""
from __future__ import annotations

import argparse
import logging
import re
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


_ATTRACT_RE = re.compile(
    r"from_step\s+(\d+)\s+steer_sae\s+([0-9,\s]+?)\s+for\s+(\d+)\s*$",
)

_AVOID_RE = re.compile(
    r"avoid_sae\s+([0-9,\s]+?)\s+activation_threshold\s+([0-9.eE+-]+)"
    r"(?:\s+max_tokens\s+(\d+))?"
    r"(?:\s+from_step\s+(\d+))?"
    r"(?:\s+sticky_duration\s+(\d+))?"
    r"\s*$",
)


def _parse_id_list(s: str) -> list[int]:
    parts = [p.strip() for p in s.split(",")]
    parts = [p for p in parts if p]
    if not parts:
        raise ValueError(f"Empty latent-id list in {s!r}")
    return [int(p) for p in parts]


def parse_rule(text: str):
    """Parse one rule string into an ``SAESteeringRule``."""
    from evaluation.baselines.sae_unsupervised_steering.config import SAESteeringRule

    text = text.strip()

    m = _AVOID_RE.match(text)
    if m is not None:
        lids = _parse_id_list(m.group(1))
        threshold = float(m.group(2))
        max_tokens = int(m.group(3)) if m.group(3) else None
        from_step = int(m.group(4)) if m.group(4) else None
        sticky = int(m.group(5)) if m.group(5) else 1
        return SAESteeringRule(
            avoid_latents=lids,
            avoid_pre_threshold=threshold,
            avoid_max_tokens=max_tokens,
            avoid_from_step=from_step,
            avoid_sticky_duration=sticky,
        )

    m = _ATTRACT_RE.match(text)
    if m is not None:
        trigger_position = int(m.group(1))
        lids = _parse_id_list(m.group(2))
        duration = int(m.group(3))
        return SAESteeringRule(
            trigger_position=trigger_position,
            target_latents=lids,
            duration=duration,
        )

    raise ValueError(
        f"Cannot parse rule: {text!r}\n"
        "Expected one of:\n"
        "  from_step <N> steer_sae <ids> for <M>\n"
        "  avoid_sae <ids> activation_threshold <T> "
        "[max_tokens <N>] [from_step <S>] [sticky_duration <D>]"
    )


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="SAE-based unsupervised steering of LLM generation",
    )

    parser.add_argument(
        "--llm-config", type=str, required=True,
        help="RET-style YAML config (e.g. configs/GptOss20B/baselines/sae_l11.yaml)",
    )
    parser.add_argument(
        "--sae-release", type=str, default="andyrdt/saes-gpt-oss-20b",
    )
    parser.add_argument(
        "--sae-id", type=str, default="resid_post_layer_11/trainer_0",
    )
    parser.add_argument("--sae-latent-names", type=str, default=None)
    parser.add_argument(
        "--direction-source", choices=["decoder", "encoder"], default="decoder",
    )

    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--prompt", type=str)
    grp.add_argument("--prompt-file", type=str)

    parser.add_argument(
        "--rule", action="append", default=[],
        help="Steering rule string. Can be specified multiple times.",
    )

    parser.add_argument("--strength", type=float, default=0.5)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k-diag", type=int, default=16)

    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args(argv)

    if args.prompt_file:
        from pathlib import Path
        args.prompt = Path(args.prompt_file).read_text(encoding="utf-8").strip()

    if not args.rule:
        parser.error("At least one --rule is required")
    rules = [parse_rule(r) for r in args.rule]

    from evaluation.baselines.sae_unsupervised_steering.config import SAESteeringConfig
    from evaluation.baselines.sae_unsupervised_steering.pipeline import run_sae_steering

    config = SAESteeringConfig(
        sae_release=args.sae_release,
        sae_id=args.sae_id,
        llm_config_path=args.llm_config,
        direction_source=args.direction_source,
        sae_latent_names_path=args.sae_latent_names,
        rules=rules,
        strength=args.strength,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        device=args.device,
        output_dir=args.output_dir,
        top_k_diag=args.top_k_diag,
    )

    logger.info("Rules: %s", rules)
    logger.info(
        "SAE steering: strength=%.3f, direction_source=%s",
        config.strength, config.direction_source,
    )

    result = run_sae_steering(config)

    print("\n=== Steered Output ===")
    print(result["text"])
    print(f"\n=== {len(result['diagnostics'])} tokens generated ===")

    steered = [d for d in result["diagnostics"] if d.get("was_steered")]
    if steered:
        print(f"Steered {len(steered)} tokens")
        deltas = [d["delta_h_norm"] for d in steered]
        print(f"  delta_h_norm range: [{min(deltas):.3f}, {max(deltas):.3f}]")


if __name__ == "__main__":
    main()
