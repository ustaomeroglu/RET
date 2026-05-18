#!/usr/bin/env python
"""
Compute SYCON-Bench metrics (ToF, NoF) from judge results.

Usage:
    python scripts/sycophancy/run_metrics.py \
        --judge_results data/sycophancy/debate/Qwen2.5-14B-Instruct/judge_results.jsonl

    # Custom output path
    python scripts/sycophancy/run_metrics.py \
        --judge_results data/sycophancy/debate/Qwen2.5-14B-Instruct/judge_results.jsonl \
        --output data/sycophancy/debate/Qwen2.5-14B-Instruct/metrics.json
"""
import argparse
import json
import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="SYCON-Bench metrics")
    p.add_argument("--judge_results", required=True, help="Path to judge_results.jsonl")
    p.add_argument("--output", default=None, help="Path to metrics.json (default: same dir)")
    p.add_argument("--num_turns", type=int, default=5)
    return p.parse_args()


def main():
    args = parse_args()

    from evaluation.datasets.sycophancy.metrics import aggregate_metrics

    output = args.output
    if output is None:
        output = os.path.join(os.path.dirname(args.judge_results), "metrics.json")

    metrics = aggregate_metrics(args.judge_results, num_turns=args.num_turns)

    # Save (without per-conversation detail for the summary)
    summary = {k: v for k, v in metrics.items() if k != "per_conversation"}
    with open(output, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== SYCON-Bench Metrics ===")
    print(f"  Conversations: {metrics['n']}")
    print(f"  Mean ToF: {metrics['mean_tof']:.3f} (std={metrics['std_tof']:.3f})")
    print(f"  Mean NoF: {metrics['mean_nof']:.3f} (std={metrics['std_nof']:.3f})")
    print(f"\n  Saved to {output}")


if __name__ == "__main__":
    main()
