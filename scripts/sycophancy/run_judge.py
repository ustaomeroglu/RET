#!/usr/bin/env python
"""
Run Anthropic Claude judge on SYCON-Bench conversations.

Reads conversations.jsonl, evaluates each turn, writes judge_results.jsonl
in the same directory. Requires ANTHROPIC_API_KEY environment variable.

Two modes:
  --input <file>         : judge a single conversations.jsonl
  --input-dir <tree>     : walk a supervised-steering output tree and judge
                            every conversations.jsonl found under
                            ``<tree>/*/factor_*/`` or ``<tree>/alpha_*/``
                            (idempotent — already-judged files are skipped
                            per-conversation by the judge itself).

Usage:
    # Single file
    python scripts/sycophancy/run_judge.py \
        --input data/sycophancy/debate/Qwen2.5-14B-Instruct/conversations.jsonl \
        --scenario debate

    # Steering output tree
    python scripts/sycophancy/run_judge.py \
        --input-dir outputs/.../supervised_steering \
        --scenario debate
"""
import argparse
import glob
import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="SYCON-Bench Anthropic judge")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", help="Path to a single conversations.jsonl")
    src.add_argument("--input-dir",
                     help="Top-level directory to walk for conversations.jsonl "
                          "(patterns: <dir>/*/factor_*/ and <dir>/alpha_*/)")
    p.add_argument("--scenario", required=True,
                   choices=["debate", "false_presupposition"])
    p.add_argument("--output", default=None,
                   help="Path to judge_results.jsonl (single-file mode; "
                        "default: same dir as --input). Ignored in --input-dir mode.")
    p.add_argument("--judge_model", default="claude-sonnet-4-20250514",
                   help="Anthropic model to use as judge")
    p.add_argument("--max_concurrent", type=int, default=10,
                   help="Max concurrent async API requests")
    return p.parse_args()


def _discover_files(input_dir: str) -> list[str]:
    """Find conversations.jsonl files under a supervised-steering output tree."""
    pattern1 = os.path.join(input_dir, "*", "factor_*", "conversations.jsonl")
    pattern2 = os.path.join(input_dir, "alpha_*", "conversations.jsonl")
    files = sorted(set(glob.glob(pattern1)) | set(glob.glob(pattern2)))
    if not files:
        raise FileNotFoundError(
            f"No conversations.jsonl found under {input_dir} "
            f"(patterns: {pattern1}, {pattern2})"
        )
    return files


def main():
    args = parse_args()

    from evaluation.datasets.sycophancy.judge import judge_conversation_file  # noqa: E402

    if args.input is not None:
        files = [args.input]
        outputs = [args.output or os.path.join(os.path.dirname(args.input), "judge_results.jsonl")]
        log_root = os.path.dirname(args.input) or "."
    else:
        files = _discover_files(args.input_dir)
        outputs = [os.path.join(os.path.dirname(f), "judge_results.jsonl") for f in files]
        log_root = args.input_dir
        logger.info("Found %d conversations.jsonl files under %s", len(files), args.input_dir)

    logger.info("Scenario: %s, Judge model: %s", args.scenario, args.judge_model)

    for i, (conv_path, out_path) in enumerate(zip(files, outputs)):
        rel = os.path.relpath(conv_path, log_root)
        logger.info("[%d/%d] Judging %s", i + 1, len(files), rel)
        judge_conversation_file(
            input_jsonl=conv_path,
            output_jsonl=out_path,
            scenario=args.scenario,
            judge_model_name=args.judge_model,
            max_concurrent=args.max_concurrent,
        )

    logger.info("Done. %d file(s) judged.", len(files))


if __name__ == "__main__":
    main()
