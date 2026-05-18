#!/usr/bin/env python3
"""Generate GPT-OSS 20B solutions for NuminaMath problems and cache as Arrow datasets.

Uses vLLM for fast inference with tensor parallelism across all available GPUs.
No model internals needed — purely text-in, text-out generation.

Loads pre-split NuminaMath data from ``data/numina_math/{train,val,test}/``,
subsamples stratified by source, generates model completions at the given
temperature, and saves the results as Arrow datasets ready for RET training.

Prompt examples are saved *before* generation starts so you can verify
formatting without waiting for the full run.

Resume support: partial results are written to a JSONL checkpoint file.
On restart the script skips already-generated indices.

Prerequisites:
    conda run -n eet pip install vllm

Usage:
    conda run -n eet python scripts/generate_numina_completions.py \
        --max-train-samples 64000 --max-val-samples 3000 --max-test-samples 1000
"""

import argparse
import json
import logging
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from datasets import Dataset, load_from_disk
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def stratified_subsample(sources: list, max_samples: int, seed: int = 42) -> list[int]:
    """Return indices for a stratified subsample of *max_samples* items.

    Rare source categories (< 50 occurrences) are grouped into "other"
    before stratification to avoid sklearn errors.
    """
    n = len(sources)
    if max_samples >= n:
        return list(range(n))

    counts = Counter(sources)
    strat_labels = [s if counts[s] >= 50 else "other" for s in sources]

    all_idx = np.arange(n)
    _, sample_idx = train_test_split(
        all_idx,
        test_size=max_samples,
        random_state=seed,
        stratify=strat_labels,
    )
    return sorted(sample_idx.tolist())


def load_checkpoint(path: Path) -> dict[int, dict]:
    """Load previously generated results from a JSONL checkpoint file.

    Returns a dict mapping original dataset index -> row dict.
    """
    results: dict[int, dict] = {}
    if not path.exists():
        return results
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            results[row["original_index"]] = row
    logger.info("Resumed %d previously generated samples from %s", len(results), path)
    return results


def append_checkpoint(path: Path, rows: list[dict]) -> None:
    """Append result rows to the JSONL checkpoint."""
    with open(path, "a") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def format_prompt(problem: str, tokenizer, system_text: str, **template_kwargs) -> str:
    """Format a NuminaMath problem as a chat prompt for generation."""
    messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": f"Problem: {problem}\n\nSolution:"},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        **template_kwargs,
    )


def save_prompt_examples(
    dataset,
    indices: list[int],
    tokenizer,
    output_path: Path,
    system_text: str,
    n_examples: int = 10,
    **template_kwargs,
) -> None:
    """Save a few fully-formatted prompt examples to JSONL for inspection."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for idx in indices[:n_examples]:
            problem = dataset[idx]["problem"]
            formatted = format_prompt(problem, tokenizer, system_text, **template_kwargs)
            entry = {
                "index": idx,
                "source": dataset[idx]["source"],
                "problem": problem,
                "formatted_prompt": formatted,
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    logger.info("Saved %d prompt examples to %s", min(n_examples, len(indices)), output_path)


# ---------------------------------------------------------------------------
# Per-split generation
# ---------------------------------------------------------------------------

def generate_for_split(
    dataset,
    indices: list[int],
    llm,
    tokenizer,
    *,
    system_text: str,
    checkpoint_path: Path,
    max_new_tokens: int,
    temperature: float,
    seed: int,
    **template_kwargs,
) -> Dataset:
    """Generate solutions for selected indices with checkpoint/resume support."""
    from vllm import SamplingParams

    # Load any previously completed results
    completed = load_checkpoint(checkpoint_path)
    todo_indices = [i for i in indices if i not in completed]
    logger.info(
        "Split: %d selected, %d already done, %d remaining",
        len(indices), len(completed), len(todo_indices),
    )

    if todo_indices:
        # Format prompts for remaining indices
        problems = [dataset[i]["problem"] for i in todo_indices]
        prompts = [
            format_prompt(prob, tokenizer, system_text, **template_kwargs)
            for prob in problems
        ]

        # Generate with vLLM
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_new_tokens,
            seed=seed,
        )
        outputs = llm.generate(prompts, sampling_params)

        # Write results to checkpoint — preserve exact token IDs via
        # decode(skip_special_tokens=False) so the text round-trips through
        # encode() to the identical token sequence the model saw during generation.
        new_rows = []
        for idx, prob, output in zip(todo_indices, problems, outputs):
            prompt_ids = list(output.prompt_token_ids)
            gen_ids = list(output.outputs[0].token_ids)
            full_ids = prompt_ids + gen_ids
            full_text = tokenizer.decode(full_ids, skip_special_tokens=False)
            gen_text = tokenizer.decode(gen_ids, skip_special_tokens=False)
            row = {
                "original_index": idx,
                "source": dataset[idx]["source"],
                "problem": prob,
                "generation_raw": gen_text,
                "text": full_text,
                "prompt_length": len(prompt_ids),
                "messages": [
                    {"role": "user", "content": f"Problem: {prob}\n\nSolution:"},
                    {"role": "assistant", "content": gen_text},
                ],
            }
            completed[idx] = row
            new_rows.append(row)
        append_checkpoint(checkpoint_path, new_rows)

    # Assemble final dataset in the order of `indices`
    rows = {"text": [], "source": [], "problem": [], "generation_raw": [], "prompt_length": [], "messages": []}
    for idx in indices:
        r = completed[idx]
        rows["text"].append(r["text"])
        rows["source"].append(r["source"])
        rows["problem"].append(r["problem"])
        rows["generation_raw"].append(r["generation_raw"])
        rows["prompt_length"].append(r["prompt_length"])
        rows["messages"].append(r["messages"])

    return Dataset.from_dict(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate GPT-OSS 20B solutions for NuminaMath and cache as Arrow.",
    )
    parser.add_argument("--input-dir", type=str, default="data/numina_math",
                        help="Directory with existing train/val/test Arrow splits")
    parser.add_argument("--output-dir", type=str, default="data/numina_math_64k_samples/GptOss20B",
                        help="Output directory for generated Arrow datasets")
    parser.add_argument("--model", type=str, default="openai/gpt-oss-20b",
                        help="HuggingFace model ID")
    parser.add_argument("--max-train-samples", type=int, default=5000)
    parser.add_argument("--max-val-samples", type=int, default=500)
    parser.add_argument("--max-test-samples", type=int, default=250)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--system-message", type=str,
                        default="You are a helpful assistant.")
    parser.add_argument("--reasoning-effort", type=str, default="medium",
                        help="reasoning_effort kwarg for chat template (GPT-OSS)")
    parser.add_argument("--tensor-parallel-size", type=int, default=0,
                        help="Number of GPUs for tensor parallelism (0 = all available)")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    template_kwargs = {"reasoning_effort": args.reasoning_effort}

    # ------------------------------------------------------------------
    # 1. Load model with vLLM
    # ------------------------------------------------------------------
    import torch
    from vllm import LLM
    from transformers import AutoTokenizer

    tp_size = args.tensor_parallel_size
    if tp_size <= 0:
        tp_size = torch.cuda.device_count()
    logger.info("Loading %s with vLLM (tensor_parallel_size=%d) ...", args.model, tp_size)

    llm = LLM(
        model=args.model,
        tensor_parallel_size=tp_size,
        trust_remote_code=True,
        seed=args.seed,
    )

    # Load tokenizer separately for prompt formatting
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ------------------------------------------------------------------
    # 2. Process each split
    # ------------------------------------------------------------------
    splits = [
        ("train", args.max_train_samples),
        ("val", args.max_val_samples),
        ("test", args.max_test_samples),
    ]

    generation_info: dict = {
        "model": args.model,
        "engine": "vllm",
        "tensor_parallel_size": tp_size,
        "temperature": args.temperature,
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "system_message": args.system_message,
        "reasoning_effort": args.reasoning_effort,
        "splits": {},
    }

    prompt_examples_saved = False

    for split_name, max_samples in splits:
        split_path = input_dir / split_name
        if not split_path.exists():
            logger.warning("Split %s not found at %s, skipping", split_name, split_path)
            continue

        logger.info("=== Processing split: %s ===", split_name)
        dataset = load_from_disk(str(split_path))
        sources = dataset["source"]

        # Stratified subsample
        indices = stratified_subsample(sources, max_samples, seed=args.seed)
        logger.info(
            "Subsampled %d / %d samples (stratified by source)",
            len(indices), len(dataset),
        )

        # Save prompt examples (once, from the first split)
        if not prompt_examples_saved:
            save_prompt_examples(
                dataset, indices, tokenizer,
                output_dir / "prompt_examples.jsonl",
                system_text=args.system_message,
                **template_kwargs,
            )
            prompt_examples_saved = True

        # Generate
        checkpoint_path = output_dir / f"_checkpoint_{split_name}.jsonl"
        gen_ds = generate_for_split(
            dataset, indices, llm, tokenizer,
            system_text=args.system_message,
            checkpoint_path=checkpoint_path,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            seed=args.seed,
            **template_kwargs,
        )

        # Save Arrow dataset
        arrow_path = output_dir / split_name
        arrow_path.mkdir(parents=True, exist_ok=True)
        gen_ds.save_to_disk(str(arrow_path))
        logger.info("Saved %s split (%d samples) -> %s", split_name, len(gen_ds), arrow_path)

        # Collect stats
        gen_lengths = [len(g) for g in gen_ds["generation_raw"]]
        generation_info["splits"][split_name] = {
            "total_in_source": len(dataset),
            "subsampled": len(indices),
            "mean_generation_chars": int(np.mean(gen_lengths)) if gen_lengths else 0,
            "median_generation_chars": int(np.median(gen_lengths)) if gen_lengths else 0,
            "max_generation_chars": int(np.max(gen_lengths)) if gen_lengths else 0,
            "min_generation_chars": int(np.min(gen_lengths)) if gen_lengths else 0,
            "empty_generations": sum(1 for g in gen_lengths if g == 0),
            "source_distribution": dict(Counter(gen_ds["source"])),
        }

    # ------------------------------------------------------------------
    # 3. Save metadata
    # ------------------------------------------------------------------
    info_path = output_dir / "generation_info.json"
    with open(info_path, "w") as f:
        json.dump(generation_info, f, indent=2)
    logger.info("Saved generation info -> %s", info_path)

    print(f"\nDone. Generated data saved under: {output_dir}")
    for split_name, max_samples in splits:
        info = generation_info["splits"].get(split_name)
        if info:
            print(f"  {split_name}: {info['subsampled']} samples "
                  f"(median gen length: {info['median_generation_chars']} chars, "
                  f"{info['empty_generations']} empty)")


if __name__ == "__main__":
    main()
