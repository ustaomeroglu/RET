#!/usr/bin/env python
"""
Cache LLM-generated responses to Alpaca prompts + hidden states for RET training.

Pipeline (per sample):
  1. Build messages = [{"role": "user", "content": instruction (+ "\n" + input)}]
  2. Apply chat template with add_generation_prompt=True
     (+ enable_thinking=True for Qwen 3.5 thinking mode)
  3. Generate response with model.generate()
  4. Run forward pass on (prompt + response) to extract hidden states at the
     target layer via a forward hook
  5. Append to the SYCON-Bench memmap cache (MemmapWriter) — the same format
     consumed by RET cached-mode training

Output layout (matches sycophancy cache format):
    <output_dir>/
        conversations.jsonl
        layer_<L>_activations_cache/
            metadata.json
            conv_index.jsonl
            hidden_states.bin
            token_ids.bin
            role_ids.bin        # 1 = user, 2 = assistant (incl. thinking)
            turn_ids.bin        # all 1 (single turn)
        split_indices.json      # {"train": [...], "val": [...], "test": []}
        sample_prompts.txt

Usage
-----
    python scripts/data_preparation_scripts/cache_alpaca_generations.py \\
        --model Qwen/Qwen2.5-14B-Instruct --layer 23 \\
        --output-dir data/alpaca/Qwen2.5-14B-Instruct \\
        --max-samples 60000 --max-new-tokens 2048 --batch-size 8

    # Qwen 3.5 with thinking mode on (default):
    python scripts/data_preparation_scripts/cache_alpaca_generations.py \\
        --model Qwen/Qwen3.5-35B-A3B --layer 19 \\
        --output-dir data/alpaca/Qwen3.5-35B-A3B \\
        --max-samples 60000 --max-new-tokens 4096 --batch-size 4
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Cache Alpaca generations + hidden states")
    p.add_argument("--model", required=True, help="HuggingFace model ID")
    p.add_argument("--layer", type=int, required=True,
                   help="Layer index to extract hidden states from (e.g. 23 for Qwen2.5-14B, 19 for Qwen3.5-35B-A3B)")
    p.add_argument("--output-dir", required=True,
                   help="Output directory (e.g. data/alpaca/Qwen2.5-14B-Instruct)")
    p.add_argument("--max-samples", type=int, default=60000,
                   help="Max number of Alpaca samples to process (dataset has ~52k)")
    p.add_argument("--max-new-tokens", type=int, default=2048,
                   help="Max tokens to generate per sample (use 4096+ for thinking-mode)")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--no-thinking", action="store_true",
                   help="Disable thinking mode (pass enable_thinking=False to chat template)")
    p.add_argument("--val-fraction", type=float, default=0.05,
                   help="Fraction of samples to put in the val split")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--start-idx", type=int, default=0,
                   help="Start index into the Alpaca dataset (for parallel/resumable runs)")
    p.add_argument("--end-idx", type=int, default=None,
                   help="End index (exclusive). Default = min(len(dataset), max_samples)")
    p.add_argument("--chunk-id", default=None,
                   help="Optional chunk identifier for parallel runs (see MemmapWriter).")
    p.add_argument("--flush-every", type=int, default=50,
                   help="Flush MemmapWriter metadata every N conversations")
    p.add_argument("--trust-remote-code", action="store_true", default=True)
    return p.parse_args()


def load_alpaca(max_samples: int, start_idx: int, end_idx: int | None) -> List[Dict]:
    """Load tatsu-lab/alpaca from HuggingFace, return list of {instruction, input, output} dicts."""
    from datasets import load_dataset

    logger.info("Loading tatsu-lab/alpaca from HF datasets...")
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    end = end_idx if end_idx is not None else min(len(ds), max_samples)
    ds = ds.select(range(start_idx, end))
    logger.info("Loaded %d Alpaca examples (indices %d..%d)", len(ds), start_idx, end)
    return list(ds)


def build_user_message(example: Dict) -> str:
    """Combine Alpaca's instruction + (optional) input into a single user message."""
    instruction = (example.get("instruction") or "").strip()
    extra_input = (example.get("input") or "").strip()
    if extra_input:
        return f"{instruction}\n\n{extra_input}"
    return instruction


@torch.no_grad()
def generate_batch(
    prompts: List[str],
    model,
    tokenizer,
    max_new_tokens: int,
) -> List[Tuple[str, int, int]]:
    """Generate responses for a batch of chat-templated prompts.

    Returns list of (response_text, prompt_token_count, response_token_count).
    """
    first_device = next(model.parameters()).device
    enc = tokenizer(
        prompts, return_tensors="pt", padding=True, truncation=False,
    )
    input_ids = enc["input_ids"].to(first_device)
    attention_mask = enc["attention_mask"].to(first_device)

    output_ids = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        do_sample=False,
    )

    results = []
    prompt_len = input_ids.shape[1]
    for i in range(len(prompts)):
        gen_ids = output_ids[i, prompt_len:]
        if tokenizer.pad_token_id is not None:
            gen_ids = gen_ids[gen_ids != tokenizer.pad_token_id]
        response = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        sample_prompt_len = int(attention_mask[i].sum().item())
        results.append((response, sample_prompt_len, int(len(gen_ids))))
    return results


@torch.no_grad()
def extract_hidden_states(
    full_text: str,
    model,
    tokenizer,
    layer_idx: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Forward pass; capture hidden states at `layer_idx`. Returns (hs [S,D], token_ids [S])."""
    from evaluation.probing.model_utils import _HiddenStateHook, _get_layer_from, _unwrap

    raw_model = _unwrap(model)
    enc = tokenizer(full_text, return_tensors="pt", truncation=False)
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    first_device = next(raw_model.parameters()).device
    input_ids = input_ids.to(first_device)
    attention_mask = attention_mask.to(first_device)

    layer_module = _get_layer_from(raw_model, layer_idx)
    hook = _HiddenStateHook()
    handle = layer_module.register_forward_hook(hook)
    try:
        raw_model(input_ids=input_ids, attention_mask=attention_mask)
    finally:
        handle.remove()

    hs = hook.output.float().squeeze(0).cpu().numpy()  # [S, D]
    tokens = input_ids.squeeze(0).cpu().numpy().astype(np.int32)  # [S]
    return hs, tokens


def compute_role_ids(
    prompt_user_only_len: int,
    total_len: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """For single-turn Alpaca: first user tokens are role=1 (including system header
    if any), the remainder (assistant scaffold + response, incl. thinking) is role=2.
    turn_ids are all 1.
    """
    role_ids = np.full(total_len, 2, dtype=np.int8)  # default assistant
    role_ids[:prompt_user_only_len] = 1
    turn_ids = np.ones(total_len, dtype=np.int8)
    return role_ids, turn_ids


def main():
    args = parse_args()

    # Imports deferred until after argparse (fast --help)
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
    from evaluation.datasets.sycophancy.storage import MemmapWriter

    os.makedirs(args.output_dir, exist_ok=True)

    # ----- Chat-template thinking flag -----
    chat_template_kwargs: Dict = {}
    if args.no_thinking:
        chat_template_kwargs["enable_thinking"] = False

    # ----- Load data -----
    alpaca = load_alpaca(args.max_samples, args.start_idx, args.end_idx)

    # ----- Model + tokenizer -----
    tok_kwargs = {"trust_remote_code": args.trust_remote_code}
    logger.info("Loading tokenizer: %s", args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model, **tok_kwargs)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    logger.info("Loading model: %s (dtype=%s, device_map=auto)", args.model, dtype)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, device_map="auto",
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()
    logger.info("Model devices: %s", getattr(model, "hf_device_map", "single"))

    # ----- Resolve hidden_dim -----
    model_config = AutoConfig.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    hidden_dim = getattr(model_config, "hidden_size", None)
    if hidden_dim is None:
        hidden_dim = model_config.text_config.hidden_size
    logger.info("Hidden dim = %d, target layer = %d", hidden_dim, args.layer)

    # ----- MemmapWriter -----
    cache_dir = os.path.join(args.output_dir, f"layer_{args.layer}_activations_cache")
    writer = MemmapWriter(
        cache_dir, args.model, args.layer, hidden_dim, chunk_id=args.chunk_id,
    )
    logger.info("MemmapWriter: %s (%d already cached)",
                cache_dir, writer.num_conversations)
    already_done = writer.completed_ids

    # ----- Conversations JSONL (append mode) -----
    conv_path = os.path.join(args.output_dir, "conversations.jsonl")

    # Track sample prompts from the first batch for verification
    sample_prompts: List[str] = []

    # ----- Build prompts -----
    remaining = []
    for i, ex in enumerate(alpaca):
        cid = f"alpaca_{args.start_idx + i:06d}"
        if cid in already_done:
            continue
        remaining.append((cid, ex))
    logger.info("Processing %d remaining examples (out of %d)", len(remaining), len(alpaca))

    # ----- Main loop: batch generate, then extract hidden states per-sample -----
    total_saved = 0
    for batch_start in range(0, len(remaining), args.batch_size):
        batch = remaining[batch_start:batch_start + args.batch_size]
        batch_cids = [c for c, _ in batch]
        batch_examples = [e for _, e in batch]

        # Build chat-templated prompts
        batch_prompts = []
        batch_user_only_lens = []
        for ex in batch_examples:
            user_msg = build_user_message(ex)
            messages = [{"role": "user", "content": user_msg}]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                **chat_template_kwargs,
            )
            user_only = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False,
                **chat_template_kwargs,
            )
            user_only_len = len(tokenizer.encode(user_only, add_special_tokens=False))
            batch_prompts.append(prompt)
            batch_user_only_lens.append(user_only_len)

        # Generate
        gen_results = generate_batch(
            batch_prompts, model, tokenizer, max_new_tokens=args.max_new_tokens,
        )

        # Extract hidden states + write (per-sample since full sequences have varying lengths)
        for (cid, ex), prompt, (response, prompt_len, resp_len), user_only_len in zip(
            batch, batch_prompts, gen_results, batch_user_only_lens,
        ):
            full_text = prompt + response
            try:
                hs, full_ids = extract_hidden_states(
                    full_text, model, tokenizer, args.layer,
                )
            except Exception as e:
                logger.error("Extraction failed for %s: %s", cid, e)
                continue

            S = hs.shape[0]
            token_ids = full_ids
            role_ids, turn_ids = compute_role_ids(user_only_len, S)

            writer.append_conversation(
                conv_id=cid,
                hidden_states=hs,
                token_ids=token_ids,
                role_ids=role_ids,
                turn_ids=turn_ids,
                num_turns=1,
            )

            # Append text JSONL
            record = {
                "id": cid,
                "scenario": "alpaca",
                "model_id": args.model,
                "metadata": {
                    "instruction": ex.get("instruction", ""),
                    "input": ex.get("input", ""),
                    "alpaca_output": ex.get("output", ""),  # reference only
                },
                "turns": [
                    {
                        "turn": 1,
                        "user_message": build_user_message(ex),
                        "assistant_response": response,
                        "prompt_token_count": prompt_len,
                        "response_token_count": resp_len,
                    }
                ],
            }
            with open(conv_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

            if len(sample_prompts) < 5:
                sample_prompts.append(full_text)

            total_saved += 1

        if (batch_start // args.batch_size) % max(1, args.flush_every // args.batch_size) == 0:
            writer.flush_metadata()
            logger.info(
                "  Batch %d/%d | saved %d convs | %d tokens total",
                batch_start // args.batch_size + 1,
                (len(remaining) + args.batch_size - 1) // args.batch_size,
                total_saved, writer.total_tokens,
            )

    writer.flush_metadata()
    logger.info("Done. Saved %d conversations, %d total tokens", total_saved, writer.total_tokens)

    # ----- Sample prompts -----
    if sample_prompts:
        sample_path = os.path.join(args.output_dir, "sample_prompts.txt")
        with open(sample_path, "w", encoding="utf-8") as f:
            for i, p in enumerate(sample_prompts):
                f.write("=" * 80 + "\n")
                f.write(f"SAMPLE {i+1}\n")
                f.write("=" * 80 + "\n")
                f.write(p)
                f.write("\n\n")
        logger.info("Saved %d sample prompts to %s", len(sample_prompts), sample_path)

    # ----- split_indices.json (only if we cached end-to-end, not chunked) -----
    if args.chunk_id is None:
        from evaluation.datasets.sycophancy.storage import MemmapReader
        reader = MemmapReader(cache_dir)
        n = len(reader)
        rng = np.random.default_rng(args.seed)
        perm = rng.permutation(n)
        n_val = max(1, int(n * args.val_fraction))
        val_idx = sorted(int(x) for x in perm[:n_val])
        train_idx = sorted(int(x) for x in perm[n_val:])
        split_path = os.path.join(args.output_dir, "split_indices.json")
        with open(split_path, "w") as f:
            json.dump({"train": train_idx, "val": val_idx, "test": []}, f)
        logger.info(
            "Wrote %s: train=%d, val=%d", split_path, len(train_idx), len(val_idx),
        )


if __name__ == "__main__":
    main()
