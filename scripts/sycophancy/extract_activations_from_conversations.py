#!/usr/bin/env python
"""
Extract hidden states from previously generated SYCON-Bench conversations.

Reads a `conversations.jsonl` produced by `run_benchmark.py`, replays each
conversation through one final-turn forward pass, and writes the resulting
hidden states + per-token boundary annotations into a `MemmapWriter` cache.
No new sampling is done — the saved `assistant_response` strings are used
verbatim, so the extracted activations are deterministic given the model.

Why this exists: when a parallel multi-job generation run corrupts the
shared activation cache via the `MemmapWriter` race (concurrent writers
sharing one binary file), the text `conversations.jsonl` is still intact
because JSONL appends are line-atomic. This script rebuilds the activation
cache from that text without re-running expensive Qwen sampling and without
invalidating any judge results.

Usage
-----
Single chunk:
    python scripts/sycophancy/extract_activations_from_conversations.py \
        --input data/sycophancy/debate/Qwen2.5-14B-Instruct/conversations.jsonl \
        --model Qwen/Qwen2.5-14B-Instruct \
        --layers 23

Parallel chunked (12 jobs, race-free via per-chunk subdirs):
    for i in 0 1 2 3 4 5 6 7 8 9 10 11; do
      start=$((i * 509)); end=$(((i + 1) * 509))
      sbatch scripts/cluster/psc/submit.sh --job-name=syc-extract-$i -- \
        python scripts/sycophancy/extract_activations_from_conversations.py \
          --input .../conversations.jsonl --model Qwen/Qwen2.5-14B-Instruct \
          --layers 23 \
          --start_idx $start --end_idx $end --chunk_id $i
    done
    # then merge:
    python -c "from evaluation.datasets.sycophancy.storage import merge_chunks; \
               merge_chunks('.../layer_23_activations_cache', delete_chunks=True)"
"""
import argparse
import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(
        description="Extract sycophancy hidden states from saved conversations.jsonl"
    )
    p.add_argument("--input", required=True,
                   help="Path to conversations.jsonl produced by run_benchmark.py")
    p.add_argument("--model", required=True, help="HuggingFace model ID")
    p.add_argument("--layers", type=int, nargs="+", default=[-1],
                   help="Layers to extract hidden states from (-1 = last)")
    p.add_argument("--no_thinking", action="store_true",
                   help="Pass enable_thinking=False to apply_chat_template "
                        "(must match the original generation run)")
    p.add_argument("--start_idx", type=int, default=0,
                   help="Slice the conversations list starting at this index")
    p.add_argument("--end_idx", type=int, default=None,
                   help="Slice end (exclusive). Default = all.")
    p.add_argument("--chunk_id", default=None,
                   help="Optional chunk identifier for parallel runs. When set, "
                        "writes to layer_X_activations_cache/chunk_{id}/ instead of "
                        "layer_X_activations_cache/.")
    p.add_argument("--output_dir", default=None,
                   help="Override output directory. By default the cache is written "
                        "next to the input conversations.jsonl.")
    return p.parse_args()


def _seed_from_record(record: dict):
    """Reconstruct a ConversationSeed from a saved conversations.jsonl record.

    The user_message stored in turns[0] is the raw initial_question
    (run_batched stores `seeds[i].initial_question` verbatim, see
    generation.py:380), and turns[t>=1].user_message holds the corresponding
    followup_messages[t-1]. So a faithful rebuild is straightforward.
    """
    from evaluation.datasets.sycophancy.data import ConversationSeed
    turns = record["turns"]
    return ConversationSeed(
        id=record["id"],
        scenario=record["scenario"],
        initial_question=turns[0]["user_message"],
        followup_messages=[t["user_message"] for t in turns[1:]],
        metadata=record.get("metadata", {}),
    )


def main():
    args = parse_args()

    # Imports deferred so --help is fast and doesn't load torch
    from evaluation.datasets.sycophancy.generation import (
        load_model_sharded, _extract_hidden_states, compute_token_boundaries,
        strip_thinking, free_model,
    )
    from evaluation.datasets.sycophancy.prompts import build_turn_messages, format_messages
    from evaluation.datasets.sycophancy.storage import (
        MemmapWriter, load_conversations, get_completed_ids,
    )

    # ----- Resolve output directory -----
    if args.output_dir is not None:
        output_dir = args.output_dir
    else:
        output_dir = os.path.dirname(os.path.abspath(args.input))

    # ----- Load conversations -----
    records = load_conversations(args.input)
    logger.info("Loaded %d conversations from %s", len(records), args.input)
    records = records[args.start_idx:args.end_idx]
    if not records:
        logger.warning("No records in slice [%s:%s]; nothing to do.",
                       args.start_idx, args.end_idx)
        return
    logger.info("Working on slice [%d:%s] -> %d records",
                args.start_idx, args.end_idx, len(records))

    # ----- Build chat_template_kwargs -----
    chat_template_kwargs = {}
    if args.no_thinking:
        chat_template_kwargs["enable_thinking"] = False

    # ----- Load model + tokenizer -----
    logger.info("Loading model: %s", args.model)
    tokenizer, model = load_model_sharded(args.model)

    # ----- Hidden dim (handles MoE configs that nest under text_config) -----
    from transformers import AutoConfig
    model_config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    hidden_dim = getattr(model_config, "hidden_size", None)
    if hidden_dim is None:
        hidden_dim = model_config.text_config.hidden_size
    logger.info("hidden_dim = %d", hidden_dim)

    # ----- Build writers (one per layer) -----
    writers = {}
    for layer in args.layers:
        layer_dir = os.path.join(output_dir, f"layer_{layer}_activations_cache")
        writers[layer] = MemmapWriter(
            layer_dir, args.model, layer, hidden_dim, chunk_id=args.chunk_id,
        )

    # ----- Resume support: skip already-extracted IDs in this chunk -----
    completed = get_completed_ids(output_dir, layers=args.layers, chunk_id=args.chunk_id)
    todo = [r for r in records if r["id"] not in completed]
    logger.info("Resume: %d records in slice, %d already done, %d remaining",
                len(records), len(records) - len(todo), len(todo))

    # ----- Compute generation suffix for this model + kwargs -----
    # When enable_thinking=False, Qwen3.5's chat template appends
    # '<think>\n\n</think>\n\n' after the assistant role header. The stored
    # assistant_response already includes this prefix (prepended during
    # generation via _get_generation_suffix). Since format_messages with
    # add_generation_prompt=True also includes it in the prompt, we must
    # strip it from the response before concatenating to avoid a double
    # <think> block that the model never saw.
    from evaluation.datasets.sycophancy.generation import _get_generation_suffix
    gen_suffix = _get_generation_suffix(tokenizer, chat_template_kwargs)
    if gen_suffix:
        logger.info("  Generation suffix detected: %r — will strip from responses", gen_suffix)

    # ----- Per-conversation extraction loop -----
    n_done = 0
    for record in todo:
        seed = _seed_from_record(record)
        turns = record["turns"]
        num_turns = len(turns)

        # Strip thinking from prior turns to mirror generation pipeline
        # (see generation.py:398)
        prior = [strip_thinking(t["assistant_response"]) for t in turns[:-1]]
        final_response = turns[-1]["assistant_response"]

        # Strip gen_suffix from response to avoid double-counting
        # (prompt already includes it via add_generation_prompt=True)
        response_for_extraction = final_response
        if gen_suffix and final_response.startswith(gen_suffix):
            response_for_extraction = final_response[len(gen_suffix):]

        # Build the same final-turn prompt the generation pipeline used
        msgs = build_turn_messages(
            seed, prior, num_turns, model_id=args.model,
        )
        final_prompt = format_messages(msgs, tokenizer, **chat_template_kwargs)
        full_text = final_prompt + response_for_extraction

        # Forward pass + hook-based hidden-state capture
        hs_dict, _ = _extract_hidden_states(full_text, model, tokenizer, args.layers)

        # Per-token role/turn annotations
        boundaries = compute_token_boundaries(
            seed, prior, num_turns, args.model, tokenizer,
            response_for_extraction, **chat_template_kwargs,
        )

        for layer_idx, hs in hs_dict.items():
            if layer_idx not in writers:
                continue
            writers[layer_idx].append_conversation(
                conv_id=record["id"],
                hidden_states=hs,
                token_ids=boundaries["token_ids"],
                role_ids=boundaries["role_ids"],
                turn_ids=boundaries["turn_ids"],
                num_turns=num_turns,
            )

        n_done += 1
        if n_done % 25 == 0 or n_done == len(todo):
            logger.info("  %d / %d extracted", n_done, len(todo))
            # Flush metadata periodically so a killed job leaves a usable
            # snapshot of progress.
            for w in writers.values():
                w.flush_metadata()

    # Final metadata flush
    for w in writers.values():
        w.flush_metadata()

    free_model(model)
    logger.info("Done. %d conversations extracted into chunk_id=%s",
                n_done, args.chunk_id)


if __name__ == "__main__":
    main()
