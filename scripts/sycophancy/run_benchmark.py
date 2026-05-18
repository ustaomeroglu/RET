#!/usr/bin/env python
"""
Run SYCON-Bench multi-turn generation with hidden-state extraction.

Conversations are batched by turn number for better GPU utilization.
All results cached under data/sycophancy/{scenario}/{model_slug}/.
Re-runs skip already-completed conversations.

Usage:
    # Smoke test (3 samples, debate only)
    python scripts/sycophancy/run_benchmark.py \
        --model Qwen/Qwen2.5-14B-Instruct --scenario debate \
        --layers -1 --max_samples 3

    # Full run (all 3 scenarios), batch_size 8
    python scripts/sycophancy/run_benchmark.py \
        --model Qwen/Qwen2.5-14B-Instruct --scenario all --layers -1 --batch_size 8

    # Skip hidden-state extraction (faster, text only)
    python scripts/sycophancy/run_benchmark.py \
        --model Qwen/Qwen2.5-14B-Instruct --scenario debate --no_hidden_states
"""
import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="SYCON-Bench generation")
    p.add_argument("--model", required=True, help="HuggingFace model ID")
    p.add_argument("--scenario", default="all",
                   choices=["debate", "false_presupposition", "all"])
    p.add_argument("--layers", type=int, nargs="+", required=True,
                   help="Layers to extract hidden states from. Required. "
                        "Pass explicit positive integers (e.g. --layers 23). "
                        "The shorthand --layers -1 is still accepted and "
                        "resolves to the last layer at hook-registration time, "
                        "but is discouraged because the on-disk cache directory "
                        "will literally be named layer_-1_activations_cache/.")
    p.add_argument("--max_new_tokens", type=int, default=4096,
                   help="Generation budget per turn. Use 4096+ for thinking-mode models "
                        "(Qwen3.5 traces can easily exceed 2048 tokens)")
    p.add_argument("--num_turns", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=4,
                   help="Number of conversations to batch per turn")
    p.add_argument("--max_samples", type=int, default=None,
                   help="Limit samples per scenario (for smoke tests)")
    p.add_argument("--start_idx", type=int, default=0,
                   help="Start index for parallel runs (inclusive)")
    p.add_argument("--end_idx", type=int, default=None,
                   help="End index for parallel runs (exclusive, default=all)")
    p.add_argument("--data_dir", default="data/sycophancy/debate",
                   help="Path to folder containing questions.txt and arguments.txt")
    p.add_argument("--no_hidden_states", action="store_true",
                   help="Skip hidden-state extraction (text-only mode)")
    p.add_argument("--no_thinking", action="store_true",
                   help="Disable thinking mode for models like Qwen3.5 "
                        "(passes enable_thinking=False to apply_chat_template)")
    p.add_argument("--chunk_id", default=None,
                   help="Optional chunk identifier for parallel runs. When set, "
                        "MemmapWriter writes to layer_X_activations_cache/chunk_{id}/ "
                        "instead of layer_X_activations_cache/ — necessary for "
                        "race-free parallel jobs. Run merge_chunks() afterward.")
    return p.parse_args()


def main():
    args = parse_args()

    import os
    from evaluation.datasets.sycophancy.data import load_seeds, SCENARIOS
    from evaluation.datasets.sycophancy.generation import (
        load_model_sharded, run_batched, free_model, strip_thinking,
        _get_generation_suffix,
    )
    from evaluation.datasets.sycophancy.prompts import build_turn_messages, format_messages
    from evaluation.datasets.sycophancy.storage import (
        get_output_dir, save_conversation, get_completed_ids, save_sample_prompts,
        MemmapWriter,
    )

    scenarios = SCENARIOS if args.scenario == "all" else [args.scenario]

    chat_template_kwargs = {}
    if args.no_thinking:
        chat_template_kwargs["enable_thinking"] = False

    logger.info("Loading model: %s", args.model)
    tokenizer, model = load_model_sharded(args.model)

    # Detect generation suffix (e.g. '<think>\n\n</think>\n\n' for Qwen3.5
    # no_thinking). Used to strip duplicate prefix when building extraction
    # text and sample_prompts.txt.
    gen_suffix = _get_generation_suffix(tokenizer, chat_template_kwargs)

    for scenario in scenarios:
        logger.info("=== Scenario: %s ===", scenario)

        # Debate always uses the assigned-stance system prompt now;
        # no mode suffix on the output directory.
        dir_mode = "no_thinking" if args.no_thinking else ""

        seeds = load_seeds(scenario, args.data_dir, args.max_samples)
        seeds = seeds[args.start_idx:args.end_idx]
        if not seeds:
            logger.warning("No seeds loaded for %s, skipping", scenario)
            continue

        output_dir = get_output_dir(args.data_dir, args.model, mode=dir_mode)
        completed = get_completed_ids(output_dir, layers=args.layers, chunk_id=args.chunk_id)
        remaining = [s for s in seeds if s.id not in completed]
        logger.info(
            "  %d seeds total, %d already done, %d remaining",
            len(seeds), len(completed), len(remaining),
        )
        if not remaining:
            logger.info("  All done, skipping.")
            continue

        # Create MemmapWriters for hidden-state layers
        writers = {}
        if not args.no_hidden_states:
            from transformers import AutoConfig
            model_config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
            # MoE configs (e.g. Qwen3.5 MoE) nest hidden_size under text_config
            hidden_dim = getattr(model_config, "hidden_size", None)
            if hidden_dim is None:
                hidden_dim = model_config.text_config.hidden_size
            for layer in args.layers:
                layer_dir = os.path.join(output_dir, f"layer_{layer}_activations_cache")
                writers[layer] = MemmapWriter(
                    layer_dir, args.model, layer, hidden_dim, chunk_id=args.chunk_id,
                )

        # Process seeds in chunks to bound memory: run_batched accumulates all
        # hidden states for its input list before returning, so processing all
        # 6,914 seeds at once OOMs (~211 GB for layer 23 of Qwen 2.5 14B).
        # Chunking matches the working debate run (which had ~100 samples).
        CHUNK_SIZE = 100
        sample_results = []  # only first 5 from first chunk for sample_prompts
        total_saved = 0
        num_chunks = (len(remaining) + CHUNK_SIZE - 1) // CHUNK_SIZE

        for chunk_start in range(0, len(remaining), CHUNK_SIZE):
            chunk = remaining[chunk_start:chunk_start + CHUNK_SIZE]
            logger.info(
                "  Processing chunk %d/%d (%d conversations)",
                chunk_start // CHUNK_SIZE + 1, num_chunks, len(chunk),
            )

            chunk_results = run_batched(
                chunk, model, tokenizer,
                layers=args.layers,
                model_id=args.model,
                max_new_tokens=args.max_new_tokens,
                num_turns=args.num_turns,
                batch_size=args.batch_size,
                extract_hidden_states=not args.no_hidden_states,
                chat_template_kwargs=chat_template_kwargs,
            )

            for conv in chunk_results:
                save_conversation(conv, output_dir, writers=writers or None)
            total_saved += len(chunk_results)

            # Flush metadata after every inner chunk so a cancelled job
            # leaves a merge-able cache (merge_chunks requires metadata.json
            # whose total_tokens matches the binary file sizes).
            for w in writers.values():
                w.flush_metadata()

            # Keep first 5 from the very first chunk for sample_prompts
            if not sample_results:
                sample_results = chunk_results[:5]

            # Drop the rest immediately so the GC can free hidden_states arrays
            del chunk_results

        # Final flush after all chunks (no-op if loop already flushed)
        for w in writers.values():
            w.flush_metadata()

        # Save sample prompts: show the EXACT string fed into the final-turn
        # forward pass for hidden-state extraction. That string is
        # final_prompt + final_response, where final_prompt is built from
        # thinking-stripped prior responses (matching the generation pipeline,
        # see generation.py:411) and final_response is the actual turn-N reply
        # produced by the model (with any <think>...</think> blocks intact).
        sample_prompts = []
        for conv in sample_results:
            final_turn = len(conv.turns)
            prior = [strip_thinking(t.assistant_response) for t in conv.turns[:-1]]
            msgs = build_turn_messages(
                conv.seed, prior, final_turn, model_id=args.model,
            )
            prompt_text = format_messages(msgs, tokenizer, **chat_template_kwargs)
            final_response = conv.turns[-1].assistant_response
            # Strip gen_suffix to avoid double <think> block (same as extraction)
            resp_for_display = final_response
            if gen_suffix and final_response.startswith(gen_suffix):
                resp_for_display = final_response[len(gen_suffix):]
            sample_prompts.append(prompt_text + resp_for_display)
        save_sample_prompts(sample_prompts, output_dir)

        logger.info("  Done. %d conversations saved to %s", total_saved, output_dir)

    free_model(model)
    logger.info("All scenarios complete.")


if __name__ == "__main__":
    main()
