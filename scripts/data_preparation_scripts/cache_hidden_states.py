#!/usr/bin/env python3
"""Extract and cache LLM hidden states to disk as numpy memmap files.

Two modes:

1. **Config mode** (``--config``): existing behaviour — reads YAML config,
   loads local Arrow datasets by split, extracts hidden states, and writes
   per-split memmap files.  All splits in ``--splits`` are processed.

2. **Dataset mode** (``--dataset``): loads a named probing/evaluation dataset
   by name (e.g. ``models_under_pressure``), formats prompts, extracts hidden
   states one prompt at a time, and writes a single flat memmap together with
   labels and metadata.

Storage layout (both modes)::

    <output_dir>/
      <split>/              ← config mode only (train/val/test)
        metadata.json        # model info, layer, hidden_dim, dtype, num_docs, total_tokens
        offsets.npy           # (num_docs,) int64 — start offset per doc in the flat array
        lengths.npy           # (num_docs,) int32 — sequence length per doc
        hidden_states.bin     # memmap (total_tokens, hidden_dim) float16/32
        input_ids.bin         # memmap (total_tokens,) int32

    <output_dir>/             ← dataset mode (flat, no split sub-dir)
        metadata.json
        offsets.npy
        lengths.npy
        hidden_states.bin
        input_ids.bin
        labels.npy / labels.npz   ← only if labels are available

Loading cached hidden states::

    import numpy as np, json
    meta = json.load(open("<dir>/metadata.json"))
    offsets = np.load("<dir>/offsets.npy")
    lengths = np.load("<dir>/lengths.npy")
    h = np.memmap("<dir>/hidden_states.bin", dtype=meta["dtype"], mode="r",
                  shape=(meta["total_tokens"], meta["hidden_dim"]))
    # Load document i:
    doc_h = h[offsets[i] : offsets[i] + lengths[i]]  # (lengths[i], hidden_dim)

No attention_mask is saved — all stored tokens are valid (padding is stripped
before writing). Reconstruct the mask from ``lengths`` during collation.

Note: token embeddings are NOT cached. Controlled-mode training and closure
predictors that need ``get_token_embeddings()`` can compute them cheaply from
the cached ``input_ids`` by loading only the LLM's embedding layer::

    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(meta["model_name"], ...)
    embed = model.get_input_embeddings()  # just the embedding table
    token_emb = embed(torch.from_numpy(doc_ids).long())  # (seq_len, hidden_dim)

When loading ``input_ids.bin``, cast from int32 to int64 before passing to
PyTorch models: ``input_ids = torch.from_numpy(doc_ids).long()``

Usage::

    # Config mode — full extraction
    conda run -n eet python scripts/data_preparation_scripts/cache_hidden_states.py \\
        --config configs/GptOss20B/numina_math_gen.yaml \\
        --splits train val test \\
        --batch-size 12

    # Config mode — quick test
    conda run -n eet python scripts/data_preparation_scripts/cache_hidden_states.py \\
        --config configs/GptOss20B/numina_math_gen.yaml \\
        --splits val --max-docs 20 --batch-size 4

    # Dataset mode — models_under_pressure
    conda run -n eet python scripts/data_preparation_scripts/cache_hidden_states.py \\
        --dataset models_under_pressure --model openai/gpt-oss-20b --layer 11
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract and cache LLM hidden states to disk.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ---- Mode selection (mutually exclusive) ----
    mode_group = parser.add_argument_group("Mode (provide exactly one)")
    mode_group.add_argument(
        "--config", type=str, default=None,
        help="YAML config path (specifies LLM and data settings) — config mode",
    )
    mode_group.add_argument(
        "--dataset", type=str, default=None,
        choices=["models_under_pressure",
                 "numina_math", "numina_math_64k_samples", "mmlu"],
        help="Dataset to load by name (alternative to --config) — dataset mode",
    )

    # ---- Dataset-mode arguments ----
    ds_group = parser.add_argument_group("Dataset mode options")
    ds_group.add_argument(
        "--model", type=str, default="openai/gpt-oss-20b",
        help="HuggingFace model ID (used with --dataset, default: openai/gpt-oss-20b)",
    )
    ds_group.add_argument(
        "--layer", type=int, default=12,
        help="Layer index for hidden state extraction (used with --dataset, default: 12)",
    )
    ds_group.add_argument(
        "--generate", action="store_true",
        help="Generate model completions before extraction. The model generates a response "
             "to each prompt, then hidden states are extracted from the full [prompt+response] sequence.",
    )
    ds_group.add_argument(
        "--max-new-tokens", type=int, default=512,
        help="Max tokens to generate per prompt (used with --generate, default: 512)",
    )
    ds_group.add_argument(
        "--temperature", type=float, default=0.0,
        help="Sampling temperature for generation (0.0 = greedy, used with --generate)",
    )
    ds_group.add_argument(
        "--strip-gpt-info", action="store_true",
        help="Strip GPT-OSS harmony system prompt, keeping only developer prompt. "
             "By default the full system prompt is preserved (in-distribution for GPT-OSS).",
    )
    ds_group.add_argument(
        "--max-samples", type=int, default=None,
        help="Max samples (used with --dataset)",
    )

    # ---- Config-mode arguments ----
    cfg_group = parser.add_argument_group("Config mode options")
    cfg_group.add_argument(
        "--splits", type=str, nargs="+", default=["train", "val", "test"],
        help="Dataset splits to extract (config mode, default: train val test)",
    )
    cfg_group.add_argument(
        "--batch-size", type=int, default=8,
        help="Document batch size for LLM forward pass (config mode, default: 8)",
    )
    cfg_group.add_argument(
        "--max-docs", type=int, default=None,
        help="Max documents per split for quick tests (config mode)",
    )

    # ---- Shared arguments ----
    shared_group = parser.add_argument_group("Shared options")
    shared_group.add_argument(
        "--output-dir", type=str, default=None,
        help=(
            "Root output directory. "
            "Config mode default: <local_dir>/cached_hidden_states/layer<N>. "
            "Dataset mode default: data/<dataset>/<model_short>/layer<N>"
        ),
    )
    shared_group.add_argument(
        "--dtype", type=str, default="float16", choices=["float16", "float32"],
        help="Dtype for saved hidden states (default: float16)",
    )
    shared_group.add_argument(
        "--device", type=str, default="cuda",
        help="Torch device (default: cuda)",
    )
    shared_group.add_argument(
        "--no-split", action="store_true",
        help="Skip automatic train/val/test splitting (default: split 90/5/5)",
    )

    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Dataset-mode helpers
# ---------------------------------------------------------------------------

def _load_dataset(name: str, tokenizer, args) -> tuple:
    """Load a named probing dataset and return (prompts, labels).

    Parameters
    ----------
    name      : dataset name key
    tokenizer : HuggingFace tokenizer (used for prompt formatting)
    args      : parsed argparse namespace

    Returns
    -------
    prompts : list[str]
    labels  : np.ndarray | dict[str, np.ndarray] | None
    """
    if name == "models_under_pressure":
        from evaluation.datasets.pressure.data import load_pressure_dataset
        prompts, labels = load_pressure_dataset(
            max_samples=args.max_samples,
            tokenizer=tokenizer,
            include_gpt_info=not args.strip_gpt_info,
        )
        return prompts, labels

    elif name in ("numina_math", "numina_math_64k_samples"):
        # Delegate to the corpus adapter (single source of truth; replaces
        # the inline `if name==` load_from_disk switch arm). Faithful:
        # data/<name>/train ds["text"], capped at max_samples, raw.
        from evaluation.datasets.numina_math.dataset import NuminaMathCorpus

        corpus = NuminaMathCorpus(variant=name)
        logger.info("Loading %s via NuminaMathCorpus", name)
        texts = list(corpus.iter_texts("train", max_texts=args.max_samples))
        return texts, None

    elif name == "mmlu":
        raise NotImplementedError(
            "mmlu dataset mode is not yet implemented. "
            "Use the probing experiment runner for MMLU."
        )

    else:
        raise ValueError(f"Unknown dataset: {name}")


def _run_dataset_mode(args) -> None:
    """Dataset mode: load named dataset, extract hidden states, save memmap."""

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU.")
        device = "cpu"

    # ------------------------------------------------------------------
    # Shared setup: LLMConfig + output_dir (needed by both branches)
    # ------------------------------------------------------------------
    from ret.config import LLMConfig, llm_short_name

    # Auto-detect dtype: use bfloat16 if supported, else float16
    model_dtype = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
    llm_config = LLMConfig(name=args.model, layer_index=args.layer, dtype=model_dtype)

    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
    else:
        model_short = llm_short_name(llm_config)
        # Nested layout: data/{dataset}/{model_short}/layer{N}/
        # Whether generation was done is recorded in metadata.json (key "generation").
        output_dir = Path(f"data/{args.dataset}/{model_short}/layer{args.layer}")
    output_dir.mkdir(parents=True, exist_ok=True)
    # Model-level dir: layer-independent artifacts (generations.jsonl,
    # prompts.jsonl, info.txt) go here so multiple layers share them.
    model_dir = output_dir.parent
    model_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: %s  (model dir: %s)", output_dir, model_dir)

    # ------------------------------------------------------------------
    # Generate path: vLLM → HiddenStateExtractor
    # Non-generate path: HiddenStateExtractor only (existing flow)
    # ------------------------------------------------------------------
    if args.generate:
        # ------------------------------------------------------------------
        # 1. Load tokenizer ONLY (no model weights yet)
        # ------------------------------------------------------------------
        from transformers import AutoTokenizer
        logger.info("Loading tokenizer for %s ...", args.model)
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # ------------------------------------------------------------------
        # 2. Load dataset + build prompts
        # ------------------------------------------------------------------
        logger.info("Loading dataset: %s", args.dataset)
        prompts, labels = _load_dataset(args.dataset, tokenizer, args)
        logger.info("Loaded %d prompts", len(prompts))

        # Track original prompt lengths BEFORE generation (used for prompts.jsonl)
        original_prompt_lengths = [
            len(tokenizer(p)["input_ids"]) for p in prompts
        ]

        # Check if already extracted
        if (output_dir / "metadata.json").exists():
            logger.info(
                "Already extracted (metadata.json exists). Skipping. "
                "Delete the directory to re-extract."
            )
            return

        # ------------------------------------------------------------------
        # Phase A: vLLM generation (with resume support)
        # ------------------------------------------------------------------
        gen_path = model_dir / "generations.jsonl"

        if gen_path.exists():
            logger.info(
                "Found existing generations.jsonl — loading saved generations (skipping vLLM)"
            )
            records = []
            with open(gen_path) as f:
                for line in f:
                    records.append(json.loads(line.strip()))
            # Use saved token IDs if available (preserves special tokens exactly)
            if "prompt_token_ids" in records[0] and "gen_token_ids" in records[0]:
                full_texts = None  # signal to use token IDs directly
                all_full_ids = [r["prompt_token_ids"] + r["gen_token_ids"] for r in records]
                logger.info("Loaded %d generations with token IDs from disk", len(records))
            else:
                full_texts = [r["full_text"] for r in records]
                all_full_ids = None
                logger.info("Loaded %d generations (text only, no token IDs) from disk", len(records))
        else:
            logger.info(
                "Generating responses with vLLM (max_new_tokens=%d, temperature=%.1f)...",
                args.max_new_tokens, args.temperature,
            )

            from vllm import LLM, SamplingParams

            tp_size = torch.cuda.device_count() or 1
            logger.info("Loading vLLM (tensor_parallel_size=%d)...", tp_size)
            vllm_model = LLM(
                model=args.model,
                tensor_parallel_size=tp_size,
                trust_remote_code=True,
            )

            sampling_params = SamplingParams(
                max_tokens=args.max_new_tokens,
                temperature=args.temperature if args.temperature > 0 else 0.0,
            )

            # Batch generate all prompts at once
            outputs = vllm_model.generate(prompts, sampling_params)

            # Use token IDs (not .text) to preserve special tokens like <|channel|>, <|message|>
            # vLLM's .text strips special tokens, but .token_ids keeps them
            all_gen_token_ids = []
            for o in outputs:
                gen_ids = list(o.outputs[0].token_ids)
                prompt_ids = list(o.prompt_token_ids)
                all_gen_token_ids.append((prompt_ids, gen_ids))

            # Save generations.jsonl with token IDs for exact reproducibility
            full_texts = []
            all_full_ids = []
            with open(gen_path, "w") as f:
                for i, (prompt_ids, gen_ids) in enumerate(all_gen_token_ids):
                    full_ids = prompt_ids + gen_ids
                    # Decode with special tokens visible for inspection
                    full_text = tokenizer.decode(full_ids, skip_special_tokens=False)
                    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=False)
                    full_texts.append(full_text)
                    all_full_ids.append(full_ids)
                    record = {
                        "idx": i,
                        "prompt": prompts[i],
                        "generation": gen_text,
                        "full_text": full_text,
                        "prompt_token_ids": prompt_ids,
                        "gen_token_ids": gen_ids,
                        "prompt_len": len(prompt_ids),
                        "gen_len": len(gen_ids),
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            logger.info("Saved %d generations to %s", len(all_gen_token_ids), gen_path)

            # Free vLLM before loading HiddenStateExtractor
            del vllm_model, outputs
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("vLLM freed from GPU")

        # ------------------------------------------------------------------
        # 3. Build token sequences for extraction
        # ------------------------------------------------------------------
        if all_full_ids is not None:
            # Use exact token IDs from vLLM (preserves special tokens)
            import torch as _torch
            all_encoded = [{"input_ids": _torch.tensor([ids])} for ids in all_full_ids]
            logger.info("Using %d token ID sequences from vLLM output", len(all_encoded))
        else:
            # Fallback: re-tokenize from text (may lose special tokens in generation)
            logger.info("Tokenizing %d full texts...", len(full_texts))
            all_encoded = [tokenizer(t, return_tensors="pt") for t in full_texts]

        # ------------------------------------------------------------------
        # 4. Load HiddenStateExtractor (now that vLLM is freed)
        # ------------------------------------------------------------------
        from ret.extraction.hooks import HiddenStateExtractor
        logger.info("Building HiddenStateExtractor for %s layer %d ...",
                    args.model, args.layer)
        extractor = HiddenStateExtractor(llm_config, device=device)
        hidden_dim = extractor.model.config.hidden_size

    else:
        # ------------------------------------------------------------------
        # Non-generate path: existing flow unchanged
        # ------------------------------------------------------------------
        from ret.extraction.hooks import HiddenStateExtractor

        logger.info("Building HiddenStateExtractor for %s layer %d ...",
                    args.model, args.layer)
        extractor = HiddenStateExtractor(llm_config, device=device)
        tokenizer = extractor.tokenizer
        # Get actual hidden_dim from the loaded model
        hidden_dim = extractor.model.config.hidden_size

        # ------------------------------------------------------------------
        # 2. Load dataset
        # ------------------------------------------------------------------
        logger.info("Loading dataset: %s", args.dataset)
        prompts, labels = _load_dataset(args.dataset, tokenizer, args)
        logger.info("Loaded %d prompts", len(prompts))

        # Check if already extracted
        if (output_dir / "metadata.json").exists():
            logger.info(
                "Already extracted (metadata.json exists). Skipping. "
                "Delete the directory to re-extract."
            )
            extractor.cleanup()
            return

        # ------------------------------------------------------------------
        # 3. Tokenize all prompts and count tokens
        # ------------------------------------------------------------------
        logger.info("Tokenizing %d prompts...", len(prompts))
        all_encoded = [tokenizer(p, return_tensors="pt") for p in prompts]
        # Track original prompt lengths (used for prompts.jsonl)
        original_prompt_lengths = [enc["input_ids"].shape[1] for enc in all_encoded]

    # ------------------------------------------------------------------
    # Common: log token count (both paths meet here)
    # ------------------------------------------------------------------
    total_tokens = sum(enc["input_ids"].shape[1] for enc in all_encoded)
    logger.info(
        "Total tokens: %d  (%.2f GB in %s)",
        total_tokens,
        total_tokens * hidden_dim * (2 if args.dtype == "float16" else 4) / 1e9,
        args.dtype,
    )

    # ------------------------------------------------------------------
    # 6. Allocate memmap files
    # ------------------------------------------------------------------
    save_dtype = np.float16 if args.dtype == "float16" else np.float32
    h_path = output_dir / "hidden_states.bin"
    ids_path = output_dir / "input_ids.bin"

    h_mmap = np.memmap(
        h_path, dtype=save_dtype, mode="w+",
        shape=(total_tokens, hidden_dim),
    )
    ids_mmap = np.memmap(
        ids_path, dtype=np.int32, mode="w+",
        shape=(total_tokens,),
    )

    offsets: list[int] = []
    lengths: list[int] = []
    offset = 0

    # ------------------------------------------------------------------
    # 7. Extract hidden states per prompt
    # ------------------------------------------------------------------
    logger.info(
        "Extracting hidden states from %d prompts (%d total tokens)...",
        len(prompts), total_tokens,
    )
    t0 = time.time()

    for i, enc in enumerate(all_encoded):
        input_ids = enc["input_ids"].to(extractor.input_device)
        with torch.no_grad():
            h = extractor.extract(input_ids)  # (1, T, D)

        seq_len = input_ids.shape[1]
        h_np = h[0, :seq_len].cpu().numpy().astype(save_dtype)
        ids_np = input_ids[0, :seq_len].cpu().numpy().astype(np.int32)

        h_mmap[offset: offset + seq_len] = h_np
        ids_mmap[offset: offset + seq_len] = ids_np
        offsets.append(offset)
        lengths.append(seq_len)
        offset += seq_len

        if (i + 1) % 100 == 0 or i == len(all_encoded) - 1:
            elapsed = time.time() - t0
            rate = (i + 1) / (elapsed / 60) if elapsed > 0 else 0
            logger.info(
                "  %d / %d prompts extracted  (%.1f prompts/min)",
                i + 1, len(all_encoded), rate,
            )

        # Flush periodically to avoid data loss on crash
        if (i + 1) % 500 == 0:
            h_mmap.flush()
            ids_mmap.flush()

    h_mmap.flush()
    ids_mmap.flush()
    del h_mmap, ids_mmap
    gc.collect()

    # ------------------------------------------------------------------
    # 8. Save index arrays
    # ------------------------------------------------------------------
    np.save(output_dir / "offsets.npy", np.array(offsets, dtype=np.int64))
    np.save(output_dir / "lengths.npy", np.array(lengths, dtype=np.int32))
    np.save(
        output_dir / "prompt_lengths.npy",
        np.array(original_prompt_lengths, dtype=np.int32),
    )
    logger.info("Saved prompt_lengths.npy (%d entries)", len(original_prompt_lengths))

    # ------------------------------------------------------------------
    # 9. Save labels if present
    # ------------------------------------------------------------------
    if labels is not None:
        if isinstance(labels, dict):
            np.savez(
                output_dir / "labels.npz",
                **{k: v for k, v in labels.items()},
            )
            logger.info(
                "Saved labels.npz with keys: %s", list(labels.keys())
            )
        else:
            np.save(output_dir / "labels.npy", labels)
            logger.info("Saved labels.npy (%d entries)", len(labels))

    # ------------------------------------------------------------------
    # 10. Compute system_prefix_len
    # ------------------------------------------------------------------
    sys_prefix_len = 0
    if getattr(tokenizer, "chat_template", None):
        try:
            from evaluation.probing.model_utils import compute_system_prefix_len
            sys_prefix_len = compute_system_prefix_len(
                tokenizer,
                "You are a helpful assistant.",
                include_gpt_info=not args.strip_gpt_info,
            )
            logger.info("system_prefix_len = %d", sys_prefix_len)
        except Exception as exc:
            logger.warning(
                "Could not compute system_prefix_len: %s", exc
            )

    # ------------------------------------------------------------------
    # 11. Save metadata
    # ------------------------------------------------------------------
    metadata = {
        "model_name": args.model,
        "layer_index": args.layer,
        "hidden_dim": hidden_dim,
        "dtype": args.dtype,
        "num_docs": len(prompts),
        "total_tokens": int(offset),
        "dataset": args.dataset,
        "system_prefix_len": sys_prefix_len,
        "include_gpt_info": not args.strip_gpt_info,
        "generation": args.generate,
    }
    if args.generate:
        metadata["max_new_tokens"] = args.max_new_tokens
        metadata["temperature"] = args.temperature
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # ------------------------------------------------------------------
    # 11b. Save prompt_examples.jsonl (first 5 samples with FULL token IDs
    #      + decoded text including special tokens — for visual verification
    #      that the LLM saw exactly what we think it saw).
    # ------------------------------------------------------------------
    import json as _json

    examples_path = model_dir / "prompt_examples.jsonl"
    with open(examples_path, "w") as f:
        for i in range(min(5, len(all_encoded))):
            ids = all_encoded[i]["input_ids"]
            if hasattr(ids, "dim") and ids.dim() > 1:
                ids = ids[0]
            ids_list = ids.tolist() if hasattr(ids, "tolist") else list(ids)
            text = tokenizer.decode(ids_list, skip_special_tokens=False)
            prompt_len = original_prompt_lengths[i]
            total_len = len(ids_list)
            gen_len = total_len - prompt_len if args.generate else 0
            record = {
                "idx": i,
                "prompt_length": prompt_len,
                "gen_length": gen_len,
                "total_length": total_len,
                "full_text_with_specials": text,
                "first_30_token_ids": ids_list[:30],
                "last_10_token_ids": ids_list[-10:],
            }
            f.write(_json.dumps(record, ensure_ascii=False) + "\n")
    logger.info("Saved prompt_examples.jsonl (5 samples for visual verification)")

    # ------------------------------------------------------------------
    # 11c. Save generation_info.json (structured JSON, replaces info.txt).
    # ------------------------------------------------------------------
    import datetime

    gen_info = {
        "dataset": args.dataset,
        "model": args.model,
        "layer": args.layer,
        "hidden_dim": hidden_dim,
        "dtype": args.dtype,
        "generation": bool(args.generate),
        "system_message": "You are a helpful assistant.",
        "include_gpt_info": bool(not args.strip_gpt_info),
        "num_samples": len(prompts),
        "total_tokens": int(offset),
        "cache_size_gb": float(offset * hidden_dim * np.dtype(save_dtype).itemsize / 1e9),
        "date": datetime.date.today().isoformat(),
    }
    if args.generate:
        gen_info["max_new_tokens"] = args.max_new_tokens
        gen_info["temperature"] = args.temperature

    # Label distribution
    if labels is not None and isinstance(labels, dict):
        dist = {}
        for k, v in labels.items():
            u, c = np.unique(v, return_counts=True)
            dist[k] = {str(int(uv)): int(cv) for uv, cv in zip(u, c)}
        gen_info["label_distribution"] = dist

    with open(model_dir / "generation_info.json", "w") as f:
        json.dump(gen_info, f, indent=2)
    logger.info("Saved generation_info.json")

    # ------------------------------------------------------------------
    # 12. Train/val/test split → per-split subdirectories
    # ------------------------------------------------------------------
    if not args.no_split:
        n = len(prompts)
        indices = np.random.default_rng(42).permutation(n)
        n_test = max(1, int(n * 0.05))
        n_val = max(1, int(n * 0.05))
        n_train = n - n_val - n_test

        split_index_map = {
            "train": indices[:n_train],
            "val": indices[n_train:n_train + n_val],
            "test": indices[n_train + n_val:],
        }

        # Load the full memmap arrays we just wrote
        all_offsets = np.load(output_dir / "offsets.npy")
        all_lengths = np.load(output_dir / "lengths.npy")
        all_prompt_lengths = np.load(output_dir / "prompt_lengths.npy")
        h_full = np.memmap(
            output_dir / "hidden_states.bin", dtype=save_dtype, mode="r",
            shape=(int(offset), hidden_dim),
        )
        ids_full = np.memmap(
            output_dir / "input_ids.bin", dtype="int32", mode="r",
            shape=(int(offset),),
        )

        for split_name, split_idx in split_index_map.items():
            split_dir = output_dir / split_name
            split_dir.mkdir(parents=True, exist_ok=True)

            s_offsets = all_offsets[split_idx]
            s_lengths = all_lengths[split_idx]
            s_prompt_lengths = all_prompt_lengths[split_idx]
            s_num_docs = len(split_idx)
            s_total_tokens = int(s_lengths.sum())

            # Write contiguous memmap for this split (re-pack with new offsets)
            s_h = np.memmap(
                split_dir / "hidden_states.bin", dtype=save_dtype, mode="w+",
                shape=(s_total_tokens, hidden_dim),
            )
            s_ids = np.memmap(
                split_dir / "input_ids.bin", dtype="int32", mode="w+",
                shape=(s_total_tokens,),
            )
            new_offsets = np.zeros(s_num_docs, dtype=np.int64)
            write_pos = 0
            for j in range(s_num_docs):
                src_start = int(s_offsets[j])
                src_len = int(s_lengths[j])
                new_offsets[j] = write_pos
                s_h[write_pos:write_pos + src_len] = h_full[src_start:src_start + src_len]
                s_ids[write_pos:write_pos + src_len] = ids_full[src_start:src_start + src_len]
                write_pos += src_len
            s_h.flush()
            s_ids.flush()
            del s_h, s_ids

            np.save(split_dir / "offsets.npy", new_offsets)
            np.save(split_dir / "lengths.npy", s_lengths.astype(np.int32))
            np.save(split_dir / "prompt_lengths.npy", s_prompt_lengths.astype(np.int32))

            # Labels
            if labels is not None:
                if isinstance(labels, dict):
                    np.savez(split_dir / "labels.npz",
                             **{k: v[split_idx] for k, v in labels.items()})
                else:
                    np.save(split_dir / "labels.npy", labels[split_idx])

            # Metadata
            split_meta = dict(metadata)
            split_meta["split"] = split_name
            split_meta["num_docs"] = s_num_docs
            split_meta["total_tokens"] = s_total_tokens
            with open(split_dir / "metadata.json", "w") as f:
                json.dump(split_meta, f, indent=2)

            logger.info("Saved %s split: %d docs, %d tokens → %s",
                        split_name, s_num_docs, s_total_tokens, split_dir)

            # When generation was performed, also write a per-split Arrow
            # dataset at model_dir/<split>/ (parallels numina_math_64k_samples's
            # GptOss20B/<split>/ Arrow datasets). Columns: text (full
            # prompt+gen), prompt, generation_raw, prompt_token_ids,
            # gen_token_ids, prompt_length, gen_length, plus any labels.
            if args.generate:
                from datasets import Dataset as _ArrowDataset
                arrow_rows = {
                    "text": [],
                    "prompt": [],
                    "generation_raw": [],
                    "prompt_token_ids": [],
                    "gen_token_ids": [],
                    "prompt_length": [],
                    "gen_length": [],
                }
                for orig_idx in split_idx.tolist():
                    ids = all_encoded[orig_idx]["input_ids"]
                    if hasattr(ids, "dim") and ids.dim() > 1:
                        ids = ids[0]
                    ids_list = ids.tolist() if hasattr(ids, "tolist") else list(ids)
                    prompt_len = int(original_prompt_lengths[orig_idx])
                    prompt_ids = ids_list[:prompt_len]
                    gen_ids = ids_list[prompt_len:]
                    arrow_rows["prompt_token_ids"].append(prompt_ids)
                    arrow_rows["gen_token_ids"].append(gen_ids)
                    arrow_rows["prompt_length"].append(prompt_len)
                    arrow_rows["gen_length"].append(len(gen_ids))
                    arrow_rows["text"].append(
                        tokenizer.decode(ids_list, skip_special_tokens=False)
                    )
                    arrow_rows["prompt"].append(
                        tokenizer.decode(prompt_ids, skip_special_tokens=False)
                    )
                    arrow_rows["generation_raw"].append(
                        tokenizer.decode(gen_ids, skip_special_tokens=False)
                    )
                if labels is not None and isinstance(labels, dict):
                    for lk, lv in labels.items():
                        arrow_rows[lk] = lv[split_idx].tolist()
                arrow_split_dir = model_dir / split_name
                arrow_split_dir.mkdir(parents=True, exist_ok=True)
                _ArrowDataset.from_dict(arrow_rows).save_to_disk(str(arrow_split_dir))
                logger.info("Saved %s Arrow gen dataset (%d rows) → %s",
                            split_name, len(split_idx), arrow_split_dir)

        del h_full, ids_full
        gc.collect()
        logger.info("Split into train=%d, val=%d, test=%d subdirectories",
                    n_train, n_val, n_test)

        # Remove the now-redundant top-level flat memmap (the splits hold
        # the same data partitioned). Saves ~hidden_states.bin in size.
        flat_files = [
            "hidden_states.bin", "input_ids.bin", "offsets.npy", "lengths.npy",
            "prompt_lengths.npy", "labels.npz", "labels.npy", "metadata.json",
        ]
        removed = 0
        for fname in flat_files:
            fp = output_dir / fname
            if fp.exists():
                fp.unlink()
                removed += 1
        logger.info("Removed %d redundant top-level flat files from %s", removed, output_dir)

    # ------------------------------------------------------------------
    # 13. Cleanup
    # ------------------------------------------------------------------
    extractor.cleanup()
    elapsed = time.time() - t0
    size_gb = offset * hidden_dim * np.dtype(save_dtype).itemsize / 1e9
    logger.info(
        "Done. %d docs, %d tokens, %.2f GB saved to %s  (%.1f min)",
        len(prompts), offset, size_gb, output_dir, elapsed / 60,
    )


# ---------------------------------------------------------------------------
# Config mode (original main logic, preserved intact)
# ---------------------------------------------------------------------------

def _run_config_mode(args) -> None:
    """Config mode: original behaviour — process splits from a YAML config."""

    # ------------------------------------------------------------------
    # 1. Load config
    # ------------------------------------------------------------------
    from ret.config import load_config

    config = load_config(args.config)
    logger.info("Config loaded: LLM=%s, layer=%d, hidden_dim=%d",
                config.llm.name, config.llm.layer_index, config.llm.hidden_dim)

    # Validate that local_dir is set (required for document-mode extraction)
    if config.data.local_dir is None:
        logger.error("This script requires data.local_dir (document mode). "
                     "HF streaming datasets are not supported for caching.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 2. Determine output directory
    # ------------------------------------------------------------------
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
    else:
        output_dir = (
            Path(config.data.local_dir)
            / "cached_hidden_states"
            / f"layer{config.llm.layer_index}"
        )
    logger.info("Output directory: %s", output_dir)

    # ------------------------------------------------------------------
    # 3. Build HiddenStateExtractor
    # ------------------------------------------------------------------
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU.")
        device = "cpu"

    from ret.extraction.hooks import HiddenStateExtractor

    logger.info("Building HiddenStateExtractor ...")
    extractor = HiddenStateExtractor(config.llm, device=device)
    hidden_dim = config.llm.hidden_dim
    save_dtype = np.float16 if args.dtype == "float16" else np.float32

    # ------------------------------------------------------------------
    # 4. Process each split
    # ------------------------------------------------------------------
    from torch.utils.data import DataLoader
    from ret.data.dataset import (
        PaddedPromptDataset,
        build_padded_dataloader,
        collate_padded,
    )
    from transformers import AutoTokenizer

    for split in args.splits:
        split_dir = Path(config.data.local_dir) / split
        if not split_dir.exists():
            logger.warning("Split directory does not exist, skipping: %s", split_dir)
            continue

        logger.info("=" * 60)
        logger.info("Processing split: %s", split)
        logger.info("=" * 60)

        out_split = output_dir / split
        out_split.mkdir(parents=True, exist_ok=True)

        # Check if already extracted
        if (out_split / "metadata.json").exists():
            logger.info("Already extracted (metadata.json exists). Skipping. "
                        "Delete the directory to re-extract.")
            continue

        # Build dataloader directly (not via build_padded_dataloader) to ensure
        # drop_last=False so ALL documents are cached, including the last
        # incomplete batch.
        loader_for_dataset = build_padded_dataloader(
            config.data, config.llm, split=split,
            batch_size=args.batch_size, shuffle=False,
        )
        dataset = loader_for_dataset.dataset
        pad_id = extractor.tokenizer.pad_token_id

        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=lambda b: collate_padded(b, pad_id),
            num_workers=config.data.num_workers,
            pin_memory=True,
            drop_last=False,
        )

        # --- Pass 1: Count total tokens ---
        logger.info("Pass 1: counting total tokens ...")
        num_docs = min(len(dataset), args.max_docs) if args.max_docs else len(dataset)
        total_tokens = 0
        for i in range(num_docs):
            total_tokens += len(dataset.samples[i])

        logger.info("  num_docs=%d, total_tokens=%d (%.1f GB for hidden states in %s)",
                    num_docs, total_tokens,
                    total_tokens * hidden_dim * np.dtype(save_dtype).itemsize / 1e9,
                    args.dtype)

        # --- Allocate memmap files ---
        h_path = out_split / "hidden_states.bin"
        ids_path = out_split / "input_ids.bin"

        h_mmap = np.memmap(
            h_path, dtype=save_dtype, mode="w+",
            shape=(total_tokens, hidden_dim),
        )
        ids_mmap = np.memmap(
            ids_path, dtype=np.int32, mode="w+",
            shape=(total_tokens,),
        )

        offsets_list: list[int] = []
        lengths_list: list[int] = []

        # --- Pass 2: Extract and write ---
        logger.info("Pass 2: extracting hidden states ...")
        offset = 0
        doc_count = 0
        done = False
        t0 = time.time()

        for batch in loader:
            input_ids = batch["input_ids"]  # (B, T)
            attention_mask = batch.get("attention_mask")  # (B, T) or None
            B = input_ids.shape[0]

            # LLM forward pass
            with torch.no_grad():
                h = extractor.extract(
                    input_ids.to(extractor.input_device),
                    attention_mask=attention_mask.to(extractor.input_device)
                    if attention_mask is not None else None,
                )  # (B, T, D) float32

            h_cpu = h.cpu().numpy()
            ids_cpu = input_ids.cpu().numpy()
            mask_cpu = (
                attention_mask.cpu().numpy()
                if attention_mask is not None
                else np.ones((B, input_ids.shape[1]), dtype=np.int64)
            )

            for b in range(B):
                # Check max_docs BEFORE writing to avoid buffer overflow
                if args.max_docs is not None and doc_count >= args.max_docs:
                    done = True
                    break

                valid_len = int(mask_cpu[b].sum())
                if valid_len == 0:
                    logger.warning("Doc %d has zero valid tokens, skipping", doc_count)
                    doc_count += 1
                    continue
                offsets_list.append(offset)
                lengths_list.append(valid_len)

                h_mmap[offset: offset + valid_len] = h_cpu[b, :valid_len].astype(save_dtype)
                ids_mmap[offset: offset + valid_len] = ids_cpu[b, :valid_len].astype(np.int32)

                offset += valid_len
                doc_count += 1

            if done:
                logger.info("  Reached --max-docs limit (%d)", args.max_docs)
                break

            # Progress logging
            if doc_count % max(args.batch_size * 10, 100) < args.batch_size:
                elapsed = time.time() - t0
                docs_per_min = doc_count / (elapsed / 60) if elapsed > 0 else 0
                eta_min = (num_docs - doc_count) / docs_per_min if docs_per_min > 0 else 0
                logger.info("  %d / %d docs  (%.1f docs/min, ETA %.0f min)",
                            doc_count, num_docs, docs_per_min, eta_min)

            # Flush periodically to avoid data loss on crash
            if doc_count % 1000 < args.batch_size:
                h_mmap.flush()
                ids_mmap.flush()

        # Final flush
        h_mmap.flush()
        ids_mmap.flush()
        del h_mmap, ids_mmap
        gc.collect()  # ensure file descriptors are closed before truncation

        # Truncate memmap if fewer tokens were written than allocated
        if offset < total_tokens:
            logger.info("  Truncating memmap: %d tokens written (allocated %d)",
                        offset, total_tokens)
            os.truncate(str(h_path), offset * hidden_dim * np.dtype(save_dtype).itemsize)
            os.truncate(str(ids_path), offset * np.dtype(np.int32).itemsize)
            total_tokens = offset

        # Save index arrays
        np.save(out_split / "offsets.npy", np.array(offsets_list, dtype=np.int64))
        np.save(out_split / "lengths.npy", np.array(lengths_list, dtype=np.int32))

        # Save metadata
        metadata = {
            "model_name": config.llm.name,
            "layer_index": config.llm.layer_index,
            "hidden_dim": hidden_dim,
            "dtype": args.dtype,
            "num_docs": len(offsets_list),
            "total_tokens": int(total_tokens),
            "split": split,
            "max_seq_length": config.data.max_seq_length or 2048,
            "config_path": str(Path(args.config).resolve()),
            "local_dir": config.data.local_dir,
            "llm_dtype": config.llm.dtype,
            "scaffolding": config.data.scaffolding,
            "source_num_docs": len(dataset),
        }
        with open(out_split / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        elapsed = time.time() - t0
        logger.info("Split '%s' done: %d docs, %d tokens in %.1f min",
                    split, doc_count, total_tokens, elapsed / 60)
        logger.info("Saved to: %s", out_split)
        size_gb = total_tokens * hidden_dim * np.dtype(save_dtype).itemsize / 1e9
        logger.info("Hidden states size: %.1f GB", size_gb)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # 5. Cleanup
    # ------------------------------------------------------------------
    extractor.cleanup()
    logger.info("Done. All splits saved to: %s", output_dir)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv=None) -> None:
    args = _parse_args(argv)

    if args.dataset is not None:
        _run_dataset_mode(args)
    elif args.config is not None:
        _run_config_mode(args)
    else:
        print(
            "Error: provide either --dataset <name> or --config <path>",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
