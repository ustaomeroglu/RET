"""Orchestrate interpretability-score runs for RET or SAE methods.

Flow per sample:
    1. Extract cluster sequence (RET) or top-K latents (SAE).
    2. Call describer LLM A (codes → description) and LLM B (text → description)
       concurrently.
    3. Call judge LLM C (two descriptions → 1–10 score).
    4. Save per-sample artifacts + append to per_sample_scores.jsonl.

Resumable: skips samples whose judge.json already exists.
"""
from __future__ import annotations

import asyncio
import json
import logging
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from evaluation.mis.llm import (
    MISLLM,
    build_codes_prompt,
    build_judge_prompt,
    build_text_prompt,
    describe_from_codes,
    describe_from_text,
    judge_similarity,
)
from evaluation.mis.naming import (
    RETNaming,
    format_ret_naming_table,
    format_sae_naming_table,
    load_sae_latent_names,
    parse_ret_naming,
)
from evaluation.mis.sequence import (
    RETInference,
    SAEInference,
    find_assistant_response_start,
    format_ret_sequence,
    format_ret_sequence_rle,
    format_ret_sequence_windowed,
    format_ret_sequence_windowed_group1,
    format_sae_sequence,
    format_sae_sequence_windowed,
    load_ret_inference,
    load_sae_inference,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class MISConfig:
    method: str                         # "ret" | "sae" | "both"
    output_dir: str
    num_samples: int = 100
    word_limit: int = 80
    model_id: str = "claude-sonnet-4-20250514"
    max_concurrent: int = 5
    device: str = "cuda"

    # RET-specific
    clustering_dir: Optional[str] = None

    # SAE-specific
    sae_checkpoint: Optional[str] = None
    sae_latent_names_path: Optional[str] = None
    top_k: int = 5
    llm_config_yaml: Optional[str] = None    # for SAE (supplies LLM name + cache dir)

    # Max tokens per sample sent to LLM (safety cap)
    max_sample_tokens: int = 20000

    # Window size (in tokens) for both RET-windowed and SAE-windowed
    # aggregations. Same value is used on both sides so the resulting
    # sequences have the same number of rows.
    # Default = 50: the paper MIS configuration (Sec 4.1 / Appendix G).
    window_size: int = 50

    # Token-trajectory variant: "full" keeps the whole chat (scaffolding +
    # user prompt + assistant response); "afterassistant" strips everything
    # before the first '<|start|>assistant' token and describes only the
    # model's generated response.
    # Default = "afterassistant": the paper MIS configuration (Appendix G).
    variant: str = "afterassistant"

    # RET sequence formatting:
    #   "window-topk"  — top-K (group.cluster) pairs per window by duration (default)
    #   "window-group1" — single most-represented Group ID per window,
    #                     with cluster-level naming omitted. Deliberate
    #                     handicap for RET against SAE.
    # Default = "window-group1": the paper MIS configuration (Sec 4.1 /
    # Appendix G — "RET reports the dominant group in each window").
    ret_format: str = "window-group1"

    # If set, the combined pipeline copies SAE describer output, text
    # describer output, and SAE-judge result from the corresponding
    # sample dirs under this path (instead of re-running those API calls).
    # Use when only the RET representation has changed.
    copy_sae_from: Optional[str] = None

    # Mirror of copy_sae_from for the RET side: copies RET describer output,
    # text describer output, and RET-judge result. Use when only the SAE
    # representation has changed.
    copy_ret_from: Optional[str] = None

    # Optional override for SAE top-K (per window) distinct from RET top-K.
    # If None, falls back to top_k.
    # Default = 10: the paper MIS configuration (Appendix G — "SAE reports
    # the top-10 latents by cumulative activation").
    sae_top_k: Optional[int] = 10

    # If True, build describers + save prompts/outputs but skip the judge call.
    skip_judge: bool = False

    def as_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Sample extraction shared helpers
# ---------------------------------------------------------------------------

def _load_test_dataset(config):
    """Load the cached test split's underlying dataset."""
    from ret.data.dataset import CachedHiddenStateDataset

    cache_dir = Path(config.data.cached_hidden_states_dir)
    split_dir = cache_dir / "test"
    if not split_dir.exists():
        raise FileNotFoundError(f"test split not found: {split_dir}")
    return CachedHiddenStateDataset(
        split_dir, llm_config=config.llm, max_seq_length=config.data.max_seq_length,
    )


def _iter_samples(dataset, num: int):
    for i in range(min(num, len(dataset))):
        yield i, dataset[i]


def _write_prompt(path: Path, system: str, user: str) -> None:
    path.write_text(
        "=" * 80 + "\nSYSTEM\n" + "=" * 80 + "\n" + system + "\n\n"
        + "=" * 80 + "\nUSER\n" + "=" * 80 + "\n" + user
    )


def _save_artifacts(
    sample_dir: Path,
    original_text: str,
    naming_header: str,
    sequence_body: str,
    codes_prompt: tuple[str, str],
    text_prompt: tuple[str, str],
    desc_codes: str,
    desc_text: str,
    judge_prompt: tuple[str, str] | None,
    judge_out: dict | None,
) -> None:
    sample_dir.mkdir(parents=True, exist_ok=True)
    (sample_dir / "original_text.txt").write_text(original_text)
    (sample_dir / "codes_sequence.txt").write_text(
        naming_header + "\n\n\n" + sequence_body
    )
    _write_prompt(sample_dir / "prompt_describe_from_codes.txt", *codes_prompt)
    _write_prompt(sample_dir / "prompt_describe_from_text.txt", *text_prompt)
    (sample_dir / "desc_from_codes.txt").write_text(desc_codes)
    (sample_dir / "desc_from_text.txt").write_text(desc_text)
    if judge_prompt is not None:
        _write_prompt(sample_dir / "prompt_judge.txt", *judge_prompt)
    if judge_out is not None:
        (sample_dir / "judge.json").write_text(json.dumps(judge_out, indent=2))


def _already_done(sample_dir: Path, skip_judge: bool) -> bool:
    if skip_judge:
        return (sample_dir / "desc_from_codes.txt").exists() and (
            sample_dir / "desc_from_text.txt"
        ).exists()
    return (sample_dir / "judge.json").exists()


# ---------------------------------------------------------------------------
# Processing one sample
# ---------------------------------------------------------------------------

async def _process_one(
    sample_dir: Path,
    original_text: str,
    naming_header: str,
    sequence_body: str,
    llm: MISLLM,
    word_limit: int,
    is_sae: bool,
    skip_judge: bool,
) -> dict | None:
    codes_prompt = build_codes_prompt(
        naming_header, sequence_body, word_limit=word_limit, is_sae=is_sae,
    )
    text_prompt = build_text_prompt(original_text, word_limit=word_limit)

    desc_codes_task = describe_from_codes(
        llm, naming_header, sequence_body, word_limit=word_limit, is_sae=is_sae,
    )
    desc_text_task = describe_from_text(llm, original_text, word_limit=word_limit)
    desc_codes, desc_text = await asyncio.gather(desc_codes_task, desc_text_task)

    judge_prompt = None
    judge_out = None
    if not skip_judge:
        judge_prompt = build_judge_prompt(desc_codes, desc_text)
        judge_out = await judge_similarity(llm, desc_codes, desc_text)

    _save_artifacts(
        sample_dir, original_text, naming_header, sequence_body,
        codes_prompt, text_prompt, desc_codes, desc_text,
        judge_prompt, judge_out,
    )
    return judge_out


# ---------------------------------------------------------------------------
# RET run
# ---------------------------------------------------------------------------

async def run_ret(cfg: MISConfig) -> dict:
    if cfg.clustering_dir is None:
        raise ValueError("clustering_dir required for RET method")

    clustering_dir = Path(cfg.clustering_dir).resolve()

    # 1. Parse naming
    naming_path = clustering_dir / "cluster_naming" / "naming.txt"
    if not naming_path.exists():
        raise FileNotFoundError(
            f"Expected {naming_path}. The clustering dir must contain a populated "
            "cluster_naming/naming.txt."
        )
    naming = parse_ret_naming(naming_path)
    full_naming_header = format_ret_naming_table(naming)
    logger.info("Loaded naming: %d groups, %d clusters", naming.num_groups, naming.K)

    # 2. Load RET inference state
    ret, config = load_ret_inference(clustering_dir, device=cfg.device, stride=1)

    # 3. Load test dataset
    dataset = _load_test_dataset(config)
    logger.info("Test dataset: %d docs", len(dataset))

    # 4. Output dir + config snapshot
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(
        {**cfg.as_dict(),
         "clustering_dir_resolved": str(clustering_dir),
         "naming_path": str(naming_path),
         "num_docs_available": len(dataset)},
        indent=2,
    ))
    (out_dir / "naming_table.txt").write_text(full_naming_header)

    # 5. LLM client
    llm = MISLLM.create(model=cfg.model_id, max_concurrent=cfg.max_concurrent)

    # 6. Main loop
    samples_dir = out_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "per_sample_scores.jsonl"
    jsonl_f = jsonl_path.open("a", encoding="utf-8")

    tasks = []
    meta_by_task = []

    for doc_idx, item in _iter_samples(dataset, cfg.num_samples):
        sample_dir = samples_dir / f"doc_{doc_idx:04d}"
        if _already_done(sample_dir, cfg.skip_judge):
            continue

        h = item["hidden_states"]               # (T, D)
        ids = item["input_ids"]                 # (T,)
        T = int(h.shape[0])
        if T > cfg.max_sample_tokens:
            logger.warning("Skipping doc %d (length %d > %d)", doc_idx, T, cfg.max_sample_tokens)
            continue

        tokens, labels, groups = ret.sequence_for_sample(h, ids)
        sequence_body = format_ret_sequence_windowed(
            tokens, labels, groups,
            window_size=cfg.window_size,
            top_k_out=cfg.top_k,
        )
        original_text = ret.tokenizer.decode(ids.tolist())

        # Restrict the naming header to only the clusters/groups this sample uses.
        active_clusters = set(int(c) for c in labels.tolist())
        naming_header = format_ret_naming_table(naming, active_clusters=active_clusters)

        coro = _process_one(
            sample_dir=sample_dir,
            original_text=original_text,
            naming_header=naming_header,
            sequence_body=sequence_body,
            llm=llm,
            word_limit=cfg.word_limit,
            is_sae=False,
            skip_judge=cfg.skip_judge,
        )
        tasks.append(coro)
        meta_by_task.append({"doc_idx": doc_idx, "length": T})

    # Execute concurrently
    scores: List[dict] = []
    if tasks:
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for meta, res in zip(meta_by_task, results):
            if isinstance(res, Exception):
                logger.error("doc %d failed: %s", meta["doc_idx"], res)
                rec = {"doc_id": meta["doc_idx"], "length": meta["length"],
                       "score": None, "error": str(res)}
            else:
                score = res.get("score") if res is not None else None
                rec = {"doc_id": meta["doc_idx"], "length": meta["length"],
                       "score": score}
            jsonl_f.write(json.dumps(rec) + "\n")
            jsonl_f.flush()
            scores.append(rec)

    jsonl_f.close()

    # 7. Summary — also read any previously-completed samples from per_sample_scores.jsonl
    summary = _summarize(out_dir, method="ret")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("RET summary: %s", summary)
    return summary


# ---------------------------------------------------------------------------
# SAE run
# ---------------------------------------------------------------------------

async def run_sae(cfg: MISConfig) -> dict:
    from evaluation.clustering.config import load_config_with_overlay

    if cfg.llm_config_yaml is None:
        raise ValueError("llm_config_yaml required for SAE method (to locate cached test data)")
    if cfg.sae_latent_names_path is None:
        raise ValueError("sae_latent_names_path required for SAE method (build it first)")

    config = load_config_with_overlay(cfg.llm_config_yaml)
    latent_names = load_sae_latent_names(cfg.sae_latent_names_path)
    logger.info("Loaded SAE naming: %d latents", len(latent_names))

    # Load SAE + tokenizer
    sae_infer = load_sae_inference(
        sae_checkpoint=cfg.sae_checkpoint,
        llm_name_for_tokenizer=config.llm.name,
        trust_remote_code=config.llm.trust_remote_code,
        top_k=cfg.top_k,
        device=cfg.device,
    )

    dataset = _load_test_dataset(config)
    logger.info("Test dataset: %d docs", len(dataset))

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(
        {**cfg.as_dict(), "num_docs_available": len(dataset)}, indent=2,
    ))

    llm = MISLLM.create(model=cfg.model_id, max_concurrent=cfg.max_concurrent)

    samples_dir = out_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "per_sample_scores.jsonl"
    jsonl_f = jsonl_path.open("a", encoding="utf-8")

    tasks = []
    meta_by_task = []

    for doc_idx, item in _iter_samples(dataset, cfg.num_samples):
        sample_dir = samples_dir / f"doc_{doc_idx:04d}"
        if _already_done(sample_dir, cfg.skip_judge):
            continue

        h = item["hidden_states"]
        ids = item["input_ids"]
        T = int(h.shape[0])
        if T > cfg.max_sample_tokens:
            logger.warning("Skipping doc %d (length %d > %d)", doc_idx, T, cfg.max_sample_tokens)
            continue

        tokens, topk_idx, topk_vals = sae_infer.sequence_for_sample(h, ids)
        sequence_body, active_ids = format_sae_sequence_windowed(
            tokens, topk_idx, topk_vals,
            window_size=cfg.window_size,
            top_k_out=cfg.top_k,
        )
        naming_header = format_sae_naming_table(latent_names, active_ids)
        original_text = sae_infer.tokenizer.decode(ids.tolist())

        coro = _process_one(
            sample_dir=sample_dir,
            original_text=original_text,
            naming_header=naming_header,
            sequence_body=sequence_body,
            llm=llm,
            word_limit=cfg.word_limit,
            is_sae=True,
            skip_judge=cfg.skip_judge,
        )
        tasks.append(coro)
        meta_by_task.append({"doc_idx": doc_idx, "length": T})

    if tasks:
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for meta, res in zip(meta_by_task, results):
            if isinstance(res, Exception):
                logger.error("doc %d failed: %s", meta["doc_idx"], res)
                rec = {"doc_id": meta["doc_idx"], "length": meta["length"],
                       "score": None, "error": str(res)}
            else:
                score = res.get("score") if res is not None else None
                rec = {"doc_id": meta["doc_idx"], "length": meta["length"],
                       "score": score}
            jsonl_f.write(json.dumps(rec) + "\n")
            jsonl_f.flush()

    jsonl_f.close()

    summary = _summarize(out_dir, method="sae")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("SAE summary: %s", summary)
    return summary


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _summarize(out_dir: Path, method: str) -> dict:
    """Aggregate per_sample_scores.jsonl into summary stats."""
    jsonl_path = out_dir / "per_sample_scores.jsonl"
    scores: List[int] = []
    errors = 0
    if jsonl_path.exists():
        with jsonl_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                s = rec.get("score")
                if isinstance(s, int) and 1 <= s <= 10:
                    scores.append(s)
                else:
                    errors += 1

    if scores:
        mean = statistics.mean(scores)
        std = statistics.pstdev(scores) if len(scores) > 1 else 0.0
        hist = {str(i): scores.count(i) for i in range(1, 11)}
    else:
        mean, std, hist = None, None, {}

    return {
        "method": method,
        "n": len(scores),
        "errors": errors,
        "mean": mean,
        "std": std,
        "hist": hist,
    }


# ---------------------------------------------------------------------------
# Combined (RET + SAE + text, with two judges) run
# ---------------------------------------------------------------------------

def _write_prompt_pair(path: Path, system: str, user: str) -> None:
    _write_prompt(path, system, user)


def _already_done_combined(sample_dir: Path, skip_judge: bool) -> bool:
    if skip_judge:
        return (
            (sample_dir / "desc_from_ret_codes.txt").exists()
            and (sample_dir / "desc_from_sae_codes.txt").exists()
            and (sample_dir / "desc_from_text.txt").exists()
        )
    return (
        (sample_dir / "judge_ret_vs_text.json").exists()
        and (sample_dir / "judge_sae_vs_text.json").exists()
    )


async def _process_one_combined(
    sample_dir: Path,
    original_text: str,
    ret_naming_header: str,
    ret_sequence_body: str,
    sae_naming_header: str,
    sae_sequence_body: str,
    llm: MISLLM,
    word_limit: int,
    skip_judge: bool,
    variant: str = "full",
    source_sample_dir_sae: Optional[Path] = None,
    source_sample_dir_ret: Optional[Path] = None,
) -> dict:
    """Run the per-sample pipeline.

    - If ``source_sample_dir_sae`` is set: copy SAE desc + text desc + SAE
      judge from that source. Only compute RET describer + RET judge.
    - If ``source_sample_dir_ret`` is set: copy RET desc + text desc + RET
      judge from that source. Only compute SAE describer + SAE judge.
    - If both are set, raises ValueError.
    - If neither is set: compute all 5 API calls as normal.
    """
    if source_sample_dir_sae is not None and source_sample_dir_ret is not None:
        raise ValueError("cannot copy both SAE and RET from source at the same time")

    ret_codes_prompt = build_codes_prompt(
        ret_naming_header, ret_sequence_body, word_limit=word_limit,
        is_sae=False, variant=variant,
    )
    sae_codes_prompt = build_codes_prompt(
        sae_naming_header, sae_sequence_body, word_limit=word_limit,
        is_sae=True, variant=variant,
    )
    text_prompt = build_text_prompt(original_text, word_limit=word_limit, variant=variant)

    sample_dir.mkdir(parents=True, exist_ok=True)

    if source_sample_dir_sae is not None:
        src = Path(source_sample_dir_sae)
        for required in ("desc_from_sae_codes.txt", "desc_from_text.txt",
                         "judge_sae_vs_text.json"):
            if not (src / required).exists():
                raise FileNotFoundError(f"Source missing {required}: {src}")
        desc_sae = (src / "desc_from_sae_codes.txt").read_text()
        desc_text = (src / "desc_from_text.txt").read_text()
        desc_ret = await describe_from_codes(
            llm, ret_naming_header, ret_sequence_body,
            word_limit=word_limit, is_sae=False, variant=variant,
        )
    elif source_sample_dir_ret is not None:
        src = Path(source_sample_dir_ret)
        for required in ("desc_from_ret_codes.txt", "desc_from_text.txt",
                         "judge_ret_vs_text.json"):
            if not (src / required).exists():
                raise FileNotFoundError(f"Source missing {required}: {src}")
        desc_ret = (src / "desc_from_ret_codes.txt").read_text()
        desc_text = (src / "desc_from_text.txt").read_text()
        desc_sae = await describe_from_codes(
            llm, sae_naming_header, sae_sequence_body,
            word_limit=word_limit, is_sae=True, variant=variant,
        )
    else:
        desc_ret, desc_sae, desc_text = await asyncio.gather(
            describe_from_codes(
                llm, ret_naming_header, ret_sequence_body,
                word_limit=word_limit, is_sae=False, variant=variant,
            ),
            describe_from_codes(
                llm, sae_naming_header, sae_sequence_body,
                word_limit=word_limit, is_sae=True, variant=variant,
            ),
            describe_from_text(llm, original_text, word_limit=word_limit, variant=variant),
        )

    (sample_dir / "original_text.txt").write_text(original_text)
    (sample_dir / "ret_codes_sequence.txt").write_text(
        ret_naming_header + "\n\n\n" + ret_sequence_body
    )
    (sample_dir / "sae_codes_sequence.txt").write_text(
        sae_naming_header + "\n\n\n" + sae_sequence_body
    )
    _write_prompt(sample_dir / "prompt_describe_from_ret_codes.txt", *ret_codes_prompt)
    _write_prompt(sample_dir / "prompt_describe_from_sae_codes.txt", *sae_codes_prompt)
    _write_prompt(sample_dir / "prompt_describe_from_text.txt", *text_prompt)
    (sample_dir / "desc_from_ret_codes.txt").write_text(desc_ret)
    (sample_dir / "desc_from_sae_codes.txt").write_text(desc_sae)
    (sample_dir / "desc_from_text.txt").write_text(desc_text)

    if skip_judge:
        return {"ret_score": None, "sae_score": None}

    ret_judge_prompt = build_judge_prompt(desc_ret, desc_text)
    sae_judge_prompt = build_judge_prompt(desc_sae, desc_text)

    if source_sample_dir_sae is not None:
        ret_judge_out = await judge_similarity(llm, desc_ret, desc_text)
        sae_judge_out = json.loads(
            (Path(source_sample_dir_sae) / "judge_sae_vs_text.json").read_text()
        )
        src_sae_prompt = Path(source_sample_dir_sae) / "prompt_judge_sae_vs_text.txt"
        if src_sae_prompt.exists():
            (sample_dir / "prompt_judge_sae_vs_text.txt").write_text(
                src_sae_prompt.read_text()
            )
        else:
            _write_prompt(sample_dir / "prompt_judge_sae_vs_text.txt", *sae_judge_prompt)
    elif source_sample_dir_ret is not None:
        sae_judge_out = await judge_similarity(llm, desc_sae, desc_text)
        ret_judge_out = json.loads(
            (Path(source_sample_dir_ret) / "judge_ret_vs_text.json").read_text()
        )
        src_ret_prompt = Path(source_sample_dir_ret) / "prompt_judge_ret_vs_text.txt"
        if src_ret_prompt.exists():
            (sample_dir / "prompt_judge_ret_vs_text.txt").write_text(
                src_ret_prompt.read_text()
            )
        else:
            _write_prompt(sample_dir / "prompt_judge_ret_vs_text.txt", *ret_judge_prompt)
        _write_prompt(sample_dir / "prompt_judge_sae_vs_text.txt", *sae_judge_prompt)
    else:
        ret_judge_out, sae_judge_out = await asyncio.gather(
            judge_similarity(llm, desc_ret, desc_text),
            judge_similarity(llm, desc_sae, desc_text),
        )
        _write_prompt(sample_dir / "prompt_judge_sae_vs_text.txt", *sae_judge_prompt)

    # Write RET judge prompt (skipped in the ret-copy branch above, which reused source)
    if source_sample_dir_ret is None:
        _write_prompt(sample_dir / "prompt_judge_ret_vs_text.txt", *ret_judge_prompt)

    (sample_dir / "judge_ret_vs_text.json").write_text(
        json.dumps(ret_judge_out, indent=2)
    )
    (sample_dir / "judge_sae_vs_text.json").write_text(
        json.dumps(sae_judge_out, indent=2)
    )
    return {
        "ret_score": ret_judge_out.get("score"),
        "sae_score": sae_judge_out.get("score"),
    }


async def run_combined(cfg: MISConfig) -> dict:
    if cfg.clustering_dir is None:
        raise ValueError("clustering_dir required for combined method")
    if cfg.sae_latent_names_path is None:
        raise ValueError("sae_latent_names_path required for combined method")
    if cfg.llm_config_yaml is None and cfg.sae_checkpoint is None:
        # RET config is loaded from clustering_meta via load_ret_inference; that
        # will supply the LLM config we need for tokenizer + cached dataset.
        pass

    clustering_dir = Path(cfg.clustering_dir).resolve()

    # --- RET setup ---
    naming_path = clustering_dir / "cluster_naming" / "naming.txt"
    if not naming_path.exists():
        raise FileNotFoundError(f"RET naming.txt not found: {naming_path}")
    naming = parse_ret_naming(naming_path)
    full_ret_naming_header = format_ret_naming_table(naming)
    logger.info("RET naming: %d groups, %d clusters", naming.num_groups, naming.K)

    ret, config = load_ret_inference(clustering_dir, device=cfg.device, stride=1)

    sae_top_k = cfg.sae_top_k if cfg.sae_top_k is not None else cfg.top_k

    # --- SAE setup (skipped when copy_sae_from is set: we only need tokenizer + names) ---
    sae_latent_names = load_sae_latent_names(cfg.sae_latent_names_path)
    logger.info("SAE naming: %d latents", len(sae_latent_names))
    sae_infer = None
    if cfg.copy_sae_from is None:
        sae_infer = load_sae_inference(
            sae_checkpoint=cfg.sae_checkpoint,
            llm_name_for_tokenizer=config.llm.name,
            trust_remote_code=config.llm.trust_remote_code,
            top_k=sae_top_k,
            device=cfg.device,
        )
    else:
        logger.info("copy_sae_from=%s — skipping SAE model load", cfg.copy_sae_from)

    # --- Shared test dataset ---
    dataset = _load_test_dataset(config)
    logger.info("Test dataset: %d docs", len(dataset))

    # --- Output dir + config ---
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(
        {**cfg.as_dict(),
         "clustering_dir_resolved": str(clustering_dir),
         "ret_naming_path": str(naming_path),
         "num_docs_available": len(dataset)},
        indent=2,
    ))
    (out_dir / "ret_naming_table.txt").write_text(full_ret_naming_header)

    llm = MISLLM.create(model=cfg.model_id, max_concurrent=cfg.max_concurrent)

    samples_dir = out_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "per_sample_scores.jsonl"
    jsonl_f = jsonl_path.open("a", encoding="utf-8")

    tasks = []
    meta_by_task = []

    for doc_idx, item in _iter_samples(dataset, cfg.num_samples):
        sample_dir = samples_dir / f"doc_{doc_idx:04d}"
        if _already_done_combined(sample_dir, cfg.skip_judge):
            continue

        h = item["hidden_states"]
        ids = item["input_ids"]
        T = int(h.shape[0])
        if T > cfg.max_sample_tokens:
            logger.warning("Skipping doc %d (length %d > %d)",
                           doc_idx, T, cfg.max_sample_tokens)
            continue

        # Run inference — always RET; SAE only when not copying
        tokens, labels, groups = ret.sequence_for_sample(h, ids)
        ids_list = ids.tolist()
        if sae_infer is not None:
            _tokens, topk_idx, topk_vals = sae_infer.sequence_for_sample(h, ids)
        else:
            topk_idx, topk_vals = None, None

        # Slice to afterassistant if requested.
        if cfg.variant == "afterassistant":
            start_idx = find_assistant_response_start(tokens)
            if start_idx is None:
                logger.warning(
                    "Skipping doc %d: no <|start|>assistant marker found",
                    doc_idx,
                )
                continue
            tokens = tokens[start_idx:]
            labels = labels[start_idx:]
            groups = groups[start_idx:]
            if topk_idx is not None:
                topk_idx = topk_idx[start_idx:]
                topk_vals = topk_vals[start_idx:]
            ids_list = ids_list[start_idx:]
            logger.info(
                "doc %d: afterassistant slice at token %d (len %d -> %d)",
                doc_idx, start_idx, T, len(tokens),
            )

        # RET sequence — formatter depends on cfg.ret_format
        active_clusters = set(int(c) for c in labels.tolist())
        if cfg.ret_format == "window-group1":
            ret_sequence_body = format_ret_sequence_windowed_group1(
                tokens, labels, groups, window_size=cfg.window_size,
            )
            ret_naming_header = format_ret_naming_table(
                naming, active_clusters=active_clusters, groups_only=True,
            )
        else:
            ret_sequence_body = format_ret_sequence_windowed(
                tokens, labels, groups,
                window_size=cfg.window_size,
                top_k_out=cfg.top_k,
            )
            ret_naming_header = format_ret_naming_table(
                naming, active_clusters=active_clusters,
            )

        # SAE sequence — compute or copy from source
        if sae_infer is not None:
            sae_sequence_body, active_ids = format_sae_sequence_windowed(
                tokens, topk_idx, topk_vals,
                window_size=cfg.window_size,
                top_k_out=sae_top_k,
            )
            sae_naming_header = format_sae_naming_table(sae_latent_names, active_ids)
        else:
            src_seq = Path(cfg.copy_sae_from) / "samples" / f"doc_{doc_idx:04d}" / "sae_codes_sequence.txt"
            if not src_seq.exists():
                logger.warning("copy_sae_from: source sae_codes_sequence missing for doc %d — skipping", doc_idx)
                continue
            src_text = src_seq.read_text()
            # sae_codes_sequence.txt is "naming\n\n\nsequence_body" — split on the blank-line separator
            parts = src_text.split("\n\n\n", 1)
            if len(parts) == 2:
                sae_naming_header, sae_sequence_body = parts
            else:
                sae_naming_header = ""
                sae_sequence_body = src_text

        # Text (decoded from same ids range as sequences)
        original_text = ret.tokenizer.decode(ids_list)

        source_sd_sae = None
        if cfg.copy_sae_from is not None:
            src = Path(cfg.copy_sae_from) / "samples" / f"doc_{doc_idx:04d}"
            if src.exists():
                source_sd_sae = src
            else:
                logger.warning(
                    "copy_sae_from: source dir %s missing — will compute SAE from scratch", src,
                )

        source_sd_ret = None
        if cfg.copy_ret_from is not None:
            src = Path(cfg.copy_ret_from) / "samples" / f"doc_{doc_idx:04d}"
            if src.exists():
                source_sd_ret = src
            else:
                logger.warning(
                    "copy_ret_from: source dir %s missing — will compute RET from scratch", src,
                )

        coro = _process_one_combined(
            sample_dir=sample_dir,
            original_text=original_text,
            ret_naming_header=ret_naming_header,
            ret_sequence_body=ret_sequence_body,
            sae_naming_header=sae_naming_header,
            sae_sequence_body=sae_sequence_body,
            llm=llm,
            word_limit=cfg.word_limit,
            skip_judge=cfg.skip_judge,
            variant=cfg.variant,
            source_sample_dir_sae=source_sd_sae,
            source_sample_dir_ret=source_sd_ret,
        )
        tasks.append(coro)
        meta_by_task.append({"doc_idx": doc_idx, "length": T})

    if tasks:
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for meta, res in zip(meta_by_task, results):
            if isinstance(res, Exception):
                logger.error("doc %d failed: %s", meta["doc_idx"], res)
                rec = {"doc_id": meta["doc_idx"], "length": meta["length"],
                       "ret_score": None, "sae_score": None,
                       "error": str(res)}
            else:
                rec = {"doc_id": meta["doc_idx"], "length": meta["length"],
                       "ret_score": res.get("ret_score"),
                       "sae_score": res.get("sae_score")}
            jsonl_f.write(json.dumps(rec) + "\n")
            jsonl_f.flush()

    jsonl_f.close()

    summary = _summarize_combined(out_dir)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("Combined summary: %s", summary)
    return summary


def _summarize_combined(out_dir: Path) -> dict:
    jsonl_path = out_dir / "per_sample_scores.jsonl"
    ret_scores: List[int] = []
    sae_scores: List[int] = []
    if jsonl_path.exists():
        with jsonl_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                e = rec.get("ret_score")
                s = rec.get("sae_score")
                if isinstance(e, int) and 1 <= e <= 10:
                    ret_scores.append(e)
                if isinstance(s, int) and 1 <= s <= 10:
                    sae_scores.append(s)

    def _stats(vals: List[int]) -> dict:
        if not vals:
            return {"n": 0, "mean": None, "std": None, "hist": {}}
        return {
            "n": len(vals),
            "mean": statistics.mean(vals),
            "std": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
            "hist": {str(i): vals.count(i) for i in range(1, 11)},
        }

    return {
        "method": "combined",
        "ret": _stats(ret_scores),
        "sae": _stats(sae_scores),
    }
