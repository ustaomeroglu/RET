"""Single probing pipeline.

`run_probing(dataset, cfg)` is the single prompt-level probing path: one
concrete extract -> (cache) -> encode -> per-task probe -> save flow,
driven by a :class:`~evaluation.datasets.base.ProbingDataset` adapter's
*declared* label tasks (no per-experiment subclassing, no run() overrides).

It subsumes, with identical metric behavior:
  - mmlu ``_run_mmlu_presplit``  (pre-split, multi-task, plain prompts)
  - pressure ``run()`` path-3/4  (single-pool, multi-subtask, chat-wrapped)
  - pressure ``_run_with_splits``(pre-split JSONL, multi-subtask)
  - the generic base ``_run_no_splits`` / ``_run_with_splits`` for the
    single-dataset case.

Out of scope (kept in ``pressure/runner.py``, P4 sign-off item): pressure's
``_run_from_memmap`` stream-RET-encode + lazy-attention path over the 64 GB
memmap cache — a scale optimization tied to the memmap format.
"""
from __future__ import annotations

import logging
import os
from typing import Dict

from evaluation.datasets.base import ProbingDataset

logger = logging.getLogger(__name__)


def _eval_split_key(splits: Dict) -> str:
    """The non-train split used for evaluation in pre-split mode."""
    for k in ("val", "test"):
        if k in splits:
            return k
    raise KeyError(
        f"pre-split dataset must expose a 'val' or 'test' split; got {sorted(splits)}"
    )


def run_probing(dataset: ProbingDataset, cfg) -> Dict:
    """Run all configured probes for every label task the adapter declares."""
    from evaluation.probing.model_utils import (
        load_model, extract_hidden_states_multi_mode, free_model,
        compute_system_prefix_len, build_prompt,
    )
    from evaluation.probing.hidden_state_io import (
        save_hidden_states, load_hidden_states, cache_exists,
        repool_hidden_states, repool_from_all_tokens,
    )
    from evaluation.probing.experiment_utils import (
        build_probe_list_for_task, needed_modes_with_ret, get_pooled_modes,
        apply_ret_encoding, apply_pca_encoding_to_cache,
        apply_sae_encoding_to_cache, save_probe_results, save_training_curves,
        probe_output_dir,
    )
    from evaluation.probing.evaluation import (
        within_dataset_eval, presplit_within_dataset_eval, print_results,
    )
    from evaluation.probing.prompt_inspector import (
        save_prompt_samples, SampleAccumulator,
    )
    from ret.utils.prompt_logger import PromptLogger

    os.makedirs(cfg.output_dir, exist_ok=True)

    exp = dataset.name
    sys_text = dataset.system_prompt
    gpt_info = dataset.include_gpt_info

    splits = dataset.load(max_samples=cfg.eval.max_samples_per_dataset)
    if not splits:
        raise RuntimeError(f"{exp}: adapter returned no splits.")
    presplit = set(splits) != {"all"}
    eval_key = _eval_split_key(splits) if presplit else None

    ret_active = cfg.ret.checkpoint_path is not None
    modes = needed_modes_with_ret(cfg, ret_active)

    use_cache = cfg.cache.use_cache and cfg.cache.cache_dir is not None
    save_cache = cfg.cache.cache_dir is not None

    def _cache_name(split: str) -> str:
        return exp if not presplit else f"{exp}_{split}"

    # ---- 1. hidden states per split (load .pt cache or extract once) -------
    hs_by_split: Dict[str, Dict] = {}

    all_cached = use_cache and all(
        cache_exists(cfg.cache.cache_dir, exp, _cache_name(s)) for s in splits
    )

    if all_cached:
        logger.info("Loading hidden states from cache: %s", cfg.cache.cache_dir)
        for s in splits:
            cached = repool_hidden_states(
                load_hidden_states(cfg.cache.cache_dir, exp, _cache_name(s))
            )
            hs_by_split[s] = cached["hidden_states"]
    else:
        tokenizer, model = load_model(
            cfg.model.model_id, cfg.model.device, n_gpus=cfg.model.n_gpus
        )
        sys_prefix_len = compute_system_prefix_len(
            tokenizer, sys_text, include_gpt_info=gpt_info
        )
        fwd_acc = SampleAccumulator(
            os.path.join(cfg.output_dir, "forward_pass_samples.jsonl"),
            tokenizer, system_text=sys_text,
            source=f"{exp}_extraction", include_gpt_info=gpt_info,
        )
        prompt_logger = PromptLogger(cfg.output_dir)

        for s, split in splits.items():
            raw = split.prompts
            if dataset.chat_wrap:
                prompts = [
                    build_prompt(t, tokenizer, system_text=sys_text,
                                 include_gpt_info=gpt_info)
                    for t in raw
                ]
            else:
                prompts = list(raw)
            prompt_logger.log_prompts(prompts, tokenizer, source="probing")

            hs = extract_hidden_states_multi_mode(
                prompts, model, tokenizer, layer=cfg.model.layer,
                modes=list(modes), batch_size=cfg.model.batch_size,
                sample_accumulator=fwd_acc,
            )
            hs = repool_from_all_tokens(hs, sys_prefix_len)
            hs_by_split[s] = hs
            logger.info("  %s/%s: extracted modes %s", exp, s, list(hs.keys()))

            tag = "" if not presplit else f"_{s}"
            save_prompt_samples(
                os.path.join(cfg.output_dir, f"prompt_samples_{exp}{tag}.jsonl"),
                prompts, tokenizer, system_text=sys_text,
                source=exp, max_samples=100, include_gpt_info=gpt_info,
            )
            if save_cache:
                first_task = next(iter(split.tasks.values()))
                meta = split.metadata or [
                    {"index": i} for i in range(len(first_task.labels))
                ]
                save_hidden_states(
                    cfg.cache.cache_dir, exp, _cache_name(s),
                    hs, first_task.labels, meta,
                    system_prefix_len=sys_prefix_len,
                )

        fwd_acc.close()
        prompt_logger.save()
        free_model(model)

    # ---- 2. encode (RET -> PCA -> SAE), per split -------------------------
    enc_modes = modes
    for s in list(hs_by_split):
        hs_by_split[s], enc_modes = apply_ret_encoding(cfg, hs_by_split[s], modes)
        hs_by_split[s] = apply_pca_encoding_to_cache(cfg, hs_by_split[s])
        hs_by_split[s] = apply_sae_encoding_to_cache(
            cfg, hs_by_split[s], device=cfg.model.device
        )
    pooled_modes = get_pooled_modes(enc_modes)

    # ---- 3. per declared label task: build probes, eval, save -------------
    tasks = splits["train" if presplit else "all"].tasks
    all_results: Dict = {}

    for task_name, ref_task in tasks.items():
        logger.info("=== task: %s (type=%s, n_classes=%d) ===",
                    task_name, ref_task.task_type, ref_task.n_classes)
        task_probes = build_probe_list_for_task(
            cfg, task_type=ref_task.task_type, n_classes=ref_task.n_classes
        )

        for probe_name, probe_factory in task_probes:
            probe_dir = probe_output_dir(cfg.output_dir, probe_name, exp)
            os.makedirs(probe_dir, exist_ok=True)
            all_results.setdefault(probe_name, {})
            probe_results = all_results[probe_name]
            cm_base = os.path.join(probe_dir, "confusion_matrices", task_name)
            os.makedirs(cm_base, exist_ok=True)

            eval_modes = (["all_tokens"] if probe_name == "attention"
                          else pooled_modes)

            for mode in eval_modes:
                if presplit:
                    tr = splits["train"].tasks[task_name].labels
                    te = splits[eval_key].tasks[task_name].labels
                    r = presplit_within_dataset_eval(
                        ["all"],
                        {"all": hs_by_split["train"][mode]}, {"all": tr},
                        {"all": hs_by_split[eval_key][mode]}, {"all": te},
                        probe_factory, output_dir=cm_base,
                        probe_name=f"{probe_name}_{task_name}_{mode}",
                    )
                else:
                    y = splits["all"].tasks[task_name].labels
                    r = within_dataset_eval(
                        ["all"], {"all": hs_by_split["all"][mode]}, {"all": y},
                        probe_factory, val_ratio=cfg.eval.val_ratio,
                        output_dir=cm_base,
                        probe_name=f"{probe_name}_{task_name}_{mode}",
                    )
                probe_results.setdefault(f"{task_name}/{mode}", {})[probe_name] = (
                    r.get("all", {})
                )
                print_results(
                    {probe_name: r},
                    title=f"{exp} | {task_name} | hs={mode} | "
                          f"{'presplit' if presplit else 'within'}",
                )

    for probe_name, probe_results in all_results.items():
        probe_dir = probe_output_dir(cfg.output_dir, probe_name, exp)
        save_probe_results(probe_dir, probe_name, probe_results)
        save_training_curves(probe_dir, probe_results, probe_name)

    return all_results
