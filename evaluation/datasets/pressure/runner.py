"""
Experiment: models_under_pressure

Probing hidden states of gpt-oss-20b on inputs from the
Arrrlex/models-under-pressure dataset.

Three subtasks:
    tone         — 4-class: casual, overly polite, vulnerable, angry
    prompt_style — 5-class: Third Person, Character Perspective, Question,
                            Response to Situation, Prompt to LLM
    high_stakes  — binary:  True / False

Pipeline:
1. Load dataset and encode labels for all subtasks.
2. Extract hidden states ONCE (all modes in a single forward pass).
3. For each subtask, train all configured probes; evaluate; save results.
No generation needed — labels come directly from the dataset.

Supports hidden-state disk caching via CacheConfig:
    cfg.cache.use_cache = True  → load from disk, skip LLM entirely.
    cfg.cache.cache_dir = path  → save extracted hidden states for reuse.

Results are saved per-probe into separate subdirectories.

Note: multi-subtask / multi-class structure differs from the binary single-
dataset base class pipeline.  This subclass overrides run() and provides
its own presplit path rather than using the inherited ones.
"""
from __future__ import annotations

import logging
import os
from typing import Dict, List

import numpy as np

from evaluation.datasets.pressure.config import PressureConfig
from evaluation.probing.model_utils import (
    load_model, extract_hidden_states_multi_mode, free_model, compute_system_prefix_len,
)
from evaluation.datasets.pressure.data import load_pressure_dataset, SUBTASKS
from evaluation.probing.evaluation import (
    within_dataset_eval, presplit_within_dataset_eval, print_results,
)
from evaluation.probing.hidden_state_io import (
    save_hidden_states, load_hidden_states, cache_exists,
    repool_hidden_states, repool_from_all_tokens,
)
from evaluation.probing.experiment_utils import (
    build_probe_list_for_task, needed_modes_with_ret, get_pooled_modes,
    apply_ret_encoding, apply_pca_encoding_to_cache, apply_sae_encoding_to_cache,
    save_probe_results, save_training_curves, probe_output_dir,
)
from evaluation.probing.prompt_inspector import save_prompt_samples, SampleAccumulator
from ret.utils.prompt_logger import PromptLogger

logger = logging.getLogger(__name__)

EXPERIMENT_NAME = "models_under_pressure"
_CACHE_DS_NAME = "pressure"
_SYS = "You are a helpful assistant."


class ModelsUnderPressureRunner:
    """Legacy models-under-pressure probing (memmap / multi-subtask path).

    Concrete, standalone. The standard pressure path now goes through
    ``run_probing``; this is retained for the 64 GB memmap-stream variant
    not yet ported.
    """

    # ------------------------------------------------------------------
    # Subtask helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _run_subtask_presplit(cfg, subtask_name, train_hs, train_labels,
                              test_hs, test_labels, pooled_modes,
                              probe_name, probe_factory, probe_results, output_dir):
        """Evaluate one probe on one subtask using pre-split data."""
        info = SUBTASKS[subtask_name]
        logger.info("=== Subtask (presplit): %s (task_type=%s, n_classes=%d) ===",
                    subtask_name, info.task_type, info.n_classes)

        cm_base = os.path.join(output_dir, "confusion_matrices", subtask_name)
        os.makedirs(cm_base, exist_ok=True)

        if probe_name == "attention":
            tr_dict = {"all": train_hs["all_tokens"]}
            te_dict = {"all": test_hs["all_tokens"]}
            r = presplit_within_dataset_eval(
                ["all"], tr_dict, {"all": train_labels}, te_dict, {"all": test_labels},
                probe_factory, output_dir=cm_base,
                probe_name=f"{probe_name}_{subtask_name}_all_tokens")
            key = f"{subtask_name}/all_tokens"
            probe_results[key] = {probe_name: r.get("all", {})}
            print_results({probe_name: r},
                          title=f"pressure presplit | {subtask_name} | hs=all_tokens")
        else:
            for hs_mode in pooled_modes:
                tr_dict = {"all": train_hs[hs_mode]}
                te_dict = {"all": test_hs[hs_mode]}
                r = presplit_within_dataset_eval(
                    ["all"], tr_dict, {"all": train_labels}, te_dict, {"all": test_labels},
                    probe_factory, output_dir=cm_base,
                    probe_name=f"{probe_name}_{subtask_name}_{hs_mode}")
                key = f"{subtask_name}/{hs_mode}"
                probe_results.setdefault(key, {})[probe_name] = r.get("all", {})
                print_results({probe_name: r},
                              title=f"pressure presplit | {subtask_name} | hs={hs_mode}")

    @staticmethod
    def _run_subtask(cfg, subtask_name, hs_cache, labels, pooled_modes,
                     probe_name, probe_factory, probe_results, output_dir):
        """Evaluate one probe on one subtask's labels."""
        info = SUBTASKS[subtask_name]
        logger.info("=== Subtask: %s (task_type=%s, n_classes=%d) ===",
                    subtask_name, info.task_type, info.n_classes)

        cm_base = os.path.join(output_dir, "confusion_matrices", subtask_name)
        os.makedirs(cm_base, exist_ok=True)

        if probe_name == "attention":
            X_dict = {"all": hs_cache["all_tokens"]}
            y_dict = {"all": labels}
            r = within_dataset_eval(
                ["all"], X_dict, y_dict, probe_factory,
                val_ratio=cfg.eval.val_ratio,
                output_dir=cm_base,
                probe_name=f"{probe_name}_{subtask_name}_all_tokens",
            )
            key = f"{subtask_name}/all_tokens"
            probe_results[key] = {probe_name: r.get("all", {})}
            print_results({probe_name: r},
                          title=f"pressure | {subtask_name} | hs=all_tokens")
        else:
            for hs_mode in pooled_modes:
                X_dict = {"all": hs_cache[hs_mode]}
                y_dict = {"all": labels}
                r = within_dataset_eval(
                    ["all"], X_dict, y_dict, probe_factory,
                    val_ratio=cfg.eval.val_ratio,
                    output_dir=cm_base,
                    probe_name=f"{probe_name}_{subtask_name}_{hs_mode}",
                )
                key = f"{subtask_name}/{hs_mode}"
                probe_results.setdefault(key, {})[probe_name] = r.get("all", {})
                print_results({probe_name: r},
                              title=f"pressure | {subtask_name} | hs={hs_mode}")

    @staticmethod
    def _run_subtask_attention_memmap(cfg, subtask_name,
                                      train_dataset, train_labels,
                                      val_all_tokens, val_labels,
                                      probe_name, probe_factory,
                                      probe_results, output_dir):
        """Run attention probe on one subtask using lazy memmap train data."""
        info = SUBTASKS[subtask_name]
        logger.info("=== Subtask (memmap attention): %s (task_type=%s, n_classes=%d) ===",
                    subtask_name, info.task_type, info.n_classes)

        cm_base = os.path.join(output_dir, "confusion_matrices", subtask_name)
        os.makedirs(cm_base, exist_ok=True)

        probe = probe_factory()
        # Train with lazy dataset (dynamic per-batch padding)
        probe.fit(train_dataset, train_labels,
                  X_val=val_all_tokens, y_val=val_labels)
        # Evaluate on val
        y_pred = probe.predict(val_all_tokens)

        from evaluation.probing.evaluation import compute_metrics, print_confusion_matrix, save_confusion_matrix
        metrics = compute_metrics(val_labels, y_pred)
        cm = np.array(metrics["confusion_matrix"])
        classes = metrics["cm_classes"]
        print_confusion_matrix(cm, classes)

        cm_path = os.path.join(
            cm_base, f"cm_within_{probe_name}_{subtask_name}_all_tokens_all.png")
        save_confusion_matrix(cm, classes, cm_path)

        if hasattr(probe, "history_"):
            metrics["training_history"] = probe.history_

        key = f"{subtask_name}/all_tokens"
        probe_results[key] = {probe_name: metrics}
        print_results({probe_name: {"all": metrics}},
                      title=f"pressure memmap | {subtask_name} | hs=all_tokens")

    # ------------------------------------------------------------------
    # Memmap cache path (with embedded train/val splits + encoding)
    # ------------------------------------------------------------------

    def _run_from_memmap(self, cfg: PressureConfig) -> Dict:
        """Load train/val from memmap splits, apply encoding, run presplit probes.

        Used when --cache_dir points to a memmap directory (has metadata.json)
        and --use_cache is set.  Loads the train and val splits from
        splits.json, applies PCA/SAE encoding if configured, and runs
        presplit within-dataset probes per subtask.

        Attention probe uses lazy per-batch loading via MemmapTokenDataset
        (dynamic padding, no full materialization).
        """
        from pathlib import Path
        from evaluation.probing.hidden_state_io import (
            load_hidden_states_from_memmap, MemmapTokenDataset,
            pad_all_tokens_from_memmap,
        )

        os.makedirs(cfg.output_dir, exist_ok=True)
        cache_dir = cfg.cache.cache_dir

        logger.info("Loading train/val splits from memmap cache: %s", cache_dir)
        train_data = load_hidden_states_from_memmap(
            cache_dir, modes={"average", "assistant_token"}, split="train",
        )
        val_data = load_hidden_states_from_memmap(
            cache_dir, modes={"average", "assistant_token"}, split="val",
        )

        train_hs = {k: v for k, v in train_data.items()
                    if k in ("average", "assistant_token")}
        val_hs = {k: v for k, v in val_data.items()
                  if k in ("average", "assistant_token")}
        train_labels_dict = train_data["labels"]
        val_labels_dict = val_data["labels"]

        # Keep raw memmap handles for attention probe lazy loading
        train_memmap = train_data["_memmap_handle"]
        train_offsets = train_data["_offsets"]
        train_prompt_lens = train_data["_prompt_lengths"]
        train_sys_prefix = train_data["_sys_prefix"]

        val_memmap = val_data["_memmap_handle"]
        val_offsets = val_data["_offsets"]
        val_prompt_lens = val_data["_prompt_lengths"]
        val_sys_prefix = val_data["_sys_prefix"]

        logger.info("Train: %d samples, Val: %d samples",
                    train_hs["average"].shape[0], val_hs["average"].shape[0])

        # Apply encoding (RET, PCA, SAE, or identity)
        ret_active = cfg.ret.checkpoint_path is not None
        has_encoding = (
            ret_active
            or (getattr(cfg, "pca", None) and cfg.pca.pca_dir)
            or (getattr(cfg, "sae", None) and (cfg.sae.checkpoint_path or cfg.sae.hf_repo))
        )

        if ret_active:
            # RET encoding: stream-encode with dynamic padding per batch.
            # Avoids materializing the full padded array (64 GB for train)
            # by iterating through MemmapTokenDataset batches (~37 MB each).
            import torch
            from torch.utils.data import DataLoader as _DL
            from evaluation.probing.ret_utils import load_ret_encoder, free_ret_encoder
            from evaluation.probing.hidden_state_io import MemmapTokenDataset, collate_pad

            ret_encoder, _ret_cfg = load_ret_encoder(
                cfg.ret.checkpoint_path, device=cfg.ret.device)
            z_dim = _ret_cfg.encoder.z_dim

            for split_tag, (mm, offs, plens, spfx) in [
                ("train", (train_memmap, train_offsets, train_prompt_lens, train_sys_prefix)),
                ("val", (val_memmap, val_offsets, val_prompt_lens, val_sys_prefix)),
            ]:
                n_docs = len(offs)
                dummy_labels = np.zeros(n_docs, dtype=np.int64)
                ds = MemmapTokenDataset(mm, offs, plens, spfx, dummy_labels)
                loader = _DL(ds, batch_size=16, shuffle=False,
                             collate_fn=collate_pad, num_workers=0)

                avg_z = np.zeros((n_docs, z_dim), dtype=np.float32)
                asst_z = np.zeros((n_docs, z_dim), dtype=np.float32)
                pos = 0
                for h_b, mask_b, _ in loader:
                    h_b = h_b.to(cfg.ret.device)
                    mask_b = mask_b.to(cfg.ret.device)
                    with torch.no_grad():
                        z_b, _ = ret_encoder(h_b, attention_mask=mask_b)
                    B = z_b.shape[0]
                    for j in range(B):
                        m = mask_b[j].bool()
                        z_valid = z_b[j][m]  # (n_real_tokens, z_dim)
                        avg_z[pos] = z_valid.mean(dim=0).cpu().numpy()
                        asst_z[pos] = z_valid[-1].cpu().numpy()
                        pos += 1
                logger.info("RET stream-encoded %s: %d docs → z_dim=%d", split_tag, n_docs, z_dim)

                if split_tag == "train":
                    train_hs = {"average": avg_z, "assistant_token": asst_z}
                else:
                    val_hs = {"average": avg_z, "assistant_token": asst_z}

            free_ret_encoder(ret_encoder)

        train_hs = apply_pca_encoding_to_cache(cfg, train_hs)
        val_hs = apply_pca_encoding_to_cache(cfg, val_hs)
        train_hs = apply_sae_encoding_to_cache(cfg, train_hs, device=cfg.model.device)
        val_hs = apply_sae_encoding_to_cache(cfg, val_hs, device=cfg.model.device)

        pooled_modes = get_pooled_modes(set(train_hs.keys()))
        logger.info("Pooled modes: %s, representation dim: %d",
                    pooled_modes, train_hs[pooled_modes[0]].shape[1])

        all_results: Dict = {}

        for subtask_name in SUBTASKS:
            if (subtask_name not in train_labels_dict
                    or subtask_name not in val_labels_dict):
                logger.warning("Subtask '%s' missing labels, skipping", subtask_name)
                continue

            info = SUBTASKS[subtask_name]
            subtask_probes = build_probe_list_for_task(
                cfg, task_type=info.task_type, n_classes=info.n_classes,
            )
            # Skip attention probe when PCA/SAE encoding is active (encoding
            # operates on pooled representations, not per-token)
            if has_encoding:
                subtask_probes = [(n, f) for n, f in subtask_probes
                                  if n != "attention"]

            for probe_name, probe_factory in subtask_probes:
                probe_dir = probe_output_dir(cfg.output_dir, probe_name, EXPERIMENT_NAME)
                os.makedirs(probe_dir, exist_ok=True)
                if probe_name not in all_results:
                    all_results[probe_name] = {}

                if probe_name == "attention":
                    # Lazy per-batch loading for attention probe
                    train_dataset = MemmapTokenDataset(
                        train_memmap, train_offsets, train_prompt_lens,
                        train_sys_prefix, train_labels_dict[subtask_name],
                    )
                    # Val: pad fully (small set, ~243 docs)
                    val_all_tokens = pad_all_tokens_from_memmap(
                        val_memmap, val_offsets, val_prompt_lens, val_sys_prefix,
                    )

                    self._run_subtask_attention_memmap(
                        cfg, subtask_name,
                        train_dataset, train_labels_dict[subtask_name],
                        val_all_tokens, val_labels_dict[subtask_name],
                        probe_name, probe_factory,
                        all_results[probe_name], output_dir=probe_dir,
                    )
                else:
                    self._run_subtask_presplit(
                        cfg, subtask_name,
                        train_hs, train_labels_dict[subtask_name],
                        val_hs, val_labels_dict[subtask_name],
                        pooled_modes, probe_name, probe_factory,
                        all_results[probe_name], output_dir=probe_dir,
                    )

        for probe_name, probe_results in all_results.items():
            probe_dir = probe_output_dir(cfg.output_dir, probe_name, EXPERIMENT_NAME)
            save_probe_results(probe_dir, probe_name, probe_results)
            save_training_curves(probe_dir, probe_results, probe_name)

        return all_results

    # ------------------------------------------------------------------
    # Main entry point (overrides base class run())
    # ------------------------------------------------------------------

    def run(self, cfg: PressureConfig) -> Dict:
        if cfg.splits.probe_train_dir and cfg.splits.probe_test_dir:
            return self._run_with_splits(cfg)

        # Detect memmap cache and use embedded splits
        if cfg.cache.use_cache and cfg.cache.cache_dir:
            from pathlib import Path
            cache_dir = Path(cfg.cache.cache_dir)
            if (cache_dir / "metadata.json").exists():
                return self._run_from_memmap(cfg)

        os.makedirs(cfg.output_dir, exist_ok=True)

        ret_active = cfg.ret.checkpoint_path is not None
        modes = needed_modes_with_ret(cfg, ret_active)

        use_cache = cfg.cache.use_cache and cfg.cache.cache_dir is not None
        save_cache = cfg.cache.cache_dir is not None

        if use_cache and cache_exists(cfg.cache.cache_dir, EXPERIMENT_NAME, _CACHE_DS_NAME):
            logger.info("Loading hidden states from cache: %s", cfg.cache.cache_dir)
            cached = repool_hidden_states(
                load_hidden_states(cfg.cache.cache_dir, EXPERIMENT_NAME, _CACHE_DS_NAME))
            hs_cache = cached["hidden_states"]
            labels_dict = cached["labels"]
        else:
            tokenizer, model = load_model(
                cfg.model.model_id, cfg.model.device, n_gpus=cfg.model.n_gpus
            )
            sys_prefix_len = compute_system_prefix_len(
                tokenizer, _SYS, include_gpt_info=cfg.include_gpt_info
            )
            fwd_acc = SampleAccumulator(
                os.path.join(cfg.output_dir, "forward_pass_samples.jsonl"),
                tokenizer, system_text=_SYS,
                source=f"{EXPERIMENT_NAME}_extraction",
                include_gpt_info=cfg.include_gpt_info,
            )

            logger.info("Loading models-under-pressure dataset...")
            prompts, labels_dict = load_pressure_dataset(
                max_samples=cfg.eval.max_samples_per_dataset,
                tokenizer=tokenizer,
                include_gpt_info=cfg.include_gpt_info,
            )
            N = len(prompts)
            logger.info("Dataset size: %d samples", N)

            prompt_logger = PromptLogger(cfg.output_dir)
            prompt_logger.log_prompts(prompts, tokenizer, source="probing")
            prompt_logger.save()

            logger.info("Extracting hidden states for modes: %s", sorted(modes))
            hs_cache = extract_hidden_states_multi_mode(
                prompts, model, tokenizer,
                layer=cfg.model.layer,
                modes=list(modes),
                batch_size=cfg.model.batch_size,
                sample_accumulator=fwd_acc,
            )
            hs_cache = repool_from_all_tokens(hs_cache, sys_prefix_len)
            logger.info("Extraction complete. Cached modes: %s", list(hs_cache.keys()))

            save_prompt_samples(
                os.path.join(cfg.output_dir, "prompt_samples_pressure.jsonl"),
                prompts, tokenizer, system_text=_SYS,
                source=EXPERIMENT_NAME, max_samples=100,
                include_gpt_info=cfg.include_gpt_info,
            )

            if save_cache:
                save_hidden_states(
                    cfg.cache.cache_dir, EXPERIMENT_NAME, _CACHE_DS_NAME,
                    hs_cache, labels_dict,
                    [{"prompt_index": i} for i in range(N)],
                    system_prefix_len=sys_prefix_len,
                )

            fwd_acc.close()
            free_model(model)

        # RET encoding
        hs_cache, modes = apply_ret_encoding(cfg, hs_cache, modes)
        pooled_modes = get_pooled_modes(modes)

        all_results: Dict = {}

        for subtask_name, labels in labels_dict.items():
            info = SUBTASKS[subtask_name]
            subtask_probes = build_probe_list_for_task(
                cfg, task_type=info.task_type, n_classes=info.n_classes
            )

            for probe_name, probe_factory in subtask_probes:
                probe_dir = probe_output_dir(cfg.output_dir, probe_name, EXPERIMENT_NAME)
                os.makedirs(probe_dir, exist_ok=True)
                if probe_name not in all_results:
                    all_results[probe_name] = {}

                self._run_subtask(
                    cfg, subtask_name, hs_cache, labels, pooled_modes,
                    probe_name, probe_factory, all_results[probe_name],
                    output_dir=probe_dir,
                )

        for probe_name, probe_results in all_results.items():
            probe_dir = probe_output_dir(cfg.output_dir, probe_name, EXPERIMENT_NAME)
            save_probe_results(probe_dir, probe_name, probe_results)
            save_training_curves(probe_dir, probe_results, probe_name)

        return all_results

    def _run_with_splits(self, cfg: PressureConfig) -> Dict:
        from evaluation.datasets.pressure.splits import load_pressure_splits
        from evaluation.probing.model_utils import build_prompt

        os.makedirs(cfg.output_dir, exist_ok=True)

        train_path = os.path.join(cfg.splits.probe_train_dir, "pressure.jsonl")
        test_path = os.path.join(cfg.splits.probe_test_dir, "pressure.jsonl")

        ret_active = cfg.ret.checkpoint_path is not None
        modes = needed_modes_with_ret(cfg, ret_active)

        use_cache = cfg.cache.use_cache and cfg.cache.cache_dir is not None
        save_cache = cfg.cache.cache_dir is not None

        train_hs_cache = None
        test_hs_cache = None
        train_labels_dict = None
        test_labels_dict = None

        all_cached = use_cache and (
            cache_exists(cfg.cache.cache_dir, EXPERIMENT_NAME, f"{_CACHE_DS_NAME}_train")
            and cache_exists(cfg.cache.cache_dir, EXPERIMENT_NAME, f"{_CACHE_DS_NAME}_test")
        )

        if all_cached:
            logger.info("Loading presplit pressure hidden states from cache.")
            c_tr = repool_hidden_states(
                load_hidden_states(cfg.cache.cache_dir, EXPERIMENT_NAME,
                                   f"{_CACHE_DS_NAME}_train"))
            train_hs_cache = c_tr["hidden_states"]; train_labels_dict = c_tr["labels"]
            c_te = repool_hidden_states(
                load_hidden_states(cfg.cache.cache_dir, EXPERIMENT_NAME,
                                   f"{_CACHE_DS_NAME}_test"))
            test_hs_cache = c_te["hidden_states"]; test_labels_dict = c_te["labels"]
        else:
            tokenizer, model = load_model(
                cfg.model.model_id, cfg.model.device, n_gpus=cfg.model.n_gpus
            )
            sys_prefix_len = compute_system_prefix_len(
                tokenizer, _SYS, include_gpt_info=cfg.include_gpt_info
            )
            fwd_acc = SampleAccumulator(
                os.path.join(cfg.output_dir, "forward_pass_samples.jsonl"),
                tokenizer, system_text=_SYS,
                source=f"{EXPERIMENT_NAME}_extraction",
                include_gpt_info=cfg.include_gpt_info,
            )

            for split_tag, split_path in [("train", train_path), ("test", test_path)]:
                texts, labels_dict = load_pressure_splits(split_path)
                prompts = [build_prompt(t, tokenizer, system_text=_SYS,
                                        include_gpt_info=cfg.include_gpt_info)
                           for t in texts]
                hs = extract_hidden_states_multi_mode(
                    prompts, model, tokenizer, layer=cfg.model.layer,
                    modes=list(modes), batch_size=cfg.model.batch_size,
                    sample_accumulator=fwd_acc,
                )
                hs = repool_from_all_tokens(hs, sys_prefix_len)
                save_prompt_samples(
                    os.path.join(cfg.output_dir,
                                 f"prompt_samples_pressure_{split_tag}.jsonl"),
                    prompts, tokenizer, system_text=_SYS,
                    source=EXPERIMENT_NAME, max_samples=100,
                    include_gpt_info=cfg.include_gpt_info,
                )

                if save_cache:
                    save_hidden_states(
                        cfg.cache.cache_dir, EXPERIMENT_NAME,
                        f"{_CACHE_DS_NAME}_{split_tag}",
                        hs, labels_dict,
                        [{"prompt_index": i} for i in range(len(prompts))],
                        system_prefix_len=sys_prefix_len,
                    )

                if split_tag == "train":
                    train_hs_cache = hs; train_labels_dict = labels_dict
                else:
                    test_hs_cache = hs; test_labels_dict = labels_dict

            fwd_acc.close()
            free_model(model)

        train_hs_cache, modes = apply_ret_encoding(cfg, train_hs_cache, modes)
        test_hs_cache, _ = apply_ret_encoding(cfg, test_hs_cache, modes)
        pooled_modes = get_pooled_modes(modes)

        all_results: Dict = {}

        for subtask_name in SUBTASKS:
            if (subtask_name not in train_labels_dict
                    or subtask_name not in test_labels_dict):
                continue
            info = SUBTASKS[subtask_name]
            subtask_probes = build_probe_list_for_task(
                cfg, task_type=info.task_type, n_classes=info.n_classes
            )

            for probe_name, probe_factory in subtask_probes:
                probe_dir = probe_output_dir(cfg.output_dir, probe_name, EXPERIMENT_NAME)
                os.makedirs(probe_dir, exist_ok=True)
                if probe_name not in all_results:
                    all_results[probe_name] = {}

                self._run_subtask_presplit(
                    cfg, subtask_name,
                    train_hs_cache, train_labels_dict[subtask_name],
                    test_hs_cache, test_labels_dict[subtask_name],
                    pooled_modes, probe_name, probe_factory,
                    all_results[probe_name], output_dir=probe_dir,
                )

        for probe_name, probe_results in all_results.items():
            probe_dir = probe_output_dir(cfg.output_dir, probe_name, EXPERIMENT_NAME)
            save_probe_results(probe_dir, probe_name, probe_results)
            save_training_curves(probe_dir, probe_results, probe_name)

        return all_results


_runner = ModelsUnderPressureRunner()


def run(cfg: PressureConfig) -> Dict:
    return _runner.run(cfg)
