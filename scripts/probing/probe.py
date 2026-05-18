"""probe.py — thin CLI for hidden-state probing.

Pairs with ``evaluation/probing/``. mmlu and models_under_pressure run
through the single ``evaluation.probing.pipeline.run_probing`` (driven by a
pure-data adapter's declared label tasks). Sycophancy uses its own concrete
per-token pipeline ``evaluation.probing.sycophancy.runner`` (kept separate:
its variable-length causal-prefix probing differs structurally).

Examples
--------
# Models-under-pressure
conda run -n eet python scripts/probing/probe.py \
    --experiment models_under_pressure --model openai/gpt-oss-20b \
    --layer 20 --hidden_state_type all --probe all --max_samples 1000

# MMLU
conda run -n eet python scripts/probing/probe.py \
    --experiment mmlu --probe all

# Sycophancy (pre-extracted hidden states; existing runner)
conda run -n eet python scripts/probing/probe.py \
    --experiment sycophancy --probe all \
    --sycophancy_data_dir data/sycophancy/debate/Qwen2.5-14B-Instruct \
    --sycophancy_layer 23 --sycophancy_mode debate
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from evaluation.probing.config import (
    CacheConfig, RETConfig, EvalConfig, ModelConfig, PCAConfig,
    ProbeConfig, SAEConfig, SplitPathsConfig,
)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run hidden-state probing experiments on LLMs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", default=None,
                   help="YAML config path (sycophancy). CLI args override YAML.")
    p.add_argument("--experiment", default=None,
                   choices=["models_under_pressure", "mmlu", "sycophancy"],
                   help="Which experiment to run.")

    # ---- Model ----
    p.add_argument("--model", default="openai/gpt-oss-20b", help="HuggingFace model ID.")
    p.add_argument("--layer", type=int, default=20, help="Layer index (0-based). -1=last.")
    p.add_argument("--hidden_state_type", choices=["average", "assistant_token", "all"],
                   default="all", help="'average', 'assistant_token', or both ('all').")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--device", default="cuda")
    p.add_argument("--n_gpus", type=int, default=0,
                   help="Number of GPUs (0 = all available).")

    # ---- Probe ----
    p.add_argument("--probe", choices=["linear", "dom", "attention", "transformer", "all"],
                   default=None, help="Probe type(s) to run.")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--l2", type=float, default=1e-4)
    p.add_argument("--probe_batch_size", type=int, default=64)

    # ---- Evaluation ----
    p.add_argument("--eval", choices=["cross_dataset", "within_dataset", "all"],
                   default="all", help="Evaluation protocol.")
    p.add_argument("--val_ratio", type=float, default=0.2)
    p.add_argument("--max_samples", type=int, default=None,
                   help="Max samples per dataset (smoke tests).")

    # ---- Paths ----
    p.add_argument("--output_dir", default=None,
                   help="Result dir. Default outputs/probing/<experiment>.")
    p.add_argument("--mmlu_split_dir", default="data/mmlu/eval",
                   help="Pre-split MMLU train/ and val/ Arrow dir.")

    # ---- Sycophancy ----
    p.add_argument("--sycophancy_data_dir",
                   default="data/sycophancy/debate/Qwen2.5-14B-Instruct")
    p.add_argument("--sycophancy_layer", type=int, default=23)
    p.add_argument("--max_conversations", type=int, default=None)
    p.add_argument("--sycophancy_mode", default=None,
                   choices=["debate", "false_presupposition"],
                   help="Label mode (REQUIRED for sycophancy).")
    p.add_argument("--truncate_on", type=int, default=None)

    # ---- Hidden-state caching ----
    p.add_argument("--cache_dir", default=None,
                   help="Dir to save/load cached hidden states (.pt).")
    p.add_argument("--use_cache", action="store_true",
                   help="Load hidden states from --cache_dir instead of the LLM.")

    # ---- Pre-split probe data ----
    p.add_argument("--probe_train_dir", default=None,
                   help="Per-subdataset train JSONL dir (prepare_probing_splits.py).")
    p.add_argument("--probe_test_dir", default=None,
                   help="Per-subdataset test JSONL dir.")

    # ---- PCA / SAE / RET baselines ----
    p.add_argument("--pca_dir", default=None)
    p.add_argument("--pca_dim", type=int, default=None)
    p.add_argument("--sae_checkpoint", default=None)
    p.add_argument("--sae_hf_repo", default=None)
    p.add_argument("--ret_checkpoint", default=None,
                   help="Trained RET checkpoint; probes run on z instead of h.")

    # ---- GPT-OSS ----
    p.add_argument("--strip_gpt_info", action="store_true",
                   help="Strip the GPT-OSS harmony system prompt.")

    p.add_argument("--log_level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def _common_cfgs(args) -> dict:
    if args.output_dir:
        output_dir = args.output_dir
    elif args.ret_checkpoint:
        output_dir = os.path.join(str(Path(args.ret_checkpoint).parent),
                                  "probing_analysis", args.experiment)
    else:
        output_dir = f"outputs/probing/{args.experiment}"
    return dict(
        model=ModelConfig(model_id=args.model, layer=args.layer,
                          hidden_state_type=args.hidden_state_type,
                          batch_size=args.batch_size, device=args.device,
                          n_gpus=args.n_gpus),
        probe=ProbeConfig(probe_type=args.probe, lr=args.lr,
                          max_epochs=args.epochs, l2=args.l2,
                          probe_batch_size=args.probe_batch_size),
        eval=EvalConfig(eval_type=args.eval, val_ratio=args.val_ratio,
                        max_samples_per_dataset=args.max_samples),
        ret=RETConfig(checkpoint_path=args.ret_checkpoint, device=args.device),
        cache=CacheConfig(cache_dir=args.cache_dir, use_cache=args.use_cache),
        splits=SplitPathsConfig(probe_train_dir=args.probe_train_dir,
                                probe_test_dir=args.probe_test_dir),
        pca=PCAConfig(pca_dir=args.pca_dir, pca_dim=args.pca_dim),
        sae=SAEConfig(checkpoint_path=args.sae_checkpoint, hf_repo=args.sae_hf_repo),
        output_dir=output_dir,
    )


def _run_pressure(args, c):
    from evaluation.datasets.pressure.config import PressureConfig
    from evaluation.datasets.pressure.dataset import PressureDataset
    from evaluation.probing.pipeline import run_probing

    cfg = PressureConfig(
        model=c["model"], probe=c["probe"], eval=c["eval"], ret=c["ret"],
        cache=c["cache"], splits=c["splits"], pca=c["pca"], sae=c["sae"],
        output_dir=c["output_dir"], include_gpt_info=not args.strip_gpt_info,
    )
    tr = te = None
    if args.probe_train_dir and args.probe_test_dir:
        tr = os.path.join(args.probe_train_dir, "pressure.jsonl")
        te = os.path.join(args.probe_test_dir, "pressure.jsonl")
    dataset = PressureDataset(train_jsonl=tr, test_jsonl=te,
                              include_gpt_info=not args.strip_gpt_info)
    return run_probing(dataset, cfg)


def _run_mmlu(args, c):
    from evaluation.datasets.mmlu.config import MMLUConfig
    from evaluation.datasets.mmlu.dataset import MMLUDataset
    from evaluation.probing.pipeline import run_probing

    cfg = MMLUConfig(
        model=c["model"], probe=c["probe"], eval=c["eval"], ret=c["ret"],
        cache=c["cache"], splits=c["splits"], pca=c["pca"], sae=c["sae"],
        mmlu_split_dir=args.mmlu_split_dir, output_dir=c["output_dir"],
    )
    return run_probing(MMLUDataset(split_dir=args.mmlu_split_dir), cfg)


def _run_sycophancy(args, c):
    """Deferred path — uses the existing sycophancy runner verbatim."""
    from evaluation.datasets.sycophancy.config import SycophancyProbingConfig
    from evaluation.probing.sycophancy.runner import run

    if args.sycophancy_mode is None:
        print("ERROR: --sycophancy_mode is required for sycophancy.", file=sys.stderr)
        sys.exit(1)
    output_dir = c["output_dir"]
    if not args.output_dir and not args.ret_checkpoint:
        data_dir = args.sycophancy_data_dir.rstrip("/")
        output_dir = os.path.join(
            "outputs", os.path.basename(data_dir), "sycophancy",
            os.path.basename(os.path.dirname(data_dir)), "baselines", "h", "probing",
        )
    cfg = SycophancyProbingConfig(
        model=c["model"], probe=c["probe"], eval=c["eval"], ret=c["ret"],
        cache=c["cache"], splits=c["splits"], pca=c["pca"], sae=c["sae"],
        data_dir=args.sycophancy_data_dir, layer=args.sycophancy_layer,
        output_dir=output_dir, max_conversations=args.max_conversations,
        mode=args.sycophancy_mode, truncate_on=args.truncate_on,
    )
    return run(cfg)


def _run_sycophancy_yaml(yaml_cfg: dict, args):
    """Deferred path — sycophancy YAML config, existing runner verbatim."""
    from evaluation.datasets.sycophancy.config import SycophancyProbingConfig
    from evaluation.probing.sycophancy.runner import run

    syc = yaml_cfg.get("sycophancy", {})
    model = yaml_cfg.get("model", {})
    probe = yaml_cfg.get("probe", {})
    eval_ = yaml_cfg.get("eval", {})
    pca = yaml_cfg.get("pca", {})
    ret = yaml_cfg.get("ret", {})
    logging_ = yaml_cfg.get("logging", {})
    device = getattr(args, "device", None) or model.get("device", "cuda")
    cfg = SycophancyProbingConfig(
        model=ModelConfig(device=device,
                          hidden_state_type=model.get("hidden_state_type", "all")),
        probe=ProbeConfig(
            probe_type=args.probe if args.probe else probe.get("probe_type", "all"),
            lr=args.lr if args.lr != 1e-3 else probe.get("lr", 1e-3),
            max_epochs=args.epochs if args.epochs != 1000 else probe.get("max_epochs", 1000),
            l2=args.l2 if args.l2 != 1e-4 else probe.get("l2", 1e-4),
            probe_batch_size=(args.probe_batch_size if args.probe_batch_size != 64
                              else probe.get("probe_batch_size", 64)),
        ),
        eval=EvalConfig(eval_type=eval_.get("eval_type", "within_dataset"),
                        val_ratio=eval_.get("val_ratio", 0.2)),
        ret=RETConfig(checkpoint_path=args.ret_checkpoint or ret.get("checkpoint_path"),
                      device=device),
        pca=PCAConfig(pca_dir=pca.get("pca_dir"), pca_dim=pca.get("pca_dim")),
        data_dir=syc.get("data_dir", "data/sycophancy/debate/Qwen2.5-14B-Instruct"),
        layer=syc.get("layer", 23),
        output_dir=args.output_dir or logging_.get("output_dir",
                    "outputs/probing/sycophancy"),
        max_conversations=args.max_conversations,
        max_tokens_per_turn=syc.get("max_tokens_per_turn"),
        mode=syc.get("mode"), truncate_on=syc.get("truncate_on"),
        split_file=syc.get("split_file", "split_indices.json"),
        user_context_only=syc.get("user_context_only", False),
        seed=int(syc.get("seed", 42)),
    )
    return run(cfg)


def main():
    args = build_arg_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.config:
        import yaml
        with open(args.config) as f:
            yaml_cfg = yaml.safe_load(f)
        experiment = args.experiment or yaml_cfg.get("experiment")
        if experiment == "sycophancy":
            _run_sycophancy_yaml(yaml_cfg, args)
            return
        if not args.experiment:
            args.experiment = experiment

    if not args.experiment:
        print("Error: --experiment is required (or --config with experiment key).",
              file=sys.stderr)
        sys.exit(1)

    c = _common_cfgs(args)
    if args.experiment == "models_under_pressure":
        _run_pressure(args, c)
    elif args.experiment == "mmlu":
        _run_mmlu(args, c)
    elif args.experiment == "sycophancy":
        _run_sycophancy(args, c)
    else:
        print(f"Unknown experiment: {args.experiment}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
