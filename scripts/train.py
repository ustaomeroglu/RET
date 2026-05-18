#!/usr/bin/env python3
"""Entry point for JEPA training."""

from accelerate import Accelerator
from accelerate.utils import set_seed

from ret.config import config_from_cli, save_config
from ret.data.dataset import build_dataloader, build_padded_dataloader, build_cached_dataloader
from ret.training.trainer import JEPATrainer
from ret.utils.logging import init_wandb, finish_wandb
from ret.utils.prompt_logger import PromptLogger

from pathlib import Path


def main():
    # split_batches=True ensures the effective batch size stays the same
    # regardless of GPU count (each batch is split across GPUs, not duplicated)
    config = config_from_cli()
    accelerator = Accelerator(
        split_batches=True,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
    )
    set_seed(config.training.seed)

    output_dir = Path(config.logging.output_dir)

    # Main process handles directory creation, config saving, wandb
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        save_config(config, output_dir / "config.yaml")
    accelerator.wait_for_everyone()

    # Data
    mode = config.data.mode
    if accelerator.is_main_process:
        print(f"Loading dataset (mode={mode})...", flush=True)
    if mode == "cached":
        train_loader = build_cached_dataloader(
            config.data, config.llm, split="train",
            batch_size=config.training.batch_size, seed=config.training.seed,
        )
        val_loader = build_cached_dataloader(
            config.data, config.llm, split="val",
            batch_size=config.training.batch_size, shuffle=False,
        )
    elif mode == "document":
        train_loader = build_padded_dataloader(
            config.data, config.llm, split="train",
            batch_size=config.training.batch_size, seed=config.training.seed,
        )
        val_loader = build_padded_dataloader(
            config.data, config.llm, split="val",
            batch_size=config.training.batch_size, shuffle=False,
        )
    else:
        train_loader = build_dataloader(
            config.data, config.llm, split="train",
            max_chunks=config.training.num_chunks,
            batch_size=config.training.batch_size, seed=config.training.seed,
        )
        val_loader = build_dataloader(
            config.data, config.llm, split="val",
            max_chunks=config.training.num_chunks,
            batch_size=config.training.batch_size, shuffle=False,
        )
    if accelerator.is_main_process:
        train_size = "streaming" if not hasattr(train_loader.dataset, "__len__") else str(len(train_loader.dataset))
        val_size = str(len(val_loader.dataset))
        unit = {"cached": "documents (cached)", "document": "documents"}.get(mode, "chunks")
        print(f"Train: {train_size} {unit}, Val: {val_size} {unit}", flush=True)

    # Init wandb (main process only)
    if accelerator.is_main_process:
        init_wandb(config)

    # Prompt logger (main process only)
    prompt_logger = PromptLogger(output_dir) if accelerator.is_main_process else None

    # Trainer
    if accelerator.is_main_process:
        print("Initializing trainer...", flush=True)
    trainer = JEPATrainer(config, accelerator=accelerator, prompt_logger=prompt_logger)

    # Resume from checkpoint if specified
    if config.training.resume_from is not None:
        trainer.resume_from_checkpoint(config.training.resume_from)

    if accelerator.is_main_process:
        print("Starting training...", flush=True)
    trainer.train(train_loader, val_loader, use_wandb=True)

    if accelerator.is_main_process:
        finish_wandb()
        print("Training complete.", flush=True)


if __name__ == "__main__":
    main()
