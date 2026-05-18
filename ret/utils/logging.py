from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

import wandb

from ret.config import ExperimentConfig


def init_wandb(config: ExperimentConfig) -> None:
    """Initialize wandb logging."""
    wandb.init(
        project=config.logging.wandb_project,
        entity=config.logging.wandb_entity,
        name=config.logging.wandb_run_name,
        config=asdict(config),
    )


def log_metrics(metrics: dict, step: int) -> None:
    """Log metrics to wandb."""
    wandb.log(metrics, step=step)


def finish_wandb() -> None:
    """Finish wandb run."""
    wandb.finish()


class LocalLogger:
    """Logs metrics to CSV files and generates loss curve plots at the end."""

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._train_rows: list[dict] = []
        self._val_rows: list[dict] = []

    def log_train(self, metrics: dict, step: int) -> None:
        self._train_rows.append({"step": step, **metrics})

    def log_val(self, metrics: dict, step: int) -> None:
        self._val_rows.append({"step": step, **metrics})

    def save(self) -> None:
        """Write CSV files, a summary JSON, and loss curve plots."""
        self._write_csv(self._train_rows, self.output_dir / "train_metrics.csv")
        if self._val_rows:
            self._write_csv(self._val_rows, self.output_dir / "val_metrics.csv")
        self._write_summary()
        self._plot_curves()

    def _write_csv(self, rows: list[dict], path: Path) -> None:
        if not rows:
            return
        all_keys = list(rows[0].keys())
        for row in rows[1:]:
            for k in row:
                if k not in all_keys:
                    all_keys.append(k)
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    def _write_summary(self) -> None:
        """Write a human-readable summary of the training run."""
        summary = {}
        if self._train_rows:
            last = self._train_rows[-1]
            first = self._train_rows[0]
            summary["total_steps"] = last["step"]
            summary["final_train_loss"] = last.get("train/total_loss")
            summary["final_pred_loss"] = last.get("train/pred_loss")
            summary["initial_train_loss"] = first.get("train/total_loss")
            summary["final_lr"] = last.get("train/lr")
            summary["final_mean_std"] = last.get("reg/mean_std")
        if self._val_rows:
            last_val = self._val_rows[-1]
            summary["final_val_pred_loss"] = last_val.get("val/pred_loss")
        with open(self.output_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    def _plot_curves(self) -> None:
        if not self._train_rows:
            return
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return

        steps = [r["step"] for r in self._train_rows]

        # -- Loss curves --
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Total loss
        ax = axes[0, 0]
        vals = [r.get("train/total_loss") for r in self._train_rows]
        ax.plot(steps, vals, linewidth=0.8, label="train")
        if self._val_rows:
            vs = [r["step"] for r in self._val_rows]
            vv = [r.get("val/pred_loss") for r in self._val_rows]
            ax.plot(vs, vv, linewidth=0.8, label="val pred")
            ax.legend(fontsize=8)
        ax.set_ylabel("Loss")
        ax.set_title("Total Loss")
        ax.grid(True, alpha=0.3)

        # Prediction loss
        ax = axes[0, 1]
        vals = [r.get("train/pred_loss") for r in self._train_rows]
        ax.plot(steps, vals, linewidth=0.8)
        ax.set_ylabel("Loss")
        ax.set_title("Prediction Loss")
        ax.grid(True, alpha=0.3)

        # Regularizer components
        ax = axes[1, 0]
        var_vals = [r.get("vicreg/variance_loss") for r in self._train_rows]
        cov_vals = [r.get("vicreg/covariance_loss") for r in self._train_rows]
        sig_vals = [r.get("sigreg/statistic") for r in self._train_rows]
        if any(v is not None for v in var_vals):
            ax.plot(steps, var_vals, linewidth=0.8, label="variance")
        if any(v is not None for v in cov_vals):
            ax.plot(steps, cov_vals, linewidth=0.8, label="covariance")
        if any(v is not None for v in sig_vals):
            ax.plot(steps, sig_vals, linewidth=0.8, label="sigreg statistic")
        ax.set_ylabel("Loss")
        ax.set_title("Regularizer Components")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Mean std + LR
        ax = axes[1, 1]
        std_vals = [r.get("reg/mean_std") for r in self._train_rows]
        if any(v is not None for v in std_vals):
            ax.plot(steps, std_vals, linewidth=0.8, color="tab:blue", label="mean std")
            ax.set_ylabel("Mean Std", color="tab:blue")
        ax2 = ax.twinx()
        lr_vals = [r.get("train/lr") for r in self._train_rows]
        if any(v is not None for v in lr_vals):
            ax2.plot(steps, lr_vals, linewidth=0.8, color="tab:orange", alpha=0.6, label="LR")
            ax2.set_ylabel("Learning Rate", color="tab:orange")
        ax.set_title("Mean Std & LR Schedule")
        ax.grid(True, alpha=0.3)

        for ax_row in axes:
            for ax in ax_row:
                ax.set_xlabel("Step")

        fig.tight_layout()
        fig.savefig(self.output_dir / "loss_curves.png", dpi=150)
        plt.close(fig)
