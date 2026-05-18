#!/usr/bin/env python3
"""Predictor-loss OOD diagnostic for unsupervised steering runs.

Hypothesis:
  Gradient steering on h can push h into OOD regions where the encoder E
  maps it adversarially to a target cluster center, even though the LLM's
  downstream behavior doesn't follow. A predictor P: z_t -> z_{t+1}
  trained on real LLM trajectories should fail (high MSE) on OOD z's.

We use the closure self-predictor (autonomous variant, MSE loss) — it is
trained specifically for self-predictivity to early-stopping convergence,
typically achieving higher R² than the RET training-time predictor.

Usage:
  python scripts/unsupervised_steering/diagnose_predictor_ood.py \\
      --run-dir outputs/.../<steering-run-dir> \\
      --closure-dir outputs/.../<ret-checkpoint-dir>/closure
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from evaluation.closure.io import pick_best_closure_predictor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ood_diag")

REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------


def load_predictor(closure_dir: Path, device: str):
    """Load the best closure autonomous self-predictor (MSE variant)."""
    predictor, info = pick_best_closure_predictor(
        closure_dir, device=device, variant="autonomous", loss_type="mse",
    )
    return predictor, "mse", info


# ---------------------------------------------------------------------------


def per_step_loss(z: np.ndarray, predictor: torch.nn.Module, loss_type: str,
                  device: str) -> np.ndarray:
    """Compute per-step prediction loss between P(z_t) and z_{t+1}.

    z: (T, d_z). Returns array of length T-1 with per-step loss
    (per-element MSE = mean squared error averaged over the d_z dims).
    """
    if z.shape[0] < 2:
        return np.zeros(0, dtype=np.float32)
    z_t = torch.from_numpy(z[:-1]).float().to(device)
    z_tp1 = torch.from_numpy(z[1:]).float().to(device)
    with torch.no_grad():
        z_hat = predictor(z_t)  # (T-1, d_z) — closure predictor takes (N, d_z) directly
    if loss_type == "mse":
        loss = ((z_hat - z_tp1) ** 2).mean(dim=-1)
    elif loss_type == "l1":
        loss = (z_hat - z_tp1).abs().mean(dim=-1)
    elif loss_type == "cosine":
        zhn = torch.nn.functional.normalize(z_hat, dim=-1)
        ztn = torch.nn.functional.normalize(z_tp1, dim=-1)
        loss = 1.0 - (zhn * ztn).sum(dim=-1)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
    return loss.cpu().numpy()


# ---------------------------------------------------------------------------


def load_condition(cond_dir: Path) -> dict | None:
    raw_path = cond_dir / "z_trajectory_raw.npy"
    proc_path = cond_dir / "z_trajectory.npy"
    pl_path = cond_dir / "prompt_len.txt"
    z_path = raw_path if raw_path.exists() else proc_path
    if not z_path.exists() or not pl_path.exists():
        logger.warning("Skipping %s — missing z_trajectory*.npy or prompt_len.txt",
                       cond_dir.name)
        return None
    z = np.load(z_path)  # (T, d_z)
    prompt_len = int(pl_path.read_text().strip())

    diag_path = cond_dir / "diagnostics.jsonl"
    steered_gen_idx: list[int] = []
    if diag_path.exists():
        with open(diag_path) as f:
            for i, line in enumerate(f):
                d = json.loads(line)
                if d.get("was_steered"):
                    steered_gen_idx.append(i)
    return {
        "name": cond_dir.name,
        "z": z,
        "prompt_len": prompt_len,
        "steered_global": [prompt_len + i for i in steered_gen_idx],
    }


# ---------------------------------------------------------------------------


def rolling(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1 or len(x) < w:
        return x
    cs = np.cumsum(np.insert(x, 0, 0.0))
    out = (cs[w:] - cs[:-w]) / w
    pad = np.full(w - 1, out[0])
    return np.concatenate([pad, out])


# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True,
                    help="Steering run directory containing condition subdirs.")
    ap.add_argument("--closure-dir", type=Path, required=True,
                    help="Directory holding hdim*/predictors.pt closure predictors.")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--rolling-window", type=int, default=5)
    args = ap.parse_args()

    if not args.run_dir.is_dir():
        raise SystemExit(f"run-dir does not exist: {args.run_dir}")

    predictor, loss_type, pred_info = load_predictor(args.closure_dir, args.device)

    cond_dirs = sorted(p for p in args.run_dir.iterdir() if p.is_dir())
    conds = [load_condition(d) for d in cond_dirs]
    conds = [c for c in conds if c is not None]
    if not conds:
        raise SystemExit(f"No valid conditions under {args.run_dir}")
    logger.info("Found %d conditions: %s", len(conds), [c["name"] for c in conds])

    # Compute per-step loss for each condition
    for c in conds:
        c["loss"] = per_step_loss(c["z"], predictor, loss_type, args.device)

    # ---- Plot --------------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    cmap = plt.get_cmap("tab10")

    for i, c in enumerate(conds):
        color = "black" if c["name"] == "baseline" else cmap(i % 10)
        x = np.arange(len(c["loss"]))
        axes[0].plot(x, c["loss"], lw=0.8, alpha=0.8, color=color, label=c["name"])
        axes[1].plot(x, rolling(c["loss"], args.rolling_window), lw=1.4,
                     color=color, label=c["name"])

        # mark prompt boundary (same x for both panels because z's are aligned)
        axes[0].axvline(c["prompt_len"] - 1, color="gray", lw=0.5, ls="--", alpha=0.4)
        # shade steered tokens for this condition along the bottom of top panel
        if c["steered_global"]:
            ymax = max(c["loss"].max() for c in conds)
            yband_low = ymax * (0.96 - 0.04 * i)
            yband_high = yband_low + ymax * 0.03
            axes[0].fill_between(
                c["steered_global"], yband_low, yband_high,
                color=color, alpha=0.5, step="pre",
            )

    val_loss_ref = pred_info["val_loss"]
    target_var = pred_info["target_variance"]
    val_r2 = pred_info["val_r2"]
    for ax in axes:
        ax.axhline(val_loss_ref, color="black", lw=1.0, ls=":",
                   label=f"closure val MSE = {val_loss_ref:.4f}")
        ax.axhline(target_var, color="darkred", lw=1.0, ls=":",
                   label=f"target variance = {target_var:.4f}")

    axes[0].set_ylabel(f"per-step {loss_type} loss")
    axes[0].set_title(
        f"Closure predictor MSE by condition  —  {args.run_dir.name}\n"
        f"closure hdim={pred_info['hidden_dim']}, val R²={val_r2:.3f}; "
        f"dashed v-line = end of prompt; coloured bars at top = steered tokens"
    )
    axes[0].legend(fontsize=7, loc="upper left", ncol=2)
    axes[0].grid(alpha=0.2)

    axes[1].set_xlabel("token index (combined prompt + generation)")
    axes[1].set_ylabel(f"rolling-{args.rolling_window} {loss_type} loss")
    axes[1].grid(alpha=0.2)
    axes[1].legend(fontsize=7, loc="upper left", ncol=2)

    fig.tight_layout()
    out_path = args.run_dir / "predictor_ood_diagnostic.png"
    fig.savefig(out_path, dpi=140)
    logger.info("Saved plot -> %s", out_path)

    # ---- Numerical summary -------------------------------------------------
    print()
    print(f"{'condition':<32}  {'prompt':>8}  {'pre':>8}  {'steered':>8}  {'post':>8}")
    print("-" * 78)
    for c in conds:
        loss = c["loss"]
        prompt_end = c["prompt_len"] - 1  # last loss index in prompt window
        steer_glob = sorted(c["steered_global"])
        steer_idx = [i for i in steer_glob if i < len(loss)]

        prompt_loss = loss[:prompt_end].mean() if prompt_end > 0 else float("nan")

        if steer_idx:
            first, last = steer_idx[0], steer_idx[-1]
            pre_mask = np.arange(prompt_end, first)
            post_mask = np.arange(last + 1, len(loss))
            steered_loss = loss[steer_idx].mean()
        else:
            pre_mask = np.arange(prompt_end, len(loss))
            post_mask = np.arange(0, 0)
            steered_loss = float("nan")

        pre_loss = loss[pre_mask].mean() if len(pre_mask) else float("nan")
        post_loss = loss[post_mask].mean() if len(post_mask) else float("nan")

        print(f"{c['name']:<32}  {prompt_loss:>8.4f}  {pre_loss:>8.4f}  "
              f"{steered_loss:>8.4f}  {post_loss:>8.4f}")
    print()


if __name__ == "__main__":
    main()
