"""Save/load closure predictors to/from disk."""
from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn

from evaluation.closure.predictors import build_closure_predictor

logger = logging.getLogger(__name__)


def save_closure_predictors(
    models: dict[str, dict],
    save_path: str | Path,
    target_variance: float | None = None,
) -> None:
    """Save trained closure predictor models and loss curves to disk.

    Args:
        models: dict mapping variant name -> {
            "model": nn.Module, "input_dim": int, "output_dim": int,
            "loss_type": str, "hidden_dim": int, "num_layers": int,
            "train_losses": list, "val_losses": list
        }
        save_path: path to .pt file
        target_variance: population-level per-element variance of the prediction
            targets, used to normalize per-step losses across methods.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_data = {"__meta__": {"target_variance": target_variance}}
    for name, info in models.items():
        train_losses = info.get("train_losses", [])
        val_losses = info.get("val_losses", [])
        save_data[name] = {
            "state_dict": info["model"].state_dict(),
            "input_dim": info["input_dim"],
            "output_dim": info["output_dim"],
            "loss_type": info["loss_type"],
            "hidden_dim": info.get("hidden_dim", 256),
            "num_layers": info.get("num_layers", 2),
            "train_losses": train_losses,
            "val_losses": val_losses,
            "train_loss_final": train_losses[-1] if train_losses else None,
            "val_loss_final": val_losses[-1] if val_losses else None,
            "val_r2_final": info.get("val_r2_final"),
        }
    torch.save(save_data, save_path)
    print(f"Saved {len(save_data) - 1} closure predictors to {save_path}")


def load_closure_predictors(
    save_path: str | Path,
    device: str = "cuda",
) -> dict | None:
    """Load cached closure predictors. Reconstructs purely from saved data (self-contained).

    Returns dict mapping name -> {"model": nn.Module, "input_dim": int, "output_dim": int,
    "loss_type": str, "hidden_dim": int, "num_layers": int,
    "train_losses": list, "val_losses": list, "train_loss_final": float,
    "val_loss_final": float, "val_r2_final": float}
    plus a special "__meta__" key with {"target_variance": float | None}.
    Returns None if file does not exist.
    """
    save_path = Path(save_path)
    if not save_path.exists():
        return None

    saved = torch.load(save_path, map_location=device, weights_only=False)

    loaded = {}

    # Extract metadata (target_variance etc.)
    meta = saved.pop("__meta__", {})
    loaded["__meta__"] = {
        "target_variance": meta.get("target_variance"),
    }

    for name, info in saved.items():
        hidden_dim = info.get("hidden_dim", 256)
        num_layers = info.get("num_layers", 2)
        input_dim = info["input_dim"]
        output_dim = info["output_dim"]

        model = build_closure_predictor(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        ).to(device)
        model.load_state_dict(info["state_dict"])
        model.eval()

        loaded[name] = {
            "model": model,
            "input_dim": input_dim,
            "output_dim": output_dim,
            "loss_type": info["loss_type"],
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "train_losses": info.get("train_losses", []),
            "val_losses": info.get("val_losses", []),
            "train_loss_final": info.get("train_loss_final"),
            "val_loss_final": info.get("val_loss_final"),
            "val_r2_final": info.get("val_r2_final"),
        }

    print(f"Loaded {len(loaded) - 1} cached closure predictors from {save_path}")
    return loaded


def pick_best_closure_predictor(
    closure_dir: str | Path,
    device: str = "cuda",
    *,
    variant: str = "autonomous",
    loss_type: str = "mse",
) -> tuple[nn.Module, dict]:
    """Walk ``closure_dir/hdim*/predictors.pt`` and return the predictor with
    the lowest validation loss.

    Used by downstream consumers (e.g. closure-regularized unsupervised
    steering) that just want "the best predictor available" without picking a
    specific hidden dim.

    Parameters
    ----------
    closure_dir : path to a ``closure/`` output dir containing per-hidden-dim
        subdirectories (``hdim64/``, ``hdim128/``, …), each with
        ``predictors.pt``.
    variant : which sub-dict to read from each ``predictors.pt``. Default
        ``"autonomous"``.
    loss_type : asserted ``loss_type`` of the chosen predictor; raises
        ``RuntimeError`` if the best candidate has a different loss type (we
        only want like-for-like comparisons).

    Returns
    -------
    (predictor, info)
        ``predictor`` is the loaded ``nn.Module`` already on ``device`` in
        eval mode. ``info`` is the saved metadata dict plus a
        ``"target_variance"`` field copied from ``__meta__``.
    """
    closure_dir = Path(closure_dir)
    candidates: list[tuple[float, Path, dict]] = []
    for p in sorted(closure_dir.glob("hdim*/predictors.pt")):
        saved = torch.load(p, map_location="cpu", weights_only=False)
        variant_info = saved.get(variant)
        if variant_info is None:
            continue
        val_losses = variant_info.get("val_losses", [])
        val_loss = min(val_losses) if val_losses else variant_info.get("val_loss_final")
        if val_loss is None:
            continue
        candidates.append(
            (val_loss, p, {**variant_info, "target_variance": saved["__meta__"]["target_variance"]})
        )
    if not candidates:
        raise FileNotFoundError(
            f"No closure predictors found under {closure_dir} "
            f"(looking for hdim*/predictors.pt with a {variant!r} entry)"
        )
    candidates.sort(key=lambda t: t[0])
    val_loss_best, path, info = candidates[0]
    if info["loss_type"] != loss_type:
        raise RuntimeError(
            f"Best predictor at {path} has loss_type={info['loss_type']!r}; "
            f"expected {loss_type!r}"
        )

    predictor = build_closure_predictor(
        input_dim=info["input_dim"],
        output_dim=info["output_dim"],
        hidden_dim=info["hidden_dim"],
        num_layers=info["num_layers"],
    )
    predictor.load_state_dict(info["state_dict"])
    predictor.to(device).eval()

    val_r2 = (
        1.0 - val_loss_best / info["target_variance"]
        if info.get("target_variance") else float("nan")
    )
    logger.info(
        "Picked closure predictor %s (hdim=%d val_MSE=%.5f val_R²=%.4f)",
        path, info["hidden_dim"], val_loss_best, val_r2,
    )
    return predictor, info
