"""Unit tests for the closure (self-predictivity) library.

Targets ``evaluation/closure/{predictors.py, io.py}`` — the core math that
produces paper Figure 2's R² numbers and the capacity-sweep machinery for
Figure 7. No GPU required.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from evaluation.closure.io import (
    load_closure_predictors,
    save_closure_predictors,
)
from evaluation.closure.predictors import (
    build_closure_predictor,
    build_token_projector,
    compute_loss,
    compute_per_element_loss,
    compute_r2,
    compute_r2_streaming,
)


# ---------------------------------------------------------------------------
# build_closure_predictor / build_token_projector
# ---------------------------------------------------------------------------

def test_build_closure_predictor_shapes() -> None:
    p = build_closure_predictor(input_dim=16, output_dim=8, hidden_dim=32, num_layers=2)
    x = torch.randn(4, 16)
    y = p(x)
    assert y.shape == (4, 8)


def test_build_closure_predictor_supports_input_ne_output() -> None:
    """Closure probes typically have input_dim ≠ output_dim (e.g. h→z or z→z)."""
    p = build_closure_predictor(input_dim=128, output_dim=32, hidden_dim=64, num_layers=2)
    y = p(torch.randn(2, 128))
    assert y.shape == (2, 32)


def test_build_token_projector_shape() -> None:
    proj = build_token_projector(d_h=64, control_dim=16)
    # TokenProjector takes (B, T, d_h) -> (B, T, control_dim)
    u = proj(torch.randn(2, 5, 64))
    assert u.shape == (2, 5, 16)


# ---------------------------------------------------------------------------
# compute_loss + compute_per_element_loss
# ---------------------------------------------------------------------------

def test_compute_loss_mse_matches_manual() -> None:
    pred = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    tgt  = torch.tensor([[1.5, 2.5], [3.5, 4.0]])
    expected = ((pred - tgt) ** 2).mean()
    assert compute_loss(pred, tgt, "mse").item() == pytest.approx(expected.item())


def test_compute_loss_cosine_perfect_alignment_is_zero() -> None:
    pred = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    tgt = pred.clone()
    # cosine = 1.0 → 1 - cos = 0.0
    assert compute_loss(pred, tgt, "cosine").item() == pytest.approx(0.0, abs=1e-6)


def test_compute_loss_cosine_orthogonal_is_one() -> None:
    pred = torch.tensor([[1.0, 0.0]])
    tgt = torch.tensor([[0.0, 1.0]])
    # cosine = 0.0 → 1 - cos = 1.0
    assert compute_loss(pred, tgt, "cosine").item() == pytest.approx(1.0, abs=1e-6)


def test_compute_loss_unsupported_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported closure loss_type"):
        compute_loss(torch.zeros(1, 1), torch.zeros(1, 1), "bogus")


def test_compute_per_element_loss_mse_per_sample() -> None:
    pred = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    tgt = torch.tensor([[1.0, 2.0], [3.0, 5.0]])
    per = compute_per_element_loss(pred, tgt, "mse")
    # Mean over last dim per sample: [(0+0)/2, (0+1)/2] = [0, 0.5]
    assert per.shape == (2,)
    assert torch.allclose(per, torch.tensor([0.0, 0.5]))


# ---------------------------------------------------------------------------
# R² (batch + streaming agreement)
# ---------------------------------------------------------------------------

def test_compute_r2_perfect_prediction() -> None:
    """R² = 1 when pred == target."""
    target = torch.randn(50, 4)
    assert compute_r2(target, target) == pytest.approx(1.0, abs=1e-6)


def test_compute_r2_constant_prediction_at_mean() -> None:
    """Predicting the column-mean gives R² ≈ 0."""
    torch.manual_seed(0)
    target = torch.randn(100, 4)
    pred = target.mean(dim=0, keepdim=True).expand_as(target)
    # SS_res ≈ SS_tot → R² ≈ 0
    assert compute_r2(pred, target) == pytest.approx(0.0, abs=1e-6)


def test_compute_r2_zero_variance_target_returns_zero() -> None:
    """All-constant target → SS_tot = 0 → return 0.0 by convention."""
    target = torch.full((20, 3), 1.5)
    pred = torch.full((20, 3), 0.0)
    assert compute_r2(pred, target) == 0.0


def test_compute_r2_streaming_matches_batch() -> None:
    """The streaming formula must agree with the batch formula on the same data."""
    torch.manual_seed(42)
    target = torch.randn(200, 8)
    pred = target + 0.1 * torch.randn(200, 8)
    # Batch
    r2_batch = compute_r2(pred, target)
    # Streaming inputs:  ss_res, target_sum, target_sq_sum, n
    ss_res = (pred - target).pow(2).sum().item()
    tgt_sum = target.sum(dim=0)
    tgt_sq_sum = target.pow(2).sum(dim=0)
    r2_stream = compute_r2_streaming(ss_res, tgt_sum, tgt_sq_sum, n=target.shape[0])
    assert r2_stream == pytest.approx(r2_batch, abs=1e-5)


# ---------------------------------------------------------------------------
# save / load round-trip
# ---------------------------------------------------------------------------

def test_save_load_round_trip(tmp_path: Path) -> None:
    """Trained model state survives save_closure_predictors → load_closure_predictors."""
    torch.manual_seed(0)
    model = build_closure_predictor(input_dim=8, output_dim=4, hidden_dim=16, num_layers=2)

    # Pretend it was trained
    info = {
        "model": model,
        "input_dim": 8,
        "output_dim": 4,
        "loss_type": "mse",
        "hidden_dim": 16,
        "num_layers": 2,
        "train_losses": [1.0, 0.5, 0.3],
        "val_losses":   [1.2, 0.7, 0.4],
        "val_r2_final": 0.6,
    }
    save_closure_predictors(
        {"autonomous": info},
        save_path=tmp_path / "predictors.pt",
        target_variance=2.5,
    )

    loaded = load_closure_predictors(tmp_path / "predictors.pt", device="cpu")
    assert "__meta__" in loaded
    assert loaded["__meta__"]["target_variance"] == pytest.approx(2.5)
    assert "autonomous" in loaded
    a = loaded["autonomous"]
    assert a["input_dim"] == 8 and a["output_dim"] == 4
    assert a["hidden_dim"] == 16 and a["num_layers"] == 2
    assert a["loss_type"] == "mse"
    assert a["train_losses"] == [1.0, 0.5, 0.3]
    assert a["val_losses"] == [1.2, 0.7, 0.4]
    assert a["val_r2_final"] == pytest.approx(0.6)

    # The loaded model produces identical output on the same input.
    x = torch.randn(3, 8)
    with torch.no_grad():
        y_orig = model(x)
        y_load = a["model"](x)
    assert torch.allclose(y_orig, y_load, atol=1e-6)


def test_load_closure_predictors_missing_file_returns_none(tmp_path: Path) -> None:
    out = load_closure_predictors(tmp_path / "does_not_exist.pt", device="cpu")
    assert out is None
