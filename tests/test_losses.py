"""Tests for loss functions."""

import pytest
import torch

from ret.losses.prediction import (
    cosine_prediction_loss, mse_prediction_loss, l1_prediction_loss,
    masked_prediction_loss,
)
from ret.losses.vicreg import variance_loss, covariance_loss, vicreg_loss
from ret.losses.sigreg import SIGReg, sigreg_loss


B, T, D_Z = 4, 63, 64


def test_cosine_loss_range():
    """Cosine loss should be in [0, 2]."""
    pred = torch.randn(B, T, D_Z)
    target = torch.randn(B, T, D_Z)
    loss = cosine_prediction_loss(pred, target)
    assert 0 <= loss.item() <= 2.0


def test_cosine_loss_identical():
    """Cosine loss should be ~0 for identical vectors."""
    x = torch.randn(B, T, D_Z)
    loss = cosine_prediction_loss(x, x.clone())
    assert loss.item() < 1e-5


def test_cosine_loss_opposite():
    """Cosine loss should be ~2 for opposite vectors."""
    x = torch.randn(B, T, D_Z)
    loss = cosine_prediction_loss(x, -x)
    assert loss.item() > 1.9


def test_mse_loss_zero():
    x = torch.randn(B, T, D_Z)
    loss = mse_prediction_loss(x, x.clone())
    assert loss.item() < 1e-10


def test_l1_loss_zero():
    """L1 loss should be ~0 for identical tensors."""
    x = torch.randn(B, T, D_Z)
    loss = l1_prediction_loss(x, x.clone())
    assert loss.item() < 1e-10


def test_l1_loss_positive():
    """L1 loss should be > 0 for different tensors."""
    pred = torch.randn(B, T, D_Z)
    target = torch.randn(B, T, D_Z)
    loss = l1_prediction_loss(pred, target)
    assert loss.item() > 0


def test_variance_loss_penalizes_collapse():
    """Variance loss should be high for collapsed (constant) representations."""
    z_collapsed = torch.ones(100, D_Z) * 5.0
    loss = variance_loss(z_collapsed)
    assert loss.item() > 0.5  # Should be ~1.0

    z_spread = torch.randn(100, D_Z) * 2.0
    loss_spread = variance_loss(z_spread)
    assert loss_spread.item() < loss.item()


def test_covariance_loss_diagonal():
    """Covariance loss should be ~0 for uncorrelated dimensions."""
    # Independent standard normal: covariance should be ~identity
    z = torch.randn(10000, D_Z)
    loss = covariance_loss(z)
    # With many samples, off-diagonal covariance should be small
    assert loss.item() < 0.01


def test_vicreg_combined():
    """Test combined VICReg loss returns correct structure."""
    z = torch.randn(B, T + 1, D_Z)
    total, metrics = vicreg_loss(z)
    assert isinstance(total, torch.Tensor)
    assert total.ndim == 0
    assert "vicreg/variance_loss" in metrics
    assert "vicreg/covariance_loss" in metrics
    assert "vicreg/mean_std" in metrics


def test_vicreg_collapse_detection():
    """VICReg should give higher loss for collapsed representations."""
    z_good = torch.randn(B, T + 1, D_Z)
    z_bad = torch.ones(B, T + 1, D_Z)

    loss_good, _ = vicreg_loss(z_good)
    loss_bad, _ = vicreg_loss(z_bad)
    assert loss_bad.item() > loss_good.item()


# --- SIGReg tests ---

def test_sigreg_forward_scalar():
    """SIGReg forward returns a non-negative scalar."""
    m = SIGReg(knots=17, num_proj=512)
    stat = m(torch.randn(200, D_Z))
    assert stat.ndim == 0
    assert stat.item() >= 0


def test_sigreg_gaussian_low_loss():
    """SIGReg statistic should be lower for N(0,I) than for shifted/scaled samples."""
    m = SIGReg(knots=17, num_proj=1024)
    z_good = torch.randn(5000, D_Z)
    z_bad = torch.randn(5000, D_Z) * 5.0 + 3.0
    assert m(z_good).item() < m(z_bad).item()


def test_sigreg_collapse_detection():
    """SIGReg should give higher loss for collapsed representations."""
    m = SIGReg(knots=17, num_proj=1024)
    z_good = torch.randn(B, T + 1, D_Z)
    z_bad = torch.ones(B, T + 1, D_Z)
    loss_good, _ = sigreg_loss(z_good, m)
    loss_bad, _ = sigreg_loss(z_bad, m)
    assert loss_bad.item() > loss_good.item()


def test_sigreg_combined_structure():
    """sigreg_loss returns correct (scalar, metrics_dict) structure."""
    m = SIGReg(knots=17, num_proj=512)
    z = torch.randn(B, T + 1, D_Z)
    total, metrics = sigreg_loss(z, m)
    assert isinstance(total, torch.Tensor)
    assert total.ndim == 0
    for key in ["sigreg/statistic", "sigreg/total", "sigreg/mean_std", "reg/mean_std"]:
        assert key in metrics


# --- masked_prediction_loss tests ---

def test_masked_prediction_loss_all_types():
    """Masked prediction loss should be > 0 for all loss types."""
    pred = torch.randn(B, T, D_Z)
    target = torch.randn(B, T, D_Z)
    mask = torch.ones(B, T)

    for loss_type in ["mse", "l1", "cosine"]:
        loss = masked_prediction_loss(pred, target, mask, loss_type)
        assert loss.item() > 0, f"{loss_type} loss should be > 0"


def test_masked_prediction_loss_ignores_padding():
    """Masked prediction loss should differ when padding positions are excluded."""
    pred = torch.randn(B, T, D_Z)
    target = torch.randn(B, T, D_Z)

    full_mask = torch.ones(B, T)
    partial_mask = torch.ones(B, T)
    partial_mask[:, -3:] = 0  # mask out last 3 positions

    for loss_type in ["mse", "l1", "cosine"]:
        loss_full = masked_prediction_loss(pred, target, full_mask, loss_type)
        loss_partial = masked_prediction_loss(pred, target, partial_mask, loss_type)
        assert loss_full.item() != loss_partial.item(), (
            f"{loss_type}: full vs partial mask should give different losses"
        )


def test_masked_prediction_loss_zero_for_identical():
    """Masked MSE/L1 loss should be ~0 for identical tensors."""
    x = torch.randn(B, T, D_Z)
    mask = torch.ones(B, T)
    assert masked_prediction_loss(x, x.clone(), mask, "mse").item() < 1e-10
    assert masked_prediction_loss(x, x.clone(), mask, "l1").item() < 1e-10
    assert masked_prediction_loss(x, x.clone(), mask, "cosine").item() < 1e-5
