"""Tests for predictor (including TokenProjector)."""

import pytest
import torch

from ret.config import PredictorConfig
from ret.models.predictor import Predictor
from ret.models.predictor import TokenProjector


D_Z = 64
D_H = 768
B, T = 2, 63  # T-1 for temporal pairs


@pytest.fixture
def autonomous_config():
    return PredictorConfig(mode="autonomous", hidden_dim=256, num_layers=2)


@pytest.fixture
def controlled_config():
    return PredictorConfig(mode="controlled", control_dim=32, hidden_dim=256, num_layers=2)


def test_autonomous_predictor_shape(autonomous_config):
    pred = Predictor(D_Z, autonomous_config).cuda()
    z_t = torch.randn(B, T, D_Z, device="cuda")
    z_hat = pred(z_t)
    assert z_hat.shape == (B, T, D_Z)


def test_controlled_predictor_shape(controlled_config):
    pred = Predictor(D_Z, controlled_config).cuda()
    z_t = torch.randn(B, T, D_Z, device="cuda")
    u_next = torch.randn(B, T, 32, device="cuda")
    z_hat = pred(z_t, u_next)
    assert z_hat.shape == (B, T, D_Z)


def test_controlled_predictor_requires_u(controlled_config):
    pred = Predictor(D_Z, controlled_config).cuda()
    z_t = torch.randn(B, T, D_Z, device="cuda")
    with pytest.raises(AssertionError):
        pred(z_t)


def test_token_projector_shape(controlled_config):
    proj = TokenProjector(D_H, controlled_config).cuda()
    emb = torch.randn(B, T + 1, D_H, device="cuda")
    u = proj(emb)
    assert u.shape == (B, T + 1, 32)


def test_predictor_gradient_flow(autonomous_config):
    """Verify gradients flow through the predictor."""
    pred = Predictor(D_Z, autonomous_config).cuda()
    z_t = torch.randn(B, T, D_Z, device="cuda", requires_grad=True)
    z_hat = pred(z_t)
    loss = z_hat.sum()
    loss.backward()
    assert z_t.grad is not None
    assert z_t.grad.abs().sum() > 0
