"""Tests for encoder modules."""

import pytest
import torch

from ret.config import EncoderConfig
from ret.models.encoders import (
    MLPEncoder, CausalTransformerEncoder,
    SparseWrapper, VQWrapper,
    build_encoder,
)


D_H = 768
B, T = 2, 64


@pytest.fixture
def mlp_config():
    return EncoderConfig(type="mlp", z_dim=64, mlp_hidden_dims=[512, 256])


@pytest.fixture
def transformer_config():
    return EncoderConfig(
        type="transformer", z_dim=64, num_heads=4, num_layers=1
    )


@pytest.fixture
def sparse_config():
    return EncoderConfig(
        type="mlp", z_dim=64, mlp_hidden_dims=[512, 256],
        z_regularization="sparse", lambda_sparse=1e-3,
    )


@pytest.fixture
def vq_config():
    return EncoderConfig(
        type="mlp", z_dim=64, mlp_hidden_dims=[512, 256],
        z_regularization="vq", codebook_size=128, commitment_weight=0.25,
    )


def test_mlp_encoder_shape(mlp_config):
    encoder = MLPEncoder(D_H, mlp_config).cuda()
    h = torch.randn(B, T, D_H, device="cuda")
    z, aux = encoder(h)
    assert z.shape == (B, T, 64)
    assert aux == {}


def test_transformer_encoder_shape(transformer_config):
    encoder = CausalTransformerEncoder(D_H, transformer_config).cuda()
    h = torch.randn(B, T, D_H, device="cuda")
    z, aux = encoder(h)
    assert z.shape == (B, T, 64)
    assert aux == {}


def test_build_encoder_factory(mlp_config, transformer_config, sparse_config, vq_config):
    mlp = build_encoder(D_H, mlp_config)
    assert isinstance(mlp, MLPEncoder)

    transformer = build_encoder(D_H, transformer_config)
    assert isinstance(transformer, CausalTransformerEncoder)

    sparse = build_encoder(D_H, sparse_config)
    assert isinstance(sparse, SparseWrapper)
    assert isinstance(sparse.base_encoder, MLPEncoder)

    vq = build_encoder(D_H, vq_config)
    assert isinstance(vq, VQWrapper)
    assert isinstance(vq.base_encoder, MLPEncoder)


def test_causal_mask_gradient():
    """Verify d z_t / d h_{t+1} == 0 for causal transformer encoder."""
    config = EncoderConfig(type="transformer", z_dim=32, num_heads=4, num_layers=1)
    encoder = CausalTransformerEncoder(D_H, config).cuda()
    encoder.eval()

    h = torch.randn(1, 8, D_H, device="cuda", requires_grad=True)
    z, _aux = encoder(h)

    # Check: gradient of z[0, t, :] w.r.t. h[0, t+1:, :] should be zero
    t = 3
    z_t = z[0, t].sum()
    z_t.backward()

    grad = h.grad[0]  # (T, D_H)
    # Positions after t should have zero gradient
    future_grad_norm = grad[t + 1 :].abs().max().item()
    assert future_grad_norm < 1e-5, f"Causal violation: future grad norm = {future_grad_norm}"


def test_mlp_encoder_pointwise(mlp_config):
    """MLP encoder should be pointwise: changing h[t'] shouldn't affect z[t]."""
    encoder = MLPEncoder(D_H, mlp_config).cuda()
    encoder.eval()

    h = torch.randn(1, 8, D_H, device="cuda")
    z1, _ = encoder(h)

    h2 = h.clone()
    h2[0, 5] = torch.randn(D_H, device="cuda")
    z2, _ = encoder(h2)

    # z at position 3 should be unchanged
    assert torch.allclose(z1[0, 3], z2[0, 3], atol=1e-6)


# --- Sparse wrapper tests ---

def test_sparse_encoder_shape(sparse_config):
    encoder = build_encoder(D_H, sparse_config).cuda()
    h = torch.randn(B, T, D_H, device="cuda")
    z, aux = encoder(h)
    assert z.shape == (B, T, 64)
    assert "aux_loss" in aux
    assert "sparse_loss" in aux


def test_sparse_loss_nonnegative(sparse_config):
    encoder = build_encoder(D_H, sparse_config).cuda()
    h = torch.randn(B, T, D_H, device="cuda")
    _, aux = encoder(h)
    assert aux["sparse_loss"].item() >= 0


def test_sparse_encoder_gradient_flows(sparse_config):
    """Verify gradients flow through sparse encoder aux_loss."""
    encoder = build_encoder(D_H, sparse_config).cuda()
    h = torch.randn(B, T, D_H, device="cuda")
    z, aux = encoder(h)
    loss = z.sum() + aux["aux_loss"]
    loss.backward()
    for p in encoder.parameters():
        assert p.grad is not None


# --- VQ wrapper tests ---

def test_vq_encoder_shape(vq_config):
    encoder = build_encoder(D_H, vq_config).cuda()
    h = torch.randn(B, T, D_H, device="cuda")
    z, aux = encoder(h)
    assert z.shape == (B, T, 64)
    assert "aux_loss" in aux
    assert "vq_loss" in aux
    assert "z_pre" in aux
    assert "codebook_perplexity" in aux
    assert aux["z_pre"].shape == z.shape


def test_vq_codebook_membership(vq_config):
    """Quantized output should match nearest codebook entry."""
    encoder = build_encoder(D_H, vq_config).cuda()
    encoder.eval()
    h = torch.randn(1, 4, D_H, device="cuda")
    with torch.no_grad():
        z_q, aux = encoder(h)
        z_pre = aux["z_pre"]

    # For each token, find nearest codebook entry and verify z_q matches
    codebook = encoder.codebook.weight  # (K, d_z)
    flat_pre = z_pre.reshape(-1, z_pre.shape[-1])
    flat_q = z_q.reshape(-1, z_q.shape[-1])
    dists = torch.cdist(flat_pre, codebook)
    nearest_idx = dists.argmin(dim=-1)
    expected = codebook[nearest_idx]
    assert torch.allclose(flat_q, expected, atol=1e-5)


def test_vq_straight_through_gradient(vq_config):
    """Verify gradients flow through VQ encoder via straight-through estimator."""
    encoder = build_encoder(D_H, vq_config).cuda()
    h = torch.randn(B, T, D_H, device="cuda")
    z, aux = encoder(h)
    loss = z.sum() + aux["aux_loss"]
    loss.backward()
    # Check that base encoder MLP weights received gradients (via straight-through)
    assert encoder.base_encoder.net[0].weight.grad is not None
    assert encoder.base_encoder.net[0].weight.grad.abs().sum() > 0


# --- Cross-product tests: transformer + sparse/vq ---

def test_transformer_sparse_shape():
    """Test transformer encoder with sparse wrapper."""
    config = EncoderConfig(
        type="transformer", z_dim=64, num_heads=4, num_layers=1,
        z_regularization="sparse", lambda_sparse=1e-3,
    )
    encoder = build_encoder(D_H, config).cuda()
    assert isinstance(encoder, SparseWrapper)
    assert isinstance(encoder.base_encoder, CausalTransformerEncoder)

    h = torch.randn(B, T, D_H, device="cuda")
    z, aux = encoder(h)
    assert z.shape == (B, T, 64)
    assert "sparse_loss" in aux
    assert aux["sparse_loss"].item() >= 0


def test_transformer_vq_shape():
    """Test transformer encoder with VQ wrapper."""
    config = EncoderConfig(
        type="transformer", z_dim=64, num_heads=4, num_layers=1,
        z_regularization="vq", codebook_size=128, commitment_weight=0.25,
    )
    encoder = build_encoder(D_H, config).cuda()
    assert isinstance(encoder, VQWrapper)
    assert isinstance(encoder.base_encoder, CausalTransformerEncoder)

    h = torch.randn(B, T, D_H, device="cuda")
    z, aux = encoder(h)
    assert z.shape == (B, T, 64)
    assert "vq_loss" in aux
    assert "codebook_perplexity" in aux
