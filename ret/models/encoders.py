from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ret.config import EncoderConfig


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def _apply_rotary_emb(
    q: torch.Tensor, k: torch.Tensor,
    cos: torch.Tensor, sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # q, k: (B, nhead, T, head_dim); cos, sin: (T, head_dim)
    cos = cos.unsqueeze(0).unsqueeze(0)   # (1, 1, T, head_dim)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin


class MLPEncoder(nn.Module):
    """Pointwise MLP encoder: z_t = MLP(h_t).

    Architecture: d_h -> 512 -> LayerNorm -> GELU -> 256 -> LayerNorm -> GELU -> d_z
    """

    def __init__(self, d_h: int, config: EncoderConfig):
        super().__init__()
        dims = [d_h] + config.mlp_hidden_dims + [config.z_dim]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.LayerNorm(dims[i + 1]))
                layers.append(nn.GELU())
        self.net = nn.Sequential(*layers)

    def forward(self, h: torch.Tensor, **kwargs) -> tuple[torch.Tensor, dict]:
        """
        Args:
            h: (B, T, d_h)
            **kwargs: ignored (accepted for API compatibility with CausalTransformerEncoder)
        Returns:
            z: (B, T, d_z)
            aux: empty dict
        """
        return self.net(h), {}


class CausalSelfAttentionWithRoPE(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % nhead == 0
        self.nhead = nhead
        self.head_dim = d_model // nhead

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop = nn.Dropout(dropout)

        # RoPE: precompute inverse frequencies (not a learned parameter)
        inv_freq = 1.0 / (10000.0 ** (
            torch.arange(0, self.head_dim, 2).float() / self.head_dim
        ))
        self.register_buffer("inv_freq", inv_freq)

    def _rope(self, T: int, device: torch.device):
        positions = torch.arange(T, device=device).float()
        freqs = torch.outer(positions, self.inv_freq)   # (T, head_dim/2)
        emb = torch.cat([freqs, freqs], dim=-1)         # (T, head_dim)
        return emb.cos(), emb.sin()

    def forward(
        self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)          # each (B, T, C)

        # Reshape to (B, nhead, T, head_dim)
        def split_heads(t):
            return t.view(B, T, self.nhead, self.head_dim).transpose(1, 2)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)

        cos, sin = self._rope(T, x.device)
        q, k = _apply_rotary_emb(q, k, cos, sin)

        if key_padding_mask is not None:
            # Combine causal mask with padding mask
            causal = torch.ones(T, T, device=x.device, dtype=x.dtype).tril()
            # key_padding_mask: (B, T) with 1=valid, 0=padding
            pad = key_padding_mask.to(device=x.device, dtype=x.dtype)[:, None, None, :]  # (B, 1, 1, T)
            attn_mask = causal[None, None, :, :] * pad   # (B, 1, T, T)
            # Convert to additive mask: 0 -> -inf
            attn_mask = attn_mask.masked_fill(attn_mask == 0, float("-inf"))
            attn_mask = attn_mask.masked_fill(attn_mask == 1, 0.0)
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, is_causal=False,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
        else:
            out = F.scaled_dot_product_attention(
                q, k, v, is_causal=True,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


class CausalTransformerLayerWithRoPE(nn.Module):
    """Pre-norm transformer layer with RoPE causal self-attention."""

    def __init__(self, d_model: int, nhead: int, ff_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttentionWithRoPE(d_model, nhead, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.drop(self.attn(self.norm1(x), key_padding_mask=key_padding_mask))
        x = x + self.drop(self.ff(self.norm2(x)))
        return x


class CausalTransformerEncoder(nn.Module):
    """Causal transformer encoder with RoPE positional encoding.

    Two projection modes controlled by ``config.proj_position``:
      - "begin": project d_h -> d_z first, run transformer at d_z.
      - "end":   run transformer at d_h, then project d_h -> d_z.
    """

    def __init__(self, d_h: int, config: EncoderConfig, max_len: int = 2048):
        super().__init__()
        d_z = config.z_dim
        self.proj_position = config.proj_position

        if self.proj_position == "begin":
            self.input_proj = nn.Linear(d_h, d_z)
            d_model = d_z
        else:
            d_model = d_h
            self.output_proj = nn.Linear(d_h, d_z)

        ff_dim = config.ff_dim if config.ff_dim is not None else 4 * d_model

        self.layers = nn.ModuleList([
            CausalTransformerLayerWithRoPE(
                d_model=d_model,
                nhead=config.num_heads,
                ff_dim=ff_dim,
                dropout=config.dropout,
            )
            for _ in range(config.num_layers)
        ])
        self.output_norm = nn.LayerNorm(d_z)

    def forward(
        self, h: torch.Tensor, attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """
        Args:
            h: (B, T, d_h)
            attention_mask: (B, T) with 1=valid, 0=padding. Optional.
        Returns:
            z: (B, T, d_z)
            aux: empty dict
        """
        if self.proj_position == "begin":
            z = self.input_proj(h)
            for layer in self.layers:
                z = layer(z, key_padding_mask=attention_mask)
        else:
            z = h
            for layer in self.layers:
                z = layer(z, key_padding_mask=attention_mask)
            z = self.output_proj(z)
        return self.output_norm(z), {}


class SparseWrapper(nn.Module):
    """Wrapper that adds L1 sparsity penalty to any base encoder's output."""

    def __init__(self, base_encoder: nn.Module, lambda_sparse: float):
        super().__init__()
        self.base_encoder = base_encoder
        self.lambda_sparse = lambda_sparse

    def forward(self, h: torch.Tensor, **kwargs) -> tuple[torch.Tensor, dict]:
        z, aux = self.base_encoder(h, **kwargs)
        sparse_loss = self.lambda_sparse * z.abs().mean()
        aux["aux_loss"] = aux.get("aux_loss", 0) + sparse_loss
        aux["sparse_loss"] = sparse_loss
        return z, aux


class VQWrapper(nn.Module):
    """Wrapper that adds VQ-VAE vector quantization to any base encoder's output.

    Quantizes the base encoder's continuous z_pre to nearest codebook entry
    using straight-through estimation.
    """

    def __init__(self, base_encoder: nn.Module, z_dim: int,
                 codebook_size: int, commitment_weight: float):
        super().__init__()
        self.base_encoder = base_encoder
        self.codebook = nn.Embedding(codebook_size, z_dim)
        self.codebook.weight.data.uniform_(-1.0 / codebook_size, 1.0 / codebook_size)
        self.commitment_weight = commitment_weight

    def forward(self, h: torch.Tensor, **kwargs) -> tuple[torch.Tensor, dict]:
        z_pre, aux = self.base_encoder(h, **kwargs)

        # Flatten for distance computation
        flat = z_pre.reshape(-1, z_pre.shape[-1])  # (N, d_z)
        codebook = self.codebook.weight  # (K, d_z)

        # Squared distances: ||z - e||^2 = ||z||^2 - 2*z@e^T + ||e||^2
        dists = (
            flat.pow(2).sum(dim=-1, keepdim=True)
            - 2 * flat @ codebook.t()
            + codebook.pow(2).sum(dim=-1, keepdim=True).t()
        )  # (N, K)

        indices = dists.argmin(dim=-1)  # (N,)
        z_q_quantized = self.codebook(indices).reshape_as(z_pre)

        # Losses
        codebook_loss = F.mse_loss(z_q_quantized, z_pre.detach())
        commitment_loss = F.mse_loss(z_pre, z_q_quantized.detach())
        vq_loss = codebook_loss + self.commitment_weight * commitment_loss

        # Straight-through estimator
        z_q = z_pre + (z_q_quantized - z_pre).detach()

        # Codebook perplexity (usage metric)
        encodings = F.one_hot(indices, codebook.shape[0]).float()
        avg_probs = encodings.mean(dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        aux["aux_loss"] = aux.get("aux_loss", 0) + vq_loss
        aux["vq_loss"] = vq_loss
        aux["z_pre"] = z_pre
        aux["codebook_indices"] = indices.reshape(z_pre.shape[:-1])  # (B, T)
        aux["codebook_perplexity"] = perplexity.item()
        return z_q, aux


class MultiLayerEncoder(nn.Module):
    """Wraps a base encoder with a learned bottleneck down-projection.

    Used when the encoder input is the concatenation of hidden states from
    multiple LLM layers: concat_dim → bottleneck_dim → out_dim → base_encoder.
    """

    def __init__(
        self,
        concat_dim: int,
        bottleneck_dim: int,
        out_dim: int,
        base_encoder: nn.Module,
    ):
        super().__init__()
        self.down_proj = nn.Sequential(
            nn.Linear(concat_dim, bottleneck_dim),
            nn.GELU(),
            nn.Linear(bottleneck_dim, out_dim),
            nn.LayerNorm(out_dim),
        )
        self.base_encoder = base_encoder

    def forward(self, h: torch.Tensor, **kwargs) -> tuple[torch.Tensor, dict]:
        h_proj = self.down_proj(h)
        return self.base_encoder(h_proj, **kwargs)


def build_encoder(
    d_h: int,
    config: EncoderConfig,
    concat_dim: int | None = None,
) -> nn.Module:
    """Factory function for building encoders.

    Args:
        d_h: single-layer hidden dimension (used as base encoder input dim).
        config: encoder configuration.
        concat_dim: total concatenated dimension when using multi-layer input.
            When set together with ``config.down_proj_bottleneck``, the encoder
            is wrapped with a bottleneck down-projection.
    """
    # Determine the base encoder's input dimension
    effective_d_h = config.down_proj_out_dim or d_h

    if config.type == "mlp":
        encoder = MLPEncoder(effective_d_h, config)
    elif config.type == "transformer":
        encoder = CausalTransformerEncoder(effective_d_h, config)
    else:
        raise ValueError(f"Unknown encoder type: {config.type}")

    if config.z_regularization == "sparse":
        encoder = SparseWrapper(encoder, config.lambda_sparse)
    elif config.z_regularization == "vq":
        encoder = VQWrapper(encoder, config.z_dim, config.codebook_size, config.commitment_weight)

    # Wrap with multi-layer bottleneck if configured
    if concat_dim is not None and config.down_proj_bottleneck is not None:
        encoder = MultiLayerEncoder(
            concat_dim=concat_dim,
            bottleneck_dim=config.down_proj_bottleneck,
            out_dim=effective_d_h,
            base_encoder=encoder,
        )

    return encoder
