"""Data collection utilities for closure evaluation."""
from __future__ import annotations

import torch
import torch.nn as nn


def extract_sequence_data(
    text: str,
    extractor,
    encoder: nn.Module,
    token_projector: nn.Module | None,
    device: str,
) -> dict[str, torch.Tensor | list[str]]:
    """Extract representations for a single text sequence.

    Returns dict with keys: h, z, h_t, h_next, z_t, z_next, u_next (if controlled),
    token_emb, tokens. All tensors are 2D (T or T-1 along dim 0), on the given device.
    """
    tokenizer = extractor.tokenizer
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)

    with torch.no_grad():
        h = extractor.extract(input_ids)  # (1, T, d_h)
        z, aux = encoder(h)  # (1, T, d_z)
        z = aux.get("z_pre", z)
        token_emb = extractor.get_token_embeddings(input_ids)  # (1, T, d_h)

    result = {
        "h": h[0],             # (T, d_h)
        "z": z[0],             # (T, d_z)
        "h_t": h[0, :-1],     # (T-1, d_h)
        "h_next": h[0, 1:],   # (T-1, d_h)
        "z_t": z[0, :-1],     # (T-1, d_z)
        "z_next": z[0, 1:],   # (T-1, d_z)
        "token_emb": token_emb[0],  # (T, d_h)
    }

    if token_projector is not None:
        with torch.no_grad():
            u = token_projector(token_emb)  # (1, T, d_u)
        result["u_next"] = u[0, 1:]  # (T-1, d_u)

    token_ids = input_ids[0].cpu().tolist()
    result["tokens"] = [tokenizer.decode([tid]) for tid in token_ids]

    return result


def collect_val_data(
    extractor,
    val_loader,
    val_batches: int,
    device: str,
    encoder: nn.Module | None = None,
    scaffold_prefix_len: int = 0,
) -> dict[str, torch.Tensor]:
    """Pre-collect hidden states and token embeddings from val split for closure eval.

    Mode-agnostic: returns raw h and token_emb. When an encoder is provided,
    also returns z_t and z_next encoded from full sequences (preserving causal
    context for CausalTransformerEncoder).

    Returns dict with keys: h_t, h_next, token_emb_next (always),
    plus z_t, z_next when encoder is provided. All CPU tensors.
    """
    all_h_t, all_h_next, all_te_next = [], [], []
    all_z_t, all_z_next = [], []

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= val_batches:
                break
            input_ids = batch["input_ids"]
            attention_mask = batch.get("attention_mask")

            # Use cached hidden states if available, otherwise extract
            if "hidden_states" in batch:
                h = batch["hidden_states"].to(device)
            else:
                h = extractor.extract(input_ids, attention_mask=attention_mask)
            token_emb = extractor.get_token_embeddings(input_ids)

            if attention_mask is not None:
                # Move mask to same device as h (needed for multi-GPU / device_map)
                attention_mask = attention_mask.to(h.device)
                pair_mask = (attention_mask[:, :-1] * attention_mask[:, 1:]).bool()
                if scaffold_prefix_len > 0:
                    pair_mask[:, :scaffold_prefix_len] = False
                all_h_t.append(h[:, :-1][pair_mask].cpu())
                all_h_next.append(h[:, 1:][pair_mask].cpu())
                all_te_next.append(token_emb[:, 1:][pair_mask].cpu())

                if encoder is not None:
                    z, _aux = encoder(h, attention_mask=attention_mask)
                    all_z_t.append(z[:, :-1][pair_mask].cpu())
                    all_z_next.append(z[:, 1:][pair_mask].cpu())
            else:
                B, T, d = h.shape
                if scaffold_prefix_len > 0:
                    # Build a content-only mask excluding scaffold prefix
                    pair_mask = torch.ones(B, T - 1, dtype=torch.bool)
                    pair_mask[:, :scaffold_prefix_len] = False
                    all_h_t.append(h[:, :-1][pair_mask].cpu())
                    all_h_next.append(h[:, 1:][pair_mask].cpu())
                    all_te_next.append(token_emb[:, 1:][pair_mask].cpu())
                else:
                    all_h_t.append(h[:, :-1].reshape(-1, h.shape[-1]).cpu())
                    all_h_next.append(h[:, 1:].reshape(-1, h.shape[-1]).cpu())
                    all_te_next.append(token_emb[:, 1:].reshape(-1, token_emb.shape[-1]).cpu())

                if encoder is not None:
                    z, _aux = encoder(h)  # (B, T, d_z) — full causal context
                    if scaffold_prefix_len > 0:
                        all_z_t.append(z[:, :-1][pair_mask].cpu())
                        all_z_next.append(z[:, 1:][pair_mask].cpu())
                    else:
                        all_z_t.append(z[:, :-1].reshape(-1, z.shape[-1]).cpu())
                        all_z_next.append(z[:, 1:].reshape(-1, z.shape[-1]).cpu())

    result = {
        "h_t": torch.cat(all_h_t),
        "h_next": torch.cat(all_h_next),
        "token_emb_next": torch.cat(all_te_next),
    }
    if encoder is not None:
        result["z_t"] = torch.cat(all_z_t)
        result["z_next"] = torch.cat(all_z_next)
    return result


def encode_in_chunks(
    encoder: nn.Module,
    h_flat: torch.Tensor,
    device: str,
    batch_size: int = 4096,
) -> torch.Tensor:
    """Batch-encode flat h tensors through a frozen encoder, returning CPU tensor.

    WARNING: Encodes each token independently (T=1). Only correct for pointwise
    encoders (MLPEncoder). Do NOT use with CausalTransformerEncoder — use
    collect_val_data(encoder=...) instead to preserve causal context.

    Args:
        encoder: frozen RET encoder (must be in eval mode)
        h_flat: (N, d_h) hidden states on CPU
        device: GPU device for encoding
        batch_size: number of samples per forward pass

    Returns:
        (N, d_z) encoded representations on CPU
    """
    chunks = []
    with torch.no_grad():
        for i in range(0, h_flat.shape[0], batch_size):
            h_batch = h_flat[i:i + batch_size].to(device)
            # Encoder expects (B, T, d_h); use T=1 for pointwise encoding
            z, _aux = encoder(h_batch.unsqueeze(1))  # (batch, 1, d_z)
            chunks.append(z.squeeze(1).cpu())
    return torch.cat(chunks, dim=0)
