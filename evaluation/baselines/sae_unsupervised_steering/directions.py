"""Build h-space steering directions from SAE latent indices.

For ``direction_source="decoder"``, the direction for latent ``i`` is the
i-th column of the SAE decoder, ``W_dec[:, i]``. Decoder columns are unit-
norm by training construction and represent what the residual stream
contains when feature ``i`` fires with unit activation. This is the
standard choice (Templeton et al. 2024, Marks et al. 2024).

For ``direction_source="encoder"``, we use the i-th row of the encoder,
``W_enc[i]`` (the gradient of the linear pre-activation w.r.t. ``h``).
Encoder rows are detection-tuned and not unit-norm in this SAE — they
work as a fallback when no decoder is available.

Multi-latent composition: per-latent unit directions are summed and the
sum is unit-normalized. This makes ``strength`` scale-invariant and
prevents one high-norm column from dominating multi-latent composites.
"""
from __future__ import annotations

import torch


_NORM_EPS = 1e-10


def get_latent_direction(
    *,
    latent_id: int,
    decoder_weight: torch.Tensor | None,
    encoder_weight: torch.Tensor,
    direction_source: str,
) -> torch.Tensor:
    """Return a unit-norm h-space direction for one SAE latent.

    Parameters
    ----------
    latent_id : int
        SAE feature index in [0, d_sae).
    decoder_weight : (d_in, d_sae) or None
        SAE decoder weight matrix; required when ``direction_source="decoder"``.
    encoder_weight : (d_sae, d_in)
        SAE encoder weight matrix; always required (used for fallback or as
        the primary direction).
    direction_source : "decoder" or "encoder"

    Returns
    -------
    Tensor of shape (d_in,), unit-norm.

    Raises
    ------
    ValueError
        If ``latent_id`` is out of range, if direction_source is invalid, if
        decoder is requested but None, or if the chosen vector has near-zero
        norm (dead latent).
    """
    d_sae = encoder_weight.shape[0]
    if not (0 <= latent_id < d_sae):
        raise ValueError(
            f"latent_id {latent_id} out of range [0, {d_sae})"
        )

    if direction_source == "decoder":
        if decoder_weight is None:
            raise ValueError(
                "direction_source='decoder' but decoder_weight is None. "
                "Pass --direction-source encoder or load a SAE checkpoint "
                "that includes decoder.weight."
            )
        if decoder_weight.shape[1] != d_sae:
            raise ValueError(
                f"decoder_weight shape {tuple(decoder_weight.shape)} does "
                f"not match encoder d_sae={d_sae}"
            )
        v = decoder_weight[:, latent_id]
    elif direction_source == "encoder":
        v = encoder_weight[latent_id]
    else:
        raise ValueError(
            f"direction_source must be 'decoder' or 'encoder', "
            f"got {direction_source!r}"
        )

    n = v.norm()
    if float(n) < _NORM_EPS:
        raise ValueError(
            f"Latent {latent_id} has near-zero norm in {direction_source} "
            f"({float(n):.2e}); refusing to steer with a dead direction."
        )
    return v / n


def build_composite_direction(
    *,
    latent_ids: list[int],
    decoder_weight: torch.Tensor | None,
    encoder_weight: torch.Tensor,
    direction_source: str,
) -> torch.Tensor:
    """Sum of unit-normed per-latent directions, then unit-normalized.

    Returns ``(d_in,)``. Raises ValueError if ``latent_ids`` is empty or if
    the composite happens to have near-zero norm (cancellation across
    latents).
    """
    if not latent_ids:
        raise ValueError("latent_ids must be a non-empty list")
    dirs = [
        get_latent_direction(
            latent_id=lid,
            decoder_weight=decoder_weight,
            encoder_weight=encoder_weight,
            direction_source=direction_source,
        )
        for lid in latent_ids
    ]
    s = torch.stack(dirs).sum(dim=0)  # (d_in,)
    n = s.norm()
    if float(n) < _NORM_EPS:
        raise ValueError(
            f"Composite direction has near-zero norm ({float(n):.2e}) for "
            f"latents {latent_ids}; check for cancelling/anti-aligned features."
        )
    return s / n
