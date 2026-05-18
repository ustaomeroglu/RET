"""Closed-form SAE steering: scale a precomputed unit direction by
``strength * ||h_t||`` and apply with a sign.

No autograd, no iterative refinement — for an additive steering vector
the gradient of the linear pre-activation w.r.t. ``h`` is constant in
``h``, so a single closed-form step is exactly equivalent to many small
re-encoded ones in this regime. (Iterative SAE steering through a TopK
non-linearity is not standard practice in published work.)
"""
from __future__ import annotations

import torch


class SAESteerer:
    """Compute ``δh = sign * strength * ||h_t|| * unit_dir``.

    Parameters
    ----------
    strength : float
        Step size as a fraction of ``||h_t||``. Parity with the RET
        ``GradientSteerer`` knob: same numeric range works comparably.
    d_h : int
        Hidden dimension (kept for API symmetry / sanity-checking inputs).
    device : str
    """

    def __init__(self, *, strength: float, d_h: int, device: str):
        self.strength = float(strength)
        self.d_h = int(d_h)
        self.device = device

    def step(
        self,
        h_t: torch.Tensor,                # (1, 1, d_h)
        composite_dir: torch.Tensor,      # (d_h,) unit-norm
        sign: int = 1,                    # +1 attractor, -1 avoid
    ) -> torch.Tensor:
        """Return ``δh`` of shape ``(1, 1, d_h)``.

        ``composite_dir`` must be unit-norm and on the same device as ``h_t``.
        ``sign`` must be ``+1`` (attract) or ``-1`` (repel).
        """
        if sign not in (1, -1):
            raise ValueError(f"sign must be +1 or -1, got {sign}")
        if h_t.shape[-1] != self.d_h:
            raise ValueError(
                f"h_t last-dim {h_t.shape[-1]} != configured d_h {self.d_h}"
            )
        if composite_dir.shape[-1] != self.d_h:
            raise ValueError(
                f"composite_dir last-dim {composite_dir.shape[-1]} != "
                f"configured d_h {self.d_h}"
            )
        h_norm = h_t.norm(dim=-1).item()  # scalar (single-token)
        delta = (sign * self.strength * h_norm) * composite_dir
        return delta.view(1, 1, self.d_h)
