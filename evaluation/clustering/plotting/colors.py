"""Small color and format utilities shared across clustering plots."""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np



def _cluster_colors(
    K: int, group_labels: np.ndarray | None = None,
) -> list[str]:
    """Return K distinct hex color strings.

    When *group_labels* (length-K array mapping cluster → group) is given,
    clusters in the same group share a base hue and differ in
    lightness/saturation so they look related but distinguishable.
    Without it, falls back to tab20 (K≤20) or turbo.
    """
    if group_labels is None:
        if K <= 20:
            cmap = plt.cm.tab20
            return [mcolors.to_hex(cmap(i % 20)) for i in range(K)]
        return [mcolors.to_hex(c) for c in plt.cm.turbo(
            np.linspace(0, 1, K, endpoint=False))]

    # All clusters in the same group share the same color.
    # Margin confidence is shown via opacity (see _margin_to_alpha),
    # so color intensity must NOT encode cluster identity within a group.
    import colorsys
    K_coarse = int(group_labels.max()) + 1
    group_hex: list[str] = []
    for g in range(K_coarse):
        h = g / K_coarse
        r, gr, b = colorsys.hls_to_rgb(h, 0.45, 0.65)
        group_hex.append(mcolors.to_hex((r, gr, b)))

    colors: list[str] = [""] * K
    for cid in range(K):
        colors[cid] = group_hex[int(group_labels[cid])]
    return colors


def _group_colors(K_coarse: int) -> list[str]:
    """Return K_coarse distinct pastel hex colors for group-level Voronoi."""
    import colorsys
    colors: list[str] = []
    for g in range(K_coarse):
        h = g / K_coarse
        r, gr, b = colorsys.hls_to_rgb(h, 0.7, 0.5)
        colors.append(mcolors.to_hex((r, gr, b)))
    return colors


def _margin_to_alpha(margin_val: float, m0: float = 0.02, m1: float = 0.12) -> float:
    """Map margin to background alpha in [0.25, 1.0]. Low margin → faint."""
    if np.isnan(margin_val):
        return 1.0
    t = (margin_val - m0) / (m1 - m0)
    return float(np.clip(t, 0.0, 1.0) * 0.75 + 0.25)


def _text_color_for_bg(hex_bg: str) -> str:
    """Return 'black' or 'white' depending on background luminance."""
    r, g, b = mcolors.to_rgb(hex_bg)
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    return "#000" if lum > 0.5 else "#fff"


