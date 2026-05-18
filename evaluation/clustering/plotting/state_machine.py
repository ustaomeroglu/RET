"""State-machine SVG rendering for clustering trajectories."""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


from evaluation.clustering.plotting.colors import _cluster_colors, _group_colors, _margin_to_alpha, _text_color_for_bg

def plot_state_machine_micro(
    P_pruned: np.ndarray,
    usage: np.ndarray,
    out_path: Path,
    title: str = "",
    node_scale: float = 3000.0,
    edge_scale: float = 5.0,
) -> None:
    """Draw a flat (non-hierarchical) state machine diagram of micro clusters.

    Nodes are sized by cluster usage frequency.  Edges are drawn where
    ``P_pruned[i, j] > 0`` (excluding self-loops); width is proportional to
    the transition probability.  Self-loops are rendered as thick node borders
    rather than literal arrows.

    Args:
        P_pruned: (K, K) pruned transition matrix (non-zero = edge to draw).
        usage: (K,) cluster frequency array.
        out_path: file path for saved figure.
        title: figure title.
        node_scale: base multiplier for node sizes.
        edge_scale: base multiplier for edge widths.
    """
    import networkx as nx

    K = P_pruned.shape[0]
    colors = _cluster_colors(K)

    G = nx.DiGraph()
    for i in range(K):
        G.add_node(i)

    edges = []
    weights = []
    for i in range(K):
        for j in range(K):
            if i != j and P_pruned[i, j] > 0:
                G.add_edge(i, j, weight=P_pruned[i, j])
                edges.append((i, j))
                weights.append(P_pruned[i, j])

    # Layout
    if K <= 12:
        pos = nx.kamada_kawai_layout(G)
    else:
        pos = nx.spring_layout(G, k=2.0 / np.sqrt(max(K, 1)), iterations=100, seed=42)

    fig, ax = plt.subplots(figsize=(max(10, K * 0.6), max(8, K * 0.5)))

    # Node sizes
    node_sizes = [max(100, usage[i] * node_scale) for i in range(K)]

    # Self-loop border widths (proportional to P[i,i])
    linewidths = [1.0 + P_pruned[i, i] * 8.0 for i in range(K)]

    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_size=node_sizes,
        node_color=colors[:K],
        edgecolors="black",
        linewidths=linewidths,
    )

    # Edge widths
    if weights:
        max_w = max(weights) if max(weights) > 0 else 1.0
        edge_widths = [w / max_w * edge_scale + 0.5 for w in weights]
    else:
        edge_widths = []

    if edges:
        nx.draw_networkx_edges(
            G, pos, ax=ax,
            edgelist=edges,
            width=edge_widths,
            alpha=0.7,
            edge_color="gray",
            arrows=True,
            arrowsize=15,
            connectionstyle="arc3,rad=0.1",
            min_source_margin=15,
            min_target_margin=15,
        )

    # Labels
    label_colors = {}
    for i in range(K):
        r, g, b = mcolors.to_rgb(colors[i])
        lum = 0.299 * r + 0.587 * g + 0.114 * b
        label_colors[i] = "black" if lum > 0.5 else "white"

    for i, (x, y) in pos.items():
        ax.text(
            x, y, str(i),
            ha="center", va="center",
            fontsize=max(6, min(10, 120 // max(K, 1))),
            fontweight="bold",
            color=label_colors[i],
        )

    # Edge labels for top-5 strongest edges
    if weights:
        top_edges = sorted(zip(weights, edges), reverse=True)[:5]
        edge_labels = {e: f"{w:.2f}" for w, e in top_edges}
        nx.draw_networkx_edge_labels(
            G, pos, ax=ax,
            edge_labels=edge_labels,
            font_size=7,
            label_pos=0.3,
        )

    # Self-loop legend annotation
    ax.text(
        0.01, 0.01,
        "Node border thickness = self-loop probability (stickiness)",
        transform=ax.transAxes, fontsize=8, alpha=0.6,
    )

    ax.set_title(title if title else "State Machine (Micro)", fontsize=12)
    ax.axis("off")

    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_state_machine_hierarchical(
    P_pruned_micro: np.ndarray,
    P_macro: np.ndarray,
    group_labels: np.ndarray,
    usage: np.ndarray,
    K_coarse: int,
    out_path: Path,
    title: str = "",
    node_scale: float = 2000.0,
    macro_edge_scale: float = 8.0,
    micro_edge_scale: float = 3.0,
    macro_alpha: float = 0.15,
    macro_top_r: int = 3,
    macro_min_prob: float = 0.03,
) -> None:
    """Draw a hierarchical state machine with macro bubbles containing micro nodes.

    Layout:
      1. Compute macro-level positions (spring/kamada_kawai for K_coarse nodes).
      2. For each macro group, compute a local micro layout (spring within a
         circle) and offset it to centre on the macro position.
      3. Draw macro bubbles as large translucent circles.
      4. Draw intra-group micro edges (thin, from pruned micro matrix).
      5. Draw inter-group macro edges (thick, from P_macro, pruned top-2).
      6. Label micro cluster IDs inside bubbles and macro group labels above.

    Args:
        P_pruned_micro: (K, K) pruned micro transition matrix.
        P_macro: (K_coarse, K_coarse) row-normalised macro transition matrix.
        group_labels: (K,) micro -> macro group mapping (0-indexed).
        usage: (K,) cluster frequency array.
        K_coarse: number of macro groups.
        out_path: output file path.
        title: figure title.
        node_scale: base node size multiplier.
        macro_edge_scale: edge width multiplier for macro arrows.
        micro_edge_scale: edge width multiplier for intra-group arrows.
        macro_alpha: transparency of macro bubble fill.
    """
    import networkx as nx

    K = P_pruned_micro.shape[0]
    colors = _cluster_colors(K, group_labels)

    # --- Macro layout (sparse graph: top_r + threshold per row) ---
    macro_G = nx.DiGraph()
    for g in range(K_coarse):
        macro_G.add_node(g)
    for g in range(K_coarse):
        row = P_macro[g].copy()
        row[g] = 0
        top_idx = np.argsort(-row)[:macro_top_r]
        for j in top_idx:
            if row[j] > 0:
                macro_G.add_edge(g, j, weight=row[j])
        for j in range(K_coarse):
            if j != g and row[j] >= macro_min_prob and not macro_G.has_edge(g, j):
                macro_G.add_edge(g, j, weight=row[j])

    macro_pos = nx.circular_layout(macro_G, scale=5.0)

    # Group membership
    groups: dict[int, list[int]] = {g: [] for g in range(K_coarse)}
    for i in range(K):
        groups[group_labels[i]].append(i)

    # Group usage (for bubble radius)
    group_usage = np.zeros(K_coarse, dtype=np.float64)
    for g in range(K_coarse):
        group_usage[g] = sum(usage[i] for i in groups[g])

    # --- Compute bubble radii ---
    bubble_radii: dict[int, float] = {}
    for g in range(K_coarse):
        bubble_radii[g] = max(0.3, np.sqrt(group_usage[g]) * 2.0)

    # --- Repulsion pass: push overlapping bubbles apart ---
    for _iter in range(50):
        moved = False
        for g1 in range(K_coarse):
            for g2 in range(g1 + 1, K_coarse):
                p1 = np.array(macro_pos[g1])
                p2 = np.array(macro_pos[g2])
                dist = np.linalg.norm(p2 - p1)
                min_dist = bubble_radii[g1] + bubble_radii[g2] + 0.6
                if dist < min_dist and dist > 1e-8:
                    direction = (p2 - p1) / dist
                    push = (min_dist - dist) / 2.0 + 0.15
                    macro_pos[g1] = tuple(p1 - direction * push)
                    macro_pos[g2] = tuple(p2 + direction * push)
                    moved = True
                elif dist <= 1e-8:
                    angle = g1 * 2 * np.pi / K_coarse
                    offset = np.array([np.cos(angle), np.sin(angle)]) * min_dist / 2
                    macro_pos[g1] = tuple(p1 + offset)
                    macro_pos[g2] = tuple(p2 - offset)
                    moved = True
        if not moved:
            break

    # --- Micro layout per group ---
    micro_pos: dict[int, np.ndarray] = {}

    for g in range(K_coarse):
        members = groups[g]
        cx, cy = macro_pos[g]
        radius = bubble_radii[g]

        if len(members) == 1:
            micro_pos[members[0]] = np.array([cx, cy])
        elif len(members) == 2:
            micro_pos[members[0]] = np.array([cx - radius * 0.3, cy])
            micro_pos[members[1]] = np.array([cx + radius * 0.3, cy])
        else:
            # Build subgraph for local layout
            sub_G = nx.DiGraph()
            for m in members:
                sub_G.add_node(m)
            for m1 in members:
                for m2 in members:
                    if m1 != m2 and P_pruned_micro[m1, m2] > 0:
                        sub_G.add_edge(m1, m2)

            local_pos = nx.spring_layout(sub_G, k=1.5, iterations=80, seed=42)

            # Scale to fit within radius and translate to macro centre
            if local_pos:
                coords = np.array(list(local_pos.values()))
                max_extent = np.abs(coords).max()
                scale = radius * 0.6 / max(max_extent, 1e-8)
                for node_id in local_pos:
                    micro_pos[node_id] = local_pos[node_id] * scale + np.array([cx, cy])

    # --- Draw ---
    fig_w = max(16, K_coarse * 4.0)
    fig_h = max(12, K_coarse * 3.2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # Macro group colours (pastel, matching group hues from _cluster_colors)
    macro_colors = _group_colors(K_coarse)

    # 1. Draw macro bubbles
    for g in range(K_coarse):
        cx, cy = macro_pos[g]
        radius = bubble_radii[g]
        circle = plt.Circle(
            (cx, cy), radius,
            facecolor=macro_colors[g],
            edgecolor="black",
            linewidth=2.0,
            alpha=macro_alpha,
            zorder=1,
        )
        ax.add_patch(circle)

        # Macro group label above the bubble
        pct = group_usage[g] * 100
        ax.text(
            cx, cy + radius + 0.12,
            f"Group {g} ({pct:.0f}%)",
            ha="center", va="bottom",
            fontsize=10, fontweight="bold",
            color="black",
        )

    # 2. Draw intra-group micro edges
    for g in range(K_coarse):
        members = groups[g]
        for m1 in members:
            for m2 in members:
                if m1 != m2 and P_pruned_micro[m1, m2] > 0:
                    x1, y1 = micro_pos[m1]
                    x2, y2 = micro_pos[m2]
                    w = P_pruned_micro[m1, m2]
                    ax.annotate(
                        "", xy=(x2, y2), xytext=(x1, y1),
                        arrowprops=dict(
                            arrowstyle="-|>",
                            color="gray",
                            lw=max(0.5, w * micro_edge_scale),
                            alpha=0.5,
                            connectionstyle="arc3,rad=0.15",
                            shrinkA=10, shrinkB=10,
                        ),
                        zorder=2,
                    )

    # 3. Draw inter-group macro arrows (pruned by top_r + min_prob)
    P_macro_pruned = np.zeros_like(P_macro)
    for g in range(K_coarse):
        row = P_macro[g].copy()
        row[g] = 0
        top_idx = np.argsort(-row)[:macro_top_r]
        for j in top_idx:
            if row[j] > 0:
                P_macro_pruned[g, j] = row[j]
        for j in range(K_coarse):
            if j != g and row[j] >= macro_min_prob:
                P_macro_pruned[g, j] = row[j]

    def _point_seg_dist(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        ab = b - a
        ab2 = np.dot(ab, ab)
        if ab2 < 1e-12:
            return float(np.linalg.norm(p - a))
        t = np.clip(np.dot(p - a, ab) / ab2, 0.0, 1.0)
        return float(np.linalg.norm(p - (a + t * ab)))

    for g1 in range(K_coarse):
        for g2 in range(K_coarse):
            if g1 != g2 and P_macro_pruned[g1, g2] > 0:
                c1 = np.array(macro_pos[g1])
                c2 = np.array(macro_pos[g2])
                d = np.linalg.norm(c2 - c1)
                if d < 1e-8:
                    continue
                direction = (c2 - c1) / d
                start = c1 + direction * (bubble_radii[g1] + 0.05)
                end = c2 - direction * (bubble_radii[g2] + 0.05)

                arc_rad = 0.15
                for g_other in range(K_coarse):
                    if g_other in (g1, g2):
                        continue
                    c_other = np.array(macro_pos[g_other])
                    if _point_seg_dist(c_other, start, end) < bubble_radii[g_other] + 0.1:
                        arc_rad = max(arc_rad, 0.45)
                        seg_vec = end - start
                        to_obs = c_other - start
                        cross = seg_vec[0] * to_obs[1] - seg_vec[1] * to_obs[0]
                        arc_rad = arc_rad if cross < 0 else -arc_rad
                        break

                w = P_macro_pruned[g1, g2]
                ax.annotate(
                    "", xy=end, xytext=start,
                    arrowprops=dict(
                        arrowstyle="-|>", color="darkblue",
                        lw=max(2.0, w * macro_edge_scale), alpha=0.8,
                        connectionstyle=f"arc3,rad={arc_rad}",
                    ),
                    zorder=4,
                )
                mid = (start + end) / 2
                ax.text(
                    mid[0], mid[1], f"{w:.3f}",
                    fontsize=7, fontweight="bold",
                    color="darkblue", alpha=0.9, ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                              alpha=0.7, edgecolor="none"),
                    zorder=5,
                )

    # 4. Draw micro nodes
    min_node_size = max(300, 5000 / max(K, 1))
    label_fontsize = max(7, min(11, 250 // max(K, 1)))
    for i in range(K):
        x, y = micro_pos[i]
        size = max(min_node_size, usage[i] * node_scale)
        border_w = 0.5 + P_pruned_micro[i, i] * 1.5
        ax.scatter(
            x, y, s=size,
            c=colors[i], edgecolors="black",
            linewidths=border_w, zorder=3,
        )
        r, g_val, b = mcolors.to_rgb(colors[i])
        lum = 0.299 * r + 0.587 * g_val + 0.114 * b
        ax.text(
            x, y, str(i),
            ha="center", va="center",
            fontsize=label_fontsize,
            fontweight="bold",
            color="black" if lum > 0.5 else "white",
            zorder=6,
        )

    # Axes
    ax.set_title(title if title else "Hierarchical State Machine", fontsize=13)
    ax.axis("off")
    ax.set_aspect("equal")

    # Auto-scale axes to fit all elements
    all_x = [micro_pos[i][0] for i in range(K)]
    all_y = [micro_pos[i][1] for i in range(K)]
    max_radius = max(bubble_radii.values()) if bubble_radii else 1.0
    pad = max_radius + 0.5
    ax.set_xlim(min(all_x) - pad, max(all_x) + pad)
    ax.set_ylim(min(all_y) - pad - 0.3, max(all_y) + pad + 0.3)

    # Legend
    ax.text(
        0.01, 0.01,
        "Thin gray = intra-group transitions | Thick blue = inter-group transitions\n"
        "Node border thickness = self-loop (stickiness) | Node size = frequency",
        transform=ax.transAxes, fontsize=7, alpha=0.6,
        va="bottom",
    )

    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def _compute_state_machine_layout(
    P_exit_micro: np.ndarray,
    P_exit: np.ndarray,
    group_labels: np.ndarray,
    usage: np.ndarray,
    K_coarse: int,
    micro_top_r: int = 3,
    micro_min_prob: float = 0.03,
    macro_top_r: int = 3,
    macro_min_prob: float = 0.03,
) -> dict:
    """Compute layout for a hierarchical state machine diagram.

    Returns dict with keys: macro_pos, micro_pos, bubble_radii, groups,
    group_usage, P_micro_pruned, P_exit_pruned.
    """
    import networkx as nx

    K = P_exit_micro.shape[0]

    # --- Macro layout (sparse graph: top_r + threshold per row) ---
    macro_G = nx.DiGraph()
    for g in range(K_coarse):
        macro_G.add_node(g)
    for g in range(K_coarse):
        row = P_exit[g].copy()
        row[g] = 0
        top_idx = np.argsort(-row)[:macro_top_r]
        for j in top_idx:
            if row[j] > 0:
                macro_G.add_edge(g, j, weight=row[j])
        for j in range(K_coarse):
            if j != g and row[j] >= macro_min_prob and not macro_G.has_edge(g, j):
                macro_G.add_edge(g, j, weight=row[j])

    macro_pos = nx.circular_layout(macro_G, scale=5.0)

    # Group membership
    groups: dict[int, list[int]] = {g: [] for g in range(K_coarse)}
    for i in range(K):
        groups[group_labels[i]].append(i)

    # Group usage (for bubble radius)
    group_usage = np.zeros(K_coarse, dtype=np.float64)
    for g in range(K_coarse):
        group_usage[g] = sum(usage[i] for i in groups[g])

    bubble_radii: dict[int, float] = {}
    for g in range(K_coarse):
        bubble_radii[g] = max(0.3, np.sqrt(group_usage[g]) * 2.0)

    # Repulsion pass — push groups farther apart
    for _iter in range(50):
        moved = False
        for g1_r in range(K_coarse):
            for g2_r in range(g1_r + 1, K_coarse):
                p1 = np.array(macro_pos[g1_r])
                p2 = np.array(macro_pos[g2_r])
                dist = np.linalg.norm(p2 - p1)
                min_dist = bubble_radii[g1_r] + bubble_radii[g2_r] + 0.6
                if dist < min_dist:
                    if dist > 1e-8:
                        direction = (p2 - p1) / dist
                        push = (min_dist - dist) / 2.0 + 0.15
                        macro_pos[g1_r] = p1 - direction * push
                        macro_pos[g2_r] = p2 + direction * push
                    else:
                        angle = g1_r * 2 * np.pi / K_coarse
                        offset = np.array([np.cos(angle), np.sin(angle)]) * min_dist / 2
                        macro_pos[g1_r] = p1 + offset
                        macro_pos[g2_r] = p2 - offset
                    moved = True
        if not moved:
            break

    # Micro node positions within bubbles
    micro_pos: dict[int, np.ndarray] = {}
    for g in range(K_coarse):
        members = groups[g]
        cx, cy = macro_pos[g]
        radius = bubble_radii[g]
        if len(members) == 1:
            micro_pos[members[0]] = np.array([cx, cy])
        elif len(members) == 2:
            micro_pos[members[0]] = np.array([cx - radius * 0.3, cy])
            micro_pos[members[1]] = np.array([cx + radius * 0.3, cy])
        else:
            sub_G = nx.DiGraph()
            for m in members:
                sub_G.add_node(m)
            for m1 in members:
                for m2 in members:
                    if m1 != m2 and P_exit_micro[m1, m2] > 0:
                        sub_G.add_edge(m1, m2, weight=P_exit_micro[m1, m2])
            local_pos = nx.spring_layout(sub_G, k=1.5, iterations=80, seed=42)
            coords = np.array(list(local_pos.values()))
            max_extent = np.abs(coords).max()
            scale = radius * 0.6 / max(max_extent, 1e-8)
            for node_id in local_pos:
                micro_pos[node_id] = local_pos[node_id] * scale + np.array([cx, cy])

    # --- Prune micro exit matrix ---
    P_micro_pruned = np.zeros_like(P_exit_micro)
    for i in range(K):
        row = P_exit_micro[i].copy()
        row[i] = 0
        top_idx = np.argsort(-row)[:micro_top_r]
        for j in top_idx:
            if row[j] > 0:
                P_micro_pruned[i, j] = row[j]
        for j in range(K):
            if j != i and row[j] >= micro_min_prob:
                P_micro_pruned[i, j] = row[j]

    # --- Prune macro exit matrix ---
    P_exit_pruned = np.zeros_like(P_exit)
    for g in range(K_coarse):
        row = P_exit[g].copy()
        row[g] = 0
        top_idx = np.argsort(-row)[:macro_top_r]
        for j in top_idx:
            if row[j] > 0:
                P_exit_pruned[g, j] = row[j]
        for j in range(K_coarse):
            if j != g and row[j] >= macro_min_prob:
                P_exit_pruned[g, j] = row[j]

    # --- Compute per-row ranks (1-indexed) for dynamic pruning UI ---
    micro_rank = np.zeros_like(P_exit_micro, dtype=int)
    for i in range(K):
        row = P_exit_micro[i].copy()
        row[i] = 0
        order = np.argsort(-row)
        for rank_pos, j in enumerate(order):
            if row[j] > 0:
                micro_rank[i, j] = rank_pos + 1

    macro_rank = np.zeros_like(P_exit, dtype=int)
    for g in range(K_coarse):
        row = P_exit[g].copy()
        row[g] = 0
        order = np.argsort(-row)
        for rank_pos, j in enumerate(order):
            if row[j] > 0:
                macro_rank[g, j] = rank_pos + 1

    return {
        "macro_pos": macro_pos,
        "micro_pos": micro_pos,
        "bubble_radii": bubble_radii,
        "groups": groups,
        "group_usage": group_usage,
        "P_micro_pruned": P_micro_pruned,
        "P_exit_pruned": P_exit_pruned,
        "P_exit_micro_full": P_exit_micro,
        "P_exit_full": P_exit,
        "micro_rank": micro_rank,
        "macro_rank": macro_rank,
    }


def _render_state_machine_svg(
    layout: dict,
    K: int,
    K_coarse: int,
    group_labels: np.ndarray,
    usage: np.ndarray,
    P_self_loop: np.ndarray | None,
    colors: list[str],
    macro_colors: list[str],
    default_micro_top_r: int = 3,
    default_micro_min_prob: float = 0.03,
    default_macro_top_r: int = 3,
    default_macro_min_prob: float = 0.03,
) -> str:
    """Generate an inline SVG state machine diagram from precomputed layout.

    When full matrices and rank arrays are available in the layout dict,
    renders ALL edges above a tiny floor with data-prob/data-rank attributes
    for client-side dynamic pruning.  Falls back to pruned matrices otherwise.
    """
    import math

    macro_pos = layout["macro_pos"]
    micro_pos = layout["micro_pos"]
    bubble_radii = layout["bubble_radii"]
    groups = layout["groups"]
    group_usage = layout["group_usage"]

    # Use full matrices + ranks for dynamic pruning if available
    has_full = "P_exit_micro_full" in layout
    P_exit_micro = layout["P_exit_micro_full"] if has_full else layout["P_micro_pruned"]
    P_exit_mat = layout["P_exit_full"] if has_full else layout["P_exit_pruned"]
    micro_rank = layout.get("micro_rank")
    macro_rank = layout.get("macro_rank")
    EDGE_FLOOR = 0.001 if has_full else 0.0

    # SVG dimensions
    svg_w = 900 if K_coarse <= 12 else 1200
    svg_h = int(svg_w * 1.1)

    # Bounding box
    all_x = [float(micro_pos[i][0]) for i in range(K)]
    all_y = [float(micro_pos[i][1]) for i in range(K)]
    max_r = max(bubble_radii.values()) if bubble_radii else 1.0
    x_min = min(all_x) - max_r - 0.5
    x_max = max(all_x) + max_r + 0.5
    y_min = min(all_y) - max_r - 0.5
    y_max = max(all_y) + max_r + 0.8

    pad_l, pad_t, pad_r_val, pad_b = 20, 30, 20, 20
    pw = svg_w - pad_l - pad_r_val
    ph = svg_h - pad_t - pad_b
    x_range = max(x_max - x_min, 1e-9)
    y_range = max(y_max - y_min, 1e-9)

    def sx(x):
        return pad_l + (x - x_min) / x_range * pw

    def sy(y):
        return pad_t + (y_max - y) / y_range * ph  # flip y

    def sr(r):
        """Convert data-space radius to SVG pixels."""
        return r / x_range * pw

    # Node visual radius
    node_r = max(4, min(16, 400 // max(K, 1)))
    label_fs = max(6, min(11, 250 // max(K, 1)))

    s = []
    s.append(f'<svg class="sm-svg" viewBox="0 0 {svg_w} {svg_h}" '
             f'xmlns="http://www.w3.org/2000/svg">')

    # Defs: arrowheads + glow filter
    s.append('<defs>')
    s.append('<marker id="sm-arrow" markerWidth="5" markerHeight="4" '
             'refX="4" refY="2" orient="auto">'
             '<polygon points="0 0, 5 2, 0 4" fill="gray" fill-opacity="0.6"/>'
             '</marker>')
    s.append('<marker id="sm-arrow-macro" markerWidth="7" markerHeight="5" '
             'refX="6" refY="2.5" orient="auto">'
             '<polygon points="0 0, 7 2.5, 0 5" fill="darkblue"/>'
             '</marker>')
    s.append('<filter id="sm-glow"><feGaussianBlur stdDeviation="3" result="blur"/>'
             '<feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/>'
             '</feMerge></filter>')
    s.append('</defs>')

    # 1. Group bubbles
    for g in range(K_coarse):
        cx_g, cy_g = float(macro_pos[g][0]), float(macro_pos[g][1])
        r_svg = sr(bubble_radii[g])
        s.append(
            f'<circle class="sm-bubble" id="sm-bubble-{g}" data-group="{g}" '
            f'cx="{sx(cx_g):.1f}" cy="{sy(cy_g):.1f}" r="{r_svg:.1f}" '
            f'fill="{macro_colors[g]}" fill-opacity="0.15" '
            f'stroke="gray" stroke-width="1.5" stroke-dasharray="6,3"/>'
        )

    # 2. Group labels above bubbles
    for g in range(K_coarse):
        cx_g, cy_g = float(macro_pos[g][0]), float(macro_pos[g][1])
        r_svg = sr(bubble_radii[g])
        pct = group_usage[g] * 100
        s.append(
            f'<text class="sm-group-label" id="sm-glabel-{g}" '
            f'x="{sx(cx_g):.1f}" y="{sy(cy_g) - r_svg - 4:.1f}" '
            f'text-anchor="middle" font-size="10" font-weight="bold" '
            f'fill="dimgray">G{g} ({pct:.0f}%)</text>'
        )

    # 3. Micro exit edges (curved) — all above floor, with data attrs for dynamic pruning
    for i in range(K):
        for j in range(K):
            if i != j and P_exit_micro[i, j] > EDGE_FLOOR:
                x1, y1 = sx(float(micro_pos[i][0])), sy(float(micro_pos[i][1]))
                x2, y2 = sx(float(micro_pos[j][0])), sy(float(micro_pos[j][1]))
                w = P_exit_micro[i, j]
                rank = int(micro_rank[i, j]) if micro_rank is not None else 1
                same_group = group_labels[i] == group_labels[j]
                # Quadratic bezier control point (perpendicular offset)
                dx, dy = x2 - x1, y2 - y1
                length = math.sqrt(dx * dx + dy * dy)
                if length < 1:
                    continue
                # Shrink endpoints by node radius
                ux, uy = dx / length, dy / length
                x1s = x1 + ux * node_r
                y1s = y1 + uy * node_r
                x2s = x2 - ux * node_r
                y2s = y2 - uy * node_r
                # Control point offset
                nx_v, ny_v = -dy / length, dx / length
                offset = 0.12 * length
                qx = (x1s + x2s) / 2 + nx_v * offset
                qy = (y1s + y2s) / 2 + ny_v * offset
                lw = max(0.5, w * 2.5)
                dash = "" if same_group else ' stroke-dasharray="4,3"'
                visible = (rank <= default_micro_top_r) and (w >= default_micro_min_prob)
                disp = '' if visible else ' display="none"'
                s.append(
                    f'<path class="sm-micro-edge" data-src="{i}" data-dst="{j}" '
                    f'data-prob="{w:.4f}" data-rank="{rank}" '
                    f'd="M {x1s:.1f} {y1s:.1f} Q {qx:.1f} {qy:.1f} {x2s:.1f} {y2s:.1f}" '
                    f'stroke="gray" stroke-width="{lw:.1f}" stroke-opacity="0.4" '
                    f'fill="none" marker-end="url(#sm-arrow)"{dash}{disp}/>'
                )

    # 4. Macro exit arrows — all above floor, with data attrs for dynamic pruning
    for g1 in range(K_coarse):
        for g2 in range(K_coarse):
            if g1 != g2 and P_exit_mat[g1, g2] > EDGE_FLOOR:
                c1x, c1y = sx(float(macro_pos[g1][0])), sy(float(macro_pos[g1][1]))
                c2x, c2y = sx(float(macro_pos[g2][0])), sy(float(macro_pos[g2][1]))
                dx, dy = c2x - c1x, c2y - c1y
                d = math.sqrt(dx * dx + dy * dy)
                if d < 1:
                    continue
                ux, uy = dx / d, dy / d
                r1 = sr(bubble_radii[g1]) + 3
                r2 = sr(bubble_radii[g2]) + 3
                x1s = c1x + ux * r1
                y1s = c1y + uy * r1
                x2s = c2x - ux * r2
                y2s = c2y - uy * r2
                # Curve control point
                nx_v, ny_v = -dy / d, dx / d
                offset = 0.08 * d
                qx = (x1s + x2s) / 2 + nx_v * offset
                qy = (y1s + y2s) / 2 + ny_v * offset
                w = P_exit_mat[g1, g2]
                rank = int(macro_rank[g1, g2]) if macro_rank is not None else 1
                lw = max(2, w * 6)
                visible = (rank <= default_macro_top_r) and (w >= default_macro_min_prob)
                disp = '' if visible else ' display="none"'
                s.append(
                    f'<path class="sm-macro-edge" '
                    f'data-src-group="{g1}" data-dst-group="{g2}" '
                    f'data-prob="{w:.4f}" data-rank="{rank}" '
                    f'd="M {x1s:.1f} {y1s:.1f} Q {qx:.1f} {qy:.1f} {x2s:.1f} {y2s:.1f}" '
                    f'stroke="darkblue" stroke-width="{lw:.1f}" stroke-opacity="0.7" '
                    f'fill="none" marker-end="url(#sm-arrow-macro)"{disp}/>'
                )
                # Percentage label at midpoint
                mx = (x1s + x2s) / 2 + nx_v * offset * 0.5
                my = (y1s + y2s) / 2 + ny_v * offset * 0.5
                s.append(
                    f'<text class="sm-macro-label" '
                    f'data-prob="{w:.4f}" data-rank="{rank}" '
                    f'x="{mx:.1f}" y="{my:.1f}" text-anchor="middle" '
                    f'font-size="9" font-weight="bold" fill="darkblue"'
                    f'{disp}>{w:.0%}</text>'
                )

    # 5. Micro nodes
    for i in range(K):
        cx_n = sx(float(micro_pos[i][0]))
        cy_n = sy(float(micro_pos[i][1]))
        g = int(group_labels[i])
        size = max(node_r, node_r * (usage[i] / max(usage.max(), 1e-9)) * 2.5)
        self_p = float(P_self_loop[i]) if P_self_loop is not None else 0.0
        bw = 0.5 + self_p * 1.5
        r_c, g_c, b_c = mcolors.to_rgb(colors[i])
        lum = 0.299 * r_c + 0.587 * g_c + 0.114 * b_c
        txt_col = "#000" if lum > 0.5 else "#fff"
        s.append(
            f'<circle class="sm-node" id="sm-node-{i}" '
            f'data-cluster="{i}" data-group="{g}" '
            f'cx="{cx_n:.1f}" cy="{cy_n:.1f}" r="{size:.1f}" '
            f'fill="{colors[i]}" stroke="black" stroke-width="{bw:.1f}"/>'
        )
        s.append(
            f'<text x="{cx_n:.1f}" y="{cy_n:.1f}" text-anchor="middle" '
            f'dominant-baseline="central" font-size="{label_fs}" '
            f'font-weight="bold" fill="{txt_col}">{i}</text>'
        )
        # Self-loop annotation
        if P_self_loop is not None:
            s.append(
                f'<text class="sm-self-loop" x="{cx_n:.1f}" '
                f'y="{cy_n + size + 10:.1f}" text-anchor="middle" '
                f'font-size="7" fill="gray">s={self_p:.2f}</text>'
            )

    s.append('</svg>')
    return "\n".join(s)


def _render_trajectory_svg(
    layout: dict,
    labels: np.ndarray,
    K: int,
    K_coarse: int,
    group_labels: np.ndarray,
    usage: np.ndarray,
    colors: list[str],
    macro_colors: list[str],
) -> str:
    """Generate an inline SVG showing the realized trajectory on the SM layout.

    Uses the same node/bubble positions as ``_render_state_machine_svg`` but
    only highlights visited states and draws arrows for realized transitions
    colored by temporal order.
    """
    import math

    macro_pos = layout["macro_pos"]
    micro_pos = layout["micro_pos"]
    bubble_radii = layout["bubble_radii"]
    groups = layout["groups"]
    group_usage = layout["group_usage"]

    # --- Run-length encode the trajectory ---
    state_seq: list[tuple[int, int, int]] = []  # (cluster_id, start, end)
    i = 0
    n = len(labels)
    while i < n:
        c = int(labels[i])
        j = i + 1
        while j < n and int(labels[j]) == c:
            j += 1
        state_seq.append((c, i, j - 1))
        i = j

    visited: set[int] = set(s[0] for s in state_seq)
    active_groups: set[int] = set(int(group_labels[c]) for c in visited if c < K)

    # Visit count per cluster
    visit_count: dict[int, int] = {}
    first_visit_order: dict[int, int] = {}
    order_idx = 0
    for c, _, _ in state_seq:
        visit_count[c] = visit_count.get(c, 0) + 1
        if c not in first_visit_order:
            first_visit_order[c] = order_idx
            order_idx += 1

    # Transitions: (src, dst) -> {count, first_idx}
    transitions: dict[tuple[int, int], dict] = {}
    for idx in range(len(state_seq) - 1):
        src, dst = state_seq[idx][0], state_seq[idx + 1][0]
        if src != dst:
            key = (src, dst)
            if key not in transitions:
                transitions[key] = {"count": 0, "first_idx": idx}
            transitions[key]["count"] += 1

    first_cluster = state_seq[0][0] if state_seq else -1
    last_cluster = state_seq[-1][0] if state_seq else -1

    # --- SVG setup (same dimensions as SM) ---
    svg_w = 900 if K_coarse <= 12 else 1200
    svg_h = int(svg_w * 1.1)

    all_x = [float(micro_pos[i][0]) for i in range(K)]
    all_y = [float(micro_pos[i][1]) for i in range(K)]
    max_r = max(bubble_radii.values()) if bubble_radii else 1.0
    x_min = min(all_x) - max_r - 0.5
    x_max = max(all_x) + max_r + 0.5
    y_min = min(all_y) - max_r - 0.5
    y_max = max(all_y) + max_r + 0.8

    pad_l, pad_t, pad_r_val, pad_b = 20, 30, 20, 20
    pw = svg_w - pad_l - pad_r_val
    ph = svg_h - pad_t - pad_b
    x_range = max(x_max - x_min, 1e-9)
    y_range = max(y_max - y_min, 1e-9)

    def sx(x):
        return pad_l + (x - x_min) / x_range * pw

    def sy(y):
        return pad_t + (y_max - y) / y_range * ph

    def sr(r):
        return r / x_range * pw

    node_r = max(4, min(16, 400 // max(K, 1)))
    label_fs = max(6, min(11, 250 // max(K, 1)))

    # Viridis color map for temporal ordering
    viridis_cm = plt.cm.viridis
    max_first_idx = max((t["first_idx"] for t in transitions.values()), default=1)
    if max_first_idx == 0:
        max_first_idx = 1
    max_traj_count = max((t["count"] for t in transitions.values()), default=1)

    s: list[str] = []
    s.append(f'<svg class="sm-svg" data-max-count="{max_traj_count}" '
             f'viewBox="0 0 {svg_w} {svg_h}" '
             f'xmlns="http://www.w3.org/2000/svg">')

    # Defs
    s.append('<defs>')
    s.append('<marker id="traj-arrow" markerWidth="5" markerHeight="4" '
             'refX="4" refY="2" orient="auto">'
             '<polygon points="0 0, 5 2, 0 4" fill="#333" fill-opacity="0.8"/>'
             '</marker>')
    s.append('<filter id="sm-glow"><feGaussianBlur stdDeviation="3" result="blur"/>'
             '<feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/>'
             '</feMerge></filter>')
    s.append('</defs>')

    # 1. Group bubbles (all, but faded if inactive)
    for g in range(K_coarse):
        cx_g, cy_g = float(macro_pos[g][0]), float(macro_pos[g][1])
        r_svg = sr(bubble_radii[g])
        active = g in active_groups
        fill_op = "0.15" if active else "0.04"
        stroke_op = "1" if active else "0.15"
        s.append(
            f'<circle class="sm-bubble" id="traj-bubble-{g}" data-group="{g}" '
            f'cx="{sx(cx_g):.1f}" cy="{sy(cy_g):.1f}" r="{r_svg:.1f}" '
            f'fill="{macro_colors[g]}" fill-opacity="{fill_op}" '
            f'stroke="gray" stroke-width="1.5" stroke-dasharray="6,3" '
            f'stroke-opacity="{stroke_op}"/>'
        )

    # 2. Group labels (only active groups)
    for g in range(K_coarse):
        cx_g, cy_g = float(macro_pos[g][0]), float(macro_pos[g][1])
        r_svg = sr(bubble_radii[g])
        if g in active_groups:
            pct = group_usage[g] * 100
            s.append(
                f'<text x="{sx(cx_g):.1f}" y="{sy(cy_g) - r_svg - 4:.1f}" '
                f'text-anchor="middle" font-size="10" font-weight="bold" '
                f'fill="dimgray">G{g} ({pct:.0f}%)</text>'
            )

    # 3. Transition arrows (colored by temporal order)
    sorted_trans = sorted(transitions.items(), key=lambda t: t[1]["first_idx"])
    for (src, dst), info in sorted_trans:
        x1 = sx(float(micro_pos[src][0]))
        y1 = sy(float(micro_pos[src][1]))
        x2 = sx(float(micro_pos[dst][0]))
        y2 = sy(float(micro_pos[dst][1]))
        dx, dy = x2 - x1, y2 - y1
        length = math.sqrt(dx * dx + dy * dy)
        if length < 1:
            continue
        ux, uy = dx / length, dy / length
        x1s = x1 + ux * node_r
        y1s = y1 + uy * node_r
        x2s = x2 - ux * node_r
        y2s = y2 - uy * node_r
        # Bezier control
        nx_v, ny_v = -dy / length, dx / length
        offset = 0.12 * length
        qx = (x1s + x2s) / 2 + nx_v * offset
        qy = (y1s + y2s) / 2 + ny_v * offset
        # Width by count, color by temporal order
        count = info["count"]
        lw = max(1.0, min(4.0, 1.0 + count * 0.5))
        t_frac = info["first_idx"] / max_first_idx
        edge_color = mcolors.to_hex(viridis_cm(t_frac))
        s.append(
            f'<path class="traj-edge" data-src="{src}" data-dst="{dst}" '
            f'data-count="{count}" '
            f'd="M {x1s:.1f} {y1s:.1f} Q {qx:.1f} {qy:.1f} {x2s:.1f} {y2s:.1f}" '
            f'stroke="{edge_color}" stroke-width="{lw:.1f}" stroke-opacity="0.8" '
            f'fill="none" marker-end="url(#traj-arrow)"/>'
        )
        # Count label if > 1
        if count > 1:
            mx = (x1s + x2s) / 2 + nx_v * offset * 0.5
            my = (y1s + y2s) / 2 + ny_v * offset * 0.5
            s.append(
                f'<text class="traj-label" data-count="{count}" '
                f'x="{mx:.1f}" y="{my:.1f}" text-anchor="middle" '
                f'font-size="7" font-weight="bold" fill="{edge_color}">'
                f'\u00d7{count}</text>'
            )

    # 4. Micro nodes (all, faded if not visited)
    max_usage = max(usage.max(), 1e-9)
    for ci in range(K):
        cx_n = sx(float(micro_pos[ci][0]))
        cy_n = sy(float(micro_pos[ci][1]))
        g = int(group_labels[ci])
        size = max(node_r, node_r * (usage[ci] / max_usage) * 2.5)

        if ci in visited:
            fill = colors[ci]
            fill_op = "1"
            stroke_col = "black"
            stroke_w = 1.0
            # Start / end markers
            if ci == first_cluster:
                stroke_col = "lime"
                stroke_w = 3.0
            elif ci == last_cluster:
                stroke_col = "red"
                stroke_w = 3.0
        else:
            fill = "#ccc"
            fill_op = "0.2"
            stroke_col = "#aaa"
            stroke_w = 0.5

        s.append(
            f'<circle class="sm-node" id="traj-node-{ci}" '
            f'data-cluster="{ci}" data-group="{g}" '
            f'cx="{cx_n:.1f}" cy="{cy_n:.1f}" r="{size:.1f}" '
            f'fill="{fill}" fill-opacity="{fill_op}" '
            f'stroke="{stroke_col}" stroke-width="{stroke_w:.1f}"/>'
        )

        if ci in visited:
            r_c, g_c, b_c = mcolors.to_rgb(colors[ci])
            lum = 0.299 * r_c + 0.587 * g_c + 0.114 * b_c
            txt_col = "#000" if lum > 0.5 else "#fff"
            s.append(
                f'<text x="{cx_n:.1f}" y="{cy_n:.1f}" text-anchor="middle" '
                f'dominant-baseline="central" font-size="{label_fs}" '
                f'font-weight="bold" fill="{txt_col}">{ci}</text>'
            )
            # Visit order annotation below node
            order = first_visit_order[ci]
            s.append(
                f'<text x="{cx_n:.1f}" y="{cy_n + size + 10:.1f}" '
                f'text-anchor="middle" font-size="7" fill="gray">'
                f'#{order + 1} ({visit_count[ci]}x)</text>'
            )

    s.append('</svg>')
    return "\n".join(s)


def _render_live_trajectory_base_svg(
    layout: dict,
    labels: np.ndarray,
    K: int,
    K_coarse: int,
    group_labels: np.ndarray,
    usage: np.ndarray,
    colors: list[str],
    macro_colors: list[str],
) -> tuple[str, dict]:
    """Generate base SVG for the Live Trajectory tab (nodes only, no edges).

    Returns ``(svg_string, metadata)`` where metadata contains node positions
    and sizes needed by the client-side JS to draw dynamic edges.
    """
    import math

    macro_pos = layout["macro_pos"]
    micro_pos = layout["micro_pos"]
    bubble_radii = layout["bubble_radii"]
    groups = layout["groups"]
    group_usage = layout["group_usage"]

    # --- RLE to find visited states ---
    state_seq: list[tuple[int, int, int]] = []
    i = 0
    n = len(labels)
    while i < n:
        c = int(labels[i])
        j = i + 1
        while j < n and int(labels[j]) == c:
            j += 1
        state_seq.append((c, i, j - 1))
        i = j

    visited: set[int] = set(s[0] for s in state_seq)
    active_groups: set[int] = set(int(group_labels[c]) for c in visited if c < K)

    visit_count: dict[int, int] = {}
    first_visit_order: dict[int, int] = {}
    order_idx = 0
    for c, _, _ in state_seq:
        visit_count[c] = visit_count.get(c, 0) + 1
        if c not in first_visit_order:
            first_visit_order[c] = order_idx
            order_idx += 1

    first_cluster = state_seq[0][0] if state_seq else -1
    last_cluster = state_seq[-1][0] if state_seq else -1

    # --- SVG setup (same dimensions as SM / Trajectory) ---
    svg_w = 900 if K_coarse <= 12 else 1200
    svg_h = int(svg_w * 1.1)

    all_x = [float(micro_pos[i][0]) for i in range(K)]
    all_y = [float(micro_pos[i][1]) for i in range(K)]
    max_r = max(bubble_radii.values()) if bubble_radii else 1.0
    x_min = min(all_x) - max_r - 0.5
    x_max = max(all_x) + max_r + 0.5
    y_min = min(all_y) - max_r - 0.5
    y_max = max(all_y) + max_r + 0.8

    pad_l, pad_t, pad_r_val, pad_b = 20, 30, 20, 20
    pw = svg_w - pad_l - pad_r_val
    ph = svg_h - pad_t - pad_b
    x_range = max(x_max - x_min, 1e-9)
    y_range = max(y_max - y_min, 1e-9)

    def sx(x):
        return pad_l + (x - x_min) / x_range * pw

    def sy(y):
        return pad_t + (y_max - y) / y_range * ph

    def sr(r):
        return r / x_range * pw

    node_r = max(4, min(16, 400 // max(K, 1)))
    label_fs = max(6, min(11, 250 // max(K, 1)))

    s: list[str] = []
    s.append(f'<svg class="sm-svg" viewBox="0 0 {svg_w} {svg_h}" '
             f'xmlns="http://www.w3.org/2000/svg">')

    # Defs: arrow markers for past (blue) and future (orange), glow filter
    s.append('<defs>')
    s.append('<marker id="live-arrow-past" markerWidth="5" markerHeight="4" '
             'refX="4" refY="2" orient="auto">'
             '<polygon points="0 0, 5 2, 0 4" fill="#4488cc"/>'
             '</marker>')
    s.append('<marker id="live-arrow-future" markerWidth="5" markerHeight="4" '
             'refX="4" refY="2" orient="auto">'
             '<polygon points="0 0, 5 2, 0 4" fill="#dd8833"/>'
             '</marker>')
    s.append('<filter id="live-glow"><feGaussianBlur stdDeviation="4" result="blur"/>'
             '<feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/>'
             '</feMerge></filter>')
    s.append('</defs>')

    # Empty group for dynamically drawn edges
    s.append('<g id="live-edges"></g>')

    # 1. Group bubbles
    for g in range(K_coarse):
        cx_g, cy_g = float(macro_pos[g][0]), float(macro_pos[g][1])
        r_svg = sr(bubble_radii[g])
        active = g in active_groups
        fill_op = "0.15" if active else "0.04"
        stroke_op = "1" if active else "0.15"
        s.append(
            f'<circle class="sm-bubble" id="live-bubble-{g}" data-group="{g}" '
            f'cx="{sx(cx_g):.1f}" cy="{sy(cy_g):.1f}" r="{r_svg:.1f}" '
            f'fill="{macro_colors[g]}" fill-opacity="{fill_op}" '
            f'stroke="gray" stroke-width="1.5" stroke-dasharray="6,3" '
            f'stroke-opacity="{stroke_op}"/>'
        )

    # 2. Group labels (active groups only)
    for g in range(K_coarse):
        cx_g, cy_g = float(macro_pos[g][0]), float(macro_pos[g][1])
        r_svg = sr(bubble_radii[g])
        if g in active_groups:
            pct = group_usage[g] * 100
            s.append(
                f'<text x="{sx(cx_g):.1f}" y="{sy(cy_g) - r_svg - 4:.1f}" '
                f'text-anchor="middle" font-size="10" font-weight="bold" '
                f'fill="dimgray">G{g} ({pct:.0f}%)</text>'
            )

    # 3. Micro nodes
    max_usage = max(usage.max(), 1e-9)
    node_pos: dict[int, tuple[float, float]] = {}
    node_sizes: dict[int, float] = {}
    for ci in range(K):
        cx_n = sx(float(micro_pos[ci][0]))
        cy_n = sy(float(micro_pos[ci][1]))
        g = int(group_labels[ci])
        size = max(node_r, node_r * (usage[ci] / max_usage) * 2.5)
        node_pos[ci] = (cx_n, cy_n)
        node_sizes[ci] = size

        if ci in visited:
            fill = colors[ci]
            fill_op = "1"
            stroke_col = "black"
            stroke_w = 1.0
            if ci == first_cluster:
                stroke_col = "lime"
                stroke_w = 3.0
            elif ci == last_cluster:
                stroke_col = "red"
                stroke_w = 3.0
        else:
            fill = "#ccc"
            fill_op = "0.2"
            stroke_col = "#aaa"
            stroke_w = 0.5

        s.append(
            f'<circle class="sm-node" id="live-node-{ci}" '
            f'data-cluster="{ci}" data-group="{g}" '
            f'cx="{cx_n:.1f}" cy="{cy_n:.1f}" r="{size:.1f}" '
            f'fill="{fill}" fill-opacity="{fill_op}" '
            f'stroke="{stroke_col}" stroke-width="{stroke_w:.1f}" '
            f'data-default-stroke="{stroke_col}" data-default-sw="{stroke_w:.1f}"/>'
        )

        if ci in visited:
            r_c, g_c, b_c = mcolors.to_rgb(colors[ci])
            lum = 0.299 * r_c + 0.587 * g_c + 0.114 * b_c
            txt_col = "#000" if lum > 0.5 else "#fff"
            s.append(
                f'<text x="{cx_n:.1f}" y="{cy_n:.1f}" text-anchor="middle" '
                f'dominant-baseline="central" font-size="{label_fs}" '
                f'font-weight="bold" fill="{txt_col}">{ci}</text>'
            )
            order = first_visit_order[ci]
            s.append(
                f'<text x="{cx_n:.1f}" y="{cy_n + size + 10:.1f}" '
                f'text-anchor="middle" font-size="7" fill="gray">'
                f'#{order + 1} ({visit_count[ci]}x)</text>'
            )

    s.append('</svg>')

    metadata = {
        "node_pos": node_pos,
        "node_r": node_r,
        "node_sizes": node_sizes,
    }
    return "\n".join(s), metadata


def plot_state_machine_exit(
    P_exit_micro: np.ndarray,
    P_exit: np.ndarray,
    group_labels: np.ndarray,
    usage: np.ndarray,
    K_coarse: int,
    out_path: Path,
    title: str = "",
    node_scale: float = 2000.0,
    macro_edge_scale: float = 8.0,
    micro_edge_scale: float = 3.0,
    macro_alpha: float = 0.15,
    micro_top_r: int = 3,
    micro_min_prob: float = 0.03,
    macro_top_r: int = 3,
    macro_min_prob: float = 0.03,
    P_self_loop: np.ndarray | None = None,
) -> None:
    """Hierarchical state machine with exit probabilities at both micro and macro level.

    Both micro and macro edges use conditional exit probabilities:
    P_exit[i,j] = P[i,j] / (1 - P[i,i]) for j!=i.

    Micro edges are drawn both within and across groups. Dwell time
    (self-loop probability) is annotated next to each node.

    Args:
        P_exit_micro: (K, K) micro-level exit matrix (diagonal = 0, rows sum to ~1).
        P_exit: (K_coarse, K_coarse) macro conditional exit matrix.
        group_labels: (K,) micro -> macro group mapping (0-indexed).
        usage: (K,) cluster frequency array.
        K_coarse: number of macro groups.
        out_path: output file path.
        micro_top_r: top-r edges to keep per micro node.
        micro_min_prob: minimum exit probability threshold for micro edges.
        macro_top_r: top-r edges to keep per macro group.
        macro_min_prob: minimum exit probability threshold for macro edges.
        P_self_loop: (K,) diagonal of P_weighted (self-loop probabilities).
    """
    K = P_exit_micro.shape[0]
    colors = _cluster_colors(K, group_labels)

    layout = _compute_state_machine_layout(
        P_exit_micro, P_exit, group_labels, usage, K_coarse,
        micro_top_r, micro_min_prob, macro_top_r, macro_min_prob,
    )
    macro_pos = layout["macro_pos"]
    micro_pos = layout["micro_pos"]
    bubble_radii = layout["bubble_radii"]
    groups = layout["groups"]
    group_usage = layout["group_usage"]
    P_micro_pruned = layout["P_micro_pruned"]
    P_exit_pruned = layout["P_exit_pruned"]

    # --- Draw ---
    fig_w = max(16, K_coarse * 4.0)
    fig_h = max(12, K_coarse * 3.2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # 1. Macro bubbles
    macro_colors = _group_colors(K_coarse)
    for g in range(K_coarse):
        cx, cy = macro_pos[g]
        circle = plt.Circle(
            (cx, cy), bubble_radii[g],
            color=macro_colors[g],
            alpha=macro_alpha, zorder=0,
        )
        ax.add_patch(circle)
        border = plt.Circle(
            (cx, cy), bubble_radii[g],
            fill=False, edgecolor="gray", linewidth=1.5, linestyle="--", zorder=1,
        )
        ax.add_patch(border)
        ax.text(
            cx, cy + bubble_radii[g] + 0.08,
            f"G{g}",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
            color="dimgray",
        )

    # 2. Micro exit edges (both intra-group and cross-group)
    for i in range(K):
        for j in range(K):
            if i != j and P_micro_pruned[i, j] > 0:
                x1, y1 = micro_pos[i]
                x2, y2 = micro_pos[j]
                w = P_micro_pruned[i, j]
                same_group = group_labels[i] == group_labels[j]
                ax.annotate(
                    "", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(
                        arrowstyle="-|>",
                        color="gray" if same_group else "darkgray",
                        lw=max(0.5, w * micro_edge_scale),
                        alpha=0.5 if same_group else 0.35,
                        connectionstyle="arc3,rad=0.15",
                        shrinkA=10, shrinkB=10,
                        linestyle="solid" if same_group else "dashed",
                    ),
                    zorder=2,
                )

    # 3. Inter-group arrows using conditional exit probabilities
    # Helper: minimum distance from point p to line segment (a, b)
    def _point_seg_dist(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        ab = b - a
        ab2 = np.dot(ab, ab)
        if ab2 < 1e-12:
            return float(np.linalg.norm(p - a))
        t = np.clip(np.dot(p - a, ab) / ab2, 0.0, 1.0)
        proj = a + t * ab
        return float(np.linalg.norm(p - proj))

    for g1 in range(K_coarse):
        for g2 in range(K_coarse):
            if g1 != g2 and P_exit_pruned[g1, g2] > 0:
                c1 = np.array(macro_pos[g1])
                c2 = np.array(macro_pos[g2])
                d = np.linalg.norm(c2 - c1)
                if d < 1e-8:
                    continue
                direction = (c2 - c1) / d
                start = c1 + direction * (bubble_radii[g1] + 0.05)
                end = c2 - direction * (bubble_radii[g2] + 0.05)

                # Avoid arrows crossing intermediate bubbles
                arc_rad = 0.15
                for g_other in range(K_coarse):
                    if g_other in (g1, g2):
                        continue
                    c_other = np.array(macro_pos[g_other])
                    dist_to_seg = _point_seg_dist(c_other, start, end)
                    if dist_to_seg < bubble_radii[g_other] + 0.1:
                        arc_rad = max(arc_rad, 0.45)
                        seg_vec = end - start
                        to_obs = c_other - start
                        cross = seg_vec[0] * to_obs[1] - seg_vec[1] * to_obs[0]
                        arc_rad = arc_rad if cross < 0 else -arc_rad
                        break

                w = P_exit_pruned[g1, g2]
                ax.annotate(
                    "", xy=end, xytext=start,
                    arrowprops=dict(
                        arrowstyle="-|>",
                        color="darkblue",
                        lw=max(2.0, w * macro_edge_scale),
                        alpha=0.8,
                        connectionstyle=f"arc3,rad={arc_rad}",
                    ),
                    zorder=4,
                )
                mid = (start + end) / 2
                ax.text(
                    mid[0], mid[1], f"{w:.0%}",
                    fontsize=8, fontweight="bold",
                    color="darkblue", alpha=0.9,
                    ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                              alpha=0.7, edgecolor="none"),
                    zorder=5,
                )

    # 4. Draw micro nodes + dwell annotations
    min_node_size = max(300, 5000 / max(K, 1))
    label_fontsize = max(7, min(11, 250 // max(K, 1)))
    # Estimate node visual radius in data coords for dwell text offset
    xlim_range = max(micro_pos[i][0] for i in range(K)) - min(micro_pos[i][0] for i in range(K))
    node_offset = max(0.08, xlim_range * 0.02)
    for i in range(K):
        x, y = micro_pos[i]
        size = max(min_node_size, usage[i] * node_scale)
        self_p = P_self_loop[i] if P_self_loop is not None else 0.0
        border_w = 0.5 + self_p * 1.5
        ax.scatter(
            x, y, s=size,
            c=colors[i], edgecolors="black",
            linewidths=border_w, zorder=3,
        )
        r, g_val, b = mcolors.to_rgb(colors[i])
        lum = 0.299 * r + 0.587 * g_val + 0.114 * b
        ax.text(
            x, y, str(i),
            ha="center", va="center",
            fontsize=label_fontsize,
            fontweight="bold",
            color="black" if lum > 0.5 else "white",
            zorder=6,
        )
        # Dwell / self-loop annotation
        if P_self_loop is not None:
            ax.text(
                x, y - node_offset, f"s={P_self_loop[i]:.2f}",
                fontsize=5, color="gray", ha="center", va="top",
                zorder=6,
            )

    ax.set_title(title if title else "Hierarchical State Machine (Exit Probabilities)",
                 fontsize=13)
    ax.axis("off")
    ax.set_aspect("equal")

    all_x = [micro_pos[i][0] for i in range(K)]
    all_y = [micro_pos[i][1] for i in range(K)]
    max_radius = max(bubble_radii.values()) if bubble_radii else 1.0
    pad = max_radius + 0.5
    ax.set_xlim(min(all_x) - pad, max(all_x) + pad)
    ax.set_ylim(min(all_y) - pad - 0.3, max(all_y) + pad + 0.3)

    ax.text(
        0.01, 0.01,
        "Blue arrows = P(target group | leaving source group)\n"
        "Gray arrows = micro exit P (solid=intra-group, dashed=cross-group)\n"
        "s=X.XX = self-loop probability (dwell) | Node size = frequency",
        transform=ax.transAxes, fontsize=7, alpha=0.6,
        va="bottom",
    )

    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# Colored-text HTML visualization
# ---------------------------------------------------------------------------


