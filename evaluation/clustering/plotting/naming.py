"""Cluster-naming input/output helpers (LLM-driven naming pipeline)."""
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

def _extract_segments(
    tokens_per_doc: list[list[str]],
    trajs_per_doc: list[np.ndarray],
    context_tokens: int,
    cos_per_doc: list[np.ndarray] | None = None,
    margin_per_doc: list[np.ndarray] | None = None,
) -> list[tuple]:
    """Run-length encode all trajectories and collect segments with context.

    Returns list of (cluster_id, doc_idx, start, end, seg_len,
                      pre_tokens, seg_tokens, post_tokens,
                      mean_margin, mean_cos).

    When *cos_per_doc* / *margin_per_doc* are ``None``, the last two
    tuple elements are ``None``.
    """
    segments = []
    for doc_idx, (toks, traj) in enumerate(zip(tokens_per_doc, trajs_per_doc)):
        n = len(traj)
        if n == 0:
            continue
        # Per-doc margin / cos arrays (may be None)
        doc_margin = (
            margin_per_doc[doc_idx]
            if margin_per_doc is not None and doc_idx < len(margin_per_doc)
            else None
        )
        doc_cos = (
            cos_per_doc[doc_idx]
            if cos_per_doc is not None and doc_idx < len(cos_per_doc)
            else None
        )
        i = 0
        while i < n:
            c = int(traj[i])
            j = i + 1
            while j < n and int(traj[j]) == c:
                j += 1
            seg_len = j - i
            pre = toks[max(0, i - context_tokens): i]
            seg = toks[i: j]
            post = toks[j: j + context_tokens]
            mean_margin = (
                float(np.mean(doc_margin[i:j]))
                if doc_margin is not None
                else None
            )
            mean_cos = (
                float(np.mean(doc_cos[i:j]))
                if doc_cos is not None
                else None
            )
            segments.append(
                (c, doc_idx, i, j - 1, seg_len, pre, seg, post,
                 mean_margin, mean_cos)
            )
            i = j
    return segments


def _select_snippets(
    segments: list[tuple],
    cluster_id: int,
    max_snippets: int,
    min_seg_len: int,
    sort_by: str = "length",
    min_margin: float | None = None,
    max_margin: float | None = None,
) -> list[tuple]:
    """Select diverse, high-quality segments for a given cluster.

    Args:
        sort_by: ``"margin"`` sorts by mean_margin descending (most
            prototypical first); ``"length"`` sorts by segment length
            descending (original behaviour).  Segments whose mean_margin
            is ``None`` are placed last when sorting by margin.
        min_margin: if set, exclude segments with ``mean_margin < min_margin``.
            Segments whose mean_margin is ``None`` are kept (no data to filter on).
        max_margin: if set, exclude segments with ``mean_margin >= max_margin``.
            Segments whose mean_margin is ``None`` are kept.
    """
    cluster_segs = [s for s in segments if s[0] == cluster_id and s[4] >= min_seg_len]
    if min_margin is not None:
        cluster_segs = [
            s for s in cluster_segs if s[8] is None or s[8] >= min_margin
        ]
    if max_margin is not None:
        cluster_segs = [
            s for s in cluster_segs if s[8] is None or s[8] < max_margin
        ]
    if sort_by == "margin":
        # Sort by mean_margin descending; None values sink to the end.
        cluster_segs.sort(
            key=lambda s: -(s[8] if s[8] is not None else -float("inf"))
        )
    else:
        cluster_segs.sort(key=lambda s: -s[4])  # longest first
    # Greedily diversify across documents
    selected: list[tuple] = []
    seen_docs: set[int] = set()
    # First pass: one per unseen doc (best margin from each doc)
    for seg in cluster_segs:
        if len(selected) >= max_snippets:
            break
        if seg[1] not in seen_docs:
            selected.append(seg)
            seen_docs.add(seg[1])
    # Second pass: prefer docs with fewest selections so far
    if len(selected) < max_snippets:
        from collections import Counter
        doc_counts = Counter(s[1] for s in selected)
        remaining = [s for s in cluster_segs if s not in selected]
        remaining.sort(key=lambda s: (doc_counts.get(s[1], 0), -(s[8] if s[8] is not None else -float("inf"))))
        for seg in remaining:
            if len(selected) >= max_snippets:
                break
            selected.append(seg)
            doc_counts[seg[1]] = doc_counts.get(seg[1], 0) + 1
    return selected


def _select_borderline_snippets(
    segments: list[tuple],
    cluster_id: int,
    max_snippets: int = 3,
    min_seg_len: int = 2,
) -> list[tuple]:
    """Select the lowest-margin segments for *cluster_id* (boundary cases).

    Only segments with a non-None ``mean_margin`` (index 8) are eligible.
    Returns up to *max_snippets* segments sorted by ascending margin
    with document diversity.
    """
    eligible = [
        s for s in segments
        if s[0] == cluster_id and s[4] >= min_seg_len and s[8] is not None
    ]
    eligible.sort(key=lambda s: s[8])  # lowest margin first
    selected: list[tuple] = []
    seen_docs: set[int] = set()
    for seg in eligible:
        if len(selected) >= max_snippets:
            break
        if seg[1] not in seen_docs:
            selected.append(seg)
            seen_docs.add(seg[1])
    if len(selected) < max_snippets:
        for seg in eligible:
            if len(selected) >= max_snippets:
                break
            if seg not in selected:
                selected.append(seg)
    return selected


def write_cluster_naming_input(
    tokens_per_doc: list[list[str]],
    trajs_per_doc: list[np.ndarray],
    K: int,
    usage: np.ndarray,
    dwell_stats: dict,
    out_path: Path,
    variant: str,
    window_size: int,
    seed: int,
    num_docs: int,
    max_snippets: int | None = None,
    context_tokens: int = 12,
    min_seg_len: int = 2,
    cos_per_doc: list[np.ndarray] | None = None,
    margin_per_doc: list[np.ndarray] | None = None,
) -> None:
    """Write a self-contained plain-text file for an LLM to name cluster centers.

    The file has three sections:
      1. Task instructions and expected output format
      2. Cluster overview table (frequency, dwell stats, optionally margin/cos)
      3. Per-cluster evidence snippets (segments with context)

    Args:
        tokens_per_doc: list of per-document token string lists.
        trajs_per_doc: list of per-document cluster label arrays.
        K: number of clusters.
        usage: (K,) cluster usage fractions from compute_cluster_usage.
        dwell_stats: dict from compute_dwell_times.
        out_path: path to write the .txt file.
        variant: representation variant name (e.g. "z", "h", "h_pca64").
        window_size: causal window size used.
        seed: k-means seed.
        num_docs: number of documents processed.
        max_snippets: snippets per cluster (default: auto based on K).
        context_tokens: tokens of context before/after each segment.
        min_seg_len: minimum segment length to include.
        cos_per_doc: optional list of per-document top-1 cosine similarity
            arrays (same shape as trajs_per_doc entries).
        margin_per_doc: optional list of per-document cosine margin arrays.
    """
    if max_snippets is None:
        max_snippets = 100

    has_margin = margin_per_doc is not None
    sort_by = "margin" if has_margin else "length"

    total_tokens = sum(len(t) for t in trajs_per_doc)

    # Compute global mean dwell for annotation
    all_dwells = dwell_stats["all_dwells"]
    global_mean_dwell = float(all_dwells.mean()) if len(all_dwells) > 0 else 1.0

    lines: list[str] = []

    # ---- Section 1: Task instructions ----
    lines.append("=" * 80)
    lines.append("TASK: NAME AND DESCRIBE CLUSTER CENTERS")
    lines.append("=" * 80)
    lines.append("")
    lines.append(
        f"You are analyzing K-means cluster assignments over token sequences from an LLM."
    )
    lines.append(
        f"There are {K} clusters. "
        f"Tokens from {num_docs} documents ({total_tokens:,} total tokens) "
        f"were embedded and assigned to one of these {K} clusters."
    )
    lines.append("")
    lines.append(
        "Your job is to infer what each cluster captures — what linguistic, semantic, "
        "or structural function the tokens in that cluster share — and assign a short "
        "name plus a one-sentence description."
    )
    lines.append("")
    lines.append("DATA FORMAT")
    lines.append("-" * 40)
    lines.append(
        "Section 2 (CLUSTER OVERVIEW) gives global statistics for every cluster."
    )
    lines.append(
        "Section 3 (CLUSTER EVIDENCE) groups actual token evidence by cluster."
    )
    lines.append(
        "  Each cluster shows text snippets extracted from real documents."
    )
    lines.append(
        "  Each snippet is a SEGMENT — a maximal run of consecutive tokens all "
        "assigned to that cluster."
    )
    lines.append(
        "  Surrounding context is shown as [PRE] ... and ... [POST]."
    )
    lines.append(
        "  The segment itself is wrapped in >> ... <<"
    )
    if has_margin:
        lines.append(
            "  Snippets are ranked by cosine margin (distance from cluster boundary)."
        )
        lines.append(
            "  HIGH CONF = strong cluster member; BORDERLINE = near decision boundary."
        )
    lines.append("")
    lines.append("OUTPUT FORMAT")
    lines.append("-" * 40)
    lines.append(f"For each cluster 0..{K-1}, produce exactly one line:")
    lines.append("  C<id>: <short_name> | <one_sentence_description>")
    lines.append("")
    lines.append("Example:")
    lines.append(
        "  C0: numeric_expressions | Tokens forming standalone numbers, indices, "
        "or mathematical constants."
    )
    lines.append(
        "  C1: narrative_connectives | Transition words and phrases linking "
        "sentences or reasoning steps."
    )
    lines.append("")
    lines.append("")

    # ---- Section 2: Cluster overview table ----
    lines.append("=" * 80)
    lines.append(
        f"SECTION 2: CLUSTER OVERVIEW  (K={K}, total_tokens={total_tokens:,}, "
        f"num_docs={num_docs})"
    )
    lines.append("=" * 80)
    lines.append("")

    lines.extend(
        _format_overview_table(
            K, usage, dwell_stats, total_tokens, global_mean_dwell,
            cos_per_doc=cos_per_doc,
            margin_per_doc=margin_per_doc,
            trajs_per_doc=trajs_per_doc,
        )
    )

    lines.append("")
    lines.append("")

    # ---- Section 3: Per-cluster evidence ----
    lines.append("=" * 80)
    lines.append("SECTION 3: CLUSTER EVIDENCE")
    lines.append("=" * 80)
    lines.append("")

    per_cluster_dwells = dwell_stats["per_cluster_dwells"]
    mean_dwell_arr = dwell_stats["mean_dwell_per_cluster"]

    segments = _extract_segments(
        tokens_per_doc, trajs_per_doc, context_tokens,
        cos_per_doc=cos_per_doc, margin_per_doc=margin_per_doc,
    )

    # Order clusters by mean dwell descending (most coherent first)
    cluster_order = sorted(
        range(K),
        key=lambda c: -(mean_dwell_arr[c] if not np.isnan(mean_dwell_arr[c]) else 0.0),
    )

    for c in cluster_order:
        m_dwell = mean_dwell_arr[c] if not np.isnan(mean_dwell_arr[c]) else 0.0
        freq_pct = usage[c] * 100
        n_runs = len(per_cluster_dwells[c])
        lines.append(
            f"--- CLUSTER {c}  (mean_dwell={m_dwell:.1f}, freq={freq_pct:.1f}%, "
            f"{n_runs} runs) ---"
        )
        lines.append("")

        snippets = _select_snippets(
            segments, c, max_snippets, min_seg_len, sort_by=sort_by,
        )

        if not snippets:
            lines.append("  (no segments of length >= 2 found)")
            lines.append("")
            continue

        for seg in snippets:
            lines.extend(_format_snippet_block(seg))

        # Borderline tier — only when margin data is available
        if has_margin:
            borderline = _select_borderline_snippets(
                segments, c, max_snippets=50, min_seg_len=min_seg_len,
            )
            if borderline:
                lines.append(
                    "  BORDERLINE EXAMPLES (lowest confidence for this cluster):"
                )
                lines.append("")
                for seg in borderline:
                    lines.extend(_format_snippet_block(seg))

        lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved {out_path}")


def _format_snippet_block(seg: tuple) -> list[str]:
    """Format a single segment tuple into text lines for cluster naming files.

    Handles both legacy 8-tuples and extended 10-tuples that carry
    ``mean_margin`` and ``mean_cos``.
    """
    # Unpack — support both old 8-tuples and new 10-tuples
    _, doc_idx, start, end, seg_len, pre, seg_toks, post = seg[:8]
    mean_margin = seg[8] if len(seg) > 8 else None
    mean_cos = seg[9] if len(seg) > 9 else None

    block: list[str] = []

    # Build header line dynamically
    header_parts = [f"doc={doc_idx}", f"tokens={start}..{end}", f"seg_len={seg_len}"]
    if mean_margin is not None:
        header_parts.append(f"mean_margin={mean_margin:.2f}")
    if mean_cos is not None:
        header_parts.append(f"mean_cos={mean_cos:.2f}")
    header = "[" + ", ".join(header_parts) + "]"

    # Confidence annotation
    if mean_margin is not None:
        if mean_margin > 0.15:
            header += "  <- HIGH CONF"
        elif mean_margin < 0.05:
            header += "  <- BORDERLINE"

    block.append(header)
    pre_text = "".join(pre)
    seg_text = "".join(seg_toks)
    post_text = "".join(post)
    if pre_text:
        block.append(f"[PRE]  ...{pre_text}")
    block.append(f">>{seg_text}<<")
    if post_text:
        block.append(f"[POST]  {post_text}...")
    block.append("")
    return block


def _format_overview_table(
    K: int,
    usage: np.ndarray,
    dwell_stats: dict,
    total_tokens: int,
    global_mean_dwell: float,
    cos_per_doc: list[np.ndarray] | None = None,
    margin_per_doc: list[np.ndarray] | None = None,
    trajs_per_doc: list[np.ndarray] | None = None,
) -> list[str]:
    """Format the cluster overview table lines (shared by overview + original).

    When *cos_per_doc*, *margin_per_doc*, and *trajs_per_doc* are all
    provided, two extra columns (``mean_margin``, ``mean_cos``) are
    appended to each row.
    """
    per_cluster_dwells = dwell_stats["per_cluster_dwells"]
    mean_dwell_arr = dwell_stats["mean_dwell_per_cluster"]

    # Pre-compute per-cluster margin/cos means if data is available
    has_margin_data = (
        cos_per_doc is not None
        and margin_per_doc is not None
        and trajs_per_doc is not None
    )
    cluster_mean_margin: dict[int, float] = {}
    cluster_mean_cos: dict[int, float] = {}
    if has_margin_data:
        # Accumulate per-cluster sums
        cluster_margin_sum: dict[int, float] = {k: 0.0 for k in range(K)}
        cluster_cos_sum: dict[int, float] = {k: 0.0 for k in range(K)}
        cluster_token_count: dict[int, int] = {k: 0 for k in range(K)}
        for doc_idx, traj in enumerate(trajs_per_doc):
            doc_m = margin_per_doc[doc_idx] if doc_idx < len(margin_per_doc) else None
            doc_c = cos_per_doc[doc_idx] if doc_idx < len(cos_per_doc) else None
            if doc_m is None or doc_c is None:
                continue
            for t_idx in range(len(traj)):
                cid = int(traj[t_idx])
                if cid < K and t_idx < len(doc_m) and t_idx < len(doc_c):
                    cluster_margin_sum[cid] += float(doc_m[t_idx])
                    cluster_cos_sum[cid] += float(doc_c[t_idx])
                    cluster_token_count[cid] += 1
        for k in range(K):
            if cluster_token_count[k] > 0:
                cluster_mean_margin[k] = cluster_margin_sum[k] / cluster_token_count[k]
                cluster_mean_cos[k] = cluster_cos_sum[k] / cluster_token_count[k]
            else:
                cluster_mean_margin[k] = 0.0
                cluster_mean_cos[k] = 0.0

    lines: list[str] = []
    base_header = (
        f"{'ID':>4}  {'freq%':>6}  {'count':>8}  {'mean_dwell':>10}  "
        f"{'med_dwell':>9}  {'num_runs':>8}"
    )
    if has_margin_data:
        header = base_header + f"  {'mean_margin':>11}  {'mean_cos':>8}"
    else:
        header = base_header
    lines.append(header)
    lines.append("-" * len(header))

    for k in range(K):
        freq_pct = usage[k] * 100
        count = int(round(usage[k] * total_tokens))
        m_dwell = mean_dwell_arr[k] if not np.isnan(mean_dwell_arr[k]) else 0.0
        med_dwell = (
            float(np.median(per_cluster_dwells[k]))
            if len(per_cluster_dwells[k]) > 0
            else 0.0
        )
        n_runs = len(per_cluster_dwells[k])
        annotation = ""
        if m_dwell > 2 * global_mean_dwell:
            annotation = "  <-- LONG DWELL"
        row = (
            f"{k:>4}  {freq_pct:>5.1f}%  {count:>8}  {m_dwell:>10.2f}  "
            f"{med_dwell:>9.1f}  {n_runs:>8}"
        )
        if has_margin_data:
            row += f"  {cluster_mean_margin[k]:>11.4f}  {cluster_mean_cos[k]:>8.4f}"
        lines.append(row + annotation)
    return lines


def _compute_group_stats(
    group_labels: np.ndarray,
    K: int,
    K_coarse: int,
    usage: np.ndarray,
    dwell_stats: dict,
    cluster_mean_margin: dict[int, float | None],
) -> dict[int, dict]:
    """Compute per-group aggregate statistics for naming prompts.

    Returns dict keyed by group id with keys:
        clusters: sorted list of cluster ids in this group
        agg_usage: sum of usage fractions
        mean_margin: average of per-cluster mean margins (None if unavailable)
        mean_dwell: average of per-cluster mean dwells
    """
    mean_dwell_arr = dwell_stats["mean_dwell_per_cluster"]
    stats: dict[int, dict] = {}
    for g in range(K_coarse):
        members = sorted(int(c) for c in range(K) if int(group_labels[c]) == g)
        agg_usage = float(sum(usage[c] for c in members)) if members else 0.0

        margin_vals = [
            cluster_mean_margin[c]
            for c in members
            if cluster_mean_margin.get(c) is not None
        ]
        mean_margin = (
            float(np.mean(margin_vals)) if margin_vals else None
        )

        dwell_vals = [
            float(mean_dwell_arr[c])
            for c in members
            if not np.isnan(mean_dwell_arr[c])
        ]
        mean_dwell = float(np.mean(dwell_vals)) if dwell_vals else 0.0

        stats[g] = {
            "clusters": members,
            "agg_usage": agg_usage,
            "mean_margin": mean_margin,
            "mean_dwell": mean_dwell,
        }
    return stats


def write_cluster_naming_overview(
    tokens_per_doc: list[list[str]],
    trajs_per_doc: list[np.ndarray],
    K: int,
    usage: np.ndarray,
    dwell_stats: dict,
    out_dir: Path,
    variant: str,
    window_size: int,
    seed: int,
    num_docs: int,
    overview_snippets: int = 5,
    context_tokens: int = 12,
    min_seg_len: int = 2,
    cos_per_doc: list[np.ndarray] | None = None,
    margin_per_doc: list[np.ndarray] | None = None,
) -> None:
    """Write a compact overview file for all clusters (for contrastive naming).

    Writes ``out_dir/overview.txt`` with task instructions, a stats table,
    and a small number of snippets per cluster so the LLM can see all
    clusters at a glance.

    When *cos_per_doc* and *margin_per_doc* are provided, snippets are
    ranked by cosine margin (most prototypical first) and a borderline
    tier is appended after the main snippets for each cluster.
    """
    has_margin = margin_per_doc is not None
    sort_by = "margin" if has_margin else "length"

    total_tokens = sum(len(t) for t in trajs_per_doc)
    all_dwells = dwell_stats["all_dwells"]
    global_mean_dwell = float(all_dwells.mean()) if len(all_dwells) > 0 else 1.0
    per_cluster_dwells = dwell_stats["per_cluster_dwells"]
    mean_dwell_arr = dwell_stats["mean_dwell_per_cluster"]

    lines: list[str] = []

    # ---- Section 1: Task instructions ----
    lines.append("=" * 80)
    lines.append("TASK: NAME AND DESCRIBE CLUSTER CENTERS")
    lines.append("=" * 80)
    lines.append("")
    lines.append(
        f"You are analyzing K-means cluster assignments over token sequences from an LLM."
    )
    lines.append(
        f"There are {K} clusters. "
        f"Tokens from {num_docs} documents ({total_tokens:,} total tokens) "
        f"were embedded and assigned to one of these {K} clusters."
    )
    lines.append("")
    lines.append(
        "Your job is to infer what each cluster captures — what linguistic, semantic, "
        "or structural function the tokens in that cluster share — and assign a short, "
        "DISTINCTIVE name plus a one-sentence description."
    )
    lines.append("")
    lines.append(
        "IMPORTANT: Names must be CONTRASTIVE — each cluster name should clearly "
        "distinguish it from all other clusters. Avoid generic labels."
    )
    lines.append("")
    lines.append("DATA FORMAT")
    lines.append("-" * 40)
    lines.append(
        "This OVERVIEW file gives global statistics and a few example snippets per cluster."
    )
    lines.append(
        f"Accompanying detail files (cluster_C00.txt .. cluster_C{K-1:02d}.txt) in the "
        f"same directory provide 15-20 longer snippets per cluster for deeper evidence."
    )
    lines.append(
        "Each snippet is a SEGMENT — a maximal run of consecutive tokens all "
        "assigned to that cluster. Context: [PRE] ... and ... [POST]. "
        "The segment itself is wrapped in >> ... <<"
    )
    if has_margin:
        lines.append(
            "Snippets are ranked by cosine margin (distance from cluster boundary)."
        )
        lines.append(
            "HIGH CONF = strong cluster member; BORDERLINE = near decision boundary."
        )
    lines.append("")
    lines.append("OUTPUT FORMAT")
    lines.append("-" * 40)
    lines.append(f"For each cluster 0..{K-1}, produce exactly one line:")
    lines.append("  C<id>: <short_name> | <one_sentence_description>")
    lines.append("")
    lines.append("Example:")
    lines.append(
        "  C0: numeric_expressions | Tokens forming standalone numbers, indices, "
        "or mathematical constants."
    )
    lines.append(
        "  C1: narrative_connectives | Transition words and phrases linking "
        "sentences or reasoning steps."
    )
    lines.append("")
    lines.append("")

    # ---- Section 2: Cluster overview table ----
    lines.append("=" * 80)
    lines.append(
        f"CLUSTER OVERVIEW  (K={K}, total_tokens={total_tokens:,}, "
        f"num_docs={num_docs})"
    )
    lines.append("=" * 80)
    lines.append("")
    lines.extend(
        _format_overview_table(
            K, usage, dwell_stats, total_tokens, global_mean_dwell,
            cos_per_doc=cos_per_doc,
            margin_per_doc=margin_per_doc,
            trajs_per_doc=trajs_per_doc,
        )
    )
    lines.append("")
    lines.append("")

    # ---- Section 3: Brief snippets per cluster ----
    lines.append("=" * 80)
    lines.append(f"CLUSTER SNIPPETS  ({overview_snippets} per cluster)")
    lines.append("=" * 80)
    lines.append("")

    segments = _extract_segments(
        tokens_per_doc, trajs_per_doc, context_tokens,
        cos_per_doc=cos_per_doc, margin_per_doc=margin_per_doc,
    )

    cluster_order = sorted(
        range(K),
        key=lambda c: -(mean_dwell_arr[c] if not np.isnan(mean_dwell_arr[c]) else 0.0),
    )

    for c in cluster_order:
        m_dwell = mean_dwell_arr[c] if not np.isnan(mean_dwell_arr[c]) else 0.0
        freq_pct = usage[c] * 100
        n_runs = len(per_cluster_dwells[c])
        lines.append(
            f"--- CLUSTER {c}  (mean_dwell={m_dwell:.1f}, freq={freq_pct:.1f}%, "
            f"{n_runs} runs) ---"
        )
        lines.append("")

        snippets = _select_snippets(
            segments, c, overview_snippets, min_seg_len, sort_by=sort_by,
        )
        if not snippets:
            lines.append("  (no segments of length >= 2 found)")
            lines.append("")
            continue

        for seg in snippets:
            lines.extend(_format_snippet_block(seg))

        # Borderline tier
        if has_margin:
            borderline = _select_borderline_snippets(
                segments, c, max_snippets=2, min_seg_len=min_seg_len,
            )
            if borderline:
                lines.append(
                    "  BORDERLINE EXAMPLES (lowest confidence for this cluster):"
                )
                lines.append("")
                for seg in borderline:
                    lines.extend(_format_snippet_block(seg))

        lines.append("")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "overview.txt"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved {out_path}")


def write_cluster_naming_details(
    tokens_per_doc: list[list[str]],
    trajs_per_doc: list[np.ndarray],
    K: int,
    usage: np.ndarray,
    dwell_stats: dict,
    out_dir: Path,
    max_snippets: int = 20,
    context_tokens: int = 12,
    min_seg_len: int = 2,
    cos_per_doc: list[np.ndarray] | None = None,
    margin_per_doc: list[np.ndarray] | None = None,
) -> None:
    """Write per-cluster detail files with exhaustive evidence snippets.

    Writes ``out_dir/cluster_C{k:02d}.txt`` for each cluster k, containing
    a header with stats and 15-20 long, diverse segments.

    When *cos_per_doc* and *margin_per_doc* are provided, snippets are
    ranked by cosine margin (most prototypical first) and per-cluster
    mean margin/cos statistics are included in the header.  A borderline
    tier (2-3 lowest-margin snippets) is appended after the main evidence.
    """
    has_margin = margin_per_doc is not None
    sort_by = "margin" if has_margin else "length"

    total_tokens = sum(len(t) for t in trajs_per_doc)
    per_cluster_dwells = dwell_stats["per_cluster_dwells"]
    mean_dwell_arr = dwell_stats["mean_dwell_per_cluster"]

    segments = _extract_segments(
        tokens_per_doc, trajs_per_doc, context_tokens,
        cos_per_doc=cos_per_doc, margin_per_doc=margin_per_doc,
    )

    # Pre-compute per-cluster margin/cos means for header stats
    cluster_mean_margin: dict[int, float | None] = {}
    cluster_mean_cos: dict[int, float | None] = {}
    if has_margin:
        for k in range(K):
            margin_vals: list[float] = []
            cos_vals: list[float] = []
            for doc_idx, traj in enumerate(trajs_per_doc):
                doc_m = margin_per_doc[doc_idx] if doc_idx < len(margin_per_doc) else None
                doc_c = (
                    cos_per_doc[doc_idx]
                    if cos_per_doc is not None and doc_idx < len(cos_per_doc)
                    else None
                )
                if doc_m is None:
                    continue
                for t_idx in range(len(traj)):
                    if int(traj[t_idx]) == k and t_idx < len(doc_m):
                        margin_vals.append(float(doc_m[t_idx]))
                        if doc_c is not None and t_idx < len(doc_c):
                            cos_vals.append(float(doc_c[t_idx]))
            cluster_mean_margin[k] = (
                float(np.mean(margin_vals)) if margin_vals else None
            )
            cluster_mean_cos[k] = (
                float(np.mean(cos_vals)) if cos_vals else None
            )

    out_dir.mkdir(parents=True, exist_ok=True)

    for k in range(K):
        lines: list[str] = []

        m_dwell = mean_dwell_arr[k] if not np.isnan(mean_dwell_arr[k]) else 0.0
        freq_pct = usage[k] * 100
        count = int(round(usage[k] * total_tokens))
        med_dwell = (
            float(np.median(per_cluster_dwells[k]))
            if len(per_cluster_dwells[k]) > 0
            else 0.0
        )
        n_runs = len(per_cluster_dwells[k])

        lines.append("=" * 60)
        lines.append(f"CLUSTER {k}  \u2014  DETAILED EVIDENCE")
        lines.append("=" * 60)
        lines.append("")
        lines.append(f"  freq     = {freq_pct:.1f}%  ({count:,} tokens)")
        lines.append(f"  mean_dwell = {m_dwell:.2f}")
        lines.append(f"  med_dwell  = {med_dwell:.1f}")
        lines.append(f"  num_runs   = {n_runs}")
        if has_margin and cluster_mean_margin.get(k) is not None:
            lines.append(f"  mean_margin = {cluster_mean_margin[k]:.4f}")
        if has_margin and cluster_mean_cos.get(k) is not None:
            lines.append(f"  mean_cos    = {cluster_mean_cos[k]:.4f}")
        lines.append("")
        lines.append("-" * 60)
        lines.append("")

        snippets = _select_snippets(
            segments, k, max_snippets, min_seg_len, sort_by=sort_by,
        )

        if not snippets:
            lines.append("  (no segments of length >= 2 found)")
        else:
            for seg in snippets:
                lines.extend(_format_snippet_block(seg))

        # Borderline tier
        if has_margin:
            borderline = _select_borderline_snippets(
                segments, k, max_snippets=3, min_seg_len=min_seg_len,
            )
            if borderline:
                lines.append(
                    "  BORDERLINE EXAMPLES (lowest confidence for this cluster):"
                )
                lines.append("")
                for seg in borderline:
                    lines.extend(_format_snippet_block(seg))

        lines.append("")

        out_path = out_dir / f"cluster_C{k:02d}.txt"
        out_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved {K} detail files to {out_dir}")


def write_cluster_group_naming(
    tokens_per_doc: list[list[str]],
    trajs_per_doc: list[np.ndarray],
    K: int,
    usage: np.ndarray,
    dwell_stats: dict,
    out_dir: Path,
    variant: str,
    window_size: int,
    seed: int,
    num_docs: int,
    group_labels: np.ndarray,
    K_coarse: int,
    snippets_per_cluster: int = 30,
    proto_count: int = 20,
    typical_count: int = 10,
    proto_min_margin: float = 0.15,
    typical_min_margin: float = 0.05,
    context_tokens: int = 12,
    min_seg_len: int = 2,
    cos_per_doc: list[np.ndarray] | None = None,
    margin_per_doc: list[np.ndarray] | None = None,
) -> None:
    """Write a group-aware naming prompt for an LLM to name both groups and clusters.

    Generates ``out_dir/group_cluster_naming.txt`` with:
      1. Task instructions (name groups + clusters together)
      2. Group structure table (which clusters belong to which group)
      3. Per-group cluster evidence with tiered snippets (PROTOTYPICAL + TYPICAL)

    When *margin_per_doc* is provided, snippets are split into confidence tiers
    and anything below *typical_min_margin* is excluded entirely.
    When *margin_per_doc* is ``None``, a single flat EXAMPLES tier is shown
    with *snippets_per_cluster* snippets sorted by length.

    Args:
        group_labels: (K,) array mapping each cluster to its macro-group id.
        K_coarse: number of macro-groups.
        snippets_per_cluster: total snippet budget per cluster (proto + typical).
        proto_count: max snippets in PROTOTYPICAL tier.
        typical_count: max snippets in TYPICAL tier.
        proto_min_margin: minimum margin for PROTOTYPICAL tier.
        typical_min_margin: minimum margin for TYPICAL tier (hard cutoff).
    """
    has_margin = margin_per_doc is not None
    total_tokens = sum(len(t) for t in trajs_per_doc)
    mean_dwell_arr = dwell_stats["mean_dwell_per_cluster"]

    # ---- Pre-compute per-cluster margin/cos means ----
    cluster_mean_margin: dict[int, float | None] = {k: None for k in range(K)}
    cluster_mean_cos: dict[int, float | None] = {k: None for k in range(K)}
    if has_margin:
        margin_sum: dict[int, float] = {k: 0.0 for k in range(K)}
        cos_sum: dict[int, float] = {k: 0.0 for k in range(K)}
        tok_count: dict[int, int] = {k: 0 for k in range(K)}
        for doc_idx, traj in enumerate(trajs_per_doc):
            doc_m = (
                margin_per_doc[doc_idx]
                if doc_idx < len(margin_per_doc) else None
            )
            doc_c = (
                cos_per_doc[doc_idx]
                if cos_per_doc is not None and doc_idx < len(cos_per_doc)
                else None
            )
            if doc_m is None:
                continue
            for t_idx in range(len(traj)):
                cid = int(traj[t_idx])
                if cid < K and t_idx < len(doc_m):
                    margin_sum[cid] += float(doc_m[t_idx])
                    tok_count[cid] += 1
                    if doc_c is not None and t_idx < len(doc_c):
                        cos_sum[cid] += float(doc_c[t_idx])
        for k in range(K):
            if tok_count[k] > 0:
                cluster_mean_margin[k] = margin_sum[k] / tok_count[k]
                cluster_mean_cos[k] = cos_sum[k] / tok_count[k]

    # ---- Group stats ----
    group_stats = _compute_group_stats(
        group_labels, K, K_coarse, usage, dwell_stats, cluster_mean_margin,
    )

    # ---- Extract segments once ----
    segments = _extract_segments(
        tokens_per_doc, trajs_per_doc, context_tokens,
        cos_per_doc=cos_per_doc, margin_per_doc=margin_per_doc,
    )

    # ---- Build text ----
    lines: list[str] = []

    # ==== Section 1: Task instructions ====
    lines.append("=" * 80)
    lines.append("TASK: NAME GROUPS AND CLUSTERS")
    lines.append("=" * 80)
    lines.append("")
    lines.append(
        "You are analyzing a two-level hierarchy of K-means clusters over LLM"
    )
    lines.append(
        "token representations."
    )
    lines.append("")
    lines.append(
        f"There are {K} clusters grouped into {K_coarse} macro-groups."
    )
    lines.append(
        f"Tokens from {num_docs} documents ({total_tokens:,} total "
        f"tokens) were embedded, clustered, and hierarchically grouped."
    )
    lines.append("")
    lines.append("HIERARCHY")
    lines.append("-" * 40)
    lines.append(
        "Clusters are grouped by a hybrid geometric + behavioural hierarchy."
    )
    lines.append(
        "Each GROUP is a set of clusters whose representations and transition"
    )
    lines.append("patterns are similar.")
    lines.append("")
    lines.append("YOUR TASK")
    lines.append("-" * 40)
    lines.append(
        "1. Name each GROUP: infer the overarching linguistic/semantic theme"
    )
    lines.append(
        "   shared by its constituent clusters. Group names should be broad"
    )
    lines.append("   categories.")
    lines.append(
        "2. Name each CLUSTER: infer the specific sub-function within its"
    )
    lines.append(
        "   group. Cluster names must be CONTRASTIVE — clearly distinguishable"
    )
    lines.append("   from sibling clusters in the same group.")
    lines.append("")
    lines.append("DATA FORMAT")
    lines.append("-" * 40)
    lines.append(
        "Section 2 shows the group structure table with aggregate statistics."
    )
    lines.append(
        "Section 3 shows evidence per group. Within each group, clusters are"
    )
    lines.append("listed in order of decreasing frequency.")
    lines.append(
        "Each snippet is a SEGMENT — a maximal run of consecutive tokens all"
    )
    lines.append(
        "assigned to that cluster. Context: [PRE] ... and ... [POST]."
    )
    lines.append("The segment itself is wrapped in >> ... <<")
    if has_margin:
        lines.append("")
        lines.append(
            "Each cluster has two tiers of examples:"
        )
        lines.append(
            f"  PROTOTYPICAL (margin >= {proto_min_margin}): These are the most"
        )
        lines.append(
            "    confident assignments. Weight these examples MOST heavily when"
        )
        lines.append("    inferring cluster meaning.")
        lines.append(
            f"  TYPICAL (margin {typical_min_margin}–{proto_min_margin}): Decent"
        )
        lines.append(
            "    examples with moderate confidence."
        )
        lines.append(
            f"  Examples below margin {typical_min_margin} are excluded entirely."
        )
    lines.append("")
    lines.append("OUTPUT FORMAT")
    lines.append("-" * 40)
    lines.append(
        f"First, output one line per group (G0..G{K_coarse - 1}):"
    )
    lines.append("  G<id>: <group_name> | <one_sentence_description_of_theme>")
    lines.append("")
    lines.append(
        f"Then, output one line per cluster (C0..C{K - 1}):"
    )
    lines.append(
        "  C<id> [G<group_id>]: <cluster_name> | <one_sentence_description>"
    )
    lines.append("")
    lines.append("Example:")
    lines.append(
        "  G0: syntactic_structure | Tokens involved in grammatical scaffolding "
        "and sentence boundaries."
    )
    lines.append(
        "  G1: semantic_content | Tokens carrying topical or narrative content "
        "information."
    )
    lines.append("")
    lines.append(
        "  C0 [G0]: sentence_boundaries | End-of-sentence punctuation and "
        "clause-final tokens."
    )
    lines.append(
        "  C1 [G0]: clause_connectives | Subordinating conjunctions and "
        "transition words starting dependent clauses."
    )
    lines.append(
        "  C2 [G1]: named_entities | Proper nouns — character names, places, "
        "and organisations."
    )
    lines.append(
        "  C3 [G1]: content_verbs | Main-clause action verbs in active voice."
    )
    lines.append("")
    lines.append("")

    # ==== Section 2: Group structure table ====
    lines.append("=" * 80)
    lines.append(
        f"SECTION 2: GROUP STRUCTURE TABLE  ({K_coarse} groups, K={K}, "
        f"total_tokens={total_tokens:,})"
    )
    lines.append("=" * 80)
    lines.append("")

    # Table header
    has_margin_col = any(
        group_stats[g]["mean_margin"] is not None for g in range(K_coarse)
    )
    hdr = f"{'GRP':>4}  {'CLUSTERS':<30}  {'AGG_FREQ%':>9}"
    if has_margin_col:
        hdr += f"  {'MEAN_MARGIN':>11}"
    hdr += f"  {'MEAN_DWELL':>10}"
    lines.append(hdr)
    lines.append("-" * len(hdr))

    # Groups sorted by aggregate usage descending
    group_order = sorted(
        range(K_coarse),
        key=lambda g: -group_stats[g]["agg_usage"],
    )

    for g in group_order:
        gs = group_stats[g]
        cluster_str = ", ".join(f"C{c}" for c in gs["clusters"])
        if len(cluster_str) > 28:
            cluster_str = cluster_str[:25] + "..."
        row = f"G{g:>3}  {cluster_str:<30}  {gs['agg_usage'] * 100:>8.1f}%"
        if has_margin_col:
            if gs["mean_margin"] is not None:
                row += f"  {gs['mean_margin']:>11.4f}"
            else:
                row += f"  {'n/a':>11}"
        row += f"  {gs['mean_dwell']:>10.2f}"
        lines.append(row)

    lines.append("")
    lines.append("")

    # ==== Section 3: Per-group cluster evidence ====
    lines.append("=" * 80)
    lines.append("SECTION 3: PER-GROUP CLUSTER EVIDENCE")
    lines.append("=" * 80)
    lines.append("")

    per_cluster_dwells = dwell_stats["per_cluster_dwells"]

    for g in group_order:
        gs = group_stats[g]
        lines.append("=" * 80)
        grp_hdr = (
            f"GROUP G{g}: {len(gs['clusters'])} clusters  "
            f"(agg_freq={gs['agg_usage'] * 100:.1f}%"
        )
        if gs["mean_margin"] is not None:
            grp_hdr += f", mean_margin={gs['mean_margin']:.3f}"
        grp_hdr += f", mean_dwell={gs['mean_dwell']:.1f})"
        lines.append(grp_hdr)
        lines.append("=" * 80)
        lines.append("")

        # Clusters within group, sorted by usage descending
        cluster_order = sorted(
            gs["clusters"], key=lambda c: -usage[c],
        )

        for c in cluster_order:
            m_dwell = (
                float(mean_dwell_arr[c])
                if not np.isnan(mean_dwell_arr[c]) else 0.0
            )
            freq_pct = usage[c] * 100
            n_runs = len(per_cluster_dwells[c])
            c_hdr = (
                f"--- CLUSTER C{c}  [G{g}]  "
                f"(freq={freq_pct:.1f}%, mean_dwell={m_dwell:.1f}"
            )
            if cluster_mean_margin[c] is not None:
                c_hdr += f", mean_margin={cluster_mean_margin[c]:.3f}"
            if cluster_mean_cos[c] is not None:
                c_hdr += f", mean_cos={cluster_mean_cos[c]:.3f}"
            c_hdr += f", {n_runs} runs) ---"
            lines.append(c_hdr)
            lines.append("")

            if has_margin:
                # PROTOTYPICAL tier
                proto_snippets = _select_snippets(
                    segments, c, proto_count, min_seg_len,
                    sort_by="margin",
                    min_margin=proto_min_margin,
                )
                if proto_snippets:
                    lines.append(
                        f"  PROTOTYPICAL EXAMPLES  "
                        f"(margin >= {proto_min_margin}, weight these most):"
                    )
                    lines.append("")
                    for seg in proto_snippets:
                        lines.extend(_format_snippet_block(seg))

                # TYPICAL tier
                typical_snippets = _select_snippets(
                    segments, c, typical_count, min_seg_len,
                    sort_by="margin",
                    min_margin=typical_min_margin,
                    max_margin=proto_min_margin,
                )
                if typical_snippets:
                    lines.append(
                        f"  TYPICAL EXAMPLES  "
                        f"(margin {typical_min_margin}–{proto_min_margin}):"
                    )
                    lines.append("")
                    for seg in typical_snippets:
                        lines.extend(_format_snippet_block(seg))

                if not proto_snippets and not typical_snippets:
                    lines.append(
                        f"  (no segments above margin {typical_min_margin} found)"
                    )
                    lines.append("")
            else:
                # No margin data — flat tier sorted by length
                flat_snippets = _select_snippets(
                    segments, c, snippets_per_cluster, min_seg_len,
                    sort_by="length",
                )
                if flat_snippets:
                    lines.append("  EXAMPLES:")
                    lines.append("")
                    for seg in flat_snippets:
                        lines.extend(_format_snippet_block(seg))
                else:
                    lines.append("  (no segments of length >= 2 found)")
                    lines.append("")

            lines.append("")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "group_cluster_naming.txt"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved {out_path}")


def _extract_segments_with_transitions(
    tokens_per_doc: list[list[str]],
    trajs_per_doc: list[np.ndarray],
    context_tokens: int,
    cos_per_doc: list[np.ndarray] | None = None,
    margin_per_doc: list[np.ndarray] | None = None,
) -> list[tuple]:
    """Like _extract_segments but appends prev_cluster and next_cluster.

    Returns 12-tuples: (cluster_id, doc_idx, start, end, seg_len,
                        pre_tokens, seg_tokens, post_tokens,
                        mean_margin, mean_cos, prev_cluster, next_cluster).

    prev_cluster / next_cluster are -1 at document boundaries.
    """
    segments = []
    for doc_idx, (toks, traj) in enumerate(zip(tokens_per_doc, trajs_per_doc)):
        n = len(traj)
        if n == 0:
            continue
        doc_margin = (
            margin_per_doc[doc_idx]
            if margin_per_doc is not None and doc_idx < len(margin_per_doc)
            else None
        )
        doc_cos = (
            cos_per_doc[doc_idx]
            if cos_per_doc is not None and doc_idx < len(cos_per_doc)
            else None
        )
        i = 0
        while i < n:
            c = int(traj[i])
            j = i + 1
            while j < n and int(traj[j]) == c:
                j += 1
            seg_len = j - i
            pre = toks[max(0, i - context_tokens): i]
            seg = toks[i: j]
            post = toks[j: j + context_tokens]
            mean_margin = (
                float(np.mean(doc_margin[i:j]))
                if doc_margin is not None else None
            )
            mean_cos = (
                float(np.mean(doc_cos[i:j]))
                if doc_cos is not None else None
            )
            prev_cluster = int(traj[i - 1]) if i > 0 else -1
            next_cluster = int(traj[j]) if j < n else -1
            segments.append(
                (c, doc_idx, i, j - 1, seg_len, pre, seg, post,
                 mean_margin, mean_cos, prev_cluster, next_cluster)
            )
            i = j
    return segments


def _format_snippet_block_v2(
    seg: tuple,
    group_labels: np.ndarray | None = None,
) -> list[str]:
    """Format a segment with transition context annotations.

    Like _format_snippet_block but appends [G_prev -> THIS -> G_next]
    when group_labels is provided and the segment has transition info.
    """
    _, doc_idx, start, end, seg_len, pre, seg_toks, post = seg[:8]
    mean_margin = seg[8] if len(seg) > 8 else None
    mean_cos = seg[9] if len(seg) > 9 else None
    prev_cluster = seg[10] if len(seg) > 10 else None
    next_cluster = seg[11] if len(seg) > 11 else None

    block: list[str] = []

    header_parts = [f"doc={doc_idx}", f"tokens={start}..{end}", f"seg_len={seg_len}"]
    if mean_margin is not None:
        header_parts.append(f"mean_margin={mean_margin:.2f}")
    if mean_cos is not None:
        header_parts.append(f"mean_cos={mean_cos:.2f}")
    header = "[" + ", ".join(header_parts) + "]"

    if mean_margin is not None:
        if mean_margin > 0.15:
            header += "  <- HIGH CONF"
        elif mean_margin < 0.05:
            header += "  <- BORDERLINE"

    # Transition context
    if prev_cluster is not None and group_labels is not None:
        prev_g = f"G{group_labels[prev_cluster]}" if prev_cluster >= 0 else "START"
        next_g = f"G{group_labels[next_cluster]}" if next_cluster >= 0 else "END"
        header += f"  [{prev_g} -> THIS -> {next_g}]"

    block.append(header)
    pre_text = "".join(pre)
    seg_text = "".join(seg_toks)
    post_text = "".join(post)
    if pre_text:
        block.append(f"[PRE]  ...{pre_text}")
    block.append(f">>{seg_text}<<")
    if post_text:
        block.append(f"[POST]  {post_text}...")
    block.append("")
    return block


def write_cluster_group_naming_v2(
    tokens_per_doc: list[list[str]],
    trajs_per_doc: list[np.ndarray],
    K: int,
    usage: np.ndarray,
    dwell_stats: dict,
    out_dir: Path,
    variant: str,
    window_size: int,
    seed: int,
    num_docs: int,
    group_labels: np.ndarray,
    K_coarse: int,
    snippets_per_cluster: int = 30,
    proto_count: int = 20,
    typical_count: int = 10,
    proto_min_margin: float = 0.15,
    typical_min_margin: float = 0.05,
    context_tokens: int = 20,
    min_seg_len: int = 2,
    cos_per_doc: list[np.ndarray] | None = None,
    margin_per_doc: list[np.ndarray] | None = None,
    reference_labels: dict[str, str] | None = None,
) -> None:
    """V2 naming prompt with transition context and optional reference labels.

    Like write_cluster_group_naming but:
      - Each snippet shows [G_prev -> THIS -> G_next] transition annotations
      - Default context_tokens=20 (up from 12)
      - Optional reference_labels dict pins known-good names in the prompt

    Outputs ``out_dir/group_cluster_naming.txt``.
    """
    has_margin = margin_per_doc is not None
    total_tokens = sum(len(t) for t in trajs_per_doc)
    mean_dwell_arr = dwell_stats["mean_dwell_per_cluster"]

    # ---- Pre-compute per-cluster margin/cos means ----
    cluster_mean_margin: dict[int, float | None] = {k: None for k in range(K)}
    cluster_mean_cos: dict[int, float | None] = {k: None for k in range(K)}
    if has_margin:
        margin_sum: dict[int, float] = {k: 0.0 for k in range(K)}
        cos_sum: dict[int, float] = {k: 0.0 for k in range(K)}
        tok_count: dict[int, int] = {k: 0 for k in range(K)}
        for doc_idx, traj in enumerate(trajs_per_doc):
            doc_m = (
                margin_per_doc[doc_idx]
                if doc_idx < len(margin_per_doc) else None
            )
            doc_c = (
                cos_per_doc[doc_idx]
                if cos_per_doc is not None and doc_idx < len(cos_per_doc)
                else None
            )
            if doc_m is None:
                continue
            for t_idx in range(len(traj)):
                cid = int(traj[t_idx])
                if cid < K and t_idx < len(doc_m):
                    margin_sum[cid] += float(doc_m[t_idx])
                    tok_count[cid] += 1
                    if doc_c is not None and t_idx < len(doc_c):
                        cos_sum[cid] += float(doc_c[t_idx])
        for k in range(K):
            if tok_count[k] > 0:
                cluster_mean_margin[k] = margin_sum[k] / tok_count[k]
                cluster_mean_cos[k] = cos_sum[k] / tok_count[k]

    # ---- Group stats ----
    group_stats = _compute_group_stats(
        group_labels, K, K_coarse, usage, dwell_stats, cluster_mean_margin,
    )

    # ---- Extract segments with transition info ----
    segments = _extract_segments_with_transitions(
        tokens_per_doc, trajs_per_doc, context_tokens,
        cos_per_doc=cos_per_doc, margin_per_doc=margin_per_doc,
    )

    # ---- Build text ----
    lines: list[str] = []

    # ==== Section 1: Task instructions ====
    lines.append("=" * 80)
    lines.append("TASK: NAME GROUPS AND CLUSTERS")
    lines.append("=" * 80)
    lines.append("")
    lines.append(
        "You are analyzing a two-level hierarchy of K-means clusters over LLM"
    )
    lines.append("token representations.")
    lines.append("")
    lines.append(
        f"There are {K} clusters grouped into {K_coarse} macro-groups."
    )
    lines.append(
        f"Tokens from {num_docs} documents ({total_tokens:,} total "
        f"tokens) were embedded, clustered, and hierarchically grouped."
    )
    lines.append("")
    lines.append("HIERARCHY")
    lines.append("-" * 40)
    lines.append(
        "Clusters are grouped by a hybrid geometric + behavioural hierarchy."
    )
    lines.append(
        "Each GROUP is a set of clusters whose representations and transition"
    )
    lines.append("patterns are similar.")
    lines.append("")
    lines.append("YOUR TASK")
    lines.append("-" * 40)
    lines.append(
        "1. Name each GROUP: infer the overarching linguistic/semantic theme"
    )
    lines.append(
        "   shared by its constituent clusters. Group names should be broad"
    )
    lines.append("   categories.")
    lines.append(
        "2. Name each CLUSTER: infer the specific sub-function within its"
    )
    lines.append(
        "   group. Cluster names must be CONTRASTIVE — clearly distinguishable"
    )
    lines.append("   from sibling clusters in the same group.")
    lines.append("")
    lines.append("DATA FORMAT")
    lines.append("-" * 40)
    lines.append(
        "Section 2 shows the group structure table with aggregate statistics."
    )
    lines.append(
        "Section 3 shows evidence per group. Within each group, clusters are"
    )
    lines.append("listed in order of decreasing frequency.")
    lines.append(
        "Each snippet is a SEGMENT — a maximal run of consecutive tokens all"
    )
    lines.append(
        "assigned to that cluster. Context: [PRE] ... and ... [POST]."
    )
    lines.append("The segment itself is wrapped in >> ... <<")
    lines.append("")
    lines.append("TRANSITION CONTEXT")
    lines.append("-" * 40)
    lines.append(
        "Each snippet header ends with [G_prev -> THIS -> G_next] showing"
    )
    lines.append(
        "which macro-group the preceding and following segments belong to."
    )
    lines.append(
        "This tells you the ROLE of this cluster in the reasoning flow."
    )
    lines.append(
        "Use this to distinguish clusters that contain similar text but"
    )
    lines.append("appear at different points in the reasoning trajectory.")
    if has_margin:
        lines.append("")
        lines.append("Each cluster has two tiers of examples:")
        lines.append(
            f"  PROTOTYPICAL (margin >= {proto_min_margin}): These are the most"
        )
        lines.append(
            "    confident assignments. Weight these examples MOST heavily when"
        )
        lines.append("    inferring cluster meaning.")
        lines.append(
            f"  TYPICAL (margin {typical_min_margin}–{proto_min_margin}): Decent"
        )
        lines.append("    examples with moderate confidence.")
        lines.append(
            f"  Examples below margin {typical_min_margin} are excluded entirely."
        )
    lines.append("")
    lines.append("OUTPUT FORMAT")
    lines.append("-" * 40)
    lines.append(
        f"First, output one line per group (G0..G{K_coarse - 1}):"
    )
    lines.append("  G<id>: <group_name> | <one_sentence_description_of_theme>")
    lines.append("")
    lines.append(
        f"Then, output one line per cluster (C0..C{K - 1}):"
    )
    lines.append(
        "  C<id> [G<group_id>]: <cluster_name> | <one_sentence_description>"
    )
    lines.append("")
    lines.append("Example:")
    lines.append(
        "  G0: syntactic_structure | Tokens involved in grammatical scaffolding "
        "and sentence boundaries."
    )
    lines.append(
        "  G1: semantic_content | Tokens carrying topical or narrative content "
        "information."
    )
    lines.append("")
    lines.append(
        "  C0 [G0]: sentence_boundaries | End-of-sentence punctuation and "
        "clause-final tokens."
    )
    lines.append(
        "  C1 [G0]: clause_connectives | Subordinating conjunctions and "
        "transition words starting dependent clauses."
    )
    lines.append(
        "  C2 [G1]: named_entities | Proper nouns — character names, places, "
        "and organisations."
    )
    lines.append(
        "  C3 [G1]: content_verbs | Main-clause action verbs in active voice."
    )
    lines.append("")

    # ==== Reference labels (optional) ====
    if reference_labels:
        lines.append("")
        lines.append("=" * 80)
        lines.append("REFERENCE LABELS (verified by human)")
        lines.append("=" * 80)
        lines.append("")
        lines.append(
            "These labels are known to be correct. Use them to calibrate"
        )
        lines.append(
            "your naming style and granularity. DO NOT contradict these."
        )
        lines.append(
            "Output these exact names for the referenced groups/clusters."
        )
        lines.append("")
        for key in sorted(reference_labels.keys(),
                          key=lambda k: (k[0], int(k[1:]))):
            lines.append(f"  {key}: {reference_labels[key]}")
        lines.append("")

    # ==== Section 2: Group structure table ====
    lines.append("=" * 80)
    lines.append(
        f"SECTION 2: GROUP STRUCTURE TABLE  ({K_coarse} groups, K={K}, "
        f"total_tokens={total_tokens:,})"
    )
    lines.append("=" * 80)
    lines.append("")

    has_margin_col = any(
        group_stats[g]["mean_margin"] is not None for g in range(K_coarse)
    )
    hdr = f"{'GRP':>4}  {'CLUSTERS':<30}  {'AGG_FREQ%':>9}"
    if has_margin_col:
        hdr += f"  {'MEAN_MARGIN':>11}"
    hdr += f"  {'MEAN_DWELL':>10}"
    lines.append(hdr)
    lines.append("-" * len(hdr))

    group_order = sorted(
        range(K_coarse),
        key=lambda g: -group_stats[g]["agg_usage"],
    )

    for g in group_order:
        gs = group_stats[g]
        cluster_str = ", ".join(f"C{c}" for c in gs["clusters"])
        if len(cluster_str) > 28:
            cluster_str = cluster_str[:25] + "..."
        row = f"G{g:>3}  {cluster_str:<30}  {gs['agg_usage'] * 100:>8.1f}%"
        if has_margin_col:
            if gs["mean_margin"] is not None:
                row += f"  {gs['mean_margin']:>11.4f}"
            else:
                row += f"  {'n/a':>11}"
        row += f"  {gs['mean_dwell']:>10.2f}"
        lines.append(row)

    lines.append("")
    lines.append("")

    # ==== Section 3: Per-group cluster evidence ====
    lines.append("=" * 80)
    lines.append("SECTION 3: PER-GROUP CLUSTER EVIDENCE")
    lines.append("=" * 80)
    lines.append("")

    per_cluster_dwells = dwell_stats["per_cluster_dwells"]

    for g in group_order:
        gs = group_stats[g]
        lines.append("=" * 80)
        grp_hdr = (
            f"GROUP G{g}: {len(gs['clusters'])} clusters  "
            f"(agg_freq={gs['agg_usage'] * 100:.1f}%"
        )
        if gs["mean_margin"] is not None:
            grp_hdr += f", mean_margin={gs['mean_margin']:.3f}"
        grp_hdr += f", mean_dwell={gs['mean_dwell']:.1f})"
        lines.append(grp_hdr)
        lines.append("=" * 80)
        lines.append("")

        cluster_order = sorted(
            gs["clusters"], key=lambda c: -usage[c],
        )

        for c in cluster_order:
            m_dwell = (
                float(mean_dwell_arr[c])
                if not np.isnan(mean_dwell_arr[c]) else 0.0
            )
            freq_pct = usage[c] * 100
            n_runs = len(per_cluster_dwells[c])
            c_hdr = (
                f"--- CLUSTER C{c}  [G{g}]  "
                f"(freq={freq_pct:.1f}%, mean_dwell={m_dwell:.1f}"
            )
            if cluster_mean_margin[c] is not None:
                c_hdr += f", mean_margin={cluster_mean_margin[c]:.3f}"
            if cluster_mean_cos[c] is not None:
                c_hdr += f", mean_cos={cluster_mean_cos[c]:.3f}"
            c_hdr += f", {n_runs} runs) ---"
            lines.append(c_hdr)
            lines.append("")

            if has_margin:
                proto_snippets = _select_snippets(
                    segments, c, proto_count, min_seg_len,
                    sort_by="margin",
                    min_margin=proto_min_margin,
                )
                if proto_snippets:
                    lines.append(
                        f"  PROTOTYPICAL EXAMPLES  "
                        f"(margin >= {proto_min_margin}, weight these most):"
                    )
                    lines.append("")
                    for seg in proto_snippets:
                        lines.extend(
                            _format_snippet_block_v2(seg, group_labels)
                        )

                typical_snippets = _select_snippets(
                    segments, c, typical_count, min_seg_len,
                    sort_by="margin",
                    min_margin=typical_min_margin,
                    max_margin=proto_min_margin,
                )
                if typical_snippets:
                    lines.append(
                        f"  TYPICAL EXAMPLES  "
                        f"(margin {typical_min_margin}–{proto_min_margin}):"
                    )
                    lines.append("")
                    for seg in typical_snippets:
                        lines.extend(
                            _format_snippet_block_v2(seg, group_labels)
                        )

                if not proto_snippets and not typical_snippets:
                    lines.append(
                        f"  (no segments above margin {typical_min_margin} found)"
                    )
                    lines.append("")
            else:
                flat_snippets = _select_snippets(
                    segments, c, snippets_per_cluster, min_seg_len,
                    sort_by="length",
                )
                if flat_snippets:
                    lines.append("  EXAMPLES:")
                    lines.append("")
                    for seg in flat_snippets:
                        lines.extend(
                            _format_snippet_block_v2(seg, group_labels)
                        )
                else:
                    lines.append("  (no segments of length >= 2 found)")
                    lines.append("")

            lines.append("")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "group_cluster_naming.txt"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved {out_path}")


def _parse_naming_file(
    path: Path,
) -> tuple[dict[int, tuple[str, str]], dict[int, tuple[str, str]]]:
    """Parse an LLM naming output file.

    Recognizes:
        G<id>: <name> | <description>
        C<id> [G<id>]: <name> | <description>
        C<id>: <name> | <description>

    Returns (cluster_names, group_names) where each value is (name, desc).
    """
    cluster_names: dict[int, tuple[str, str]] = {}
    group_names: dict[int, tuple[str, str]] = {}

    group_re = re.compile(r"^\s*G(\d+)\s*:\s*(.+?)\s*\|\s*(.+?)\s*$")
    cluster_re = re.compile(
        r"^\s*C(\d+)\s*(?:\[G\d+\])?\s*:\s*(.+?)\s*\|\s*(.+?)\s*$"
    )

    text = path.read_text(encoding="utf-8")
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = group_re.match(line)
        if m:
            gid = int(m.group(1))
            group_names[gid] = (m.group(2).strip(), m.group(3).strip())
            continue
        m = cluster_re.match(line)
        if m:
            cid = int(m.group(1))
            cluster_names[cid] = (m.group(2).strip(), m.group(3).strip())
            continue

    return cluster_names, group_names


