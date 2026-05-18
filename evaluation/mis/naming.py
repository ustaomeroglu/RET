"""RET cluster-name parser + non-LLM SAE latent naming.

Two code paths:
    parse_ret_naming     — read an existing `naming.txt` produced by the
                            clustering pipeline and return structured group/
                            cluster metadata.
    build_sae_latent_names — stream hidden states through a TopKSAE over a
                              corpus and record the top-K most-activating token
                              strings per latent. Non-LLM naming baseline.

Formatting helpers produce the naming-table headers consumed by the
describe-from-codes prompt.
"""
from __future__ import annotations

import heapq
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# RET naming.txt parser (reuses regex semantics from scripts/clustering/plotting/inject_names.py)
# ---------------------------------------------------------------------------

_GROUP_RE = re.compile(r"^\s*G(\d+)\s*:\s*(.+?)\s*\|\s*(.+?)\s*$")
_CLUSTER_RE = re.compile(r"^\s*C(\d+)\s*(?:\[G(\d+)\])?\s*:\s*(.+?)\s*\|\s*(.+?)\s*$")


@dataclass
class RETNaming:
    groups: Dict[int, Dict[str, str]]         # gid -> {name, desc}
    clusters: Dict[int, Dict[str, object]]    # cid -> {name, desc, group_id}

    @property
    def K(self) -> int:
        return len(self.clusters)

    @property
    def num_groups(self) -> int:
        return len(self.groups)


def parse_ret_naming(naming_txt_path: Path | str) -> RETNaming:
    """Parse an LLM-produced naming.txt with G/C lines.

    Recognized formats per line:
        G<id>: <name> | <description>
        C<id> [G<id>]: <name> | <description>
        C<id>: <name> | <description>     (ungrouped fallback)
    Other lines are silently skipped.
    """
    path = Path(naming_txt_path)
    groups: Dict[int, Dict[str, str]] = {}
    clusters: Dict[int, Dict[str, object]] = {}

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue

        m = _GROUP_RE.match(line)
        if m:
            gid = int(m.group(1))
            groups[gid] = {"name": m.group(2).strip(), "desc": m.group(3).strip()}
            continue

        m = _CLUSTER_RE.match(line)
        if m:
            cid = int(m.group(1))
            gid = int(m.group(2)) if m.group(2) is not None else -1
            clusters[cid] = {
                "name": m.group(3).strip(),
                "desc": m.group(4).strip(),
                "group_id": gid,
            }

    return RETNaming(groups=groups, clusters=clusters)


def format_ret_naming_table(
    naming: RETNaming,
    active_clusters: Optional[Iterable[int]] = None,
    groups_only: bool = False,
) -> str:
    """Render the naming table as the header shown to the describer LLM.

    If ``active_clusters`` is provided, only clusters in that set are
    included, and only groups referenced by at least one active cluster.
    If ``groups_only`` is True, only the Groups section is rendered
    (cluster-level entries are omitted).
    """
    if active_clusters is not None:
        active_set = {int(c) for c in active_clusters}
    else:
        active_set = None

    # Determine groups to include.
    if active_set is not None:
        active_group_ids = {
            int(naming.clusters[cid]["group_id"])
            for cid in active_set
            if cid in naming.clusters and int(naming.clusters[cid]["group_id"]) != -1
        }
    else:
        active_group_ids = None

    lines: List[str] = []
    lines.append("# State code naming table")
    lines.append("")
    if naming.groups:
        lines.append("## Groups (high-level roles)")
        group_ids = sorted(naming.groups)
        if active_group_ids is not None:
            group_ids = [g for g in group_ids if g in active_group_ids]
        for gid in group_ids:
            g = naming.groups[gid]
            lines.append(f"G{gid} — {g['name']}: {g['desc']}")
        lines.append("")

    if not groups_only:
        lines.append("## Clusters (fine-grained states)")
        cluster_ids = sorted(naming.clusters)
        if active_set is not None:
            cluster_ids = [c for c in cluster_ids if c in active_set]
        for cid in cluster_ids:
            c = naming.clusters[cid]
            gid = c["group_id"]
            tag = f"[G{gid}] " if gid != -1 else ""
            lines.append(f"C{cid} {tag}— {c['name']}: {c['desc']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# SAE latent naming (non-LLM; top max-activating tokens per latent)
# ---------------------------------------------------------------------------

class _TopKBucket:
    """Per-latent dict of token_id -> max activation, pruned periodically.

    At finalize time, returns the top-K distinct tokens by activation.
    Amortized O(1) per push (prune is O(N log K) but rare).
    """

    __slots__ = ("k", "prune_threshold", "best_by_id")

    def __init__(self, k: int, prune_mult: int = 8):
        self.k = k
        self.prune_threshold = k * prune_mult
        self.best_by_id: Dict[int, float] = {}

    def push(self, score: float, token_id: int) -> None:
        prev = self.best_by_id.get(token_id)
        if prev is None or score > prev:
            self.best_by_id[token_id] = score
        if len(self.best_by_id) > self.prune_threshold:
            self._prune()

    def _prune(self) -> None:
        keep = heapq.nlargest(self.k, self.best_by_id.items(), key=lambda x: x[1])
        self.best_by_id = dict(keep)

    def top_ids(self) -> List[int]:
        top = heapq.nlargest(self.k, self.best_by_id.items(), key=lambda x: x[1])
        return [tid for tid, _ in top]


@torch.no_grad()
def build_sae_latent_names(
    sae,
    hidden_loader,
    tokenizer,
    device: str = "cuda",
    top_k: int = 5,
    max_tokens: Optional[int] = None,
    log_every: int = 10,
) -> Dict[int, List[str]]:
    """Collect top-K max-activating distinct token strings per SAE latent.

    Streams through ``hidden_loader`` (yields ``{"hidden_states", "input_ids",
    "attention_mask"}``); for each token, encodes through the SAE to get the
    top-k active latents with values, and for each active (latent, score, tok)
    triple updates a per-latent bounded top-K over scores.

    Parameters
    ----------
    sae            : TopKSAE in eval mode
    hidden_loader  : iterator of batches with ``hidden_states`` (B,T,D),
                     ``input_ids`` (B,T), ``attention_mask`` (B,T)
    tokenizer      : HF tokenizer, used to decode token ids to strings
    device         : torch device
    top_k          : number of distinct tokens to keep per latent
    max_tokens     : optional cap on total tokens streamed (for quick builds)

    Returns
    -------
    Dict[int, List[str]]  mapping latent_id -> list of up to top_k decoded
                          token strings (most-activating first).
    """
    d_sae = sae.d_sae
    buckets: Dict[int, _TopKBucket] = {}
    total_tokens = 0
    batch_count = 0

    # Encode a bounded-size composite key (lat, tok) as lat * _VOCAB_CAP + tok.
    # gpt-oss-20b vocab is ~200k so 1M gives plenty of headroom.
    _VOCAB_CAP = 1_000_000

    for batch in hidden_loader:
        h = batch["hidden_states"].to(device=device, dtype=torch.float32)
        mask = batch.get("attention_mask")
        ids = batch["input_ids"]

        B, T, D = h.shape
        flat_h = h.reshape(-1, D)
        flat_ids = ids.reshape(-1).to(torch.int64)
        if mask is not None:
            flat_mask = mask.reshape(-1).bool()
        else:
            flat_mask = torch.ones(B * T, dtype=torch.bool, device=h.device)

        chunk_size = 4096
        for start in range(0, flat_h.shape[0], chunk_size):
            end = min(start + chunk_size, flat_h.shape[0])
            chunk = flat_h[start:end]
            chunk_ids = flat_ids[start:end].to(h.device)
            chunk_mask = flat_mask[start:end].to(h.device)

            centered = chunk - sae.b_dec
            pre = centered @ sae.encoder_weight.T + sae.encoder_bias
            top_vals, top_idx = pre.topk(sae.k, dim=-1)  # (N, k) each
            top_vals = torch.relu(top_vals)

            # Flatten to triples (lat, tok, act) on GPU, then filter.
            N = top_vals.shape[0]
            K = top_vals.shape[1]
            tok_b = chunk_ids.unsqueeze(-1).expand(N, K)                  # (N, k)
            keep = chunk_mask.unsqueeze(-1).expand(N, K) & (top_vals > 0)

            lats = top_idx[keep].to(torch.int64)
            toks = tok_b[keep].to(torch.int64)
            acts = top_vals[keep]

            # Per-batch dedup: for each unique (lat, tok) pair, keep the max act.
            # Use scatter_reduce on a composite key.
            composite = lats * _VOCAB_CAP + toks
            uniq, inverse = torch.unique(composite, return_inverse=True)
            max_acts = torch.full(
                (uniq.shape[0],), float("-inf"),
                device=uniq.device, dtype=acts.dtype,
            )
            max_acts.scatter_reduce_(
                0, inverse, acts, reduce="amax", include_self=True,
            )

            # Move back to CPU for the Python merge with global state.
            uniq_np = uniq.cpu().numpy()
            max_acts_np = max_acts.cpu().numpy()
            batch_lats = (uniq_np // _VOCAB_CAP).astype(np.int64)
            batch_toks = (uniq_np % _VOCAB_CAP).astype(np.int64)

            # One push per unique (lat, tok) in this chunk (much fewer than N*k).
            for lat_id, tok_id, val in zip(
                batch_lats.tolist(), batch_toks.tolist(), max_acts_np.tolist(),
            ):
                bucket = buckets.get(lat_id)
                if bucket is None:
                    bucket = _TopKBucket(top_k)
                    buckets[lat_id] = bucket
                bucket.push(val, tok_id)

            # Token count: number of valid tokens processed in this chunk.
            total_tokens += int(chunk_mask.sum().item())

            if max_tokens is not None and total_tokens >= max_tokens:
                break

        batch_count += 1
        if batch_count % log_every == 0:
            logger.info(
                "SAE naming: streamed %d batches, %d tokens, %d latents seen",
                batch_count, total_tokens, len(buckets),
            )
        if max_tokens is not None and total_tokens >= max_tokens:
            break

    latent_names: Dict[int, List[str]] = {}
    for lat_id, bucket in buckets.items():
        tok_ids = bucket.top_ids()
        strings: List[str] = []
        for tid in tok_ids:
            try:
                s = tokenizer.decode([tid])
            except Exception:
                s = f"<tok{tid}>"
            strings.append(s.replace("\n", "\\n").replace("\r", "\\r"))
        latent_names[lat_id] = strings

    logger.info(
        "SAE naming complete: %d latents named from %d tokens",
        len(latent_names), total_tokens,
    )
    return latent_names


def save_sae_latent_names(
    latent_names: Dict[int, List[str]],
    out_path: Path | str,
    meta: Optional[dict] = None,
) -> None:
    """Persist latent_names as JSON. Keys are stringified latent IDs."""
    out = {
        "meta": meta or {},
        "latent_names": {str(k): v for k, v in latent_names.items()},
    }
    Path(out_path).write_text(json.dumps(out, indent=2, ensure_ascii=False))


def load_sae_latent_names(path: Path | str) -> Dict[int, List[str]]:
    data = json.loads(Path(path).read_text())
    return {int(k): list(v) for k, v in data["latent_names"].items()}


def format_sae_naming_table(
    latent_names: Dict[int, List[str]],
    active_latent_ids: Iterable[int],
) -> str:
    """Render only the latents that actually appear in the current sample."""
    lines: List[str] = []
    lines.append("# SAE latent naming table (top max-activating tokens per latent)")
    lines.append("")
    seen = sorted({int(i) for i in active_latent_ids})
    for lat_id in seen:
        toks = latent_names.get(lat_id, [])
        if toks:
            pretty = ", ".join(repr(t) for t in toks)
            lines.append(f"L{lat_id}: {pretty}")
        else:
            lines.append(f"L{lat_id}: <no naming data>")
    return "\n".join(lines)
