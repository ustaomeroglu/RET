"""Keyword search over the SAE latent-name table.

The names file (``sae_latent_names.json``) maps latent_id -> list of top-K
tokens that maximally activate that feature on the training corpus. We
provide simple substring search (case-insensitive by default) so the user
can quickly find candidate latents for a steering experiment.
"""
from __future__ import annotations

import json
from pathlib import Path


def load_sae_latent_names(path: str | Path) -> dict[int, list[str]]:
    """Load the names JSON. Top-level may be either a flat
    ``{lid: [tokens]}`` dict or the wrapped form ``{"meta": {...},
    "latent_names": {lid: [tokens]}}``.

    Raises FileNotFoundError if the path does not exist (no silent default).
    Returns a dict with int keys.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"SAE latent names file not found: {p}")
    with open(p) as f:
        data = json.load(f)
    if isinstance(data, dict) and "latent_names" in data:
        raw = data["latent_names"]
    else:
        raw = data
    if not isinstance(raw, dict):
        raise ValueError(
            f"Unrecognized latent-names JSON shape at {p}: "
            f"top-level type {type(raw).__name__}"
        )
    return {int(k): list(v) for k, v in raw.items()}


def search_latents_by_keyword(
    sae_latent_names: dict[int, list[str]],
    keywords: list[str],
    *,
    case_sensitive: bool = False,
    require_all: bool = False,
    max_results: int | None = None,
) -> list[tuple[int, list[str]]]:
    """Return ``(latent_id, top_tokens)`` pairs whose token list contains any
    (or all, if ``require_all=True``) of the keywords as a substring.

    Sorted by latent_id ascending. Truncated to ``max_results`` if set.
    """
    if not keywords:
        raise ValueError("keywords must be a non-empty list")

    if case_sensitive:
        kws = list(keywords)
        norm = lambda s: s
    else:
        kws = [k.lower() for k in keywords]
        norm = lambda s: s.lower()

    out: list[tuple[int, list[str]]] = []
    for lid, toks in sae_latent_names.items():
        normed_toks = [norm(t) for t in toks]
        if require_all:
            ok = all(any(kw in t for t in normed_toks) for kw in kws)
        else:
            ok = any(any(kw in t for t in normed_toks) for kw in kws)
        if ok:
            out.append((lid, list(toks)))
    out.sort(key=lambda x: x[0])
    if max_results is not None:
        out = out[:max_results]
    return out


def pretty_print_latent(latent_id: int, top_tokens: list[str]) -> str:
    """One-line repr like ``L1770: [' derivative', ' derived', ...]``."""
    return f"L{latent_id}: {top_tokens!r}"
