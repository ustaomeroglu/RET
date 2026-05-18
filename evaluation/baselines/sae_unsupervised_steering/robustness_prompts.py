"""Single source for the paper Appendix-L.1 33-prompt robustness set.

The G7 SAE-baseline scripts (and, by parity, the RET side) used to hard-code
these 33 prompts inline. The canonical copy is
``data/math_problems/paper_robustness_set_33.json``; this loader makes that
the single source of truth.

The legacy inline label for each prompt is exactly the JSON ``id`` minus its
``paper_robustness_<NN>_`` prefix (verified 33/33 by exact prompt text), so
output-dir / condition names are preserved byte-for-byte. Prompts are returned
in canonical id order (paper_robustness_01..33).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

_DEFAULT_JSON = (
    Path(__file__).resolve().parents[3]
    / "data" / "math_problems" / "paper_robustness_set_33.json"
)
_ID_PREFIX = re.compile(r"^paper_robustness_\d+_")


def load_robustness_33(path: str | Path | None = None) -> dict[str, str]:
    """Return ``{legacy_label: prompt}`` for the 33-prompt robustness set.

    ``legacy_label`` = JSON ``id`` with the ``paper_robustness_<NN>_`` prefix
    stripped (== the historical inline keys). Ordered by id.
    """
    entries = json.loads(Path(path or _DEFAULT_JSON).read_text())
    out: dict[str, str] = {}
    for e in sorted(entries, key=lambda e: e["id"]):
        out[_ID_PREFIX.sub("", e["id"])] = e["prompt"]
    if len(out) != len(entries):
        raise ValueError(
            f"label collision after prefix strip: {len(entries)} entries -> "
            f"{len(out)} labels"
        )
    return out
