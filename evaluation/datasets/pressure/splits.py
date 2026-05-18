"""Load pre-split pressure JSONL files."""
from __future__ import annotations

import json
import logging
from typing import Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def _read_jsonl(path: str) -> List[dict]:
    records = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_pressure_splits(
    path: str,
) -> Tuple[List[str], Dict[str, np.ndarray]]:
    """Load pressure JSONL file.

    The file ``pressure.jsonl`` has fields: text, tone, prompt_style, high_stakes.

    Returns ``(texts, labels_dict)`` matching ``load_pressure_dataset`` format.
    """
    records = _read_jsonl(path)
    texts = [r["text"] for r in records]
    labels_dict = {
        "tone": np.array([r["tone"] for r in records], dtype=int),
        "prompt_style": np.array([r["prompt_style"] for r in records], dtype=int),
        "high_stakes": np.array([r["high_stakes"] for r in records], dtype=int),
    }
    logger.info("Loaded pressure split %s: %d samples", path, len(records))
    return texts, labels_dict
