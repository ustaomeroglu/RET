"""
Data loader for the Arrrlex/models-under-pressure HuggingFace dataset.

Training split columns used:
    inputs       : str  — the user prompt text
    tone         : str  — ['casual', 'overly polite', 'vulnerable', 'angry']
    prompt_style : str  — ['Third Person', 'Character Perspective', 'Question',
                           'Response to Situation', 'Prompt to LLM']
    high_stakes  : bool — True / False  (derived from the 'high-stakes' label field)

Public API
----------
SUBTASKS : dict[str, SubtaskInfo]
    Describes each subtask: n_classes, task_type, label_map

load_pressure_dataset(max_samples) -> (prompts, labels_dict)
    Returns formatted prompt strings and a dict {subtask_name -> np.ndarray of int labels}
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = "You are a helpful assistant."

# ---------------------------------------------------------------------------
# Subtask definitions
# ---------------------------------------------------------------------------

@dataclass
class SubtaskInfo:
    task_type: str         # "binary" | "multiclass"
    n_classes: int
    label_map: Dict        # raw_value -> int index
    class_names: List[str] # int index -> display name


# Canonical class orderings (alphabetical / as-they-appear in dataset)
SUBTASKS: Dict[str, SubtaskInfo] = {
    "tone": SubtaskInfo(
        task_type="multiclass",
        n_classes=4,
        label_map={
            "casual":        0,
            "overly polite": 1,
            "vulnerable":    2,
            "angry":         3,
        },
        class_names=["casual", "overly polite", "vulnerable", "angry"],
    ),
    "prompt_style": SubtaskInfo(
        task_type="multiclass",
        n_classes=5,
        label_map={
            "Third Person":            0,
            "Character Perspective":   1,
            "Question":                2,
            "Response to Situation":   3,
            "Prompt to LLM":           4,
        },
        class_names=["Third Person", "Character Perspective", "Question",
                     "Response to Situation", "Prompt to LLM"],
    ),
    "high_stakes": SubtaskInfo(
        task_type="binary",
        n_classes=2,
        label_map={True: 1, False: 0, "true": 1, "false": 0,
                   "True": 1, "False": 0},
        class_names=["low_stakes", "high_stakes"],
    ),
}


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def _encode_label(value, subtask: SubtaskInfo) -> Optional[int]:
    """Convert a raw label value to an integer index; None if unrecognised."""
    if value in subtask.label_map:
        return subtask.label_map[value]
    # Case-insensitive string fallback
    if isinstance(value, str):
        for k, v in subtask.label_map.items():
            if isinstance(k, str) and k.lower() == value.lower():
                return v
    return None


def load_pressure_dataset(
    max_samples: Optional[int] = None,
    tokenizer: Optional[AutoTokenizer] = None,
    include_gpt_info: bool = True,
) -> Tuple[List[str], Dict[str, np.ndarray]]:
    """Load the training split of models-under-pressure.

    Parameters
    ----------
    max_samples : optional cap (for quick tests).
    tokenizer   : used to build Harmony prompts; if None, returns raw prompts.

    Returns
    -------
    prompts     : List[str]  — one fully-formatted Harmony prompt per sample
    labels_dict : {subtask_name -> np.ndarray[N] of int labels}
        Samples that have unrecognised labels for a subtask are excluded
        from that subtask's array (they still appear in the other subtasks'
        arrays if their labels are valid there).
        For simplicity, we keep only samples that are valid for ALL subtasks.
    """
    logger.info("Loading Arrrlex/models-under-pressure (training split)...")
    ds = load_dataset("Arrrlex/models-under-pressure", "training", split="train")
    logger.info("Loaded %d raw samples", len(ds))

    n_before = len(ds)
    ds = ds.filter(lambda row: row.get("language", "") == "English")
    logger.info("Filtered to English only: %d → %d samples", n_before, len(ds))

    col_names = ds.column_names

    valid_rows: List[int] = []
    all_labels: Dict[str, List[int]] = {name: [] for name in SUBTASKS}

    for i, row in enumerate(ds):
        text = row.get("inputs") or row.get("input") or ""
        if not text:
            continue

        # --- high_stakes: look in 'high_stakes' col OR derive from 'labels' ---
        if "high_stakes" in col_names:
            hs_raw = row["high_stakes"]
        elif "labels" in col_names:
            # labels is often a list like ["high-stakes"] or ["low-stakes"]
            lab_list = row["labels"]
            if isinstance(lab_list, list):
                hs_raw = any("high" in str(l).lower() for l in lab_list)
            else:
                hs_raw = "high" in str(lab_list).lower()
        else:
            hs_raw = None

        # --- Encode all subtasks ---
        tone_raw   = row.get("tone")
        style_raw  = row.get("prompt_style")

        enc_tone   = _encode_label(tone_raw,   SUBTASKS["tone"])
        enc_style  = _encode_label(style_raw,  SUBTASKS["prompt_style"])
        enc_hs     = _encode_label(hs_raw,     SUBTASKS["high_stakes"])

        if enc_tone is None or enc_style is None or enc_hs is None:
            continue  # skip samples with missing / unrecognised labels

        valid_rows.append(i)
        all_labels["tone"].append(enc_tone)
        all_labels["prompt_style"].append(enc_style)
        all_labels["high_stakes"].append(enc_hs)

    logger.info("Valid samples (all labels present): %d", len(valid_rows))

    if max_samples is not None and len(valid_rows) > max_samples:
        rng = np.random.default_rng(42)
        sel = rng.choice(len(valid_rows), size=max_samples, replace=False)
        sel.sort()
        valid_rows = [valid_rows[i] for i in sel]
        all_labels = {k: [v[i] for i in sel] for k, v in all_labels.items()}
        logger.info("Subsampled to %d samples", max_samples)

    # Build prompts
    prompts: List[str] = []
    if tokenizer is not None:
        # Lazy: only chat-templating needs torch-heavy model_utils. Keeps the
        # pure-data path (tokenizer=None, used by the dataset adapter) import-light.
        from evaluation.probing.model_utils import build_prompt
    for i in valid_rows:
        text = ds[i].get("inputs") or ds[i].get("input") or ""
        if tokenizer is not None:
            p = build_prompt(text, tokenizer, system_text=SYSTEM_PROMPT,
                             include_gpt_info=include_gpt_info)
        else:
            p = text
        prompts.append(p)

    labels_dict = {k: np.array(v, dtype=int) for k, v in all_labels.items()}

    for name, arr in labels_dict.items():
        info = SUBTASKS[name]
        logger.info(
            "Subtask '%s': %d samples, classes=%s, dist=%s",
            name, len(arr), info.class_names,
            {cls: int((arr == i).sum()) for i, cls in enumerate(info.class_names)},
        )

    return prompts, labels_dict
