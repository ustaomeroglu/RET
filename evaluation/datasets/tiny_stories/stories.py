"""Shared TinyStories loading utilities."""
from __future__ import annotations

import numpy as np
from datasets import load_dataset


def load_stories(
    num_stories: int,
    seed: int,
    min_tokens: int = 60,
    max_tokens: int = 150,
) -> list[str]:
    """Load stories from TinyStories validation split, filtered by token count.

    Returns list of story strings (more than num_stories for downstream filtering).
    """
    ds = load_dataset("roneneldan/TinyStories", split="validation", streaming=True)
    rng = np.random.default_rng(seed)

    # Collect a pool of candidate stories then sample
    candidates = []
    # Use a rough char-based filter first (avg ~4 chars/token), then exact check later
    for i, row in enumerate(ds):
        text = row["text"].strip()
        char_len = len(text)
        if char_len < min_tokens * 3 or char_len > max_tokens * 6:
            continue
        candidates.append(text)
        if len(candidates) >= num_stories * 20:
            break

    if len(candidates) < num_stories:
        raise ValueError(
            f"Only found {len(candidates)} candidate stories, need {num_stories}. "
            "Try relaxing token length filters."
        )
    # Shuffle and return first num_stories (exact token filtering done later)
    indices = rng.permutation(len(candidates))
    return [candidates[i] for i in indices[: num_stories * 5]]  # return extras for filtering


def filter_by_token_count(
    stories: list[str],
    tokenizer,
    min_tokens: int,
    max_tokens: int,
    num_stories: int,
) -> list[str]:
    """Filter stories to those with token count in [min_tokens, max_tokens]."""
    filtered = []
    for text in stories:
        ids = tokenizer.encode(text)
        if min_tokens <= len(ids) <= max_tokens:
            filtered.append(text)
            if len(filtered) >= num_stories:
                break
    return filtered
