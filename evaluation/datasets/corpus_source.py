"""Single corpus source resolver for the data layer.

One place that turns ``(hf_dataset | local_dir)`` into a raw HuggingFace
dataset object. ``ret.data.dataset`` and the ``CorpusDataset`` adapters
both delegate here so the ``load_dataset`` / ``load_from_disk`` decision
lives once. Shuffling / tokenization / chunking stay with the caller —
this function only acquires the source, with the exact semantics the
training loaders already used (HF -> streaming train split; otherwise
Arrow-on-disk).
"""
from __future__ import annotations


def load_corpus(*, hf_dataset: str | None = None, local_dir: str | None = None):
    """Return the raw dataset object (no shuffle, no iteration applied).

    - ``hf_dataset`` set: ``load_dataset(hf_dataset, split="train",
      streaming=True)`` (matches ret's chunked/streaming source).
    - else: ``load_from_disk(local_dir)`` (Arrow-on-disk; the caller passes
      the already-resolved split subdirectory when applicable).
    """
    from datasets import load_dataset, load_from_disk

    if hf_dataset is not None:
        return load_dataset(hf_dataset, split="train", streaming=True)
    if local_dir is None:
        raise ValueError("load_corpus: provide exactly one of hf_dataset / local_dir")
    return load_from_disk(local_dir)
