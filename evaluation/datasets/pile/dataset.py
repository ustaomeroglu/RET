"""Pile as a :class:`~evaluation.datasets.base.CorpusDataset`.

Streaming raw-text source over a HuggingFace dataset (default
``monology/pile-uncopyrighted``). Mirrors the source resolution used by
``ret.data.dataset.StreamingTokenChunkDataset``; chunking / shuffling /
tokenization stay with the consumer (wired in P6).
"""
from __future__ import annotations

from typing import Iterator


class PileCorpus:
    """Streaming HF text corpus (unbounded)."""

    def __init__(self, hf_dataset: str = "monology/pile-uncopyrighted",
                 text_field: str = "text"):
        self.name = "pile"
        self.hf_dataset = hf_dataset
        self.text_field = text_field

    def is_streaming(self, split: str) -> bool:
        return True

    def iter_texts(self, split: str = "train", *,
                   max_texts: int | None = None) -> Iterator[str]:
        from evaluation.datasets.corpus_source import load_corpus

        ds = load_corpus(hf_dataset=self.hf_dataset)
        n = 0
        for example in ds:
            text = example[self.text_field]
            if not isinstance(text, str) or not text.strip():
                continue
            yield text
            n += 1
            if max_texts is not None and n >= max_texts:
                break
