"""TinyStories as a :class:`~evaluation.datasets.base.CorpusDataset`.

Pure text source over the pre-built Arrow dataset
(``data/tiny_stories/<split>``). The separate streaming-validation
helpers ``load_stories`` / ``filter_by_token_count`` (used by the
embedding-trajectory / plot_analysis scripts) live in ``.stories`` and are
re-exported from this package's ``__init__``.
"""
from __future__ import annotations

from typing import Iterator


class TinyStoriesCorpus:
    """Arrow-on-disk TinyStories corpus."""

    def __init__(self, base: str = "data", text_field: str = "text"):
        self.name = "tiny_stories"
        self.base = base
        self.text_field = text_field

    def is_streaming(self, split: str) -> bool:
        return False

    def iter_texts(self, split: str = "train", *,
                   max_texts: int | None = None) -> Iterator[str]:
        from evaluation.datasets.corpus_source import load_corpus

        ds = load_corpus(local_dir=f"{self.base}/tiny_stories/{split}")
        texts = ds[self.text_field]
        if max_texts is not None:
            texts = texts[:max_texts]
        for t in texts:
            yield t
