"""NuminaMath as a :class:`~evaluation.datasets.base.CorpusDataset`.

Pure text source over the pre-built Arrow datasets produced by
``scripts/data_preparation_scripts/prepare_numina_splits.py`` /
``prepare_numina_math_64k.py``. Tokenization/chunking stay with the
consumer (ret/data/dataset.py, cache_hidden_states.py).
"""
from __future__ import annotations

from typing import Iterator


class NuminaMathCorpus:
    """Arrow-on-disk NuminaMath corpus.

    ``variant`` selects ``data/<variant>/<split>`` (e.g. ``numina_math`` or
    ``numina_math_64k_samples``; pass ``numina_math_64k_samples/GptOss20B``
    for the generation-augmented variant).
    """

    def __init__(self, variant: str = "numina_math", base: str = "data",
                 text_field: str = "text"):
        self.name = variant
        self.variant = variant
        self.base = base
        self.text_field = text_field

    def is_streaming(self, split: str) -> bool:
        return False

    def iter_texts(self, split: str = "train", *,
                   max_texts: int | None = None) -> Iterator[str]:
        from evaluation.datasets.corpus_source import load_corpus

        ds = load_corpus(local_dir=f"{self.base}/{self.variant}/{split}")
        texts = ds[self.text_field]
        if max_texts is not None:
            texts = texts[:max_texts]
        # Faithful raw passthrough (matches the legacy cache_hidden_states
        # `list(ds["text"])[:max]` behavior exactly — no content filtering;
        # consumers that need empty-skipping already do it themselves).
        for t in texts:
            yield t
