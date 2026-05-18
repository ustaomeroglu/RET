"""Dataset-layer protocols.

`evaluation/datasets/` is a pure data layer: it loads/produces inputs, labels,
and splits. It contains no model, no hidden states, no probes. Experiment logic
(the probing pipeline, steering, etc.) consumes these adapters; it does not live
here.

Two shapes, one home:

- :class:`ProbingDataset` — labelled, sample-level data for probing
  (mmlu, pressure, sycophancy-data). Yields raw user texts + one or more
  :class:`LabelTask` per split. The chat template / system prompt is applied
  downstream by the probing pipeline (which owns the tokenizer); adapters stay
  tokenizer-free so they remain pure data.

- :class:`CorpusDataset` — unlabelled document corpora used for RET training and
  hidden-state caching (numina_math, tiny_stories, pile). Yields raw document
  text per split. Tokenization, chunking, scaffolding and padding remain the
  responsibility of the consumer (``ret/data/dataset.py``); the adapter only
  resolves *where the raw text comes from*.

These Protocols are intentionally minimal and dependency-light (typing,
dataclasses, numpy only) so any dataset folder can implement one without
pulling in torch/transformers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Protocol, runtime_checkable

import numpy as np


# ---------------------------------------------------------------------------
# ProbingDataset: labelled, sample-level data
# ---------------------------------------------------------------------------

@dataclass
class LabelTask:
    """One classification target over a split's samples.

    A dataset may declare several (e.g. MMLU: subject / group_4way /
    group_7way / question_idx; pressure: tone / prompt_style / high_stakes).
    The pipeline loops over the declared tasks — it does not subclass.
    """

    name: str
    labels: np.ndarray            # shape (n_samples,), integer-encoded
    task_type: str                # "binary" | "multiclass"
    n_classes: int
    class_names: list[str] | None = None


@dataclass
class ProbingSplit:
    """One split's worth of probing data.

    ``prompts`` are *raw user texts* (pre chat-template). The probing pipeline
    applies the system prompt / chat template using the model tokenizer, so the
    adapter never needs a tokenizer.
    """

    prompts: list[str]
    tasks: dict[str, LabelTask]
    metadata: list[dict] = field(default_factory=list)


@runtime_checkable
class ProbingDataset(Protocol):
    """Pure data adapter for a probing experiment.

    The single place dataset identity / loading lives; consumed by
    ``run_probing``. Implementations live in ``evaluation/datasets/<name>/``.
    """

    #: short identifier, e.g. "mmlu", "models_under_pressure"
    name: str
    #: system prompt the pipeline should prepend (data-side default)
    system_prompt: str
    #: whether to include GPT-OSS harmony system metadata in the prefix
    include_gpt_info: bool
    #: whether the pipeline must generate model outputs before labelling
    needs_generation: bool
    #: whether the pipeline should wrap raw prompts with the chat template +
    #: system prompt before extraction (pressure=True). When False the raw
    #: text is fed to the LLM unwrapped (mmlu feeds the plain question).
    chat_wrap: bool

    def load(self, *, max_samples: int | None = None) -> dict[str, ProbingSplit]:
        """Return splits.

        Pre-split datasets return ``{"train": ..., "val": ...}`` (or
        ``"test"``). Single-pool datasets return ``{"all": ...}`` and the
        pipeline performs the within-dataset split. ``max_samples`` caps
        samples per split when set.
        """
        ...


# ---------------------------------------------------------------------------
# CorpusDataset: unlabelled document corpora for training / caching
# ---------------------------------------------------------------------------

@runtime_checkable
class CorpusDataset(Protocol):
    """Pure text-source adapter for a training/caching corpus.

    The consumer (``ret/data/dataset.py``, ``cache_hidden_states.py``) keeps
    all tokenize/chunk/scaffold/pad mechanics; this adapter only resolves the
    raw-text source for a given split (HF stream vs Arrow-on-disk).
    """

    #: short identifier, e.g. "numina_math", "tiny_stories", "pile"
    name: str

    def is_streaming(self, split: str) -> bool:
        """True if the split is an unbounded stream (e.g. pile train)."""
        ...

    def iter_texts(
        self, split: str, *, max_texts: int | None = None
    ) -> Iterable[str]:
        """Yield raw document text strings for ``split``.

        Finite for Arrow-on-disk corpora (numina_math, tiny_stories);
        potentially unbounded for streaming corpora (pile). ``max_texts``
        caps the number yielded when set.
        """
        ...
