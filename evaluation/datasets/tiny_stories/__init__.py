"""TinyStories data layer.

- :class:`TinyStoriesCorpus` — Arrow-on-disk training corpus.
- :func:`load_stories` / :func:`filter_by_token_count` — the legacy
  HF-validation streaming + char/token filter helpers (used by the
  embedding-trajectory and plot_analysis scripts).
"""
from evaluation.datasets.tiny_stories.dataset import TinyStoriesCorpus
from evaluation.datasets.tiny_stories.stories import (
    filter_by_token_count,
    load_stories,
)

__all__ = ["TinyStoriesCorpus", "load_stories", "filter_by_token_count"]
