"""MMLU as a :class:`~evaluation.datasets.base.ProbingDataset` (pure data)."""
from __future__ import annotations

import numpy as np

from evaluation.datasets.base import LabelTask, ProbingSplit
from evaluation.datasets.mmlu.data import load_mmlu_splits

_SYSTEM_PROMPT = "You are a helpful assistant."

# (task_name, dataframe column)
_TASKS = [
    ("subject", "subject"),
    ("group_4way", "group_4way"),
    ("group_7way", "group_7way"),
    ("question_idx", "question_idx"),
]


class MMLUDataset:
    """Pre-split MMLU probing data declaring four label tasks over a shared
    set of question prompts: subject (57-class), group_4way, group_7way,
    question_idx (contextual)."""

    name = "mmlu"
    system_prompt = _SYSTEM_PROMPT
    include_gpt_info = True
    needs_generation = False
    # Legacy MMLU runner fed the plain formatted question to extraction
    # (no chat-template wrapping).
    chat_wrap = False

    def __init__(self, split_dir: str = "data/mmlu/eval"):
        self.split_dir = split_dir

    def load(self, *, max_samples: int | None = None) -> dict[str, ProbingSplit]:
        train_dict, val_dict = load_mmlu_splits(self.split_dir, max_samples=max_samples)
        train_df = train_dict["mmlu"]
        val_df = val_dict["mmlu"]

        tasks_train: dict[str, LabelTask] = {}
        tasks_val: dict[str, LabelTask] = {}
        for task_name, col in _TASKS:
            tr_raw = train_df[col].to_numpy()
            va_raw = val_df[col].to_numpy()
            if col == "question_idx":
                tr_lab = tr_raw.astype(np.int64)
                va_lab = va_raw.astype(np.int64)
                n_classes = int(len(np.unique(np.concatenate([tr_lab, va_lab]))))
                class_names = None
            else:
                # One encoder fit on train+val union. When class sets are
                # identical (MMLU is stratified) this reproduces the legacy
                # per-split LabelEncoder integers exactly and yields the
                # correct cross-split n_classes.
                from sklearn.preprocessing import LabelEncoder

                enc = LabelEncoder()
                enc.fit(np.concatenate([tr_raw.astype(str), va_raw.astype(str)]))
                tr_lab = enc.transform(tr_raw.astype(str)).astype(np.int64)
                va_lab = enc.transform(va_raw.astype(str)).astype(np.int64)
                n_classes = int(len(enc.classes_))
                class_names = [str(c) for c in enc.classes_]
            task_type = "binary" if n_classes == 2 else "multiclass"
            tasks_train[task_name] = LabelTask(
                task_name, tr_lab, task_type, n_classes, class_names
            )
            tasks_val[task_name] = LabelTask(
                task_name, va_lab, task_type, n_classes, class_names
            )

        def _meta(df):
            return [
                {"subject": str(s), "question_idx": int(q)}
                for s, q in zip(df["subject"], df["question_idx"])
            ]

        return {
            "train": ProbingSplit(
                prompts=train_df["text"].tolist(),
                tasks=tasks_train,
                metadata=_meta(train_df),
            ),
            "val": ProbingSplit(
                prompts=val_df["text"].tolist(),
                tasks=tasks_val,
                metadata=_meta(val_df),
            ),
        }
