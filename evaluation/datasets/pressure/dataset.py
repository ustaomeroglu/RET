"""models-under-pressure as a :class:`~evaluation.datasets.base.ProbingDataset`
(pure data)."""
from __future__ import annotations

from evaluation.datasets.base import LabelTask, ProbingSplit
from evaluation.datasets.pressure.data import SUBTASKS, load_pressure_dataset


def _tasks_from_labels(labels_dict) -> dict[str, LabelTask]:
    return {
        name: LabelTask(
            name=name,
            labels=labels_dict[name],
            task_type=info.task_type,
            n_classes=info.n_classes,
            class_names=list(info.class_names),
        )
        for name, info in SUBTASKS.items()
    }


class PressureDataset:
    """models-under-pressure probing data declaring three label tasks
    (tone 4-class, prompt_style 5-class, high_stakes binary) over a shared
    prompt set. Single-pool from the HF training split by default (the
    pipeline performs the within-dataset split); pre-split JSONL when
    train/test paths are given."""

    name = "models_under_pressure"
    system_prompt = "You are a helpful assistant."
    needs_generation = False
    # Legacy pressure runner Harmony/chat-wrapped each prompt via build_prompt.
    chat_wrap = True

    def __init__(
        self,
        train_jsonl: str | None = None,
        test_jsonl: str | None = None,
        include_gpt_info: bool = True,
    ):
        self.train_jsonl = train_jsonl
        self.test_jsonl = test_jsonl
        # Instance attr (overrides nothing; satisfies the ProbingDataset
        # protocol). False == legacy --strip_gpt_info.
        self.include_gpt_info = include_gpt_info

    def load(self, *, max_samples: int | None = None) -> dict[str, ProbingSplit]:
        if self.train_jsonl and self.test_jsonl:
            from evaluation.datasets.pressure.splits import load_pressure_splits

            out: dict[str, ProbingSplit] = {}
            for split, path in (("train", self.train_jsonl), ("test", self.test_jsonl)):
                texts, labels_dict = load_pressure_splits(path)
                out[split] = ProbingSplit(
                    prompts=texts, tasks=_tasks_from_labels(labels_dict)
                )
            return out

        prompts, labels_dict = load_pressure_dataset(
            max_samples=max_samples, tokenizer=None
        )
        return {"all": ProbingSplit(prompts=prompts, tasks=_tasks_from_labels(labels_dict))}
