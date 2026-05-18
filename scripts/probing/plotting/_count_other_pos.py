"""One-off: count SPACE / X / INTJ breakdown within OTHER bucket on MMLU."""
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import spacy
from datasets import load_dataset
from transformers import AutoTokenizer

from evaluation.datasets.mmlu.data import POS_MERGE, format_question


def _load_mmlu_subsampled(max_questions_per_subject: int):
    ds = load_dataset("cais/mmlu", "all", split="test")
    subject_rows: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(ds):
        subject_rows[row["subject"]].append(i)
    rng = np.random.default_rng(42)
    selected: list[int] = []
    for _s, idxs in sorted(subject_rows.items()):
        chosen = rng.choice(idxs, size=min(len(idxs), max_questions_per_subject), replace=False)
        selected.extend(chosen.tolist())
    selected.sort()
    return ds.select(selected)


def main():
    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")
    tokenizer.pad_token = tokenizer.eos_token
    dataset = _load_mmlu_subsampled(100)
    print(f"Questions: {len(dataset)}")

    raw_counts: Counter = Counter()
    total_real = 0
    for row in dataset:
        text = format_question(row)
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=256,
                       padding="max_length", return_offsets_mapping=True)
        offsets = enc["offset_mapping"][0].tolist()
        mask = enc["attention_mask"][0].tolist()
        doc = nlp(text)
        char_to_pos: dict[int, str] = {}
        for tok in doc:
            for i in range(tok.idx, tok.idx + len(tok.text)):
                char_to_pos[i] = tok.pos_
        for (s, e), m in zip(offsets, mask):
            if m == 0 or s == e:
                continue
            total_real += 1
            raw = char_to_pos.get(s, "X")
            if POS_MERGE.get(raw, raw) == "OTHER":
                raw_counts[raw] += 1

    other_total = sum(raw_counts.values())
    print(f"\nTotal non-special tokens: {total_real}")
    print(f"Tokens in OTHER bucket:   {other_total} ({100 * other_total / total_real:.2f}%)")
    print("\nBreakdown within OTHER:")
    for tag in ("SPACE", "X", "INTJ"):
        c = raw_counts.get(tag, 0)
        pct_other = 100 * c / other_total if other_total else 0.0
        pct_all = 100 * c / total_real if total_real else 0.0
        print(f"  {tag:6s}: {c:7d}  ({pct_other:5.2f}% of OTHER, {pct_all:.3f}% of all)")


if __name__ == "__main__":
    main()
