"""MMLU subject groupings, POS merge map, and question formatter.

Shared by the MMLU probing adapter, scripts/probing/data/prepare_mmlu_splits.py,
and the MMLU plotting scripts.
"""

# ---------------------------------------------------------------------------
# Subject → broad group mapping
# ---------------------------------------------------------------------------
# Official MMLU 4-way grouping (Hendrycks et al., 2021)
SUBJECT_GROUPS_4WAY = {
    "STEM": [
        "abstract_algebra", "astronomy", "college_biology", "college_chemistry",
        "college_computer_science", "college_mathematics", "college_physics",
        "computer_security", "conceptual_physics", "electrical_engineering",
        "elementary_mathematics", "high_school_biology", "high_school_chemistry",
        "high_school_computer_science", "high_school_mathematics",
        "high_school_physics", "high_school_statistics", "machine_learning",
    ],
    "Humanities": [
        "formal_logic", "high_school_european_history",
        "high_school_us_history", "high_school_world_history",
        "international_law", "jurisprudence", "logical_fallacies",
        "moral_disputes", "moral_scenarios", "philosophy", "prehistory",
        "professional_law", "world_religions",
    ],
    "Social Sciences": [
        "econometrics", "high_school_geography",
        "high_school_government_and_politics", "high_school_macroeconomics",
        "high_school_microeconomics", "high_school_psychology",
        "human_sexuality", "professional_psychology", "public_relations",
        "security_studies", "sociology", "us_foreign_policy",
    ],
    "Other": [
        "anatomy", "business_ethics", "clinical_knowledge", "college_medicine",
        "global_facts", "human_aging", "management", "marketing",
        "medical_genetics", "miscellaneous", "nutrition",
        "professional_accounting", "professional_medicine", "virology",
    ],
}

# Fine-grained 7-way grouping
SUBJECT_GROUPS_7WAY = {
    "STEM": [
        "abstract_algebra", "astronomy", "college_biology", "college_chemistry",
        "college_computer_science", "college_mathematics", "college_physics",
        "computer_security", "conceptual_physics", "electrical_engineering",
        "elementary_mathematics", "high_school_biology", "high_school_chemistry",
        "high_school_computer_science", "high_school_mathematics",
        "high_school_physics", "high_school_statistics", "machine_learning",
    ],
    "Humanities": [
        "formal_logic", "high_school_european_history",
        "high_school_us_history", "high_school_world_history",
        "logical_fallacies", "moral_disputes", "moral_scenarios",
        "philosophy", "prehistory", "world_religions",
    ],
    "Social Sciences": [
        "econometrics", "high_school_geography",
        "high_school_government_and_politics", "high_school_macroeconomics",
        "high_school_microeconomics", "high_school_psychology",
        "human_sexuality", "professional_psychology", "public_relations",
        "security_studies", "sociology", "us_foreign_policy",
    ],
    "Medicine & Health": [
        "anatomy", "clinical_knowledge", "college_medicine", "human_aging",
        "medical_genetics", "nutrition", "professional_medicine", "virology",
    ],
    "Law": ["international_law", "jurisprudence", "professional_law"],
    "Business": [
        "business_ethics", "management", "marketing", "professional_accounting",
    ],
    "Other": [
        "global_facts", "miscellaneous",
    ],
}

# Default grouping alias (used by probe_mmlu.py and other importers)
SUBJECT_GROUPS = SUBJECT_GROUPS_4WAY

# Build reverse maps: subject -> group
SUBJECT_TO_GROUP_4WAY: dict[str, str] = {}
for _group, _subjects in SUBJECT_GROUPS_4WAY.items():
    for _s in _subjects:
        SUBJECT_TO_GROUP_4WAY[_s] = _group

SUBJECT_TO_GROUP_7WAY: dict[str, str] = {}
for _group, _subjects in SUBJECT_GROUPS_7WAY.items():
    for _s in _subjects:
        SUBJECT_TO_GROUP_7WAY[_s] = _group

# Default alias
SUBJECT_TO_GROUP = SUBJECT_TO_GROUP_4WAY

# POS tags to merge into broader categories
POS_MERGE = {
    "CCONJ": "CONJ",
    "SCONJ": "CONJ",
    "SPACE": "OTHER",
    "X": "OTHER",
    "INTJ": "OTHER",
}


def format_question(row: dict) -> str:
    """Format an MMLU row into a plain-text question string."""
    q = row["question"]
    c = row["choices"]
    return f"Question: {q}\nA) {c[0]}\nB) {c[1]}\nC) {c[2]}\nD) {c[3]}"


# ---------------------------------------------------------------------------
# Data loading helpers (prompt-level probing)
# ---------------------------------------------------------------------------

def load_mmlu_datasets(
    split_dir: str,
    max_samples: "int | None" = None,
) -> "dict[str, 'pd.DataFrame']":
    """Load pre-split MMLU data as DataFrames.

    Expects Arrow datasets at ``split_dir/train/`` and ``split_dir/val/``.
    Returns a dict with a single key ``"mmlu"`` whose value is a DataFrame
    with columns: text, subject, question_idx, group_4way, group_7way.

    Both train and val rows are combined here; the caller (MMLURunner) uses
    ``load_splits()`` to keep them separate.  This function is provided for
    the no-splits path but the MMLU experiment always uses pre-split data.
    """
    import pandas as pd
    from datasets import load_from_disk
    from pathlib import Path

    split_dir = Path(split_dir)
    train_path = split_dir / "train"
    val_path = split_dir / "val"

    rows = []
    for split_name, path in [("train", train_path), ("val", val_path)]:
        if not path.exists():
            raise FileNotFoundError(
                f"MMLU split not found at {path}. "
                "Run: python scripts/probing/data/prepare_mmlu_splits.py"
            )
        ds = load_from_disk(str(path))
        for idx, row in enumerate(ds):
            rows.append({
                "text": row.get("text") or format_question(row),
                "subject": row["subject"],
                "question_idx": idx,
                "split": split_name,
            })

    df = pd.DataFrame(rows)
    df["group_4way"] = df["subject"].map(
        lambda s: SUBJECT_TO_GROUP_4WAY.get(s, "Other")
    )
    df["group_7way"] = df["subject"].map(
        lambda s: SUBJECT_TO_GROUP_7WAY.get(s, "Other")
    )

    if max_samples is not None:
        df = df.head(max_samples).reset_index(drop=True)

    return {"mmlu": df}


def load_mmlu_splits(
    split_dir: str,
    max_samples: "int | None" = None,
) -> "tuple[dict, dict]":
    """Load pre-split MMLU data and return (train_datasets, val_datasets).

    Each returned dict has key ``"mmlu"`` -> DataFrame.
    """
    import pandas as pd
    from datasets import load_from_disk
    from pathlib import Path

    split_dir = Path(split_dir)

    def _load_one(path, label):
        if not path.exists():
            raise FileNotFoundError(
                f"MMLU {label} split not found at {path}. "
                "Run: python scripts/probing/data/prepare_mmlu_splits.py"
            )
        ds = load_from_disk(str(path))
        rows = []
        for idx, row in enumerate(ds):
            rows.append({
                "text": row.get("text") or format_question(row),
                "subject": row["subject"],
                "question_idx": idx,
            })
        df = pd.DataFrame(rows)
        df["group_4way"] = df["subject"].map(
            lambda s: SUBJECT_TO_GROUP_4WAY.get(s, "Other")
        )
        df["group_7way"] = df["subject"].map(
            lambda s: SUBJECT_TO_GROUP_7WAY.get(s, "Other")
        )
        if max_samples is not None:
            df = df.head(max_samples).reset_index(drop=True)
        return {"mmlu": df}

    train_dict = _load_one(split_dir / "train", "train")
    val_dict = _load_one(split_dir / "val", "val")
    return train_dict, val_dict


def build_mmlu_prompts(df: "pd.DataFrame") -> "list[str]":
    """Return prompt strings from an MMLU DataFrame (already formatted)."""
    return df["text"].tolist()
