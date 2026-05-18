#!/usr/bin/env python
"""Train linear probes on cached MMLU embeddings: semantic vs syntactic.

For each method (h, h^PCA128, SAE, RET z=128, RET z=256):
  - Semantic task: classify MMLU 7-way subject group
  - Syntactic task: classify Universal POS tag (X tokens filtered as
    alignment artifacts — same filter used in the t-SNE paper figure).

Train/test split is 80/20 at the QUESTION level (no token-level leakage).
Features are per-method standardized (with_mean=False for sparse SAE).

Outputs to Paper_figures_2/mmlu_analysis/:
  linear_probe_results.json
  linear_probe_accuracy.png
  linear_probe_accuracy.pdf
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

from evaluation.datasets.mmlu.data import SUBJECT_TO_GROUP_7WAY
from scripts.probing.plotting.mmlu_comparison import _build_raw_pos_tags

ROOT = Path(__file__).resolve().parents[3]  # repo root (scripts/probing/plotting/ -> 3 up)
OUT_DIR = ROOT / "Paper_figures_2" / "mmlu_analysis"
SEED = 42
TRAIN_FRAC = 0.8

# (label, cache_path, is_sparse)
METHODS = [
    ("h (768)",                      ROOT / "outputs/Pythia160M/pile/baselines/h/analysis/mmlu_embeddings_hh.npz", False),
    ("h^PCA128 (MMLU-train-fit)",    ROOT / "outputs/Pythia160M/pile/baselines/h_pca128/analysis/mmlu_embeddings_pca_128_mmlutrain.npz", False),
    ("h^PCA256 (pile-fit)",          ROOT / "outputs/Pythia160M/pile/baselines/h_pca256/analysis/mmlu_embeddings_pca_256_pile.npz", False),
    ("h^PCA256 (MMLU-train-fit)",    ROOT / "outputs/Pythia160M/pile/baselines/h_pca256/analysis/mmlu_embeddings_pca_256_mmlu_train.npz", False),
    ("SAE (32k, k=32)",              ROOT / "outputs/Pythia160M/pile/baselines/sae/analysis/mmlu_embeddings_sae.npz", True),
    ("RET pile z=128",               ROOT / "outputs/Pythia160M/pile/ret_autonomous/analysis/mmlu_embeddings.npz", False),
    ("RET pile z=256",               ROOT / "outputs/Pythia160M/pile/ret_dense_autonomous_z256/analysis/mmlu_embeddings.npz", False),
    ("RET MMLU z=128",               ROOT / "outputs/Pythia160M/mmlu/ret_dense_autonomous/analysis/mmlu_embeddings.npz", False),
    ("RET MMLU supervised z=128",    ROOT / "outputs/Pythia160M/mmlu/ret_dense_autonomous_supervised/analysis/mmlu_embeddings.npz", False),
]


def _question_level_split(q_indices: np.ndarray, seed: int = SEED):
    """Return (train_mask, test_mask) partitioning by MMLU question index."""
    unique = np.unique(q_indices)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    cut = int(len(unique) * TRAIN_FRAC)
    train_q = set(unique[:cut].tolist())
    train_mask = np.array([q in train_q for q in q_indices], dtype=bool)
    return train_mask, ~train_mask


def _fit_probe(X_tr, y_tr, X_te, y_te, is_sparse: bool) -> dict:
    """Fit multinomial logistic regression, return metrics."""
    # lbfgs is much faster on dense; saga is needed for sparse SAE.
    solver = "saga" if is_sparse else "lbfgs"
    clf = LogisticRegression(
        max_iter=200,
        C=1.0,
        solver=solver,
        n_jobs=-1,
        tol=1e-3,
        verbose=1,
    )
    print(f"    solver={solver}")
    clf.fit(X_tr, y_tr)
    pred_tr = clf.predict(X_tr)
    pred_te = clf.predict(X_te)
    return {
        "train_acc": float((pred_tr == y_tr).mean()),
        "test_acc": float((pred_te == y_te).mean()),
        "test_f1_macro": float(f1_score(y_te, pred_te, average="macro")),
        "n_train": int(len(y_tr)),
        "n_test": int(len(y_te)),
        "n_classes": int(len(np.unique(y_tr))),
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Load reference labels (shared across methods) ---
    print("Loading reference labels from first cache ...")
    ref = np.load(METHODS[0][1], allow_pickle=True)
    subjects_all = np.array(ref["subjects"])
    q_indices_all = ref["q_indices"]
    print(f"  {len(subjects_all)} tokens.")

    # --- Rebuild raw POS tags and mask X (alignment artifacts) ---
    cache_raw = OUT_DIR / "_raw_pos_tags_cache.npy"
    if cache_raw.exists():
        raw_tags = np.load(cache_raw, allow_pickle=True)
        print(f"Loaded raw POS tags from cache: {cache_raw} ({len(raw_tags)} tokens)")
    else:
        raw_tags = np.array(_build_raw_pos_tags())
        np.save(cache_raw, raw_tags)
        print(f"Cached raw POS tags to {cache_raw}")
    assert len(raw_tags) == len(subjects_all), "POS tag alignment mismatch"
    keep = raw_tags != "X"
    print(f"Filtering X: keeping {keep.sum()}/{len(keep)} tokens.")

    # --- Build labels (only on kept indices) ---
    subjects_kept = subjects_all[keep]
    pos_kept = np.array([t for t, k in zip(ref["pos_tags"], keep) if k])
    q_kept = q_indices_all[keep]

    sem_labels = np.array([SUBJECT_TO_GROUP_7WAY.get(s, "Other") for s in subjects_kept])
    syn_labels = pos_kept  # merged POS, X already filtered

    le_sem = LabelEncoder().fit(sem_labels)
    le_syn = LabelEncoder().fit(syn_labels)
    y_sem = le_sem.transform(sem_labels)
    y_syn = le_syn.transform(syn_labels)

    train_mask, test_mask = _question_level_split(q_kept)
    print(f"Question-level split: {train_mask.sum()} train, {test_mask.sum()} test.")
    print(f"Semantic classes ({len(le_sem.classes_)}): {list(le_sem.classes_)}")
    print(f"Syntactic classes ({len(le_syn.classes_)}): {list(le_syn.classes_)}")

    # --- Per-method: load z, standardize, train probes ---
    results: dict[str, dict] = {}
    for label, cache_path, is_sparse in METHODS:
        print(f"\n=== {label} ===")
        npz = np.load(cache_path, allow_pickle=True)
        z = npz["z_all"][keep]
        print(f"  z shape: {z.shape} dtype={z.dtype}")

        if is_sparse:
            X = sp.csr_matrix(z)
            print(f"  converted to CSR, nnz={X.nnz}")
            scaler = StandardScaler(with_mean=False)
            X = scaler.fit_transform(X)
        else:
            scaler = StandardScaler()
            X = scaler.fit_transform(z.astype(np.float64))

        # Slice
        if is_sparse:
            X_tr, X_te = X[train_mask], X[test_mask]
        else:
            X_tr, X_te = X[train_mask], X[test_mask]

        print(f"  fitting semantic probe ...")
        sem = _fit_probe(X_tr, y_sem[train_mask], X_te, y_sem[test_mask], is_sparse)
        print(f"    semantic: test_acc={sem['test_acc']:.3f} f1_macro={sem['test_f1_macro']:.3f}")

        print(f"  fitting syntactic probe ...")
        syn = _fit_probe(X_tr, y_syn[train_mask], X_te, y_syn[test_mask], is_sparse)
        print(f"    syntactic: test_acc={syn['test_acc']:.3f} f1_macro={syn['test_f1_macro']:.3f}")

        results[label] = {"semantic": sem, "syntactic": syn, "dim": int(z.shape[1])}

        # Free memory
        del z, X, X_tr, X_te, npz

    # --- Save JSON ---
    results_out = {
        "config": {
            "seed": SEED,
            "train_frac": TRAIN_FRAC,
            "semantic_task": "MMLU 7-way subject group",
            "syntactic_task": "Universal POS tag (X filtered)",
            "n_kept_tokens": int(keep.sum()),
            "n_train_tokens": int(train_mask.sum()),
            "n_test_tokens": int(test_mask.sum()),
            "semantic_classes": list(le_sem.classes_),
            "syntactic_classes": list(le_syn.classes_),
            "probe": "LogisticRegression(max_iter=500, C=1.0, solver=saga, multinomial)",
        },
        "results": results,
    }
    json_path = OUT_DIR / "linear_probe_results.json"
    with open(json_path, "w") as f:
        json.dump(results_out, f, indent=2)
    print(f"\nSaved {json_path}")

    # --- Bar chart ---
    labels = [m[0] for m in METHODS]
    sem_acc = [results[l]["semantic"]["test_acc"] for l in labels]
    syn_acc = [results[l]["syntactic"]["test_acc"] for l in labels]

    x = np.arange(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(13.5, 5.0))
    b1 = ax.bar(x - w/2, sem_acc, w, label="Semantic (7-way subject group)", color="#3366cc")
    b2 = ax.bar(x + w/2, syn_acc, w, label="Syntactic (POS tag)", color="#dc8a44")

    # Annotate bars
    for bars in (b1, b2):
        for rect in bars:
            h = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2, h + 0.008,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Test accuracy")
    ax.set_ylim(0, 1.0)
    ax.set_title("Linear probes on MMLU token representations (Pythia-160M L6)")
    ax.legend(loc="upper left", frameon=True)
    ax.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.5)

    fig.tight_layout()
    png_path = OUT_DIR / "linear_probe_accuracy.png"
    pdf_path = OUT_DIR / "linear_probe_accuracy.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")


if __name__ == "__main__":
    main()
