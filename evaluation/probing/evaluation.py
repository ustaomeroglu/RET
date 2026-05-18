"""
Evaluation utilities: metrics, confusion matrix, train/eval splits.

Public API
----------
compute_metrics(y_true, y_pred)             -> dict
confusion_matrix(y_true, y_pred, classes)   -> np.ndarray [K, K]
print_confusion_matrix(cm, classes, title)
save_confusion_matrix(cm, classes, path)
train_and_eval_probe(probe, X_tr, y_tr, X_val, y_val) -> dict
cross_dataset_eval(...)                     -> dict
within_dataset_eval(...)                    -> dict
print_results(results_by_probe, title)

Note on X formats
-----------------
All functions accept X as either:
  - np.ndarray [N, D]       (Linear / DOM probes)
  - dict{"hidden_states": [N,S,D], "attention_mask": [N,S]}  (Attention probe)

The helper ``_index_X`` handles subsetting either format by row index.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Callable, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Row-indexing helper (works for ndarray and dict)
# ---------------------------------------------------------------------------

def _index_X(X, idx: np.ndarray):
    """Subset X (ndarray or dict) by row indices."""
    if isinstance(X, dict):
        return {k: v[idx] for k, v in X.items()}
    return X[idx]


def _len_X(X) -> int:
    if isinstance(X, dict):
        return len(next(iter(X.values())))
    return len(X)


def _concat_X(Xs: list):
    """Concatenate a list of X values (ndarray or dict).

    For dict inputs (AttentionProbe's all_tokens mode), handles different
    sequence lengths by left-padding shorter sequences to the maximum
    sequence length before concatenation.
    """
    if not Xs:
        return None
    if isinstance(Xs[0], dict):
        # Find the max sequence length across all X dicts
        max_S = max(x["hidden_states"].shape[1] for x in Xs)
        D = Xs[0]["hidden_states"].shape[2]
        has_orig = "original_attention_mask" in Xs[0]
        padded_hs, padded_masks, padded_orig = [], [], []
        for x in Xs:
            N, S, _ = x["hidden_states"].shape
            if S < max_S:
                ph = np.zeros((N, max_S, D), dtype=np.float32)
                pm = np.zeros((N, max_S), dtype=np.float32)
                ph[:, max_S - S:, :] = x["hidden_states"]
                pm[:, max_S - S:] = x["attention_mask"]
                padded_hs.append(ph)
                padded_masks.append(pm)
                if has_orig:
                    po = np.zeros((N, max_S), dtype=np.float32)
                    po[:, max_S - S:] = x["original_attention_mask"]
                    padded_orig.append(po)
            else:
                padded_hs.append(x["hidden_states"])
                padded_masks.append(x["attention_mask"])
                if has_orig:
                    padded_orig.append(x["original_attention_mask"])
        result = {
            "hidden_states": np.concatenate(padded_hs, axis=0),
            "attention_mask": np.concatenate(padded_masks, axis=0),
        }
        if has_orig:
            result["original_attention_mask"] = np.concatenate(padded_orig, axis=0)
        return result
    return np.concatenate(Xs, axis=0)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """Compute accuracy, balanced accuracy, macro-F1, and confusion matrix."""
    y_true   = np.asarray(y_true)
    y_pred   = np.asarray(y_pred)
    classes  = np.unique(np.concatenate([y_true, y_pred]))

    accuracy = float((y_true == y_pred).mean())

    # Balanced accuracy
    recalls = []
    for c in classes:
        mask = y_true == c
        if mask.sum() == 0:
            continue
        recalls.append(float((y_pred[mask] == c).mean()))
    balanced_acc = float(np.mean(recalls)) if recalls else 0.0

    # F1
    if len(classes) == 2:
        pos = int(classes.max())
        tp = float(((y_pred == pos) & (y_true == pos)).sum())
        fp = float(((y_pred == pos) & (y_true != pos)).sum())
        fn = float(((y_pred != pos) & (y_true == pos)).sum())
        prec = tp / (tp + fp + 1e-8)
        rec  = tp / (tp + fn + 1e-8)
        f1   = 2 * prec * rec / (prec + rec + 1e-8)
    else:
        f1s = []
        for c in classes:
            tp = float(((y_pred == c) & (y_true == c)).sum())
            fp = float(((y_pred == c) & (y_true != c)).sum())
            fn = float(((y_pred != c) & (y_true == c)).sum())
            prec = tp / (tp + fp + 1e-8)
            rec  = tp / (tp + fn + 1e-8)
            f1s.append(2 * prec * rec / (prec + rec + 1e-8))
        f1 = float(np.mean(f1s))

    cm = confusion_matrix(y_true, y_pred, classes=sorted(classes.tolist()))

    return {
        "accuracy":          accuracy,
        "balanced_accuracy": balanced_acc,
        "f1":                f1,
        "n_samples":         int(len(y_true)),
        "n_positive":        int((y_true == y_true.max()).sum()),
        "confusion_matrix":  cm.tolist(),
        "cm_classes":        sorted(classes.tolist()),
    }


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------

def confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: Optional[List[int]] = None,
) -> np.ndarray:
    """Return a K x K confusion matrix (rows=true, cols=predicted)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if classes is None:
        classes = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    K = len(classes)
    cls_idx = {c: i for i, c in enumerate(classes)}
    cm = np.zeros((K, K), dtype=int)
    for t, p in zip(y_true.tolist(), y_pred.tolist()):
        if t in cls_idx and p in cls_idx:
            cm[cls_idx[t], cls_idx[p]] += 1
    return cm


def print_confusion_matrix(
    cm: np.ndarray,
    classes: List[int],
    title: str = "Confusion Matrix",
) -> None:
    """Pretty-print a confusion matrix to stdout."""
    print(f"\n  {title}")
    labels = [str(c) for c in classes]
    col_w  = max(max(len(l) for l in labels), 6) + 2
    # Header
    print(" " * (col_w + 2) + "".join(f"{l:>{col_w}}" for l in labels) + "  <- predicted")
    # Rows
    for i, label in enumerate(labels):
        row_vals = "".join(f"{cm[i, j]:>{col_w}}" for j in range(len(classes)))
        print(f"  {label:>{col_w}}{row_vals}")
    print("  ^ true\n")


def save_confusion_matrix(
    cm: np.ndarray,
    classes: List[int],
    path: str,
) -> None:
    """Save a confusion matrix as PNG heatmap (or JSON if path ends in .json)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    if path.endswith(".json"):
        data = {"classes": classes, "matrix": cm.tolist()}
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    else:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        K = len(classes)
        fig, ax = plt.subplots(figsize=(max(5, K * 0.8), max(4, K * 0.7)))
        im = ax.imshow(cm, cmap="Blues", aspect="auto")
        fig.colorbar(im, ax=ax)

        labels = [str(c) for c in classes]
        ax.set_xticks(range(K))
        ax.set_yticks(range(K))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

        for i in range(K):
            for j in range(K):
                color = "white" if cm[i, j] > cm.max() / 2 else "black"
                ax.text(j, i, str(int(cm[i, j])), ha="center", va="center",
                        color=color, fontsize=max(8, 14 - K))

        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    logger.info("Confusion matrix saved to %s", path)


# ---------------------------------------------------------------------------
# Single train/eval run
# ---------------------------------------------------------------------------

def train_and_eval_probe(
    probe,
    X_train,
    y_train: np.ndarray,
    X_val,
    y_val: np.ndarray,
    cm_save_path: Optional[str] = None,
    model_save_path: Optional[str] = None,
) -> Dict:
    """Fit probe, evaluate, return metrics (including confusion matrix).

    Parameters
    ----------
    cm_save_path    : if given, save confusion matrix to this path (.png or .json).
    model_save_path : if given, save trained probe weights to this path.
    """
    probe.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    y_pred   = probe.predict(X_val)
    metrics  = compute_metrics(y_val, y_pred)
    classes  = metrics["cm_classes"]
    cm       = np.array(metrics["confusion_matrix"])

    print_confusion_matrix(cm, classes)

    if cm_save_path:
        save_confusion_matrix(cm, classes, cm_save_path)

    if model_save_path and hasattr(probe, "save"):
        os.makedirs(os.path.dirname(model_save_path) or ".", exist_ok=True)
        probe.save(model_save_path)
        logger.info("Probe weights saved to %s", model_save_path)

    # Attach training history if the probe tracked it
    if hasattr(probe, "history_"):
        metrics["training_history"] = probe.history_

    return metrics


def train_and_eval_probe_multisplit(
    probe,
    X_train,
    y_train: np.ndarray,
    eval_splits: Dict[str, tuple],
    cm_save_dir: Optional[str] = None,
    cm_name_prefix: str = "probe",
    model_save_path: Optional[str] = None,
) -> tuple:
    """Train probe once, evaluate on multiple splits.

    Parameters
    ----------
    eval_splits     : {"val": (X_val, y_val), "test": (X_test, y_test)}
                      The first split is also used for validation during training.
    cm_save_dir     : directory to save confusion matrix PNGs
    cm_name_prefix  : prefix for confusion matrix filenames
    model_save_path : if given, save trained probe weights

    Returns
    -------
    (probe, {split_name: metrics_dict})
    """
    # Use the first split for validation during training
    first_split = next(iter(eval_splits.values()))
    X_val_train, y_val_train = first_split

    probe.fit(X_train, y_train, X_val=X_val_train, y_val=y_val_train)

    if model_save_path and hasattr(probe, "save"):
        os.makedirs(os.path.dirname(model_save_path) or ".", exist_ok=True)
        probe.save(model_save_path)
        logger.info("Probe weights saved to %s", model_save_path)

    split_metrics = {}
    for split_name, (X_split, y_split) in eval_splits.items():
        y_pred = probe.predict(X_split)
        metrics = compute_metrics(y_split, y_pred)
        classes = metrics["cm_classes"]
        cm = np.array(metrics["confusion_matrix"])

        print_confusion_matrix(cm, classes, title=f"{cm_name_prefix} ({split_name})")

        if cm_save_dir:
            os.makedirs(cm_save_dir, exist_ok=True)
            cm_path = os.path.join(cm_save_dir, f"{cm_name_prefix}_{split_name}.png")
            save_confusion_matrix(cm, classes, cm_path)

        split_metrics[split_name] = metrics

    # Attach training history to the first split's metrics
    if hasattr(probe, "history_"):
        first_key = next(iter(split_metrics))
        split_metrics[first_key]["training_history"] = probe.history_

    return probe, split_metrics


# ---------------------------------------------------------------------------
# Shared fit/eval for one (key) — used by all four eval loops below
# ---------------------------------------------------------------------------

def _fit_eval_one(
    key: str,
    X_train,
    y_train: np.ndarray,
    X_val,
    y_val: np.ndarray,
    probe_factory: Callable,
    *,
    kind: str,                       # "cross" | "within" — only the cm filename
    output_dir: Optional[str],
    probe_name: str,
    model_save_dir: Optional[str],
) -> Dict:
    """Build cm/model paths, fit a fresh probe, return metrics.

    This is the block that was duplicated verbatim across the four eval
    loops; callers keep their own skip-logic and (distinct) log lines.
    """
    cm_path = (
        os.path.join(output_dir, f"cm_{kind}_{probe_name}_{key}.png")
        if output_dir else None
    )
    model_path = (
        os.path.join(model_save_dir, f"model_{key}.pt")
        if model_save_dir else None
    )
    probe = probe_factory()
    return train_and_eval_probe(
        probe, X_train, y_train, X_val, y_val,
        cm_save_path=cm_path, model_save_path=model_path,
    )


# ---------------------------------------------------------------------------
# Cross-dataset evaluation (leave-one-out)
# ---------------------------------------------------------------------------

def cross_dataset_eval(
    dataset_names: List[str],
    hidden_states: Dict,
    labels: Dict[str, np.ndarray],
    probe_factory: Callable,
    output_dir: Optional[str] = None,
    probe_name: str = "probe",
    model_save_dir: Optional[str] = None,
) -> Dict[str, Dict]:
    """Leave-one-out cross-dataset evaluation.

    Works with both ndarray and dict X values (for AttentionProbe).
    """
    results: Dict[str, Dict] = {}

    for held_out in dataset_names:
        train_X_list, train_y_list = [], []
        for name in dataset_names:
            if name == held_out:
                continue
            if _len_X(hidden_states[name]) == 0:
                continue
            train_X_list.append(hidden_states[name])
            train_y_list.append(labels[name])

        if not train_X_list:
            logger.warning("No training data after removing %s; skipping.", held_out)
            continue

        X_train = _concat_X(train_X_list)
        y_train = np.concatenate(train_y_list, axis=0)
        X_val   = hidden_states[held_out]
        y_val   = labels[held_out]

        if _len_X(X_val) == 0:
            logger.warning("Held-out dataset %s is empty; skipping.", held_out)
            continue

        metrics = _fit_eval_one(
            held_out, X_train, y_train, X_val, y_val, probe_factory,
            kind="cross", output_dir=output_dir, probe_name=probe_name,
            model_save_dir=model_save_dir,
        )
        results[held_out] = metrics
        logger.info(
            "Cross-dataset [held-out=%s]: acc=%.3f  bal_acc=%.3f  f1=%.3f  (n_val=%d)",
            held_out, metrics["accuracy"], metrics["balanced_accuracy"],
            metrics["f1"], metrics["n_samples"],
        )

    return results


# ---------------------------------------------------------------------------
# Within-dataset evaluation (train/val split)
# ---------------------------------------------------------------------------

def within_dataset_eval(
    dataset_names: List[str],
    hidden_states: Dict,
    labels: Dict[str, np.ndarray],
    probe_factory: Callable,
    val_ratio: float = 0.2,
    seed: int = 42,
    output_dir: Optional[str] = None,
    probe_name: str = "probe",
    model_save_dir: Optional[str] = None,
) -> Dict[str, Dict]:
    """Per-dataset train/val split evaluation (works for ndarray or dict X)."""
    results: Dict[str, Dict] = {}
    rng = np.random.default_rng(seed)

    for name in dataset_names:
        X = hidden_states[name]
        y = labels[name]
        if _len_X(X) == 0:
            logger.warning("Dataset %s is empty; skipping.", name)
            continue

        n       = _len_X(X)
        idx     = rng.permutation(n)
        n_val   = max(1, int(n * val_ratio))
        val_idx   = idx[:n_val]
        train_idx = idx[n_val:]

        X_train = _index_X(X, train_idx)
        y_train = y[train_idx]
        X_val   = _index_X(X, val_idx)
        y_val   = y[val_idx]

        metrics = _fit_eval_one(
            name, X_train, y_train, X_val, y_val, probe_factory,
            kind="within", output_dir=output_dir, probe_name=probe_name,
            model_save_dir=model_save_dir,
        )
        results[name] = metrics
        logger.info(
            "Within-dataset [%s]: acc=%.3f  bal_acc=%.3f  f1=%.3f  (n_train=%d, n_val=%d)",
            name, metrics["accuracy"], metrics["balanced_accuracy"],
            metrics["f1"], len(train_idx), len(val_idx),
        )

    return results


# ---------------------------------------------------------------------------
# Pre-split cross-dataset evaluation
# ---------------------------------------------------------------------------

def presplit_cross_dataset_eval(
    dataset_names: List[str],
    train_hidden_states: Dict,
    train_labels: Dict[str, np.ndarray],
    test_hidden_states: Dict,
    test_labels: Dict[str, np.ndarray],
    probe_factory: Callable,
    output_dir: Optional[str] = None,
    probe_name: str = "probe",
    model_save_dir: Optional[str] = None,
) -> Dict[str, Dict]:
    """Cross-dataset evaluation with pre-split train/test data.

    For each held-out dataset:
        train probe on TRAIN data from all OTHER datasets,
        evaluate on TEST data from the held-out dataset.
    """
    results: Dict[str, Dict] = {}

    for held_out in dataset_names:
        train_X_list, train_y_list = [], []
        for name in dataset_names:
            if name == held_out:
                continue
            if name not in train_hidden_states or _len_X(train_hidden_states[name]) == 0:
                continue
            train_X_list.append(train_hidden_states[name])
            train_y_list.append(train_labels[name])

        if not train_X_list:
            logger.warning("No training data after removing %s; skipping.", held_out)
            continue

        if held_out not in test_hidden_states or _len_X(test_hidden_states[held_out]) == 0:
            logger.warning("No test data for held-out %s; skipping.", held_out)
            continue

        X_train = _concat_X(train_X_list)
        y_train = np.concatenate(train_y_list, axis=0)
        X_val   = test_hidden_states[held_out]
        y_val   = test_labels[held_out]

        metrics = _fit_eval_one(
            held_out, X_train, y_train, X_val, y_val, probe_factory,
            kind="cross", output_dir=output_dir, probe_name=probe_name,
            model_save_dir=model_save_dir,
        )
        results[held_out] = metrics
        logger.info(
            "Presplit cross [held-out=%s]: acc=%.3f  bal_acc=%.3f  f1=%.3f  (n_train=%d, n_val=%d)",
            held_out, metrics["accuracy"], metrics["balanced_accuracy"],
            metrics["f1"], len(y_train), metrics["n_samples"],
        )

    return results


# ---------------------------------------------------------------------------
# Pre-split within-dataset evaluation
# ---------------------------------------------------------------------------

def presplit_within_dataset_eval(
    dataset_names: List[str],
    train_hidden_states: Dict,
    train_labels: Dict[str, np.ndarray],
    test_hidden_states: Dict,
    test_labels: Dict[str, np.ndarray],
    probe_factory: Callable,
    output_dir: Optional[str] = None,
    probe_name: str = "probe",
    model_save_dir: Optional[str] = None,
) -> Dict[str, Dict]:
    """Within-dataset evaluation using pre-split train/test data (no internal splitting)."""
    results: Dict[str, Dict] = {}

    for name in dataset_names:
        if name not in train_hidden_states or _len_X(train_hidden_states[name]) == 0:
            logger.warning("No train data for %s; skipping.", name)
            continue
        if name not in test_hidden_states or _len_X(test_hidden_states[name]) == 0:
            logger.warning("No test data for %s; skipping.", name)
            continue

        X_train = train_hidden_states[name]
        y_train = train_labels[name]
        X_val   = test_hidden_states[name]
        y_val   = test_labels[name]

        metrics = _fit_eval_one(
            name, X_train, y_train, X_val, y_val, probe_factory,
            kind="within", output_dir=output_dir, probe_name=probe_name,
            model_save_dir=model_save_dir,
        )
        results[name] = metrics
        logger.info(
            "Presplit within [%s]: acc=%.3f  bal_acc=%.3f  f1=%.3f  (n_train=%d, n_val=%d)",
            name, metrics["accuracy"], metrics["balanced_accuracy"],
            metrics["f1"], len(y_train), metrics["n_samples"],
        )

    return results


# ---------------------------------------------------------------------------
# Result printing
# ---------------------------------------------------------------------------

def print_results(
    results_by_probe: Dict[str, Dict[str, Dict]],
    title: str = "Results",
) -> None:
    """Print a formatted results table."""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")
    col_w = 16
    probe_header = " " * 24 + "".join(
        f"{name:>{3 * col_w}}" for name in results_by_probe
    )
    header = f"{'Dataset':<24}" + "".join(
        f"{'Acc':>{col_w}}{'BAcc':>{col_w}}{'F1':>{col_w}}"
        for _ in results_by_probe
    )
    print(probe_header)
    print(header)
    print("-" * 70)

    all_datasets: set = set()
    for probe_res in results_by_probe.values():
        all_datasets.update(probe_res.keys())

    for ds in sorted(all_datasets):
        row = f"{ds:<24}"
        for probe_res in results_by_probe.values():
            m    = probe_res.get(ds, {})
            acc  = m.get("accuracy",          float("nan"))
            bacc = m.get("balanced_accuracy", float("nan"))
            f1   = m.get("f1",                float("nan"))
            row += f"{acc:>{col_w}.3f}{bacc:>{col_w}.3f}{f1:>{col_w}.3f}"
        print(row)
    print()
