"""Characterization tests for the probing eval loops.

These pin the *current* behavior of the four eval functions in
``evaluation.probing.evaluation`` and the two probe-loop drivers in
``evaluation.probing.experiment_utils`` BEFORE the P3 dedup, so the
helper-extraction refactor can be proven byte-equivalent.

A deterministic majority-class fake probe is used so results depend only
on label distributions / split logic, never on training randomness.
"""

import os

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Deterministic fake probe (no .history_, no .save -> clean metrics dicts)
# ---------------------------------------------------------------------------

class _MajorityProbe:
    def __init__(self):
        self.cls = None

    def fit(self, X, y, X_val=None, y_val=None):
        vals, cnts = np.unique(np.asarray(y), return_counts=True)
        self.cls = int(vals[int(np.argmax(cnts))])

    def predict(self, X):
        n = X["hidden_states"].shape[0] if isinstance(X, dict) else len(X)
        return np.full(n, self.cls, dtype=int)


def _factory():
    return _MajorityProbe()


def _toy_pooled(seed=0):
    """Two datasets, ndarray X, binary labels with different majority class."""
    rng = np.random.default_rng(seed)
    hs = {
        "A": rng.standard_normal((20, 8)).astype(np.float32),
        "B": rng.standard_normal((16, 8)).astype(np.float32),
    }
    labels = {
        "A": np.array([0] * 14 + [1] * 6),   # majority 0
        "B": np.array([1] * 11 + [0] * 5),   # majority 1
    }
    return hs, labels


# ---------------------------------------------------------------------------
# cross_dataset_eval / within_dataset_eval
# ---------------------------------------------------------------------------

def test_cross_dataset_eval_characterization():
    from evaluation.probing.evaluation import cross_dataset_eval

    hs, labels = _toy_pooled()
    r = cross_dataset_eval(["A", "B"], hs, labels, _factory)

    assert sorted(r) == ["A", "B"]
    # held-out A: train on B (majority 1) -> predict all 1 on A (6/20 correct)
    assert r["A"]["accuracy"] == pytest.approx(6 / 20)
    assert r["A"]["n_samples"] == 20
    # held-out B: train on A (majority 0) -> predict all 0 on B (5/16 correct)
    assert r["B"]["accuracy"] == pytest.approx(5 / 16)
    assert r["B"]["n_samples"] == 16


def test_within_dataset_eval_characterization():
    from evaluation.probing.evaluation import within_dataset_eval

    hs, labels = _toy_pooled()
    r = within_dataset_eval(["A", "B"], hs, labels, _factory,
                            val_ratio=0.25, seed=42)

    assert sorted(r) == ["A", "B"]
    for name in ("A", "B"):
        n = len(labels[name])
        n_val = max(1, int(n * 0.25))
        assert r[name]["n_samples"] == n_val
        assert 0.0 <= r[name]["accuracy"] <= 1.0


def test_presplit_within_dataset_eval_characterization():
    from evaluation.probing.evaluation import presplit_within_dataset_eval

    hs, labels = _toy_pooled()
    test_hs, test_labels = _toy_pooled(seed=1)
    r = presplit_within_dataset_eval(
        ["A", "B"], hs, labels, test_hs, test_labels, _factory,
    )
    assert sorted(r) == ["A", "B"]
    # train A majority 0; test A labels = 14x0 + 6x1 -> 14/20
    assert r["A"]["accuracy"] == pytest.approx(14 / 20)
    # train B majority 1; test B labels = 11x1 + 5x0 -> 11/16
    assert r["B"]["accuracy"] == pytest.approx(11 / 16)


def test_presplit_cross_dataset_eval_characterization():
    from evaluation.probing.evaluation import presplit_cross_dataset_eval

    hs, labels = _toy_pooled()
    test_hs, test_labels = _toy_pooled(seed=1)
    r = presplit_cross_dataset_eval(
        ["A", "B"], hs, labels, test_hs, test_labels, _factory,
    )
    assert sorted(r) == ["A", "B"]
    # held-out A: train on B-train (majority 1) -> predict 1 on A-test (6/20)
    assert r["A"]["accuracy"] == pytest.approx(6 / 20)
    # held-out B: train on A-train (majority 0) -> predict 0 on B-test (5/16)
    assert r["B"]["accuracy"] == pytest.approx(5 / 16)


def test_eval_loops_write_confusion_matrices(tmp_path):
    from evaluation.probing.evaluation import within_dataset_eval

    hs, labels = _toy_pooled()
    within_dataset_eval(["A", "B"], hs, labels, _factory,
                        output_dir=str(tmp_path), probe_name="linear")
    assert (tmp_path / "cm_within_linear_A.png").exists()
    assert (tmp_path / "cm_within_linear_B.png").exists()


# ---------------------------------------------------------------------------
# run_probes_and_save / run_probes_and_save_presplit (skeleton drivers)
# ---------------------------------------------------------------------------

class _Eval:
    def __init__(self, eval_type="within_dataset", val_ratio=0.25):
        self.eval_type = eval_type
        self.val_ratio = val_ratio


class _DriverCfg:
    def __init__(self, output_dir, eval_type="within_dataset"):
        self.output_dir = output_dir
        self.eval = _Eval(eval_type=eval_type)


def test_run_probes_and_save_characterization(tmp_path):
    from evaluation.probing.experiment_utils import run_probes_and_save

    hs0, labels = _toy_pooled()
    hs_cache = {"A": {"average": hs0["A"]}, "B": {"average": hs0["B"]}}
    cfg = _DriverCfg(str(tmp_path), eval_type="within_dataset")

    out = run_probes_and_save(
        cfg, [("linear", _factory)], hs_cache, labels,
        ["A", "B"], ["average"], "toy",
    )
    assert list(out) == ["linear"]
    assert "average/within_dataset" in out["linear"]
    assert sorted(out["linear"]["average/within_dataset"]["linear"]) == ["A", "B"]
    assert os.path.exists(os.path.join(tmp_path, "linear_probe", "results.json"))


def test_run_probes_and_save_presplit_characterization(tmp_path):
    from evaluation.probing.experiment_utils import run_probes_and_save_presplit

    hs0, labels = _toy_pooled()
    hs1, test_labels = _toy_pooled(seed=1)
    train_hs = {"A": {"average": hs0["A"]}, "B": {"average": hs0["B"]}}
    test_hs = {"A": {"average": hs1["A"]}, "B": {"average": hs1["B"]}}
    cfg = _DriverCfg(str(tmp_path))

    out = run_probes_and_save_presplit(
        cfg, [("linear", _factory)], train_hs, labels, test_hs, test_labels,
        ["A", "B"], ["average"], "toy", eval_type="within_dataset",
    )
    assert list(out) == ["linear"]
    assert "average/within_dataset" in out["linear"]
    r = out["linear"]["average/within_dataset"]["linear"]
    assert r["A"]["accuracy"] == pytest.approx(14 / 20)
    assert r["B"]["accuracy"] == pytest.approx(11 / 16)
    assert os.path.exists(os.path.join(tmp_path, "linear_probe", "results.json"))
