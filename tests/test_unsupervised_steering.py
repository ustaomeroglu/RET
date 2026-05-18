"""Unit tests for the unsupervised-steering library.

Covers the small, fast-to-test surface:
- ``build_rules_for_condition`` for all condition shapes
- ``SteeringRule`` defaults and field validation
- ``pick_best_closure_predictor`` end-to-end on a synthetic closure dir
- public-API importability of ``attach_steering_hook``

No GPU, no LLM load, no API.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from evaluation.closure.io import pick_best_closure_predictor, save_closure_predictors
from evaluation.closure.predictors import build_closure_predictor
from evaluation.unsupervised_steering import (
    attach_steering_hook,  # importability check
    build_rules_for_condition,
)
from evaluation.unsupervised_steering.config import SteeringRule


# ---------------------------------------------------------------------------
# SteeringRule defaults
# ---------------------------------------------------------------------------

def test_steering_rule_empty_spec_raises() -> None:
    """A SteeringRule with no target and no avoid is rejected — the
    dataclass enforces 'exactly one of target_group/target_cluster' OR an
    avoid spec."""
    with pytest.raises(ValueError, match="target_group .*target_cluster"):
        SteeringRule()


def test_steering_rule_attractor_construction() -> None:
    r = SteeringRule(trigger_position=0, target_cluster=29, duration=50)
    assert r.trigger_position == 0
    assert r.target_cluster == 29
    assert r.duration == 50
    assert r.avoid_cluster is None  # repulsor fields untouched


def test_steering_rule_target_group_and_cluster_conflict_raises() -> None:
    """Cannot set both target_group and target_cluster — XOR is enforced."""
    with pytest.raises(ValueError, match="target_group .*target_cluster"):
        SteeringRule(target_group=1, target_cluster=29, trigger_position=0)


def test_steering_rule_repulsor_construction() -> None:
    r = SteeringRule(
        avoid_cluster=41,
        avoid_margin_threshold=0.15,
        avoid_sticky_duration=5,
        avoid_from_step=10,
    )
    assert r.avoid_cluster == 41
    assert r.avoid_margin_threshold == pytest.approx(0.15)
    assert r.avoid_sticky_duration == 5
    assert r.target_cluster is None  # attractor fields untouched


# ---------------------------------------------------------------------------
# build_rules_for_condition — all 9 shapes
# ---------------------------------------------------------------------------

def test_baseline_condition_produces_no_rules() -> None:
    label, rules = build_rules_for_condition(("baseline",))
    assert label == "baseline"
    assert rules == []


def test_steer_C_single_cluster() -> None:
    label, rules = build_rules_for_condition(
        ("steer_C", 29), attractor_duration=50, trigger_position=0,
    )
    assert label == "steer_C29_dur50"
    assert len(rules) == 1
    r = rules[0]
    assert r.target_cluster == 29
    assert r.target_group is None
    assert r.trigger_position == 0
    assert r.duration == 50


def test_steer_G_single_group() -> None:
    label, rules = build_rules_for_condition(
        ("steer_G", 7), attractor_duration=100, trigger_position=5,
    )
    assert label == "steer_G7_dur100"
    assert len(rules) == 1
    r = rules[0]
    assert r.target_group == 7
    assert r.target_cluster is None
    assert r.trigger_position == 5
    assert r.duration == 100


def test_avoid_C_single_cluster() -> None:
    label, rules = build_rules_for_condition(
        ("avoid_C", 41),
        avoid_margin_threshold=0.20, avoid_sticky_duration=8, avoid_from_step=3,
    )
    assert label == "avoid_C41_sticky_st8"
    assert len(rules) == 1
    r = rules[0]
    assert r.avoid_cluster == 41
    assert r.avoid_margin_threshold == pytest.approx(0.20)
    assert r.avoid_sticky_duration == 8
    assert r.avoid_from_step == 3


def test_avoid_G_single_group() -> None:
    _, rules = build_rules_for_condition(("avoid_G", 5))
    assert len(rules) == 1
    assert rules[0].avoid_group == 5
    assert rules[0].avoid_cluster is None


def test_steer_Cs_multi_cluster() -> None:
    label, rules = build_rules_for_condition(
        ("steer_Cs", [10, 25, 38]), attractor_duration=40,
    )
    assert label == "steer_C10_C25_C38_dur40"
    assert len(rules) == 3
    assert {r.target_cluster for r in rules} == {10, 25, 38}
    assert all(r.duration == 40 for r in rules)


def test_steer_Gs_multi_group() -> None:
    label, rules = build_rules_for_condition(("steer_Gs", [3, 7]), attractor_duration=20)
    assert label == "steer_G3_G7_dur20"
    assert len(rules) == 2
    assert {r.target_group for r in rules} == {3, 7}


def test_avoid_Cs_multi_cluster() -> None:
    label, rules = build_rules_for_condition(
        ("avoid_Cs", [41, 50]), avoid_sticky_duration=5,
    )
    assert label == "avoid_C41_C50_sticky_st5"
    assert len(rules) == 2
    assert {r.avoid_cluster for r in rules} == {41, 50}


def test_avoid_Gs_multi_group() -> None:
    _, rules = build_rules_for_condition(("avoid_Gs", [1, 4]))
    assert len(rules) == 2
    assert {r.avoid_group for r in rules} == {1, 4}


def test_mix_combined_steer_avoid() -> None:
    label, rules = build_rules_for_condition(
        ("mix", [10, 25], [41]),
        attractor_duration=30, avoid_sticky_duration=4,
    )
    # 2 attractors + 1 repulsor = 3 rules
    assert len(rules) == 3
    attractors = [r for r in rules if r.target_cluster is not None]
    repulsors = [r for r in rules if r.avoid_cluster is not None]
    assert {r.target_cluster for r in attractors} == {10, 25}
    assert {r.avoid_cluster for r in repulsors} == {41}
    assert label.startswith("mix_steer_C10_C25_avoid_C41")


def test_mix_empty_steer_only_avoid() -> None:
    _, rules = build_rules_for_condition(("mix", [], [41]))
    assert len(rules) == 1
    assert rules[0].avoid_cluster == 41


def test_unknown_condition_kind_raises() -> None:
    with pytest.raises(ValueError, match="Unknown condition kind"):
        build_rules_for_condition(("not_a_real_kind",))


# ---------------------------------------------------------------------------
# attach_steering_hook — public-API importability
# ---------------------------------------------------------------------------

def test_attach_steering_hook_is_callable() -> None:
    """Public re-export resolves to a real callable. Full invocation needs an
    LLM context which is too heavy for unit tests."""
    assert callable(attach_steering_hook)


# ---------------------------------------------------------------------------
# pick_best_closure_predictor
# ---------------------------------------------------------------------------

def _write_fake_predictors_pt(
    path: Path,
    *,
    hidden_dim: int,
    input_dim: int = 8,
    output_dim: int = 8,
    val_losses: list[float],
    target_variance: float = 1.0,
    loss_type: str = "mse",
    variant: str = "autonomous",
) -> None:
    """Write a predictors.pt that mimics what save_closure_predictors produces."""
    path.parent.mkdir(parents=True, exist_ok=True)
    model = build_closure_predictor(input_dim, output_dim, hidden_dim, num_layers=2)
    save_closure_predictors(
        models={variant: {
            "model": model,
            "input_dim": input_dim,
            "output_dim": output_dim,
            "loss_type": loss_type,
            "hidden_dim": hidden_dim,
            "num_layers": 2,
            "train_losses": list(val_losses),
            "val_losses": list(val_losses),
            "val_r2_final": 0.5,
        }},
        save_path=path,
        target_variance=target_variance,
    )


def test_pick_best_closure_predictor_chooses_lowest_val_loss(tmp_path: Path) -> None:
    _write_fake_predictors_pt(tmp_path / "hdim64"  / "predictors.pt", hidden_dim=64,  val_losses=[0.50, 0.30])
    _write_fake_predictors_pt(tmp_path / "hdim128" / "predictors.pt", hidden_dim=128, val_losses=[0.40, 0.10])
    _write_fake_predictors_pt(tmp_path / "hdim256" / "predictors.pt", hidden_dim=256, val_losses=[0.20, 0.20])

    predictor, info = pick_best_closure_predictor(tmp_path, device="cpu")
    assert info["hidden_dim"] == 128  # minimum val_loss = 0.10
    assert info["loss_type"] == "mse"
    # The returned module should be on CPU + eval mode
    assert next(predictor.parameters()).device.type == "cpu"
    assert not predictor.training


def test_pick_best_closure_predictor_raises_on_empty_dir(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        pick_best_closure_predictor(tmp_path, device="cpu")


def test_pick_best_closure_predictor_raises_on_loss_type_mismatch(tmp_path: Path) -> None:
    _write_fake_predictors_pt(
        tmp_path / "hdim64" / "predictors.pt",
        hidden_dim=64, val_losses=[0.1], loss_type="cosine",
    )
    with pytest.raises(RuntimeError, match="loss_type"):
        pick_best_closure_predictor(tmp_path, device="cpu", loss_type="mse")


def test_pick_best_closure_predictor_round_trip_shape(tmp_path: Path) -> None:
    _write_fake_predictors_pt(
        tmp_path / "hdim32" / "predictors.pt",
        hidden_dim=32, input_dim=16, output_dim=16, val_losses=[0.05],
    )
    predictor, info = pick_best_closure_predictor(tmp_path, device="cpu")
    assert info["input_dim"] == 16 and info["output_dim"] == 16
    # Verify the loaded predictor produces the expected output shape.
    x = torch.randn(4, 16)
    with torch.no_grad():
        y = predictor(x)
    assert y.shape == (4, 16)
