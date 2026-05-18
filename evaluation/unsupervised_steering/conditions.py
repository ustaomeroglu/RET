"""Steering condition specifications.

A *condition* is a short spec describing what rules to apply for one steered
generation. The spec format is a tuple whose first element is a kind tag::

    ("baseline",)                       # no steering — diagnostics-only run
    ("steer_C", cluster_id)             # attract to one cluster
    ("steer_G", group_id)               # attract to one group
    ("avoid_C", cluster_id)             # repel from one cluster
    ("avoid_G", group_id)               # repel from one group
    ("steer_Cs", [c1, c2, ...])         # attract to multiple clusters (parallel)
    ("steer_Gs", [g1, g2, ...])         # attract to multiple groups (parallel)
    ("avoid_Cs", [c1, c2, ...])         # repel from multiple clusters
    ("avoid_Gs", [g1, g2, ...])         # repel from multiple groups
    ("mix", [steer_cs], [avoid_cs])     # combine steer + avoid clusters

Each is turned into ``(label: str, rules: list[SteeringRule])`` by
:func:`build_rules_for_condition`. The label is a deterministic
human-readable tag suitable as a subdirectory name.

Steering parameters (duration, margin threshold, sticky duration, trigger
position) are passed alongside as keyword arguments so the same condition
spec can be reused across sweeps with different timing.
"""
from __future__ import annotations

from evaluation.unsupervised_steering.config import SteeringRule


def build_rules_for_condition(
    condition: tuple,
    *,
    attractor_duration: int = 50,
    avoid_margin_threshold: float = 0.15,
    avoid_sticky_duration: int = 5,
    trigger_position: int = 0,
    avoid_from_step: int = 0,
) -> tuple[str, list[SteeringRule]]:
    """Translate a condition spec into ``(label, [SteeringRule])``.

    ``label`` is a deterministic snake_case tag suitable as a subdirectory
    name (e.g. ``"steer_C29_dur50"`` or ``"avoid_C41_sticky_st5"``).
    """
    kind = condition[0]

    if kind == "baseline":
        return ("baseline", [])

    if kind == "steer_C":
        c = condition[1]
        return (
            f"steer_C{c}_dur{attractor_duration}",
            [SteeringRule(
                trigger_position=trigger_position,
                target_cluster=c,
                duration=attractor_duration,
            )],
        )

    if kind == "steer_G":
        g = condition[1]
        return (
            f"steer_G{g}_dur{attractor_duration}",
            [SteeringRule(
                trigger_position=trigger_position,
                target_group=g,
                duration=attractor_duration,
            )],
        )

    if kind == "avoid_C":
        c = condition[1]
        return (
            f"avoid_C{c}_sticky_st{avoid_sticky_duration}",
            [SteeringRule(
                avoid_cluster=c,
                avoid_margin_threshold=avoid_margin_threshold,
                avoid_sticky_duration=avoid_sticky_duration,
                avoid_from_step=avoid_from_step,
            )],
        )

    if kind == "avoid_G":
        g = condition[1]
        return (
            f"avoid_G{g}_sticky_st{avoid_sticky_duration}",
            [SteeringRule(
                avoid_group=g,
                avoid_margin_threshold=avoid_margin_threshold,
                avoid_sticky_duration=avoid_sticky_duration,
                avoid_from_step=avoid_from_step,
            )],
        )

    if kind == "steer_Cs":
        cs = list(condition[1])
        cstr = "_".join(f"C{c}" for c in cs)
        return (
            f"steer_{cstr}_dur{attractor_duration}",
            [SteeringRule(
                trigger_position=trigger_position,
                target_cluster=c,
                duration=attractor_duration,
            ) for c in cs],
        )

    if kind == "steer_Gs":
        gs = list(condition[1])
        gstr = "_".join(f"G{g}" for g in gs)
        return (
            f"steer_{gstr}_dur{attractor_duration}",
            [SteeringRule(
                trigger_position=trigger_position,
                target_group=g,
                duration=attractor_duration,
            ) for g in gs],
        )

    if kind == "avoid_Cs":
        cs = list(condition[1])
        cstr = "_".join(f"C{c}" for c in cs)
        return (
            f"avoid_{cstr}_sticky_st{avoid_sticky_duration}",
            [SteeringRule(
                avoid_cluster=c,
                avoid_margin_threshold=avoid_margin_threshold,
                avoid_sticky_duration=avoid_sticky_duration,
                avoid_from_step=avoid_from_step,
            ) for c in cs],
        )

    if kind == "avoid_Gs":
        gs = list(condition[1])
        gstr = "_".join(f"G{g}" for g in gs)
        return (
            f"avoid_{gstr}_sticky_st{avoid_sticky_duration}",
            [SteeringRule(
                avoid_group=g,
                avoid_margin_threshold=avoid_margin_threshold,
                avoid_sticky_duration=avoid_sticky_duration,
                avoid_from_step=avoid_from_step,
            ) for g in gs],
        )

    if kind == "mix":
        steer_cs = list(condition[1])
        avoid_cs = list(condition[2])
        sstr = "_".join(f"C{c}" for c in steer_cs) if steer_cs else "X"
        astr = "_".join(f"C{c}" for c in avoid_cs) if avoid_cs else "X"
        rules = [
            SteeringRule(
                trigger_position=trigger_position,
                target_cluster=c,
                duration=attractor_duration,
            )
            for c in steer_cs
        ] + [
            SteeringRule(
                avoid_cluster=c,
                avoid_margin_threshold=avoid_margin_threshold,
                avoid_sticky_duration=avoid_sticky_duration,
                avoid_from_step=avoid_from_step,
            )
            for c in avoid_cs
        ]
        return (
            f"mix_steer_{sstr}_avoid_{astr}_dur{attractor_duration}_st{avoid_sticky_duration}",
            rules,
        )

    raise ValueError(f"Unknown condition kind: {kind!r} (in {condition!r})")
