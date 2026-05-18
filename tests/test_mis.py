"""Unit tests for the Mental-state Interpretability Score (MIS) pipeline.

No API calls, no GPU — exercises the prompt-construction surface that
``run_mis.py`` would feed to Claude judges.
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np
import pytest

from evaluation.mis.llm import (
    _CODES_SYS,
    _JUDGE_SYS,
    _SAE_CODES_SYS,
    _TEXT_SYS,
    build_codes_prompt,
    build_judge_prompt,
    build_text_prompt,
)
from evaluation.mis.naming import (
    RETNaming,
    format_ret_naming_table,
    parse_ret_naming,
)
from evaluation.mis.sequence import format_ret_sequence


# ---------------------------------------------------------------------------
# parse_ret_naming
# ---------------------------------------------------------------------------

def _write_naming(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "naming.txt"
    p.write_text(textwrap.dedent(body).strip() + "\n")
    return p


def test_parse_ret_naming_minimal(tmp_path: Path) -> None:
    fp = _write_naming(tmp_path, """
        G0: scaffolding | low-level routing tokens
        G1: planning | strategy / setup
        C0 [G0]: greeting | hello-style tokens
        C1 [G0]: punctuation | comma/period/etc.
        C2 [G1]: outline | plan-step markers
    """)
    naming = parse_ret_naming(fp)
    assert naming.num_groups == 2
    assert naming.K == 3
    assert naming.groups[0]["name"] == "scaffolding"
    assert naming.groups[1]["desc"].startswith("strategy")
    assert naming.clusters[0]["name"] == "greeting"
    assert naming.clusters[0]["group_id"] == 0
    assert naming.clusters[2]["group_id"] == 1


def test_parse_ret_naming_ungrouped_fallback(tmp_path: Path) -> None:
    fp = _write_naming(tmp_path, """
        G0: scaffolding | hi
        C0: stray | not yet assigned to a group
    """)
    naming = parse_ret_naming(fp)
    # Ungrouped clusters should still parse without crashing
    assert naming.K == 1
    assert naming.clusters[0]["name"] == "stray"


def test_parse_ret_naming_ignores_noise(tmp_path: Path) -> None:
    fp = _write_naming(tmp_path, """
        # arbitrary comment
        Some preamble paragraph that mentions stuff.

        G0: scaffolding | x
        C0 [G0]: greeting | y
    """)
    naming = parse_ret_naming(fp)
    assert naming.num_groups == 1 and naming.K == 1


# ---------------------------------------------------------------------------
# format_ret_naming_table
# ---------------------------------------------------------------------------

def test_format_ret_naming_table_contains_all_entries(tmp_path: Path) -> None:
    fp = _write_naming(tmp_path, """
        G0: scaffolding | low-level routing
        G1: planning | strategy
        C0 [G0]: greeting | hi/hello
        C1 [G1]: outline | plan-step markers
    """)
    naming = parse_ret_naming(fp)
    out = format_ret_naming_table(naming)
    # Every cluster id and every group id appears in the table.
    for token in ("G0", "G1", "C0", "C1", "scaffolding", "planning", "greeting", "outline"):
        assert token in out, f"missing {token!r}"


# ---------------------------------------------------------------------------
# format_ret_sequence
# ---------------------------------------------------------------------------

def test_format_ret_sequence_shape() -> None:
    tokens = ["The", " sum", " is"]
    labels = np.array([5, 5, 12])
    groups = np.array([0, 0, 1])
    out = format_ret_sequence(tokens, labels, groups)
    # Format: "G<g>.C<c>" comma-separated.
    parts = [p.strip() for p in out.split(",")]
    assert parts == ["G0.C5", "G0.C5", "G1.C12"]


def test_format_ret_sequence_length_match() -> None:
    tokens = ["a", "b", "c", "d"]
    labels = np.array([1, 2, 3, 4])
    groups = np.array([0, 0, 1, 1])
    out = format_ret_sequence(tokens, labels, groups)
    assert out.count(",") == len(tokens) - 1


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def test_build_codes_prompt_ret_contains_inputs() -> None:
    sys_msg, user_msg = build_codes_prompt(
        "G0 scaffolding: C0 greeting\nG1 planning: C1 outline",
        "G0.C0, G1.C1",
        word_limit=80, is_sae=False,
    )
    assert sys_msg == _CODES_SYS, "system prompt drifted from module constant"
    assert "G0 scaffolding" in user_msg
    assert "G0.C0, G1.C1" in user_msg
    assert "80" in user_msg, "word_limit not surfaced"


def test_build_codes_prompt_sae_uses_sae_system() -> None:
    sys_msg, _ = build_codes_prompt(
        "L0: token_a\nL1: token_b",
        "Token 0: L0,L1",
        word_limit=80, is_sae=True,
    )
    assert sys_msg == _SAE_CODES_SYS
    assert sys_msg != _CODES_SYS


def test_build_text_prompt_contains_text() -> None:
    text = "The answer is 42 because of foo."
    sys_msg, user_msg = build_text_prompt(text, word_limit=50)
    assert sys_msg == _TEXT_SYS
    assert text in user_msg
    assert "50" in user_msg


def test_build_judge_prompt_contains_both_descriptions() -> None:
    desc_a = "DESC-FROM-CODES-XYZ"
    desc_b = "DESC-FROM-TEXT-ABC"
    sys_msg, user_msg = build_judge_prompt(desc_a, desc_b)
    assert sys_msg == _JUDGE_SYS
    assert desc_a in user_msg
    assert desc_b in user_msg


def test_system_prompts_are_distinct() -> None:
    """Sanity: the four system prompts should not have collapsed to the same string."""
    constants = {
        "codes": _CODES_SYS,
        "sae_codes": _SAE_CODES_SYS,
        "text": _TEXT_SYS,
        "judge": _JUDGE_SYS,
    }
    assert len(set(constants.values())) == 4, "two system prompts are identical"
