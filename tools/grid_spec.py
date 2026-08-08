"""Shared steering-grid specification (runner + asset builder + UI manifest).

Multi-prompt; each prompt may override the knob axes. Targets are the
paper's actual steering examples (Figs 6, 24, 27-29, 30 + the G7 sweep the
user asked for). Cluster- and group-level (high-level) targets supported.
"""
from __future__ import annotations

# default knob axes
STRENGTHS = [0.1, 0.3, 0.5, 0.7, 1.0]
STARTS = [0, 1, 2, 4, 8, 16, 32, 64]
DURATIONS = [5, 10, 20, 50]

# ---- Sum 1..11 : the two examples the paper shows (Fig 6, Fig 24) ----------
SUM_TARGETS = [
    {"key": "C35", "kind": "steer_C", "value": 35,
     "label": "→ C35 known-formula retrieval (G7) · Fig 6",
     "desc": "Pulls the model into the known-formula-retrieval phase: instead "
             "of adding 1 + 2 + … + 11 term by term it recalls the closed form "
             "n(n+1)/2 and applies it."},
    {"key": "avC32", "kind": "avoid_C", "value": 32,
     "label": "✕ away from C32 factor/exponent bookkeeping (G9) · Fig 24",
     "desc": "Repels the model away from C32’s factor/exponent bookkeeping "
             "phase. The model reaches the same answer via a different "
             "reasoning route (the dual of Figure 6, opposite direction)."},
]

# ---- Held-out NuminaMath #0 : paper appendix HIGH-LEVEL steering ------------
# Figs 27-29 (C18/C20/C27, each held active for the full generation) + the
# full G7 cluster sweep the user asked for + Fig-24-style ✕C32 repulsor.
HIGHLEVEL_TARGETS = [
    {"key": "C18", "kind": "steer_C", "value": 18,
     "label": "→ C18 verification closure (G1) · Fig 27",
     "desc": "Pulls the response into a verification-closure phase: the model "
             "commits a compact summary confirming the candidates satisfy the "
             "stated conditions rather than re-deriving the count.",
     # user: drop the strength-1 variant for C18 (overpowers the trace)
     "strengths": [0.1, 0.3, 0.5]},
    {"key": "C20", "kind": "steer_C", "value": 20,
     "label": "→ C20 analysis entry stub (G4) · Fig 28",
     "desc": "Holds the analysis-entry phase open: the response repeatedly "
             "reframes the setup with “We need to…” openers and never quite "
             "leaves the orientation step."},
    {"key": "C27", "kind": "steer_C", "value": 27,
     "label": "→ C27 boxed choice conclusions (G2) · Fig 29",
     "desc": "Forces the boxed-conclusion phase: the model jumps straight to "
             "a brief final answer with minimal intermediate work.",
     # user: drop the strength-1 variant for C27 (overpowers the trace)
     "strengths": [0.1, 0.3, 0.5]},
    {"key": "C38", "kind": "steer_C", "value": 38,
     "label": "→ C38 error-checking & reinterpretation (G7)",
     "desc": "Holds the model in an error-checking phase: it repeatedly "
             "catches itself, questions earlier moves, and re-interprets the "
             "problem mid-derivation."},
]

# ---- Fig 30 : an ambiguous-denomination word problem → C57 -----------------
# Exactly the paper's Fig 30: a single → C57 target (baseline = strength 0).
FIG30_TARGETS = [
    {"key": "C57", "kind": "steer_C", "value": 57,
     "label": "→ C57 wording disambiguation (G7) · Fig 30",
     "desc": "Pulls the model into a wording-disambiguation phase. The "
             "baseline skips past the ambiguous “Rs.” denomination; steering "
             "toward C57 the model reasons about what “Rs.” refers to, "
             "identifies it as Indian Rupees, and answers in that context."},
]
FIG30_PROMPT_FILE = (
    "outputs/GptOss20B/numina_math/ret_jepa_auth_l11/clustering/w1/K64/seed1/"
    "analysis/weighted_tau0.1/grouped_alpha1.0/unsupervised_steering/"
    "wording_disambiguation_C57/numina_test_5/user_prompt.txt"
)

PROMPTS = {
    "paper_fig6_sum_1_to_11": {
        "label": "Sum 1 + 2 + … + 11   ·   Fig 6 / Fig 24 (cluster-level)",
        "blurb": "Without steering the model adds 1 + 2 + … + 11 term-by-term. "
                 "→ C35 flips it into the closed-form-retrieval phase "
                 "(Figure 6); ✕ C32 reaches the same shift by pushing the "
                 "model away from the bookkeeping phase (Figure 24).",
        "source": {"problem_id": "paper_fig6_sum_1_to_11"},
        "targets": SUM_TARGETS,
        # user: drop strength 0.1 from the slider (treat as nonexistent).
        # cells exist on disk but the exact-key filter omits them.
        "strengths": [0.3, 0.5, 0.7, 1.0],
        "max_new_tokens": 256,
    },
    "figs2729_highlevel": {
        "label": "Four-digit numbers (Figs 27-29)   ·   high-level steering",
        "blurb": "Appendix high-level steering on the held-out NuminaMath "
                 "problem “How many four-digit numbers satisfy the property "
                 "that the second digit is the average of the first and third "
                 "digits?”. The chosen target cluster is held active for up to "
                 "the full generation. → C18 / C20 / C27 reproduce Figures "
                 "27-29; → C38 holds the model in an error-checking-and-"
                 "reinterpretation phase.",
        "source": {"user_prompt": "How many four-digit numbers satisfy the "
                   "property that the second digit is the average of the first "
                   "and the third digits?"},
        "targets": HIGHLEVEL_TARGETS,
        # paper used 2048 — this hard counting problem needs ≳1k tokens just to
        # finish; at 768 even the baseline truncated mid-analysis so steering's
        # behavioural effect (C27's short boxed answer, C18's verification
        # closure) was invisible.
        "max_new_tokens": 2048,
        # (target, s0.5, p0, d600) == the paper's EXACT Figs 27-29 recipe
        # (cluster_sweep: s=0.5, K=3, cosine, trigger 0, dur 600, 2048 tok);
        # those cells are overlaid token-identical from the persisted runs by
        # build_figs2729_paper.py --overlay. d100/d300 + s0.1/0.3/1.0 + the
        # group-level G7 target are freshly generated to compare regimes.
        "strengths": [0.1, 0.3, 0.5, 1.0],
        "starts": [0],
        "durations": [100, 300, 600],
        # open the playground on the paper's Figure 27 (C18, s0.5, full d600)
        "defaults": {"target": "C18", "strength": 0.5, "start": 0, "dur": 600},
        # individual (target,strength,dur) combos the user explicitly asked
        # to drop from the UI (cell .js skipped + duration slider filtered)
        "cells_omit": ["C27_s0p5_p0_d300"],
    },
    "fig30_rs_C57": {
        "label": "Sheela’s savings (“Rs.”)   ·   Fig 30 (exact paper prompt)",
        "blurb": "The exact held-out problem from paper Figure 30, fed "
                 "verbatim. Without steering the model walks past the "
                 "ambiguous “Rs.” denomination; → C57 pulls it into a "
                 "wording-disambiguation phase where it reads “Rs.” as Indian "
                 "Rupees and answers in that context.",
        "source": {"prompt_file": FIG30_PROMPT_FILE},
        "targets": FIG30_TARGETS,
        # paper used 2048 (same truncation lesson as Figs 27-29)
        "max_new_tokens": 2048,
        # paper steered C57 with duration 75; sweep short-active … long
        "durations": [20, 75, 200, 768],
    },
}

AVOID_MARGIN = 0.15
# Small fixed per-fire stickiness for repulsors ("as old usual"); the
# `duration` knob now window-limits the repulsor via avoid_until_step
# instead of overloading the sticky parameter.
AVOID_STICKY = 5
GRADIENT_STEPS = 3
TEMPERATURE = 0.7
TOP_P = 0.9
TORCH_SEED = 12345


def axes(pid: str):
    """Per-prompt knob axes (strengths, starts, durations)."""
    s = PROMPTS[pid]
    return (s.get("strengths", STRENGTHS),
            s.get("starts", STARTS),
            s.get("durations", DURATIONS))


def cell_key(target_key: str, strength: float, start: int, dur: int) -> str:
    s = f"{strength:g}".replace(".", "p")
    return f"{target_key}_s{s}_p{start}_d{dur}"


def enumerate_cells(pid: str):
    """Yield every grid cell for a prompt, plus the single baseline cell."""
    spec = PROMPTS[pid]
    strengths, starts, durations = axes(pid)
    omit = set(spec.get("cells_omit", []))
    yield {"key": "baseline", "target_key": "baseline", "kind": "baseline",
           "value": None, "strength": 0.0, "start": 0, "dur": 0}
    for t in spec["targets"]:
        for s in t.get("strengths", strengths):   # optional per-target override
            for p in starts:
                for d in durations:
                    k = cell_key(t["key"], s, p, d)
                    if k in omit:
                        continue
                    yield {
                        "key": k,
                        "target_key": t["key"],
                        "kind": t["kind"],
                        "value": t["value"],
                        "strength": s,
                        "start": p,
                        "dur": d,
                    }


def n_cells(pid: str) -> int:
    return sum(1 for _ in enumerate_cells(pid))
