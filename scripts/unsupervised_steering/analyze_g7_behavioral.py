#!/usr/bin/env python3
"""Analyze steering outputs for G7 behavioral markers.

For each (prompt, condition) read ``steered_output.txt`` and count keyword
families that signal G7 (exploratory search / disambiguation) behavioral
signature. Reports per-condition aggregates + a per-prompt breakdown.

Produces the Table 4 "behavioral signature appears in 30/33 prompts" numbers
from the paper (Appendix L.1).

Usage:
    python scripts/unsupervised_steering/analyze_g7_behavioral.py \\
        --input-dir outputs/.../<clustering-analysis-dir>/unsupervised_steering/<sweep_root>/ \\
        [--strength str0.5 | str0.7]
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# Each marker family: list of patterns (case-insensitive substrings or regex).
# We use word-boundary-ish matching on lowercased text.
MARKER_FAMILIES = {
    "hedging": [
        r"\bmaybe\b", r"\bperhaps\b", r"\bpossibly\b", r"\bprobably\b",
        r"\bmight\b", r"\bnot sure\b", r"\bunclear\b", r"\bi think\b",
    ],
    "verification": [
        r"\bwait\b", r"\blet's check\b", r"\blet me check\b",
        r"\bdouble[- ]?check\b", r"\bactually\b", r"\bhmm\b",
        r"\bverify\b", r"\bsanity check\b", r"\brecompute\b",
    ],
    "disambiguation": [
        r"\binterpretation\b", r"\bambiguous\b", r"\binterpret\b",
        r"\bcould mean\b", r"\bcould be\b", r"\beither\b",
        r"\bwhich one\b", r"\bclarify\b", r"\bquestion is\b",
    ],
    "backtracking": [
        r"\bwait, ?no\b", r"\bno, ?actually\b", r"\bi made an error\b",
        r"\boh wait\b", r"\bscratch that\b", r"\bredo\b",
        r"\bmistake\b", r"\bcorrection\b", r"\binstead\b",
    ],
    "alternation": [
        r"\balternatively\b", r"\balternative\b", r"\bon the other hand\b",
        r"\banother way\b", r"\bequivalently\b",
    ],
    "exploratory": [
        r"\bsuppose\b", r"\bconsider\b", r"\bif we\b", r"\blet's try\b",
        r"\blet's see\b", r"\bone approach\b", r"\bidea\b",
    ],
    "stutter_repeat": [
        # repeated "X = X = X" or "X. X. X" type patterns
        # we detect by scanning text directly later
    ],
}


def count_stutter(text: str) -> int:
    """Count stutter repetition: same short substring appearing 3+ times in
    immediate succession (separated by ' = ' or '. ' or just space)."""
    # find sequences like "tok = tok = tok = tok" (3+ repeats)
    n = 0
    n += len(re.findall(r"(\b\S{1,15}\b)( = \1){2,}", text))
    n += len(re.findall(r"(\b\S{1,15}\b)\.\s+(\1\.\s+){2,}", text))
    return n


def count_markers(text: str, patterns: list[str]) -> int:
    t = text.lower()
    return sum(len(re.findall(p, t)) for p in patterns)


def analyze_text(text: str) -> dict:
    chars = len(text)
    counts = {fam: count_markers(text, pats) for fam, pats in MARKER_FAMILIES.items()}
    counts["stutter_repeat"] = count_stutter(text)
    counts["chars"] = chars
    counts["total_markers"] = sum(v for k, v in counts.items()
                                  if k not in ("chars",))
    return counts


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--input-dir", type=Path, required=True,
                    help="Root directory containing <prompt>/<condition>/steered_output.txt.")
    ap.add_argument("--strength", choices=["str0.5", "str0.7"], default="str0.5",
                    help="Which strength sweep to read (controls the dir-name suffix).")
    ap.add_argument("--clusters", type=int, nargs="+",
                    default=[2, 30, 33, 35, 36, 38, 57],
                    help="Cluster ids in the sweep (default: G7's seven canonical clusters).")
    ap.add_argument("--duration", type=int, default=600,
                    help="Attractor duration that appears in the condition dir names.")
    args = ap.parse_args()

    root: Path = args.input_dir
    suffix = "_str0.7" if args.strength == "str0.7" else ""
    conditions = ["baseline"] + [
        f"steer_C{c}_dur{args.duration}{suffix}" for c in args.clusters
    ] + [f"steer_G7_dur{args.duration}{suffix}"]

    rows = []
    prompt_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
    print(f"Found {len(prompt_dirs)} prompt dirs in {root}")
    for pdir in prompt_dirs:
        ptag = pdir.name
        for cond in conditions:
            fp = pdir / cond / "steered_output.txt"
            if not fp.exists():
                continue
            txt = fp.read_text()
            stats = analyze_text(txt)
            rows.append({"prompt": ptag, "cond": cond, **stats})

    # Aggregate per condition
    by_cond = {}
    for r in rows:
        by_cond.setdefault(r["cond"], []).append(r)

    print()
    print("AGGREGATE: average raw marker counts per condition")
    print(f"{'cond':<20} {'n':>3} {'chars':>8} {'hedge':>6} {'verif':>6} "
          f"{'disamb':>6} {'backtr':>6} {'altern':>6} {'explor':>6} "
          f"{'stutter':>7} {'total':>6}")
    print("-" * 95)
    for cond in conditions:
        rs = by_cond.get(cond, [])
        if not rs:
            continue
        n = len(rs)
        avg = lambda k: sum(r[k] for r in rs) / n
        print(f"{cond:<20} {n:>3} {avg('chars'):>8.0f} "
              f"{avg('hedging'):>6.2f} {avg('verification'):>6.2f} "
              f"{avg('disambiguation'):>6.2f} {avg('backtracking'):>6.2f} "
              f"{avg('alternation'):>6.2f} {avg('exploratory'):>6.2f} "
              f"{avg('stutter_repeat'):>7.2f} {avg('total_markers'):>6.2f}")

    # Density: markers per 1000 chars (controls for length)
    print()
    print("DENSITY: markers per 1000 chars (length-controlled)")
    print(f"{'cond':<20} {'n':>3} {'hedge':>6} {'verif':>6} {'disamb':>6} "
          f"{'backtr':>6} {'altern':>6} {'explor':>6} {'stutter':>7} {'total':>6}")
    print("-" * 95)
    for cond in conditions:
        rs = by_cond.get(cond, [])
        if not rs:
            continue
        n = len(rs)
        densities = {}
        for k in ("hedging", "verification", "disambiguation",
                  "backtracking", "alternation", "exploratory",
                  "stutter_repeat", "total_markers"):
            densities[k] = sum(r[k] / max(r["chars"], 1) for r in rs) / n * 1000
        print(f"{cond:<20} {n:>3} "
              f"{densities['hedging']:>6.2f} {densities['verification']:>6.2f} "
              f"{densities['disambiguation']:>6.2f} {densities['backtracking']:>6.2f} "
              f"{densities['alternation']:>6.2f} {densities['exploratory']:>6.2f} "
              f"{densities['stutter_repeat']:>7.2f} {densities['total_markers']:>6.2f}")

    # Per-prompt breakdown for conditions vs baseline
    print()
    print("Per-prompt: G7 marker count (conds minus baseline)")
    print("(positive means steered shows MORE G7 behavioral markers)")
    print(f"{'prompt':<20}  " + "  ".join(f"{c.replace('steer_','').replace(f'_dur{args.duration}',''):>6}" for c in conditions[1:]))
    for ptag in sorted({r['prompt'] for r in rows}):
        b = next((r for r in rows if r['prompt'] == ptag and r['cond'] == 'baseline'), None)
        if b is None:
            continue
        b_total = b['total_markers']
        deltas = []
        for cond in conditions[1:]:
            s = next((r for r in rows if r['prompt'] == ptag and r['cond'] == cond), None)
            if s is None:
                deltas.append("  -  ")
            else:
                deltas.append(f"{s['total_markers'] - b_total:+6}")
        print(f"{ptag:<20}  " + "  ".join(deltas))

    # Save full table as JSON
    out_json = root / f"behavioral_marker_counts_{args.strength}.json"
    with open(out_json, "w") as f:
        json.dump({"rows": rows, "conditions": conditions,
                   "marker_families": list(MARKER_FAMILIES.keys()) + ["stutter_repeat"]},
                  f, indent=2)
    print(f"\nWrote {out_json}")


if __name__ == "__main__":
    main()
