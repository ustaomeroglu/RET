# Extra steering-playground prompt ideas (for review)

The flagship grid (`paper_fig6_sum_1_to_11`) is live in the playground. All
prompts below are **short** (cheap: ≈4 s/cell on one H100, ≈10 min for a full
640-cell grid across 4 GPUs) and chosen because steering produces a *visible
strategy change*, not just reworded output. Each can be added by appending an
entry to `webpage/tools/grid_spec.py::PROMPTS` and re-running
`steer_grid.py` + `build_grid_assets.py`.

Pick the ones you want; targets use the paper's cluster vocabulary
(`cluster_naming/naming.txt`).

| # | prompt (problems.json id) | why it's a good demo | suggested targets (toward / away) |
|---|---|---|---|
| 1 | **Telescoping sum** `run_telescoping_sum` — `1/(1·2)+…+1/(10·11)` | baseline often telescopes; can be flipped to brute enumeration or pushed off the telescope trick | → C63 exhaustive enumeration · → C35 known-formula · ✕ away C6 telescoping (kills the trick → forces long arithmetic) |
| 2 | **Min of x + 4/x** `amgm_vs_calc` (x>0) | classic two-method problem: AM–GM vs calculus derivative; steering switches the method | → C50 monotonicity/derivative analysis (G8) · → C29 structural relation (algebra) · ✕ away C50 (blocks calculus → forces AM–GM) |
| 3 | **Max xy given x+y=10** `p8_max_xy_perimeter` | AM–GM vs substitution vs calculus; clean strategy fork on a one-liner | → C50 derivative · → C35 known-formula · → C63 enumeration |
| 4 | **P(exactly 2 heads in 3 flips)** `p3_coin_3flips_2heads` | enumeration of outcomes vs binomial formula — very legible flip | → C63 exhaustive tally · → C35 formula retrieval · → C28 direct arrangement counting |
| 5 | **Triangle area, verts (0,0),(6,0),(2,4)** `p5_triangle_area_026_24` | base×height vs shoelace vs coordinate setup — the paper's Fig-25 effect (C40) | → C40 diagram sanity checks (G5) · → C43 coordinate-frame setup · → C31 formula substitution |
| 6 | **Min x²+y² s.t. 3x+4y=12** `run_avoid_geometry` | Lagrange vs geometric distance vs substitution; rich method space | → C43 coordinate setup · → C50 derivative · → C30 spatial-config interpretation |
| 7 | **Why a²−b² = (a−b)(a+b)** `diff_of_squares` | proof-framing shift: expand-and-verify vs factor-narrative vs worked-example exposition | → C47 worked-example explanation (G2) · → C26 implication-chain algebra · → C61 expansion arithmetic |
| 8 | **Sum 1 to 61** (custom, like Fig-26) | honest *limit* example: paper shows steering **fails** here (model uses closed form regardless). Good for the "honest scope" point on the page. | → C63 enumeration (shows it cannot force term-by-term) · ✕ away C6 |

Notes / recommendations:
- **Strongest visual contrasts:** #2 (AM–GM vs calculus) and #4 (enumeration vs
  binomial) — both flip a recognizable human method on a one-line prompt.
- **#8 is worth including** even though steering "fails": it visibly demonstrates
  the paper's limitation claim ("steering shifts among paths already plausible
  under the prompt") and pairs well with the existing honest-scope callout.
- The current 3 fixed steering demos (train word-problem → C30, means → C57) can
  be **retired** once 2–3 playground prompts exist, since the playground
  subsumes them with far more interactivity.
- Each added prompt ≈ 5 strengths × 4 targets × 8 starts × 4 durations + 1
  baseline = 641 cells ≈ 10 min on 4×H100, ≈ 5 MB of `.js` assets.
