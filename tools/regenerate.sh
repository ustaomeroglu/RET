#!/usr/bin/env bash
# Rebuild the webpage demo traces from scratch (idempotent).
# Read-only w.r.t. outputs/: copies pristine demo HTMLs from outputs/, then
# computes 3D z_t coords and post-processes them in webpage/ only.
#
#   bash webpage/tools/regenerate.sh        # from repo root
set -euo pipefail
cd "$(dirname "$0")/../.."                  # repo root

GA="outputs/GptOss20B/numina_math/ret_jepa_auth_l11/clustering/w1/K64/seed1/analysis/weighted_tau0.1/grouped_alpha1.0"
ET="$GA/example_test_trajectories"
US="$GA/unsupervised_steering"
M="webpage/demos/mental-state"
S="webpage/demos/steering"

echo "[1/3] copy pristine demo HTMLs from outputs/ (read-only source)"
cp "$ET/interactive_doc_0.html" "$M/trace-1-function-range.html"
cp "$ET/interactive_doc_1.html" "$M/trace-2-hyperbola.html"
cp "$ET/interactive_doc_2.html" "$M/trace-3-statistics.html"
cp "$ET/interactive_doc_3.html" "$M/trace-4-word-problem.html"
cp "$US/paper_fig6_sum_1_to_11/n11_str0.5_gs3/baseline/interactive_trajectory.html"        "$S/sum_baseline.html"
cp "$US/paper_fig6_sum_1_to_11/n11_str0.5_gs3/steer_C35_dur50/interactive_trajectory.html" "$S/sum_steered_C35.html"
cp "$US/robustness_exploration_disambiguation/word_problem/baseline/interactive_trajectory.html"        "$S/train_baseline.html"
cp "$US/robustness_exploration_disambiguation/word_problem/steer_C30_dur600/interactive_trajectory.html" "$S/train_steered_C30.html"
cp "$US/wording_disambiguation_C57/numina_test_4/baseline/interactive_trajectory.html"        "$S/mean_baseline.html"
cp "$US/wording_disambiguation_C57/numina_test_4/steer_C57_dur300/interactive_trajectory.html" "$S/mean_steered_C57.html"

echo "[2/3] compute 3D z_t UMAP coords (steering: cached; mental-state: GPU encoder over cached h)"
conda run -n eet python webpage/tools/build_3d.py

echo "[3/3] post-process HTMLs (strip title/live-info, relabel, inject 3D viewer)"
python webpage/tools/enhance_demos.py

echo "Done. webpage/demos/ regenerated."
