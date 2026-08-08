#!/usr/bin/env bash
# Regenerate the steering knob-grid end to end (4×H100, ~15 min) then build
# the compact web assets. Read-only w.r.t. outputs/. Resumable: existing
# staged cells are skipped, so re-running only fills gaps.
#
#   bash webpage/tools/regenerate_grid.sh [PID]      # from repo root
set -euo pipefail
cd "$(dirname "$0")/../.."
PID="${1:-paper_fig6_sum_1_to_11}"
STAGE=/scratch/mustaome/ret_grid
mkdir -p "$STAGE/logs"

# global PCA+UMAP projection (shared by every demo) — fit once if missing
if [ ! -f webpage_files/global.npz ]; then
  echo "[0] global projection missing — fitting it"
  conda run -n eet python webpage/tools/fit_global_proj.py
fi

echo "[1/2] generate grid for $PID on 4 GPUs (sharded, resumable)"
pids=()
for s in 0 1 2 3; do
  conda run -n eet python webpage/tools/steer_grid.py \
    --pid "$PID" --shard "$s" --nshards 4 --gpu "$s" --stage "$STAGE" \
    > "$STAGE/logs/shard$s.log" 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
grep -h DONE "$STAGE"/logs/shard*.log || true
nfail=$(grep -hc FAIL "$STAGE"/logs/shard*.log 2>/dev/null | paste -sd+ | bc || echo 0)
echo "  failures: ${nfail:-0}"

echo "[2/2] build compact web assets (shared 3D UMAP + per-cell JSONP)"
conda run -n eet python webpage/tools/build_grid_assets.py --pid "$PID"
echo "Done. webpage/demos/steering_grid/$PID regenerated."
