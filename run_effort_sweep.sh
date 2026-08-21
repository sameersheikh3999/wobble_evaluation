#!/usr/bin/env bash
# Does thinking depth change the wobble?
#
# Opus 5 exposes no temperature/top_p/top_k - they were removed from the model,
# not merely hidden by Claude Code. `effort` is the only sampling-adjacent dial
# it has, so this is the closest available substitute for a temperature sweep.
#
# The two EXTREMES are run here (low, max). `high` already exists for these same
# three lessons in multi_binary_clean, giving three points on the curve. If the
# flip rate is flat from low to max, the residual wobble is not a compute
# problem - it is ambiguity in the rubric, and no setting will remove it.
set -eu
PY=".venv/Scripts/python.exe"
export PATH="$HOME/.local/bin:$PATH"
for EFF in low max; do
  echo "=== effort=$EFF ==="
  $PY run_multi_binary.py --dir _effort_subset -n 10 --exclude unreliable \
      --effort "$EFF" --out "effort_${EFF}_out" --resume
done
echo "=== EFFORT SWEEP COMPLETE ==="
