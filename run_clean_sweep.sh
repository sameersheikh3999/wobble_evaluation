#!/usr/bin/env bash
# Fresh scoring of both scales with the seven unreliable indicators removed from
# the prompt entirely. Sequential on purpose: running both at once would put 8
# concurrent CLI calls on the subscription and risk throttling.
set -eu   # -e: a failed phase halts the chain, so step 3 can never run a
          # comparison over partially-scored data
PY=".venv/Scripts/python.exe"
export PATH="$HOME/.local/bin:$PATH"

echo "=== [1/3] BINARY, 30 indicators ==="
$PY run_multi_binary.py --dir Transcripts -n 10 --exclude unreliable \
    --out multi_binary_clean --resume
echo "=== [2/3] 1-4 SCALE, 30 indicators ==="
$PY run_multi.py --dir Transcripts --iterations 10 --exclude unreliable \
    --out multi_scale14_clean --no-per-session-charts
echo "=== [3/3] COMPARISON ==="
$PY compare_scales.py --binary multi_binary_clean --scale14 multi_scale14_clean \
    --exclude unreliable --out scale_comparison_fresh
echo "=== SWEEP COMPLETE ==="
