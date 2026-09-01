#!/usr/bin/env bash
# FICO / HOTS / TEACH on the SAME three observations, through the SAME pipeline.
# Three is the MINIMUM for the discrimination test - an indicator has to differ
# sharply across observations to register, so expect more UNINFORMATIVE verdicts
# than a ten-observation run would give. That is the sample size, not the
# frameworks, and the comparison holds only for RELATIVE differences between them.
#
# Running FICO again rather than reusing the existing 10-transcript data is
# deliberate: the older run used a different prompt builder, and a prompt
# difference between arms would be indistinguishable from a framework
# difference. Same transcripts, same scorer, same analysis - so what is left is
# the framework.
set -eu
PY=".venv/Scripts/python.exe"
export PATH="$HOME/.local/bin:$PATH"
export PYTHONIOENCODING=utf-8
for FW in taleemabad hots teach; do
  echo "=== $FW ==="
  $PY run_framework_wobble.py run --framework "frameworks/${FW}.yaml" \
      --dir _cmp3 -n 10 --out "cmp_${FW}" --resume
done
echo "=== ALL THREE DONE ==="
