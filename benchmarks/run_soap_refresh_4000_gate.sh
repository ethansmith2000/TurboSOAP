#!/usr/bin/env bash
set -euo pipefail

repo=/workspace/TurboSOAP
python_bin=/venv/main/bin/python
control_config="$repo/configs/llm_openwebtext_modern_768x8_soap_fixed10_current_lr000125_batch32_seq1024_4000.json"
candidate_config="$repo/configs/llm_openwebtext_modern_768x8_soap_fixed10to20_warmup200_resetage400_lr000125_batch32_seq1024_4000.json"
control_output=/workspace/optimizer_replay_results/llm_openwebtext_modern_768x8_soap_fixed10_current_lr000125_batch32_seq1024_4000.json
candidate_output=/workspace/optimizer_replay_results/llm_openwebtext_modern_768x8_soap_fixed10to20_warmup200_resetage400_lr000125_batch32_seq1024_4000.json
control_checkpoint=/workspace/optimizer_checkpoints/llm_openwebtext_modern_768x8_soap_fixed10_current_lr000125_batch32_seq1024_4000.pt
candidate_checkpoint=/workspace/optimizer_checkpoints/llm_openwebtext_modern_768x8_soap_fixed10to20_warmup200_resetage400_lr000125_batch32_seq1024_4000.pt

is_complete() {
  "$python_bin" - "$1" <<'PY'
import json
import sys
from pathlib import Path
try:
    result = json.loads(Path(sys.argv[1]).read_text())
    progress = result.get("progress", {})
    complete = (
        progress.get("completed") is True
        and progress.get("completed_steps") == 4000
        and progress.get("target_steps") == 4000
    )
except (OSError, ValueError, AttributeError):
    complete = False
sys.exit(0 if complete else 1)
PY
}

run_arm() {
  local config=$1
  local output=$2
  local checkpoint=$3
  if is_complete "$output"; then
    echo "[soap-refresh-gate] already complete: $output"
    return
  fi
  local resume=()
  if [[ -f "$checkpoint" ]]; then
    resume=(--resume "$checkpoint")
  fi
  "$python_bin" "$repo/train_llm.py" --config "$config" "${resume[@]}"
}

if [[ "${1:-}" == "--inside-claim" ]]; then
  cd "$repo"
  run_arm "$control_config" "$control_output" "$control_checkpoint"
  run_arm "$candidate_config" "$candidate_output" "$candidate_checkpoint"
  exit 0
fi

if is_complete "$control_output" && is_complete "$candidate_output"; then
  echo "[soap-refresh-gate] both arms are already complete"
  exit 0
fi

exec /workspace/bin/gpu-claim run \
  --owner optimizer-research \
  --job soap-refresh-4000-seed123 \
  --gpu any \
  -- "$0" --inside-claim
