#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/home/mjp/Project/PLM/UnifyImmun
PYTHON_BIN=/home/mjp/miniconda3/envs/unifyimmun/bin/python
CONFIG_PATH="$PROJECT_ROOT/configs/plm_stage6.yaml"
REQUESTED_CONDITION="${1:-all}"

CONDITIONS=(
  stage6_phla_only_matched
  stage6_joint_phla_full
  stage6_phla_only_full
)

run_condition() {
  local condition="$1"
  "$PYTHON_BIN" "$PROJECT_ROOT/source/train_plm_stage6.py" \
    --config "$CONFIG_PATH" \
    --condition "$condition" \
    --fold 4 \
    --seed 2026 \
    --device cuda \
    --swanlab-mode online
}

if [[ "$REQUESTED_CONDITION" == "all" ]]; then
  for condition in "${CONDITIONS[@]}"; do
    run_condition "$condition"
  done
else
  valid=0
  for condition in "${CONDITIONS[@]}"; do
    if [[ "$REQUESTED_CONDITION" == "$condition" ]]; then
      valid=1
      break
    fi
  done
  if [[ "$valid" -ne 1 ]]; then
    echo "Unknown Stage-6 condition: $REQUESTED_CONDITION" >&2
    exit 2
  fi
  run_condition "$REQUESTED_CONDITION"
fi
