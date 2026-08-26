#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/home/mjp/Project/PLM/UnifyImmun
PYTHON_BIN=/home/mjp/miniconda3/envs/unifyimmun/bin/python
CONFIG="$PROJECT_ROOT/configs/plm_stage8.yaml"

ACTION=${1:-audit}
if [[ $# -gt 0 ]]; then
  shift
fi

case "$ACTION" in
  audit)
    exec "$PYTHON_BIN" "$PROJECT_ROOT/source/audit_plm_stage8_data.py" --config "$CONFIG" "$@"
    ;;
  warmup)
    exec "$PYTHON_BIN" "$PROJECT_ROOT/source/train_plm_stage8.py" --config "$CONFIG" --warmup-only "$@"
    ;;
  train)
    exec "$PYTHON_BIN" "$PROJECT_ROOT/source/train_plm_stage8.py" --config "$CONFIG" "$@"
    ;;
  ablations)
    exec "$PYTHON_BIN" "$PROJECT_ROOT/source/run_plm_stage8_ablations.py" --config "$CONFIG" "$@"
    ;;
  *)
    echo "Usage: $0 {audit|warmup|train|ablations} [options]" >&2
    exit 2
    ;;
esac
