#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/home/mjp/Project/PLM/UnifyImmun
PYTHON_BIN=/home/mjp/miniconda3/envs/unifyimmun/bin/python
CONFIG="$PROJECT_ROOT/configs/plm_stage9.yaml"

ACTION=${1:-audit}
if [[ $# -gt 0 ]]; then
  shift
fi

case "$ACTION" in
  audit)
    exec "$PYTHON_BIN" "$PROJECT_ROOT/source/audit_plm_stage9_data.py" --config "$CONFIG" "$@"
    ;;
  train)
    exec "$PYTHON_BIN" "$PROJECT_ROOT/source/train_plm_stage9.py" --config "$CONFIG" "$@"
    ;;
  evaluate)
    exec "$PYTHON_BIN" "$PROJECT_ROOT/source/evaluate_plm_stage9.py" --config "$CONFIG" "$@"
    ;;
  export)
    exec "$PYTHON_BIN" "$PROJECT_ROOT/source/export_plm_stage9_embeddings.py" --config "$CONFIG" "$@"
    ;;
  *)
    echo "Usage: $0 {audit|train|evaluate|export} [options]" >&2
    exit 2
    ;;
esac
