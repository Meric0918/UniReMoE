#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
if [[ -z ${PYTHON_BIN:-} ]]; then
  if [[ -x /home/mjp/miniconda3/envs/unifyimmun/bin/python ]]; then
    PYTHON_BIN=/home/mjp/miniconda3/envs/unifyimmun/bin/python
  else
    PYTHON_BIN=$(command -v python3)
  fi
fi
CONFIG=${STAGE14_CONFIG:-"$PROJECT_ROOT/configs/plm_stage14.yaml"}
ACTION=${1:-audit}
if [[ $# -gt 0 ]]; then
  shift
fi

case "$ACTION" in
  audit)
    exec "$PYTHON_BIN" "$PROJECT_ROOT/source/audit_plm_stage14_data.py" --config "$CONFIG" "$@"
    ;;
  warmup)
    exec "$PYTHON_BIN" "$PROJECT_ROOT/source/train_plm_stage14.py" --config "$CONFIG" --warmup-only "$@"
    ;;
  train)
    exec "$PYTHON_BIN" "$PROJECT_ROOT/source/train_plm_stage14.py" --config "$CONFIG" "$@"
    ;;
  ablations)
    exec "$PYTHON_BIN" "$PROJECT_ROOT/source/run_plm_stage14_ablations.py" --config "$CONFIG" "$@"
    ;;
  evaluate)
    exec "$PYTHON_BIN" "$PROJECT_ROOT/source/evaluate_plm_stage14.py" --config "$CONFIG" "$@"
    ;;
  benchmarks)
    exec "$PYTHON_BIN" "$PROJECT_ROOT/source/evaluate_plm_stage14_benchmarks.py" --config "$CONFIG" "$@"
    ;;
  resume)
    if [[ $# -lt 1 ]]; then
      echo "Usage: $0 resume STAGE14_LAST_CHECKPOINT [options]" >&2
      exit 2
    fi
    CHECKPOINT=$1
    shift
    exec "$PYTHON_BIN" "$PROJECT_ROOT/source/train_plm_stage14.py" --config "$CONFIG" --resume "$CHECKPOINT" "$@"
    ;;
  *)
    echo "Usage: $0 {audit|warmup|train|ablations|resume|evaluate|benchmarks} [options]" >&2
    exit 2
    ;;
esac
