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
CONFIG=${STAGE_PHT_CONFIG:-"$PROJECT_ROOT/configs/plm_stage_PHT.yaml"}
ACTION=${1:-audit}
if [[ $# -gt 0 ]]; then shift; fi

case "$ACTION" in
  audit)
    exec "$PYTHON_BIN" "$PROJECT_ROOT/source/audit_plm_stage_PHT_data.py" --config "$CONFIG" "$@"
    ;;
  warmup)
    exec "$PYTHON_BIN" "$PROJECT_ROOT/source/train_plm_stage_PHT.py" --config "$CONFIG" --warmup-only "$@"
    ;;
  train)
    exec "$PYTHON_BIN" "$PROJECT_ROOT/source/train_plm_stage_PHT.py" --config "$CONFIG" "$@"
    ;;
  resume)
    if [[ ${1:-} == "--help" || ${1:-} == "-h" ]]; then
      exec "$PYTHON_BIN" "$PROJECT_ROOT/source/train_plm_stage_PHT.py" --help
    fi
    exec "$PYTHON_BIN" "$PROJECT_ROOT/source/train_plm_stage_PHT.py" --config "$CONFIG" --resume "$@"
    ;;
  evaluate)
    exec "$PYTHON_BIN" "$PROJECT_ROOT/source/evaluate_plm_stage_PHT.py" --config "$CONFIG" "$@"
    ;;
  benchmark)
    exec "$PYTHON_BIN" "$PROJECT_ROOT/source/evaluate_plm_stage_PHT_benchmarks.py" --config "$CONFIG" "$@"
    ;;
  *)
    echo "Usage: $0 {audit|warmup|train|resume|evaluate|benchmark} [options]" >&2
    exit 2
    ;;
esac
