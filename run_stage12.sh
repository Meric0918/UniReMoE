#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PYTHON_BIN=${PYTHON_BIN:-/home/mjp/miniconda3/envs/unifyimmun/bin/python}
CONFIG=${STAGE12_CONFIG:-"$PROJECT_ROOT/configs/plm_stage12.yaml"}
if [[ -z ${CUDA_HOME:-} && -d /home/mjp/miniconda3/envs/unifyimmun/cuda ]]; then
  export CUDA_HOME=/home/mjp/miniconda3/envs/unifyimmun/cuda
fi
COMMAND=${1:-}
if [[ -n "$COMMAND" ]]; then
  shift
fi

case "$COMMAND" in
  audit)
    exec "$PYTHON_BIN" "$PROJECT_ROOT/source/train_plm_stage12.py" \
      --config "$CONFIG" --audit-only "$@"
    ;;
  train)
    exec "$PYTHON_BIN" "$PROJECT_ROOT/source/train_plm_stage12.py" \
      --config "$CONFIG" "$@"
    ;;
  resume)
    if [[ $# -lt 1 ]]; then
      echo "Usage: $0 resume STAGE12_LAST_CHECKPOINT [options]" >&2
      exit 2
    fi
    CHECKPOINT=$1
    shift
    exec "$PYTHON_BIN" "$PROJECT_ROOT/source/train_plm_stage12.py" \
      --config "$CONFIG" --resume "$CHECKPOINT" "$@"
    ;;
  evaluate)
    exec "$PYTHON_BIN" "$PROJECT_ROOT/source/evaluate_plm_stage12_benchmarks.py" \
      --config "$CONFIG" "$@"
    ;;
  *)
    echo "Usage: $0 {audit|train|resume|evaluate} [options]" >&2
    exit 2
    ;;
esac
