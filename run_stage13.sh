#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PYTHON_BIN=${PYTHON_BIN:-/home/mjp/miniconda3/envs/unifyimmun/bin/python}
CONFIG=${STAGE13_CONFIG:-"$PROJECT_ROOT/configs/plm_stage13.yaml"}
COMMAND=${1:-}
if [[ -n "$COMMAND" ]]; then
  shift
fi

case "$COMMAND" in
  build-graph)
    exec "$PYTHON_BIN" "$PROJECT_ROOT/source/build_plm_stage13_graph.py" --config "$CONFIG" "$@"
    ;;
  train)
    exec "$PYTHON_BIN" "$PROJECT_ROOT/source/train_plm_stage13.py" --config "$CONFIG" "$@"
    ;;
  resume)
    if [[ $# -lt 1 ]]; then
      echo "Usage: $0 resume STAGE13_LAST_CHECKPOINT [options]" >&2
      exit 2
    fi
    CHECKPOINT=$1
    shift
    exec "$PYTHON_BIN" "$PROJECT_ROOT/source/train_plm_stage13.py" --config "$CONFIG" --resume "$CHECKPOINT" "$@"
    ;;
  evaluate)
    exec "$PYTHON_BIN" "$PROJECT_ROOT/source/evaluate_plm_stage13.py" --config "$CONFIG" "$@"
    ;;
  *)
    echo "Usage: $0 {build-graph|train|resume|evaluate} [options]" >&2
    exit 2
    ;;
esac
