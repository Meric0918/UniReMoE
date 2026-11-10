#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${STAGE15_CONFIG:-${ROOT_DIR}/configs/plm_stage15.yaml}"
COMMAND="${1:-help}"
shift || true

case "${COMMAND}" in
  train)
    exec python "${ROOT_DIR}/source/train_plm_stage15.py" --config "${CONFIG}" "$@"
    ;;
  resume)
    exec python "${ROOT_DIR}/source/train_plm_stage15.py" --config "${CONFIG}" --resume "$@"
    ;;
  evaluate)
    exec python "${ROOT_DIR}/source/evaluate_plm_stage15.py" --config "${CONFIG}" "$@"
    ;;
  benchmark)
    exec python "${ROOT_DIR}/source/evaluate_plm_stage15_benchmarks.py" --config "${CONFIG}" "$@"
    ;;
  seeds)
    exec python "${ROOT_DIR}/source/run_plm_stage15_seeds.py" --config "${CONFIG}" "$@"
    ;;
  export)
    exec python "${ROOT_DIR}/source/export_plm_stage15_task_models.py" "$@"
    ;;
  help|*)
    echo "Usage: $0 {train|resume|evaluate|benchmark|seeds|export} [arguments...]"
    ;;
esac
