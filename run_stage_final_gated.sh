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
CONFIG=${STAGE_FINAL_GATED_CONFIG:-"$PROJECT_ROOT/configs/plm_stage_final_gated.yaml"}
ACTION=${1:-audit}
if [[ $# -gt 0 ]]; then shift; fi
case "$ACTION" in
  audit) SCRIPT=audit_plm_stage_final_gated_data.py ;;
  cache) SCRIPT=precompute_plm_stage_final_gated_embeddings.py ;;
  train) SCRIPT=train_plm_stage_final_gated.py ;;
  train-sequential) SCRIPT=train_plm_stage_final_gated_sequential.py ;;
  resume) SCRIPT=train_plm_stage_final_gated.py ;;
  resume-sequential) SCRIPT=train_plm_stage_final_gated_sequential.py ;;
  evaluate) SCRIPT=evaluate_plm_stage_final_gated.py ;;
  benchmark) SCRIPT=evaluate_plm_stage_final_gated_benchmarks.py ;;
  *) echo "Usage: $0 {audit|cache|train|train-sequential|resume|resume-sequential|evaluate|benchmark} [options]" >&2; exit 2 ;;
esac
if [[ "$ACTION" == resume || "$ACTION" == resume-sequential ]]; then
  HAS_RESUME=false
  for ARG in "$@"; do
    if [[ "$ARG" == --resume || "$ARG" == --resume=* ]]; then HAS_RESUME=true; fi
  done
  if [[ "$HAS_RESUME" != true ]]; then echo 'resume requires --resume PATH' >&2; exit 2; fi
fi
exec "$PYTHON_BIN" "$PROJECT_ROOT/source/$SCRIPT" --config "$CONFIG" "$@"
