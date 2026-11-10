#!/usr/bin/env bash
set -euo pipefail

STAGE11_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
STAGE11_PYTHON="${PYTHON_BIN:-/home/mjp/miniconda3/envs/unifyimmun/bin/python}"
ACTION="${1:-audit}"
if [[ $# -gt 0 ]]; then shift; fi

case "$ACTION" in
  audit) SCRIPT=audit_plm_stage11_data.py ;;
  precompute) SCRIPT=precompute_plm_stage11_embeddings.py ;;
  train) SCRIPT=train_plm_stage11.py ;;
  evaluate) SCRIPT=evaluate_plm_stage11.py ;;
  *) echo "Usage: $0 {audit|precompute|train|evaluate} [options]" >&2; exit 2 ;;
esac
exec "$STAGE11_PYTHON" "$STAGE11_ROOT/source/$SCRIPT" --config "$STAGE11_ROOT/configs/plm_stage11.yaml" "$@"
