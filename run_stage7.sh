#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/home/mjp/Project/PLM/UnifyImmun
PYTHON_BIN=/home/mjp/miniconda3/envs/unifyimmun/bin/python
STAGE3_RUN_DIR="$PROJECT_ROOT/trained_model/PLM_Triplet/fold_4/seed_2026"
STAGE3_CHECKPOINT="$STAGE3_RUN_DIR/triplet_best_20260802_215908_285880.pt"

exec "$PYTHON_BIN" "$PROJECT_ROOT/source/evaluate_plm_stage7.py" \
  --config "$PROJECT_ROOT/configs/plm_stage7.yaml" \
  --stage3-checkpoint "$STAGE3_CHECKPOINT" \
  --stage1-phla-checkpoint "$PROJECT_ROOT/trained_model/PLM_Unified/fold_4/seed_2026/best_phla.pt" \
  --stage1-ptcr-checkpoint "$PROJECT_ROOT/trained_model/PLM_Unified/fold_4/seed_2026/best_ptcr.pt" \
  --device cuda \
  "$@"
