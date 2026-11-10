#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT=/home/mjp/Project/PLM/UnifyImmun
RUN_TIMESTAMP=${STAGE_ADAPTER_RUN_TIMESTAMP:?set STAGE_ADAPTER_RUN_TIMESTAMP}
CONFIG="$PROJECT_ROOT/configs/plm_stage_adapter_20261007_stage2b_round16.yaml"
SOURCE_CHECKPOINT="$PROJECT_ROOT/trained_model/PLM_Unified/fold_1/seed_3407/stage2b_best_joint_20260928_112359_821461.pt"
WARMUP_CHECKPOINT="$PROJECT_ROOT/trained_model/PLM_StageAdapter/fold_1/seed_3407/warmup/stage_adapter_warmup_${RUN_TIMESTAMP}.pt"
LOG="$PROJECT_ROOT/trained_model/PLM_StageAdapter/logs/stage_adapter_stage2b_round16_${RUN_TIMESTAMP}.log"

cd "$PROJECT_ROOT"
mkdir -p "$(dirname "$LOG")"
export PYTHONPATH="$PROJECT_ROOT/source"
export STAGE_ADAPTER_CONFIG="$CONFIG"

{
  echo "[LAUNCH] timestamp=$RUN_TIMESTAMP"
  echo "[LAUNCH] source=$SOURCE_CHECKPOINT"
  echo "[LAUNCH] config=$CONFIG"
  ./run_stage_adapter.sh warmup \
    --stage1-checkpoint "$SOURCE_CHECKPOINT" \
    --run-timestamp "$RUN_TIMESTAMP" \
    --device cuda \
    --num-workers 2 \
    --swanlab-mode online
  ./run_stage_adapter.sh train \
    --warmup-checkpoint "$WARMUP_CHECKPOINT" \
    --run-timestamp "$RUN_TIMESTAMP" \
    --device cuda \
    --num-workers 2 \
    --swanlab-mode online
  echo "[COMPLETE] timestamp=$RUN_TIMESTAMP"
} 2>&1 | tee -a "$LOG"
