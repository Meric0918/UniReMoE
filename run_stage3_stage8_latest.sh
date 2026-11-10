#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="/home/mjp/Project/PLM/UnifyImmun"
PYTHON_BIN="/home/mjp/miniconda3/envs/unifyimmun/bin/python"
STAGE1_CHECKPOINT="$PROJECT_ROOT/trained_model/PLM_Unified_PH_PT/fold_1/seed_42/best_round/best_joint_20260831_174042_678934.pt"

cd "$PROJECT_ROOT"
mkdir -p "$PROJECT_ROOT/training_logs"

STAGE8_TIMESTAMP="${STAGE8_TIMESTAMP:-$(date +%Y%m%d_%H%M%S_%6N)}"
STAGE8_WARMUP="$PROJECT_ROOT/trained_model/PLM_Stage8/fold_1/seed_42/warmup/stage8a_best_pht_${STAGE8_TIMESTAMP}.pt"
STAGE8_WARMUP_LOG="$PROJECT_ROOT/training_logs/stage8_warmup_from_stage1_${STAGE8_TIMESTAMP}.log"
STAGE8_FULL_LOG="$PROJECT_ROOT/training_logs/stage8_full_from_stage1_${STAGE8_TIMESTAMP}.log"

STAGE3_TIMESTAMP="${STAGE3_TIMESTAMP:-$(date +%Y%m%d_%H%M%S_%6N)}"
STAGE3_LOG="$PROJECT_ROOT/training_logs/stage3_from_stage1_${STAGE3_TIMESTAMP}.log"

run_stage3() {
  echo "[pipeline] Stage 3 start: $(date --iso-8601=seconds)"
  echo "[pipeline] Stage 3 source: $STAGE1_CHECKPOINT"
  "$PYTHON_BIN" source/train_plm_triplet.py \
    --config configs/plm_triplet.yaml \
    --stage1-checkpoint "$STAGE1_CHECKPOINT" \
    --fold 1 \
    --seed 42 \
    --device cuda:0 \
    --num-workers 2 \
    --swanlab-mode online \
    --run-timestamp "$STAGE3_TIMESTAMP" \
    2>&1 | tee "$STAGE3_LOG"
}

run_stage8() {
  : > "$PROJECT_ROOT/training_logs/.stage8_parallel_running"
  trap 'rm -f "$PROJECT_ROOT/training_logs/.stage8_parallel_running"' EXIT
  echo "[pipeline] Stage 8A warmup start: $(date --iso-8601=seconds)"
  echo "[pipeline] Stage 8 source: $STAGE1_CHECKPOINT"
  "$PYTHON_BIN" source/train_plm_stage8.py \
    --config configs/plm_stage8.yaml \
    --run-timestamp "$STAGE8_TIMESTAMP" \
    --warmup-only \
    --stage1-checkpoint "$STAGE1_CHECKPOINT" \
    --device cuda:0 \
    --num-workers 2 \
    --swanlab-mode online \
    2>&1 | tee "$STAGE8_WARMUP_LOG"

  test -f "$STAGE8_WARMUP"

  echo "[pipeline] Stage 8B full start: $(date --iso-8601=seconds)"
  "$PYTHON_BIN" source/train_plm_stage8.py \
    --config configs/plm_stage8.yaml \
    --run-timestamp "$STAGE8_TIMESTAMP" \
    --condition full \
    --warmup-checkpoint "$STAGE8_WARMUP" \
    --device cuda:0 \
    --num-workers 2 \
    --swanlab-mode online \
    2>&1 | tee "$STAGE8_FULL_LOG"
}

# Used when one branch is already running and only the other branch is needed.
if [[ "${STAGE8_ONLY:-0}" == "1" ]]; then
  run_stage8
  exit 0
fi

if [[ "${STAGE3_ONLY:-0}" == "1" ]]; then
  run_stage3
  exit 0
fi

echo "[pipeline] Stage 3 and Stage 8 start concurrently: $(date --iso-8601=seconds)"
run_stage3 &
stage3_pid=$!
run_stage8 &
stage8_pid=$!

pipeline_status=0
if ! wait "$stage3_pid"; then
  echo "[pipeline] Stage 3 failed" >&2
  pipeline_status=1
fi
if ! wait "$stage8_pid"; then
  echo "[pipeline] Stage 8 failed" >&2
  pipeline_status=1
fi

if (( pipeline_status != 0 )); then
  exit "$pipeline_status"
fi
echo "[pipeline] complete: $(date --iso-8601=seconds)"
