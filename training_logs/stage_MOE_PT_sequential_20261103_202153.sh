#!/usr/bin/env bash
set -uo pipefail
cd /home/mjp/Project/PLM/UnifyImmun
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1
/home/mjp/miniconda3/envs/unifyimmun/bin/python -u /home/mjp/Project/PLM/UnifyImmun/source/train_plm_stage_MOE_PT_sequential.py --config /home/mjp/Project/PLM/UnifyImmun/training_logs/stage_MOE_PT_sequential_20261103_202153.yaml --swanlab-mode online --output-dir /home/mjp/Project/PLM/UnifyImmun/trained_model/PLM_Stage_MOE_PT/sequential_seed42_20261103_202153 > /home/mjp/Project/PLM/UnifyImmun/training_logs/stage_MOE_PT_sequential_20261103_202153.log 2>&1
run_status=$?
printf '%s\n' "$run_status" > /home/mjp/Project/PLM/UnifyImmun/training_logs/stage_MOE_PT_sequential_20261103_202153.exit
exit "$run_status"
