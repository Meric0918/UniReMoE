#!/usr/bin/env bash
set -uo pipefail
cd /home/mjp/Project/PLM/UnifyImmun
export PYTHON_BIN=/home/mjp/miniconda3/envs/unifyimmun/bin/python
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false
bash /home/mjp/Project/PLM/UnifyImmun/run_stage_MOE.sh train-sequential --config /home/mjp/Project/PLM/UnifyImmun/configs/plm_stage_MOE.yaml --device cuda:0 --num-workers 2 --swanlab-mode online --output-dir /home/mjp/Project/PLM/UnifyImmun/trained_model/PLM_Stage_MOE/seed_42/sequential_online_20261027_193829 > /home/mjp/Project/PLM/UnifyImmun/training_logs/stage_MOE_online_20261027_193829/sequential.log 2>&1
status=$?
printf '%s\n' "$status" > /home/mjp/Project/PLM/UnifyImmun/training_logs/stage_MOE_online_20261027_193829/sequential.exit_code
exit "$status"
