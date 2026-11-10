#!/usr/bin/env bash
set -uo pipefail
cd -- /home/mjp/Project/PLM/UnifyImmun
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export CUDA_HOME=/home/mjp/miniconda3/envs/unifyimmun/cuda
/home/mjp/miniconda3/envs/unifyimmun/bin/python -u /home/mjp/Project/PLM/UnifyImmun/source/train_plm_stage1_encoder.py --config /home/mjp/Project/PLM/UnifyImmun/training_logs/stage1_encoder_20261013_191934_818965.yaml --fold 1 --seed 42 --device cuda --swanlab-mode online --run-timestamp 20261013_191934_818965 2>&1 | tee /home/mjp/Project/PLM/UnifyImmun/training_logs/stage1_encoder_20261013_191934_818965.log
train_status=${PIPESTATUS[0]}
printf "\nTRAIN_EXIT_CODE=%s\n" "$train_status" >> /home/mjp/Project/PLM/UnifyImmun/training_logs/stage1_encoder_20261013_191934_818965.log
exit "$train_status"
