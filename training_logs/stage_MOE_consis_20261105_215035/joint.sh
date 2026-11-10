#!/usr/bin/env bash
set -uo pipefail
cd /home/mjp/Project/PLM/UnifyImmun
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=0
/home/mjp/Project/PLM/UnifyImmun/run_stage_MOE_consis.sh train --config /home/mjp/Project/PLM/UnifyImmun/training_logs/stage_MOE_consis_20261105_215035/config.yaml --output-dir /home/mjp/Project/PLM/UnifyImmun/trained_model/PLM_Stage_MOE_consis/seed_42/joint_20261105_215035 --device cuda --swanlab-mode online 2>&1 | tee /home/mjp/Project/PLM/UnifyImmun/training_logs/stage_MOE_consis_20261105_215035/joint.log
status=${PIPESTATUS[0]}
printf "%s\n" "$status" > /home/mjp/Project/PLM/UnifyImmun/training_logs/stage_MOE_consis_20261105_215035/joint.exit_code
exit "$status"
