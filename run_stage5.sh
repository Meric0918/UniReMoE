#!/usr/bin/env bash
set -euo pipefail

# One-click Stage-5 training for fold 4 / seed 2026.
#
# Default:
#   ./run_stage5.sh
#
# Resume the newest Stage-5 checkpoint:
#   ./run_stage5.sh --resume
#
# Optional environment overrides:
#   STAGE5_GPU=0
#   STAGE5_NUM_WORKERS=2
#   STAGE5_SWANLAB_MODE=online   # online, local, offline, or disabled
#   STAGE5_PYTHON=/path/to/python
#   STAGE5_DRY_RUN=1

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${STAGE5_PYTHON:-/home/mjp/miniconda3/envs/unifyimmun/bin/python}"
CONFIG="configs/plm_stage5.yaml"
TRAIN_SCRIPT="source/train_plm_stage5.py"
OUTPUT_DIR="/home/mjp/Project/PLM/UnifyImmun/trained_model/PLM_Stage5/fold_5/seed_2026"
LOG_DIR="training_logs"
FOLD=4
SEED=2026
DEVICE="cuda"
GPU_ID="${STAGE5_GPU:-0}"
NUM_WORKERS="${STAGE5_NUM_WORKERS:-2}"
SWANLAB_MODE="${STAGE5_SWANLAB_MODE:-online}"
DRY_RUN="${STAGE5_DRY_RUN:-0}"
CUDA_TOOLKIT="${CUDA_HOME:-/home/mjp/miniconda3/envs/unifyimmun/cuda}"
NVML_COMPAT_DIR="/home/mjp/.local/lib/nvidia-595.71.05"
LATEST_PID_FILE="${LOG_DIR}/stage5_fold5_seed2026_latest.pid"
RESUME=0

usage() {
    cat <<'EOF'
Usage:
  ./run_stage5.sh             Start a new Stage-5 run in the background
  ./run_stage5.sh --resume    Resume the newest stage5_last checkpoint
  ./run_stage5.sh --dry-run   Validate inputs and print the command only
  ./run_stage5.sh --help      Show this help

Environment overrides:
  STAGE5_GPU=0
  STAGE5_NUM_WORKERS=2
  STAGE5_SWANLAB_MODE=online
  STAGE5_PYTHON=/path/to/python
  STAGE5_DRY_RUN=1
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --resume)
            RESUME=1
            shift
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "[ERROR] Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "[ERROR] Python executable not found: ${PYTHON_BIN}" >&2
    exit 2
fi
for required_file in "${CONFIG}" "${TRAIN_SCRIPT}"; do
    if [[ ! -f "${required_file}" ]]; then
        echo "[ERROR] Required file not found: ${required_file}" >&2
        exit 2
    fi
done
if [[ ! "${NUM_WORKERS}" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] STAGE5_NUM_WORKERS must be a non-negative integer" >&2
    exit 2
fi
if [[ ! "${SWANLAB_MODE}" =~ ^(online|local|offline|disabled)$ ]]; then
    echo "[ERROR] STAGE5_SWANLAB_MODE must be online, local, offline, or disabled" >&2
    exit 2
fi
if [[ "${DRY_RUN}" != "0" && "${DRY_RUN}" != "1" ]]; then
    echo "[ERROR] STAGE5_DRY_RUN must be 0 or 1" >&2
    exit 2
fi
if [[ ! -d "${CUDA_TOOLKIT}" ]]; then
    echo "[ERROR] CUDA toolkit directory not found: ${CUDA_TOOLKIT}" >&2
    exit 2
fi

mkdir -p "${LOG_DIR}"

# Refuse to launch a second copy while the latest recorded Stage-5 process lives.
if [[ -f "${LATEST_PID_FILE}" ]]; then
    RECORDED_PID="$(<"${LATEST_PID_FILE}")"
    if [[ "${RECORDED_PID}" =~ ^[1-9][0-9]*$ ]] \
        && kill -0 "${RECORDED_PID}" 2>/dev/null; then
        PROCESS_COMMAND="$(tr '\0' ' ' < "/proc/${RECORDED_PID}/cmdline" 2>/dev/null || true)"
        if [[ "${PROCESS_COMMAND}" == *"train_plm_stage5.py"* ]]; then
            echo "[ERROR] Stage 5 is already running with PID ${RECORDED_PID}." >&2
            echo "        Monitor: tail -f ${LOG_DIR}/stage5_fold5_seed2026_latest.log" >&2
            exit 1
        fi
    fi
fi

INIT_ARGS=()
if [[ "${RESUME}" == "1" ]]; then
    if [[ ! -d "${OUTPUT_DIR}" ]]; then
        echo "[ERROR] Stage-5 output directory not found: ${OUTPUT_DIR}" >&2
        exit 2
    fi
    RESUME_CHECKPOINT="$(
        find "${OUTPUT_DIR}" -maxdepth 1 -type f -name 'stage5_last_*.pt' \
            -printf '%T@ %p\n' 2>/dev/null \
            | sort -nr \
            | sed -n '1s/^[^ ]* //p'
    )"
    if [[ -z "${RESUME_CHECKPOINT}" || ! -f "${RESUME_CHECKPOINT}" ]]; then
        echo "[ERROR] No Stage-5 resume checkpoint found in ${OUTPUT_DIR}" >&2
        exit 2
    fi
    INIT_ARGS=(--resume "${RESUME_CHECKPOINT}")
    echo "[INIT] Resuming from ${RESUME_CHECKPOINT}"
else
    echo "[INIT] Starting Stage 5 with random downstream parameters"
fi

# A package update left nvidia-smi's system NVML out of sync on this host. Use
# the existing compatibility library when its project helper verifies it.
if command -v nvidia-smi >/dev/null 2>&1 \
    && ! nvidia-smi -L >/dev/null 2>&1 \
    && [[ -x "${PROJECT_ROOT}/nvidia-smi-compatible.sh" ]] \
    && "${PROJECT_ROOT}/nvidia-smi-compatible.sh" -L >/dev/null 2>&1; then
    export LD_LIBRARY_PATH="${NVML_COMPAT_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    echo "[NVML] Using the local compatibility library; a host reboot is still recommended."
fi

RUN_TAG="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/stage5_fold5_seed2026_${RUN_TAG}.log"
PID_FILE="${LOG_DIR}/stage5_fold5_seed2026_${RUN_TAG}.pid"

TRAIN_COMMAND=(
    "${PYTHON_BIN}" -u "${TRAIN_SCRIPT}"
    --config "${CONFIG}"
    "${INIT_ARGS[@]}"
    --fold "${FOLD}"
    --seed "${SEED}"
    --device "${DEVICE}"
    --num-workers "${NUM_WORKERS}"
    --swanlab-mode "${SWANLAB_MODE}"
)

printf '[COMMAND] CUDA_VISIBLE_DEVICES=%q CUDA_HOME=%q' "${GPU_ID}" "${CUDA_TOOLKIT}"
printf ' %q' "${TRAIN_COMMAND[@]}"
printf '\n'

if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[DRY RUN] Validation passed; training was not started."
    exit 0
fi

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export CUDA_HOME="${CUDA_TOOLKIT}"
export PYTHONUNBUFFERED=1

nohup "${TRAIN_COMMAND[@]}" >"${LOG_FILE}" 2>&1 </dev/null &
TRAIN_PID=$!
printf '%s\n' "${TRAIN_PID}" >"${PID_FILE}"
printf '%s\n' "${TRAIN_PID}" >"${LATEST_PID_FILE}"
ln -sfn "$(basename "${LOG_FILE}")" "${LOG_DIR}/stage5_fold5_seed2026_latest.log"

# Detect immediate startup failures while keeping successful training detached.
sleep 2
if ! kill -0 "${TRAIN_PID}" 2>/dev/null; then
    echo "[ERROR] Stage 5 exited during startup. Recent log output:" >&2
    tail -n 60 "${LOG_FILE}" >&2 || true
    exit 1
fi

echo "[OK] Stage 5 started in the background."
echo "     PID: ${TRAIN_PID}"
echo "     Log: ${PROJECT_ROOT}/${LOG_FILE}"
echo "     SwanLab mode: ${SWANLAB_MODE}"
echo
echo "Monitor log:"
echo "  tail -f ${PROJECT_ROOT}/${LOG_DIR}/stage5_fold5_seed2026_latest.log"
echo
echo "Check process:"
echo "  ps -fp ${TRAIN_PID}"
