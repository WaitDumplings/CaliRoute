#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "common_ablation.sh is meant to be sourced." >&2
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

if [[ -z "${DATA_ROOT:-}" ]]; then
  if [[ -d "$ROOT_DIR/../Routing-D" ]]; then
    DATA_ROOT="$ROOT_DIR/../Routing-D"
  else
    DATA_ROOT="$ROOT_DIR/../Route-D"
  fi
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
PROBLEM="${PROBLEM:-evrptw}"
CUSTOMERS="${CUSTOMERS:-100}"
CS="${CS:-20}"
SEED="${SEED:-3009}"
EPOCHS="${EPOCHS:-1500}"
INIT_EPOCH="${INIT_EPOCH:-100}"
NUM_ENVS="${NUM_ENVS:-24}"
N_TRAJ="${N_TRAJ:-50}"
EVAL_N_TRAJ="${EVAL_N_TRAJ:-50}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-160}"
EVAL_MAX_STEPS="${EVAL_MAX_STEPS:-160}"
PPO_STEP_CHUNK_SIZE="${PPO_STEP_CHUNK_SIZE:-16}"
NUM_MINIBATCHES="${NUM_MINIBATCHES:-4}"
EVAL_INTERVAL="${EVAL_INTERVAL:-20}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-128}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-50}"
GPU_LIST_DEFAULT="${GPU_LIST_DEFAULT:-0,1,2,3}"
MAX_GPU_MEM_MIB="${MAX_GPU_MEM_MIB:-11000}"
GPU_MEM_POLL_SECONDS="${GPU_MEM_POLL_SECONDS:-15}"
POLL_SECONDS="${POLL_SECONDS:-60}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${LOG_ROOT:-results/launch_logs/ablation}"

BASE_RUN="CALIROUTE_${PROBLEM^^}_CUS${CUSTOMERS}_CS${CS}"
DEFAULT_INIT_RUN="${BASE_RUN}_PPO_SEED${SEED}_E${EPOCHS}_N${NUM_ENVS}_R${ROLLOUT_STEPS}_2080TI"
LEGACY_INIT_CKPT="../EVRPTW-OFFLINE2ONLINE/results/checkpoints/Cus_${CUSTOMERS}_CS_${CS}/O2O_CUS${CUSTOMERS}_PPO_ROUTE_POS_SEED${SEED}_E1500_N128_CHUNK32_EVAL20/seed_${SEED}/checkpoint_epoch_$(printf '%04d' "$INIT_EPOCH").pt"
LOCAL_INIT_CKPT="results/checkpoints/Cus_${CUSTOMERS}_CS_${CS}/${DEFAULT_INIT_RUN}/seed_${SEED}/checkpoint_epoch_$(printf '%04d' "$INIT_EPOCH").pt"
if [[ -z "${INIT_CKPT:-}" ]]; then
  if [[ -s "$LEGACY_INIT_CKPT" ]]; then
    INIT_CKPT="$LEGACY_INIT_CKPT"
  else
    INIT_CKPT="$LOCAL_INIT_CKPT"
  fi
fi

COMMON_ARGS=(
  --problem "$PROBLEM"
  --customers "$CUSTOMERS"
  --charging-stations "$CS"
  --data-root "$DATA_ROOT"
  --seed "$SEED"
  --epochs "$EPOCHS"
  --num-envs "$NUM_ENVS"
  --n-traj "$N_TRAJ"
  --rollout-steps "$ROLLOUT_STEPS"
  --ppo-step-chunk-size "$PPO_STEP_CHUNK_SIZE"
  --num-minibatches "$NUM_MINIBATCHES"
  --eval-interval "$EVAL_INTERVAL"
  --eval-n-traj "$EVAL_N_TRAJ"
  --eval-max-steps "$EVAL_MAX_STEPS"
  --eval-batch-size "$EVAL_BATCH_SIZE"
  --checkpoint-interval "$CHECKPOINT_INTERVAL"
  --mixed-precision
)

IFS=',' read -r -a GPU_LIST <<<"${GPU_LIST:-$GPU_LIST_DEFAULT}"
RUN_DIR="${RUN_DIR:-$LOG_ROOT/${SCRIPT_TAG:-ablation}_seed${SEED}_${STAMP}}"
mkdir -p "$RUN_DIR"

gpu_mem_used() {
  local gpu="$1"
  local used
  used="$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -n1 | tr -dc '0-9' || true)"
  echo "${used:-0}"
}

monitor_gpu_delta() {
  local gpu="$1"
  local baseline="$2"
  local pid="$3"
  local tag="$4"
  while kill -0 "$pid" 2>/dev/null; do
    local used delta
    used="$(gpu_mem_used "$gpu")"
    delta=$((used - baseline))
    echo "$(date '+%F %T') gpu=${gpu} tag=${tag} used=${used}MiB baseline=${baseline}MiB delta=${delta}MiB"
    if (( delta > MAX_GPU_MEM_MIB )); then
      echo "[GPU budget exceeded] tag=${tag} gpu=${gpu} delta=${delta}MiB > ${MAX_GPU_MEM_MIB}MiB"
      kill "$pid" 2>/dev/null || true
      sleep 5
      kill -9 "$pid" 2>/dev/null || true
      exit 99
    fi
    sleep "$GPU_MEM_POLL_SECONDS"
  done
}

PIDS=()
TAGS=()

wait_for_init_checkpoint() {
  echo "[Wait] PPO init checkpoint: ${INIT_CKPT}"
  while [[ ! -s "$INIT_CKPT" ]]; do
    sleep "$POLL_SECONDS"
  done
  echo "[Ready] ${INIT_CKPT}"
}

start_job() {
  local physical_gpu="$1"
  local tag="$2"
  shift 2
  local baseline
  baseline="$(gpu_mem_used "$physical_gpu")"
  echo "[Launch] ${tag} on GPU${physical_gpu}; baseline=${baseline}MiB"
  (
    export CUDA_VISIBLE_DEVICES="$physical_gpu"
    export PYTHONUNBUFFERED=1
    "$PYTHON_BIN" train.py "$@" --device cuda:0
  ) >"$RUN_DIR/${tag}.log" 2>&1 &
  local pid="$!"
  echo "$pid" >"$RUN_DIR/${tag}.pid"
  monitor_gpu_delta "$physical_gpu" "$baseline" "$pid" "$tag" >"$RUN_DIR/${tag}.gpu_mem.log" 2>&1 &
  PIDS+=("$pid")
  TAGS+=("$tag")
}

wait_batch() {
  local status=0
  for i in "${!PIDS[@]}"; do
    if wait "${PIDS[$i]}"; then
      echo "[Done] ${TAGS[$i]}"
    else
      echo "[Failed] ${TAGS[$i]}"
      status=1
    fi
  done
  PIDS=()
  TAGS=()
  return "$status"
}

trap 'echo "[Abort] stopping child jobs"; jobs -pr | xargs -r kill; exit 130' INT TERM
