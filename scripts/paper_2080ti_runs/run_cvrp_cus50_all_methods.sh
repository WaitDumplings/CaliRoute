#!/usr/bin/env bash
set -euo pipefail

PROBLEM="cvrp"
CUSTOMERS=50
CS=0
SCRIPT_TAG="cvrp_cus50"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

source "$ROOT_DIR/scripts/paper_2080ti_runs/common_2080ti_config.sh"

GPU_PPO="${GPU_PPO:-0}"
GPU_DAPG="${GPU_DAPG:-1}"
GPU_SLPPO="${GPU_SLPPO:-2}"
GPU_AWBC="${GPU_AWBC:-3}"

usage() {
  cat <<EOF
Usage: $(basename "$0") {ppo|gpu0|offline|gpu123|all} [--foreground|--no-detach|--detach] [train.py args...]

Run groups:
  ppo, gpu0        Run PPO only on GPU_PPO (default: GPU0).
  offline, gpu123  Run DAPG/SLPPO/AWBC only on GPU_DAPG/GPU_SLPPO/GPU_AWBC.
  all              Run the original full pipeline: PPO, then DAPG/SLPPO/AWBC.

Detach:
  --detach          Run under nohup in the background (default).
  --foreground,
  --no-detach       Run in the current terminal for debugging.
EOF
}

if [[ $# -eq 0 ]]; then
  usage >&2
  exit 2
fi

case "${1:-}" in
  --mode|--group)
    shift
    if [[ $# -eq 0 ]]; then
      echo "[Error] missing run group after --mode/--group" >&2
      usage >&2
      exit 2
    fi
    RUN_GROUP="$1"
    shift
    ;;
  -h|--help)
    usage
    exit 0
    ;;
  *)
    RUN_GROUP="$1"
    shift
    ;;
esac

case "$RUN_GROUP" in
  ppo|gpu0)
    RUN_GROUP="ppo"
    ;;
  offline|gpu123|gpu1-3|gpu1,2,3)
    RUN_GROUP="offline"
    ;;
  all)
    RUN_GROUP="all"
    ;;
  *)
    echo "[Error] unknown run group: ${RUN_GROUP}" >&2
    usage >&2
    exit 2
    ;;
esac

DETACH="${DETACH:-1}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --detach)
      DETACH=1
      shift
      ;;
    --foreground|--no-detach)
      DETACH=0
      shift
      ;;
    --)
      shift
      break
      ;;
    *)
      break
      ;;
  esac
done
EXTRA_ARGS=("$@")

LOG_ROOT="${LOG_ROOT:-results/launch_logs/paper_2080ti_runs}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-$LOG_ROOT/${SCRIPT_TAG}_${RUN_GROUP}_seed${SEED}_${STAMP}}"
mkdir -p "$RUN_DIR"

if [[ "$DETACH" != "0" && "${CALIROUTE_NOHUP_CHILD:-0}" != "1" ]]; then
  SCRIPT_PATH="$ROOT_DIR/scripts/paper_2080ti_runs/$(basename "${BASH_SOURCE[0]}")"
  LAUNCH_LOG="$RUN_DIR/launcher.log"
  {
    echo "[Start] $(date '+%F %T') launching ${SCRIPT_TAG} ${RUN_GROUP} under nohup"
    echo "[Command] ${SCRIPT_PATH} ${RUN_GROUP} ${EXTRA_ARGS[*]}"
    echo "[Run dir] ${RUN_DIR}"
  } >"$LAUNCH_LOG"
  if command -v setsid >/dev/null 2>&1; then
    nohup setsid env CALIROUTE_NOHUP_CHILD=1 DETACH=0 STAMP="$STAMP" RUN_DIR="$RUN_DIR" \
      bash "$SCRIPT_PATH" "$RUN_GROUP" "${EXTRA_ARGS[@]}" >>"$LAUNCH_LOG" 2>&1 &
  else
    nohup env CALIROUTE_NOHUP_CHILD=1 DETACH=0 STAMP="$STAMP" RUN_DIR="$RUN_DIR" \
      bash "$SCRIPT_PATH" "$RUN_GROUP" "${EXTRA_ARGS[@]}" >>"$LAUNCH_LOG" 2>&1 &
  fi
  launcher_pid="$!"
  echo "$launcher_pid" >"$RUN_DIR/launcher.pid"
  echo "[Detached] ${SCRIPT_TAG} ${RUN_GROUP} pid=${launcher_pid}"
  echo "[Detached] run dir: ${RUN_DIR}"
  echo "[Detached] launcher log: ${LAUNCH_LOG}"
  exit 0
fi

trap 'echo "[Abort] stopping child jobs"; jobs -pr | xargs -r kill; exit 130' INT TERM

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
LAST_PID=""

start_job() {
  local physical_gpu="$1"
  local tag="$2"
  shift 2
  local baseline
  baseline="$(gpu_mem_used "$physical_gpu")"
  echo "[Launch] ${tag} on physical GPU${physical_gpu}; baseline=${baseline}MiB; log=${RUN_DIR}/${tag}.log"
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
  LAST_PID="$pid"
}

wait_for_checkpoint() {
  local ppo_pid="$1"
  local ckpt="$2"
  echo "[Wait] waiting for PPO init checkpoint: ${ckpt}"
  while [[ ! -s "$ckpt" ]]; do
    if ! kill -0 "$ppo_pid" 2>/dev/null; then
      echo "[Error] PPO exited before writing ${ckpt}"
      tail -80 "$RUN_DIR/${SCRIPT_TAG}_ppo.log" || true
      exit 1
    fi
    sleep "$POLL_SECONDS"
  done
  sleep 10
  echo "[Ready] found PPO init checkpoint: ${ckpt}"
}

wait_for_checkpoint_file() {
  local ckpt="$1"
  echo "[Wait] waiting for PPO init checkpoint: ${ckpt}"
  while [[ ! -s "$ckpt" ]]; do
    sleep "$POLL_SECONDS"
  done
  sleep 10
  echo "[Ready] found PPO init checkpoint: ${ckpt}"
}

BASE_RUN="CALIROUTE_${PROBLEM^^}_CUS${CUSTOMERS}_CS${CS}"
PPO_RUN="${BASE_RUN}_PPO_SEED${SEED}_E${EPOCHS}_N${NUM_ENVS}_R${ROLLOUT_STEPS}_2080TI"
DAPG_RUN="${BASE_RUN}_DAPG_SEED${SEED}_E${EPOCHS}_N${NUM_ENVS}_R${ROLLOUT_STEPS}_FROM_PPO${INIT_EPOCH}_2080TI"
SLPPO_RUN="${BASE_RUN}_SLPPO_${SLPPO_POOL^^}POOL_SEED${SEED}_E${EPOCHS}_N${NUM_ENVS}_R${ROLLOUT_STEPS}_FROM_PPO${INIT_EPOCH}_2080TI"
AWBC_RUN="${BASE_RUN}_AWBC_SEED${SEED}_E${EPOCHS}_N${NUM_ENVS}_R${ROLLOUT_STEPS}_FROM_PPO${INIT_EPOCH}_2080TI"
INIT_CKPT="results/checkpoints/Cus_${CUSTOMERS}_CS_${CS}/${PPO_RUN}/seed_${SEED}/checkpoint_epoch_$(printf '%04d' "$INIT_EPOCH").pt"

BASE_ARGS=(
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

launch_ppo() {
  start_job "$GPU_PPO" "${SCRIPT_TAG}_ppo" \
    "${BASE_ARGS[@]}" --offline-method ppo --ppo-update-epochs 3 --run-name "$PPO_RUN" "${EXTRA_ARGS[@]}"
  PPO_PID="$LAST_PID"
}

launch_offline() {
  start_job "$GPU_DAPG" "${SCRIPT_TAG}_dapg" \
    "${BASE_ARGS[@]}" --offline-method dapg --ppo-update-epochs 3 --init-checkpoint "$INIT_CKPT" --run-name "$DAPG_RUN" "${EXTRA_ARGS[@]}"
  start_job "$GPU_SLPPO" "${SCRIPT_TAG}_slppo" \
    "${BASE_ARGS[@]}" --offline-method slppo --pool "$SLPPO_POOL" --ppo-update-epochs 4 --init-checkpoint "$INIT_CKPT" --run-name "$SLPPO_RUN" "${EXTRA_ARGS[@]}"
  start_job "$GPU_AWBC" "${SCRIPT_TAG}_awbc" \
    "${BASE_ARGS[@]}" --offline-method awbc --ppo-update-epochs 3 --init-checkpoint "$INIT_CKPT" --run-name "$AWBC_RUN" "${EXTRA_ARGS[@]}"
}

case "$RUN_GROUP" in
  ppo)
    launch_ppo
    ;;
  offline)
    wait_for_checkpoint_file "$INIT_CKPT"
    launch_offline
    ;;
  all)
    launch_ppo
    wait_for_checkpoint "$PPO_PID" "$INIT_CKPT"
    launch_offline
    ;;
esac

status=0
for i in "${!PIDS[@]}"; do
  if wait "${PIDS[$i]}"; then
    echo "[Done] ${TAGS[$i]}"
  else
    echo "[Failed] ${TAGS[$i]}"
    status=1
  fi
done
exit "$status"
