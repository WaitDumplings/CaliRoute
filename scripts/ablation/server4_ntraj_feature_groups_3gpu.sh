#!/usr/bin/env bash
set -euo pipefail

SCRIPT_TAG="server4_ntraj_feature_groups"
GPU_LIST_DEFAULT="${GPU_LIST_DEFAULT:-0,1,2}"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common_ablation.sh"

if stage_runs_init; then
  run_init_ppo "${GPU_LIST[0]}"
fi
if [[ "$AB_STAGE" == "init" ]]; then
  echo "[Done] stage=init"
  exit 0
fi

wait_for_init_checkpoint

FULL_ARCH_ARGS=(
  --distance-source road
  --rdi-embedding none
  --rdi-encoder-bias on
  --rdi-encoder-norm softmax
  --dde on
  --qkv-delta none
  --action-key on
  --action-bias on
)

job_idx=0
launch_job() {
  local label="$1"
  local method="$2"
  local updates="$3"
  shift 3
  local gpu="${GPU_LIST[$((job_idx % ${#GPU_LIST[@]}))]}"
  local run_name="${BASE_RUN}_ABL_S4_${label^^}_${method^^}_SEED${SEED}"
  start_job "$gpu" "s4_${label}" \
    "${COMMON_ARGS[@]}" \
    --offline-method "$method" \
    --ppo-update-epochs "$updates" \
    --init-checkpoint "$INIT_CKPT" \
    --pool weighted \
    --run-name "$run_name" \
    "$@"
  job_idx=$((job_idx + 1))
  if (( job_idx % ${#GPU_LIST[@]} == 0 )); then
    wait_batch
  fi
}

if stage_runs_ppo_jobs; then
  # Part B: AGDA dynamic-feature group ablation, fixed full RDI and plain PPO.
  launch_job "without_distance_features" "ppo" 3 \
    "${FULL_ARCH_ARGS[@]}" \
    --agda-drop-groups distance

  launch_job "without_capacity_features" "ppo" 3 \
    "${FULL_ARCH_ARGS[@]}" \
    --agda-drop-groups capacity

  launch_job "without_battery_features" "ppo" 3 \
    "${FULL_ARCH_ARGS[@]}" \
    --agda-drop-groups battery

  wait_batch
  job_idx=0
fi

if stage_runs_offline_jobs; then
  # Part A: run K=100 first because it is the OOM-risk cell.
  launch_job "ntraj100" "slppo" 4 \
    --n-traj 100 \
    "${FULL_ARCH_ARGS[@]}"
  wait_batch

  job_idx=0
  launch_job "ntraj5" "slppo" 4 \
    --n-traj 5 \
    "${FULL_ARCH_ARGS[@]}"

  launch_job "ntraj15" "slppo" 4 \
    --n-traj 15 \
    "${FULL_ARCH_ARGS[@]}"
fi

wait_batch
echo "[All done] logs: ${RUN_DIR}"
