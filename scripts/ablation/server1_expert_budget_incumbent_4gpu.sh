#!/usr/bin/env bash
set -euo pipefail

SCRIPT_TAG="server1_expert_budget_incumbent"
GPU_LIST_DEFAULT="${GPU_LIST_DEFAULT:-0,1,2,3}"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common_ablation.sh"

wait_for_init_checkpoint

EXPERT_TRACE="${EXPERT_TRACE:-$DATA_ROOT/${PROBLEM}/train/Cus${CUSTOMERS}/gurobi_time_trace.csv}"
BUDGETS=(${BUDGETS:-60 300 900 3600 7200})
INCUMBENT_SETTINGS=(${INCUMBENT_SETTINGS:-on off})

job_idx=0
for budget in "${BUDGETS[@]}"; do
  for incumbent in "${INCUMBENT_SETTINGS[@]}"; do
    gpu="${GPU_LIST[$((job_idx % ${#GPU_LIST[@]}))]}"
    tag="s1_budget${budget}_inc${incumbent}"
    run_name="${BASE_RUN}_ABL_S1_BUDGET${budget}_INC${incumbent^^}_SLPPO_SEED${SEED}"
    start_job "$gpu" "$tag" \
      "${COMMON_ARGS[@]}" \
      --offline-method slppo \
      --pool weighted \
      --ppo-update-epochs 4 \
      --init-checkpoint "$INIT_CKPT" \
      --expert-time-trace "$EXPERT_TRACE" \
      --expert-checkpoint-s "$budget" \
      --memory-incumbent "$incumbent" \
      --run-name "$run_name"
    job_idx=$((job_idx + 1))
    if (( job_idx % ${#GPU_LIST[@]} == 0 )); then
      wait_batch
    fi
  done
done

wait_batch
echo "[All done] logs: ${RUN_DIR}"
