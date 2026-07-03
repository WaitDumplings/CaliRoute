#!/usr/bin/env bash
set -euo pipefail

SCRIPT_TAG="server3_agda_progressive_slppo"
GPU_LIST_DEFAULT="${GPU_LIST_DEFAULT:-0,1,2,3}"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common_ablation.sh"

if stage_runs_init; then
  run_init_ppo "${GPU_LIST[0]}"
fi
if [[ "$AB_STAGE" == "init" ]]; then
  echo "[Done] stage=init"
  exit 0
fi

wait_for_init_checkpoint

job_idx=0
launch_job() {
  local label="$1"
  local method="$2"
  local updates="$3"
  shift 3
  local gpu="${GPU_LIST[$((job_idx % ${#GPU_LIST[@]}))]}"
  local run_name="${BASE_RUN}_ABL_S3_${label^^}_${method^^}_SEED${SEED}"
  start_job "$gpu" "s3_${label}" \
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
  # Part A: AGDA injection location, fixed RDI=encoder bias, PPO.
  launch_job "agda_kv_only" "ppo" 3 \
    --distance-source road --rdi-embedding none --rdi-encoder-bias on --rdi-encoder-norm softmax \
    --dde on --qkv-delta kv --action-key off --action-bias off

  launch_job "agda_action_key_only" "ppo" 3 \
    --distance-source road --rdi-embedding none --rdi-encoder-bias on --rdi-encoder-norm softmax \
    --dde on --qkv-delta none --action-key on --action-bias on

  launch_job "agda_both" "ppo" 3 \
    --distance-source road --rdi-embedding none --rdi-encoder-bias on --rdi-encoder-norm softmax \
    --dde on --qkv-delta kv --action-key on --action-bias on

  # Part B: progressive PPO corner.
  launch_job "progressive_rdi_off_agda_on_ppo" "ppo" 3 \
    --distance-source none --rdi-embedding none --rdi-encoder-bias off --rdi-encoder-norm softmax \
    --dde on --qkv-delta none --action-key on --action-bias on

  wait_batch
  job_idx=0
fi

if stage_runs_offline_jobs; then
  # Part B: progressive SL-PPO corners.
  launch_job "progressive_rdi_off_agda_off_slppo" "slppo" 4 \
    --distance-source none --rdi-embedding none --rdi-encoder-bias off --rdi-encoder-norm softmax \
    --dde off --qkv-delta none --action-key off --action-bias off

  launch_job "progressive_rdi_on_agda_off_slppo" "slppo" 4 \
    --distance-source road --rdi-embedding none --rdi-encoder-bias on --rdi-encoder-norm softmax \
    --dde off --qkv-delta none --action-key off --action-bias off

  launch_job "progressive_rdi_off_agda_on_slppo" "slppo" 4 \
    --distance-source none --rdi-embedding none --rdi-encoder-bias off --rdi-encoder-norm softmax \
    --dde on --qkv-delta none --action-key on --action-bias on

  # Part C: solution-level advantage components, full calibrated architecture.
  launch_job "slppo_group_only" "slppo" 4 \
    --distance-source road --rdi-embedding none --rdi-encoder-bias on --rdi-encoder-norm softmax \
    --dde on --qkv-delta none --action-key on --action-bias on \
    --group-advantage on --reference-advantage off --sl-candidate off --memory-incumbent on

  launch_job "slppo_reference_only" "slppo" 4 \
    --distance-source road --rdi-embedding none --rdi-encoder-bias on --rdi-encoder-norm softmax \
    --dde on --qkv-delta none --action-key on --action-bias on \
    --group-advantage off --reference-advantage on --sl-candidate off --memory-incumbent on
fi

wait_batch
echo "[All done] logs: ${RUN_DIR}"
