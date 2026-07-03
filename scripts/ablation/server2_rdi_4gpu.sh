#!/usr/bin/env bash
set -euo pipefail

SCRIPT_TAG="server2_rdi"
GPU_LIST_DEFAULT="${GPU_LIST_DEFAULT:-0,1,2,3}"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common_ablation.sh"

if stage_runs_init; then
  run_init_ppo "${GPU_LIST[0]}"
fi
if [[ "$AB_STAGE" == "init" ]]; then
  echo "[Done] stage=init"
  exit 0
fi
if stage_runs_offline_jobs && ! stage_runs_ppo_jobs; then
  echo "[Done] stage=offline; Server2 has no non-PPO jobs."
  exit 0
fi

wait_for_init_checkpoint

NN_MATCH_CSV="$RUN_DIR/rdi_nn_match.csv"

record_nn_match() {
  local label="$1"
  local representation="$2"
  "$PYTHON_BIN" scripts/ablation/compute_rdi_nn_match.py \
    --dataset "$DATA_ROOT/${PROBLEM}/val/Cus${CUSTOMERS}" \
    --problem "$PROBLEM" \
    --customers "$CUSTOMERS" \
    --charging-stations "$CS" \
    --representation "$representation" \
    --svd-rank 10 \
    --output "$NN_MATCH_CSV" >"$RUN_DIR/${label}.nn_match.log" 2>&1 || true
}

launch_rdi() {
  local gpu="$1"
  local label="$2"
  local nn_rep="$3"
  shift 3
  local run_name="${BASE_RUN}_ABL_S2_RDI_${label^^}_PPO_SEED${SEED}"
  record_nn_match "s2_${label}" "$nn_rep"
  start_job "$gpu" "s2_${label}" \
    "${COMMON_ARGS[@]}" \
    --offline-method ppo \
    --ppo-update-epochs 3 \
    --init-checkpoint "$INIT_CKPT" \
    --dde off \
    --qkv-delta none \
    --action-key off \
    --action-bias off \
    --run-name "$run_name" \
    "$@"
}

job_idx=0
launch_next() {
  local label="$1"
  local nn_rep="$2"
  shift 2
  local gpu="${GPU_LIST[$((job_idx % ${#GPU_LIST[@]}))]}"
  launch_rdi "$gpu" "$label" "$nn_rep" "$@"
  job_idx=$((job_idx + 1))
  if (( job_idx % ${#GPU_LIST[@]} == 0 )); then
    wait_batch
  fi
}

launch_next "base" "none" \
  --distance-source none --rdi-embedding none --rdi-encoder-bias off --rdi-encoder-norm softmax

# Euclidean uses the same additive-bias slot as our RDI, but with the wrong metric.
launch_next "euclidean" "euclidean" \
  --distance-source euclidean --rdi-embedding none --rdi-encoder-bias on --rdi-encoder-norm softmax

launch_next "embedding_svd_only" "svd" \
  --distance-source road --rdi-embedding svd --rdi-encoder-bias off --rdi-encoder-norm softmax

launch_next "encoder_sinkhorn_only" "road" \
  --distance-source road --rdi-embedding none --rdi-encoder-bias off --rdi-encoder-norm sinkhorn

launch_next "encoder_bias_only" "road" \
  --distance-source road --rdi-embedding none --rdi-encoder-bias on --rdi-encoder-norm softmax

launch_next "embedding_svd_encoder_sinkhorn" "road" \
  --distance-source road --rdi-embedding svd --rdi-encoder-bias off --rdi-encoder-norm sinkhorn

launch_next "embedding_svd_encoder_bias" "road" \
  --distance-source road --rdi-embedding svd --rdi-encoder-bias on --rdi-encoder-norm softmax

launch_next "embedding_svd_sinkhorn_bias" "road" \
  --distance-source road --rdi-embedding svd --rdi-encoder-bias on --rdi-encoder-norm sinkhorn

wait_batch
echo "[All done] logs: ${RUN_DIR}"
