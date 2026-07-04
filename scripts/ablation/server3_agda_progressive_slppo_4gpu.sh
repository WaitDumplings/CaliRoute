#!/usr/bin/env bash
set -euo pipefail

SCRIPT_TAG="server3_agda_progressive_slppo"
GPU_LIST_DEFAULT="${GPU_LIST_DEFAULT:-0,1,2,3}"

print_server3_agda_usage() {
  cat <<EOF_USAGE
Usage:
  bash scripts/ablation/server3_agda_progressive_slppo_4gpu.sh GPU_ID DYNAMIC_KV DYNAMIC_ACTION_LOGITS [--detach|--foreground|--no-detach]
  bash scripts/ablation/server3_agda_progressive_slppo_4gpu.sh [init|ppo|offline|all] [--detach|--foreground|--no-detach]

Examples:
  bash scripts/ablation/server3_agda_progressive_slppo_4gpu.sh 0 true true
  bash scripts/ablation/server3_agda_progressive_slppo_4gpu.sh 1 true false
  bash scripts/ablation/server3_agda_progressive_slppo_4gpu.sh 2 false true
  bash scripts/ablation/server3_agda_progressive_slppo_4gpu.sh 3 false false

Notes:
  GPU_ID can be one GPU id, such as 2, or a comma list, such as 0,1,2,3.
  AGDA single-run mode always uses road graph distance with encoder bias only.
  Query/DDE is on by default; DYNAMIC_KV controls --qkv-delta kv vs none.
  DYNAMIC_ACTION_LOGITS controls both --action-key and --action-bias.
  false false reuses the Server2 RDI encoder_bias_only result and launches no job.
EOF_USAGE
}

normalize_bool_arg() {
  local value="${1,,}"
  case "$value" in
    true|1|yes|y|on)
      echo "true"
      ;;
    false|0|no|n|off)
      echo "false"
      ;;
    *)
      echo "[Error] expected boolean true/false, got: $1" >&2
      exit 2
      ;;
  esac
}

AGDA_SINGLE_RUN="${AGDA_SINGLE_RUN:-0}"
AGDA_DYNAMIC_KV="${AGDA_DYNAMIC_KV:-true}"
AGDA_DYNAMIC_ACTION_LOGITS="${AGDA_DYNAMIC_ACTION_LOGITS:-true}"

if [[ $# -gt 0 ]]; then
  case "$1" in
    -h|--help|help)
      print_server3_agda_usage
      exit 0
      ;;
    init|ppo|offline|all|--detach|--foreground|--no-detach|--)
      ;;
    *)
      GPU_LIST="$1"
      GPU_LIST_DEFAULT="$1"
      shift
      if [[ $# -eq 0 || "$1" == --* ]]; then
        echo "[Error] missing DYNAMIC_KV argument." >&2
        print_server3_agda_usage >&2
        exit 2
      fi
      AGDA_DYNAMIC_KV="$(normalize_bool_arg "$1")"
      shift
      if [[ $# -eq 0 || "$1" == --* ]]; then
        echo "[Error] missing DYNAMIC_ACTION_LOGITS argument." >&2
        print_server3_agda_usage >&2
        exit 2
      fi
      AGDA_DYNAMIC_ACTION_LOGITS="$(normalize_bool_arg "$1")"
      shift
      AGDA_SINGLE_RUN=1
      export GPU_LIST AGDA_SINGLE_RUN AGDA_DYNAMIC_KV AGDA_DYNAMIC_ACTION_LOGITS
      set -- ppo "$@"
      ;;
  esac
fi

if [[ "$AGDA_SINGLE_RUN" == "1" ]]; then
  AGDA_DYNAMIC_KV="$(normalize_bool_arg "$AGDA_DYNAMIC_KV")"
  AGDA_DYNAMIC_ACTION_LOGITS="$(normalize_bool_arg "$AGDA_DYNAMIC_ACTION_LOGITS")"
fi

if [[ "$AGDA_SINGLE_RUN" == "1" && "$AGDA_DYNAMIC_KV" == "false" && "$AGDA_DYNAMIC_ACTION_LOGITS" == "false" ]]; then
  echo "[Reuse] dynamic_kv=false dynamic_action_logits=false; use Server2 RDI encoder_bias_only PPO result."
  echo "[Reuse] No AGDA job launched."
  exit 0
fi

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

agda_single_label() {
  local dynamic_kv="$1"
  local dynamic_action_logits="$2"
  if [[ "$dynamic_kv" == "true" && "$dynamic_action_logits" == "true" ]]; then
    echo "agda_both"
  elif [[ "$dynamic_kv" == "true" && "$dynamic_action_logits" == "false" ]]; then
    echo "agda_dynamic_kv"
  elif [[ "$dynamic_kv" == "false" && "$dynamic_action_logits" == "true" ]]; then
    echo "agda_dynamic_action_logits"
  else
    echo "[Error] false false should reuse Server2 RDI encoder_bias_only instead of launching." >&2
    exit 2
  fi
}

launch_agda_single() {
  local qkv_delta="none"
  local action_switch="off"
  local label
  if [[ "$AGDA_DYNAMIC_KV" == "true" ]]; then
    qkv_delta="kv"
  fi
  if [[ "$AGDA_DYNAMIC_ACTION_LOGITS" == "true" ]]; then
    action_switch="on"
  fi
  label="$(agda_single_label "$AGDA_DYNAMIC_KV" "$AGDA_DYNAMIC_ACTION_LOGITS")"
  echo "[AGDA] label=${label} graph=road rdi=encoder_bias query=on dynamic_kv=${AGDA_DYNAMIC_KV} dynamic_action_logits=${AGDA_DYNAMIC_ACTION_LOGITS}"
  launch_job "$label" "ppo" 3 \
    --distance-source road --rdi-embedding none --rdi-encoder-bias on --rdi-encoder-norm softmax \
    --dde on --qkv-delta "$qkv_delta" --action-key "$action_switch" --action-bias "$action_switch"
}

if stage_runs_ppo_jobs; then
  if [[ "$AGDA_SINGLE_RUN" == "1" ]]; then
    launch_agda_single
    wait_batch
    job_idx=0
  else
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
