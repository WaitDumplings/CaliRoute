#!/usr/bin/env bash
set -euo pipefail

SCRIPT_TAG="server8_general_rdi_agda_slppo"
GPU_LIST_DEFAULT="${GPU_LIST_DEFAULT:-0,1,2,3}"

print_usage() {
  cat <<EOF
Usage:
  bash scripts/ablation/server8_general_rdi_agda_slppo.sh GPU_ID RDI AGDA SLPPO [--detach|--foreground|--no-detach]
  bash scripts/ablation/server8_general_rdi_agda_slppo.sh GPU_ID all [--detach|--foreground|--no-detach]
  bash scripts/ablation/server8_general_rdi_agda_slppo.sh [ppo|offline|all] [--detach|--foreground|--no-detach]

Examples:
  bash scripts/ablation/server8_general_rdi_agda_slppo.sh 0 true true true
  bash scripts/ablation/server8_general_rdi_agda_slppo.sh 1 true false false
  bash scripts/ablation/server8_general_rdi_agda_slppo.sh 0,1,2,3 all

Factors:
  RDI=true   road distance via encoder bias only.
  RDI=false  no distance injection.
  AGDA=true  our dynamic action logits only.
  AGDA=false AGDA off.
  SLPPO=true SL-PPO with expert budget=7200s and incumbent=on.
  SLPPO=false plain PPO.

The shared PPO epoch-100 init checkpoint must already exist.
EOF
}

normalize_bool() {
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

SERVER8_MODE="${SERVER8_MODE:-all}"
SERVER8_RDI="${SERVER8_RDI:-true}"
SERVER8_AGDA="${SERVER8_AGDA:-true}"
SERVER8_SLPPO="${SERVER8_SLPPO:-true}"

if [[ $# -gt 0 ]]; then
  case "$1" in
    -h|--help|help)
      print_usage
      exit 0
      ;;
    init|ppo|offline|all|--detach|--foreground|--no-detach|--)
      ;;
    *)
      GPU_LIST="$1"
      GPU_LIST_DEFAULT="$1"
      shift
      if [[ $# -eq 0 || "$1" == --* ]]; then
        echo "[Error] missing RDI|all argument." >&2
        print_usage >&2
        exit 2
      fi
      if [[ "${1,,}" == "all" ]]; then
        SERVER8_MODE="all"
        shift
        export GPU_LIST SERVER8_MODE
        set -- all "$@"
      else
        if [[ $# -lt 3 ]]; then
          echo "[Error] missing RDI AGDA SLPPO arguments." >&2
          print_usage >&2
          exit 2
        fi
        SERVER8_MODE="single"
        SERVER8_RDI="$(normalize_bool "$1")"
        SERVER8_AGDA="$(normalize_bool "$2")"
        SERVER8_SLPPO="$(normalize_bool "$3")"
        shift 3
        export GPU_LIST SERVER8_MODE SERVER8_RDI SERVER8_AGDA SERVER8_SLPPO
        if [[ "$SERVER8_SLPPO" == "true" ]]; then
          set -- offline "$@"
        else
          set -- ppo "$@"
        fi
      fi
      ;;
  esac
fi

AB_REQUIRE_INIT_CKPT=1
export AB_REQUIRE_INIT_CKPT
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common_ablation.sh"

if [[ "$AB_STAGE" == "init" ]]; then
  echo "[Done] stage=init; use scripts/ablation/init_ppo100_1gpu.sh if the shared init checkpoint is missing."
  exit 0
fi

EXPERT_TRACE="${EXPERT_TRACE:-$DATA_ROOT/${PROBLEM}/train/Cus${CUSTOMERS}/gurobi_time_trace.csv}"
job_idx=0

bool_label() {
  if [[ "$1" == "true" ]]; then
    echo "on"
  else
    echo "off"
  fi
}

launch_combo() {
  local rdi agda slppo
  rdi="$(normalize_bool "$1")"
  agda="$(normalize_bool "$2")"
  slppo="$(normalize_bool "$3")"

  local method updates label run_name gpu rdi_state agda_state slppo_state
  rdi_state="$(bool_label "$rdi")"
  agda_state="$(bool_label "$agda")"
  slppo_state="$(bool_label "$slppo")"
  method="ppo"
  updates=3
  if [[ "$slppo" == "true" ]]; then
    method="slppo"
    updates=4
  fi

  label="rdi_${rdi_state}_agda_${agda_state}_slppo_${slppo_state}"
  run_name="${BASE_RUN}_ABL_S8_RDI_${rdi_state^^}_AGDA_${agda_state^^}_SLPPO_${slppo_state^^}_SEED${SEED}"
  gpu="${GPU_LIST[$((job_idx % ${#GPU_LIST[@]}))]}"

  rdi_args=(--distance-source none --rdi-embedding none --rdi-encoder-bias off --rdi-encoder-norm softmax)
  if [[ "$rdi" == "true" ]]; then
    rdi_args=(--distance-source road --rdi-embedding none --rdi-encoder-bias on --rdi-encoder-norm softmax)
  fi

  agda_args=(--dde off --qkv-delta none --action-key off --action-bias off)
  if [[ "$agda" == "true" ]]; then
    agda_args=(--dde on --qkv-delta none --action-key on --action-bias on)
  fi

  method_args=(--offline-method "$method" --pool weighted --ppo-update-epochs "$updates" --init-checkpoint "$INIT_CKPT")
  if [[ "$slppo" == "true" ]]; then
    method_args+=(
      --expert-time-trace "$EXPERT_TRACE"
      --expert-checkpoint-s 7200
      --memory-incumbent on
      --group-advantage on
      --reference-advantage on
    )
  fi

  echo "[Server8] label=${label} rdi=${rdi} agda_dynamic_action_logits=${agda} slppo=${slppo} gpu=${gpu}"
  start_job "$gpu" "s8_${label}" \
    "${COMMON_ARGS[@]}" \
    "${method_args[@]}" \
    "${rdi_args[@]}" \
    "${agda_args[@]}" \
    --run-name "$run_name"

  job_idx=$((job_idx + 1))
  if (( job_idx % ${#GPU_LIST[@]} == 0 )); then
    wait_batch
  fi
}

if [[ "$SERVER8_MODE" == "single" ]]; then
  launch_combo "$SERVER8_RDI" "$SERVER8_AGDA" "$SERVER8_SLPPO"
else
  if stage_runs_ppo_jobs; then
    launch_combo false false false
    launch_combo true false false
    launch_combo false true false
    launch_combo true true false
  fi
  if stage_runs_offline_jobs; then
    launch_combo false false true
    launch_combo true false true
    launch_combo false true true
    launch_combo true true true
  fi
fi

wait_batch
echo "[All done] logs: ${RUN_DIR}"
