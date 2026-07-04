#!/usr/bin/env bash
set -euo pipefail

SCRIPT_TAG="server5_advantage_terms"
GPU_LIST_DEFAULT="${GPU_LIST_DEFAULT:-0}"

print_usage() {
  cat <<EOF
Usage:
  bash scripts/ablation/server5_advantage_terms_slppo.sh GPU_ID GROUP_USE REFERENCE_USE [--detach|--foreground|--no-detach]

Examples:
  bash scripts/ablation/server5_advantage_terms_slppo.sh 0 true false
  bash scripts/ablation/server5_advantage_terms_slppo.sh 1 false true
  bash scripts/ablation/server5_advantage_terms_slppo.sh 2 true true

Fixed: full architecture, SL-PPO, expert budget=7200s, incumbent=on.
true true reuses the Main Results SL-PPO row and launches no job.
false false is outside this ablation and is rejected.
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

SERVER5_SINGLE_RUN="${SERVER5_SINGLE_RUN:-0}"
SERVER5_GROUP_USE="${SERVER5_GROUP_USE:-true}"
SERVER5_REFERENCE_USE="${SERVER5_REFERENCE_USE:-true}"

if [[ $# -gt 0 ]]; then
  case "$1" in
    -h|--help|help)
      print_usage
      exit 0
      ;;
    init|ppo|offline|all|--detach|--foreground|--no-detach|--)
      ;;
    *)
      if [[ $# -lt 3 ]]; then
        echo "[Error] missing arguments." >&2
        print_usage >&2
        exit 2
      fi
      GPU_LIST="$1"
      GPU_LIST_DEFAULT="$1"
      SERVER5_GROUP_USE="$(normalize_bool "$2")"
      SERVER5_REFERENCE_USE="$(normalize_bool "$3")"
      shift 3
      if [[ "$SERVER5_GROUP_USE" == "true" && "$SERVER5_REFERENCE_USE" == "true" ]]; then
        echo "[Reuse] group_use=true reference_use=true; use the Main Results SL-PPO row."
        echo "[Reuse] No Server5 job launched."
        exit 0
      fi
      if [[ "$SERVER5_GROUP_USE" == "false" && "$SERVER5_REFERENCE_USE" == "false" ]]; then
        echo "[Error] group_use=false reference_use=false is outside the Server5 advantage-terms ablation." >&2
        exit 2
      fi
      SERVER5_SINGLE_RUN=1
      export GPU_LIST SERVER5_SINGLE_RUN SERVER5_GROUP_USE SERVER5_REFERENCE_USE
      set -- ppo "$@"
      ;;
  esac
fi

AB_REQUIRE_INIT_CKPT=1
export AB_REQUIRE_INIT_CKPT
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common_ablation.sh"

if [[ "$SERVER5_SINGLE_RUN" != "1" ]]; then
  echo "[Error] Server5 expects positional input: GPU_ID GROUP_USE REFERENCE_USE" >&2
  print_usage >&2
  exit 2
fi

SERVER5_GROUP_USE="$(normalize_bool "$SERVER5_GROUP_USE")"
SERVER5_REFERENCE_USE="$(normalize_bool "$SERVER5_REFERENCE_USE")"
if [[ "$SERVER5_GROUP_USE" == "true" && "$SERVER5_REFERENCE_USE" == "false" ]]; then
  label="group_only"
elif [[ "$SERVER5_GROUP_USE" == "false" && "$SERVER5_REFERENCE_USE" == "true" ]]; then
  label="reference_only"
else
  echo "[Error] Server5 only launches group_only or reference_only. both is reused from Main Results." >&2
  exit 2
fi

EXPERT_TRACE="${EXPERT_TRACE:-$DATA_ROOT/${PROBLEM}/train/Cus${CUSTOMERS}/gurobi_time_trace.csv}"
run_name="${BASE_RUN}_ABL_S5_${label^^}_SLPPO_SEED${SEED}"

echo "[Ready] PPO init checkpoint: ${INIT_CKPT}"
echo "[Server5] label=${label} group_use=${SERVER5_GROUP_USE} reference_use=${SERVER5_REFERENCE_USE} gpu=${GPU_LIST[0]}"
start_job "${GPU_LIST[0]}" "s5_${label}" \
  "${COMMON_ARGS[@]}" \
  --offline-method slppo \
  --pool weighted \
  --ppo-update-epochs 4 \
  --init-checkpoint "$INIT_CKPT" \
  --expert-time-trace "$EXPERT_TRACE" \
  --expert-checkpoint-s 7200 \
  --memory-incumbent on \
  --group-advantage "$( [[ "$SERVER5_GROUP_USE" == "true" ]] && echo on || echo off )" \
  --reference-advantage "$( [[ "$SERVER5_REFERENCE_USE" == "true" ]] && echo on || echo off )" \
  --distance-source road \
  --rdi-embedding none \
  --rdi-encoder-bias on \
  --rdi-encoder-norm softmax \
  --dde on \
  --qkv-delta none \
  --action-key on \
  --action-bias on \
  --run-name "$run_name"

wait_batch
echo "[All done] logs: ${RUN_DIR}"
