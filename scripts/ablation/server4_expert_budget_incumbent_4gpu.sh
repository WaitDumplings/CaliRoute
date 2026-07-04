#!/usr/bin/env bash
set -euo pipefail

SCRIPT_TAG="server4_expert_budget_incumbent"
GPU_LIST_DEFAULT="${GPU_LIST_DEFAULT:-0}"

print_usage() {
  cat <<EOF
Usage:
  bash scripts/ablation/server4_expert_budget_incumbent_4gpu.sh GPU_ID EXPERT_BUDGET_S INCUMBENT [--detach|--foreground|--no-detach]

Examples:
  bash scripts/ablation/server4_expert_budget_incumbent_4gpu.sh 0 60 true
  bash scripts/ablation/server4_expert_budget_incumbent_4gpu.sh 1 7200 false

Fixed: EVRPTW Cus100, single seed, full architecture, SL-PPO,
RDI=encoder_bias, AGDA=action_key. EXPERT_BUDGET_S must be one of
60, 300, 900, 3600, 7200. INCUMBENT accepts true/false or on/off.
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

bool_to_on_off() {
  if [[ "$1" == "true" ]]; then
    echo "on"
  else
    echo "off"
  fi
}

validate_budget() {
  case "$1" in
    60|300|900|3600|7200)
      ;;
    *)
      echo "[Error] EXPERT_BUDGET_S must be one of 60, 300, 900, 3600, 7200; got: $1" >&2
      exit 2
      ;;
  esac
}

SERVER4_SINGLE_RUN="${SERVER4_SINGLE_RUN:-0}"
SERVER4_EXPERT_BUDGET="${SERVER4_EXPERT_BUDGET:-7200}"
SERVER4_INCUMBENT="${SERVER4_INCUMBENT:-true}"

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
      SERVER4_EXPERT_BUDGET="$2"
      SERVER4_INCUMBENT="$(normalize_bool "$3")"
      shift 3
      validate_budget "$SERVER4_EXPERT_BUDGET"
      SERVER4_SINGLE_RUN=1
      export GPU_LIST SERVER4_SINGLE_RUN SERVER4_EXPERT_BUDGET SERVER4_INCUMBENT
      set -- ppo "$@"
      ;;
  esac
fi

AB_REQUIRE_INIT_CKPT=1
export AB_REQUIRE_INIT_CKPT
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common_ablation.sh"

if [[ "$SERVER4_SINGLE_RUN" != "1" ]]; then
  echo "[Error] Server4 expects positional input: GPU_ID EXPERT_BUDGET_S INCUMBENT" >&2
  print_usage >&2
  exit 2
fi

validate_budget "$SERVER4_EXPERT_BUDGET"
SERVER4_INCUMBENT="$(normalize_bool "$SERVER4_INCUMBENT")"
incumbent_switch="$(bool_to_on_off "$SERVER4_INCUMBENT")"
EXPERT_TRACE="${EXPERT_TRACE:-$DATA_ROOT/${PROBLEM}/train/Cus${CUSTOMERS}/gurobi_time_trace.csv}"
label="budget${SERVER4_EXPERT_BUDGET}_inc${incumbent_switch}"
run_name="${BASE_RUN}_ABL_S4_BUDGET${SERVER4_EXPERT_BUDGET}_INC${incumbent_switch^^}_SLPPO_SEED${SEED}"

echo "[Ready] PPO init checkpoint: ${INIT_CKPT}"
echo "[Server4] budget=${SERVER4_EXPERT_BUDGET}s incumbent=${incumbent_switch} gpu=${GPU_LIST[0]}"
start_job "${GPU_LIST[0]}" "s4_${label}" \
  "${COMMON_ARGS[@]}" \
  --offline-method slppo \
  --pool weighted \
  --ppo-update-epochs 4 \
  --init-checkpoint "$INIT_CKPT" \
  --expert-time-trace "$EXPERT_TRACE" \
  --expert-checkpoint-s "$SERVER4_EXPERT_BUDGET" \
  --memory-incumbent "$incumbent_switch" \
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
