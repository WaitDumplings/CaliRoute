#!/usr/bin/env bash
set -euo pipefail

SCRIPT_TAG="server7_agda_feature_groups"
GPU_LIST_DEFAULT="${GPU_LIST_DEFAULT:-0}"

print_usage() {
  cat <<EOF
Usage:
  bash scripts/ablation/server7_agda_feature_groups_ppo.sh GPU_ID DISTANCE_FEATURE CAPACITY_FEATURE BATTERY_FEATURE [--detach|--foreground|--no-detach]

Examples:
  bash scripts/ablation/server7_agda_feature_groups_ppo.sh 0 false true true
  bash scripts/ablation/server7_agda_feature_groups_ppo.sh 1 true false true
  bash scripts/ablation/server7_agda_feature_groups_ppo.sh 2 true true false
  bash scripts/ablation/server7_agda_feature_groups_ppo.sh 3 true true true

Fixed: full architecture, plain PPO, EVRPTW Cus100, single seed,
RDI=encoder_bias, AGDA=action_key. Exactly one feature group should be false.
true true true reuses Server3 action_key_only and launches no job.
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

SERVER7_SINGLE_RUN="${SERVER7_SINGLE_RUN:-0}"
SERVER7_DISTANCE_FEATURE="${SERVER7_DISTANCE_FEATURE:-true}"
SERVER7_CAPACITY_FEATURE="${SERVER7_CAPACITY_FEATURE:-true}"
SERVER7_BATTERY_FEATURE="${SERVER7_BATTERY_FEATURE:-true}"

if [[ $# -gt 0 ]]; then
  case "$1" in
    -h|--help|help)
      print_usage
      exit 0
      ;;
    init|ppo|offline|all|--detach|--foreground|--no-detach|--)
      ;;
    *)
      if [[ $# -lt 4 ]]; then
        echo "[Error] missing arguments." >&2
        print_usage >&2
        exit 2
      fi
      GPU_LIST="$1"
      GPU_LIST_DEFAULT="$1"
      SERVER7_DISTANCE_FEATURE="$(normalize_bool "$2")"
      SERVER7_CAPACITY_FEATURE="$(normalize_bool "$3")"
      SERVER7_BATTERY_FEATURE="$(normalize_bool "$4")"
      shift 4
      false_count=0
      [[ "$SERVER7_DISTANCE_FEATURE" == "false" ]] && false_count=$((false_count + 1))
      [[ "$SERVER7_CAPACITY_FEATURE" == "false" ]] && false_count=$((false_count + 1))
      [[ "$SERVER7_BATTERY_FEATURE" == "false" ]] && false_count=$((false_count + 1))
      if [[ "$false_count" -eq 0 ]]; then
        echo "[Reuse] all AGDA feature groups enabled; use Server3 action_key_only."
        echo "[Reuse] No Server7 job launched."
        exit 0
      fi
      if [[ "$false_count" -gt 1 ]]; then
        echo "[Error] Server7 launches exactly one removed feature group; multiple-false is outside this ablation." >&2
        exit 2
      fi
      SERVER7_SINGLE_RUN=1
      export GPU_LIST SERVER7_SINGLE_RUN SERVER7_DISTANCE_FEATURE SERVER7_CAPACITY_FEATURE SERVER7_BATTERY_FEATURE
      set -- ppo "$@"
      ;;
  esac
fi

AB_REQUIRE_INIT_CKPT=1
export AB_REQUIRE_INIT_CKPT
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common_ablation.sh"

if [[ "$SERVER7_SINGLE_RUN" != "1" ]]; then
  echo "[Error] Server7 expects positional input: GPU_ID DISTANCE_FEATURE CAPACITY_FEATURE BATTERY_FEATURE" >&2
  print_usage >&2
  exit 2
fi

SERVER7_DISTANCE_FEATURE="$(normalize_bool "$SERVER7_DISTANCE_FEATURE")"
SERVER7_CAPACITY_FEATURE="$(normalize_bool "$SERVER7_CAPACITY_FEATURE")"
SERVER7_BATTERY_FEATURE="$(normalize_bool "$SERVER7_BATTERY_FEATURE")"
false_count=0
drop_group=""
if [[ "$SERVER7_DISTANCE_FEATURE" == "false" ]]; then
  false_count=$((false_count + 1))
  drop_group="distance"
fi
if [[ "$SERVER7_CAPACITY_FEATURE" == "false" ]]; then
  false_count=$((false_count + 1))
  drop_group="capacity"
fi
if [[ "$SERVER7_BATTERY_FEATURE" == "false" ]]; then
  false_count=$((false_count + 1))
  drop_group="battery"
fi
if [[ "$false_count" -ne 1 ]]; then
  echo "[Error] Server7 launches exactly one removed feature group; all-true is reused and multiple-false is outside this ablation." >&2
  exit 2
fi

label="without_${drop_group}_features"
run_name="${BASE_RUN}_ABL_S7_${label^^}_PPO_SEED${SEED}"

echo "[Ready] PPO init checkpoint: ${INIT_CKPT}"
echo "[Server7] label=${label} distance_feature=${SERVER7_DISTANCE_FEATURE} capacity_feature=${SERVER7_CAPACITY_FEATURE} battery_feature=${SERVER7_BATTERY_FEATURE} gpu=${GPU_LIST[0]}"
start_job "${GPU_LIST[0]}" "s7_${label}" \
  "${COMMON_ARGS[@]}" \
  --offline-method ppo \
  --ppo-update-epochs 3 \
  --init-checkpoint "$INIT_CKPT" \
  --distance-source road \
  --rdi-embedding none \
  --rdi-encoder-bias on \
  --rdi-encoder-norm softmax \
  --dde on \
  --qkv-delta none \
  --action-key on \
  --action-bias on \
  --agda-drop-groups "$drop_group" \
  --run-name "$run_name"

wait_batch
echo "[All done] logs: ${RUN_DIR}"
