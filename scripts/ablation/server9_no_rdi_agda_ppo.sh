#!/usr/bin/env bash
set -euo pipefail

SCRIPT_TAG="server9_no_rdi_agda_ppo"
GPU_LIST_DEFAULT="${GPU_LIST_DEFAULT:-0}"

# This ablation is intentionally plain PPO from scratch with both RDI and AGDA
# disabled. The only experimental knobs are problem family, customer size, and
# seed; GPU can be passed positionally for scheduling convenience.
NUM_ENVS="${NUM_ENVS:-256}"
N_TRAJ="${N_TRAJ:-50}"
EPOCHS="${EPOCHS:-1500}"
PPO_STEP_CHUNK_SIZE="${PPO_STEP_CHUNK_SIZE:-32}"
NUM_MINIBATCHES="${NUM_MINIBATCHES:-2}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-128}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-150}"
EVAL_MAX_STEPS="${EVAL_MAX_STEPS:-150}"

if [[ -z "${PYTHON_BIN:-}" && -x /home/npg/miniconda3/envs/maojie/bin/python ]]; then
  PYTHON_BIN="/home/npg/miniconda3/envs/maojie/bin/python"
fi

print_usage() {
  cat <<EOF
Usage:
  bash scripts/ablation/server9_no_rdi_agda_ppo.sh GPU_ID PROBLEM CUSTOMERS [SEED] [--detach|--foreground|--no-detach]
  GPU_LIST=GPU_ID bash scripts/ablation/server9_no_rdi_agda_ppo.sh PROBLEM CUSTOMERS [SEED] [--detach|--foreground|--no-detach]

Examples:
  bash scripts/ablation/server9_no_rdi_agda_ppo.sh 0 evrptw 100
  bash scripts/ablation/server9_no_rdi_agda_ppo.sh 3 vrptw 100 3009
  EPOCHS=11 bash scripts/ablation/server9_no_rdi_agda_ppo.sh 0 evrptw 100 --foreground

Fixed:
  plain PPO from scratch, RDI=off, AGDA=off, SL-PPO=off,
  num_envs=${NUM_ENVS}, n_traj=${N_TRAJ}, ppo_step_chunk_size=${PPO_STEP_CHUNK_SIZE},
  num_minibatches=${NUM_MINIBATCHES}, eval every 20 epochs and at the final epoch.

Inputs:
  PROBLEM    evrptw | vrptw | cvrp
  CUSTOMERS  15 | 50 | 100
  SEED       default 3009
EOF
}

is_problem() {
  case "${1,,}" in
    evrptw|vrptw|cvrptw|cvrp) return 0 ;;
    *) return 1 ;;
  esac
}

normalize_problem() {
  case "${1,,}" in
    evrptw) echo "evrptw" ;;
    vrptw|cvrptw) echo "vrptw" ;;
    cvrp) echo "cvrp" ;;
    *)
      echo "[Error] PROBLEM must be one of: evrptw, vrptw, cvrp; got: $1" >&2
      exit 2
      ;;
  esac
}

validate_customers() {
  case "$1" in
    15|50|100) ;;
    *)
      echo "[Error] CUSTOMERS must be one of: 15, 50, 100; got: $1" >&2
      exit 2
      ;;
  esac
}

POSITIONAL=()
FLAGS=()
REQUESTED_STAGE="ppo"
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help|help)
      print_usage
      exit 0
      ;;
    init|ppo|offline|all)
      REQUESTED_STAGE="$1"
      shift
      ;;
    --detach|--foreground|--no-detach|--)
      FLAGS+=("$1")
      shift
      ;;
    *)
      POSITIONAL+=("$1")
      shift
      ;;
  esac
done

if [[ "${#POSITIONAL[@]}" -lt 2 ]]; then
  if [[ -n "${PROBLEM:-}" && -n "${CUSTOMERS:-}" ]]; then
    PROBLEM="$(normalize_problem "$PROBLEM")"
    CUSTOMERS="$CUSTOMERS"
    SEED="${SEED:-3009}"
  else
    echo "[Error] missing PROBLEM/CUSTOMERS arguments." >&2
    print_usage >&2
    exit 2
  fi
elif is_problem "${POSITIONAL[0]}"; then
  PROBLEM="$(normalize_problem "${POSITIONAL[0]}")"
  CUSTOMERS="${POSITIONAL[1]}"
  SEED="${POSITIONAL[2]:-${SEED:-3009}}"
else
  if [[ "${#POSITIONAL[@]}" -lt 3 ]]; then
    echo "[Error] missing PROBLEM/CUSTOMERS after GPU_ID." >&2
    print_usage >&2
    exit 2
  fi
  GPU_LIST="${POSITIONAL[0]}"
  GPU_LIST_DEFAULT="${POSITIONAL[0]}"
  PROBLEM="$(normalize_problem "${POSITIONAL[1]}")"
  CUSTOMERS="${POSITIONAL[2]}"
  SEED="${POSITIONAL[3]:-${SEED:-3009}}"
fi

validate_customers "$CUSTOMERS"
if [[ "$PROBLEM" == "evrptw" ]]; then
  CS="$((CUSTOMERS / 5))"
else
  CS=0
fi

export PYTHON_BIN GPU_LIST PROBLEM CUSTOMERS SEED CS
export NUM_ENVS N_TRAJ EPOCHS PPO_STEP_CHUNK_SIZE NUM_MINIBATCHES
export EVAL_BATCH_SIZE ROLLOUT_STEPS EVAL_MAX_STEPS
set -- "$REQUESTED_STAGE" "${FLAGS[@]}"

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common_ablation.sh"

if [[ "$AB_STAGE" == "init" ]]; then
  echo "[Done] stage=init; Server9 is plain PPO from scratch and has no init checkpoint."
  exit 0
fi
if stage_runs_offline_jobs && ! stage_runs_ppo_jobs; then
  echo "[Done] stage=offline; Server9 has no non-PPO jobs."
  exit 0
fi

label="${PROBLEM}_cus${CUSTOMERS}_no_rdi_no_agda_ppo"
run_name="${BASE_RUN}_ABL_S9_NO_RDI_NO_AGDA_PPO_SEED${SEED}_E${EPOCHS}_N${NUM_ENVS}_R${ROLLOUT_STEPS}"
gpu="${GPU_LIST[0]}"

echo "[Server9] problem=${PROBLEM} customers=${CUSTOMERS} cs=${CS} seed=${SEED} gpu=${gpu}"
echo "[Server9] fixed: plain PPO, RDI=off, AGDA=off, SL-PPO=off, num_envs=${NUM_ENVS}, n_traj=${N_TRAJ}"
start_job "$gpu" "s9_${label}" \
  "${COMMON_ARGS[@]}" \
  --offline-method ppo \
  --ppo-update-epochs 3 \
  --distance-source none \
  --rdi-embedding none \
  --rdi-encoder-bias off \
  --rdi-encoder-norm softmax \
  --dde off \
  --qkv-delta none \
  --action-key off \
  --action-bias off \
  --run-name "$run_name"

wait_batch
echo "[All done] logs: ${RUN_DIR}"
