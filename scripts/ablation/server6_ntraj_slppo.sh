#!/usr/bin/env bash
set -euo pipefail

SCRIPT_TAG="server6_ntraj"
GPU_LIST_DEFAULT="${GPU_LIST_DEFAULT:-0}"

print_usage() {
  cat <<EOF
Usage:
  bash scripts/ablation/server6_ntraj_slppo.sh [N_TRAJ] [--detach|--foreground|--no-detach]
  bash scripts/ablation/server6_ntraj_slppo.sh GPU_ID N_TRAJ [--detach|--foreground|--no-detach]

Examples:
  bash scripts/ablation/server6_ntraj_slppo.sh 100
  bash scripts/ablation/server6_ntraj_slppo.sh 0 5
  bash scripts/ablation/server6_ntraj_slppo.sh 1 15

Fixed: full architecture, SL-PPO, expert budget=7200s, incumbent=on.
Eval N_TRAJ is forced to 50; only the training N_TRAJ changes.
Run N_TRAJ=100 first on 2080Ti because it is the OOM-risk cell.
EOF
}

validate_ntraj() {
  case "$1" in
    ''|*[!0-9]*)
      echo "[Error] N_TRAJ must be a positive integer; got: $1" >&2
      exit 2
      ;;
    0)
      echo "[Error] N_TRAJ must be positive." >&2
      exit 2
      ;;
  esac
}

SERVER6_SINGLE_RUN="${SERVER6_SINGLE_RUN:-0}"
SERVER6_N_TRAJ="${SERVER6_N_TRAJ:-${N_TRAJ:-50}}"

if [[ $# -gt 0 ]]; then
  case "$1" in
    -h|--help|help)
      print_usage
      exit 0
      ;;
    init|ppo|offline|all|--detach|--foreground|--no-detach|--)
      ;;
    *)
      if [[ $# -eq 1 || ( $# -gt 1 && "$2" == --* ) ]]; then
        SERVER6_N_TRAJ="$1"
        shift
      else
        GPU_LIST="$1"
        GPU_LIST_DEFAULT="$1"
        SERVER6_N_TRAJ="$2"
        shift 2
      fi
      validate_ntraj "$SERVER6_N_TRAJ"
      SERVER6_SINGLE_RUN=1
      export GPU_LIST SERVER6_SINGLE_RUN SERVER6_N_TRAJ
      set -- ppo "$@"
      ;;
  esac
else
  SERVER6_SINGLE_RUN=1
  export SERVER6_SINGLE_RUN SERVER6_N_TRAJ
  set -- ppo
fi

AB_REQUIRE_INIT_CKPT=1
export AB_REQUIRE_INIT_CKPT
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common_ablation.sh"

if [[ "$SERVER6_SINGLE_RUN" != "1" ]]; then
  echo "[Error] Server6 expects positional input: [GPU_ID] N_TRAJ" >&2
  print_usage >&2
  exit 2
fi

validate_ntraj "$SERVER6_N_TRAJ"
EXPERT_TRACE="${EXPERT_TRACE:-$DATA_ROOT/${PROBLEM}/train/Cus${CUSTOMERS}/gurobi_time_trace.csv}"
label="ntraj${SERVER6_N_TRAJ}"
run_name="${BASE_RUN}_ABL_S6_NTRAJ${SERVER6_N_TRAJ}_SLPPO_SEED${SEED}"

echo "[Ready] PPO init checkpoint: ${INIT_CKPT}"
echo "[Server6] train_n_traj=${SERVER6_N_TRAJ} eval_n_traj=50 gpu=${GPU_LIST[0]}"
start_job "${GPU_LIST[0]}" "s6_${label}" \
  "${COMMON_ARGS[@]}" \
  --n-traj "$SERVER6_N_TRAJ" \
  --eval-n-traj 50 \
  --offline-method slppo \
  --pool weighted \
  --ppo-update-epochs 4 \
  --init-checkpoint "$INIT_CKPT" \
  --expert-time-trace "$EXPERT_TRACE" \
  --expert-checkpoint-s 7200 \
  --memory-incumbent on \
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
