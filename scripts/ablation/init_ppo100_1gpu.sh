#!/usr/bin/env bash
set -euo pipefail

SCRIPT_TAG="init_ppo100"
GPU_LIST_DEFAULT="${GPU_LIST_DEFAULT:-0}"

print_usage() {
  cat <<EOF
Usage:
  bash scripts/ablation/init_ppo100_1gpu.sh [GPU_ID] [--detach|--foreground|--no-detach]

Examples:
  bash scripts/ablation/init_ppo100_1gpu.sh 0
  bash scripts/ablation/init_ppo100_1gpu.sh 0 --foreground

Trains or verifies the seed-matched PPO epoch-100 initial checkpoint used by
SL-PPO and init-checkpoint PPO ablations.
EOF
}

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
      export GPU_LIST
      shift
      set -- init "$@"
      ;;
  esac
else
  set -- init
fi

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common_ablation.sh"

run_init_ppo "${GPU_LIST[0]}"
echo "[Done] PPO init checkpoint: ${INIT_CKPT}"
