#!/usr/bin/env bash
set -euo pipefail

SCRIPT_TAG="server2_rdi"
GPU_LIST_DEFAULT="${GPU_LIST_DEFAULT:-0,1,2,3}"

# RDI option model:
#   RDI_OPTION=all       launches the 8 rows required by Section 5.3.
#   RDI_OPTION=base      no distance injection.
#   RDI_OPTION=euclidean injects Euclidean distance through the encoder-bias slot.
#   RDI_OPTION=graph     uses road-network distance; choose the switches below.
# For RDI_OPTION=graph, the three switches are true/false knobs:
#   RDI_EMBEDDING_SVD=true|false
#   RDI_ENCODER_SINKHORN=true|false
#   RDI_ENCODER_BIAS=true|false
RDI_OPTION="${RDI_OPTION:-all}"
RDI_EMBEDDING_SVD="${RDI_EMBEDDING_SVD:-false}"
RDI_ENCODER_SINKHORN="${RDI_ENCODER_SINKHORN:-false}"
RDI_ENCODER_BIAS="${RDI_ENCODER_BIAS:-false}"
RDI_SINKHORN_ITERS=10
RDI_SVD_RANK=10

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common_ablation.sh"

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

on_off() {
  if [[ "$1" == "true" ]]; then
    echo "on"
  else
    echo "off"
  fi
}

softmax_or_sinkhorn() {
  if [[ "$1" == "true" ]]; then
    echo "sinkhorn"
  else
    echo "softmax"
  fi
}

rdi_label_for() {
  local distance_option="$1"
  local embedding_svd="$2"
  local encoder_sinkhorn="$3"
  local encoder_bias="$4"
  case "$distance_option" in
    base|euclidean)
      echo "$distance_option"
      ;;
    graph)
      local parts=()
      [[ "$embedding_svd" == "true" ]] && parts+=("embedding_svd")
      [[ "$encoder_sinkhorn" == "true" ]] && parts+=("encoder_sinkhorn")
      [[ "$encoder_bias" == "true" ]] && parts+=("encoder_bias")
      if [[ "${#parts[@]}" -eq 0 ]]; then
        echo "[Error] RDI_OPTION=graph needs at least one graph-distance injection switch enabled." >&2
        echo "Use RDI_OPTION=base for the no-distance lower bound." >&2
        exit 2
      fi
      local joined
      local IFS=_
      joined="${parts[*]}"
      echo "$joined"
      ;;
    *)
      echo "[Error] unknown RDI_OPTION: ${distance_option}; choose all, base, euclidean, or graph" >&2
      exit 2
      ;;
  esac
}

nn_representation_for() {
  local distance_option="$1"
  local embedding_svd="$2"
  local encoder_sinkhorn="$3"
  local encoder_bias="$4"
  case "$distance_option" in
    base|euclidean)
      # With no road graph in the row, the geometric baseline is coordinate/Euclidean NN.
      echo "euclidean"
      ;;
    graph)
      if [[ "$embedding_svd" == "true" && "$encoder_sinkhorn" == "false" && "$encoder_bias" == "false" ]]; then
        echo "svd"
      else
        echo "road"
      fi
      ;;
  esac
}

add_rdi_experiment() {
  local label="$1"
  local distance_option="$2"
  local embedding_svd
  local encoder_sinkhorn
  local encoder_bias
  embedding_svd="$(normalize_bool "$3")"
  encoder_sinkhorn="$(normalize_bool "$4")"
  encoder_bias="$(normalize_bool "$5")"
  RDI_EXPERIMENTS+=("${label}|${distance_option}|${embedding_svd}|${encoder_sinkhorn}|${encoder_bias}")
}

build_rdi_experiments() {
  RDI_EXPERIMENTS=()
  local option="${RDI_OPTION,,}"
  case "$option" in
    all)
      add_rdi_experiment "base" "base" false false false
      add_rdi_experiment "euclidean" "euclidean" false false true
      add_rdi_experiment "embedding_svd_only" "graph" true false false
      add_rdi_experiment "encoder_sinkhorn_only" "graph" false true false
      add_rdi_experiment "encoder_bias_only" "graph" false false true
      add_rdi_experiment "embedding_svd_encoder_sinkhorn" "graph" true true false
      add_rdi_experiment "embedding_svd_encoder_bias" "graph" true false true
      add_rdi_experiment "embedding_svd_encoder_sinkhorn_encoder_bias" "graph" true true true
      ;;
    base)
      add_rdi_experiment "base" "base" false false false
      ;;
    euclidean)
      add_rdi_experiment "euclidean" "euclidean" false false true
      ;;
    graph)
      local embedding_svd
      local encoder_sinkhorn
      local encoder_bias
      embedding_svd="$(normalize_bool "$RDI_EMBEDDING_SVD")"
      encoder_sinkhorn="$(normalize_bool "$RDI_ENCODER_SINKHORN")"
      encoder_bias="$(normalize_bool "$RDI_ENCODER_BIAS")"
      local label
      label="$(rdi_label_for graph "$embedding_svd" "$encoder_sinkhorn" "$encoder_bias")"
      add_rdi_experiment "$label" "graph" "$embedding_svd" "$encoder_sinkhorn" "$encoder_bias"
      ;;
    *)
      echo "[Error] unknown RDI_OPTION: ${RDI_OPTION}; choose all, base, euclidean, or graph" >&2
      exit 2
      ;;
  esac
}

if [[ "$AB_STAGE" == "init" ]]; then
  echo "[Done] stage=init; Server2 RDI uses plain PPO from scratch and has no shared init."
  exit 0
fi
if stage_runs_offline_jobs && ! stage_runs_ppo_jobs; then
  echo "[Done] stage=offline; Server2 has no non-PPO jobs."
  exit 0
fi

build_rdi_experiments

NN_MATCH_CSV="$RUN_DIR/rdi_nn_match.csv"

record_nn_match() {
  local label="$1"
  local distance_option="$2"
  local embedding_svd="$3"
  local encoder_sinkhorn="$4"
  local encoder_bias="$5"
  local representation="$6"
  "$PYTHON_BIN" scripts/ablation/compute_rdi_nn_match.py \
    --dataset "$DATA_ROOT/${PROBLEM}/val/Cus${CUSTOMERS}" \
    --problem "$PROBLEM" \
    --customers "$CUSTOMERS" \
    --charging-stations "$CS" \
    --label "$label" \
    --distance-option "$distance_option" \
    --embedding-svd "$embedding_svd" \
    --encoder-sinkhorn "$encoder_sinkhorn" \
    --encoder-bias "$encoder_bias" \
    --representation "$representation" \
    --svd-rank "$RDI_SVD_RANK" \
    --output "$NN_MATCH_CSV" >"$RUN_DIR/s2_${label}.nn_match.log" 2>&1 || true
}

launch_rdi() {
  local gpu="$1"
  local label="$2"
  local distance_option="$3"
  local embedding_svd="$4"
  local encoder_sinkhorn="$5"
  local encoder_bias="$6"

  local distance_source
  local rdi_embedding
  local rdi_encoder_norm
  local rdi_encoder_bias
  local nn_representation
  case "$distance_option" in
    base)
      distance_source="none"
      ;;
    euclidean)
      distance_source="euclidean"
      ;;
    graph)
      distance_source="road"
      ;;
    *)
      echo "[Error] unknown distance option: ${distance_option}" >&2
      exit 2
      ;;
  esac
  if [[ "$embedding_svd" == "true" ]]; then
    rdi_embedding="svd"
  else
    rdi_embedding="none"
  fi
  rdi_encoder_norm="$(softmax_or_sinkhorn "$encoder_sinkhorn")"
  rdi_encoder_bias="$(on_off "$encoder_bias")"
  nn_representation="$(nn_representation_for "$distance_option" "$embedding_svd" "$encoder_sinkhorn" "$encoder_bias")"

  local run_name="${BASE_RUN}_ABL_S2_RDI_${label^^}_PPO_SEED${SEED}"
  echo "[RDI] label=${label} option=${distance_option} embedding_svd=${embedding_svd} encoder_sinkhorn=${encoder_sinkhorn} encoder_bias=${encoder_bias} nn_rep=${nn_representation}"
  record_nn_match "$label" "$distance_option" "$embedding_svd" "$encoder_sinkhorn" "$encoder_bias" "$nn_representation"
  start_job "$gpu" "s2_${label}" \
    "${COMMON_ARGS[@]}" \
    --offline-method ppo \
    --ppo-update-epochs 3 \
    --distance-source "$distance_source" \
    --rdi-embedding "$rdi_embedding" \
    --rdi-encoder-bias "$rdi_encoder_bias" \
    --rdi-encoder-norm "$rdi_encoder_norm" \
    --sinkhorn-iters "$RDI_SINKHORN_ITERS" \
    --dde off \
    --qkv-delta none \
    --action-key off \
    --action-bias off \
    --run-name "$run_name"
}

job_idx=0
for spec in "${RDI_EXPERIMENTS[@]}"; do
  IFS='|' read -r label distance_option embedding_svd encoder_sinkhorn encoder_bias <<<"$spec"
  gpu="${GPU_LIST[$((job_idx % ${#GPU_LIST[@]}))]}"
  launch_rdi "$gpu" "$label" "$distance_option" "$embedding_svd" "$encoder_sinkhorn" "$encoder_bias"
  job_idx=$((job_idx + 1))
  if (( job_idx % ${#GPU_LIST[@]} == 0 )); then
    wait_batch
  fi
done

wait_batch
echo "[All done] logs: ${RUN_DIR}"
