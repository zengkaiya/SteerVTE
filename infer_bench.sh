#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
export PYTHONPATH=$PYTHONPATH:$(pwd)

WEIGHT_PATH=${WEIGHT_PATH:-models/SteerVTE/SteerVTE.safetensors}
DATASET_SPECS=${DATASET_SPECS:-demo_data/infer.csv}
CSV_ROOT=${CSV_ROOT:-VTE-Bench}
INFER_ROOT=${INFER_ROOT:-outputs/bench}
NUM_GPUS=${NUM_GPUS:-1}
BASE_GPU=${BASE_GPU:-0}
NUM_FRAMES=${NUM_FRAMES:-49}
NUM_INFERENCE_STEPS=${NUM_INFERENCE_STEPS:-20}
SEED=${SEED:-1}
VACE_PREPROCESS_TYPE=${VACE_PREPROCESS_TYPE:-remain_all}
STYLE_ENCODER_TYPE=${STYLE_ENCODER_TYPE:-qwen_vlm}
PROJ_TYPE=${PROJ_TYPE:-linear}

is_integer() { [[ "${1:-}" =~ ^[0-9]+$ ]]; }
if ! is_integer "$NUM_GPUS" || [ "$NUM_GPUS" -le 0 ]; then
  echo "[ERROR] NUM_GPUS must be a positive integer: $NUM_GPUS" >&2
  exit 1
fi
if ! is_integer "$BASE_GPU"; then
  echo "[ERROR] BASE_GPU must be a non-negative integer: $BASE_GPU" >&2
  exit 1
fi
if [ ! -f "$WEIGHT_PATH" ]; then
  echo "[ERROR] Weight not found: $WEIGHT_PATH" >&2
  exit 1
fi

csv_rows() {
  python -c 'import csv, sys; print(max(0, sum(1 for _ in csv.reader(open(sys.argv[1], newline="", encoding="utf-8"))) - 1))' "$1"
}

datasets=()
total_samples=0
IFS=';' read -r -a raw_specs <<< "$DATASET_SPECS"
for raw in "${raw_specs[@]}"; do
  spec="${raw#"${raw%%[![:space:]]*}"}"
  spec="${spec%"${spec##*[![:space:]]}"}"
  [ -z "$spec" ] && continue
  IFS='|' read -r csv total offset extra <<< "$spec"
  total=${total:-all}
  offset=${offset:-0}
  if [ -n "${extra:-}" ] || [ -z "$csv" ] || [ ! -f "$csv" ]; then
    echo "[ERROR] Invalid or missing DATASET_SPECS item: $spec" >&2
    exit 1
  fi
  [[ "${total,,}" == "all" ]] && total=$(csv_rows "$csv")
  if ! is_integer "$total" || ! is_integer "$offset"; then
    echo "[ERROR] total and offset must be integers (or total=all): $spec" >&2
    exit 1
  fi
  datasets+=("$csv|$total|$offset")
  total_samples=$((total_samples + total))
done
[ "${#datasets[@]}" -gt 0 ] || { echo "[ERROR] No datasets specified" >&2; exit 1; }

active_gpus=$NUM_GPUS
[ "$total_samples" -lt "$active_gpus" ] && active_gpus=$total_samples
[ "$active_gpus" -gt 0 ] || { echo "[WARN] No rows to infer"; exit 0; }
worker_specs=()
for ((worker=0; worker<active_gpus; worker++)); do worker_specs+=(""); done

# Split each CSV independently. This keeps one CSV's row ranges contiguous and
# gives every GPU the same share of every dataset.
for dataset in "${datasets[@]}"; do
  IFS='|' read -r csv total offset <<< "$dataset"
  for ((worker=0; worker<active_gpus; worker++)); do
    start=$((offset + (total * worker) / active_gpus))
    end=$((offset + (total * (worker + 1)) / active_gpus))
    if [ "$start" -lt "$end" ]; then
      item="$csv|$start|$end"
      [ -n "${worker_specs[$worker]}" ] && worker_specs[$worker]+=";"
      worker_specs[$worker]+="$item"
    fi
  done
done

mkdir -p "$INFER_ROOT"
pids=()
cleanup_workers() {
  trap - INT TERM EXIT
  if [ "${#pids[@]}" -gt 0 ]; then
    kill -TERM "${pids[@]}" 2>/dev/null || true
    wait "${pids[@]}" 2>/dev/null || true
  fi
}
trap cleanup_workers INT TERM EXIT

for ((worker=0; worker<active_gpus; worker++)); do
  gpu=$((BASE_GPU + worker))
  echo "[GPU $gpu] specs=${worker_specs[$worker]}"
  CUDA_VISIBLE_DEVICES="$gpu" python infer_bench.py \
    --weight_path "$WEIGHT_PATH" \
    --dataset_specs "${worker_specs[$worker]}" \
    --csv_root "$CSV_ROOT" \
    --infer_root "$INFER_ROOT" \
    --vace_preprocess_type "$VACE_PREPROCESS_TYPE" \
    --style_encoder_type "$STYLE_ENCODER_TYPE" \
    --proj_type "$PROJ_TYPE" \
    --num_frames "$NUM_FRAMES" \
    --num_inference_steps "$NUM_INFERENCE_STEPS" \
    --seed "$SEED" \
    --gpu_label "$gpu" &
  pids+=("$!")
done
for pid in "${pids[@]}"; do
  wait "$pid"
done
echo "Inference finished: $INFER_ROOT"
