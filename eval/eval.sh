#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONPATH=$PYTHONPATH:$(pwd)

DATASET_SPECS=${DATASET_SPECS:-"VTE-Bench/Synth.csv;VTE-Bench/Real.csv;VTE-Bench/SceneText.csv"}
CSV_ROOT=${CSV_ROOT:-VTE-Bench}
INFER_ROOT=${INFER_ROOT:-outputs/bench}
RESULT_ROOT=${RESULT_ROOT:-outputs/eval}
TMP_ROOT=${TMP_ROOT:-outputs/eval_tmp}
NUM_REGIONS=${NUM_REGIONS:-1}
MIN_COMPONENT_AREA=${MIN_COMPONENT_AREA:-32}
FRAME_NUMBER=${FRAME_NUMBER:-20}
SQUARE_SIZE=${SQUARE_SIZE:-224}
USE_GPU=${USE_GPU:-1}
USE_FP16=${USE_FP16:-0}
SKIP_MISSING=${SKIP_MISSING:-1}
PREPARE_WORKERS=${PREPARE_WORKERS:-8}

if [ -z "$DATASET_SPECS" ]; then
  echo "[ERROR] DATASET_SPECS is empty" >&2
  exit 1
fi
if [ ! -d "$INFER_ROOT" ]; then
  echo "[ERROR] Inference root not found: $INFER_ROOT" >&2
  exit 1
fi

IFS=';' read -r -a datasets <<< "$DATASET_SPECS"
valid_count=0
for raw in "${datasets[@]}"; do
  csv="${raw#"${raw%%[![:space:]]*}"}"
  csv="${csv%"${csv##*[![:space:]]}"}"
  [ -z "$csv" ] && continue
  if [ ! -f "$csv" ]; then
    echo "[ERROR] CSV not found: $csv" >&2
    exit 1
  fi
  dataset_name=$(basename "$csv" .csv)
  if [ ! -d "$INFER_ROOT/$dataset_name" ]; then
    echo "[ERROR] Inference output directory not found: $INFER_ROOT/$dataset_name" >&2
    exit 1
  fi
  valid_count=$((valid_count + 1))
done
[ "$valid_count" -gt 0 ] || { echo "[ERROR] No datasets specified" >&2; exit 1; }

mkdir -p "$RESULT_ROOT" "$TMP_ROOT"

cmd=(
  python eval/eval_multi_csv.py
  --dataset_specs "$DATASET_SPECS"
  --csv_root "$CSV_ROOT"
  --infer_root "$INFER_ROOT"
  --result_root "$RESULT_ROOT"
  --tmp_root "$TMP_ROOT"
  --num_regions "$NUM_REGIONS"
  --min_component_area "$MIN_COMPONENT_AREA"
  --frame_number "$FRAME_NUMBER"
  --square_size "$SQUARE_SIZE"
  --prepare_workers "$PREPARE_WORKERS"
)

if [[ "${USE_GPU,,}" =~ ^(1|true|yes|y|on)$ ]]; then
  cmd+=(--use_gpu)
fi
if [[ "${USE_FP16,,}" =~ ^(1|true|yes|y|on)$ ]]; then
  cmd+=(--use_fp16)
fi
if [[ "${SKIP_MISSING,,}" =~ ^(1|true|yes|y|on)$ ]]; then
  cmd+=(--skip_missing)
fi

echo "DATASET_SPECS: $DATASET_SPECS"
echo "CSV_ROOT:      $CSV_ROOT"
echo "INFER_ROOT:    $INFER_ROOT"
echo "RESULT_ROOT:   $RESULT_ROOT"
echo "SKIP_MISSING:  $SKIP_MISSING"
echo "PREPARE_WORKERS: $PREPARE_WORKERS"
"${cmd[@]}"
