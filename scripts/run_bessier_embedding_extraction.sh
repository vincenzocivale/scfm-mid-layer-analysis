#!/usr/bin/env bash
set -u

cd "$(dirname "${BASH_SOURCE[0]}")/.."

mkdir -p logs/embedding_bessier

MODELS_TO_RUN=(scfoundation tahoe scgpt uce genecompass)
INPUTS=(
  data/raw/pseudotime/bessier_et_al/bro_timecourse_fm_ready.h5ad
  data/raw/pseudotime/bessier_et_al/dmgo_fm_ready.h5ad
)

for input in "${INPUTS[@]}"; do
  stem="$(basename "$input" .h5ad)"
  for model in "${MODELS_TO_RUN[@]}"; do
    log="logs/embedding_bessier/${stem}_${model}.log"
    echo "[$(date -Is)] START model=${model} input=${input}" | tee -a "$log"
    MODELS="$model" BATCH_SIZE="${BATCH_SIZE:-32}" CHUNK_SIZE="${CHUNK_SIZE:-5000}" \
      bash run_embedding_extraction.sh "$input" >> "$log" 2>&1
    status=$?
    echo "[$(date -Is)] END model=${model} input=${input} status=${status}" | tee -a "$log"
  done
done
