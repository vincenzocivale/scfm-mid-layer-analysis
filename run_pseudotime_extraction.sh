#!/bin/bash
# Estrazione embedding stage-1 per tutti i modelli sul dataset inner_ear (pseudotime).
# Lancia in sequenza: scfoundation → tahoe → scgpt → cellfm → uce → genecompass
# I chunk già presenti vengono saltati automaticamente.
#
# Avvio: nohup bash run_pseudotime_extraction.sh > logs/pseudotime_extraction.log 2>&1 &

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INPUT="$REPO_ROOT/data/raw/pseudotime/inner_ear_development.h5ad"
LOG_DIR="$REPO_ROOT/logs"
mkdir -p "$LOG_DIR"

run_model() {
    local model="$1"
    local batch_size="$2"
    local chunk_size="$3"
    local extra_args="${4:-}"

    echo ""
    echo "========================================================"
    echo "  MODEL: $model   batch_size=$batch_size   chunk_size=$chunk_size"
    echo "  START: $(date)"
    echo "========================================================"

    MODELS="$model" \
    BATCH_SIZE="$batch_size" \
    CHUNK_SIZE="$chunk_size" \
    bash "$REPO_ROOT/run_embedding_extraction.sh" "$INPUT" $extra_args

    echo "  DONE:  $(date)"
}

# scfoundation — batch 8 (chunk 0 già presente, salterà automaticamente)
run_model scfoundation 8 50000

# tahoe-1b
run_model tahoe 16 50000

# scgpt
run_model scgpt 32 50000

# cellfm (MindSpore, fp16 non usato ma compatibile)
run_model cellfm 4 20000 "--no-fp16"

# uce
run_model uce 16 50000

# genecompass
run_model genecompass 32 50000

echo ""
echo "========================================================"
echo "  Estrazione completata: $(date)"
echo "========================================================"
