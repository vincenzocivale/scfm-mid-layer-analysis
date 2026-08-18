#!/bin/bash
# Extract all available per-layer embeddings for the benchmark-ready
# GSE277032 organoid pseudotime dataset. CellFM is intentionally skipped.
#
# Launch:
#   nohup bash run_gse277032_pseudotime_extraction.sh > logs/pseudotime_gse277032_extraction.log 2>&1 &

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INPUT="$REPO_ROOT/data/raw/pseudotime/GSE277032_organoids.h5ad"
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
    echo "  INPUT: $INPUT"
    echo "  START: $(date)"
    echo "========================================================"

    MODELS="$model" \
    BATCH_SIZE="$batch_size" \
    CHUNK_SIZE="$chunk_size" \
    bash "$REPO_ROOT/run_embedding_extraction.sh" "$INPUT" $extra_args

    echo "  DONE:  $(date)"
}

run_model scfoundation 8 50000
run_model tahoe 16 50000
run_model scgpt 32 50000
run_model uce 16 50000
run_model genecompass 32 50000

echo ""
echo "========================================================"
echo "  GSE277032 extraction completed: $(date)"
echo "========================================================"
