#!/bin/bash
#
# Stage-1 orchestrator: chunked, all-layer embedding extraction for one or
# more input h5ad files.
#
# Configuration (env vars, all optional):
#   MODELS              space-separated list of model keys     (default: scfoundation)
#   TAHOE_MODEL_SIZE    size variant for Tahoe (70m/1b/3b)     (default: 1b)
#   CHUNK_SIZE          cells per chunk                        (default: 50000)
#   BATCH_SIZE          forward-pass batch size                (default: 1)
#   OUTPUT_DIR          embeddings root                        (default: data/embeddings)
#
# Usage:
#   ./run_embedding_extraction.sh <input1.h5ad> [input2.h5ad ...] [--no-fp16]
#
set -euo pipefail
export PYTORCH_ALLOC_CONF=expandable_segments:True

# Make `scfm_eval` importable without requiring `pip install -e .`.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

INPUT_FILES=()
EXTRA_ARGS=()
for arg in "$@"; do
    if [[ "$arg" == --* ]]; then EXTRA_ARGS+=("$arg"); else INPUT_FILES+=("$arg"); fi
done

if [ ${#INPUT_FILES[@]} -eq 0 ]; then
    echo "Usage: $0 <input1.h5ad> [input2.h5ad ...] [--no-fp16]"
    exit 1
fi

OUTPUT_DIR="${OUTPUT_DIR:-data/embeddings}"
BATCH_SIZE="${BATCH_SIZE:-1}"
CHUNK_SIZE="${CHUNK_SIZE:-50000}"
TAHOE_MODEL_SIZE="${TAHOE_MODEL_SIZE:-1b}"
MODELS=(${MODELS:-scfoundation})

# Default Python from the main venv; models with dedicated envs override below.
PYTHON="${PYTHON:-./.venv/bin/python}"
# Python for CellFM (MindSpore env). Override via CELLFM_PYTHON env var.
CELLFM_PYTHON="${CELLFM_PYTHON:-$(conda run -n cellfm-env which python 2>/dev/null || echo '')}"

echo "Models:     ${MODELS[*]}"
echo "Inputs:     ${#INPUT_FILES[@]}"
echo "Chunk size: $CHUNK_SIZE   Batch size: $BATCH_SIZE   Output: $OUTPUT_DIR"
echo "========================================================================"

for INPUT_FILE in "${INPUT_FILES[@]}"; do
    INPUT_NAME=$(basename "$INPUT_FILE" .h5ad)
    for MODEL in "${MODELS[@]}"; do
        if [ "$MODEL" == "tahoe" ]; then
            MODEL_OUT_DIR="${OUTPUT_DIR}/${INPUT_NAME}_${MODEL}_${TAHOE_MODEL_SIZE}"
            SIZE_ARG=(--model-size "$TAHOE_MODEL_SIZE")
        else
            MODEL_OUT_DIR="${OUTPUT_DIR}/${INPUT_NAME}_${MODEL}"
            SIZE_ARG=()
        fi
        mkdir -p "$MODEL_OUT_DIR"

        # Select Python interpreter: CellFM needs its MindSpore env.
        if [ "$MODEL" == "cellfm" ]; then
            if [ -z "$CELLFM_PYTHON" ]; then
                echo "ERROR: cellfm-env not found. Create it first:"
                echo "  conda env create -f envs/cellfm-env.yml && pip install -e ."
                echo "Or set CELLFM_PYTHON=/path/to/cellfm-env/bin/python"
                exit 1
            fi
            RUN_PYTHON="$CELLFM_PYTHON"
        else
            RUN_PYTHON="$PYTHON"
        fi

        echo ""
        echo "── $MODEL on $INPUT_NAME (python: $RUN_PYTHON) ──"
        PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
        "$RUN_PYTHON" -m scfm_eval.extraction.chunked \
            --model "$MODEL" \
            "${SIZE_ARG[@]}" \
            --input "$INPUT_FILE" \
            --output-dir "$MODEL_OUT_DIR" \
            --batch-size "$BATCH_SIZE" \
            --chunk-size "$CHUNK_SIZE" \
            "${EXTRA_ARGS[@]}"

        # Free GPU memory between (input, model) pairs (torch optional).
        "$RUN_PYTHON" - <<'EOF' 2>/dev/null || true
import gc
try:
    import torch; torch.cuda.empty_cache()
except ImportError:
    pass
gc.collect()
EOF
    done
done

echo ""
echo "========================================================================"
echo "Extraction complete."
