#!/bin/bash
set -e

# --------------------------------------------------
# PyTorch memory settings
# --------------------------------------------------
export PYTORCH_ALLOC_CONF=expandable_segments:True

# --------------------------------------------------
# Separate input files from options (e.g. --no-fp16)
# --------------------------------------------------
INPUT_FILES=()
EXTRA_ARGS=()

for arg in "$@"; do
    if [[ "$arg" == --* ]]; then
        EXTRA_ARGS+=("$arg")
    else
        INPUT_FILES+=("$arg")
    fi
done

# --------------------------------------------------
# Config (overridable via env)
# --------------------------------------------------
OUTPUT_DIR="${OUTPUT_DIR:-data/embeddings}"
BATCH_SIZE="${BATCH_SIZE:-1}"
CHUNK_SIZE="${CHUNK_SIZE:-20000}"
TAHOE_MODEL_SIZE="${TAHOE_MODEL_SIZE:-1b}"
MODELS=(${MODELS:-scfoundation})

# --------------------------------------------------
# Sanity check
# --------------------------------------------------
if [ ${#INPUT_FILES[@]} -eq 0 ]; then
    echo "Usage: $0 <input1.h5ad> [input2.h5ad ...] [--no-fp16]"
    exit 1
fi

echo "Starting CHUNKED analysis"
echo "Inputs      : ${#INPUT_FILES[@]}"
echo "Models      : ${MODELS[*]}"
echo "Chunk size  : $CHUNK_SIZE"
echo "Batch size  : $BATCH_SIZE"
echo "Output dir  : $OUTPUT_DIR"
echo "========================================================================"

# --------------------------------------------------
# Main loop
# --------------------------------------------------
for INPUT_FILE in "${INPUT_FILES[@]}"; do
    INPUT_NAME=$(basename "$INPUT_FILE" .h5ad)

    for MODEL in "${MODELS[@]}"; do
        echo ""
        echo "========================================================================"
        echo "PROCESSING (chunked): Model '$MODEL' on '$INPUT_NAME'"
        echo "========================================================================"

        # --------------------------------------------------
        # Detect number of layers
        # --------------------------------------------------
        if [ "$MODEL" == "tahoe" ]; then
            NUM_LAYERS=$(python models/get_model_info.py \
                --model "$MODEL" \
                --tahoe_size "$TAHOE_MODEL_SIZE" | grep -Eo '^[0-9]+$')
        else
            NUM_LAYERS=$(python models/get_model_info.py \
                --model "$MODEL" | grep -Eo '^[0-9]+$')
        fi

        if [ -z "$NUM_LAYERS" ]; then
            echo "ERROR: Could not determine number of layers for $MODEL"
            continue
        fi

        LAYERS=$(seq 0 $((NUM_LAYERS - 1)) | tr '\n' ' ')
        echo "Detected $NUM_LAYERS layers: $LAYERS"

        # --------------------------------------------------
        # Output directory (per model + input)
        # --------------------------------------------------
        if [ "$MODEL" == "tahoe" ]; then
            MODEL_OUT_DIR="${OUTPUT_DIR}/${INPUT_NAME}_${MODEL}_${TAHOE_MODEL_SIZE}"
        else
            MODEL_OUT_DIR="${OUTPUT_DIR}/${INPUT_NAME}_${MODEL}"
        fi

        mkdir -p "$MODEL_OUT_DIR"

        # --------------------------------------------------
        # Run chunked extraction
        # --------------------------------------------------
        echo "--- RUNNING CHUNKED EMBEDDING EXTRACTION ---"

        python models/extract_embeddings_chunked_all_layers.py \
            --model "$MODEL" \
            --model-size "$TAHOE_MODEL_SIZE" \
            --input "$INPUT_FILE" \
            --output-dir "$MODEL_OUT_DIR" \
            --batch-size "$BATCH_SIZE" \
            --chunk-size "$CHUNK_SIZE" \
            --layers $LAYERS \
            "${EXTRA_ARGS[@]}"

        # --------------------------------------------------
        # Cleanup GPU
        # --------------------------------------------------
        python - <<EOF
import torch, gc
torch.cuda.empty_cache()
gc.collect()
EOF

        sleep 3
    done
done

echo ""
echo "========================================================================"
echo "CHUNKED ANALYSIS COMPLETE ✅"
echo "========================================================================"
