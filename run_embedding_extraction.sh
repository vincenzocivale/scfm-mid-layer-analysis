#!/bin/bash
set -e

export PYTORCH_ALLOC_CONF=expandable_segments:True


# Separate input files from options (e.g. --no-fp16)
INPUT_FILES=()
EXTRA_ARGS=()
for arg in "$@"; do
    if [[ "$arg" == --* ]]; then
        EXTRA_ARGS+=("$arg")
    else
        INPUT_FILES+=("$arg")
    fi
done

OUTPUT_DIR="${OUTPUT_DIR:-data/embeddings}"
BATCH_SIZE="${BATCH_SIZE:-8}"
TAHOE_MODEL_SIZE="${TAHOE_MODEL_SIZE:-1b}"

if [ ${#INPUT_FILES[@]} -eq 0 ]; then
    echo "Usage: $0 <input1.h5ad> [input2.h5ad ...] [--no-fp16]"
    exit 1
fi

MODELS=(${MODELS:-tahoe})

echo "Starting analysis for ${#INPUT_FILES[@]} file(s) and ${#MODELS[@]} model(s)"
echo "========================================================================"

for INPUT_FILE in "${INPUT_FILES[@]}"; do
    INPUT_NAME=$(basename "$INPUT_FILE" .h5ad)
    
    for MODEL in "${MODELS[@]}"; do
        echo ""
        echo "========================================================================"
        echo "PROCESSING: Model '$MODEL' on '$INPUT_NAME'"
        echo "========================================================================"
        
        # Get the number of layers dynamically (filtra solo il numero)
        if [ "$MODEL" == "tahoe" ]; then
            NUM_LAYERS=$(python models/get_model_info.py --model "$MODEL" --tahoe_size "$TAHOE_MODEL_SIZE" | grep -Eo '^[0-9]+$')
        else
            NUM_LAYERS=$(python models/get_model_info.py --model "$MODEL" | grep -Eo '^[0-9]+$')
        fi

        if [ $? -ne 0 ]; then
            echo "ERROR: Could not determine number of layers for model $MODEL. Skipping."
            continue
        fi
        
        LAYERS=""
        for i in $(seq 0 $((NUM_LAYERS - 1))); do
            LAYERS="$LAYERS $i"
        done
        # Trim leading space
        LAYERS="${LAYERS# }"
        
        echo "Detected $NUM_LAYERS layers for $MODEL. Processing layers: $LAYERS"

        if [ -z "$LAYERS" ]; then
            echo "ERROR: Nessun layer trovato per il modello $MODEL. Skipping."
            continue
        fi

        if [ "$MODEL" == "tahoe" ]; then
            FINAL_OUTPUT_FILE="${OUTPUT_DIR}/${INPUT_NAME}_${MODEL}_${TAHOE_MODEL_SIZE}_embeddings.h5ad"
        else
            FINAL_OUTPUT_FILE="${OUTPUT_DIR}/${INPUT_NAME}_${MODEL}_embeddings.h5ad"
        fi

        mkdir -p "$OUTPUT_DIR"
        if [ -f "$FINAL_OUTPUT_FILE" ]; then
            echo "Output file $FINAL_OUTPUT_FILE already exists. Skipping."
            continue
        fi

        echo "--- EXTRACTING AND SAVING EMBEDDINGS to $FINAL_OUTPUT_FILE ---"
        # Esegui solo python, non interpretare output come comando
        if [ "$MODEL" == "tahoe" ]; then
            python models/extract_embeddings.py \
                --model "$MODEL" \
                --model-size "$TAHOE_MODEL_SIZE" \
                --input "$INPUT_FILE" \
                --output "$FINAL_OUTPUT_FILE" \
                --batch-size "$BATCH_SIZE" \
                --layers $LAYERS \
                "${EXTRA_ARGS[@]}"
        else
            python models/extract_embeddings.py \
                --model "$MODEL" \
                --input "$INPUT_FILE" \
                --output "$FINAL_OUTPUT_FILE" \
                --batch-size "$BATCH_SIZE" \
                --layers $LAYERS \
                "${EXTRA_ARGS[@]}"
        fi
        
        if [ $? -ne 0 ]; then
            echo "ERROR: Embedding extraction failed for $MODEL on $INPUT_NAME"
            continue
        fi
        
        # Cleanup GPU
        python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
        sleep 3
    done
done

echo ""
echo "========================================================================"
echo "COMPLETE"
echo "========================================================================"
