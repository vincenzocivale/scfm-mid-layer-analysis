
#!/bin/bash
# Pipeline completa per i dataset tabula: extraction → merge → classification.
# Esegue sequenzialmente scGPT → GeneCompass → Tahoe (su blood e immune).
# scFoundation è già in corso / già completato — viene saltato.
#
# Uso: bash run_tabula_pipeline.sh [--skip-immune]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
PYTHON="./.venv/bin/python"

BLOOD="data/raw/classification/tabula/tabula-blood.h5ad"
IMMUNE="data/raw/classification/tabula/tabula-immune.h5ad"
EMBS="data/embeddings"

SKIP_IMMUNE=0
[[ "${1:-}" == "--skip-immune" ]] && SKIP_IMMUNE=1

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ─── Helpers ──────────────────────────────────────────────────────────────────

run_extraction() {
    local model="$1" dataset="$2" batch_size="$3" chunk_size="$4"
    local stem; stem=$(basename "$dataset" .h5ad)
    local out_dir="${EMBS}/${stem}_${model}"
    [[ "$model" == "tahoe" ]] && out_dir="${EMBS}/${stem}_${model}_1b"

    # Count already-written chunks to decide if we need to run at all.
    local n_done=0
    [[ -d "$out_dir" ]] && n_done=$(find "$out_dir" -maxdepth 1 -name "*_chunk_*.h5ad" 2>/dev/null | wc -l)

    local n_cells; n_cells=$($PYTHON -c "
import scanpy as sc, sys
a = sc.read_h5ad('$dataset', backed='r'); print(a.n_obs)" 2>/dev/null)
    local n_chunks=$(( (n_cells - 1) / chunk_size + 1 ))

    if (( n_done >= n_chunks )); then
        log "SKIP $model on $stem — $n_done/$n_chunks chunks already present"
        return 0
    fi

    log "START $model on $stem — $n_done/$n_chunks chunks done, batch=$batch_size chunk=$chunk_size"
    MODELS="$model" BATCH_SIZE="$batch_size" CHUNK_SIZE="$chunk_size" \
        bash run_embedding_extraction.sh "$dataset"
    log "DONE  $model on $stem"
}

merge_and_classify() {
    local model="$1" dataset="$2"
    local stem; stem=$(basename "$dataset" .h5ad)
    local suffix="${stem}_${model}"
    [[ "$model" == "tahoe" ]] && suffix="${stem}_${model}_1b"
    local emb_dir="${EMBS}/${suffix}"
    local merged="${emb_dir}/${suffix}.h5ad"

    # Verify all expected chunks are present before merging.
    local n_cells; n_cells=$($PYTHON -c "
import scanpy as sc; a = sc.read_h5ad('$dataset', backed='r'); print(a.n_obs)" 2>/dev/null)
    local chunk_size=50000
    local n_expected=$(( (n_cells - 1) / chunk_size + 1 ))
    local n_actual=0
    [[ -d "$emb_dir" ]] && n_actual=$(ls "$emb_dir"/*_chunk_*.h5ad 2>/dev/null | wc -l)

    if (( n_actual < n_expected )); then
        log "SKIP merge $suffix — only $n_actual/$n_expected chunks ready (extraction still running?)"
        return 0
    fi

    if [[ ! -f "$merged" ]]; then
        log "MERGE $suffix"
        $PYTHON scripts/merge_chunks.py --input-dir "$emb_dir"
    else
        log "SKIP merge $suffix — already merged"
    fi

    local result_path="data/classification_results/${suffix}_classification_results.csv"
    if [[ -f "$result_path" ]]; then
        log "SKIP classify $suffix — results exist"
        return 0
    fi

    log "CLASSIFY $suffix"
    $PYTHON scripts/run_cell_type_classification.py \
        --input "$merged" \
        --cell_type_column cell_type
    log "DONE classify $suffix"
}

# ─── Stage 1: Extraction ──────────────────────────────────────────────────────

# tabula-blood
run_extraction scgpt  "$BLOOD"  32 50000
run_extraction genecompass "$BLOOD" 32 50000
run_extraction tahoe  "$BLOOD"  16 50000

# tabula-immune (potenzialmente molto lungo — 592k celle)
if (( SKIP_IMMUNE == 0 )); then
    run_extraction scgpt  "$IMMUNE" 32 50000
    run_extraction genecompass "$IMMUNE" 32 50000
    run_extraction tahoe  "$IMMUNE" 16 50000
else
    log "SKIP immune (--skip-immune passato)"
fi

# ─── Stage 2: Merge + Classification ─────────────────────────────────────────

DATASETS=("$BLOOD")
(( SKIP_IMMUNE == 0 )) && DATASETS+=("$IMMUNE")

for ds in "${DATASETS[@]}"; do
    for model in scfoundation scgpt genecompass tahoe; do
        merge_and_classify "$model" "$ds"
    done
done

log "=== PIPELINE COMPLETATA ==="
