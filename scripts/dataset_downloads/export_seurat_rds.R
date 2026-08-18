#!/usr/bin/env Rscript
# Export a Seurat .rds into language-neutral pieces that assemble.py turns
# into an h5ad: raw-count matrix (MatrixMarket, genes x cells), gene ids,
# cell barcodes, and the full cell metadata table.
#
# Usage: Rscript export_seurat_rds.R <rds_path> <out_dir> [assay]
suppressMessages({library(Seurat); library(Matrix)})

args   <- commandArgs(trailingOnly = TRUE)
rds    <- args[1]
outdir <- args[2]
assay  <- if (length(args) >= 3) args[3] else "RNA"
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

cat("reading", rds, "\n"); flush.console()
obj <- readRDS(rds)
if (!inherits(obj, "Seurat")) stop("not a Seurat object: ", paste(class(obj), collapse=","))

# Raw counts: try the v5 `layer=` API, fall back to the v4 `slot=` API.
counts <- tryCatch(
  SeuratObject::GetAssayData(obj, assay = assay, layer = "counts"),
  error = function(e) SeuratObject::GetAssayData(obj, assay = assay, slot = "counts"))
counts <- as(counts, "CsparseMatrix")            # genes x cells, dgCMatrix
cat("counts:", nrow(counts), "genes x", ncol(counts), "cells\n")

Matrix::writeMM(counts, file.path(outdir, "counts.mtx"))
writeLines(rownames(counts), file.path(outdir, "genes.txt"))
writeLines(colnames(counts), file.path(outdir, "barcodes.txt"))
write.csv(obj@meta.data, file.path(outdir, "meta.csv"), row.names = TRUE)
cat("exported counts.mtx / genes.txt / barcodes.txt / meta.csv to", outdir, "\n")
