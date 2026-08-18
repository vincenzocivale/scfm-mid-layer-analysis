#!/usr/bin/env Rscript
# Export a (possibly double-gzipped) Seurat .RData into language-neutral
# pieces that assemble.py / prepare_*.py turn into an h5ad: raw-count matrix
# (MatrixMarket, genes x cells), gene ids, cell barcodes, and the full cell
# metadata table (which for GSE303158 already carries Mixscape perturbation
# calls in `target` / `guide` / `mixscape_class*`).
#
# GSE303158's supplementary RData files are gzip-compressed *twice* (the
# upstream .gz wraps an already gzip-compressed RData stream), which R's
# load() cannot auto-detect — so we decompress to a plain RDX3 file first.
#
# Usage: Rscript export_seurat_rdata.R <rdata_gz_path> <out_dir> [assay]
suppressMessages({library(Seurat); library(Matrix)})

args   <- commandArgs(trailingOnly = TRUE)
src    <- args[1]
outdir <- args[2]
assay  <- if (length(args) >= 3) args[3] else "RNA"
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

# Decompress until we hit the RDX magic header (handles single or double gzip).
# Ping-pong between two scratch paths so we never read from the file we're writing.
scratch <- c(file.path(outdir, "_decompressed_a.RData"),
             file.path(outdir, "_decompressed_b.RData"))
cur <- src
produced <- character(0)
for (i in 1:3) {
  magic <- readBin(cur, "raw", n = 5)
  if (identical(magic[1:4], as.raw(c(0x52, 0x44, 0x58, 0x33)))) break  # "RDX3"
  dest <- scratch[((i - 1) %% 2) + 1]
  cat("decompressing layer", i, "of", basename(cur), "->", basename(dest), "\n"); flush.console()
  con_in  <- gzfile(cur, "rb")
  con_out <- file(dest, "wb")
  while (length(chunk <- readBin(con_in, "raw", n = 1e8)) > 0) writeBin(chunk, con_out)
  close(con_in); close(con_out)
  cur <- dest
  produced <- c(produced, dest)
}

cat("loading", cur, "\n"); flush.console()
e <- new.env()
load(cur, envir = e)
for (p in unique(produced)) if (file.exists(p)) file.remove(p)

seurat_objs <- Filter(function(n) inherits(get(n, envir = e), "Seurat"), ls(e))
if (length(seurat_objs) == 0) stop("no Seurat object found in ", src)
obj <- get(seurat_objs[1], envir = e)
cat("using object '", seurat_objs[1], "' (", paste(dim(obj), collapse = " x "), ")\n", sep = "")

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
