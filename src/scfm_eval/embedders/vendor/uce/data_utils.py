"""UCE data utilities — vendored subset of UCE/data_proc/data_utils.py (MIT License).

Only the functions needed for in-memory inference are retained:
  - SPECIES_PE_FILENAMES: species → protein-embedding filename mapping.
  - load_species_pe_genes: returns ordered list of gene symbols for a species.
  - get_spec_chrom_csv: loads the gene→chromosome position CSV.
"""
from pathlib import Path
from typing import List

import pandas as pd
import torch

# Per-species protein-embedding file names (from UCE/data_proc/gene_embeddings.py)
SPECIES_PE_FILENAMES = {
    "human":              "Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt",
    "mouse":              "Mus_musculus.GRCm39.gene_symbol_to_embedding_ESM2.pt",
    "frog":               "Xenopus_tropicalis.Xenopus_tropicalis_v9.1.gene_symbol_to_embedding_ESM2.pt",
    "zebrafish":          "Danio_rerio.GRCz11.gene_symbol_to_embedding_ESM2.pt",
    "mouse_lemur":        "Microcebus_murinus.Mmur_3.0.gene_symbol_to_embedding_ESM2.pt",
    "pig":                "Sus_scrofa.Sscrofa11.1.gene_symbol_to_embedding_ESM2.pt",
    "macaca_fascicularis": "Macaca_fascicularis.Macaca_fascicularis_6.0.gene_symbol_to_embedding_ESM2.pt",
    "macaca_mulatta":     "Macaca_mulatta.Mmul_10.gene_symbol_to_embedding_ESM2.pt",
}


def load_species_pe_genes(protein_embeddings_dir: Path, species: str) -> List[str]:
    """Return the ordered list of gene symbols in the UCE vocabulary for *species*.

    The order matches the row ordering in `all_tokens.torch` for that species
    (offset-shifted). Gene symbols are returned as-is from the file (typically mixed
    case); the caller should uppercase before comparisons.
    """
    if species not in SPECIES_PE_FILENAMES:
        # Try reading the extra-species CSV that ships with UCE model files
        extra_csv = Path(protein_embeddings_dir).parent / "new_species_protein_embeddings.csv"
        if extra_csv.exists():
            extra = pd.read_csv(str(extra_csv)).set_index("species")["path"].to_dict()
            if species in extra:
                pt_path = Path(protein_embeddings_dir) / extra[species]
            else:
                raise ValueError(
                    f"Species {species!r} not recognised. "
                    f"Known species: {sorted(SPECIES_PE_FILENAMES)}. "
                    "Add it to SPECIES_PE_FILENAMES or the extra-species CSV."
                )
        else:
            raise ValueError(
                f"Species {species!r} not recognised. Known: {sorted(SPECIES_PE_FILENAMES)}."
            )
    else:
        pt_path = Path(protein_embeddings_dir) / SPECIES_PE_FILENAMES[species]

    if not pt_path.exists():
        raise FileNotFoundError(
            f"Protein embedding file not found: {pt_path}\n"
            "Run UCEEmbedder with auto_download=True (default) to fetch model files."
        )
    pe_dict = torch.load(str(pt_path), map_location="cpu")
    return list(pe_dict.keys())  # pe_dict GC'd after return


def get_spec_chrom_csv(path: str) -> pd.DataFrame:
    """Load the gene-to-chromosome-position CSV and add a combined 'spec_chrom' Categorical.

    Returns a DataFrame with columns: species, gene_symbol, chromosome, start, spec_chrom.
    The Categorical codes in spec_chrom correspond to the chromosome token indices used
    in UCE cell sentences (offset by CHROM_TOKEN_OFFSET = 143574).
    """
    df = pd.read_csv(path)
    df["spec_chrom"] = pd.Categorical(df["species"] + "_" + df["chromosome"])
    return df
