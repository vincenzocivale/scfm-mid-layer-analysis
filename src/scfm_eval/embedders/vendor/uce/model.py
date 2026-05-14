"""UCE TransformerModel — vendored from UCE/model.py (MIT License).

Changes vs. upstream:
  - Removed sys.path.append('../') hack.
  - No other logic changes.
"""
import math
import warnings

import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer

warnings.filterwarnings("ignore")


def full_block(in_features, out_features, p_drop=0.1):
    return nn.Sequential(
        nn.Linear(in_features, out_features, bias=True),
        nn.LayerNorm(out_features),
        nn.GELU(),
        nn.Dropout(p=p_drop),
    )


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1536):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class TransformerModel(nn.Module):
    """UCE cell-sentence transformer.

    Input  (forward): src [seq_len, batch, token_dim], mask [batch, seq_len]
    Output (forward): (gene_output [seq_len, batch, output_dim],
                        embedding   [batch, output_dim])  ← L2-normalised CLS
    """

    def __init__(
        self,
        token_dim: int,
        d_model: int,
        nhead: int,
        d_hid: int,
        nlayers: int,
        output_dim: int,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.model_type = "Transformer"
        self.d_model = d_model
        self.dropout = dropout

        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.encoder = nn.Sequential(
            nn.Linear(token_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.decoder = nn.Sequential(
            full_block(d_model, 1024, dropout),
            full_block(1024, output_dim, dropout),
            full_block(output_dim, output_dim, dropout),
            nn.Linear(output_dim, output_dim),
        )
        self.binary_decoder = nn.Sequential(
            full_block(output_dim + 1280, 2048, dropout),
            full_block(2048, 512, dropout),
            full_block(512, 128, dropout),
            nn.Linear(128, 1),
        )
        self.gene_embedding_layer = nn.Sequential(
            nn.Linear(token_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.pe_embedding = None

    def forward(self, src: Tensor, mask: Tensor):
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_key_padding_mask=(1 - mask))
        gene_output = self.decoder(output)
        embedding = gene_output[0, :, :]
        embedding = nn.functional.normalize(embedding, dim=1)
        return gene_output, embedding

    def predict(self, cell_embedding, gene_embeddings):
        gene_embeddings = self.gene_embedding_layer(gene_embeddings)
        return self.binary_decoder(torch.hstack((cell_embedding, gene_embeddings)))
