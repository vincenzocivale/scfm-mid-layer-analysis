"""Minimal CellFM Encoder for inference — extracted from get_gene_emb.py.

get_gene_emb.py cannot be imported directly because it contains module-level
training/evaluation code. This module re-exposes only the Encoder class.
"""
import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops.operations as P
from mindspore.common.initializer import initializer, XavierNormal

from model import ValueEncoder, ValueDecoder
from retention import RetentionLayer


class Encoder(nn.Cell):
    def __init__(self, n_genes, used_gene, cfg, shard=None):
        super().__init__()
        self.depth = cfg.enc_nlayers
        self.n_genes = n_genes
        self.gene_emb = ms.Parameter(
            initializer('normal', [n_genes + 1 + (-n_genes - 1) % 8, cfg.enc_dims])
        )
        self.zero_emb = ms.Parameter(initializer('normal', [1, 1, cfg.enc_dims]))
        self.pert_token = ms.Parameter(initializer('normal', [3, cfg.enc_dims]))
        self.pert_token[0] = 0
        self.pert_token[1] = -self.pert_token[0]
        self.pert_token[2] = 0
        self.used_gene = ms.Tensor(used_gene, ms.int32)
        self.value_enc = ValueEncoder(cfg.enc_dims)
        self.encoder = nn.CellList([
            RetentionLayer(
                cfg.enc_dims, cfg.enc_num_heads, cfg.enc_nlayers,
                cfg.enc_dropout * i / cfg.enc_nlayers, cfg.lora,
                cfg.recompute,
            )
            for i in range(cfg.enc_nlayers)
        ])
        self.value_dec = ValueDecoder(cfg.enc_dims, cfg.dropout)
        self.less = P.Less()
        self.one = P.Ones()
        self.zero = P.Zeros()
        self.tile = P.Tile()
        self.sum = P.ReduceSum(True)
        self.softmax = P.Softmax(-1)
        self.gather1 = P.Gather()
        self.gather2 = P.Gather()
        self.maskmul = P.Mul()
        self.mul = P.Mul()
        self.add = P.Add()
        self.posa = P.Add()
        self.rsqrt = P.Rsqrt()
        self.detach = P.StopGradient()

    def construct(self, expr, if_pert, pert_id, zero_mask):
        b, l = if_pert.shape
        len_scale = self.detach(self.rsqrt(self.sum(zero_mask, -1)).reshape(b, 1, 1, 1))
        gene_emb = self.gather1(self.gene_emb, self.used_gene, 0).reshape(1, l, -1)
        init_emb, unmask = self.value_enc(expr)
        expr_emb = init_emb + gene_emb
        attn_mask = None
        for i in range(self.depth):
            expr_emb = self.encoder[i](expr_emb, v_pos=len_scale, seq_mask=attn_mask)
        return expr_emb
