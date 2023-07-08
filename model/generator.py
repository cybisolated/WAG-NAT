import torch
from torch import nn

from .sub_models import *
from .embeddings import LearnablePositionEmbedding, SinusoidalPositionEmbedding


class QueryGenerator(nn.Module):
    def __init__(self,
                 L_out,
                 d_model,
                 num_heads,
                 dropout):
        super().__init__()

        # self.pos_table = LearnablePositionEmbedding(L_out, d_model)
        self.pos_table = SinusoidalPositionEmbedding(d_model)

        self.attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.add_norm = AddNorm(d_model, dropout)
        self.query_attn = MultiHeadAttention(d_model, num_heads, dropout)

    def forward(self, gen_input, enc_input, enc_out, mask=None):
        pos_vec = self.pos_table(gen_input).expand_as(gen_input)

        y, attn_map = self.attn(pos_vec, gen_input, gen_input)
        y = self.add_norm(gen_input, y)

        queries, query_attn_map = self.query_attn(y, enc_out, enc_input, mask)

        return queries
