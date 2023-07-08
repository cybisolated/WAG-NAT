import torch
from torch import nn

from .sub_models import *
from .embeddings import SinusoidalPositionEmbedding


class Decoder(nn.Module):
    def __init__(self,
                 d_model,
                 num_heads,
                 num_layers,
                 ffn_hidden_size,
                 activation,
                 dropout):
        super().__init__()

        self.dec_layers = nn.ModuleList()
        self.dec_layers.extend([
            DecoderLayer(d_model, num_heads, ffn_hidden_size, activation, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, encoder_out, src_mask=None, tgt_mask=None):
        for dec_layer in self.dec_layers:
            x = dec_layer(x, encoder_out, src_mask, tgt_mask)
        return x


class DecoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 num_heads,
                 ffn_hidden_size,
                 activation,
                 dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.add_norm1 = AddNorm(d_model, dropout)

        self.pos_embed = SinusoidalPositionEmbedding(d_model)
        self.pos_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.add_norm2 = AddNorm(d_model, dropout)

        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.add_norm3 = AddNorm(d_model, dropout)

        self.ffn = PositionWiseFFN(d_model, ffn_hidden_size, activation, dropout)
        self.add_norm4 = AddNorm(d_model, dropout)

        self.self_attn_logits = None
        self.cross_attn_logits = None

    def forward(self, x, encoder_out, src_mask=None, tgt_mask=None):
        y, self.self_attn_logits = self.self_attn(x, x, x, tgt_mask)
        y = self.add_norm1(x, y)

        shortcut = y
        pos_vector = self.pos_embed(y).expand_as(y)
        y, _ = self.pos_attn(pos_vector, y, y, tgt_mask)
        y = self.add_norm2(shortcut, y)

        z, self.cross_attn_logits = self.cross_attn(y, encoder_out, encoder_out, src_mask)
        z = self.add_norm3(y, z)

        out = self.ffn(z)
        out = self.add_norm4(z, out)

        return out
