import math
import torch
from torch import nn
import torch.nn.functional as F


def attention(query, key, value, dropout, mask=None):
    dim_k = query.shape[-1]
    scores = query @ key.transpose(-2, -1) / math.sqrt(dim_k)

    # mask
    if mask is not None:
        scores = torch.masked_fill(scores, mask == 0, float('-inf'))

    # softmax normalization
    attn_map = F.softmax(scores, dim=-1)

    # dropout
    if dropout:
        attn_map = dropout(attn_map)

    out = attn_map @ value

    return out, attn_map


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout):
        super().__init__()
        assert d_model % num_heads == 0

        self.dim_per_head = d_model // num_heads
        self.num_heads = num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

        # store the attention scores (softmaxed)

    def forward(self, query, key, value, mask=None):
        # query, key, value: [N, L, d_model]
        if mask is not None:
            # add a dimension on to match `num_heads`
            mask = mask.unsqueeze(1)

        batch_size = query.shape[0]

        # [N, L, d_model] ==> [N, L, num_heads, dim_per_head] ==> [N, num_heads, L, dim_per_head]
        query = self.W_q(query).view(batch_size, -1, self.num_heads, self.dim_per_head).transpose(1, 2)
        key = self.W_q(key).view(batch_size, -1, self.num_heads, self.dim_per_head).transpose(1, 2)
        value = self.W_q(value).view(batch_size, -1, self.num_heads, self.dim_per_head).transpose(1, 2)

        # out: [N, num_heads, L, dim_per_head]
        # attn_logits: [N, num_heads, L, L] (softmaxed)
        out, attn_logits = attention(query, key, value, self.dropout, mask)

        # [N, num_heads, L, dim_per_head] ==> [N, L, num_heads, dim_per_head] ==> [N, L, d_model]
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.dim_per_head)

        out = self.W_o(out)

        return out, attn_logits


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_length=1000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_length, d_model)

        positon = torch.arange(max_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(positon * div_term)
        pe[:, 1::2] = torch.cos(positon * div_term)

        # add batch dim
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        self.pe = self.pe.to(x.device)
        x += self.pe[:, : x.shape[1], :]

        return self.dropout(x)


class PositionWiseFFN(nn.Module):
    def __init__(self, d_model, ffn_hidden_size, activation, dropout):
        super().__init__()
        self.linear1 = nn.Linear(d_model, ffn_hidden_size)
        self.activation = getattr(nn.functional, activation)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(ffn_hidden_size, d_model)

    def forward(self, x):
        x = self.linear1(x)
        x = self.dropout(self.activation(x))
        return self.linear2(x)


class AddNorm(nn.Module):
    def __init__(self, norm_shape, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        # layer norm
        self.ln = nn.LayerNorm(norm_shape)

    def forward(self, x, y):
        return self.ln(self.dropout(y) + x)
