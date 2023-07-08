import math
import numpy as np
import torch
from torch import nn
from torch.nn import Parameter
from dataset import BaseDataset
from .embeddings import DataEmbedding


class ConvTransformer(nn.Module, BaseDataset):
    def __init__(
        self,
        num_classes,
        d_model,
        num_heads,
        num_layers,
        L_in,
        L_out,
        q_len,
        scale_attn=True,
        sub_len=1,
        dropout=0.0,
        sparse_attn=False,
        target=[],
        time_varying_known_reals=[],
        time_varying_known_categoricals=[],
        time_varying_unknown_reals=[],
        time_varying_unknown_categoricals=[],
        static_reals=[],
        static_categoricals=[],
        **kwargs
    ):
        super().__init__()

        self.target = target
        self.time_varying_known_reals = time_varying_known_reals
        self.time_varying_known_categoricals = time_varying_known_categoricals
        self.time_varying_unknown_reals = time_varying_unknown_reals
        self.time_varying_unknown_categoricals = time_varying_unknown_categoricals
        self.static_reals = static_reals
        self.static_categoricals = static_categoricals
        self.L_in = L_in
        self.L_out = L_out

        self.cat_embeddings = nn.ModuleDict()

        real_size = len(self.encoder_reals) + len(self.targets)
        encoder_num_classes = []
        for name in self.encoder_cats:
            encoder_num_classes.append(num_classes[name])

        output_size = len(self.targets)

        self.input_embedding = DataEmbedding(real_size, encoder_num_classes, d_model, dropout, pos_type='learn', L=L_in)

        self.stem = TransformerModel(
            num_heads, num_layers, d_model, L_in, q_len, scale_attn, sub_len, dropout, sparse_attn
        )

        self.out_proj = nn.Linear(d_model, output_size)
        self.predict_layer = nn.Linear(L_in, L_out)

        nn.init.normal_(self.input_embedding.pos_embedding.embed.weight, std=0.02)

    def forward(self, x):
        # x_reals: [N, L_in, D]
        x_reals = torch.concat([x['encoder_reals'], x['encoder_target']], dim=-1)
        x_cats = x['encoder_cats']

        x = self.input_embedding(x_reals, x_cats)

        # out: [N, L_in, d_model]
        out = self.stem(x)
        # [N, L_in, output_size]
        out = self.out_proj(out)
        if out.ndim == 2:
            out = out.unsqueeze(-1)
        # out: [N, output_size, L_in]
        out = out.transpose(1, 2)

        # [N, output_size, L_out]
        out = self.predict_layer(out)
        # [N, L_out, output_size]
        return out.transpose(1, 2)


class TransformerModel(nn.Module):
    def __init__(self, num_heads, num_layers, d_model, L_in, q_len, scale_attn, sub_len, dropout, sparse_attn):
        super().__init__()
        self.d_model = (d_model,)
        self.L_in = L_in

        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                Block(num_heads, d_model, L_in, scale_attn, q_len, sub_len, sparse_attn, dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)

        return x


class Block(nn.Module):
    def __init__(self, num_heads, d_model, L_in, scale_attn, q_len, sub_len, sparse_attn, dropout):
        super().__init__()
        self.attn = Attention(num_heads, d_model, L_in, scale_attn, q_len, sub_len, sparse_attn, dropout)
        self.ln1 = LayerNorm(d_model)
        self.mlp = MLP(d_model, dropout)
        self.ln2 = LayerNorm(d_model)

    def forward(self, x):
        attn = self.attn(x)
        ln1 = self.ln1(x + attn)
        mlp = self.mlp(ln1)
        out = self.ln2(ln1 + mlp)

        return out


class Attention(nn.Module):
    def __init__(self, num_heads, d_model, L_in, scale_attn, q_len, sub_len, sparse, dropout):
        super(Attention, self).__init__()

        if sparse:
            # use sparse attention
            mask = self.log_mask(L_in, sub_len)
        else:
            mask = torch.tril(torch.ones(L_in, L_in)).view(1, 1, L_in, L_in)

        self.register_buffer('mask_tri', mask)
        self.num_heads = num_heads
        self.split_size = d_model * self.num_heads
        self.scale = scale_attn
        self.q_len = q_len
        self.query_key = nn.Conv1d(d_model, d_model * num_heads * 2, self.q_len)
        self.value = Conv1D(d_model, d_model * num_heads)
        self.c_proj = Conv1D(d_model * num_heads, d_model)
        self.dropout = nn.Dropout(dropout)

    def log_mask(self, L_in, sub_len):
        mask = torch.zeros((L_in, L_in), dtype=torch.float)
        for i in range(L_in):
            mask[i] = self.row_mask(i, sub_len, L_in)
        return mask.view(1, 1, mask.size(0), mask.size(1))

    def row_mask(self, index, sub_len, L_in):
        """
        Remark:
        1 . Currently, dense matrices with sparse multiplication are not supported by Pytorch. Efficient implementation
            should deal with CUDA kernel, which we haven't implemented yet.
        2 . Our default setting here use Local attention and Restart attention.
        3 . For index-th row, if its past is smaller than the number of cells the last
            cell can attend, we can allow current cell to attend all past cells to fully
            utilize parallel computing in dense matrices with sparse multiplication."""
        log_l = math.ceil(np.log2(sub_len))
        mask = torch.zeros(L_in, dtype=torch.float)
        if (L_in // sub_len) * 2 * (log_l) > index:
            mask[: (index + 1)] = 1
        else:
            while index >= 0:
                if (index - log_l + 1) < 0:
                    mask[:index] = 1
                    break
                mask[index - log_l + 1 : (index + 1)] = 1  # Local attention
                for i in range(0, log_l):
                    new_index = index - log_l + 1 - 2**i
                    if (index - new_index) <= sub_len and new_index >= 0:
                        mask[new_index] = 1
                index -= sub_len
        return mask

    def attn(self, query: torch.Tensor, key, value: torch.Tensor):
        activation = torch.softmax
        pre_att = torch.matmul(query, key)
        if self.scale:
            pre_att = pre_att / math.sqrt(value.size(-1))
        mask = self.mask_tri[:, :, : pre_att.size(-2), : pre_att.size(-1)]
        pre_att = pre_att * mask + -1e9 * (1 - mask)
        pre_att = activation(pre_att, dim=-1)
        pre_att = self.dropout(pre_att)
        attn = torch.matmul(pre_att, value)

        return attn

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.num_heads, x.size(-1) // self.num_heads)
        x = x.view(*new_x_shape)
        if k:
            return x.permute(0, 2, 3, 1)
        else:
            return x.permute(0, 2, 1, 3)

    def forward(self, x):
        value = self.value(x)
        qk_x = nn.functional.pad(x.permute(0, 2, 1), pad=(self.q_len - 1, 0))
        query_key = self.query_key(qk_x).permute(0, 2, 1)
        query, key = query_key.split(self.split_size, dim=2)
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)
        attn = self.attn(query, key, value)
        attn = self.merge_heads(attn)
        attn = self.c_proj(attn)
        attn = self.dropout(attn)
        return attn


class MLP(nn.Module):
    def __init__(self, d_model, dropout):
        super().__init__()
        hidden_size = d_model * 4
        self.fc = nn.Sequential(
            Conv1D(d_model, hidden_size), nn.ReLU(), Conv1D(hidden_size, d_model), nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.fc(x)


class Conv1D(nn.Module):
    def __init__(self, in_dim, out_dim, rf=1):
        super(Conv1D, self).__init__()
        self.rf = rf
        self.out_dim = out_dim
        if rf == 1:
            w = torch.empty(in_dim, out_dim)
            nn.init.normal_(w, std=0.02)
            self.w = Parameter(w)
            self.b = Parameter(torch.zeros(out_dim))
        else:
            raise NotImplementedError

    def forward(self, x):
        if self.rf == 1:
            size_out = x.size()[:-1] + (self.out_dim,)
            x = torch.addmm(self.b, x.view(-1, x.size(-1)), self.w)
            x = x.view(*size_out)
        else:
            raise NotImplementedError
        return x


class LayerNorm(nn.Module):
    def __init__(self, d_model, e=1e-5):
        super().__init__()
        self.g = nn.Parameter(torch.ones(d_model))
        self.b = nn.Parameter(torch.zeros(d_model))
        self.e = e

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = (x - mu).pow(2).mean(-1, keepdim=True)
        x = (x - mu) / torch.sqrt(sigma + self.e)
        return self.g * x + self.b


if __name__ == '__main__':
    model = ConvTransformer(15, 64, 1, 2, 1, 72, 24, 3, sparse_attn=False, dropout=0.1)
    x = torch.rand(32, 72, 15)
    yhat = model(x)
    print(yhat.shape)
