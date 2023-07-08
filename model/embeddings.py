import torch
import math
from torch import nn


class RealEmbedding(nn.Module):
    def __init__(self, input_size, d_model):
        super().__init__()
        # self.conv = nn.Conv1d(input_size, d_model, kernel_size=3,
        #                       padding=1, padding_mode='circular')

        # # init weights
        # for m in self.modules():
        #     if isinstance(m, nn.Conv1d):
        #         nn.init.kaiming_normal_(m.weight)
        self.linear = nn.Linear(input_size, d_model)

    def forward(self, x):
        # x: [N, L, input_size]
        # x = self.conv(x.transpose(1, 2)).transpose(1, 2)
        # return x.contiguous()
        return self.linear(x)


class CategoricalEmbedding(nn.Module):
    def __init__(self, num_classes, d_model):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(num_class, d_model) for num_class in num_classes])

    def forward(self, x):
        out = 0.0
        for idx, embedding in enumerate(self.embeddings):
            out += embedding(x[..., idx])
        return out


class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, d_model, max_length=1000):
        super().__init__()

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
        return self.pe[:, : x.shape[1]]


class LearnablePositionEmbedding(nn.Module):
    def __init__(self, L, d_model):
        super().__init__()
        self.embed = nn.Embedding(L, d_model)
        nn.init.constant_(self.embed.weight, 0.0)

    def forward(self, x):
        pos = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        pos = self.embed(pos)
        return pos.expand_as(x)


class DataEmbedding(nn.Module):
    def __init__(self, real_size, num_classes, d_model, dropout, pos_type='sin', L=None):
        super().__init__()

        self.real_embedding = RealEmbedding(real_size, d_model)
        self.cat_embedding = CategoricalEmbedding(num_classes, d_model)
        if pos_type == 'sin':
            self.pos_embedding = SinusoidalPositionEmbedding(d_model)
        elif pos_type == 'learn':
            assert L is not None
            self.pos_embedding = LearnablePositionEmbedding(L, d_model)
        else:
            self.pos_embedding = None

        self.dropout = nn.Dropout(dropout)

    def forward(self, x_reals, x_cats):
        x = self.real_embedding(x_reals) + self.cat_embedding(x_cats)
        if self.pos_embedding is not None:
            x += self.pos_embedding(x)

        return self.dropout(x)
