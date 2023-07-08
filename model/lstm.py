import numpy as np
import torch
from torch import nn
from .embeddings import DataEmbedding
from dataset import TimeSeriesDataset


class LSTM(nn.Module, TimeSeriesDataset):
    def __init__(
        self,
        L_in,
        L_out,
        d_model,
        hidden_size,
        num_layers,
        dropout,
        num_classes={},
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

        real_size = len(self.encoder_reals) + len(self.targets)
        output_size = len(self.targets)

        encoder_num_classes = []
        for name in self.encoder_cats:
            encoder_num_classes.append(num_classes[name])

        self.embedding = DataEmbedding(real_size, encoder_num_classes, d_model, dropout, pos_type='None')

        lstm_params = dict(
            input_size=d_model,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.stem = nn.LSTM(**lstm_params)
        self.proj = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x_reals = torch.concat([x['encoder_reals'], x['encoder_target']], dim=-1)
        x_cats = x['encoder_cats']

        # x: [N, L_in, d_model]
        x = self.embedding(x_reals, x_cats)
        out, _ = self.stem(x)
        out = self.proj(out[:, -self.L_out :, :])

        return out
