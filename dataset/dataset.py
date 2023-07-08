from torch.utils.data import Dataset
import numpy as np
import torch
import pandas as pd

from dataset.base import BaseDataset


class TimeSeriesDataset(Dataset, BaseDataset):
    def __init__(self,
                 df_data: pd.DataFrame,
                 L_in,
                 L_out,
                 group_id=[],
                 target=[],
                 time_varying_known_reals=[],
                 time_varying_known_categoricals=[],
                 time_varying_unknown_reals=[],
                 time_varying_unknown_categoricals=[],
                 static_reals=[],
                 static_categoricals=[],
                 num_samples=None,
                 **kwargs):
        super().__init__()

        self.group_id = group_id
        self.target = target
        self.time_varying_known_reals = time_varying_known_reals
        self.time_varying_known_categoricals = time_varying_known_categoricals
        self.time_varying_unknown_reals = time_varying_unknown_reals
        self.time_varying_unknown_categoricals = time_varying_unknown_categoricals
        self.static_reals = static_reals
        self.static_categoricals = static_categoricals
        self.L_in = L_in
        self.L_out = L_out
        self.L_window = L_in + L_out
        self.num_samples = num_samples

        self.data = df_data.copy()
        # Each item in `self.index[List]` is a `int`,
        # indicating start_idx of `self.data`.
        self.index = self.construct_index()
        # convert pd.DataFrame to torch.Tensor to accelerate Dataloader instantiation
        self.data = self.construct_tensor_data()

        # construct data using sliding window by `id` separately
        # self.data = []
        # for idx, df_sliced in df_data.groupby(by=group_id):
        #     df = df_sliced.copy()
        #     df_slided = [df[i:i+L_window] for i in range(len(df) - L_window + 1)]
        #     self.data.extend(df_slided)

    def construct_index(self):
        index = []
        offset = 0
        for idx, df_sliced in self.data.groupby(by=self.group_id):
            num_entries = len(df_sliced)
            # Number of `df_sliced` must greater equal than `self.L_window`,
            # otherwise cannot construc even a single data using sliding window.
            if num_entries < self.L_window:
                continue
            index.extend([
                i + offset
                for i in range(num_entries - self.L_window + 1)
            ])
            # update offset
            offset += len(df_sliced)

        # random sample `self.num_samples` samples from index
        if self.num_samples is not None:
            if len(index) > self.num_samples:
                index = np.random.choice(index, self.num_samples)

        return index

    def construct_tensor_data(self):
        return dict(
            encoder_reals=torch.tensor(self.data[self.encoder_reals].to_numpy(np.float32), dtype=torch.float),
            encoder_cats=torch.tensor(self.data[self.encoder_cats].to_numpy(np.int32), dtype=torch.long),
            encoder_target=torch.tensor(self.data[self.targets].to_numpy(np.float32), dtype=torch.float),
            decoder_reals=torch.tensor(self.data[self.decoder_reals].to_numpy(np.float32), dtype=torch.float),
            decoder_cats=torch.tensor(self.data[self.decoder_cats].to_numpy(np.int32), dtype=torch.long),
            decoder_target=torch.tensor(self.data[self.targets].to_numpy(np.float32), dtype=torch.float),
        )

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        s = self.index[idx]

        return dict(
            encoder_reals=self.data['encoder_reals'][s:s+self.L_in, :],
            encoder_cats=self.data['encoder_cats'][s:s+self.L_in, :],
            encoder_target=self.data['encoder_target'][s:s+self.L_in, :],
            decoder_reals=self.data['decoder_reals'][s+self.L_in:s+self.L_window, :],
            decoder_cats=self.data['decoder_cats'][s+self.L_in:s+self.L_window, :],
            decoder_target=self.data['decoder_target'][s+self.L_in:s+self.L_window, :],
        )
