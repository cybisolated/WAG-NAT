import torch
from torch import nn

from .sub_models import *


def window_partition(x, window_size):
    # x: [N, L, D]
    N, L, D = x.shape
    assert L % window_size == 0
    # [N, L, D] ==> [N, M, W, D], M is the number of windows
    x = x.view(N, L // window_size, window_size, D)
    # [N, M, W, D] ==> [N * M, W, D]
    return x.view(-1, window_size, D)


def window_restore(windows, window_size, L):
    # L is the encoder sequence length
    assert L % window_size == 0
    # windows: [N * M, W, D]
    N = int(windows.shape[0] / (L / window_size))
    # [N * M, W, D] ==> [N, M, W, D]
    x = windows.view(N, L // window_size, window_size, -1)
    # [N, M, W, D] ==> [N, L, D]
    x = x.view(N, L, -1)

    return x


class Encoder(nn.Module):
    def __init__(self, d_model, window_size, kernel_size, num_heads, num_layers, ffn_hidden_size, activation, dropout):
        super().__init__()
        self.enc_layers = nn.ModuleList(
            [
                EncoderLayer(d_model, window_size, kernel_size, num_heads, ffn_hidden_size, activation, dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x):
        attns = []
        for enc_layer in self.enc_layers:
            x, attn = enc_layer(x)
            attns.append(attn)

        return x, attns


class CanonicalEncoder(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, ffn_hidden_size, activation, dropout):
        super().__init__()
        self.enc_layers = nn.ModuleList(
            [CanonicalEncoderLayer(d_model, num_heads, ffn_hidden_size, activation, dropout) for _ in range(num_layers)]
        )

    def forward(self, x):
        attns = []
        for enc_layer in self.enc_layers:
            x, attn = enc_layer(x)
            attns.append(attn)

        return x, attns


# 经典的编码器架构，用于比较和窗口注意力的attention map
class CanonicalEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, ffn_hidden_size, activation, dropout):
        super().__init__()

        self.attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.add_norm1 = AddNorm(d_model, dropout)

        self.ffn = PositionWiseFFN(d_model, ffn_hidden_size, activation, dropout)
        self.add_norm2 = AddNorm(d_model, dropout)

    def forward(self, x):
        shortcut = x
        x, attn_map = self.attn(x, x, x)

        x = self.add_norm1(shortcut, x)

        # ffn
        y = self.ffn(x)
        y = self.add_norm2(x, y)

        return y, attn_map


class EncoderLayer(nn.Module):
    def __init__(self, d_model, window_size, kernel_size, num_heads, ffn_hidden_size, activation, dropout):
        super().__init__()

        self.w_attn_blk = WindowAttentionBlock(d_model, window_size, kernel_size, num_heads, dropout)

        self.ffn = PositionWiseFFN(d_model, ffn_hidden_size, activation, dropout)
        self.add_norm = AddNorm(d_model, dropout)

    def forward(self, x):
        x, attn_map = self.w_attn_blk(x)

        # ffn
        y = self.ffn(x)
        y = self.add_norm(x, y)

        return y, attn_map


class WindowAttentionBlock(nn.Module):
    def __init__(self, d_model, window_size, kernel_size, num_heads, dropout):
        super().__init__()

        self.window_size = window_size
        self.kernel_size = kernel_size

        self.w_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.add_norm1 = AddNorm(d_model, dropout)

        self.interaction_layer = nn.Sequential(
            nn.Conv1d(
                d_model * window_size, d_model * window_size, kernel_size, stride=1, groups=window_size, bias=False
            ),
            nn.BatchNorm1d(d_model * window_size),
            nn.ELU(),
        )
        self.add_norm2 = AddNorm(d_model, dropout)

    def forward(self, x):
        # x: [N, L, D]
        N, L, D = x.shape
        shortcut = x

        # pad x to ensure L % window_size == 0
        # only pad the right part
        pad_r = (self.window_size - L % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, 0, pad_r))
        num_windows = x.shape[1] // self.window_size

        # construct windows
        # windows: [N * M, W, D], M is the number of windows, W is `self.window_size`
        windows = window_partition(x, self.window_size)

        # window self attention
        attn_windows, attn_map = self.w_attn(windows, windows, windows)

        # turn windows back into sequence
        x = window_restore(attn_windows, self.window_size, x.shape[1])
        if pad_r > 0:
            # remove the padding part
            x = x[:, :L, :]

        x = self.add_norm1(shortcut, x)
        shortcut = x

        # window interaction layer
        x = F.pad(x, (0, 0, 0, pad_r))
        windows_inter = window_partition(x, self.window_size)
        # [N * M, W, D] ==> [N, M, W, D]
        windows_inter = windows_inter.view(N, num_windows, -1, D)
        # [N, M, W, D] ==> [N, W, D, M]
        windows_inter = windows_inter.permute(0, 2, 3, 1).contiguous()
        # [N, W, D, M] ==> [N, W * D, M]
        windows_inter = windows_inter.view(N, -1, num_windows)

        windows_inter = self.interaction_layer(F.pad(windows_inter, (self.kernel_size - 1, 0)))
        # [N, W * D, M] ==> [N, W, D, M]
        windows_inter = windows_inter.view(N, self.window_size, D, num_windows)
        # [N, W, D, M] ==> [N, M, W, D]
        windows_inter = windows_inter.permute(0, 3, 1, 2).contiguous()
        # [N, M, W, D] ==> [N * M, W, D]
        windows_inter = windows_inter.view(N * num_windows, self.window_size, D)

        x = window_restore(windows_inter, self.window_size, x.shape[1])
        if pad_r > 0:
            x = x[:, :L, :]

        x = self.add_norm2(shortcut, x)

        return x, attn_map
